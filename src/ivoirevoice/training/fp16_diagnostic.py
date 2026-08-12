"""Bounded train-only FP16 diagnostics for the full Whisper workflow."""

from __future__ import annotations

import csv
import gc
import hashlib
import hmac
import importlib
import json
import math
import secrets
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeVar, cast

from ivoirevoice.data.audio import sha256_file
from ivoirevoice.exceptions import ConfigError
from ivoirevoice.models.whisper import load_whisper_settings
from ivoirevoice.training.audit import REQUIRED_COLUMNS, ManifestRow, _parse_manifest_row
from ivoirevoice.training.full_settings import FullTrainingSettings
from ivoirevoice.training.pilot_finetune import (
    PilotCollator,
    _chunks,
    _linear_warmup_scheduler,
    _load_model,
    _ordered_train_rows,
    _seed_everything,
    _to_device,
)
from ivoirevoice.training.pilot_settings import PilotSettings
from ivoirevoice.training.whisper_finetune import (
    directory_sha256,
    git_provenance,
    identity_sha256,
    load_json_object,
    require_matching_identity,
    selection_sha256,
    write_json_atomic,
)

DIAGNOSTIC_DIRECTORY_NAME = "fp16_diagnostic"
DIAGNOSTIC_SUMMARY_FILENAME = "fp16_diagnostic_summary.json"
PREFLIGHT_REPORT_FILENAME = "preflight_report.json"
FINAL_EVALUATION_RECEIPT_FILENAME = "final_holdout_evaluation_receipt.json"
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class DiagnosticContext:
    """Only the train rows and immutable identities needed by the diagnostic."""

    train_rows: tuple[ManifestRow, ...]
    model_settings: Any
    code_commit: str
    config_sha256: str
    manifest_sha256: str
    initial_checkpoint_sha256: str
    train_selection_sha256: str
    run_id: str


@dataclass(frozen=True, slots=True)
class AmpStepOutcome:
    """Public GradScaler evidence for one attempted optimizer update."""

    scale_before: float
    scale_after: float
    optimizer_step_executed: bool
    amp_skip: bool


@dataclass(slots=True)
class DiagnosticProgress:
    """Counters that distinguish attempts from successful optimizer steps."""

    optimizer_attempts: int = 0
    successful_optimizer_steps: int = 0
    amp_skipped_steps: int = 0
    current_consecutive_skips: int = 0
    max_consecutive_skips: int = 0

    def record(self, outcome: AmpStepOutcome) -> None:
        self.optimizer_attempts += 1
        if outcome.amp_skip:
            self.amp_skipped_steps += 1
            self.current_consecutive_skips += 1
            self.max_consecutive_skips = max(
                self.max_consecutive_skips,
                self.current_consecutive_skips,
            )
            return
        self.successful_optimizer_steps += 1
        self.current_consecutive_skips = 0


def _chunk_sequence(values: Sequence[T], size: int) -> Sequence[Sequence[T]]:
    """Group an arbitrary sequence while preserving its element type."""

    return [values[index : index + size] for index in range(0, len(values), size)]


def bounded_optimizer_groups(
    rows: Sequence[ManifestRow],
    *,
    batch_size: int,
    accumulation_steps: int,
    max_optimizer_attempts: int,
) -> Sequence[Sequence[Sequence[ManifestRow]]]:
    """Build only the optimizer groups permitted by the diagnostic budget."""

    micro_batches = _chunks(rows, batch_size)
    groups = _chunk_sequence(micro_batches, accumulation_steps)
    return groups[:max_optimizer_attempts]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ConfigError(f"Impossible de calculer l'empreinte de {path.name}.") from exc
    return digest.hexdigest()


def _load_metadata(path: Path) -> dict[str, Any]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigError("Métadonnées dataset illisibles pour le diagnostic.") from exc
    if not isinstance(raw, Mapping) or not all(isinstance(key, str) for key in raw):
        raise ConfigError("Métadonnées dataset invalides pour le diagnostic.")
    return cast(dict[str, Any], dict(raw))


def _validate_train_audio(row: ManifestRow, dataset_root: Path) -> None:
    path = dataset_root / row.audio_path
    if not path.is_file():
        raise ConfigError("Un audio train attendu est absent du diagnostic.")
    if sha256_file(path) != row.audio_sha256:
        raise ConfigError("Un audio train du diagnostic ne correspond plus au manifeste.")


def load_diagnostic_train_rows(
    settings: FullTrainingSettings,
) -> tuple[tuple[ManifestRow, ...], str]:
    """Load and validate train rows without constructing any non-train row."""

    manifest_sha256 = sha256_file(settings.manifest_path)
    if manifest_sha256 != settings.expected_manifest_sha256:
        raise ConfigError("Le hash du manifeste diagnostic n'est plus approuvé.")
    metadata = _load_metadata(settings.dataset_metadata_path)
    expected_metadata = {
        "manifest_sha256": manifest_sha256,
        "dataset_version": settings.expected_dataset_version,
        "publication_allowed": False,
        "model_derivative_publication_allowed": False,
        "usage_scope": "local_research_only",
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise ConfigError(f"Métadonnée diagnostic inattendue : {field}.")
    counts = metadata.get("audio_count_by_split")
    if not isinstance(counts, Mapping) or counts.get(settings.train_split) != (
        settings.train_audio_count
    ):
        raise ConfigError("Le compte train agrégé a changé avant le diagnostic.")

    rows: list[ManifestRow] = []
    try:
        with settings.manifest_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or ()))
            if missing:
                raise ConfigError(f"Colonnes train manquantes : {missing}.")
            for line_number, raw in enumerate(reader, 2):
                if raw.get("split") != settings.train_split:
                    continue
                rows.append(_parse_manifest_row(raw, line_number))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ConfigError("Impossible de lire le train gelé pour le diagnostic.") from exc

    train_rows = tuple(rows)
    observed = {
        "audio_count": len(train_rows),
        "speaker_count": len({row.speaker_id for row in train_rows}),
    }
    expected = {
        "audio_count": settings.train_audio_count,
        "speaker_count": settings.train_speaker_count,
    }
    if observed != expected:
        raise ConfigError(f"Comptes train inattendus pour le diagnostic : {observed}.")
    if len({row.utterance_id for row in train_rows}) != len(train_rows):
        raise ConfigError("Le train diagnostic contient un utterance_id dupliqué.")
    if len({row.audio_sha256 for row in train_rows}) != len(train_rows):
        raise ConfigError("Le train diagnostic contient un audio dupliqué.")
    for row in train_rows:
        if (
            row.split != settings.train_split
            or row.language != "dyu"
            or row.usage_scope != "local_research_only"
            or not row.target_text.strip()
            or not 0 < row.duration_seconds <= settings.max_audio_seconds
            or row.sample_rate_hz != 16_000
            or row.channels != 1
        ):
            raise ConfigError("Une ligne train est incompatible avec le diagnostic.")
        _validate_train_audio(row, settings.dataset_root)
    return train_rows, manifest_sha256


def _load_context(settings: FullTrainingSettings) -> DiagnosticContext:
    root = Path(__file__).resolve().parents[3]
    code_commit, clean = git_provenance(root)
    if not clean:
        raise ConfigError("Le diagnostic FP16 exige un commit Git propre.")
    receipt = settings.artifact_output_directory / FINAL_EVALUATION_RECEIPT_FILENAME
    if receipt.exists():
        raise ConfigError("Un reçu final_holdout interdit tout diagnostic d'entraînement.")
    output_directory = settings.fp16_diagnostic_output_directory
    if output_directory.exists():
        raise ConfigError("Un diagnostic FP16 existe déjà : écrasement refusé.")

    train_rows, manifest_sha256 = load_diagnostic_train_rows(settings)
    config_sha256 = _sha256_file(settings.config_path)
    initial_checkpoint_sha256 = directory_sha256(settings.initial_checkpoint_path)
    train_selection_sha256 = selection_sha256(
        {"train": tuple((row.utterance_id, row.audio_sha256) for row in train_rows)}
    )
    preflight = load_json_object(settings.artifact_output_directory / PREFLIGHT_REPORT_FILENAME)
    require_matching_identity(
        preflight,
        {
            "status": "passed",
            "code_commit": code_commit,
            "config_sha256": config_sha256,
            "manifest_sha256": manifest_sha256,
            "initial_checkpoint_sha256": initial_checkpoint_sha256,
            "train_audio_count": len(train_rows),
            "historical_test_decoded": False,
            "final_holdout_decoded": False,
        },
        description="Le préflight du diagnostic FP16",
    )
    model_settings = load_whisper_settings(root / settings.model_config_path)
    if (
        model_settings.model_id != settings.expected_model_id
        or model_settings.model_revision != settings.expected_model_revision
        or model_settings.task != "transcribe"
        or model_settings.language is not None
    ):
        raise ConfigError("Le modèle du diagnostic ne correspond plus au protocole.")
    run_identity = {
        "stage": "fp16-diagnostic",
        "selection_sha256": train_selection_sha256,
        "initial_checkpoint_sha256": initial_checkpoint_sha256,
        "config_sha256": config_sha256,
        "code_commit": code_commit,
    }
    return DiagnosticContext(
        train_rows=train_rows,
        model_settings=model_settings,
        code_commit=code_commit,
        config_sha256=config_sha256,
        manifest_sha256=manifest_sha256,
        initial_checkpoint_sha256=initial_checkpoint_sha256,
        train_selection_sha256=train_selection_sha256,
        run_id=identity_sha256(run_identity),
    )


def _require_cuda(torch: Any) -> Any:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise ConfigError("nvidia-smi absent : diagnostic FP16 refusé.")
    try:
        result = subprocess.run(
            [executable, "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ConfigError("nvidia-smi inutilisable pour le diagnostic.") from exc
    if result.returncode or not result.stdout.strip() or not torch.cuda.is_available():
        raise ConfigError("CUDA indisponible : diagnostic FP16 refusé.")
    device = torch.device("cuda:0")
    probe = torch.ones((2, 2), dtype=torch.float16, device=device)
    if not bool((probe @ probe).isfinite().all()):
        raise ConfigError("Le calcul FP16 CUDA du diagnostic a échoué.")
    torch.cuda.synchronize(device)
    return device


def inspect_gradients(model: Any, torch: Any) -> dict[str, Any]:
    """Inspect unscaled gradients without modifying them."""

    tensors_with_grad = 0
    tensors_with_inf = 0
    tensors_with_nan = 0
    nonfinite_values = 0
    max_abs_finite = 0.0
    squared_norm = 0.0
    all_finite = True
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        tensors_with_grad += 1
        gradient = parameter.grad.detach()
        values = gradient.coalesce().values() if gradient.is_sparse else gradient
        inf_mask = torch.isinf(values)
        nan_mask = torch.isnan(values)
        finite_mask = torch.isfinite(values)
        inf_count = int(inf_mask.sum().item())
        nan_count = int(nan_mask.sum().item())
        if inf_count:
            tensors_with_inf += 1
        if nan_count:
            tensors_with_nan += 1
        nonfinite_values += inf_count + nan_count
        all_finite = all_finite and inf_count == 0 and nan_count == 0
        if bool(finite_mask.any()):
            finite_values = values[finite_mask]
            max_abs_finite = max(
                max_abs_finite,
                float(finite_values.abs().max().detach().cpu()),
            )
            tensor_norm = float(torch.linalg.vector_norm(finite_values.float()).detach().cpu())
            squared_norm += tensor_norm * tensor_norm
    return {
        "gradient_global_norm": math.sqrt(squared_norm) if all_finite else None,
        "gradient_global_norm_finite": all_finite,
        "parameter_tensors_with_grad": tensors_with_grad,
        "gradient_tensors_with_inf": tensors_with_inf,
        "gradient_tensors_with_nan": tensors_with_nan,
        "nonfinite_gradient_values": nonfinite_values,
        "max_abs_finite_gradient": max_abs_finite,
    }


def apply_amp_step(
    optimizer: Any,
    scaler: Any,
    scheduler: Any,
) -> AmpStepOutcome:
    """Attempt one update and advance the scheduler only when AdamW ran."""

    scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    amp_skip = scale_after < scale_before
    if not amp_skip:
        scheduler.step()
    return AmpStepOutcome(
        scale_before=scale_before,
        scale_after=scale_after,
        optimizer_step_executed=not amp_skip,
        amp_skip=amp_skip,
    )


def _sample_metadata(
    rows: Sequence[ManifestRow],
    batch: Mapping[str, Any],
    hash_key: bytes,
) -> list[dict[str, Any]]:
    features = batch["input_features"]
    labels = batch["labels"]
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        result.append(
            {
                "utterance_hash": hmac.new(
                    hash_key,
                    row.utterance_id.encode("utf-8"),
                    hashlib.sha256,
                ).hexdigest(),
                "duration_seconds": row.duration_seconds,
                "feature_length": int(features[index].shape[-1]),
                "label_length": int((labels[index] != -100).sum().item()),
            }
        )
    return result


def _parameters_are_finite(model: Any, torch: Any) -> bool:
    return all(bool(torch.isfinite(parameter.detach()).all()) for parameter in model.parameters())


def _diagnostic_attempt(
    *,
    optimizer_attempt: int,
    row_batches: Sequence[Sequence[ManifestRow]],
    collator: PilotCollator,
    model: Any,
    optimizer: Any,
    scaler: Any,
    scheduler: Any,
    device: Any,
    torch: Any,
    max_grad_norm: float,
    hash_key: bytes,
) -> tuple[dict[str, Any], AmpStepOutcome | None]:
    optimizer.zero_grad(set_to_none=True)
    divisor = len(row_batches)
    losses: list[float] = []
    samples: list[dict[str, Any]] = []
    terminal_error: str | None = None
    for row_batch in row_batches:
        batch = _to_device(collator(row_batch), device)
        samples.extend(_sample_metadata(row_batch, batch, hash_key))
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            raw_loss = model(**batch).loss
            scaled_loss = raw_loss / divisor
        value = float(raw_loss.detach().cpu())
        if not math.isfinite(value):
            terminal_error = "nonfinite_loss"
            break
        if value > 50:
            terminal_error = "loss_above_50"
            break
        scaler.scale(scaled_loss).backward()
        losses.append(value)

    attempt: dict[str, Any] = {
        "optimizer_attempt": optimizer_attempt,
        "micro_batches_accumulated": len(losses),
        "mean_raw_loss": sum(losses) / len(losses) if losses else None,
        "min_raw_loss": min(losses) if losses else None,
        "max_raw_loss": max(losses) if losses else None,
        "loss_finite": terminal_error != "nonfinite_loss",
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "cuda_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "samples": samples,
        "terminal_error": terminal_error,
    }
    if terminal_error is not None:
        attempt.update(
            {
                "scale_before": float(scaler.get_scale()),
                "scale_after": float(scaler.get_scale()),
                "gradient_global_norm": None,
                "gradient_global_norm_finite": False,
                "parameter_tensors_with_grad": sum(
                    parameter.grad is not None for parameter in model.parameters()
                ),
                "gradient_tensors_with_inf": 0,
                "gradient_tensors_with_nan": 0,
                "nonfinite_gradient_values": 0,
                "max_abs_finite_gradient": 0.0,
                "optimizer_step_executed": False,
                "amp_skip": False,
                "parameters_finite": _parameters_are_finite(model, torch),
            }
        )
        optimizer.zero_grad(set_to_none=True)
        return attempt, None

    scaler.unscale_(optimizer)
    gradient_stats = inspect_gradients(model, torch)
    attempt.update(gradient_stats)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    outcome = apply_amp_step(optimizer, scaler, scheduler)
    attempt.update(
        {
            "scale_before": outcome.scale_before,
            "scale_after": outcome.scale_after,
            "optimizer_step_executed": outcome.optimizer_step_executed,
            "amp_skip": outcome.amp_skip,
            "parameters_finite": _parameters_are_finite(model, torch),
        }
    )
    return attempt, outcome


def _technical_outlier_signal(attempts: Sequence[Mapping[str, Any]]) -> bool:
    skipped = [attempt for attempt in attempts if attempt.get("amp_skip") is True]
    successful = [attempt for attempt in attempts if attempt.get("optimizer_step_executed") is True]
    if len(skipped) < 2 or not successful:
        return False
    for field in ("duration_seconds", "feature_length", "label_length"):
        skipped_values = [
            float(sample[field])
            for attempt in skipped
            for sample in cast(list[dict[str, Any]], attempt["samples"])
        ]
        successful_values = [
            float(sample[field])
            for attempt in successful
            for sample in cast(list[dict[str, Any]], attempt["samples"])
        ]
        if min(skipped_values) > max(successful_values) or max(skipped_values) < min(
            successful_values
        ):
            return True
    return False


def classify_diagnostic(
    attempts: Sequence[Mapping[str, Any]],
    progress: DiagnosticProgress,
) -> tuple[str, str]:
    """Classify observed behavior without treating one skip as failure."""

    if any(
        attempt.get("terminal_error") is not None or attempt.get("parameters_finite") is False
        for attempt in attempts
    ):
        return "CATEGORY_E_OTHER_NUMERICAL_INSTABILITY", "terminal_numerical_error"
    if _technical_outlier_signal(attempts):
        return "CATEGORY_D_DATA_BATCH_SPECIFIC_FAILURE", "technical_outlier_separation"
    if progress.max_consecutive_skips >= 3 or (
        progress.amp_skipped_steps > progress.successful_optimizer_steps
    ):
        return "CATEGORY_C_REPEATED_FP16_OVERFLOW", "repeated_or_dominant_skips"
    skipped_indices = [
        int(attempt["optimizer_attempt"]) for attempt in attempts if attempt.get("amp_skip") is True
    ]
    if skipped_indices and max(skipped_indices) <= 4:
        trailing_successes = len(attempts) - max(skipped_indices)
        if trailing_successes >= 8:
            return "CATEGORY_A_INITIAL_SCALE_CALIBRATION", "early_skip_then_stable"
    return "CATEGORY_B_OCCASIONAL_AMP_SKIP", "stable_or_isolated_skips"


def _build_summary(
    *,
    settings: FullTrainingSettings,
    context: DiagnosticContext,
    attempts: Sequence[Mapping[str, Any]],
    progress: DiagnosticProgress,
    gpu_name: str,
    bf16_supported: bool,
) -> dict[str, Any]:
    scales_before = [float(attempt["scale_before"]) for attempt in attempts]
    scales_after = [float(attempt["scale_after"]) for attempt in attempts]
    category, category_reason = classify_diagnostic(attempts, progress)
    return {
        "schema_version": 1,
        "status": "completed",
        "diagnostic_only": True,
        "run_id": context.run_id,
        "code_commit": context.code_commit,
        "config_sha256": context.config_sha256,
        "manifest_sha256": context.manifest_sha256,
        "initial_checkpoint_name": settings.initial_checkpoint_path.name,
        "initial_checkpoint_sha256": context.initial_checkpoint_sha256,
        "train_selection_sha256": context.train_selection_sha256,
        "optimizer_attempts": progress.optimizer_attempts,
        "successful_optimizer_steps": progress.successful_optimizer_steps,
        "amp_skipped_steps": progress.amp_skipped_steps,
        "first_scale": scales_before[0],
        "final_scale": scales_after[-1],
        "minimum_scale": min((*scales_before, *scales_after)),
        "nonfinite_gradient_attempts": sum(
            int(attempt["nonfinite_gradient_values"]) > 0 for attempt in attempts
        ),
        "max_consecutive_skips": progress.max_consecutive_skips,
        "category": category,
        "category_reason": category_reason,
        "gpu_name": gpu_name,
        "bf16_supported": bf16_supported,
        "training_parameters": {
            "precision": "fp16",
            "learning_rate": settings.learning_rate,
            "train_batch_size": settings.train_batch_size,
            "gradient_accumulation_steps": settings.gradient_accumulation_steps,
            "max_grad_norm": settings.max_grad_norm,
            "seed": settings.seed,
            "max_optimizer_attempts": settings.fp16_diagnostic_max_optimizer_attempts,
        },
        "attempts": list(attempts),
        "isolation": {
            "train_only": True,
            "validation_used": False,
            "historical_test_used": False,
            "final_holdout_used": False,
            "checkpoint_created": False,
            "model_output_created": False,
            "resume_allowed": False,
        },
        "privacy": {
            "contains_transcriptions": False,
            "contains_speaker_ids": False,
            "contains_audio": False,
            "contains_local_paths": False,
            "utterance_hash_key_persisted": False,
        },
    }


def write_diagnostic_report(
    settings: FullTrainingSettings,
    summary: Mapping[str, Any],
) -> Path:
    """Write one report while refusing to overwrite an earlier diagnostic."""

    output_directory = settings.fp16_diagnostic_output_directory
    try:
        output_directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ConfigError("Un diagnostic FP16 existe déjà : écrasement refusé.") from exc
    path = output_directory / DIAGNOSTIC_SUMMARY_FILENAME
    write_json_atomic(path, summary)
    return path


def run_fp16_diagnostic(settings: FullTrainingSettings) -> dict[str, Any]:
    """Run bounded FP16 attempts on train only, without any checkpoint output."""

    context = _load_context(settings)
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    device = _require_cuda(torch)
    _seed_everything(settings.seed, torch)
    processor = transformers.WhisperProcessor.from_pretrained(
        settings.initial_checkpoint_path,
        cache_dir=str(context.model_settings.cache_dir),
        local_files_only=True,
        trust_remote_code=False,
    )
    processor.tokenizer.set_prefix_tokens(task="transcribe")
    collator = PilotCollator(
        cast(
            PilotSettings,
            SimpleNamespace(
                train_split=settings.train_split,
                validation_split=settings.train_split,
                dataset_root=settings.dataset_root,
                max_audio_seconds=settings.max_audio_seconds,
            ),
        ),
        processor,
        context.model_settings.expected_sampling_rate_hz,
        torch,
    )
    model = _load_model(
        source=settings.initial_checkpoint_path,
        model_revision=None,
        cache_dir=context.model_settings.cache_dir,
        local_files_only=True,
        device=device,
        transformers=transformers,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    total_development_steps = 1_722
    scheduler = _linear_warmup_scheduler(
        optimizer,
        total_steps=total_development_steps,
        warmup_steps=math.ceil(total_development_steps * settings.warmup_ratio),
        torch=torch,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    ordered = _ordered_train_rows(context.train_rows, settings.seed, 0)
    groups = bounded_optimizer_groups(
        ordered,
        batch_size=settings.train_batch_size,
        accumulation_steps=settings.gradient_accumulation_steps,
        max_optimizer_attempts=settings.fp16_diagnostic_max_optimizer_attempts,
    )
    progress = DiagnosticProgress()
    attempts: list[dict[str, Any]] = []
    hash_key = secrets.token_bytes(32)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for optimizer_attempt, row_batches in enumerate(
            groups,
            1,
        ):
            model.train()
            attempt, outcome = _diagnostic_attempt(
                optimizer_attempt=optimizer_attempt,
                row_batches=row_batches,
                collator=collator,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                device=device,
                torch=torch,
                max_grad_norm=settings.max_grad_norm,
                hash_key=hash_key,
            )
            if outcome is not None:
                progress.record(outcome)
            else:
                progress.optimizer_attempts += 1
            attempt["successful_optimizer_steps"] = progress.successful_optimizer_steps
            attempts.append(attempt)
            print(
                "fp16-diagnostic "
                f"attempt={optimizer_attempt:02d}/"
                f"{settings.fp16_diagnostic_max_optimizer_attempts} "
                f"executed={attempt['optimizer_step_executed']} "
                f"scale={attempt['scale_before']:.0f}->{attempt['scale_after']:.0f}"
            )
            if attempt.get("terminal_error") or not attempt["parameters_finite"]:
                break
        summary = _build_summary(
            settings=settings,
            context=context,
            attempts=attempts,
            progress=progress,
            gpu_name=str(torch.cuda.get_device_name(device)),
            bf16_supported=bool(torch.cuda.is_bf16_supported()),
        )
        write_diagnostic_report(settings, summary)
        return summary
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
