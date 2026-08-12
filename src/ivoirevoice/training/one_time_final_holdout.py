"""One-time, final-refit-only and aggregate-only holdout evaluation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any, cast

from ivoirevoice.evaluation.baseline import normalize_evaluation_text
from ivoirevoice.evaluation.metrics import edit_counts
from ivoirevoice.exceptions import ConfigError, IvoireVoiceError
from ivoirevoice.models.whisper import load_whisper_settings
from ivoirevoice.training.audit import AuditedDataset, ManifestRow, load_audited_dataset
from ivoirevoice.training.full_finetune import _require_cuda
from ivoirevoice.training.full_selection import (
    PRIVATE_SELECTION_FILENAME,
    validate_final_holdout_files,
)
from ivoirevoice.training.full_settings import (
    FullTrainingSettings,
    load_full_training_settings,
)
from ivoirevoice.training.pilot_finetune import (
    PilotCollator,
    _chunks,
    _load_model,
    _runtime_autocast,
    _seed_everything,
    _to_device,
)
from ivoirevoice.training.pilot_settings import PilotSettings
from ivoirevoice.training.whisper_finetune import (
    directory_sha256,
    free_disk_gib,
    git_provenance,
    load_json_object,
    require_matching_identity,
    selection_sha256,
    write_json_atomic,
)

CONFIG_PATH = Path("configs/experiments/full_finetune_whisper_tiny_dy.yaml")
APPROVAL_PATH = Path("configs/experiments/final_holdout_refit_approval.json")
CONFIRMATION_ENVIRONMENT_VARIABLE = "IVOIREVOICE_CONFIRM_FINAL_HOLDOUT"
CONFIRMATION_VALUE = "EVALUATE_FROZEN_REFIT_ONCE"
FINAL_MODEL_MANIFEST_FILENAME = "final_model_manifest.json"
DEVELOPMENT_DECISION_FILENAME = "development_decision.json"
TRAINER_STATE_FILENAME = "full_trainer_state.json"
FROZEN_FILENAME = "FROZEN.json"
PUBLIC_SELECTION_FILENAME = "full_selection.json"
LEGACY_RECEIPT_FILENAME = "final_holdout_evaluation_receipt.json"
LEGACY_METRICS_FILENAME = "final_holdout_metrics.json"
OUTPUT_DIRECTORY_NAME = "final_holdout"
STATE_FILENAME = "state.json"
PREFLIGHT_FILENAME = "preflight.json"
METRICS_FILENAME = "final_metrics.json"
SEALED = "SEALED"
IN_PROGRESS = "EVALUATION_IN_PROGRESS"
EVALUATED = "EVALUATED"
FAILED_AFTER_ACCESS = "EVALUATION_FAILED_AFTER_ACCESS"
EXPECTED_SUCCESSFUL_STEPS = 2_052
ALLOWED_AGGREGATE_FIELDS = frozenset(
    {
        "holdout_selection_sha256",
        "holdout_audio_count",
        "holdout_speaker_count",
        "wer",
        "cer",
        "rtf",
        "substitutions",
        "insertions",
        "deletions",
        "character_substitutions",
        "character_insertions",
        "character_deletions",
        "total_reference_words",
        "total_reference_characters",
        "exact_match_count",
        "total_audio_duration_seconds",
        "total_inference_seconds",
        "mean_latency_seconds",
        "final_loss",
    }
)


@dataclass(frozen=True, slots=True)
class GuardedFinalModel:
    """Verified identities available before the holdout access boundary."""

    final_checkpoint: Path
    final_checkpoint_name: str
    final_checkpoint_sha256: str
    final_step: int
    refit_run_id: str
    training_git_commit: str
    evaluation_git_commit: str
    config_sha256: str
    manifest_sha256: str
    refit_selection_sha256: str
    source_checkpoint_sha256: str
    historical_expected_holdout_speaker_count: int


@dataclass(slots=True)
class PreparedRuntime:
    """Final-model runtime created before any holdout data access."""

    model: Any
    processor: Any
    collator: Any
    torch: Any
    transformers: Any
    device: Any
    torch_version: str
    transformers_version: str
    cuda_version: str
    gpu_name: str


@dataclass(slots=True)
class StreamingASRAggregates:
    """In-memory aggregate counters that never retain sample text or predictions."""

    audio_count: int = 0
    speaker_tokens: set[str] = field(default_factory=set)
    substitutions: int = 0
    insertions: int = 0
    deletions: int = 0
    character_substitutions: int = 0
    character_insertions: int = 0
    character_deletions: int = 0
    total_reference_words: int = 0
    total_reference_characters: int = 0
    exact_match_count: int = 0
    total_audio_duration_seconds: float = 0.0
    total_inference_seconds: float = 0.0
    loss_sum: float = 0.0
    loss_count: int = 0

    def add(
        self,
        *,
        reference: str,
        hypothesis: str,
        speaker_token: str,
        audio_duration_seconds: float,
        inference_seconds: float,
        loss: float,
    ) -> None:
        """Consume one temporary result and retain only cumulative statistics."""

        reference_normalized = normalize_evaluation_text(
            reference,
            lowercase=True,
            remove_punctuation=True,
        )
        hypothesis_normalized = normalize_evaluation_text(
            hypothesis,
            lowercase=True,
            remove_punctuation=True,
        )
        reference_words = tuple(reference_normalized.split())
        hypothesis_words = tuple(hypothesis_normalized.split())
        reference_characters = tuple(reference_normalized)
        hypothesis_characters = tuple(hypothesis_normalized)
        word_counts = edit_counts(reference_words, hypothesis_words)
        character_counts = edit_counts(reference_characters, hypothesis_characters)
        numeric = (audio_duration_seconds, inference_seconds, loss)
        if not all(math.isfinite(value) for value in numeric):
            raise ConfigError("Une valeur agrégée final_holdout n'est pas finie.")
        if audio_duration_seconds <= 0 or inference_seconds < 0:
            raise ConfigError("Une durée final_holdout agrégée est invalide.")
        if not reference_words or not reference_characters:
            raise ConfigError("Une référence final_holdout normalisée est vide.")
        self.audio_count += 1
        self.speaker_tokens.add(speaker_token)
        self.substitutions += word_counts.substitutions
        self.insertions += word_counts.insertions
        self.deletions += word_counts.deletions
        self.character_substitutions += character_counts.substitutions
        self.character_insertions += character_counts.insertions
        self.character_deletions += character_counts.deletions
        self.total_reference_words += len(reference_words)
        self.total_reference_characters += len(reference_characters)
        self.exact_match_count += reference_normalized == hypothesis_normalized
        self.total_audio_duration_seconds += audio_duration_seconds
        self.total_inference_seconds += inference_seconds
        self.loss_sum += loss
        self.loss_count += 1

    def result(self, *, selection_sha256: str) -> dict[str, Any]:
        """Return the privacy-safe aggregate result and no speaker tokens."""

        if self.audio_count <= 0 or self.loss_count != self.audio_count:
            raise ConfigError("L'agrégation final_holdout est vide ou incomplète.")
        word_errors = self.substitutions + self.insertions + self.deletions
        character_errors = (
            self.character_substitutions + self.character_insertions + self.character_deletions
        )
        speaker_count = len(self.speaker_tokens)
        self.speaker_tokens.clear()
        return {
            "holdout_selection_sha256": selection_sha256,
            "holdout_audio_count": self.audio_count,
            "holdout_speaker_count": speaker_count,
            "wer": word_errors / self.total_reference_words,
            "cer": character_errors / self.total_reference_characters,
            "rtf": self.total_inference_seconds / self.total_audio_duration_seconds,
            "substitutions": self.substitutions,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "character_substitutions": self.character_substitutions,
            "character_insertions": self.character_insertions,
            "character_deletions": self.character_deletions,
            "total_reference_words": self.total_reference_words,
            "total_reference_characters": self.total_reference_characters,
            "exact_match_count": self.exact_match_count,
            "total_audio_duration_seconds": self.total_audio_duration_seconds,
            "total_inference_seconds": self.total_inference_seconds,
            "mean_latency_seconds": self.total_inference_seconds / self.audio_count,
            "final_loss": self.loss_sum / self.loss_count,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ConfigError(f"Impossible de calculer l'empreinte de {path.name}.") from exc
    return digest.hexdigest()


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ConfigError(f"{label} doit être un objet JSON.")
    return cast(dict[str, Any], dict(value))


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{label} doit être un entier positif.")
    return value


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _effective_approval_path(path: Path | None) -> Path:
    result = path or APPROVAL_PATH
    return result if result.is_absolute() else _root() / result


def _output_directory(settings: FullTrainingSettings) -> Path:
    return settings.artifact_output_directory / OUTPUT_DIRECTORY_NAME


def _state_path(settings: FullTrainingSettings) -> Path:
    return _output_directory(settings) / STATE_FILENAME


def _identity(guard: GuardedFinalModel) -> dict[str, Any]:
    return {
        "final_checkpoint_name": guard.final_checkpoint_name,
        "final_checkpoint_sha256": guard.final_checkpoint_sha256,
        "final_model_step": guard.final_step,
        "refit_run_id": guard.refit_run_id,
        "training_git_commit": guard.training_git_commit,
        "evaluation_git_commit": guard.evaluation_git_commit,
        "config_sha256": guard.config_sha256,
        "manifest_sha256": guard.manifest_sha256,
        "refit_selection_sha256": guard.refit_selection_sha256,
        "source_checkpoint_sha256": guard.source_checkpoint_sha256,
        "historical_expected_holdout_speaker_count": (
            guard.historical_expected_holdout_speaker_count
        ),
    }


def _require_no_legacy_evaluation(settings: FullTrainingSettings) -> None:
    legacy_paths = (
        settings.artifact_output_directory / LEGACY_RECEIPT_FILENAME,
        settings.artifact_output_directory / LEGACY_METRICS_FILENAME,
    )
    if any(path.exists() for path in legacy_paths):
        raise ConfigError("Une activité final_holdout legacy existe déjà.")
    legacy_directory = settings.artifact_output_directory / "final_evaluation"
    if legacy_directory.is_dir() and any(legacy_directory.iterdir()):
        raise ConfigError("Un cache final_holdout legacy existe déjà.")


def _load_guarded_final_model(
    settings: FullTrainingSettings,
    *,
    approval_path: Path | None = None,
    git_reader: Callable[[Path], tuple[str, bool]] = git_provenance,
    directory_hasher: Callable[[Path], str] = directory_sha256,
) -> GuardedFinalModel:
    """Validate every immutable identity without reading holdout data."""

    evaluation_commit, clean = git_reader(_root())
    if not clean:
        raise ConfigError("Le final_holdout exige un worktree Git propre.")
    config_sha256 = _sha256_file(settings.config_path)
    approval = load_json_object(_effective_approval_path(approval_path))
    if (
        approval.get("schema_version") != 1
        or approval.get("status") != "approved_for_one_time_final_holdout"
        or approval.get("remaining_model_decisions") is not False
    ):
        raise ConfigError("L'autorisation one-time final_holdout est invalide.")
    expected_approval = {
        "config_sha256": config_sha256,
        "manifest_sha256": settings.expected_manifest_sha256,
        "source_checkpoint_name": settings.expected_initial_checkpoint_name,
        "successful_optimizer_steps": EXPECTED_SUCCESSFUL_STEPS,
        "refit_audio_count": settings.refit_audio_count,
        "refit_speaker_count": settings.refit_speaker_count,
        "final_holdout_count": settings.final_holdout_count,
        "historical_pilot_count": settings.historical_pilot_test_count,
        "final_holdout_evaluation_count": 0,
        "final_holdout_accessed": False,
        "final_holdout_decoded": False,
        "remaining_model_decisions": False,
    }
    require_matching_identity(
        approval,
        expected_approval,
        description="L'autorisation one-time final_holdout",
    )
    final_manifest = load_json_object(
        settings.artifact_output_directory / FINAL_MODEL_MANIFEST_FILENAME
    )
    final_checkpoint_name = str(approval.get("final_checkpoint_name", ""))
    final_step = _positive_int(
        approval.get("successful_optimizer_steps"),
        "approval.successful_optimizer_steps",
    )
    if final_checkpoint_name != f"checkpoint-{final_step:06d}":
        raise ConfigError("Le checkpoint final approuvé ne correspond pas au step.")
    final_checkpoint = settings.refit_checkpoint_directory / final_checkpoint_name
    final_hash = directory_hasher(final_checkpoint)
    expected_manifest = {
        "status": "frozen",
        "code_commit": approval.get("training_git_commit"),
        "config_sha256": config_sha256,
        "manifest_sha256": settings.expected_manifest_sha256,
        "refit_selection_sha256": approval.get("refit_selection_sha256"),
        "initial_checkpoint_name": settings.expected_initial_checkpoint_name,
        "initial_checkpoint_sha256": approval.get("source_checkpoint_sha256"),
        "train_audio_count": settings.refit_audio_count,
        "train_speaker_count": settings.refit_speaker_count,
        "optimizer_steps": EXPECTED_SUCCESSFUL_STEPS,
        "successful_optimizer_steps": EXPECTED_SUCCESSFUL_STEPS,
        "final_checkpoint_name": final_checkpoint_name,
        "final_checkpoint_sha256": approval.get("final_checkpoint_sha256"),
        "historical_test_used": False,
        "final_holdout_used": False,
        "publication_allowed": False,
    }
    require_matching_identity(
        final_manifest,
        expected_manifest,
        description="Le manifeste du refit final",
    )
    if final_hash != approval.get("final_checkpoint_sha256"):
        raise ConfigError("Le hash du modèle refit final ne correspond pas.")
    if not (final_checkpoint / "model.safetensors").is_file():
        raise ConfigError("Les poids sûrs du modèle final sont absents.")
    trainer_state = load_json_object(final_checkpoint / TRAINER_STATE_FILENAME)
    expected_state = {
        "stage": "refit",
        "run_id": approval.get("refit_run_id"),
        "code_commit": approval.get("training_git_commit"),
        "config_sha256": config_sha256,
        "selection_sha256": approval.get("refit_selection_sha256"),
        "initial_checkpoint_sha256": approval.get("source_checkpoint_sha256"),
        "global_step": EXPECTED_SUCCESSFUL_STEPS,
        "successful_optimizer_steps": EXPECTED_SUCCESSFUL_STEPS,
        "total_steps": EXPECTED_SUCCESSFUL_STEPS,
        "completed": True,
        "optimizer_initialized_from": "fresh",
        "scheduler_initialized_from": "fresh",
        "pilot_optimizer_state_loaded": False,
        "precision": "fp16",
    }
    require_matching_identity(
        trainer_state,
        expected_state,
        description="L'état final du refit",
    )
    frozen = load_json_object(final_checkpoint / FROZEN_FILENAME)
    require_matching_identity(
        frozen,
        {
            "status": "immutable_final_checkpoint",
            "run_id": approval.get("refit_run_id"),
            "optimizer_steps": EXPECTED_SUCCESSFUL_STEPS,
        },
        description="Le scellé du checkpoint final",
    )
    decision = load_json_object(settings.artifact_output_directory / DEVELOPMENT_DECISION_FILENAME)
    require_matching_identity(
        decision,
        {
            "status": "frozen",
            "development_selection_finalized": True,
            "finalization_git_commit": approval.get("training_git_commit"),
            "config_sha256": config_sha256,
            "manifest_sha256": settings.expected_manifest_sha256,
            "refit_selection_sha256": approval.get("refit_selection_sha256"),
            "initial_checkpoint_sha256": approval.get("source_checkpoint_sha256"),
            "refit_step_budget": EXPECTED_SUCCESSFUL_STEPS,
            "final_holdout_used": False,
        },
        description="La décision de développement finalisée",
    )
    selection = load_json_object(settings.shareable_output_directory / PUBLIC_SELECTION_FILENAME)
    final_holdout = _mapping(selection.get("final_holdout"), "Le scellé final_holdout")
    historical = _mapping(selection.get("historical_test"), "Le pilote historique")
    require_matching_identity(
        selection,
        {
            "manifest_sha256": settings.expected_manifest_sha256,
        },
        description="Le résumé de sélection",
    )
    require_matching_identity(
        final_holdout,
        {"audio_count": settings.final_holdout_count, "decoded": False},
        description="Le final_holdout scellé",
    )
    require_matching_identity(
        historical,
        {"audio_count": settings.historical_pilot_test_count, "decoded": False},
        description="Le pilote historique exclu",
    )
    _require_no_legacy_evaluation(settings)
    return GuardedFinalModel(
        final_checkpoint=final_checkpoint,
        final_checkpoint_name=final_checkpoint_name,
        final_checkpoint_sha256=final_hash,
        final_step=final_step,
        refit_run_id=str(approval["refit_run_id"]),
        training_git_commit=str(approval["training_git_commit"]),
        evaluation_git_commit=evaluation_commit,
        config_sha256=config_sha256,
        manifest_sha256=settings.expected_manifest_sha256,
        refit_selection_sha256=str(approval["refit_selection_sha256"]),
        source_checkpoint_sha256=str(approval["source_checkpoint_sha256"]),
        historical_expected_holdout_speaker_count=_positive_int(
            approval.get("historical_expected_holdout_speaker_count"),
            "approval.historical_expected_holdout_speaker_count",
        ),
    )


def _load_state(settings: FullTrainingSettings) -> dict[str, Any] | None:
    path = _state_path(settings)
    return load_json_object(path) if path.is_file() else None


def _require_state_identity(
    state: Mapping[str, Any],
    guard: GuardedFinalModel,
) -> None:
    require_matching_identity(
        state,
        {"schema_version": 1, **_identity(guard)},
        description="L'état one-time final_holdout",
    )


def _sealed_state(guard: GuardedFinalModel) -> dict[str, Any]:
    return {
        "schema_version": 1,
        **_identity(guard),
        "status": SEALED,
        "evaluation_count": 0,
        "final_holdout_accessed": False,
        "final_holdout_decoded": False,
    }


def _probe_environment(settings: FullTrainingSettings) -> dict[str, Any]:
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    device = _require_cuda(torch)
    free_gib = free_disk_gib(settings.artifact_output_directory)
    if free_gib < settings.minimum_free_disk_gib:
        raise ConfigError("Espace disque insuffisant pour l'évaluation finale.")
    return {
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
        "cuda_version": str(torch.version.cuda),
        "gpu_name": str(torch.cuda.get_device_name(device)),
        "free_disk_gib": free_gib,
    }


def run_final_holdout_preflight(
    settings: FullTrainingSettings,
    *,
    approval_path: Path | None = None,
    git_reader: Callable[[Path], tuple[str, bool]] = git_provenance,
    directory_hasher: Callable[[Path], str] = directory_sha256,
    environment_probe: Callable[[FullTrainingSettings], dict[str, Any]] = (_probe_environment),
) -> dict[str, Any]:
    """Seal identities and runtime readiness without constructing holdout rows."""

    guard = _load_guarded_final_model(
        settings,
        approval_path=approval_path,
        git_reader=git_reader,
        directory_hasher=directory_hasher,
    )
    existing = _load_state(settings)
    if existing is not None:
        _require_state_identity(existing, guard)
        if existing.get("status") != SEALED or existing.get("evaluation_count") != 0:
            raise ConfigError("Le final_holdout n'est plus dans l'état SEALED.")
    runtime = environment_probe(settings)
    state = existing or _sealed_state(guard)
    output = _output_directory(settings)
    output.mkdir(parents=True, exist_ok=True)
    if existing is None:
        write_json_atomic(_state_path(settings), state)
    report_path = output / PREFLIGHT_FILENAME
    expected_report = {
        "schema_version": 1,
        **_identity(guard),
        "status": "READY_FOR_ONE_TIME_FINAL_HOLDOUT",
        "state": SEALED,
        "evaluation_count": 0,
        "final_holdout_accessed": False,
        "final_holdout_decoded": False,
        "final_holdout_count": settings.final_holdout_count,
        "remaining_model_decisions": False,
        "environment": runtime,
    }
    if report_path.is_file():
        existing_report = load_json_object(report_path)
        require_matching_identity(
            existing_report,
            {key: value for key, value in expected_report.items() if key != "environment"},
            description="Le preflight final_holdout existant",
        )
    write_json_atomic(report_path, expected_report)
    return expected_report


def _prepare_runtime(
    settings: FullTrainingSettings,
    guard: GuardedFinalModel,
) -> PreparedRuntime:
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    device = _require_cuda(torch)
    _seed_everything(settings.seed, torch)
    model_settings = load_whisper_settings(_root() / settings.model_config_path)
    if (
        model_settings.model_id != settings.expected_model_id
        or model_settings.model_revision != settings.expected_model_revision
        or model_settings.task != "transcribe"
        or model_settings.language is not None
    ):
        raise ConfigError("La configuration Whisper finale a changé.")
    processor = transformers.WhisperProcessor.from_pretrained(
        guard.final_checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )
    processor.tokenizer.set_prefix_tokens(task="transcribe")
    collator_settings = cast(
        PilotSettings,
        SimpleNamespace(
            train_split=settings.forbidden_split,
            validation_split=settings.forbidden_split,
            dataset_root=settings.dataset_root,
            max_audio_seconds=settings.max_audio_seconds,
        ),
    )
    collator = PilotCollator(
        collator_settings,
        processor,
        model_settings.expected_sampling_rate_hz,
        torch,
    )
    model = _load_model(
        source=guard.final_checkpoint,
        model_revision=None,
        cache_dir=model_settings.cache_dir,
        local_files_only=True,
        device=device,
        transformers=transformers,
    )
    model.config.use_cache = True
    return PreparedRuntime(
        model=model,
        processor=processor,
        collator=collator,
        torch=torch,
        transformers=transformers,
        device=device,
        torch_version=str(torch.__version__),
        transformers_version=str(transformers.__version__),
        cuda_version=str(torch.version.cuda),
        gpu_name=str(torch.cuda.get_device_name(device)),
    )


def _release_runtime(runtime: PreparedRuntime) -> None:
    del runtime.model
    gc.collect()
    runtime.torch.cuda.empty_cache()


def _private_identifiers(
    value: object,
    *,
    field_name: str,
    expected_count: int,
) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or len(value) != expected_count
        or not all(isinstance(item, str) and item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ConfigError(f"La sélection privée {field_name} est invalide.")
    return tuple(cast(list[str], value))


def _select_frozen_final_holdout(
    dataset: AuditedDataset,
    settings: FullTrainingSettings,
) -> tuple[tuple[ManifestRow, ...], str]:
    """Resolve the prior frozen selection after the access boundary, without audio I/O."""

    if (
        dataset.manifest_sha256 != settings.expected_manifest_sha256
        or dataset.dataset_version != settings.expected_dataset_version
    ):
        raise ConfigError("Le dataset final_holdout ne correspond plus au corpus gelé.")
    frozen = load_json_object(settings.artifact_output_directory / PRIVATE_SELECTION_FILENAME)
    require_matching_identity(
        frozen,
        {
            "schema_version": 1,
            "manifest_sha256": settings.expected_manifest_sha256,
        },
        description="La sélection privée gelée",
    )
    final_ids = _private_identifiers(
        frozen.get("final_holdout_audio_ids"),
        field_name="final_holdout_audio_ids",
        expected_count=settings.final_holdout_count,
    )
    historical_ids = _private_identifiers(
        frozen.get("historical_test_audio_ids"),
        field_name="historical_test_audio_ids",
        expected_count=settings.historical_pilot_test_count,
    )
    if set(final_ids) & set(historical_ids):
        raise ConfigError("Le final_holdout recouvre le pilote historique.")
    test_rows = tuple(row for row in dataset.rows if row.split == settings.forbidden_split)
    test_by_id = {row.utterance_id: row for row in test_rows}
    if (
        len(test_by_id) != len(test_rows)
        or len(test_rows) != settings.final_holdout_count + settings.historical_pilot_test_count
        or set(final_ids) | set(historical_ids) != set(test_by_id)
    ):
        raise ConfigError("La partition test ne correspond plus à la sélection gelée.")
    rows = tuple(test_by_id[identifier] for identifier in final_ids)
    observed_hash = selection_sha256(
        {
            "final_holdout": tuple((row.utterance_id, row.audio_sha256) for row in rows),
        }
    )
    expected_hash = frozen.get("final_holdout_selection_sha256")
    if (
        not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or observed_hash != expected_hash
    ):
        raise ConfigError("L'empreinte du final_holdout gelé ne correspond plus.")
    return rows, observed_hash


def _evaluate_final_refit_aggregate(
    settings: FullTrainingSettings,
    runtime: PreparedRuntime,
) -> dict[str, Any]:
    """Open the holdout after sealing and retain cumulative metrics only."""

    dataset = load_audited_dataset(settings)
    rows, holdout_selection_sha256 = _select_frozen_final_holdout(dataset, settings)
    validate_final_holdout_files(rows, settings)
    aggregates = StreamingASRAggregates()
    runtime.model.eval()
    autocast_enabled, autocast_dtype = _runtime_autocast(
        runtime.torch,
        str(runtime.device.type),
        settings.fp16,
    )
    with runtime.torch.inference_mode():
        for row_batch in _chunks(rows, settings.eval_batch_size):
            batch = _to_device(runtime.collator(row_batch), runtime.device)
            if any(
                tensor.device != next(runtime.model.parameters()).device
                for tensor in batch.values()
            ):
                raise ConfigError("Le modèle final et ses tensors sont sur des devices différents.")
            with runtime.torch.autocast(
                device_type=str(runtime.device.type),
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                output = runtime.model(**batch)
            batch_loss = float(output.loss.detach().cpu())
            if not math.isfinite(batch_loss):
                raise ConfigError("La loss final_holdout n'est pas finie.")
            runtime.torch.cuda.synchronize(runtime.device)
            started = perf_counter()
            generated = runtime.model.generate(
                input_features=batch["input_features"],
                attention_mask=batch["attention_mask"],
                task="transcribe",
                do_sample=False,
                max_new_tokens=128,
            )
            runtime.torch.cuda.synchronize(runtime.device)
            elapsed = perf_counter() - started
            decoded = runtime.processor.batch_decode(
                generated,
                skip_special_tokens=True,
            )
            item_latency = elapsed / len(row_batch)
            for row, hypothesis in zip(row_batch, decoded, strict=True):
                aggregates.add(
                    reference=row.target_text,
                    hypothesis=str(hypothesis).strip(),
                    speaker_token=row.speaker_id,
                    audio_duration_seconds=row.duration_seconds,
                    inference_seconds=item_latency,
                    loss=batch_loss,
                )
            del decoded, generated, output, batch
    return aggregates.result(selection_sha256=holdout_selection_sha256)


def _validate_aggregate_result(
    result: Mapping[str, Any],
    settings: FullTrainingSettings,
) -> dict[str, Any]:
    if set(result) != set(ALLOWED_AGGREGATE_FIELDS):
        raise ConfigError("Le résultat final contient des champs absents ou non agrégés.")
    payload = dict(result)
    if payload.get("holdout_audio_count") != settings.final_holdout_count:
        raise ConfigError("L'évaluation ne couvre pas tout le final_holdout.")
    _positive_int(payload.get("holdout_speaker_count"), "holdout_speaker_count")
    integer_fields = (
        "substitutions",
        "insertions",
        "deletions",
        "character_substitutions",
        "character_insertions",
        "character_deletions",
        "total_reference_words",
        "total_reference_characters",
        "exact_match_count",
    )
    for field_name in integer_fields:
        value = payload.get(field_name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ConfigError(f"La métrique agrégée {field_name} est invalide.")
    if payload["total_reference_words"] <= 0 or payload["total_reference_characters"] <= 0:
        raise ConfigError("Les dénominateurs WER/CER sont vides.")
    float_fields = (
        "wer",
        "cer",
        "rtf",
        "total_audio_duration_seconds",
        "total_inference_seconds",
        "mean_latency_seconds",
        "final_loss",
    )
    for field_name in float_fields:
        value = payload.get(field_name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ConfigError(f"La métrique agrégée {field_name} est invalide.")
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ConfigError(f"La métrique agrégée {field_name} n'est pas finie.")
    expected_wer = (
        payload["substitutions"] + payload["insertions"] + payload["deletions"]
    ) / payload["total_reference_words"]
    expected_cer = (
        payload["character_substitutions"]
        + payload["character_insertions"]
        + payload["character_deletions"]
    ) / payload["total_reference_characters"]
    if not math.isclose(float(payload["wer"]), expected_wer, abs_tol=1e-15):
        raise ConfigError("Le WER agrégé est incohérent.")
    if not math.isclose(float(payload["cer"]), expected_cer, abs_tol=1e-15):
        raise ConfigError("Le CER agrégé est incohérent.")
    selection_hash = payload.get("holdout_selection_sha256")
    if not isinstance(selection_hash, str) or len(selection_hash) != 64:
        raise ConfigError("L'empreinte de sélection final_holdout est invalide.")
    return payload


def _require_preflight(
    settings: FullTrainingSettings,
    guard: GuardedFinalModel,
) -> dict[str, Any]:
    path = _output_directory(settings) / PREFLIGHT_FILENAME
    report = load_json_object(path)
    require_matching_identity(
        report,
        {
            "schema_version": 1,
            **_identity(guard),
            "status": "READY_FOR_ONE_TIME_FINAL_HOLDOUT",
            "state": SEALED,
            "evaluation_count": 0,
            "final_holdout_accessed": False,
            "final_holdout_decoded": False,
            "final_holdout_count": settings.final_holdout_count,
            "remaining_model_decisions": False,
        },
        description="Le preflight one-time final_holdout",
    )
    return report


def run_one_time_final_holdout(
    settings: FullTrainingSettings,
    *,
    approval_path: Path | None = None,
    git_reader: Callable[[Path], tuple[str, bool]] = git_provenance,
    directory_hasher: Callable[[Path], str] = directory_sha256,
    runtime_preparer: Callable[
        [FullTrainingSettings, GuardedFinalModel], PreparedRuntime
    ] = _prepare_runtime,
    aggregate_evaluator: Callable[
        [FullTrainingSettings, PreparedRuntime], dict[str, Any]
    ] = _evaluate_final_refit_aggregate,
    runtime_releaser: Callable[[PreparedRuntime], None] = _release_runtime,
) -> dict[str, Any]:
    """Evaluate the frozen refit once and permanently consume the access event."""

    if os.getenv(CONFIRMATION_ENVIRONMENT_VARIABLE) != CONFIRMATION_VALUE:
        raise ConfigError("La confirmation one-time final_holdout exacte est absente.")
    guard = _load_guarded_final_model(
        settings,
        approval_path=approval_path,
        git_reader=git_reader,
        directory_hasher=directory_hasher,
    )
    state = _load_state(settings)
    if state is None:
        raise ConfigError("Le preflight final_holdout officiel est absent.")
    _require_state_identity(state, guard)
    status = state.get("status")
    if status == EVALUATED:
        return {
            "status": "FINAL_HOLDOUT_ALREADY_EVALUATED",
            "evaluation_count": state.get("evaluation_count"),
        }
    if status in {IN_PROGRESS, FAILED_AFTER_ACCESS}:
        raise ConfigError("L'accès final_holdout est déjà consommé et non relançable.")
    if (
        status != SEALED
        or state.get("evaluation_count") != 0
        or state.get("final_holdout_accessed") is not False
        or state.get("final_holdout_decoded") is not False
    ):
        raise ConfigError("L'état SEALED du final_holdout est incohérent.")
    _require_preflight(settings, guard)
    runtime = runtime_preparer(settings, guard)
    in_progress = {
        **state,
        "status": IN_PROGRESS,
        "evaluation_count": 1,
        "final_holdout_accessed": True,
        "final_holdout_decoded": False,
        "access_started_at": datetime.now(UTC).isoformat(),
    }
    write_json_atomic(_state_path(settings), in_progress)
    try:
        raw_result = aggregate_evaluator(settings, runtime)
        aggregate = _validate_aggregate_result(raw_result, settings)
        if directory_hasher(guard.final_checkpoint) != guard.final_checkpoint_sha256:
            raise ConfigError("Le modèle final a changé pendant l'évaluation.")
        timestamp = datetime.now(UTC).isoformat()
        report = {
            "schema_version": 1,
            "status": "evaluated",
            "evaluation_count": 1,
            **_identity(guard),
            **aggregate,
            "final_model_hash": guard.final_checkpoint_sha256,
            "config_hash": guard.config_sha256,
            "git_commit": guard.evaluation_git_commit,
            "holdout_speaker_count_historical_expectation": (
                guard.historical_expected_holdout_speaker_count
            ),
            "holdout_speaker_count_anomaly": (
                aggregate["holdout_speaker_count"]
                != guard.historical_expected_holdout_speaker_count
            ),
            "evaluation_timestamp": timestamp,
            "torch_version": runtime.torch_version,
            "transformers_version": runtime.transformers_version,
            "cuda_version": runtime.cuda_version,
            "gpu_name": runtime.gpu_name,
            "no_retraining_after_holdout": True,
            "remaining_model_decisions": False,
            "privacy": {
                "contains_audio": False,
                "contains_paths": False,
                "contains_transcriptions": False,
                "contains_predictions": False,
                "contains_sample_identifiers": False,
                "contains_speaker_identifiers": False,
            },
        }
        metrics_path = _output_directory(settings) / METRICS_FILENAME
        if metrics_path.exists():
            raise ConfigError("Un rapport final_holdout existe déjà.")
        write_json_atomic(metrics_path, report)
        evaluated_state = {
            **in_progress,
            "status": EVALUATED,
            "final_holdout_decoded": True,
            "evaluation_completed_at": timestamp,
            "aggregate_report_sha256": _sha256_file(metrics_path),
        }
        write_json_atomic(_state_path(settings), evaluated_state)
        return report
    except BaseException as exc:
        failed_state = {
            **in_progress,
            "status": FAILED_AFTER_ACCESS,
            "final_holdout_decoded": True,
            "failure_timestamp": datetime.now(UTC).isoformat(),
            "failure_type": type(exc).__name__,
        }
        write_json_atomic(_state_path(settings), failed_state)
        raise
    finally:
        runtime_releaser(runtime)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Évaluation one-time du seul modèle refit gelé.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument(
        "--stage",
        choices=("preflight", "final-holdout-evaluate-refit-once"),
        required=True,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        settings = load_full_training_settings(args.config)
        result = (
            run_final_holdout_preflight(settings)
            if args.stage == "preflight"
            else run_one_time_final_holdout(settings)
        )
    except IvoireVoiceError as exc:
        print(f"ERREUR: {exc}")
        return 1
    print(f"Stage {args.stage} : {result['status']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
