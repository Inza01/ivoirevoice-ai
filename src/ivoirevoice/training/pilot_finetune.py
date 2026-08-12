"""Phase 4C Whisper Tiny pilot training and same-validation comparison."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from ivoirevoice.evaluation.baseline import normalize_evaluation_text
from ivoirevoice.evaluation.metrics import (
    ScoredItem,
    compute_evaluation_metrics,
    edit_counts,
)
from ivoirevoice.exceptions import ConfigError, IvoireVoiceError
from ivoirevoice.models.whisper import load_whisper_settings, runtime_labels
from ivoirevoice.training.audit import AuditedDataset, ManifestRow, load_audited_dataset
from ivoirevoice.training.pilot_selection import (
    PilotSelection,
    build_pilot_selection,
    write_selection_report,
)
from ivoirevoice.training.pilot_settings import PilotSettings, load_pilot_settings
from ivoirevoice.training.whisper_finetune import (
    accumulation_group_sizes,
    checkpoint_directories,
    free_disk_gib,
    latest_complete_checkpoint,
    save_checkpoint_atomic,
)

BASELINE_CACHE_FILENAME = "baseline_validation_private.json"
TRAINING_STATE_FILENAME = "trainer_state.json"
PILOT_METRICS_FILENAME = "pilot_training_metrics.json"
HISTORY_FILENAME = "pilot_training_history.csv"
CURVES_FILENAME = "pilot_training_curves.png"
PREDICTIONS_FILENAME = "pilot_validation_predictions.csv"
COMPARISON_JSON_FILENAME = "pilot_comparison.json"
COMPARISON_MARKDOWN_FILENAME = "pilot_comparison.md"
RESOURCE_ESTIMATE_FILENAME = "pilot_resource_estimate.md"


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Metrics and private predictions for one fixed validation pass."""

    metrics: dict[str, Any]
    predictions: tuple[dict[str, Any], ...]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        content = json.dumps(
            dict(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Métriques non sérialisables dans {path.name} : {exc}") from exc
    temporary.write_text(content + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Impossible de lire {path.name} : {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"{path.name} doit contenir un objet JSON.")
    return cast(dict[str, Any], value)


def _seed_everything(seed: int, torch: Any) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    numpy = importlib.import_module("numpy")
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def _load_audio(path: Path, expected_rate: int, maximum_seconds: float) -> Any:
    soundfile = importlib.import_module("soundfile")
    numpy = importlib.import_module("numpy")
    try:
        audio, sample_rate = soundfile.read(
            str(path),
            dtype="float32",
            always_2d=False,
        )
    except (OSError, RuntimeError) as exc:
        raise ConfigError(f"Décodage audio impossible pour {path.name} : {exc}") from exc
    array = numpy.asarray(audio, dtype="float32")
    if int(sample_rate) != expected_rate:
        raise ConfigError(f"{path.name} n'est pas décodé à {expected_rate} Hz.")
    if array.ndim != 1 or not array.size:
        raise ConfigError(f"{path.name} n'est pas un signal mono non vide.")
    if array.size / expected_rate > maximum_seconds:
        raise ConfigError(f"{path.name} dépasse la durée maximale du pilote.")
    if not numpy.isfinite(array).all():
        raise ConfigError(f"{path.name} contient NaN ou Inf.")
    return array


class PilotCollator:
    """Decode private audio on demand and create Whisper inputs."""

    def __init__(
        self,
        settings: PilotSettings,
        processor: Any,
        expected_rate: int,
        torch: Any,
    ) -> None:
        self.settings = settings
        self.processor = processor
        self.expected_rate = expected_rate
        self.torch = torch

    def __call__(self, rows: Sequence[ManifestRow]) -> dict[str, Any]:
        feature_items: list[Any] = []
        attention_items: list[Any] = []
        label_items: list[Any] = []
        for row in rows:
            if row.split not in {
                self.settings.train_split,
                self.settings.validation_split,
            }:
                raise ConfigError("Le collator refuse tout audio hors train/validation.")
            audio = _load_audio(
                self.settings.dataset_root / row.audio_path,
                self.expected_rate,
                self.settings.max_audio_seconds,
            )
            encoded = self.processor.feature_extractor(
                audio,
                sampling_rate=self.expected_rate,
                return_tensors="pt",
                return_attention_mask=True,
            )
            features = encoded.input_features[0]
            attention = encoded.attention_mask[0]
            labels = self.processor.tokenizer(
                row.target_text,
                return_tensors="pt",
            ).input_ids[0]
            if int(labels.numel()) < 2:
                raise ConfigError("Un label pilote est vide après tokenisation.")
            if not bool(features.isfinite().all()) or not bool(attention.isfinite().all()):
                raise ConfigError("Les features ou masques contiennent NaN/Inf.")
            feature_items.append(features)
            attention_items.append(attention)
            label_items.append(labels)
        maximum = max(int(labels.numel()) for labels in label_items)
        padded_labels = self.torch.full(
            (len(rows), maximum),
            -100,
            dtype=self.torch.long,
        )
        for index, labels in enumerate(label_items):
            padded_labels[index, : labels.numel()] = labels
        decoder_start_token_id = self.processor.tokenizer.bos_token_id
        if decoder_start_token_id is None:
            raise ConfigError("Le tokenizer Whisper ne définit pas de token de début.")
        if bool((padded_labels[:, 0] == decoder_start_token_id).all()):
            padded_labels = padded_labels[:, 1:]
        return {
            "input_features": self.torch.stack(feature_items),
            "attention_mask": self.torch.stack(attention_items),
            "labels": padded_labels,
        }


def _chunks(rows: Sequence[ManifestRow], size: int) -> Sequence[Sequence[ManifestRow]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def _to_device(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {name: tensor.to(device, non_blocking=True) for name, tensor in batch.items()}


def _runtime_autocast(torch: Any, device_name: str, fp16: bool) -> tuple[bool, Any]:
    enabled = device_name == "cuda" and fp16
    return enabled, torch.float16 if enabled else torch.float32


def _evaluate_validation(
    *,
    model: Any,
    processor: Any,
    rows: Sequence[ManifestRow],
    collator: PilotCollator,
    batch_size: int,
    device: Any,
    torch: Any,
    fp16: bool,
) -> EvaluationResult:
    """Evaluate one model on the frozen validation subset without post-correction."""

    model.eval()
    autocast_enabled, autocast_dtype = _runtime_autocast(
        torch, str(device.type), fp16
    )
    scored: list[ScoredItem] = []
    predictions: list[dict[str, Any]] = []
    weighted_loss = 0.0
    total_rows = 0
    total_inference_seconds = 0.0
    with torch.inference_mode():
        for row_batch in _chunks(rows, batch_size):
            batch = _to_device(collator(row_batch), device)
            if any(
                tensor.device != next(model.parameters()).device
                for tensor in batch.values()
            ):
                raise ConfigError(
                    "Évaluation refusée : tensors et modèle sur des devices différents."
                )
            with torch.autocast(
                device_type=str(device.type),
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                output = model(**batch)
            batch_loss = float(output.loss.detach().cpu())
            if not math.isfinite(batch_loss):
                raise ConfigError("Validation loss non finie.")
            weighted_loss += batch_loss * len(row_batch)
            total_rows += len(row_batch)
            started = perf_counter()
            generated = model.generate(
                input_features=batch["input_features"],
                attention_mask=batch["attention_mask"],
                task="transcribe",
                do_sample=False,
                max_new_tokens=128,
            )
            if str(device.type) == "cuda":
                torch.cuda.synchronize(device)
            elapsed = perf_counter() - started
            total_inference_seconds += elapsed
            decoded = processor.batch_decode(generated, skip_special_tokens=True)
            amortized_latency = elapsed / len(row_batch)
            for row, prediction in zip(row_batch, decoded, strict=True):
                reference_normalized = normalize_evaluation_text(
                    row.target_text,
                    lowercase=True,
                    remove_punctuation=True,
                )
                prediction_text = str(prediction).strip()
                prediction_normalized = normalize_evaluation_text(
                    prediction_text,
                    lowercase=True,
                    remove_punctuation=True,
                )
                word_counts = edit_counts(
                    tuple(reference_normalized.split()),
                    tuple(prediction_normalized.split()),
                )
                character_counts = edit_counts(
                    tuple(reference_normalized),
                    tuple(prediction_normalized),
                )
                scored.append(
                    ScoredItem(
                        speaker_id=row.speaker_id,
                        reference_normalized=reference_normalized,
                        prediction_normalized=prediction_normalized,
                        audio_duration_seconds=row.duration_seconds,
                        processing_time_seconds=amortized_latency,
                    )
                )
                predictions.append(
                    {
                        "audio_id_anonymized": row.utterance_id,
                        "speaker_id_anonymized": row.speaker_id,
                        "text_raw": row.text_raw,
                        "text_no_tones": row.text_no_tones,
                        "target_text_mvp": row.target_text,
                        "prediction": prediction_text,
                        "word_substitutions": word_counts.substitutions,
                        "word_deletions": word_counts.deletions,
                        "word_insertions": word_counts.insertions,
                        "character_errors": character_counts.errors,
                    }
                )
    metrics = compute_evaluation_metrics(tuple(scored))
    metrics.update(
        {
            "validation_loss": weighted_loss / total_rows,
            "inference_seconds": total_inference_seconds,
            "latency_definition": "amortized_per_item_within_each_batch",
            "post_correction_used": False,
        }
    )
    return EvaluationResult(metrics=metrics, predictions=tuple(predictions))


def relative_reduction(baseline: float, adapted: float) -> float:
    """Return percentage error reduction, with a defined zero-baseline behavior."""

    if baseline == 0:
        return 0.0 if adapted == 0 else float("-inf")
    return (baseline - adapted) / baseline * 100.0


def comparison_metrics(
    baseline: Mapping[str, Any],
    adapted: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare the two models on exactly the same validation selection."""

    baseline_wer = float(baseline["wer_micro"])
    adapted_wer = float(adapted["wer_micro"])
    baseline_cer = float(baseline["cer_micro"])
    adapted_cer = float(adapted["cer_micro"])
    return {
        "baseline": dict(baseline),
        "adapted": dict(adapted),
        "wer_absolute_reduction": baseline_wer - adapted_wer,
        "wer_relative_reduction_percent": relative_reduction(
            baseline_wer, adapted_wer
        ),
        "cer_absolute_reduction": baseline_cer - adapted_cer,
        "cer_relative_reduction_percent": relative_reduction(
            baseline_cer, adapted_cer
        ),
        "wer_improved": adapted_wer < baseline_wer,
        "cer_improved": adapted_cer < baseline_cer,
    }


def find_latest_checkpoint(directory: Path) -> Path | None:
    """Return the latest complete numeric checkpoint suitable for resume."""

    return latest_complete_checkpoint(
        directory,
        (
            TRAINING_STATE_FILENAME,
            "optimizer.pt",
            "scheduler.pt",
            "config.json",
        ),
    )


def _free_disk_gib(path: Path) -> float:
    return free_disk_gib(path)


def _check_disk(settings: PilotSettings) -> None:
    available = _free_disk_gib(settings.checkpoint_directory)
    if available < settings.minimum_free_disk_gib:
        raise ConfigError(
            f"Espace disque insuffisant : {available:.2f} GiB disponibles, "
            f"{settings.minimum_free_disk_gib:.2f} GiB requis."
        )


def _checkpoint_paths(directory: Path) -> list[Path]:
    return checkpoint_directories(directory)


def _save_checkpoint(
    *,
    settings: PilotSettings,
    model: Any,
    processor: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    state: Mapping[str, Any],
    torch: Any,
) -> Path:
    return save_checkpoint_atomic(
        directory=settings.checkpoint_directory,
        minimum_free_disk_gib=settings.minimum_free_disk_gib,
        save_total_limit=settings.save_total_limit,
        best_checkpoint_name=str(state["best_checkpoint_name"]),
        state_filename=TRAINING_STATE_FILENAME,
        model=model,
        processor=processor,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        state=state,
        torch=torch,
    )


def _linear_warmup_scheduler(
    optimizer: Any,
    *,
    total_steps: int,
    warmup_steps: int,
    torch: Any,
) -> Any:
    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)
        remaining = max(1, total_steps - warmup_steps)
        return max(0.0, (total_steps - step) / remaining)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _ordered_train_rows(
    rows: Sequence[ManifestRow],
    seed: int,
    epoch: int,
) -> list[ManifestRow]:
    ordered = list(rows)
    random.Random(seed + epoch).shuffle(ordered)
    return ordered


def _load_model(
    *,
    source: str | Path,
    model_revision: str | None,
    cache_dir: Path,
    local_files_only: bool,
    device: Any,
    transformers: Any,
) -> Any:
    arguments: dict[str, Any] = {
        "cache_dir": str(cache_dir),
        "local_files_only": local_files_only,
        "trust_remote_code": False,
        "attn_implementation": "eager",
    }
    if model_revision is not None:
        arguments["revision"] = model_revision
    model = transformers.WhisperForConditionalGeneration.from_pretrained(
        source,
        **arguments,
    ).to(device)
    if next(model.parameters()).device != device:
        raise ConfigError("Le modèle pilote n'est pas sur le device attendu.")
    return model


def _train_pilot(
    *,
    settings: PilotSettings,
    selection: PilotSelection,
    processor: Any,
    collator: PilotCollator,
    model_settings: Any,
    device: Any,
    torch: Any,
    transformers: Any,
) -> tuple[dict[str, Any], Path]:
    """Train at most two epochs, save best WER checkpoint and support resume."""

    settings.checkpoint_directory.mkdir(parents=True, exist_ok=True)
    latest = (
        find_latest_checkpoint(settings.checkpoint_directory)
        if settings.resume_from_checkpoint
        else None
    )
    if latest is None:
        model = _load_model(
            source=model_settings.model_id,
            model_revision=model_settings.model_revision,
            cache_dir=model_settings.cache_dir,
            local_files_only=model_settings.local_files_only,
            device=device,
            transformers=transformers,
        )
        state: dict[str, Any] = {
            "schema_version": 1,
            "selection_sha256": selection.selection_sha256,
            "global_step": 0,
            "micro_batches_completed": 0,
            "epoch": 0,
            "best_wer": None,
            "best_checkpoint_name": "",
            "early_stopping_bad_evaluations": 0,
            "early_stopping_triggered": False,
            "fp16_optimizer_steps_skipped": 0,
            "completed_run_reused": False,
            "log_history": [],
        }
    else:
        state = _load_json(latest / TRAINING_STATE_FILENAME)
        if state.get("selection_sha256") != selection.selection_sha256:
            raise ConfigError("Le checkpoint de reprise utilise une autre sélection.")
        if state.get("completed") is True:
            best_path = settings.checkpoint_directory / str(
                state["best_checkpoint_name"]
            )
            if not best_path.is_dir():
                raise ConfigError("Le meilleur checkpoint du run terminé est introuvable.")
            state["completed_run_reused"] = True
            print(f"Run déjà terminé ; réutilisation de {best_path.name}.")
            return state, best_path
        model = _load_model(
            source=latest,
            model_revision=None,
            cache_dir=model_settings.cache_dir,
            local_files_only=True,
            device=device,
            transformers=transformers,
        )
        print(f"Reprise depuis {latest.name}.")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    micro_batches_per_epoch = math.ceil(
        len(selection.train_rows) / settings.train_batch_size
    )
    optimizer_steps_per_epoch = math.ceil(
        micro_batches_per_epoch / settings.gradient_accumulation_steps
    )
    total_steps = optimizer_steps_per_epoch * settings.num_train_epochs
    warmup_steps = math.ceil(total_steps * settings.warmup_ratio)
    scheduler = _linear_warmup_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        torch=torch,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=settings.fp16,
        init_scale=4096.0,
        growth_interval=2000,
    )
    if latest is not None:
        optimizer.load_state_dict(
            torch.load(latest / "optimizer.pt", map_location=device, weights_only=True)
        )
        scheduler.load_state_dict(
            torch.load(latest / "scheduler.pt", map_location=device, weights_only=True)
        )
        scaler_path = latest / "scaler.pt"
        if scaler_path.is_file():
            scaler.load_state_dict(
                torch.load(scaler_path, map_location=device, weights_only=True)
            )
    started = perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    accumulation_loss = 0.0
    accumulation_count = 0
    optimizer.zero_grad(set_to_none=True)
    stop_training = False

    for epoch in range(settings.num_train_epochs):
        ordered = _ordered_train_rows(selection.train_rows, settings.seed, epoch)
        batches = _chunks(ordered, settings.train_batch_size)
        group_sizes = accumulation_group_sizes(
            len(batches),
            settings.gradient_accumulation_steps,
        )
        completed = (
            int(state["micro_batches_completed"]) if epoch == int(state["epoch"]) else 0
        )
        if epoch < int(state["epoch"]):
            continue
        for batch_index, row_batch in enumerate(batches):
            if batch_index < completed:
                continue
            model.train()
            batch = _to_device(collator(row_batch), device)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=settings.fp16,
            ):
                output = model(**batch)
                raw_loss = output.loss
                group_index = batch_index // settings.gradient_accumulation_steps
                loss = raw_loss / group_sizes[group_index]
            loss_value = float(raw_loss.detach().cpu())
            if not math.isfinite(loss_value):
                raise ConfigError("Loss train NaN/Inf : arrêt immédiat.")
            if loss_value > 50:
                raise ConfigError("Divergence forte détectée (loss > 50).")
            scaler.scale(loss).backward()
            accumulation_loss += loss_value
            accumulation_count += 1
            state["micro_batches_completed"] = batch_index + 1
            is_last_batch = batch_index + 1 == len(batches)
            should_step = (
                accumulation_count == settings.gradient_accumulation_steps
                or is_last_batch
            )
            if not should_step:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.max_grad_norm)
            scale_before_step = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            optimizer_step_skipped = float(scaler.get_scale()) < scale_before_step
            if optimizer_step_skipped:
                state["fp16_optimizer_steps_skipped"] = (
                    int(state.get("fp16_optimizer_steps_skipped", 0)) + 1
                )
            optimizer.zero_grad(set_to_none=True)
            if optimizer_step_skipped:
                accumulation_loss = 0.0
                accumulation_count = 0
                continue
            scheduler.step()
            state["global_step"] = int(state["global_step"]) + 1
            global_step = int(state["global_step"])
            train_loss = accumulation_loss / accumulation_count
            accumulation_loss = 0.0
            accumulation_count = 0
            log_row: dict[str, Any] = {
                "step": global_step,
                "epoch": epoch + (batch_index + 1) / len(batches),
                "train_loss": train_loss,
                "validation_loss": None,
                "validation_wer": None,
                "validation_cer": None,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "elapsed_seconds": perf_counter() - started,
            }
            cast(list[dict[str, Any]], state["log_history"]).append(log_row)
            if global_step == 1 or global_step % settings.logging_steps == 0:
                print(
                    f"step={global_step:03d}/{total_steps} "
                    f"train_loss={train_loss:.6f} "
                    f"lr={log_row['learning_rate']:.8f}"
                )
            should_evaluate = (
                global_step % settings.evaluation_steps == 0
                or global_step == total_steps
                or is_last_batch
            )
            if not should_evaluate:
                continue
            validation = _evaluate_validation(
                model=model,
                processor=processor,
                rows=selection.validation_rows,
                collator=collator,
                batch_size=settings.eval_batch_size,
                device=device,
                torch=torch,
                fp16=settings.fp16,
            )
            log_row["validation_loss"] = validation.metrics["validation_loss"]
            log_row["validation_wer"] = validation.metrics["wer_micro"]
            log_row["validation_cer"] = validation.metrics["cer_micro"]
            current_wer = float(validation.metrics["wer_micro"])
            previous_best = state.get("best_wer")
            improved = previous_best is None or current_wer < (
                float(previous_best) - settings.early_stopping_threshold
            )
            checkpoint_name = f"checkpoint-{global_step:06d}"
            if improved:
                state["best_wer"] = current_wer
                state["best_checkpoint_name"] = checkpoint_name
                state["early_stopping_bad_evaluations"] = 0
            else:
                state["early_stopping_bad_evaluations"] = (
                    int(state["early_stopping_bad_evaluations"]) + 1
                )
            state["epoch"] = epoch
            _save_checkpoint(
                settings=settings,
                model=model,
                processor=processor,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                state=state,
                torch=torch,
            )
            print(
                f"validation step={global_step:03d} "
                f"loss={validation.metrics['validation_loss']:.6f} "
                f"WER={validation.metrics['wer_micro']:.6f} "
                f"CER={validation.metrics['cer_micro']:.6f}"
            )
            if (
                int(state["early_stopping_bad_evaluations"])
                >= settings.early_stopping_patience
            ):
                state["early_stopping_triggered"] = True
                stop_training = True
                break
        state["epoch"] = epoch + 1
        state["micro_batches_completed"] = 0
        if stop_training:
            break
    if not state.get("best_checkpoint_name"):
        raise ConfigError("Aucun meilleur checkpoint n'a été créé.")
    torch.cuda.synchronize(device)
    state["training_duration_seconds"] = perf_counter() - started
    state["max_gpu_memory_mib"] = (
        float(torch.cuda.max_memory_allocated(device)) / 1024**2
    )
    state["total_optimizer_steps_planned"] = total_steps
    state["warmup_steps"] = warmup_steps
    state["completed"] = True
    best_path = settings.checkpoint_directory / str(state["best_checkpoint_name"])
    for checkpoint in _checkpoint_paths(settings.checkpoint_directory):
        _write_json(checkpoint / TRAINING_STATE_FILENAME, state)
    return state, best_path


def _write_history(path: Path, history: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "step",
        "epoch",
        "train_loss",
        "validation_loss",
        "validation_wer",
        "validation_cer",
        "learning_rate",
        "elapsed_seconds",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)


def _write_predictions(
    path: Path,
    rows: Sequence[ManifestRow],
    baseline: Sequence[Mapping[str, Any]],
    adapted: Sequence[Mapping[str, Any]],
) -> None:
    if not len(rows) == len(baseline) == len(adapted):
        raise ConfigError("Les prédictions baseline/adaptées ne sont pas alignées.")
    fields = (
        "audio_id_anonymized",
        "speaker_id_anonymized",
        "text_raw",
        "text_no_tones",
        "target_text_mvp",
        "baseline_prediction",
        "adapted_prediction",
        "baseline_word_substitutions",
        "baseline_word_deletions",
        "baseline_word_insertions",
        "adapted_word_substitutions",
        "adapted_word_deletions",
        "adapted_word_insertions",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row, before, after in zip(rows, baseline, adapted, strict=True):
            if (
                before["audio_id_anonymized"] != row.utterance_id
                or after["audio_id_anonymized"] != row.utterance_id
            ):
                raise ConfigError("Ordre des prédictions incohérent.")
            writer.writerow(
                {
                    "audio_id_anonymized": row.utterance_id,
                    "speaker_id_anonymized": row.speaker_id,
                    "text_raw": row.text_raw,
                    "text_no_tones": row.text_no_tones,
                    "target_text_mvp": row.target_text,
                    "baseline_prediction": before["prediction"],
                    "adapted_prediction": after["prediction"],
                    "baseline_word_substitutions": before["word_substitutions"],
                    "baseline_word_deletions": before["word_deletions"],
                    "baseline_word_insertions": before["word_insertions"],
                    "adapted_word_substitutions": after["word_substitutions"],
                    "adapted_word_deletions": after["word_deletions"],
                    "adapted_word_insertions": after["word_insertions"],
                }
            )


def _write_curves(path: Path, history: Sequence[Mapping[str, Any]]) -> None:
    image_module = importlib.import_module("PIL.Image")
    draw_module = importlib.import_module("PIL.ImageDraw")
    image = image_module.new("RGB", (1200, 760), "white")
    draw = draw_module.Draw(image)
    panels = (
        ("Train loss", "train_loss", (80, 70, 1140, 340), "#0057b8"),
        ("Validation WER", "validation_wer", (80, 420, 1140, 690), "#c73e1d"),
    )
    for title, field, box, color in panels:
        left, top, right, bottom = box
        draw.rectangle(box, outline="#222222", width=2)
        values = [
            (int(row["step"]), float(row[field]))
            for row in history
            if row.get(field) is not None
        ]
        draw.text((left, top - 30), title, fill="#111111")
        if not values:
            continue
        minimum = min(value for _, value in values)
        maximum = max(value for _, value in values)
        span = maximum - minimum or 1.0
        maximum_step = max(step for step, _ in values)
        points = [
            (
                left + (right - left) * step / max(1, maximum_step),
                bottom - (bottom - top) * (value - minimum) / span,
            )
            for step, value in values
        ]
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        else:
            draw.line(points, fill=color, width=4)
        draw.text((10, top), f"{maximum:.4f}", fill="#111111")
        draw.text((10, bottom - 10), f"{minimum:.4f}", fill="#111111")
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(temporary, format="PNG")
    temporary.replace(path)


def _checkpoint_size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _public_comparison(comparison: Mapping[str, Any]) -> dict[str, Any]:
    """Remove per-speaker linkable data from the shareable aggregate report."""

    payload = dict(comparison)
    for field in ("baseline", "adapted"):
        metrics = payload.get(field)
        if isinstance(metrics, Mapping):
            public_metrics = dict(metrics)
            public_metrics.pop("speaker_metrics", None)
            payload[field] = public_metrics
    return payload


def _write_comparison_markdown(
    path: Path,
    comparison: Mapping[str, Any],
) -> None:
    baseline = cast(Mapping[str, Any], comparison["baseline"])
    adapted = cast(Mapping[str, Any], comparison["adapted"])
    content = "\n".join(
        (
            "# Comparaison validation — pilote Whisper Tiny dioula",
            "",
            "> Rapport agrégé sans identifiant, chemin, référence ni prédiction. "
            "Les métriques utilisent le même sous-ensemble de validation gelé ; "
            "aucun audio test n'a été chargé ou transcrit.",
            "",
            "| Métrique | Baseline | Adapté |",
            "|---|---:|---:|",
            f"| WER micro | {float(baseline['wer_micro']):.6f} | "
            f"{float(adapted['wer_micro']):.6f} |",
            f"| CER micro | {float(baseline['cer_micro']):.6f} | "
            f"{float(adapted['cer_micro']):.6f} |",
            f"| RTF | {float(baseline['rtf']):.6f} | {float(adapted['rtf']):.6f} |",
            f"| Latence moyenne (s) | {float(baseline['mean_latency_seconds']):.6f} | "
            f"{float(adapted['mean_latency_seconds']):.6f} |",
            f"| Substitutions | {baseline['word_substitutions']} | "
            f"{adapted['word_substitutions']} |",
            f"| Insertions | {baseline['word_insertions']} | "
            f"{adapted['word_insertions']} |",
            f"| Suppressions | {baseline['word_deletions']} | "
            f"{adapted['word_deletions']} |",
            "",
            f"- réduction absolue WER : {float(comparison['wer_absolute_reduction']):.6f}",
            "- réduction relative WER : "
            f"{float(comparison['wer_relative_reduction_percent']):.2f} %",
            f"- réduction absolue CER : {float(comparison['cer_absolute_reduction']):.6f}",
            "- réduction relative CER : "
            f"{float(comparison['cer_relative_reduction_percent']):.2f} %",
            "",
            "Ces résultats servent au choix expérimental sur validation uniquement. "
            "Ils ne constituent pas une évaluation finale.",
            "",
        )
    )
    path.write_text(content, encoding="utf-8")


def _write_resource_estimate(
    path: Path,
    *,
    settings: PilotSettings,
    dataset: AuditedDataset,
    training_state: Mapping[str, Any],
    checkpoint_size_bytes: int,
) -> None:
    full_train_count = sum(row.split == "train" for row in dataset.rows)
    scale = full_train_count / settings.train_sample_count
    one_epoch_seconds = float(training_state["training_duration_seconds"]) * scale
    two_epoch_seconds = one_epoch_seconds * 2
    checkpoint_gib = checkpoint_size_bytes / 1024**3
    content = f"""# Estimation des ressources — entraînement complet dioula

Cette estimation extrapole le pilote Phase 4C ; aucun entraînement complet n'a
été lancé.

- audios pilote train : {settings.train_sample_count}
- audios train complets : {full_train_count}
- facteur de volume : {scale:.3f}
- durée pilote observée : {float(training_state["training_duration_seconds"]):.3f} s
- estimation 1 époque complète : {one_epoch_seconds / 60:.2f} min
- estimation 2 époques complètes : {two_epoch_seconds / 60:.2f} min
- pic VRAM observé : {float(training_state["max_gpu_memory_mib"]):.2f} MiB
- taille d'un checkpoint observé : {checkpoint_gib:.3f} GiB
- réserve recommandée pour deux checkpoints et les journaux :
  {max(5.0, checkpoint_gib * 2.5):.2f} GiB

## Configuration complète proposée, non exécutée

- base : `openai/whisper-tiny` à la même révision ;
- train complet : 13 764 audios, validation complète : 2 661 audios ;
- 2 époques maximum avec early stopping ;
- batch CUDA : {settings.train_batch_size} ;
- accumulation : {settings.gradient_accumulation_steps} ;
- learning rate initial : `{settings.learning_rate}` ;
- warmup ratio : `{settings.warmup_ratio}` ;
- weight decay : `{settings.weight_decay}` ;
- fp16 et gradient checkpointing conservés ;
- meilleur checkpoint choisi sur WER validation ;
- test final strictement réservé à l'évaluation finale.

L'extrapolation est linéaire et doit conserver une marge d'au moins 30 % pour
les évaluations régulières, les entrées/sorties et les variations matérielles.
"""
    path.write_text(content, encoding="utf-8")


def _load_or_run_baseline(
    *,
    settings: PilotSettings,
    selection: PilotSelection,
    processor: Any,
    collator: PilotCollator,
    model_settings: Any,
    device: Any,
    torch: Any,
    transformers: Any,
) -> EvaluationResult:
    cache_path = settings.artifact_output_directory / BASELINE_CACHE_FILENAME
    if cache_path.is_file():
        cached = _load_json(cache_path)
        if (
            cached.get("selection_sha256") == selection.selection_sha256
            and cached.get("model_revision") == model_settings.model_revision
        ):
            return EvaluationResult(
                metrics=cast(dict[str, Any], cached["metrics"]),
                predictions=tuple(cast(list[dict[str, Any]], cached["predictions"])),
            )
    model = _load_model(
        source=model_settings.model_id,
        model_revision=model_settings.model_revision,
        cache_dir=model_settings.cache_dir,
        local_files_only=model_settings.local_files_only,
        device=device,
        transformers=transformers,
    )
    result = _evaluate_validation(
        model=model,
        processor=processor,
        rows=selection.validation_rows,
        collator=collator,
        batch_size=settings.eval_batch_size,
        device=device,
        torch=torch,
        fp16=settings.fp16,
    )
    _write_json(
        cache_path,
        {
            "selection_sha256": selection.selection_sha256,
            "model_id": model_settings.model_id,
            "model_revision": model_settings.model_revision,
            "hardware": {
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device),
            },
            "metrics": result.metrics,
            "predictions": list(result.predictions),
        },
    )
    del model
    torch.cuda.empty_cache()
    return result


def run_pilot(settings: PilotSettings) -> dict[str, Any]:
    """Execute only Phase 4C and produce local, privacy-aware evidence."""

    dataset = load_audited_dataset(settings)
    selection, selection_report = build_pilot_selection(dataset, settings)
    write_selection_report(settings, selection_report)
    model_settings = load_whisper_settings(settings.model_config_path)
    if (
        model_settings.model_id != settings.expected_model_id
        or model_settings.model_revision != settings.expected_model_revision
    ):
        raise ConfigError("Le modèle Whisper Tiny ou sa révision ne correspond pas.")
    if model_settings.task != "transcribe" or model_settings.language is not None:
        raise ConfigError("Le pilote exige transcribe sans token de langue dyu forcé.")
    device_name, _ = runtime_labels(model_settings)
    if device_name != "cuda":
        raise ConfigError("La Phase 4C exige CUDA sur cette machine.")
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    _seed_everything(settings.seed, torch)
    device = torch.device("cuda:0")
    _check_disk(settings)
    settings.artifact_output_directory.mkdir(parents=True, exist_ok=True)
    settings.report_output_directory.mkdir(parents=True, exist_ok=True)
    processor = transformers.WhisperProcessor.from_pretrained(
        model_settings.model_id,
        revision=model_settings.model_revision,
        cache_dir=str(model_settings.cache_dir),
        local_files_only=model_settings.local_files_only,
        trust_remote_code=False,
    )
    processor.tokenizer.set_prefix_tokens(task="transcribe")
    collator = PilotCollator(
        settings,
        processor,
        model_settings.expected_sampling_rate_hz,
        torch,
    )

    baseline = _load_or_run_baseline(
        settings=settings,
        selection=selection,
        processor=processor,
        collator=collator,
        model_settings=model_settings,
        device=device,
        torch=torch,
        transformers=transformers,
    )
    print(
        "Baseline validation — "
        f"WER={baseline.metrics['wer_micro']:.6f} "
        f"CER={baseline.metrics['cer_micro']:.6f} "
        f"RTF={baseline.metrics['rtf']:.6f}"
    )
    training_state, best_checkpoint = _train_pilot(
        settings=settings,
        selection=selection,
        processor=processor,
        collator=collator,
        model_settings=model_settings,
        device=device,
        torch=torch,
        transformers=transformers,
    )
    del processor
    torch.cuda.empty_cache()
    processor = transformers.WhisperProcessor.from_pretrained(
        best_checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )
    processor.tokenizer.set_prefix_tokens(task="transcribe")
    collator = PilotCollator(
        settings,
        processor,
        model_settings.expected_sampling_rate_hz,
        torch,
    )
    adapted_model = _load_model(
        source=best_checkpoint,
        model_revision=None,
        cache_dir=model_settings.cache_dir,
        local_files_only=True,
        device=device,
        transformers=transformers,
    )
    adapted = _evaluate_validation(
        model=adapted_model,
        processor=processor,
        rows=selection.validation_rows,
        collator=collator,
        batch_size=settings.eval_batch_size,
        device=device,
        torch=torch,
        fp16=settings.fp16,
    )
    comparison = comparison_metrics(baseline.metrics, adapted.metrics)
    comparison.update(
        {
            "schema_version": 1,
            "selection_sha256": selection.selection_sha256,
            "same_validation_subset": True,
            "validation_audio_count": len(selection.validation_rows),
            "official_test_used": False,
            "pilot_test_used": False,
            "final_holdout_used": False,
            "post_correction_used": False,
            "model_id": model_settings.model_id,
            "model_revision": model_settings.model_revision,
            "best_checkpoint_name": best_checkpoint.name,
            "hardware": {
                "device": str(device),
                "gpu_name": torch.cuda.get_device_name(device),
                "pytorch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
            },
        }
    )
    report_root = settings.report_output_directory
    history = cast(list[dict[str, Any]], training_state["log_history"])
    _write_history(report_root / HISTORY_FILENAME, history)
    _write_curves(report_root / CURVES_FILENAME, history)
    _write_predictions(
        settings.artifact_output_directory / PREDICTIONS_FILENAME,
        selection.validation_rows,
        baseline.predictions,
        adapted.predictions,
    )
    _write_json(
        report_root / COMPARISON_JSON_FILENAME,
        _public_comparison(comparison),
    )
    _write_comparison_markdown(
        report_root / COMPARISON_MARKDOWN_FILENAME,
        comparison,
    )
    checkpoint_size = _checkpoint_size_bytes(best_checkpoint)
    training_metrics = {
        "schema_version": 1,
        "status": (
            "succeeded"
            if comparison["wer_improved"] or comparison["cer_improved"]
            else "stable_but_no_validation_improvement"
        ),
        "experiment_id": settings.experiment_id,
        "selection_sha256": selection.selection_sha256,
        "train_audio_count": len(selection.train_rows),
        "validation_audio_count": len(selection.validation_rows),
        "train_speaker_count": len({row.speaker_id for row in selection.train_rows}),
        "validation_speaker_count": len(
            {row.speaker_id for row in selection.validation_rows}
        ),
        "seed": settings.seed,
        "model_id": model_settings.model_id,
        "model_revision": model_settings.model_revision,
        "task": "transcribe",
        "forced_language_token": None,
        "canonical_text_column": settings.canonical_text_column,
        "epochs_configured": settings.num_train_epochs,
        "steps_completed": training_state["global_step"],
        "learning_rate": settings.learning_rate,
        "warmup_steps": training_state["warmup_steps"],
        "weight_decay": settings.weight_decay,
        "train_batch_size": settings.train_batch_size,
        "gradient_accumulation_steps": settings.gradient_accumulation_steps,
        "effective_batch_size": (
            settings.train_batch_size * settings.gradient_accumulation_steps
        ),
        "fp16": settings.fp16,
        "gradient_checkpointing": settings.gradient_checkpointing,
        "fp16_optimizer_steps_skipped": training_state[
            "fp16_optimizer_steps_skipped"
        ],
        "training_duration_seconds": training_state["training_duration_seconds"],
        "max_gpu_memory_mib": training_state["max_gpu_memory_mib"],
        "best_validation_wer": training_state["best_wer"],
        "best_checkpoint_name": best_checkpoint.name,
        "best_checkpoint_reload_succeeded": True,
        "checkpoint_size_bytes": checkpoint_size,
        "checkpoints_retained": [
            path.name for path in _checkpoint_paths(settings.checkpoint_directory)
        ],
        "save_total_limit": settings.save_total_limit,
        "early_stopping_triggered": training_state["early_stopping_triggered"],
        "early_stopping_bad_evaluations": training_state[
            "early_stopping_bad_evaluations"
        ],
        "resume_supported": settings.resume_from_checkpoint,
        "completed_checkpoint_resume_verified": bool(
            training_state.get("completed_run_reused", False)
        ),
        "hardware": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "nan_or_inf_detected": False,
        "cuda_error_detected": False,
        "split_integrity_passed": selection_report["overall_passed"],
        "official_test_used": False,
        "pilot_test_used": False,
        "final_holdout_used": False,
        "publication_allowed": False,
    }
    _write_json(report_root / PILOT_METRICS_FILENAME, training_metrics)
    _write_resource_estimate(
        report_root / RESOURCE_ESTIMATE_FILENAME,
        settings=settings,
        dataset=dataset,
        training_state=training_state,
        checkpoint_size_bytes=checkpoint_size,
    )
    print(
        "Adapté validation — "
        f"WER={adapted.metrics['wer_micro']:.6f} "
        f"CER={adapted.metrics['cer_micro']:.6f} "
        f"RTF={adapted.metrics['rtf']:.6f}"
    )
    print(
        "Réduction relative — "
        f"WER={comparison['wer_relative_reduction_percent']:.2f}% "
        f"CER={comparison['cer_relative_reduction_percent']:.2f}%"
    )
    return {
        "training": training_metrics,
        "comparison": comparison,
        "baseline_predictions": baseline.predictions,
        "adapted_predictions": adapted.predictions,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the bounded Phase 4C run."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        default="configs/experiments/pilot_finetune_whisper_tiny_dy.yaml",
    )
    arguments = parser.parse_args(argv)
    try:
        result = run_pilot(load_pilot_settings(arguments.experiment))
    except IvoireVoiceError as exc:
        parser.error(str(exc))
    training = cast(dict[str, Any], result["training"])
    print(f"Statut pilote : {training['status']}.")
    print(f"Meilleur checkpoint : {training['best_checkpoint_name']}.")
    print(
        f"Durée train={training['training_duration_seconds']:.3f}s, "
        f"pic VRAM={training['max_gpu_memory_mib']:.2f} MiB."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
