"""Leakage-safe direct PyTorch smoke-overfit for local Whisper Tiny adaptation."""

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
from ivoirevoice.evaluation.metrics import ScoredItem, compute_evaluation_metrics
from ivoirevoice.exceptions import ConfigError, IvoireVoiceError
from ivoirevoice.models.whisper import load_whisper_settings, runtime_labels
from ivoirevoice.training.audit import (
    AuditedDataset,
    ManifestRow,
    build_split_integrity_report,
    load_audited_dataset,
    load_or_initialize_annotations,
    select_representative_train_rows,
    selection_sha256,
    validated_rows,
)
from ivoirevoice.training.settings import SmokeSettings, load_smoke_settings


@dataclass(frozen=True, slots=True)
class PreparedSample:
    """One human-validated micro-train sample held in CPU memory."""

    row: ManifestRow
    input_features: Any
    attention_mask: Any
    labels: Any


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_loss_history(path: Path, history: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["step", "loss", "learning_rate", "elapsed_seconds"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(path)


def _write_private_predictions(
    path: Path,
    samples: Sequence[PreparedSample],
    before: Sequence[str],
    after: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "audio_id_anonymized",
                "speaker_id_anonymized",
                "text_raw",
                "text_no_tones",
                "target_text_mvp",
                "prediction_before",
                "prediction_after",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        for sample, prediction_before, prediction_after in zip(
            samples, before, after, strict=True
        ):
            writer.writerow(
                {
                    "audio_id_anonymized": sample.row.utterance_id,
                    "speaker_id_anonymized": sample.row.speaker_id,
                    "text_raw": sample.row.text_raw,
                    "text_no_tones": sample.row.text_no_tones,
                    "target_text_mvp": sample.row.target_text,
                    "prediction_before": prediction_before,
                    "prediction_after": prediction_after,
                }
            )
    temporary.replace(path)


def _seed_everything(seed: int, torch: Any) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    try:
        numpy = importlib.import_module("numpy")
    except ImportError as exc:
        raise ConfigError("NumPy est requis pour le smoke-overfit.") from exc
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def _load_audio(path: Path, expected_rate: int) -> Any:
    try:
        soundfile = importlib.import_module("soundfile")
        numpy = importlib.import_module("numpy")
    except ImportError as exc:
        raise ConfigError("soundfile et NumPy sont requis pour charger les audios.") from exc
    try:
        audio, sample_rate = soundfile.read(
            str(path),
            dtype="float32",
            always_2d=False,
        )
    except (OSError, RuntimeError) as exc:
        raise ConfigError(f"Lecture audio impossible pour {path.name} : {exc}") from exc
    if int(sample_rate) != expected_rate:
        raise ConfigError(
            f"{path.name} est à {sample_rate} Hz au lieu de {expected_rate} Hz."
        )
    array = numpy.asarray(audio, dtype="float32")
    if array.ndim != 1 or not array.size:
        raise ConfigError(f"{path.name} doit être un signal mono non vide.")
    if not numpy.isfinite(array).all():
        raise ConfigError(f"{path.name} contient des valeurs audio non finies.")
    return array


def _prepare_samples(
    settings: SmokeSettings,
    rows: Sequence[ManifestRow],
    processor: Any,
    expected_rate: int,
) -> tuple[PreparedSample, ...]:
    prepared: list[PreparedSample] = []
    for row in rows:
        audio = _load_audio(settings.dataset_root / row.audio_path, expected_rate)
        actual_duration = float(audio.size) / expected_rate
        if actual_duration > 30.0:
            raise ConfigError(
                "Un audio validé dépasse les 30 secondes acceptées sans découpage par Whisper."
            )
        encoded = processor.feature_extractor(
            audio,
            sampling_rate=expected_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        features = encoded.input_features[0]
        attention_mask = encoded.attention_mask[0]
        labels = processor.tokenizer(row.target_text, return_tensors="pt").input_ids[0]
        if int(labels.numel()) < 2:
            raise ConfigError("Un label tokenisé est vide ou invalide.")
        if not bool(features.isfinite().all()):
            raise ConfigError("Les features Whisper contiennent NaN ou Inf.")
        if not bool(attention_mask.isfinite().all()):
            raise ConfigError("Le masque d'attention Whisper contient NaN ou Inf.")
        if not bool(labels.isfinite().all()):
            raise ConfigError("Les labels Whisper contiennent NaN ou Inf.")
        prepared.append(
            PreparedSample(
                row=row,
                input_features=features,
                attention_mask=attention_mask,
                labels=labels,
            )
        )
    return tuple(prepared)


def _collate(
    samples: Sequence[PreparedSample],
    indices: Sequence[int],
    torch: Any,
) -> tuple[Any, Any, Any]:
    features = torch.stack([samples[index].input_features for index in indices])
    attention_masks = torch.stack([samples[index].attention_mask for index in indices])
    maximum = max(int(samples[index].labels.numel()) for index in indices)
    labels = torch.full((len(indices), maximum), -100, dtype=torch.long)
    for batch_index, sample_index in enumerate(indices):
        sample_labels = samples[sample_index].labels
        labels[batch_index, : sample_labels.numel()] = sample_labels
        padding = labels[batch_index, sample_labels.numel() :]
        if padding.numel() and not bool((padding == -100).all()):
            raise ConfigError("Le padding des labels doit être intégralement égal à -100.")
    return features, attention_masks, labels


def _prediction_metrics(
    samples: Sequence[PreparedSample],
    predictions: Sequence[str],
) -> dict[str, Any]:
    scored = tuple(
        ScoredItem(
            speaker_id=sample.row.speaker_id,
            reference_normalized=normalize_evaluation_text(
                sample.row.target_text,
                lowercase=True,
                remove_punctuation=True,
            ),
            prediction_normalized=normalize_evaluation_text(
                prediction,
                lowercase=True,
                remove_punctuation=True,
            ),
            audio_duration_seconds=sample.row.duration_seconds,
            processing_time_seconds=0.0,
        )
        for sample, prediction in zip(samples, predictions, strict=True)
    )
    metrics = compute_evaluation_metrics(scored)
    return {
        "wer_micro": metrics["wer_micro"],
        "cer_micro": metrics["cer_micro"],
        "word_substitutions": metrics["word_substitutions"],
        "word_deletions": metrics["word_deletions"],
        "word_insertions": metrics["word_insertions"],
    }


def _generate_predictions(
    model: Any,
    processor: Any,
    samples: Sequence[PreparedSample],
    device: Any,
    torch: Any,
) -> tuple[str, ...]:
    model.eval()
    maximum_label_length = max(int(sample.labels.numel()) for sample in samples)
    predictions: list[str] = []
    with torch.inference_mode():
        for sample in samples:
            input_features = sample.input_features.unsqueeze(0).to(device)
            attention_mask = sample.attention_mask.unsqueeze(0).to(device)
            generated = model.generate(
                input_features=input_features,
                attention_mask=attention_mask,
                task="transcribe",
                do_sample=False,
                max_new_tokens=min(maximum_label_length + 12, 96),
            )
            prediction = processor.batch_decode(
                generated,
                skip_special_tokens=True,
            )[0]
            predictions.append(str(prediction).strip())
    return tuple(predictions)


def _resolve_precision(settings: SmokeSettings, torch: Any, device_name: str) -> tuple[bool, Any]:
    if settings.mixed_precision == "no" or device_name == "cpu":
        return False, torch.float32
    requested = settings.mixed_precision
    if requested == "auto":
        requested = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if requested == "bf16" and not torch.cuda.is_bf16_supported():
        raise ConfigError("Le GPU ne prend pas en charge le mixed precision bf16.")
    return True, torch.bfloat16 if requested == "bf16" else torch.float16


def _train(
    settings: SmokeSettings,
    samples: Sequence[PreparedSample],
    model: Any,
    device: Any,
    torch: Any,
) -> list[dict[str, Any]]:
    model.train()
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
    device_name = str(device.type)
    autocast_enabled, autocast_dtype = _resolve_precision(settings, torch, device_name)
    use_scaler = autocast_enabled and autocast_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(settings.seed)
    order: list[int] = []
    cursor = 0
    history: list[dict[str, Any]] = []
    started = perf_counter()

    for step in range(1, settings.max_steps + 1):
        if cursor + settings.batch_size > len(order):
            order.extend(torch.randperm(len(samples), generator=generator).tolist())
        indices = order[cursor : cursor + settings.batch_size]
        cursor += settings.batch_size
        features, attention_mask, labels = _collate(samples, indices, torch)
        features = features.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        model_device = next(model.parameters()).device
        if (
            features.device != model_device
            or attention_mask.device != model_device
            or labels.device != model_device
        ):
            raise ConfigError(
                "Le modèle, les features et les labels ne sont pas sur le même device."
            )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device_name,
            dtype=autocast_dtype,
            enabled=autocast_enabled,
        ):
            outputs = model(
                input_features=features,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
        loss_value = float(loss.detach().cpu())
        if settings.stop_on_nan and not math.isfinite(loss_value):
            raise ConfigError(f"Loss non finie détectée au step {step}; arrêt immédiat.")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), settings.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        history.append(
            {
                "step": step,
                "loss": loss_value,
                "learning_rate": settings.learning_rate,
                "elapsed_seconds": perf_counter() - started,
            }
        )
        if step == 1 or step % settings.evaluation_steps == 0:
            print(f"step={step:03d} loss={loss_value:.6f}")
    return history


def _loss_summary(history: Sequence[dict[str, Any]]) -> dict[str, Any]:
    window = min(10, max(1, len(history) // 4))
    first_mean = sum(float(item["loss"]) for item in history[:window]) / window
    final_mean = sum(float(item["loss"]) for item in history[-window:]) / window
    reduction = (first_mean - final_mean) / first_mean if first_mean else 0.0
    checkpoints = [
        {"step": int(item["step"]), "loss": float(item["loss"])}
        for item in history
        if int(item["step"]) == 1
        or int(item["step"]) % 10 == 0
        or int(item["step"]) == len(history)
    ]
    return {
        "first_step_loss": float(history[0]["loss"]),
        "last_step_loss": float(history[-1]["loss"]),
        "initial_window_size": window,
        "initial_mean_loss": first_mean,
        "final_mean_loss": final_mean,
        "loss_reduction_fraction": reduction,
        "checkpoints": checkpoints,
    }


def _write_loss_plot(path: Path, history: Sequence[dict[str, Any]]) -> None:
    """Render a dependency-light PNG loss curve with Pillow."""

    try:
        image_module = importlib.import_module("PIL.Image")
        draw_module = importlib.import_module("PIL.ImageDraw")
    except ImportError as exc:
        raise ConfigError("Pillow est requis pour produire la courbe PNG.") from exc
    width, height = 1000, 600
    left, top, right, bottom = 90, 60, 960, 520
    image = image_module.new("RGB", (width, height), "white")
    draw = draw_module.Draw(image)
    draw.rectangle((left, top, right, bottom), outline="#222222", width=2)
    losses = [float(item["loss"]) for item in history]
    minimum, maximum = min(losses), max(losses)
    span = maximum - minimum or 1.0
    denominator = max(1, len(losses) - 1)
    points = [
        (
            left + (right - left) * index / denominator,
            bottom - (bottom - top) * (loss - minimum) / span,
        )
        for index, loss in enumerate(losses)
    ]
    draw.line(points, fill="#0057b8", width=4)
    draw.text((left, 20), "Whisper Tiny smoke-overfit - training loss", fill="#111111")
    draw.text((left, bottom + 20), f"step 1 -> {len(history)}", fill="#111111")
    draw.text((10, top), f"max {maximum:.4f}", fill="#111111")
    draw.text((10, bottom - 10), f"min {minimum:.4f}", fill="#111111")
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(temporary, format="PNG")
    temporary.replace(path)


def _write_smoke_report(
    path: Path,
    metrics: Mapping[str, Any],
) -> None:
    loss = cast(Mapping[str, Any], metrics["loss"])
    before_metrics = cast(Mapping[str, Any], metrics["micro_train_metrics_before"])
    after_metrics = cast(Mapping[str, Any], metrics["micro_train_metrics_after"])
    checkpoints = cast(Sequence[Mapping[str, Any]], loss["checkpoints"])
    checkpoint_rows = [
        f"| {item['step']} | {float(item['loss']):.6f} |" for item in checkpoints
    ]
    content = "\n".join(
        (
            "# Smoke-overfit Whisper Tiny dioula — Phase 4B",
            "",
            "> Rapport public agrégé, sans identifiant, transcription ni prédiction. "
            "Aucune conclusion de généralisation.",
            "",
            f"- Statut : **{metrics['validation_status']}**",
            f"- Audios validés du train : {metrics['sample_count']}",
            f"- Test officiel utilisé : {metrics['official_test_used']}",
            f"- Validation utilisée : {metrics['validation_used']}",
            f"- Pilote utilisé : {metrics['pilot_used']}",
            f"- Modèle : `{metrics['model_id']}`",
            f"- Révision : `{metrics['model_revision']}`",
            f"- Tâche : `{metrics['task']}`",
            f"- Langue : `{metrics['language_strategy']}`",
            f"- Device : `{metrics['device']}`",
            f"- Seed : {metrics['seed']}",
            f"- Steps : {metrics['max_steps']}",
            f"- Learning rate : {metrics['learning_rate']}",
            f"- Temps total : {float(metrics['total_duration_seconds']):.3f} s",
            f"- Mémoire GPU maximale : {float(metrics['max_gpu_memory_mib']):.2f} MiB",
            "",
            "## Loss de mémorisation",
            "",
            f"- Première loss : {float(loss['first_step_loss']):.6f}",
            f"- Dernière loss : {float(loss['last_step_loss']):.6f}",
            f"- Moyenne initiale : {float(loss['initial_mean_loss']):.6f}",
            f"- Moyenne finale : {float(loss['final_mean_loss']):.6f}",
            f"- Réduction des moyennes : {float(loss['loss_reduction_fraction']):.2%}",
            "",
            "| Step | Loss |",
            "|---:|---:|",
            *checkpoint_rows,
            "",
            "## Métriques sur le micro-train uniquement",
            "",
            "| Mesure de mémorisation | Avant | Après |",
            "|---|---:|---:|",
            f"| WER micro-train | {float(before_metrics['wer_micro']):.6f} | "
            f"{float(after_metrics['wer_micro']):.6f} |",
            f"| CER micro-train | {float(before_metrics['cer_micro']):.6f} | "
            f"{float(after_metrics['cer_micro']):.6f} |",
            "",
            "Ces valeurs sont calculées sur les mêmes audios que ceux de l'entraînement. "
            "Elles mesurent la mémorisation, pas la généralisation.",
            "",
            "## Anomalies d'exécution",
            "",
            f"- NaN/Inf : {metrics['nan_detected']}",
            f"- Crash : {metrics['crash_detected']}",
            f"- Fuite de split : {not bool(metrics['split_integrity_passed'])}",
            "",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _preflight(
    settings: SmokeSettings,
) -> tuple[AuditedDataset, tuple[ManifestRow, ...], dict[str, Any]]:
    dataset = load_audited_dataset(settings)
    selected = select_representative_train_rows(
        dataset.rows,
        settings.sample_count,
        settings.seed,
    )
    annotations = load_or_initialize_annotations(settings, dataset, selected)
    rows = validated_rows(settings, selected, annotations)
    pilot_ids: set[str] = set()
    for path in settings.pilot_prediction_files:
        try:
            with path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                pilot_ids.update(
                    row["utterance_id"] for row in reader if row.get("utterance_id")
                )
        except (OSError, UnicodeError, csv.Error, KeyError) as exc:
            raise ConfigError(f"Impossible de revérifier le pilote {path.name} : {exc}") from exc
    split_report = build_split_integrity_report(dataset, rows, pilot_ids)
    if split_report["overall_passed"] is not True:
        raise ConfigError("Préflight refusé : train/test/pilote ne sont pas strictement séparés.")
    return dataset, rows, split_report


def run_smoke_overfit(settings: SmokeSettings) -> dict[str, Any]:
    """Run a bounded memorization diagnostic after all manual and split gates pass."""

    total_started = perf_counter()
    dataset, rows, split_report = _preflight(settings)
    model_settings = load_whisper_settings(settings.model_config_path)
    if model_settings.model_id != settings.expected_model_id:
        raise ConfigError("Le modèle chargé ne correspond pas à Whisper Tiny.")
    if model_settings.task != "transcribe":
        raise ConfigError("Le smoke-overfit interdit toute tâche autre que transcribe.")
    if model_settings.language is not None:
        raise ConfigError(
            "Whisper ne possède pas de token dyu : aucun token de langue ne doit être forcé."
        )
    device_name, configured_dtype = runtime_labels(model_settings)
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise ConfigError("PyTorch et Transformers sont requis pour le smoke-overfit.") from exc
    _seed_everything(settings.seed, torch)
    device = torch.device("cuda:0" if device_name == "cuda" else "cpu")
    model_settings.cache_dir.mkdir(parents=True, exist_ok=True)
    common_arguments = {
        "revision": model_settings.model_revision,
        "cache_dir": str(model_settings.cache_dir),
        "local_files_only": model_settings.local_files_only,
        "trust_remote_code": False,
    }
    processor = transformers.WhisperProcessor.from_pretrained(
        model_settings.model_id,
        **common_arguments,
    )
    processor.tokenizer.set_prefix_tokens(task="transcribe")
    model = transformers.WhisperForConditionalGeneration.from_pretrained(
        model_settings.model_id,
        attn_implementation="eager",
        **common_arguments,
    ).to(device)
    if next(model.parameters()).device != device:
        raise ConfigError("Le modèle Whisper n'a pas été placé sur le device attendu.")
    if device_name == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    prepared = _prepare_samples(
        settings,
        rows,
        processor,
        model_settings.expected_sampling_rate_hz,
    )
    output_directory = settings.artifacts_root / settings.output_relative_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    report_directory = settings.report_output_directory
    report_directory.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": settings.experiment_id,
        "status": "in_progress",
        "objective": "memorization_diagnostic_only",
        "generalization_claim_allowed": False,
        "split": "train",
        "official_test_used": False,
        "validation_used": False,
        "pilot_used": False,
        "seed": settings.seed,
        "manifest_sha256": dataset.manifest_sha256,
        "selection_sha256": selection_sha256(rows),
        "sample_count": len(rows),
        "speaker_count": len({row.speaker_id for row in rows}),
        "model_id": model_settings.model_id,
        "model_revision": model_settings.model_revision,
        "device": device_name,
        "configured_dtype": configured_dtype,
        "mixed_precision": settings.mixed_precision,
        "task": "transcribe",
        "forced_language_token": None,
        "language_strategy": "multilingual_without_forced_dyu_token",
        "canonical_text_column": settings.canonical_text_column,
        "max_steps": settings.max_steps,
        "batch_size": settings.batch_size,
        "learning_rate": settings.learning_rate,
        "save_model": False,
        "publication_allowed": False,
        "split_integrity_passed": split_report["overall_passed"],
        "preflight_checks": {
            "human_validation_minimum_met": len(rows)
            >= settings.minimum_correct_samples,
            "decoded_mono": True,
            "decoded_sample_rate_hz": model_settings.expected_sampling_rate_hz,
            "audio_duration_at_most_30_seconds": True,
            "labels_non_empty": True,
            "label_padding_value": -100,
            "features_finite": True,
            "attention_mask_present_and_finite": True,
            "labels_finite": True,
            "model_and_tensors_same_device": True,
            "deterministic_algorithms_enforced": True,
            "attention_implementation": "eager",
        },
    }
    _write_json(output_directory / "run_metadata.json", metadata)

    try:
        predictions_before = _generate_predictions(
            model, processor, prepared, device, torch
        )
        before_metrics = _prediction_metrics(prepared, predictions_before)
        history = _train(settings, prepared, model, device, torch)
        _write_loss_history(output_directory / "loss_history.csv", history)
        predictions_after = _generate_predictions(
            model, processor, prepared, device, torch
        )
        after_metrics = _prediction_metrics(prepared, predictions_after)
        loss = _loss_summary(history)
        wer_improved = float(after_metrics["wer_micro"]) < float(
            before_metrics["wer_micro"]
        )
        cer_improved = float(after_metrics["cer_micro"]) < float(
            before_metrics["cer_micro"]
        )
        finite_loss = all(math.isfinite(float(item["loss"])) for item in history)
        loss_clearly_decreased = float(loss["loss_reduction_fraction"]) >= 0.20
        success = loss_clearly_decreased and wer_improved and cer_improved and finite_loss
        partial = finite_loss and (loss_clearly_decreased or wer_improved or cer_improved)
        validation_status = (
            "réussi" if success else "partiellement réussi" if partial else "échoué"
        )
        if device_name == "cuda":
            torch.cuda.synchronize(device)
            max_gpu_memory_mib = float(torch.cuda.max_memory_allocated(device)) / 1024**2
        else:
            max_gpu_memory_mib = 0.0
        total_duration = perf_counter() - total_started
        metrics = {
            **metadata,
            "status": "succeeded" if success else "criteria_not_met",
            "validation_status": validation_status,
            "loss": loss,
            "micro_train_metrics_before": before_metrics,
            "micro_train_metrics_after": after_metrics,
            "wer_improved": wer_improved,
            "cer_improved": cer_improved,
            "loss_clearly_decreased": loss_clearly_decreased,
            "nan_detected": False,
            "crash_detected": False,
            "training_duration_seconds": float(history[-1]["elapsed_seconds"]),
            "total_duration_seconds": total_duration,
            "max_gpu_memory_mib": max_gpu_memory_mib,
            "report_files": [
                "smoke_overfit_metrics.json",
                "smoke_overfit_report.md",
                "smoke_overfit_loss.csv",
                "smoke_overfit_loss.png",
            ],
            "smoke_overfit_succeeded": success,
        }
        _write_private_predictions(
            output_directory / "predictions_private.csv",
            prepared,
            predictions_before,
            predictions_after,
        )
        _write_loss_history(report_directory / "smoke_overfit_loss.csv", history)
        _write_loss_plot(report_directory / "smoke_overfit_loss.png", history)
        _write_smoke_report(
            report_directory / "smoke_overfit_report.md",
            metrics,
        )
        _write_json(report_directory / "smoke_overfit_metrics.json", metrics)
        _write_json(output_directory / "metrics.json", metrics)
        _write_json(output_directory / "run_metadata.json", metrics)
        return metrics
    except Exception as exc:
        metadata.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "validation_status": "échoué",
                "crash_detected": True,
                "smoke_overfit_succeeded": False,
            }
        )
        _write_json(output_directory / "run_metadata.json", metadata)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint with compact, non-private loss and metric output."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        default="configs/experiments/smoke_overfit_whisper_tiny_dy.yaml",
    )
    arguments = parser.parse_args(argv)
    try:
        metrics = run_smoke_overfit(load_smoke_settings(arguments.experiment))
    except IvoireVoiceError as exc:
        parser.error(str(exc))
    print("\nLoss (agrégée, micro-train) :")
    print("step\tloss")
    loss = cast(dict[str, Any], metrics["loss"])
    for checkpoint in cast(list[dict[str, Any]], loss["checkpoints"]):
        print(f"{checkpoint['step']}\t{checkpoint['loss']:.6f}")
    print(f"Réduction moyenne de loss : {loss['loss_reduction_fraction']:.2%}")
    before = cast(dict[str, Any], metrics["micro_train_metrics_before"])
    after = cast(dict[str, Any], metrics["micro_train_metrics_after"])
    print(
        "Micro-train avant/après — "
        f"WER {before['wer_micro']:.4f} -> {after['wer_micro']:.4f}, "
        f"CER {before['cer_micro']:.4f} -> {after['cer_micro']:.4f}"
    )
    status = str(metrics["validation_status"]).upper()
    print(f"Smoke-overfit : {status}. Aucune métrique de généralisation n'est revendiquée.")
    return 0 if metrics["smoke_overfit_succeeded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
