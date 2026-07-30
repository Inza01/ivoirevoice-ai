"""Leakage-safe direct PyTorch smoke-overfit for local Whisper Tiny adaptation."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
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
                "utterance_id",
                "speaker_id",
                "reference",
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
                    "utterance_id": sample.row.utterance_id,
                    "speaker_id": sample.row.speaker_id,
                    "reference": sample.row.target_text,
                    "prediction_before": prediction_before,
                    "prediction_after": prediction_after,
                }
            )
    temporary.replace(path)


def _seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    try:
        numpy = importlib.import_module("numpy")
    except ImportError as exc:
        raise ConfigError("NumPy est requis pour le smoke-overfit.") from exc
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False


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
        features = processor.feature_extractor(
            audio,
            sampling_rate=expected_rate,
            return_tensors="pt",
        ).input_features[0]
        labels = processor.tokenizer(row.target_text, return_tensors="pt").input_ids[0]
        if int(labels.numel()) < 2:
            raise ConfigError("Un label tokenisé est vide ou invalide.")
        prepared.append(PreparedSample(row=row, input_features=features, labels=labels))
    return tuple(prepared)


def _collate(
    samples: Sequence[PreparedSample],
    indices: Sequence[int],
    torch: Any,
) -> tuple[Any, Any]:
    features = torch.stack([samples[index].input_features for index in indices])
    maximum = max(int(samples[index].labels.numel()) for index in indices)
    labels = torch.full((len(indices), maximum), -100, dtype=torch.long)
    for batch_index, sample_index in enumerate(indices):
        sample_labels = samples[sample_index].labels
        labels[batch_index, : sample_labels.numel()] = sample_labels
    return features, labels


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
            generated = model.generate(
                input_features=input_features,
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
        features, labels = _collate(samples, indices, torch)
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device_name,
            dtype=autocast_dtype,
            enabled=autocast_enabled,
        ):
            outputs = model(input_features=features, labels=labels)
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
        "initial_window_size": window,
        "initial_mean_loss": first_mean,
        "final_mean_loss": final_mean,
        "loss_reduction_fraction": reduction,
        "checkpoints": checkpoints,
    }


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

    dataset, rows, split_report = _preflight(settings)
    model_settings = load_whisper_settings(settings.model_config_path)
    if model_settings.model_id != settings.expected_model_id:
        raise ConfigError("Le modèle chargé ne correspond pas à Whisper Tiny.")
    device_name, configured_dtype = runtime_labels(model_settings)
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise ConfigError("PyTorch et Transformers sont requis pour le smoke-overfit.") from exc
    _seed_everything(settings.seed, torch)
    device = torch.device(device_name)
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
    model = transformers.WhisperForConditionalGeneration.from_pretrained(
        model_settings.model_id,
        **common_arguments,
    ).to(device)
    prepared = _prepare_samples(
        settings,
        rows,
        processor,
        model_settings.expected_sampling_rate_hz,
    )
    output_directory = settings.artifacts_root / settings.output_relative_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": settings.experiment_id,
        "status": "in_progress",
        "objective": "memorization_diagnostic_only",
        "generalization_claim_allowed": False,
        "split": "train",
        "official_test_used": False,
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
        "max_steps": settings.max_steps,
        "batch_size": settings.batch_size,
        "learning_rate": settings.learning_rate,
        "save_model": False,
        "publication_allowed": False,
        "split_integrity_passed": split_report["overall_passed"],
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
        predictions_improved = (
            float(after_metrics["wer_micro"]) < float(before_metrics["wer_micro"])
            or float(after_metrics["cer_micro"]) < float(before_metrics["cer_micro"])
        )
        success = (
            float(loss["loss_reduction_fraction"]) >= 0.20
            and predictions_improved
            and all(math.isfinite(float(item["loss"])) for item in history)
        )
        metrics = {
            **metadata,
            "status": "succeeded" if success else "criteria_not_met",
            "loss": loss,
            "micro_train_metrics_before": before_metrics,
            "micro_train_metrics_after": after_metrics,
            "predictions_improved": predictions_improved,
            "nan_detected": False,
            "smoke_overfit_succeeded": success,
        }
        _write_private_predictions(
            output_directory / "predictions_private.csv",
            prepared,
            predictions_before,
            predictions_after,
        )
        _write_json(output_directory / "metrics.json", metrics)
        _write_json(output_directory / "run_metadata.json", metrics)
        return metrics
    except Exception as exc:
        metadata.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
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
    status = "RÉUSSI" if metrics["smoke_overfit_succeeded"] else "NON VALIDÉ"
    print(f"Smoke-overfit : {status}. Aucune métrique de généralisation n'est revendiquée.")
    return 0 if metrics["smoke_overfit_succeeded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
