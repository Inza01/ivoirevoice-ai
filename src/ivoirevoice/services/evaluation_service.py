"""Reference-aware metrics and structured benchmark/error-analysis loading."""

from __future__ import annotations

import csv
import html
import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, cast

from ivoirevoice.evaluation.baseline import normalize_evaluation_text
from ivoirevoice.evaluation.metrics import edit_counts
from ivoirevoice.exceptions import ConfigError

NO_REFERENCE_MESSAGE = "Non disponible : aucune référence fournie"


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Individual reference/hypothesis comparison."""

    available: bool
    wer: float | None
    cer: float | None
    substitutions: int | None
    deletions: int | None
    insertions: int | None
    reference_normalized: str | None
    prediction_normalized: str | None
    message: str


@dataclass(frozen=True, slots=True)
class BenchmarkView:
    """Aggregate-only benchmark facts safe for the public demo tab."""

    rows: tuple[dict[str, Any], ...]
    dataset_name: str
    split: str
    speaker_count: int
    seed: int
    hardware: str
    run_date: str
    normalization: str
    experiment_id: str
    experiment_title: str
    audio_count: int
    comparison_scope: str


@dataclass(frozen=True, slots=True)
class ErrorSample:
    """One private local sample joined across baseline predictions."""

    audio_id: str
    relative_audio_path: str
    reference: str
    predictions: dict[str, str]


def relative_reduction_percent(baseline: float, adapted: float) -> float | None:
    """Return a lower-is-better relative reduction percentage."""

    if baseline == 0:
        return None
    return (baseline - adapted) / baseline * 100


class EvaluationService:
    """Apply the exact Phase 4A normalization and error metrics."""

    def evaluate(self, reference: str | None, prediction: str) -> EvaluationResult:
        if reference is None or not reference.strip():
            return EvaluationResult(
                available=False,
                wer=None,
                cer=None,
                substitutions=None,
                deletions=None,
                insertions=None,
                reference_normalized=None,
                prediction_normalized=None,
                message=NO_REFERENCE_MESSAGE,
            )
        reference_normalized = normalize_evaluation_text(
            reference,
            lowercase=True,
            remove_punctuation=True,
        )
        prediction_normalized = normalize_evaluation_text(
            prediction,
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
        wer = (
            word_counts.errors / word_counts.reference_units
            if word_counts.reference_units
            else (0.0 if word_counts.errors == 0 else 1.0)
        )
        cer = (
            character_counts.errors / character_counts.reference_units
            if character_counts.reference_units
            else (0.0 if character_counts.errors == 0 else 1.0)
        )
        return EvaluationResult(
            available=True,
            wer=wer,
            cer=cer,
            substitutions=word_counts.substitutions,
            deletions=word_counts.deletions,
            insertions=word_counts.insertions,
            reference_normalized=reference_normalized,
            prediction_normalized=prediction_normalized,
            message="Métriques calculées sur les textes normalisés.",
        )

    def render_word_diff(self, reference: str, prediction: str) -> str:
        reference_words = reference.split()
        prediction_words = prediction.split()
        matcher = SequenceMatcher(a=reference_words, b=prediction_words, autojunk=False)
        parts: list[str] = []
        for operation, start_ref, end_ref, start_hyp, end_hyp in matcher.get_opcodes():
            reference_chunk = html.escape(" ".join(reference_words[start_ref:end_ref]))
            prediction_chunk = html.escape(" ".join(prediction_words[start_hyp:end_hyp]))
            if operation == "equal":
                parts.append(f"<span class='diff-equal'>{reference_chunk}</span>")
            elif operation == "delete":
                parts.append(f"<del class='diff-delete'>{reference_chunk}</del>")
            elif operation == "insert":
                parts.append(f"<ins class='diff-insert'>{prediction_chunk}</ins>")
            else:
                parts.append(
                    f"<del class='diff-delete'>{reference_chunk}</del> "
                    f"<ins class='diff-insert'>{prediction_chunk}</ins>"
                )
        return " ".join(part for part in parts if part)


def load_benchmark_view(comparison_path: Path, environment_path: Path) -> BenchmarkView:
    """Load only structured aggregate JSON, never Markdown or predictions."""

    try:
        comparison: object = json.loads(comparison_path.read_text(encoding="utf-8"))
        environment: object = json.loads(environment_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigError("Les artefacts structurés du benchmark sont indisponibles.") from exc
    if not isinstance(comparison, dict) or not isinstance(environment, dict):
        raise ConfigError("Les artefacts du benchmark doivent être des objets JSON.")
    serialized = json.dumps(comparison, ensure_ascii=False)
    forbidden = ("prediction_raw", "reference_raw", "/home/", "\\Users\\")
    if any(token in serialized for token in forbidden):
        raise ConfigError("Le benchmark agrégé contient une donnée privée.")
    raw_models = comparison.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ConfigError("Le benchmark ne contient aucun modèle.")
    rows: list[dict[str, Any]] = []
    for model in raw_models:
        if not isinstance(model, dict):
            raise ConfigError("Une ligne du benchmark est invalide.")
        rows.append(
            {
                "model": str(model["model_id"]).replace(
                    "openai/whisper-tiny",
                    "Whisper Tiny — baseline",
                ).replace(
                    "openai/whisper-small",
                    "Whisper Small — baseline",
                ),
                "type": "baseline",
                "audios": model["evaluated_audio_count"],
                "successes": model["successful_audio_count"],
                "wer_percent": float(model["wer_micro"]) * 100,
                "cer_percent": float(model["cer_micro"]) * 100,
                "rtf": float(model["rtf"]),
                "processing_time_seconds": float(model["processing_time_seconds"]),
                "device": model["device"],
                "date": model.get("generated_at_utc", comparison.get("generated_at_utc", "")),
                "seed": int(comparison.get("seed", 42)),
                "revision": model["model_revision"],
                "validation_loss": None,
                "substitutions": None,
                "insertions": None,
                "deletions": None,
                "wer_absolute_reduction_points": None,
                "wer_relative_reduction_percent": None,
                "cer_absolute_reduction_points": None,
                "cer_relative_reduction_percent": None,
            }
        )
    torch_info = environment.get("torch")
    hardware = (
        str(torch_info.get("gpu_name") or "CPU")
        if isinstance(torch_info, dict)
        else "Non disponible"
    )
    return BenchmarkView(
        rows=tuple(rows),
        dataset_name=str(comparison.get("dataset_name", "Dioula v0.1 local")),
        split=str(comparison.get("split", "test")),
        speaker_count=int(comparison["selected_speaker_count"]),
        seed=int(comparison.get("seed", 42)),
        hardware=hardware,
        run_date=str(comparison.get("generated_at_utc", "")),
        normalization=str(
            comparison.get(
                "normalization",
                "NFC, espaces normalisés, minuscules, ponctuation retirée, ↘ retiré",
            )
        ),
        experiment_id="historical_pilot",
        experiment_title="Expérience B — pilote historique",
        audio_count=int(comparison.get("selected_audio_count", len(rows))),
        comparison_scope=(
            "150 audios du pilote historique : Whisper Tiny baseline "
            "contre Whisper Small baseline."
        ),
    )


def load_pilot_adaptation_benchmark(comparison_path: Path) -> BenchmarkView:
    """Load the Phase 4C validation-only baseline/adapted comparison."""

    try:
        payload: object = json.loads(comparison_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigError("Le benchmark d'adaptation pilote est indisponible.") from exc
    if not isinstance(payload, dict):
        raise ConfigError("Le benchmark d'adaptation pilote doit être un objet JSON.")
    serialized = json.dumps(payload, ensure_ascii=False)
    if any(token in serialized for token in ("/home/", "\\Users\\", "speaker_name")):
        raise ConfigError("Le benchmark d'adaptation contient une donnée privée.")
    if payload.get("same_validation_subset") is not True:
        raise ConfigError("Les deux modèles ne partagent pas la même validation.")
    if any(
        payload.get(field) is not False
        for field in ("official_test_used", "pilot_test_used", "final_holdout_used")
    ):
        raise ConfigError("Un split interdit apparaît dans le benchmark d'adaptation.")

    baseline = payload.get("baseline")
    adapted = payload.get("adapted")
    if not isinstance(baseline, dict) or not isinstance(adapted, dict):
        raise ConfigError("Les métriques baseline/adapté sont incomplètes.")
    audio_count = int(payload.get("validation_audio_count", 0))
    if audio_count <= 0:
        raise ConfigError("Le nombre d'audios de validation est invalide.")

    baseline_wer = float(baseline["wer_micro"])
    adapted_wer = float(adapted["wer_micro"])
    baseline_cer = float(baseline["cer_micro"])
    adapted_cer = float(adapted["cer_micro"])
    wer_relative = relative_reduction_percent(baseline_wer, adapted_wer)
    cer_relative = relative_reduction_percent(baseline_cer, adapted_cer)
    expected_wer_relative = float(payload["wer_relative_reduction_percent"])
    expected_cer_relative = float(payload["cer_relative_reduction_percent"])
    if (
        wer_relative is None
        or cer_relative is None
        or abs(wer_relative - expected_wer_relative) > 1e-9
        or abs(cer_relative - expected_cer_relative) > 1e-9
    ):
        raise ConfigError("Les réductions relatives du benchmark sont incohérentes.")

    common = {
        "audios": audio_count,
        "successes": audio_count,
        "date": "Non enregistré",
        "seed": 42,
        "device": str(
            cast(dict[str, Any], payload.get("hardware", {})).get("device", "Non disponible")
        ),
    }
    rows = (
        {
            **common,
            "model": "Whisper Tiny — baseline",
            "type": "baseline",
            "wer_percent": baseline_wer * 100,
            "cer_percent": baseline_cer * 100,
            "rtf": float(baseline["rtf"]),
            "processing_time_seconds": float(baseline["processing_time_seconds"]),
            "revision": str(payload["model_revision"]),
            "validation_loss": float(baseline["validation_loss"]),
            "substitutions": int(baseline["word_substitutions"]),
            "insertions": int(baseline["word_insertions"]),
            "deletions": int(baseline["word_deletions"]),
            "wer_absolute_reduction_points": None,
            "wer_relative_reduction_percent": None,
            "cer_absolute_reduction_points": None,
            "cer_relative_reduction_percent": None,
        },
        {
            **common,
            "model": "Whisper Tiny Dioula — adapté pilote",
            "type": "pilote adapté",
            "wer_percent": adapted_wer * 100,
            "cer_percent": adapted_cer * 100,
            "rtf": float(adapted["rtf"]),
            "processing_time_seconds": float(adapted["processing_time_seconds"]),
            "revision": str(payload["best_checkpoint_name"]),
            "validation_loss": float(adapted["validation_loss"]),
            "substitutions": int(adapted["word_substitutions"]),
            "insertions": int(adapted["word_insertions"]),
            "deletions": int(adapted["word_deletions"]),
            "wer_absolute_reduction_points": float(
                payload["wer_absolute_reduction"]
            )
            * 100,
            "wer_relative_reduction_percent": expected_wer_relative,
            "cer_absolute_reduction_points": float(
                payload["cer_absolute_reduction"]
            )
            * 100,
            "cer_relative_reduction_percent": expected_cer_relative,
        },
    )
    hardware = payload.get("hardware")
    hardware_label = (
        str(hardware.get("gpu_name") or hardware.get("device") or "Non disponible")
        if isinstance(hardware, dict)
        else "Non disponible"
    )
    return BenchmarkView(
        rows=rows,
        dataset_name="Validation pilote dioula",
        split="validation",
        speaker_count=3,
        seed=42,
        hardware=hardware_label,
        run_date="Non enregistré",
        normalization=(
            "target_text_mvp ; NFC ; minuscules et ponctuation retirée "
            "pour le calcul WER/CER"
        ),
        experiment_id="pilot_adaptation_validation",
        experiment_title="Expérience A — validation pilote",
        audio_count=audio_count,
        comparison_scope=(
            "600 mêmes audios et références : Whisper Tiny baseline "
            "contre Whisper Tiny adapté pilote."
        ),
    )


def load_error_samples(tiny_csv: Path, small_csv: Path) -> tuple[ErrorSample, ...]:
    """Join local private pilot rows while exposing only anonymized identifiers."""

    def read_rows(path: Path) -> dict[str, dict[str, str]]:
        try:
            with path.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                required = {
                    "utterance_id",
                    "audio_path",
                    "reference_raw",
                    "prediction_raw",
                    "status",
                }
                if not required.issubset(reader.fieldnames or []):
                    raise ConfigError("Le fichier privé d'analyse est incomplet.")
                return {row["utterance_id"]: row for row in reader if row["status"] == "success"}
        except (OSError, UnicodeError, csv.Error) as exc:
            raise ConfigError("Les prédictions privées sont indisponibles.") from exc

    tiny = read_rows(tiny_csv)
    small = read_rows(small_csv)
    common_ids = sorted(set(tiny).intersection(small))
    samples: list[ErrorSample] = []
    for audio_id in common_ids:
        tiny_row = tiny[audio_id]
        small_row = small[audio_id]
        if tiny_row["reference_raw"] != small_row["reference_raw"]:
            raise ConfigError("Les références privées ne correspondent pas.")
        samples.append(
            ErrorSample(
                audio_id=audio_id,
                relative_audio_path=tiny_row["audio_path"],
                reference=tiny_row["reference_raw"],
                predictions={
                    "Whisper Tiny — baseline": tiny_row["prediction_raw"],
                    "Whisper Small — baseline": small_row["prediction_raw"],
                },
            )
        )
    if not samples:
        raise ConfigError("Aucun échantillon commun n'est disponible.")
    return tuple(samples)


def load_adaptation_error_samples(
    predictions_path: Path,
    manifest_path: Path,
) -> tuple[ErrorSample, ...]:
    """Join shareable Phase 4C predictions to private validation audio paths."""

    try:
        with manifest_path.open(encoding="utf-8", newline="") as stream:
            manifest_reader = csv.DictReader(stream)
            required_manifest = {"utterance_id", "audio_path", "split"}
            if not required_manifest.issubset(manifest_reader.fieldnames or []):
                raise ConfigError("Le manifeste privé est incomplet.")
            validation_paths = {
                row["utterance_id"]: row["audio_path"]
                for row in manifest_reader
                if row["split"] == "validation"
            }
        with predictions_path.open(encoding="utf-8", newline="") as stream:
            prediction_reader = csv.DictReader(stream)
            required_predictions = {
                "audio_id_anonymized",
                "target_text_mvp",
                "baseline_prediction",
                "adapted_prediction",
            }
            if not required_predictions.issubset(prediction_reader.fieldnames or []):
                raise ConfigError("Les prédictions d'adaptation sont incomplètes.")
            prediction_rows = list(prediction_reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ConfigError("L'analyse d'erreurs adaptée est indisponible.") from exc

    samples: list[ErrorSample] = []
    for row in prediction_rows:
        audio_id = row["audio_id_anonymized"]
        relative_audio_path = validation_paths.get(audio_id)
        if relative_audio_path is None:
            raise ConfigError("Une prédiction ne correspond pas au split validation.")
        samples.append(
            ErrorSample(
                audio_id=audio_id,
                relative_audio_path=relative_audio_path,
                reference=row["target_text_mvp"],
                predictions={
                    "Whisper Tiny — baseline": row["baseline_prediction"],
                    "Whisper Tiny Dioula — adapté pilote": row["adapted_prediction"],
                },
            )
        )
    if not samples:
        raise ConfigError("Aucun échantillon d'adaptation n'est disponible.")
    return tuple(samples)
