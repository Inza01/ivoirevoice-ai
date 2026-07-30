"""Reference-aware metrics and structured benchmark/error-analysis loading."""

from __future__ import annotations

import csv
import html
import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True, slots=True)
class ErrorSample:
    """One private local sample joined across baseline predictions."""

    audio_id: str
    relative_audio_path: str
    reference: str
    predictions: dict[str, str]


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
                "model": model["model_id"],
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
