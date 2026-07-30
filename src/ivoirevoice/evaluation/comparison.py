"""Aggregate-only comparison of completed Dioula ASR pilot runs."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ivoirevoice.exceptions import ConfigError, IvoireVoiceError

PILOT_RUNS = (
    "baseline-dy-whisper-tiny-pilot",
    "baseline-dy-whisper-small-pilot",
)
NUMERIC_METRICS = (
    "evaluated_audio_count",
    "successful_audio_count",
    "failed_audio_count",
    "wer_micro",
    "cer_micro",
    "wer_macro_speakers",
    "cer_macro_speakers",
    "mean_latency_seconds",
    "latency_p50_seconds",
    "latency_p95_seconds",
    "audio_duration_seconds",
    "processing_time_seconds",
    "rtf",
    "estimated_full_processing_seconds",
)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Impossible de lire {label} : {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"{label} doit être un objet JSON.")
    return value


def _validated_run(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = _load_json(directory / "run_metadata.json", "les métadonnées du pilote")
    metrics = _load_json(directory / "metrics.json", "les métriques du pilote")
    if metadata.get("status") != "completed" or metadata.get("level") != "pilot":
        raise ConfigError("La comparaison exige deux pilotes terminés.")
    if metadata.get("publication_allowed") is not False:
        raise ConfigError("La comparaison refuse un run publiable.")
    if metrics.get("level") != "pilot":
        raise ConfigError("Le niveau des métriques doit être pilot.")
    if metrics.get("model_id") != metadata.get("model_id"):
        raise ConfigError("Le modèle des métriques ne correspond pas aux métadonnées.")
    for field in NUMERIC_METRICS:
        value = metrics.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ConfigError(f"La métrique {field} est absente ou invalide.")
    if metrics["evaluated_audio_count"] != metadata.get("selected_audio_count"):
        raise ConfigError("Le nombre d'audios évalués ne correspond pas à la sélection.")
    return metadata, metrics


def compare_pilot_runs(run_directories: tuple[Path, Path]) -> dict[str, Any]:
    """Compare two aggregate pilot outputs on one identical frozen selection."""

    runs = tuple(_validated_run(directory) for directory in run_directories)
    first_metadata = runs[0][0]
    for metadata, _metrics in runs[1:]:
        for field in ("manifest_sha256", "selection_sha256", "selected_audio_count"):
            if metadata.get(field) != first_metadata.get(field):
                raise ConfigError(f"Les pilotes ne partagent pas le champ {field}.")

    models = []
    for metadata, metrics in runs:
        models.append(
            {
                "model_id": metadata["model_id"],
                "model_revision": metadata["model_revision"],
                "device": metadata["device"],
                "torch_dtype": metadata["torch_dtype"],
                "pipeline_commit_sha": metadata["pipeline_commit_sha"],
                "pipeline_git_dirty": metadata["pipeline_git_dirty"],
                "pipeline_source_sha256": metadata["pipeline_source_sha256"],
                "generated_at_utc": metrics["generated_at_utc"],
                "batch_size": metrics["batch_size"],
                **{field: metrics[field] for field in NUMERIC_METRICS},
            }
        )
    ranked = sorted(
        models,
        key=lambda model: (
            model["wer_micro"],
            model["cer_micro"],
            model["rtf"],
            model["model_id"],
        ),
    )
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_name": "Dioula v0.1 local",
        "level": "pilot",
        "split": "test",
        "seed": first_metadata["seed"],
        "normalization": (
            "Unicode NFC, espaces normalisés, minuscules, ponctuation retirée, marqueur ↘ retiré"
        ),
        "manifest_sha256": first_metadata["manifest_sha256"],
        "selection_sha256": first_metadata["selection_sha256"],
        "selected_audio_count": first_metadata["selected_audio_count"],
        "selected_speaker_count": first_metadata["selected_speaker_count"],
        "models": models,
        "recommendation": {
            "model_id": ranked[0]["model_id"],
            "criterion": "lowest_wer_then_cer_then_rtf",
            "scope": "first_controlled_fine_tuning_validation",
            "fine_tuning_performed": False,
        },
        "privacy_checks": {
            "transcriptions_absent": True,
            "personal_paths_absent": True,
            "publication_allowed": False,
        },
    }
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    if "/home/" in serialized or "\\Users\\" in serialized:
        raise ConfigError("La comparaison contient un chemin personnel.")
    return report


def comparison_markdown(report: dict[str, Any]) -> str:
    """Render a shareable aggregate table without corpus text."""

    rows = "\n".join(
        (
            f"| `{model['model_id']}` | {model['wer_micro']:.4f} | "
            f"{model['cer_micro']:.4f} | {model['wer_macro_speakers']:.4f} | "
            f"{model['cer_macro_speakers']:.4f} | {model['rtf']:.4f} | "
            f"{model['failed_audio_count']} |"
        )
        for model in report["models"]
    )
    return f"""# Comparaison des pilotes ASR dioula

- niveau : `pilot`
- sélection identique : {report["selected_audio_count"]} audios /
  {report["selected_speaker_count"]} locuteurs
- manifeste SHA-256 : `{report["manifest_sha256"]}`
- sélection SHA-256 : `{report["selection_sha256"]}`

| Modèle | WER micro | CER micro | WER macro | CER macro | RTF | Échecs |
|---|---:|---:|---:|---:|---:|---:|
{rows}

Décision provisoire : retenir `{report["recommendation"]["model_id"]}` pour
la première validation d'adaptation contrôlée. Aucun fine-tuning n'a été
effectué pendant cette phase.

Cette synthèse ne contient ni transcription, ni prédiction, ni chemin local.
Le corpus, les prédictions et tout modèle dérivé restent non publiables.
"""


def _atomic_write(path: Path, content: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(path)
    except OSError as exc:
        raise ConfigError(f"Impossible d'écrire {path.name} : {exc}") from exc


def main() -> int:
    """Build JSON and Markdown comparisons from the two local pilot runs."""

    raw_root = os.getenv("IVOIREVOICE_ARTIFACTS_DIR")
    try:
        if not raw_root:
            raise ConfigError("IVOIREVOICE_ARTIFACTS_DIR doit être défini.")
        artifacts_root = Path(raw_root).expanduser().resolve()
        run_directories = tuple(artifacts_root / "baselines" / run for run in PILOT_RUNS)
        report = compare_pilot_runs(run_directories)  # type: ignore[arg-type]
        output_root = artifacts_root / "reports" / "baselines"
        _atomic_write(
            output_root / "baseline_dy_pilot_comparison.json",
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        _atomic_write(
            output_root / "baseline_dy_pilot_comparison.md",
            comparison_markdown(report),
        )
    except IvoireVoiceError as exc:
        print(f"ERREUR: {exc}")
        return 1
    print(f"selected_audio_count={report['selected_audio_count']}")
    print(f"recommended_model={report['recommendation']['model_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
