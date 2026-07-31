"""Strict configuration for the Phase 4B Whisper Tiny smoke-overfit."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from ivoirevoice.exceptions import ConfigError


@dataclass(frozen=True, slots=True)
class SmokeSettings:
    """Validated paths and hyperparameters for one local smoke experiment."""

    experiment_id: str
    language: str
    split: str
    seed: int
    model_config_path: Path
    expected_model_id: str
    manifest_path: Path
    dataset_metadata_path: Path
    dataset_root: Path
    artifacts_root: Path
    reports_root: Path
    output_relative_directory: Path
    report_output_directory: Path
    pilot_prediction_files: tuple[Path, ...]
    sample_count: int
    minimum_correct_samples: int
    canonical_text_column: str
    max_steps: int
    batch_size: int
    learning_rate: float
    max_grad_norm: float
    logging_steps: int
    evaluation_steps: int
    mixed_precision: str
    stop_on_nan: bool
    save_model: bool
    publication_allowed: bool


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ConfigError(f"La section '{name}' doit être un objet YAML.")
    return cast(dict[str, Any], dict(value))


def _required_string(data: dict[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Le champ 'experiment.{field}' doit être une chaîne non vide.")
    return value.strip()


def _positive_int(data: dict[str, Any], field: str) -> int:
    value = data.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"Le champ 'experiment.{field}' doit être un entier positif.")
    return value


def _positive_float(data: dict[str, Any], field: str) -> float:
    value = data.get(field)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or float(value) <= 0
    ):
        raise ConfigError(f"Le champ 'experiment.{field}' doit être strictement positif.")
    return float(value)


def _safe_relative_path(value: str, field: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ConfigError(f"experiment.{field} doit être un chemin relatif sûr.")
    return path


def _environment_root(variable_name: str) -> Path:
    raw_value = os.getenv(variable_name)
    if not raw_value:
        raise ConfigError(f"La variable obligatoire '{variable_name}' n'est pas définie.")
    return Path(raw_value).expanduser().resolve()


def load_smoke_settings(path: str | Path) -> SmokeSettings:
    """Load and reject any configuration that could use the official test set."""

    config_path = Path(path)
    try:
        with config_path.open(encoding="utf-8") as stream:
            raw: object = yaml.safe_load(stream)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"Impossible de lire l'expérience smoke : {exc}") from exc
    root = _mapping(raw, "racine")
    experiment = _mapping(root.get("experiment"), "experiment")

    if _required_string(experiment, "mode") != "smoke_overfit":
        raise ConfigError("La Phase 4B exige experiment.mode=smoke_overfit.")
    if _required_string(experiment, "language") != "dyu":
        raise ConfigError("La Phase 4B est limitée au dioula.")
    if _required_string(experiment, "split") != "train":
        raise ConfigError("Le smoke-overfit doit utiliser exclusivement le split train.")
    if _required_string(experiment, "expected_model_id") != "openai/whisper-tiny":
        raise ConfigError("La Phase 4B autorise uniquement openai/whisper-tiny.")
    if experiment.get("publication_allowed") is not False:
        raise ConfigError("experiment.publication_allowed doit rester false.")
    if experiment.get("save_model") is not False:
        raise ConfigError("Le smoke test ne doit pas sauvegarder de modèle.")
    if experiment.get("stop_on_nan") is not True:
        raise ConfigError("experiment.stop_on_nan doit rester true.")

    sample_count = _positive_int(experiment, "sample_count")
    if not 10 <= sample_count <= 20:
        raise ConfigError("experiment.sample_count doit être compris entre 10 et 20.")
    minimum_correct = _positive_int(experiment, "minimum_correct_samples")
    if not 10 <= minimum_correct <= sample_count:
        raise ConfigError(
            "experiment.minimum_correct_samples doit être compris entre 10 et sample_count."
        )
    canonical = _required_string(experiment, "canonical_text_column")
    if canonical != "target_text_mvp":
        raise ConfigError("La colonne canonique Phase 4B doit rester target_text_mvp.")
    mixed_precision = _required_string(experiment, "mixed_precision").lower()
    if mixed_precision not in {"auto", "no", "fp16", "bf16"}:
        raise ConfigError("experiment.mixed_precision doit valoir auto, no, fp16 ou bf16.")

    raw_pilot_files = experiment.get("pilot_prediction_files")
    if (
        not isinstance(raw_pilot_files, list)
        or not raw_pilot_files
        or not all(isinstance(item, str) and item.strip() for item in raw_pilot_files)
    ):
        raise ConfigError("experiment.pilot_prediction_files doit être une liste non vide.")
    artifacts_root = _environment_root("IVOIREVOICE_ARTIFACTS_DIR")
    pilot_files = tuple(
        artifacts_root / _safe_relative_path(item.strip(), "pilot_prediction_files")
        for item in raw_pilot_files
    )

    reports_override = os.getenv("IVOIREVOICE_TRAINING_REPORTS_DIR")
    reports_root = (
        Path(reports_override).expanduser().resolve()
        if reports_override
        else (Path.cwd() / "reports" / "data").resolve()
    )

    return SmokeSettings(
        experiment_id=_required_string(experiment, "id"),
        language="dyu",
        split="train",
        seed=_positive_int(experiment, "seed"),
        model_config_path=_safe_relative_path(
            _required_string(experiment, "model_config"), "model_config"
        ),
        expected_model_id="openai/whisper-tiny",
        manifest_path=artifacts_root
        / _safe_relative_path(_required_string(experiment, "manifest_path"), "manifest_path"),
        dataset_metadata_path=artifacts_root
        / _safe_relative_path(
            _required_string(experiment, "dataset_metadata_path"),
            "dataset_metadata_path",
        ),
        dataset_root=_environment_root("IVOIREVOICE_DIOULA_DATA_DIR"),
        artifacts_root=artifacts_root,
        reports_root=reports_root,
        output_relative_directory=_safe_relative_path(
            _required_string(experiment, "output_directory"),
            "output_directory",
        ),
        report_output_directory=(
            Path(__file__).resolve().parents[3]
            / _safe_relative_path(
                _required_string(experiment, "report_output_directory"),
                "report_output_directory",
            )
        ),
        pilot_prediction_files=pilot_files,
        sample_count=sample_count,
        minimum_correct_samples=minimum_correct,
        canonical_text_column=canonical,
        max_steps=_positive_int(experiment, "max_steps"),
        batch_size=_positive_int(experiment, "batch_size"),
        learning_rate=_positive_float(experiment, "learning_rate"),
        max_grad_norm=_positive_float(experiment, "max_grad_norm"),
        logging_steps=_positive_int(experiment, "logging_steps"),
        evaluation_steps=_positive_int(experiment, "evaluation_steps"),
        mixed_precision=mixed_precision,
        stop_on_nan=True,
        save_model=False,
        publication_allowed=False,
    )
