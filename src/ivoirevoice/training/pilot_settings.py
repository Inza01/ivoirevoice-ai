"""Strict Phase 4C pilot fine-tuning configuration."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from ivoirevoice.exceptions import ConfigError


@dataclass(frozen=True, slots=True)
class PilotSettings:
    """Validated experiment, storage and bounded-training parameters."""

    experiment_id: str
    language: str
    train_split: str
    validation_split: str
    forbidden_split: str
    seed: int
    model_config_path: Path
    expected_model_id: str
    expected_model_revision: str
    manifest_path: Path
    dataset_metadata_path: Path
    pilot_prediction_file: Path
    dataset_root: Path
    artifacts_root: Path
    artifact_output_directory: Path
    report_output_directory: Path
    checkpoint_directory: Path
    train_sample_count: int
    validation_sample_count: int
    expected_pilot_test_count: int
    expected_final_holdout_count: int
    canonical_text_column: str
    num_train_epochs: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    max_grad_norm: float
    logging_steps: int
    evaluation_steps: int
    fp16: bool
    gradient_checkpointing: bool
    early_stopping_patience: int
    early_stopping_threshold: float
    save_total_limit: int
    max_audio_seconds: float
    minimum_free_disk_gib: float
    resume_from_checkpoint: bool
    task: str
    forced_language_token: None
    post_correction: bool
    publication_allowed: bool


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ConfigError(f"La section '{name}' doit être un objet YAML.")
    return cast(dict[str, Any], dict(value))


def _required_string(data: Mapping[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"experiment.{field} doit être une chaîne non vide.")
    return value.strip()


def _positive_int(data: Mapping[str, Any], field: str) -> int:
    value = data.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"experiment.{field} doit être un entier positif.")
    return value


def _number(data: Mapping[str, Any], field: str, *, allow_zero: bool = False) -> float:
    value = data.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"experiment.{field} doit être numérique.")
    numeric = float(value)
    if numeric < 0 or (not allow_zero and numeric == 0):
        qualifier = "positif ou nul" if allow_zero else "strictement positif"
        raise ConfigError(f"experiment.{field} doit être {qualifier}.")
    return numeric


def _required_bool(data: Mapping[str, Any], field: str, expected: bool) -> bool:
    value = data.get(field)
    if value is not expected:
        raise ConfigError(f"experiment.{field} doit rester {str(expected).lower()}.")
    return bool(expected)


def _safe_relative(value: str, field: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ConfigError(f"experiment.{field} doit être un chemin relatif sûr.")
    return path


def _environment_root(name: str) -> Path:
    value = os.getenv(name)
    if not value:
        raise ConfigError(f"La variable obligatoire '{name}' n'est pas définie.")
    return Path(value).expanduser().resolve()


def load_pilot_settings(path: str | Path) -> PilotSettings:
    """Load Phase 4C while rejecting test use and unbounded training."""

    config_path = Path(path)
    try:
        with config_path.open(encoding="utf-8") as stream:
            raw: object = yaml.safe_load(stream)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"Impossible de lire la configuration pilote : {exc}") from exc
    experiment = _mapping(_mapping(raw, "racine").get("experiment"), "experiment")
    exact_strings = {
        "mode": "pilot_finetune",
        "language": "dyu",
        "train_split": "train",
        "validation_split": "validation",
        "forbidden_split": "test",
        "expected_model_id": "openai/whisper-tiny",
        "canonical_text_column": "target_text_mvp",
        "task": "transcribe",
    }
    for field, expected in exact_strings.items():
        if _required_string(experiment, field) != expected:
            raise ConfigError(f"experiment.{field} doit rester {expected!r}.")
    expected_revision = _required_string(experiment, "expected_model_revision")
    if len(expected_revision) != 40:
        raise ConfigError("La révision Whisper Tiny doit être un commit SHA de 40 caractères.")
    train_count = _positive_int(experiment, "train_sample_count")
    validation_count = _positive_int(experiment, "validation_sample_count")
    epochs = _positive_int(experiment, "num_train_epochs")
    if not 2_000 <= train_count <= 2_500:
        raise ConfigError("train_sample_count doit être compris entre 2000 et 2500.")
    if not 500 <= validation_count <= 800:
        raise ConfigError("validation_sample_count doit être compris entre 500 et 800.")
    if epochs > 2:
        raise ConfigError("Le pilote est limité à deux époques.")
    warmup_ratio = _number(experiment, "warmup_ratio", allow_zero=True)
    if warmup_ratio >= 1:
        raise ConfigError("warmup_ratio doit être inférieur à 1.")
    forced_language = experiment.get("forced_language_token")
    if forced_language is not None:
        raise ConfigError("Aucun token de langue dyu ne doit être forcé.")
    artifacts_root = _environment_root("IVOIREVOICE_ARTIFACTS_DIR")
    checkpoint_variable = _required_string(
        experiment, "checkpoint_environment_variable"
    )
    checkpoint_root = _environment_root(checkpoint_variable)
    repository_root = Path(__file__).resolve().parents[3]

    return PilotSettings(
        experiment_id=_required_string(experiment, "id"),
        language="dyu",
        train_split="train",
        validation_split="validation",
        forbidden_split="test",
        seed=_positive_int(experiment, "seed"),
        model_config_path=_safe_relative(
            _required_string(experiment, "model_config"), "model_config"
        ),
        expected_model_id="openai/whisper-tiny",
        expected_model_revision=expected_revision,
        manifest_path=artifacts_root
        / _safe_relative(_required_string(experiment, "manifest_path"), "manifest_path"),
        dataset_metadata_path=artifacts_root
        / _safe_relative(
            _required_string(experiment, "dataset_metadata_path"),
            "dataset_metadata_path",
        ),
        pilot_prediction_file=artifacts_root
        / _safe_relative(
            _required_string(experiment, "pilot_prediction_file"),
            "pilot_prediction_file",
        ),
        dataset_root=_environment_root("IVOIREVOICE_DIOULA_DATA_DIR"),
        artifacts_root=artifacts_root,
        artifact_output_directory=artifacts_root
        / _safe_relative(
            _required_string(experiment, "artifact_output_directory"),
            "artifact_output_directory",
        ),
        report_output_directory=repository_root
        / _safe_relative(
            _required_string(experiment, "report_output_directory"),
            "report_output_directory",
        ),
        checkpoint_directory=checkpoint_root
        / _safe_relative(_required_string(experiment, "id"), "id"),
        train_sample_count=train_count,
        validation_sample_count=validation_count,
        expected_pilot_test_count=_positive_int(
            experiment, "expected_pilot_test_count"
        ),
        expected_final_holdout_count=_positive_int(
            experiment, "expected_final_holdout_count"
        ),
        canonical_text_column="target_text_mvp",
        num_train_epochs=epochs,
        learning_rate=_number(experiment, "learning_rate"),
        warmup_ratio=warmup_ratio,
        weight_decay=_number(experiment, "weight_decay", allow_zero=True),
        train_batch_size=_positive_int(experiment, "train_batch_size"),
        eval_batch_size=_positive_int(experiment, "eval_batch_size"),
        gradient_accumulation_steps=_positive_int(
            experiment, "gradient_accumulation_steps"
        ),
        max_grad_norm=_number(experiment, "max_grad_norm"),
        logging_steps=_positive_int(experiment, "logging_steps"),
        evaluation_steps=_positive_int(experiment, "evaluation_steps"),
        fp16=_required_bool(experiment, "fp16", True),
        gradient_checkpointing=_required_bool(
            experiment, "gradient_checkpointing", True
        ),
        early_stopping_patience=_positive_int(
            experiment, "early_stopping_patience"
        ),
        early_stopping_threshold=_number(
            experiment, "early_stopping_threshold", allow_zero=True
        ),
        save_total_limit=_positive_int(experiment, "save_total_limit"),
        max_audio_seconds=_number(experiment, "max_audio_seconds"),
        minimum_free_disk_gib=_number(experiment, "minimum_free_disk_gib"),
        resume_from_checkpoint=_required_bool(
            experiment, "resume_from_checkpoint", True
        ),
        task="transcribe",
        forced_language_token=None,
        post_correction=_required_bool(experiment, "post_correction", False),
        publication_allowed=_required_bool(
            experiment, "publication_allowed", False
        ),
    )
