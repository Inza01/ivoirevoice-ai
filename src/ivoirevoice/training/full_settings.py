"""Strict configuration for full Dioula development, refit and final evaluation."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from ivoirevoice.exceptions import ConfigError


@dataclass(frozen=True, slots=True)
class FullTrainingSettings:
    """Validated immutable contract for the full local training workflow."""

    config_path: Path
    experiment_id: str
    seed: int
    model_config_path: Path
    expected_model_id: str
    expected_model_revision: str
    expected_manifest_sha256: str
    expected_dataset_version: str
    manifest_path: Path
    dataset_metadata_path: Path
    historical_pilot_prediction_file: Path
    dataset_root: Path
    artifacts_root: Path
    artifact_output_directory: Path
    checkpoint_directory: Path
    initial_checkpoint_path: Path
    expected_initial_checkpoint_name: str
    train_split: str
    validation_split: str
    forbidden_split: str
    train_audio_count: int
    train_speaker_count: int
    validation_audio_count: int
    validation_speaker_count: int
    refit_audio_count: int
    refit_speaker_count: int
    historical_pilot_test_count: int
    final_holdout_count: int
    canonical_text_column: str
    development_epochs: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    max_grad_norm: float
    logging_steps: int
    evaluations_per_epoch: int
    early_stopping_patience: int
    save_total_limit: int
    refit_save_steps: int
    refit_save_total_limit: int
    max_audio_seconds: float
    minimum_free_disk_gib: float
    final_holdout_confirmation: str
    fp16: bool = True
    gradient_checkpointing: bool = True
    task: str = "transcribe"
    forced_language_token: None = None
    post_correction: bool = False
    publication_allowed: bool = False

    @property
    def development_checkpoint_directory(self) -> Path:
        return self.checkpoint_directory / "development"

    @property
    def refit_checkpoint_directory(self) -> Path:
        return self.checkpoint_directory / "refit"

    @property
    def shareable_output_directory(self) -> Path:
        """Aggregate-only runtime reports, still kept outside Git."""

        return self.artifact_output_directory / "shareable"


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ConfigError(f"La section '{label}' doit être un objet YAML.")
    return cast(dict[str, Any], dict(value))


def _string(data: Mapping[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"experiment.{field} doit être une chaîne non vide.")
    return value.strip()


def _positive_int(data: Mapping[str, Any], field: str) -> int:
    value = data.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"experiment.{field} doit être un entier positif.")
    return value


def _number(data: Mapping[str, Any], field: str, *, zero_allowed: bool = False) -> float:
    value = data.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"experiment.{field} doit être numérique.")
    result = float(value)
    if result < 0 or (result == 0 and not zero_allowed):
        qualifier = "positif ou nul" if zero_allowed else "strictement positif"
        raise ConfigError(f"experiment.{field} doit être {qualifier}.")
    return result


def _exact_bool(data: Mapping[str, Any], field: str, expected: bool) -> bool:
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
    raw_value = os.getenv(name)
    if not raw_value:
        raise ConfigError(f"La variable obligatoire '{name}' n'est pas définie.")
    return Path(raw_value).expanduser().resolve()


def _require_exact(data: Mapping[str, Any], field: str, expected: object) -> None:
    if data.get(field) != expected:
        raise ConfigError(f"experiment.{field} doit rester {expected!r}.")


def load_full_training_settings(path: str | Path) -> FullTrainingSettings:
    """Load the full workflow while rejecting test use and mutable protocol choices."""

    config_path = Path(path).resolve()
    try:
        raw: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"Impossible de lire la configuration complète : {exc}") from exc
    experiment = _mapping(_mapping(raw, "racine").get("experiment"), "experiment")
    exact_values: dict[str, object] = {
        "id": "full-finetune-whisper-tiny-dy",
        "mode": "full_finetune",
        "language": "dyu",
        "train_split": "train",
        "validation_split": "validation",
        "forbidden_split": "test",
        "seed": 42,
        "model_config": "configs/models/whisper_tiny.yaml",
        "expected_model_id": "openai/whisper-tiny",
        "expected_model_revision": "be0ba7c2f24f0127b27863a23a08002af4c2c279",
        "expected_manifest_sha256": (
            "3b680d108b8d2d106bf04708d79ea54599c9f998cba289356ae4ba0ff36e5572"
        ),
        "expected_dataset_version": "0.1.0-local",
        "checkpoint_environment_variable": "IVOIREVOICE_CHECKPOINT_DIR",
        "initial_checkpoint_environment_variable": (
            "IVOIREVOICE_DIOULA_PILOT_MODEL_PATH"
        ),
        "expected_initial_checkpoint_name": "checkpoint-000140",
        "canonical_text_column": "target_text_mvp",
        "task": "transcribe",
        "forced_language_token": None,
        "train_audio_count": 13_764,
        "train_speaker_count": 15,
        "validation_audio_count": 2_661,
        "validation_speaker_count": 3,
        "refit_audio_count": 16_425,
        "refit_speaker_count": 18,
        "historical_pilot_test_count": 150,
        "final_holdout_count": 2_624,
        "development_epochs": 2,
        "learning_rate": 0.00001,
        "warmup_ratio": 0.05,
        "weight_decay": 0.01,
        "train_batch_size": 4,
        "eval_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "max_grad_norm": 1.0,
        "logging_steps": 10,
        "evaluations_per_epoch": 4,
        "early_stopping_patience": 3,
        "save_total_limit": 3,
        "refit_save_steps": 250,
        "refit_save_total_limit": 2,
        "max_audio_seconds": 30.0,
        "minimum_free_disk_gib": 15.0,
        "final_holdout_confirmation": "EVALUATE_FROZEN_MODEL_ONCE",
    }
    for field, expected in exact_values.items():
        _require_exact(experiment, field, expected)
    revision = _string(experiment, "expected_model_revision")
    manifest_sha256 = _string(experiment, "expected_manifest_sha256")
    if len(revision) != 40 or len(manifest_sha256) != 64:
        raise ConfigError("Les empreintes modèle/manifeste n'ont pas la longueur attendue.")
    warmup_ratio = _number(experiment, "warmup_ratio", zero_allowed=True)
    if warmup_ratio >= 1:
        raise ConfigError("experiment.warmup_ratio doit être inférieur à 1.")
    artifacts_root = _environment_root("IVOIREVOICE_ARTIFACTS_DIR")
    checkpoint_root = _environment_root(_string(experiment, "checkpoint_environment_variable"))
    initial_checkpoint = _environment_root(
        _string(experiment, "initial_checkpoint_environment_variable")
    )
    expected_checkpoint_name = _string(experiment, "expected_initial_checkpoint_name")
    if initial_checkpoint.name != expected_checkpoint_name:
        raise ConfigError(
            f"Le checkpoint initial doit être {expected_checkpoint_name}, "
            f"pas {initial_checkpoint.name}."
        )
    settings = FullTrainingSettings(
        config_path=config_path,
        experiment_id=_string(experiment, "id"),
        seed=_positive_int(experiment, "seed"),
        model_config_path=_safe_relative(
            _string(experiment, "model_config"), "model_config"
        ),
        expected_model_id="openai/whisper-tiny",
        expected_model_revision=revision,
        expected_manifest_sha256=manifest_sha256,
        expected_dataset_version=_string(experiment, "expected_dataset_version"),
        manifest_path=artifacts_root
        / _safe_relative(_string(experiment, "manifest_path"), "manifest_path"),
        dataset_metadata_path=artifacts_root
        / _safe_relative(
            _string(experiment, "dataset_metadata_path"), "dataset_metadata_path"
        ),
        historical_pilot_prediction_file=artifacts_root
        / _safe_relative(
            _string(experiment, "historical_pilot_prediction_file"),
            "historical_pilot_prediction_file",
        ),
        dataset_root=_environment_root("IVOIREVOICE_DIOULA_DATA_DIR"),
        artifacts_root=artifacts_root,
        artifact_output_directory=artifacts_root
        / _safe_relative(
            _string(experiment, "artifact_output_directory"),
            "artifact_output_directory",
        ),
        checkpoint_directory=checkpoint_root
        / _safe_relative(_string(experiment, "id"), "id"),
        initial_checkpoint_path=initial_checkpoint,
        expected_initial_checkpoint_name=expected_checkpoint_name,
        train_split="train",
        validation_split="validation",
        forbidden_split="test",
        train_audio_count=13_764,
        train_speaker_count=15,
        validation_audio_count=2_661,
        validation_speaker_count=3,
        refit_audio_count=16_425,
        refit_speaker_count=18,
        historical_pilot_test_count=150,
        final_holdout_count=2_624,
        canonical_text_column="target_text_mvp",
        development_epochs=2,
        learning_rate=_number(experiment, "learning_rate"),
        warmup_ratio=warmup_ratio,
        weight_decay=_number(experiment, "weight_decay", zero_allowed=True),
        train_batch_size=_positive_int(experiment, "train_batch_size"),
        eval_batch_size=_positive_int(experiment, "eval_batch_size"),
        gradient_accumulation_steps=_positive_int(
            experiment, "gradient_accumulation_steps"
        ),
        max_grad_norm=_number(experiment, "max_grad_norm"),
        logging_steps=_positive_int(experiment, "logging_steps"),
        evaluations_per_epoch=_positive_int(experiment, "evaluations_per_epoch"),
        early_stopping_patience=_positive_int(
            experiment, "early_stopping_patience"
        ),
        save_total_limit=_positive_int(experiment, "save_total_limit"),
        refit_save_steps=_positive_int(experiment, "refit_save_steps"),
        refit_save_total_limit=_positive_int(
            experiment, "refit_save_total_limit"
        ),
        max_audio_seconds=_number(experiment, "max_audio_seconds"),
        minimum_free_disk_gib=_number(experiment, "minimum_free_disk_gib"),
        final_holdout_confirmation=_string(
            experiment, "final_holdout_confirmation"
        ),
        fp16=_exact_bool(experiment, "fp16", True),
        gradient_checkpointing=_exact_bool(
            experiment, "gradient_checkpointing", True
        ),
        post_correction=_exact_bool(experiment, "post_correction", False),
        publication_allowed=_exact_bool(
            experiment, "publication_allowed", False
        ),
    )
    if settings.refit_audio_count != (
        settings.train_audio_count + settings.validation_audio_count
    ):
        raise ConfigError("Le refit doit contenir exactement train + validation.")
    return settings
