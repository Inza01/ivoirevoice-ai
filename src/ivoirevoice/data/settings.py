"""Validated settings for the local Dioula data pipeline."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from ivoirevoice.exceptions import ConfigError


@dataclass(frozen=True, slots=True)
class SplitSettings:
    """Deterministic speaker-split parameters."""

    seed: int
    train_ratio: float
    validation_ratio: float
    test_ratio: float


@dataclass(frozen=True, slots=True)
class DioulaDataSettings:
    """Resolved local paths and audit options."""

    dataset_root: Path
    artifacts_root: Path
    language: str
    license_status: str
    usage_scope: str
    hash_audio: bool
    split: SplitSettings
    manifest_relative_path: Path
    report_relative_directory: Path

    @property
    def manifest_path(self) -> Path:
        return self.artifacts_root / self.manifest_relative_path

    @property
    def report_directory(self) -> Path:
        return self.artifacts_root / self.report_relative_directory


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ConfigError(f"La section '{name}' doit être un objet YAML.")
    return cast(dict[str, Any], dict(value))


def _required_string(data: dict[str, Any], field: str, section: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Le champ '{section}.{field}' doit être une chaîne non vide.")
    return value.strip()


def _ratio(data: dict[str, Any], field: str) -> float:
    value = data.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"Le champ 'dataset.split.{field}' doit être numérique.")
    numeric_value = float(value)
    if not 0 < numeric_value < 1:
        raise ConfigError(f"Le champ 'dataset.split.{field}' doit être compris entre 0 et 1.")
    return numeric_value


def _environment_path(variable_name: str) -> Path:
    raw_path = os.getenv(variable_name)
    if not raw_path:
        raise ConfigError(
            f"La variable d'environnement obligatoire '{variable_name}' n'est pas définie."
        )
    return Path(raw_path).expanduser().resolve()


def _relative_output_path(data: dict[str, Any], field: str) -> Path:
    value = Path(_required_string(data, field, "artifacts"))
    if value.is_absolute() or ".." in value.parts:
        raise ConfigError(f"Le champ 'artifacts.{field}' doit être un chemin relatif sûr.")
    return value


def load_dioula_settings(config_path: str | Path) -> DioulaDataSettings:
    """Load data settings while resolving local roots exclusively from the environment."""

    path = Path(config_path)
    try:
        with path.open(encoding="utf-8") as stream:
            raw: object = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Impossible de charger la configuration de données : {exc}") from exc

    root = _mapping(raw, "racine")
    dataset = _mapping(root.get("dataset"), "dataset")
    artifacts = _mapping(root.get("artifacts"), "artifacts")
    split_data = _mapping(dataset.get("split"), "dataset.split")

    data_env = _required_string(
        dataset,
        "source_environment_variable",
        "dataset",
    )
    artifacts_env = _required_string(
        artifacts,
        "root_environment_variable",
        "artifacts",
    )
    dataset_root = _environment_path(data_env)
    if not dataset_root.is_dir():
        raise ConfigError(f"Le répertoire indiqué par '{data_env}' n'existe pas.")

    artifacts_root = _environment_path(artifacts_env)
    if artifacts_root == dataset_root or artifacts_root.is_relative_to(dataset_root):
        raise ConfigError("Le répertoire d'artefacts ne peut pas se trouver dans le corpus brut.")
    hash_audio = dataset.get("hash_audio")
    hash_override = os.getenv("IVOIREVOICE_HASH_AUDIO")
    if hash_override is not None:
        normalized_override = hash_override.strip().lower()
        if normalized_override not in {"true", "false", "1", "0", "yes", "no"}:
            raise ConfigError("IVOIREVOICE_HASH_AUDIO doit être un booléen.")
        hash_audio = normalized_override in {"true", "1", "yes"}
    if not isinstance(hash_audio, bool):
        raise ConfigError("Le champ 'dataset.hash_audio' doit être un booléen.")

    seed = split_data.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ConfigError("Le champ 'dataset.split.seed' doit être un entier.")
    train_ratio = _ratio(split_data, "train_ratio")
    validation_ratio = _ratio(split_data, "validation_ratio")
    test_ratio = _ratio(split_data, "test_ratio")
    if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-9:
        raise ConfigError("Les ratios train, validation et test doivent totaliser 1.")

    return DioulaDataSettings(
        dataset_root=dataset_root,
        artifacts_root=artifacts_root,
        language=_required_string(dataset, "language_code", "dataset"),
        license_status=_required_string(dataset, "license_status", "dataset"),
        usage_scope=_required_string(dataset, "usage_scope", "dataset"),
        hash_audio=hash_audio,
        split=SplitSettings(
            seed=seed,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
        ),
        manifest_relative_path=_relative_output_path(artifacts, "manifest_path"),
        report_relative_directory=_relative_output_path(artifacts, "report_directory"),
    )
