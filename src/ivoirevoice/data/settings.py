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
class CurationSettings:
    """Paths and policy for producing the local training candidate."""

    source_manifest_relative_path: Path
    candidate_manifest_relative_path: Path
    metadata_relative_path: Path
    report_relative_directory: Path
    target_text: str
    recover_missing_audio: bool
    recovery_output_environment_variable: str


@dataclass(frozen=True, slots=True)
class FreezeSettings:
    """Immutable local dataset release policy and output paths."""

    split_comparison_relative_path: Path
    manifest_relative_path: Path
    metadata_relative_path: Path
    report_relative_path: Path
    split_report_relative_path: Path
    dataset_version: str
    dataset_status: str
    split_strategy: str
    expected_audio_count: int
    expected_speaker_count: int
    expected_speaker_counts: dict[str, int]
    publication_allowed: bool
    model_derivative_publication_allowed: bool


@dataclass(frozen=True, slots=True)
class DioulaDataSettings:
    """Resolved local paths and audit options."""

    dataset_root: Path
    artifacts_root: Path
    language: str
    license_status: str
    consent_status: str
    usage_scope: str
    hash_audio: bool
    split: SplitSettings
    curation: CurationSettings
    freeze: FreezeSettings
    manifest_relative_path: Path
    report_relative_directory: Path

    @property
    def manifest_path(self) -> Path:
        return self.artifacts_root / self.manifest_relative_path

    @property
    def report_directory(self) -> Path:
        return self.artifacts_root / self.report_relative_directory

    @property
    def source_manifest_path(self) -> Path:
        return self.artifacts_root / self.curation.source_manifest_relative_path

    @property
    def candidate_manifest_path(self) -> Path:
        return self.artifacts_root / self.curation.candidate_manifest_relative_path

    @property
    def candidate_metadata_path(self) -> Path:
        return self.artifacts_root / self.curation.metadata_relative_path

    @property
    def curation_report_directory(self) -> Path:
        return self.artifacts_root / self.curation.report_relative_directory

    @property
    def split_comparison_path(self) -> Path:
        return self.artifacts_root / self.freeze.split_comparison_relative_path

    @property
    def frozen_manifest_path(self) -> Path:
        return self.artifacts_root / self.freeze.manifest_relative_path

    @property
    def frozen_metadata_path(self) -> Path:
        return self.artifacts_root / self.freeze.metadata_relative_path

    @property
    def frozen_report_path(self) -> Path:
        return self.artifacts_root / self.freeze.report_relative_path

    @property
    def frozen_split_report_path(self) -> Path:
        return self.artifacts_root / self.freeze.split_report_relative_path


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


def _positive_int(data: dict[str, Any], field: str, section: str) -> int:
    value = data.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"Le champ '{section}.{field}' doit être un entier positif.")
    return value


def _required_bool(data: dict[str, Any], field: str, section: str) -> bool:
    value = data.get(field)
    if not isinstance(value, bool):
        raise ConfigError(f"Le champ '{section}.{field}' doit être un booléen.")
    return value


def _environment_path(variable_name: str) -> Path:
    raw_path = os.getenv(variable_name)
    if not raw_path:
        raise ConfigError(
            f"La variable d'environnement obligatoire '{variable_name}' n'est pas définie."
        )
    return Path(raw_path).expanduser().resolve()


def _relative_output_path(
    data: dict[str, Any],
    field: str,
    section: str = "artifacts",
) -> Path:
    value = Path(_required_string(data, field, section))
    if value.is_absolute() or ".." in value.parts:
        raise ConfigError(f"Le champ '{section}.{field}' doit être un chemin relatif sûr.")
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
    curation = _mapping(root.get("curation"), "curation")
    freeze = _mapping(root.get("freeze"), "freeze")
    split_data = _mapping(dataset.get("split"), "dataset.split")
    freeze_speaker_counts = _mapping(
        freeze.get("expected_speaker_counts"),
        "freeze.expected_speaker_counts",
    )

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

    recover_missing_audio = curation.get("recover_missing_audio")
    if not isinstance(recover_missing_audio, bool):
        raise ConfigError("Le champ 'curation.recover_missing_audio' doit être un booléen.")
    if recover_missing_audio:
        raise ConfigError("La Phase 3A.2 interdit la récupération et la conversion audio.")
    target_text = _required_string(curation, "target_text", "curation")
    if target_text != "text_without_tones_nfc":
        raise ConfigError("La Phase 3A.1 exige 'curation.target_text=text_without_tones_nfc'.")

    license_status = _required_string(dataset, "license_status", "dataset")
    consent_status = _required_string(dataset, "consent_status", "dataset")
    usage_scope = _required_string(dataset, "usage_scope", "dataset")
    if (license_status, consent_status, usage_scope) != (
        "unknown",
        "unknown",
        "local_research_only",
    ):
        raise ConfigError(
            "La Phase 3A.2 exige license_status=unknown, consent_status=unknown "
            "et usage_scope=local_research_only."
        )

    split_strategy = _required_string(freeze, "split_strategy", "freeze")
    if split_strategy != "B_15_3_3":
        raise ConfigError("La Phase 3A.2 exige le split 'B_15_3_3'.")
    dataset_version = _required_string(freeze, "dataset_version", "freeze")
    dataset_status = _required_string(freeze, "dataset_status", "freeze")
    if dataset_version != "0.1.0-local" or dataset_status != "frozen_candidate":
        raise ConfigError(
            "Le gel local exige dataset_version=0.1.0-local et dataset_status=frozen_candidate."
        )
    publication_allowed = _required_bool(freeze, "publication_allowed", "freeze")
    derivative_publication_allowed = _required_bool(
        freeze,
        "model_derivative_publication_allowed",
        "freeze",
    )
    if publication_allowed or derivative_publication_allowed:
        raise ConfigError("La publication du corpus ou d'un modèle dérivé doit rester interdite.")
    expected_speaker_counts = {
        split_name: _positive_int(
            freeze_speaker_counts,
            split_name,
            "freeze.expected_speaker_counts",
        )
        for split_name in ("train", "validation", "test")
    }
    expected_speaker_count = _positive_int(
        freeze,
        "expected_speaker_count",
        "freeze",
    )
    if expected_speaker_counts != {"train": 15, "validation": 3, "test": 3}:
        raise ConfigError("Le gel local exige exactement 15/3/3 locuteurs.")
    if sum(expected_speaker_counts.values()) != expected_speaker_count:
        raise ConfigError("Les nombres de locuteurs par split doivent couvrir le total attendu.")

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
        license_status=license_status,
        consent_status=consent_status,
        usage_scope=usage_scope,
        hash_audio=hash_audio,
        split=SplitSettings(
            seed=seed,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
        ),
        curation=CurationSettings(
            source_manifest_relative_path=_relative_output_path(
                curation,
                "source_manifest_path",
                "curation",
            ),
            candidate_manifest_relative_path=_relative_output_path(
                curation,
                "candidate_manifest_path",
                "curation",
            ),
            metadata_relative_path=_relative_output_path(
                curation,
                "metadata_path",
                "curation",
            ),
            report_relative_directory=_relative_output_path(
                curation,
                "report_directory",
                "curation",
            ),
            target_text=target_text,
            recover_missing_audio=recover_missing_audio,
            recovery_output_environment_variable=_required_string(
                curation,
                "recovery_output_environment_variable",
                "curation",
            ),
        ),
        freeze=FreezeSettings(
            split_comparison_relative_path=_relative_output_path(
                freeze,
                "split_comparison_path",
                "freeze",
            ),
            manifest_relative_path=_relative_output_path(
                freeze,
                "manifest_path",
                "freeze",
            ),
            metadata_relative_path=_relative_output_path(
                freeze,
                "metadata_path",
                "freeze",
            ),
            report_relative_path=_relative_output_path(
                freeze,
                "report_path",
                "freeze",
            ),
            split_report_relative_path=_relative_output_path(
                freeze,
                "split_report_path",
                "freeze",
            ),
            dataset_version=dataset_version,
            dataset_status=dataset_status,
            split_strategy=split_strategy,
            expected_audio_count=_positive_int(
                freeze,
                "expected_audio_count",
                "freeze",
            ),
            expected_speaker_count=expected_speaker_count,
            expected_speaker_counts=expected_speaker_counts,
            publication_allowed=publication_allowed,
            model_derivative_publication_allowed=derivative_publication_allowed,
        ),
        manifest_relative_path=_relative_output_path(artifacts, "manifest_path"),
        report_relative_directory=_relative_output_path(artifacts, "report_directory"),
    )
