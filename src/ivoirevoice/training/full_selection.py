"""Exhaustive, privacy-aware selections for full development and refit."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ivoirevoice.exceptions import ConfigError
from ivoirevoice.training.audit import AuditedDataset, ManifestRow
from ivoirevoice.training.full_settings import FullTrainingSettings
from ivoirevoice.training.whisper_finetune import selection_sha256, write_json_atomic

PRIVATE_SELECTION_FILENAME = "full_selection_private.json"
PUBLIC_SELECTION_FILENAME = "full_selection.json"


@dataclass(frozen=True, slots=True)
class FullSelection:
    """Complete train/validation rows and metadata-only test partitions."""

    train_rows: tuple[ManifestRow, ...]
    validation_rows: tuple[ManifestRow, ...]
    refit_rows: tuple[ManifestRow, ...]
    historical_pilot_ids: frozenset[str]
    final_holdout_ids: frozenset[str]
    development_selection_sha256: str
    refit_selection_sha256: str
    final_holdout_selection_sha256: str


def _historical_ids(path: Path) -> frozenset[str]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if "utterance_id" not in (reader.fieldnames or ()):
                raise ConfigError("Le pilote historique ne contient pas utterance_id.")
            return frozenset(
                row["utterance_id"] for row in reader if row.get("utterance_id")
            )
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ConfigError(f"Impossible de lire le pilote historique : {exc}") from exc


def _validate_training_row(row: ManifestRow, settings: FullTrainingSettings) -> None:
    if row.split not in {settings.train_split, settings.validation_split}:
        raise ConfigError("Un audio test a atteint la validation train/refit.")
    if not row.target_text.strip():
        raise ConfigError("Le dataset complet contient un label vide.")
    if not 0 < row.duration_seconds <= settings.max_audio_seconds:
        raise ConfigError("Le dataset complet contient une durée incompatible.")
    if row.sample_rate_hz != 16_000 or row.channels != 1:
        raise ConfigError("Tous les audios train/validation doivent être mono 16 kHz.")
    if row.usage_scope != "local_research_only":
        raise ConfigError("Le scope de gouvernance du dataset a changé.")
    _validate_audio_file(row, settings.dataset_root, stage="train/validation")


def _validate_audio_file(row: ManifestRow, dataset_root: Path, *, stage: str) -> None:
    audio_path = dataset_root / row.audio_path
    if not audio_path.is_file():
        raise ConfigError(f"Un audio {stage} attendu est absent.")
    digest = hashlib.sha256()
    try:
        with audio_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ConfigError(f"Un audio {stage} est illisible.") from exc
    if digest.hexdigest() != row.audio_sha256:
        raise ConfigError(f"Un audio {stage} ne correspond plus au manifeste.")


def validate_final_holdout_files(
    rows: Sequence[ManifestRow],
    settings: FullTrainingSettings,
) -> None:
    """Validate holdout bytes only inside the explicitly confirmed final event."""

    if len(rows) != settings.final_holdout_count:
        raise ConfigError("La sélection final_holdout a changé.")
    for row in rows:
        if row.split != settings.forbidden_split:
            raise ConfigError("Un élément final_holdout n'appartient pas au split test.")
        if row.sample_rate_hz != 16_000 or row.channels != 1:
            raise ConfigError("Un audio final_holdout n'est pas mono 16 kHz.")
        if not row.target_text.strip():
            raise ConfigError("Un label final_holdout est vide.")
        _validate_audio_file(row, settings.dataset_root, stage="final_holdout")


def _require_unique(rows: Sequence[ManifestRow], field: str) -> None:
    values = [
        row.utterance_id if field == "utterance_id" else row.audio_sha256 for row in rows
    ]
    duplicates = len(values) - len(set(values))
    if duplicates:
        raise ConfigError(f"{duplicates} doublons {field} détectés dans le dataset.")


def _summary(rows: Sequence[ManifestRow]) -> dict[str, Any]:
    return {
        "audio_count": len(rows),
        "speaker_count": len({row.speaker_id for row in rows}),
        "duration_seconds": sum(row.duration_seconds for row in rows),
        "gender_folder_counts": dict(Counter(row.gender_folder for row in rows)),
    }


def build_full_selection(
    dataset: AuditedDataset,
    settings: FullTrainingSettings,
) -> tuple[FullSelection, dict[str, Any], dict[str, Any]]:
    """Select every train/validation row and prove test isolation."""

    if dataset.manifest_sha256 != settings.expected_manifest_sha256:
        raise ConfigError("Le hash du manifeste complet n'est pas celui qui a été approuvé.")
    if dataset.dataset_version != settings.expected_dataset_version:
        raise ConfigError("La version du dataset complet n'est pas celle qui a été approuvée.")
    _require_unique(dataset.rows, "utterance_id")
    _require_unique(dataset.rows, "audio_sha256")

    train = tuple(row for row in dataset.rows if row.split == settings.train_split)
    validation = tuple(
        row for row in dataset.rows if row.split == settings.validation_split
    )
    test = tuple(row for row in dataset.rows if row.split == settings.forbidden_split)
    for row in (*train, *validation):
        _validate_training_row(row, settings)

    observed = {
        "train_audio_count": len(train),
        "train_speaker_count": len({row.speaker_id for row in train}),
        "validation_audio_count": len(validation),
        "validation_speaker_count": len({row.speaker_id for row in validation}),
        "refit_audio_count": len(train) + len(validation),
        "refit_speaker_count": len(
            {row.speaker_id for row in (*train, *validation)}
        ),
    }
    expected = {
        "train_audio_count": settings.train_audio_count,
        "train_speaker_count": settings.train_speaker_count,
        "validation_audio_count": settings.validation_audio_count,
        "validation_speaker_count": settings.validation_speaker_count,
        "refit_audio_count": settings.refit_audio_count,
        "refit_speaker_count": settings.refit_speaker_count,
    }
    if observed != expected:
        raise ConfigError(f"Comptes exhaustifs inattendus : {observed}.")

    train_speakers = {row.speaker_id for row in train}
    validation_speakers = {row.speaker_id for row in validation}
    test_speakers = {row.speaker_id for row in test}
    if (
        train_speakers & validation_speakers
        or train_speakers & test_speakers
        or validation_speakers & test_speakers
    ):
        raise ConfigError("Les locuteurs ne sont plus disjoints entre les splits.")

    historical = _historical_ids(settings.historical_pilot_prediction_file)
    test_by_id = {row.utterance_id: row for row in test}
    if len(historical) != settings.historical_pilot_test_count:
        raise ConfigError("Le pilote historique ne contient pas exactement 150 audios.")
    if historical - test_by_id.keys():
        raise ConfigError("Le pilote historique contient un identifiant hors test.")
    final_ids = frozenset(test_by_id.keys() - historical)
    if len(final_ids) != settings.final_holdout_count:
        raise ConfigError("Le final_holdout ne contient pas exactement 2624 audios.")

    development_hash = selection_sha256(
        {
            "train": tuple((row.utterance_id, row.audio_sha256) for row in train),
            "validation": tuple(
                (row.utterance_id, row.audio_sha256) for row in validation
            ),
        }
    )
    refit = tuple((*train, *validation))
    refit_hash = selection_sha256(
        {
            "refit": tuple((row.utterance_id, row.audio_sha256) for row in refit),
        }
    )
    holdout_hash = selection_sha256(
        {
            "final_holdout": tuple(
                (identifier, test_by_id[identifier].audio_sha256)
                for identifier in final_ids
            ),
        }
    )
    selection = FullSelection(
        train_rows=train,
        validation_rows=validation,
        refit_rows=refit,
        historical_pilot_ids=historical,
        final_holdout_ids=final_ids,
        development_selection_sha256=development_hash,
        refit_selection_sha256=refit_hash,
        final_holdout_selection_sha256=holdout_hash,
    )
    public_report = {
        "schema_version": 1,
        "dataset_version": dataset.dataset_version,
        "manifest_sha256": dataset.manifest_sha256,
        "train": _summary(train),
        "validation": _summary(validation),
        "refit": _summary(refit),
        "historical_test": {
            "audio_count": len(historical),
            "decoded": False,
        },
        "final_holdout": {
            "audio_count": len(final_ids),
            "decoded": False,
            "access_policy": "sealed_until_final_evaluation",
        },
        "integrity": {
            "speaker_disjoint": True,
            "utterance_id_unique": True,
            "audio_sha256_unique": True,
            "all_train_validation_audio_present": True,
        },
        "privacy": {
            "contains_sample_identifiers": False,
            "contains_transcriptions": False,
            "contains_local_paths": False,
        },
    }
    private_report = {
        **public_report,
        "train_audio_ids": [row.utterance_id for row in train],
        "validation_audio_ids": [row.utterance_id for row in validation],
        "historical_test_audio_ids": sorted(historical),
        "final_holdout_audio_ids": sorted(final_ids),
        "development_selection_sha256": development_hash,
        "refit_selection_sha256": refit_hash,
        "final_holdout_selection_sha256": holdout_hash,
    }
    return selection, public_report, private_report


def write_full_selection_reports(
    settings: FullTrainingSettings,
    public_report: dict[str, Any],
    private_report: dict[str, Any],
) -> tuple[Path, Path]:
    """Persist aggregate public evidence and detailed private evidence separately."""

    private_path = settings.artifact_output_directory / PRIVATE_SELECTION_FILENAME
    public_path = settings.shareable_output_directory / PUBLIC_SELECTION_FILENAME
    write_json_atomic(private_path, private_report)
    write_json_atomic(public_path, public_report)
    return public_path, private_path
