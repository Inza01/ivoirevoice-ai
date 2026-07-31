"""Deterministic Phase 4C subset selection without decoding test audio."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from statistics import median
from typing import Any

from ivoirevoice.exceptions import ConfigError
from ivoirevoice.training.audit import AuditedDataset, ManifestRow
from ivoirevoice.training.pilot_settings import PilotSettings

SELECTION_FILENAME = "pilot_selection.json"
PRIVATE_SELECTION_FILENAME = "pilot_selection_private.json"


@dataclass(frozen=True, slots=True)
class PilotSelection:
    """Frozen train/validation rows plus metadata-only test partitions."""

    train_rows: tuple[ManifestRow, ...]
    validation_rows: tuple[ManifestRow, ...]
    pilot_test_ids: frozenset[str]
    final_holdout_ids: frozenset[str]
    selection_sha256: str


def _stable_rank(row: ManifestRow, seed: int) -> str:
    return sha256(f"{seed}:{row.utterance_id}".encode()).hexdigest()


def _quantile_bin(rank: int, total: int, bin_count: int = 4) -> int:
    return min(bin_count - 1, rank * bin_count // max(1, total))


def _speaker_stratified_rows(
    rows: Sequence[ManifestRow],
    quota: int,
    seed: int,
) -> list[ManifestRow]:
    duration_order = sorted(rows, key=lambda row: (row.duration_seconds, row.utterance_id))
    text_order = sorted(rows, key=lambda row: (len(row.target_text.split()), row.utterance_id))
    duration_rank = {row.utterance_id: index for index, row in enumerate(duration_order)}
    text_rank = {row.utterance_id: index for index, row in enumerate(text_order)}
    strata: dict[tuple[int, int], list[ManifestRow]] = defaultdict(list)
    for row in rows:
        key = (
            _quantile_bin(duration_rank[row.utterance_id], len(rows)),
            _quantile_bin(text_rank[row.utterance_id], len(rows)),
        )
        strata[key].append(row)
    for values in strata.values():
        values.sort(key=lambda row: _stable_rank(row, seed))
    keys = sorted(strata)
    selected: list[ManifestRow] = []
    cursor = 0
    while len(selected) < quota:
        key = keys[cursor % len(keys)]
        if strata[key]:
            selected.append(strata[key].pop(0))
        cursor += 1
        if cursor > quota * len(keys) * 2:
            raise ConfigError("Impossible de satisfaire le quota stratifié.")
    return selected


def select_balanced_subset(
    rows: Sequence[ManifestRow],
    *,
    split: str,
    count: int,
    seed: int,
) -> tuple[ManifestRow, ...]:
    """Balance speakers exactly, then cover duration/text quartiles."""

    by_speaker: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in rows:
        if row.split == split:
            by_speaker[row.speaker_id].append(row)
    if not by_speaker:
        raise ConfigError(f"Aucun audio éligible dans le split {split}.")
    speakers = sorted(by_speaker, key=lambda value: sha256(value.encode()).hexdigest())
    quotas = _balanced_quotas(
        {speaker: len(by_speaker[speaker]) for speaker in speakers},
        count,
        speakers,
    )
    selected: list[ManifestRow] = []
    for index, speaker in enumerate(speakers):
        quota = quotas[speaker]
        selected.extend(
            _speaker_stratified_rows(by_speaker[speaker], quota, seed + index)
        )
    return tuple(sorted(selected, key=lambda row: _stable_rank(row, seed)))


def _balanced_quotas(
    availability: Mapping[str, int],
    count: int,
    ordered_speakers: Sequence[str],
) -> dict[str, int]:
    if sum(availability.values()) < count:
        raise ConfigError("Le split ne contient pas assez d'audios éligibles.")
    quotas = {speaker: 0 for speaker in ordered_speakers}
    active = list(ordered_speakers)
    remaining = count
    while active:
        base, remainder = divmod(remaining, len(active))
        capped = [
            speaker
            for speaker in active
            if availability[speaker] < base + (active.index(speaker) < remainder)
        ]
        if not capped:
            for index, speaker in enumerate(active):
                quotas[speaker] = base + (index < remainder)
            break
        for speaker in capped:
            quotas[speaker] = availability[speaker]
            remaining -= quotas[speaker]
            active.remove(speaker)
    if sum(quotas.values()) != count or any(quota <= 0 for quota in quotas.values()):
        raise ConfigError("Impossible de construire des quotas locuteurs équilibrés.")
    return quotas


def _eligible_rows(
    dataset: AuditedDataset,
    settings: PilotSettings,
) -> tuple[tuple[ManifestRow, ...], dict[str, int]]:
    reasons: Counter[str] = Counter()
    eligible: list[ManifestRow] = []
    for row in dataset.rows:
        if row.split not in {settings.train_split, settings.validation_split}:
            continue
        if not row.target_text.strip():
            reasons["empty_target"] += 1
        elif not 0 < row.duration_seconds <= settings.max_audio_seconds:
            reasons["duration_out_of_bounds"] += 1
        elif row.sample_rate_hz != 16_000:
            reasons["sample_rate_not_16khz"] += 1
        elif row.channels != 1:
            reasons["not_mono"] += 1
        elif not (settings.dataset_root / row.audio_path).is_file():
            reasons["audio_missing"] += 1
        else:
            eligible.append(row)
    return tuple(eligible), dict(reasons)


def _pilot_test_ids(path: Path) -> frozenset[str]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if "utterance_id" not in (reader.fieldnames or ()):
                raise ConfigError("Le fichier pilote historique ne contient pas utterance_id.")
            identifiers = frozenset(
                row["utterance_id"] for row in reader if row.get("utterance_id")
            )
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ConfigError(f"Impossible de lire le pilote historique : {exc}") from exc
    return identifiers


def _selection_hash(train: Sequence[ManifestRow], validation: Sequence[ManifestRow]) -> str:
    material = "\n".join(
        [*(f"train:{row.utterance_id}" for row in train)]
        + [*(f"validation:{row.utterance_id}" for row in validation)]
    )
    return sha256(material.encode()).hexdigest()


def build_pilot_selection(
    dataset: AuditedDataset,
    settings: PilotSettings,
) -> tuple[PilotSelection, dict[str, Any]]:
    """Freeze Phase 4C subsets and prove that no test audio can be decoded."""

    eligible, excluded_reasons = _eligible_rows(dataset, settings)
    train = select_balanced_subset(
        eligible,
        split=settings.train_split,
        count=settings.train_sample_count,
        seed=settings.seed,
    )
    validation = select_balanced_subset(
        eligible,
        split=settings.validation_split,
        count=settings.validation_sample_count,
        seed=settings.seed,
    )
    pilot_ids = _pilot_test_ids(settings.pilot_prediction_file)
    row_by_id = {row.utterance_id: row for row in dataset.rows}
    test_ids = {
        row.utterance_id for row in dataset.rows if row.split == settings.forbidden_split
    }
    missing_pilot = pilot_ids - row_by_id.keys()
    pilot_non_test = {
        identifier
        for identifier in pilot_ids
        if identifier in row_by_id
        and row_by_id[identifier].split != settings.forbidden_split
    }
    final_holdout = frozenset(test_ids - pilot_ids)
    train_ids = {row.utterance_id for row in train}
    validation_ids = {row.utterance_id for row in validation}
    train_hashes = {row.audio_sha256 for row in train}
    validation_hashes = {row.audio_sha256 for row in validation}
    test_hashes = {
        row.audio_sha256 for row in dataset.rows if row.split == settings.forbidden_split
    }
    violations = {
        "missing_pilot_ids": len(missing_pilot),
        "pilot_ids_outside_test": len(pilot_non_test),
        "train_validation_id_overlap": len(train_ids & validation_ids),
        "train_test_id_overlap": len(train_ids & test_ids),
        "validation_test_id_overlap": len(validation_ids & test_ids),
        "train_validation_hash_overlap": len(train_hashes & validation_hashes),
        "train_test_hash_overlap": len(train_hashes & test_hashes),
        "validation_test_hash_overlap": len(validation_hashes & test_hashes),
    }
    if len(pilot_ids) != settings.expected_pilot_test_count:
        raise ConfigError("Le pilot_test historique ne contient pas exactement 150 audios.")
    if len(final_holdout) != settings.expected_final_holdout_count:
        raise ConfigError("Le final_holdout ne contient pas exactement 2624 audios.")
    if any(violations.values()):
        raise ConfigError(f"Fuite de split détectée : {violations}.")
    digest = _selection_hash(train, validation)
    selection = PilotSelection(
        train_rows=train,
        validation_rows=validation,
        pilot_test_ids=pilot_ids,
        final_holdout_ids=final_holdout,
        selection_sha256=digest,
    )
    report = {
        "schema_version": 1,
        "seed": settings.seed,
        "manifest_sha256": dataset.manifest_sha256,
        "selection_sha256": digest,
        "train": _subset_summary(train),
        "validation": _subset_summary(validation),
        "pilot_test": {
            "audio_count": len(pilot_ids),
            "source": "historical_test_pilot_metadata_only",
            "audio_decoded": False,
        },
        "final_holdout": {
            "audio_count": len(final_holdout),
            "access_policy": "metadata_partition_only_never_decode_or_transcribe_phase_4c",
            "audio_decoded": False,
        },
        "excluded_invalid_train_or_validation": excluded_reasons,
        "integrity_violations": violations,
        "overall_passed": not any(violations.values()),
        "train_audio_ids": [row.utterance_id for row in train],
        "validation_audio_ids": [row.utterance_id for row in validation],
        "pilot_test_audio_ids": sorted(pilot_ids),
        "final_holdout_audio_ids": sorted(final_holdout),
        "privacy": {
            "contains_real_speaker_names": False,
            "contains_local_paths": False,
            "contains_audio": False,
        },
    }
    return selection, report


def _subset_summary(rows: Sequence[ManifestRow]) -> dict[str, Any]:
    durations = [row.duration_seconds for row in rows]
    word_counts = [len(row.target_text.split()) for row in rows]
    return {
        "audio_count": len(rows),
        "speaker_count": len({row.speaker_id for row in rows}),
        "audio_count_by_anonymized_speaker": dict(
            sorted(Counter(row.speaker_id for row in rows).items())
        ),
        "duration_seconds": {
            "minimum": min(durations),
            "median": median(durations),
            "maximum": max(durations),
        },
        "target_word_count": {
            "minimum": min(word_counts),
            "median": median(word_counts),
            "maximum": max(word_counts),
        },
    }


def write_selection_report(
    settings: PilotSettings,
    report: Mapping[str, Any],
) -> Path:
    """Persist private IDs externally and an aggregate-only public report."""

    settings.artifact_output_directory.mkdir(parents=True, exist_ok=True)
    private_path = settings.artifact_output_directory / PRIVATE_SELECTION_FILENAME
    private_path.write_text(
        json.dumps(dict(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    public_report = dict(report)
    for field in (
        "train_audio_ids",
        "validation_audio_ids",
        "pilot_test_audio_ids",
        "final_holdout_audio_ids",
    ):
        public_report.pop(field, None)
    for split in ("train", "validation"):
        summary = public_report.get(split)
        if isinstance(summary, dict):
            public_report[split] = {
                key: value
                for key, value in summary.items()
                if key != "audio_count_by_anonymized_speaker"
            }
    public_report["privacy"] = {
        "contains_audio": False,
        "contains_local_paths": False,
        "contains_real_speaker_names": False,
        "contains_sample_identifiers": False,
    }
    settings.report_output_directory.mkdir(parents=True, exist_ok=True)
    path = settings.report_output_directory / SELECTION_FILENAME
    path.write_text(
        json.dumps(public_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
