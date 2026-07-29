"""Deterministic, gender-aware speaker split proposal."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from ivoirevoice.data.records import ManifestRow
from ivoirevoice.data.settings import SplitSettings

SPLIT_NAMES = ("train", "validation", "test")


def _group_sizes(
    speaker_count: int,
    settings: SplitSettings,
) -> tuple[dict[str, int], str | None]:
    if speaker_count == 0:
        return dict.fromkeys(SPLIT_NAMES, 0), "groupe vide"
    if speaker_count == 1:
        return {"train": 1, "validation": 0, "test": 0}, "groupe trop petit"
    if speaker_count == 2:
        return {"train": 1, "validation": 1, "test": 0}, "groupe trop petit"

    validation_count = max(1, round(speaker_count * settings.validation_ratio))
    test_count = max(1, round(speaker_count * settings.test_ratio))
    while validation_count + test_count >= speaker_count:
        if validation_count >= test_count and validation_count > 1:
            validation_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            break
    return {
        "train": speaker_count - validation_count - test_count,
        "validation": validation_count,
        "test": test_count,
    }, None


def propose_speaker_split(
    rows: tuple[ManifestRow, ...],
    settings: SplitSettings,
) -> dict[str, Any]:
    """Propose disjoint speaker groups without modifying manifest rows."""

    speaker_gender: dict[str, str] = {}
    record_counts: dict[str, int] = defaultdict(int)
    durations: dict[str, float] = defaultdict(float)
    seen_audio_paths: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        speaker_gender[row.speaker_id] = row.gender_folder
        record_counts[row.speaker_id] += 1
        if row.audio_path and row.audio_path not in seen_audio_paths[row.speaker_id]:
            durations[row.speaker_id] += row.duration_seconds or 0.0
            seen_audio_paths[row.speaker_id].add(row.audio_path)

    by_gender: dict[str, list[str]] = defaultdict(list)
    for speaker_id, gender in speaker_gender.items():
        by_gender[gender].append(speaker_id)

    assignments: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    warnings: list[str] = []
    for gender in sorted(by_gender):
        speakers = sorted(by_gender[gender])
        random.Random(f"{settings.seed}:{gender}").shuffle(speakers)
        sizes, warning = _group_sizes(len(speakers), settings)
        if warning:
            warnings.append(f"{gender}: {warning} ({len(speakers)} locuteur(s))")
        train_end = sizes["train"]
        validation_end = train_end + sizes["validation"]
        assignments["train"].extend(speakers[:train_end])
        assignments["validation"].extend(speakers[train_end:validation_end])
        assignments["test"].extend(speakers[validation_end:])

    for split_name in SPLIT_NAMES:
        assignments[split_name].sort()

    all_assigned = [speaker for split in assignments.values() for speaker in split]
    leakage_free = len(all_assigned) == len(set(all_assigned)) == len(speaker_gender)
    gender_counts = {
        split_name: {
            gender: sum(speaker_gender[speaker] == gender for speaker in speakers)
            for gender in sorted(by_gender)
        }
        for split_name, speakers in assignments.items()
    }
    return {
        "status": "proposal_only_human_validation_required",
        "seed": settings.seed,
        "ratios": {
            "train": settings.train_ratio,
            "validation": settings.validation_ratio,
            "test": settings.test_ratio,
        },
        "speaker_ids": assignments,
        "speaker_counts": {
            split_name: len(speakers) for split_name, speakers in assignments.items()
        },
        "record_counts": {
            split_name: sum(record_counts[speaker] for speaker in speakers)
            for split_name, speakers in assignments.items()
        },
        "duration_seconds": {
            split_name: sum(durations[speaker] for speaker in speakers)
            for split_name, speakers in assignments.items()
        },
        "gender_speaker_counts": gender_counts,
        "leakage_free": leakage_free,
        "warnings": warnings,
    }
