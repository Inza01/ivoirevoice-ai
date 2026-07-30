from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from ivoirevoice.exceptions import ConfigError
from ivoirevoice.training.audit import (
    AuditedDataset,
    ManifestRow,
    analyze_text,
    build_split_integrity_report,
    select_representative_train_rows,
    selection_sha256,
    validated_rows,
)
from ivoirevoice.training.settings import SmokeSettings


def _row(
    index: int,
    *,
    speaker: str,
    gender: str,
    split: str = "train",
    audio_hash: str | None = None,
) -> ManifestRow:
    return ManifestRow(
        utterance_id=f"utt_{index:04d}",
        speaker_id=speaker,
        gender_folder=gender,
        language="dyu",
        text_raw="Á hakili kà go ↘" if index % 2 else "A hakili ka go",
        text_no_tones="A hakili ka go ↘" if index % 2 else "A hakili ka go",
        target_text="A hakili ka go",
        audio_path=f"{gender}/anonymous/audio_{index:04d}.wav",
        duration_seconds=0.75 + index * 0.21,
        sample_rate_hz=16_000,
        channels=1,
        audio_sha256=audio_hash or f"{index + 1:064x}",
        split=split,
        usage_scope="local_research_only",
    )


def _representative_rows() -> tuple[ManifestRow, ...]:
    rows: list[ManifestRow] = []
    index = 0
    for gender, prefix in (("men", "m"), ("women", "w")):
        for speaker_number in range(4):
            for _ in range(5):
                rows.append(
                    _row(
                        index,
                        speaker=f"spk_{prefix}{speaker_number}",
                        gender=gender,
                    )
                )
                index += 1
    return tuple(rows)


def test_representative_selection_is_deterministic_balanced_and_train_only() -> None:
    rows = _representative_rows()

    selected = select_representative_train_rows(rows, count=16, seed=42)

    assert selected == select_representative_train_rows(rows, count=16, seed=42)
    assert len(selected) == len({row.utterance_id for row in selected}) == 16
    assert {row.split for row in selected} == {"train"}
    assert {row.gender_folder for row in selected} == {"men", "women"}
    assert len({row.speaker_id for row in selected}) == 8
    assert max(row.duration_seconds for row in selected) > 6
    assert min(row.duration_seconds for row in selected) < 2


def test_text_analysis_compares_variants_and_flags_unicode_conditions() -> None:
    rows = (
        _row(1, speaker="spk_m0", gender="men"),
        _row(2, speaker="spk_w0", gender="women"),
    )

    analysis = analyze_text(rows)

    assert analysis["row_count"] == 2
    assert analysis["different_raw_vs_no_tones"] == 1
    assert analysis["identical_raw_vs_no_tones"] == 1
    assert analysis["raw_with_tones"] == 1
    assert analysis["falling_marker_rows"] == 1
    assert analysis["target_empty"] == 0


def test_split_integrity_accepts_disjoint_splits_and_test_only_pilot() -> None:
    train = _row(1, speaker="spk_train", gender="men")
    validation = _row(
        2,
        speaker="spk_validation",
        gender="women",
        split="validation",
    )
    test = _row(3, speaker="spk_test", gender="women", split="test")
    dataset = AuditedDataset(
        rows=(train, validation, test),
        manifest_sha256="a" * 64,
        dataset_version="0.1.0-local",
    )

    report = build_split_integrity_report(dataset, (train,), {test.utterance_id})

    assert report["overall_passed"] is True
    assert report["counts"]["pilot_unique_audio_ids"] == 1
    assert all(check["passed"] for check in report["checks"].values())


@pytest.mark.parametrize("leak_kind", ["speaker", "hash"])
def test_split_integrity_detects_speaker_or_hash_leakage(leak_kind: str) -> None:
    train = _row(1, speaker="spk_train", gender="men")
    test = _row(
        2,
        speaker=train.speaker_id if leak_kind == "speaker" else "spk_test",
        gender="women",
        split="test",
        audio_hash=train.audio_sha256 if leak_kind == "hash" else None,
    )
    dataset = AuditedDataset(
        rows=(train, test),
        manifest_sha256="b" * 64,
        dataset_version="0.1.0-local",
    )

    report = build_split_integrity_report(dataset, (train,), {test.utterance_id})

    assert report["overall_passed"] is False
    check_name = (
        "speaker_ids_disjoint_across_splits"
        if leak_kind == "speaker"
        else "audio_sha256_disjoint_across_splits"
    )
    assert report["checks"][check_name]["passed"] is False


def test_training_gate_requires_ten_genuinely_correct_annotations() -> None:
    selected = _representative_rows()[:16]
    settings = cast(
        SmokeSettings,
        SimpleNamespace(minimum_correct_samples=10),
    )
    annotations = {
        "statuses": {
            row.utterance_id: {
                "status": "correct" if index < 9 else "à vérifier",
                "anomaly": "",
            }
            for index, row in enumerate(selected)
        }
    }

    with pytest.raises(ConfigError, match="9/10"):
        validated_rows(settings, selected, annotations)

    annotations["statuses"][selected[9].utterance_id]["status"] = "correct"
    assert len(validated_rows(settings, selected, annotations)) == 10


def test_selection_hash_changes_with_order() -> None:
    selected = _representative_rows()[:10]

    assert selection_sha256(selected) != selection_sha256(tuple(reversed(selected)))
