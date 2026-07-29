from __future__ import annotations

import csv
from pathlib import Path

import pytest

from ivoirevoice.data.curation import curate_manifest, write_curation_outputs
from ivoirevoice.data.recovery import execute_recovery
from ivoirevoice.data.settings import (
    CurationSettings,
    DioulaDataSettings,
    SplitSettings,
)
from ivoirevoice.exceptions import ConfigError

DRAFT_FIELDS = [
    "utterance_id",
    "sentence_id",
    "speaker_id",
    "gender_folder",
    "language",
    "text_raw",
    "text_normalized",
    "audio_path",
    "audio_filename",
    "audio_match_status",
    "duration_seconds",
    "sample_rate_hz",
    "channels",
    "file_size_bytes",
    "audio_sha256",
    "audio_status",
    "source_json",
    "record_index",
    "license_status",
]


def _settings(tmp_path: Path) -> DioulaDataSettings:
    return DioulaDataSettings(
        dataset_root=tmp_path / "dataset",
        artifacts_root=tmp_path / "artifacts",
        language="dyu",
        license_status="unknown",
        usage_scope="local_research_only",
        hash_audio=True,
        split=SplitSettings(
            seed=42,
            train_ratio=0.8,
            validation_ratio=0.1,
            test_ratio=0.1,
        ),
        curation=CurationSettings(
            source_manifest_relative_path=Path("manifests/draft.csv"),
            candidate_manifest_relative_path=Path("manifests/candidate.csv"),
            metadata_relative_path=Path("manifests/metadata.json"),
            report_relative_directory=Path("reports/data_curation"),
            target_text="text_without_tones_nfc",
            recover_missing_audio=False,
            recovery_output_environment_variable="IVOIREVOICE_DIOULA_INTERIM_DIR",
        ),
        manifest_relative_path=Path("manifests/draft.csv"),
        report_relative_directory=Path("reports/data_audit"),
    )


def _row(
    utterance_id: str,
    *,
    audio_filename: str,
    audio_hash: str,
    text: str = "A\u0301n bɛ taa.\r",
    sentence_id: str = "sentence-shared",
    record_index: int = 0,
) -> dict[str, str]:
    return {
        "utterance_id": utterance_id,
        "sentence_id": sentence_id,
        "speaker_id": "spk_test",
        "gender_folder": "women",
        "language": "dyu",
        "text_raw": text,
        "text_normalized": text.strip(),
        "audio_path": f"women/speaker_01/wav/{audio_filename}",
        "audio_filename": audio_filename,
        "audio_match_status": "matched",
        "duration_seconds": "1.0",
        "sample_rate_hz": "16000",
        "channels": "1",
        "file_size_bytes": "32044",
        "audio_sha256": audio_hash,
        "audio_status": "readable",
        "source_json": "women/speaker_01/clips.json",
        "record_index": str(record_index),
        "license_status": "unknown",
    }


def _write_inputs(
    settings: DioulaDataSettings,
    rows: list[dict[str, str]],
    no_tones: dict[str, str],
) -> None:
    speaker = settings.dataset_root / "women" / "speaker_01"
    speaker.mkdir(parents=True)
    (speaker / "text-no-tones").write_text(
        "".join(
            f"{Path(filename).with_suffix('.mp4').name} {text}\n"
            for filename, text in no_tones.items()
        ),
        encoding="utf-8",
    )
    settings.source_manifest_path.parent.mkdir(parents=True)
    with settings.source_manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=DRAFT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_duplicate_reference_selects_deterministic_canonical_and_preserves_raw(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    audio_hash = "a" * 64
    later = _row(
        "utt_later",
        audio_filename="sample.wav",
        audio_hash=audio_hash,
        record_index=2,
    )
    earlier = _row(
        "utt_earlier",
        audio_filename="sample.wav",
        audio_hash=audio_hash,
        record_index=1,
    )
    _write_inputs(settings, [later, earlier], {"sample.wav": "An bɛ taa."})

    result = curate_manifest(settings)

    assert len(result.candidate_rows) == 1
    assert result.candidate_rows[0].utterance_id == "utt_earlier"
    assert result.candidate_rows[0].text_raw == "A\u0301n bɛ taa.\r"
    assert result.candidate_rows[0].text_with_tones_nfc == "Án bɛ taa."
    assert result.candidate_rows[0].text_without_tones_nfc == "An bɛ taa."
    assert result.candidate_rows[0].target_text_mvp == "An bɛ taa."
    assert len(result.duplicate_references) == 1


def test_same_hash_deduplicates_paths_but_same_sentence_with_distinct_audio_stays(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    rows = [
        _row("utt_a", audio_filename="a.wav", audio_hash="a" * 64),
        _row("utt_b", audio_filename="b.wav", audio_hash="a" * 64),
        _row("utt_c", audio_filename="c.wav", audio_hash="c" * 64),
    ]
    _write_inputs(
        settings,
        rows,
        {
            "a.wav": "An bɛ taa.",
            "b.wav": "An bɛ taa.",
            "c.wav": "An bɛ taa.",
        },
    )

    result = curate_manifest(settings)

    assert len(result.candidate_rows) == 2
    assert len(result.duplicate_hashes) == 1
    assert {row.sentence_id for row in result.candidate_rows} == {"sentence-shared"}


def test_same_audio_with_different_transcriptions_is_quarantined(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    rows = [
        _row("utt_a", audio_filename="a.wav", audio_hash="a" * 64, text="Án bɛ taa."),
        _row("utt_b", audio_filename="a.wav", audio_hash="a" * 64, text="U bɛ taa."),
    ]
    _write_inputs(settings, rows, {"a.wav": "An bɛ taa."})

    result = curate_manifest(settings)

    assert not result.candidate_rows
    assert result.conflict_group_count == 1
    assert len(result.conflicts) == 2
    assert len(result.quarantined) == 2


def test_same_hash_with_different_transcriptions_is_quarantined(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    rows = [
        _row("utt_a", audio_filename="a.wav", audio_hash="a" * 64, text="Án bɛ taa."),
        _row("utt_b", audio_filename="b.wav", audio_hash="a" * 64, text="U bɛ taa."),
    ]
    _write_inputs(
        settings,
        rows,
        {"a.wav": "An bɛ taa.", "b.wav": "U bɛ taa."},
    )

    result = curate_manifest(settings)

    assert not result.candidate_rows
    assert len(result.conflicts) == 2
    assert len(result.quarantined) == 2


def test_written_candidate_contains_no_url_absolute_path_or_split(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    _write_inputs(
        settings,
        [_row("utt_a", audio_filename="a.wav", audio_hash="a" * 64)],
        {"a.wav": "An bɛ taa."},
    )
    result = curate_manifest(settings)

    write_curation_outputs(result, settings, config_reference="configs/data/dioula.yaml")

    content = settings.candidate_manifest_path.read_text(encoding="utf-8")
    assert "https://" not in content
    assert str(tmp_path) not in content
    assert ",local_research_only,eligible,," in content
    assert settings.candidate_metadata_path.is_file()


def test_missing_audio_recovery_is_disabled_by_default(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    with pytest.raises(ConfigError, match="désactivée"):
        execute_recovery(settings, explicit_confirmation=True)
