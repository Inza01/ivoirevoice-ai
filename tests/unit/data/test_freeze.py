from __future__ import annotations

import csv
import json
from dataclasses import asdict, fields, replace
from pathlib import Path

import pytest

from ivoirevoice.data.audio import sha256_file
from ivoirevoice.data.freeze import (
    FreezeCandidateRow,
    SplitPlan,
    build_frozen_rows,
    freeze_dataset,
    load_frozen_manifest,
    validate_frozen_dataset,
    validate_frozen_rows,
)
from ivoirevoice.data.settings import (
    CurationSettings,
    DioulaDataSettings,
    FreezeSettings,
    SplitSettings,
)
from ivoirevoice.exceptions import ConfigError

EXPECTED_SPEAKER_COUNTS = {"train": 15, "validation": 3, "test": 3}


def _candidate_rows() -> tuple[FreezeCandidateRow, ...]:
    rows: list[FreezeCandidateRow] = []
    for index in range(21):
        falling = index % 2 == 0
        marker = " ↘" if falling else ""
        rows.append(
            FreezeCandidateRow(
                utterance_id=f"utt_{index:02d}",
                sentence_id=f"sentence_{index:02d}",
                speaker_id=f"spk_{index:02d}",
                gender_folder="men" if index < 11 else "women",
                language="dyu",
                text_raw=f"À kà{marker}\r",
                text_with_tones_nfc=f"À kà{marker}",
                text_without_tones_nfc=f"A ka{marker}",
                audio_path=f"speaker_{index:02d}/audio.wav",
                duration_seconds=float(index + 1),
                sample_rate_hz=16_000,
                channels=1,
                file_size_bytes=1_000 + index,
                audio_sha256=f"{index:064x}",
                source_json=f"speaker_{index:02d}/clips.json",
                license_status="unknown",
                usage_scope="local_research_only",
                eligibility_status="eligible",
                exclusion_reason="",
                split="",
            )
        )
    return tuple(rows)


def _split_plan() -> SplitPlan:
    return SplitPlan(
        strategy="B_15_3_3",
        seed=42,
        speaker_ids={
            "train": tuple(
                [f"spk_{index:02d}" for index in range(7)]
                + [f"spk_{index:02d}" for index in range(11, 19)]
            ),
            "validation": ("spk_07", "spk_08", "spk_19"),
            "test": ("spk_09", "spk_10", "spk_20"),
        },
    )


def test_applies_split_b_and_preserves_text_while_removing_marker_from_target() -> None:
    frozen = build_frozen_rows(
        _candidate_rows(),
        _split_plan(),
        license_status="unknown",
        consent_status="unknown",
        usage_scope="local_research_only",
    )

    metrics = validate_frozen_rows(
        frozen,
        expected_audio_count=21,
        expected_speaker_count=21,
        expected_speaker_counts=EXPECTED_SPEAKER_COUNTS,
        language="dyu",
        license_status="unknown",
        consent_status="unknown",
        usage_scope="local_research_only",
    )

    assert frozen[0].text_raw == "À kà ↘\r"
    assert frozen[0].text_with_tones_nfc == "À kà ↘"
    assert frozen[0].text_without_tones_nfc == "A ka ↘"
    assert frozen[0].target_text_mvp == "A ka"
    assert frozen[0].intonation_falling is True
    assert frozen[1].intonation_falling is False
    assert metrics["speaker_count_by_split"] == EXPECTED_SPEAKER_COUNTS
    assert all(metrics["audio_count_by_split"][split] > 0 for split in EXPECTED_SPEAKER_COUNTS)
    assert metrics["intonation_falling_rows"] == 11


def test_rejects_speaker_leakage() -> None:
    frozen = build_frozen_rows(
        _candidate_rows(),
        _split_plan(),
        license_status="unknown",
        consent_status="unknown",
        usage_scope="local_research_only",
    )
    leaked = replace(
        frozen[0],
        utterance_id="utt_leaked",
        audio_path="speaker_00/second.wav",
        audio_sha256="f" * 64,
        split="validation",
    )

    with pytest.raises(ConfigError, match="Fuite de locuteur"):
        validate_frozen_rows(
            (*frozen, leaked),
            expected_audio_count=22,
            expected_speaker_count=21,
            expected_speaker_counts=EXPECTED_SPEAKER_COUNTS,
            language="dyu",
            license_status="unknown",
            consent_status="unknown",
            usage_scope="local_research_only",
        )


def _write_candidate(path: Path, rows: tuple[FreezeCandidateRow, ...]) -> None:
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[field.name for field in fields(FreezeCandidateRow)],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _settings(tmp_path: Path) -> tuple[DioulaDataSettings, Path]:
    dataset_root = tmp_path / "dataset"
    artifacts_root = tmp_path / "artifacts"
    dataset_root.mkdir()
    config_path = tmp_path / "dioula.yaml"
    config_path.write_text("test_configuration: true\n", encoding="utf-8")
    settings = DioulaDataSettings(
        dataset_root=dataset_root,
        artifacts_root=artifacts_root,
        language="dyu",
        license_status="unknown",
        consent_status="unknown",
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
            metadata_relative_path=Path("manifests/candidate_metadata.json"),
            report_relative_directory=Path("reports/data_curation"),
            target_text="text_without_tones_nfc",
            recover_missing_audio=False,
            recovery_output_environment_variable="IVOIREVOICE_DIOULA_INTERIM_DIR",
        ),
        freeze=FreezeSettings(
            split_comparison_relative_path=Path("reports/data_curation/split_comparison.json"),
            manifest_relative_path=Path("manifests/dioula_manifest_v0.1.csv"),
            metadata_relative_path=Path("manifests/dioula_dataset_v0.1_metadata.json"),
            report_relative_path=Path("reports/data_curation/dioula_v0.1_report.md"),
            split_report_relative_path=Path("reports/data_curation/dioula_v0.1_split_report.json"),
            dataset_version="0.1.0-local",
            dataset_status="frozen_candidate",
            split_strategy="B_15_3_3",
            expected_audio_count=21,
            expected_speaker_count=21,
            expected_speaker_counts=EXPECTED_SPEAKER_COUNTS,
            publication_allowed=False,
            model_derivative_publication_allowed=False,
        ),
        manifest_relative_path=Path("manifests/draft.csv"),
        report_relative_directory=Path("reports/data_audit"),
    )
    return settings, config_path


def test_freeze_is_reproducible_local_and_publication_forbidden(tmp_path: Path) -> None:
    settings, config_path = _settings(tmp_path)
    candidates = _candidate_rows()
    _write_candidate(settings.candidate_manifest_path, candidates)
    settings.candidate_metadata_path.write_text(
        json.dumps(
            {
                "candidate_manifest_sha256": sha256_file(settings.candidate_manifest_path),
                "included_rows": len(candidates),
                "license_status": "unknown",
                "usage_scope": "local_research_only",
                "recovery_executed": False,
            }
        ),
        encoding="utf-8",
    )
    settings.split_comparison_path.parent.mkdir(parents=True)
    settings.split_comparison_path.write_text(
        json.dumps(
            {
                "strategies": [
                    {
                        "strategy": "B_15_3_3",
                        "seed": 42,
                        "speaker_ids": {
                            split: list(speakers)
                            for split, speakers in _split_plan().speaker_ids.items()
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    first = freeze_dataset(
        settings,
        config_path=config_path,
        pipeline_commit_sha="a" * 40,
    )
    first_bytes = settings.frozen_manifest_path.read_bytes()
    second = freeze_dataset(
        settings,
        config_path=config_path,
        pipeline_commit_sha="a" * 40,
    )
    validated = validate_frozen_dataset(settings, config_path=config_path)
    metadata = json.loads(settings.frozen_metadata_path.read_text(encoding="utf-8"))
    loaded = load_frozen_manifest(settings.frozen_manifest_path)

    assert first == second == validated
    assert settings.frozen_manifest_path.read_bytes() == first_bytes
    assert loaded[0].text_raw == candidates[0].text_raw
    assert all(row.split for row in loaded)
    assert all(row.usage_scope == "local_research_only" for row in loaded)
    assert all(row.consent_status == "unknown" for row in loaded)
    assert metadata["dataset_version"] == "0.1.0-local"
    assert metadata["dataset_status"] == "frozen_candidate"
    assert metadata["publication_allowed"] is False
    assert metadata["model_derivative_publication_allowed"] is False
