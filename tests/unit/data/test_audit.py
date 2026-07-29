from __future__ import annotations

import json
import wave
from pathlib import Path

from ivoirevoice.data.audit import run_audit, write_audit_outputs
from ivoirevoice.data.settings import DioulaDataSettings, SplitSettings


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\x00\x00" * 1_600)


def test_audit_counts_unique_audio_and_writes_privacy_safe_reports(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    speaker = dataset_root / "men" / "speaker_01"
    _write_wav(speaker / "wav" / "dyu-speaker-000000-01.wav")
    record = {
        "id": "clip-1",
        "sequence": 0,
        "audioSrc": "https://audio.invalid/source.wav?token=secret",
        "sentence": {
            "id": "sentence-1",
            "sequence": 0,
            "text": "Án bɛ taa.\r",
        },
    }
    (speaker / "clips.json").write_text(
        json.dumps([record, record], ensure_ascii=False),
        encoding="utf-8",
    )
    (speaker / "text").write_text("Án bɛ taa.\n", encoding="utf-8")
    (speaker / "text-no-tones").write_text("An bɛ taa.\n", encoding="utf-8")
    artifacts_root = tmp_path / "artifacts"
    settings = DioulaDataSettings(
        dataset_root=dataset_root,
        artifacts_root=artifacts_root,
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
        manifest_relative_path=Path("manifests/dioula_manifest_draft.csv"),
        report_relative_directory=Path("reports/data_audit"),
    )

    result = run_audit(settings)
    write_audit_outputs(result, settings)

    assert result.summary["matching"]["matched"] == 2
    assert result.summary["audio"]["unique_matched_audio_files"] == 1
    assert result.summary["audio"]["duration_seconds"]["total"] == 0.1
    assert result.summary["audio"]["duplicate_audio_path_reference_groups"] == 1
    assert result.summary["audio"]["duplicate_audio_path_extra_occurrences"] == 1
    assert result.summary["audio"]["duplicate_sha256_groups"] == 0
    assert result.summary["matching"]["methods"]["sequence"] == 2
    assert result.summary["text"]["records_with_carriage_return"] == 2
    assert result.split_proposal["leakage_free"] is True

    summary_path = settings.report_directory / "dioula_summary.json"
    report_content = summary_path.read_text(encoding="utf-8")
    assert "https://" not in report_content
    assert "token=secret" not in report_content
    assert (settings.report_directory / "dioula_split_proposal.json").is_file()
    assert settings.manifest_path.is_file()
