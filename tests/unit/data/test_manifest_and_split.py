from __future__ import annotations

import json
import wave
from dataclasses import replace
from pathlib import Path

from ivoirevoice.data.manifest import build_manifest, write_manifest_outputs
from ivoirevoice.data.records import ManifestRow
from ivoirevoice.data.settings import CurationSettings, DioulaDataSettings, SplitSettings
from ivoirevoice.data.split import propose_speaker_split

BASE_ROW = ManifestRow(
    utterance_id="utt_01",
    clip_id="clip_01",
    sentence_id="sentence_01",
    speaker_id="spk_01",
    gender_folder="men",
    language="dyu",
    text_raw="Àn bɛ taa.",
    text_nfc="Àn bɛ taa.",
    text_normalized="Àn bɛ taa.",
    audio_path="men/speaker_01/sample.wav",
    audio_filename="sample.wav",
    audio_match_status="matched",
    audio_match_method="exact_filename",
    duration_seconds=1.0,
    sample_rate_hz=16_000,
    channels=1,
    num_samples=16_000,
    audio_format="WAV",
    file_size_bytes=32_044,
    audio_sha256="abc",
    audio_status="readable",
    source_json="men/speaker_01/clips.json",
    record_index=0,
    license_status="unknown",
    split="",
    validation_issues="",
)


def _settings(dataset_root: Path, artifacts_root: Path) -> DioulaDataSettings:
    return DioulaDataSettings(
        dataset_root=dataset_root,
        artifacts_root=artifacts_root,
        language="dyu",
        license_status="unknown",
        usage_scope="local_research_only",
        hash_audio=False,
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
        manifest_relative_path=Path("manifests/dioula_manifest_draft.csv"),
        report_relative_directory=Path("reports/data_audit"),
    )


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(8_000)
        stream.writeframes(b"\x00\x00" * 800)


def test_manifest_never_contains_signed_url_or_absolute_dataset_path(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "raw"
    speaker = dataset_root / "women" / "speaker_01"
    audio_path = speaker / "nested" / "sample.wav"
    _write_wav(audio_path)
    (speaker / "clips.json").write_text(
        json.dumps(
            [
                {
                    "id": "clip-1",
                    "sequence": 0,
                    "audioSrc": "https://audio.invalid/sample.wav?signature=secret",
                    "sentence": {"id": "sentence-1", "text": "Án bɛ taa.", "sequence": 0},
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    settings = _settings(dataset_root, tmp_path / "artifacts")

    build = build_manifest(settings)
    write_manifest_outputs(build, settings)

    manifest_content = settings.manifest_path.read_text(encoding="utf-8")
    assert "audioSrc" not in manifest_content
    assert "https://" not in manifest_content
    assert "signature=secret" not in manifest_content
    assert str(dataset_root) not in manifest_content
    assert build.rows[0].audio_path == "women/speaker_01/nested/sample.wav"
    assert build.rows[0].split == ""


def test_split_is_deterministic_and_has_no_speaker_leakage() -> None:
    rows = tuple(
        replace(
            BASE_ROW,
            utterance_id=f"utt_{index}",
            speaker_id=f"spk_{index}",
            gender_folder="men" if index < 5 else "women",
        )
        for index in range(10)
    )
    settings = SplitSettings(
        seed=7,
        train_ratio=0.8,
        validation_ratio=0.1,
        test_ratio=0.1,
    )

    first = propose_speaker_split(rows, settings)
    second = propose_speaker_split(rows, settings)

    assert first == second
    assert first["leakage_free"] is True
    split_sets = [set(speakers) for speakers in first["speaker_ids"].values()]
    assert split_sets[0].isdisjoint(split_sets[1])
    assert split_sets[0].isdisjoint(split_sets[2])
    assert split_sets[1].isdisjoint(split_sets[2])
