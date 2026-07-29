from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from ivoirevoice.data.clips import ClipsParseError, extract_wav_filename, parse_clips_file
from ivoirevoice.data.records import SpeakerSource


def _speaker(tmp_path: Path, root: object) -> SpeakerSource:
    speaker_directory = tmp_path / "men" / "speaker_01"
    speaker_directory.mkdir(parents=True)
    clips_json = speaker_directory / "clips.json"
    clips_json.write_text(json.dumps(root, ensure_ascii=False), encoding="utf-8")
    return SpeakerSource(
        speaker_id="spk_test",
        gender_folder="men",
        directory=speaker_directory,
        relative_directory="men/speaker_01",
        clips_json=clips_json,
        source_json="men/speaker_01/clips.json",
        wav_files=(),
    )


def _valid_record() -> dict[str, object]:
    return {
        "id": "clip-1",
        "sequence": 0,
        "audioSrc": "https://audio.invalid/sample.wav?signature=secret",
        "sentence": {
            "id": "sentence-1",
            "text": "  Á hákílí\r\nkà gò ↘  ",
            "sequence": 0,
        },
    }


def test_parse_valid_record_preserves_raw_tones_and_normalizes_whitespace(
    tmp_path: Path,
) -> None:
    parsed = parse_clips_file(_speaker(tmp_path, [_valid_record()]))

    assert len(parsed) == 1
    assert parsed[0].text_raw == "  Á hákílí\r\nkà gò ↘  "
    assert parsed[0].text_nfc == parsed[0].text_raw
    assert parsed[0].text_normalized == "Á hákílí kà gò ↘"
    assert "↘" in parsed[0].text_normalized
    assert parsed[0].audio_filename == "sample.wav"
    assert parsed[0].valid is True


def test_rejects_non_list_json_root(tmp_path: Path) -> None:
    speaker = _speaker(tmp_path, {"records": []})

    with pytest.raises(ClipsParseError, match="doit être une liste"):
        parse_clips_file(speaker)


def test_missing_sentence_is_retained_as_invalid_record(tmp_path: Path) -> None:
    speaker = _speaker(
        tmp_path,
        [{"id": "clip-1", "sequence": 0, "audioSrc": "https://invalid/sample.wav"}],
    )

    parsed = parse_clips_file(speaker)

    assert len(parsed) == 1
    assert parsed[0].valid is False
    assert "missing_sentence" in parsed[0].validation_issues


def test_signed_url_is_reduced_to_wav_basename(tmp_path: Path) -> None:
    signed_url = "https://audio.invalid/path/sample%20one.wav?token=top-secret#fragment"

    assert extract_wav_filename(signed_url) == "sample one.wav"

    record = _valid_record()
    record["audioSrc"] = signed_url
    parsed = parse_clips_file(_speaker(tmp_path, [record]))[0]
    serialized = json.dumps(asdict(parsed), ensure_ascii=False)
    assert "https://" not in serialized
    assert "top-secret" not in serialized
