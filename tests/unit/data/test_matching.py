from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ivoirevoice.data.matching import match_audio
from ivoirevoice.data.records import ParsedClip

BASE_RECORD = ParsedClip(
    clip_id="clip-id",
    sentence_id="sentence-id",
    sequence=0,
    text_raw="Àn bɛ taa.",
    text_nfc="Àn bɛ taa.",
    text_normalized="Àn bɛ taa.",
    audio_filename="expected.wav",
    source_json="women/speaker_01/clips.json",
    record_index=0,
    speaker_id="spk_test",
    gender_folder="women",
    valid=True,
    validation_issues=(),
)


def test_exact_filename_match_is_local_and_unique(tmp_path: Path) -> None:
    expected = tmp_path / "speaker" / "nested" / "expected.wav"
    other = tmp_path / "speaker" / "other.wav"

    result = match_audio(BASE_RECORD, (expected, other))

    assert result.status == "matched"
    assert result.method == "exact_filename"
    assert result.path == expected


def test_missing_match(tmp_path: Path) -> None:
    result = match_audio(BASE_RECORD, (tmp_path / "unrelated.wav",))

    assert result.status == "missing"
    assert result.method == "none"
    assert result.path is None


def test_ambiguous_exact_filename_match(tmp_path: Path) -> None:
    first = tmp_path / "one" / "expected.wav"
    second = tmp_path / "two" / "expected.wav"

    result = match_audio(BASE_RECORD, (first, second))

    assert result.status == "ambiguous"
    assert result.method == "exact_filename"
    assert result.candidate_count == 2


def test_falls_back_to_sentence_then_clip_identifier(tmp_path: Path) -> None:
    sentence_path = tmp_path / "prefix_sentence-id_suffix.wav"
    clip_path = tmp_path / "prefix_clip-id_suffix.wav"

    sentence_result = match_audio(
        replace(BASE_RECORD, audio_filename="not-found.wav"),
        (sentence_path, clip_path),
    )
    clip_result = match_audio(
        replace(
            BASE_RECORD,
            audio_filename="not-found.wav",
            sentence_id="absent-sentence",
        ),
        (sentence_path, clip_path),
    )

    assert sentence_result.path == sentence_path
    assert sentence_result.method == "sentence_id"
    assert clip_result.path == clip_path
    assert clip_result.method == "clip_id"


def test_prefers_converted_wav_matched_by_sequence(tmp_path: Path) -> None:
    converted = tmp_path / "wav" / "dyu-speaker-000000-01.wav"
    mislabeled_source = tmp_path / "source" / "expected.wav"

    result = match_audio(BASE_RECORD, (mislabeled_source, converted))

    assert result.path == converted
    assert result.method == "sequence"
