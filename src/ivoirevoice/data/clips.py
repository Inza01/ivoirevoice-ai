"""Robust clips.json parsing that never persists signed source URLs."""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit

from ivoirevoice.data.records import ParsedClip, SpeakerSource
from ivoirevoice.exceptions import IvoireVoiceError


class ClipsParseError(IvoireVoiceError):
    """Raised when a clips.json file itself is not parseable."""


def extract_wav_filename(audio_source: object) -> str:
    """Extract only a safe WAV basename, discarding query parameters and fragments."""

    if not isinstance(audio_source, str) or not audio_source.strip():
        return ""
    url_path = urlsplit(audio_source).path
    filename = PurePosixPath(unquote(url_path)).name
    if not filename.lower().endswith(".wav"):
        return ""
    return Path(filename).name


def normalize_transcription(text: str) -> str:
    """Collapse surrounding/internal whitespace and normalize to NFC without removing tones."""

    collapsed = " ".join(text.split())
    return unicodedata.normalize("NFC", collapsed)


def _string_identifier(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _sequence(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _parse_record(
    record: object,
    *,
    record_index: int,
    speaker: SpeakerSource,
) -> ParsedClip:
    issues: list[str] = []
    if not isinstance(record, Mapping):
        record_data: Mapping[str, Any] = {}
        issues.append("record_not_object")
    else:
        record_data = record

    clip_id = _string_identifier(record_data.get("id"))
    if not clip_id:
        issues.append("missing_clip_id")

    sentence_value = record_data.get("sentence")
    if not isinstance(sentence_value, Mapping):
        sentence: Mapping[str, Any] = {}
        issues.append("missing_sentence")
    else:
        sentence = sentence_value

    sentence_id = _string_identifier(sentence.get("id"))
    if not sentence_id:
        issues.append("missing_sentence_id")

    text_value = sentence.get("text")
    if isinstance(text_value, str):
        text_raw = text_value
    else:
        text_raw = ""
        issues.append("invalid_sentence_text")

    sequence = _sequence(record_data.get("sequence"))
    if sequence is None:
        sequence = _sequence(sentence.get("sequence"))
    if sequence is None:
        issues.append("invalid_sequence")

    audio_filename = extract_wav_filename(record_data.get("audioSrc"))
    if not audio_filename:
        issues.append("missing_audio_filename")

    fatal_issues = {
        "record_not_object",
        "missing_clip_id",
        "missing_sentence",
        "missing_sentence_id",
        "invalid_sentence_text",
        "invalid_sequence",
    }
    return ParsedClip(
        clip_id=clip_id,
        sentence_id=sentence_id,
        sequence=sequence,
        text_raw=text_raw,
        text_nfc=unicodedata.normalize("NFC", text_raw),
        text_normalized=normalize_transcription(text_raw),
        audio_filename=audio_filename,
        source_json=speaker.source_json,
        record_index=record_index,
        speaker_id=speaker.speaker_id,
        gender_folder=speaker.gender_folder,
        valid=not any(issue in fatal_issues for issue in issues),
        validation_issues=tuple(issues),
    )


def parse_clips_file(speaker: SpeakerSource) -> list[ParsedClip]:
    """Parse all records while keeping malformed records observable."""

    try:
        with speaker.clips_json.open(encoding="utf-8") as stream:
            root: object = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ClipsParseError(f"Impossible de lire {speaker.source_json} : {exc}") from exc
    if not isinstance(root, list):
        raise ClipsParseError(f"La racine de {speaker.source_json} doit être une liste.")
    return [
        _parse_record(record, record_index=index, speaker=speaker)
        for index, record in enumerate(root)
    ]
