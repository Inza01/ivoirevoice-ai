"""Internal immutable records shared by the data pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

AudioMatchStatus = Literal["matched", "missing", "ambiguous", "invalid"]
AudioMatchMethod = Literal["sequence", "exact_filename", "sentence_id", "clip_id", "none"]


@dataclass(frozen=True, slots=True)
class SpeakerSource:
    """A structurally inferred speaker directory."""

    speaker_id: str
    gender_folder: str
    directory: Path
    relative_directory: str
    clips_json: Path
    source_json: str
    wav_files: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class ParsedClip:
    """Sanitized representation of one clips.json record."""

    clip_id: str
    sentence_id: str
    sequence: int | None
    text_raw: str
    text_nfc: str
    text_normalized: str
    audio_filename: str
    source_json: str
    record_index: int
    speaker_id: str
    gender_folder: str
    valid: bool
    validation_issues: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AudioMatch:
    """Result of matching one record to a local WAV."""

    status: AudioMatchStatus
    method: AudioMatchMethod
    path: Path | None
    candidate_count: int


@dataclass(frozen=True, slots=True)
class AudioMetadata:
    """Metadata read without decoding an entire audio file."""

    duration_seconds: float | None
    sample_rate_hz: int | None
    channels: int | None
    num_samples: int | None
    audio_format: str
    file_size_bytes: int | None
    audio_sha256: str
    audio_status: Literal["readable", "corrupted", "format_mismatch", "not_applicable"]


@dataclass(frozen=True, slots=True)
class ManifestRow:
    """Canonical local Dioula manifest row."""

    utterance_id: str
    clip_id: str
    sentence_id: str
    speaker_id: str
    gender_folder: str
    language: str
    text_raw: str
    text_nfc: str
    text_normalized: str
    audio_path: str
    audio_filename: str
    audio_match_status: AudioMatchStatus
    audio_match_method: AudioMatchMethod
    duration_seconds: float | None
    sample_rate_hz: int | None
    channels: int | None
    num_samples: int | None
    audio_format: str
    file_size_bytes: int | None
    audio_sha256: str
    audio_status: str
    source_json: str
    record_index: int
    license_status: str
    split: str
    validation_issues: str
