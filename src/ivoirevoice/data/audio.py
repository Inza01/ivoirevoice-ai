"""Streaming audio metadata and hashing helpers."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import soundfile

from ivoirevoice.data.records import AudioMetadata

HASH_CHUNK_SIZE = 1024 * 1024
ISO_BASE_MEDIA_SIGNATURE = b"ftyp"


def sha256_file(path: Path) -> str:
    """Hash a file incrementally."""

    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def empty_audio_metadata() -> AudioMetadata:
    """Return metadata for a row that has no unique audio match."""

    return AudioMetadata(
        duration_seconds=None,
        sample_rate_hz=None,
        channels=None,
        num_samples=None,
        audio_format="",
        file_size_bytes=None,
        audio_sha256="",
        audio_status="not_applicable",
    )


def inspect_audio(path: Path, *, hash_audio: bool) -> AudioMetadata:
    """Read file-level metadata without decoding the whole waveform."""

    try:
        file_size = path.stat().st_size
        with path.open("rb") as stream:
            header = stream.read(12)
    except OSError:
        return AudioMetadata(
            duration_seconds=None,
            sample_rate_hz=None,
            channels=None,
            num_samples=None,
            audio_format="",
            file_size_bytes=None,
            audio_sha256="",
            audio_status="corrupted",
        )

    audio_hash = ""
    if hash_audio:
        try:
            audio_hash = sha256_file(path)
        except OSError:
            return AudioMetadata(
                duration_seconds=None,
                sample_rate_hz=None,
                channels=None,
                num_samples=None,
                audio_format="",
                file_size_bytes=file_size,
                audio_sha256="",
                audio_status="corrupted",
            )

    if len(header) >= 8 and header[4:8] == ISO_BASE_MEDIA_SIGNATURE:
        return AudioMetadata(
            duration_seconds=None,
            sample_rate_hz=None,
            channels=None,
            num_samples=None,
            audio_format="ISO_BASE_MEDIA",
            file_size_bytes=file_size,
            audio_sha256=audio_hash,
            audio_status="format_mismatch",
        )

    try:
        info = soundfile.info(str(path))
        return AudioMetadata(
            duration_seconds=float(info.duration),
            sample_rate_hz=int(info.samplerate),
            channels=int(info.channels),
            num_samples=int(info.frames),
            audio_format=str(info.format),
            file_size_bytes=file_size,
            audio_sha256=audio_hash,
            audio_status="readable",
        )
    except (OSError, RuntimeError, ValueError, soundfile.LibsndfileError):
        return AudioMetadata(
            duration_seconds=None,
            sample_rate_hz=None,
            channels=None,
            num_samples=None,
            audio_format="",
            file_size_bytes=file_size,
            audio_sha256=audio_hash,
            audio_status="corrupted",
        )
