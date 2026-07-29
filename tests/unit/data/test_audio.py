from __future__ import annotations

import wave
from pathlib import Path

from ivoirevoice.data.audio import inspect_audio


def _write_pcm_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\x00\x00" * 1_600)


def test_inspects_small_readable_wav_without_loading_corpus(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    _write_pcm_wav(audio_path)

    metadata = inspect_audio(audio_path, hash_audio=True)

    assert metadata.audio_status == "readable"
    assert metadata.sample_rate_hz == 16_000
    assert metadata.channels == 1
    assert metadata.num_samples == 1_600
    assert metadata.duration_seconds == 0.1
    assert len(metadata.audio_sha256) == 64


def test_marks_corrupted_audio(tmp_path: Path) -> None:
    audio_path = tmp_path / "corrupted.wav"
    audio_path.write_bytes(b"this is not a wav")

    metadata = inspect_audio(audio_path, hash_audio=False)

    assert metadata.audio_status == "corrupted"
    assert metadata.duration_seconds is None


def test_detects_mp4_container_with_wav_extension(tmp_path: Path) -> None:
    audio_path = tmp_path / "mislabeled.wav"
    audio_path.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00")

    metadata = inspect_audio(audio_path, hash_audio=False)

    assert metadata.audio_status == "format_mismatch"
    assert metadata.audio_format == "ISO_BASE_MEDIA"
    assert metadata.duration_seconds is None
