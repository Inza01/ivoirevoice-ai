from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "web_asr_smoke.py"
SPEC = importlib.util.spec_from_file_location("ivoirevoice_web_asr_smoke", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
web_asr_smoke = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = web_asr_smoke
SPEC.loader.exec_module(web_asr_smoke)

SmokeConfigurationError = web_asr_smoke.SmokeConfigurationError
_validate_audio_path = web_asr_smoke._validate_audio_path
_validate_base_url = web_asr_smoke._validate_base_url
_validate_discovery = web_asr_smoke._validate_discovery
_validate_transcription_payload = web_asr_smoke._validate_transcription_payload


def test_smoke_accepts_only_loopback_origin() -> None:
    assert _validate_base_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000"
    with pytest.raises(SmokeConfigurationError, match="loopback"):
        _validate_base_url("https://example.test")


def test_smoke_rejects_repository_audio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audio = tmp_path / "safe.wav"
    audio.write_bytes(b"synthetic")
    monkeypatch.setattr(web_asr_smoke, "REPOSITORY_ROOT", tmp_path)

    with pytest.raises(SmokeConfigurationError, match="protégé"):
        _validate_audio_path(str(audio), label="FR")


def test_smoke_rejects_holdout_like_path(tmp_path: Path) -> None:
    protected = tmp_path / "final-holdout" / "safe.wav"
    protected.parent.mkdir()
    protected.write_bytes(b"synthetic")

    with pytest.raises(SmokeConfigurationError, match="ressemble"):
        _validate_audio_path(str(protected), label="EN")


def test_smoke_validates_public_discovery_contract() -> None:
    supported = _validate_discovery(
        {"languages": [{"code": "fr"}, {"code": "en"}, {"code": "dyu"}]},
        {
            "models": [
                {
                    "id": "whisper_tiny_baseline",
                    "supported_languages": ["fr", "en", "dyu"],
                },
                {
                    "id": "whisper_tiny_dioula_final",
                    "supported_languages": ["dyu"],
                },
            ]
        },
    )

    assert supported["whisper_tiny_baseline"] == frozenset({"fr", "en", "dyu"})


def test_smoke_rejects_empty_transcription_result() -> None:
    payload = {
        "id": "tr_synthetic",
        "status": "completed",
        "language": "en",
        "model_id": "whisper_tiny_baseline",
        "text": "",
        "audio_duration_seconds": 1.0,
        "processing_time_seconds": 0.1,
        "rtf": 0.1,
    }

    with pytest.raises(RuntimeError, match="invalide ou vide"):
        _validate_transcription_payload(
            payload,
            language="en",
            model_id="whisper_tiny_baseline",
        )
