from __future__ import annotations

import io
import logging
import wave
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from ivoirevoice.api.app import create_app
from ivoirevoice.api.asr_routes import ASRAPIServices
from ivoirevoice.api.uploads import SecureUploadStager
from ivoirevoice.models.base import ASRBackend, AudioInput, TranscriptionResult
from ivoirevoice.models.registry import ModelRegistry
from ivoirevoice.services.transcription_service import (
    ModelCatalog,
    ModelDefinition,
    TranscriptionService,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _wav_bytes(*, duration_seconds: float = 0.1, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\x00\x00" * round(16_000 * duration_seconds) * channels)
    return buffer.getvalue()


def _definition(key: str, languages: tuple[str, ...]) -> ModelDefinition:
    return ModelDefinition(
        key=key,
        display_name={
            "whisper_tiny_baseline": "Whisper Tiny — Baseline",
            "whisper_small_baseline": "Whisper Small — Baseline",
            "whisper_tiny_dioula_final": "Whisper Tiny — Dioula Final",
        }[key],
        backend="whisper",
        status="final_adapted" if key.endswith("dioula_final") else "baseline",
        model_id="private/model/path-must-not-leak",
        revision="a" * 40,
        config_path=None,
        device="cpu",
        languages=languages,
        enabled=True,
    )


class RecordingBackend(ASRBackend):
    def __init__(
        self,
        events: list[str],
        observed_paths: list[Path],
        *,
        failure: bool = False,
        non_finite: bool = False,
    ) -> None:
        self.events = events
        self.observed_paths = observed_paths
        self.failure = failure
        self.non_finite = non_finite

    @property
    def model_name(self) -> str:
        return "/private/checkpoint/model"

    @property
    def supported_languages(self) -> tuple[str, ...]:
        return ("fr", "en", "dyu")

    def load(self) -> None:
        self.events.append("load")

    def transcribe(self, audio: AudioInput, language: str) -> TranscriptionResult:
        self.events.append(f"transcribe:{language}")
        path = Path(audio)
        assert path.is_file()
        self.observed_paths.append(path)
        if self.failure:
            raise RuntimeError("private text at /home/private/checkpoint")
        return TranscriptionResult(
            text="résultat synthétique",
            language=language,
            processing_time_seconds=float("nan") if self.non_finite else 0.01,
            model_name=self.model_name,
            duration_seconds=0.1,
        )

    def unload(self) -> None:
        self.events.append("unload")


def _services(
    temporary_root: Path,
    *,
    failure: bool = False,
    non_finite: bool = False,
    factory_observer: list[str] | None = None,
) -> tuple[ASRAPIServices, list[str], list[Path]]:
    catalog = ModelCatalog(
        models=(
            _definition("whisper_tiny_baseline", ("fr", "en", "dyu")),
            _definition("whisper_small_baseline", ("fr", "en", "dyu")),
            _definition("whisper_tiny_dioula_final", ("dyu",)),
        ),
        max_audio_size_bytes=1024 * 1024,
        max_audio_duration_seconds=30,
        allowed_extensions=(".wav", ".mp3", ".flac", ".ogg"),
    )
    events: list[str] = []
    observed_paths: list[Path] = []
    registry = ModelRegistry()

    def factory() -> RecordingBackend:
        if factory_observer is not None:
            factory_observer.append("created")
        return RecordingBackend(
            events,
            observed_paths,
            failure=failure,
            non_finite=non_finite,
        )

    for definition in catalog.enabled_models:
        registry.register(definition.key, factory)
    return (
        ASRAPIServices(
            catalog=catalog,
            transcription=TranscriptionService(catalog, registry),
            uploads=SecureUploadStager(
                max_size_bytes=catalog.max_audio_size_bytes,
                temporary_root=temporary_root,
            ),
            language_codes=("fr", "en", "dyu"),
        ),
        events,
        observed_paths,
    )


async def _client(services: ASRAPIServices) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=create_app(asr_services=services))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def test_health_and_discovery_do_not_construct_a_backend(tmp_path: Path) -> None:
    constructed: list[str] = []
    services, _, _ = _services(tmp_path / "uploads", factory_observer=constructed)
    async for client in _client(services):
        health = await client.get("/api/health")
        languages = await client.get("/api/v1/languages")
        models = await client.get("/api/v1/models")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert [item["code"] for item in languages.json()["languages"]] == ["fr", "en", "dyu"]
    assert all(item["asr"] == "experimental" for item in languages.json()["languages"])
    assert [item["id"] for item in models.json()["models"]] == [
        "whisper_tiny_baseline",
        "whisper_small_baseline",
        "whisper_tiny_dioula_final",
    ]
    serialized = models.text + languages.text
    assert "config_path" not in serialized
    assert "checkpoint" not in serialized
    assert "/private/" not in serialized
    assert constructed == []


@pytest.mark.parametrize("language", ["fr", "en", "dyu"])
async def test_transcription_contract_uses_one_backend_and_cleans_upload(
    tmp_path: Path,
    language: str,
) -> None:
    services, events, observed_paths = _services(tmp_path / "uploads")
    model = "whisper_tiny_dioula_final" if language == "dyu" else "whisper_tiny_baseline"
    async for client in _client(services):
        response = await client.post(
            "/api/v1/transcriptions",
            data={"language": language, "model": model},
            files={"audio": ("user-name.wav", _wav_bytes(), "audio/wav")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "id": payload["id"],
        "status": "completed",
        "language": language,
        "model_id": model,
        "text": "résultat synthétique",
        "audio_duration_seconds": 0.1,
        "processing_time_seconds": 0.01,
        "rtf": payload["rtf"],
    }
    assert payload["rtf"] == pytest.approx(0.1)
    assert payload["id"].startswith("tr_")
    assert "/private/" not in response.text
    assert events == ["load", f"transcribe:{language}", "unload"]
    assert len(observed_paths) == 1
    assert observed_paths[0].name != "user-name.wav"
    assert not observed_paths[0].exists()
    upload_root = tmp_path / "uploads"
    assert not upload_root.exists() or list(upload_root.iterdir()) == []


@pytest.mark.parametrize(
    ("data", "files", "code"),
    [
        ({"language": "zz", "model": "whisper_tiny_baseline"}, {}, "unknown_language"),
        ({"language": "fr", "model": "missing"}, {}, "unknown_model"),
        (
            {"language": "fr", "model": "whisper_tiny_dioula_final"},
            {},
            "incompatible_model_language",
        ),
    ],
)
async def test_invalid_language_and_model_choices_are_safe(
    tmp_path: Path,
    data: dict[str, str],
    files: dict[str, object],
    code: str,
) -> None:
    services, events, _ = _services(tmp_path / "uploads")
    selected_files = files or {"audio": ("sample.wav", _wav_bytes(), "audio/wav")}
    async for client in _client(services):
        response = await client.post(
            "/api/v1/transcriptions",
            data=data,
            files=selected_files,
        )

    assert response.status_code in {404, 422}
    assert response.json()["error"]["code"] == code
    assert events == []


@pytest.mark.parametrize(
    ("filename", "content", "mime"),
    [
        ("sample.txt", b"not-audio", "text/plain"),
        ("sample.wav", b"not-a-wave", "audio/wav"),
        ("sample.wav", _wav_bytes(), "audio/mpeg"),
        ("sample.m4a", b"m4a", "audio/mp4"),
    ],
)
async def test_invalid_uploads_are_rejected_without_inference(
    tmp_path: Path,
    filename: str,
    content: bytes,
    mime: str,
) -> None:
    services, events, _ = _services(tmp_path / "uploads")
    async for client in _client(services):
        response = await client.post(
            "/api/v1/transcriptions",
            data={"language": "fr", "model": "whisper_tiny_baseline"},
            files={"audio": (filename, content, mime)},
        )

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_audio"
    assert events == []
    upload_root = tmp_path / "uploads"
    assert not upload_root.exists() or list(upload_root.iterdir()) == []


async def test_per_file_and_declared_body_size_limits_are_enforced(tmp_path: Path) -> None:
    services, events, _ = _services(tmp_path / "uploads")
    async for client in _client(services):
        file_response = await client.post(
            "/api/v1/transcriptions",
            data={"language": "fr", "model": "whisper_tiny_baseline"},
            files={"audio": ("sample.wav", b"RIFF" + b"x" * (1024 * 1024), "audio/wav")},
        )
        declared_response = await client.post(
            "/api/v1/transcriptions",
            headers={"Content-Length": str(26 * 1024 * 1024)},
            content=b"small",
        )

    assert file_response.status_code == 413
    assert file_response.json()["error"]["code"] == "payload_too_large"
    assert declared_response.status_code == 413
    assert declared_response.json()["error"]["code"] == "payload_too_large"
    assert events == []


async def test_content_length_is_required_and_must_be_positive_decimal(tmp_path: Path) -> None:
    services, events, _ = _services(tmp_path / "uploads")

    async def streamed_body() -> AsyncIterator[bytes]:
        yield b"small"

    async for client in _client(services):
        missing = await client.post(
            "/api/v1/transcriptions",
            content=streamed_body(),
        )
        invalid = await client.post(
            "/api/v1/transcriptions",
            content=b"small",
            headers={"Content-Length": "invalid"},
        )
        zero = await client.post(
            "/api/v1/transcriptions",
            content=b"small",
            headers={"Content-Length": "0"},
        )

    assert missing.status_code == 411
    assert invalid.status_code == 400
    assert zero.status_code == 400
    assert {response.json()["error"]["code"] for response in (missing, invalid, zero)} == {
        "invalid_request"
    }
    assert events == []


async def test_audio_longer_than_thirty_seconds_is_rejected(tmp_path: Path) -> None:
    services, events, _ = _services(tmp_path / "uploads")
    async for client in _client(services):
        response = await client.post(
            "/api/v1/transcriptions",
            data={"language": "fr", "model": "whisper_tiny_baseline"},
            files={"audio": ("long.wav", _wav_bytes(duration_seconds=30.1), "audio/wav")},
        )

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_audio"
    assert events == []


async def test_model_failure_is_sanitized_logged_and_cleaned(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    services, events, observed_paths = _services(tmp_path / "uploads", failure=True)
    with caplog.at_level(logging.INFO, logger="ivoirevoice.api.asr_routes"):
        async for client in _client(services):
            response = await client.post(
                "/api/v1/transcriptions",
                data={"language": "fr", "model": "whisper_tiny_baseline"},
                files={"audio": ("private-name.wav", _wav_bytes(), "audio/wav")},
            )

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "transcription_failed"
    assert "/home/" not in response.text + caplog.text
    assert "private-name" not in response.text + caplog.text
    assert "résultat synthétique" not in caplog.text
    assert events == ["load", "transcribe:fr", "unload"]
    assert observed_paths and not observed_paths[0].exists()
    assert list((tmp_path / "uploads").iterdir()) == []


async def test_non_finite_backend_metrics_are_rejected_safely(tmp_path: Path) -> None:
    services, events, _ = _services(tmp_path / "uploads", non_finite=True)
    async for client in _client(services):
        response = await client.post(
            "/api/v1/transcriptions",
            data={"language": "fr", "model": "whisper_tiny_baseline"},
            files={"audio": ("sample.wav", _wav_bytes(), "audio/wav")},
        )

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "transcription_failed"
    assert "NaN" not in response.text
    assert events == ["load", "transcribe:fr", "unload"]
