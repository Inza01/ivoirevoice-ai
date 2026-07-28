from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest

from ivoirevoice.api.app import create_app

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as test_client:
        yield test_client


async def test_health_endpoint(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "IvoireVoice AI",
        "version": "0.1.0",
    }


async def test_models_endpoint_exposes_only_dummy_backend(client: httpx.AsyncClient) -> None:
    response = await client.get("/models")

    assert response.status_code == 200
    assert response.json() == {
        "models": [
            {
                "name": "dummy",
                "supported_languages": ["fr", "dyu"],
                "implementation": "DummyBackend",
            }
        ]
    }


async def test_transcribe_rejects_missing_form_fields_with_structured_error(
    client: httpx.AsyncClient,
) -> None:
    response = await client.post("/transcribe")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


async def test_transcribe_rejects_unsupported_content_type(
    client: httpx.AsyncClient,
) -> None:
    response = await client.post(
        "/transcribe",
        data={"language": "fr"},
        files={"file": ("sample.txt", b"not audio", "text/plain")},
    )

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_audio_type"


async def test_transcribe_uses_dummy_backend_without_model_download(
    client: httpx.AsyncClient,
) -> None:
    response = await client.post(
        "/transcribe",
        data={"language": "dyu"},
        files={"file": ("sample.wav", b"RIFF-development-placeholder", "audio/wav")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"] == "dummy"
    assert payload["language"] == "dyu"
    assert payload["confidence"] is None
    assert "fictive" in payload["text"]
