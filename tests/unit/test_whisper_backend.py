from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ivoirevoice.exceptions import BackendNotLoadedError, ConfigError
from ivoirevoice.models.whisper import (
    WhisperBackend,
    WhisperSettings,
    load_whisper_settings,
)


def _settings(tmp_path: Path) -> WhisperSettings:
    return WhisperSettings(
        backend_name="whisper-tiny",
        model_id="openai/whisper-tiny",
        model_revision="a" * 40,
        device="cpu",
        torch_dtype="float32",
        batch_size=1,
        chunk_length_seconds=30,
        stride_length_seconds=5,
        task="transcribe",
        language=None,
        cache_dir=tmp_path / "cache",
        local_files_only=True,
        max_new_tokens=128,
        expected_sampling_rate_hz=16_000,
        supported_languages=("dyu", "fr", "en"),
    )


def test_backend_loads_once_and_never_forces_dyu_token(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    factory_calls = 0

    class FakePipeline:
        def __call__(self, _inputs: Any, **kwargs: Any) -> dict[str, str]:
            calls.append(kwargs)
            return {"text": "A ka taa"}

    def factory(_settings: WhisperSettings) -> FakePipeline:
        nonlocal factory_calls
        factory_calls += 1
        return FakePipeline()

    backend = WhisperBackend(
        _settings(tmp_path),
        pipeline_factory=factory,
        audio_loader=lambda _audio, _rate: ({"array": [0.0]}, 1.25),
    )
    backend.load()
    backend.load()

    result = backend.transcribe(b"fake-audio", "dyu")

    assert factory_calls == 1
    assert result.text == "A ka taa"
    assert result.duration_seconds == 1.25
    assert result.model_name.endswith("@" + "a" * 40)
    assert calls[0]["generate_kwargs"]["task"] == "transcribe"
    assert "language" not in calls[0]["generate_kwargs"]
    backend.unload()


def test_backend_requires_explicit_load(tmp_path: Path) -> None:
    backend = WhisperBackend(
        _settings(tmp_path),
        pipeline_factory=lambda _settings: pytest.fail("factory should not run"),
        audio_loader=lambda _audio, _rate: ({"array": [0.0]}, 1.0),
    )

    with pytest.raises(BackendNotLoadedError):
        backend.transcribe(b"fake", "dyu")


@pytest.mark.parametrize("language", ["fr", "en"])
def test_backend_forces_supported_french_and_english_tokens(
    tmp_path: Path,
    language: str,
) -> None:
    calls: list[dict[str, Any]] = []

    class FakePipeline:
        def __call__(self, _inputs: Any, **kwargs: Any) -> dict[str, str]:
            calls.append(kwargs)
            return {"text": "synthetic"}

    backend = WhisperBackend(
        _settings(tmp_path),
        pipeline_factory=lambda _settings: FakePipeline(),
        audio_loader=lambda _audio, _rate: ({"array": [0.0]}, 1.0),
    )
    backend.load()

    backend.transcribe(b"fake", language)

    assert calls[0]["generate_kwargs"] == {
        "task": "transcribe",
        "max_new_tokens": 128,
        "language": language,
    }


def test_config_rejects_forced_dyu_language(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IVOIREVOICE_MODEL_CACHE_DIR", str(tmp_path / "cache"))
    config = tmp_path / "whisper.yaml"
    config.write_text(
        """
model:
  name: whisper
  family: whisper
  model_id: openai/whisper-tiny
  model_revision: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  device: cpu
  torch_dtype: float32
  batch_size: 1
  chunk_length_seconds: 30
  stride_length_seconds: 5
  task: transcribe
  language: dyu
  cache_environment_variable: IVOIREVOICE_MODEL_CACHE_DIR
  local_files_only: true
  max_new_tokens: 128
  expected_sampling_rate_hz: 16000
  supported_languages: [dyu]
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="ne doit pas être forcé"):
        load_whisper_settings(config)
