from __future__ import annotations

import csv
import json
import wave
from pathlib import Path

import pytest

from ivoirevoice.exceptions import ConfigError
from ivoirevoice.models.base import ASRBackend, AudioInput, TranscriptionResult
from ivoirevoice.models.dummy import DummyBackend
from ivoirevoice.models.registry import ModelRegistry
from ivoirevoice.services.comparison_service import ComparisonService
from ivoirevoice.services.evaluation_service import (
    NO_REFERENCE_MESSAGE,
    EvaluationService,
    load_benchmark_view,
)
from ivoirevoice.services.export_service import ExportService
from ivoirevoice.services.transcription_service import (
    ModelCatalog,
    ModelDefinition,
    TranscriptionService,
    build_model_registry,
    load_model_catalog,
)


def _wav(path: Path) -> Path:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\x00\x00" * 1_600)
    return path


def _definition(key: str) -> ModelDefinition:
    return ModelDefinition(
        key=key,
        display_name=f"Modèle {key}",
        backend="whisper",
        status="baseline",
        model_id=f"test/{key}",
        revision="a" * 40,
        config_path=None,
        device="cpu",
        languages=("fr", "dyu"),
        enabled=True,
    )


def _catalog(*keys: str) -> ModelCatalog:
    return ModelCatalog(
        models=tuple(_definition(key) for key in keys),
        max_audio_size_bytes=1024 * 1024,
        max_audio_duration_seconds=10.0,
        allowed_extensions=(".wav", ".mp3", ".m4a", ".ogg"),
    )


def _service(registry: ModelRegistry, *keys: str) -> ComparisonService:
    transcription = TranscriptionService(_catalog(*keys), registry)
    return ComparisonService(transcription, EvaluationService())


def test_comparison_service_runs_two_dummy_backends(tmp_path: Path) -> None:
    registry = ModelRegistry()
    registry.register("tiny", DummyBackend)
    registry.register("small", DummyBackend)
    service = _service(registry, "tiny", "small")

    run = service.compare(
        audio_path=_wav(tmp_path / "sample.wav"),
        language="dyu",
        model_keys=("tiny", "small"),
    )

    assert len(run.results) == 2
    assert all(result.success for result in run.results)
    assert all(result.evaluation.message == NO_REFERENCE_MESSAGE for result in run.results)
    assert len(run.audio_id) == 16


def test_one_model_failure_does_not_abort_other_model(tmp_path: Path) -> None:
    class FailingBackend(ASRBackend):
        @property
        def model_name(self) -> str:
            return "failing"

        @property
        def supported_languages(self) -> tuple[str, ...]:
            return ("dyu",)

        def load(self) -> None:
            raise RuntimeError("private technical details")

        def transcribe(self, audio: AudioInput, language: str) -> TranscriptionResult:
            raise AssertionError((audio, language))

        def unload(self) -> None:
            return None

    registry = ModelRegistry()
    registry.register("tiny", DummyBackend)
    registry.register("small", FailingBackend)
    service = _service(registry, "tiny", "small")

    run = service.compare(
        audio_path=_wav(tmp_path / "sample.wav"),
        language="dyu",
        model_keys=("tiny", "small"),
    )

    assert run.results[0].success is True
    assert run.results[1].success is False
    assert "private technical details" not in (run.results[1].error or "")


def test_evaluation_metrics_are_optional_and_exact() -> None:
    service = EvaluationService()

    unavailable = service.evaluate(None, "n be taa")
    available = service.evaluate("n bɛ taa", "n bɛ ta")

    assert unavailable.available is False
    assert unavailable.wer is None
    assert unavailable.message == NO_REFERENCE_MESSAGE
    assert available.available is True
    assert available.wer == pytest.approx(1 / 3)
    assert available.cer is not None
    assert available.substitutions == 1


def test_exports_json_csv_and_txt_without_local_path(tmp_path: Path) -> None:
    registry = ModelRegistry()
    registry.register("tiny", DummyBackend)
    service = _service(registry, "tiny")
    run = service.compare(
        audio_path=_wav(tmp_path / "sample.wav"),
        language="fr",
        model_keys=("tiny",),
        reference="bonjour",
    )
    exports = ExportService(tmp_path / "exports")

    json_path, csv_path, txt_path = exports.export_all(run)
    json_content = Path(json_path).read_text(encoding="utf-8")
    with Path(csv_path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    txt_content = Path(txt_path).read_text(encoding="utf-8")

    assert json.loads(json_content)["audio_id"] == run.audio_id
    assert rows[0]["model_key"] == "tiny"
    assert "IvoireVoice AI" in txt_content
    assert "/home/" not in json_content + txt_content
    exports.cleanup()
    assert exports.export_json(run).is_file()
    exports.cleanup()


def test_export_rejects_private_path_in_reference(tmp_path: Path) -> None:
    registry = ModelRegistry()
    registry.register("tiny", DummyBackend)
    service = _service(registry, "tiny")
    run = service.compare(
        audio_path=_wav(tmp_path / "sample.wav"),
        language="fr",
        model_keys=("tiny",),
        reference="/home/personne/secret",
    )

    with pytest.raises(ConfigError, match="chemin local privé"):
        ExportService(tmp_path / "exports").export_json(run)


def test_audio_extension_validation_rejects_unknown_format() -> None:
    service = TranscriptionService(_catalog("tiny"), ModelRegistry())

    assert service.validate_extension("sample.MP3") == ".mp3"
    with pytest.raises(ConfigError, match="Format audio refusé"):
        service.validate_extension("sample.exe")


def test_benchmark_loader_uses_structured_json(tmp_path: Path) -> None:
    comparison = {
        "generated_at_utc": "2026-07-29T10:00:00+00:00",
        "dataset_name": "Dioula v0.1 local",
        "split": "test",
        "seed": 42,
        "selected_speaker_count": 3,
        "normalization": "NFC et espaces",
        "models": [
            {
                "model_id": "openai/whisper-tiny",
                "model_revision": "a" * 40,
                "evaluated_audio_count": 150,
                "successful_audio_count": 150,
                "wer_micro": 1.14,
                "cer_micro": 0.74,
                "rtf": 0.03,
                "processing_time_seconds": 20.0,
                "device": "cuda",
                "generated_at_utc": "2026-07-29T09:00:00+00:00",
            }
        ],
    }
    environment = {"torch": {"gpu_name": "GPU de test"}}
    comparison_path = tmp_path / "comparison.json"
    environment_path = tmp_path / "environment.json"
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
    environment_path.write_text(json.dumps(environment), encoding="utf-8")

    view = load_benchmark_view(comparison_path, environment_path)

    assert view.dataset_name == "Dioula v0.1 local"
    assert view.hardware == "GPU de test"
    assert view.rows[0]["wer_percent"] == pytest.approx(114.0)


def test_real_model_catalog_registers_factories_without_loading_weights() -> None:
    catalog = load_model_catalog()
    registry = build_model_registry(catalog)

    assert [model.key for model in catalog.enabled_models] == [
        "whisper_tiny_baseline",
        "whisper_small_baseline",
        "whisper_tiny_dioula_pilot",
    ]
    assert registry.available_models == (
        "whisper_small_baseline",
        "whisper_tiny_baseline",
        "whisper_tiny_dioula_pilot",
    )
