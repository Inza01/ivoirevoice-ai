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
from ivoirevoice.models.whisper import WhisperBackend
from ivoirevoice.services.comparison_service import ComparisonService
from ivoirevoice.services.evaluation_service import (
    EvaluationService,
    load_adaptation_error_samples,
    load_benchmark_view,
    load_pilot_adaptation_benchmark,
    relative_reduction_percent,
)
from ivoirevoice.services.export_service import ExportService
from ivoirevoice.services.transcription_service import (
    ModelCatalog,
    ModelDefinition,
    TranscriptionService,
    build_model_registry,
    load_model_catalog,
)
from ivoirevoice.ui.components import render_comparison_cards, render_error_sample


def _wav(path: Path) -> Path:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\x00\x00" * 1_600)
    return path


def _definition(key: str, *, status: str = "baseline") -> ModelDefinition:
    pilot = status == "pilot_adapted"
    return ModelDefinition(
        key=key,
        display_name=(
            "Whisper Tiny Dioula — adapté pilote" if pilot else f"Modèle {key}"
        ),
        backend="whisper",
        status=status,
        model_id="openai/whisper-tiny",
        revision="a" * 40,
        config_path=None,
        device="cpu",
        languages=("dyu",),
        enabled=True,
        model_path=Path("/home/personne/checkpoint-000140") if pilot else None,
        checkpoint_name="checkpoint-000140" if pilot else None,
        configured_language="dyu" if pilot else None,
        training_audio_count=2250 if pilot else None,
        validation_audio_count=600 if pilot else None,
    )


def _comparison(
    tmp_path: Path,
    definitions: tuple[ModelDefinition, ...],
    registry: ModelRegistry,
):
    catalog = ModelCatalog(
        models=definitions,
        max_audio_size_bytes=1024 * 1024,
        max_audio_duration_seconds=10,
        allowed_extensions=(".wav",),
    )
    service = ComparisonService(
        TranscriptionService(catalog, registry),
        EvaluationService(),
    )
    return service.compare(
        audio_path=_wav(tmp_path / "sample.wav"),
        language="dyu",
        model_keys=tuple(item.key for item in definitions),
        reference="a ka taa",
    )


def test_pilot_catalog_uses_environment_path_without_exposing_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint-000140"
    checkpoint.mkdir()
    for filename in ("config.json", "model.safetensors", "preprocessor_config.json"):
        (checkpoint / filename).touch()
    monkeypatch.setenv("IVOIREVOICE_DIOULA_PILOT_MODEL_PATH", str(checkpoint))
    monkeypatch.setenv("IVOIREVOICE_MODEL_CACHE_DIR", str(tmp_path / "cache"))

    catalog = load_model_catalog()
    pilot = catalog.definition("whisper_tiny_dioula_pilot")
    backend = build_model_registry(catalog).create(pilot.key)

    assert pilot.model_path == checkpoint.resolve()
    assert pilot.status == "pilot_adapted"
    assert pilot.checkpoint_name == "checkpoint-000140"
    assert pilot.training_audio_count == 2250
    assert pilot.validation_audio_count == 600
    assert isinstance(backend, WhisperBackend)
    assert backend.settings.model_id == str(checkpoint.resolve())
    assert backend.settings.local_files_only is True


def test_comparison_and_exports_support_three_models_without_private_path(
    tmp_path: Path,
) -> None:
    definitions = (
        _definition("tiny"),
        _definition("small"),
        _definition("pilot", status="pilot_adapted"),
    )
    registry = ModelRegistry()
    for definition in definitions:
        registry.register(definition.key, DummyBackend)

    run = _comparison(tmp_path, definitions, registry)
    exports = ExportService(tmp_path / "exports")
    paths = exports.export_all(run)
    content = "".join(Path(path).read_text(encoding="utf-8") for path in paths)
    rendered = render_comparison_cards(run)

    assert len(run.results) == 3
    assert all(result.success for result in run.results)
    assert "modèle pilote adapté" in rendered
    assert "checkpoint-000140" in rendered
    assert "/home/personne" not in rendered + content
    with Path(paths[1]).open(encoding="utf-8", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 3


def test_pilot_failure_is_isolated_from_baseline(tmp_path: Path) -> None:
    class FailingPilot(ASRBackend):
        @property
        def model_name(self) -> str:
            return "pilot"

        @property
        def supported_languages(self) -> tuple[str, ...]:
            return ("dyu",)

        def load(self) -> None:
            raise RuntimeError("/home/personne/secret")

        def transcribe(self, audio: AudioInput, language: str) -> TranscriptionResult:
            raise AssertionError((audio, language))

        def unload(self) -> None:
            return None

    definitions = (
        _definition("tiny"),
        _definition("pilot", status="pilot_adapted"),
    )
    registry = ModelRegistry()
    registry.register("tiny", DummyBackend)
    registry.register("pilot", FailingPilot)

    run = _comparison(tmp_path, definitions, registry)

    assert run.results[0].success is True
    assert run.results[1].success is False
    assert "/home/" not in (run.results[1].error or "")


def test_benchmark_experiments_are_separate_and_reductions_are_exact(
    tmp_path: Path,
) -> None:
    adaptation = load_pilot_adaptation_benchmark(
        Path("reports/training/pilot_comparison.json")
    )
    historical_payload = {
        "dataset_name": "Dioula v0.1 local",
        "split": "test",
        "selected_audio_count": 150,
        "selected_speaker_count": 3,
        "seed": 42,
        "models": [
            {
                "model_id": "openai/whisper-tiny",
                "model_revision": "a" * 40,
                "evaluated_audio_count": 150,
                "successful_audio_count": 150,
                "wer_micro": 1.14,
                "cer_micro": 0.73,
                "rtf": 0.03,
                "processing_time_seconds": 20,
                "device": "cuda",
            }
        ],
    }
    comparison_path = tmp_path / "historical.json"
    environment_path = tmp_path / "environment.json"
    comparison_path.write_text(json.dumps(historical_payload), encoding="utf-8")
    environment_path.write_text(
        json.dumps({"torch": {"gpu_name": "GPU test"}}),
        encoding="utf-8",
    )
    historical = load_benchmark_view(comparison_path, environment_path)

    assert adaptation.experiment_id == "pilot_adaptation_validation"
    assert adaptation.split == "validation"
    assert adaptation.audio_count == 600
    assert adaptation.rows[1]["wer_relative_reduction_percent"] == pytest.approx(
        relative_reduction_percent(1.1545268890401634, 0.7821647379169503)
    )
    assert historical.experiment_id == "historical_pilot"
    assert historical.audio_count == 150
    assert historical.split == "test"
    assert adaptation.dataset_name != historical.dataset_name


def test_adaptation_error_analysis_uses_validation_and_labels_change(
    tmp_path: Path,
) -> None:
    predictions = tmp_path / "predictions.csv"
    predictions.write_text(
        "audio_id_anonymized,target_text_mvp,baseline_prediction,adapted_prediction\n"
        "utt_safe,a ka taa,a ta,a ka taa\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "utterance_id,audio_path,split\n"
        "utt_safe,validation/sample.wav,validation\n",
        encoding="utf-8",
    )

    samples = load_adaptation_error_samples(predictions, manifest)
    rendered = render_error_sample(samples[0], EvaluationService())

    assert "L’adaptation améliore cet échantillon" in rendered
    assert "Whisper Tiny — baseline" in rendered
    assert "Whisper Tiny Dioula — adapté pilote" in rendered
    assert "Substitutions" in rendered
    assert "validation/sample.wav" not in rendered

    invalid_manifest = tmp_path / "invalid_manifest.csv"
    invalid_manifest.write_text(
        "utterance_id,audio_path,split\nutt_safe,test/sample.wav,test\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="split validation"):
        load_adaptation_error_samples(predictions, invalid_manifest)
