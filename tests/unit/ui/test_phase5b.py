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
    final = status == "final_adapted"
    return ModelDefinition(
        key=key,
        display_name="Whisper Tiny — Dioula Final" if final else f"Modèle {key}",
        backend="whisper",
        status=status,
        model_id="openai/whisper-tiny",
        revision="a" * 40,
        config_path=None,
        device="cpu",
        languages=("dyu",),
        enabled=True,
        model_path=Path("/synthetic/checkpoint-002052") if final else None,
        checkpoint_name="checkpoint-002052" if final else None,
        configured_language="dyu" if final else None,
        training_audio_count=16_425 if final else None,
        validation_audio_count=2_661 if final else None,
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


def test_final_catalog_uses_environment_path_without_exposing_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint-002052"
    checkpoint.mkdir()
    for filename in ("config.json", "model.safetensors", "preprocessor_config.json"):
        (checkpoint / filename).touch()
    monkeypatch.setenv("IVOIREVOICE_DIOULA_FINAL_MODEL_PATH", str(checkpoint))
    monkeypatch.setenv("IVOIREVOICE_MODEL_CACHE_DIR", str(tmp_path / "cache"))

    catalog = load_model_catalog()
    final = catalog.definition("whisper_tiny_dioula_final")
    backend = build_model_registry(catalog).create(final.key)

    assert final.model_path == checkpoint.resolve()
    assert final.status == "final_adapted"
    assert final.display_name == "Whisper Tiny — Dioula Final"
    assert final.checkpoint_name == "checkpoint-002052"
    assert final.training_audio_count == 16_425
    assert final.validation_audio_count == 2_661
    assert isinstance(backend, WhisperBackend)
    assert backend.settings.model_id == str(checkpoint.resolve())
    assert backend.settings.local_files_only is True


def test_comparison_and_exports_support_three_models_without_private_path(
    tmp_path: Path,
) -> None:
    definitions = (
        _definition("tiny"),
        _definition("small"),
        _definition("final", status="final_adapted"),
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
    assert "modèle final gelé" in rendered
    assert "Whisper Tiny — Dioula Final" in rendered
    assert "checkpoint-002052" in rendered
    assert "/synthetic/" not in rendered + content
    with Path(paths[1]).open(encoding="utf-8", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 3


def test_final_failure_is_isolated_from_baseline(tmp_path: Path) -> None:
    class FailingFinal(ASRBackend):
        @property
        def model_name(self) -> str:
            return "final"

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
        _definition("final", status="final_adapted"),
    )
    registry = ModelRegistry()
    registry.register("tiny", DummyBackend)
    registry.register("final", FailingFinal)

    run = _comparison(tmp_path, definitions, registry)

    assert run.results[0].success is True
    assert run.results[1].success is False
    assert "/home/" not in (run.results[1].error or "")


def test_missing_final_checkpoint_fails_clearly_and_keeps_baselines_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("IVOIREVOICE_DIOULA_FINAL_MODEL_PATH", raising=False)
    monkeypatch.setenv("IVOIREVOICE_MODEL_CACHE_DIR", str(tmp_path / "cache"))
    catalog = load_model_catalog()
    registry = build_model_registry(catalog)

    assert registry.create("whisper_tiny_baseline") is not None
    assert registry.create("whisper_small_baseline") is not None
    with pytest.raises(ConfigError, match="IVOIREVOICE_DIOULA_FINAL_MODEL_PATH"):
        registry.create("whisper_tiny_dioula_final")


def test_final_catalog_rejects_a_different_checkpoint_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint-000140"
    checkpoint.mkdir()
    for filename in ("config.json", "model.safetensors", "preprocessor_config.json"):
        (checkpoint / filename).touch()
    monkeypatch.setenv("IVOIREVOICE_DIOULA_FINAL_MODEL_PATH", str(checkpoint))
    monkeypatch.setenv("IVOIREVOICE_MODEL_CACHE_DIR", str(tmp_path / "cache"))

    catalog = load_model_catalog()
    with pytest.raises(ConfigError, match="ne correspond pas au modèle déclaré"):
        build_model_registry(catalog).create("whisper_tiny_dioula_final")


def test_final_catalog_has_exact_live_labels_and_make_wiring() -> None:
    catalog = load_model_catalog()
    choices = {model.key: model.display_name for model in catalog.enabled_models}
    makefile = Path("Makefile").read_text(encoding="utf-8")

    assert choices == {
        "whisper_tiny_baseline": "Whisper Tiny — Baseline",
        "whisper_small_baseline": "Whisper Small — Baseline",
        "whisper_tiny_dioula_final": "Whisper Tiny — Dioula Final",
    }
    assert 'IVOIREVOICE_DIOULA_FINAL_MODEL_PATH="$(DIOULA_FINAL_MODEL_PATH)"' in makefile


def test_public_final_report_is_aggregate_only() -> None:
    report_path = Path("reports/final_holdout_metrics.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    serialized = json.dumps(report)

    assert report["model"] == "Whisper Tiny Dioula Final"
    assert report["final_step"] == 2_052
    assert report["holdout_audio_count"] == 2_624
    assert report["holdout_speaker_count"] == 3
    assert report["wer"] == pytest.approx(0.33262458967472397)
    assert report["cer"] == pytest.approx(0.12380383285671603)
    assert report["rtf"] == pytest.approx(0.007853382888764181)
    assert report["final_loss"] == pytest.approx(0.34643741533523653)
    assert report["substitutions"] == 5_690
    assert report["insertions"] == 1_363
    assert report["deletions"] == 1_864
    assert report["exact_matches"] == 334
    assert report["evaluation_count"] == 1
    assert report["individual_predictions_persisted"] is False
    assert {"speaker_id", "utterance_id", "audio_path", "transcript", "prediction"}.isdisjoint(
        report
    )
    assert "/home/" not in serialized


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
