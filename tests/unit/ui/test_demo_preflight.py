from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from ivoirevoice.models.dummy import DummyBackend
from ivoirevoice.models.registry import ModelRegistry
from ivoirevoice.services.comparison_service import ComparisonService
from ivoirevoice.services.evaluation_service import EvaluationService
from ivoirevoice.services.export_service import ExportService
from ivoirevoice.services.transcription_service import (
    ModelCatalog,
    ModelDefinition,
    TranscriptionService,
)
from ivoirevoice.ui.app import DemoServices, build_demo_services
from ivoirevoice.ui.components import ABOUT_MARKDOWN

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "demo_preflight.py"
SPEC = importlib.util.spec_from_file_location("ivoirevoice_demo_preflight", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
demo_preflight = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = demo_preflight
SPEC.loader.exec_module(demo_preflight)

SMOKE_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "demo_smoke.py"
SMOKE_SPEC = importlib.util.spec_from_file_location("ivoirevoice_demo_smoke", SMOKE_SCRIPT)
assert SMOKE_SPEC is not None and SMOKE_SPEC.loader is not None
demo_smoke = importlib.util.module_from_spec(SMOKE_SPEC)
sys.modules[SMOKE_SPEC.name] = demo_smoke
SMOKE_SPEC.loader.exec_module(demo_smoke)


def _checkpoint(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "checkpoint-002052"
    checkpoint.mkdir()
    for name in demo_preflight.REQUIRED_MODEL_FILES:
        (checkpoint / name).touch()
    return checkpoint


def _cache(tmp_path: Path) -> Path:
    cache = tmp_path / "cache"
    revisions = {
        "openai/whisper-tiny": "be0ba7c2f24f0127b27863a23a08002af4c2c279",
        "openai/whisper-small": "973afd24965f72e36ca33b3055d56a652f456b4d",
    }
    for model_id, revision in revisions.items():
        snapshot = cache / ("models--" + model_id.replace("/", "--")) / "snapshots" / revision
        snapshot.mkdir(parents=True)
        for name in demo_preflight.REQUIRED_MODEL_FILES:
            (snapshot / name).touch()
    return cache


def _ready_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    cache = _cache(tmp_path)
    monkeypatch.setenv("IVOIREVOICE_DIOULA_FINAL_MODEL_PATH", str(checkpoint))
    monkeypatch.setenv("IVOIREVOICE_MODEL_CACHE_DIR", str(cache))
    monkeypatch.setenv("IVOIREVOICE_UI_HOST", "127.0.0.1")
    monkeypatch.setenv("IVOIREVOICE_UI_PORT", "7860")
    monkeypatch.delenv("IVOIREVOICE_DEMO_AUDIO_PATH", raising=False)
    monkeypatch.delenv("IVOIREVOICE_DEMO_AUDIO_CONFIRMATION", raising=False)
    monkeypatch.setattr(demo_preflight, "_git_state", lambda root: ("main", True))
    monkeypatch.setattr(
        demo_preflight,
        "_python_runtime",
        lambda root: (True, "Python 3.11 / .venv"),
    )
    monkeypatch.setattr(
        demo_preflight,
        "_torch_runtime",
        lambda: (True, True, demo_preflight.EXPECTED_GPU, 12.0),
    )
    monkeypatch.setattr(demo_preflight, "_port_available", lambda host, port: True)
    monkeypatch.setattr(demo_preflight, "_demo_inputs_ignored", lambda root: True)
    monkeypatch.setattr(
        demo_preflight,
        "directory_sha256",
        lambda path: "d9dd6469cd102e98b17d1e0750e51fa9107f3eb0847f130984cf993f033151c1",
    )
    monkeypatch.setattr(
        demo_preflight,
        "_load_final_model",
        lambda: demo_preflight.ModelProbe(
            gpu_name=demo_preflight.EXPECTED_GPU,
            vram_gib=12.0,
            load_seconds=1.0,
            peak_vram_mib=100.0,
        ),
    )


def test_demo_preflight_passes_without_selecting_corpus_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _ready_runtime(tmp_path, monkeypatch)

    report = demo_preflight.run_demo_preflight()

    assert report.ready is True
    assert report.tiny_cached is True
    assert report.small_cached is True
    assert report.final_model_local is True
    assert report.demo_audio_name is None
    assert report.warnings == ("DEMO AUDIO REQUIRED",)
    assert report.demo_audio_required is True

    demo_preflight._print_report(report)

    assert capsys.readouterr().out.rstrip().endswith(
        "READY WITH DEMO AUDIO REQUIRED"
    )


def test_demo_preflight_refuses_non_main_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ready_runtime(tmp_path, monkeypatch)
    model_loaded = False

    def unexpected_load() -> demo_preflight.ModelProbe:
        nonlocal model_loaded
        model_loaded = True
        raise AssertionError("model load must remain unreachable")

    monkeypatch.setattr(demo_preflight, "_git_state", lambda root: ("feat/demo", True))
    monkeypatch.setattr(demo_preflight, "_load_final_model", unexpected_load)

    report = demo_preflight.run_demo_preflight()

    assert report.ready is False
    assert model_loaded is False
    assert next(check for check in report.checks if check.name == "git_branch").passed is False


def test_demo_preflight_refuses_non_loopback_gradio_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ready_runtime(tmp_path, monkeypatch)
    monkeypatch.setenv("IVOIREVOICE_UI_HOST", "0.0.0.0")

    report = demo_preflight.run_demo_preflight()

    assert report.ready is False
    assert next(check for check in report.checks if check.name == "gradio_port").passed is False


def test_demo_preflight_accepts_only_explicit_external_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ready_runtime(tmp_path, monkeypatch)
    external = tmp_path / "external" / "today.wav"
    external.parent.mkdir()
    external.touch()
    monkeypatch.setenv("IVOIREVOICE_DEMO_AUDIO_PATH", str(external))
    monkeypatch.setenv(
        "IVOIREVOICE_DEMO_AUDIO_CONFIRMATION",
        demo_preflight.DEMO_AUDIO_CONFIRMATION,
    )

    report = demo_preflight.run_demo_preflight()

    assert report.ready is True
    assert report.demo_audio_name == "today.wav"
    assert not report.warnings


def test_demo_preflight_rejects_audio_under_private_data_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ready_runtime(tmp_path, monkeypatch)
    data_root = tmp_path / "private_data"
    audio = data_root / "train" / "sample.wav"
    audio.parent.mkdir(parents=True)
    audio.touch()
    monkeypatch.setenv("IVOIREVOICE_DIOULA_DATA_DIR", str(data_root))
    monkeypatch.setenv("IVOIREVOICE_DEMO_AUDIO_PATH", str(audio))
    monkeypatch.setenv(
        "IVOIREVOICE_DEMO_AUDIO_CONFIRMATION",
        demo_preflight.DEMO_AUDIO_CONFIRMATION,
    )

    report = demo_preflight.run_demo_preflight()

    assert report.ready is False
    check = next(item for item in report.checks if item.name == "demo_audio")
    assert check.detail == "fichier situé sous une racine privée"


def test_demo_make_target_withholds_private_roots_and_uses_preflight() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    demo_recipe = makefile.split("demo: demo-preflight", maxsplit=1)[1].split(
        "\naudit-dioula:", maxsplit=1
    )[0]

    assert 'IVOIREVOICE_DIOULA_DATA_DIR=""' in demo_recipe
    assert 'IVOIREVOICE_ARTIFACTS_DIR=""' in demo_recipe
    assert 'IVOIREVOICE_DIOULA_PILOT_MODEL_PATH=""' in demo_recipe
    assert "IVOIREVOICE_DIOULA_FINAL_MODEL_PATH" in demo_recipe
    assert 'HF_HUB_OFFLINE="1"' in demo_recipe
    assert 'TRANSFORMERS_OFFLINE="1"' in demo_recipe
    assert "one_time_final_holdout" not in demo_recipe
    assert "full_finetune" not in demo_recipe


def test_public_final_results_are_visible_without_holdout_access() -> None:
    assert "33,26 % de WER" in ABOUT_MARKDOWN
    assert "12,38 %" in ABOUT_MARKDOWN
    assert "RTF de 0,00785" in ABOUT_MARKDOWN
    assert "2 624 audios" in ABOUT_MARKDOWN
    assert "3 locuteurs" in ABOUT_MARKDOWN
    assert "2 052 steps réussis" in ABOUT_MARKDOWN
    assert "Independent final holdout — evaluated once" in ABOUT_MARKDOWN
    assert "Le holdout n'est jamais réévalué" in ABOUT_MARKDOWN


def test_secure_demo_explains_why_private_error_analysis_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("IVOIREVOICE_ARTIFACTS_DIR", raising=False)
    monkeypatch.delenv("IVOIREVOICE_DIOULA_DATA_DIR", raising=False)

    services = build_demo_services()

    assert services.error_samples == ()
    assert (
        services.error_samples_error
        == "Analyse détaillée non disponible en mode démonstration sécurisé."
    )


def _synthetic_demo_services(tmp_path: Path) -> DemoServices:
    definitions = tuple(
        ModelDefinition(
            key=key,
            display_name=key,
            backend="whisper",
            status="baseline",
            model_id="synthetic",
            revision="a" * 40,
            config_path=None,
            device="cpu",
            languages=("dyu",),
            enabled=True,
        )
        for key in demo_smoke.MODEL_KEYS
    )
    catalog = ModelCatalog(
        models=definitions,
        max_audio_size_bytes=1024 * 1024,
        max_audio_duration_seconds=10,
        allowed_extensions=(".wav",),
    )
    registry = ModelRegistry()
    for key in demo_smoke.MODEL_KEYS:
        registry.register(key, DummyBackend)
    evaluation = EvaluationService()
    return DemoServices(
        catalog=catalog,
        comparison=ComparisonService(
            TranscriptionService(catalog, registry),
            evaluation,
        ),
        evaluation=evaluation,
        exports=ExportService(tmp_path / "exports"),
        pilot_benchmark=None,
        pilot_benchmark_error=None,
        historical_benchmark=None,
        historical_benchmark_error=None,
        error_samples=(),
        error_samples_error=None,
        dataset_root=None,
    )


def test_versioned_demo_smoke_uses_only_synthetic_audio(tmp_path: Path) -> None:
    services = _synthetic_demo_services(tmp_path)

    report = demo_smoke.run_demo_smoke(services)

    assert report.passed is True
    assert report.model_statuses == tuple((key, True) for key in demo_smoke.MODEL_KEYS)
    assert report.export_suffixes == (".json", ".csv", ".txt")
    assert report.private_path_exposed is False
    assert not (tmp_path / "synthetic_silence.wav").exists()


def test_demo_smoke_target_cannot_reach_training_or_holdout() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    recipe = makefile.split("demo-smoke: demo-preflight", maxsplit=1)[1].split(
        "\ndemo: demo-preflight", maxsplit=1
    )[0]

    assert 'IVOIREVOICE_DIOULA_DATA_DIR=""' in recipe
    assert 'IVOIREVOICE_ARTIFACTS_DIR=""' in recipe
    assert "scripts/demo_smoke.py" in recipe
    assert "one_time_final_holdout" not in recipe
    assert "full_finetune" not in recipe
