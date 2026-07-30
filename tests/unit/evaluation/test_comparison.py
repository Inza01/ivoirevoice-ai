from __future__ import annotations

import json
from pathlib import Path

import pytest

from ivoirevoice.evaluation.comparison import compare_pilot_runs, comparison_markdown
from ivoirevoice.exceptions import ConfigError


def _run(
    root: Path,
    *,
    model_id: str,
    selection_sha256: str,
    wer: float,
    cer: float,
    rtf: float,
) -> Path:
    root.mkdir()
    metadata = {
        "status": "completed",
        "level": "pilot",
        "publication_allowed": False,
        "model_id": model_id,
        "model_revision": "a" * 40,
        "device": "cuda",
        "torch_dtype": "float16",
        "pipeline_commit_sha": "b" * 40,
        "pipeline_git_dirty": True,
        "pipeline_source_sha256": "c" * 64,
        "manifest_sha256": "d" * 64,
        "selection_sha256": selection_sha256,
        "selected_audio_count": 150,
        "selected_speaker_count": 3,
        "seed": 42,
    }
    metrics = {
        "level": "pilot",
        "model_id": model_id,
        "generated_at_utc": "2026-07-29T10:00:00+00:00",
        "batch_size": 1,
        "evaluated_audio_count": 150,
        "successful_audio_count": 150,
        "failed_audio_count": 0,
        "wer_micro": wer,
        "cer_micro": cer,
        "wer_macro_speakers": wer,
        "cer_macro_speakers": cer,
        "mean_latency_seconds": 0.2,
        "latency_p50_seconds": 0.1,
        "latency_p95_seconds": 0.4,
        "audio_duration_seconds": 600.0,
        "processing_time_seconds": rtf * 600.0,
        "rtf": rtf,
        "estimated_full_processing_seconds": 300.0,
    }
    (root / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (root / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return root


def test_comparison_requires_same_selection_and_recommends_lowest_wer(
    tmp_path: Path,
) -> None:
    tiny = _run(
        tmp_path / "tiny",
        model_id="openai/whisper-tiny",
        selection_sha256="e" * 64,
        wer=1.1,
        cer=0.7,
        rtf=0.03,
    )
    small = _run(
        tmp_path / "small",
        model_id="openai/whisper-small",
        selection_sha256="e" * 64,
        wer=1.5,
        cer=0.8,
        rtf=0.09,
    )

    report = compare_pilot_runs((tiny, small))
    markdown = comparison_markdown(report)

    assert report["recommendation"]["model_id"] == "openai/whisper-tiny"
    assert report["selected_audio_count"] == 150
    assert "/home/" not in markdown
    assert "transcription privée" not in markdown


def test_comparison_rejects_different_selections(tmp_path: Path) -> None:
    tiny = _run(
        tmp_path / "tiny",
        model_id="openai/whisper-tiny",
        selection_sha256="e" * 64,
        wer=1.1,
        cer=0.7,
        rtf=0.03,
    )
    small = _run(
        tmp_path / "small",
        model_id="openai/whisper-small",
        selection_sha256="f" * 64,
        wer=1.0,
        cer=0.6,
        rtf=0.09,
    )

    with pytest.raises(ConfigError, match="selection_sha256"):
        compare_pilot_runs((tiny, small))
