from __future__ import annotations

import pytest

from ivoirevoice.evaluation.metrics import (
    ScoredItem,
    compute_evaluation_metrics,
    edit_counts,
    percentile,
)


def test_edit_counts_and_micro_macro_metrics() -> None:
    counts = edit_counts(("a", "b", "c"), ("a", "x", "c", "d"))
    items = (
        ScoredItem(
            speaker_id="spk_a",
            reference_normalized="a b c",
            prediction_normalized="a x c d",
            audio_duration_seconds=2.0,
            processing_time_seconds=1.0,
        ),
        ScoredItem(
            speaker_id="spk_b",
            reference_normalized="un deux",
            prediction_normalized="un deux",
            audio_duration_seconds=3.0,
            processing_time_seconds=1.5,
        ),
    )

    metrics = compute_evaluation_metrics(items)

    assert counts.substitutions == 1
    assert counts.insertions == 1
    assert counts.deletions == 0
    assert metrics["wer_micro"] == pytest.approx(2 / 5)
    assert metrics["wer_macro_speakers"] == pytest.approx((2 / 3) / 2)
    assert metrics["rtf"] == 0.5
    assert metrics["failure_rate"] == 0.0


def test_latency_percentiles_and_failures() -> None:
    items = (
        ScoredItem("spk", "a", "a", 1.0, 1.0),
        ScoredItem("spk", "b", "b", 1.0, 2.0),
        ScoredItem("spk", "c", "c", 1.0, 3.0),
        ScoredItem("spk", "d", "", 1.0, 0.0, "RuntimeError"),
    )

    metrics = compute_evaluation_metrics(items)

    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert metrics["latency_p50_seconds"] == 2.0
    assert metrics["latency_p95_seconds"] == pytest.approx(2.9)
    assert metrics["failure_rate"] == 0.25
