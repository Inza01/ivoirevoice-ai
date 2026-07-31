from __future__ import annotations

import csv
from pathlib import Path

import torch

from ivoirevoice.training.audit import ManifestRow
from ivoirevoice.training.smoke_overfit import (
    PreparedSample,
    _collate,
    _loss_summary,
    _write_loss_plot,
    _write_private_predictions,
)


def _row(index: int) -> ManifestRow:
    return ManifestRow(
        utterance_id=f"utt_{index}",
        speaker_id=f"spk_{index}",
        gender_folder="women",
        language="dyu",
        text_raw="Á hakili kà go ↘",
        text_no_tones="A hakili ka go ↘",
        target_text="A hakili ka go",
        audio_path=f"women/anonymous/{index}.wav",
        duration_seconds=2.0,
        sample_rate_hz=16_000,
        channels=1,
        audio_sha256=f"{index + 1:064x}",
        split="train",
        usage_scope="local_research_only",
    )


def test_collate_uses_minus_one_hundred_only_for_label_padding() -> None:
    samples = (
        PreparedSample(
            _row(0),
            torch.zeros((80, 20)),
            torch.ones(20, dtype=torch.long),
            torch.tensor([1, 2]),
        ),
        PreparedSample(
            _row(1),
            torch.ones((80, 20)),
            torch.ones(20, dtype=torch.long),
            torch.tensor([3, 4, 5]),
        ),
    )

    features, attention_masks, labels = _collate(samples, (0, 1), torch)

    assert features.shape == (2, 80, 20)
    assert attention_masks.shape == (2, 20)
    assert labels.tolist() == [[1, 2, -100], [3, 4, 5]]


def test_loss_summary_exposes_first_last_and_regular_checkpoints() -> None:
    history = [
        {
            "step": step,
            "loss": 2.0 / step,
            "learning_rate": 0.0001,
            "elapsed_seconds": float(step),
        }
        for step in range(1, 21)
    ]

    summary = _loss_summary(history)

    assert summary["first_step_loss"] == 2.0
    assert summary["last_step_loss"] == 0.1
    assert summary["loss_reduction_fraction"] > 0.8
    assert [item["step"] for item in summary["checkpoints"]] == [1, 10, 20]


def test_loss_plot_and_predictions_have_required_private_fields(tmp_path: Path) -> None:
    history = [
        {
            "step": step,
            "loss": 2.0 / step,
            "learning_rate": 0.0001,
            "elapsed_seconds": float(step),
        }
        for step in range(1, 4)
    ]
    plot = tmp_path / "loss.png"
    predictions = tmp_path / "predictions.csv"
    samples = (
        PreparedSample(
            _row(0),
            torch.zeros((80, 20)),
            torch.ones(20, dtype=torch.long),
            torch.tensor([1, 2]),
        ),
    )

    _write_loss_plot(plot, history)
    _write_private_predictions(predictions, samples, ("avant",), ("après",))

    assert plot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    with predictions.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["text_raw"] == samples[0].row.text_raw
    assert row["text_no_tones"] == samples[0].row.text_no_tones
    assert row["target_text_mvp"] == samples[0].row.target_text
    assert "audio_path" not in row
