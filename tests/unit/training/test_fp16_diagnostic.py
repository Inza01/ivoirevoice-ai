from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from ivoirevoice.exceptions import ConfigError
from ivoirevoice.training.audit import REQUIRED_COLUMNS
from ivoirevoice.training.fp16_diagnostic import (
    AmpStepOutcome,
    DiagnosticContext,
    DiagnosticProgress,
    _build_summary,
    apply_amp_step,
    bounded_optimizer_groups,
    classify_diagnostic,
    inspect_gradients,
    load_diagnostic_train_rows,
    write_diagnostic_report,
)
from ivoirevoice.training.full_settings import FullTrainingSettings


class CountingScheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1


def _optimizer_state_step(optimizer: Any, parameter: Any) -> int:
    value = optimizer.state.get(parameter, {}).get("step", 0)
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


def _scaled_cpu_runtime() -> tuple[Any, Any, Any, Any]:
    model = torch.nn.Linear(2, 1)
    parameter = next(model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cpu")
    scaler.scale(model(torch.ones(1, 2)).sum()).backward()
    return model, parameter, optimizer, scaler


def test_valid_adamw_none_return_still_executes_step() -> None:
    model, parameter, optimizer, scaler = _scaled_cpu_runtime()
    scheduler = CountingScheduler()
    scaler.unscale_(optimizer)

    outcome = apply_amp_step(optimizer, scaler, scheduler)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert outcome == AmpStepOutcome(65_536.0, 65_536.0, True, False)
    assert _optimizer_state_step(optimizer, parameter) == 1
    assert scheduler.steps == 1
    assert all(torch.isfinite(value).all() for value in model.parameters())


def test_real_overflow_skips_step_and_does_not_advance_scheduler() -> None:
    model, parameter, optimizer, scaler = _scaled_cpu_runtime()
    scheduler = CountingScheduler()
    for current in model.parameters():
        if current.grad is not None:
            current.grad.fill_(float("inf"))
    scaler.unscale_(optimizer)

    stats = inspect_gradients(model, torch)
    outcome = apply_amp_step(optimizer, scaler, scheduler)

    assert stats["gradient_tensors_with_inf"] > 0
    assert stats["nonfinite_gradient_values"] > 0
    assert stats["gradient_global_norm"] is None
    assert outcome == AmpStepOutcome(65_536.0, 32_768.0, False, True)
    assert _optimizer_state_step(optimizer, parameter) == 0
    assert scheduler.steps == 0


def test_progress_continues_after_skip_without_counting_success() -> None:
    progress = DiagnosticProgress()

    progress.record(AmpStepOutcome(65_536.0, 32_768.0, False, True))
    assert progress.optimizer_attempts == 1
    assert progress.successful_optimizer_steps == 0
    assert progress.amp_skipped_steps == 1

    progress.record(AmpStepOutcome(32_768.0, 32_768.0, True, False))
    assert progress.optimizer_attempts == 2
    assert progress.successful_optimizer_steps == 1
    assert progress.max_consecutive_skips == 1


def test_optimizer_groups_are_strictly_bounded_to_32_attempts() -> None:
    rows = cast(tuple[Any, ...], tuple(object() for _ in range(600)))

    groups = bounded_optimizer_groups(
        rows,
        batch_size=4,
        accumulation_steps=4,
        max_optimizer_attempts=32,
    )

    assert len(groups) == 32
    assert all(len(group) == 4 for group in groups)
    assert sum(len(batch) for group in groups for batch in group) == 512


def _manifest_payload(
    *,
    split: str,
    identifier: str,
    audio_path: str,
    audio_hash: str,
    speaker: str,
) -> dict[str, str]:
    return {
        "utterance_id": identifier,
        "speaker_id": speaker,
        "gender_folder": "anonymous",
        "language": "dyu",
        "text_raw": "PRIVATE TEXT MUST NOT LEAK",
        "text_without_tones_nfc": "PRIVATE TEXT MUST NOT LEAK",
        "target_text_mvp": "train label" if split == "train" else "PRIVATE HOLDOUT",
        "audio_path": audio_path,
        "duration_seconds": "1.0",
        "sample_rate_hz": "16000",
        "channels": "1",
        "audio_sha256": audio_hash,
        "split": split,
        "usage_scope": "local_research_only",
    }


def test_train_loader_never_constructs_or_touches_non_train_rows(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    train_path = data_root / "train/anonymous/train.wav"
    train_path.parent.mkdir(parents=True)
    train_path.write_bytes(b"train-audio")
    train_hash = hashlib.sha256(b"train-audio").hexdigest()
    missing_hash = hashlib.sha256(b"not-present").hexdigest()
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerows(
            [
                _manifest_payload(
                    split="train",
                    identifier="train-1",
                    audio_path="train/anonymous/train.wav",
                    audio_hash=train_hash,
                    speaker="train-speaker",
                ),
                _manifest_payload(
                    split="validation",
                    identifier="validation-private",
                    audio_path="validation/private/missing.wav",
                    audio_hash=missing_hash,
                    speaker="validation-private",
                ),
                _manifest_payload(
                    split="test",
                    identifier="final-holdout-private",
                    audio_path="test/private/missing.wav",
                    audio_hash=missing_hash,
                    speaker="test-private",
                ),
            ]
        )
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "manifest_sha256": manifest_hash,
                "dataset_version": "0.1.0-local",
                "publication_allowed": False,
                "model_derivative_publication_allowed": False,
                "usage_scope": "local_research_only",
                "audio_count_by_split": {"train": 1, "validation": 1, "test": 1},
            }
        ),
        encoding="utf-8",
    )
    settings = cast(
        FullTrainingSettings,
        SimpleNamespace(
            manifest_path=manifest_path,
            dataset_metadata_path=metadata_path,
            expected_manifest_sha256=manifest_hash,
            expected_dataset_version="0.1.0-local",
            train_split="train",
            train_audio_count=1,
            train_speaker_count=1,
            max_audio_seconds=30.0,
            dataset_root=data_root,
        ),
    )

    rows, actual_hash = load_diagnostic_train_rows(settings)

    assert actual_hash == manifest_hash
    assert len(rows) == 1
    assert rows[0].split == "train"
    assert rows[0].utterance_id == "train-1"


def _attempt(index: int, *, skipped: bool) -> dict[str, Any]:
    scale_before = 65_536.0 if index == 1 else 32_768.0
    scale_after = 32_768.0
    return {
        "optimizer_attempt": index,
        "scale_before": scale_before,
        "scale_after": scale_after,
        "optimizer_step_executed": not skipped,
        "amp_skip": skipped,
        "nonfinite_gradient_values": 1 if skipped else 0,
        "terminal_error": None,
        "parameters_finite": True,
        "samples": [
            {
                "utterance_hash": "a" * 64,
                "duration_seconds": 1.0,
                "feature_length": 3000,
                "label_length": 8,
            }
        ],
    }


def test_initial_skip_followed_by_stability_is_category_a() -> None:
    attempts = [_attempt(1, skipped=True)] + [
        _attempt(index, skipped=False) for index in range(2, 11)
    ]
    progress = DiagnosticProgress()
    progress.record(AmpStepOutcome(65_536.0, 32_768.0, False, True))
    for _ in range(9):
        progress.record(AmpStepOutcome(32_768.0, 32_768.0, True, False))

    category, _ = classify_diagnostic(attempts, progress)

    assert category == "CATEGORY_A_INITIAL_SCALE_CALIBRATION"


def test_report_is_private_checkpoint_free_non_resumable_and_not_overwritten(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "artifacts/fp16_diagnostic"
    settings = cast(
        FullTrainingSettings,
        SimpleNamespace(
            initial_checkpoint_path=Path("checkpoint-000140"),
            learning_rate=1e-5,
            train_batch_size=4,
            gradient_accumulation_steps=4,
            max_grad_norm=1.0,
            seed=42,
            fp16_diagnostic_max_optimizer_attempts=32,
            fp16_diagnostic_output_directory=output_directory,
        ),
    )
    context = DiagnosticContext(
        train_rows=(),
        model_settings=SimpleNamespace(),
        code_commit="a" * 40,
        config_sha256="b" * 64,
        manifest_sha256="c" * 64,
        initial_checkpoint_sha256="d" * 64,
        train_selection_sha256="e" * 64,
        run_id="f" * 64,
    )
    attempts = [_attempt(1, skipped=True)]
    progress = DiagnosticProgress()
    progress.record(AmpStepOutcome(65_536.0, 32_768.0, False, True))
    summary = _build_summary(
        settings=settings,
        context=context,
        attempts=attempts,
        progress=progress,
        gpu_name="test-gpu",
        bf16_supported=True,
    )

    report = write_diagnostic_report(settings, summary)
    serialized = report.read_text(encoding="utf-8")

    assert summary["optimizer_attempts"] == 1
    assert summary["successful_optimizer_steps"] == 0
    assert summary["isolation"] == {
        "train_only": True,
        "validation_used": False,
        "historical_test_used": False,
        "final_holdout_used": False,
        "checkpoint_created": False,
        "model_output_created": False,
        "resume_allowed": False,
    }
    assert "PRIVATE" not in serialized
    assert '"audio_path":' not in serialized
    assert '"speaker_id":' not in serialized
    assert not tuple(tmp_path.rglob("checkpoint-*"))
    with pytest.raises(ConfigError, match="écrasement"):
        write_diagnostic_report(settings, summary)
