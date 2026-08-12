from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from ivoirevoice.exceptions import ConfigError
from ivoirevoice.training.full_finetune import (
    FullContext,
    _initial_training_state,
    _latest_checkpoint,
    _optimizer_group,
)
from ivoirevoice.training.whisper_finetune import (
    OptimizerGroupResult,
    amp_metrics,
    apply_optimizer_outcome,
    initialize_or_validate_amp_state,
    restore_runtime_states,
)


class CountingScheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1


class LoadableState:
    def __init__(self) -> None:
        self.loaded: dict[str, Any] | None = None

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.loaded = state


class TorchProxy:
    nn = torch.nn
    float16 = torch.float16
    isfinite = staticmethod(torch.isfinite)

    @staticmethod
    def autocast(**_: Any) -> Any:
        return nullcontext()


class GradientMultiplier(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, multiplier: float) -> torch.Tensor:
        ctx.multiplier = multiplier
        return value

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return gradient * ctx.multiplier, None


class ControlledGradientModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.multiplier = 1e34

    def forward(self, **_: Any) -> Any:
        return SimpleNamespace(
            loss=GradientMultiplier.apply(self.weight, self.multiplier)
        )


def _state() -> dict[str, Any]:
    context = cast(
        FullContext,
        SimpleNamespace(
            initial_checkpoint_sha256="a" * 64,
            config_sha256="b" * 64,
            code_commit="c" * 40,
        ),
    )
    state = _initial_training_state(
        stage="development",
        selection_hash="d" * 64,
        context=context,
        total_steps=1_722,
    )
    initialize_or_validate_amp_state(
        state,
        SimpleNamespace(get_scale=lambda: 65_536.0),
        resumed=False,
    )
    return state


def _result(*, before: float, after: float) -> OptimizerGroupResult:
    skipped = after < before
    return OptimizerGroupResult(
        train_loss=2.0,
        scale_before=before,
        scale_after=after,
        optimizer_step_executed=not skipped,
        amp_skipped=skipped,
    )


def test_real_gradscaler_overflow_returns_skip_then_allows_next_update() -> None:
    model = ControlledGradientModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler("cpu", init_scale=65_536.0)

    first = _optimizer_group(
        row_batches=((cast(Any, object()),),),
        collator=cast(Any, lambda _: {"input_features": torch.ones(1)}),
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        device=torch.device("cpu"),
        torch=TorchProxy,
        max_grad_norm=1.0,
    )

    assert first.optimizer_step_executed is False
    assert first.amp_skipped is True
    assert first.scale_before == 65_536.0
    assert first.scale_after == 32_768.0
    assert optimizer.state == {}

    model.multiplier = 1.0
    second = _optimizer_group(
        row_batches=((cast(Any, object()),),),
        collator=cast(Any, lambda _: {"input_features": torch.ones(1)}),
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        device=torch.device("cpu"),
        torch=TorchProxy,
        max_grad_norm=1.0,
    )

    assert second.optimizer_step_executed is True
    assert second.amp_skipped is False
    assert second.scale_before == 32_768.0
    assert second.scale_after == 32_768.0
    assert optimizer.state


def test_initial_skip_then_stable_updates_advance_only_real_step_clocks() -> None:
    state = _state()
    scheduler = CountingScheduler()
    evaluation_events: list[int] = []
    checkpoint_events: list[int] = []

    updated = apply_optimizer_outcome(
        state=state,
        result=_result(before=65_536.0, after=32_768.0),
        scheduler=scheduler,
        max_consecutive_amp_skips=4,
        stage="development",
    )
    if updated and state["global_step"] == 1:
        evaluation_events.append(1)
        checkpoint_events.append(1)

    assert updated is False
    assert state["optimizer_attempts"] == 1
    assert state["successful_optimizer_steps"] == 0
    assert state["global_step"] == 0
    assert state["amp_skipped_steps"] == 1
    assert state["consecutive_amp_skips"] == 1
    assert state["final_grad_scale"] == 32_768.0
    assert scheduler.steps == 0
    assert evaluation_events == []
    assert checkpoint_events == []

    for expected_step in range(1, 5):
        updated = apply_optimizer_outcome(
            state=state,
            result=_result(before=32_768.0, after=32_768.0),
            scheduler=scheduler,
            max_consecutive_amp_skips=4,
            stage="development",
        )
        assert updated is True
        assert state["global_step"] == expected_step
        assert state["successful_optimizer_steps"] == expected_step

    assert state["optimizer_attempts"] == 5
    assert state["amp_skipped_steps"] == 1
    assert state["consecutive_amp_skips"] == 0
    assert state["max_consecutive_amp_skips_observed"] == 1
    assert scheduler.steps == 4


def test_fifth_consecutive_amp_skip_stops_with_detailed_guard() -> None:
    state = _state()
    scheduler = CountingScheduler()

    for index in range(4):
        assert (
            apply_optimizer_outcome(
                state=state,
                result=_result(
                    before=65_536.0 / (2**index),
                    after=65_536.0 / (2 ** (index + 1)),
                ),
                scheduler=scheduler,
                max_consecutive_amp_skips=4,
                stage="refit",
            )
            is False
        )

    with pytest.raises(ConfigError, match="5 > 4"):
        apply_optimizer_outcome(
            state=state,
            result=_result(before=4_096.0, after=2_048.0),
            scheduler=scheduler,
            max_consecutive_amp_skips=4,
            stage="refit",
        )

    assert state["optimizer_attempts"] == 5
    assert state["amp_skipped_steps"] == 5
    assert state["global_step"] == 0
    assert scheduler.steps == 0


def test_resume_restores_scaler_32768_and_all_mutable_runtime_states(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint-000010"
    checkpoint.mkdir()
    torch.save({"optimizer": 10}, checkpoint / "optimizer.pt")
    torch.save({"scheduler": 10}, checkpoint / "scheduler.pt")
    saved_scaler = torch.amp.GradScaler("cpu", init_scale=32_768.0)
    torch.save(saved_scaler.state_dict(), checkpoint / "scaler.pt")
    optimizer = LoadableState()
    scheduler = LoadableState()
    resumed_scaler = torch.amp.GradScaler("cpu", init_scale=65_536.0)

    restore_runtime_states(
        checkpoint=checkpoint,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=resumed_scaler,
        torch=torch,
        device=torch.device("cpu"),
    )
    state = _state()
    state.update(
        {
            "global_step": 10,
            "successful_optimizer_steps": 10,
            "optimizer_attempts": 11,
            "amp_skipped_steps": 1,
            "consecutive_amp_skips": 0,
            "max_consecutive_amp_skips_observed": 1,
            "initial_grad_scale": 65_536.0,
            "final_grad_scale": 32_768.0,
        }
    )
    initialize_or_validate_amp_state(state, resumed_scaler, resumed=True)

    assert optimizer.loaded == {"optimizer": 10}
    assert scheduler.loaded == {"scheduler": 10}
    assert resumed_scaler.get_scale() == 32_768.0


def test_resume_rejects_missing_or_inconsistent_amp_state() -> None:
    state = _state()
    state.pop("amp_skipped_steps")
    with pytest.raises(ConfigError, match="amp_skipped_steps"):
        initialize_or_validate_amp_state(
            state,
            SimpleNamespace(get_scale=lambda: 32_768.0),
            resumed=True,
        )

    state = _state()
    state["final_grad_scale"] = 32_768.0
    with pytest.raises(ConfigError, match="incohérent"):
        initialize_or_validate_amp_state(
            state,
            SimpleNamespace(get_scale=lambda: 65_536.0),
            resumed=True,
        )


def test_checkpoint_without_scaler_is_never_resumable(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-000010"
    checkpoint.mkdir()
    for name in (
        "full_trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "config.json",
    ):
        (checkpoint / name).write_text("{}\n", encoding="utf-8")

    assert _latest_checkpoint(tmp_path) is None
    (checkpoint / "scaler.pt").write_bytes(b"scaler")
    assert _latest_checkpoint(tmp_path) == checkpoint


def test_amp_report_is_aggregate_numeric_and_contains_no_private_payload() -> None:
    state = _state()
    scheduler = CountingScheduler()
    apply_optimizer_outcome(
        state=state,
        result=_result(before=65_536.0, after=32_768.0),
        scheduler=scheduler,
        max_consecutive_amp_skips=4,
        stage="development",
    )

    report = amp_metrics(state)
    serialized = json.dumps(report)

    assert report == {
        "precision": "fp16",
        "initial_grad_scale": 65_536.0,
        "final_grad_scale": 32_768.0,
        "amp_skipped_steps": 1,
        "successful_optimizer_steps": 0,
        "optimizer_attempts": 1,
        "consecutive_amp_skips": 1,
        "max_consecutive_amp_skips_observed": 1,
    }
    assert "transcription" not in serialized
    assert "audio_path" not in serialized
    assert "/home/" not in serialized
