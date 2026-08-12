from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from ivoirevoice.exceptions import ConfigError
from ivoirevoice.training.development_selection_finalizer import (
    CANDIDATE_PUBLIC_FILENAME,
    CONFIRMATION_ENVIRONMENT_VARIABLE,
    CONFIRMATION_VALUE,
    DEVELOPMENT_DECISION_FILENAME,
    FINAL_EVALUATION_RECEIPT_FILENAME,
    FINAL_MODEL_MANIFEST_FILENAME,
    FINALIZATION_PUBLIC_FILENAME,
    METRIC_POLICY,
    REFIT_STARTED_FILENAME,
    TRAINER_STATE_FILENAME,
    _load_review,
    run_development_selection_finalizer,
)
from ivoirevoice.training.full_settings import FullTrainingSettings
from ivoirevoice.training.whisper_finetune import (
    directory_sha256,
    identity_sha256,
    metric_rank,
    refit_step_budget,
)


def _metrics(wer: float, cer: float, loss: float) -> dict[str, Any]:
    return {
        "validation_loss": loss,
        "wer_micro": wer,
        "cer_micro": cer,
        "rtf": 0.015,
        "word_substitutions": 100,
        "word_insertions": 20,
        "word_deletions": 30,
        "audio_count": 2_661,
        "speaker_count": 3,
    }


def _checkpoint_state(
    *,
    step: int,
    completed: bool,
    selection_hash: str,
    pilot_hash: str,
    config_hash: str,
    development_commit: str,
    run_id: str,
) -> dict[str, Any]:
    return {
        "stage": "development",
        "selection_sha256": selection_hash,
        "initial_checkpoint_sha256": pilot_hash,
        "config_sha256": config_hash,
        "code_commit": development_commit,
        "run_id": run_id,
        "global_step": step,
        "completed": completed,
    }


def _write_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.mkdir(parents=True)
    (path / TRAINER_STATE_FILENAME).write_text(json.dumps(state), encoding="utf-8")
    (path / "model.safetensors").write_bytes(f"weights-{path.name}".encode())
    (path / "config.json").write_text("{}\n", encoding="utf-8")


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FullTrainingSettings, Path, Path, Path]:
    artifact_root = tmp_path / "artifacts"
    shareable_root = artifact_root / "shareable"
    checkpoint_root = tmp_path / "checkpoints/full-finetune-whisper-tiny-dy"
    development_root = checkpoint_root / "development"
    pilot_root = tmp_path / "pilot/checkpoint-000140"
    artifact_root.mkdir(parents=True)
    shareable_root.mkdir()
    pilot_root.mkdir(parents=True)
    (pilot_root / "model.safetensors").write_bytes(b"pilot")
    config_path = tmp_path / "full.yaml"
    config_path.write_text("immutable-config\n", encoding="utf-8")
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    pilot_hash = directory_sha256(pilot_root)
    selection_hash = "d" * 64
    development_commit = "a" * 40
    evaluation_commit = "b" * 40
    finalization_commit = "f" * 40
    manifest_hash = "9" * 64
    validation_fingerprint = "7" * 64
    run_identity = {
        "stage": "development",
        "selection_sha256": selection_hash,
        "initial_checkpoint_sha256": pilot_hash,
        "config_sha256": config_hash,
        "code_commit": development_commit,
    }
    run_id = identity_sha256(run_identity)
    current_path = development_root / "checkpoint-001507"
    candidate_path = development_root / "checkpoint-001720"
    _write_checkpoint(
        current_path,
        _checkpoint_state(
            step=1_507,
            completed=False,
            selection_hash=selection_hash,
            pilot_hash=pilot_hash,
            config_hash=config_hash,
            development_commit=development_commit,
            run_id=run_id,
        ),
    )
    _write_checkpoint(
        candidate_path,
        _checkpoint_state(
            step=1_720,
            completed=True,
            selection_hash=selection_hash,
            pilot_hash=pilot_hash,
            config_hash=config_hash,
            development_commit=development_commit,
            run_id=run_id,
        ),
    )
    current_metrics = _metrics(0.446934179925431, 0.1855351458496065, 0.5085993105645737)
    candidate_metrics = _metrics(0.4412861309018421, 0.1836537573176775, 0.5047171142082473)
    current_rank = metric_rank(current_metrics, 1_507)
    candidate_rank = metric_rank(candidate_metrics, 1_720)
    proposed_budget = refit_step_budget(1_720, 861, 1_027)
    decision = {
        "schema_version": 1,
        "status": "frozen",
        "code_commit": development_commit,
        "config_sha256": config_hash,
        "manifest_sha256": manifest_hash,
        "development_selection_sha256": selection_hash,
        "refit_selection_sha256": "8" * 64,
        "initial_checkpoint_sha256": pilot_hash,
        "best_checkpoint_name": current_path.name,
        "best_checkpoint_sha256": directory_sha256(current_path),
        "best_development_step": 1_507,
        "development_steps_per_epoch": 861,
        "selected_epoch_fraction": 1_507 / 861,
        "refit_steps_per_epoch": 1_027,
        "refit_step_budget": refit_step_budget(1_507, 861, 1_027),
        "best_validation_metrics": {
            **current_metrics,
            "evaluated_audio_count": 2_661,
        },
        "historical_test_used": False,
        "final_holdout_used": False,
    }
    (artifact_root / DEVELOPMENT_DECISION_FILENAME).write_text(
        json.dumps(decision), encoding="utf-8"
    )
    candidate_report = {
        "schema_version": 1,
        "status": "candidate_evaluated_not_finalized",
        "evaluation_identity": {
            "stage": "development-final-validation",
            "development_run_id": run_id,
            "development_git_commit": development_commit,
            "evaluation_git_commit": evaluation_commit,
            "config_hash": config_hash,
            "pilot_checkpoint_hash": pilot_hash,
            "current_checkpoint_hash": directory_sha256(current_path),
            "candidate_checkpoint_hash": directory_sha256(candidate_path),
            "validation_manifest_hash": manifest_hash,
            "validation_fingerprint": validation_fingerprint,
            "decoding_config_hash": "e" * 64,
            "metric_policy": list(METRIC_POLICY),
        },
        "current_best": {
            "checkpoint_name": current_path.name,
            "step": 1_507,
            "metrics": current_metrics,
            "metric_rank": list(current_rank),
        },
        "candidate": {
            "checkpoint_name": candidate_path.name,
            "step": 1_720,
            "metrics": candidate_metrics,
            "metric_rank": list(candidate_rank),
        },
        "proposed_best": candidate_path.name,
        "proposed_best_step": 1_720,
        "proposed_refit_budget": proposed_budget,
        "metric_policy": list(METRIC_POLICY),
        "frozen_decision_modified": False,
        "historical_test_accessed": False,
        "final_holdout_accessed": False,
        "final_holdout_decoded": False,
        "privacy": {
            "contains_transcriptions": False,
            "contains_predictions": False,
            "contains_sample_identifiers": False,
            "contains_speaker_identifiers": False,
            "contains_local_paths": False,
        },
    }
    (shareable_root / CANDIDATE_PUBLIC_FILENAME).write_text(
        json.dumps(candidate_report), encoding="utf-8"
    )
    candidate_path_public = shareable_root / CANDIDATE_PUBLIC_FILENAME
    approval_path = tmp_path / "development_selection_approval.json"
    approval_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "approved",
                "methodological_authorization": ("FINALIZE DEVELOPMENT SELECTION APPROUVÉ"),
                "candidate_report_filename": CANDIDATE_PUBLIC_FILENAME,
                "candidate_report_sha256": hashlib.sha256(
                    candidate_path_public.read_bytes()
                ).hexdigest(),
                "evaluation_identity": {
                    field: candidate_report["evaluation_identity"][field]
                    for field in (
                        "development_run_id",
                        "development_git_commit",
                        "evaluation_git_commit",
                        "config_hash",
                        "candidate_checkpoint_hash",
                        "validation_fingerprint",
                    )
                },
                "metric_policy": list(METRIC_POLICY),
            }
        ),
        encoding="utf-8",
    )
    settings = cast(
        FullTrainingSettings,
        SimpleNamespace(
            config_path=config_path,
            initial_checkpoint_path=pilot_root,
            artifact_output_directory=artifact_root,
            shareable_output_directory=shareable_root,
            development_checkpoint_directory=development_root,
            refit_checkpoint_directory=checkpoint_root / "refit",
            validation_audio_count=2_661,
            validation_speaker_count=3,
            approval_path=approval_path,
        ),
    )
    monkeypatch.setenv(CONFIRMATION_ENVIRONMENT_VARIABLE, CONFIRMATION_VALUE)
    monkeypatch.setattr(
        "ivoirevoice.training.development_selection_finalizer.git_provenance",
        lambda _: (finalization_commit, True),
    )
    monkeypatch.setattr(
        "ivoirevoice.training.development_selection_finalizer.APPROVAL_PATH",
        approval_path,
    )
    return settings, current_path, candidate_path, artifact_root


def _mutate_candidate_report(
    settings: FullTrainingSettings,
    mutation: Any,
    *,
    renew_approval: bool = True,
) -> None:
    path = settings.shareable_output_directory / CANDIDATE_PUBLIC_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutation(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    if renew_approval:
        approval_path = cast(Any, settings).approval_path
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
        approval["candidate_report_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        for field in approval["evaluation_identity"]:
            approval["evaluation_identity"][field] = payload["evaluation_identity"][field]
        approval_path.write_text(json.dumps(approval), encoding="utf-8")


def test_finalizer_consumes_report_recalculates_and_preserves_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, current, candidate, artifact_root = _fixture(tmp_path, monkeypatch)
    checkpoint_hashes_before = (directory_sha256(current), directory_sha256(candidate))

    result = run_development_selection_finalizer(settings)

    assert result["status"] == "development_selection_finalized"
    assert result["new_best"]["checkpoint"] == "checkpoint-001720"
    assert result["new_best"]["refit_budget"] == refit_step_budget(1_720, 861, 1_027)
    decision = json.loads(
        (artifact_root / DEVELOPMENT_DECISION_FILENAME).read_text(encoding="utf-8")
    )
    assert decision["development_selection_finalized"] is True
    assert decision["best_checkpoint_name"] == "checkpoint-001720"
    assert decision["best_development_step"] == 1_720
    assert (
        decision["best_validation_metrics"]["wer_micro"]
        < (decision["previous_best_metrics"]["wer_micro"])
    )
    assert decision["refit_step_budget"] == refit_step_budget(1_720, 861, 1_027)
    assert decision["previous_best_checkpoint"] == "checkpoint-001507"
    assert decision["previous_best_step"] == 1_507
    assert decision["previous_refit_budget"] == refit_step_budget(1_507, 861, 1_027)
    assert decision["selection_history"][-1]["checkpoint"] == "checkpoint-001507"
    assert checkpoint_hashes_before == (
        directory_sha256(current),
        directory_sha256(candidate),
    )
    assert not tuple(artifact_root.rglob("*.tmp"))


def test_finalizer_is_idempotent_for_exact_same_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, _, _, artifact_root = _fixture(tmp_path, monkeypatch)
    run_development_selection_finalizer(settings)
    decision_path = artifact_root / DEVELOPMENT_DECISION_FILENAME
    report_path = settings.shareable_output_directory / FINALIZATION_PUBLIC_FILENAME
    decision_before = decision_path.read_bytes()
    report_before = report_path.read_bytes()

    result = run_development_selection_finalizer(settings)

    assert result["status"] == "already_finalized_same_decision"
    assert decision_path.read_bytes() == decision_before
    assert report_path.read_bytes() == report_before


def test_confirmation_is_required_before_reading_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, _, _, _ = _fixture(tmp_path, monkeypatch)
    monkeypatch.delenv(CONFIRMATION_ENVIRONMENT_VARIABLE)
    monkeypatch.setattr(
        "ivoirevoice.training.development_selection_finalizer._load_review",
        lambda _: pytest.fail("report must remain unread"),
    )

    with pytest.raises(ConfigError, match="confirmation"):
        run_development_selection_finalizer(settings)


@pytest.mark.parametrize("activity", ("refit-marker", "final-model", "holdout", "checkpoint"))
def test_refit_or_holdout_activity_blocks_before_report_consumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    activity: str,
) -> None:
    settings, _, _, artifact_root = _fixture(tmp_path, monkeypatch)
    if activity == "refit-marker":
        (artifact_root / REFIT_STARTED_FILENAME).write_text("{}", encoding="utf-8")
    elif activity == "final-model":
        (artifact_root / FINAL_MODEL_MANIFEST_FILENAME).write_text("{}", encoding="utf-8")
    elif activity == "holdout":
        (artifact_root / FINAL_EVALUATION_RECEIPT_FILENAME).write_text("{}", encoding="utf-8")
    else:
        settings.refit_checkpoint_directory.mkdir(parents=True)
        (settings.refit_checkpoint_directory / "checkpoint-000001").mkdir()
    monkeypatch.setattr(
        "ivoirevoice.training.development_selection_finalizer._load_review",
        lambda _: pytest.fail("report must remain unread"),
    )

    with pytest.raises(ConfigError, match="refit|final_holdout"):
        run_development_selection_finalizer(settings)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data.update(status="unexpected"), "rapport candidat"),
        (
            lambda data: data["evaluation_identity"].update(candidate_checkpoint_hash="0" * 64),
            "checkpoint candidat",
        ),
        (
            lambda data: data["evaluation_identity"].update(config_hash="0" * 64),
            "config hash",
        ),
        (
            lambda data: data["evaluation_identity"].update(validation_fingerprint="invalid"),
            "validation_fingerprint",
        ),
        (
            lambda data: data["evaluation_identity"].update(development_run_id="0" * 64),
            "run ID",
        ),
        (lambda data: data.update(metric_policy=["cer_micro"]), "politique"),
        (
            lambda data: data["candidate"]["metrics"].pop("wer_micro"),
            "métriques",
        ),
        (lambda data: data.update(proposed_refit_budget=1), "recalcul"),
    ],
)
def test_incoherent_candidate_report_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Any,
    message: str,
) -> None:
    settings, _, _, _ = _fixture(tmp_path, monkeypatch)
    _mutate_candidate_report(settings, mutation)

    with pytest.raises(ConfigError, match=message):
        _load_review(settings)


def test_modified_candidate_report_without_renewed_approval_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, _, _, _ = _fixture(tmp_path, monkeypatch)
    _mutate_candidate_report(
        settings,
        lambda data: data["candidate"]["metrics"].update(wer_micro=0.1),
        renew_approval=False,
    )

    with pytest.raises(ConfigError, match="rapport candidat diffère"):
        _load_review(settings)


def test_candidate_is_selected_by_metric_rank_not_persisted_proposal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, _, _, _ = _fixture(tmp_path, monkeypatch)
    _mutate_candidate_report(
        settings,
        lambda data: data.update(proposed_best="checkpoint-001507"),
    )

    with pytest.raises(ConfigError, match="recalcul"):
        _load_review(settings)


def test_changed_finalized_decision_is_never_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, _, _, artifact_root = _fixture(tmp_path, monkeypatch)
    run_development_selection_finalizer(settings)
    path = artifact_root / DEVELOPMENT_DECISION_FILENAME
    decision = json.loads(path.read_text(encoding="utf-8"))
    decision["refit_step_budget"] += 1
    path.write_text(json.dumps(decision), encoding="utf-8")

    with pytest.raises(ConfigError, match="déjà finalisée"):
        run_development_selection_finalizer(settings)


def test_public_finalization_report_contains_only_aggregates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, _, _, _ = _fixture(tmp_path, monkeypatch)

    report = run_development_selection_finalizer(settings)
    serialized = json.dumps(report)

    assert "/home/" not in serialized
    assert "PRIVATE TRANSCRIPTION" not in serialized
    assert "PRIVATE PREDICTION" not in serialized
    assert report["final_holdout_accessed"] is False
    assert report["privacy"]["contains_transcriptions"] is False


def test_finalizer_source_has_no_evaluation_audio_or_model_path() -> None:
    from ivoirevoice.training import development_selection_finalizer as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "_evaluate_validation",
        "generate(",
        "PilotCollator",
        "load_audited_dataset",
        "build_full_selection",
        "validate_final_holdout_files",
        "_load_model",
        "torch",
        "transformers",
        "soundfile",
        "dataset_root",
        "manifest_path",
        "target_text",
        '"predictions":',
        ".backward(",
        "optimizer.step(",
    ):
        assert forbidden not in source
    assert "2052" not in source


def test_refit_runner_still_loads_pilot_weights_not_development_best() -> None:
    from ivoirevoice.training import full_finetune

    source = Path(full_finetune.__file__).read_text(encoding="utf-8")
    refit_source = source[source.index("def run_refit") : source.index("def _evaluate_model")]
    runtime_source = source[
        source.index("def _new_training_runtime") : source.index("def _optimizer_group")
    ]

    assert "settings.initial_checkpoint_path" in refit_source
    assert "source=settings.initial_checkpoint_path" in runtime_source
    assert 'decision["best_checkpoint_name"]' not in refit_source
