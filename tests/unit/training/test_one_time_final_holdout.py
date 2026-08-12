from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from ivoirevoice.evaluation.baseline import normalize_evaluation_text
from ivoirevoice.evaluation.metrics import edit_counts
from ivoirevoice.exceptions import ConfigError
from ivoirevoice.training.audit import AuditedDataset, ManifestRow
from ivoirevoice.training.full_finetune import (
    _require_training_still_allowed,
    run_final_evaluation,
)
from ivoirevoice.training.full_settings import FullTrainingSettings
from ivoirevoice.training.one_time_final_holdout import (
    CONFIRMATION_ENVIRONMENT_VARIABLE,
    CONFIRMATION_VALUE,
    EVALUATED,
    FAILED_AFTER_ACCESS,
    IN_PROGRESS,
    METRICS_FILENAME,
    PREFLIGHT_FILENAME,
    SEALED,
    STATE_FILENAME,
    GuardedFinalModel,
    PreparedRuntime,
    StreamingASRAggregates,
    _output_directory,
    _select_frozen_final_holdout,
    run_final_holdout_preflight,
    run_one_time_final_holdout,
)
from ivoirevoice.training.whisper_finetune import directory_sha256, selection_sha256

TRAINING_COMMIT = "a" * 40
EVALUATION_COMMIT = "e" * 40
MANIFEST_HASH = "b" * 64
REFIT_SELECTION_HASH = "c" * 64
SOURCE_HASH = "d" * 64
RUN_ID = "f" * 64


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FullTrainingSettings, Path]:
    artifact_root = tmp_path / "artifacts/training/full_finetune_whisper_tiny_dy"
    shareable_root = artifact_root / "shareable"
    checkpoint_root = tmp_path / "checkpoints/full-finetune-whisper-tiny-dy/refit"
    final_checkpoint = checkpoint_root / "checkpoint-002052"
    final_checkpoint.mkdir(parents=True)
    (final_checkpoint / "model.safetensors").write_bytes(b"synthetic-final-weights")
    _write_json(
        final_checkpoint / "full_trainer_state.json",
        {
            "schema_version": 1,
            "stage": "refit",
            "run_id": RUN_ID,
            "code_commit": TRAINING_COMMIT,
            "config_sha256": "pending",
            "selection_sha256": REFIT_SELECTION_HASH,
            "initial_checkpoint_sha256": SOURCE_HASH,
            "global_step": 2_052,
            "successful_optimizer_steps": 2_052,
            "total_steps": 2_052,
            "completed": True,
            "optimizer_initialized_from": "fresh",
            "scheduler_initialized_from": "fresh",
            "pilot_optimizer_state_loaded": False,
            "precision": "fp16",
        },
    )
    _write_json(
        final_checkpoint / "FROZEN.json",
        {
            "schema_version": 1,
            "status": "immutable_final_checkpoint",
            "run_id": RUN_ID,
            "optimizer_steps": 2_052,
        },
    )
    config_path = tmp_path / "full.yaml"
    config_path.write_text("immutable-training-config\n", encoding="utf-8")
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    trainer_state_path = final_checkpoint / "full_trainer_state.json"
    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    trainer_state["config_sha256"] = config_hash
    _write_json(trainer_state_path, trainer_state)
    final_hash = directory_sha256(final_checkpoint)
    _write_json(
        artifact_root / "final_model_manifest.json",
        {
            "schema_version": 1,
            "status": "frozen",
            "code_commit": TRAINING_COMMIT,
            "config_sha256": config_hash,
            "manifest_sha256": MANIFEST_HASH,
            "refit_selection_sha256": REFIT_SELECTION_HASH,
            "initial_checkpoint_name": "checkpoint-000140",
            "initial_checkpoint_sha256": SOURCE_HASH,
            "train_audio_count": 16_425,
            "train_speaker_count": 18,
            "optimizer_steps": 2_052,
            "successful_optimizer_steps": 2_052,
            "final_checkpoint_name": "checkpoint-002052",
            "final_checkpoint_sha256": final_hash,
            "historical_test_used": False,
            "final_holdout_used": False,
            "publication_allowed": False,
        },
    )
    _write_json(
        artifact_root / "development_decision.json",
        {
            "schema_version": 1,
            "status": "frozen",
            "development_selection_finalized": True,
            "finalization_git_commit": TRAINING_COMMIT,
            "config_sha256": config_hash,
            "manifest_sha256": MANIFEST_HASH,
            "refit_selection_sha256": REFIT_SELECTION_HASH,
            "initial_checkpoint_sha256": SOURCE_HASH,
            "refit_step_budget": 2_052,
            "final_holdout_used": False,
        },
    )
    _write_json(
        shareable_root / "full_selection.json",
        {
            "schema_version": 1,
            "manifest_sha256": MANIFEST_HASH,
            "final_holdout": {"audio_count": 2_624, "decoded": False},
            "historical_test": {"audio_count": 150, "decoded": False},
        },
    )
    approval_path = tmp_path / "approval.json"
    _write_json(
        approval_path,
        {
            "schema_version": 1,
            "status": "approved_for_one_time_final_holdout",
            "training_git_commit": TRAINING_COMMIT,
            "config_sha256": config_hash,
            "manifest_sha256": MANIFEST_HASH,
            "refit_run_id": RUN_ID,
            "refit_selection_sha256": REFIT_SELECTION_HASH,
            "source_checkpoint_name": "checkpoint-000140",
            "source_checkpoint_sha256": SOURCE_HASH,
            "final_checkpoint_name": "checkpoint-002052",
            "final_checkpoint_sha256": final_hash,
            "successful_optimizer_steps": 2_052,
            "refit_audio_count": 16_425,
            "refit_speaker_count": 18,
            "final_holdout_count": 2_624,
            "historical_pilot_count": 150,
            "historical_expected_holdout_speaker_count": 3,
            "final_holdout_evaluation_count": 0,
            "final_holdout_accessed": False,
            "final_holdout_decoded": False,
            "remaining_model_decisions": False,
        },
    )
    settings = cast(
        FullTrainingSettings,
        SimpleNamespace(
            config_path=config_path,
            expected_manifest_sha256=MANIFEST_HASH,
            expected_dataset_version="0.1.0-local",
            forbidden_split="test",
            expected_initial_checkpoint_name="checkpoint-000140",
            refit_audio_count=16_425,
            refit_speaker_count=18,
            final_holdout_count=2_624,
            historical_pilot_test_count=150,
            artifact_output_directory=artifact_root,
            shareable_output_directory=shareable_root,
            refit_checkpoint_directory=checkpoint_root,
            minimum_free_disk_gib=1.0,
        ),
    )
    monkeypatch.setenv(CONFIRMATION_ENVIRONMENT_VARIABLE, CONFIRMATION_VALUE)
    return settings, approval_path


def _git_reader(_: Path) -> tuple[str, bool]:
    return EVALUATION_COMMIT, True


def _environment(_: FullTrainingSettings) -> dict[str, Any]:
    return {
        "torch_version": "synthetic-torch",
        "transformers_version": "synthetic-transformers",
        "cuda_version": "synthetic-cuda",
        "gpu_name": "synthetic-gpu",
        "free_disk_gib": 99.0,
    }


def _runtime() -> PreparedRuntime:
    return cast(
        PreparedRuntime,
        SimpleNamespace(
            model=SimpleNamespace(weight=7),
            processor=object(),
            collator=object(),
            torch=object(),
            transformers=object(),
            device=object(),
            torch_version="synthetic-torch",
            transformers_version="synthetic-transformers",
            cuda_version="synthetic-cuda",
            gpu_name="synthetic-gpu",
        ),
    )


def _aggregate_result() -> dict[str, Any]:
    return {
        "holdout_selection_sha256": "1" * 64,
        "holdout_audio_count": 2_624,
        "holdout_speaker_count": 3,
        "wer": 0.3,
        "cer": 0.15,
        "rtf": 0.5,
        "substitutions": 1,
        "insertions": 1,
        "deletions": 1,
        "character_substitutions": 1,
        "character_insertions": 1,
        "character_deletions": 1,
        "total_reference_words": 10,
        "total_reference_characters": 20,
        "exact_match_count": 5,
        "total_audio_duration_seconds": 100.0,
        "total_inference_seconds": 50.0,
        "mean_latency_seconds": 50.0 / 2_624,
        "final_loss": 0.4,
    }


def _preflight(
    settings: FullTrainingSettings,
    approval_path: Path,
) -> dict[str, Any]:
    return run_final_holdout_preflight(
        settings,
        approval_path=approval_path,
        git_reader=_git_reader,
        environment_probe=_environment,
    )


def _run(
    settings: FullTrainingSettings,
    approval_path: Path,
    *,
    prepare: Any = None,
    evaluate: Any = None,
    release: Any = None,
) -> dict[str, Any]:
    return run_one_time_final_holdout(
        settings,
        approval_path=approval_path,
        git_reader=_git_reader,
        runtime_preparer=prepare or (lambda _settings, _guard: _runtime()),
        aggregate_evaluator=evaluate or (lambda _settings, _prepared: _aggregate_result()),
        runtime_releaser=release or (lambda _runtime_value: None),
    )


def _read_state(settings: FullTrainingSettings) -> dict[str, Any]:
    return json.loads((_output_directory(settings) / STATE_FILENAME).read_text(encoding="utf-8"))


def test_confirmation_is_required_before_any_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, approval = _fixture(tmp_path, monkeypatch)
    monkeypatch.delenv(CONFIRMATION_ENVIRONMENT_VARIABLE)

    with pytest.raises(ConfigError, match="confirmation"):
        run_one_time_final_holdout(
            settings,
            approval_path=approval,
            git_reader=lambda _: pytest.fail("metadata guard must not run"),
        )


def test_preflight_is_metadata_only_and_seals_without_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, approval = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "ivoirevoice.training.one_time_final_holdout.load_audited_dataset",
        lambda _: pytest.fail("holdout dataset must remain unread"),
    )

    report = _preflight(settings, approval)
    state = _read_state(settings)

    assert report["status"] == "READY_FOR_ONE_TIME_FINAL_HOLDOUT"
    assert state["status"] == SEALED
    assert state["evaluation_count"] == 0
    assert state["final_holdout_accessed"] is False
    assert state["final_holdout_decoded"] is False


def test_preflight_refreshes_volatile_environment_without_consuming_holdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, approval = _fixture(tmp_path, monkeypatch)
    report = _preflight(settings, approval)

    refreshed = run_final_holdout_preflight(
        settings,
        approval_path=approval,
        git_reader=_git_reader,
        environment_probe=lambda _: {
            **_environment(settings),
            "free_disk_gib": 98.5,
        },
    )

    assert report["environment"]["free_disk_gib"] == 99.0
    assert refreshed["environment"]["free_disk_gib"] == 98.5
    assert _read_state(settings)["status"] == SEALED
    assert _read_state(settings)["evaluation_count"] == 0


def test_sealing_preflight_blocks_all_later_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, approval = _fixture(tmp_path, monkeypatch)
    _preflight(settings, approval)

    with pytest.raises(ConfigError, match="scellé"):
        _require_training_still_allowed(settings)


@pytest.mark.parametrize(
    ("filename", "field", "value", "message"),
    [
        ("final_model_manifest.json", "status", "not-frozen", "manifeste du refit"),
        (
            "final_model_manifest.json",
            "successful_optimizer_steps",
            2_051,
            "manifeste du refit",
        ),
        (
            "development_decision.json",
            "development_selection_finalized",
            False,
            "décision de développement",
        ),
    ],
)
def test_pre_access_frozen_guards_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    field: str,
    value: object,
    message: str,
) -> None:
    settings, approval = _fixture(tmp_path, monkeypatch)
    path = settings.artifact_output_directory / filename
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    _write_json(path, payload)

    with pytest.raises(ConfigError, match=message):
        _preflight(settings, approval)
    assert not _state_path_for_test(settings).exists()


def _state_path_for_test(settings: FullTrainingSettings) -> Path:
    return _output_directory(settings) / STATE_FILENAME


def test_changed_model_hash_stops_before_holdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, approval = _fixture(tmp_path, monkeypatch)
    final_model = settings.refit_checkpoint_directory / "checkpoint-002052/model.safetensors"
    final_model.write_bytes(b"changed")

    with pytest.raises(ConfigError, match="hash du modèle refit"):
        _preflight(settings, approval)


def test_old_development_checkpoint_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, approval = _fixture(tmp_path, monkeypatch)
    manifest_path = settings.artifact_output_directory / "final_model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["final_checkpoint_name"] = "checkpoint-001720"
    _write_json(manifest_path, manifest)

    with pytest.raises(ConfigError, match="manifeste du refit"):
        _preflight(settings, approval)


def test_nonzero_count_in_sealed_state_stops_before_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, approval = _fixture(tmp_path, monkeypatch)
    _preflight(settings, approval)
    state = _read_state(settings)
    state["evaluation_count"] = 1
    _write_json(_state_path_for_test(settings), state)

    with pytest.raises(ConfigError, match="SEALED"):
        _run(
            settings,
            approval,
            prepare=lambda *_: pytest.fail("runtime must remain unloaded"),
        )


def test_success_transitions_once_and_second_run_never_evaluates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, approval = _fixture(tmp_path, monkeypatch)
    _preflight(settings, approval)
    result = _run(settings, approval)

    assert result["status"] == "evaluated"
    assert _read_state(settings)["status"] == EVALUATED
    assert _read_state(settings)["evaluation_count"] == 1
    second = _run(
        settings,
        approval,
        evaluate=lambda *_: pytest.fail("a second evaluation is impossible"),
    )
    assert second == {
        "status": "FINAL_HOLDOUT_ALREADY_EVALUATED",
        "evaluation_count": 1,
    }


def test_state_transition_is_persisted_before_aggregate_evaluator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, approval = _fixture(tmp_path, monkeypatch)
    _preflight(settings, approval)

    def evaluate(*_: object) -> dict[str, Any]:
        state = _read_state(settings)
        assert state["status"] == IN_PROGRESS
        assert state["evaluation_count"] == 1
        assert state["final_holdout_accessed"] is True
        return _aggregate_result()

    _run(settings, approval, evaluate=evaluate)
    assert _read_state(settings)["status"] == EVALUATED


def test_error_before_access_keeps_holdout_sealed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, approval = _fixture(tmp_path, monkeypatch)
    _preflight(settings, approval)

    with pytest.raises(RuntimeError, match="model preparation"):
        _run(
            settings,
            approval,
            prepare=lambda *_: (_ for _ in ()).throw(RuntimeError("model preparation")),
        )
    assert _read_state(settings)["status"] == SEALED
    assert _read_state(settings)["evaluation_count"] == 0


def test_error_after_access_is_terminal_and_cannot_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, approval = _fixture(tmp_path, monkeypatch)
    _preflight(settings, approval)

    with pytest.raises(RuntimeError, match="synthetic access failure"):
        _run(
            settings,
            approval,
            evaluate=lambda *_: (_ for _ in ()).throw(RuntimeError("synthetic access failure")),
        )
    state = _read_state(settings)
    assert state["status"] == FAILED_AFTER_ACCESS
    assert state["evaluation_count"] == 1
    assert state["failure_type"] == "RuntimeError"
    assert "synthetic access failure" not in json.dumps(state)

    with pytest.raises(ConfigError, match="déjà consommé"):
        _run(
            settings,
            approval,
            evaluate=lambda *_: pytest.fail("automatic recovery is forbidden"),
        )


def test_report_is_aggregate_only_and_creates_no_prediction_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, approval = _fixture(tmp_path, monkeypatch)
    _preflight(settings, approval)
    result = _run(settings, approval)
    output = _output_directory(settings)
    serialized = json.dumps(result)

    assert {path.name for path in output.iterdir()} == {
        STATE_FILENAME,
        PREFLIGHT_FILENAME,
        METRICS_FILENAME,
    }
    for forbidden in (
        "PRIVATE REFERENCE",
        "PRIVATE HYPOTHESIS",
        "speaker_secret",
        "utterance_secret",
        "/home/",
    ):
        assert forbidden not in serialized
    assert result["privacy"] == {
        "contains_audio": False,
        "contains_paths": False,
        "contains_transcriptions": False,
        "contains_predictions": False,
        "contains_sample_identifiers": False,
        "contains_speaker_identifiers": False,
    }
    assert result["holdout_speaker_count_historical_expectation"] == 3
    assert result["holdout_speaker_count_anomaly"] is False
    assert result["final_model_hash"] == result["final_checkpoint_sha256"]
    assert result["config_hash"] == result["config_sha256"]
    assert result["git_commit"] == EVALUATION_COMMIT


def test_unexpected_speaker_count_is_reported_without_rejecting_real_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, approval = _fixture(tmp_path, monkeypatch)
    _preflight(settings, approval)
    aggregate = _aggregate_result()
    aggregate["holdout_speaker_count"] = 4

    result = _run(settings, approval, evaluate=lambda *_: aggregate)

    assert result["holdout_speaker_count"] == 4
    assert result["holdout_speaker_count_historical_expectation"] == 3
    assert result["holdout_speaker_count_anomaly"] is True


def test_only_final_checkpoint_reaches_runtime_preparer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, approval = _fixture(tmp_path, monkeypatch)
    _preflight(settings, approval)

    def prepare(
        _settings: FullTrainingSettings,
        guard: GuardedFinalModel,
    ) -> PreparedRuntime:
        assert guard.final_checkpoint.name == "checkpoint-002052"
        assert guard.final_checkpoint_sha256 == directory_sha256(guard.final_checkpoint)
        return _runtime()

    _run(settings, approval, prepare=prepare)


def test_dummy_model_weights_remain_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, approval = _fixture(tmp_path, monkeypatch)
    _preflight(settings, approval)
    runtime = _runtime()
    before = runtime.model.weight

    _run(
        settings,
        approval,
        prepare=lambda *_: runtime,
        evaluate=lambda _settings, prepared: (
            _aggregate_result()
            if prepared.model.weight == before
            else pytest.fail("model weight changed")
        ),
    )
    assert runtime.model.weight == before


def test_streaming_aggregates_compute_metrics_and_drop_speaker_tokens() -> None:
    aggregate = StreamingASRAggregates()
    samples = (
        ("a b c", "a x c", "speaker_a", 2.0, 0.2, 0.4),
        ("d e", "d e plus", "speaker_b", 3.0, 0.3, 0.6),
    )
    for reference, hypothesis, speaker, duration, latency, loss in samples:
        aggregate.add(
            reference=reference,
            hypothesis=hypothesis,
            speaker_token=speaker,
            audio_duration_seconds=duration,
            inference_seconds=latency,
            loss=loss,
        )
    result = aggregate.result(selection_sha256="2" * 64)
    expected_character_errors = 0
    expected_characters = 0
    for reference, hypothesis, *_ in samples:
        normalized_reference = normalize_evaluation_text(
            reference, lowercase=True, remove_punctuation=True
        )
        normalized_hypothesis = normalize_evaluation_text(
            hypothesis, lowercase=True, remove_punctuation=True
        )
        counts = edit_counts(tuple(normalized_reference), tuple(normalized_hypothesis))
        expected_character_errors += counts.errors
        expected_characters += len(normalized_reference)

    assert result["holdout_audio_count"] == 2
    assert result["holdout_speaker_count"] == 2
    assert result["wer"] == pytest.approx(2 / 5)
    assert result["cer"] == pytest.approx(expected_character_errors / expected_characters)
    assert result["rtf"] == pytest.approx(0.1)
    assert result["substitutions"] == 1
    assert result["insertions"] == 1
    assert result["deletions"] == 0
    assert result["final_loss"] == pytest.approx(0.5)
    assert aggregate.speaker_tokens == set()


def test_frozen_holdout_selection_uses_only_prior_private_partition_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, _ = _fixture(tmp_path, monkeypatch)
    settings.final_holdout_count = 1
    settings.historical_pilot_test_count = 1
    historical = ManifestRow(
        utterance_id="historical-pilot",
        speaker_id="speaker-test",
        gender_folder="unknown",
        language="dyu",
        text_raw="historical private text",
        text_no_tones="historical private text",
        target_text="historical private text",
        audio_path="historical.wav",
        duration_seconds=1.0,
        sample_rate_hz=16_000,
        channels=1,
        audio_sha256="3" * 64,
        split="test",
        usage_scope="local_research_only",
    )
    final = ManifestRow(
        utterance_id="final-only",
        speaker_id="speaker-test",
        gender_folder="unknown",
        language="dyu",
        text_raw="final private text",
        text_no_tones="final private text",
        target_text="final private text",
        audio_path="final.wav",
        duration_seconds=1.0,
        sample_rate_hz=16_000,
        channels=1,
        audio_sha256="4" * 64,
        split="test",
        usage_scope="local_research_only",
    )
    expected_hash = selection_sha256({"final_holdout": ((final.utterance_id, final.audio_sha256),)})
    _write_json(
        settings.artifact_output_directory / "full_selection_private.json",
        {
            "schema_version": 1,
            "manifest_sha256": MANIFEST_HASH,
            "historical_test_audio_ids": [historical.utterance_id],
            "final_holdout_audio_ids": [final.utterance_id],
            "final_holdout_selection_sha256": expected_hash,
        },
    )
    dataset = AuditedDataset(
        rows=(historical, final),
        manifest_sha256=MANIFEST_HASH,
        dataset_version="0.1.0-local",
    )

    rows, observed_hash = _select_frozen_final_holdout(dataset, settings)

    assert rows == (final,)
    assert observed_hash == expected_hash


def test_legacy_three_model_stage_fails_before_context_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ivoirevoice.training.full_finetune._load_context",
        lambda *_args, **_kwargs: pytest.fail("legacy must stop before data access"),
    )

    with pytest.raises(ConfigError, match="historique à trois modèles"):
        run_final_evaluation(cast(FullTrainingSettings, SimpleNamespace()))


def test_legacy_artifact_blocks_new_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, approval = _fixture(tmp_path, monkeypatch)
    _write_json(
        settings.artifact_output_directory / "final_holdout_evaluation_receipt.json",
        {"status": "started"},
    )

    with pytest.raises(ConfigError, match="legacy"):
        _preflight(settings, approval)


def test_source_has_no_optimizer_backward_or_forbidden_model_sources() -> None:
    from ivoirevoice.training import one_time_final_holdout as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "whisper_tiny_baseline",
        "whisper_small",
        "checkpoint-001507",
        "checkpoint-001720",
        "optimizer.step(",
        "scheduler.step(",
        ".backward(",
        "_write_private_evaluation",
        "predictions.csv",
        "predictions.jsonl",
        "decoded_outputs.json",
    ):
        assert forbidden not in source


def test_makefile_exposes_new_targets_and_keeps_legacy_guard() -> None:
    root = Path(__file__).resolve().parents[3]
    makefile = (root / "Makefile").read_text(encoding="utf-8")

    assert "final-holdout-preflight:" in makefile
    assert "evaluate-final-holdout-refit-once:" in makefile
    assert "--stage final-holdout-evaluate-refit-once" in makefile
    assert "evaluate-final-holdout-dy:" in makefile
    assert CONFIRMATION_VALUE not in makefile
