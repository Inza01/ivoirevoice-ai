from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from ivoirevoice.exceptions import ConfigError
from ivoirevoice.training.audit import REQUIRED_COLUMNS, ManifestRow
from ivoirevoice.training.development_final_validation import (
    CANDIDATE_PRIVATE_FILENAME,
    CANDIDATE_PUBLIC_FILENAME,
    METRIC_POLICY,
    DevelopmentArtifacts,
    TerminalValidationContext,
    _aggregate_metrics,
    _existing_report,
    _load_development_artifacts,
    _require_current_preflight,
    _require_stage_available,
    build_candidate_report,
    load_terminal_validation_rows,
    run_development_final_validation,
)
from ivoirevoice.training.full_finetune import (
    DEVELOPMENT_DECISION_FILENAME,
    FINAL_EVALUATION_RECEIPT_FILENAME,
    FINAL_MODEL_MANIFEST_FILENAME,
    PREFLIGHT_REPORT_FILENAME,
    REFIT_STARTED_FILENAME,
    TRAINER_STATE_FILENAME,
)
from ivoirevoice.training.full_settings import FullTrainingSettings
from ivoirevoice.training.pilot_finetune import EvaluationResult
from ivoirevoice.training.whisper_finetune import (
    directory_sha256,
    identity_sha256,
)


def _raw_row(
    *,
    split: str,
    identifier: str,
    speaker: str,
    audio_path: str,
    audio_hash: str,
    target: str,
) -> dict[str, str]:
    return {
        "utterance_id": identifier,
        "speaker_id": speaker,
        "gender_folder": "anonymous",
        "language": "dyu",
        "text_raw": target,
        "text_without_tones_nfc": target,
        "target_text_mvp": target,
        "audio_path": audio_path,
        "duration_seconds": "1.0",
        "sample_rate_hz": "16000",
        "channels": "1",
        "audio_sha256": audio_hash,
        "split": split,
        "usage_scope": "local_research_only",
    }


def _validation_settings(tmp_path: Path) -> FullTrainingSettings:
    data_root = tmp_path / "data"
    validation_audio = data_root / "validation/anonymous/valid.wav"
    validation_audio.parent.mkdir(parents=True)
    validation_audio.write_bytes(b"validation-audio")
    validation_hash = hashlib.sha256(b"validation-audio").hexdigest()
    missing_hash = hashlib.sha256(b"never-read").hexdigest()
    rows = [
        _raw_row(
            split="train",
            identifier="train-private",
            speaker="train-private",
            audio_path="train/private/missing.wav",
            audio_hash=missing_hash,
            target="TRAIN TRANSCRIPTION MUST NOT MATERIALIZE",
        ),
        _raw_row(
            split="validation",
            identifier="validation-1",
            speaker="validation-speaker",
            audio_path="validation/anonymous/valid.wav",
            audio_hash=validation_hash,
            target="validation target",
        ),
        _raw_row(
            split="test",
            identifier="holdout-private",
            speaker="holdout-private",
            audio_path="test/private/missing.wav",
            audio_hash=missing_hash,
            target="HOLDOUT TRANSCRIPTION MUST NOT MATERIALIZE",
        ),
    ]
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
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
    return cast(
        FullTrainingSettings,
        SimpleNamespace(
            manifest_path=manifest,
            dataset_metadata_path=metadata,
            expected_manifest_sha256=manifest_hash,
            expected_dataset_version="0.1.0-local",
            validation_split="validation",
            validation_audio_count=1,
            validation_speaker_count=1,
            max_audio_seconds=30.0,
            dataset_root=data_root,
        ),
    )


def test_validation_loader_materializes_only_validation_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _validation_settings(tmp_path)
    parsed_splits: list[str] = []
    from ivoirevoice.training import development_final_validation as module

    original = module._parse_manifest_row

    def recording_parser(raw: dict[str, str], line_number: int) -> ManifestRow:
        parsed_splits.append(raw["split"])
        return original(raw, line_number)

    monkeypatch.setattr(module, "_parse_manifest_row", recording_parser)

    rows, manifest_hash, fingerprint = load_terminal_validation_rows(settings)

    assert parsed_splits == ["validation"]
    assert [row.split for row in rows] == ["validation"]
    assert rows[0].target_text == "validation target"
    assert manifest_hash == settings.expected_manifest_sha256
    assert len(fingerprint) == 64


def _guard_settings(tmp_path: Path) -> FullTrainingSettings:
    return cast(
        FullTrainingSettings,
        SimpleNamespace(
            artifact_output_directory=tmp_path / "artifacts",
            refit_checkpoint_directory=tmp_path / "checkpoints/refit",
        ),
    )


def test_final_holdout_receipt_blocks_before_context_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _guard_settings(tmp_path)
    settings.artifact_output_directory.mkdir(parents=True)
    (settings.artifact_output_directory / FINAL_EVALUATION_RECEIPT_FILENAME).write_text(
        "{}\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        "ivoirevoice.training.development_final_validation._load_context",
        lambda _: pytest.fail("context must remain unreachable"),
    )

    with pytest.raises(ConfigError, match="final_holdout"):
        run_development_final_validation(settings)


@pytest.mark.parametrize(
    "activity",
    ("marker", "manifest", "checkpoint"),
)
def test_any_refit_activity_blocks_terminal_validation(
    tmp_path: Path,
    activity: str,
) -> None:
    settings = _guard_settings(tmp_path)
    settings.artifact_output_directory.mkdir(parents=True)
    if activity == "marker":
        (settings.artifact_output_directory / REFIT_STARTED_FILENAME).write_text(
            "{}\n", encoding="utf-8"
        )
    elif activity == "manifest":
        (settings.artifact_output_directory / FINAL_MODEL_MANIFEST_FILENAME).write_text(
            "{}\n", encoding="utf-8"
        )
    else:
        settings.refit_checkpoint_directory.mkdir(parents=True)
        (settings.refit_checkpoint_directory / "checkpoint-000001").mkdir()

    with pytest.raises(ConfigError, match="refit"):
        _require_stage_available(settings)


def _metric_values(wer: float, cer: float, loss: float) -> dict[str, Any]:
    return {
        "validation_loss": loss,
        "wer_micro": wer,
        "cer_micro": cer,
        "rtf": 0.01,
        "word_substitutions": 10,
        "word_insertions": 2,
        "word_deletions": 3,
        "evaluated_audio_count": 2_661,
    }


@pytest.mark.parametrize(
    ("candidate", "expected_name", "expected_budget"),
    [
        (_metric_values(0.44, 0.30, 0.60), "checkpoint-001720", 2_052),
        (_metric_values(0.46, 0.10, 0.10), "checkpoint-001507", 1_798),
        (_metric_values(0.45, 0.19, 0.90), "checkpoint-001720", 2_052),
        (_metric_values(0.45, 0.20, 0.49), "checkpoint-001720", 2_052),
        (_metric_values(0.45, 0.20, 0.50), "checkpoint-001507", 1_798),
    ],
)
def test_candidate_selection_uses_frozen_rank_and_budget_formula(
    candidate: dict[str, Any],
    expected_name: str,
    expected_budget: int,
) -> None:
    report = build_candidate_report(
        identity={"immutable": "identity"},
        current_name="checkpoint-001507",
        current_step=1_507,
        current_metrics=_metric_values(0.45, 0.20, 0.50),
        candidate_name="checkpoint-001720",
        candidate_step=1_720,
        candidate_metrics=candidate,
        development_steps_per_epoch=861,
        refit_steps_per_epoch=1_027,
        evaluation_timestamp="2026-08-11T00:00:00+00:00",
    )

    assert tuple(report["metric_policy"]) == METRIC_POLICY
    assert report["proposed_best"] == expected_name
    assert report["proposed_refit_budget"] == expected_budget
    assert report["frozen_decision_modified"] is False


def test_aggregate_metrics_excludes_speaker_details_and_private_text() -> None:
    metrics = {
        **_metric_values(0.4, 0.2, 0.5),
        "speaker_metrics": {"private-speaker": {"wer": 0.4}},
        "private_text": "MUST NOT LEAK",
    }

    public = _aggregate_metrics(metrics, audio_count=2_661, speaker_count=3)

    assert set(public) == {
        "validation_loss",
        "wer_micro",
        "cer_micro",
        "rtf",
        "word_substitutions",
        "word_insertions",
        "word_deletions",
        "audio_count",
        "speaker_count",
    }
    assert "private-speaker" not in json.dumps(public)
    assert "MUST NOT LEAK" not in json.dumps(public)


def _checkpoint_state(
    *,
    step: int,
    completed: bool,
    run_id: str,
    selection_hash: str,
    config_hash: str,
    pilot_hash: str,
    commit: str,
) -> dict[str, Any]:
    return {
        "stage": "development",
        "selection_sha256": selection_hash,
        "initial_checkpoint_sha256": pilot_hash,
        "config_sha256": config_hash,
        "code_commit": commit,
        "run_id": run_id,
        "global_step": step,
        "completed": completed,
    }


def _development_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FullTrainingSettings, TerminalValidationContext, Path, Path]:
    artifacts_root = tmp_path / "artifacts"
    development_root = tmp_path / "checkpoints/development"
    current = development_root / "checkpoint-000100"
    candidate = development_root / "checkpoint-000120"
    current.mkdir(parents=True)
    candidate.mkdir(parents=True)
    selection_hash = "d" * 64
    config_hash = "c" * 64
    pilot_hash = "p" * 64
    commit = "a" * 40
    run_id = identity_sha256(
        {
            "stage": "development",
            "selection_sha256": selection_hash,
            "initial_checkpoint_sha256": pilot_hash,
            "config_sha256": config_hash,
            "code_commit": commit,
        }
    )
    for path, step, completed in (
        (current, 100, False),
        (candidate, 120, True),
    ):
        state = _checkpoint_state(
            step=step,
            completed=completed,
            run_id=run_id,
            selection_hash=selection_hash,
            config_hash=config_hash,
            pilot_hash=pilot_hash,
            commit=commit,
        )
        (path / TRAINER_STATE_FILENAME).write_text(json.dumps(state), encoding="utf-8")
        for filename in ("optimizer.pt", "scheduler.pt", "scaler.pt", "config.json"):
            (path / filename).write_text(filename, encoding="utf-8")
    settings = cast(
        FullTrainingSettings,
        SimpleNamespace(
            artifact_output_directory=artifacts_root,
            development_checkpoint_directory=development_root,
        ),
    )
    context = TerminalValidationContext(
        validation_rows=(),
        validation_fingerprint="v" * 64,
        manifest_sha256="m" * 64,
        config_sha256=config_hash,
        pilot_checkpoint_sha256=pilot_hash,
        evaluation_git_commit="e" * 40,
        model_settings=SimpleNamespace(),
    )
    artifacts_root.mkdir(parents=True)
    decision = {
        "status": "frozen",
        "code_commit": commit,
        "config_sha256": config_hash,
        "manifest_sha256": context.manifest_sha256,
        "initial_checkpoint_sha256": pilot_hash,
        "development_selection_sha256": selection_hash,
        "historical_test_used": False,
        "final_holdout_used": False,
        "best_checkpoint_name": current.name,
        "best_development_step": 100,
        "best_checkpoint_sha256": directory_sha256(current),
    }
    decision_path = artifacts_root / DEVELOPMENT_DECISION_FILENAME
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    monkeypatch.setattr(
        "ivoirevoice.training.development_final_validation._require_current_preflight",
        lambda *args: None,
    )
    return settings, context, current, candidate


def test_development_must_be_frozen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, context, _, _ = _development_fixture(tmp_path, monkeypatch)
    path = settings.artifact_output_directory / DEVELOPMENT_DECISION_FILENAME
    decision = json.loads(path.read_text(encoding="utf-8"))
    decision["status"] = "draft"
    path.write_text(json.dumps(decision), encoding="utf-8")

    with pytest.raises(ConfigError, match="frozen"):
        _load_development_artifacts(settings, context)


@pytest.mark.parametrize("field", ("config_sha256", "initial_checkpoint_sha256"))
def test_frozen_decision_must_match_config_and_pilot_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    settings, context, _, _ = _development_fixture(tmp_path, monkeypatch)
    path = settings.artifact_output_directory / DEVELOPMENT_DECISION_FILENAME
    decision = json.loads(path.read_text(encoding="utf-8"))
    decision[field] = "x" * 64
    path.write_text(json.dumps(decision), encoding="utf-8")

    with pytest.raises(ConfigError, match=field):
        _load_development_artifacts(settings, context)


def test_candidate_checkpoint_must_belong_to_same_development_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, context, _, candidate = _development_fixture(tmp_path, monkeypatch)
    state_path = candidate / TRAINER_STATE_FILENAME
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["run_id"] = "wrong-run"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ConfigError, match="run_id"):
        _load_development_artifacts(settings, context)


def test_current_preflight_must_match_evaluation_commit_and_frozen_selection(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    settings = cast(
        FullTrainingSettings,
        SimpleNamespace(artifact_output_directory=root),
    )
    context = TerminalValidationContext(
        validation_rows=cast(tuple[ManifestRow, ...], (object(),)),
        validation_fingerprint="v" * 64,
        manifest_sha256="m" * 64,
        config_sha256="c" * 64,
        pilot_checkpoint_sha256="p" * 64,
        evaluation_git_commit="e" * 40,
        model_settings=SimpleNamespace(),
    )
    decision = {"development_selection_sha256": "s" * 64}
    preflight = {
        "status": "passed",
        "code_commit": context.evaluation_git_commit,
        "config_sha256": context.config_sha256,
        "manifest_sha256": context.manifest_sha256,
        "initial_checkpoint_sha256": context.pilot_checkpoint_sha256,
        "validation_audio_count": 1,
        "development_selection_sha256": decision["development_selection_sha256"],
        "historical_test_decoded": False,
        "final_holdout_decoded": False,
    }
    path = root / PREFLIGHT_REPORT_FILENAME
    path.write_text(json.dumps(preflight), encoding="utf-8")

    _require_current_preflight(settings, context, decision)
    preflight["code_commit"] = "old"
    path.write_text(json.dumps(preflight), encoding="utf-8")
    with pytest.raises(ConfigError, match="code_commit"):
        _require_current_preflight(settings, context, decision)


def test_existing_identical_report_is_reused_without_gpu(
    tmp_path: Path,
) -> None:
    settings = cast(
        FullTrainingSettings,
        SimpleNamespace(
            artifact_output_directory=tmp_path / "private",
            shareable_output_directory=tmp_path / "public",
        ),
    )
    identity = {"candidate_checkpoint_hash": "a" * 64, "config_hash": "b" * 64}
    private = {
        "evaluation_identity": identity,
        "predictions": [{"private": "kept outside public"}],
    }
    public = {
        "status": "candidate_evaluated_not_finalized",
        "evaluation_identity": identity,
    }
    settings.artifact_output_directory.mkdir()
    settings.shareable_output_directory.mkdir()
    (settings.artifact_output_directory / CANDIDATE_PRIVATE_FILENAME).write_text(
        json.dumps(private), encoding="utf-8"
    )
    (settings.shareable_output_directory / CANDIDATE_PUBLIC_FILENAME).write_text(
        json.dumps(public), encoding="utf-8"
    )

    assert _existing_report(settings, identity) == public


def test_existing_mismatched_or_partial_report_fails_closed(tmp_path: Path) -> None:
    settings = cast(
        FullTrainingSettings,
        SimpleNamespace(
            artifact_output_directory=tmp_path / "private",
            shareable_output_directory=tmp_path / "public",
        ),
    )
    settings.artifact_output_directory.mkdir()
    settings.shareable_output_directory.mkdir()
    private_path = settings.artifact_output_directory / CANDIDATE_PRIVATE_FILENAME
    private_path.write_text('{"evaluation_identity":{"hash":"old"}}', encoding="utf-8")

    with pytest.raises(ConfigError, match="incomplète"):
        _existing_report(settings, {"hash": "new"})
    public_path = settings.shareable_output_directory / CANDIDATE_PUBLIC_FILENAME
    public_path.write_text(
        json.dumps(
            {
                "status": "candidate_evaluated_not_finalized",
                "evaluation_identity": {"hash": "old"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="hash"):
        _existing_report(settings, {"hash": "new"})


def test_stage_source_has_no_training_or_holdout_execution_path() -> None:
    from ivoirevoice.training import development_final_validation as module

    source = Path(module.__file__).read_text(encoding="utf-8")

    for forbidden in (
        ".backward(",
        "torch.optim",
        "GradScaler",
        "_save_checkpoint(",
        "run_final_evaluation(",
        "validate_final_holdout_files(",
        "final_holdout_ids",
    ):
        assert forbidden not in source


def test_public_candidate_report_contains_no_private_values() -> None:
    report = build_candidate_report(
        identity={"candidate_checkpoint_hash": "a" * 64},
        current_name="checkpoint-001507",
        current_step=1_507,
        current_metrics=_metric_values(0.4, 0.2, 0.5),
        candidate_name="checkpoint-001720",
        candidate_step=1_720,
        candidate_metrics=_metric_values(0.3, 0.1, 0.4),
        development_steps_per_epoch=861,
        refit_steps_per_epoch=1_027,
        evaluation_timestamp="2026-08-11T00:00:00+00:00",
    )
    serialized = json.dumps(report)

    assert "PRIVATE TRANSCRIPTION" not in serialized
    assert "/home/" not in serialized
    assert report["privacy"] == {
        "contains_transcriptions": False,
        "contains_predictions": False,
        "contains_sample_identifiers": False,
        "contains_speaker_identifiers": False,
        "contains_local_paths": False,
    }


def test_idempotent_run_returns_before_cuda_or_model_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _guard_settings(tmp_path)
    context = cast(TerminalValidationContext, SimpleNamespace())
    artifacts = cast(DevelopmentArtifacts, SimpleNamespace())
    monkeypatch.setattr(
        "ivoirevoice.training.development_final_validation._load_context",
        lambda _: context,
    )
    monkeypatch.setattr(
        "ivoirevoice.training.development_final_validation._load_development_artifacts",
        lambda *_: artifacts,
    )
    monkeypatch.setattr(
        "ivoirevoice.training.development_final_validation._decoding_config_hash",
        lambda *_: "d" * 64,
    )
    monkeypatch.setattr(
        "ivoirevoice.training.development_final_validation._evaluation_identity",
        lambda *args, **kwargs: {"identity": "same"},
    )
    monkeypatch.setattr(
        "ivoirevoice.training.development_final_validation._existing_report",
        lambda *_: {"status": "candidate_evaluated_not_finalized"},
    )
    monkeypatch.setattr(
        "ivoirevoice.training.development_final_validation._require_cuda",
        lambda _: pytest.fail("CUDA must not be reached for an identical report"),
    )

    result = run_development_final_validation(settings)

    assert result["status"] == "candidate_evaluated_not_finalized"


def test_simulated_stage_writes_candidate_without_changing_decision_or_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ivoirevoice.training import development_final_validation as module

    artifact_root = tmp_path / "artifacts"
    shareable_root = artifact_root / "shareable"
    checkpoint_root = tmp_path / "checkpoints/development"
    current_checkpoint = checkpoint_root / "checkpoint-001507"
    candidate_checkpoint = checkpoint_root / "checkpoint-001720"
    current_checkpoint.mkdir(parents=True)
    candidate_checkpoint.mkdir(parents=True)
    (current_checkpoint / "sentinel").write_text("unchanged", encoding="utf-8")
    (candidate_checkpoint / "sentinel").write_text("unchanged", encoding="utf-8")
    artifact_root.mkdir(parents=True)
    decision_path = artifact_root / DEVELOPMENT_DECISION_FILENAME
    decision = {
        "best_checkpoint_name": current_checkpoint.name,
        "best_development_step": 1_507,
        "development_steps_per_epoch": 861,
        "refit_steps_per_epoch": 1_027,
        "best_validation_metrics": {
            **_metric_values(0.45, 0.20, 0.50),
            "evaluated_audio_count": 1,
        },
    }
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    decision_before = decision_path.read_bytes()
    decision_hash = hashlib.sha256(decision_before).hexdigest()
    row = ManifestRow(
        utterance_id="validation-anonymous",
        speaker_id="speaker-anonymous",
        gender_folder="anonymous",
        language="dyu",
        text_raw="private reference",
        text_no_tones="private reference",
        target_text="private reference",
        audio_path="validation/anonymous.wav",
        duration_seconds=1.0,
        sample_rate_hz=16_000,
        channels=1,
        audio_sha256="a" * 64,
        split="validation",
        usage_scope="local_research_only",
    )
    context = TerminalValidationContext(
        validation_rows=(row,),
        validation_fingerprint="v" * 64,
        manifest_sha256="m" * 64,
        config_sha256="c" * 64,
        pilot_checkpoint_sha256="p" * 64,
        evaluation_git_commit="e" * 40,
        model_settings=SimpleNamespace(cache_dir=tmp_path / "cache"),
    )
    artifacts = DevelopmentArtifacts(
        decision=decision,
        decision_sha256=decision_hash,
        current_checkpoint=current_checkpoint,
        current_state={"run_id": "run", "code_commit": "d" * 40},
        current_checkpoint_sha256="1" * 64,
        candidate_checkpoint=candidate_checkpoint,
        candidate_state={
            "run_id": "run",
            "code_commit": "d" * 40,
            "global_step": 1_720,
        },
        candidate_checkpoint_sha256="2" * 64,
    )
    settings = cast(
        FullTrainingSettings,
        SimpleNamespace(
            artifact_output_directory=artifact_root,
            shareable_output_directory=shareable_root,
            refit_checkpoint_directory=tmp_path / "checkpoints/refit",
            eval_batch_size=8,
            fp16=True,
            seed=42,
        ),
    )

    class Model:
        config = SimpleNamespace(use_cache=True)

    class Cuda:
        @staticmethod
        def empty_cache() -> None:
            return None

    fake_torch = SimpleNamespace(__version__="torch-test", cuda=Cuda())
    fake_transformers = SimpleNamespace(__version__="transformers-test")
    original_import = module.importlib.import_module

    def fake_import(name: str) -> Any:
        if name == "torch":
            return fake_torch
        if name == "transformers":
            return fake_transformers
        return original_import(name)

    seen_rows: list[ManifestRow] = []

    def fake_evaluate(**kwargs: Any) -> EvaluationResult:
        seen_rows.extend(kwargs["rows"])
        return EvaluationResult(
            metrics={**_metric_values(0.40, 0.18, 0.45), "evaluated_audio_count": 1},
            predictions=({"prediction": "PRIVATE PREDICTION"},),
        )

    monkeypatch.setattr(module.importlib, "import_module", fake_import)
    monkeypatch.setattr(module, "_load_context", lambda _: context)
    monkeypatch.setattr(module, "_load_development_artifacts", lambda *_: artifacts)
    monkeypatch.setattr(module, "_decoding_config_hash", lambda *_: "z" * 64)
    monkeypatch.setattr(module, "_require_cuda", lambda _: SimpleNamespace(type="cuda"))
    monkeypatch.setattr(module, "_seed_everything", lambda *_: None)
    monkeypatch.setattr(module, "_processor_and_collator", lambda *args: (object(), object()))
    monkeypatch.setattr(module, "_load_model", lambda **kwargs: Model())
    monkeypatch.setattr(module, "_evaluate_validation", fake_evaluate)

    result = run_development_final_validation(settings)

    assert seen_rows == [row]
    assert result["proposed_best"] == "checkpoint-001720"
    assert result["proposed_refit_budget"] == 2_052
    assert result["frozen_decision_modified"] is False
    assert decision_path.read_bytes() == decision_before
    assert (current_checkpoint / "sentinel").read_text(encoding="utf-8") == "unchanged"
    assert (candidate_checkpoint / "sentinel").read_text(encoding="utf-8") == "unchanged"
    private_text = (artifact_root / CANDIDATE_PRIVATE_FILENAME).read_text(encoding="utf-8")
    public_text = (shareable_root / CANDIDATE_PUBLIC_FILENAME).read_text(encoding="utf-8")
    assert "PRIVATE PREDICTION" in private_text
    assert "PRIVATE PREDICTION" not in public_text
    assert '"final_holdout_accessed": false' in public_text
