from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import yaml

from ivoirevoice.exceptions import ConfigError
from ivoirevoice.training.audit import AuditedDataset, ManifestRow
from ivoirevoice.training.full_finetune import (
    FINAL_EVALUATION_RECEIPT_FILENAME,
    FullContext,
    _initial_training_state,
    _require_cuda,
    _require_training_still_allowed,
)
from ivoirevoice.training.full_selection import (
    build_full_selection,
    validate_final_holdout_files,
    write_full_selection_reports,
)
from ivoirevoice.training.full_settings import (
    FullTrainingSettings,
    load_full_training_settings,
)
from ivoirevoice.training.whisper_finetune import (
    accumulation_group_sizes,
    compute_step_geometry,
    identity_sha256,
    metric_rank,
    prune_checkpoints,
    refit_step_budget,
    require_matching_identity,
    save_checkpoint_atomic,
    validation_milestones,
)

ROOT = Path(__file__).resolve().parents[3]
FULL_CONFIG = ROOT / "configs/experiments/full_finetune_whisper_tiny_dy.yaml"
MANIFEST_HASH = "3b680d108b8d2d106bf04708d79ea54599c9f998cba289356ae4ba0ff36e5572"


def _set_environment(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setenv("IVOIREVOICE_DIOULA_DATA_DIR", str(root / "data"))
    monkeypatch.setenv("IVOIREVOICE_ARTIFACTS_DIR", str(root / "artifacts"))
    monkeypatch.setenv("IVOIREVOICE_CHECKPOINT_DIR", str(root / "checkpoints"))
    monkeypatch.setenv(
        "IVOIREVOICE_DIOULA_PILOT_MODEL_PATH",
        str(root / "models/checkpoint-000140"),
    )


def _configured_copy(tmp_path: Path, **overrides: object) -> Path:
    payload = yaml.safe_load(FULL_CONFIG.read_text(encoding="utf-8"))
    experiment = payload["experiment"]
    experiment.update(overrides)
    path = tmp_path / "full.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_full_settings_lock_protocol_and_external_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_environment(monkeypatch, tmp_path)
    path = _configured_copy(tmp_path)

    settings = load_full_training_settings(path)

    assert settings.config_path == path
    assert settings.train_audio_count == 13_764
    assert settings.validation_audio_count == 2_661
    assert settings.refit_audio_count == 16_425
    assert settings.fp16_diagnostic_max_optimizer_attempts == 32
    assert settings.max_consecutive_amp_skips == 4
    assert settings.learning_rate == 1e-5
    assert settings.train_batch_size == 4
    assert settings.gradient_accumulation_steps == 4
    assert settings.max_grad_norm == 1.0
    assert settings.fp16 is True
    assert settings.initial_checkpoint_path.name == "checkpoint-000140"
    assert settings.shareable_output_directory.is_relative_to(tmp_path / "artifacts")


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"train_audio_count": 13_763}, "train_audio_count"),
        ({"learning_rate": 0.00002}, "learning_rate"),
        ({"forbidden_split": "validation"}, "forbidden_split"),
        ({"forced_language_token": "dyu"}, "forced_language_token"),
        ({"expected_model_revision": "a" * 40}, "expected_model_revision"),
        (
            {"fp16_diagnostic_max_optimizer_attempts": 31},
            "fp16_diagnostic_max_optimizer_attempts",
        ),
        ({"max_consecutive_amp_skips": 5}, "max_consecutive_amp_skips"),
    ],
)
def test_full_settings_reject_protocol_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
    field: str,
) -> None:
    _set_environment(monkeypatch, tmp_path)

    with pytest.raises(ConfigError, match=field):
        load_full_training_settings(_configured_copy(tmp_path, **overrides))


def test_step_geometry_milestones_and_refit_budget_are_exact() -> None:
    development = compute_step_geometry(13_764, 4, 4)
    refit = compute_step_geometry(16_425, 4, 4)

    assert development.micro_batches_per_epoch == 3_441
    assert development.optimizer_steps_per_epoch == 861
    assert development.optimizer_steps_per_epoch * 2 == 1_722
    assert refit.micro_batches_per_epoch == 4_107
    assert refit.optimizer_steps_per_epoch == 1_027
    assert refit_step_budget(140, 861, 1_027) == 167
    assert refit_step_budget(1_722, 861, 1_027) == 2_054
    assert validation_milestones(861, 2, 4) == frozenset(
        {216, 431, 646, 861, 1_077, 1_292, 1_507, 1_722}
    )


def test_partial_accumulation_uses_real_tail_divisor() -> None:
    development = accumulation_group_sizes(3_441, 4)
    refit = accumulation_group_sizes(4_107, 4)

    assert len(development) == 861
    assert development[-1] == 1
    assert sum(development) == 3_441
    assert len(refit) == 1_027
    assert refit[-1] == 3
    assert sum(refit) == 4_107


def test_metric_rank_prefers_wer_then_cer_loss_and_earliest_step() -> None:
    metrics = {"wer_micro": 0.8, "cer_micro": 0.3, "validation_loss": 1.2}

    assert metric_rank(metrics, 100) < metric_rank(metrics, 200)
    assert metric_rank(
        {"wer_micro": 0.7, "cer_micro": 0.9, "validation_loss": 9.0},
        300,
    ) < metric_rank(metrics, 100)
    with pytest.raises(ConfigError, match="finies"):
        metric_rank(
            {"wer_micro": float("nan"), "cer_micro": 0.3, "validation_loss": 1.2},
            1,
        )


def _row(
    root: Path,
    index: int,
    *,
    speaker: str,
    split: str,
    create_file: bool,
) -> ManifestRow:
    relative = f"{split}/anonymous/{index:04d}.wav"
    content = f"audio-{split}-{index}".encode()
    if create_file:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return ManifestRow(
        utterance_id=f"utt_{split}_{index:04d}",
        speaker_id=speaker,
        gender_folder="women" if index % 2 else "men",
        language="dyu",
        text_raw=f"Á hakili {index}",
        text_no_tones=f"A hakili {index}",
        target_text=f"A hakili {index}",
        audio_path=relative,
        duration_seconds=1.0 + index / 10,
        sample_rate_hz=16_000,
        channels=1,
        audio_sha256=hashlib.sha256(content).hexdigest(),
        split=split,
        usage_scope="local_research_only",
    )


def _selection_fixture(
    tmp_path: Path,
) -> tuple[AuditedDataset, FullTrainingSettings, tuple[ManifestRow, ...]]:
    data_root = tmp_path / "data"
    train = (
        _row(data_root, 1, speaker="train_a", split="train", create_file=True),
        _row(data_root, 2, speaker="train_b", split="train", create_file=True),
    )
    validation = (
        _row(
            data_root,
            3,
            speaker="validation_a",
            split="validation",
            create_file=True,
        ),
    )
    test = (
        _row(data_root, 4, speaker="test_a", split="test", create_file=False),
        _row(data_root, 5, speaker="test_a", split="test", create_file=False),
    )
    pilot_path = tmp_path / "pilot.csv"
    with pilot_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["utterance_id"])
        writer.writeheader()
        writer.writerow({"utterance_id": test[0].utterance_id})
    settings = cast(
        FullTrainingSettings,
        SimpleNamespace(
            train_split="train",
            validation_split="validation",
            forbidden_split="test",
            dataset_root=data_root,
            max_audio_seconds=30.0,
            expected_manifest_sha256=MANIFEST_HASH,
            expected_dataset_version="0.1.0-local",
            train_audio_count=2,
            train_speaker_count=2,
            validation_audio_count=1,
            validation_speaker_count=1,
            refit_audio_count=3,
            refit_speaker_count=3,
            historical_pilot_prediction_file=pilot_path,
            historical_pilot_test_count=1,
            final_holdout_count=1,
            artifact_output_directory=tmp_path / "artifacts/private",
            shareable_output_directory=tmp_path / "artifacts/shareable",
        ),
    )
    dataset = AuditedDataset(
        rows=(*train, *validation, *test),
        manifest_sha256=MANIFEST_HASH,
        dataset_version="0.1.0-local",
    )
    return dataset, settings, test


def test_full_selection_is_exhaustive_private_and_test_metadata_only(
    tmp_path: Path,
) -> None:
    dataset, settings, _ = _selection_fixture(tmp_path)

    selection, public, private = build_full_selection(dataset, settings)
    public_path, private_path = write_full_selection_reports(
        settings,
        public,
        private,
    )

    assert len(selection.train_rows) == 2
    assert len(selection.validation_rows) == 1
    assert len(selection.refit_rows) == 3
    assert len(selection.historical_pilot_ids) == 1
    assert len(selection.final_holdout_ids) == 1
    assert public["final_holdout"]["decoded"] is False
    assert "train_audio_ids" not in public
    assert json.loads(public_path.read_text(encoding="utf-8"))["privacy"][
        "contains_sample_identifiers"
    ] is False
    assert len(json.loads(private_path.read_text(encoding="utf-8"))["train_audio_ids"]) == 2


def test_full_selection_rejects_changed_training_audio(tmp_path: Path) -> None:
    dataset, settings, _ = _selection_fixture(tmp_path)
    changed = settings.dataset_root / dataset.rows[0].audio_path
    changed.write_bytes(b"changed")

    with pytest.raises(ConfigError, match="manifeste"):
        build_full_selection(dataset, settings)


def test_full_selection_rejects_speaker_leakage(tmp_path: Path) -> None:
    dataset, settings, _ = _selection_fixture(tmp_path)
    validation = dataset.rows[2]
    leaked = replace(validation, speaker_id=dataset.rows[0].speaker_id)
    changed_dataset = AuditedDataset(
        rows=(*dataset.rows[:2], leaked, *dataset.rows[3:]),
        manifest_sha256=dataset.manifest_sha256,
        dataset_version=dataset.dataset_version,
    )

    with pytest.raises(ConfigError, match="Comptes exhaustifs|disjoints"):
        build_full_selection(changed_dataset, settings)


def test_holdout_bytes_are_touched_only_by_explicit_final_validator(
    tmp_path: Path,
) -> None:
    dataset, settings, test = _selection_fixture(tmp_path)
    selection, _, _ = build_full_selection(dataset, settings)
    final_row = next(
        row for row in test if row.utterance_id in selection.final_holdout_ids
    )

    with pytest.raises(ConfigError, match="absent"):
        validate_final_holdout_files((final_row,), settings)

    path = settings.dataset_root / final_row.audio_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"audio-test-{final_row.utterance_id[-4:]}".encode())
    with pytest.raises(ConfigError, match="manifeste"):
        validate_final_holdout_files((final_row,), settings)
    path.write_bytes(b"audio-test-5")
    validate_final_holdout_files((final_row,), settings)


def test_run_and_receipt_identities_fail_closed() -> None:
    identity = {
        "code_commit": "a" * 40,
        "config_sha256": "b" * 64,
        "selection_sha256": "c" * 64,
    }

    assert identity_sha256(identity) == identity_sha256(dict(reversed(identity.items())))
    require_matching_identity(identity, identity, description="Le run")
    with pytest.raises(ConfigError, match="final_checkpoint_sha256"):
        require_matching_identity(
            {"final_checkpoint_sha256": "old"},
            {"final_checkpoint_sha256": "new"},
            description="Le reçu final_holdout",
        )


def test_development_and_refit_start_from_fresh_optimizer_state() -> None:
    context = cast(
        FullContext,
        SimpleNamespace(
            initial_checkpoint_sha256="a" * 64,
            config_sha256="b" * 64,
            code_commit="c" * 40,
        ),
    )

    development = _initial_training_state(
        stage="development",
        selection_hash="d" * 64,
        context=context,
        total_steps=1_722,
    )
    refit = _initial_training_state(
        stage="refit",
        selection_hash="e" * 64,
        context=context,
        total_steps=167,
    )

    for state in (development, refit):
        assert state["global_step"] == 0
        assert state["successful_optimizer_steps"] == 0
        assert state["optimizer_attempts"] == 0
        assert state["amp_skipped_steps"] == 0
        assert state["consecutive_amp_skips"] == 0
        assert state["precision"] == "fp16"
        assert state["optimizer_initialized_from"] == "fresh"
        assert state["scheduler_initialized_from"] == "fresh"
        assert state["pilot_optimizer_state_loaded"] is False
    assert development["run_id"] != refit["run_id"]


def test_started_holdout_receipt_blocks_all_further_training(tmp_path: Path) -> None:
    settings = cast(
        FullTrainingSettings,
        SimpleNamespace(artifact_output_directory=tmp_path),
    )
    _require_training_still_allowed(settings)
    (tmp_path / FINAL_EVALUATION_RECEIPT_FILENAME).write_text(
        '{"status": "started"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="réentraînement"):
        _require_training_still_allowed(settings)


def test_checkpoint_retention_includes_best_latest_and_previous(tmp_path: Path) -> None:
    for step in (100, 200, 300, 400):
        (tmp_path / f"checkpoint-{step:06d}").mkdir()

    prune_checkpoints(
        tmp_path,
        3,
        best_checkpoint_name="checkpoint-000100",
    )

    assert {path.name for path in tmp_path.iterdir()} == {
        "checkpoint-000100",
        "checkpoint-000300",
        "checkpoint-000400",
    }


def test_shared_checkpoint_writer_is_atomic_complete_and_bounded(
    tmp_path: Path,
) -> None:
    class Model:
        def save_pretrained(
            self,
            path: Path,
            *,
            safe_serialization: bool,
        ) -> None:
            assert safe_serialization is True
            (path / "config.json").write_text("{}\n", encoding="utf-8")

    class Processor:
        def save_pretrained(self, path: Path) -> None:
            (path / "processor.json").write_text("{}\n", encoding="utf-8")

    class Stateful:
        def state_dict(self) -> dict[str, int]:
            return {"value": 1}

    class Torch:
        @staticmethod
        def save(value: object, path: Path) -> None:
            path.write_text(repr(value), encoding="utf-8")

    for step in (1, 2, 3):
        save_checkpoint_atomic(
            directory=tmp_path,
            minimum_free_disk_gib=0.0,
            save_total_limit=2,
            best_checkpoint_name="checkpoint-000001",
            state_filename="trainer_state.json",
            model=Model(),
            processor=Processor(),
            optimizer=Stateful(),
            scheduler=Stateful(),
            scaler=Stateful(),
            state={"global_step": step, "finite": True},
            torch=Torch(),
        )

    retained = {path.name for path in tmp_path.glob("checkpoint-*")}
    assert retained == {"checkpoint-000001", "checkpoint-000003"}
    assert not tuple(tmp_path.glob(".checkpoint-*.tmp"))
    assert (
        tmp_path / "checkpoint-000003/trainer_state.json"
    ).is_file()
    assert (tmp_path / "checkpoint-000003/optimizer.pt").is_file()
    assert (tmp_path / "checkpoint-000003/scaler.pt").is_file()


def test_cuda_has_no_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ivoirevoice.training.full_finetune.shutil.which",
        lambda _: None,
    )

    with pytest.raises(ConfigError, match="nvidia-smi"):
        _require_cuda(SimpleNamespace())
