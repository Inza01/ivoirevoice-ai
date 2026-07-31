from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from ivoirevoice.exceptions import ConfigError
from ivoirevoice.training.audit import AuditedDataset, ManifestRow
from ivoirevoice.training.pilot_finetune import (
    _public_comparison,
    _write_json,
    comparison_metrics,
    find_latest_checkpoint,
    relative_reduction,
)
from ivoirevoice.training.pilot_selection import (
    build_pilot_selection,
    select_balanced_subset,
    write_selection_report,
)
from ivoirevoice.training.pilot_settings import PilotSettings, load_pilot_settings


def _row(index: int, *, speaker: str, split: str) -> ManifestRow:
    return ManifestRow(
        utterance_id=f"utt_{split}_{index:04d}",
        speaker_id=speaker,
        gender_folder="women" if index % 2 else "men",
        language="dyu",
        text_raw=f"Á hakili kà go {index} ↘",
        text_no_tones=f"A hakili ka go {index} ↘",
        target_text="A hakili ka go " + "wa " * (index % 8),
        audio_path=f"{split}/anonymous/{index:04d}.wav",
        duration_seconds=0.8 + index * 0.03,
        sample_rate_hz=16_000,
        channels=1,
        audio_sha256=f"{index + {'train': 1, 'validation': 1000, 'test': 2000}[split]:064x}",
        split=split,
        usage_scope="local_research_only",
    )


def _yaml(**overrides: object) -> str:
    values: dict[str, object] = {
        "train_split": "train",
        "num_train_epochs": 1,
        "forced_language_token": "null",
    }
    values.update(overrides)
    return f"""
experiment:
  id: pilot-test
  mode: pilot_finetune
  language: dyu
  train_split: {values["train_split"]}
  validation_split: validation
  forbidden_split: test
  seed: 42
  model_config: configs/models/whisper_tiny.yaml
  expected_model_id: openai/whisper-tiny
  expected_model_revision: be0ba7c2f24f0127b27863a23a08002af4c2c279
  manifest_path: manifests/manifest.csv
  dataset_metadata_path: manifests/metadata.json
  pilot_prediction_file: baselines/pilot.csv
  artifact_output_directory: training/pilot
  report_output_directory: reports/training
  checkpoint_environment_variable: IVOIREVOICE_CHECKPOINT_DIR
  train_sample_count: 2250
  validation_sample_count: 600
  expected_pilot_test_count: 150
  expected_final_holdout_count: 2624
  canonical_text_column: target_text_mvp
  num_train_epochs: {values["num_train_epochs"]}
  learning_rate: 0.00001
  warmup_ratio: 0.05
  weight_decay: 0.01
  train_batch_size: 4
  eval_batch_size: 8
  gradient_accumulation_steps: 4
  max_grad_norm: 1.0
  logging_steps: 5
  evaluation_steps: 35
  fp16: true
  gradient_checkpointing: true
  early_stopping_patience: 2
  early_stopping_threshold: 0.0
  save_total_limit: 2
  max_audio_seconds: 30.0
  minimum_free_disk_gib: 5.0
  resume_from_checkpoint: true
  task: transcribe
  forced_language_token: {values["forced_language_token"]}
  post_correction: false
  publication_allowed: false
"""


def _set_environment(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setenv("IVOIREVOICE_DIOULA_DATA_DIR", str(root / "data"))
    monkeypatch.setenv("IVOIREVOICE_ARTIFACTS_DIR", str(root / "artifacts"))
    monkeypatch.setenv("IVOIREVOICE_CHECKPOINT_DIR", str(root / "checkpoints"))


def test_pilot_settings_accept_bounded_train_validation_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_environment(monkeypatch, tmp_path)
    path = tmp_path / "pilot.yaml"
    path.write_text(_yaml(), encoding="utf-8")

    settings = load_pilot_settings(path)

    assert settings.train_sample_count == 2250
    assert settings.validation_sample_count == 600
    assert settings.train_split == "train"
    assert settings.forbidden_split == "test"
    assert settings.checkpoint_directory == tmp_path / "checkpoints" / "pilot-test"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"train_split": "test"}, "train_split"),
        ({"num_train_epochs": 3}, "deux époques"),
        ({"forced_language_token": "dyu"}, "Aucun token"),
    ],
)
def test_pilot_settings_reject_test_unbounded_or_forced_dyu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
    message: str,
) -> None:
    _set_environment(monkeypatch, tmp_path)
    path = tmp_path / "invalid.yaml"
    path.write_text(_yaml(**overrides), encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_pilot_settings(path)


def test_balanced_selection_is_deterministic_and_covers_each_speaker() -> None:
    rows = tuple(
        _row(speaker_index * 20 + item, speaker=f"spk_{speaker_index}", split="train")
        for speaker_index in range(4)
        for item in range(20)
    )

    selected = select_balanced_subset(rows, split="train", count=40, seed=42)

    assert selected == select_balanced_subset(rows, split="train", count=40, seed=42)
    assert len(selected) == 40
    assert {row.speaker_id for row in selected} == {f"spk_{index}" for index in range(4)}
    assert all(
        sum(row.speaker_id == speaker for row in selected) == 10
        for speaker in {row.speaker_id for row in selected}
    )


def test_pilot_partition_excludes_validation_pilot_and_final_holdout(
    tmp_path: Path,
) -> None:
    train = tuple(
        _row(index, speaker=f"train_{index // 10}", split="train")
        for index in range(30)
    )
    validation = tuple(
        _row(index, speaker=f"validation_{index // 10}", split="validation")
        for index in range(20)
    )
    test = tuple(_row(index, speaker="test_speaker", split="test") for index in range(8))
    for row in (*train, *validation):
        path = tmp_path / "data" / row.audio_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    pilot_path = tmp_path / "pilot.csv"
    with pilot_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["utterance_id"])
        writer.writeheader()
        writer.writerows({"utterance_id": row.utterance_id} for row in test[:3])
    settings = cast(
        PilotSettings,
        SimpleNamespace(
            train_split="train",
            validation_split="validation",
            forbidden_split="test",
            seed=42,
            dataset_root=tmp_path / "data",
            max_audio_seconds=30.0,
            train_sample_count=15,
            validation_sample_count=10,
            pilot_prediction_file=pilot_path,
            expected_pilot_test_count=3,
            expected_final_holdout_count=5,
        ),
    )
    dataset = AuditedDataset(
        rows=(*train, *validation, *test),
        manifest_sha256="a" * 64,
        dataset_version="test",
    )

    selection, report = build_pilot_selection(dataset, settings)

    assert report["overall_passed"] is True
    assert len(selection.train_rows) == 15
    assert len(selection.validation_rows) == 10
    assert len(selection.pilot_test_ids) == 3
    assert len(selection.final_holdout_ids) == 5
    assert all(value == 0 for value in report["integrity_violations"].values())


def test_relative_reductions_and_comparison_are_exact() -> None:
    baseline = {"wer_micro": 1.0, "cer_micro": 0.5}
    adapted = {"wer_micro": 0.75, "cer_micro": 0.25}

    comparison = comparison_metrics(baseline, adapted)

    assert relative_reduction(1.0, 0.75) == 25.0
    assert comparison["wer_absolute_reduction"] == 0.25
    assert comparison["wer_relative_reduction_percent"] == 25.0
    assert comparison["cer_relative_reduction_percent"] == 50.0


def test_resume_uses_latest_complete_checkpoint(tmp_path: Path) -> None:
    complete = tmp_path / "checkpoint-000035"
    incomplete = tmp_path / "checkpoint-000070"
    complete.mkdir()
    incomplete.mkdir()
    for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "config.json"):
        (complete / name).touch()
    (incomplete / "trainer_state.json").touch()

    assert find_latest_checkpoint(tmp_path) == complete


def test_metric_serialization_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    payload = {"wer": 0.25, "stable": True, "checkpoint": "checkpoint-000035"}

    _write_json(path, payload)

    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_metric_serialization_rejects_non_finite_values(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="non sérialisables"):
        _write_json(tmp_path / "metrics.json", {"loss": float("nan")})


def test_public_reports_remove_speaker_and_sample_identifiers(tmp_path: Path) -> None:
    public_comparison = _public_comparison(
        {
            "baseline": {
                "wer_micro": 1.0,
                "speaker_metrics": {"spk_private": {"wer": 1.0}},
            },
            "adapted": {
                "wer_micro": 0.5,
                "speaker_metrics": {"spk_private": {"wer": 0.5}},
            },
        }
    )
    settings = cast(
        PilotSettings,
        SimpleNamespace(
            artifact_output_directory=tmp_path / "private",
            report_output_directory=tmp_path / "public",
        ),
    )
    report = {
        "train": {
            "audio_count": 2,
            "audio_count_by_anonymized_speaker": {"spk_private": 2},
        },
        "validation": {
            "audio_count": 1,
            "audio_count_by_anonymized_speaker": {"spk_private": 1},
        },
        "train_audio_ids": ["utt_private"],
        "validation_audio_ids": ["utt_private_validation"],
        "pilot_test_audio_ids": ["utt_private_test"],
        "final_holdout_audio_ids": ["utt_private_holdout"],
    }

    public_path = write_selection_report(settings, report)
    public_selection = json.loads(public_path.read_text(encoding="utf-8"))
    private_selection = json.loads(
        (tmp_path / "private/pilot_selection_private.json").read_text(
            encoding="utf-8"
        )
    )

    assert "speaker_metrics" not in public_comparison["baseline"]
    assert "speaker_metrics" not in public_comparison["adapted"]
    assert "train_audio_ids" not in public_selection
    assert "audio_count_by_anonymized_speaker" not in public_selection["train"]
    assert public_selection["privacy"]["contains_sample_identifiers"] is False
    assert private_selection["train_audio_ids"] == ["utt_private"]
