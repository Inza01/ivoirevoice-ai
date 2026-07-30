from __future__ import annotations

from pathlib import Path

import pytest

from ivoirevoice.exceptions import ConfigError
from ivoirevoice.training.settings import load_smoke_settings


def _config(split: str = "train", sample_count: int = 16) -> str:
    return f"""
experiment:
  id: smoke-test
  mode: smoke_overfit
  language: dyu
  split: {split}
  seed: 42
  model_config: configs/models/whisper_tiny.yaml
  expected_model_id: openai/whisper-tiny
  manifest_path: manifests/manifest.csv
  dataset_metadata_path: manifests/metadata.json
  pilot_prediction_files:
    - baselines/tiny/predictions_private.csv
  output_directory: training/smoke
  sample_count: {sample_count}
  minimum_correct_samples: 10
  canonical_text_column: target_text_mvp
  max_steps: 20
  batch_size: 2
  learning_rate: 0.0001
  max_grad_norm: 1.0
  logging_steps: 1
  evaluation_steps: 5
  mixed_precision: auto
  stop_on_nan: true
  save_model: false
  publication_allowed: false
"""


def _environment(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setenv("IVOIREVOICE_DIOULA_DATA_DIR", str(root / "data"))
    monkeypatch.setenv("IVOIREVOICE_ARTIFACTS_DIR", str(root / "artifacts"))
    monkeypatch.setenv("IVOIREVOICE_MODEL_CACHE_DIR", str(root / "cache"))
    monkeypatch.setenv("IVOIREVOICE_TRAINING_REPORTS_DIR", str(root / "reports"))


def test_load_smoke_settings_accepts_bounded_train_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _environment(monkeypatch, tmp_path)
    config = tmp_path / "smoke.yaml"
    config.write_text(_config(), encoding="utf-8")

    settings = load_smoke_settings(config)

    assert settings.split == "train"
    assert settings.sample_count == 16
    assert settings.minimum_correct_samples == 10
    assert settings.expected_model_id == "openai/whisper-tiny"
    assert settings.save_model is False


@pytest.mark.parametrize(
    ("split", "sample_count", "message"),
    [
        ("test", 16, "exclusivement le split train"),
        ("validation", 16, "exclusivement le split train"),
        ("train", 9, "compris entre 10 et 20"),
        ("train", 21, "compris entre 10 et 20"),
    ],
)
def test_load_smoke_settings_rejects_test_or_invalid_sample_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    split: str,
    sample_count: int,
    message: str,
) -> None:
    _environment(monkeypatch, tmp_path)
    config = tmp_path / "invalid.yaml"
    config.write_text(_config(split=split, sample_count=sample_count), encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_smoke_settings(config)
