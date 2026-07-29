from __future__ import annotations

from pathlib import Path

import pytest

from ivoirevoice.data.settings import load_dioula_settings
from ivoirevoice.exceptions import ConfigError

CONFIG = """
dataset:
  language_code: dyu
  source_environment_variable: IVOIREVOICE_DIOULA_DATA_DIR
  license_status: unknown
  consent_status: unknown
  usage_scope: local_research_only
  hash_audio: true
  split:
    seed: 42
    train_ratio: 0.8
    validation_ratio: 0.1
    test_ratio: 0.1
artifacts:
  root_environment_variable: IVOIREVOICE_ARTIFACTS_DIR
  manifest_path: manifests/dioula.csv
  report_directory: reports/data_audit
curation:
  source_manifest_path: manifests/dioula.csv
  candidate_manifest_path: manifests/dioula_candidate.csv
  metadata_path: manifests/metadata.json
  report_directory: reports/data_curation
  target_text: text_without_tones_nfc
  recover_missing_audio: false
  recovery_output_environment_variable: IVOIREVOICE_DIOULA_INTERIM_DIR
freeze:
  split_comparison_path: reports/data_curation/split_comparison.json
  manifest_path: manifests/dioula_v0.1.csv
  metadata_path: manifests/dioula_v0.1_metadata.json
  report_path: reports/data_curation/dioula_v0.1.md
  split_report_path: reports/data_curation/dioula_v0.1_split.json
  dataset_version: 0.1.0-local
  dataset_status: frozen_candidate
  split_strategy: B_15_3_3
  expected_audio_count: 21
  expected_speaker_count: 21
  expected_speaker_counts:
    train: 15
    validation: 3
    test: 3
  publication_allowed: false
  model_derivative_publication_allowed: false
"""


def _config(tmp_path: Path, content: str = CONFIG) -> Path:
    path = tmp_path / "dioula.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_paths_only_from_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    artifacts = tmp_path / "artifacts"
    dataset.mkdir()
    monkeypatch.setenv("IVOIREVOICE_DIOULA_DATA_DIR", str(dataset))
    monkeypatch.setenv("IVOIREVOICE_ARTIFACTS_DIR", str(artifacts))
    monkeypatch.setenv("IVOIREVOICE_HASH_AUDIO", "false")

    settings = load_dioula_settings(_config(tmp_path))

    assert settings.dataset_root == dataset
    assert settings.artifacts_root == artifacts
    assert settings.hash_audio is False
    assert settings.consent_status == "unknown"
    assert settings.usage_scope == "local_research_only"
    assert settings.freeze.expected_speaker_counts == {
        "train": 15,
        "validation": 3,
        "test": 3,
    }


def test_rejects_artifacts_inside_raw_corpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    monkeypatch.setenv("IVOIREVOICE_DIOULA_DATA_DIR", str(dataset))
    monkeypatch.setenv("IVOIREVOICE_ARTIFACTS_DIR", str(dataset / "generated"))

    with pytest.raises(ConfigError, match="corpus brut"):
        load_dioula_settings(_config(tmp_path))


def test_rejects_output_path_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    monkeypatch.setenv("IVOIREVOICE_DIOULA_DATA_DIR", str(dataset))
    monkeypatch.setenv("IVOIREVOICE_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    invalid_config = CONFIG.replace(
        "manifest_path: manifests/dioula.csv",
        "manifest_path: ../outside.csv",
    )

    with pytest.raises(ConfigError, match="chemin relatif sûr"):
        load_dioula_settings(_config(tmp_path, invalid_config))
