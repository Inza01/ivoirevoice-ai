from __future__ import annotations

from pathlib import Path

import pytest

from ivoirevoice.config import load_config
from ivoirevoice.exceptions import ConfigError

VALID_CONFIG = """
project:
  name: Test Voice
  version: 0.1.0
  default_language: fr
  supported_languages: [fr, dyu]
  default_model: dummy
api:
  max_upload_size_mb: 2
  allowed_content_types: [audio/wav]
  audio_retention: delete_immediately
"""


def write_config(path: Path, content: str = VALID_CONFIG) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_valid_configuration(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path / "project.yaml"))

    assert config.project.name == "Test Voice"
    assert config.project.supported_languages == ("fr", "dyu")
    assert config.api.max_upload_size_bytes == 2 * 1024 * 1024
    assert config.api.audio_retention == "delete_immediately"


def test_environment_overrides_nested_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_config(tmp_path / "project.yaml")
    monkeypatch.setenv("IVOIREVOICE_PROJECT__DEFAULT_LANGUAGE", "dyu")
    monkeypatch.setenv("IVOIREVOICE_API__MAX_UPLOAD_SIZE_MB", "3")

    config = load_config(config_path)

    assert config.project.default_language == "dyu"
    assert config.api.max_upload_size_mb == 3


def test_rejects_default_language_outside_supported_list(tmp_path: Path) -> None:
    invalid_config = VALID_CONFIG.replace("default_language: fr", "default_language: baoulé")

    with pytest.raises(ConfigError, match="default_language"):
        load_config(write_config(tmp_path / "invalid.yaml", invalid_config))


def test_rejects_persistent_audio_retention(tmp_path: Path) -> None:
    invalid_config = VALID_CONFIG.replace("delete_immediately", "keep_forever")

    with pytest.raises(ConfigError, match="delete_immediately"):
        load_config(write_config(tmp_path / "invalid-retention.yaml", invalid_config))
