"""YAML configuration loading with environment-variable overrides."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from ivoirevoice.exceptions import ConfigError

ENV_PREFIX = "IVOIREVOICE_"
CONFIG_PATH_ENV = f"{ENV_PREFIX}CONFIG_PATH"

ConfigMapping = dict[str, Any]


@dataclass(frozen=True, slots=True)
class ProjectSettings:
    """Project-wide settings."""

    name: str
    version: str
    default_language: str
    supported_languages: tuple[str, ...]
    default_model: str


@dataclass(frozen=True, slots=True)
class APISettings:
    """HTTP API limits and accepted formats."""

    max_upload_size_mb: int
    allowed_content_types: tuple[str, ...]
    audio_retention: str

    @property
    def max_upload_size_bytes(self) -> int:
        """Return the upload limit expressed in bytes."""

        return self.max_upload_size_mb * 1024 * 1024


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Validated application configuration."""

    project: ProjectSettings
    api: APISettings
    source_path: Path


def _default_config_path() -> Path:
    configured_path = os.getenv(CONFIG_PATH_ENV)
    if configured_path:
        return Path(configured_path).expanduser()

    working_directory_candidate = Path.cwd() / "configs" / "project.yaml"
    if working_directory_candidate.is_file():
        return working_directory_candidate

    repository_candidate = Path(__file__).resolve().parents[2] / "configs" / "project.yaml"
    return repository_candidate


def _load_yaml(path: Path) -> ConfigMapping:
    try:
        with path.open(encoding="utf-8") as stream:
            raw: object = yaml.safe_load(stream)
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration introuvable : {path}") from exc
    except OSError as exc:
        raise ConfigError(f"Impossible de lire la configuration {path} : {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML invalide dans {path} : {exc}") from exc

    if not isinstance(raw, Mapping):
        raise ConfigError(f"La racine de {path} doit être un objet YAML.")
    if not all(isinstance(key, str) for key in raw):
        raise ConfigError(f"Toutes les clés de {path} doivent être des chaînes.")
    return cast(ConfigMapping, dict(raw))


def _coerce_environment_value(raw_value: str, current_value: Any) -> Any:
    if isinstance(current_value, list) and not raw_value.lstrip().startswith("["):
        return [item.strip() for item in raw_value.split(",") if item.strip()]
    try:
        return yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Valeur de surcharge invalide : {raw_value!r}") from exc


def _apply_environment_overrides(data: ConfigMapping) -> None:
    for variable_name, raw_value in os.environ.items():
        if not variable_name.startswith(ENV_PREFIX):
            continue

        nested_name = variable_name.removeprefix(ENV_PREFIX)
        if "__" not in nested_name:
            continue

        path = [part.lower() for part in nested_name.split("__") if part]
        if not path:
            continue

        cursor = data
        for part in path[:-1]:
            existing = cursor.get(part)
            if existing is None:
                child: ConfigMapping = {}
                cursor[part] = child
                cursor = child
            elif isinstance(existing, dict):
                cursor = existing
            else:
                raise ConfigError(
                    f"La variable {variable_name} cible une valeur non imbriquée à '{part}'."
                )

        leaf = path[-1]
        cursor[leaf] = _coerce_environment_value(raw_value, cursor.get(leaf))


def _section(data: ConfigMapping, name: str) -> ConfigMapping:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"La section obligatoire '{name}' doit être un objet YAML.")
    return value


def _required_string(section: ConfigMapping, field: str, section_name: str) -> str:
    value = section.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Le champ '{section_name}.{field}' doit être une chaîne non vide.")
    return value.strip()


def _required_string_tuple(
    section: ConfigMapping,
    field: str,
    section_name: str,
) -> tuple[str, ...]:
    value = section.get(field)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ConfigError(
            f"Le champ '{section_name}.{field}' doit être une liste non vide de chaînes."
        )
    return tuple(item.strip() for item in value)


def _validate(data: ConfigMapping, source_path: Path) -> AppConfig:
    project_data = _section(data, "project")
    api_data = _section(data, "api")

    supported_languages = _required_string_tuple(project_data, "supported_languages", "project")
    default_language = _required_string(project_data, "default_language", "project")
    if default_language not in supported_languages:
        raise ConfigError(
            "Le champ 'project.default_language' doit appartenir à 'project.supported_languages'."
        )

    max_upload_size_mb = api_data.get("max_upload_size_mb")
    if (
        not isinstance(max_upload_size_mb, int)
        or isinstance(max_upload_size_mb, bool)
        or max_upload_size_mb <= 0
    ):
        raise ConfigError("Le champ 'api.max_upload_size_mb' doit être un entier positif.")

    project = ProjectSettings(
        name=_required_string(project_data, "name", "project"),
        version=_required_string(project_data, "version", "project"),
        default_language=default_language,
        supported_languages=supported_languages,
        default_model=_required_string(project_data, "default_model", "project"),
    )
    api = APISettings(
        max_upload_size_mb=max_upload_size_mb,
        allowed_content_types=_required_string_tuple(api_data, "allowed_content_types", "api"),
        audio_retention=_required_string(api_data, "audio_retention", "api"),
    )
    if api.audio_retention != "delete_immediately":
        raise ConfigError(
            "Le champ 'api.audio_retention' doit valoir delete_immediately pour ce MVP."
        )
    return AppConfig(project=project, api=api, source_path=source_path)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load, override and validate the main project configuration."""

    source_path = Path(path).expanduser() if path is not None else _default_config_path()
    data = _load_yaml(source_path)
    _apply_environment_overrides(data)
    return _validate(data, source_path)
