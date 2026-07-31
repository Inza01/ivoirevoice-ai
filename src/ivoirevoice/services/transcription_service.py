"""Validated, lazy and sequential access to registered ASR backends."""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import soundfile
import yaml

from ivoirevoice.data.audio import sha256_file
from ivoirevoice.exceptions import ConfigError
from ivoirevoice.models.base import TranscriptionResult
from ivoirevoice.models.registry import BackendFactory, ModelRegistry
from ivoirevoice.models.whisper import WhisperBackend, load_whisper_settings

MODEL_STATUS = {"baseline", "adapted", "pilot_adapted"}
MODEL_BACKENDS = {"whisper"}


@dataclass(frozen=True, slots=True)
class ModelDefinition:
    """Display and factory metadata for one configurable ASR model."""

    key: str
    display_name: str
    backend: str
    status: str
    model_id: str
    revision: str
    config_path: Path | None
    device: str
    languages: tuple[str, ...]
    enabled: bool
    model_path: Path | None = None
    checkpoint_name: str | None = None
    task: str = "transcribe"
    configured_language: str | None = None
    training_audio_count: int | None = None
    validation_audio_count: int | None = None


@dataclass(frozen=True, slots=True)
class ModelCatalog:
    """Validated UI limits and enabled model definitions."""

    models: tuple[ModelDefinition, ...]
    max_audio_size_bytes: int
    max_audio_duration_seconds: float
    allowed_extensions: tuple[str, ...]

    @property
    def enabled_models(self) -> tuple[ModelDefinition, ...]:
        return tuple(model for model in self.models if model.enabled)

    def definition(self, key: str) -> ModelDefinition:
        for model in self.enabled_models:
            if model.key == key:
                return model
        raise ConfigError("Le modèle demandé n'est pas activé.")


@dataclass(frozen=True, slots=True)
class AudioMetadata:
    """Safe aggregate facts about one user-provided audio."""

    audio_id: str
    duration_seconds: float
    size_bytes: int
    extension: str


@dataclass(frozen=True, slots=True)
class TranscriptionOutput:
    """One successful backend result enriched with safe model metadata."""

    model_key: str
    display_name: str
    model_status: str
    model_id: str
    model_revision: str
    device: str
    hardware: str
    transcription: str
    processing_time_seconds: float
    audio_duration_seconds: float
    rtf: float
    checkpoint_name: str | None
    task: str
    configured_language: str | None
    training_audio_count: int | None
    validation_audio_count: int | None


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ConfigError(f"La section {label} doit être un objet YAML.")
    return cast(dict[str, Any], dict(value))


def _string(data: dict[str, Any], field: str, label: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label}.{field} doit être une chaîne non vide.")
    return value.strip()


def _positive_number(data: dict[str, Any], field: str, label: str) -> float:
    value = data.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{label}.{field} doit être strictement positif.")
    return float(value)


def _string_list(data: dict[str, Any], field: str, label: str) -> tuple[str, ...]:
    value = data.get(field)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        raise ConfigError(f"{label}.{field} doit être une liste non vide.")
    return tuple(cast(str, item).strip() for item in value)


def _optional_positive_int(data: dict[str, Any], field: str, label: str) -> int | None:
    value = data.get(field)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{label}.{field} doit être un entier strictement positif.")
    return value


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_config_path(data: dict[str, Any], *, enabled: bool, label: str) -> Path | None:
    raw_path = data.get("config_path")
    environment_variable = data.get("config_path_environment_variable")
    if raw_path is not None and environment_variable is not None:
        raise ConfigError(f"{label} ne peut définir qu'une source de configuration.")
    if isinstance(raw_path, str) and raw_path.strip():
        relative = Path(raw_path.strip())
        if relative.is_absolute() or ".." in relative.parts:
            raise ConfigError(f"{label}.config_path doit être un chemin relatif sûr.")
        return (_repository_root() / relative).resolve()
    if isinstance(environment_variable, str) and environment_variable.strip():
        configured = os.getenv(environment_variable.strip())
        if configured:
            return Path(configured).expanduser().resolve()
        if enabled:
            raise ConfigError(
                f"La variable {environment_variable.strip()} est requise pour {label}."
            )
        return None
    if enabled:
        raise ConfigError(f"{label} activé exige une configuration de modèle.")
    return None


def _resolve_model_path(data: dict[str, Any], *, label: str) -> Path | None:
    raw_value = data.get("model_path")
    if raw_value is None:
        return None
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ConfigError(f"{label}.model_path doit être une chaîne non vide.")
    placeholder = raw_value.strip()
    if not (placeholder.startswith("${") and placeholder.endswith("}")):
        raise ConfigError(
            f"{label}.model_path doit référencer uniquement une variable d'environnement."
        )
    variable_name = placeholder[2:-1]
    if not variable_name or not variable_name.replace("_", "").isalnum():
        raise ConfigError(f"{label}.model_path contient une variable invalide.")
    configured = os.getenv(variable_name)
    return Path(configured).expanduser().resolve() if configured else None


def _optional_safe_string(data: dict[str, Any], field: str, label: str) -> str | None:
    value = data.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label}.{field} doit être une chaîne non vide.")
    normalized = value.strip()
    if "/" in normalized or "\\" in normalized:
        raise ConfigError(f"{label}.{field} ne doit pas contenir de chemin.")
    return normalized


def load_model_catalog(path: str | Path | None = None) -> ModelCatalog:
    """Load the UI model catalog without loading or downloading model weights."""

    source = (
        Path(path) if path is not None else _repository_root() / "configs" / "ui" / "models.yaml"
    )
    try:
        raw: object = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"Impossible de lire le catalogue des modèles : {exc}") from exc
    root = _mapping(raw, "racine")
    ui = _mapping(root.get("ui"), "ui")
    raw_models = _mapping(root.get("models"), "models")

    max_audio_size_mb = _positive_number(ui, "max_audio_size_mb", "ui")
    max_audio_duration = _positive_number(ui, "max_audio_duration_seconds", "ui")
    allowed_extensions = tuple(
        extension.lower() for extension in _string_list(ui, "allowed_extensions", "ui")
    )
    if any(not extension.startswith(".") for extension in allowed_extensions):
        raise ConfigError("Chaque extension autorisée doit commencer par un point.")

    models: list[ModelDefinition] = []
    for key, value in raw_models.items():
        label = f"models.{key}"
        data = _mapping(value, label)
        enabled = data.get("enabled")
        if not isinstance(enabled, bool):
            raise ConfigError(f"{label}.enabled doit être booléen.")
        backend = _string(data, "backend", label).lower()
        status = _string(data, "status", label).lower()
        device = _string(data, "device", label).lower()
        if backend not in MODEL_BACKENDS:
            raise ConfigError(f"{label}.backend n'est pas pris en charge.")
        if status not in MODEL_STATUS:
            raise ConfigError(f"{label}.status n'est pas pris en charge.")
        if device not in {"auto", "cpu", "cuda"}:
            raise ConfigError(f"{label}.device est invalide.")
        task = str(data.get("task", "transcribe")).strip()
        if task != "transcribe":
            raise ConfigError(f"{label}.task doit valoir transcribe.")
        configured_language = data.get("language")
        if configured_language is not None and (
            not isinstance(configured_language, str) or not configured_language.strip()
        ):
            raise ConfigError(f"{label}.language doit être une chaîne non vide.")
        models.append(
            ModelDefinition(
                key=key.strip().lower(),
                display_name=_string(data, "display_name", label),
                backend=backend,
                status=status,
                model_id=_string(data, "model_id", label),
                revision=_string(data, "revision", label),
                config_path=_resolve_config_path(data, enabled=enabled, label=label),
                device=device,
                languages=_string_list(data, "languages", label),
                enabled=enabled,
                model_path=_resolve_model_path(data, label=label),
                checkpoint_name=_optional_safe_string(
                    data,
                    "checkpoint_name",
                    label,
                ),
                task=task,
                configured_language=(
                    configured_language.strip()
                    if isinstance(configured_language, str)
                    else None
                ),
                training_audio_count=_optional_positive_int(
                    data,
                    "training_audio_count",
                    label,
                ),
                validation_audio_count=_optional_positive_int(
                    data,
                    "validation_audio_count",
                    label,
                ),
            )
        )
    enabled_models = [model for model in models if model.enabled]
    if len(enabled_models) < 1:
        raise ConfigError("Le catalogue doit activer au moins un modèle.")
    keys = [model.key for model in models]
    if len(keys) != len(set(keys)):
        raise ConfigError("Les clés de modèles doivent être uniques.")
    return ModelCatalog(
        models=tuple(models),
        max_audio_size_bytes=round(max_audio_size_mb * 1024 * 1024),
        max_audio_duration_seconds=max_audio_duration,
        allowed_extensions=allowed_extensions,
    )


def _backend_factory(definition: ModelDefinition) -> WhisperBackend:
    if definition.config_path is None:
        raise ConfigError("La configuration du modèle activé est absente.")
    settings = load_whisper_settings(definition.config_path)
    if settings.model_id != definition.model_id:
        raise ConfigError("Le model_id du catalogue ne correspond pas à sa configuration.")
    if settings.model_revision != definition.revision:
        raise ConfigError("La révision du catalogue ne correspond pas à sa configuration.")
    if definition.model_path is not None:
        required_files = ("config.json", "model.safetensors", "preprocessor_config.json")
        if not definition.model_path.is_dir() or any(
            not (definition.model_path / filename).is_file()
            for filename in required_files
        ):
            raise ConfigError("Le checkpoint pilote configuré est absent ou incomplet.")
        settings = replace(
            settings,
            backend_name=definition.key,
            model_id=str(definition.model_path),
            device=definition.device,
            task=definition.task,
            language=None,
            local_files_only=True,
            supported_languages=definition.languages,
        )
    elif definition.status == "pilot_adapted":
        raise ConfigError(
            "IVOIREVOICE_DIOULA_PILOT_MODEL_PATH est requis pour le modèle pilote."
        )
    return WhisperBackend(settings)


def _factory_for(definition: ModelDefinition) -> BackendFactory:
    def factory() -> WhisperBackend:
        return _backend_factory(definition)

    return factory


def build_model_registry(catalog: ModelCatalog) -> ModelRegistry:
    """Register lightweight factories; no model is loaded at this stage."""

    registry = ModelRegistry()
    for definition in catalog.enabled_models:
        registry.register(definition.key, _factory_for(definition))
    return registry


def _effective_device(configured_device: str) -> str:
    if configured_device != "auto":
        return configured_device
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return "cpu"
    return "cuda" if bool(torch.cuda.is_available()) else "cpu"


def _hardware_label(device: str) -> str:
    if device != "cuda":
        return "CPU"
    try:
        torch = importlib.import_module("torch")
        return str(torch.cuda.get_device_name(0))
    except (ImportError, RuntimeError):
        return "GPU CUDA"


class TranscriptionService:
    """Validate inputs and use exactly one registered model at a time."""

    def __init__(self, catalog: ModelCatalog, registry: ModelRegistry) -> None:
        self.catalog = catalog
        self.registry = registry

    def validate_extension(self, filename: str | Path) -> str:
        extension = Path(filename).suffix.lower()
        if extension not in self.catalog.allowed_extensions:
            allowed = ", ".join(self.catalog.allowed_extensions)
            raise ConfigError(f"Format audio refusé. Formats autorisés : {allowed}.")
        return extension

    def validate_audio(self, audio_path: str | Path) -> AudioMetadata:
        path = Path(audio_path)
        extension = self.validate_extension(path)
        try:
            size_bytes = path.stat().st_size
        except OSError as exc:
            raise ConfigError("Le fichier audio est introuvable ou illisible.") from exc
        if size_bytes <= 0:
            raise ConfigError("Le fichier audio est vide.")
        if size_bytes > self.catalog.max_audio_size_bytes:
            raise ConfigError("Le fichier audio dépasse la taille maximale autorisée.")
        try:
            info = soundfile.info(path)
        except (OSError, RuntimeError) as exc:
            raise ConfigError("Le fichier audio ne peut pas être décodé.") from exc
        duration = float(info.duration)
        if duration <= 0:
            raise ConfigError("La durée audio doit être strictement positive.")
        if duration > self.catalog.max_audio_duration_seconds:
            raise ConfigError("La durée audio dépasse la limite autorisée.")
        return AudioMetadata(
            audio_id=sha256_file(path)[:16],
            duration_seconds=duration,
            size_bytes=size_bytes,
            extension=extension,
        )

    def transcribe(
        self,
        *,
        model_key: str,
        audio_path: str | Path,
        language: str,
        audio_metadata: AudioMetadata,
    ) -> TranscriptionOutput:
        definition = self.catalog.definition(model_key)
        if language not in definition.languages:
            raise ConfigError("La langue choisie n'est pas activée pour ce modèle.")
        backend = self.registry.create(model_key)
        try:
            backend.load()
            result: TranscriptionResult = backend.transcribe(audio_path, language)
        finally:
            backend.unload()
        duration = result.duration_seconds or audio_metadata.duration_seconds
        effective_device = _effective_device(definition.device)
        return TranscriptionOutput(
            model_key=definition.key,
            display_name=definition.display_name,
            model_status=definition.status,
            model_id=definition.model_id,
            model_revision=definition.revision,
            device=effective_device,
            hardware=_hardware_label(effective_device),
            transcription=result.text,
            processing_time_seconds=result.processing_time_seconds,
            audio_duration_seconds=duration,
            rtf=(result.processing_time_seconds / duration if duration else 0.0),
            checkpoint_name=definition.checkpoint_name,
            task=definition.task,
            configured_language=definition.configured_language,
            training_audio_count=definition.training_audio_count,
            validation_audio_count=definition.validation_audio_count,
        )
