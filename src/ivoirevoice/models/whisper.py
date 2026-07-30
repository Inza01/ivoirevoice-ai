"""Lazy, configurable Hugging Face Whisper backend for local-only inference."""

from __future__ import annotations

import gc
import importlib
import io
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, cast

import yaml

from ivoirevoice.exceptions import (
    BackendNotLoadedError,
    ConfigError,
    UnsupportedLanguageError,
)
from ivoirevoice.models.base import ASRBackend, AudioInput, TranscriptionResult


@dataclass(frozen=True, slots=True)
class WhisperSettings:
    """Pinned model and inference parameters."""

    backend_name: str
    model_id: str
    model_revision: str
    device: str
    torch_dtype: str
    batch_size: int
    chunk_length_seconds: float
    stride_length_seconds: float
    task: str
    language: str | None
    cache_dir: Path
    local_files_only: bool
    max_new_tokens: int
    expected_sampling_rate_hz: int
    supported_languages: tuple[str, ...]


class WhisperPipeline(Protocol):
    """Minimal callable surface used from a Transformers ASR pipeline."""

    def __call__(self, inputs: Any, **kwargs: Any) -> Mapping[str, Any]:
        """Run one transcription."""


PipelineFactory = Callable[[WhisperSettings], WhisperPipeline]
AudioLoader = Callable[[AudioInput, int], tuple[Any, float]]


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ConfigError(f"La section '{name}' doit être un objet YAML.")
    return cast(dict[str, Any], dict(value))


def _required_string(data: dict[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Le champ 'model.{field}' doit être une chaîne non vide.")
    return value.strip()


def _positive_int(data: dict[str, Any], field: str) -> int:
    value = data.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"Le champ 'model.{field}' doit être un entier positif.")
    return value


def _non_negative_float(data: dict[str, Any], field: str) -> float:
    value = data.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise ConfigError(f"Le champ 'model.{field}' doit être un nombre positif ou nul.")
    return float(value)


def load_whisper_settings(path: str | Path) -> WhisperSettings:
    """Load and strictly validate one pinned Whisper configuration."""

    config_path = Path(path)
    try:
        with config_path.open(encoding="utf-8") as stream:
            raw: object = yaml.safe_load(stream)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"Impossible de lire la configuration Whisper : {exc}") from exc
    root = _mapping(raw, "racine")
    model = _mapping(root.get("model"), "model")

    family = _required_string(model, "family")
    if family != "whisper":
        raise ConfigError("Le backend Whisper refuse une autre famille de modèle.")
    task = _required_string(model, "task")
    if task != "transcribe":
        raise ConfigError("La baseline exige model.task=transcribe.")
    device = _required_string(model, "device").lower()
    if device not in {"auto", "cpu", "cuda"}:
        raise ConfigError("model.device doit valoir auto, cpu ou cuda.")
    torch_dtype = _required_string(model, "torch_dtype").lower()
    if torch_dtype not in {"auto", "float32", "float16", "bfloat16"}:
        raise ConfigError("model.torch_dtype n'est pas pris en charge.")

    language_value = model.get("language")
    if language_value is not None and (
        not isinstance(language_value, str) or not language_value.strip()
    ):
        raise ConfigError("model.language doit être null ou une chaîne non vide.")
    language = language_value.strip() if isinstance(language_value, str) else None
    if language == "dyu":
        raise ConfigError("Le token de langue dyu ne doit pas être forcé dans Whisper.")

    cache_environment_variable = _required_string(
        model,
        "cache_environment_variable",
    )
    raw_cache_dir = os.getenv(cache_environment_variable)
    if not raw_cache_dir:
        raise ConfigError(
            f"La variable obligatoire '{cache_environment_variable}' n'est pas définie."
        )
    cache_dir = Path(raw_cache_dir).expanduser().resolve()

    local_files_only = model.get("local_files_only")
    if not isinstance(local_files_only, bool):
        raise ConfigError("model.local_files_only doit être un booléen.")
    raw_languages = model.get("supported_languages")
    if (
        not isinstance(raw_languages, list)
        or not raw_languages
        or not all(isinstance(item, str) and item.strip() for item in raw_languages)
    ):
        raise ConfigError("model.supported_languages doit être une liste non vide.")

    chunk_length = _non_negative_float(model, "chunk_length_seconds")
    stride_length = _non_negative_float(model, "stride_length_seconds")
    if chunk_length and stride_length * 2 >= chunk_length:
        raise ConfigError("Le stride Whisper doit être inférieur à la moitié du chunk.")

    return WhisperSettings(
        backend_name=_required_string(model, "name"),
        model_id=_required_string(model, "model_id"),
        model_revision=_required_string(model, "model_revision"),
        device=device,
        torch_dtype=torch_dtype,
        batch_size=_positive_int(model, "batch_size"),
        chunk_length_seconds=chunk_length,
        stride_length_seconds=stride_length,
        task=task,
        language=language,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        max_new_tokens=_positive_int(model, "max_new_tokens"),
        expected_sampling_rate_hz=_positive_int(
            model,
            "expected_sampling_rate_hz",
        ),
        supported_languages=tuple(cast(list[str], [item.strip() for item in raw_languages])),
    )


def _resolve_runtime(settings: WhisperSettings, torch: Any) -> tuple[int, Any]:
    cuda_available = bool(torch.cuda.is_available())
    if settings.device == "cuda" and not cuda_available:
        raise ConfigError("CUDA est demandé mais indisponible.")
    use_cuda = settings.device == "cuda" or (settings.device == "auto" and cuda_available)
    device = 0 if use_cuda else -1

    requested_dtype = settings.torch_dtype
    if requested_dtype == "auto":
        requested_dtype = "float16" if use_cuda else "float32"
    if not use_cuda and requested_dtype == "float16":
        raise ConfigError("float16 n'est pas autorisé pour cette baseline CPU.")
    if requested_dtype == "bfloat16" and use_cuda:
        checker = getattr(torch.cuda, "is_bf16_supported", None)
        if callable(checker) and not checker():
            raise ConfigError("Le GPU ne prend pas en charge bfloat16.")
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[requested_dtype]
    return device, dtype


def runtime_labels(settings: WhisperSettings) -> tuple[str, str]:
    """Resolve the effective device and dtype without loading model weights."""

    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise ConfigError("PyTorch doit être installé pour résoudre le runtime.") from exc
    device, dtype = _resolve_runtime(settings, torch)
    dtype_name = str(dtype).removeprefix("torch.")
    return ("cuda" if device == 0 else "cpu"), dtype_name


def _create_transformers_pipeline(settings: WhisperSettings) -> WhisperPipeline:
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise ConfigError(
            "PyTorch et Transformers doivent être installés pour charger Whisper."
        ) from exc

    device, dtype = _resolve_runtime(settings, torch)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    pipeline = transformers.pipeline(
        "automatic-speech-recognition",
        model=settings.model_id,
        revision=settings.model_revision,
        device=device,
        dtype=dtype,
        trust_remote_code=False,
        model_kwargs={
            "cache_dir": str(settings.cache_dir),
            "local_files_only": settings.local_files_only,
            "use_safetensors": True,
        },
    )
    return cast(WhisperPipeline, pipeline)


def _load_audio(audio: AudioInput, expected_sampling_rate_hz: int) -> tuple[Any, float]:
    try:
        numpy = importlib.import_module("numpy")
        soundfile = importlib.import_module("soundfile")
    except ImportError as exc:
        raise ConfigError("NumPy et SoundFile sont requis pour lire les audios.") from exc

    source: Any
    if isinstance(audio, bytes):
        if not audio:
            raise ValueError("Le contenu audio est vide.")
        source = io.BytesIO(audio)
    else:
        path = Path(audio)
        if not path.is_file():
            raise ValueError("Le fichier audio local est introuvable.")
        source = str(path)
    try:
        samples, sampling_rate = soundfile.read(
            source,
            dtype="float32",
            always_2d=True,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("Le fichier audio ne peut pas être décodé.") from exc
    if len(samples) == 0 or sampling_rate <= 0:
        raise ValueError("Le fichier audio ne contient aucun échantillon.")

    mono = samples.mean(axis=1, dtype=numpy.float32)
    duration_seconds = float(len(mono) / sampling_rate)
    if sampling_rate != expected_sampling_rate_hz:
        target_length = max(
            1,
            round(len(mono) * expected_sampling_rate_hz / sampling_rate),
        )
        source_positions = numpy.arange(len(mono), dtype=numpy.float64)
        target_positions = numpy.linspace(
            0,
            len(mono) - 1,
            target_length,
            dtype=numpy.float64,
        )
        mono = numpy.interp(target_positions, source_positions, mono).astype(numpy.float32)
    return {
        "array": mono,
        "sampling_rate": expected_sampling_rate_hz,
    }, duration_seconds


class WhisperBackend(ASRBackend):
    """Whisper inference backend that never forces an unsupported Dioula token."""

    def __init__(
        self,
        settings: WhisperSettings,
        *,
        pipeline_factory: PipelineFactory | None = None,
        audio_loader: AudioLoader | None = None,
    ) -> None:
        self.settings = settings
        self._pipeline_factory = pipeline_factory or _create_transformers_pipeline
        self._audio_loader = audio_loader or _load_audio
        self._pipeline: WhisperPipeline | None = None

    @property
    def model_name(self) -> str:
        return f"{self.settings.model_id}@{self.settings.model_revision}"

    @property
    def supported_languages(self) -> tuple[str, ...]:
        return self.settings.supported_languages

    def load(self) -> None:
        if self._pipeline is None:
            try:
                self._pipeline = self._pipeline_factory(self.settings)
            except ConfigError:
                raise
            except Exception as exc:
                raise ConfigError(f"Impossible de charger Whisper ({type(exc).__name__}).") from exc

    def transcribe(self, audio: AudioInput, language: str) -> TranscriptionResult:
        if self._pipeline is None:
            raise BackendNotLoadedError("Le backend Whisper doit être chargé avant utilisation.")
        if language not in self.supported_languages:
            raise UnsupportedLanguageError(
                f"Langue '{language}' non prise en charge par {self.model_name}."
            )

        inputs, duration_seconds = self._audio_loader(
            audio,
            self.settings.expected_sampling_rate_hz,
        )
        generation_kwargs: dict[str, Any] = {
            "task": self.settings.task,
            "max_new_tokens": self.settings.max_new_tokens,
        }
        if self.settings.language is not None:
            generation_kwargs["language"] = self.settings.language
        inference_kwargs: dict[str, Any] = {
            "generate_kwargs": generation_kwargs,
            "batch_size": self.settings.batch_size,
        }
        if self.settings.chunk_length_seconds:
            inference_kwargs["chunk_length_s"] = self.settings.chunk_length_seconds
            inference_kwargs["stride_length_s"] = self.settings.stride_length_seconds

        started_at = perf_counter()
        result = self._pipeline(inputs, **inference_kwargs)
        processing_time = perf_counter() - started_at
        text = result.get("text")
        if not isinstance(text, str):
            raise ValueError("Whisper n'a pas retourné un texte valide.")
        return TranscriptionResult(
            text=text,
            language=language,
            confidence=None,
            duration_seconds=duration_seconds,
            processing_time_seconds=processing_time,
            model_name=self.model_name,
        )

    def unload(self) -> None:
        self._pipeline = None
        gc.collect()
        try:
            torch = importlib.import_module("torch")
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
