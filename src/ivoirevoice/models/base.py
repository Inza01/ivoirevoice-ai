"""Stable contracts implemented by every ASR backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

AudioInput: TypeAlias = bytes | str | Path


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Normalized result returned by every ASR backend."""

    text: str
    language: str
    processing_time_seconds: float
    model_name: str
    confidence: float | None = None
    duration_seconds: float | None = None


class ASRBackend(ABC):
    """Abstract interface shielding the application from model frameworks."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the stable registry name of the backend."""

    @property
    @abstractmethod
    def supported_languages(self) -> tuple[str, ...]:
        """Return the ISO-like language codes supported by the backend."""

    @abstractmethod
    def load(self) -> None:
        """Load resources required for transcription."""

    @abstractmethod
    def transcribe(self, audio: AudioInput, language: str) -> TranscriptionResult:
        """Transcribe audio into a normalized result."""

    @abstractmethod
    def unload(self) -> None:
        """Release resources owned by the backend."""
