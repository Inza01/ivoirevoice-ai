"""Deterministic lightweight backend used by tests and development surfaces."""

from __future__ import annotations

from time import perf_counter

from ivoirevoice.exceptions import BackendNotLoadedError, UnsupportedLanguageError
from ivoirevoice.models.base import ASRBackend, AudioInput, TranscriptionResult


class DummyBackend(ASRBackend):
    """Backend that exercises contracts without loading or inferring a model."""

    def __init__(self) -> None:
        self._is_loaded = False

    @property
    def model_name(self) -> str:
        return "dummy"

    @property
    def supported_languages(self) -> tuple[str, ...]:
        return ("fr", "dyu")

    def load(self) -> None:
        self._is_loaded = True

    def transcribe(self, audio: AudioInput, language: str) -> TranscriptionResult:
        if not self._is_loaded:
            raise BackendNotLoadedError("Le backend fictif doit être chargé avant utilisation.")
        if language not in self.supported_languages:
            raise UnsupportedLanguageError(
                f"Langue '{language}' non prise en charge par {self.model_name}."
            )
        if isinstance(audio, bytes) and not audio:
            raise ValueError("Le contenu audio est vide.")

        started_at = perf_counter()
        text = "[Transcription fictive : aucun modèle ASR n'est chargé]"
        processing_time = perf_counter() - started_at
        return TranscriptionResult(
            text=text,
            language=language,
            confidence=None,
            duration_seconds=None,
            processing_time_seconds=processing_time,
            model_name=self.model_name,
        )

    def unload(self) -> None:
        self._is_loaded = False
