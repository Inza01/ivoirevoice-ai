"""Domain exceptions used across IvoireVoice."""

from __future__ import annotations

from typing import Any


class IvoireVoiceError(Exception):
    """Base class for controlled application errors."""


class ConfigError(IvoireVoiceError):
    """Raised when a configuration cannot be loaded or validated."""


class ModelRegistryError(IvoireVoiceError):
    """Raised when a backend cannot be registered or resolved."""


class BackendNotLoadedError(IvoireVoiceError):
    """Raised when inference is requested before loading a backend."""


class UnsupportedLanguageError(IvoireVoiceError):
    """Raised when a backend does not support the requested language."""


class APIError(IvoireVoiceError):
    """An error that can safely be exposed through the HTTP API."""

    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        details: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details
