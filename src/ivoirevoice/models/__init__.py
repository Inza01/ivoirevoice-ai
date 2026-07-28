"""ASR backend contracts and registry."""

from ivoirevoice.models.base import ASRBackend, AudioInput, TranscriptionResult
from ivoirevoice.models.dummy import DummyBackend
from ivoirevoice.models.registry import ModelRegistry, create_default_registry

__all__ = [
    "ASRBackend",
    "AudioInput",
    "DummyBackend",
    "ModelRegistry",
    "TranscriptionResult",
    "create_default_registry",
]
