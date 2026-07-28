from __future__ import annotations

import pytest

from ivoirevoice.exceptions import BackendNotLoadedError, ModelRegistryError
from ivoirevoice.models.dummy import DummyBackend
from ivoirevoice.models.registry import ModelRegistry, create_default_registry


def test_default_registry_creates_dummy_backend() -> None:
    registry = create_default_registry()

    backend = registry.create("dummy")

    assert isinstance(backend, DummyBackend)
    assert backend.supported_languages == ("fr", "dyu")


def test_registry_rejects_duplicate_names() -> None:
    registry = ModelRegistry()
    registry.register("dummy", DummyBackend)

    with pytest.raises(ModelRegistryError, match="déjà enregistré"):
        registry.register("dummy", DummyBackend)


def test_registry_reports_unknown_backend() -> None:
    registry = create_default_registry()

    with pytest.raises(ModelRegistryError, match="Backend inconnu"):
        registry.create("whisper")


def test_dummy_backend_requires_explicit_load() -> None:
    backend = DummyBackend()

    with pytest.raises(BackendNotLoadedError):
        backend.transcribe(b"audio", "fr")
