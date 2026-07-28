"""Registry for resolving ASR backends without framework coupling."""

from __future__ import annotations

from collections.abc import Callable

from ivoirevoice.exceptions import ModelRegistryError
from ivoirevoice.models.base import ASRBackend
from ivoirevoice.models.dummy import DummyBackend

BackendFactory = Callable[[], ASRBackend]


class ModelRegistry:
    """Map stable backend names to lightweight factories."""

    def __init__(self) -> None:
        self._factories: dict[str, BackendFactory] = {}

    def register(self, name: str, factory: BackendFactory, *, replace: bool = False) -> None:
        normalized_name = name.strip().lower()
        if not normalized_name:
            raise ModelRegistryError("Le nom du backend ne peut pas être vide.")
        if normalized_name in self._factories and not replace:
            raise ModelRegistryError(f"Le backend '{normalized_name}' est déjà enregistré.")
        self._factories[normalized_name] = factory

    def create(self, name: str) -> ASRBackend:
        normalized_name = name.strip().lower()
        try:
            factory = self._factories[normalized_name]
        except KeyError as exc:
            available = ", ".join(self.available_models) or "aucun"
            raise ModelRegistryError(
                f"Backend inconnu '{normalized_name}'. Backends disponibles : {available}."
            ) from exc
        return factory()

    @property
    def available_models(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


def create_default_registry() -> ModelRegistry:
    """Create the Phase 2 registry without any heavyweight backend."""

    registry = ModelRegistry()
    registry.register("dummy", DummyBackend)
    return registry
