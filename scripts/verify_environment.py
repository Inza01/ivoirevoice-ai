"""Perform lightweight, offline checks of the development environment."""

from __future__ import annotations

import platform
import sys

from ivoirevoice.config import load_config
from ivoirevoice.models.registry import create_default_registry


def main() -> int:
    """Check Python, configuration and the lightweight backend registry."""

    print(f"Python: {platform.python_version()}")
    if sys.version_info[:2] != (3, 11):
        print("ERREUR: IvoireVoice cible Python 3.11.")
        return 1

    config = load_config()
    print(f"Configuration: {config.source_path}")
    print(f"Langues: {', '.join(config.project.supported_languages)}")

    registry = create_default_registry()
    print(f"Backends: {', '.join(registry.available_models)}")
    if config.project.default_model not in registry.available_models:
        print("ERREUR: le backend par défaut n'est pas enregistré.")
        return 1

    print("Environnement léger valide (aucun téléchargement requis).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
