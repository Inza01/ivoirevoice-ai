"""Serializable session state for the local Gradio demonstration."""

from __future__ import annotations

from dataclasses import dataclass

from ivoirevoice.services.comparison_service import ComparisonRun


@dataclass(frozen=True, slots=True)
class UIState:
    """Latest private comparison and its temporary downloadable exports."""

    run: ComparisonRun | None = None
    export_paths: tuple[str, str, str] | None = None


def empty_state() -> UIState:
    return UIState()
