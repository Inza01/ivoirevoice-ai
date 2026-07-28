"""Minimal Gradio interface backed by DummyBackend."""

from __future__ import annotations

from typing import Any, cast

import gradio as gr

from ivoirevoice.models.base import ASRBackend
from ivoirevoice.models.dummy import DummyBackend


def create_interface(backend: ASRBackend | None = None) -> Any:
    """Create the Phase 2 interface without loading an ML model."""

    selected_backend = backend or DummyBackend()

    def transcribe_audio(audio_path: str | None, language: str) -> tuple[str, str, str]:
        if not audio_path:
            return "Veuillez fournir un fichier audio.", selected_backend.model_name, "0.000 s"

        selected_backend.load()
        try:
            result = selected_backend.transcribe(audio_path, language)
        finally:
            selected_backend.unload()
        return (
            result.text,
            result.model_name,
            f"{result.processing_time_seconds:.3f} s",
        )

    with gr.Blocks(title="IvoireVoice AI — Démonstrateur fictif") as interface:
        gr.Markdown(
            "# IvoireVoice AI\nInterface de développement : aucun modèle ASR réel n'est chargé."
        )
        audio = gr.Audio(
            sources=["upload", "microphone"],
            type="filepath",
            label="Audio",
        )
        language = gr.Dropdown(
            choices=[("Français", "fr"), ("Dioula", "dyu")],
            value="fr",
            label="Langue",
        )
        transcribe_button = gr.Button("Transcrire", variant="primary")
        transcript = gr.Textbox(label="Texte")
        model_name = gr.Textbox(label="Modèle")
        processing_time = gr.Textbox(label="Temps de traitement")
        cast(Any, transcribe_button).click(
            fn=transcribe_audio,
            inputs=[audio, language],
            outputs=[transcript, model_name, processing_time],
        )

    return interface


def main() -> None:
    """Launch the local Gradio development server."""

    create_interface().launch()


if __name__ == "__main__":
    main()
