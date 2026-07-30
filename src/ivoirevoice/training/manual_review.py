"""Loopback-only Gradio tool for genuine auditory validation of train samples."""

from __future__ import annotations

import argparse
import importlib
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from ivoirevoice.exceptions import IvoireVoiceError
from ivoirevoice.training.audit import (
    VALIDATION_STATUSES,
    AuditedDataset,
    ManifestRow,
    run_audit,
    save_annotations,
    write_manual_report,
)
from ivoirevoice.training.settings import SmokeSettings, load_smoke_settings


class ReviewSession:
    """Stateful local reviewer backed by an ignored JSON annotation file."""

    def __init__(
        self,
        settings: SmokeSettings,
        dataset: AuditedDataset,
        selected: Sequence[ManifestRow],
        annotations: dict[str, Any],
    ) -> None:
        self.settings = settings
        self.dataset = dataset
        self.selected = tuple(selected)
        self.annotations = annotations

    def _row(self, one_based_index: float | int) -> ManifestRow:
        index = max(1, min(int(one_based_index), len(self.selected))) - 1
        return self.selected[index]

    def display(self, one_based_index: float | int) -> tuple[Any, ...]:
        """Return one local audio path and only anonymized textual metadata."""

        row = self._row(one_based_index)
        statuses = cast(dict[str, Any], self.annotations["statuses"])
        entry = cast(dict[str, Any], statuses[row.utterance_id])
        audio_path = (self.settings.dataset_root / row.audio_path).resolve()
        return (
            str(audio_path),
            row.utterance_id,
            row.speaker_id,
            f"{row.duration_seconds:.3f} s",
            row.text_raw.strip(),
            row.text_no_tones.strip(),
            entry.get("status", "à vérifier"),
            entry.get("anomaly", ""),
        )

    def save(
        self,
        one_based_index: float | int,
        status: str,
        anomaly: str,
    ) -> str:
        """Validate and persist one human decision, then refresh the Markdown report."""

        if status not in VALIDATION_STATUSES:
            raise ValueError("Statut de validation inconnu.")
        row = self._row(one_based_index)
        statuses = cast(dict[str, Any], self.annotations["statuses"])
        statuses[row.utterance_id] = {
            "status": status,
            "anomaly": anomaly.strip(),
            "reviewed_at_utc": datetime.now(UTC).isoformat(),
        }
        save_annotations(self.settings, self.annotations)
        write_manual_report(
            self.settings,
            self.dataset,
            self.selected,
            self.annotations,
        )
        correct = sum(
            isinstance(value, dict) and value.get("status") == "correct"
            for value in statuses.values()
        )
        return (
            f"Décision enregistrée localement. Audios corrects : {correct}/"
            f"{self.settings.minimum_correct_samples} requis."
        )


def create_review_interface(session: ReviewSession) -> Any:
    """Build the local review UI without enabling any public sharing."""

    gradio = importlib.import_module("gradio")
    initial = session.display(1)
    with gradio.Blocks(title="IvoireVoice — validation privée Phase 4B") as interface:
        gradio.Markdown(
            "# Validation auditive privée — Phase 4B\n"
            "Cette interface tourne uniquement en local. Écoutez chaque extrait avant "
            "d'attribuer un statut. Aucun statut n'est prérempli comme « correct »."
        )
        index = gradio.Slider(
            minimum=1,
            maximum=len(session.selected),
            value=1,
            step=1,
            label="Numéro de l'échantillon",
        )
        audio = gradio.Audio(value=initial[0], label="Audio train local", interactive=False)
        with gradio.Row():
            audio_id = gradio.Textbox(
                value=initial[1], label="audio_id anonymisé", interactive=False
            )
            speaker_id = gradio.Textbox(
                value=initial[2], label="speaker_id anonymisé", interactive=False
            )
            duration = gradio.Textbox(value=initial[3], label="Durée", interactive=False)
        text_raw = gradio.Textbox(
            value=initial[4], label="text_raw", lines=2, interactive=False
        )
        text_no_tones = gradio.Textbox(
            value=initial[5], label="text_no_tones", lines=2, interactive=False
        )
        status = gradio.Radio(
            choices=list(VALIDATION_STATUSES),
            value=initial[6],
            label="Statut après écoute",
        )
        anomaly = gradio.Textbox(
            value=initial[7],
            label="Anomalie ou remarque (facultatif)",
            lines=2,
        )
        save_button = gradio.Button("Enregistrer cette décision", variant="primary")
        confirmation = gradio.Markdown()
        outputs = [
            audio,
            audio_id,
            speaker_id,
            duration,
            text_raw,
            text_no_tones,
            status,
            anomaly,
        ]
        index.change(
            session.display,
            inputs=[index],
            outputs=outputs,
            api_name=False,
            show_api=False,
        )
        save_button.click(
            session.save,
            inputs=[index, status, anomaly],
            outputs=[confirmation],
            api_name=False,
            show_api=False,
        )
    return interface


def main(argv: Sequence[str] | None = None) -> int:
    """Audit first, then expose the private sample on a loopback-only server."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        default="configs/experiments/smoke_overfit_whisper_tiny_dy.yaml",
    )
    arguments = parser.parse_args(argv)
    try:
        settings = load_smoke_settings(arguments.experiment)
        result = run_audit(settings)
    except IvoireVoiceError as exc:
        parser.error(str(exc))
    session = ReviewSession(
        settings=settings,
        dataset=cast(AuditedDataset, result["dataset"]),
        selected=cast(tuple[ManifestRow, ...], result["selected"]),
        annotations=cast(dict[str, Any], result["annotations"]),
    )
    host = os.getenv("IVOIREVOICE_REVIEW_HOST", "127.0.0.1")
    if host not in {"127.0.0.1", "localhost"}:
        parser.error("L'outil de validation privée doit rester sur l'interface loopback.")
    try:
        port = int(os.getenv("IVOIREVOICE_REVIEW_PORT", "7861"))
    except ValueError:
        parser.error("IVOIREVOICE_REVIEW_PORT doit être un entier.")
    create_review_interface(session).launch(
        server_name=host,
        server_port=port,
        share=False,
        inbrowser=False,
        allowed_paths=[str(Path(settings.dataset_root))],
        show_error=True,
        show_api=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
