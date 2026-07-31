"""Private, structured exports with anonymized audio identifiers."""

from __future__ import annotations

import atexit
import csv
import json
import shutil
import tempfile
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ivoirevoice.exceptions import ConfigError
from ivoirevoice.services.comparison_service import ComparisonRun


class ExportService:
    """Create and clean temporary JSON, CSV, TXT and audio-preview files."""

    def __init__(self, temporary_root: Path | None = None) -> None:
        self._root = temporary_root or Path(tempfile.mkdtemp(prefix="ivoirevoice-ui-"))
        self._root.mkdir(parents=True, exist_ok=True)
        self._created_paths: set[Path] = set()
        atexit.register(self.cleanup)

    def _path(self, stem: str, suffix: str) -> Path:
        self._root.mkdir(parents=True, exist_ok=True)
        safe_stem = "".join(
            character for character in stem if character.isalnum() or character in "-_"
        )
        path = self._root / f"{safe_stem}{suffix}"
        self._created_paths.add(path)
        return path

    @staticmethod
    def _assert_private_paths_absent(payload: object) -> None:
        serialized = json.dumps(payload, ensure_ascii=False)
        if "/home/" in serialized or "\\Users\\" in serialized:
            raise ConfigError("Un export contient un chemin local privé.")

    def export_json(self, run: ComparisonRun) -> Path:
        payload = run.to_dict()
        self._assert_private_paths_absent(payload)
        path = self._path(run.experiment_id, ".json")
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    def export_csv(self, run: ComparisonRun) -> Path:
        path = self._path(run.experiment_id, ".csv")
        rows: list[dict[str, Any]] = []
        for result in run.results:
            evaluation = asdict(result.evaluation)
            rows.append(
                {
                    "experiment_id": run.experiment_id,
                    "generated_at_utc": run.generated_at_utc,
                    "audio_id": run.audio_id,
                    "language": run.language,
                    "reference": run.reference or "",
                    "model_key": result.model_key,
                    "model_id": result.model_id,
                    "model_revision": result.model_revision,
                    "model_status": result.model_status,
                    "checkpoint_name": result.checkpoint_name or "",
                    "task": result.task,
                    "configured_language": result.configured_language or "",
                    "training_audio_count": result.training_audio_count,
                    "validation_audio_count": result.validation_audio_count,
                    "device": result.device,
                    "hardware": result.hardware,
                    "success": result.success,
                    "transcription": result.transcription,
                    "processing_time_seconds": result.processing_time_seconds,
                    "audio_duration_seconds": result.audio_duration_seconds,
                    "rtf": result.rtf,
                    "wer": evaluation["wer"],
                    "cer": evaluation["cer"],
                    "substitutions": evaluation["substitutions"],
                    "deletions": evaluation["deletions"],
                    "insertions": evaluation["insertions"],
                    "error": result.error or "",
                }
            )
        self._assert_private_paths_absent(rows)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def export_txt(self, run: ComparisonRun) -> Path:
        lines = [
            "IvoireVoice AI — comparaison ASR",
            f"Expérience : {run.experiment_id}",
            f"Date UTC : {run.generated_at_utc}",
            f"Audio ID : {run.audio_id}",
            f"Langue : {run.language}",
            f"Référence : {run.reference or 'non fournie'}",
            "",
        ]
        for result in run.results:
            lines.extend(
                [
                    result.display_name,
                    f"Statut : {'succès' if result.success else 'échec'}",
                    f"Révision : {result.model_revision}",
                    f"Checkpoint : {result.checkpoint_name or 'non applicable'}",
                    f"Tâche : {result.task}",
                    (
                        "Configuration de langue : "
                        f"{result.configured_language or 'multilingue sans token forcé'}"
                    ),
                    (
                        "Audios d'entraînement : "
                        f"{result.training_audio_count or 'non applicable'}"
                    ),
                    (
                        "Audios de validation : "
                        f"{result.validation_audio_count or 'non applicable'}"
                    ),
                    f"Appareil : {result.device}",
                    f"Matériel : {result.hardware}",
                    f"Transcription : {result.transcription}",
                    f"WER : {result.evaluation.wer}",
                    f"CER : {result.evaluation.cer}",
                    f"RTF : {result.rtf}",
                    f"Erreur : {result.error or 'aucune'}",
                    "",
                ]
            )
        payload = "\n".join(lines)
        self._assert_private_paths_absent(payload)
        path = self._path(run.experiment_id, ".txt")
        path.write_text(payload, encoding="utf-8")
        return path

    def export_all(self, run: ComparisonRun) -> tuple[str, str, str]:
        return (
            str(self.export_json(run)),
            str(self.export_csv(run)),
            str(self.export_txt(run)),
        )

    def prepare_audio_preview(self, source: Path, audio_id: str) -> str:
        suffix = source.suffix.lower() if source.suffix else ".wav"
        destination = self._path(f"sample-{audio_id}", suffix)
        try:
            shutil.copyfile(source, destination)
        except OSError as exc:
            raise ConfigError("Impossible de préparer l'aperçu audio privé.") from exc
        return str(destination)

    def cleanup(self) -> None:
        for path in tuple(self._created_paths):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                continue
            self._created_paths.discard(path)
        with suppress(OSError):
            self._root.rmdir()
