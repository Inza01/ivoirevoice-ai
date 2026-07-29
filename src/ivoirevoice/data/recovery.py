"""Explicitly gated ffmpeg recovery for missing converted WAV files."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

from ivoirevoice.data.audio import inspect_audio
from ivoirevoice.data.settings import DioulaDataSettings, load_dioula_settings
from ivoirevoice.exceptions import ConfigError, IvoireVoiceError


def _load_plan(settings: DioulaDataSettings) -> dict[str, Any]:
    plan_path = settings.curation_report_directory / "missing_audio_recovery_plan.json"
    try:
        with plan_path.open(encoding="utf-8") as stream:
            value: object = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError("Le plan de récupération doit être généré par la curation.") from exc
    if not isinstance(value, dict) or not isinstance(value.get("proposed_rows"), list):
        raise ConfigError("Le plan de récupération est invalide.")
    return value


def _recovery_root(settings: DioulaDataSettings) -> Path:
    variable_name = settings.curation.recovery_output_environment_variable
    raw_path = os.getenv(variable_name)
    if not raw_path:
        raise ConfigError(f"La variable '{variable_name}' est obligatoire pour la récupération.")
    output_root = Path(raw_path).expanduser().resolve()
    repository_root = Path(__file__).resolve().parents[3]
    if output_root == settings.dataset_root or output_root.is_relative_to(settings.dataset_root):
        raise ConfigError("La récupération ne peut pas écrire dans le corpus brut.")
    if output_root == repository_root or output_root.is_relative_to(repository_root):
        raise ConfigError("La récupération doit écrire hors du dépôt Git.")
    return output_root


def _safe_source_path(value: object, dataset_root: Path) -> Path:
    if not isinstance(value, str):
        raise ConfigError("Chemin source de récupération invalide.")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ConfigError("Chemin source de récupération non sûr.")
    source = dataset_root / Path(*relative.parts)
    if not source.is_file():
        raise ConfigError("Une source planifiée n'existe plus.")
    return source


def execute_recovery(
    settings: DioulaDataSettings,
    *,
    explicit_confirmation: bool,
) -> list[dict[str, Any]]:
    """Convert only planned missing sources after two explicit safety gates."""

    if not settings.curation.recover_missing_audio:
        raise ConfigError("La récupération est désactivée dans la configuration.")
    if not explicit_confirmation:
        raise ConfigError("L'option explicite --execute est obligatoire.")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ConfigError("ffmpeg est introuvable.")

    plan = _load_plan(settings)
    output_root = _recovery_root(settings)
    output_root.mkdir(parents=True, exist_ok=True)
    provenance: list[dict[str, Any]] = []
    for item in plan["proposed_rows"]:
        if not isinstance(item, dict) or item.get("status") != "source_found":
            continue
        utterance_id = str(item.get("utterance_id", ""))
        if not utterance_id.startswith("utt_"):
            continue
        source = _safe_source_path(item.get("source_path"), settings.dataset_root)
        output = output_root / f"{utterance_id}.wav"
        command = [
            ffmpeg,
            "-nostdin",
            "-n",
            "-v",
            "error",
            "-i",
            str(source),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            status = "conversion_failed"
            metadata = None
        else:
            metadata = inspect_audio(output, hash_audio=True)
            status = (
                "converted_and_validated"
                if metadata.audio_status == "readable"
                else "converted_but_invalid"
            )
        provenance.append(
            {
                "utterance_id": utterance_id,
                "source_path": source.relative_to(settings.dataset_root).as_posix(),
                "output_path": output.relative_to(output_root).as_posix(),
                "status": status,
                "audio_sha256": metadata.audio_sha256 if metadata else "",
                "duration_seconds": metadata.duration_seconds if metadata else "",
                "sample_rate_hz": metadata.sample_rate_hz if metadata else "",
                "channels": metadata.channels if metadata else "",
            }
        )
    return provenance


def write_provenance(
    rows: list[dict[str, Any]],
    settings: DioulaDataSettings,
) -> None:
    """Write a path-relative conversion log outside Git."""

    path = settings.curation_report_directory / "missing_audio_recovery_provenance.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".csv.tmp")
    fieldnames = [
        "utterance_id",
        "source_path",
        "output_path",
        "status",
        "audio_sha256",
        "duration_seconds",
        "sample_rate_hz",
        "channels",
    ]
    with temporary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Récupérer les WAV dioula manquants.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Confirme explicitement la conversion ffmpeg locale.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI that remains blocked unless config and command both authorize execution."""

    args = _parse_args()
    try:
        settings = load_dioula_settings(args.config)
        provenance = execute_recovery(
            settings,
            explicit_confirmation=bool(args.execute),
        )
        write_provenance(provenance, settings)
    except IvoireVoiceError as exc:
        print(f"ERREUR: {exc}")
        return 1
    status_counts: dict[str, int] = {}
    for row in provenance:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    print(f"planned_conversions={len(provenance)}")
    for status, count in sorted(status_counts.items()):
        print(f"{status}={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
