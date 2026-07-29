"""Build the canonical draft manifest without altering raw data."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, fields
from hashlib import sha256
from pathlib import Path
from typing import Any

from ivoirevoice.data.audio import empty_audio_metadata, inspect_audio
from ivoirevoice.data.clips import parse_clips_file
from ivoirevoice.data.discovery import CorpusInventory, discover_corpus
from ivoirevoice.data.matching import match_audio
from ivoirevoice.data.records import ManifestRow, ParsedClip
from ivoirevoice.data.settings import DioulaDataSettings, load_dioula_settings
from ivoirevoice.exceptions import IvoireVoiceError


@dataclass(frozen=True, slots=True)
class ManifestBuild:
    """In-memory result reused by the audit layer."""

    inventory: CorpusInventory
    rows: tuple[ManifestRow, ...]


def _utterance_id(record: ParsedClip) -> str:
    identity = f"{record.source_json}:{record.record_index}:{record.clip_id}"
    return f"utt_{sha256(identity.encode('utf-8')).hexdigest()[:16]}"


def _build_row(
    record: ParsedClip,
    *,
    wav_files: tuple[Path, ...],
    settings: DioulaDataSettings,
) -> ManifestRow:
    audio_match = match_audio(record, wav_files)
    if audio_match.status == "matched" and audio_match.path is not None:
        metadata = inspect_audio(audio_match.path, hash_audio=settings.hash_audio)
        audio_path = audio_match.path.relative_to(settings.dataset_root).as_posix()
        audio_filename = audio_match.path.name
    else:
        metadata = empty_audio_metadata()
        audio_path = ""
        audio_filename = record.audio_filename

    return ManifestRow(
        utterance_id=_utterance_id(record),
        clip_id=record.clip_id,
        sentence_id=record.sentence_id,
        speaker_id=record.speaker_id,
        gender_folder=record.gender_folder,
        language=settings.language,
        text_raw=record.text_raw,
        text_nfc=record.text_nfc,
        text_normalized=record.text_normalized,
        audio_path=audio_path,
        audio_filename=audio_filename,
        audio_match_status=audio_match.status,
        audio_match_method=audio_match.method,
        duration_seconds=metadata.duration_seconds,
        sample_rate_hz=metadata.sample_rate_hz,
        channels=metadata.channels,
        num_samples=metadata.num_samples,
        audio_format=metadata.audio_format,
        file_size_bytes=metadata.file_size_bytes,
        audio_sha256=metadata.audio_sha256,
        audio_status=metadata.audio_status,
        source_json=record.source_json,
        record_index=record.record_index,
        license_status=settings.license_status,
        split="",
        validation_issues=";".join(record.validation_issues),
    )


def build_manifest(settings: DioulaDataSettings) -> ManifestBuild:
    """Discover, parse, match and inspect the corpus sequentially."""

    inventory = discover_corpus(settings.dataset_root)
    rows: list[ManifestRow] = []
    for speaker in inventory.speakers:
        records = parse_clips_file(speaker)
        rows.extend(
            _build_row(record, wav_files=speaker.wav_files, settings=settings) for record in records
        )
    return ManifestBuild(inventory=inventory, rows=tuple(rows))


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    temporary_path.replace(path)


def write_manifest_outputs(build: ManifestBuild, settings: DioulaDataSettings) -> None:
    """Write the draft manifest, inventory and matching issue extracts."""

    fieldnames = [field.name for field in fields(ManifestRow)]
    serialized_rows = [asdict(row) for row in build.rows]
    _atomic_csv(settings.manifest_path, serialized_rows, fieldnames)
    _atomic_json(
        settings.report_directory / "dioula_inventory.json",
        build.inventory.to_shareable_dict(),
    )

    issue_fieldnames = [
        "utterance_id",
        "clip_id",
        "sentence_id",
        "speaker_id",
        "audio_filename",
        "audio_match_status",
        "audio_match_method",
        "source_json",
        "record_index",
    ]
    unmatched = [
        row for row in serialized_rows if row["audio_match_status"] in {"missing", "invalid"}
    ]
    ambiguous = [row for row in serialized_rows if row["audio_match_status"] == "ambiguous"]
    _atomic_csv(
        settings.report_directory / "unmatched_audio.csv",
        unmatched,
        issue_fieldnames,
    )
    _atomic_csv(
        settings.report_directory / "ambiguous_audio.csv",
        ambiguous,
        issue_fieldnames,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construire le manifeste dioula provisoire.")
    parser.add_argument("--config", required=True, help="Configuration YAML des données.")
    return parser.parse_args()


def main() -> int:
    """CLI entry point with privacy-safe aggregate output."""

    args = _parse_args()
    try:
        settings = load_dioula_settings(args.config)
        build = build_manifest(settings)
        write_manifest_outputs(build, settings)
    except IvoireVoiceError as exc:
        print(f"ERREUR: {exc}")
        return 1

    status_counts = {
        status: sum(row.audio_match_status == status for row in build.rows)
        for status in ("matched", "missing", "ambiguous", "invalid")
    }
    print(f"clips_json={len(build.inventory.files_by_kind['clips_json'])}")
    print(f"wav={len(build.inventory.files_by_kind['wav'])}")
    print(f"records={len(build.rows)}")
    for status, count in status_counts.items():
        print(f"{status}={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
