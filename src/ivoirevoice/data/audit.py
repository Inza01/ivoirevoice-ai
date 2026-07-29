"""Reproducible, read-only audit of the local Dioula corpus."""

from __future__ import annotations

import argparse
import json
import statistics
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ivoirevoice.data.manifest import (
    ManifestBuild,
    build_manifest,
    write_manifest_outputs,
)
from ivoirevoice.data.records import ManifestRow, SpeakerSource
from ivoirevoice.data.settings import DioulaDataSettings, load_dioula_settings
from ivoirevoice.data.split import propose_speaker_split
from ivoirevoice.exceptions import IvoireVoiceError


@dataclass(frozen=True, slots=True)
class AuditResult:
    """Aggregate audit result."""

    build: ManifestBuild
    summary: dict[str, Any]
    split_proposal: dict[str, Any]


def _numeric_statistics(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "total": 0.0,
            "mean": None,
            "median": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "count": len(values),
        "total": sum(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _distribution(values: list[int]) -> dict[str, int]:
    counts = Counter(values)
    return {str(value): counts[value] for value in sorted(counts)}


def _duplicate_statistics(values: list[str]) -> dict[str, int]:
    counts = Counter(value for value in values if value)
    duplicates = [count for count in counts.values() if count > 1]
    return {
        "groups": len(duplicates),
        "records": sum(duplicates),
        "extra_occurrences": sum(count - 1 for count in duplicates),
    }


def _unicode_statistics(rows: tuple[ManifestRow, ...]) -> list[dict[str, Any]]:
    counts = Counter(
        character for row in rows for character in row.text_nfc if not character.isspace()
    )
    return [
        {
            "codepoint": f"U+{ord(character):04X}",
            "character": character,
            "name": unicodedata.name(character, "UNKNOWN"),
            "count": count,
        }
        for character, count in sorted(counts.items(), key=lambda item: ord(item[0]))
    ]


def _text_variant_statistics(speakers: tuple[SpeakerSource, ...]) -> dict[str, int]:
    paired_speakers = 0
    aligned_speakers = 0
    line_count_mismatches = 0
    compared_lines = 0
    differing_lines = 0
    for speaker in speakers:
        tones_path = speaker.directory / "text"
        no_tones_path = speaker.directory / "text-no-tones"
        if not tones_path.is_file() or not no_tones_path.is_file():
            continue
        paired_speakers += 1
        try:
            tones_lines = tones_path.read_text(encoding="utf-8").splitlines()
            no_tones_lines = no_tones_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            continue
        if len(tones_lines) != len(no_tones_lines):
            line_count_mismatches += 1
            continue
        aligned_speakers += 1
        compared_lines += len(tones_lines)
        differing_lines += sum(
            tones_line != no_tones_line
            for tones_line, no_tones_line in zip(tones_lines, no_tones_lines, strict=True)
        )
    return {
        "paired_speakers": paired_speakers,
        "aligned_speakers": aligned_speakers,
        "line_count_mismatches": line_count_mismatches,
        "compared_lines": compared_lines,
        "differing_lines": differing_lines,
    }


def _group_audio_statistics(
    rows: tuple[ManifestRow, ...],
    attribute: str,
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in rows:
        grouped[str(getattr(row, attribute))].append(row)

    result: dict[str, dict[str, float | int]] = {}
    for group, group_rows in sorted(grouped.items()):
        unique_audio = {row.audio_path: row for row in group_rows if row.audio_path}
        unique_rows = list(unique_audio.values())
        durations = [
            row.duration_seconds for row in unique_rows if row.duration_seconds is not None
        ]
        result[group] = {
            "records": len(group_rows),
            "matched": sum(row.audio_match_status == "matched" for row in group_rows),
            "unique_audio_files": len(unique_rows),
            "readable": sum(row.audio_status == "readable" for row in unique_rows),
            "corrupted": sum(row.audio_status == "corrupted" for row in unique_rows),
            "format_mismatch": sum(row.audio_status == "format_mismatch" for row in unique_rows),
            "duration_seconds": sum(durations),
        }
    return result


def _text_summary(rows: tuple[ManifestRow, ...]) -> dict[str, Any]:
    lengths = [float(len(row.text_normalized)) for row in rows]
    text_to_audio: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.text_normalized and row.audio_path:
            text_to_audio[row.text_normalized].add(row.audio_path)

    return {
        "empty_texts": sum(not row.text_normalized for row in rows),
        "duplicate_text": _duplicate_statistics([row.text_normalized for row in rows]),
        "duplicate_sentence_id": _duplicate_statistics([row.sentence_id for row in rows]),
        "identical_text_multiple_audio_groups": sum(
            len(audio_paths) > 1 for audio_paths in text_to_audio.values()
        ),
        "records_with_down_arrow": sum("↘" in row.text_raw for row in rows),
        "down_arrow_occurrences": sum(row.text_raw.count("↘") for row in rows),
        "records_with_carriage_return": sum("\r" in row.text_raw for row in rows),
        "length_characters": _numeric_statistics(lengths),
        "unicode_characters": _unicode_statistics(rows),
    }


def _container_signature_statistics(paths: tuple[Path, ...]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for path in paths:
        try:
            with path.open("rb") as stream:
                header = stream.read(12)
        except OSError:
            counts["unreadable"] += 1
            continue
        if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WAVE":
            counts["riff_wave"] += 1
        elif len(header) >= 8 and header[4:8] == b"ftyp":
            counts["iso_base_media_mislabeled_wav"] += 1
        else:
            counts["other_or_unknown"] += 1
    return {
        name: counts[name]
        for name in (
            "riff_wave",
            "iso_base_media_mislabeled_wav",
            "other_or_unknown",
            "unreadable",
        )
    }


def _audio_summary(
    rows: tuple[ManifestRow, ...],
    *,
    all_wav_files: tuple[Path, ...],
    hash_enabled: bool,
) -> dict[str, Any]:
    unique_audio = {row.audio_path: row for row in rows if row.audio_path}
    unique_rows = list(unique_audio.values())
    readable_rows = [row for row in unique_rows if row.audio_status == "readable"]
    hashes = Counter(row.audio_sha256 for row in unique_rows if row.audio_sha256)
    duplicate_hash_counts = [count for count in hashes.values() if count > 1]
    path_references = Counter(row.audio_path for row in rows if row.audio_path)
    duplicate_path_counts = [count for count in path_references.values() if count > 1]
    return {
        "duration_seconds": _numeric_statistics(
            [row.duration_seconds for row in readable_rows if row.duration_seconds is not None]
        ),
        "sample_rate_hz_distribution": _distribution(
            [row.sample_rate_hz for row in readable_rows if row.sample_rate_hz is not None]
        ),
        "channels_distribution": _distribution(
            [row.channels for row in readable_rows if row.channels is not None]
        ),
        "readable": len(readable_rows),
        "unique_matched_audio_files": len(unique_rows),
        "corrupted": sum(row.audio_status == "corrupted" for row in unique_rows),
        "format_mismatch": sum(row.audio_status == "format_mismatch" for row in unique_rows),
        "missing": sum(row.audio_match_status == "missing" for row in rows),
        "ambiguous": sum(row.audio_match_status == "ambiguous" for row in rows),
        "duplicate_audio_path_reference_groups": len(duplicate_path_counts),
        "duplicate_audio_path_reference_records": sum(duplicate_path_counts),
        "duplicate_audio_path_extra_occurrences": sum(count - 1 for count in duplicate_path_counts),
        "hash_enabled": hash_enabled,
        "duplicate_sha256_groups": len(duplicate_hash_counts) if hash_enabled else None,
        "duplicate_sha256_records": (sum(duplicate_hash_counts) if hash_enabled else None),
        "wav_extension_container_signatures": _container_signature_statistics(all_wav_files),
        "by_speaker": _group_audio_statistics(rows, "speaker_id"),
        "by_gender_folder": _group_audio_statistics(rows, "gender_folder"),
    }


def build_audit_summary(
    build: ManifestBuild,
    settings: DioulaDataSettings,
    split_proposal: dict[str, Any],
) -> dict[str, Any]:
    """Compute privacy-safe aggregate text, audio and structure statistics."""

    rows = build.rows
    matching = Counter(row.audio_match_status for row in rows)
    match_methods = Counter(row.audio_match_method for row in rows)
    return {
        "corpus": {
            "language": settings.language,
            "clips_json": len(build.inventory.files_by_kind["clips_json"]),
            "wav": len(build.inventory.files_by_kind["wav"]),
            "mp4_ignored": len(build.inventory.files_by_kind["mp4"]),
            "records": len(rows),
            "speaker_count_estimate": len(build.inventory.speakers),
            "gender_speaker_counts": build.inventory.gender_speaker_counts,
            "unexpected_structures": list(build.inventory.unexpected_structures),
        },
        "matching": {
            "matched": matching["matched"],
            "missing": matching["missing"],
            "ambiguous": matching["ambiguous"],
            "invalid": matching["invalid"],
            "methods": {
                "sequence": match_methods["sequence"],
                "exact_filename": match_methods["exact_filename"],
                "sentence_id": match_methods["sentence_id"],
                "clip_id": match_methods["clip_id"],
                "none": match_methods["none"],
            },
        },
        "text": {
            **_text_summary(rows),
            "tone_variant_files": _text_variant_statistics(build.inventory.speakers),
        },
        "audio": _audio_summary(
            rows,
            all_wav_files=build.inventory.files_by_kind["wav"],
            hash_enabled=settings.hash_audio,
        ),
        "governance": {
            "license_status": settings.license_status,
            "usage_scope": settings.usage_scope,
            "consent_status": "unknown",
            "signed_audio_urls_persisted": False,
            "raw_files_modified": False,
        },
        "split_proposal": split_proposal,
    }


def run_audit(settings: DioulaDataSettings) -> AuditResult:
    """Run the complete read-only audit and write only to the artifact root."""

    build = build_manifest(settings)
    split_proposal = propose_speaker_split(build.rows, settings.split)
    summary = build_audit_summary(build, settings, split_proposal)
    return AuditResult(build=build, summary=summary, split_proposal=split_proposal)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    temporary_path.replace(path)


def _audit_markdown(summary: dict[str, Any]) -> str:
    corpus = summary["corpus"]
    matching = summary["matching"]
    text = summary["text"]
    audio = summary["audio"]
    split = summary["split_proposal"]
    duration = audio["duration_seconds"]
    return f"""# Audit du corpus dioula

## Statut

Audit en lecture seule. La licence et le consentement restent à confirmer.
Aucune URL signée n'est conservée. Les identifiants de locuteurs sont
pseudonymisés et les chemins du manifeste sont relatifs au corpus.

## Inventaire

- `clips.json` : {corpus["clips_json"]}
- WAV : {corpus["wav"]}
- MP4 ignorés : {corpus["mp4_ignored"]}
- enregistrements : {corpus["records"]}
- locuteurs estimés : {corpus["speaker_count_estimate"]}

## Correspondance et audio

- matched : {matching["matched"]}
- missing : {matching["missing"]}
- ambiguous : {matching["ambiguous"]}
- invalid : {matching["invalid"]}
- corrompus : {audio["corrupted"]}
- fichiers sélectionnés avec format incohérent : {audio["format_mismatch"]}
- fichiers bruts suffixés WAV mais au format ISO Base Media :
  {audio["wav_extension_container_signatures"]["iso_base_media_mislabeled_wav"]}
- durée totale lisible (secondes) : {duration["total"]}
- références audio répétées (occurrences supplémentaires) :
  {audio["duplicate_audio_path_extra_occurrences"]}
- groupes de doublons SHA-256 : {audio["duplicate_sha256_groups"]}

## Texte

- textes vides : {text["empty_texts"]}
- groupes sentence_id dupliqués : {text["duplicate_sentence_id"]["groups"]}
- enregistrements contenant `↘` : {text["records_with_down_arrow"]}
- enregistrements contenant un retour chariot brut : {text["records_with_carriage_return"]}

## Proposition de split

Cette proposition n'est pas écrite dans le manifeste et exige une validation
humaine.

- train : {split["speaker_counts"]["train"]} locuteurs
- validation : {split["speaker_counts"]["validation"]} locuteurs
- test : {split["speaker_counts"]["test"]} locuteurs
- absence de fuite de locuteur : {split["leakage_free"]}

## Décisions humaines requises

- confirmer la licence et le consentement ;
- valider la définition structurelle d'un locuteur ;
- examiner les correspondances manquantes, ambiguës et les fichiers corrompus ;
- décider de la politique sur les tons sans les supprimer du texte brut ;
- approuver le split avant de renseigner la colonne `split`.
"""


def write_audit_outputs(result: AuditResult, settings: DioulaDataSettings) -> None:
    """Write all requested reports outside the Git repository."""

    write_manifest_outputs(result.build, settings)
    _atomic_json(
        settings.report_directory / "dioula_summary.json",
        result.summary,
    )
    _atomic_json(
        settings.report_directory / "dioula_split_proposal.json",
        result.split_proposal,
    )
    report_path = settings.report_directory / "dioula_audit.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = report_path.with_suffix(".md.tmp")
    temporary_path.write_text(_audit_markdown(result.summary), encoding="utf-8")
    temporary_path.replace(report_path)


def _print_safe_summary(summary: dict[str, Any]) -> None:
    corpus = summary["corpus"]
    matching = summary["matching"]
    text = summary["text"]
    audio = summary["audio"]
    split = summary["split_proposal"]
    print(f"clips_json={corpus['clips_json']}")
    print(f"wav={corpus['wav']}")
    print(f"records={corpus['records']}")
    print(f"speakers_estimated={corpus['speaker_count_estimate']}")
    print(f"duration_total_seconds={audio['duration_seconds']['total']}")
    print(f"matched={matching['matched']}")
    print(f"missing={matching['missing']}")
    print(f"ambiguous={matching['ambiguous']}")
    print(f"corrupted={audio['corrupted']}")
    print(f"empty_texts={text['empty_texts']}")
    print(f"duplicate_audio_groups={audio['duplicate_sha256_groups']}")
    print(f"duplicate_sentence_id_groups={text['duplicate_sentence_id']['groups']}")
    for split_name in ("train", "validation", "test"):
        print(
            f"split_{split_name}="
            f"{split['speaker_counts'][split_name]} speakers,"
            f"{split['record_counts'][split_name]} records"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auditer le corpus dioula local.")
    parser.add_argument("--config", required=True, help="Configuration YAML des données.")
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""

    args = _parse_args()
    try:
        settings = load_dioula_settings(args.config)
        result = run_audit(settings)
        write_audit_outputs(result, settings)
    except IvoireVoiceError as exc:
        print(f"ERREUR: {exc}")
        return 1
    _print_safe_summary(result.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
