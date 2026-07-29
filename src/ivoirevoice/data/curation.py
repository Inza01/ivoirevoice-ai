"""Deterministic curation of the audited Dioula draft manifest."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any

from ivoirevoice import __version__
from ivoirevoice.data.audio import sha256_file
from ivoirevoice.data.clips import normalize_transcription
from ivoirevoice.data.settings import DioulaDataSettings, load_dioula_settings
from ivoirevoice.exceptions import ConfigError, IvoireVoiceError

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SEQUENCE_AUDIO_PATTERN = re.compile(r"-(?P<sequence>\d{6})-\d{2}\.mp4$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class DraftRow:
    """Fields consumed from the Phase 3A draft manifest."""

    utterance_id: str
    sentence_id: str
    speaker_id: str
    gender_folder: str
    language: str
    text_raw: str
    text_normalized: str
    audio_path: str
    audio_filename: str
    audio_match_status: str
    duration_seconds: float | None
    sample_rate_hz: int | None
    channels: int | None
    file_size_bytes: int | None
    audio_sha256: str
    audio_status: str
    source_json: str
    record_index: int
    license_status: str


@dataclass(frozen=True, slots=True)
class PreparedRow:
    """Eligible row enriched with both text variants."""

    draft: DraftRow
    text_with_tones_nfc: str
    text_without_tones_nfc: str

    @property
    def transcription_key(self) -> tuple[str, str]:
        return self.text_with_tones_nfc, self.text_without_tones_nfc


@dataclass(frozen=True, slots=True)
class CuratedRow:
    """One canonical training-candidate row per selected audio."""

    utterance_id: str
    sentence_id: str
    speaker_id: str
    gender_folder: str
    language: str
    text_raw: str
    text_with_tones_nfc: str
    text_without_tones_nfc: str
    target_text_mvp: str
    audio_path: str
    duration_seconds: float
    sample_rate_hz: int
    channels: int
    file_size_bytes: int
    audio_sha256: str
    source_json: str
    license_status: str
    usage_scope: str
    eligibility_status: str
    exclusion_reason: str
    split: str


@dataclass(frozen=True, slots=True)
class CurationResult:
    """Curated rows and privacy-safe issue reports."""

    source_rows: tuple[DraftRow, ...]
    candidate_rows: tuple[CuratedRow, ...]
    duplicate_references: tuple[dict[str, Any], ...]
    duplicate_hashes: tuple[dict[str, Any], ...]
    conflicts: tuple[dict[str, Any], ...]
    quarantined: tuple[dict[str, Any], ...]
    text_report: dict[str, Any]
    recovery_plan: dict[str, Any]

    @property
    def conflict_group_count(self) -> int:
        """Count distinct path/hash transcription conflicts."""

        groups = {
            (
                str(row.get("reason", "")),
                (
                    str(row.get("audio_sha256", ""))
                    if row.get("reason") == "audio_hash_transcription_conflict"
                    else str(row.get("audio_path", ""))
                ),
            )
            for row in self.conflicts
        }
        return len(groups)


def _optional_float(value: str) -> float | None:
    try:
        return float(value) if value else None
    except ValueError:
        return None


def _optional_int(value: str) -> int | None:
    try:
        return int(value) if value else None
    except ValueError:
        return None


def _safe_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


def load_draft_manifest(path: Path) -> tuple[DraftRow, ...]:
    """Load the external draft manifest without retaining unneeded fields."""

    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {
                "utterance_id",
                "sentence_id",
                "speaker_id",
                "gender_folder",
                "language",
                "text_raw",
                "text_normalized",
                "audio_path",
                "audio_filename",
                "audio_match_status",
                "duration_seconds",
                "sample_rate_hz",
                "channels",
                "file_size_bytes",
                "audio_sha256",
                "audio_status",
                "source_json",
                "record_index",
                "license_status",
            }
            missing_columns = required.difference(reader.fieldnames or [])
            if missing_columns:
                raise ConfigError(
                    "Colonnes absentes du manifeste source : " + ", ".join(sorted(missing_columns))
                )
            rows = tuple(
                DraftRow(
                    utterance_id=row["utterance_id"],
                    sentence_id=row["sentence_id"],
                    speaker_id=row["speaker_id"],
                    gender_folder=row["gender_folder"],
                    language=row["language"],
                    text_raw=row["text_raw"],
                    text_normalized=row["text_normalized"],
                    audio_path=row["audio_path"],
                    audio_filename=row["audio_filename"],
                    audio_match_status=row["audio_match_status"],
                    duration_seconds=_optional_float(row["duration_seconds"]),
                    sample_rate_hz=_optional_int(row["sample_rate_hz"]),
                    channels=_optional_int(row["channels"]),
                    file_size_bytes=_optional_int(row["file_size_bytes"]),
                    audio_sha256=row["audio_sha256"],
                    audio_status=row["audio_status"],
                    source_json=row["source_json"],
                    record_index=int(row["record_index"]),
                    license_status=row["license_status"],
                )
                for row in reader
            )
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        raise ConfigError(f"Impossible de lire le manifeste source : {exc}") from exc
    return rows


def _load_no_tones_index(
    rows: tuple[DraftRow, ...],
    dataset_root: Path,
) -> tuple[dict[tuple[str, str], str], set[tuple[str, str]]]:
    index: dict[tuple[str, str], str] = {}
    conflicts: set[tuple[str, str]] = set()
    source_directories = {
        PurePosixPath(row.source_json).parent.as_posix()
        for row in rows
        if _safe_relative_path(row.source_json)
    }
    for relative_directory in sorted(source_directories):
        text_path = dataset_root / relative_directory / "text-no-tones"
        if not text_path.is_file():
            continue
        try:
            lines = text_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            continue
        for line in lines:
            audio_token, separator, text = line.partition(" ")
            if not separator or not audio_token:
                continue
            wav_filename = str(PurePosixPath(audio_token).with_suffix(".wav").name)
            key = relative_directory, wav_filename
            normalized_text = normalize_transcription(text)
            existing = index.get(key)
            if existing is not None and existing != normalized_text:
                conflicts.add(key)
            else:
                index[key] = normalized_text
    return index, conflicts


def _eligibility_reasons(row: DraftRow) -> list[str]:
    reasons: list[str] = []
    if row.audio_match_status != "matched":
        reasons.append("audio_not_matched")
    if row.audio_status != "readable":
        reasons.append("audio_not_readable")
    if not normalize_transcription(row.text_raw):
        reasons.append("empty_transcription")
    if not row.speaker_id:
        reasons.append("missing_speaker_id")
    if row.duration_seconds is None or row.duration_seconds <= 0:
        reasons.append("invalid_duration")
    if row.sample_rate_hz is None or row.sample_rate_hz <= 0:
        reasons.append("invalid_sample_rate")
    if row.channels is None or row.channels <= 0:
        reasons.append("invalid_channels")
    if row.file_size_bytes is None or row.file_size_bytes <= 0:
        reasons.append("invalid_file_size")
    if not SHA256_PATTERN.fullmatch(row.audio_sha256):
        reasons.append("missing_or_invalid_sha256")
    if not _safe_relative_path(row.audio_path):
        reasons.append("unsafe_or_missing_audio_path")
    if not _safe_relative_path(row.source_json):
        reasons.append("unsafe_source_json")
    return reasons


def _issue_record(
    row: DraftRow,
    *,
    reason: str,
    canonical_utterance_id: str = "",
) -> dict[str, Any]:
    return {
        "utterance_id": row.utterance_id,
        "sentence_id": row.sentence_id,
        "speaker_id": row.speaker_id,
        "audio_path": row.audio_path,
        "audio_sha256": row.audio_sha256,
        "source_json": row.source_json,
        "record_index": row.record_index,
        "reason": reason,
        "canonical_utterance_id": canonical_utterance_id,
    }


def _text_fingerprint(prepared: PreparedRow) -> str:
    combined = "\n".join(prepared.transcription_key)
    return sha256(combined.encode("utf-8")).hexdigest()


def _conflict_record(row: PreparedRow, reason: str) -> dict[str, Any]:
    record = _issue_record(row.draft, reason=reason)
    record["transcription_fingerprint"] = _text_fingerprint(row)
    return record


def _prepare_rows(
    rows: tuple[DraftRow, ...],
    settings: DioulaDataSettings,
) -> tuple[list[PreparedRow], list[dict[str, Any]]]:
    no_tones_index, no_tones_conflicts = _load_no_tones_index(
        rows,
        settings.dataset_root,
    )
    prepared: list[PreparedRow] = []
    quarantined: list[dict[str, Any]] = []
    for row in rows:
        reasons = _eligibility_reasons(row)
        source_directory = PurePosixPath(row.source_json).parent.as_posix()
        variant_key = source_directory, row.audio_filename
        no_tones = no_tones_index.get(variant_key, "")
        if row.audio_match_status == "matched":
            if variant_key in no_tones_conflicts:
                reasons.append("conflicting_text_without_tones")
            elif not no_tones:
                reasons.append("missing_text_without_tones")
        if reasons:
            quarantined.append(_issue_record(row, reason=";".join(reasons)))
            continue
        prepared.append(
            PreparedRow(
                draft=row,
                text_with_tones_nfc=normalize_transcription(row.text_raw),
                text_without_tones_nfc=no_tones,
            )
        )
    return prepared, quarantined


def _deduplicate(
    prepared_rows: list[PreparedRow],
) -> tuple[
    list[PreparedRow],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    by_path: dict[str, list[PreparedRow]] = defaultdict(list)
    for row in prepared_rows:
        by_path[row.draft.audio_path].append(row)

    path_canonical: list[PreparedRow] = []
    duplicate_references: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    for audio_path in sorted(by_path):
        group = sorted(
            by_path[audio_path],
            key=lambda row: (
                row.draft.source_json,
                row.draft.record_index,
                row.draft.utterance_id,
            ),
        )
        if len({row.transcription_key for row in group}) > 1:
            for row in group:
                conflicts.append(_conflict_record(row, "audio_path_transcription_conflict"))
                quarantined.append(
                    _issue_record(row.draft, reason="audio_path_transcription_conflict")
                )
            continue
        canonical = group[0]
        path_canonical.append(canonical)
        duplicate_references.extend(
            _issue_record(
                duplicate.draft,
                reason="duplicate_audio_path_reference",
                canonical_utterance_id=canonical.draft.utterance_id,
            )
            for duplicate in group[1:]
        )

    by_hash: dict[str, list[PreparedRow]] = defaultdict(list)
    for row in path_canonical:
        by_hash[row.draft.audio_sha256].append(row)

    selected: list[PreparedRow] = []
    duplicate_hashes: list[dict[str, Any]] = []
    for audio_hash in sorted(by_hash):
        group = sorted(
            by_hash[audio_hash],
            key=lambda row: (
                row.draft.audio_path,
                row.draft.source_json,
                row.draft.record_index,
            ),
        )
        if len({row.transcription_key for row in group}) > 1:
            for row in group:
                conflicts.append(_conflict_record(row, "audio_hash_transcription_conflict"))
                quarantined.append(
                    _issue_record(row.draft, reason="audio_hash_transcription_conflict")
                )
            continue
        canonical = group[0]
        selected.append(canonical)
        duplicate_hashes.extend(
            _issue_record(
                duplicate.draft,
                reason="duplicate_audio_sha256",
                canonical_utterance_id=canonical.draft.utterance_id,
            )
            for duplicate in group[1:]
        )
    return selected, duplicate_references, duplicate_hashes, conflicts, quarantined


def _curated_row(row: PreparedRow, settings: DioulaDataSettings) -> CuratedRow:
    draft = row.draft
    if (
        draft.duration_seconds is None
        or draft.sample_rate_hz is None
        or draft.channels is None
        or draft.file_size_bytes is None
    ):
        raise ConfigError("Une ligne sélectionnée a perdu ses métadonnées audio.")
    return CuratedRow(
        utterance_id=draft.utterance_id,
        sentence_id=draft.sentence_id,
        speaker_id=draft.speaker_id,
        gender_folder=draft.gender_folder,
        language=draft.language,
        text_raw=draft.text_raw,
        text_with_tones_nfc=row.text_with_tones_nfc,
        text_without_tones_nfc=row.text_without_tones_nfc,
        target_text_mvp=row.text_without_tones_nfc,
        audio_path=draft.audio_path,
        duration_seconds=draft.duration_seconds,
        sample_rate_hz=draft.sample_rate_hz,
        channels=draft.channels,
        file_size_bytes=draft.file_size_bytes,
        audio_sha256=draft.audio_sha256,
        source_json=draft.source_json,
        license_status=settings.license_status,
        usage_scope=settings.usage_scope,
        eligibility_status="eligible",
        exclusion_reason="",
        split="",
    )


def _character_vocabulary(texts: list[str]) -> list[dict[str, Any]]:
    counts = Counter(character for text in texts for character in text)
    return [
        {
            "character": character,
            "codepoint": f"U+{ord(character):04X}",
            "name": unicodedata.name(character, "UNKNOWN"),
            "count": count,
        }
        for character, count in sorted(
            counts.items(),
            key=lambda item: (-item[1], ord(item[0])),
        )
    ]


def _text_report(rows: tuple[CuratedRow, ...]) -> dict[str, Any]:
    with_tones = [row.text_with_tones_nfc for row in rows]
    without_tones = [row.text_without_tones_nfc for row in rows]
    raw_to_tones = [row for row in rows if row.text_raw != row.text_with_tones_nfc]
    tones_to_without = [
        row for row in rows if row.text_with_tones_nfc != row.text_without_tones_nfc
    ]
    row_count = len(rows)
    return {
        "row_count": row_count,
        "with_tones": {
            "character_count_unique": len(set("".join(with_tones))),
            "average_length": statistics.fmean(map(len, with_tones)) if rows else 0.0,
            "vocabulary": _character_vocabulary(with_tones),
        },
        "without_tones": {
            "character_count_unique": len(set("".join(without_tones))),
            "average_length": (statistics.fmean(map(len, without_tones)) if rows else 0.0),
            "vocabulary": _character_vocabulary(without_tones),
        },
        "normalization_changes": {
            "raw_to_with_tones_nfc": {
                "rows": len(raw_to_tones),
                "rate": len(raw_to_tones) / row_count if row_count else 0.0,
                "anonymized_example_ids": [row.utterance_id for row in raw_to_tones[:5]],
            },
            "with_tones_to_without_tones_nfc": {
                "rows": len(tones_to_without),
                "rate": len(tones_to_without) / row_count if row_count else 0.0,
                "anonymized_example_ids": [row.utterance_id for row in tones_to_without[:5]],
            },
        },
        "down_arrow": {
            "rows_with_symbol": sum("↘" in row.target_text_mvp for row in rows),
            "occurrences": sum(row.target_text_mvp.count("↘") for row in rows),
        },
        "target_text_mvp": "text_without_tones_nfc",
    }


def _recovery_plan(
    rows: tuple[DraftRow, ...],
    settings: DioulaDataSettings,
) -> dict[str, Any]:
    missing_rows = [row for row in rows if row.audio_match_status == "missing"]
    planned: list[dict[str, Any]] = []
    directory_indexes: dict[
        str,
        tuple[dict[str, list[Path]], dict[int, list[Path]]],
    ] = {}
    sequence_cache: dict[str, list[int | None]] = {}
    seen_sources: dict[str, str] = {}
    for row in missing_rows:
        relative_directory = PurePosixPath(row.source_json).parent.as_posix()
        if not _safe_relative_path(row.source_json):
            continue
        if relative_directory not in directory_indexes:
            files_by_name: dict[str, list[Path]] = defaultdict(list)
            files_by_sequence: dict[int, list[Path]] = defaultdict(list)
            speaker_directory = settings.dataset_root / relative_directory
            if speaker_directory.is_dir():
                for path in speaker_directory.rglob("*"):
                    if path.is_file():
                        files_by_name[path.name].append(path)
                        sequence_match = SEQUENCE_AUDIO_PATTERN.search(path.name)
                        if path.parent.name.lower() == "mp4" and sequence_match is not None:
                            files_by_sequence[int(sequence_match.group("sequence"))].append(path)
            directory_indexes[relative_directory] = (
                files_by_name,
                files_by_sequence,
            )
        if row.source_json not in sequence_cache:
            sequence_cache[row.source_json] = _load_record_sequences(
                settings.dataset_root / Path(*PurePosixPath(row.source_json).parts)
            )
        files_by_name, files_by_sequence = directory_indexes[relative_directory]
        candidates = files_by_name.get(row.audio_filename, [])
        source_candidates = [
            path for path in candidates if path.is_file() and _is_iso_base_media(path)
        ]
        match_method = "exact_filename"
        if not source_candidates:
            sequences = sequence_cache.get(row.source_json, [])
            sequence = (
                sequences[row.record_index] if 0 <= row.record_index < len(sequences) else None
            )
            if sequence is not None:
                source_candidates = files_by_sequence.get(sequence, [])
                match_method = "sequence"
        if len(source_candidates) == 1:
            source_path = source_candidates[0].relative_to(settings.dataset_root).as_posix()
            if source_path in seen_sources:
                status = "duplicate_source_reference"
                canonical_utterance_id = seen_sources[source_path]
            else:
                status = "source_found"
                canonical_utterance_id = ""
                seen_sources[source_path] = row.utterance_id
        elif len(source_candidates) > 1:
            status = "ambiguous_source"
            source_path = ""
            canonical_utterance_id = ""
        else:
            status = "source_missing"
            source_path = ""
            canonical_utterance_id = ""
            match_method = "none"
        planned.append(
            {
                "utterance_id": row.utterance_id,
                "source_path": source_path,
                "output_filename": f"{row.utterance_id}.wav",
                "status": status,
                "match_method": match_method,
                "canonical_utterance_id": canonical_utterance_id,
            }
        )
    status_counts = Counter(item["status"] for item in planned)
    return {
        "enabled": settings.curation.recover_missing_audio,
        "execution_performed": False,
        "missing_rows": len(missing_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "ffmpeg_command_template": (
            "ffmpeg -nostdin -n -i <source> -ar 16000 -ac 1 -c:a pcm_s16le <external_output.wav>"
        ),
        "proposed_rows": planned,
    }


def _load_record_sequences(path: Path) -> list[int | None]:
    try:
        with path.open(encoding="utf-8") as stream:
            root: object = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(root, list):
        return []
    sequences: list[int | None] = []
    for record in root:
        if not isinstance(record, dict):
            sequences.append(None)
            continue
        sentence = record.get("sentence")
        sequence_value: object = None
        if isinstance(sentence, dict):
            sequence_value = sentence.get("sequence")
        if sequence_value is None:
            sequence_value = record.get("sequence")
        sequences.append(
            sequence_value
            if isinstance(sequence_value, int) and not isinstance(sequence_value, bool)
            else None
        )
    return sequences


def _is_iso_base_media(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            header = stream.read(8)
    except OSError:
        return False
    return len(header) >= 8 and header[4:8] == b"ftyp"


def curate_manifest(settings: DioulaDataSettings) -> CurationResult:
    """Apply eligibility, text-variant and two-level deduplication rules."""

    source_rows = load_draft_manifest(settings.source_manifest_path)
    prepared, initial_quarantine = _prepare_rows(source_rows, settings)
    selected, duplicate_refs, duplicate_hashes, conflicts, conflict_quarantine = _deduplicate(
        prepared
    )
    candidate_rows = tuple(
        sorted(
            (_curated_row(row, settings) for row in selected),
            key=lambda row: row.audio_path,
        )
    )
    return CurationResult(
        source_rows=source_rows,
        candidate_rows=candidate_rows,
        duplicate_references=tuple(duplicate_refs),
        duplicate_hashes=tuple(duplicate_hashes),
        conflicts=tuple(conflicts),
        quarantined=tuple(initial_quarantine + conflict_quarantine),
        text_report=_text_report(candidate_rows),
        recovery_plan=_recovery_plan(source_rows, settings),
    )


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


def _curation_markdown(result: CurationResult, settings: DioulaDataSettings) -> str:
    duration = sum(row.duration_seconds for row in result.candidate_rows)
    return f"""# Curation du candidat dioula

## Statut

- licence : `{settings.license_status}`
- périmètre : `{settings.usage_scope}`
- split : en attente de validation humaine
- récupération des audios manquants : non exécutée

## Résultat

- lignes source : {len(result.source_rows)}
- lignes candidates : {len(result.candidate_rows)}
- durée candidate (secondes) : {duration}
- références audio dédupliquées : {len(result.duplicate_references)}
- doublons SHA-256 dédupliqués : {len(result.duplicate_hashes)}
- groupes de conflits de transcription : {result.conflict_group_count}
- lignes impliquées dans un conflit : {len(result.conflicts)}
- lignes en quarantaine : {len(result.quarantined)}

Le candidat contient une seule ligne par audio et conserve `split` vide.
`target_text_mvp` utilise la variante NFC sans tons. Cette décision est
technique et limitée au premier MVP ; le texte brut et la variante avec tons
restent conservés.
"""


def write_curation_outputs(
    result: CurationResult,
    settings: DioulaDataSettings,
    *,
    config_reference: str,
) -> None:
    """Write the candidate, provenance metadata and issue reports externally."""

    candidate_dicts = [asdict(row) for row in result.candidate_rows]
    _atomic_csv(
        settings.candidate_manifest_path,
        candidate_dicts,
        [field.name for field in fields(CuratedRow)],
    )
    report_directory = settings.curation_report_directory
    issue_fields = [
        "utterance_id",
        "sentence_id",
        "speaker_id",
        "audio_path",
        "audio_sha256",
        "source_json",
        "record_index",
        "reason",
        "canonical_utterance_id",
        "transcription_fingerprint",
    ]
    for filename, rows in (
        ("duplicate_audio_references.csv", result.duplicate_references),
        ("duplicate_audio_hashes.csv", result.duplicate_hashes),
        ("conflicting_transcriptions.csv", result.conflicts),
        ("quarantined_rows.csv", result.quarantined),
    ):
        _atomic_csv(report_directory / filename, list(rows), issue_fields)
    _atomic_json(report_directory / "text_variants_report.json", result.text_report)
    _atomic_json(
        report_directory / "missing_audio_recovery_plan.json",
        result.recovery_plan,
    )

    source_hash = sha256_file(settings.source_manifest_path)
    candidate_hash = sha256_file(settings.candidate_manifest_path)
    included_duration = sum(row.duration_seconds for row in result.candidate_rows)
    metadata = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "pipeline_version": __version__,
        "source_manifest_sha256": source_hash,
        "candidate_manifest_sha256": candidate_hash,
        "configuration": config_reference,
        "seed": settings.split.seed,
        "source_rows": len(result.source_rows),
        "included_rows": len(result.candidate_rows),
        "excluded_rows": len(result.source_rows) - len(result.candidate_rows),
        "included_duration_seconds": included_duration,
        "conflict_groups": result.conflict_group_count,
        "conflicting_rows": len(result.conflicts),
        "normalization_rules": [
            "Unicode NFC",
            "trim leading and trailing whitespace",
            "collapse whitespace and line breaks",
            "preserve text_raw",
            "target_text_mvp=text_without_tones_nfc",
        ],
        "deduplication_rules": [
            "one canonical row per audio_path",
            "one canonical path per audio_sha256",
            "quarantine conflicting normalized transcriptions",
            "do not deduplicate sentence_id across distinct audio",
        ],
        "license_status": settings.license_status,
        "usage_scope": settings.usage_scope,
        "split_status": "pending_human_validation",
        "recovery_executed": False,
    }
    _atomic_json(settings.candidate_metadata_path, metadata)

    report_path = report_directory / "curation_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = report_path.with_suffix(".md.tmp")
    temporary_path.write_text(_curation_markdown(result, settings), encoding="utf-8")
    temporary_path.replace(report_path)


def _safe_config_reference(config_path: str) -> str:
    path = PurePosixPath(config_path)
    if path.is_absolute() or ".." in path.parts:
        return path.name
    return path.as_posix()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curater le candidat dioula local.")
    parser.add_argument("--config", required=True, help="Configuration YAML des données.")
    return parser.parse_args()


def main() -> int:
    """CLI entry point with aggregate-only output."""

    args = _parse_args()
    try:
        settings = load_dioula_settings(args.config)
        result = curate_manifest(settings)
        write_curation_outputs(
            result,
            settings,
            config_reference=_safe_config_reference(args.config),
        )
    except IvoireVoiceError as exc:
        print(f"ERREUR: {exc}")
        return 1

    print(f"candidate_rows={len(result.candidate_rows)}")
    print(f"unique_audio={len(result.candidate_rows)}")
    print(f"duration_total_seconds={sum(row.duration_seconds for row in result.candidate_rows)}")
    print(f"removed_audio_references={len(result.duplicate_references)}")
    print(f"removed_duplicate_hashes={len(result.duplicate_hashes)}")
    print(f"conflict_groups={result.conflict_group_count}")
    print(f"conflicting_rows={len(result.conflicts)}")
    print(f"quarantined_rows={len(result.quarantined)}")
    print("split=pending_human_validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
