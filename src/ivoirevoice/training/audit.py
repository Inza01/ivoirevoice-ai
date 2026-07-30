"""Phase 4B data audit, representative selection and leakage checks."""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, cast

from ivoirevoice.data.audio import sha256_file
from ivoirevoice.exceptions import ConfigError, IvoireVoiceError
from ivoirevoice.training.settings import SmokeSettings, load_smoke_settings

VALIDATION_STATUSES = (
    "correct",
    "texte partiellement incorrect",
    "mauvais alignement",
    "audio inutilisable",
    "à vérifier",
)
ANNOTATIONS_FILENAME = "manual_validation_annotations.json"
MANUAL_REPORT_FILENAME = "manual_validation_report.md"
NORMALIZATION_REPORT_FILENAME = "text_normalization_analysis.md"
SPLIT_REPORT_FILENAME = "split_integrity_report.json"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_COLUMNS = {
    "utterance_id",
    "speaker_id",
    "gender_folder",
    "language",
    "text_raw",
    "text_without_tones_nfc",
    "target_text_mvp",
    "audio_path",
    "duration_seconds",
    "sample_rate_hz",
    "channels",
    "audio_sha256",
    "split",
    "usage_scope",
}


@dataclass(frozen=True, slots=True)
class ManifestRow:
    """Relevant fields from one private frozen-manifest row."""

    utterance_id: str
    speaker_id: str
    gender_folder: str
    language: str
    text_raw: str
    text_no_tones: str
    target_text: str
    audio_path: str
    duration_seconds: float
    sample_rate_hz: int
    channels: int
    audio_sha256: str
    split: str
    usage_scope: str


@dataclass(frozen=True, slots=True)
class AuditedDataset:
    """Validated manifest content and immutable provenance."""

    rows: tuple[ManifestRow, ...]
    manifest_sha256: str
    dataset_version: str


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ConfigError(f"{name} doit être un objet JSON.")
    return cast(dict[str, Any], dict(value))


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            payload: object = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Impossible de lire {description} : {exc}") from exc
    return _mapping(payload, description)


def _safe_audio_path(value: str) -> bool:
    path = PurePosixPath(value)
    return (
        bool(value)
        and not path.is_absolute()
        and ".." not in path.parts
        and "://" not in value
        and "\\" not in value
    )


def _parse_manifest_row(raw: dict[str, str], line_number: int) -> ManifestRow:
    prefix = f"Manifeste ligne {line_number}"
    try:
        duration = float(raw["duration_seconds"])
        sample_rate = int(raw["sample_rate_hz"])
        channels = int(raw["channels"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"{prefix} : métadonnées audio invalides.") from exc
    if duration <= 0 or sample_rate <= 0 or channels <= 0:
        raise ConfigError(f"{prefix} : valeurs audio non positives.")
    audio_path = raw.get("audio_path", "")
    if not _safe_audio_path(audio_path):
        raise ConfigError(f"{prefix} : audio_path non sûr.")
    audio_hash = raw.get("audio_sha256", "")
    if not SHA256_PATTERN.fullmatch(audio_hash):
        raise ConfigError(f"{prefix} : audio_sha256 invalide.")
    split = raw.get("split", "")
    if split not in {"train", "validation", "test"}:
        raise ConfigError(f"{prefix} : split inconnu.")
    utterance_id = raw.get("utterance_id", "")
    speaker_id = raw.get("speaker_id", "")
    if not utterance_id or not speaker_id:
        raise ConfigError(f"{prefix} : identifiant vide.")
    return ManifestRow(
        utterance_id=utterance_id,
        speaker_id=speaker_id,
        gender_folder=raw.get("gender_folder", ""),
        language=raw.get("language", ""),
        text_raw=raw.get("text_raw", ""),
        text_no_tones=raw.get("text_without_tones_nfc", ""),
        target_text=raw.get("target_text_mvp", ""),
        audio_path=audio_path,
        duration_seconds=duration,
        sample_rate_hz=sample_rate,
        channels=channels,
        audio_sha256=audio_hash,
        split=split,
        usage_scope=raw.get("usage_scope", ""),
    )


def load_audited_dataset(settings: SmokeSettings) -> AuditedDataset:
    """Validate governance, provenance and the complete frozen manifest."""

    metadata = _load_json(settings.dataset_metadata_path, "les métadonnées v0.1")
    manifest_hash = sha256_file(settings.manifest_path)
    if metadata.get("manifest_sha256") != manifest_hash:
        raise ConfigError("Le manifeste ne correspond plus aux métadonnées v0.1.")
    governance = {
        "publication_allowed": False,
        "model_derivative_publication_allowed": False,
        "usage_scope": "local_research_only",
    }
    for field, expected in governance.items():
        if metadata.get(field) != expected:
            raise ConfigError(f"La gouvernance exige {field}={expected!r}.")
    dataset_version = metadata.get("dataset_version")
    if not isinstance(dataset_version, str) or not dataset_version:
        raise ConfigError("dataset_version est absent des métadonnées.")

    try:
        with settings.manifest_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fieldnames = set(reader.fieldnames or ())
            missing = sorted(REQUIRED_COLUMNS - fieldnames)
            if missing:
                raise ConfigError(f"Colonnes manquantes dans le manifeste : {missing}.")
            rows = tuple(_parse_manifest_row(row, index) for index, row in enumerate(reader, 2))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ConfigError(f"Impossible de lire le manifeste gelé : {exc}") from exc
    if not rows:
        raise ConfigError("Le manifeste gelé est vide.")

    utterance_ids = [row.utterance_id for row in rows]
    if len(utterance_ids) != len(set(utterance_ids)):
        raise ConfigError("Le manifeste contient des utterance_id dupliqués.")
    if any(row.language != "dyu" for row in rows):
        raise ConfigError("Le manifeste Phase 4B doit contenir uniquement language=dyu.")
    if any(row.usage_scope != "local_research_only" for row in rows):
        raise ConfigError("Toutes les lignes doivent rester local_research_only.")

    expected_counts = metadata.get("audio_count_by_split")
    actual_counts = Counter(row.split for row in rows)
    if not isinstance(expected_counts, Mapping) or any(
        expected_counts.get(split) != actual_counts[split]
        for split in ("train", "validation", "test")
    ):
        raise ConfigError("Les volumes par split ne correspondent plus aux métadonnées.")
    return AuditedDataset(
        rows=rows,
        manifest_sha256=manifest_hash,
        dataset_version=dataset_version,
    )


def _stable_rank(value: str, seed: int) -> str:
    return sha256(f"{seed}:{value}".encode()).hexdigest()


def select_representative_train_rows(
    rows: Sequence[ManifestRow],
    count: int,
    seed: int,
) -> tuple[ManifestRow, ...]:
    """Select a deterministic speaker-balanced duration/text-length sample."""

    if not 10 <= count <= 20:
        raise ConfigError("La sélection manuelle doit contenir entre 10 et 20 audios.")
    train_rows = [row for row in rows if row.split == "train"]
    by_speaker: dict[str, list[ManifestRow]] = defaultdict(list)
    speaker_gender: dict[str, str] = {}
    for row in train_rows:
        by_speaker[row.speaker_id].append(row)
        speaker_gender[row.speaker_id] = row.gender_folder
    if len(by_speaker) < 2:
        raise ConfigError("Le train doit contenir plusieurs locuteurs.")

    speakers_by_gender: dict[str, list[str]] = defaultdict(list)
    for speaker, gender in speaker_gender.items():
        speakers_by_gender[gender].append(speaker)
    for speakers in speakers_by_gender.values():
        speakers.sort(key=lambda value: _stable_rank(value, seed))
    gender_names = sorted(speakers_by_gender)
    interleaved_speakers: list[str] = []
    maximum = max(len(values) for values in speakers_by_gender.values())
    for index in range(maximum):
        for gender in gender_names:
            speakers = speakers_by_gender[gender]
            if index < len(speakers):
                interleaved_speakers.append(speakers[index])

    assignments = [
        interleaved_speakers[index % len(interleaved_speakers)] for index in range(count)
    ]
    occurrences: Counter[str] = Counter()
    selected: list[ManifestRow] = []
    target_quantiles = (0.0, 1.0, 0.5, 0.25, 0.75)
    for index, speaker in enumerate(assignments):
        candidates = sorted(
            by_speaker[speaker],
            key=lambda row: (
                row.duration_seconds,
                len(row.target_text.split()),
                _stable_rank(row.utterance_id, seed),
            ),
        )
        target = target_quantiles[index % len(target_quantiles)]
        base_index = round(target * (len(candidates) - 1))
        offset = occurrences[speaker]
        candidate_index = min(base_index + offset, len(candidates) - 1)
        candidate = candidates[candidate_index]
        if candidate in selected:
            candidate = next(row for row in candidates if row not in selected)
        selected.append(candidate)
        occurrences[speaker] += 1
    return tuple(selected)


def selection_sha256(rows: Sequence[ManifestRow]) -> str:
    """Hash the ordered private selection without exposing its identifiers."""

    payload = "\n".join(row.utterance_id for row in rows).encode()
    return sha256(payload).hexdigest()


def _default_annotations(
    dataset: AuditedDataset,
    selected: Sequence[ManifestRow],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "manifest_sha256": dataset.manifest_sha256,
        "selection_sha256": selection_sha256(selected),
        "statuses": {
            row.utterance_id: {
                "status": "à vérifier",
                "anomaly": "",
                "reviewed_at_utc": None,
            }
            for row in selected
        },
    }


def load_or_initialize_annotations(
    settings: SmokeSettings,
    dataset: AuditedDataset,
    selected: Sequence[ManifestRow],
) -> dict[str, Any]:
    """Preserve annotations only while manifest and deterministic selection match."""

    settings.reports_root.mkdir(parents=True, exist_ok=True)
    path = settings.reports_root / ANNOTATIONS_FILENAME
    expected = _default_annotations(dataset, selected)
    if path.is_file():
        existing = _load_json(path, "les annotations manuelles")
        if (
            existing.get("schema_version") == 1
            and existing.get("manifest_sha256") == dataset.manifest_sha256
            and existing.get("selection_sha256") == selection_sha256(selected)
        ):
            statuses = existing.get("statuses")
            if isinstance(statuses, Mapping) and set(statuses) == {
                row.utterance_id for row in selected
            }:
                return existing
    path.write_text(
        json.dumps(expected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return expected


def save_annotations(settings: SmokeSettings, annotations: Mapping[str, Any]) -> None:
    """Persist private human labels in the ignored local report directory."""

    path = settings.reports_root / ANNOTATIONS_FILENAME
    path.write_text(
        json.dumps(dict(annotations), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _status_entry(annotations: Mapping[str, Any], utterance_id: str) -> dict[str, Any]:
    statuses = annotations.get("statuses")
    if not isinstance(statuses, Mapping):
        return {"status": "à vérifier", "anomaly": ""}
    value = statuses.get(utterance_id)
    return dict(value) if isinstance(value, Mapping) else {"status": "à vérifier", "anomaly": ""}


def validated_rows(
    settings: SmokeSettings,
    selected: Sequence[ManifestRow],
    annotations: Mapping[str, Any],
) -> tuple[ManifestRow, ...]:
    """Return human-confirmed rows or stop before any training begins."""

    correct = tuple(
        row
        for row in selected
        if _status_entry(annotations, row.utterance_id).get("status") == "correct"
    )
    if len(correct) < settings.minimum_correct_samples:
        raise ConfigError(
            "Smoke test verrouillé : "
            f"{len(correct)}/{settings.minimum_correct_samples} audios ont été validés "
            "'correct'. Lancez d'abord make review-dioula-training."
        )
    return correct


def _automatic_anomalies(row: ManifestRow, settings: SmokeSettings) -> list[str]:
    anomalies: list[str] = []
    audio_file = (settings.dataset_root / row.audio_path).resolve()
    try:
        audio_file.relative_to(settings.dataset_root)
    except ValueError:
        anomalies.append("chemin audio hors racine")
    if not audio_file.is_file():
        anomalies.append("fichier audio absent")
    if row.sample_rate_hz != 16_000:
        anomalies.append(f"fréquence={row.sample_rate_hz} Hz")
    if row.channels != 1:
        anomalies.append(f"canaux={row.channels}")
    if not row.text_raw.strip():
        anomalies.append("text_raw vide")
    if not row.text_no_tones.strip():
        anomalies.append("text_no_tones vide")
    if not row.target_text.strip():
        anomalies.append("target_text_mvp vide")
    return anomalies


def _markdown_cell(value: object) -> str:
    return str(value).replace("\r", " ").replace("\n", " ").replace("|", "\\|").strip()


def write_manual_report(
    settings: SmokeSettings,
    dataset: AuditedDataset,
    selected: Sequence[ManifestRow],
    annotations: Mapping[str, Any],
) -> Path:
    """Render the requested private report without real names or private paths."""

    rows: list[str] = []
    for row in selected:
        entry = _status_entry(annotations, row.utterance_id)
        status = entry.get("status", "à vérifier")
        if status not in VALIDATION_STATUSES:
            status = "à vérifier"
        automatic = _automatic_anomalies(row, settings)
        manual_anomaly = str(entry.get("anomaly", "")).strip()
        anomaly = "; ".join([*automatic, *([manual_anomaly] if manual_anomaly else [])])
        rows.append(
            "| "
            + " | ".join(
                (
                    _markdown_cell(row.utterance_id),
                    _markdown_cell(row.speaker_id),
                    f"{row.duration_seconds:.3f}",
                    _markdown_cell(row.text_raw),
                    _markdown_cell(row.text_no_tones),
                    _markdown_cell(status),
                    _markdown_cell(anomaly or "aucune détectée automatiquement"),
                )
            )
            + " |"
        )
    status_counts = Counter(
        str(_status_entry(annotations, row.utterance_id).get("status", "à vérifier"))
        for row in selected
    )
    content = "\n".join(
        (
            "# Validation manuelle assistée — échantillon train dioula",
            "",
            "> **Rapport privé local.** Il contient des transcriptions du dataset. "
            "Il est ignoré par Git et ne doit pas être publié.",
            "",
            f"- Dataset : `{dataset.dataset_version}`",
            f"- Empreinte manifeste : `{dataset.manifest_sha256}`",
            f"- Taille de l'échantillon : {len(selected)}",
            f"- Locuteurs anonymisés : {len({row.speaker_id for row in selected})}",
            "- Répartition dossiers de genre : "
            f"{dict(Counter(row.gender_folder for row in selected))}",
            f"- Statuts : {dict(status_counts)}",
            "",
            "| audio_id anonymisé | speaker_id anonymisé | durée (s) | text_raw | "
            "text_no_tones | statut | anomalie éventuelle |",
            "|---|---|---:|---|---|---|---|",
            *rows,
            "",
            "Les statuts auditifs sont saisis exclusivement avec l'outil local. "
            "« à vérifier » ne constitue pas une validation.",
            "",
        )
    )
    path = settings.reports_root / MANUAL_REPORT_FILENAME
    path.write_text(content, encoding="utf-8")
    return path


def _contains_tone(text: str) -> bool:
    decomposed = unicodedata.normalize("NFD", text)
    tone_marks = {"\u0300", "\u0301", "\u0302", "\u0304", "\u030c"}
    return any(character in tone_marks for character in decomposed)


def analyze_text(rows: Sequence[ManifestRow]) -> dict[str, Any]:
    """Compute aggregate Unicode and normalization diagnostics."""

    character_counts: Counter[str] = Counter()
    raw_no_tone_differences = 0
    falling_marker_rows = 0
    stats: Counter[str] = Counter()
    for row in rows:
        raw = row.text_raw
        no_tones = row.text_no_tones
        character_counts.update(raw)
        raw_no_tone_differences += raw != no_tones
        falling_marker_rows += "↘" in raw or "↘" in no_tones
        stats["raw_not_nfc"] += raw != unicodedata.normalize("NFC", raw)
        stats["no_tones_not_nfc"] += no_tones != unicodedata.normalize("NFC", no_tones)
        stats["raw_empty"] += not raw.strip()
        stats["no_tones_empty"] += not no_tones.strip()
        stats["target_empty"] += not row.target_text.strip()
        stats["raw_with_tones"] += _contains_tone(raw)
        stats["raw_with_apostrophe"] += "'" in raw or "’" in raw
        stats["raw_with_punctuation"] += any(
            unicodedata.category(character).startswith("P") for character in raw
        )
        stats["raw_with_digits"] += any(character.isdigit() for character in raw)
        stats["raw_with_multiple_spaces"] += bool(re.search(r"\s{2,}", raw))
    rare = [
        {
            "character": character,
            "codepoint": f"U+{ord(character):04X}",
            "unicode_name": unicodedata.name(character, "UNKNOWN"),
            "count": count,
        }
        for character, count in sorted(character_counts.items(), key=lambda item: ord(item[0]))
        if count < 5 and not character.isspace()
    ]
    return {
        "row_count": len(rows),
        "different_raw_vs_no_tones": raw_no_tone_differences,
        "identical_raw_vs_no_tones": len(rows) - raw_no_tone_differences,
        "falling_marker_rows": falling_marker_rows,
        **stats,
        "rare_characters_under_5_occurrences": rare,
    }


def write_normalization_report(
    settings: SmokeSettings,
    dataset: AuditedDataset,
) -> Path:
    """Write only aggregate text statistics to the shareable analysis report."""

    train_rows = tuple(row for row in dataset.rows if row.split == "train")
    stats = analyze_text(train_rows)
    total = cast(int, stats["row_count"])
    difference = cast(int, stats["different_raw_vs_no_tones"])
    rare = cast(list[dict[str, Any]], stats["rare_characters_under_5_occurrences"])
    rare_table = [
        f"| `{item['codepoint']}` | {item['unicode_name']} | {item['count']} |" for item in rare
    ]
    if not rare_table:
        rare_table.append("| — | aucun | 0 |")
    content = "\n".join(
        (
            "# Analyse de normalisation textuelle — train dioula v0.1",
            "",
            "Ce rapport est agrégé : aucun chemin, nom réel ou texte du dataset n'y figure.",
            "",
            "## Résultats",
            "",
            "| Contrôle | Nombre de lignes |",
            "|---|---:|",
            f"| Lignes train analysées | {total} |",
            f"| text_raw différent de text_no_tones | {difference} "
            f"({difference / total:.2%}) |",
            f"| text_raw identique à text_no_tones | {stats['identical_raw_vs_no_tones']} |",
            f"| text_raw non NFC | {stats['raw_not_nfc']} |",
            f"| text_no_tones non NFC | {stats['no_tones_not_nfc']} |",
            f"| text_raw avec marques de ton | {stats['raw_with_tones']} |",
            f"| text_raw avec apostrophe droite ou typographique | "
            f"{stats['raw_with_apostrophe']} |",
            f"| text_raw avec ponctuation Unicode | {stats['raw_with_punctuation']} |",
            f"| text_raw avec chiffres | {stats['raw_with_digits']} |",
            f"| text_raw avec espaces multiples | {stats['raw_with_multiple_spaces']} |",
            f"| text_raw vide | {stats['raw_empty']} |",
            f"| text_no_tones vide | {stats['no_tones_empty']} |",
            f"| target_text_mvp vide | {stats['target_empty']} |",
            f"| Lignes avec marque d'intonation descendante | {stats['falling_marker_rows']} |",
            "",
            "## Caractères rares dans text_raw",
            "",
            "| Codepoint | Nom Unicode | Occurrences |",
            "|---|---|---:|",
            *rare_table,
            "",
            "## Colonne canonique proposée",
            "",
            "**`target_text_mvp`** est retenue pour le smoke-overfit. Elle conserve la variante "
            "NFC sans tons et retire la marque prosodique `↘`, qui ne correspond pas à une "
            "unité lexicale à prédire. Ce choix réduit la sparsité orthographique sur seulement "
            "10 à 20 exemples et reste cohérent avec l'évaluation des baselines.",
            "",
            "`text_raw`, `text_with_tones_nfc`, `text_without_tones_nfc` et "
            "`target_text_mvp` restent toutes conservées dans le manifeste gelé. "
            "Aucune variante n'est supprimée ou réécrite.",
            "",
        )
    )
    path = settings.reports_root / NORMALIZATION_REPORT_FILENAME
    path.write_text(content, encoding="utf-8")
    return path


def _pilot_ids(paths: Iterable[Path]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        try:
            with path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if "utterance_id" not in (reader.fieldnames or ()):
                    raise ConfigError(f"Le pilote {path.name} ne contient pas utterance_id.")
                ids.update(row["utterance_id"] for row in reader if row.get("utterance_id"))
        except (OSError, UnicodeError, csv.Error) as exc:
            raise ConfigError(
                f"Impossible de lire les résultats pilote {path.name} : {exc}"
            ) from exc
    return ids


def build_split_integrity_report(
    dataset: AuditedDataset,
    selected: Sequence[ManifestRow],
    pilot_ids: set[str],
) -> dict[str, Any]:
    """Prove speaker/hash separation and isolate pilot and smoke selections."""

    speaker_splits: dict[str, set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    id_to_row = {row.utterance_id: row for row in dataset.rows}
    for row in dataset.rows:
        speaker_splits[row.speaker_id].add(row.split)
        hash_splits[row.audio_sha256].add(row.split)
    speaker_leaks = sum(len(splits) > 1 for splits in speaker_splits.values())
    hash_leaks = sum(len(splits) > 1 for splits in hash_splits.values())
    missing_pilot = pilot_ids - id_to_row.keys()
    pilot_non_test = {
        utterance_id
        for utterance_id in pilot_ids
        if utterance_id in id_to_row and id_to_row[utterance_id].split != "test"
    }
    test_ids = {row.utterance_id for row in dataset.rows if row.split == "test"}
    test_hashes = {row.audio_sha256 for row in dataset.rows if row.split == "test"}
    smoke_non_train = sum(row.split != "train" for row in selected)
    smoke_test_id_overlap = sum(row.utterance_id in test_ids for row in selected)
    smoke_test_hash_overlap = sum(row.audio_sha256 in test_hashes for row in selected)
    smoke_pilot_overlap = sum(row.utterance_id in pilot_ids for row in selected)
    checks = {
        "speaker_ids_disjoint_across_splits": {
            "passed": speaker_leaks == 0,
            "violation_count": speaker_leaks,
        },
        "audio_sha256_disjoint_across_splits": {
            "passed": hash_leaks == 0,
            "violation_count": hash_leaks,
        },
        "pilot_membership_resolved": {
            "passed": not missing_pilot,
            "violation_count": len(missing_pilot),
        },
        "pilot_exclusively_test": {
            "passed": not pilot_non_test and bool(pilot_ids),
            "violation_count": len(pilot_non_test),
        },
        "smoke_exclusively_train": {
            "passed": smoke_non_train == 0,
            "violation_count": smoke_non_train,
        },
        "smoke_disjoint_from_test_ids": {
            "passed": smoke_test_id_overlap == 0,
            "violation_count": smoke_test_id_overlap,
        },
        "smoke_disjoint_from_test_hashes": {
            "passed": smoke_test_hash_overlap == 0,
            "violation_count": smoke_test_hash_overlap,
        },
        "smoke_disjoint_from_pilot": {
            "passed": smoke_pilot_overlap == 0,
            "violation_count": smoke_pilot_overlap,
        },
    }
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_version": dataset.dataset_version,
        "manifest_sha256": dataset.manifest_sha256,
        "counts": {
            "manifest_rows": len(dataset.rows),
            "rows_by_split": dict(sorted(Counter(row.split for row in dataset.rows).items())),
            "speakers_by_split": {
                split: len({row.speaker_id for row in dataset.rows if row.split == split})
                for split in ("train", "validation", "test")
            },
            "pilot_unique_audio_ids": len(pilot_ids),
            "smoke_selected_audio_ids": len(selected),
        },
        "checks": checks,
        "overall_passed": all(cast(dict[str, Any], value)["passed"] for value in checks.values()),
        "privacy": {
            "contains_private_paths": False,
            "contains_audio_or_speaker_ids": False,
        },
    }


def write_split_report(
    settings: SmokeSettings,
    dataset: AuditedDataset,
    selected: Sequence[ManifestRow],
) -> tuple[Path, dict[str, Any]]:
    """Write aggregate split evidence and fail closed if any check is violated."""

    report = build_split_integrity_report(
        dataset,
        selected,
        _pilot_ids(settings.pilot_prediction_files),
    )
    path = settings.reports_root / SPLIT_REPORT_FILENAME
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if report["overall_passed"] is not True:
        raise ConfigError("L'intégrité des splits a échoué ; consultez le rapport JSON local.")
    return path, report


def run_audit(settings: SmokeSettings) -> dict[str, Any]:
    """Generate all pre-training Phase 4B reports."""

    dataset = load_audited_dataset(settings)
    selected = select_representative_train_rows(
        dataset.rows,
        settings.sample_count,
        settings.seed,
    )
    annotations = load_or_initialize_annotations(settings, dataset, selected)
    manual_path = write_manual_report(settings, dataset, selected, annotations)
    normalization_path = write_normalization_report(settings, dataset)
    split_path, split_report = write_split_report(settings, dataset, selected)
    return {
        "dataset": dataset,
        "selected": selected,
        "annotations": annotations,
        "manual_report": manual_path,
        "normalization_report": normalization_path,
        "split_report": split_path,
        "split_overall_passed": split_report["overall_passed"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint used before both manual review and smoke training."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        default="configs/experiments/smoke_overfit_whisper_tiny_dy.yaml",
    )
    arguments = parser.parse_args(argv)
    try:
        result = run_audit(load_smoke_settings(arguments.experiment))
    except IvoireVoiceError as exc:
        parser.error(str(exc))
    selected = cast(tuple[ManifestRow, ...], result["selected"])
    annotations = cast(dict[str, Any], result["annotations"])
    correct = sum(
        _status_entry(annotations, row.utterance_id).get("status") == "correct"
        for row in selected
    )
    print(f"Audit Phase 4B réussi : {len(selected)} audios train sélectionnés.")
    print("Intégrité des splits : OK.")
    print(f"Validation auditive : {correct}/{len(selected)} marqués correct.")
    print(f"Rapports privés locaux : {cast(Path, result['manual_report']).parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
