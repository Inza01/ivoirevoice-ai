"""Freeze and validate the immutable local Dioula dataset candidate v0.1."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, cast

from ivoirevoice.data.audio import sha256_file
from ivoirevoice.data.clips import normalize_transcription
from ivoirevoice.data.settings import DioulaDataSettings, load_dioula_settings
from ivoirevoice.exceptions import ConfigError, IvoireVoiceError

FALLING_INTONATION_MARKER = "↘"
SPLIT_NAMES = ("train", "validation", "test")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40,64}$")


@dataclass(frozen=True, slots=True)
class FreezeCandidateRow:
    """Curated candidate fields needed to produce the frozen manifest."""

    utterance_id: str
    sentence_id: str
    speaker_id: str
    gender_folder: str
    language: str
    text_raw: str
    text_with_tones_nfc: str
    text_without_tones_nfc: str
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
class FrozenRow:
    """One final local candidate row with its human-approved speaker split."""

    utterance_id: str
    sentence_id: str
    speaker_id: str
    gender_folder: str
    language: str
    text_raw: str
    text_with_tones_nfc: str
    text_without_tones_nfc: str
    target_text_mvp: str
    intonation_falling: bool
    audio_path: str
    duration_seconds: float
    sample_rate_hz: int
    channels: int
    file_size_bytes: int
    audio_sha256: str
    source_json: str
    split: str
    license_status: str
    consent_status: str
    usage_scope: str


@dataclass(frozen=True, slots=True)
class SplitPlan:
    """Human-approved deterministic speaker assignment."""

    strategy: str
    seed: int
    speaker_ids: dict[str, tuple[str, ...]]


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            payload: object = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Impossible de lire {description} : {exc}") from exc
    if not isinstance(payload, Mapping) or not all(isinstance(key, str) for key in payload):
        raise ConfigError(f"{description.capitalize()} doit contenir un objet JSON.")
    return cast(dict[str, Any], dict(payload))


def _required_candidate_columns() -> set[str]:
    return {
        "utterance_id",
        "sentence_id",
        "speaker_id",
        "gender_folder",
        "language",
        "text_raw",
        "text_with_tones_nfc",
        "text_without_tones_nfc",
        "audio_path",
        "duration_seconds",
        "sample_rate_hz",
        "channels",
        "file_size_bytes",
        "audio_sha256",
        "source_json",
        "license_status",
        "usage_scope",
        "eligibility_status",
        "exclusion_reason",
        "split",
    }


def load_freeze_candidate(path: Path) -> tuple[FreezeCandidateRow, ...]:
    """Load the curated candidate and reject any non-eligible or assigned row."""

    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            missing = _required_candidate_columns().difference(reader.fieldnames or [])
            if missing:
                raise ConfigError("Colonnes absentes du candidat : " + ", ".join(sorted(missing)))
            rows = tuple(
                FreezeCandidateRow(
                    utterance_id=row["utterance_id"],
                    sentence_id=row["sentence_id"],
                    speaker_id=row["speaker_id"],
                    gender_folder=row["gender_folder"],
                    language=row["language"],
                    text_raw=row["text_raw"],
                    text_with_tones_nfc=row["text_with_tones_nfc"],
                    text_without_tones_nfc=row["text_without_tones_nfc"],
                    audio_path=row["audio_path"],
                    duration_seconds=float(row["duration_seconds"]),
                    sample_rate_hz=int(row["sample_rate_hz"]),
                    channels=int(row["channels"]),
                    file_size_bytes=int(row["file_size_bytes"]),
                    audio_sha256=row["audio_sha256"],
                    source_json=row["source_json"],
                    license_status=row["license_status"],
                    usage_scope=row["usage_scope"],
                    eligibility_status=row["eligibility_status"],
                    exclusion_reason=row["exclusion_reason"],
                    split=row["split"],
                )
                for row in reader
            )
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        raise ConfigError(f"Impossible de lire le manifeste candidat : {exc}") from exc

    invalid = [
        row
        for row in rows
        if row.eligibility_status != "eligible" or row.exclusion_reason or row.split
    ]
    if invalid:
        raise ConfigError(
            "Le candidat contient une ligne non éligible, en quarantaine ou déjà assignée."
        )
    return rows


def _speaker_list(value: object, name: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ConfigError(f"Le champ de split '{name}' doit être une liste non vide.")
    speakers = tuple(sorted(cast(list[str], value)))
    if len(speakers) != len(set(speakers)):
        raise ConfigError(f"Le champ de split '{name}' contient un locuteur dupliqué.")
    return speakers


def load_split_plan(path: Path, *, strategy: str, seed: int) -> SplitPlan:
    """Select the exact human-approved strategy from the comparison report."""

    comparison = _load_json(path, "le rapport de comparaison des splits")
    raw_strategies = comparison.get("strategies")
    if not isinstance(raw_strategies, list):
        raise ConfigError("Le rapport de comparaison ne contient pas de stratégies.")

    selected: dict[str, Any] | None = None
    for raw_strategy in raw_strategies:
        if not isinstance(raw_strategy, Mapping):
            continue
        strategy_data = cast(dict[str, Any], dict(raw_strategy))
        if strategy_data.get("strategy") == strategy:
            selected = strategy_data
            break
    if selected is None:
        raise ConfigError(f"La stratégie humaine '{strategy}' est absente du rapport.")
    if selected.get("seed") != seed:
        raise ConfigError("La seed du split approuvé diffère de la configuration.")
    raw_speaker_ids = selected.get("speaker_ids")
    if not isinstance(raw_speaker_ids, Mapping):
        raise ConfigError("La stratégie approuvée ne contient pas les locuteurs par split.")
    speaker_mapping = cast(dict[str, object], dict(raw_speaker_ids))
    speaker_ids = {split: _speaker_list(speaker_mapping.get(split), split) for split in SPLIT_NAMES}
    all_speakers = [
        speaker for split_speakers in speaker_ids.values() for speaker in split_speakers
    ]
    if len(all_speakers) != len(set(all_speakers)):
        raise ConfigError("La stratégie approuvée contient une fuite de locuteur.")
    return SplitPlan(strategy=strategy, seed=seed, speaker_ids=speaker_ids)


def _speaker_to_split(plan: SplitPlan) -> dict[str, str]:
    return {speaker: split for split, speakers in plan.speaker_ids.items() for speaker in speakers}


def target_text_without_falling_marker(text_without_tones_nfc: str) -> str:
    """Remove only the falling-intonation marker from the normalized MVP target."""

    return normalize_transcription(text_without_tones_nfc.replace(FALLING_INTONATION_MARKER, ""))


def build_frozen_rows(
    candidate_rows: tuple[FreezeCandidateRow, ...],
    plan: SplitPlan,
    *,
    license_status: str,
    consent_status: str,
    usage_scope: str,
) -> tuple[FrozenRow, ...]:
    """Apply the approved split and target policy without changing source text."""

    speaker_to_split = _speaker_to_split(plan)
    candidate_speakers = {row.speaker_id for row in candidate_rows}
    if candidate_speakers != set(speaker_to_split):
        missing_from_plan = candidate_speakers.difference(speaker_to_split)
        missing_from_candidate = set(speaker_to_split).difference(candidate_speakers)
        raise ConfigError(
            "Le plan de split et le candidat ne couvrent pas les mêmes locuteurs "
            f"({len(missing_from_plan)} sans split, "
            f"{len(missing_from_candidate)} sans audio)."
        )

    rows: list[FrozenRow] = []
    for row in candidate_rows:
        if row.license_status != license_status or row.usage_scope != usage_scope:
            raise ConfigError("La gouvernance du candidat diffère de la politique de gel.")
        target_text = target_text_without_falling_marker(row.text_without_tones_nfc)
        if not target_text:
            raise ConfigError("Le retrait du marqueur produit une cible MVP vide.")
        rows.append(
            FrozenRow(
                utterance_id=row.utterance_id,
                sentence_id=row.sentence_id,
                speaker_id=row.speaker_id,
                gender_folder=row.gender_folder,
                language=row.language,
                text_raw=row.text_raw,
                text_with_tones_nfc=row.text_with_tones_nfc,
                text_without_tones_nfc=row.text_without_tones_nfc,
                target_text_mvp=target_text,
                intonation_falling=FALLING_INTONATION_MARKER in row.text_raw,
                audio_path=row.audio_path,
                duration_seconds=row.duration_seconds,
                sample_rate_hz=row.sample_rate_hz,
                channels=row.channels,
                file_size_bytes=row.file_size_bytes,
                audio_sha256=row.audio_sha256,
                source_json=row.source_json,
                split=speaker_to_split[row.speaker_id],
                license_status=license_status,
                consent_status=consent_status,
                usage_scope=usage_scope,
            )
        )
    return tuple(sorted(rows, key=lambda item: (item.audio_path, item.utterance_id)))


def _safe_relative_reference(value: str) -> bool:
    path = PurePosixPath(value)
    return (
        bool(value)
        and not path.is_absolute()
        and ".." not in path.parts
        and "://" not in value
        and "?" not in value
        and "#" not in value
        and "\\" not in value
    )


def _frozen_metrics(rows: tuple[FrozenRow, ...]) -> dict[str, Any]:
    speaker_sets = {
        split: {row.speaker_id for row in rows if row.split == split} for split in SPLIT_NAMES
    }
    speaker_genders: dict[str, str] = {}
    for row in rows:
        previous = speaker_genders.setdefault(row.speaker_id, row.gender_folder)
        if previous != row.gender_folder:
            raise ConfigError("Un locuteur possède plusieurs catégories de genre.")
    audio_counts = Counter(row.split for row in rows)
    duration_seconds = {
        split: sum(row.duration_seconds for row in rows if row.split == split)
        for split in SPLIT_NAMES
    }
    gender_counts = {
        split: {
            gender: sum(speaker_genders[speaker] == gender for speaker in speakers)
            for gender in ("men", "women", "unknown")
        }
        for split, speakers in speaker_sets.items()
    }
    total_duration = sum(duration_seconds.values())
    return {
        "audio_count": len(rows),
        "speaker_count": len(speaker_genders),
        "duration_total_seconds": total_duration,
        "audio_count_by_split": {split: audio_counts.get(split, 0) for split in SPLIT_NAMES},
        "speaker_count_by_split": {split: len(speaker_sets[split]) for split in SPLIT_NAMES},
        "duration_seconds_by_split": duration_seconds,
        "duration_fraction_by_split": {
            split: duration_seconds[split] / total_duration if total_duration else 0.0
            for split in SPLIT_NAMES
        },
        "gender_speaker_count_by_split": gender_counts,
        "intonation_falling_rows": sum(row.intonation_falling for row in rows),
        "privacy_checks": {
            "audio_paths_relative": all(_safe_relative_reference(row.audio_path) for row in rows),
            "source_json_paths_relative": all(
                _safe_relative_reference(row.source_json) for row in rows
            ),
            "signed_urls_absent": all(
                "://" not in row.audio_path
                and "?" not in row.audio_path
                and "://" not in row.source_json
                and "?" not in row.source_json
                for row in rows
            ),
            "absolute_personal_paths_absent": all(
                not PurePosixPath(row.audio_path).is_absolute()
                and not PurePosixPath(row.source_json).is_absolute()
                for row in rows
            ),
        },
    }


def validate_frozen_rows(
    rows: tuple[FrozenRow, ...],
    *,
    expected_audio_count: int,
    expected_speaker_count: int,
    expected_speaker_counts: dict[str, int],
    language: str,
    license_status: str,
    consent_status: str,
    usage_scope: str,
) -> dict[str, Any]:
    """Run all row-, audio-, speaker-, split-, target- and privacy-level checks."""

    if len(rows) != expected_audio_count:
        raise ConfigError(
            f"Le manifeste gelé contient {len(rows)} audios au lieu de {expected_audio_count}."
        )
    if len({row.utterance_id for row in rows}) != len(rows):
        raise ConfigError("Le manifeste gelé contient des utterance_id dupliqués.")
    if len({row.audio_path for row in rows}) != len(rows):
        raise ConfigError("Le manifeste gelé contient des chemins audio dupliqués.")
    if len({row.audio_sha256 for row in rows}) != len(rows):
        raise ConfigError("Le manifeste gelé contient des SHA-256 audio dupliqués.")
    if any(not SHA256_PATTERN.fullmatch(row.audio_sha256) for row in rows):
        raise ConfigError("Le manifeste gelé contient un SHA-256 invalide.")
    if any(row.split not in SPLIT_NAMES for row in rows):
        raise ConfigError("Le manifeste gelé contient un split vide ou inconnu.")
    if any(
        not _safe_relative_reference(row.audio_path)
        or not _safe_relative_reference(row.source_json)
        for row in rows
    ):
        raise ConfigError("Le manifeste gelé contient une URL ou un chemin non relatif.")
    if any(
        FALLING_INTONATION_MARKER in row.target_text_mvp or not row.target_text_mvp for row in rows
    ):
        raise ConfigError("Une cible MVP est vide ou contient encore le symbole ↘.")
    if any(row.intonation_falling != (FALLING_INTONATION_MARKER in row.text_raw) for row in rows):
        raise ConfigError("Le booléen d'intonation ne correspond pas au texte brut.")
    if any(
        row.target_text_mvp != target_text_without_falling_marker(row.text_without_tones_nfc)
        for row in rows
    ):
        raise ConfigError("Une cible MVP ne dérive pas uniquement du texte NFC sans tons.")
    if any(
        not row.text_raw or not row.text_with_tones_nfc or not row.text_without_tones_nfc
        for row in rows
    ):
        raise ConfigError("Le manifeste gelé a perdu une variante textuelle.")
    if any(
        not unicodedata.is_normalized("NFC", row.text_with_tones_nfc)
        or not unicodedata.is_normalized("NFC", row.text_without_tones_nfc)
        or not unicodedata.is_normalized("NFC", row.target_text_mvp)
        for row in rows
    ):
        raise ConfigError("Une variante textuelle annoncée NFC ne l'est pas.")
    if any(
        row.language != language
        or row.license_status != license_status
        or row.consent_status != consent_status
        or row.usage_scope != usage_scope
        for row in rows
    ):
        raise ConfigError("Une ligne ne respecte pas la langue ou la gouvernance locale.")
    if any(
        row.duration_seconds <= 0
        or row.sample_rate_hz <= 0
        or row.channels <= 0
        or row.file_size_bytes <= 0
        for row in rows
    ):
        raise ConfigError("Une ligne contient des métadonnées audio invalides.")

    speaker_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        speaker_splits[row.speaker_id].add(row.split)
    if any(len(splits) != 1 for splits in speaker_splits.values()):
        raise ConfigError("Fuite de locuteur détectée entre les splits.")

    metrics = _frozen_metrics(rows)
    if metrics["speaker_count"] != expected_speaker_count:
        raise ConfigError("Le nombre total de locuteurs diffère de la décision humaine.")
    if metrics["speaker_count_by_split"] != expected_speaker_counts:
        raise ConfigError("La distribution des locuteurs n'est pas exactement 15/3/3.")
    if any(metrics["audio_count_by_split"][split] == 0 for split in SPLIT_NAMES):
        raise ConfigError("Au moins un split ne contient aucun audio.")
    gender_counts = metrics["gender_speaker_count_by_split"]
    if any(
        gender_counts[split]["men"] == 0 or gender_counts[split]["women"] == 0
        for split in SPLIT_NAMES
    ):
        raise ConfigError("Chaque split doit conserver des locuteurs men et women.")
    if not all(metrics["privacy_checks"].values()):
        raise ConfigError("Un contrôle de confidentialité a échoué.")
    return metrics


def assert_frozen_matches_candidate(
    frozen_rows: tuple[FrozenRow, ...],
    candidate_rows: tuple[FreezeCandidateRow, ...],
) -> None:
    """Prove that the freeze neither adds nor removes rows and preserves source text."""

    candidate_by_audio = {row.audio_path: row for row in candidate_rows}
    if len(candidate_by_audio) != len(candidate_rows):
        raise ConfigError("Le candidat source contient encore des chemins audio dupliqués.")
    if set(candidate_by_audio) != {row.audio_path for row in frozen_rows}:
        raise ConfigError("Le gel a ajouté ou retiré un audio du candidat validé.")
    for frozen in frozen_rows:
        source = candidate_by_audio[frozen.audio_path]
        if (
            frozen.utterance_id != source.utterance_id
            or frozen.speaker_id != source.speaker_id
            or frozen.text_raw != source.text_raw
            or frozen.text_with_tones_nfc != source.text_with_tones_nfc
            or frozen.text_without_tones_nfc != source.text_without_tones_nfc
            or frozen.audio_sha256 != source.audio_sha256
        ):
            raise ConfigError("Le gel a altéré une donnée source qui devait être conservée.")


def _manifest_content(rows: tuple[FrozenRow, ...]) -> bytes:
    stream = io.StringIO(newline="")
    fieldnames = [field.name for field in fields(FrozenRow)]
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        payload = asdict(row)
        payload["intonation_falling"] = str(row.intonation_falling).lower()
        writer.writerow(payload)
    return stream.getvalue().encode("utf-8")


def _json_content(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _split_report(plan: SplitPlan, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "strategy": plan.strategy,
        "seed": plan.seed,
        "leakage_free": True,
        "speaker_ids": {split: list(plan.speaker_ids[split]) for split in SPLIT_NAMES},
        "speaker_count_by_split": metrics["speaker_count_by_split"],
        "audio_count_by_split": metrics["audio_count_by_split"],
        "duration_seconds_by_split": metrics["duration_seconds_by_split"],
        "duration_fraction_by_split": metrics["duration_fraction_by_split"],
        "gender_speaker_count_by_split": metrics["gender_speaker_count_by_split"],
    }


def _safe_config_reference(config_path: Path) -> str:
    try:
        relative = config_path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        return config_path.name
    return relative.as_posix()


def _pipeline_commit_sha() -> str:
    repository_root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ConfigError(f"Impossible de déterminer le commit du pipeline : {exc}") from exc
    commit_sha = result.stdout.strip().lower()
    if not GIT_SHA_PATTERN.fullmatch(commit_sha):
        raise ConfigError("Le commit SHA du pipeline est invalide.")
    return commit_sha


def _validate_candidate_provenance(
    settings: DioulaDataSettings,
    candidate_rows: tuple[FreezeCandidateRow, ...],
) -> str:
    metadata = _load_json(settings.candidate_metadata_path, "les métadonnées du candidat")
    candidate_hash = sha256_file(settings.candidate_manifest_path)
    if metadata.get("candidate_manifest_sha256") != candidate_hash:
        raise ConfigError("Le hash du candidat ne correspond pas à ses métadonnées.")
    if metadata.get("included_rows") != len(candidate_rows):
        raise ConfigError("Le nombre de lignes du candidat ne correspond pas à ses métadonnées.")
    if metadata.get("license_status") != settings.license_status:
        raise ConfigError("La licence du candidat ne correspond pas à la configuration.")
    if metadata.get("usage_scope") != settings.usage_scope:
        raise ConfigError("Le périmètre du candidat ne correspond pas à la configuration.")
    if metadata.get("recovery_executed") is not False:
        raise ConfigError("Le gel refuse un candidat issu d'une récupération audio.")
    return candidate_hash


def _metadata(
    settings: DioulaDataSettings,
    *,
    config_path: Path,
    pipeline_commit_sha: str,
    manifest_hash: str,
    candidate_hash: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset_version": settings.freeze.dataset_version,
        "dataset_status": settings.freeze.dataset_status,
        "publication_allowed": settings.freeze.publication_allowed,
        "model_derivative_publication_allowed": (
            settings.freeze.model_derivative_publication_allowed
        ),
        "pipeline_commit_sha": pipeline_commit_sha,
        "manifest_sha256": manifest_hash,
        "source_candidate_manifest_sha256": candidate_hash,
        "source_candidate_metadata_sha256": sha256_file(settings.candidate_metadata_path),
        "source_split_comparison_sha256": sha256_file(settings.split_comparison_path),
        "configuration": _safe_config_reference(config_path),
        "configuration_sha256": sha256_file(config_path),
        "seed": settings.split.seed,
        "split_strategy": settings.freeze.split_strategy,
        "audio_count": metrics["audio_count"],
        "speaker_count": metrics["speaker_count"],
        "duration_total_seconds": metrics["duration_total_seconds"],
        "audio_count_by_split": metrics["audio_count_by_split"],
        "speaker_count_by_split": metrics["speaker_count_by_split"],
        "duration_seconds_by_split": metrics["duration_seconds_by_split"],
        "duration_fraction_by_split": metrics["duration_fraction_by_split"],
        "gender_speaker_count_by_split": metrics["gender_speaker_count_by_split"],
        "intonation_falling_rows": metrics["intonation_falling_rows"],
        "license_status": settings.license_status,
        "consent_status": settings.consent_status,
        "usage_scope": settings.usage_scope,
        "privacy_checks": metrics["privacy_checks"],
        "curation_rules": [
            "include only eligible rows from the curated candidate",
            "one row per unique audio_path",
            "one row per unique audio_sha256",
            "keep all quarantined rows outside v0.1",
            "do not recover missing audio",
            "do not run ffmpeg",
            "apply the human-approved B_15_3_3 speaker assignment",
        ],
        "normalization_rules": [
            "preserve text_raw byte-for-byte through CSV parsing and writing",
            "preserve text_with_tones_nfc",
            "preserve text_without_tones_nfc",
            "target_text_mvp=text_without_tones_nfc with ↘ removed",
            "collapse whitespace left by marker removal and normalize target to NFC",
            "intonation_falling=true exactly when text_raw contains ↘",
        ],
        "open_decisions": [
            "written dataset license authorization",
            "written participant consent confirmation",
            "permission to publish the corpus, manifest or derived model",
        ],
    }


def _report_markdown(metadata: dict[str, Any]) -> str:
    audio_counts = metadata["audio_count_by_split"]
    speaker_counts = metadata["speaker_count_by_split"]
    durations = metadata["duration_seconds_by_split"]
    genders = metadata["gender_speaker_count_by_split"]
    table_rows = "\n".join(
        (
            f"| {split} | {speaker_counts[split]} | {audio_counts[split]} | "
            f"{durations[split]} | {genders[split]['men']} | "
            f"{genders[split]['women']} |"
        )
        for split in SPLIT_NAMES
    )
    return f"""# Dataset dioula v0.1 — gel local

## Statut

- version : `{metadata["dataset_version"]}`
- statut : `{metadata["dataset_status"]}`
- licence : `{metadata["license_status"]}`
- consentement : `{metadata["consent_status"]}`
- périmètre : `{metadata["usage_scope"]}`
- publication du corpus ou du manifeste : interdite
- publication d'un modèle dérivé : interdite

## Résultat

- audios uniques : {metadata["audio_count"]}
- locuteurs : {metadata["speaker_count"]}
- durée totale (secondes) : {metadata["duration_total_seconds"]}
- lignes avec intonation descendante : {metadata["intonation_falling_rows"]}
- SHA-256 du manifeste : `{metadata["manifest_sha256"]}`

| Split | Locuteurs | Audios | Durée (s) | Men | Women |
|---|---:|---:|---:|---:|---:|
{table_rows}

Les 1 885 lignes sans audio récupérable et les deux lignes du conflit SHA-256
restent hors du manifeste. Aucun audio n'a été converti ou récupéré. Le texte
brut et les deux variantes NFC sont conservés ; seul `target_text_mvp` retire
le marqueur `↘`.
"""


def _write_immutable_outputs(outputs: dict[Path, bytes]) -> None:
    for path, content in outputs.items():
        try:
            if path.exists() and path.read_bytes() != content:
                raise ConfigError(
                    f"L'artefact local immuable existe avec un contenu différent : {path.name}"
                )
        except OSError as exc:
            raise ConfigError(f"Impossible de contrôler l'artefact {path.name} : {exc}") from exc

    for path, content in outputs.items():
        try:
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = path.with_suffix(f"{path.suffix}.tmp")
            temporary_path.write_bytes(content)
            temporary_path.replace(path)
        except OSError as exc:
            raise ConfigError(f"Impossible d'écrire l'artefact {path.name} : {exc}") from exc


def freeze_dataset(
    settings: DioulaDataSettings,
    *,
    config_path: Path,
    pipeline_commit_sha: str | None = None,
) -> dict[str, Any]:
    """Build all frozen outputs, writing only if immutable contents agree."""

    candidate_rows = load_freeze_candidate(settings.candidate_manifest_path)
    candidate_hash = _validate_candidate_provenance(settings, candidate_rows)
    plan = load_split_plan(
        settings.split_comparison_path,
        strategy=settings.freeze.split_strategy,
        seed=settings.split.seed,
    )
    if {
        split: len(plan.speaker_ids[split]) for split in SPLIT_NAMES
    } != settings.freeze.expected_speaker_counts:
        raise ConfigError("Le plan approuvé ne contient pas exactement 15/3/3 locuteurs.")

    rows = build_frozen_rows(
        candidate_rows,
        plan,
        license_status=settings.license_status,
        consent_status=settings.consent_status,
        usage_scope=settings.usage_scope,
    )
    metrics = validate_frozen_rows(
        rows,
        expected_audio_count=settings.freeze.expected_audio_count,
        expected_speaker_count=settings.freeze.expected_speaker_count,
        expected_speaker_counts=settings.freeze.expected_speaker_counts,
        language=settings.language,
        license_status=settings.license_status,
        consent_status=settings.consent_status,
        usage_scope=settings.usage_scope,
    )
    assert_frozen_matches_candidate(rows, candidate_rows)

    manifest_content = _manifest_content(rows)
    manifest_hash = sha256(manifest_content).hexdigest()
    commit_sha = pipeline_commit_sha or _pipeline_commit_sha()
    if not GIT_SHA_PATTERN.fullmatch(commit_sha):
        raise ConfigError("Le commit SHA fourni pour le pipeline est invalide.")
    metadata = _metadata(
        settings,
        config_path=config_path,
        pipeline_commit_sha=commit_sha,
        manifest_hash=manifest_hash,
        candidate_hash=candidate_hash,
        metrics=metrics,
    )
    split_report = _split_report(plan, metrics)
    outputs = {
        settings.frozen_manifest_path: manifest_content,
        settings.frozen_metadata_path: _json_content(metadata),
        settings.frozen_report_path: _report_markdown(metadata).encode("utf-8"),
        settings.frozen_split_report_path: _json_content(split_report),
    }
    _write_immutable_outputs(outputs)
    return validate_frozen_dataset(settings, config_path=config_path)


def _bool_value(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ConfigError("La colonne intonation_falling doit contenir true ou false.")
    return normalized == "true"


def load_frozen_manifest(path: Path) -> tuple[FrozenRow, ...]:
    """Load a frozen manifest for independent post-write validation."""

    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {field.name for field in fields(FrozenRow)}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ConfigError(
                    "Colonnes absentes du manifeste gelé : " + ", ".join(sorted(missing))
                )
            rows = tuple(
                FrozenRow(
                    utterance_id=row["utterance_id"],
                    sentence_id=row["sentence_id"],
                    speaker_id=row["speaker_id"],
                    gender_folder=row["gender_folder"],
                    language=row["language"],
                    text_raw=row["text_raw"],
                    text_with_tones_nfc=row["text_with_tones_nfc"],
                    text_without_tones_nfc=row["text_without_tones_nfc"],
                    target_text_mvp=row["target_text_mvp"],
                    intonation_falling=_bool_value(row["intonation_falling"]),
                    audio_path=row["audio_path"],
                    duration_seconds=float(row["duration_seconds"]),
                    sample_rate_hz=int(row["sample_rate_hz"]),
                    channels=int(row["channels"]),
                    file_size_bytes=int(row["file_size_bytes"]),
                    audio_sha256=row["audio_sha256"],
                    source_json=row["source_json"],
                    split=row["split"],
                    license_status=row["license_status"],
                    consent_status=row["consent_status"],
                    usage_scope=row["usage_scope"],
                )
                for row in reader
            )
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        raise ConfigError(f"Impossible de lire le manifeste gelé : {exc}") from exc
    return rows


def validate_frozen_dataset(
    settings: DioulaDataSettings,
    *,
    config_path: Path,
) -> dict[str, Any]:
    """Re-open and validate the complete frozen release and its provenance."""

    rows = load_frozen_manifest(settings.frozen_manifest_path)
    metrics = validate_frozen_rows(
        rows,
        expected_audio_count=settings.freeze.expected_audio_count,
        expected_speaker_count=settings.freeze.expected_speaker_count,
        expected_speaker_counts=settings.freeze.expected_speaker_counts,
        language=settings.language,
        license_status=settings.license_status,
        consent_status=settings.consent_status,
        usage_scope=settings.usage_scope,
    )
    candidate_rows = load_freeze_candidate(settings.candidate_manifest_path)
    candidate_hash = _validate_candidate_provenance(settings, candidate_rows)
    assert_frozen_matches_candidate(rows, candidate_rows)
    plan = load_split_plan(
        settings.split_comparison_path,
        strategy=settings.freeze.split_strategy,
        seed=settings.split.seed,
    )
    speaker_to_split = _speaker_to_split(plan)
    if any(speaker_to_split.get(row.speaker_id) != row.split for row in rows):
        raise ConfigError("Le manifeste ne respecte plus l'affectation B approuvée.")

    metadata = _load_json(settings.frozen_metadata_path, "les métadonnées v0.1")
    manifest_hash = sha256_file(settings.frozen_manifest_path)
    required_metadata = {
        "dataset_version": settings.freeze.dataset_version,
        "dataset_status": settings.freeze.dataset_status,
        "publication_allowed": False,
        "model_derivative_publication_allowed": False,
        "manifest_sha256": manifest_hash,
        "source_candidate_manifest_sha256": candidate_hash,
        "configuration_sha256": sha256_file(config_path),
        "seed": settings.split.seed,
        "split_strategy": settings.freeze.split_strategy,
        "license_status": settings.license_status,
        "consent_status": settings.consent_status,
        "usage_scope": settings.usage_scope,
        "audio_count": metrics["audio_count"],
        "speaker_count": metrics["speaker_count"],
        "audio_count_by_split": metrics["audio_count_by_split"],
        "speaker_count_by_split": metrics["speaker_count_by_split"],
        "duration_seconds_by_split": metrics["duration_seconds_by_split"],
        "gender_speaker_count_by_split": metrics["gender_speaker_count_by_split"],
        "intonation_falling_rows": metrics["intonation_falling_rows"],
        "privacy_checks": metrics["privacy_checks"],
    }
    for field_name, expected in required_metadata.items():
        if metadata.get(field_name) != expected:
            raise ConfigError(f"Les métadonnées v0.1 sont incohérentes : {field_name}.")
    commit_sha = metadata.get("pipeline_commit_sha")
    if not isinstance(commit_sha, str) or not GIT_SHA_PATTERN.fullmatch(commit_sha):
        raise ConfigError("Les métadonnées ne contiennent pas un commit pipeline valide.")

    split_report = _load_json(settings.frozen_split_report_path, "le rapport du split v0.1")
    if split_report != _split_report(plan, metrics):
        raise ConfigError("Le rapport du split v0.1 ne correspond plus au manifeste.")
    expected_report = _report_markdown(metadata)
    try:
        actual_report = settings.frozen_report_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ConfigError(f"Impossible de lire le rapport v0.1 : {exc}") from exc
    if actual_report != expected_report:
        raise ConfigError("Le rapport Markdown v0.1 ne correspond plus aux métadonnées.")

    return {
        **metrics,
        "manifest_sha256": manifest_hash,
        "publication_allowed": False,
        "model_derivative_publication_allowed": False,
    }


def _print_summary(summary: dict[str, Any]) -> None:
    audio_counts = summary["audio_count_by_split"]
    speaker_counts = summary["speaker_count_by_split"]
    genders = summary["gender_speaker_count_by_split"]
    print(f"audio_count={summary['audio_count']}")
    print(f"speaker_count={summary['speaker_count']}")
    print(f"duration_total_seconds={summary['duration_total_seconds']}")
    print(
        "split_distribution="
        f"train:{audio_counts['train']} audios/{speaker_counts['train']} speakers,"
        f"validation:{audio_counts['validation']} audios/"
        f"{speaker_counts['validation']} speakers,"
        f"test:{audio_counts['test']} audios/{speaker_counts['test']} speakers"
    )
    print(
        "gender_presence="
        + ",".join(
            f"{split}:men={genders[split]['men'] > 0}/women={genders[split]['women'] > 0}"
            for split in SPLIT_NAMES
        )
    )
    print(f"intonation_falling_rows={summary['intonation_falling_rows']}")
    print(f"manifest_sha256={summary['manifest_sha256']}")
    print(
        "privacy_checks="
        + ",".join(f"{name}={value}" for name, value in summary["privacy_checks"].items())
        + ",publication_allowed=False,model_derivative_publication_allowed=False"
    )
    print(
        "open_decisions=written_license,consent_confirmation,"
        "corpus_manifest_and_model_publication_authorization"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Geler le dataset dioula local v0.1.")
    parser.add_argument("--config", required=True, help="Configuration YAML des données.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Valider les artefacts existants sans les modifier.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point with aggregate and privacy-safe output only."""

    args = _parse_args()
    config_path = Path(args.config)
    try:
        settings = load_dioula_settings(config_path)
        if args.validate_only:
            summary = validate_frozen_dataset(settings, config_path=config_path)
        else:
            summary = freeze_dataset(settings, config_path=config_path)
    except IvoireVoiceError as exc:
        print(f"ERREUR: {exc}")
        return 1
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
