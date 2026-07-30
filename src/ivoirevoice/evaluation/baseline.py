"""Reproducible, resumable and privacy-aware Dioula ASR baseline runner."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

import yaml

from ivoirevoice.data.audio import sha256_file
from ivoirevoice.data.clips import normalize_transcription
from ivoirevoice.evaluation.metrics import ScoredItem, compute_evaluation_metrics
from ivoirevoice.exceptions import ConfigError, IvoireVoiceError
from ivoirevoice.models.whisper import (
    WhisperBackend,
    WhisperSettings,
    load_whisper_settings,
    runtime_labels,
)

EvaluationLevel = Literal["smoke", "pilot", "full"]
SPLIT_NAME = "test"
FALLING_MARKER = "↘"
GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40,64}$")


@dataclass(frozen=True, slots=True)
class BaselineSettings:
    """Validated experiment and external-storage configuration."""

    experiment_id: str
    language: str
    split: str
    seed: int
    model_config_path: Path
    expected_model_id: str
    manifest_path: Path
    dataset_metadata_path: Path
    dataset_root: Path
    artifacts_root: Path
    output_relative_directory: Path
    report_relative_directory: Path
    expected_test_audio_count: int
    expected_test_speaker_count: int
    smoke_per_speaker: int
    pilot_per_speaker: int
    timeout_seconds: float
    lowercase: bool
    remove_punctuation: bool
    publication_allowed: bool


@dataclass(frozen=True, slots=True)
class BaselineItem:
    """One private test item selected from the frozen manifest."""

    utterance_id: str
    speaker_id: str
    gender_folder: str
    audio_path: str
    audio_sha256: str
    audio_duration_seconds: float
    reference_raw: str


@dataclass(frozen=True, slots=True)
class BaselineDataset:
    """Validated test items and immutable dataset provenance."""

    items: tuple[BaselineItem, ...]
    manifest_sha256: str
    dataset_version: str


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """Private progressive result; never written to a shareable report."""

    utterance_id: str
    speaker_id: str
    gender_folder: str
    audio_path: str
    audio_sha256: str
    audio_duration_seconds: float
    reference_raw: str
    reference_normalized: str
    prediction_raw: str
    prediction_normalized: str
    processing_time_seconds: float
    model_name: str
    status: str
    error_type: str


BackendFactory = Callable[[WhisperSettings], WhisperBackend]


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ConfigError(f"La section '{name}' doit être un objet YAML.")
    return cast(dict[str, Any], dict(value))


def _required_string(data: dict[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Le champ 'experiment.{field}' doit être une chaîne non vide.")
    return value.strip()


def _positive_int(data: dict[str, Any], field: str) -> int:
    value = data.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"Le champ 'experiment.{field}' doit être un entier positif.")
    return value


def _safe_relative_path(value: str, field: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ConfigError(f"experiment.{field} doit être un chemin relatif sûr.")
    return path


def _environment_root(variable_name: str) -> Path:
    raw_value = os.getenv(variable_name)
    if not raw_value:
        raise ConfigError(f"La variable obligatoire '{variable_name}' n'est pas définie.")
    return Path(raw_value).expanduser().resolve()


def load_baseline_settings(path: str | Path) -> BaselineSettings:
    """Load one baseline experiment without accepting training configuration."""

    config_path = Path(path)
    try:
        with config_path.open(encoding="utf-8") as stream:
            raw: object = yaml.safe_load(stream)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"Impossible de lire l'expérience baseline : {exc}") from exc
    root = _mapping(raw, "racine")
    experiment = _mapping(root.get("experiment"), "experiment")
    if _required_string(experiment, "mode") != "evaluation":
        raise ConfigError("La Phase 4A interdit tout mode autre que evaluation.")
    if _required_string(experiment, "language") != "dyu":
        raise ConfigError("La Phase 4A est limitée au dioula.")
    if _required_string(experiment, "split") != SPLIT_NAME:
        raise ConfigError("La baseline doit utiliser exclusivement le split test.")

    publication_allowed = experiment.get("publication_allowed")
    if publication_allowed is not False:
        raise ConfigError("experiment.publication_allowed doit rester false.")
    lowercase = experiment.get("lowercase")
    remove_punctuation = experiment.get("remove_punctuation")
    if not isinstance(lowercase, bool) or not isinstance(remove_punctuation, bool):
        raise ConfigError("Les politiques de normalisation doivent être booléennes.")
    timeout_value = experiment.get("timeout_seconds")
    if (
        not isinstance(timeout_value, (int, float))
        or isinstance(timeout_value, bool)
        or timeout_value <= 0
    ):
        raise ConfigError("experiment.timeout_seconds doit être strictement positif.")

    model_config_path = _safe_relative_path(
        _required_string(experiment, "model_config"),
        "model_config",
    )
    return BaselineSettings(
        experiment_id=_required_string(experiment, "id"),
        language="dyu",
        split=SPLIT_NAME,
        seed=_positive_int(experiment, "seed"),
        model_config_path=model_config_path,
        expected_model_id=_required_string(experiment, "expected_model_id"),
        manifest_path=_environment_root("IVOIREVOICE_ARTIFACTS_DIR")
        / _safe_relative_path(
            _required_string(experiment, "manifest_path"),
            "manifest_path",
        ),
        dataset_metadata_path=_environment_root("IVOIREVOICE_ARTIFACTS_DIR")
        / _safe_relative_path(
            _required_string(experiment, "dataset_metadata_path"),
            "dataset_metadata_path",
        ),
        dataset_root=_environment_root("IVOIREVOICE_DIOULA_DATA_DIR"),
        artifacts_root=_environment_root("IVOIREVOICE_ARTIFACTS_DIR"),
        output_relative_directory=_safe_relative_path(
            _required_string(experiment, "output_directory"),
            "output_directory",
        ),
        report_relative_directory=_safe_relative_path(
            _required_string(experiment, "report_directory"),
            "report_directory",
        ),
        expected_test_audio_count=_positive_int(
            experiment,
            "expected_test_audio_count",
        ),
        expected_test_speaker_count=_positive_int(
            experiment,
            "expected_test_speaker_count",
        ),
        smoke_per_speaker=_positive_int(experiment, "smoke_per_speaker"),
        pilot_per_speaker=_positive_int(experiment, "pilot_per_speaker"),
        timeout_seconds=float(timeout_value),
        lowercase=lowercase,
        remove_punctuation=remove_punctuation,
        publication_allowed=False,
    )


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            payload: object = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Impossible de lire {description} : {exc}") from exc
    return _mapping(payload, description)


def _safe_manifest_audio_path(value: str) -> bool:
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


def load_test_dataset(settings: BaselineSettings) -> BaselineDataset:
    """Validate governance and load only test rows from the frozen manifest."""

    metadata = _load_json(settings.dataset_metadata_path, "les métadonnées v0.1")
    manifest_hash = sha256_file(settings.manifest_path)
    if metadata.get("manifest_sha256") != manifest_hash:
        raise ConfigError("Le manifeste ne correspond plus aux métadonnées v0.1.")
    if metadata.get("publication_allowed") is not False:
        raise ConfigError("La publication du dataset doit rester interdite.")
    if metadata.get("model_derivative_publication_allowed") is not False:
        raise ConfigError("La publication d'un modèle dérivé doit rester interdite.")
    if metadata.get("usage_scope") != "local_research_only":
        raise ConfigError("La baseline exige usage_scope=local_research_only.")
    dataset_version = metadata.get("dataset_version")
    if not isinstance(dataset_version, str):
        raise ConfigError("La version du dataset est absente des métadonnées.")

    try:
        with settings.manifest_path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {
                "utterance_id",
                "speaker_id",
                "gender_folder",
                "audio_path",
                "audio_sha256",
                "duration_seconds",
                "target_text_mvp",
                "split",
                "usage_scope",
            }
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ConfigError(
                    "Colonnes absentes du manifeste v0.1 : " + ", ".join(sorted(missing))
                )
            items = tuple(
                BaselineItem(
                    utterance_id=row["utterance_id"],
                    speaker_id=row["speaker_id"],
                    gender_folder=row["gender_folder"],
                    audio_path=row["audio_path"],
                    audio_sha256=row["audio_sha256"],
                    audio_duration_seconds=float(row["duration_seconds"]),
                    reference_raw=row["target_text_mvp"],
                )
                for row in reader
                if row["split"] == settings.split and row["usage_scope"] == "local_research_only"
            )
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        raise ConfigError(f"Impossible de charger le split test : {exc}") from exc

    if len(items) != settings.expected_test_audio_count:
        raise ConfigError("Le nombre d'audios test ne correspond pas au gel v0.1.")
    if len({item.speaker_id for item in items}) != settings.expected_test_speaker_count:
        raise ConfigError("Le nombre de locuteurs test ne correspond pas au gel v0.1.")
    if len({item.utterance_id for item in items}) != len(items):
        raise ConfigError("Le split test contient des utterance_id dupliqués.")
    if len({item.audio_path for item in items}) != len(items):
        raise ConfigError("Le split test contient des chemins audio dupliqués.")
    if any(
        not _safe_manifest_audio_path(item.audio_path)
        or not item.reference_raw
        or FALLING_MARKER in item.reference_raw
        or item.audio_duration_seconds <= 0
        for item in items
    ):
        raise ConfigError("Une ligne test viole la cible ou les règles de confidentialité.")
    return BaselineDataset(
        items=tuple(sorted(items, key=lambda item: item.utterance_id)),
        manifest_sha256=manifest_hash,
        dataset_version=dataset_version,
    )


def _duration_spread(
    items: list[BaselineItem],
    count: int,
    seed: int,
) -> tuple[BaselineItem, ...]:
    if len(items) < count:
        raise ConfigError("Un locuteur ne contient pas assez d'audios pour ce niveau.")
    ordered = sorted(
        items,
        key=lambda item: (
            item.audio_duration_seconds,
            sha256(f"{seed}:{item.utterance_id}".encode()).hexdigest(),
        ),
    )
    if count == 1:
        return (ordered[0],)
    indices = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    return tuple(ordered[index] for index in indices)


def select_evaluation_items(
    items: tuple[BaselineItem, ...],
    *,
    level: EvaluationLevel,
    seed: int,
    smoke_per_speaker: int = 2,
    pilot_per_speaker: int = 50,
) -> tuple[BaselineItem, ...]:
    """Select identical deterministic test subsets for every compared model."""

    grouped: dict[str, list[BaselineItem]] = defaultdict(list)
    for item in items:
        grouped[item.speaker_id].append(item)
    selected: list[BaselineItem] = []
    for speaker_id, speaker_items in sorted(grouped.items()):
        if level == "smoke":
            chosen = tuple(
                sorted(
                    speaker_items,
                    key=lambda item: (
                        item.audio_duration_seconds,
                        item.utterance_id,
                    ),
                )[:smoke_per_speaker]
            )
            if len(chosen) != smoke_per_speaker:
                raise ConfigError(f"Locuteur test incomplet pour le smoke : {speaker_id}.")
        elif level == "pilot":
            chosen = _duration_spread(speaker_items, pilot_per_speaker, seed)
        else:
            chosen = tuple(sorted(speaker_items, key=lambda item: item.utterance_id))
        selected.extend(chosen)
    return tuple(sorted(selected, key=lambda item: (item.speaker_id, item.utterance_id)))


def normalize_evaluation_text(
    text: str,
    *,
    lowercase: bool,
    remove_punctuation: bool,
) -> str:
    """Apply the same controlled evaluation policy to references and predictions."""

    normalized = unicodedata.normalize("NFC", text.replace(FALLING_MARKER, ""))
    if lowercase:
        normalized = normalized.casefold()
    if remove_punctuation:
        normalized = "".join(
            (
                " "
                if unicodedata.category(character).startswith("P") and character not in {"'", "’"}
                else character
            )
            for character in normalized
        )
    return normalize_transcription(normalized)


def require_full_confirmation(level: EvaluationLevel, confirmation: str | None) -> None:
    """Refuse full evaluation unless the caller supplies an explicit confirmation."""

    if level == "full" and confirmation != "1":
        raise ConfigError("L'évaluation complète exige CONFIRM_FULL=1 après examen du pilote.")


def completed_utterance_ids(path: Path) -> set[str]:
    """Read progressive completion state without exposing private contents."""

    if not path.exists():
        return set()
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if "utterance_id" not in (reader.fieldnames or []):
                raise ConfigError("Le fichier de reprise ne contient pas utterance_id.")
            identifiers = [row["utterance_id"] for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ConfigError(f"Impossible de lire l'état de reprise : {exc}") from exc
    if len(identifiers) != len(set(identifiers)):
        raise ConfigError("Le fichier de reprise contient des résultats dupliqués.")
    return set(identifiers)


def pending_items(
    selected: tuple[BaselineItem, ...],
    completed_ids: set[str],
) -> tuple[BaselineItem, ...]:
    """Return only items not durably recorded by an earlier run."""

    return tuple(item for item in selected if item.utterance_id not in completed_ids)


def _selection_hash(items: tuple[BaselineItem, ...]) -> str:
    material = "\n".join(f"{item.utterance_id},{item.audio_sha256}" for item in items)
    return sha256(material.encode("utf-8")).hexdigest()


def _git_commit_sha() -> str:
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
        raise ConfigError("Le commit du pipeline est invalide.")
    return commit_sha


def _git_is_dirty() -> bool:
    repository_root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ConfigError(f"Impossible de vérifier l'état Git du pipeline : {exc}") from exc
    return bool(result.stdout.strip())


def _pipeline_source_sha256() -> str:
    repository_root = Path(__file__).resolve().parents[3]
    source_root = repository_root / "src" / "ivoirevoice"
    digest = sha256()
    try:
        source_files = sorted(source_root.rglob("*.py"))
        if not source_files:
            raise ConfigError("Aucun fichier source du pipeline n'a été trouvé.")
        for source_path in source_files:
            relative_path = source_path.relative_to(repository_root).as_posix()
            digest.update(relative_path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(source_path.read_bytes())
            digest.update(b"\0")
    except OSError as exc:
        raise ConfigError(f"Impossible de calculer l'empreinte du pipeline : {exc}") from exc
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        with temporary_path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        temporary_path.replace(path)
    except OSError as exc:
        raise ConfigError(f"Impossible d'écrire {path.name} : {exc}") from exc


def _write_text(path: Path, content: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(path)
    except OSError as exc:
        raise ConfigError(f"Impossible d'écrire {path.name} : {exc}") from exc


def _append_prediction(path: Path, record: PredictionRecord) -> None:
    fieldnames = [field.name for field in fields(PredictionRecord)]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=fieldnames,
                lineterminator="\n",
            )
            if write_header:
                writer.writeheader()
            writer.writerow(asdict(record))
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise ConfigError(f"Impossible d'enregistrer une prédiction privée : {exc}") from exc


def _load_predictions(path: Path) -> tuple[PredictionRecord, ...]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {field.name for field in fields(PredictionRecord)}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ConfigError(
                    "Colonnes absentes des prédictions : " + ", ".join(sorted(missing))
                )
            records = tuple(
                PredictionRecord(
                    utterance_id=row["utterance_id"],
                    speaker_id=row["speaker_id"],
                    gender_folder=row["gender_folder"],
                    audio_path=row["audio_path"],
                    audio_sha256=row["audio_sha256"],
                    audio_duration_seconds=float(row["audio_duration_seconds"]),
                    reference_raw=row["reference_raw"],
                    reference_normalized=row["reference_normalized"],
                    prediction_raw=row["prediction_raw"],
                    prediction_normalized=row["prediction_normalized"],
                    processing_time_seconds=float(row["processing_time_seconds"]),
                    model_name=row["model_name"],
                    status=row["status"],
                    error_type=row["error_type"],
                )
                for row in reader
            )
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        raise ConfigError(f"Impossible de relire les prédictions privées : {exc}") from exc
    return records


def _scored_items(records: tuple[PredictionRecord, ...]) -> tuple[ScoredItem, ...]:
    return tuple(
        ScoredItem(
            speaker_id=record.speaker_id,
            reference_normalized=record.reference_normalized,
            prediction_normalized=record.prediction_normalized,
            audio_duration_seconds=record.audio_duration_seconds,
            processing_time_seconds=record.processing_time_seconds,
            error_type=record.error_type,
        )
        for record in records
    )


def _write_speaker_metrics(path: Path, speaker_metrics: dict[str, Any]) -> None:
    rows = [
        {"speaker_id": speaker_id, **metrics} for speaker_id, metrics in speaker_metrics.items()
    ]
    fieldnames = list(rows[0]) if rows else ["speaker_id"]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        with temporary_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=fieldnames,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(path)
    except OSError as exc:
        raise ConfigError(f"Impossible d'écrire les métriques locuteur : {exc}") from exc


def _private_error_analysis(
    records: tuple[PredictionRecord, ...],
    metrics: dict[str, Any],
) -> str:
    categories: Counter[str] = Counter()
    for record in records:
        if record.error_type:
            categories["inference_failure"] += 1
        elif not record.prediction_normalized:
            categories["empty_output"] += 1
        if record.audio_duration_seconds >= 15:
            categories["long_audio"] += 1
        tokens = record.prediction_normalized.split()
        if len(tokens) >= 4 and len(set(tokens[-4:])) == 1:
            categories["apparent_repetition"] += 1
    examples = [
        record for record in records if record.error_type or not record.prediction_normalized
    ][:10]
    lines = [
        "# Analyse privée des erreurs",
        "",
        "Ce fichier contient des extraits privés et ne doit pas être publié.",
        "",
        f"- substitutions de mots : {metrics['word_substitutions']}",
        f"- suppressions de mots : {metrics['word_deletions']}",
        f"- insertions de mots : {metrics['word_insertions']}",
    ]
    lines.extend(f"- {category} : {count}" for category, count in sorted(categories.items()))
    lines.extend(["", "## Exemples locaux", ""])
    for record in examples:
        lines.extend(
            [
                f"- statut : `{record.status}` / `{record.error_type or 'none'}`",
                f"  - référence : `{record.reference_raw[:120]}`",
                f"  - prédiction : `{record.prediction_raw[:120]}`",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def shareable_summary(
    *,
    experiment_id: str,
    level: EvaluationLevel,
    model_id: str,
    model_revision: str,
    metrics: dict[str, Any],
    manifest_hash: str,
) -> str:
    """Render an aggregate-only report without paths or transcriptions."""

    return f"""# Baseline ASR dioula — {experiment_id} — {level}

- modèle : `{model_id}`
- révision : `{model_revision}`
- niveau : `{level}`
- manifeste SHA-256 : `{manifest_hash}`
- audios évalués : {metrics["evaluated_audio_count"]}
- échecs : {metrics["failed_audio_count"]}
- WER micro : {metrics["wer_micro"]}
- CER micro : {metrics["cer_micro"]}
- WER macro locuteurs : {metrics["wer_macro_speakers"]}
- CER macro locuteurs : {metrics["cer_macro_speakers"]}
- latence moyenne (s) : {metrics["mean_latency_seconds"]}
- latence p50 (s) : {metrics["latency_p50_seconds"]}
- latence p95 (s) : {metrics["latency_p95_seconds"]}
- RTF : {metrics["rtf"]}

Aucune transcription, prédiction, URL ou chemin local n'est inclus dans cette
synthèse. Les sorties privées restent hors Git et la publication d'un modèle
dérivé demeure interdite.
"""


def _validate_shareable(content: str, records: tuple[PredictionRecord, ...]) -> None:
    if "/home/" in content or "\\Users\\" in content or "://" in content:
        raise ConfigError("La synthèse partageable contient un chemin ou une URL.")
    for record in records:
        for private_text in (record.reference_raw, record.prediction_raw):
            if private_text and len(private_text) >= 8 and private_text in content:
                raise ConfigError("La synthèse partageable contient une transcription.")


def _run_metadata(
    settings: BaselineSettings,
    model: WhisperSettings,
    dataset: BaselineDataset,
    selected: tuple[BaselineItem, ...],
    level: EvaluationLevel,
) -> dict[str, Any]:
    effective_device, effective_dtype = runtime_labels(model)
    return {
        "experiment_id": settings.experiment_id,
        "level": level,
        "language": settings.language,
        "split": settings.split,
        "seed": settings.seed,
        "dataset_version": dataset.dataset_version,
        "manifest_sha256": dataset.manifest_sha256,
        "selection_sha256": _selection_hash(selected),
        "selected_audio_count": len(selected),
        "selected_speaker_count": len({item.speaker_id for item in selected}),
        "model_id": model.model_id,
        "model_revision": model.model_revision,
        "configured_device": model.device,
        "configured_torch_dtype": model.torch_dtype,
        "device": effective_device,
        "torch_dtype": effective_dtype,
        "batch_size": model.batch_size,
        "chunk_length_seconds": model.chunk_length_seconds,
        "stride_length_seconds": model.stride_length_seconds,
        "task": model.task,
        "forced_language": model.language,
        "timeout_seconds": settings.timeout_seconds,
        "pipeline_commit_sha": _git_commit_sha(),
        "pipeline_git_dirty": _git_is_dirty(),
        "pipeline_source_sha256": _pipeline_source_sha256(),
        "publication_allowed": False,
        "model_derivative_publication_allowed": False,
        "status": "in_progress",
    }


def _validate_resume_metadata(
    path: Path,
    expected: dict[str, Any],
) -> dict[str, Any]:
    if not path.exists():
        return expected
    existing = _load_json(path, "les métadonnées de reprise")
    immutable_fields = (
        "experiment_id",
        "level",
        "seed",
        "manifest_sha256",
        "selection_sha256",
        "model_id",
        "model_revision",
        "configured_device",
        "configured_torch_dtype",
        "batch_size",
        "chunk_length_seconds",
        "stride_length_seconds",
        "task",
        "forced_language",
        "pipeline_source_sha256",
    )
    for field_name in immutable_fields:
        if existing.get(field_name) != expected.get(field_name):
            raise ConfigError(f"La reprise est incompatible avec le champ {field_name}.")
    return existing


def _build_record(
    item: BaselineItem,
    settings: BaselineSettings,
    backend: WhisperBackend,
) -> PredictionRecord:
    reference_normalized = normalize_evaluation_text(
        item.reference_raw,
        lowercase=settings.lowercase,
        remove_punctuation=settings.remove_punctuation,
    )
    try:
        result = backend.transcribe(
            settings.dataset_root / item.audio_path,
            settings.language,
        )
        if result.processing_time_seconds > settings.timeout_seconds:
            raise TimeoutError("soft_timeout")
        prediction_raw = result.text
        prediction_normalized = normalize_evaluation_text(
            prediction_raw,
            lowercase=settings.lowercase,
            remove_punctuation=settings.remove_punctuation,
        )
        status = "success"
        error_type = ""
        processing_time = result.processing_time_seconds
        model_name = result.model_name
    except Exception as exc:  # noqa: BLE001 - isolate each private audio failure
        prediction_raw = ""
        prediction_normalized = ""
        status = "error"
        error_type = type(exc).__name__
        processing_time = 0.0
        model_name = backend.model_name
    return PredictionRecord(
        utterance_id=item.utterance_id,
        speaker_id=item.speaker_id,
        gender_folder=item.gender_folder,
        audio_path=item.audio_path,
        audio_sha256=item.audio_sha256,
        audio_duration_seconds=item.audio_duration_seconds,
        reference_raw=item.reference_raw,
        reference_normalized=reference_normalized,
        prediction_raw=prediction_raw,
        prediction_normalized=prediction_normalized,
        processing_time_seconds=processing_time,
        model_name=model_name,
        status=status,
        error_type=error_type,
    )


def run_baseline(
    settings: BaselineSettings,
    *,
    level: EvaluationLevel,
    backend_factory: BackendFactory | None = None,
    full_confirmation: str | None = None,
) -> dict[str, Any]:
    """Run or resume one local baseline and generate private/shareable outputs."""

    require_full_confirmation(level, full_confirmation)
    dataset = load_test_dataset(settings)
    selected = select_evaluation_items(
        dataset.items,
        level=level,
        seed=settings.seed,
        smoke_per_speaker=settings.smoke_per_speaker,
        pilot_per_speaker=settings.pilot_per_speaker,
    )
    model = load_whisper_settings(settings.model_config_path)
    if model.model_id != settings.expected_model_id:
        raise ConfigError("Le modèle chargé ne correspond pas à l'expérience.")

    run_id = f"{settings.experiment_id}-{level}"
    output_directory = settings.artifacts_root / settings.output_relative_directory / run_id
    predictions_path = output_directory / "predictions_private.csv"
    metadata_path = output_directory / "run_metadata.json"
    expected_metadata = _run_metadata(settings, model, dataset, selected, level)
    resume_metadata = _validate_resume_metadata(metadata_path, expected_metadata)
    if not metadata_path.exists():
        _atomic_json(metadata_path, resume_metadata)

    completed_ids = completed_utterance_ids(predictions_path)
    selected_ids = {item.utterance_id for item in selected}
    if not completed_ids.issubset(selected_ids):
        raise ConfigError("Le fichier de reprise contient un audio hors sélection.")
    remaining = pending_items(selected, completed_ids)
    if remaining:
        factory = backend_factory or (lambda config: WhisperBackend(config))
        backend = factory(model)
        backend.load()
        try:
            for item in remaining:
                audio_path = settings.dataset_root / item.audio_path
                if not audio_path.is_file():
                    raise ConfigError("Un audio sélectionné est introuvable localement.")
                _append_prediction(
                    predictions_path,
                    _build_record(item, settings, backend),
                )
        finally:
            backend.unload()

    records = _load_predictions(predictions_path)
    if len(records) != len(selected):
        raise ConfigError("La baseline n'a pas enregistré toute la sélection.")
    if {record.utterance_id for record in records} != selected_ids:
        raise ConfigError("Les prédictions ne correspondent pas à la sélection.")
    metrics = compute_evaluation_metrics(_scored_items(records))
    effective_device, effective_dtype = runtime_labels(model)
    metrics.update(
        {
            "experiment_id": settings.experiment_id,
            "level": level,
            "model_id": model.model_id,
            "model_revision": model.model_revision,
            "configured_device": model.device,
            "configured_torch_dtype": model.torch_dtype,
            "device": effective_device,
            "torch_dtype": effective_dtype,
            "batch_size": model.batch_size,
            "seed": settings.seed,
            "manifest_sha256": dataset.manifest_sha256,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "pipeline_commit_sha": _git_commit_sha(),
            "pipeline_git_dirty": _git_is_dirty(),
            "pipeline_source_sha256": _pipeline_source_sha256(),
        }
    )
    full_audio_duration = sum(item.audio_duration_seconds for item in dataset.items)
    metrics["estimated_full_processing_seconds"] = metrics["rtf"] * full_audio_duration
    _atomic_json(output_directory / "metrics.json", metrics)
    _write_speaker_metrics(
        output_directory / "speaker_metrics.csv",
        metrics["speaker_metrics"],
    )
    _atomic_json(
        output_directory / "latency.json",
        {
            "mean_seconds": metrics["mean_latency_seconds"],
            "p50_seconds": metrics["latency_p50_seconds"],
            "p95_seconds": metrics["latency_p95_seconds"],
            "total_processing_seconds": metrics["processing_time_seconds"],
            "audio_duration_seconds": metrics["audio_duration_seconds"],
            "rtf": metrics["rtf"],
        },
    )
    _write_text(
        output_directory / "error_analysis_private.md",
        _private_error_analysis(records, metrics),
    )
    successful_audio_count = metrics["successful_audio_count"]
    completed_metadata = {
        **expected_metadata,
        "status": "completed" if successful_audio_count else "failed",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "successful_audio_count": successful_audio_count,
        "failed_audio_count": metrics["failed_audio_count"],
    }
    _atomic_json(metadata_path, completed_metadata)
    if not successful_audio_count:
        raise ConfigError("Toutes les transcriptions ont échoué ; consultez les artefacts privés.")

    summary = shareable_summary(
        experiment_id=settings.experiment_id,
        level=level,
        model_id=model.model_id,
        model_revision=model.model_revision,
        metrics=metrics,
        manifest_hash=dataset.manifest_sha256,
    )
    _validate_shareable(summary, records)
    _write_text(
        settings.artifacts_root / settings.report_relative_directory / f"{run_id}_summary.md",
        summary,
    )
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exécuter une baseline ASR dioula.")
    parser.add_argument("--experiment", required=True)
    parser.add_argument(
        "--level",
        choices=("smoke", "pilot", "full"),
        required=True,
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point with aggregate-only output."""

    args = _parse_args()
    level = cast(EvaluationLevel, args.level)
    try:
        settings = load_baseline_settings(args.experiment)
        metrics = run_baseline(
            settings,
            level=level,
            full_confirmation=os.getenv("IVOIREVOICE_CONFIRM_FULL"),
        )
    except IvoireVoiceError as exc:
        print(f"ERREUR: {exc}")
        return 1
    print(f"experiment_id={metrics['experiment_id']}")
    print(f"level={metrics['level']}")
    print(f"evaluated_audio_count={metrics['evaluated_audio_count']}")
    print(f"failed_audio_count={metrics['failed_audio_count']}")
    print(f"wer_micro={metrics['wer_micro']}")
    print(f"cer_micro={metrics['cer_micro']}")
    print(f"wer_macro_speakers={metrics['wer_macro_speakers']}")
    print(f"cer_macro_speakers={metrics['cer_macro_speakers']}")
    print(f"latency_p50_seconds={metrics['latency_p50_seconds']}")
    print(f"latency_p95_seconds={metrics['latency_p95_seconds']}")
    print(f"rtf={metrics['rtf']}")
    print(f"estimated_full_processing_seconds={metrics['estimated_full_processing_seconds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
