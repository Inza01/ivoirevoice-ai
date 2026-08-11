"""Guarded validation-only review of a terminal development checkpoint."""

from __future__ import annotations

import csv
import gc
import hashlib
import importlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from ivoirevoice.data.audio import sha256_file
from ivoirevoice.exceptions import ConfigError
from ivoirevoice.models.whisper import load_whisper_settings
from ivoirevoice.training.audit import (
    REQUIRED_COLUMNS,
    ManifestRow,
    _parse_manifest_row,
)
from ivoirevoice.training.full_finetune import (
    DEVELOPMENT_DECISION_FILENAME,
    FINAL_EVALUATION_RECEIPT_FILENAME,
    FINAL_MODEL_MANIFEST_FILENAME,
    PREFLIGHT_REPORT_FILENAME,
    REFIT_STARTED_FILENAME,
    TRAINER_STATE_FILENAME,
    _latest_checkpoint,
    _processor_and_collator,
    _require_cuda,
)
from ivoirevoice.training.full_settings import FullTrainingSettings
from ivoirevoice.training.pilot_finetune import (
    _evaluate_validation,
    _load_model,
    _seed_everything,
)
from ivoirevoice.training.whisper_finetune import (
    directory_sha256,
    git_provenance,
    identity_sha256,
    load_json_object,
    metric_rank,
    refit_step_budget,
    require_matching_identity,
    selection_sha256,
    write_json_atomic,
)

CANDIDATE_PRIVATE_FILENAME = "development_final_validation_candidate_private.json"
CANDIDATE_PUBLIC_FILENAME = "development_final_validation_candidate.json"
METRIC_POLICY = (
    "wer_micro",
    "cer_micro",
    "validation_loss",
    "earliest_step",
)
PROCESSOR_FILES = (
    "added_tokens.json",
    "merges.txt",
    "normalizer.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
MODEL_DECODING_FILES = ("config.json", "generation_config.json")


@dataclass(frozen=True, slots=True)
class TerminalValidationContext:
    """Validation-only rows and immutable evaluation provenance."""

    validation_rows: tuple[ManifestRow, ...]
    validation_fingerprint: str
    manifest_sha256: str
    config_sha256: str
    pilot_checkpoint_sha256: str
    evaluation_git_commit: str
    model_settings: Any


@dataclass(frozen=True, slots=True)
class DevelopmentArtifacts:
    """Frozen decision and the two checkpoints eligible for comparison."""

    decision: dict[str, Any]
    decision_sha256: str
    current_checkpoint: Path
    current_state: dict[str, Any]
    current_checkpoint_sha256: str
    candidate_checkpoint: Path
    candidate_state: dict[str, Any]
    candidate_checkpoint_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ConfigError(f"Impossible de calculer l'empreinte de {path.name}.") from exc
    return digest.hexdigest()


def _load_metadata(path: Path) -> dict[str, Any]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigError("Métadonnées dataset illisibles pour la validation terminale.") from exc
    if not isinstance(raw, Mapping) or not all(isinstance(key, str) for key in raw):
        raise ConfigError("Métadonnées dataset invalides pour la validation terminale.")
    return cast(dict[str, Any], dict(raw))


def load_terminal_validation_rows(
    settings: FullTrainingSettings,
) -> tuple[tuple[ManifestRow, ...], str, str]:
    """Materialize validation rows only; never construct a test/holdout row."""

    manifest_sha256 = sha256_file(settings.manifest_path)
    if manifest_sha256 != settings.expected_manifest_sha256:
        raise ConfigError("Le hash du manifeste n'est plus approuvé.")
    metadata = _load_metadata(settings.dataset_metadata_path)
    expected_metadata = {
        "manifest_sha256": manifest_sha256,
        "dataset_version": settings.expected_dataset_version,
        "publication_allowed": False,
        "model_derivative_publication_allowed": False,
        "usage_scope": "local_research_only",
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise ConfigError(f"Métadonnée validation inattendue : {field}.")
    counts = metadata.get("audio_count_by_split")
    if not isinstance(counts, Mapping) or counts.get(settings.validation_split) != (
        settings.validation_audio_count
    ):
        raise ConfigError("Le compte validation agrégé a changé.")

    validation_rows: list[ManifestRow] = []
    try:
        with settings.manifest_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or ()))
            if missing:
                raise ConfigError(f"Colonnes validation manquantes : {missing}.")
            for line_number, raw in enumerate(reader, 2):
                if raw.get("split") != settings.validation_split:
                    continue
                validation_rows.append(_parse_manifest_row(raw, line_number))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ConfigError("Impossible de lire la validation gelée.") from exc

    rows = tuple(validation_rows)
    observed = {
        "audio_count": len(rows),
        "speaker_count": len({row.speaker_id for row in rows}),
    }
    expected = {
        "audio_count": settings.validation_audio_count,
        "speaker_count": settings.validation_speaker_count,
    }
    if observed != expected:
        raise ConfigError(f"Comptes validation inattendus : {observed}.")
    if len({row.utterance_id for row in rows}) != len(rows):
        raise ConfigError("La validation contient un utterance_id dupliqué.")
    if len({row.audio_sha256 for row in rows}) != len(rows):
        raise ConfigError("La validation contient un audio dupliqué.")
    for row in rows:
        if (
            row.split != settings.validation_split
            or row.language != "dyu"
            or row.usage_scope != "local_research_only"
            or not row.target_text.strip()
            or not 0 < row.duration_seconds <= settings.max_audio_seconds
            or row.sample_rate_hz != 16_000
            or row.channels != 1
        ):
            raise ConfigError("Une ligne validation est incompatible avec le protocole.")
        audio_path = settings.dataset_root / row.audio_path
        if not audio_path.is_file():
            raise ConfigError("Un audio validation attendu est absent.")
        if sha256_file(audio_path) != row.audio_sha256:
            raise ConfigError("Un audio validation ne correspond plus au manifeste.")
    fingerprint = selection_sha256(
        {"validation": tuple((row.utterance_id, row.audio_sha256) for row in rows)}
    )
    return rows, manifest_sha256, fingerprint


def _refit_has_started(settings: FullTrainingSettings) -> bool:
    if (settings.artifact_output_directory / REFIT_STARTED_FILENAME).is_file() or (
        settings.artifact_output_directory / FINAL_MODEL_MANIFEST_FILENAME
    ).is_file():
        return True
    directory = settings.refit_checkpoint_directory
    return directory.is_dir() and any(directory.iterdir())


def _require_stage_available(settings: FullTrainingSettings) -> None:
    if (settings.artifact_output_directory / FINAL_EVALUATION_RECEIPT_FILENAME).exists():
        raise ConfigError("Le final_holdout a déjà été ouvert : validation refusée.")
    if _refit_has_started(settings):
        raise ConfigError("Le refit a déjà démarré : validation terminale refusée.")


def _load_context(settings: FullTrainingSettings) -> TerminalValidationContext:
    root = Path(__file__).resolve().parents[3]
    evaluation_git_commit, clean = git_provenance(root)
    if not clean:
        raise ConfigError("La validation terminale exige un commit Git propre.")
    validation_rows, manifest_sha256, validation_fingerprint = load_terminal_validation_rows(
        settings
    )
    model_settings = load_whisper_settings(root / settings.model_config_path)
    if (
        model_settings.model_id != settings.expected_model_id
        or model_settings.model_revision != settings.expected_model_revision
        or model_settings.task != "transcribe"
        or model_settings.language is not None
    ):
        raise ConfigError("Le modèle de validation ne correspond plus au protocole.")
    return TerminalValidationContext(
        validation_rows=validation_rows,
        validation_fingerprint=validation_fingerprint,
        manifest_sha256=manifest_sha256,
        config_sha256=_sha256_file(settings.config_path),
        pilot_checkpoint_sha256=directory_sha256(settings.initial_checkpoint_path),
        evaluation_git_commit=evaluation_git_commit,
        model_settings=model_settings,
    )


def _require_current_preflight(
    settings: FullTrainingSettings,
    context: TerminalValidationContext,
    decision: Mapping[str, Any],
) -> None:
    preflight = load_json_object(settings.artifact_output_directory / PREFLIGHT_REPORT_FILENAME)
    require_matching_identity(
        preflight,
        {
            "status": "passed",
            "code_commit": context.evaluation_git_commit,
            "config_sha256": context.config_sha256,
            "manifest_sha256": context.manifest_sha256,
            "initial_checkpoint_sha256": context.pilot_checkpoint_sha256,
            "validation_audio_count": len(context.validation_rows),
            "development_selection_sha256": decision.get("development_selection_sha256"),
            "historical_test_decoded": False,
            "final_holdout_decoded": False,
        },
        description="Le préflight de validation terminale",
    )


def _state_run_identity(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": state.get("stage"),
        "selection_sha256": state.get("selection_sha256"),
        "initial_checkpoint_sha256": state.get("initial_checkpoint_sha256"),
        "config_sha256": state.get("config_sha256"),
        "code_commit": state.get("code_commit"),
    }


def _load_development_artifacts(
    settings: FullTrainingSettings,
    context: TerminalValidationContext,
) -> DevelopmentArtifacts:
    decision_path = settings.artifact_output_directory / DEVELOPMENT_DECISION_FILENAME
    decision = load_json_object(decision_path)
    if decision.get("status") != "frozen":
        raise ConfigError("Le développement doit être frozen avant cette validation.")
    require_matching_identity(
        decision,
        {
            "config_sha256": context.config_sha256,
            "manifest_sha256": context.manifest_sha256,
            "initial_checkpoint_sha256": context.pilot_checkpoint_sha256,
            "historical_test_used": False,
            "final_holdout_used": False,
        },
        description="La décision de développement gelée",
    )
    _require_current_preflight(settings, context, decision)

    current_name = decision.get("best_checkpoint_name")
    if not isinstance(current_name, str) or not current_name.startswith("checkpoint-"):
        raise ConfigError("Le meilleur checkpoint gelé est invalide.")
    current_checkpoint = settings.development_checkpoint_directory / current_name
    candidate_checkpoint = _latest_checkpoint(settings.development_checkpoint_directory)
    if candidate_checkpoint is None:
        raise ConfigError("Le checkpoint terminal de développement est absent.")
    current_state = load_json_object(current_checkpoint / TRAINER_STATE_FILENAME)
    candidate_state = load_json_object(candidate_checkpoint / TRAINER_STATE_FILENAME)

    current_step = int(decision.get("best_development_step", 0))
    candidate_step = int(candidate_state.get("global_step", 0))
    if current_checkpoint.name != f"checkpoint-{current_step:06d}":
        raise ConfigError("Le checkpoint sélectionné ne correspond pas à son step.")
    if candidate_checkpoint.name != f"checkpoint-{candidate_step:06d}":
        raise ConfigError("Le checkpoint terminal ne correspond pas à son step.")
    if candidate_step <= current_step or candidate_state.get("completed") is not True:
        raise ConfigError("Le checkpoint candidat n'est pas l'état terminal ultérieur.")

    expected_run = {
        "stage": "development",
        "selection_sha256": decision.get("development_selection_sha256"),
        "initial_checkpoint_sha256": context.pilot_checkpoint_sha256,
        "config_sha256": context.config_sha256,
        "code_commit": decision.get("code_commit"),
    }
    expected_run_id = identity_sha256(expected_run)
    for label, state in (("sélectionné", current_state), ("terminal", candidate_state)):
        require_matching_identity(
            state,
            {**expected_run, "run_id": expected_run_id},
            description=f"Le checkpoint {label}",
        )
    if _state_run_identity(current_state) != _state_run_identity(candidate_state):
        raise ConfigError("Les checkpoints ne proviennent pas du même run.")

    current_hash = directory_sha256(current_checkpoint)
    if current_hash != decision.get("best_checkpoint_sha256"):
        raise ConfigError("Le checkpoint sélectionné ne correspond plus à la décision.")
    return DevelopmentArtifacts(
        decision=decision,
        decision_sha256=_sha256_file(decision_path),
        current_checkpoint=current_checkpoint,
        current_state=current_state,
        current_checkpoint_sha256=current_hash,
        candidate_checkpoint=candidate_checkpoint,
        candidate_state=candidate_state,
        candidate_checkpoint_sha256=directory_sha256(candidate_checkpoint),
    )


def _matching_file_hashes(paths: Sequence[Path], filenames: Sequence[str]) -> dict[str, str]:
    reference: dict[str, str] | None = None
    for path in paths:
        observed: dict[str, str] = {}
        for filename in filenames:
            file_path = path / filename
            if not file_path.is_file():
                raise ConfigError(f"Le checkpoint ne contient pas {filename}.")
            observed[filename] = _sha256_file(file_path)
        if reference is None:
            reference = observed
        elif observed != reference:
            raise ConfigError("Le processor ou le décodage diffère entre checkpoints.")
    return reference or {}


def _decoding_config_hash(
    settings: FullTrainingSettings,
    artifacts: DevelopmentArtifacts,
) -> str:
    checkpoints = (
        settings.initial_checkpoint_path,
        artifacts.current_checkpoint,
        artifacts.candidate_checkpoint,
    )
    processor_hashes = _matching_file_hashes(checkpoints, PROCESSOR_FILES)
    model_hashes = _matching_file_hashes(checkpoints, MODEL_DECODING_FILES)
    evaluator_path = Path(__file__).with_name("pilot_finetune.py")
    return identity_sha256(
        {
            "task": "transcribe",
            "do_sample": False,
            "max_new_tokens": 128,
            "normalization": "lowercase_remove_punctuation",
            "post_correction": False,
            "eval_batch_size": settings.eval_batch_size,
            "fp16": settings.fp16,
            "processor_files": processor_hashes,
            "model_decoding_files": model_hashes,
            "evaluator_source_sha256": _sha256_file(evaluator_path),
        }
    )


def _evaluation_identity(
    context: TerminalValidationContext,
    artifacts: DevelopmentArtifacts,
    *,
    decoding_config_hash: str,
    torch_version: str,
    transformers_version: str,
) -> dict[str, Any]:
    return {
        "stage": "development-final-validation",
        "development_run_id": artifacts.candidate_state["run_id"],
        "development_git_commit": artifacts.candidate_state["code_commit"],
        "evaluation_git_commit": context.evaluation_git_commit,
        "config_hash": context.config_sha256,
        "pilot_checkpoint_hash": context.pilot_checkpoint_sha256,
        "current_checkpoint_hash": artifacts.current_checkpoint_sha256,
        "candidate_checkpoint_hash": artifacts.candidate_checkpoint_sha256,
        "validation_manifest_hash": context.manifest_sha256,
        "validation_fingerprint": context.validation_fingerprint,
        "decoding_config_hash": decoding_config_hash,
        "torch_version": torch_version,
        "transformers_version": transformers_version,
        "metric_policy": list(METRIC_POLICY),
    }


def _aggregate_metrics(
    metrics: Mapping[str, Any],
    *,
    audio_count: int,
    speaker_count: int,
) -> dict[str, Any]:
    if int(metrics.get("evaluated_audio_count", 0)) != audio_count:
        raise ConfigError("L'évaluation n'a pas couvert toute la validation.")
    return {
        "validation_loss": float(metrics["validation_loss"]),
        "wer_micro": float(metrics["wer_micro"]),
        "cer_micro": float(metrics["cer_micro"]),
        "rtf": float(metrics["rtf"]),
        "word_substitutions": int(metrics["word_substitutions"]),
        "word_insertions": int(metrics["word_insertions"]),
        "word_deletions": int(metrics["word_deletions"]),
        "audio_count": audio_count,
        "speaker_count": speaker_count,
    }


def build_candidate_report(
    *,
    identity: Mapping[str, Any],
    current_name: str,
    current_step: int,
    current_metrics: Mapping[str, Any],
    candidate_name: str,
    candidate_step: int,
    candidate_metrics: Mapping[str, Any],
    development_steps_per_epoch: int,
    refit_steps_per_epoch: int,
    evaluation_timestamp: str,
) -> dict[str, Any]:
    """Propose one immutable candidate without mutating the frozen decision."""

    current_rank = metric_rank(current_metrics, current_step)
    candidate_rank = metric_rank(candidate_metrics, candidate_step)
    if candidate_rank < current_rank:
        proposed_name = candidate_name
        proposed_step = candidate_step
    else:
        proposed_name = current_name
        proposed_step = current_step
    proposed_budget = refit_step_budget(
        proposed_step,
        development_steps_per_epoch,
        refit_steps_per_epoch,
    )
    return {
        "schema_version": 1,
        "status": "candidate_evaluated_not_finalized",
        "methodological_amendment": (
            "terminal development checkpoint validation decided before final_holdout access"
        ),
        "evaluation_identity": dict(identity),
        "evaluation_timestamp": evaluation_timestamp,
        "current_best": {
            "checkpoint_name": current_name,
            "step": current_step,
            "metrics": dict(current_metrics),
            "metric_rank": list(current_rank),
        },
        "candidate": {
            "checkpoint_name": candidate_name,
            "step": candidate_step,
            "metrics": dict(candidate_metrics),
            "metric_rank": list(candidate_rank),
        },
        "proposed_best": proposed_name,
        "proposed_best_step": proposed_step,
        "proposed_refit_budget": proposed_budget,
        "metric_policy": list(METRIC_POLICY),
        "frozen_decision_modified": False,
        "historical_test_accessed": False,
        "final_holdout_accessed": False,
        "final_holdout_decoded": False,
        "privacy": {
            "contains_transcriptions": False,
            "contains_predictions": False,
            "contains_sample_identifiers": False,
            "contains_speaker_identifiers": False,
            "contains_local_paths": False,
        },
    }


def _existing_report(
    settings: FullTrainingSettings,
    identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    private_path = settings.artifact_output_directory / CANDIDATE_PRIVATE_FILENAME
    public_path = settings.shareable_output_directory / CANDIDATE_PUBLIC_FILENAME
    exists = (private_path.is_file(), public_path.is_file())
    if exists == (False, False):
        return None
    if exists != (True, True):
        raise ConfigError("L'évaluation candidate existante est incomplète.")
    private = load_json_object(private_path)
    public = load_json_object(public_path)
    for label, payload in (("privé", private), ("public", public)):
        actual = payload.get("evaluation_identity")
        if not isinstance(actual, Mapping):
            raise ConfigError(f"Le rapport candidat {label} n'a pas d'identité.")
        require_matching_identity(
            actual,
            identity,
            description=f"Le rapport candidat {label}",
        )
    if public.get("status") != "candidate_evaluated_not_finalized":
        raise ConfigError("Le rapport candidat public a un statut inattendu.")
    return public


def run_development_final_validation(
    settings: FullTrainingSettings,
) -> dict[str, Any]:
    """Evaluate the terminal development checkpoint on validation exactly once."""

    _require_stage_available(settings)
    context = _load_context(settings)
    artifacts = _load_development_artifacts(settings, context)
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    decoding_hash = _decoding_config_hash(settings, artifacts)
    identity = _evaluation_identity(
        context,
        artifacts,
        decoding_config_hash=decoding_hash,
        torch_version=str(torch.__version__),
        transformers_version=str(transformers.__version__),
    )
    existing = _existing_report(settings, identity)
    if existing is not None:
        return existing

    device = _require_cuda(torch)
    _seed_everything(settings.seed, torch)
    processor, collator = _processor_and_collator(
        settings, context.model_settings, transformers, torch
    )
    model = _load_model(
        source=artifacts.candidate_checkpoint,
        model_revision=None,
        cache_dir=context.model_settings.cache_dir,
        local_files_only=True,
        device=device,
        transformers=transformers,
    )
    model.config.use_cache = False
    try:
        evaluation = _evaluate_validation(
            model=model,
            processor=processor,
            rows=context.validation_rows,
            collator=collator,
            batch_size=settings.eval_batch_size,
            device=device,
            torch=torch,
            fp16=settings.fp16,
        )
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()

    speaker_count = len({row.speaker_id for row in context.validation_rows})
    candidate_metrics = _aggregate_metrics(
        evaluation.metrics,
        audio_count=len(context.validation_rows),
        speaker_count=speaker_count,
    )
    current_raw = artifacts.decision.get("best_validation_metrics")
    if not isinstance(current_raw, Mapping):
        raise ConfigError("Les métriques du meilleur checkpoint gelé sont absentes.")
    current_metrics = _aggregate_metrics(
        current_raw,
        audio_count=len(context.validation_rows),
        speaker_count=speaker_count,
    )
    timestamp = datetime.now(UTC).isoformat()
    report = build_candidate_report(
        identity=identity,
        current_name=artifacts.current_checkpoint.name,
        current_step=int(artifacts.decision["best_development_step"]),
        current_metrics=current_metrics,
        candidate_name=artifacts.candidate_checkpoint.name,
        candidate_step=int(artifacts.candidate_state["global_step"]),
        candidate_metrics=candidate_metrics,
        development_steps_per_epoch=int(artifacts.decision["development_steps_per_epoch"]),
        refit_steps_per_epoch=int(artifacts.decision["refit_steps_per_epoch"]),
        evaluation_timestamp=timestamp,
    )
    decision_path = settings.artifact_output_directory / DEVELOPMENT_DECISION_FILENAME
    if _sha256_file(decision_path) != artifacts.decision_sha256:
        raise ConfigError("La décision gelée a changé pendant l'évaluation.")
    write_json_atomic(
        settings.artifact_output_directory / CANDIDATE_PRIVATE_FILENAME,
        {
            "schema_version": 1,
            "evaluation_identity": identity,
            "evaluation_timestamp": timestamp,
            "metrics": evaluation.metrics,
            "predictions": list(evaluation.predictions),
        },
    )
    write_json_atomic(
        settings.shareable_output_directory / CANDIDATE_PUBLIC_FILENAME,
        report,
    )
    return report
