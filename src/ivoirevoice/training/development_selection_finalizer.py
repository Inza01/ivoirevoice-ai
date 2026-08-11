"""Decision-only finalization of the terminal development selection."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from ivoirevoice.exceptions import ConfigError, IvoireVoiceError
from ivoirevoice.training.full_settings import (
    FullTrainingSettings,
    load_full_training_settings,
)
from ivoirevoice.training.whisper_finetune import (
    directory_sha256,
    git_provenance,
    identity_sha256,
    load_json_object,
    metric_rank,
    refit_step_budget,
    require_matching_identity,
    write_json_atomic,
)

CONFIG_PATH = Path("configs/experiments/full_finetune_whisper_tiny_dy.yaml")
APPROVAL_PATH = Path("configs/experiments/development_selection_finalization_approval.json")
METHODOLOGICAL_AUTHORIZATION = "FINALIZE DEVELOPMENT SELECTION APPROUVÉ"
CONFIRMATION_ENVIRONMENT_VARIABLE = "IVOIREVOICE_CONFIRM_DEVELOPMENT_SELECTION"
CONFIRMATION_VALUE = "FINALIZE_DEVELOPMENT_SELECTION_APPROVED"
DEVELOPMENT_DECISION_FILENAME = "development_decision.json"
CANDIDATE_PUBLIC_FILENAME = "development_final_validation_candidate.json"
FINALIZATION_PUBLIC_FILENAME = "development_selection_finalization.json"
FINAL_MODEL_MANIFEST_FILENAME = "final_model_manifest.json"
FINAL_EVALUATION_RECEIPT_FILENAME = "final_holdout_evaluation_receipt.json"
REFIT_STARTED_FILENAME = "refit_started.json"
TRAINER_STATE_FILENAME = "full_trainer_state.json"
METRIC_POLICY = (
    "wer_micro",
    "cer_micro",
    "validation_loss",
    "earliest_step",
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
METRIC_FIELDS = (
    "validation_loss",
    "wer_micro",
    "cer_micro",
    "rtf",
    "word_substitutions",
    "word_insertions",
    "word_deletions",
    "audio_count",
    "speaker_count",
)
PRIVACY_FIELDS = (
    "contains_transcriptions",
    "contains_predictions",
    "contains_sample_identifiers",
    "contains_speaker_identifiers",
    "contains_local_paths",
)


@dataclass(frozen=True, slots=True)
class FinalizationReview:
    """Fully checked inputs and independently recomputed decision."""

    decision: dict[str, Any]
    candidate_report: dict[str, Any]
    candidate_report_sha256: str
    current_name: str
    current_step: int
    current_metrics: dict[str, Any]
    candidate_name: str
    candidate_step: int
    candidate_metrics: dict[str, Any]
    selected_name: str
    selected_step: int
    selected_metrics: dict[str, Any]
    selected_checkpoint_sha256: str
    previous_refit_budget: int
    selected_refit_budget: int
    development_run_id: str
    development_git_commit: str
    evaluation_git_commit: str
    finalization_git_commit: str
    config_sha256: str
    pilot_checkpoint_sha256: str
    candidate_checkpoint_sha256: str
    validation_fingerprint: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ConfigError(f"Impossible de calculer l'empreinte de {path.name}.") from exc
    return digest.hexdigest()


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ConfigError(f"{label} doit être un objet JSON.")
    return cast(dict[str, Any], dict(value))


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{label} doit être un entier positif.")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ConfigError(f"{label} doit être un entier positif ou nul.")
    return value


def _finite_float(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"{label} doit être numérique.")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{label} doit être fini.")
    return result


def _metrics(value: object, label: str, settings: FullTrainingSettings) -> dict[str, Any]:
    raw = _mapping(value, label)
    missing = sorted(set(METRIC_FIELDS) - raw.keys())
    if missing:
        raise ConfigError(f"{label} ne contient pas toutes les métriques : {missing}.")
    metrics = {
        "validation_loss": _finite_float(raw["validation_loss"], f"{label}.validation_loss"),
        "wer_micro": _finite_float(raw["wer_micro"], f"{label}.wer_micro"),
        "cer_micro": _finite_float(raw["cer_micro"], f"{label}.cer_micro"),
        "rtf": _finite_float(raw["rtf"], f"{label}.rtf"),
        "word_substitutions": _nonnegative_int(
            raw["word_substitutions"], f"{label}.word_substitutions"
        ),
        "word_insertions": _nonnegative_int(raw["word_insertions"], f"{label}.word_insertions"),
        "word_deletions": _nonnegative_int(raw["word_deletions"], f"{label}.word_deletions"),
        "audio_count": _positive_int(raw["audio_count"], f"{label}.audio_count"),
        "speaker_count": _positive_int(raw["speaker_count"], f"{label}.speaker_count"),
    }
    if (
        metrics["audio_count"] != settings.validation_audio_count
        or metrics["speaker_count"] != settings.validation_speaker_count
    ):
        raise ConfigError(f"{label} ne couvre pas la validation gelée.")
    return metrics


def _require_stage_available(settings: FullTrainingSettings) -> None:
    if (settings.artifact_output_directory / FINAL_EVALUATION_RECEIPT_FILENAME).exists():
        raise ConfigError("Le final_holdout a déjà été ouvert : finalisation refusée.")
    if (settings.artifact_output_directory / REFIT_STARTED_FILENAME).exists() or (
        settings.artifact_output_directory / FINAL_MODEL_MANIFEST_FILENAME
    ).exists():
        raise ConfigError("Le refit a déjà démarré : finalisation refusée.")
    refit_directory = settings.refit_checkpoint_directory
    if refit_directory.is_dir() and any(refit_directory.iterdir()):
        raise ConfigError("Un checkpoint refit existe : finalisation refusée.")


def _checkpoint_state(path: Path) -> dict[str, Any]:
    state = load_json_object(path / TRAINER_STATE_FILENAME)
    step = _positive_int(state.get("global_step"), "checkpoint.global_step")
    if path.name != f"checkpoint-{step:06d}":
        raise ConfigError("Le nom du checkpoint ne correspond pas à son step.")
    return state


def _rank_from_report(
    value: object,
    expected: tuple[float, float, float, int],
    label: str,
) -> None:
    if not isinstance(value, list) or len(value) != 4:
        raise ConfigError(f"{label} est absent ou invalide.")
    observed = (
        _finite_float(value[0], f"{label}[0]"),
        _finite_float(value[1], f"{label}[1]"),
        _finite_float(value[2], f"{label}[2]"),
        _positive_int(value[3], f"{label}[3]"),
    )
    if observed != expected:
        raise ConfigError(f"{label} ne correspond pas au classement recalculé.")


def _require_report_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    if report.get("status") != "candidate_evaluated_not_finalized":
        raise ConfigError("Le rapport candidat n'attend pas une finalisation.")
    if report.get("frozen_decision_modified") is not False:
        raise ConfigError("Le rapport candidat indique une décision déjà modifiée.")
    if tuple(report.get("metric_policy", ())) != METRIC_POLICY:
        raise ConfigError("La politique métrique du rapport candidat a changé.")
    if (
        report.get("final_holdout_accessed") is not False
        or report.get("final_holdout_decoded") is not False
        or report.get("historical_test_accessed") is not False
    ):
        raise ConfigError("Le rapport candidat indique un accès interdit.")
    privacy = _mapping(report.get("privacy"), "La preuve de confidentialité")
    if set(privacy) != set(PRIVACY_FIELDS) or any(
        privacy[field] is not False for field in PRIVACY_FIELDS
    ):
        raise ConfigError("Le rapport candidat contient une donnée privée.")
    identity = _mapping(report.get("evaluation_identity"), "L'identité d'évaluation")
    if identity.get("stage") != "development-final-validation":
        raise ConfigError("L'identité ne provient pas de la validation terminale.")
    if tuple(identity.get("metric_policy", ())) != METRIC_POLICY:
        raise ConfigError("La politique métrique de l'identité a changé.")
    _validate_hash(identity.get("decoding_config_hash"), "decoding_config_hash")
    return identity


def _require_approval_contract(
    path: Path,
    *,
    report_sha256: str,
    identity: Mapping[str, Any],
) -> None:
    approval = load_json_object(path)
    if (
        approval.get("schema_version") != 1
        or approval.get("status") != "approved"
        or approval.get("methodological_authorization") != METHODOLOGICAL_AUTHORIZATION
        or approval.get("candidate_report_filename") != CANDIDATE_PUBLIC_FILENAME
    ):
        raise ConfigError("Le reçu d'autorisation méthodologique est invalide.")
    expected_report_hash = _validate_hash(
        approval.get("candidate_report_sha256"),
        "approval.candidate_report_sha256",
    )
    if report_sha256 != expected_report_hash:
        raise ConfigError("Le rapport candidat diffère du rapport approuvé.")
    approved_identity = _mapping(
        approval.get("evaluation_identity"),
        "L'identité approuvée",
    )
    if tuple(approval.get("metric_policy", ())) != METRIC_POLICY:
        raise ConfigError("La politique métrique approuvée a changé.")
    require_matching_identity(
        identity,
        approved_identity,
        description="Le rapport candidat approuvé",
    )


def _validate_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise ConfigError(f"{label} n'est pas une empreinte SHA-256 valide.")
    return value


def _validate_commit(value: object, label: str) -> str:
    if not isinstance(value, str) or not COMMIT_PATTERN.fullmatch(value):
        raise ConfigError(f"{label} n'est pas un commit valide.")
    return value


def _load_review(
    settings: FullTrainingSettings,
    approval_path: Path | None = None,
) -> FinalizationReview:
    root = Path(__file__).resolve().parents[3]
    finalization_commit, clean = git_provenance(root)
    if not clean:
        raise ConfigError("Le finaliseur exige un commit Git propre.")
    config_sha256 = _sha256_file(settings.config_path)
    pilot_sha256 = directory_sha256(settings.initial_checkpoint_path)
    decision_path = settings.artifact_output_directory / DEVELOPMENT_DECISION_FILENAME
    report_path = settings.shareable_output_directory / CANDIDATE_PUBLIC_FILENAME
    decision = load_json_object(decision_path)
    report = load_json_object(report_path)
    if decision.get("status") != "frozen":
        raise ConfigError("Le développement doit rester frozen.")
    identity = _require_report_contract(report)
    report_sha256 = _sha256_file(report_path)
    effective_approval_path = approval_path or APPROVAL_PATH
    if not effective_approval_path.is_absolute():
        effective_approval_path = root / effective_approval_path
    _require_approval_contract(
        effective_approval_path,
        report_sha256=report_sha256,
        identity=identity,
    )

    report_config = _validate_hash(identity.get("config_hash"), "config_hash")
    report_pilot = _validate_hash(identity.get("pilot_checkpoint_hash"), "pilot_checkpoint_hash")
    validation_fingerprint = _validate_hash(
        identity.get("validation_fingerprint"), "validation_fingerprint"
    )
    validation_manifest = _validate_hash(
        identity.get("validation_manifest_hash"), "validation_manifest_hash"
    )
    development_commit = _validate_commit(
        identity.get("development_git_commit"), "development_git_commit"
    )
    evaluation_commit = _validate_commit(
        identity.get("evaluation_git_commit"), "evaluation_git_commit"
    )
    development_run_id = _validate_hash(identity.get("development_run_id"), "development_run_id")
    if report_config != config_sha256 or report_config != decision.get("config_sha256"):
        raise ConfigError("Le config hash ne correspond pas à la décision.")
    if report_pilot != pilot_sha256 or report_pilot != decision.get("initial_checkpoint_sha256"):
        raise ConfigError("Le hash du checkpoint pilote ne correspond pas.")
    if validation_manifest != decision.get("manifest_sha256"):
        raise ConfigError("Le hash du manifeste de validation ne correspond pas.")
    if development_commit != decision.get("code_commit"):
        raise ConfigError("Le commit du développement ne correspond pas.")

    current = _mapping(report.get("current_best"), "current_best")
    candidate = _mapping(report.get("candidate"), "candidate")
    current_name = str(current.get("checkpoint_name", ""))
    candidate_name = str(candidate.get("checkpoint_name", ""))
    current_step = _positive_int(current.get("step"), "current_best.step")
    candidate_step = _positive_int(candidate.get("step"), "candidate.step")
    if current_name != f"checkpoint-{current_step:06d}":
        raise ConfigError("Le checkpoint courant du rapport est incohérent.")
    if candidate_name != f"checkpoint-{candidate_step:06d}":
        raise ConfigError("Le checkpoint candidat du rapport est incohérent.")
    current_metrics = _metrics(current.get("metrics"), "current_best.metrics", settings)
    candidate_metrics = _metrics(candidate.get("metrics"), "candidate.metrics", settings)
    current_rank = metric_rank(current_metrics, current_step)
    candidate_rank = metric_rank(candidate_metrics, candidate_step)
    _rank_from_report(current.get("metric_rank"), current_rank, "current_best.metric_rank")
    _rank_from_report(candidate.get("metric_rank"), candidate_rank, "candidate.metric_rank")

    if candidate_rank < current_rank:
        selected_name = candidate_name
        selected_step = candidate_step
        selected_metrics = candidate_metrics
    else:
        selected_name = current_name
        selected_step = current_step
        selected_metrics = current_metrics
    development_steps = _positive_int(
        decision.get("development_steps_per_epoch"),
        "development_steps_per_epoch",
    )
    refit_steps = _positive_int(decision.get("refit_steps_per_epoch"), "refit_steps_per_epoch")
    selected_budget = refit_step_budget(selected_step, development_steps, refit_steps)
    if (
        report.get("proposed_best") != selected_name
        or report.get("proposed_best_step") != selected_step
        or report.get("proposed_refit_budget") != selected_budget
    ):
        raise ConfigError("La proposition persistée ne correspond pas au recalcul.")

    current_path = settings.development_checkpoint_directory / current_name
    candidate_path = settings.development_checkpoint_directory / candidate_name
    current_state = _checkpoint_state(current_path)
    candidate_state = _checkpoint_state(candidate_path)
    expected_run = {
        "stage": "development",
        "selection_sha256": decision.get("development_selection_sha256"),
        "initial_checkpoint_sha256": pilot_sha256,
        "config_sha256": config_sha256,
        "code_commit": development_commit,
    }
    expected_run_id = identity_sha256(expected_run)
    if expected_run_id != development_run_id:
        raise ConfigError("Le development run ID ne correspond pas à son identité.")
    for label, state in (("courant", current_state), ("candidat", candidate_state)):
        require_matching_identity(
            state,
            {**expected_run, "run_id": development_run_id},
            description=f"Le checkpoint {label}",
        )
    if candidate_state.get("completed") is not True:
        raise ConfigError("Le checkpoint candidat n'est pas terminal.")

    current_hash = directory_sha256(current_path)
    candidate_hash = directory_sha256(candidate_path)
    if current_hash != identity.get("current_checkpoint_hash"):
        raise ConfigError("Le hash du checkpoint courant ne correspond pas.")
    if candidate_hash != identity.get("candidate_checkpoint_hash"):
        raise ConfigError("Le hash du checkpoint candidat ne correspond pas.")
    selected_hash = candidate_hash if selected_name == candidate_name else current_hash

    finalized = decision.get("development_selection_finalized") is True
    if finalized:
        expected_current = {
            "checkpoint": decision.get("previous_best_checkpoint"),
            "step": decision.get("previous_best_step"),
            "budget": decision.get("previous_refit_budget"),
        }
    else:
        expected_current = {
            "checkpoint": decision.get("best_checkpoint_name"),
            "step": decision.get("best_development_step"),
            "budget": decision.get("refit_step_budget"),
        }
    if expected_current["checkpoint"] != current_name or expected_current["step"] != current_step:
        raise ConfigError("L'ancienne décision ne correspond pas au rapport candidat.")
    previous_budget = _positive_int(expected_current["budget"], "previous_refit_budget")
    original_metrics = (
        decision.get("previous_best_metrics")
        if finalized
        else (decision.get("best_validation_metrics"))
    )
    original = _mapping(original_metrics, "Les métriques de l'ancienne décision")
    comparable_original = {field: original[field] for field in METRIC_FIELDS if field in original}
    for field in ("validation_loss", "wer_micro", "cer_micro", "rtf"):
        if comparable_original.get(field) != current_metrics[field]:
            raise ConfigError("Les métriques historiques ne correspondent plus.")

    return FinalizationReview(
        decision=decision,
        candidate_report=report,
        candidate_report_sha256=report_sha256,
        current_name=current_name,
        current_step=current_step,
        current_metrics=current_metrics,
        candidate_name=candidate_name,
        candidate_step=candidate_step,
        candidate_metrics=candidate_metrics,
        selected_name=selected_name,
        selected_step=selected_step,
        selected_metrics=selected_metrics,
        selected_checkpoint_sha256=selected_hash,
        previous_refit_budget=previous_budget,
        selected_refit_budget=selected_budget,
        development_run_id=development_run_id,
        development_git_commit=development_commit,
        evaluation_git_commit=evaluation_commit,
        finalization_git_commit=finalization_commit,
        config_sha256=config_sha256,
        pilot_checkpoint_sha256=pilot_sha256,
        candidate_checkpoint_sha256=candidate_hash,
        validation_fingerprint=validation_fingerprint,
    )


def _trace(review: FinalizationReview, timestamp: str) -> dict[str, Any]:
    return {
        "development_run_id": review.development_run_id,
        "development_git_commit": review.development_git_commit,
        "evaluation_git_commit": review.evaluation_git_commit,
        "finalization_git_commit": review.finalization_git_commit,
        "config_hash": review.config_sha256,
        "pilot_checkpoint_hash": review.pilot_checkpoint_sha256,
        "candidate_checkpoint_hash": review.candidate_checkpoint_sha256,
        "candidate_report_hash": review.candidate_report_sha256,
        "validation_fingerprint": review.validation_fingerprint,
        "metric_policy": list(METRIC_POLICY),
        "finalization_timestamp": timestamp,
    }


def _build_finalized_decision(
    review: FinalizationReview,
    timestamp: str,
) -> dict[str, Any]:
    decision = dict(review.decision)
    previous_entry = {
        "role": "provisional_before_terminal_validation",
        "checkpoint": review.current_name,
        "step": review.current_step,
        "refit_budget": review.previous_refit_budget,
        "metrics": review.current_metrics,
    }
    history = decision.get("selection_history", [])
    if not isinstance(history, list):
        raise ConfigError("L'historique de sélection existant est invalide.")
    trace = _trace(review, timestamp)
    decision.update(
        {
            "best_checkpoint_name": review.selected_name,
            "best_checkpoint_sha256": review.selected_checkpoint_sha256,
            "best_development_step": review.selected_step,
            "best_validation_metrics": review.selected_metrics,
            "selected_epoch_fraction": (
                review.selected_step / int(decision["development_steps_per_epoch"])
            ),
            "refit_step_budget": review.selected_refit_budget,
            "best_checkpoint": review.selected_name,
            "best_checkpoint_hash": review.selected_checkpoint_sha256,
            "best_step": review.selected_step,
            "selection_metric": "WER",
            "WER": review.selected_metrics["wer_micro"],
            "CER": review.selected_metrics["cer_micro"],
            "validation_loss": review.selected_metrics["validation_loss"],
            "RTF": review.selected_metrics["rtf"],
            "refit_budget": review.selected_refit_budget,
            "development_selection_finalized": True,
            "finalized_from": "development-final-validation",
            "previous_best_checkpoint": review.current_name,
            "previous_best_step": review.current_step,
            "previous_best_metrics": review.current_metrics,
            "previous_refit_budget": review.previous_refit_budget,
            "development_run_id": review.development_run_id,
            "evaluation_git_commit": review.evaluation_git_commit,
            "finalization_git_commit": review.finalization_git_commit,
            "validation_fingerprint": review.validation_fingerprint,
            "metric_policy": list(METRIC_POLICY),
            "finalization_timestamp": timestamp,
            "candidate_report_sha256": review.candidate_report_sha256,
            "selection_history": [*history, previous_entry],
            "finalization_trace": trace,
        }
    )
    return decision


def _public_report(
    review: FinalizationReview,
    timestamp: str,
    *,
    status: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": status,
        **_trace(review, timestamp),
        "previous_best": {
            "checkpoint": review.current_name,
            "step": review.current_step,
            "metrics": review.current_metrics,
            "refit_budget": review.previous_refit_budget,
        },
        "new_best": {
            "checkpoint": review.selected_name,
            "step": review.selected_step,
            "metrics": review.selected_metrics,
            "refit_budget": review.selected_refit_budget,
        },
        "previous_refit_budget": review.previous_refit_budget,
        "new_refit_budget": review.selected_refit_budget,
        "selection_reason": "lowest metric_rank under frozen policy",
        "refit": {
            "source_checkpoint": "checkpoint-000140",
            "training_audio_count": 16_425,
            "training_speaker_count": 18,
            "fresh_optimizer": True,
            "fresh_scheduler": True,
            "fresh_grad_scaler": True,
            "precision": "fp16",
            "learning_rate": 1e-5,
            "per_device_batch": 4,
            "gradient_accumulation": 4,
            "effective_batch": 16,
        },
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


def _require_same_finalized_decision(
    review: FinalizationReview,
) -> str:
    decision = review.decision
    timestamp = decision.get("finalization_timestamp")
    if not isinstance(timestamp, str) or not timestamp:
        raise ConfigError("La décision finalisée n'a pas de timestamp.")
    expected = {
        "best_checkpoint_name": review.selected_name,
        "best_checkpoint_sha256": review.selected_checkpoint_sha256,
        "best_development_step": review.selected_step,
        "refit_step_budget": review.selected_refit_budget,
        "development_selection_finalized": True,
        "finalized_from": "development-final-validation",
        "previous_best_checkpoint": review.current_name,
        "previous_best_step": review.current_step,
        "previous_refit_budget": review.previous_refit_budget,
        "development_run_id": review.development_run_id,
        "evaluation_git_commit": review.evaluation_git_commit,
        "finalization_git_commit": review.finalization_git_commit,
        "validation_fingerprint": review.validation_fingerprint,
        "metric_policy": list(METRIC_POLICY),
        "candidate_report_sha256": review.candidate_report_sha256,
    }
    require_matching_identity(
        decision,
        expected,
        description="La décision déjà finalisée",
    )
    best_metrics = _mapping(
        decision.get("best_validation_metrics"),
        "Les métriques de la décision finalisée",
    )
    if any(best_metrics.get(field) != review.selected_metrics[field] for field in METRIC_FIELDS):
        raise ConfigError("Les métriques de la décision finalisée ont changé.")
    aliases = {
        "best_checkpoint": review.selected_name,
        "best_checkpoint_hash": review.selected_checkpoint_sha256,
        "best_step": review.selected_step,
        "selection_metric": "WER",
        "WER": review.selected_metrics["wer_micro"],
        "CER": review.selected_metrics["cer_micro"],
        "validation_loss": review.selected_metrics["validation_loss"],
        "RTF": review.selected_metrics["rtf"],
        "refit_budget": review.selected_refit_budget,
    }
    require_matching_identity(
        decision,
        aliases,
        description="Les alias de la décision finalisée",
    )
    return timestamp


def run_development_selection_finalizer(
    settings: FullTrainingSettings,
) -> dict[str, Any]:
    """Finalize the persisted candidate decision without evaluating any data."""

    if os.getenv(CONFIRMATION_ENVIRONMENT_VARIABLE) != CONFIRMATION_VALUE:
        raise ConfigError("La confirmation explicite de finalisation est absente.")
    _require_stage_available(settings)
    review = _load_review(settings)
    finalization_path = settings.shareable_output_directory / FINALIZATION_PUBLIC_FILENAME
    if review.decision.get("development_selection_finalized") is True:
        timestamp = _require_same_finalized_decision(review)
        existing = load_json_object(finalization_path)
        expected_report = _public_report(
            review,
            timestamp,
            status="development_selection_finalized",
        )
        if existing != expected_report:
            raise ConfigError("Le rapport de finalisation existant est incohérent.")
        return {**existing, "status": "already_finalized_same_decision"}

    if finalization_path.exists():
        raise ConfigError("Un rapport de finalisation existe avant la décision.")
    timestamp = datetime.now(UTC).isoformat()
    finalized_decision = _build_finalized_decision(review, timestamp)
    write_json_atomic(
        settings.artifact_output_directory / DEVELOPMENT_DECISION_FILENAME,
        finalized_decision,
    )
    public = _public_report(
        review,
        timestamp,
        status="development_selection_finalized",
    )
    write_json_atomic(finalization_path, public)
    return public


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalisation décisionnelle du développement Whisper Tiny."
    )
    parser.add_argument(
        "--config",
        default=str(CONFIG_PATH),
        help="Configuration complète locale.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        settings = load_full_training_settings(args.config)
        result = run_development_selection_finalizer(settings)
    except IvoireVoiceError as exc:
        print(f"ERREUR: {exc}")
        return 1
    print(f"Stage development-finalize-selection : {result['status']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
