"""Full Whisper Tiny development, train+validation refit and sealed evaluation."""

from __future__ import annotations

import argparse
import gc
import importlib
import math
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any, cast

from ivoirevoice.exceptions import ConfigError, IvoireVoiceError
from ivoirevoice.models.whisper import load_whisper_settings
from ivoirevoice.training.audit import AuditedDataset, ManifestRow, load_audited_dataset
from ivoirevoice.training.full_selection import (
    FullSelection,
    build_full_selection,
    validate_final_holdout_files,
    write_full_selection_reports,
)
from ivoirevoice.training.full_settings import (
    FullTrainingSettings,
    load_full_training_settings,
)
from ivoirevoice.training.pilot_finetune import (
    EvaluationResult,
    PilotCollator,
    _chunks,
    _evaluate_validation,
    _linear_warmup_scheduler,
    _load_model,
    _ordered_train_rows,
    _seed_everything,
    _to_device,
)
from ivoirevoice.training.pilot_settings import PilotSettings
from ivoirevoice.training.whisper_finetune import (
    OptimizerGroupResult,
    accumulation_group_sizes,
    amp_metrics,
    apply_optimizer_outcome,
    compute_step_geometry,
    directory_sha256,
    free_disk_gib,
    git_provenance,
    identity_sha256,
    initialize_or_validate_amp_state,
    latest_complete_checkpoint,
    load_json_object,
    metric_rank,
    refit_step_budget,
    require_matching_identity,
    restore_runtime_states,
    save_checkpoint_atomic,
    validation_milestones,
    write_json_atomic,
)

CONFIG_PATH = Path("configs/experiments/full_finetune_whisper_tiny_dy.yaml")
TRAINER_STATE_FILENAME = "full_trainer_state.json"
DEVELOPMENT_DECISION_FILENAME = "development_decision.json"
FINAL_MODEL_MANIFEST_FILENAME = "final_model_manifest.json"
REFIT_STARTED_FILENAME = "refit_started.json"
FINAL_EVALUATION_RECEIPT_FILENAME = "final_holdout_evaluation_receipt.json"
FINAL_EVALUATION_METRICS_FILENAME = "final_holdout_metrics.json"
PREFLIGHT_REPORT_FILENAME = "preflight_report.json"


@dataclass(frozen=True, slots=True)
class FullContext:
    """Validated data, selection and immutable fingerprints."""

    dataset: AuditedDataset
    selection: FullSelection
    model_settings: Any
    config_sha256: str
    initial_checkpoint_sha256: str
    code_commit: str


def _sha256_file(path: Path) -> str:
    hashlib = importlib.import_module("hashlib")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return str(digest.hexdigest())


def _require_cuda(torch: Any) -> Any:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise ConfigError("nvidia-smi est absent : entraînement complet refusé.")
    try:
        result = subprocess.run(
            [executable, "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ConfigError(f"nvidia-smi est inutilisable : {exc}") from exc
    if result.returncode or not result.stdout.strip():
        raise ConfigError("Le pilote NVIDIA ne répond pas : entraînement refusé.")
    if not bool(torch.cuda.is_available()):
        raise ConfigError("PyTorch ne détecte pas CUDA : entraînement refusé.")
    device = torch.device("cuda:0")
    probe = torch.zeros(1, device=device)
    if probe.device != device or not bool(probe.isfinite().all()):
        raise ConfigError("Le test tensor CUDA a échoué.")
    fp16_probe = torch.ones((2, 2), dtype=torch.float16, device=device)
    fp16_result = fp16_probe @ fp16_probe
    torch.cuda.synchronize(device)
    if not bool(fp16_result.isfinite().all()):
        raise ConfigError("Le test de calcul FP16 CUDA a échoué.")
    return device


def _public_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(metrics)
    result.pop("speaker_metrics", None)
    return result


def _holdout_receipt(settings: FullTrainingSettings) -> dict[str, Any] | None:
    path = settings.artifact_output_directory / FINAL_EVALUATION_RECEIPT_FILENAME
    return load_json_object(path) if path.is_file() else None


def _require_training_still_allowed(settings: FullTrainingSettings) -> None:
    receipt = _holdout_receipt(settings)
    if receipt is not None:
        raise ConfigError(
            "Un événement final_holdout a déjà commencé : tout réentraînement est interdit."
        )


def _load_context(
    settings: FullTrainingSettings,
    *,
    require_clean: bool,
    write_reports: bool,
) -> FullContext:
    root = Path(__file__).resolve().parents[3]
    code_commit, clean = git_provenance(root)
    if require_clean and not clean:
        raise ConfigError(
            "Le workflow complet exige un commit Git propre avant tout calcul ML."
        )
    dataset = load_audited_dataset(settings)
    selection, public_report, private_report = build_full_selection(dataset, settings)
    if write_reports:
        write_full_selection_reports(settings, public_report, private_report)
    model_settings = load_whisper_settings(root / settings.model_config_path)
    if (
        model_settings.model_id != settings.expected_model_id
        or model_settings.model_revision != settings.expected_model_revision
    ):
        raise ConfigError("Le modèle Whisper Tiny épinglé ne correspond pas.")
    if model_settings.task != "transcribe" or model_settings.language is not None:
        raise ConfigError("Le workflow exige transcribe sans token de langue forcé.")
    if free_disk_gib(settings.checkpoint_directory) < settings.minimum_free_disk_gib:
        raise ConfigError("L'espace disque libre est inférieur à 15 GiB.")
    return FullContext(
        dataset=dataset,
        selection=selection,
        model_settings=model_settings,
        config_sha256=_sha256_file(settings.config_path),
        initial_checkpoint_sha256=directory_sha256(
            settings.initial_checkpoint_path
        ),
        code_commit=code_commit,
    )


def _processor_and_collator(
    settings: FullTrainingSettings,
    model_settings: Any,
    transformers: Any,
    torch: Any,
) -> tuple[Any, PilotCollator]:
    processor = transformers.WhisperProcessor.from_pretrained(
        settings.initial_checkpoint_path,
        cache_dir=str(model_settings.cache_dir),
        local_files_only=True,
    )
    collator_settings = cast(
        PilotSettings,
        SimpleNamespace(
            train_split=settings.train_split,
            validation_split=settings.validation_split,
            dataset_root=settings.dataset_root,
            max_audio_seconds=settings.max_audio_seconds,
        ),
    )
    return processor, PilotCollator(
        collator_settings,
        processor,
        model_settings.expected_sampling_rate_hz,
        torch,
    )


def run_preflight(settings: FullTrainingSettings) -> dict[str, Any]:
    """Validate every local prerequisite and reload the pilot checkpoint on CUDA."""

    context = _load_context(settings, require_clean=True, write_reports=True)
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    device = _require_cuda(torch)
    _seed_everything(settings.seed, torch)
    processor, collator = _processor_and_collator(
        settings, context.model_settings, transformers, torch
    )
    model = _load_model(
        source=settings.initial_checkpoint_path,
        model_revision=None,
        cache_dir=context.model_settings.cache_dir,
        local_files_only=True,
        device=device,
        transformers=transformers,
    )
    sample = collator((context.selection.train_rows[0],))
    sample = _to_device(sample, device)
    with torch.inference_mode():
        generated = model.generate(
            input_features=sample["input_features"],
            attention_mask=sample["attention_mask"],
            task="transcribe",
            do_sample=False,
            max_new_tokens=2,
        )
    if int(generated.shape[0]) != 1:
        raise ConfigError("L'inférence de contrôle du checkpoint a échoué.")
    report = {
        "schema_version": 1,
        "status": "passed",
        "code_commit": context.code_commit,
        "config_sha256": context.config_sha256,
        "manifest_sha256": context.dataset.manifest_sha256,
        "initial_checkpoint_name": settings.initial_checkpoint_path.name,
        "initial_checkpoint_sha256": context.initial_checkpoint_sha256,
        "train_audio_count": len(context.selection.train_rows),
        "validation_audio_count": len(context.selection.validation_rows),
        "refit_audio_count": len(context.selection.refit_rows),
        "development_selection_sha256": (
            context.selection.development_selection_sha256
        ),
        "refit_selection_sha256": context.selection.refit_selection_sha256,
        "historical_test_decoded": False,
        "final_holdout_decoded": False,
        "cuda_available": True,
        "gpu_name": str(torch.cuda.get_device_name(device)),
        "fp16": settings.fp16,
    }
    write_json_atomic(
        settings.artifact_output_directory / PREFLIGHT_REPORT_FILENAME, report
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return report


def _latest_checkpoint(directory: Path) -> Path | None:
    return latest_complete_checkpoint(
        directory,
        (
            TRAINER_STATE_FILENAME,
            "optimizer.pt",
            "scheduler.pt",
            "scaler.pt",
            "config.json",
        ),
    )


def _save_checkpoint(
    *,
    directory: Path,
    minimum_free_disk_gib: float,
    save_total_limit: int,
    best_checkpoint_name: str | None,
    model: Any,
    processor: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    state: Mapping[str, Any],
    torch: Any,
) -> Path:
    return save_checkpoint_atomic(
        directory=directory,
        minimum_free_disk_gib=minimum_free_disk_gib,
        save_total_limit=save_total_limit,
        best_checkpoint_name=best_checkpoint_name,
        state_filename=TRAINER_STATE_FILENAME,
        model=model,
        processor=processor,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        state=state,
        torch=torch,
    )


def _initial_training_state(
    *,
    stage: str,
    selection_hash: str,
    context: FullContext,
    total_steps: int,
) -> dict[str, Any]:
    run_identity = {
        "stage": stage,
        "selection_sha256": selection_hash,
        "initial_checkpoint_sha256": context.initial_checkpoint_sha256,
        "config_sha256": context.config_sha256,
        "code_commit": context.code_commit,
    }
    return {
        "schema_version": 1,
        **run_identity,
        "run_id": identity_sha256(run_identity),
        "global_step": 0,
        "successful_optimizer_steps": 0,
        "optimizer_attempts": 0,
        "amp_skipped_steps": 0,
        "consecutive_amp_skips": 0,
        "max_consecutive_amp_skips_observed": 0,
        "initial_grad_scale": None,
        "final_grad_scale": None,
        "precision": "fp16",
        "epoch": 0,
        "groups_completed_in_epoch": 0,
        "total_steps": total_steps,
        "log_history": [],
        "completed": False,
        "optimizer_initialized_from": "fresh",
        "scheduler_initialized_from": "fresh",
        "pilot_optimizer_state_loaded": False,
    }


def _new_training_runtime(
    *,
    settings: FullTrainingSettings,
    context: FullContext,
    directory: Path,
    selection_hash: str,
    stage: str,
    total_steps: int,
    torch: Any,
    transformers: Any,
    device: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, dict[str, Any]]:
    processor, collator = _processor_and_collator(
        settings, context.model_settings, transformers, torch
    )
    latest = _latest_checkpoint(directory)
    if latest is None:
        model = _load_model(
            source=settings.initial_checkpoint_path,
            model_revision=None,
            cache_dir=context.model_settings.cache_dir,
            local_files_only=True,
            device=device,
            transformers=transformers,
        )
        state = _initial_training_state(
            stage=stage,
            selection_hash=selection_hash,
            context=context,
            total_steps=total_steps,
        )
    else:
        state = load_json_object(latest / TRAINER_STATE_FILENAME)
        expected_identity = {
            "stage": stage,
            "selection_sha256": selection_hash,
            "initial_checkpoint_sha256": context.initial_checkpoint_sha256,
            "config_sha256": context.config_sha256,
            "code_commit": context.code_commit,
        }
        expected = {
            "stage": stage,
            "selection_sha256": selection_hash,
            "initial_checkpoint_sha256": context.initial_checkpoint_sha256,
            "config_sha256": context.config_sha256,
            "code_commit": context.code_commit,
            "total_steps": total_steps,
            "run_id": identity_sha256(expected_identity),
        }
        require_matching_identity(
            state,
            expected,
            description="Le checkpoint de reprise",
        )
        model = _load_model(
            source=latest,
            model_revision=None,
            cache_dir=context.model_settings.cache_dir,
            local_files_only=True,
            device=device,
            transformers=transformers,
        )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    warmup_steps = math.ceil(total_steps * settings.warmup_ratio)
    scheduler = _linear_warmup_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        torch=torch,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=settings.fp16)
    if latest is not None:
        restore_runtime_states(
            checkpoint=latest,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            torch=torch,
            device=device,
        )
    initialize_or_validate_amp_state(state, scaler, resumed=latest is not None)
    return model, processor, collator, optimizer, scheduler, scaler, state


def _persist_latest_state(directory: Path, state: Mapping[str, Any]) -> None:
    """Finalize the state that corresponds to the latest retained optimizer state."""

    latest = _latest_checkpoint(directory)
    if latest is None or int(latest.name.removeprefix("checkpoint-")) != int(
        state["global_step"]
    ):
        raise ConfigError("Le dernier état d'entraînement n'a pas de checkpoint associé.")
    write_json_atomic(latest / TRAINER_STATE_FILENAME, state)


def _optimizer_group(
    *,
    row_batches: Sequence[Sequence[ManifestRow]],
    collator: PilotCollator,
    model: Any,
    optimizer: Any,
    scaler: Any,
    device: Any,
    torch: Any,
    max_grad_norm: float,
) -> OptimizerGroupResult:
    optimizer.zero_grad(set_to_none=True)
    divisor = len(row_batches)
    losses: list[float] = []
    for row_batch in row_batches:
        batch = _to_device(collator(row_batch), device)
        model_device = next(model.parameters()).device
        if any(tensor.device != model_device for tensor in batch.values()):
            raise ConfigError("Les tensors train et le modèle ne sont pas sur le même device.")
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            raw_loss = model(**batch).loss
            scaled_loss = raw_loss / divisor
        value = float(raw_loss.detach().cpu())
        if not math.isfinite(value):
            raise ConfigError("Loss NaN/Inf : arrêt immédiat.")
        if value > 50:
            raise ConfigError("Divergence forte détectée (loss > 50).")
        scaler.scale(scaled_loss).backward()
        losses.append(value)
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    amp_skipped = scale_after < scale_before
    if not amp_skipped and not all(
        bool(torch.isfinite(parameter.detach()).all()) for parameter in model.parameters()
    ):
        raise ConfigError("Paramètres modèle NaN/Inf après optimizer step : arrêt immédiat.")
    return OptimizerGroupResult(
        train_loss=sum(losses) / len(losses),
        scale_before=scale_before,
        scale_after=scale_after,
        optimizer_step_executed=not amp_skipped,
        amp_skipped=amp_skipped,
    )


def _write_private_evaluation(
    path: Path,
    result: EvaluationResult,
    *,
    evaluation_identity: Mapping[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "metrics": result.metrics,
        "predictions": list(result.predictions),
    }
    if evaluation_identity is not None:
        payload["evaluation_identity"] = dict(evaluation_identity)
    write_json_atomic(
        path,
        payload,
    )


def _require_passed_preflight(
    settings: FullTrainingSettings,
    context: FullContext,
) -> None:
    path = settings.artifact_output_directory / PREFLIGHT_REPORT_FILENAME
    if not path.is_file():
        raise ConfigError("Préflight complet absent : exécutez make full-finetune-preflight.")
    require_matching_identity(
        load_json_object(path),
        {
            "status": "passed",
            "code_commit": context.code_commit,
            "config_sha256": context.config_sha256,
            "manifest_sha256": context.dataset.manifest_sha256,
            "initial_checkpoint_sha256": context.initial_checkpoint_sha256,
            "development_selection_sha256": (
                context.selection.development_selection_sha256
            ),
            "refit_selection_sha256": context.selection.refit_selection_sha256,
        },
        description="Le préflight complet",
    )


def run_development(settings: FullTrainingSettings) -> dict[str, Any]:
    """Train on full train and freeze the train+validation refit budget."""

    _require_training_still_allowed(settings)
    context = _load_context(settings, require_clean=True, write_reports=True)
    decision_path = settings.artifact_output_directory / DEVELOPMENT_DECISION_FILENAME
    if decision_path.is_file():
        decision = load_json_object(decision_path)
        require_matching_identity(
            decision,
            {
                "status": "frozen",
                "code_commit": context.code_commit,
                "config_sha256": context.config_sha256,
                "manifest_sha256": context.dataset.manifest_sha256,
                "development_selection_sha256": (
                    context.selection.development_selection_sha256
                ),
                "initial_checkpoint_sha256": context.initial_checkpoint_sha256,
            },
            description="La décision de développement",
        )
        return decision
    _require_passed_preflight(settings, context)
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    device = _require_cuda(torch)
    _seed_everything(settings.seed, torch)
    geometry = compute_step_geometry(
        len(context.selection.train_rows),
        settings.train_batch_size,
        settings.gradient_accumulation_steps,
    )
    total_steps = geometry.optimizer_steps_per_epoch * settings.development_epochs
    if geometry.optimizer_steps_per_epoch != 861 or total_steps != 1_722:
        raise ConfigError("La géométrie du développement complet a changé.")
    runtime = _new_training_runtime(
        settings=settings,
        context=context,
        directory=settings.development_checkpoint_directory,
        selection_hash=context.selection.development_selection_sha256,
        stage="development",
        total_steps=total_steps,
        torch=torch,
        transformers=transformers,
        device=device,
    )
    model, processor, collator, optimizer, scheduler, scaler, state = runtime
    if state.get("completed") is True:
        decision = load_json_object(
            settings.artifact_output_directory / DEVELOPMENT_DECISION_FILENAME
        )
        return decision

    initial_path = (
        settings.artifact_output_directory / "development_initial_validation_private.json"
    )
    if not initial_path.is_file():
        initial = _evaluate_validation(
            model=model,
            processor=processor,
            rows=context.selection.validation_rows,
            collator=collator,
            batch_size=settings.eval_batch_size,
            device=device,
            torch=torch,
            fp16=True,
        )
        _write_private_evaluation(initial_path, initial)
        initial_metrics = _public_metrics(initial.metrics)
    else:
        initial_metrics = _public_metrics(
            cast(dict[str, Any], load_json_object(initial_path)["metrics"])
        )

    milestones = validation_milestones(
        geometry.optimizer_steps_per_epoch,
        settings.development_epochs,
        settings.evaluations_per_epoch,
    )
    started = perf_counter()
    duration_before_resume = float(state.get("training_duration_seconds", 0.0))
    torch.cuda.reset_peak_memory_stats(device)
    stored_rank = state.get("best_rank")
    best_rank: tuple[float, float, float, int] | None = None
    if isinstance(stored_rank, list) and len(stored_rank) == 4:
        best_rank = (
            float(stored_rank[0]),
            float(stored_rank[1]),
            float(stored_rank[2]),
            int(stored_rank[3]),
        )
    bad_evaluations = int(state.get("bad_evaluations", 0))
    stop = False
    for epoch in range(settings.development_epochs):
        if epoch < int(state["epoch"]):
            continue
        ordered = _ordered_train_rows(context.selection.train_rows, settings.seed, epoch)
        micro_batches = _chunks(ordered, settings.train_batch_size)
        sizes = accumulation_group_sizes(
            len(micro_batches), settings.gradient_accumulation_steps
        )
        groups: list[Sequence[Sequence[ManifestRow]]] = []
        cursor = 0
        for size in sizes:
            groups.append(micro_batches[cursor : cursor + size])
            cursor += size
        completed_groups = (
            int(state["groups_completed_in_epoch"])
            if epoch == int(state["epoch"])
            else 0
        )
        for group_index, group in enumerate(groups):
            if group_index < completed_groups:
                continue
            model.train()
            optimizer_result = _optimizer_group(
                row_batches=group,
                collator=collator,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                torch=torch,
                max_grad_norm=settings.max_grad_norm,
            )
            state["epoch"] = epoch
            state["groups_completed_in_epoch"] = group_index + 1
            optimizer_updated = apply_optimizer_outcome(
                state=state,
                result=optimizer_result,
                scheduler=scheduler,
                max_consecutive_amp_skips=settings.max_consecutive_amp_skips,
                stage="development",
            )
            step = int(state["global_step"])
            log_row: dict[str, Any] = {
                "step": step,
                "optimizer_attempt": int(state["optimizer_attempts"]),
                "successful_optimizer_steps": int(
                    state["successful_optimizer_steps"]
                ),
                "optimizer_step_executed": optimizer_updated,
                "amp_skipped": optimizer_result.amp_skipped,
                "scale_before": optimizer_result.scale_before,
                "scale_after": optimizer_result.scale_after,
                "epoch": epoch + (group_index + 1) / len(groups),
                "train_loss": optimizer_result.train_loss,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "validation": None,
            }
            cast(list[dict[str, Any]], state["log_history"]).append(log_row)
            if not optimizer_updated:
                continue
            if step == 1 or step % settings.logging_steps == 0:
                print(
                    f"development step={step:04d}/{total_steps} "
                    f"loss={optimizer_result.train_loss:.6f}"
                )
            if step not in milestones:
                continue
            validation = _evaluate_validation(
                model=model,
                processor=processor,
                rows=context.selection.validation_rows,
                collator=collator,
                batch_size=settings.eval_batch_size,
                device=device,
                torch=torch,
                fp16=True,
            )
            rank = metric_rank(validation.metrics, step)
            improved = best_rank is None or rank < best_rank
            checkpoint_name = f"checkpoint-{step:06d}"
            if improved:
                best_rank = rank
                bad_evaluations = 0
                state["best_checkpoint_name"] = checkpoint_name
                state["best_step"] = step
                state["best_metrics"] = _public_metrics(validation.metrics)
                state["best_rank"] = list(rank)
            else:
                bad_evaluations += 1
            state["bad_evaluations"] = bad_evaluations
            log_row["validation"] = _public_metrics(validation.metrics)
            _write_private_evaluation(
                settings.artifact_output_directory
                / f"development_validation_step_{step:06d}_private.json",
                validation,
            )
            state["training_duration_seconds"] = duration_before_resume + (
                perf_counter() - started
            )
            state["max_gpu_memory_mib"] = max(
                float(state.get("max_gpu_memory_mib", 0.0)),
                float(torch.cuda.max_memory_allocated(device)) / 1024**2,
            )
            _save_checkpoint(
                directory=settings.development_checkpoint_directory,
                minimum_free_disk_gib=settings.minimum_free_disk_gib,
                save_total_limit=settings.save_total_limit,
                best_checkpoint_name=cast(
                    str | None, state.get("best_checkpoint_name")
                ),
                model=model,
                processor=processor,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                state=state,
                torch=torch,
            )
            if bad_evaluations >= settings.early_stopping_patience:
                state["early_stopping_triggered"] = True
                stop = True
                break
        state["epoch"] = epoch + 1
        state["groups_completed_in_epoch"] = 0
        if stop:
            break
    if not state.get("best_checkpoint_name") or not state.get("best_step"):
        raise ConfigError("Aucun checkpoint de développement n'a été sélectionné.")
    state["completed"] = True
    state["training_duration_seconds"] = duration_before_resume + (
        perf_counter() - started
    )
    state["max_gpu_memory_mib"] = max(
        float(state.get("max_gpu_memory_mib", 0.0)),
        float(torch.cuda.max_memory_allocated(device)) / 1024**2,
    )
    latest = _latest_checkpoint(settings.development_checkpoint_directory)
    if latest is None or int(latest.name.removeprefix("checkpoint-")) != int(
        state["global_step"]
    ):
        _save_checkpoint(
            directory=settings.development_checkpoint_directory,
            minimum_free_disk_gib=settings.minimum_free_disk_gib,
            save_total_limit=settings.save_total_limit,
            best_checkpoint_name=cast(str | None, state.get("best_checkpoint_name")),
            model=model,
            processor=processor,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            state=state,
            torch=torch,
        )
    else:
        _persist_latest_state(settings.development_checkpoint_directory, state)
    best_step = int(state["best_step"])
    refit_geometry = compute_step_geometry(
        len(context.selection.refit_rows),
        settings.train_batch_size,
        settings.gradient_accumulation_steps,
    )
    if refit_geometry.optimizer_steps_per_epoch != 1_027:
        raise ConfigError("La géométrie du refit complet a changé.")
    refit_steps = refit_step_budget(
        best_step,
        geometry.optimizer_steps_per_epoch,
        refit_geometry.optimizer_steps_per_epoch,
    )
    decision = {
        "schema_version": 1,
        "status": "frozen",
        "code_commit": context.code_commit,
        "config_sha256": context.config_sha256,
        "manifest_sha256": context.dataset.manifest_sha256,
        "development_selection_sha256": (
            context.selection.development_selection_sha256
        ),
        "refit_selection_sha256": context.selection.refit_selection_sha256,
        "initial_checkpoint_sha256": context.initial_checkpoint_sha256,
        "best_checkpoint_name": state["best_checkpoint_name"],
        "best_checkpoint_sha256": directory_sha256(
            settings.development_checkpoint_directory
            / str(state["best_checkpoint_name"])
        ),
        "best_development_step": best_step,
        "development_steps_per_epoch": geometry.optimizer_steps_per_epoch,
        "selected_epoch_fraction": best_step / geometry.optimizer_steps_per_epoch,
        "refit_steps_per_epoch": refit_geometry.optimizer_steps_per_epoch,
        "refit_step_budget": refit_steps,
        "initial_validation_metrics": initial_metrics,
        "best_validation_metrics": state["best_metrics"],
        **amp_metrics(state),
        "historical_test_used": False,
        "final_holdout_used": False,
    }
    if decision_path.is_file() and load_json_object(decision_path) != decision:
        raise ConfigError("Une décision de développement différente existe déjà.")
    write_json_atomic(decision_path, decision)
    write_json_atomic(
        settings.shareable_output_directory / "full_development_metrics.json",
        {
            key: value
            for key, value in decision.items()
            if not key.endswith("_sha256") or key in {"manifest_sha256"}
        },
    )
    return decision


def run_refit(settings: FullTrainingSettings) -> dict[str, Any]:
    """Refit from pilot weights on train+validation for the frozen budget."""

    _require_training_still_allowed(settings)
    context = _load_context(settings, require_clean=True, write_reports=False)
    decision = load_json_object(
        settings.artifact_output_directory / DEVELOPMENT_DECISION_FILENAME
    )
    expected_decision = {
        "status": "frozen",
        "development_selection_finalized": True,
        "finalization_git_commit": context.code_commit,
        "finalized_from": "development-final-validation",
        "config_sha256": context.config_sha256,
        "manifest_sha256": context.dataset.manifest_sha256,
        "refit_selection_sha256": context.selection.refit_selection_sha256,
        "initial_checkpoint_sha256": context.initial_checkpoint_sha256,
    }
    require_matching_identity(
        decision,
        expected_decision,
        description="La décision de développement",
    )
    total_steps = int(decision["refit_step_budget"])
    if total_steps <= 0 or total_steps > 2_054:
        raise ConfigError("Le budget de refit gelé est hors limites.")

    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    device = _require_cuda(torch)
    _seed_everything(settings.seed, torch)
    runtime = _new_training_runtime(
        settings=settings,
        context=context,
        directory=settings.refit_checkpoint_directory,
        selection_hash=context.selection.refit_selection_sha256,
        stage="refit",
        total_steps=total_steps,
        torch=torch,
        transformers=transformers,
        device=device,
    )
    model, processor, collator, optimizer, scheduler, scaler, state = runtime
    refit_started_path = settings.artifact_output_directory / REFIT_STARTED_FILENAME
    refit_started = {
        "schema_version": 1,
        "status": "started",
        "code_commit": context.code_commit,
        "config_sha256": context.config_sha256,
        "manifest_sha256": context.dataset.manifest_sha256,
        "initial_checkpoint_sha256": context.initial_checkpoint_sha256,
        "refit_selection_sha256": context.selection.refit_selection_sha256,
        "refit_step_budget": total_steps,
    }
    if refit_started_path.is_file():
        require_matching_identity(
            load_json_object(refit_started_path),
            refit_started,
            description="Le marqueur de démarrage du refit",
        )
    else:
        write_json_atomic(refit_started_path, refit_started)
    if state.get("completed") is True:
        return load_json_object(
            settings.artifact_output_directory / FINAL_MODEL_MANIFEST_FILENAME
        )
    geometry = compute_step_geometry(
        len(context.selection.refit_rows),
        settings.train_batch_size,
        settings.gradient_accumulation_steps,
    )
    if geometry.optimizer_steps_per_epoch != 1_027:
        raise ConfigError("La géométrie du refit complet a changé.")
    started = perf_counter()
    duration_before_resume = float(state.get("training_duration_seconds", 0.0))
    torch.cuda.reset_peak_memory_stats(device)
    epoch = int(state["epoch"])
    while int(state["global_step"]) < total_steps:
        ordered = _ordered_train_rows(context.selection.refit_rows, settings.seed, epoch)
        micro_batches = _chunks(ordered, settings.train_batch_size)
        sizes = accumulation_group_sizes(
            len(micro_batches), settings.gradient_accumulation_steps
        )
        groups: list[Sequence[Sequence[ManifestRow]]] = []
        cursor = 0
        for size in sizes:
            groups.append(micro_batches[cursor : cursor + size])
            cursor += size
        completed_groups = (
            int(state["groups_completed_in_epoch"])
            if epoch == int(state["epoch"])
            else 0
        )
        for group_index, group in enumerate(groups):
            if group_index < completed_groups:
                continue
            if int(state["global_step"]) >= total_steps:
                break
            model.train()
            optimizer_result = _optimizer_group(
                row_batches=group,
                collator=collator,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                torch=torch,
                max_grad_norm=settings.max_grad_norm,
            )
            state["epoch"] = epoch
            state["groups_completed_in_epoch"] = group_index + 1
            optimizer_updated = apply_optimizer_outcome(
                state=state,
                result=optimizer_result,
                scheduler=scheduler,
                max_consecutive_amp_skips=settings.max_consecutive_amp_skips,
                stage="refit",
            )
            step = int(state["global_step"])
            cast(list[dict[str, Any]], state["log_history"]).append(
                {
                    "step": step,
                    "optimizer_attempt": int(state["optimizer_attempts"]),
                    "successful_optimizer_steps": int(
                        state["successful_optimizer_steps"]
                    ),
                    "optimizer_step_executed": optimizer_updated,
                    "amp_skipped": optimizer_result.amp_skipped,
                    "scale_before": optimizer_result.scale_before,
                    "scale_after": optimizer_result.scale_after,
                    "epoch": epoch + (group_index + 1) / len(groups),
                    "train_loss": optimizer_result.train_loss,
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                }
            )
            if not optimizer_updated:
                continue
            if step == 1 or step % settings.logging_steps == 0:
                print(
                    f"refit step={step:04d}/{total_steps} "
                    f"loss={optimizer_result.train_loss:.6f}"
                )
            if step % settings.refit_save_steps == 0 or step == total_steps:
                state["training_duration_seconds"] = duration_before_resume + (
                    perf_counter() - started
                )
                state["max_gpu_memory_mib"] = max(
                    float(state.get("max_gpu_memory_mib", 0.0)),
                    float(torch.cuda.max_memory_allocated(device)) / 1024**2,
                )
                _save_checkpoint(
                    directory=settings.refit_checkpoint_directory,
                    minimum_free_disk_gib=settings.minimum_free_disk_gib,
                    save_total_limit=settings.refit_save_total_limit,
                    best_checkpoint_name=None,
                    model=model,
                    processor=processor,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    state=state,
                    torch=torch,
                )
        epoch += 1
        state["epoch"] = epoch
        state["groups_completed_in_epoch"] = 0
    state["completed"] = True
    state["training_duration_seconds"] = duration_before_resume + (
        perf_counter() - started
    )
    state["max_gpu_memory_mib"] = max(
        float(state.get("max_gpu_memory_mib", 0.0)),
        float(torch.cuda.max_memory_allocated(device)) / 1024**2,
    )
    final_checkpoint = (
        settings.refit_checkpoint_directory
        / f"checkpoint-{int(state['global_step']):06d}"
    )
    write_json_atomic(final_checkpoint / TRAINER_STATE_FILENAME, state)
    write_json_atomic(
        final_checkpoint / "FROZEN.json",
        {
            "schema_version": 1,
            "status": "immutable_final_checkpoint",
            "run_id": state["run_id"],
            "optimizer_steps": state["global_step"],
        },
    )
    final_manifest = {
        "schema_version": 1,
        "status": "frozen",
        "code_commit": context.code_commit,
        "config_sha256": context.config_sha256,
        "manifest_sha256": context.dataset.manifest_sha256,
        "model_id": settings.expected_model_id,
        "model_revision": settings.expected_model_revision,
        "seed": settings.seed,
        "initial_checkpoint_name": settings.initial_checkpoint_path.name,
        "initial_checkpoint_sha256": context.initial_checkpoint_sha256,
        "refit_selection_sha256": context.selection.refit_selection_sha256,
        "train_audio_count": len(context.selection.refit_rows),
        "train_speaker_count": len(
            {row.speaker_id for row in context.selection.refit_rows}
        ),
        "optimizer_steps": int(state["global_step"]),
        "learning_rate": settings.learning_rate,
        **amp_metrics(state),
        "final_checkpoint_name": final_checkpoint.name,
        "final_checkpoint_sha256": directory_sha256(final_checkpoint),
        "training_duration_seconds": state["training_duration_seconds"],
        "max_gpu_memory_mib": state["max_gpu_memory_mib"],
        "historical_test_used": False,
        "final_holdout_used": False,
        "publication_allowed": False,
    }
    write_json_atomic(
        settings.artifact_output_directory / FINAL_MODEL_MANIFEST_FILENAME,
        final_manifest,
    )
    write_json_atomic(
        settings.shareable_output_directory / "full_refit_metrics.json",
        {
            "schema_version": 1,
            "status": "succeeded",
            "train_audio_count": final_manifest["train_audio_count"],
            "train_speaker_count": final_manifest["train_speaker_count"],
            "optimizer_steps": final_manifest["optimizer_steps"],
            "precision": final_manifest["precision"],
            "initial_grad_scale": final_manifest["initial_grad_scale"],
            "final_grad_scale": final_manifest["final_grad_scale"],
            "amp_skipped_steps": final_manifest["amp_skipped_steps"],
            "successful_optimizer_steps": final_manifest[
                "successful_optimizer_steps"
            ],
            "optimizer_attempts": final_manifest["optimizer_attempts"],
            "consecutive_amp_skips": final_manifest["consecutive_amp_skips"],
            "max_consecutive_amp_skips_observed": final_manifest[
                "max_consecutive_amp_skips_observed"
            ],
            "training_duration_seconds": final_manifest["training_duration_seconds"],
            "max_gpu_memory_mib": final_manifest["max_gpu_memory_mib"],
            "test_used": False,
            "publication_allowed": False,
        },
    )
    return final_manifest


def _evaluate_model(
    *,
    label: str,
    source: str | Path,
    revision: str | None,
    settings: FullTrainingSettings,
    context: FullContext,
    rows: Sequence[ManifestRow],
    torch: Any,
    transformers: Any,
    device: Any,
    evaluation_identity: Mapping[str, Any],
) -> EvaluationResult:
    processor = transformers.WhisperProcessor.from_pretrained(
        source,
        revision=revision,
        cache_dir=str(context.model_settings.cache_dir),
        local_files_only=True,
    )
    collator_settings = cast(
        PilotSettings,
        SimpleNamespace(
            train_split="test",
            validation_split="test",
            dataset_root=settings.dataset_root,
            max_audio_seconds=settings.max_audio_seconds,
        ),
    )
    collator = PilotCollator(
        collator_settings,
        processor,
        context.model_settings.expected_sampling_rate_hz,
        torch,
    )
    model = _load_model(
        source=source,
        model_revision=revision,
        cache_dir=context.model_settings.cache_dir,
        local_files_only=True,
        device=device,
        transformers=transformers,
    )
    result = _evaluate_validation(
        model=model,
        processor=processor,
        rows=rows,
        collator=collator,
        batch_size=settings.eval_batch_size,
        device=device,
        torch=torch,
        fp16=True,
    )
    _write_private_evaluation(
        settings.artifact_output_directory
        / "final_evaluation"
        / f"{label}_private.json",
        result,
        evaluation_identity=evaluation_identity,
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_final_evaluation(settings: FullTrainingSettings) -> dict[str, Any]:
    """Evaluate one frozen model event on final_holdout and seal the receipt."""

    if (
        os.getenv("IVOIREVOICE_CONFIRM_FINAL_HOLDOUT")
        != settings.final_holdout_confirmation
    ):
        raise ConfigError("Confirmation explicite final_holdout absente.")
    context = _load_context(settings, require_clean=True, write_reports=False)
    final_manifest = load_json_object(
        settings.artifact_output_directory / FINAL_MODEL_MANIFEST_FILENAME
    )
    final_checkpoint = (
        settings.refit_checkpoint_directory
        / str(final_manifest.get("final_checkpoint_name", ""))
    )
    expected = {
        "status": "frozen",
        "code_commit": context.code_commit,
        "config_sha256": context.config_sha256,
        "manifest_sha256": context.dataset.manifest_sha256,
        "refit_selection_sha256": context.selection.refit_selection_sha256,
        "final_checkpoint_sha256": directory_sha256(final_checkpoint),
    }
    require_matching_identity(
        final_manifest,
        expected,
        description="Le modèle final",
    )
    receipt_path = (
        settings.artifact_output_directory / FINAL_EVALUATION_RECEIPT_FILENAME
    )
    receipt_identity = {
        "schema_version": 1,
        "final_checkpoint_sha256": final_manifest["final_checkpoint_sha256"],
        "final_holdout_selection_sha256": (
            context.selection.final_holdout_selection_sha256
        ),
        "manifest_sha256": context.dataset.manifest_sha256,
        "code_commit": context.code_commit,
    }
    receipt: dict[str, Any] | None = None
    if receipt_path.is_file():
        receipt = load_json_object(receipt_path)
        require_matching_identity(
            receipt,
            receipt_identity,
            description="Le reçu final_holdout",
        )
        if receipt.get("status") == "completed":
            return load_json_object(
                settings.artifact_output_directory / FINAL_EVALUATION_METRICS_FILENAME
            )
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    device = _require_cuda(torch)
    _seed_everything(settings.seed, torch)
    holdout_rows = tuple(
        row
        for row in context.dataset.rows
        if row.utterance_id in context.selection.final_holdout_ids
    )
    validate_final_holdout_files(holdout_rows, settings)
    if receipt is None:
        receipt = {
            **receipt_identity,
            "status": "started",
            "completed_models": [],
        }
        write_json_atomic(receipt_path, receipt)
    sources: tuple[tuple[str, str | Path, str | None, str], ...] = (
        (
            "whisper_tiny_baseline",
            settings.expected_model_id,
            settings.expected_model_revision,
            settings.expected_model_revision,
        ),
        (
            "whisper_tiny_pilot",
            settings.initial_checkpoint_path,
            None,
            context.initial_checkpoint_sha256,
        ),
        (
            "whisper_tiny_full_refit",
            final_checkpoint,
            None,
            str(final_manifest["final_checkpoint_sha256"]),
        ),
    )
    metrics: dict[str, Any] = {}
    completed = set(cast(list[str], receipt.get("completed_models", [])))
    for label, source, revision, model_identity in sources:
        cached_path = (
            settings.artifact_output_directory
            / "final_evaluation"
            / f"{label}_private.json"
        )
        evaluation_identity = {
            "label": label,
            "model_identity": model_identity,
            "final_holdout_selection_sha256": (
                context.selection.final_holdout_selection_sha256
            ),
        }
        if cached_path.is_file():
            cached = load_json_object(cached_path)
            cached_identity = cached.get("evaluation_identity")
            if not isinstance(cached_identity, Mapping):
                raise ConfigError("Un cache final_holdout ne porte pas son identité.")
            require_matching_identity(
                cached_identity,
                evaluation_identity,
                description=f"Le cache final_holdout {label}",
            )
            metrics[label] = _public_metrics(
                cast(dict[str, Any], cached["metrics"])
            )
            completed.add(label)
            receipt["completed_models"] = sorted(completed)
            write_json_atomic(receipt_path, receipt)
            continue
        result = _evaluate_model(
            label=label,
            source=source,
            revision=revision,
            settings=settings,
            context=context,
            rows=holdout_rows,
            torch=torch,
            transformers=transformers,
            device=device,
            evaluation_identity=evaluation_identity,
        )
        metrics[label] = _public_metrics(result.metrics)
        completed.add(label)
        receipt["completed_models"] = sorted(completed)
        write_json_atomic(receipt_path, receipt)
    baseline = cast(dict[str, Any], metrics["whisper_tiny_baseline"])
    final = cast(dict[str, Any], metrics["whisper_tiny_full_refit"])
    output = {
        "schema_version": 1,
        "status": "completed",
        "dataset": "final_holdout",
        "audio_count": len(holdout_rows),
        "historical_pilot_mixed": False,
        "models": metrics,
        "wer_absolute_reduction": float(baseline["wer_micro"])
        - float(final["wer_micro"]),
        "wer_relative_reduction_percent": (
            (float(baseline["wer_micro"]) - float(final["wer_micro"]))
            / float(baseline["wer_micro"])
            * 100
            if float(baseline["wer_micro"])
            else 0.0
        ),
        "cer_absolute_reduction": float(baseline["cer_micro"])
        - float(final["cer_micro"]),
        "cer_relative_reduction_percent": (
            (float(baseline["cer_micro"]) - float(final["cer_micro"]))
            / float(baseline["cer_micro"])
            * 100
            if float(baseline["cer_micro"])
            else 0.0
        ),
        "scientific_conclusion": (
            "improved"
            if (
                float(final["wer_micro"]) < float(baseline["wer_micro"])
                and float(final["cer_micro"]) < float(baseline["cer_micro"])
            )
            else "not_demonstrated"
        ),
        "no_retraining_after_holdout": True,
        "publication_allowed": False,
    }
    write_json_atomic(
        settings.artifact_output_directory / FINAL_EVALUATION_METRICS_FILENAME,
        output,
    )
    receipt["status"] = "completed"
    write_json_atomic(receipt_path, receipt)
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning complet Whisper Tiny.")
    parser.add_argument(
        "--config",
        default=str(CONFIG_PATH),
        help="Configuration complète locale.",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "preflight",
            "fp16-diagnostic",
            "development",
            "development-final-validation",
            "refit",
            "final-evaluation",
        ),
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point that never advances to another stage implicitly."""

    args = _parse_args()
    try:
        settings = load_full_training_settings(args.config)
        runners = {
            "preflight": run_preflight,
            "development": run_development,
            "refit": run_refit,
            "final-evaluation": run_final_evaluation,
        }
        if args.stage == "fp16-diagnostic":
            from ivoirevoice.training.fp16_diagnostic import run_fp16_diagnostic

            result = run_fp16_diagnostic(settings)
        elif args.stage == "development-final-validation":
            from ivoirevoice.training.development_final_validation import (
                run_development_final_validation,
            )

            result = run_development_final_validation(settings)
        else:
            result = runners[args.stage](settings)
    except IvoireVoiceError as exc:
        print(f"ERREUR: {exc}")
        return 1
    print(f"Stage {args.stage} : {result.get('status', 'completed')}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
