"""Shared deterministic mechanics for Whisper development and refit workflows."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from ivoirevoice.exceptions import ConfigError


@dataclass(frozen=True, slots=True)
class StepGeometry:
    """Exact micro-batch and optimizer-step geometry."""

    audio_count: int
    batch_size: int
    gradient_accumulation_steps: int
    micro_batches_per_epoch: int
    optimizer_steps_per_epoch: int


@dataclass(frozen=True, slots=True)
class OptimizerGroupResult:
    """Outcome of one accumulated optimizer attempt."""

    train_loss: float
    scale_before: float
    scale_after: float
    optimizer_step_executed: bool
    amp_skipped: bool


def restore_runtime_states(
    *,
    checkpoint: Path,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    torch: Any,
    device: Any,
) -> None:
    """Restore every mutable runtime state required for a true resume."""

    optimizer.load_state_dict(
        torch.load(checkpoint / "optimizer.pt", map_location=device, weights_only=True)
    )
    scheduler.load_state_dict(
        torch.load(checkpoint / "scheduler.pt", map_location=device, weights_only=True)
    )
    scaler.load_state_dict(
        torch.load(checkpoint / "scaler.pt", map_location=device, weights_only=True)
    )


def initialize_or_validate_amp_state(
    state: dict[str, Any],
    scaler: Any,
    *,
    resumed: bool,
) -> None:
    """Initialize AMP provenance or reject an inconsistent resumed state."""

    current_scale = float(scaler.get_scale())
    if not resumed:
        state["initial_grad_scale"] = current_scale
        state["final_grad_scale"] = current_scale
        return
    required = {
        "global_step",
        "successful_optimizer_steps",
        "optimizer_attempts",
        "amp_skipped_steps",
        "consecutive_amp_skips",
        "max_consecutive_amp_skips_observed",
        "initial_grad_scale",
        "final_grad_scale",
        "precision",
    }
    missing = sorted(required - state.keys())
    if missing:
        raise ConfigError(
            "Le checkpoint de reprise ne contient pas l'état AMP complet "
            f"(champs : {', '.join(missing)})."
        )
    successful = int(state["successful_optimizer_steps"])
    skipped = int(state["amp_skipped_steps"])
    attempts = int(state["optimizer_attempts"])
    consecutive = int(state["consecutive_amp_skips"])
    maximum = int(state["max_consecutive_amp_skips_observed"])
    initial_scale = float(state["initial_grad_scale"])
    final_scale = float(state["final_grad_scale"])
    counters_valid = (
        min(successful, skipped, attempts, consecutive, maximum) >= 0
        and successful == int(state["global_step"])
        and attempts == successful + skipped
        and consecutive <= skipped
        and maximum >= consecutive
    )
    scales_valid = (
        math.isfinite(initial_scale)
        and math.isfinite(final_scale)
        and initial_scale > 0
        and final_scale > 0
        and final_scale == current_scale
    )
    if not counters_valid or not scales_valid or state["precision"] != "fp16":
        raise ConfigError("L'état AMP du checkpoint de reprise est incohérent.")


def amp_metrics(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return aggregate-only AMP provenance for local reports."""

    return {
        "precision": "fp16",
        "initial_grad_scale": float(state["initial_grad_scale"]),
        "final_grad_scale": float(state["final_grad_scale"]),
        "amp_skipped_steps": int(state["amp_skipped_steps"]),
        "successful_optimizer_steps": int(state["successful_optimizer_steps"]),
        "optimizer_attempts": int(state["optimizer_attempts"]),
        "consecutive_amp_skips": int(state["consecutive_amp_skips"]),
        "max_consecutive_amp_skips_observed": int(
            state["max_consecutive_amp_skips_observed"]
        ),
    }


def apply_optimizer_outcome(
    *,
    state: dict[str, Any],
    result: OptimizerGroupResult,
    scheduler: Any,
    max_consecutive_amp_skips: int,
    stage: str,
) -> bool:
    """Advance update clocks only when GradScaler executed the optimizer step."""

    if result.optimizer_step_executed == result.amp_skipped:
        raise ConfigError("L'issue AMP de la tentative optimizer est incohérente.")
    state["optimizer_attempts"] = int(state["optimizer_attempts"]) + 1
    state["final_grad_scale"] = result.scale_after
    if result.amp_skipped:
        state["amp_skipped_steps"] = int(state["amp_skipped_steps"]) + 1
        consecutive = int(state["consecutive_amp_skips"]) + 1
        state["consecutive_amp_skips"] = consecutive
        state["max_consecutive_amp_skips_observed"] = max(
            int(state["max_consecutive_amp_skips_observed"]),
            consecutive,
        )
        print(
            f"{stage} amp_skip attempt={state['optimizer_attempts']} "
            f"consecutive={consecutive}/{max_consecutive_amp_skips} "
            f"scale={result.scale_before:.0f}->{result.scale_after:.0f}"
        )
        if consecutive > max_consecutive_amp_skips:
            raise ConfigError(
                "Instabilité FP16 : nombre maximal de skips AMP consécutifs dépassé "
                f"({consecutive} > {max_consecutive_amp_skips}, "
                f"scale {result.scale_before:.0f}->{result.scale_after:.0f})."
            )
        return False
    scheduler.step()
    state["global_step"] = int(state["global_step"]) + 1
    state["successful_optimizer_steps"] = int(state["successful_optimizer_steps"]) + 1
    state["consecutive_amp_skips"] = 0
    return True


def compute_step_geometry(
    audio_count: int,
    batch_size: int,
    gradient_accumulation_steps: int,
) -> StepGeometry:
    """Return exact ceil-based step counts."""

    if min(audio_count, batch_size, gradient_accumulation_steps) <= 0:
        raise ConfigError("La géométrie d'entraînement exige des valeurs positives.")
    micro_batches = math.ceil(audio_count / batch_size)
    optimizer_steps = math.ceil(micro_batches / gradient_accumulation_steps)
    return StepGeometry(
        audio_count=audio_count,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        micro_batches_per_epoch=micro_batches,
        optimizer_steps_per_epoch=optimizer_steps,
    )


def refit_step_budget(
    best_development_step: int,
    development_steps_per_epoch: int,
    refit_steps_per_epoch: int,
) -> int:
    """Map the selected development exposure to train+validation deterministically."""

    if min(
        best_development_step,
        development_steps_per_epoch,
        refit_steps_per_epoch,
    ) <= 0:
        raise ConfigError("Le budget de refit exige des nombres de steps positifs.")
    scaled = (
        best_development_step * refit_steps_per_epoch / development_steps_per_epoch
    )
    return math.floor(scaled + 0.5)


def validation_milestones(
    steps_per_epoch: int,
    epoch_count: int,
    evaluations_per_epoch: int,
) -> frozenset[int]:
    """Return unique global quarter-like validation milestones."""

    if min(steps_per_epoch, epoch_count, evaluations_per_epoch) <= 0:
        raise ConfigError("Le calendrier de validation doit être strictement positif.")
    milestones: set[int] = set()
    for epoch in range(epoch_count):
        offset = epoch * steps_per_epoch
        for index in range(1, evaluations_per_epoch + 1):
            local_step = math.ceil(index * steps_per_epoch / evaluations_per_epoch)
            milestones.add(offset + local_step)
    return frozenset(milestones)


def accumulation_group_sizes(
    micro_batch_count: int,
    accumulation_steps: int,
) -> tuple[int, ...]:
    """Expose the real divisor for every accumulation group, including the tail."""

    if min(micro_batch_count, accumulation_steps) <= 0:
        raise ConfigError("L'accumulation exige des valeurs positives.")
    full_groups, remainder = divmod(micro_batch_count, accumulation_steps)
    sizes = [accumulation_steps] * full_groups
    if remainder:
        sizes.append(remainder)
    return tuple(sizes)


def metric_rank(metrics: Mapping[str, Any], step: int) -> tuple[float, float, float, int]:
    """Rank checkpoints by WER, CER, loss, then earliest step."""

    if step <= 0:
        raise ConfigError("Le step d'un checkpoint doit être positif.")
    values = (
        float(metrics["wer_micro"]),
        float(metrics["cer_micro"]),
        float(metrics["validation_loss"]),
    )
    if not all(math.isfinite(value) for value in values):
        raise ConfigError("Les métriques de sélection doivent être finies.")
    return (*values, step)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write strict JSON atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        serialized = json.dumps(
            dict(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"JSON non sérialisable pour {path.name} : {exc}") from exc
    temporary.write_text(serialized + "\n", encoding="utf-8")
    temporary.replace(path)


def free_disk_gib(path: Path) -> float:
    """Return available capacity after ensuring the external checkpoint root."""

    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1024**3


def checkpoint_directories(directory: Path) -> list[Path]:
    """Return numeric checkpoint directories in optimizer-step order."""

    return sorted(
        (
            path
            for path in directory.glob("checkpoint-*")
            if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
        ),
        key=lambda path: int(path.name.removeprefix("checkpoint-")),
    )


def latest_complete_checkpoint(
    directory: Path,
    required_filenames: Sequence[str],
) -> Path | None:
    """Return the latest checkpoint containing every required resume file."""

    valid = [
        path
        for path in checkpoint_directories(directory)
        if all((path / name).is_file() for name in required_filenames)
    ]
    return valid[-1] if valid else None


def prune_checkpoints(
    directory: Path,
    limit: int,
    *,
    best_checkpoint_name: str | None,
) -> None:
    """Keep at most limit checkpoints: best plus recent recoverable ones."""

    if limit <= 0:
        raise ConfigError("La rétention des checkpoints doit être positive.")
    checkpoints = checkpoint_directories(directory)
    keep: set[str] = set()
    recent_limit = limit
    if best_checkpoint_name:
        keep.add(best_checkpoint_name)
        recent_limit = max(0, limit - 1)
    if recent_limit:
        keep.update(path.name for path in checkpoints[-recent_limit:])
    for path in checkpoints:
        if path.name not in keep:
            shutil.rmtree(path)


def save_checkpoint_atomic(
    *,
    directory: Path,
    minimum_free_disk_gib: float,
    save_total_limit: int,
    best_checkpoint_name: str | None,
    state_filename: str,
    model: Any,
    processor: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    state: Mapping[str, Any],
    torch: Any,
) -> Path:
    """Persist model and optimizer state through one atomic directory replace."""

    if free_disk_gib(directory) < minimum_free_disk_gib:
        raise ConfigError("Espace disque insuffisant avant checkpoint.")
    step = int(state["global_step"])
    final_path = directory / f"checkpoint-{step:06d}"
    temporary = directory / f".checkpoint-{step:06d}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary, safe_serialization=True)
    processor.save_pretrained(temporary)
    torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
    torch.save(scheduler.state_dict(), temporary / "scheduler.pt")
    torch.save(scaler.state_dict(), temporary / "scaler.pt")
    write_json_atomic(temporary / state_filename, state)
    if final_path.exists():
        shutil.rmtree(final_path)
    temporary.replace(final_path)
    prune_checkpoints(
        directory,
        save_total_limit,
        best_checkpoint_name=best_checkpoint_name,
    )
    return final_path


def load_json_object(path: Path) -> dict[str, Any]:
    """Load one JSON object with a privacy-safe error."""

    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Impossible de lire {path.name} : {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path.name} doit contenir un objet JSON.")
    return cast(dict[str, Any], raw)


def directory_sha256(path: Path) -> str:
    """Hash checkpoint files and relative names without exposing their root."""

    if not path.is_dir():
        raise ConfigError("Le checkpoint attendu est absent.")
    files = tuple(sorted(item for item in path.rglob("*") if item.is_file()))
    if not files:
        raise ConfigError("Le checkpoint attendu est vide.")
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def selection_sha256(groups: Mapping[str, Sequence[tuple[str, str]]]) -> str:
    """Hash stage, pseudonymous ID and audio hash for deterministic selections."""

    digest = hashlib.sha256()
    for stage, rows in sorted(groups.items()):
        for identifier, audio_hash in sorted(rows):
            digest.update(f"{stage}:{identifier}:{audio_hash}\n".encode())
    return digest.hexdigest()


def identity_sha256(identity: Mapping[str, Any]) -> str:
    """Hash one canonical, path-free run identity."""

    try:
        encoded = json.dumps(
            dict(identity),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Identité de run non sérialisable : {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def require_matching_identity(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    description: str,
) -> None:
    """Fail closed when any immutable identity component changes."""

    mismatched = sorted(
        key for key, expected_value in expected.items() if actual.get(key) != expected_value
    )
    if mismatched:
        raise ConfigError(
            f"{description} ne correspond plus au run gelé "
            f"(champs : {', '.join(mismatched)})."
        )


def git_provenance(root: Path) -> tuple[str, bool]:
    """Return HEAD and whether tracked/untracked publication candidates are clean."""

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ConfigError(f"Impossible de lire la provenance Git : {exc}") from exc
    return commit, not bool(status.strip())
