"""Fail-closed checks for the local, corpus-free Gradio demonstration."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from ivoirevoice.models.whisper import WhisperBackend, runtime_labels
from ivoirevoice.services.transcription_service import (
    build_model_registry,
    load_model_catalog,
)
from ivoirevoice.training.whisper_finetune import directory_sha256

EXPECTED_BRANCH = "main"
EXPECTED_CHECKPOINT = "checkpoint-002052"
EXPECTED_GPU = "NVIDIA GeForce RTX 5070 Ti Laptop GPU"
EXPECTED_HOST = "127.0.0.1"
MIN_VRAM_GIB = 10.0
MIN_FREE_DISK_BYTES = 2 * 1024**3
DEMO_AUDIO_CONFIRMATION = "SAFE_EXTERNAL_DEMO_AUDIO"
REQUIRED_MODEL_FILES = frozenset(
    {
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "tokenizer.json",
    }
)


@dataclass(frozen=True, slots=True)
class Check:
    """One concise, privacy-safe preflight result."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class ModelProbe:
    """Aggregate-only result from loading the final model on CUDA."""

    gpu_name: str
    vram_gib: float
    load_seconds: float
    peak_vram_mib: float


@dataclass(frozen=True, slots=True)
class DemoPreflightReport:
    """Complete result without checkpoint or audio paths."""

    checks: tuple[Check, ...]
    warnings: tuple[str, ...]
    tiny_cached: bool
    small_cached: bool
    final_model_local: bool
    model_probe: ModelProbe | None
    demo_audio_name: str | None

    @property
    def ready(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def demo_audio_required(self) -> bool:
        return "DEMO AUDIO REQUIRED" in self.warnings


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_state(root: Path) -> tuple[str, bool]:
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=root, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=root,
        text=True,
    )
    return branch, not bool(status.strip())


def _python_runtime(root: Path) -> tuple[bool, str]:
    expected = (root / ".venv" / "bin" / "python").resolve()
    active = Path(sys.executable).resolve()
    valid = sys.version_info[:2] == (3, 11) and active == expected
    return valid, f"Python {sys.version_info.major}.{sys.version_info.minor} / .venv"


def _torch_runtime() -> tuple[bool, bool, str, float]:
    try:
        import torch
    except ImportError:
        return False, False, "indisponible", 0.0
    cuda = bool(torch.cuda.is_available())
    if not cuda:
        return True, False, "indisponible", 0.0
    properties = torch.cuda.get_device_properties(0)
    return True, True, str(properties.name), float(properties.total_memory / 1024**3)


def _port_available(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.bind((host, port))
    except OSError:
        return False
    return True


def _model_snapshot_cached(cache_root: Path, model_id: str, revision: str) -> bool:
    cache_name = "models--" + model_id.replace("/", "--")
    snapshot = cache_root / cache_name / "snapshots" / revision
    return snapshot.is_dir() and all((snapshot / name).is_file() for name in REQUIRED_MODEL_FILES)


def _demo_inputs_ignored(root: Path) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "demo_inputs/test.wav"],
        cwd=root,
        check=False,
    )
    return result.returncode == 0


def _load_expected_hash(root: Path) -> str:
    metrics_path = root / "reports" / "final_holdout_metrics.json"
    raw: object = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Le rapport final public est invalide.")
    value = raw.get("final_model_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("Le hash final public est invalide.")
    return value


def _load_final_model() -> ModelProbe:
    import torch

    catalog = load_model_catalog()
    backend = build_model_registry(catalog).create("whisper_tiny_dioula_final")
    if not isinstance(backend, WhisperBackend):
        raise TypeError("Le backend final n'est pas Whisper.")
    device, precision = runtime_labels(backend.settings)
    if device != "cuda" or precision != "float16":
        raise RuntimeError("Le modèle final doit utiliser CUDA FP16.")
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    try:
        backend.load()
        pipeline = getattr(backend, "_pipeline", None)
        model = getattr(pipeline, "model", None)
        parameter = next(model.parameters()) if model is not None else None
        if parameter is None or parameter.device.type != "cuda":
            raise RuntimeError("Le modèle final n'est pas placé sur CUDA.")
        probe = torch.ones(1024, device="cuda:0", dtype=torch.float16)
        if not bool(torch.isfinite(probe).all()):
            raise RuntimeError("Le tenseur CUDA de contrôle n'est pas fini.")
        torch.cuda.synchronize()
        load_seconds = perf_counter() - started
        peak_vram_mib = float(torch.cuda.max_memory_allocated() / 1024**2)
        properties = torch.cuda.get_device_properties(0)
        return ModelProbe(
            gpu_name=str(properties.name),
            vram_gib=float(properties.total_memory / 1024**3),
            load_seconds=load_seconds,
            peak_vram_mib=peak_vram_mib,
        )
    finally:
        backend.unload()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _demo_audio_check(
    root: Path,
    allowed_extensions: tuple[str, ...],
) -> tuple[Check | None, str | None, tuple[str, ...]]:
    raw_path = os.getenv("IVOIREVOICE_DEMO_AUDIO_PATH", "").strip()
    if not raw_path:
        return None, None, ("DEMO AUDIO REQUIRED",)
    path = Path(raw_path).expanduser().resolve()
    confirmation = os.getenv("IVOIREVOICE_DEMO_AUDIO_CONFIRMATION", "")
    if confirmation != DEMO_AUDIO_CONFIRMATION:
        return Check("demo_audio", False, "confirmation explicite absente"), None, ()
    if not path.is_file() or path.suffix.lower() not in allowed_extensions:
        return Check("demo_audio", False, "fichier absent ou format interdit"), None, ()
    protected_names = (
        "IVOIREVOICE_DIOULA_DATA_DIR",
        "IVOIREVOICE_ARTIFACTS_DIR",
        "IVOIREVOICE_CHECKPOINT_DIR",
    )
    protected = tuple(
        Path(value).expanduser().resolve()
        for name in protected_names
        if (value := os.getenv(name, "").strip())
    )
    if any(_is_relative_to(path, protected_root) for protected_root in protected):
        return Check("demo_audio", False, "fichier situé sous une racine privée"), None, ()
    if _is_relative_to(path, root):
        relative = path.relative_to(root)
        tracked = (
            subprocess.run(
                ["git", "ls-files", "--error-unmatch", str(relative)],
                cwd=root,
                check=False,
                capture_output=True,
            ).returncode
            == 0
        )
        if tracked:
            return Check("demo_audio", False, "audio suivi par Git"), None, ()
    return Check("demo_audio", True, "audio externe explicitement confirmé"), path.name, ()


def run_demo_preflight(root: Path | None = None) -> DemoPreflightReport:
    """Run metadata/runtime checks only; never build a dataset or transcribe audio."""

    repository_root = (root or _repository_root()).resolve()
    checks: list[Check] = []
    warnings: list[str] = []

    try:
        branch, clean = _git_state(repository_root)
    except (OSError, subprocess.SubprocessError):
        branch, clean = "indisponible", False
    checks.append(Check("git_branch", branch == EXPECTED_BRANCH, branch))
    checks.append(Check("git_clean", clean, "propre" if clean else "modifications présentes"))

    python_valid, python_detail = _python_runtime(repository_root)
    checks.append(Check("python", python_valid, python_detail))
    torch_installed, cuda_available, gpu_name, vram_gib = _torch_runtime()
    checks.append(Check("torch", torch_installed, "installé" if torch_installed else "absent"))
    checks.append(Check("cuda", cuda_available, gpu_name))
    gpu_valid = cuda_available and gpu_name == EXPECTED_GPU and vram_gib >= MIN_VRAM_GIB
    checks.append(
        Check(
            "gpu_target",
            gpu_valid,
            f"{gpu_name} / {vram_gib:.2f} GiB",
        )
    )

    checkpoint_raw = os.getenv("IVOIREVOICE_DIOULA_FINAL_MODEL_PATH", "").strip()
    checkpoint = Path(checkpoint_raw).expanduser().resolve() if checkpoint_raw else None
    final_local = bool(
        checkpoint
        and checkpoint.name == EXPECTED_CHECKPOINT
        and checkpoint.is_dir()
        and all((checkpoint / name).is_file() for name in REQUIRED_MODEL_FILES)
    )
    checks.append(
        Check(
            "final_checkpoint",
            final_local,
            EXPECTED_CHECKPOINT if final_local else "absent, incomplet ou identité incorrecte",
        )
    )

    hash_valid = False
    if final_local and checkpoint is not None:
        try:
            hash_valid = directory_sha256(checkpoint) == _load_expected_hash(repository_root)
        except (OSError, ValueError):
            hash_valid = False
    checks.append(Check("final_hash", hash_valid, "identité gelée" if hash_valid else "invalide"))

    catalog = None
    try:
        catalog = load_model_catalog()
        labels = tuple(model.display_name for model in catalog.enabled_models)
        expected_labels = (
            "Whisper Tiny — Baseline",
            "Whisper Small — Baseline",
            "Whisper Tiny — Dioula Final",
        )
        config_valid = labels == expected_labels
    except Exception:
        config_valid = False
    checks.append(Check("ui_config", config_valid, "trois modèles attendus"))

    cache_raw = os.getenv("IVOIREVOICE_MODEL_CACHE_DIR", "").strip()
    cache_root = Path(cache_raw).expanduser().resolve() if cache_raw else Path()
    tiny_cached = bool(cache_raw) and _model_snapshot_cached(
        cache_root,
        "openai/whisper-tiny",
        "be0ba7c2f24f0127b27863a23a08002af4c2c279",
    )
    small_cached = bool(cache_raw) and _model_snapshot_cached(
        cache_root,
        "openai/whisper-small",
        "973afd24965f72e36ca33b3055d56a652f456b4d",
    )
    checks.append(Check("tiny_cache", tiny_cached, "révision épinglée"))
    checks.append(Check("small_cache", small_cached, "révision épinglée"))
    demo_inputs_ignored = _demo_inputs_ignored(repository_root)
    checks.append(
        Check(
            "demo_inputs_ignored",
            demo_inputs_ignored,
            "protégé par .gitignore" if demo_inputs_ignored else "non protégé",
        )
    )

    host = os.getenv("IVOIREVOICE_UI_HOST", "127.0.0.1")
    try:
        port = int(os.getenv("IVOIREVOICE_UI_PORT", "7860"))
        port_free = host == EXPECTED_HOST and 1 <= port <= 65535 and _port_available(host, port)
    except ValueError:
        port, port_free = 0, False
    checks.append(Check("gradio_port", port_free, f"{host}:{port}"))

    try:
        free_disk = shutil.disk_usage(repository_root).free
    except OSError:
        free_disk = 0
    checks.append(
        Check(
            "disk",
            free_disk >= MIN_FREE_DISK_BYTES,
            f"{free_disk / 1024**3:.1f} GiB libres",
        )
    )

    demo_audio_name: str | None = None
    if catalog is not None:
        demo_check, demo_audio_name, audio_warnings = _demo_audio_check(
            repository_root,
            catalog.allowed_extensions,
        )
        warnings.extend(audio_warnings)
        if demo_check is not None:
            checks.append(demo_check)

    model_probe: ModelProbe | None = None
    prerequisites = (
        branch == EXPECTED_BRANCH
        and clean
        and python_valid
        and torch_installed
        and cuda_available
        and gpu_valid
        and final_local
        and hash_valid
        and config_valid
    )
    if prerequisites:
        try:
            model_probe = _load_final_model()
            load_valid = model_probe.gpu_name == gpu_name and model_probe.vram_gib > 0
        except Exception as exc:
            load_valid = False
            warnings.append(f"Chargement final refusé ({type(exc).__name__}).")
        checks.append(Check("final_model_load", load_valid, "CUDA FP16"))
    else:
        checks.append(Check("final_model_load", False, "prérequis non satisfaits"))

    return DemoPreflightReport(
        checks=tuple(checks),
        warnings=tuple(warnings),
        tiny_cached=tiny_cached,
        small_cached=small_cached,
        final_model_local=final_local,
        model_probe=model_probe,
        demo_audio_name=demo_audio_name,
    )


def _print_report(report: DemoPreflightReport) -> None:
    for check in report.checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}: {check.detail}")
    for warning in report.warnings:
        print(f"[WARN] {warning}")
    print(f"BASELINE TINY CACHED: {'YES' if report.tiny_cached else 'NO'}")
    print(f"BASELINE SMALL CACHED: {'YES' if report.small_cached else 'NO'}")
    print(f"FINAL MODEL LOCAL: {'YES' if report.final_model_local else 'NO'}")
    offline = report.tiny_cached and report.small_cached and report.final_model_local
    print(f"DEMO CAN RUN OFFLINE: {'YES' if offline else 'NO'}")
    if report.model_probe is not None:
        print(f"GPU: {report.model_probe.gpu_name}")
        print(f"VRAM: {report.model_probe.vram_gib:.2f} GiB")
        print(f"FINAL MODEL LOAD: {report.model_probe.load_seconds:.3f} s")
        print(f"FINAL MODEL PEAK VRAM: {report.model_probe.peak_vram_mib:.2f} MiB")
    if not report.ready:
        print("NOT READY FOR DEMO")
    elif report.demo_audio_required:
        print("READY WITH DEMO AUDIO REQUIRED")
    else:
        print("READY FOR DEMO")


def main() -> int:
    try:
        report = run_demo_preflight()
    except Exception as exc:
        print(f"[FAIL] demo_preflight: {type(exc).__name__}")
        print("NOT READY FOR DEMO")
        return 1
    _print_report(report)
    return 0 if report.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
