"""Privacy-safe diagnostics for the optional local ASR environment."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ivoirevoice.exceptions import ConfigError, IvoireVoiceError

PACKAGE_NAMES = (
    "torch",
    "torchaudio",
    "transformers",
    "datasets",
    "jiwer",
    "numpy",
    "soundfile",
)


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGE_NAMES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _memory_bytes() -> tuple[int | None, int | None]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            name, separator, raw_value = line.partition(":")
            if not separator:
                continue
            amount = raw_value.strip().split(maxsplit=1)[0]
            if amount.isdigit():
                values[name] = int(amount) * 1024
    except (OSError, UnicodeError):
        return None, None
    return values.get("MemTotal"), values.get("MemAvailable")


def _existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _disk_bytes(path: Path) -> tuple[int, int]:
    try:
        usage = shutil.disk_usage(_existing_parent(path))
    except OSError as exc:
        raise ConfigError(f"Impossible d'inspecter l'espace disque : {exc}") from exc
    return usage.total, usage.free


def _nvidia_smi_available() -> bool:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return False
    try:
        result = subprocess.run(
            [executable, "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def _torch_capabilities() -> dict[str, Any]:
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return {
            "installed": False,
            "version": None,
            "cuda_available": False,
            "cuda_version": None,
            "gpu_name": None,
            "gpu_memory_total_bytes": None,
            "gpu_memory_free_bytes": None,
            "float16_supported": False,
            "bfloat16_supported": False,
        }

    cuda_available = bool(torch.cuda.is_available())
    gpu_name: str | None = None
    total_memory: int | None = None
    free_memory: int | None = None
    bfloat16_supported = False
    if cuda_available:
        gpu_name = str(torch.cuda.get_device_name(0))
        try:
            free_memory, total_memory = (int(value) for value in torch.cuda.mem_get_info(0))
        except (AttributeError, RuntimeError):
            properties = torch.cuda.get_device_properties(0)
            total_memory = int(properties.total_memory)
        checker = getattr(torch.cuda, "is_bf16_supported", None)
        bfloat16_supported = bool(checker()) if callable(checker) else False
    return {
        "installed": True,
        "version": str(torch.__version__),
        "cuda_available": cuda_available,
        "cuda_version": str(torch.version.cuda) if torch.version.cuda else None,
        "gpu_name": gpu_name,
        "gpu_memory_total_bytes": total_memory,
        "gpu_memory_free_bytes": free_memory,
        "float16_supported": cuda_available,
        "bfloat16_supported": bfloat16_supported,
    }


def collect_environment_report(artifacts_root: Path) -> dict[str, Any]:
    """Collect only aggregate system facts and never persist a local absolute path."""

    ram_total, ram_available = _memory_bytes()
    disk_total, disk_available = _disk_bytes(artifacts_root)
    torch_capabilities = _torch_capabilities()
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "cpu_logical_count": os.cpu_count(),
        "ram_total_bytes": ram_total,
        "ram_available_bytes": ram_available,
        "disk_total_bytes": disk_total,
        "disk_available_bytes": disk_available,
        "nvidia_driver_usable": _nvidia_smi_available(),
        "packages": _package_versions(),
        "torch": torch_capabilities,
        "recommended_device": ("cuda" if torch_capabilities["cuda_available"] else "cpu"),
        "recommended_dtype": ("float16" if torch_capabilities["cuda_available"] else "float32"),
        "privacy_checks": {
            "absolute_personal_paths_absent": True,
            "environment_values_absent": True,
        },
    }
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    if "/home/" in serialized or "\\Users\\" in serialized:
        raise ConfigError("Le diagnostic contient un chemin personnel.")
    return report


def write_environment_report(report: dict[str, Any], path: Path) -> None:
    """Atomically write the external aggregate report."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        with temporary_path.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        temporary_path.replace(path)
    except OSError as exc:
        raise ConfigError(f"Impossible d'écrire le diagnostic ML : {exc}") from exc


def _artifacts_root() -> Path:
    raw_root = os.getenv("IVOIREVOICE_ARTIFACTS_DIR")
    if not raw_root:
        raise ConfigError("IVOIREVOICE_ARTIFACTS_DIR doit être défini.")
    return Path(raw_root).expanduser().resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnostiquer l'environnement ML local.")
    parser.add_argument(
        "--output",
        default="reports/baselines/environment_report.json",
        help="Chemin de sortie relatif à IVOIREVOICE_ARTIFACTS_DIR.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point with aggregate-only output."""

    args = _parse_args()
    output = Path(args.output)
    if output.is_absolute() or ".." in output.parts:
        print("ERREUR: --output doit être un chemin relatif sûr.")
        return 1
    try:
        artifacts_root = _artifacts_root()
        report = collect_environment_report(artifacts_root)
        write_environment_report(report, artifacts_root / output)
    except IvoireVoiceError as exc:
        print(f"ERREUR: {exc}")
        return 1

    packages = report["packages"]
    torch_info = report["torch"]
    print(f"python_version={report['python_version']}")
    print(f"pytorch_version={packages['torch'] or 'not_installed'}")
    print(f"cuda_available={torch_info['cuda_available']}")
    print(f"gpu_name={torch_info['gpu_name'] or 'none'}")
    print(f"ram_available_bytes={report['ram_available_bytes']}")
    print(f"disk_available_bytes={report['disk_available_bytes']}")
    print(f"float16_supported={torch_info['float16_supported']}")
    print(f"bfloat16_supported={torch_info['bfloat16_supported']}")
    print(f"recommended_device={report['recommended_device']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
