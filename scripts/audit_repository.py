"""Fail closed when a Git submission candidate contains unsafe local files."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

MAX_FILE_BYTES = 100 * 1024 * 1024
FORBIDDEN_SUFFIXES = {
    ".aac",
    ".ckpt",
    ".flac",
    ".key",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ogg",
    ".p12",
    ".pem",
    ".pt",
    ".pth",
    ".safetensors",
    ".wav",
}
FORBIDDEN_PARTS = {
    ".credentials",
    ".venv",
    "artifacts",
    "checkpoints",
    "corpus",
    "datasets",
    "huggingface_cache",
    "secrets",
    "torch_cache",
    "transformers_cache",
    "voices_data",
}
PRIVATE_REPORT_NAMES = {
    "manual_validation_annotations.json",
    "manual_validation_report.md",
    "pilot_validation_predictions.csv",
    "smoke_overfit_predictions.csv",
}
TOKEN_PATTERN = re.compile(
    r"(?:github_pat_[A-Za-z0-9_]{20,}|ghp_[A-Za-z0-9]{20,}|"
    r"hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,})"
)
PRIVATE_KEY_PATTERN = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
)


def _candidate_paths(root: Path) -> tuple[Path, ...]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return tuple(
        root / value.decode()
        for value in result.stdout.split(b"\0")
        if value
    )


def audit_repository(root: Path) -> tuple[int, int, str]:
    """Return candidate count and largest file, or raise with safe filenames."""

    issues: list[str] = []
    paths = _candidate_paths(root)
    largest_size = 0
    largest_name = ""
    actual_home = str(Path.home().resolve())
    for path in paths:
        relative = path.relative_to(root)
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > largest_size:
            largest_size = size
            largest_name = relative.as_posix()
        if size >= MAX_FILE_BYTES:
            issues.append(f"{relative}: fichier supérieur ou égal à 100 Mio")
        if relative.name in PRIVATE_REPORT_NAMES or "predictions_private" in relative.name:
            issues.append(f"{relative}: rapport privé")
        if relative.name == ".env" or (
            relative.name.startswith(".env.") and relative.name != ".env.example"
        ):
            issues.append(f"{relative}: fichier d'environnement privé")
        if relative.suffix.lower() in FORBIDDEN_SUFFIXES:
            issues.append(f"{relative}: extension binaire ou privée interdite")
        if FORBIDDEN_PARTS.intersection(relative.parts):
            issues.append(f"{relative}: répertoire local interdit")
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if actual_home and actual_home in content:
            issues.append(f"{relative}: chemin personnel de la machine")
        if TOKEN_PATTERN.search(content):
            issues.append(f"{relative}: jeton potentiel")
        if PRIVATE_KEY_PATTERN.search(content):
            issues.append(f"{relative}: clé privée potentielle")
    if issues:
        raise RuntimeError(
            "Audit de publication refusé :\n- " + "\n- ".join(sorted(set(issues)))
        )
    return len(paths), largest_size, largest_name


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    try:
        count, largest_size, largest_name = audit_repository(root)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(str(exc))
        return 1
    print(f"Fichiers candidats : {count}")
    print(f"Plus gros fichier candidat : {largest_size} octets ({largest_name})")
    print("Audit de publication : réussi.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
