from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import ivoirevoice

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_repository_harness_contract_is_valid() -> None:
    assert ivoirevoice.__name__ == "ivoirevoice"
    result = subprocess.run(
        [sys.executable, "scripts/check_harness.py"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Harness engineering : valide." in result.stdout
