from __future__ import annotations

import json
from pathlib import Path

from ivoirevoice.evaluation.compatibility import model_compatibility_report
from ivoirevoice.evaluation.environment import collect_environment_report


def test_environment_report_contains_no_personal_path(tmp_path: Path) -> None:
    report = collect_environment_report(tmp_path / "artifacts")
    serialized = json.dumps(report)

    assert report["python_version"]
    assert report["disk_available_bytes"] > 0
    assert "/home/" not in serialized
    assert "environment_values_absent" in report["privacy_checks"]


def test_compatibility_excludes_non_asr_models() -> None:
    report = model_compatibility_report()
    by_id = {model["model_id"]: model for model in report["models"]}

    assert by_id["openai/whisper-tiny"]["compatible"] is True
    assert by_id["openai/whisper-small"]["compatible"] is True
    assert by_id["linekeita/whisper-dioula-mt"]["architecture"] == "MarianMTModel"
    assert by_id["linekeita/whisper-dioula-mt"]["accepts_audio"] is False
    assert by_id["RobotsMali/lau-soloni-114m-mse-k1"]["compatible"] is False
    assert report["decision"]["fine_tuning_allowed"] is False
