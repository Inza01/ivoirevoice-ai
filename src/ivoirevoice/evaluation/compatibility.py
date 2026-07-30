"""Pinned model-compatibility facts for the local Dioula ASR baseline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from ivoirevoice.exceptions import ConfigError, IvoireVoiceError


def model_compatibility_report() -> dict[str, Any]:
    """Return facts verified from pinned public model cards and configurations."""

    return {
        "status": "verified_for_phase_4a",
        "verification_date": "2026-07-29",
        "models": [
            {
                "model_id": "openai/whisper-tiny",
                "revision": "be0ba7c2f24f0127b27863a23a08002af4c2c279",
                "architecture": "WhisperForConditionalGeneration",
                "task": "automatic-speech-recognition",
                "accepts_audio": True,
                "license": "apache-2.0",
                "phase_4a_role": "smoke_and_pilot_first",
                "compatible": True,
                "language_policy": "automatic_detection_no_forced_dyu_token",
                "source": "https://huggingface.co/openai/whisper-tiny",
            },
            {
                "model_id": "openai/whisper-small",
                "revision": "973afd24965f72e36ca33b3055d56a652f456b4d",
                "architecture": "WhisperForConditionalGeneration",
                "task": "automatic-speech-recognition",
                "accepts_audio": True,
                "license": "apache-2.0",
                "weights_format": "safetensors",
                "weights_sha256": (
                    "1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b"
                ),
                "phase_4a_role": "after_tiny_results_and_explicit_decision",
                "compatible": True,
                "language_policy": "automatic_detection_no_forced_dyu_token",
                "source": "https://huggingface.co/openai/whisper-small",
            },
            {
                "model_id": "linekeita/whisper-dioula-mt",
                "revision": "b9535ae44288056960f8c4e739d5305322a56ac7",
                "architecture": "MarianMTModel",
                "task": "text2text-generation",
                "accepts_audio": False,
                "license": "unknown",
                "phase_4a_role": "excluded",
                "compatible": False,
                "reason": (
                    "Pinned config declares model_type=marian and the card does "
                    "not document a license or an audio processor."
                ),
                "source": "https://huggingface.co/linekeita/whisper-dioula-mt",
            },
            {
                "model_id": "RobotsMali/lau-soloni-114m-mse-k1",
                "revision": None,
                "architecture": "HybridRNNTCTCLAUModel",
                "task": "speech-translation_bambara_to_french",
                "accepts_audio": True,
                "license": "cc-by-4.0",
                "phase_4a_role": "excluded",
                "compatible": False,
                "reason": (
                    "The model translates Bambara speech to French text and "
                    "requires a custom NeMo class."
                ),
                "source": "https://huggingface.co/RobotsMali/lau-soloni-114m-mse-k1",
            },
        ],
        "decision": {
            "first_model": "openai/whisper-tiny",
            "small_requires_prior_tiny_results": True,
            "fine_tuning_allowed": False,
            "external_audio_upload_allowed": False,
        },
    }


def write_compatibility_report(report: dict[str, Any], path: Path) -> None:
    """Write the aggregate compatibility report outside Git."""

    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if "/home/" in serialized or "\\Users\\" in serialized:
        raise ConfigError("Le rapport de compatibilité contient un chemin personnel.")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        temporary_path.write_text(serialized, encoding="utf-8")
        temporary_path.replace(path)
    except OSError as exc:
        raise ConfigError(f"Impossible d'écrire le rapport de compatibilité : {exc}") from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Documenter la compatibilité des modèles.")
    parser.add_argument(
        "--output",
        default="reports/baselines/model_compatibility_report.json",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""

    args = _parse_args()
    raw_root = os.getenv("IVOIREVOICE_ARTIFACTS_DIR")
    output = Path(args.output)
    try:
        if not raw_root:
            raise ConfigError("IVOIREVOICE_ARTIFACTS_DIR doit être défini.")
        if output.is_absolute() or ".." in output.parts:
            raise ConfigError("--output doit être un chemin relatif sûr.")
        report = model_compatibility_report()
        write_compatibility_report(
            report,
            Path(raw_root).expanduser().resolve() / output,
        )
    except IvoireVoiceError as exc:
        print(f"ERREUR: {exc}")
        return 1
    compatible = [model["model_id"] for model in report["models"] if model["compatible"]]
    excluded = [model["model_id"] for model in report["models"] if not model["compatible"]]
    print(f"compatible_models={','.join(compatible)}")
    print(f"excluded_models={','.join(excluded)}")
    print("first_model=openai/whisper-tiny")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
