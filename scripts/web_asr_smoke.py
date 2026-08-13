"""Guarded real-model smoke for the local versioned ASR API.

This script never reads project datasets. French and English inputs must be
new synthetic speech or externally licensed audio explicitly confirmed by the
operator. Dioula remains a discovery-contract check when no equally safe input
is provided.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import httpx

CONFIRMATION = "SAFE_EXTERNAL_OR_SYNTHETIC_AUDIO_NOT_FROM_PROJECT_DATA"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
SUPPORTED_AUDIO_TYPES: Mapping[str, str] = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
}
PROTECTED_ROOT_VARIABLES = (
    "IVOIREVOICE_DIOULA_DATA_DIR",
    "IVOIREVOICE_DIOULA_INTERIM_DIR",
    "IVOIREVOICE_ARTIFACTS_DIR",
    "IVOIREVOICE_MODEL_CACHE_DIR",
    "IVOIREVOICE_CHECKPOINT_DIR",
    "IVOIREVOICE_DIOULA_PILOT_MODEL_PATH",
    "IVOIREVOICE_DIOULA_FINAL_MODEL_PATH",
)
PROTECTED_PATH_MARKERS = frozenset(
    {"artifacts", "checkpoint", "checkpoints", "finalholdout", "voicesdata"}
)


class SmokeConfigurationError(ValueError):
    """Raised before any network call when smoke inputs are unsafe."""


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_base_url(raw_value: str) -> str:
    parsed = urlparse(raw_value.strip())
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise SmokeConfigurationError("Le smoke ASR doit cibler une origine HTTP loopback.")
    return raw_value.strip().rstrip("/")


def _protected_roots() -> tuple[Path, ...]:
    roots = [REPOSITORY_ROOT.resolve()]
    for variable in PROTECTED_ROOT_VARIABLES:
        value = os.getenv(variable, "").strip()
        if value:
            roots.append(Path(value).expanduser().resolve())
    return tuple(roots)


def _validate_audio_path(raw_value: str, *, label: str) -> tuple[Path, str]:
    path = Path(raw_value).expanduser().resolve()
    if not path.is_file():
        raise SmokeConfigurationError(f"{label}: le fichier externe est absent.")
    if path.stat().st_size <= 0:
        raise SmokeConfigurationError(f"{label}: le fichier externe est vide.")
    if any(_is_relative_to(path, root) for root in _protected_roots()):
        raise SmokeConfigurationError(f"{label}: un actif protégé du projet est interdit.")
    normalized_parts = {
        "".join(character for character in part.lower() if character.isalnum())
        for part in path.parts
    }
    if any(
        part.startswith(marker)
        for part in normalized_parts
        for marker in PROTECTED_PATH_MARKERS
    ):
        raise SmokeConfigurationError(f"{label}: le chemin ressemble à un actif protégé.")
    content_type = SUPPORTED_AUDIO_TYPES.get(path.suffix.lower())
    if content_type is None:
        raise SmokeConfigurationError(f"{label}: format externe non pris en charge.")
    return path, content_type


def _as_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise RuntimeError(f"Réponse {label} invalide.")
    return cast(Mapping[str, Any], value)


def _validate_discovery(
    languages_payload: object,
    models_payload: object,
) -> Mapping[str, frozenset[str]]:
    languages = _as_mapping(languages_payload, label="languages").get("languages")
    if not isinstance(languages, list):
        raise RuntimeError("Réponse languages invalide.")
    codes = {
        entry.get("code")
        for entry in languages
        if isinstance(entry, dict) and isinstance(entry.get("code"), str)
    }
    if codes != {"fr", "en", "dyu"}:
        raise RuntimeError("Le registre runtime ne contient pas exactement fr, en et dyu.")

    models = _as_mapping(models_payload, label="models").get("models")
    if not isinstance(models, list):
        raise RuntimeError("Réponse models invalide.")
    supported_by_model: dict[str, frozenset[str]] = {}
    for entry in models:
        if not isinstance(entry, dict):
            raise RuntimeError("Entrée modèle invalide.")
        model_id = entry.get("id")
        supported = entry.get("supported_languages")
        if not isinstance(model_id, str) or not isinstance(supported, list) or not all(
            isinstance(item, str) for item in supported
        ):
            raise RuntimeError("Entrée modèle invalide.")
        supported_by_model[model_id] = frozenset(cast(list[str], supported))
    return supported_by_model


def _validate_transcription_payload(
    payload: object,
    *,
    language: str,
    model_id: str,
) -> None:
    result = _as_mapping(payload, label="transcription")
    text = result.get("text")
    duration = result.get("audio_duration_seconds")
    processing = result.get("processing_time_seconds")
    rtf = result.get("rtf")
    if (
        result.get("status") != "completed"
        or result.get("language") != language
        or result.get("model_id") != model_id
        or not isinstance(text, str)
        or not text.strip()
        or not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or not math.isfinite(duration)
        or duration <= 0
        or not isinstance(processing, (int, float))
        or isinstance(processing, bool)
        or not math.isfinite(processing)
        or processing < 0
        or not isinstance(rtf, (int, float))
        or isinstance(rtf, bool)
        or not math.isfinite(rtf)
        or rtf < 0
    ):
        raise RuntimeError("Réponse de transcription invalide ou vide.")


def _transcribe(
    client: httpx.Client,
    *,
    path: Path,
    content_type: str,
    language: str,
    model_id: str,
) -> None:
    with path.open("rb") as stream:
        response = client.post(
            "/api/v1/transcriptions",
            data={"language": language, "model": model_id},
            files={"audio": ("external-smoke" + path.suffix.lower(), stream, content_type)},
        )
    response.raise_for_status()
    _validate_transcription_payload(response.json(), language=language, model_id=model_id)


def main() -> int:
    """Run guarded local FR/EN smoke and optional DYU smoke."""

    if os.getenv("IVOIREVOICE_WEB_ASR_SMOKE_CONFIRMATION") != CONFIRMATION:
        raise SmokeConfigurationError(
            "Définir IVOIREVOICE_WEB_ASR_SMOKE_CONFIRMATION avec la confirmation documentée."
        )

    base_url = _validate_base_url(
        os.getenv("IVOIREVOICE_WEB_ASR_SMOKE_BASE_URL", DEFAULT_BASE_URL)
    )
    required_inputs = {
        "fr": _validate_audio_path(
            os.getenv("IVOIREVOICE_WEB_ASR_SMOKE_FR_AUDIO_PATH", ""),
            label="FR",
        ),
        "en": _validate_audio_path(
            os.getenv("IVOIREVOICE_WEB_ASR_SMOKE_EN_AUDIO_PATH", ""),
            label="EN",
        ),
    }
    model_by_language = {
        "fr": "whisper_tiny_baseline",
        "en": "whisper_tiny_baseline",
        "dyu": "whisper_tiny_dioula_final",
    }

    with httpx.Client(base_url=base_url, timeout=120.0) as client:
        health = client.get("/api/health")
        health.raise_for_status()
        if _as_mapping(health.json(), label="health").get("status") != "ok":
            raise RuntimeError("Le service ASR local n'est pas sain.")

        languages = client.get("/api/v1/languages")
        models = client.get("/api/v1/models")
        languages.raise_for_status()
        models.raise_for_status()
        supported_by_model = _validate_discovery(languages.json(), models.json())

        for language, model_id in model_by_language.items():
            if language not in supported_by_model.get(model_id, frozenset()):
                raise RuntimeError(f"Contrat modèle/langue absent pour {language}.")

        for language, (path, content_type) in required_inputs.items():
            _transcribe(
                client,
                path=path,
                content_type=content_type,
                language=language,
                model_id=model_by_language[language],
            )
            print(f"ASR {language.upper()}: PASSED (texte non affiché)")

        raw_dyu_path = os.getenv("IVOIREVOICE_WEB_ASR_SMOKE_DYU_AUDIO_PATH", "").strip()
        if raw_dyu_path:
            dyu_path, dyu_content_type = _validate_audio_path(raw_dyu_path, label="DYU")
            _transcribe(
                client,
                path=dyu_path,
                content_type=dyu_content_type,
                language="dyu",
                model_id=model_by_language["dyu"],
            )
            print("ASR DYU: PASSED (texte non affiché)")
        else:
            print("ASR DYU: CONTRACT ONLY (aucun audio externe sûr fourni)")

    print("WEB ASR REAL-MODEL SMOKE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
