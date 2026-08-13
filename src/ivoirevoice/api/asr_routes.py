"""Versioned, privacy-safe HTTP adapter for the existing ASR services."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from time import perf_counter
from typing import Annotated, cast
from uuid import uuid4

from fastapi import APIRouter, File, Form, Request, UploadFile

from ivoirevoice.api.schemas import (
    APIHealthResponse,
    LanguageCode,
    LanguagesResponse,
    PublicASRModel,
    PublicLanguage,
    PublicModelsResponse,
    PublicTranscriptionResponse,
)
from ivoirevoice.api.uploads import SecureUploadStager, UploadValidationError
from ivoirevoice.config import AppConfig
from ivoirevoice.exceptions import (
    APIError,
    BackendNotLoadedError,
    ConfigError,
    ModelRegistryError,
    UnsupportedLanguageError,
)
from ivoirevoice.services.transcription_service import (
    AUDIO_MIME_TYPES,
    ModelCatalog,
    TranscriptionService,
    build_model_registry,
    load_model_catalog,
)

LOGGER = logging.getLogger(__name__)

LANGUAGE_NAMES: dict[str, str] = {
    "fr": "Français",
    "en": "English",
    "dyu": "Dioula",
}
EXPECTED_MODEL_LANGUAGES: dict[str, frozenset[str]] = {
    "whisper_tiny_baseline": frozenset({"fr", "en", "dyu"}),
    "whisper_small_baseline": frozenset({"fr", "en", "dyu"}),
    "whisper_tiny_dioula_final": frozenset({"dyu"}),
}


@dataclass(frozen=True, slots=True)
class ASRAPIServices:
    """Injectable dependencies for discovery, upload staging and inference."""

    catalog: ModelCatalog
    transcription: TranscriptionService
    uploads: SecureUploadStager
    language_codes: tuple[str, ...]


def build_asr_api_services(config: AppConfig) -> ASRAPIServices:
    """Build metadata and lazy factories without loading model weights."""

    catalog = load_model_catalog()
    if catalog.max_audio_size_bytes != config.api.max_upload_size_bytes:
        raise ConfigError("Les limites d'upload API et catalogue doivent être identiques.")
    if catalog.max_audio_duration_seconds != 30:
        raise ConfigError("La durée audio maximale du catalogue doit être de 30 secondes.")
    if set(catalog.allowed_extensions) != set(AUDIO_MIME_TYPES):
        raise ConfigError("Le catalogue doit déclarer exactement les formats audio supportés.")
    expected_mimes = {
        mime
        for extension in catalog.allowed_extensions
        for mime in AUDIO_MIME_TYPES[extension]
    }
    if set(config.api.allowed_content_types) != expected_mimes:
        raise ConfigError("Les types MIME API ne correspondent pas aux formats audio supportés.")
    if set(config.project.supported_languages) != set(LANGUAGE_NAMES):
        raise ConfigError("Le registre de langues public doit déclarer fr, en et dyu.")
    if config.api.audio_retention != "delete_immediately":
        raise ConfigError("La rétention audio permanente est interdite pour ce MVP.")
    definitions = {item.key: item for item in catalog.enabled_models}
    if set(definitions) != set(EXPECTED_MODEL_LANGUAGES):
        raise ConfigError("Le catalogue ASR public doit contenir les trois modèles approuvés.")
    for definition in definitions.values():
        if not set(definition.languages).issubset(LANGUAGE_NAMES):
            raise ConfigError("Un modèle déclare une langue absente du registre public.")
        if set(definition.languages) != EXPECTED_MODEL_LANGUAGES[definition.key]:
            raise ConfigError("Les capacités linguistiques d'un modèle public sont invalides.")
        if definition.task != "transcribe":
            raise ConfigError("Tous les modèles ASR publics doivent utiliser la tâche transcribe.")
    for key in ("whisper_tiny_baseline", "whisper_small_baseline"):
        if definitions[key].status != "baseline" or definitions[key].checkpoint_name is not None:
            raise ConfigError("Une baseline ASR publique contient des métadonnées invalides.")
    final_definition = definitions["whisper_tiny_dioula_final"]
    if (
        final_definition.status != "final_adapted"
        or final_definition.checkpoint_name != "checkpoint-002052"
        or final_definition.configured_language != "dyu"
    ):
        raise ConfigError("La définition publique du modèle Dioula final est invalide.")

    registry = build_model_registry(catalog)
    return ASRAPIServices(
        catalog=catalog,
        transcription=TranscriptionService(catalog, registry),
        uploads=SecureUploadStager(max_size_bytes=catalog.max_audio_size_bytes),
        language_codes=config.project.supported_languages,
    )


def _request_id(request: Request) -> str:
    value = getattr(request.state, "request_id", None)
    return value if isinstance(value, str) and value else f"req_{uuid4().hex}"


def _log_result(
    *,
    request_id: str,
    model_id: str,
    language: str,
    status: str,
    elapsed_seconds: float,
    error_code: str | None = None,
) -> None:
    LOGGER.info(
        "asr_request request_id=%s model_id=%s language=%s status=%s "
        "processing_duration_seconds=%.6f error_code=%s",
        request_id,
        model_id,
        language,
        status,
        elapsed_seconds,
        error_code or "none",
    )


def _failure(
    *,
    request_id: str,
    model_id: str,
    language: str,
    started_at: float,
    status_code: int,
    code: str,
    message: str,
) -> APIError:
    _log_result(
        request_id=request_id,
        model_id=model_id,
        language=language,
        status="failed",
        elapsed_seconds=perf_counter() - started_at,
        error_code=code,
    )
    return APIError(
        status_code=status_code,
        code=code,
        message=message,
    )


def _public_languages(language_codes: tuple[str, ...]) -> list[PublicLanguage]:
    return [
        PublicLanguage(
            code=cast(LanguageCode, code),
            name=LANGUAGE_NAMES[code],
            asr="experimental",
            learning="coming_soon",
            translation_targets={
                target: "coming_soon" for target in language_codes if target != code
            },
        )
        for code in language_codes
    ]


def _public_models(catalog: ModelCatalog) -> list[PublicASRModel]:
    return [
        PublicASRModel(
            id=definition.key,
            display_name=definition.display_name,
            status="experimental",
            supported_languages=[cast(LanguageCode, code) for code in definition.languages],
        )
        for definition in catalog.enabled_models
    ]


def create_asr_router(services: ASRAPIServices) -> APIRouter:
    """Create versioned routes over injected services and fakes."""

    router = APIRouter()

    @router.get("/api/health", response_model=APIHealthResponse, tags=["system"])
    async def api_health() -> APIHealthResponse:
        return APIHealthResponse(status="ok")

    @router.get("/api/v1/languages", response_model=LanguagesResponse, tags=["discovery"])
    async def languages() -> LanguagesResponse:
        return LanguagesResponse(languages=_public_languages(services.language_codes))

    @router.get("/api/v1/models", response_model=PublicModelsResponse, tags=["discovery"])
    async def models() -> PublicModelsResponse:
        return PublicModelsResponse(models=_public_models(services.catalog))

    @router.post(
        "/api/v1/transcriptions",
        response_model=PublicTranscriptionResponse,
        tags=["transcription"],
    )
    async def transcribe(
        request: Request,
        audio: Annotated[UploadFile, File(description="Audio utilisateur éphémère")],
        language: Annotated[str, Form(description="Code canonique de la langue")],
        model: Annotated[str, Form(description="Identifiant public du modèle")],
    ) -> PublicTranscriptionResponse:
        request_id = _request_id(request)
        started_at = perf_counter()
        normalized_language = language.strip().lower()
        normalized_model = model.strip().lower()
        safe_language = (
            normalized_language if normalized_language in services.language_codes else "unknown"
        )
        known_models = {item.key: item for item in services.catalog.enabled_models}
        safe_model = normalized_model if normalized_model in known_models else "unknown"

        try:
            if normalized_language not in services.language_codes:
                raise _failure(
                    request_id=request_id,
                    model_id=safe_model,
                    language="unknown",
                    started_at=started_at,
                    status_code=422,
                    code="unknown_language",
                    message="Langue inconnue.",
                )
            definition = known_models.get(normalized_model)
            if definition is None:
                raise _failure(
                    request_id=request_id,
                    model_id="unknown",
                    language=safe_language,
                    started_at=started_at,
                    status_code=404,
                    code="unknown_model",
                    message="Modèle inconnu.",
                )
            if normalized_language not in definition.languages:
                raise _failure(
                    request_id=request_id,
                    model_id=definition.key,
                    language=safe_language,
                    started_at=started_at,
                    status_code=422,
                    code="incompatible_model_language",
                    message="Langue incompatible avec ce modèle.",
                )

            try:
                async with services.uploads.stage(audio) as staged:
                    executor = ThreadPoolExecutor(
                        max_workers=1,
                        thread_name_prefix="ivoirevoice-asr",
                    )
                    try:
                        try:
                            metadata = await asyncio.get_running_loop().run_in_executor(
                                executor,
                                services.transcription.validate_audio,
                                staged.path,
                            )
                        except ConfigError as exc:
                            raise _failure(
                                request_id=request_id,
                                model_id=definition.key,
                                language=safe_language,
                                started_at=started_at,
                                status_code=415,
                                code="unsupported_audio",
                                message="Format audio non supporté.",
                            ) from exc

                        try:
                            output = await asyncio.get_running_loop().run_in_executor(
                                executor,
                                partial(
                                    services.transcription.transcribe,
                                    model_key=definition.key,
                                    audio_path=staged.path,
                                    language=normalized_language,
                                    audio_metadata=metadata,
                                ),
                            )
                        except UnsupportedLanguageError as exc:
                            raise _failure(
                                request_id=request_id,
                                model_id=definition.key,
                                language=safe_language,
                                started_at=started_at,
                                status_code=422,
                                code="incompatible_model_language",
                                message="Langue incompatible avec ce modèle.",
                            ) from exc
                        except (BackendNotLoadedError, ConfigError, ModelRegistryError) as exc:
                            raise _failure(
                                request_id=request_id,
                                model_id=definition.key,
                                language=safe_language,
                                started_at=started_at,
                                status_code=503,
                                code="model_unavailable",
                                message="Modèle indisponible.",
                            ) from exc
                        except Exception as exc:
                            raise _failure(
                                request_id=request_id,
                                model_id=definition.key,
                                language=safe_language,
                                started_at=started_at,
                                status_code=500,
                                code="transcription_failed",
                                message="Transcription impossible.",
                            ) from exc
                    finally:
                        # The explicit bounded pool avoids using the event loop's
                        # global executor and is always drained before the
                        # temporary audio is removed.
                        executor.shutdown(wait=True, cancel_futures=True)
            except UploadValidationError as exc:
                raise _failure(
                    request_id=request_id,
                    model_id=definition.key,
                    language=safe_language,
                    started_at=started_at,
                    status_code=exc.status_code,
                    code=exc.code,
                    message=exc.message,
                ) from exc

            _log_result(
                request_id=request_id,
                model_id=definition.key,
                language=safe_language,
                status="completed",
                elapsed_seconds=perf_counter() - started_at,
            )
            return PublicTranscriptionResponse(
                id=f"tr_{uuid4().hex}",
                status="completed",
                language=cast(LanguageCode, normalized_language),
                model_id=definition.key,
                text=output.transcription,
                audio_duration_seconds=output.audio_duration_seconds,
                processing_time_seconds=output.processing_time_seconds,
                rtf=output.rtf,
            )
        finally:
            # Selection failures happen before the stager is entered; close the
            # multipart spool in that case as well. close() is idempotent.
            with suppress(Exception):
                await audio.close()

    return router
