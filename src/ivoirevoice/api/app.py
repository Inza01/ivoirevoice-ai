"""FastAPI entry point for legacy development and versioned ASR routes."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import FastAPI, File, Form, Request, Response, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ivoirevoice.api.asr_routes import (
    ASRAPIServices,
    build_asr_api_services,
    create_asr_router,
)
from ivoirevoice.api.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    TranscriptionResponse,
)
from ivoirevoice.config import AppConfig, load_config
from ivoirevoice.exceptions import (
    APIError,
    BackendNotLoadedError,
    ModelRegistryError,
    UnsupportedLanguageError,
)
from ivoirevoice.models.registry import ModelRegistry, create_default_registry

LOGGER = logging.getLogger(__name__)


def _request_error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    request_id: str,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id,
            }
        },
        headers={"X-Request-ID": request_id},
    )


def create_app(
    config: AppConfig | None = None,
    registry: ModelRegistry | None = None,
    asr_services: ASRAPIServices | None = None,
) -> FastAPI:
    """Build an application with injectable configuration and backend registry."""

    selected_config = config or load_config()
    selected_registry = registry or create_default_registry()
    selected_asr_services = asr_services or build_asr_api_services(selected_config)
    application = FastAPI(
        title=selected_config.project.name,
        version=selected_config.project.version,
        description="API locale IvoireVoice avec compatibilité legacy et ASR versionné.",
    )

    @application.middleware("http")
    async def request_context(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = f"req_{uuid4().hex}"
        request.state.request_id = request_id
        if request.method == "POST" and request.url.path == "/api/v1/transcriptions":
            raw_length = request.headers.get("Content-Length")
            if raw_length is None:
                return _request_error_response(
                    status_code=411,
                    code="invalid_request",
                    message="La taille de la requête est obligatoire.",
                    request_id=request_id,
                )
            if not raw_length.isdecimal() or int(raw_length) <= 0:
                return _request_error_response(
                    status_code=400,
                    code="invalid_request",
                    message="La taille de la requête est invalide.",
                    request_id=request_id,
                )
            multipart_limit = selected_config.api.max_upload_size_bytes + 256 * 1024
            if int(raw_length) > multipart_limit:
                return _request_error_response(
                    status_code=413,
                    code="payload_too_large",
                    message="Audio trop volumineux.",
                    request_id=request_id,
                )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @application.exception_handler(APIError)
    async def handle_api_error(request: Request, error: APIError) -> JSONResponse:
        payload = {
            "error": {
                "code": error.code,
                "message": error.message,
                "details": None,
                "request_id": getattr(request.state, "request_id", None),
            }
        }
        return JSONResponse(status_code=error.status_code, content=jsonable_encoder(payload))

    @application.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request,
        _error: RequestValidationError,
    ) -> JSONResponse:
        error_code = (
            "invalid_request"
            if request.url.path.startswith("/api/v1/")
            else "validation_error"
        )
        payload = {
            "error": {
                "code": error_code,
                "message": "La requête ne respecte pas le contrat de l'API.",
                "details": None,
                "request_id": getattr(request.state, "request_id", None),
            }
        }
        return JSONResponse(status_code=422, content=jsonable_encoder(payload))

    @application.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, _error: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", f"req_{uuid4().hex}")
        LOGGER.error(
            "unhandled_api_error request_id=%s status=failed error_code=service_unavailable",
            request_id,
        )
        return _request_error_response(
            status_code=500,
            code="service_unavailable",
            message="Serveur momentanément indisponible.",
            request_id=request_id,
        )

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            service=selected_config.project.name,
            version=selected_config.project.version,
        )

    @application.get("/models", response_model=ModelsResponse, tags=["models"])
    async def models() -> ModelsResponse:
        model_items: list[ModelInfo] = []
        for name in selected_registry.available_models:
            backend = selected_registry.create(name)
            model_items.append(
                ModelInfo(
                    name=name,
                    supported_languages=list(backend.supported_languages),
                    implementation="legacy",
                )
            )
        return ModelsResponse(models=model_items)

    @application.post(
        "/transcribe",
        response_model=TranscriptionResponse,
        responses={
            400: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
        },
        tags=["transcription"],
    )
    async def transcribe(
        file: Annotated[UploadFile, File(description="Fichier audio à transcrire")],
        language: Annotated[Literal["fr", "dyu"], Form(description="Langue de l'audio")],
        model: Annotated[str | None, Form(description="Backend enregistré")] = None,
    ) -> TranscriptionResponse:
        content_type = (file.content_type or "").lower()
        if content_type not in selected_config.api.allowed_content_types:
            raise APIError(
                status_code=415,
                code="unsupported_audio_type",
                message="Type audio non pris en charge.",
            )

        try:
            audio = await file.read(selected_config.api.max_upload_size_bytes + 1)
        finally:
            await file.close()

        if not audio:
            raise APIError(
                status_code=400,
                code="empty_audio",
                message="Le fichier audio est vide.",
            )
        if len(audio) > selected_config.api.max_upload_size_bytes:
            raise APIError(
                status_code=413,
                code="audio_too_large",
                message="Le fichier dépasse la taille maximale configurée.",
                details={"max_upload_size_mb": selected_config.api.max_upload_size_mb},
            )

        selected_model = model or selected_config.project.default_model
        try:
            backend = selected_registry.create(selected_model)
        except ModelRegistryError as exc:
            raise APIError(
                status_code=404,
                code="unknown_model",
                message="Modèle inconnu.",
            ) from exc

        try:
            backend.load()
            result = backend.transcribe(audio, language)
        except (BackendNotLoadedError, UnsupportedLanguageError, ValueError) as exc:
            raise APIError(
                status_code=422,
                code="transcription_error",
                message="Transcription impossible.",
            ) from exc
        finally:
            backend.unload()

        return TranscriptionResponse(
            text=result.text,
            language=result.language,
            confidence=result.confidence,
            duration_seconds=result.duration_seconds,
            processing_time_seconds=result.processing_time_seconds,
            model_name=selected_model,
        )

    application.include_router(create_asr_router(selected_asr_services))
    return application


app = create_app()
