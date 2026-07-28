"""FastAPI entry point using the Phase 2 dummy backend."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

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


def create_app(
    config: AppConfig | None = None,
    registry: ModelRegistry | None = None,
) -> FastAPI:
    """Build an application with injectable configuration and backend registry."""

    selected_config = config or load_config()
    selected_registry = registry or create_default_registry()
    application = FastAPI(
        title=selected_config.project.name,
        version=selected_config.project.version,
        description="API de développement IvoireVoice utilisant un backend fictif.",
    )

    @application.exception_handler(APIError)
    async def handle_api_error(_request: Request, error: APIError) -> JSONResponse:
        payload = {
            "error": {
                "code": error.code,
                "message": error.message,
                "details": error.details,
            }
        }
        return JSONResponse(status_code=error.status_code, content=jsonable_encoder(payload))

    @application.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        payload = {
            "error": {
                "code": "validation_error",
                "message": "La requête ne respecte pas le contrat de l'API.",
                "details": error.errors(),
            }
        }
        return JSONResponse(status_code=422, content=jsonable_encoder(payload))

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
                    name=backend.model_name,
                    supported_languages=list(backend.supported_languages),
                    implementation=type(backend).__name__,
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
                message=f"Type audio non pris en charge : {content_type or 'non renseigné'}.",
                details={"allowed_content_types": selected_config.api.allowed_content_types},
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
                message=str(exc),
            ) from exc

        backend.load()
        try:
            result = backend.transcribe(audio, language)
        except (BackendNotLoadedError, UnsupportedLanguageError, ValueError) as exc:
            raise APIError(
                status_code=422,
                code="transcription_error",
                message=str(exc),
            ) from exc
        finally:
            backend.unload()

        return TranscriptionResponse(
            text=result.text,
            language=result.language,
            confidence=result.confidence,
            duration_seconds=result.duration_seconds,
            processing_time_seconds=result.processing_time_seconds,
            model_name=result.model_name,
        )

    return application


app = create_app()
