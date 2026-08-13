"""Public HTTP response schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, FiniteFloat

CapabilityStatus = Literal["available", "experimental", "coming_soon"]
LanguageCode = Literal["fr", "en", "dyu"]


class HealthResponse(BaseModel):
    """Service liveness response."""

    status: Literal["ok"]
    service: str
    version: str


class APIHealthResponse(BaseModel):
    """Minimal liveness contract used by the web platform."""

    status: Literal["ok"]


class ModelInfo(BaseModel):
    """A backend exposed by the registry."""

    name: str
    supported_languages: list[str]
    implementation: str


class ModelsResponse(BaseModel):
    """Available backend collection."""

    models: list[ModelInfo]


class PublicLanguage(BaseModel):
    """Public language identity and capability states."""

    code: LanguageCode
    name: str
    asr: CapabilityStatus
    learning: CapabilityStatus
    translation_targets: dict[str, CapabilityStatus]


class LanguagesResponse(BaseModel):
    """Runtime language registry exposed to the web client."""

    languages: list[PublicLanguage]


class PublicASRModel(BaseModel):
    """Privacy-safe model capability without a path or runtime detail."""

    id: str
    display_name: str
    status: CapabilityStatus
    supported_languages: list[LanguageCode]


class PublicModelsResponse(BaseModel):
    """Configured ASR models safe for browser discovery."""

    models: list[PublicASRModel]


class TranscriptionResponse(BaseModel):
    """Normalized transcription payload."""

    text: str
    language: str
    confidence: float | None = None
    duration_seconds: float | None = None
    processing_time_seconds: float = Field(ge=0)
    model_name: str


class PublicTranscriptionResponse(BaseModel):
    """Synchronous MVP resource shape, compatible with future job states."""

    id: str
    status: Literal["completed"]
    language: LanguageCode
    model_id: str
    text: str
    audio_duration_seconds: FiniteFloat = Field(gt=0)
    processing_time_seconds: FiniteFloat = Field(ge=0)
    rtf: FiniteFloat = Field(ge=0)


class ErrorDetail(BaseModel):
    """Machine-readable API error."""

    code: str
    message: str
    details: Any | None = None
    request_id: str | None = None


class ErrorResponse(BaseModel):
    """Stable error envelope."""

    error: ErrorDetail
