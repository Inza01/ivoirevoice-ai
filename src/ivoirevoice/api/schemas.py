"""Public HTTP response schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Service liveness response."""

    status: Literal["ok"]
    service: str
    version: str


class ModelInfo(BaseModel):
    """A backend exposed by the registry."""

    name: str
    supported_languages: list[str]
    implementation: str


class ModelsResponse(BaseModel):
    """Available backend collection."""

    models: list[ModelInfo]


class TranscriptionResponse(BaseModel):
    """Normalized transcription payload."""

    text: str
    language: str
    confidence: float | None = None
    duration_seconds: float | None = None
    processing_time_seconds: float = Field(ge=0)
    model_name: str


class ErrorDetail(BaseModel):
    """Machine-readable API error."""

    code: str
    message: str
    details: Any | None = None


class ErrorResponse(BaseModel):
    """Stable error envelope."""

    error: ErrorDetail
