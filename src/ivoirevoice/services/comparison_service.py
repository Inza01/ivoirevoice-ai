"""Fault-isolated, sequential comparison of registered ASR models."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from ivoirevoice.services.evaluation_service import EvaluationResult, EvaluationService
from ivoirevoice.services.transcription_service import (
    AudioMetadata,
    ModelDefinition,
    TranscriptionOutput,
    TranscriptionService,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModelComparisonResult:
    """Successful or isolated failed result for one selected model."""

    model_key: str
    display_name: str
    model_status: str
    model_id: str
    model_revision: str
    device: str
    hardware: str
    success: bool
    transcription: str
    processing_time_seconds: float | None
    audio_duration_seconds: float
    rtf: float | None
    checkpoint_name: str | None
    task: str
    configured_language: str | None
    training_audio_count: int | None
    validation_audio_count: int | None
    evaluation: EvaluationResult
    error: str | None


@dataclass(frozen=True, slots=True)
class ComparisonRun:
    """One UI comparison safe to export without a local audio path."""

    experiment_id: str
    generated_at_utc: str
    audio_id: str
    language: str
    reference: str | None
    audio_duration_seconds: float
    results: tuple[ModelComparisonResult, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "generated_at_utc": self.generated_at_utc,
            "audio_id": self.audio_id,
            "language": self.language,
            "reference": self.reference,
            "audio_duration_seconds": self.audio_duration_seconds,
            "results": [asdict(result) for result in self.results],
        }


def _failed_result(
    definition: ModelDefinition,
    *,
    duration: float,
    evaluation_service: EvaluationService,
    exception: Exception,
) -> ModelComparisonResult:
    LOGGER.error(
        "Échec isolé du modèle %s (%s).",
        definition.key,
        type(exception).__name__,
    )
    return ModelComparisonResult(
        model_key=definition.key,
        display_name=definition.display_name,
        model_status=definition.status,
        model_id=definition.model_id,
        model_revision=definition.revision,
        device="indisponible",
        hardware="indisponible",
        success=False,
        transcription="",
        processing_time_seconds=None,
        audio_duration_seconds=duration,
        rtf=None,
        checkpoint_name=definition.checkpoint_name,
        task=definition.task,
        configured_language=definition.configured_language,
        training_audio_count=definition.training_audio_count,
        validation_audio_count=definition.validation_audio_count,
        evaluation=evaluation_service.evaluate(None, ""),
        error=f"Échec isolé du modèle ({type(exception).__name__}).",
    )


def _successful_result(
    output: TranscriptionOutput,
    evaluation: EvaluationResult,
) -> ModelComparisonResult:
    return ModelComparisonResult(
        model_key=output.model_key,
        display_name=output.display_name,
        model_status=output.model_status,
        model_id=output.model_id,
        model_revision=output.model_revision,
        device=output.device,
        hardware=output.hardware,
        success=True,
        transcription=output.transcription,
        processing_time_seconds=output.processing_time_seconds,
        audio_duration_seconds=output.audio_duration_seconds,
        rtf=output.rtf,
        checkpoint_name=output.checkpoint_name,
        task=output.task,
        configured_language=output.configured_language,
        training_audio_count=output.training_audio_count,
        validation_audio_count=output.validation_audio_count,
        evaluation=evaluation,
        error=None,
    )


class ComparisonService:
    """Run selected models sequentially and preserve partial success."""

    def __init__(
        self,
        transcription_service: TranscriptionService,
        evaluation_service: EvaluationService,
    ) -> None:
        self.transcription_service = transcription_service
        self.evaluation_service = evaluation_service

    def compare(
        self,
        *,
        audio_path: str | Path,
        language: str,
        model_keys: tuple[str, ...],
        reference: str | None = None,
    ) -> ComparisonRun:
        if not model_keys:
            raise ValueError("Sélectionnez au moins un modèle.")
        metadata: AudioMetadata = self.transcription_service.validate_audio(audio_path)
        results: list[ModelComparisonResult] = []
        for model_key in model_keys:
            definition = self.transcription_service.catalog.definition(model_key)
            try:
                output = self.transcription_service.transcribe(
                    model_key=model_key,
                    audio_path=audio_path,
                    language=language,
                    audio_metadata=metadata,
                )
                evaluation = self.evaluation_service.evaluate(reference, output.transcription)
                results.append(_successful_result(output, evaluation))
            except Exception as exc:  # noqa: BLE001 - one model must not abort the others
                results.append(
                    _failed_result(
                        definition,
                        duration=metadata.duration_seconds,
                        evaluation_service=self.evaluation_service,
                        exception=exc,
                    )
                )
        normalized_reference = reference.strip() if reference and reference.strip() else None
        return ComparisonRun(
            experiment_id=f"ui-{uuid4().hex}",
            generated_at_utc=datetime.now(UTC).isoformat(),
            audio_id=metadata.audio_id,
            language=language,
            reference=normalized_reference,
            audio_duration_seconds=metadata.duration_seconds,
            results=tuple(results),
        )
