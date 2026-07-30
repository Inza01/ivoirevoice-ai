"""Dependency-light ASR error, latency and real-time metrics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class EditCounts:
    """Levenshtein operations for one reference/hypothesis pair."""

    substitutions: int
    deletions: int
    insertions: int
    reference_units: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions


@dataclass(frozen=True, slots=True)
class ScoredItem:
    """Minimal private evaluation row consumed by aggregate metrics."""

    speaker_id: str
    reference_normalized: str
    prediction_normalized: str
    audio_duration_seconds: float
    processing_time_seconds: float
    error_type: str = ""


def edit_counts(reference: tuple[str, ...], hypothesis: tuple[str, ...]) -> EditCounts:
    """Compute deterministic minimum Levenshtein operation counts."""

    previous: list[tuple[int, int, int, int]] = [
        (index, 0, 0, index) for index in range(len(hypothesis) + 1)
    ]
    for reference_index, reference_unit in enumerate(reference, start=1):
        current: list[tuple[int, int, int, int]] = [(reference_index, 0, reference_index, 0)]
        for hypothesis_index, hypothesis_unit in enumerate(hypothesis, start=1):
            if reference_unit == hypothesis_unit:
                current.append(previous[hypothesis_index - 1])
                continue
            diagonal = previous[hypothesis_index - 1]
            deletion = previous[hypothesis_index]
            insertion = current[hypothesis_index - 1]
            candidates = (
                (
                    diagonal[0] + 1,
                    diagonal[1] + 1,
                    diagonal[2],
                    diagonal[3],
                ),
                (
                    deletion[0] + 1,
                    deletion[1],
                    deletion[2] + 1,
                    deletion[3],
                ),
                (
                    insertion[0] + 1,
                    insertion[1],
                    insertion[2],
                    insertion[3] + 1,
                ),
            )
            current.append(min(candidates))
        previous = current
    _, substitutions, deletions, insertions = previous[-1]
    return EditCounts(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_units=len(reference),
    )


def _sum_counts(counts: list[EditCounts]) -> EditCounts:
    return EditCounts(
        substitutions=sum(item.substitutions for item in counts),
        deletions=sum(item.deletions for item in counts),
        insertions=sum(item.insertions for item in counts),
        reference_units=sum(item.reference_units for item in counts),
    )


def _error_rate(counts: EditCounts) -> float:
    if counts.reference_units == 0:
        return 0.0 if counts.errors == 0 else 1.0
    return counts.errors / counts.reference_units


def percentile(values: list[float], percentage: float) -> float:
    """Return a linearly interpolated percentile for a non-empty list."""

    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentage
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction


def _speaker_metrics(items: list[ScoredItem]) -> dict[str, Any]:
    successful = [item for item in items if not item.error_type]
    word_counts = _sum_counts(
        [
            edit_counts(
                tuple(item.reference_normalized.split()),
                tuple(item.prediction_normalized.split()),
            )
            for item in successful
        ]
    )
    character_counts = _sum_counts(
        [
            edit_counts(
                tuple(item.reference_normalized),
                tuple(item.prediction_normalized),
            )
            for item in successful
        ]
    )
    processing_time = sum(item.processing_time_seconds for item in successful)
    audio_duration = sum(item.audio_duration_seconds for item in successful)
    return {
        "audio_count": len(items),
        "successful_audio_count": len(successful),
        "failed_audio_count": len(items) - len(successful),
        "wer": _error_rate(word_counts),
        "cer": _error_rate(character_counts),
        "word_substitutions": word_counts.substitutions,
        "word_deletions": word_counts.deletions,
        "word_insertions": word_counts.insertions,
        "reference_word_count": word_counts.reference_units,
        "character_substitutions": character_counts.substitutions,
        "character_deletions": character_counts.deletions,
        "character_insertions": character_counts.insertions,
        "reference_character_count": character_counts.reference_units,
        "audio_duration_seconds": audio_duration,
        "processing_time_seconds": processing_time,
        "rtf": processing_time / audio_duration if audio_duration else 0.0,
    }


def compute_evaluation_metrics(items: tuple[ScoredItem, ...]) -> dict[str, Any]:
    """Compute micro/macro accuracy, latency, failures and RTF."""

    grouped: dict[str, list[ScoredItem]] = defaultdict(list)
    for item in items:
        grouped[item.speaker_id].append(item)
    speaker_metrics = {
        speaker: _speaker_metrics(speaker_items)
        for speaker, speaker_items in sorted(grouped.items())
    }
    successful = [item for item in items if not item.error_type]
    global_metrics = _speaker_metrics(list(items))
    latencies = [item.processing_time_seconds for item in successful]
    macro_wer = (
        sum(metrics["wer"] for metrics in speaker_metrics.values()) / len(speaker_metrics)
        if speaker_metrics
        else 0.0
    )
    macro_cer = (
        sum(metrics["cer"] for metrics in speaker_metrics.values()) / len(speaker_metrics)
        if speaker_metrics
        else 0.0
    )
    return {
        "evaluated_audio_count": len(items),
        "successful_audio_count": len(successful),
        "failed_audio_count": len(items) - len(successful),
        "failure_rate": ((len(items) - len(successful)) / len(items) if items else 0.0),
        "wer_micro": global_metrics["wer"],
        "cer_micro": global_metrics["cer"],
        "wer_macro_speakers": macro_wer,
        "cer_macro_speakers": macro_cer,
        "word_substitutions": global_metrics["word_substitutions"],
        "word_deletions": global_metrics["word_deletions"],
        "word_insertions": global_metrics["word_insertions"],
        "character_substitutions": global_metrics["character_substitutions"],
        "character_deletions": global_metrics["character_deletions"],
        "character_insertions": global_metrics["character_insertions"],
        "audio_duration_seconds": global_metrics["audio_duration_seconds"],
        "processing_time_seconds": global_metrics["processing_time_seconds"],
        "mean_latency_seconds": (sum(latencies) / len(latencies) if latencies else 0.0),
        "latency_p50_seconds": percentile(latencies, 0.50),
        "latency_p95_seconds": percentile(latencies, 0.95),
        "rtf": global_metrics["rtf"],
        "speaker_metrics": speaker_metrics,
    }
