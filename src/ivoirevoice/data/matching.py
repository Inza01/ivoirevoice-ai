"""Speaker-local matching between parsed records and existing WAV files."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from ivoirevoice.data.records import AudioMatch, AudioMatchMethod, ParsedClip

CONVERTED_SEQUENCE_PATTERN = re.compile(r"-(?P<sequence>\d{6})-\d{2}\.wav$", re.IGNORECASE)


def _result(
    candidates: Iterable[Path],
    method: AudioMatchMethod,
) -> AudioMatch:
    unique_candidates = tuple(sorted(set(candidates), key=lambda path: path.as_posix()))
    if len(unique_candidates) == 1:
        return AudioMatch(
            status="matched",
            method=method,
            path=unique_candidates[0],
            candidate_count=1,
        )
    if len(unique_candidates) > 1:
        return AudioMatch(
            status="ambiguous",
            method=method,
            path=None,
            candidate_count=len(unique_candidates),
        )
    return AudioMatch(status="missing", method="none", path=None, candidate_count=0)


def _identifier_candidates(identifier: str, wav_files: tuple[Path, ...]) -> list[Path]:
    if not identifier:
        return []
    return [path for path in wav_files if identifier in path.stem]


def _converted_sequence_candidates(
    sequence: int | None,
    wav_files: tuple[Path, ...],
) -> list[Path]:
    if sequence is None:
        return []
    candidates: list[Path] = []
    for path in wav_files:
        if path.parent.name.lower() != "wav":
            continue
        match = CONVERTED_SEQUENCE_PATTERN.search(path.name)
        if match and int(match.group("sequence")) == sequence:
            candidates.append(path)
    return candidates


def match_audio(record: ParsedClip, wav_files: tuple[Path, ...]) -> AudioMatch:
    """Match using ordered strategies without leaving the inferred speaker directory."""

    if not record.valid:
        return AudioMatch(status="invalid", method="none", path=None, candidate_count=0)

    sequence_result = _result(
        _converted_sequence_candidates(record.sequence, wav_files),
        "sequence",
    )
    if sequence_result.status != "missing":
        return sequence_result

    if record.audio_filename:
        exact_result = _result(
            (path for path in wav_files if path.name == record.audio_filename),
            "exact_filename",
        )
        if exact_result.status != "missing":
            return exact_result

    sentence_result = _result(
        _identifier_candidates(record.sentence_id, wav_files),
        "sentence_id",
    )
    if sentence_result.status != "missing":
        return sentence_result

    clip_result = _result(
        _identifier_candidates(record.clip_id, wav_files),
        "clip_id",
    )
    if clip_result.status != "missing":
        return clip_result

    return AudioMatch(status="missing", method="none", path=None, candidate_count=0)
