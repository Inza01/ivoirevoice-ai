"""Read-only discovery of the local Dioula corpus structure."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from ivoirevoice.data.records import SpeakerSource

KNOWN_GENDER_FOLDERS = {"men", "women"}


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _speaker_id(relative_directory: str) -> str:
    digest = sha256(relative_directory.encode("utf-8")).hexdigest()[:12]
    return f"spk_{digest}"


@dataclass(frozen=True, slots=True)
class CorpusInventory:
    """Deterministic inventory containing only dataset-relative paths."""

    root: Path
    files_by_kind: dict[str, tuple[Path, ...]]
    speakers: tuple[SpeakerSource, ...]
    gender_speaker_counts: dict[str, int]
    unexpected_structures: tuple[str, ...]

    def to_shareable_dict(self) -> dict[str, Any]:
        """Serialize without exposing the absolute dataset root."""

        return {
            "dataset_root": ".",
            "file_counts": {kind: len(paths) for kind, paths in sorted(self.files_by_kind.items())},
            "files": {
                kind: [_relative(path, self.root) for path in paths]
                for kind, paths in sorted(self.files_by_kind.items())
            },
            "speaker_count_estimate": len(self.speakers),
            "gender_speaker_counts": dict(sorted(self.gender_speaker_counts.items())),
            "speaker_directories": [
                {
                    "speaker_id": speaker.speaker_id,
                    "gender_folder": speaker.gender_folder,
                    "relative_directory": speaker.relative_directory,
                    "source_json": speaker.source_json,
                    "wav_count": len(speaker.wav_files),
                }
                for speaker in self.speakers
            ],
            "unexpected_structures": list(self.unexpected_structures),
        }


def discover_corpus(dataset_root: Path) -> CorpusInventory:
    """Recursively inventory files and infer speakers from clips.json locations."""

    root = dataset_root.resolve()
    all_files = tuple(
        sorted(
            (path for path in root.rglob("*") if path.is_file()),
            key=lambda path: _relative(path, root),
        )
    )
    files: dict[str, list[Path]] = {
        "clips_json": [],
        "wav": [],
        "mp4": [],
        "text": [],
        "text_no_tones": [],
        "other": [],
    }
    for path in all_files:
        lower_name = path.name.lower()
        lower_suffix = path.suffix.lower()
        if lower_name == "clips.json":
            kind = "clips_json"
        elif lower_suffix == ".wav":
            kind = "wav"
        elif lower_suffix == ".mp4":
            kind = "mp4"
        elif lower_name == "text-no-tones":
            kind = "text_no_tones"
        elif lower_name == "text":
            kind = "text"
        else:
            kind = "other"
        files[kind].append(path)

    unexpected: list[str] = []
    top_level_directories = sorted(path for path in root.iterdir() if path.is_dir())
    for directory in top_level_directories:
        if directory.name.lower() not in KNOWN_GENDER_FOLDERS:
            unexpected.append(f"unexpected_top_level_directory:{directory.name}")

    speakers: list[SpeakerSource] = []
    gender_counts = {"men": 0, "women": 0, "unknown": 0}
    for clips_json in files["clips_json"]:
        relative_json = clips_json.relative_to(root)
        parts = relative_json.parts
        expected_structure = (
            len(parts) == 3
            and parts[0].lower() in KNOWN_GENDER_FOLDERS
            and parts[-1].lower() == "clips.json"
        )
        gender = parts[0].lower() if expected_structure else "unknown"
        if not expected_structure:
            unexpected.append(f"unexpected_clips_json_location:{relative_json.as_posix()}")

        speaker_directory = clips_json.parent
        relative_directory = _relative(speaker_directory, root)
        wav_files = tuple(
            sorted(
                (
                    path
                    for path in speaker_directory.rglob("*")
                    if path.is_file() and path.suffix.lower() == ".wav"
                ),
                key=lambda path: _relative(path, root),
            )
        )
        speakers.append(
            SpeakerSource(
                speaker_id=_speaker_id(relative_directory),
                gender_folder=gender,
                directory=speaker_directory,
                relative_directory=relative_directory,
                clips_json=clips_json,
                source_json=relative_json.as_posix(),
                wav_files=wav_files,
            )
        )
        gender_counts[gender] += 1

    frozen_files = {kind: tuple(paths) for kind, paths in files.items()}
    return CorpusInventory(
        root=root,
        files_by_kind=frozen_files,
        speakers=tuple(sorted(speakers, key=lambda speaker: speaker.source_json)),
        gender_speaker_counts=gender_counts,
        unexpected_structures=tuple(sorted(unexpected)),
    )
