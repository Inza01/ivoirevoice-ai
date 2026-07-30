from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ivoirevoice.data.audio import sha256_file
from ivoirevoice.evaluation.baseline import (
    BaselineItem,
    BaselineSettings,
    completed_utterance_ids,
    normalize_evaluation_text,
    pending_items,
    require_full_confirmation,
    run_baseline,
    select_evaluation_items,
    shareable_summary,
)
from ivoirevoice.exceptions import ConfigError
from ivoirevoice.models.base import TranscriptionResult


def _items() -> tuple[BaselineItem, ...]:
    rows: list[BaselineItem] = []
    for speaker_index in range(3):
        for audio_index in range(60):
            rows.append(
                BaselineItem(
                    utterance_id=f"utt_{speaker_index}_{audio_index:02d}",
                    speaker_id=f"spk_{speaker_index}",
                    gender_folder="men" if speaker_index < 2 else "women",
                    audio_path=f"speaker_{speaker_index}/{audio_index:02d}.wav",
                    audio_sha256=f"{speaker_index * 100 + audio_index:064x}",
                    audio_duration_seconds=float(audio_index + 1),
                    reference_raw="A ka taa",
                )
            )
    return tuple(rows)


def test_smoke_and_pilot_use_only_deterministic_test_speaker_samples() -> None:
    items = _items()

    smoke = select_evaluation_items(items, level="smoke", seed=42)
    first_pilot = select_evaluation_items(items, level="pilot", seed=42)
    second_pilot = select_evaluation_items(items, level="pilot", seed=42)

    assert len(smoke) == 6
    assert len(first_pilot) == 150
    assert first_pilot == second_pilot
    assert {item.speaker_id for item in smoke} == {"spk_0", "spk_1", "spk_2"}
    assert all(
        sum(item.speaker_id == speaker for item in smoke) == 2
        for speaker in {"spk_0", "spk_1", "spk_2"}
    )
    pilot_durations = [
        item.audio_duration_seconds for item in first_pilot if item.speaker_id == "spk_0"
    ]
    assert min(pilot_durations) == 1.0
    assert max(pilot_durations) == 60.0


def test_full_requires_explicit_confirmation() -> None:
    with pytest.raises(ConfigError, match="CONFIRM_FULL=1"):
        require_full_confirmation("full", None)

    require_full_confirmation("full", "1")
    require_full_confirmation("smoke", None)


def test_normalization_is_consistent_and_removes_falling_marker() -> None:
    reference = normalize_evaluation_text(
        "  À KÀ ↘, taa! ",
        lowercase=True,
        remove_punctuation=True,
    )
    prediction = normalize_evaluation_text(
        "à kà taa",
        lowercase=True,
        remove_punctuation=True,
    )

    assert reference == prediction == "à kà taa"


def test_resume_skips_durably_recorded_items(tmp_path: Path) -> None:
    path = tmp_path / "predictions_private.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["utterance_id"])
        writer.writeheader()
        writer.writerow({"utterance_id": "utt_0_00"})

    completed = completed_utterance_ids(path)
    selected = select_evaluation_items(_items(), level="smoke", seed=42)
    remaining = pending_items(selected, completed)

    assert completed == {"utt_0_00"}
    assert len(remaining) == 5
    assert all(item.utterance_id != "utt_0_00" for item in remaining)


def test_shareable_report_contains_no_transcription_or_personal_path() -> None:
    metrics = {
        "evaluated_audio_count": 6,
        "failed_audio_count": 0,
        "wer_micro": 1.0,
        "cer_micro": 0.9,
        "wer_macro_speakers": 1.0,
        "cer_macro_speakers": 0.9,
        "mean_latency_seconds": 2.0,
        "latency_p50_seconds": 2.0,
        "latency_p95_seconds": 3.0,
        "rtf": 0.5,
    }

    report = shareable_summary(
        experiment_id="baseline-dy-whisper-tiny",
        level="smoke",
        model_id="openai/whisper-tiny",
        model_revision="a" * 40,
        metrics=metrics,
        manifest_hash="b" * 64,
    )

    assert "/home/" not in report
    assert "A ka taa" not in report
    assert "prediction_raw" not in report
    assert "publication" in report


def test_runner_resumes_without_recomputing_and_keeps_predictions_private(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = tmp_path / "artifacts"
    dataset = tmp_path / "dataset"
    manifest = artifacts / "manifests/v0.1.csv"
    manifest.parent.mkdir(parents=True)
    fieldnames = [
        "utterance_id",
        "speaker_id",
        "gender_folder",
        "audio_path",
        "audio_sha256",
        "duration_seconds",
        "target_text_mvp",
        "split",
        "usage_scope",
    ]
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for speaker_index in range(3):
            for audio_index in range(2):
                relative_audio = f"speaker_{speaker_index}/{audio_index}.wav"
                audio_path = dataset / relative_audio
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                audio_path.write_bytes(b"fake")
                writer.writerow(
                    {
                        "utterance_id": f"utt_{speaker_index}_{audio_index}",
                        "speaker_id": f"spk_{speaker_index}",
                        "gender_folder": ("men" if speaker_index < 2 else "women"),
                        "audio_path": relative_audio,
                        "audio_sha256": f"{speaker_index * 2 + audio_index:064x}",
                        "duration_seconds": "1.0",
                        "target_text_mvp": "A ka taa",
                        "split": "test",
                        "usage_scope": "local_research_only",
                    }
                )
    metadata = artifacts / "manifests/metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "manifest_sha256": sha256_file(manifest),
                "dataset_version": "0.1.0-local",
                "publication_allowed": False,
                "model_derivative_publication_allowed": False,
                "usage_scope": "local_research_only",
            }
        ),
        encoding="utf-8",
    )
    model_config = tmp_path / "whisper.yaml"
    model_config.write_text(
        """
model:
  name: whisper-tiny
  family: whisper
  model_id: openai/whisper-tiny
  model_revision: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  device: cpu
  torch_dtype: float32
  batch_size: 1
  chunk_length_seconds: 30
  stride_length_seconds: 5
  task: transcribe
  language: null
  cache_environment_variable: IVOIREVOICE_MODEL_CACHE_DIR
  local_files_only: true
  max_new_tokens: 128
  expected_sampling_rate_hz: 16000
  supported_languages: [dyu]
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("IVOIREVOICE_MODEL_CACHE_DIR", str(tmp_path / "cache"))
    settings = BaselineSettings(
        experiment_id="baseline-dy-whisper-tiny",
        language="dyu",
        split="test",
        seed=42,
        model_config_path=model_config,
        expected_model_id="openai/whisper-tiny",
        manifest_path=manifest,
        dataset_metadata_path=metadata,
        dataset_root=dataset,
        artifacts_root=artifacts,
        output_relative_directory=Path("baselines"),
        report_relative_directory=Path("reports/baselines"),
        expected_test_audio_count=6,
        expected_test_speaker_count=3,
        smoke_per_speaker=2,
        pilot_per_speaker=2,
        timeout_seconds=10,
        lowercase=True,
        remove_punctuation=True,
        publication_allowed=False,
    )
    transcribe_calls = 0

    class FakeBackend:
        model_name = "openai/whisper-tiny@" + "a" * 40

        def load(self) -> None:
            return None

        def transcribe(self, _audio: Path, language: str) -> TranscriptionResult:
            nonlocal transcribe_calls
            transcribe_calls += 1
            return TranscriptionResult(
                text="prediction privée",
                language=language,
                processing_time_seconds=0.5,
                model_name=self.model_name,
                duration_seconds=1.0,
            )

        def unload(self) -> None:
            return None

    factory = lambda _config: FakeBackend()  # noqa: E731
    first = run_baseline(
        settings,
        level="smoke",
        backend_factory=factory,  # type: ignore[arg-type]
    )
    second = run_baseline(
        settings,
        level="smoke",
        backend_factory=factory,  # type: ignore[arg-type]
    )
    output = artifacts / "baselines/baseline-dy-whisper-tiny-smoke"
    summary = (artifacts / "reports/baselines/baseline-dy-whisper-tiny-smoke_summary.md").read_text(
        encoding="utf-8"
    )
    private_predictions = (output / "predictions_private.csv").read_text(encoding="utf-8")
    run_metadata = json.loads((output / "run_metadata.json").read_text())

    assert transcribe_calls == 6
    assert first["evaluated_audio_count"] == second["evaluated_audio_count"] == 6
    assert "prediction privée" in private_predictions
    assert "prediction privée" not in summary
    assert "/home/" not in summary
    assert run_metadata["publication_allowed"] is False
