"""Synthetic end-to-end smoke for the three local demo models."""

from __future__ import annotations

import json
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from ivoirevoice.services.comparison_service import ComparisonRun
from ivoirevoice.ui.app import DemoServices, build_demo_services

MODEL_KEYS = (
    "whisper_tiny_baseline",
    "whisper_small_baseline",
    "whisper_tiny_dioula_final",
)


@dataclass(frozen=True, slots=True)
class DemoSmokeReport:
    """Technical, aggregate-only output from a synthetic demo run."""

    model_statuses: tuple[tuple[str, bool], ...]
    comparison_seconds: float
    export_suffixes: tuple[str, ...]
    wer_cer_available: bool
    private_path_exposed: bool

    @property
    def passed(self) -> bool:
        return (
            all(status for _, status in self.model_statuses)
            and self.export_suffixes == (".json", ".csv", ".txt")
            and self.wer_cer_available
            and not self.private_path_exposed
        )


def _synthetic_wav(path: Path) -> None:
    """Create one second of silence; it contains no speech or corpus content."""

    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\x00\x00" * 16_000)


def _contains_private_path(run: ComparisonRun) -> bool:
    serialized = json.dumps(run.to_dict(), ensure_ascii=False)
    return "/home/" in serialized or "\\Users\\" in serialized


def run_demo_smoke(services: DemoServices | None = None) -> DemoSmokeReport:
    """Exercise UI services without private audio or scientific claims."""

    selected_services = services or build_demo_services()
    try:
        with tempfile.TemporaryDirectory(prefix="ivoirevoice-demo-smoke-") as directory:
            audio_path = Path(directory) / "synthetic_silence.wav"
            _synthetic_wav(audio_path)
            started = perf_counter()
            run = selected_services.comparison.compare(
                audio_path=audio_path,
                language="dyu",
                model_keys=MODEL_KEYS,
                reference="synthetic technical reference",
            )
            comparison_seconds = perf_counter() - started
            exports = selected_services.exports.export_all(run)
            return DemoSmokeReport(
                model_statuses=tuple((result.model_key, result.success) for result in run.results),
                comparison_seconds=comparison_seconds,
                export_suffixes=tuple(Path(path).suffix for path in exports),
                wer_cer_available=all(
                    result.evaluation.wer is not None and result.evaluation.cer is not None
                    for result in run.results
                ),
                private_path_exposed=_contains_private_path(run),
            )
    finally:
        selected_services.exports.cleanup()


def main() -> int:
    report = run_demo_smoke()
    for model_key, passed in report.model_statuses:
        print(f"{model_key}: {'PASSED' if passed else 'FAILED'}")
    print(f"COMPARISON: {report.comparison_seconds:.3f} s")
    print(f"WER/CER EXAMPLE: {'PASSED' if report.wer_cer_available else 'FAILED'}")
    print(f"EXPORTS: {','.join(report.export_suffixes)}")
    print(f"PRIVATE PATH EXPOSED: {'YES' if report.private_path_exposed else 'NO'}")
    print("DEMO SMOKE PASSED" if report.passed else "DEMO SMOKE FAILED")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
