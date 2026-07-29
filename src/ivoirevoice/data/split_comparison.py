"""Compare deterministic speaker-level split strategies for the curated candidate."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ivoirevoice.data.settings import DioulaDataSettings, load_dioula_settings
from ivoirevoice.exceptions import ConfigError, IvoireVoiceError

SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class CandidateAudio:
    """Minimal candidate fields required for split comparison."""

    utterance_id: str
    speaker_id: str
    gender_folder: str
    audio_path: str
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class SpeakerProfile:
    """Aggregated statistics for one pseudonymized speaker."""

    speaker_id: str
    gender_folder: str
    audio_count: int
    duration_seconds: float


def load_candidate(path: Path) -> tuple[CandidateAudio, ...]:
    """Load the curated candidate while rejecting already assigned splits."""

    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {
                "utterance_id",
                "speaker_id",
                "gender_folder",
                "audio_path",
                "duration_seconds",
                "split",
            }
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ConfigError("Colonnes absentes du candidat : " + ", ".join(sorted(missing)))
            rows: list[CandidateAudio] = []
            for row in reader:
                if row["split"]:
                    raise ConfigError("Le manifeste candidat doit conserver 'split' vide.")
                rows.append(
                    CandidateAudio(
                        utterance_id=row["utterance_id"],
                        speaker_id=row["speaker_id"],
                        gender_folder=row["gender_folder"],
                        audio_path=row["audio_path"],
                        duration_seconds=float(row["duration_seconds"]),
                    )
                )
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        raise ConfigError(f"Impossible de lire le manifeste candidat : {exc}") from exc
    if len({row.audio_path for row in rows}) != len(rows):
        raise ConfigError("Le manifeste candidat contient encore des chemins audio dupliqués.")
    return tuple(rows)


def _speaker_profiles(rows: tuple[CandidateAudio, ...]) -> tuple[SpeakerProfile, ...]:
    grouped: dict[str, list[CandidateAudio]] = defaultdict(list)
    for row in rows:
        grouped[row.speaker_id].append(row)
    return tuple(
        SpeakerProfile(
            speaker_id=speaker_id,
            gender_folder=group[0].gender_folder,
            audio_count=len(group),
            duration_seconds=sum(row.duration_seconds for row in group),
        )
        for speaker_id, group in sorted(grouped.items())
    )


def _assignment_metrics(
    assignments: dict[str, tuple[str, ...]],
    profiles: dict[str, SpeakerProfile],
    *,
    strategy: str,
    target_ratios: dict[str, float],
    seed: int,
) -> dict[str, Any]:
    all_speakers = [
        speaker for split_speakers in assignments.values() for speaker in split_speakers
    ]
    leakage_free = len(all_speakers) == len(set(all_speakers)) == len(profiles)
    total_duration = sum(profile.duration_seconds for profile in profiles.values())
    duration_by_split = {
        split: sum(profiles[speaker].duration_seconds for speaker in speakers)
        for split, speakers in assignments.items()
    }
    audio_by_split = {
        split: sum(profiles[speaker].audio_count for speaker in speakers)
        for split, speakers in assignments.items()
    }
    duration_percentages = {
        split: duration / total_duration if total_duration else 0.0
        for split, duration in duration_by_split.items()
    }
    duration_deviation = sum(
        abs(duration_percentages[split] - target_ratios[split]) for split in SPLIT_NAMES
    )
    gender_counts = {
        split: {
            gender: sum(profiles[speaker].gender_folder == gender for speaker in speakers)
            for gender in ("men", "women", "unknown")
        }
        for split, speakers in assignments.items()
    }
    gender_missing_penalty = sum(
        gender_counts[split]["men"] == 0 or gender_counts[split]["women"] == 0
        for split in SPLIT_NAMES
    )
    small_evaluation_penalty = max(0, 3 - len(assignments["validation"])) + max(
        0,
        3 - len(assignments["test"]),
    )
    total_audio_count = sum(profile.audio_count for profile in profiles.values())
    record_deviation = sum(
        abs((audio_by_split[split] / total_audio_count) - target_ratios[split])
        for split in SPLIT_NAMES
    )
    balance_score = (
        duration_deviation * 100
        + record_deviation * 20
        + gender_missing_penalty * 100
        + small_evaluation_penalty * 200
        + (0 if leakage_free else 10_000)
    )
    return {
        "strategy": strategy,
        "seed": seed,
        "target_duration_ratios": target_ratios,
        "speaker_ids": {split: list(assignments[split]) for split in SPLIT_NAMES},
        "speaker_counts": {split: len(assignments[split]) for split in SPLIT_NAMES},
        "gender_speaker_counts": gender_counts,
        "line_counts": audio_by_split,
        "unique_audio_counts": audio_by_split,
        "duration_seconds": duration_by_split,
        "mean_duration_seconds": {
            split: (
                duration_by_split[split] / audio_by_split[split] if audio_by_split[split] else 0.0
            )
            for split in SPLIT_NAMES
        },
        "duration_percentages": duration_percentages,
        "duration_target_absolute_deviation": duration_deviation,
        "record_target_absolute_deviation": record_deviation,
        "leakage_free": leakage_free,
        "balance_score": balance_score,
    }


def _search(
    profiles: tuple[SpeakerProfile, ...],
    *,
    strategy: str,
    seed: int,
    target_ratios: dict[str, float],
    validation_counts: tuple[int, ...],
    test_counts: tuple[int, ...],
    iterations: int,
) -> dict[str, Any]:
    profile_map = {profile.speaker_id: profile for profile in profiles}
    speaker_ids = sorted(profile_map)
    if not speaker_ids:
        raise ConfigError("Aucun locuteur disponible pour proposer un split.")
    rng = random.Random(f"{seed}:{strategy}")
    best: dict[str, Any] | None = None
    best_key: tuple[Any, ...] | None = None
    for _ in range(iterations):
        validation_count = rng.choice(validation_counts)
        test_count = rng.choice(test_counts)
        if validation_count + test_count >= len(speaker_ids):
            continue
        shuffled = speaker_ids.copy()
        rng.shuffle(shuffled)
        validation_end = validation_count
        test_end = validation_end + test_count
        assignments = {
            "validation": tuple(sorted(shuffled[:validation_end])),
            "test": tuple(sorted(shuffled[validation_end:test_end])),
            "train": tuple(sorted(shuffled[test_end:])),
        }
        metrics = _assignment_metrics(
            assignments,
            profile_map,
            strategy=strategy,
            target_ratios=target_ratios,
            seed=seed,
        )
        candidate_key = (
            metrics["balance_score"],
            tuple(metrics["speaker_ids"]["validation"]),
            tuple(metrics["speaker_ids"]["test"]),
        )
        if best_key is None or candidate_key < best_key:
            best = metrics
            best_key = candidate_key
    if best is None:
        raise ConfigError(f"Impossible de construire la stratégie de split {strategy}.")
    return best


def compare_split_strategies(
    rows: tuple[CandidateAudio, ...],
    *,
    seed: int,
    iterations: int = 30_000,
) -> dict[str, Any]:
    """Compare fixed-count and duration-oriented deterministic searches."""

    profiles = _speaker_profiles(rows)
    if len(profiles) < 7:
        raise ConfigError("Au moins sept locuteurs sont requis pour comparer les splits.")
    strategies = [
        _search(
            profiles,
            strategy="A_17_2_2",
            seed=seed,
            target_ratios={"train": 0.8, "validation": 0.1, "test": 0.1},
            validation_counts=(2,),
            test_counts=(2,),
            iterations=iterations,
        ),
        _search(
            profiles,
            strategy="B_15_3_3",
            seed=seed,
            target_ratios={"train": 0.72, "validation": 0.14, "test": 0.14},
            validation_counts=(3,),
            test_counts=(3,),
            iterations=iterations,
        ),
        _search(
            profiles,
            strategy="C_duration_75_12_5_12_5",
            seed=seed,
            target_ratios={"train": 0.75, "validation": 0.125, "test": 0.125},
            validation_counts=(3, 4, 5),
            test_counts=(3, 4, 5),
            iterations=iterations,
        ),
    ]
    eligible_recommendations = [
        strategy
        for strategy in strategies
        if strategy["leakage_free"]
        and strategy["speaker_counts"]["validation"] >= 3
        and strategy["speaker_counts"]["test"] >= 3
        and strategy["gender_speaker_counts"]["validation"]["men"] > 0
        and strategy["gender_speaker_counts"]["validation"]["women"] > 0
        and strategy["gender_speaker_counts"]["test"]["men"] > 0
        and strategy["gender_speaker_counts"]["test"]["women"] > 0
    ]
    recommendation_pool = eligible_recommendations or strategies
    recommended = min(
        recommendation_pool,
        key=lambda strategy: (strategy["balance_score"], strategy["strategy"]),
    )
    return {
        "status": "comparison_only_human_validation_required",
        "speaker_count": len(profiles),
        "candidate_audio_count": len(rows),
        "strategies": strategies,
        "recommended_strategy": recommended["strategy"],
        "recommendation_reason": (
            "zéro fuite, au moins trois locuteurs en validation et test, "
            "représentation men/women et meilleur score d'équilibre"
        ),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    temporary_path.replace(path)


def _comparison_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# Comparaison des splits dioula",
        "",
        "Propositions uniquement : aucun split n'est écrit dans le manifeste.",
        "",
        "| Stratégie | Locuteurs T/V/Test | Durée T/V/Test | Score | Fuite |",
        "|---|---|---|---:|---|",
    ]
    for strategy in comparison["strategies"]:
        speakers = strategy["speaker_counts"]
        percentages = strategy["duration_percentages"]
        lines.append(
            f"| {strategy['strategy']} | "
            f"{speakers['train']}/{speakers['validation']}/{speakers['test']} | "
            f"{percentages['train']:.3%}/{percentages['validation']:.3%}/"
            f"{percentages['test']:.3%} | "
            f"{strategy['balance_score']:.4f} | "
            f"{'non' if strategy['leakage_free'] else 'oui'} |"
        )
    lines.extend(
        [
            "",
            f"Recommandation : `{comparison['recommended_strategy']}`.",
            "",
            "Une validation humaine reste obligatoire avant le gel du split.",
            "",
        ]
    )
    return "\n".join(lines)


def write_split_comparison(
    comparison: dict[str, Any],
    settings: DioulaDataSettings,
) -> None:
    """Write JSON and Markdown comparison reports externally."""

    directory = settings.curation_report_directory
    _atomic_json(directory / "split_comparison.json", comparison)
    markdown_path = directory / "split_comparison.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = markdown_path.with_suffix(".md.tmp")
    temporary_path.write_text(_comparison_markdown(comparison), encoding="utf-8")
    temporary_path.replace(markdown_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comparer les splits dioula.")
    parser.add_argument("--config", required=True, help="Configuration YAML des données.")
    return parser.parse_args()


def main() -> int:
    """CLI entry point with aggregate-only output."""

    args = _parse_args()
    try:
        settings = load_dioula_settings(args.config)
        rows = load_candidate(settings.candidate_manifest_path)
        comparison = compare_split_strategies(rows, seed=settings.split.seed)
        write_split_comparison(comparison, settings)
    except IvoireVoiceError as exc:
        print(f"ERREUR: {exc}")
        return 1

    for strategy in comparison["strategies"]:
        counts = strategy["speaker_counts"]
        percentages = strategy["duration_percentages"]
        print(
            f"{strategy['strategy']}="
            f"{counts['train']}/{counts['validation']}/{counts['test']} speakers,"
            f"{percentages['train']:.6f}/{percentages['validation']:.6f}/"
            f"{percentages['test']:.6f} duration,"
            f"score={strategy['balance_score']:.6f}"
        )
    print(f"recommended_split={comparison['recommended_strategy']}")
    print("split_written=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
