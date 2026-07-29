from __future__ import annotations

from ivoirevoice.data.split_comparison import CandidateAudio, compare_split_strategies


def _candidate_rows() -> tuple[CandidateAudio, ...]:
    rows: list[CandidateAudio] = []
    for speaker_index in range(21):
        gender = "men" if speaker_index < 11 else "women"
        for audio_index in range(2):
            rows.append(
                CandidateAudio(
                    utterance_id=f"utt_{speaker_index}_{audio_index}",
                    speaker_id=f"spk_{speaker_index:02d}",
                    gender_folder=gender,
                    audio_path=f"{gender}/speaker_{speaker_index}/audio_{audio_index}.wav",
                    duration_seconds=float(1 + speaker_index + audio_index),
                )
            )
    return tuple(rows)


def test_compares_three_reproducible_leakage_free_strategies() -> None:
    rows = _candidate_rows()

    first = compare_split_strategies(rows, seed=42, iterations=2_000)
    second = compare_split_strategies(rows, seed=42, iterations=2_000)

    assert first == second
    assert len(first["strategies"]) == 3
    by_name = {strategy["strategy"]: strategy for strategy in first["strategies"]}
    assert by_name["A_17_2_2"]["speaker_counts"] == {
        "train": 17,
        "validation": 2,
        "test": 2,
    }
    assert by_name["B_15_3_3"]["speaker_counts"] == {
        "train": 15,
        "validation": 3,
        "test": 3,
    }
    assert all(strategy["leakage_free"] for strategy in first["strategies"])
    assert first["recommended_strategy"] != "A_17_2_2"
