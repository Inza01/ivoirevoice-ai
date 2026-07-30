from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from ivoirevoice.training.audit import AuditedDataset, ManifestRow
from ivoirevoice.training.manual_review import ReviewSession, create_review_interface
from ivoirevoice.training.settings import SmokeSettings


def _session(tmp_path: Path) -> ReviewSession:
    row = ManifestRow(
        utterance_id="utt_anonymous",
        speaker_id="spk_anonymous",
        gender_folder="women",
        language="dyu",
        text_raw="Á hakili kà go ↘",
        text_no_tones="A hakili ka go ↘",
        target_text="A hakili ka go",
        audio_path="women/anonymous/audio.wav",
        duration_seconds=2.5,
        sample_rate_hz=16_000,
        channels=1,
        audio_sha256="a" * 64,
        split="train",
        usage_scope="local_research_only",
    )
    settings = cast(
        SmokeSettings,
        SimpleNamespace(
            dataset_root=tmp_path / "data",
            reports_root=tmp_path / "reports",
            minimum_correct_samples=1,
        ),
    )
    settings.reports_root.mkdir()
    dataset = AuditedDataset(
        rows=(row,),
        manifest_sha256="b" * 64,
        dataset_version="0.1.0-local",
    )
    annotations = {
        "schema_version": 1,
        "manifest_sha256": dataset.manifest_sha256,
        "selection_sha256": "c" * 64,
        "statuses": {
            row.utterance_id: {
                "status": "à vérifier",
                "anomaly": "",
                "reviewed_at_utc": None,
            }
        },
    }
    return ReviewSession(settings, dataset, (row,), annotations)


def test_review_session_persists_human_status_and_private_report(tmp_path: Path) -> None:
    session = _session(tmp_path)

    confirmation = session.save(1, "correct", "écoute confirmée")

    saved = json.loads(
        (tmp_path / "reports" / "manual_validation_annotations.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved["statuses"]["utt_anonymous"]["status"] == "correct"
    assert "1/1" in confirmation
    assert (tmp_path / "reports" / "manual_validation_report.md").is_file()


def test_review_session_rejects_unknown_status(tmp_path: Path) -> None:
    session = _session(tmp_path)

    with pytest.raises(ValueError, match="inconnu"):
        session.save(1, "inventé", "")


def test_review_interface_builds_without_starting_server(tmp_path: Path) -> None:
    interface = create_review_interface(_session(tmp_path))

    assert interface is not None
