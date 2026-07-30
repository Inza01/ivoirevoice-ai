from __future__ import annotations

import pytest

from ivoirevoice.ui.app import build_demo_services, create_interface


def test_interface_builds_without_browser_network_or_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("IVOIREVOICE_ARTIFACTS_DIR", raising=False)
    monkeypatch.delenv("IVOIREVOICE_DIOULA_DATA_DIR", raising=False)
    services = build_demo_services()

    interface = create_interface(services)

    assert interface.enable_queue is True
    assert len(interface.blocks) >= 50
    services.exports.cleanup()
