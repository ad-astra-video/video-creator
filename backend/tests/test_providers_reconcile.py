"""Regression tests for Livepeer client reconciliation on discovery URL updates.

The LivepeerClient is created once at startup and snapshots the discovery URL
at __init__, so a Refresh after the user updates the discovery URL (or
configures one after startup) must rebuild the client. These tests pin that
behavior so the "0 runners / stale URL" bug doesn't come back.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402

from _routes.providers import reconcile_livepeer_client  # noqa: E402


class RecordingClient:
    """Stand-in for LivepeerClient that records how it was constructed."""

    instances: list["RecordingClient"] = []

    def __init__(self, discovery_url: str, results_dir, api_key: str = "") -> None:
        self.discovery_url = discovery_url
        self.results_dir = results_dir
        self.api_key = api_key
        RecordingClient.instances.append(self)


@pytest.fixture(autouse=True)
def _reset_recorder():
    RecordingClient.instances = []
    yield
    RecordingClient.instances = []


def test_no_url_returns_none(test_state, monkeypatch):
    monkeypatch.setattr("services.livepeer_client.LivepeerClient", RecordingClient)
    assert test_state.state.app_settings.livepeer_discovery_url == ""
    assert reconcile_livepeer_client(test_state) is None
    assert RecordingClient.instances == []


def test_creates_client_when_none_exists(test_state, monkeypatch):
    monkeypatch.setattr("services.livepeer_client.LivepeerClient", RecordingClient)
    test_state.state.app_settings.livepeer_discovery_url = "https://orch:8935/discovery"

    client = reconcile_livepeer_client(test_state)

    assert client is not None
    assert len(RecordingClient.instances) == 1
    assert RecordingClient.instances[0].discovery_url == "https://orch:8935/discovery"
    # Stored on state so generation handlers pick it up.
    assert test_state.state._livepeer_client is client


def test_rebuilds_client_after_url_change(test_state, monkeypatch):
    monkeypatch.setattr("services.livepeer_client.LivepeerClient", RecordingClient)
    test_state.state.app_settings.livepeer_discovery_url = "https://orch:8935/discovery"
    first = reconcile_livepeer_client(test_state)

    # User updates the discovery URL to a new orchestrator.
    test_state.state.app_settings.livepeer_discovery_url = "https://new-orch:8935/discovery"
    second = reconcile_livepeer_client(test_state)

    assert first is not second
    assert second.discovery_url == "https://new-orch:8935/discovery"
    assert test_state.state._livepeer_client is second
    # A fresh client was built (and reused, not leaked on every call).
    assert len(RecordingClient.instances) == 2


def test_reuses_client_when_url_unchanged(test_state, monkeypatch):
    monkeypatch.setattr("services.livepeer_client.LivepeerClient", RecordingClient)
    test_state.state.app_settings.livepeer_discovery_url = "https://orch:8935/discovery"

    first = reconcile_livepeer_client(test_state)
    second = reconcile_livepeer_client(test_state)

    assert first is second
    assert len(RecordingClient.instances) == 1
