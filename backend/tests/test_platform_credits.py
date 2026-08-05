"""Tests for Module B: platform credits (balance/checkout), recovery, and the
remote-generation balance gate."""

from __future__ import annotations

import pytest

from api_types import GenerateVideoRequest
from _routes._errors import HTTPError
from tests.fakes.fake_platform_client import FakePlatformClient
from tests.fakes.services import FakeServices


def _provision(test_state) -> None:
    """Configure the platform base URL and provision a key via the fake."""
    test_state.state.app_settings.platform_base_url = "https://platform.example"
    test_state.platform.ensure_ready()
    assert test_state.state.app_settings.platform_api_key


def test_balance_returns_config_snapshot(client, test_state):
    _provision(test_state)
    r = client.get("/api/platform/balance")
    assert r.status_code == 200
    data = r.json()
    assert data["configured"] is True
    assert data["hasAccess"] is True
    assert data["balanceUsdMicros"] == 5_000_000
    assert data["remainingUsdMicros"] == 5_000_000
    assert "platformApiKey" not in data


def test_balance_unconfigured_returns_zeros(client, test_state):
    r = client.get("/api/platform/balance")
    assert r.status_code == 200
    data = r.json()
    assert data["configured"] is False
    assert data["hasAccess"] is False
    assert data["balanceUsdMicros"] == 0


def test_checkout_returns_hosted_url(client, test_state):
    _provision(test_state)
    r = client.post("/api/platform/checkout", json={"tier": 1000})
    assert r.status_code == 200
    data = r.json()
    assert data["configured"] is True
    assert data["url"].startswith("https://checkout")


def test_link_email_and_recovery_rotate_key(client, test_state, fake_services: FakeServices):
    fake: FakePlatformClient = fake_services.platform_client
    _provision(test_state)
    old_key = test_state.state.app_settings.platform_api_key

    r = client.post("/api/platform/link-email", json={"email": "user@example.com"})
    assert r.status_code == 200
    assert test_state.state.app_settings.platform_recovery_email == "user@example.com"
    assert fake.link_email_calls and fake.link_email_calls[-1][1] == "user@example.com"

    r = client.post("/api/platform/recover/request", json={"email": "user@example.com"})
    assert r.status_code == 200
    assert fake.recover_request_calls == ["user@example.com"]

    r = client.post(
        "/api/platform/recover/confirm",
        json={"email": "user@example.com", "code": "123456"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["configured"] is True
    assert data["hasApiKey"] is True
    new_key = test_state.state.app_settings.platform_api_key
    assert new_key and new_key != old_key


def test_recover_confirm_bad_code_is_400(client, test_state):
    _provision(test_state)
    r = client.post(
        "/api/platform/recover/confirm",
        json={"email": "user@example.com", "code": "wrong"},
    )
    assert r.status_code == 400


def test_balance_gate_blocks_remote_generation_when_no_credits(
    test_state, fake_services: FakeServices
):
    """Remote Livepeer generation is blocked (403) when platform configured + no access."""
    fake: FakePlatformClient = fake_services.platform_client
    _provision(test_state)
    fake.balance_result["hasAccess"] = False

    test_state.state.app_settings.livepeer_video_enabled = True
    test_state.state.app_settings.livepeer_discovery_url = "https://discovery.example"

    req = GenerateVideoRequest(prompt="a test prompt")
    with pytest.raises(HTTPError) as excinfo:
        test_state.video_generation.generate(req)
    assert excinfo.value.status_code == 403
    assert "Insufficient credits" in excinfo.value.detail


def test_balance_gate_noop_when_platform_not_configured(test_state):
    """Without platform config the gate is a no-op (remote path proceeds — here it
    fails on the missing remote client, NOT on a 403 credits gate)."""
    test_state.state.app_settings.livepeer_video_enabled = True
    test_state.state.app_settings.livepeer_discovery_url = "https://discovery.example"

    req = GenerateVideoRequest(prompt="a test prompt")
    with pytest.raises(HTTPError) as excinfo:
        test_state.video_generation.generate(req)
    assert excinfo.value.status_code == 503  # remote not initialized — gate did not block


def test_gate_passes_through_when_credits_present(
    test_state, fake_services: FakeServices
):
    """With credits (hasAccess True) the gate lets the request through to dispatch."""
    fake: FakePlatformClient = fake_services.platform_client
    _provision(test_state)
    fake.balance_result["hasAccess"] = True
    test_state.state.app_settings.livepeer_video_enabled = True
    test_state.state.app_settings.livepeer_discovery_url = "https://discovery.example"

    req = GenerateVideoRequest(prompt="a test prompt")
    with pytest.raises(HTTPError) as excinfo:
        test_state.video_generation.generate(req)
    # Not a 403 credit gate — it proceeds to dispatch and fails on missing client.
    assert excinfo.value.status_code == 503
