"""Tests for the platform credits API integration (Module A: provisioning)."""

from __future__ import annotations

import uuid

from tests.conftest import TEST_ADMIN_TOKEN
from tests.fakes.fake_platform_client import FakePlatformClient
from tests.fakes.services import FakeServices


def test_settings_response_masks_platform_api_key(client, test_state):
    """GET /api/settings surfaces platform fields but never the api key secret."""
    test_state.state.app_settings.platform_base_url = "https://platform.example"
    test_state.state.app_settings.platform_api_key = "super-secret-key"
    test_state.state.app_settings.platform_user_id = "00000000-0000-4000-8000-000000000001"
    test_state.state.app_settings.platform_recovery_email = "user@example.com"

    r = client.get("/api/settings")
    assert r.status_code == 200
    data = r.json()
    assert data["platformBaseUrl"] == "https://platform.example"
    assert data["hasPlatformBaseUrl"] is True
    assert data["platformUserId"] == "00000000-0000-4000-8000-000000000001"
    assert data["platformRecoveryEmail"] == "user@example.com"
    assert data["hasPlatformApiKey"] is True
    assert "platformApiKey" not in data


def test_settings_patch_accepts_platform_fields(client, test_state):
    """POST /api/settings accepts the non-secret platform fields and persists them."""
    r = client.post(
        "/api/settings",
        json={
            "platformBaseUrl": "https://platform.example",
            "platformRecoveryEmail": "ops@example.com",
        },
        headers={"X-Admin-Token": TEST_ADMIN_TOKEN},
    )
    assert r.status_code == 200
    assert test_state.state.app_settings.platform_base_url == "https://platform.example"
    assert test_state.state.app_settings.platform_recovery_email == "ops@example.com"


def test_ensure_ready_provisions_once_and_reuses_key(test_state, fake_services: FakeServices):
    fake: FakePlatformClient = fake_services.platform_client
    test_state.state.app_settings.platform_base_url = "https://platform.example"

    test_state.platform.ensure_ready()

    user_id = test_state.state.app_settings.platform_user_id
    assert user_id
    # uuid4 -> version 4
    assert uuid.UUID(user_id).version == 4
    assert fake.provision_calls == [user_id]
    key = test_state.state.app_settings.platform_api_key
    assert key

    # Idempotent: second call must NOT provision again.
    test_state.platform.ensure_ready()
    assert fake.provision_calls == [user_id]
    assert test_state.state.app_settings.platform_api_key == key


def test_ensure_ready_does_not_provision_without_base_url(test_state, fake_services: FakeServices):
    fake: FakePlatformClient = fake_services.platform_client
    test_state.platform.ensure_ready()
    assert fake.provision_calls == []
    assert test_state.state.app_settings.platform_user_id  # identity still minted


def test_platform_status_route(client, test_state):
    """GET /api/platform/status provisions and returns a secret-free status."""
    test_state.state.app_settings.platform_base_url = "https://platform.example"

    r = client.get("/api/platform/status")
    assert r.status_code == 200
    data = r.json()
    assert data["configured"] is True
    assert data["hasApiKey"] is True
    assert data["baseUrl"] == "https://platform.example"
    assert data["userId"]
    # No secret ever exposed.
    assert "apiKey" not in data
    assert "platformApiKey" not in data
