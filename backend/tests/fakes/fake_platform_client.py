"""In-memory fake for the platform credits API client (tests only)."""

from __future__ import annotations

from typing import Any

from services.platform_client import PlatformError


class FakePlatformClient:
    """Behavioral fake: mirrors the real provision/balance/checkout contract.

    Provisioning is idempotent per ``external_user_id`` — the first provision issues a
    key, subsequent provisions reuse it (and record the extra call so tests can assert
    the handler only provisions once).
    """

    def __init__(self) -> None:
        self.keys_by_user: dict[str, str] = {}
        self.provision_calls: list[str] = []
        self.balance_calls: list[str] = []
        self.checkout_calls: list[tuple[str, int]] = []
        self.balance_result: dict[str, Any] = {
            "hasAccess": True,
            "balanceUsdMicros": "5000000",
            "remainingUsdMicros": "5000000",
            "consumedUsdMicros": "0",
            "lifetimeGrantedUsdMicros": "5000000",
        }
        self.checkout_result = "https://checkout.example/session/abc"

    def provision(self, *, base_url: str, external_user_id: str) -> str:
        del base_url
        self.provision_calls.append(external_user_id)
        existing = self.keys_by_user.get(external_user_id)
        if existing is not None:
            return existing
        key = f"platform-key-{external_user_id[:8]}"
        self.keys_by_user[external_user_id] = key
        return key

    def get_balance(self, *, base_url: str, api_key: str) -> dict[str, Any]:
        del base_url
        self.balance_calls.append(api_key)
        if api_key not in self.keys_by_user.values():
            raise PlatformError(401, "invalid api key")
        return dict(self.balance_result)

    def create_checkout(self, *, base_url: str, api_key: str, tier_credits_cents: int) -> str:
        del base_url
        self.checkout_calls.append((api_key, tier_credits_cents))
        return self.checkout_result
