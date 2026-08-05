"""In-memory fake for the platform credits API client (tests only)."""

from __future__ import annotations

from typing import Any

from services.platform_client import PlatformError


class FakePlatformClient:
    """Behavioral fake: mirrors the real provision/balance/checkout/recovery contract.

    Provisioning is idempotent per ``external_user_id`` — the first provision issues a
    key, subsequent provisions reuse it (and record the extra call so tests can assert
    the handler only provisions once). ``recover_confirm`` accepts a hardcoded recovery
    code and returns a brand-new key (recording the call) so tests can assert key
    rotation.
    """

    def __init__(self) -> None:
        self.keys_by_user: dict[str, str] = {}
        self.provision_calls: list[str] = []
        self.balance_calls: list[str] = []
        self.checkout_calls: list[tuple[str, int]] = []
        self.link_email_calls: list[tuple[str, str]] = []
        self.recover_request_calls: list[str] = []
        self.recover_confirm_calls: list[tuple[str, str]] = []
        self.recovery_code = "123456"
        self._recover_count = 0
        self.balance_result: dict[str, Any] = {
            "hasAccess": True,
            "balanceUsdMicros": "5000000",
            "remainingUsdMicros": "5000000",
            "consumedUsdMicros": "0",
            "lifetimeGrantedUsdMicros": "5000000",
        }
        self.checkout_result = "https://checkout.example/session/abc"

    def _issue_key(self, owner: str) -> str:
        key = f"platform-{owner}-{len(self.keys_by_user) + 1}"
        self.keys_by_user[owner] = key
        return key

    def provision(self, *, base_url: str, external_user_id: str) -> str:
        del base_url
        self.provision_calls.append(external_user_id)
        existing = self.keys_by_user.get(external_user_id)
        if existing is not None:
            return existing
        return self._issue_key(external_user_id[:8])

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

    def link_email(self, *, base_url: str, api_key: str, email: str) -> None:
        del base_url
        self.link_email_calls.append((api_key, email))

    def recover_request(self, *, base_url: str, email: str) -> None:
        del base_url
        self.recover_request_calls.append(email)

    def recover_confirm(self, *, base_url: str, email: str, code: str) -> str:
        del base_url
        self.recover_confirm_calls.append((email, code))
        if code != self.recovery_code:
            raise PlatformError(400, "invalid recovery code")
        self._recover_count += 1
        return self._issue_key(f"recovered-{self._recover_count}")
