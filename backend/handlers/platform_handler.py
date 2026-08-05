"""Per-install platform credits provisioning + recovery + balance handler.

Module A scope: give the install a stable identity (UUID) and provision a per-user
API key from the remote platform exactly once. This module extends that with the
credits/balance surface (balance, checkout) and key-recovery (link-email,
recover/request, recover/confirm), plus the remote-generation balance gate.
"""

from __future__ import annotations

import logging
import uuid
from threading import RLock
from typing import TYPE_CHECKING, Any

from api_types import (
    PlatformBalanceResponse,
    PlatformCheckoutResponse,
    PlatformLinkEmailResponse,
    PlatformRecoverConfirmResponse,
    PlatformStatusResponse,
)
from _routes._errors import HTTPError
from handlers.base import StateHandlerBase
from services.platform_client import PlatformError
from state.app_state_types import AppState

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig
    from services.platform_client import PlatformClient
    from handlers.settings_handler import SettingsHandler

logger = logging.getLogger(__name__)


def _to_int(value: Any) -> int:
    """Best-effort parse of a numeric platform payload field to an int."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


class PlatformHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        config: RuntimeConfig,
        settings_handler: SettingsHandler,
        platform_client: PlatformClient,
    ) -> None:
        super().__init__(state, lock, config)
        self._settings = settings_handler
        self._platform_client = platform_client

    def ensure_ready(self) -> None:
        """Ensure the install has identity + a provisioned platform API key.

        Idempotent: the UUID is generated once and persisted; the platform key is
        provisioned once (only when a base URL is set and no key is stored yet).
        Network IO (provisioning) happens outside the state lock.
        """
        with self.lock:
            settings = self.state.app_settings
            if not settings.platform_user_id:
                settings.platform_user_id = str(uuid.uuid4())
                self._settings.save_settings()
            user_id = settings.platform_user_id
            base_url = settings.platform_base_url.strip()
            if not base_url or settings.platform_api_key.strip():
                return

        # Outside the lock: slow network IO against the platform.
        key = self._platform_client.provision(base_url=base_url, external_user_id=user_id)

        with self.lock:
            settings = self.state.app_settings
            if not settings.platform_api_key.strip():
                settings.platform_api_key = key
                self._settings.save_settings()

    def get_status(self) -> PlatformStatusResponse:
        """Provision (if needed) and return a secret-free provisioning status."""
        self.ensure_ready()
        with self.lock:
            settings = self.state.app_settings
            return PlatformStatusResponse(
                user_id=settings.platform_user_id,
                configured=bool(
                    settings.platform_base_url.strip() and settings.platform_api_key.strip()
                ),
                has_api_key=bool(settings.platform_api_key.strip()),
                base_url=settings.platform_base_url,
            )

    def get_balance(self) -> PlatformBalanceResponse:
        """Return the current platform balance/access snapshot.

        When the platform is not configured, returns zeros with ``configured=False``
        rather than erroring. Network IO happens outside the lock.
        """
        with self.lock:
            settings = self.state.app_settings
            base_url = settings.platform_base_url.strip()
            api_key = settings.platform_api_key.strip()
            if not base_url or not api_key:
                return PlatformBalanceResponse(
                    has_access=False,
                    balance_usd_micros=0,
                    remaining_usd_micros=0,
                    consumed_usd_micros=0,
                    lifetime_granted_usd_micros=0,
                    configured=False,
                )
        payload = self._platform_client.get_balance(base_url=base_url, api_key=api_key)
        return PlatformBalanceResponse(
            has_access=bool(payload.get("hasAccess")),
            balance_usd_micros=_to_int(payload.get("balanceUsdMicros")),
            remaining_usd_micros=_to_int(payload.get("remainingUsdMicros")),
            consumed_usd_micros=_to_int(payload.get("consumedUsdMicros")),
            lifetime_granted_usd_micros=_to_int(payload.get("lifetimeGrantedUsdMicros")),
            configured=True,
        )

    def create_checkout(self, tier: int) -> PlatformCheckoutResponse:
        """Create a hosted credit top-up checkout session for ``tier`` cents."""
        with self.lock:
            settings = self.state.app_settings
            base_url = settings.platform_base_url.strip()
            api_key = settings.platform_api_key.strip()
            if not base_url or not api_key:
                return PlatformCheckoutResponse(url="", configured=False)
        url = self._platform_client.create_checkout(
            base_url=base_url, api_key=api_key, tier_credits_cents=tier
        )
        return PlatformCheckoutResponse(url=url, configured=True)

    def link_recovery_email(self, email: str) -> PlatformLinkEmailResponse:
        """Persist the recovery email locally and associate it with the platform key."""
        with self.lock:
            settings = self.state.app_settings
            settings.platform_recovery_email = email
            self._settings.save_settings()
            base_url = settings.platform_base_url.strip()
            api_key = settings.platform_api_key.strip()
            configured = bool(base_url and api_key)
        if configured:
            self._platform_client.link_email(base_url=base_url, api_key=api_key, email=email)
        return PlatformLinkEmailResponse(configured=configured)

    def request_recovery(self, email: str) -> None:
        """Ask the platform to email a one-time recovery code to ``email``."""
        with self.lock:
            settings = self.state.app_settings
            settings.platform_recovery_email = email
            self._settings.save_settings()
            base_url = settings.platform_base_url.strip()
        if base_url:
            self._platform_client.recover_request(base_url=base_url, email=email)

    def confirm_recovery(self, email: str, code: str) -> PlatformRecoverConfirmResponse:
        """Confirm a recovery code and rotate the stored platform API key.

        On success the NEW key from the platform is persisted to settings (replacing
        the lost one). Network IO happens outside the lock; the settings write-back
        happens under the lock.
        """
        with self.lock:
            base_url = self.state.app_settings.platform_base_url.strip()
            if not base_url:
                return PlatformRecoverConfirmResponse(has_api_key=False, configured=False)
        try:
            new_key = self._platform_client.recover_confirm(base_url=base_url, email=email, code=code)
        except PlatformError:
            raise HTTPError(400, "Invalid or expired recovery code") from None
        with self.lock:
            settings = self.state.app_settings
            settings.platform_api_key = new_key
            self._settings.save_settings()
        return PlatformRecoverConfirmResponse(has_api_key=True, configured=True)

    def ensure_generation_allowed(self) -> None:
        """Balance gate: block remote Livepeer generation when the platform is
        configured but has no access (no credits).

        No-op when the platform is not configured, and fail-open if the balance cannot
        be queried — only a definitive ``hasAccess=False`` blocks generation.
        """
        with self.lock:
            settings = self.state.app_settings
            base_url = settings.platform_base_url.strip()
            api_key = settings.platform_api_key.strip()
            if not base_url or not api_key:
                return
        try:
            payload = self._platform_client.get_balance(base_url=base_url, api_key=api_key)
        except PlatformError:
            logger.warning("Platform balance check failed; allowing generation", exc_info=True)
            return
        if not payload.get("hasAccess"):
            raise HTTPError(403, "Insufficient credits - top up in the Credits panel")
