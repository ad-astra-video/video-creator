"""Per-install platform credits provisioning handler.

Module A scope: give the install a stable identity (UUID) and provision a per-user
API key from the remote platform exactly once. Balance/checkout are later modules;
this handler only guarantees the install is ready to talk to the platform.
"""

from __future__ import annotations

import uuid
from threading import RLock
from typing import TYPE_CHECKING

from api_types import PlatformStatusResponse
from handlers.base import StateHandlerBase
from state.app_state_types import AppState

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig
    from services.platform_client import PlatformClient
    from handlers.settings_handler import SettingsHandler


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
