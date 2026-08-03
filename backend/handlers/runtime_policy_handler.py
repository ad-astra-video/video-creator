"""Runtime policy query handler."""

from __future__ import annotations

from collections.abc import Callable

from api_types import RuntimePolicyResponse
from runtime_config.runtime_config import RuntimeConfig


class RuntimePolicyHandler:
    def __init__(
        self,
        config: RuntimeConfig,
        remote_livepeer_active: Callable[[], bool] | None = None,
    ) -> None:
        self._config = config
        # A callable the caller wires to current settings so this handler can
        # check whether remote Livepeer inference is enabled+configured at
        # request time (settings can change between requests).
        self._remote_livepeer_active = remote_livepeer_active

    def get_runtime_policy(self) -> RuntimePolicyResponse:
        # Livepeer remote inference is a first-class generation path that needs
        # neither an LTX API key nor a FAL AI key. When it's active we must NOT
        # force API-key-only generation, even though local generation is
        # unsupported on this box.
        remote_active = bool(self._remote_livepeer_active and self._remote_livepeer_active())
        force_api = self._config.force_api_generations and not remote_active
        # Server-side single source of truth for forced API mode.
        return RuntimePolicyResponse(force_api_generations=force_api)
