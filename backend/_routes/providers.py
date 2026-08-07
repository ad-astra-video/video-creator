"""Provider discovery and management routes."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from app_handler import AppHandler
from state import get_state_service
from state.app_settings import AppSettingsPatch

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/providers", tags=["providers"])


class SelectProviderRequest(BaseModel):
    runner_id: str


class ExcludeProviderRequest(BaseModel):
    runner_id: str


@router.get("")
async def get_providers(
    request: Request,
    handler: AppHandler = Depends(get_state_service),
) -> dict[str, Any]:
    """List discovered providers with selection/exclusion state."""
    settings = handler.settings.get_settings_snapshot()
    livepeer_client = getattr(handler.state, "_livepeer_client", None)

    runners = {}
    if livepeer_client:
        runners = {r.runner_id: r for r in livepeer_client._runners.values()}

    providers = []
    for runner_id, runner in runners.items():
        providers.append({
            "runner_id": runner_id,
            "url": runner.url,
            "gpu": runner.gpu,
            "price_info": runner.price_info,
            "selected": runner_id == settings.livepeer_selected_runner_id,
            "excluded": runner_id in settings.livepeer_excluded_runner_ids,
            "status": runner.status,
        })

    return {"providers": providers, "total": len(providers), "online": len(providers)}


def reconcile_livepeer_client(handler: AppHandler) -> Any | None:
    """Return a LivepeerClient that matches the *current* discovery settings.

    The LivepeerClient is created once at startup and snapshots the discovery
    URL at __init__, so after the user updates the discovery URL (or configures
    one that wasn't set at startup) the cached client would keep pointing at a
    stale/empty endpoint. Reconcile against current settings here, replacing
    ``handler.state._livepeer_client`` when the URL or API key changed. Returns
    None when no discovery URL is configured.
    """
    settings = handler.settings.get_settings_snapshot()
    if not settings.livepeer_discovery_url:
        return None

    livepeer_client = getattr(handler.state, "_livepeer_client", None)
    if (
        livepeer_client is not None
        and livepeer_client.discovery_url == settings.livepeer_discovery_url
        and livepeer_client.api_key == settings.livepeer_api_key
    ):
        return livepeer_client

    from pathlib import Path
    from services.livepeer_client import LivepeerClient

    results_dir = Path(handler.config.outputs_dir) / "livepeer"
    livepeer_client = LivepeerClient(
        discovery_url=settings.livepeer_discovery_url,
        results_dir=results_dir,
        api_key=settings.livepeer_api_key,
    )
    # Store on handler state so generation handlers keep using the fresh client.
    handler.state._livepeer_client = livepeer_client  # type: ignore[attr-defined]
    return livepeer_client


@router.post("/discover")
async def discover_providers(
    request: Request,
    handler: AppHandler = Depends(get_state_service),
) -> dict[str, Any]:
    """Trigger a fresh discovery sweep against the *current* discovery URL."""
    settings = handler.settings.get_settings_snapshot()
    livepeer_client = reconcile_livepeer_client(handler)
    if not livepeer_client:
        return {"error": "Discovery URL not configured"}

    runners = await livepeer_client.discover()
    if not runners:
        return {"providers": [], "error": "No runners discovered"}

    # Auto-select first if no explicit selection
    if not settings.livepeer_selected_runner_id and runners:
        patch = AppSettingsPatch(livepeer_selected_runner_id=runners[0].runner_id)
        handler.settings.update_settings(patch)

    return {"providers": [r.to_dict() for r in runners], "total": len(runners)}


@router.post("/select")
async def select_provider(
    req: SelectProviderRequest,
    request: Request,
    handler: AppHandler = Depends(get_state_service),
) -> dict[str, Any]:
    """Select a provider for inference."""
    patch = AppSettingsPatch(livepeer_selected_runner_id=req.runner_id)
    handler.settings.update_settings(patch)
    return {"ok": True, "runner_id": req.runner_id}


@router.post("/exclude")
async def exclude_provider(
    req: ExcludeProviderRequest,
    request: Request,
    handler: AppHandler = Depends(get_state_service),
) -> dict[str, Any]:
    """Toggle exclusion for a provider."""
    settings = handler.settings.get_settings_snapshot()
    excluded = list(settings.livepeer_excluded_runner_ids)
    runner_id = req.runner_id

    if runner_id in excluded:
        excluded.remove(runner_id)
    else:
        excluded.append(runner_id)

    # If excluded runner was selected, clear selection
    if settings.livepeer_selected_runner_id == runner_id and runner_id in excluded:
        patch = AppSettingsPatch(
            livepeer_excluded_runner_ids=excluded,
            livepeer_selected_runner_id="",
        )
    else:
        patch = AppSettingsPatch(livepeer_excluded_runner_ids=excluded)

    handler.settings.update_settings(patch)
    return {"ok": True}
