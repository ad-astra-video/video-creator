"""Platform credits provisioning + status routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api_types import PlatformStatusResponse
from app_handler import AppHandler
from state import get_state_service

router = APIRouter(prefix="/api", tags=["platform"])


@router.get("/platform/status", response_model=PlatformStatusResponse)
def route_platform_status(
    handler: AppHandler = Depends(get_state_service),
) -> PlatformStatusResponse:
    return handler.platform.get_status()
