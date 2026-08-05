"""Platform credits provisioning, balance, checkout + recovery routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api_types import (
    PlatformBalanceResponse,
    PlatformCheckoutRequest,
    PlatformCheckoutResponse,
    PlatformLinkEmailRequest,
    PlatformLinkEmailResponse,
    PlatformRecoverConfirmRequest,
    PlatformRecoverConfirmResponse,
    PlatformRecoverRequest,
    PlatformStatusResponse,
    StatusResponse,
)
from app_handler import AppHandler
from state import get_state_service

router = APIRouter(prefix="/api", tags=["platform"])


@router.get("/platform/status", response_model=PlatformStatusResponse)
def route_platform_status(
    handler: AppHandler = Depends(get_state_service),
) -> PlatformStatusResponse:
    return handler.platform.get_status()


@router.get("/platform/balance", response_model=PlatformBalanceResponse)
def route_platform_balance(
    handler: AppHandler = Depends(get_state_service),
) -> PlatformBalanceResponse:
    return handler.platform.get_balance()


@router.post("/platform/checkout", response_model=PlatformCheckoutResponse)
def route_platform_checkout(
    req: PlatformCheckoutRequest,
    handler: AppHandler = Depends(get_state_service),
) -> PlatformCheckoutResponse:
    return handler.platform.create_checkout(req.tier)


@router.post("/platform/link-email", response_model=PlatformLinkEmailResponse)
def route_platform_link_email(
    req: PlatformLinkEmailRequest,
    handler: AppHandler = Depends(get_state_service),
) -> PlatformLinkEmailResponse:
    return handler.platform.link_recovery_email(req.email)


@router.post("/platform/recover/request", response_model=StatusResponse)
def route_platform_recover_request(
    req: PlatformRecoverRequest,
    handler: AppHandler = Depends(get_state_service),
) -> StatusResponse:
    handler.platform.request_recovery(req.email)
    return StatusResponse(status="sent")


@router.post("/platform/recover/confirm", response_model=PlatformRecoverConfirmResponse)
def route_platform_recover_confirm(
    req: PlatformRecoverConfirmRequest,
    handler: AppHandler = Depends(get_state_service),
) -> PlatformRecoverConfirmResponse:
    return handler.platform.confirm_recovery(req.email, req.code)
