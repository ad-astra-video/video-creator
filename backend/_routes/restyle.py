"""Route handler for POST /api/restyle and POST /api/restyle/extract-first-frame."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api_types import (
    ExtractFirstFrameRequest,
    ExtractFirstFrameResponse,
    RestyleRequest,
    RestyleResponse,
)
from state import get_state_service
from app_handler import AppHandler

router = APIRouter(prefix="/api", tags=["restyle"])


@router.post("/restyle", response_model=RestyleResponse)
def route_restyle(req: RestyleRequest, handler: AppHandler = Depends(get_state_service)) -> RestyleResponse:
    return handler.restyle.run(req)


@router.post("/restyle/extract-first-frame", response_model=ExtractFirstFrameResponse)
def route_extract_first_frame(
    req: ExtractFirstFrameRequest,
    handler: AppHandler = Depends(get_state_service),
) -> ExtractFirstFrameResponse:
    return handler.restyle.extract_first_frame(req)
