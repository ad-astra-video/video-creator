"""Route handler for POST /api/restyle."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api_types import RestyleRequest, RestyleResponse
from state import get_state_service
from app_handler import AppHandler

router = APIRouter(prefix="/api", tags=["restyle"])


@router.post("/restyle", response_model=RestyleResponse)
def route_restyle(req: RestyleRequest, handler: AppHandler = Depends(get_state_service)) -> RestyleResponse:
    return handler.restyle.run(req)
