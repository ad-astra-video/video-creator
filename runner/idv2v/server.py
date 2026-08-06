"""ID-V2V worker HTTP service (aiohttp).

A swappable worker container driven by the `live-runner` edge on the internal
Docker network. It does NOT register with the Livepeer Orchestrator or do
heartbeats — that is the live-runner's job. It exposes the control + inference
surface the live-runner needs:

    GET  /health          — liveness + model-loaded status
    POST /load            — build the model (int8 DiT+VACE, CPU offload)
    POST /evict           — drop the model, free GPU/CPU memory
    POST /v1/restyle      — accept a restylization job (base64 in -> base64 out)

Auth: every POST requires the shared `X-Worker-Token` header (WORKER_TOKEN env,
auto-generated if blank), which the live-runner attaches on every call.

Ported/adapted from the standalone id-v2v runner (`runner.py`) plus the control
surface /load and /evict.
"""

import asyncio
import logging
import os
import sys
import time

from aiohttp import web

from . import config
from . import run as run_mod
from .model import ModelManager, health_check

logger = logging.getLogger("video_creator.runner.idv2v.server")


def _resolve_token() -> str:
    """Resolve the worker auth token, auto-generating a stable one if blank.

    When blank at first call, generates a token and persists it back to the env
    so every process in the container agrees on it.
    """
    if config.WORKER_TOKEN:
        return config.WORKER_TOKEN
    tok = config._random_token()
    os.environ["WORKER_TOKEN"] = tok
    config.WORKER_TOKEN = tok
    logger.info("WORKER_TOKEN was blank — auto-generated (won't be shown again)")
    return tok


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _require_token(request: web.Request) -> None:
    """Reject the request unless it carries the shared worker token."""
    expected = _resolve_token()
    provided = request.headers.get("X-Worker-Token", "")
    if not provided or provided != expected:
        raise web.HTTPForbidden(reason="missing/mismatched X-Worker-Token")


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------

# One ModelManager instance owned by this worker process.
_model: ModelManager | None = None
_model_lock = asyncio.Lock()


def _get_model() -> ModelManager:
    global _model
    if _model is None:
        _model = ModelManager(device=config.GPU_DEVICE)
    return _model


async def handle_load(request: web.Request) -> web.Response:
    _require_token(request)
    global _model
    async with _model_lock:
        model = _get_model()
        if model.is_ready:
            return web.json_response({"loaded": True, "already_loaded": True})
        try:
            await asyncio.wait_for(model.load(), timeout=3600)
        except asyncio.TimeoutError:
            return web.json_response({"error": "model load timed out"}, status=504)
    return web.json_response({"loaded": True})


async def handle_evict(request: web.Request) -> web.Response:
    _require_token(request)
    global _model
    async with _model_lock:
        if _model is not None:
            _model.evict()
        _model = None
    return web.json_response({"evicted": True})


async def handle_health(request: web.Request) -> web.Response:
    model = _model
    info = health_check(model) if model is not None else {
        "status": "unloaded", "model_loaded": False, "device": config.GPU_DEVICE,
        "precision": config.IDV2V_QUANT, "offload": config.IDV2V_OFFLOAD,
    }
    return web.json_response({"status": "ok", **info})


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

async def handle_restyle(request: web.Request) -> web.Response:
    _require_token(request)
    body = await request.json()
    model = _get_model()
    if not model.is_ready:
        return web.json_response(
            {"error": "Model not loaded yet — send POST /load first"}, status=409,
        )
    try:
        result = await run_mod.process_job(model, body)
    except Exception as exc:
        logger.error("Restyle job failed: %s", exc, exc_info=True)
        return web.json_response({"error": str(exc)}, status=500)
    return web.json_response(result)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> web.Application:
    app = web.Application(client_max_size=config.MAX_BODY_BYTES)
    app.router.add_get("/health", handle_health)
    app.router.add_post("/load", handle_load)
    app.router.add_post("/evict", handle_evict)
    app.router.add_post("/v1/restyle", handle_restyle)
    return app


async def _run() -> None:
    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, config.HOST, config.PORT)
    await site.start()
    logger.info("ID-V2V worker listening on %s:%d", config.HOST, config.PORT)
    await asyncio.Event().wait()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        stream=sys.stdout)
    # Resolve auth token eagerly so a blank one is generated + logged once.
    _resolve_token()
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
