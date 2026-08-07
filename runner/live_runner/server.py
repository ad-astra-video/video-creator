"""Live-runner HTTP service — the single Livepeer-facing edge.

Registers + heartbeats with the Livepeer Orchestrator as app="video-creator",
owns the shared-GPU swap policy (ResidentWorkerManager), routes each
/video-creator/v1/* request to the LTX or ID-V2V worker, and proxies
request/response. Heartbeat metadata carries the warm model + worker up/down
status so the desktop can prefer a warm restyle runner.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys

import aiohttp
from aiohttp import web

from livepeer_gateway.live_runner import (
    LiveRunnerGPU,
    register_runner,
)

from . import config
from .routing import CAPABILITIES, ROUTES, proxy
from .swap import HttpWorkerTransport, ResidentWorkerManager

logger = logging.getLogger("video_creator.runner.live_runner.server")

_max_body = int(os.environ.get("MAX_BODY_BYTES", "3000000000"))

# Global state
_session: aiohttp.ClientSession | None = None
_worker_manager: ResidentWorkerManager | None = None
_registration = None
_ready = False
_generation_sem = None  # asyncio.Semaphore(1) — single GPU, one inference at a time


def _need(request: web.Request):
    """Return the (worker_manager, session, token) trio, raising 503 if not up."""
    if _worker_manager is None or _session is None:
        raise web.HTTPServiceUnavailable(reason="live-runner not ready")
    return _worker_manager, _session, config.worker_token()


async def handle_health(_req: web.Request) -> web.Response:
    return web.json_response({"ok": True, "ready": _ready, "app": config.APP_ID})


async def handle_info(_req: web.Request) -> web.Response:
    wm, session, token = _need(_req)
    meta = await wm.check_health() if wm else {}
    return web.json_response({
        "runner_id": _registration.runner_id if _registration else "",
        "app": config.APP_ID,
        "capabilities": CAPABILITIES,
        "ready": _ready,
        "gpu": {"name": config.GPU_NAME, "vram_mb": config.GPU_VRAM_MB},
        "metadata": meta,
    })


async def handle_generic(req: web.Request) -> web.Response:
    """Proxy a /video-creator/v1/{endpoint} request to its worker.

    The endpoint name is the last non-empty path segment. Body is forward as-is
    (base64 in -> base64 out); the swap policy makes the right model resident.
    """
    endpoint = req.match_info.get("endpoint", "")
    worker = ROUTES.get(endpoint)
    if worker is None:
        return web.json_response({"error": f"unknown endpoint: {endpoint}"}, status=404)

    wm, session, token = _need(req)
    body = await req.json()

    # Serialize inference: the shared GPU runs one generation at a time.
    async with _generation_sem:
        return await proxy(wm, session, token, worker, endpoint, body)


async def on_startup(_app: web.Application) -> None:
    global _session, _worker_manager, _registration, _ready, _generation_sem
    _session = aiohttp.ClientSession()
    _worker_manager = ResidentWorkerManager(
        transport=HttpWorkerTransport(_session, config.worker_token()),
        workers=dict(config.WORKERS),
    )
    _generation_sem = asyncio.Semaphore(1)

    gpu = LiveRunnerGPU(name=config.GPU_NAME, vram_mb=config.GPU_VRAM_MB)
    _registration = await register_runner(
        config.ORCHESTRATOR_URL,
        secret=config.ORCHESTRATOR_SECRET,
        runner_url=config.RUNNER_URL,
        app=config.APP_ID,
        mode="single-shot",
        price=config.PRICE,
        unit=config.PRICE_UNIT,
        gpu=gpu,
        label="restyle",
        metadata=json.dumps({
            "capabilities": CAPABILITIES,
            "ltx_worker_up": False,
            "idv2v_worker_up": False,
            "warm_model": None,
        }),
        heartbeat_interval_s=config.HEARTBEAT_INTERVAL_S,
    )
    await _registration.start()
    logger.info("Registered live-runner %s (app=%s)", _registration.runner_id, config.APP_ID)

    # Refresh heartbeat metadata each beat from the swap policy + live worker /health.
    asyncio.create_task(_refresh_metadata_loop())
    _ready = True
    logger.info("Live-runner READY")


async def _refresh_metadata_loop() -> None:
    while True:
        try:
            if _worker_manager is not None and _registration is not None:
                meta = await _worker_manager.check_health()
                meta["capabilities"] = CAPABILITIES
                # The registration is an in-process object owned by this runner;
                # set its payload metadata so the next heartbeat advertises the
                # current warm-model + worker up/down status. (No SDK change needed.)
                _registration._metadata = json.dumps(meta)
        except Exception:
            logger.warning("metadata refresh failed", exc_info=True)
        await asyncio.sleep(config.HEARTBEAT_INTERVAL_S)


async def on_cleanup(_app: web.Application) -> None:
    global _registration, _session
    if _registration is not None:
        try:
            await _registration.close()
        except Exception:
            logger.debug("unregister failed", exc_info=True)
        _registration = None
    if _session is not None:
        await _session.close()
        _session = None


def create_app() -> web.Application:
    app = web.Application(client_max_size=_max_body)
    p = "/video-creator/v1"
    app.router.add_get(f"{p}/health", handle_health)
    app.router.add_get(f"{p}/info", handle_info)
    # One parameterized route so handle_generic can read the endpoint from
    # match_info["endpoint"] and look it up in ROUTES. (Previously these were
    # registered as static paths with no {endpoint} placeholder, so match_info
    # was empty and every proxied call returned 404 "unknown endpoint: ".)
    app.router.add_post(f"{p}/{{endpoint}}", handle_generic)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stdout)
    # Resolve auth token eagerly so a blank one is generated + logged once.
    config.worker_token()
    app = create_app()
    web.run_app(app, host=config.HOST, port=config.PORT)


if __name__ == "__main__":
    main()
