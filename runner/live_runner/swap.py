"""ResidentWorkerManager — owns GPU residency across the worker containers.

Only ONE worker has its model resident on the shared 32 GB GPU at a time.
``ensure(name)`` evicts the current resident (POST /evict) before loading the
target (POST /load), and records the new resident. Serialized by an asyncio lock
so concurrent requests can never race two models into VRAM simultaneously.

The ``transport`` is a small abstract over the HTTP calls so the TDD
ResidentWorkerManager test can inject an in-memory fake (see
runner/tests/test_live_runner_router.py).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("video_creator.runner.live_runner.swap")


class WorkerTransport:
    """HTTP transport to worker containers (control + health calls).

    ``base`` is the worker's root URL (e.g. http://ltx-worker:8991). ``post``
    appends the path for POST /load and /evict; ``health`` GETs /health.
    """

    async def post(self, base: str, path: str, payload: dict | None = None) -> dict:
        raise NotImplementedError

    async def health(self, base: str) -> dict:
        raise NotImplementedError


class HttpWorkerTransport(WorkerTransport):
    """Real aiohttp transport: adds X-Worker-Token to control POSTs."""

    def __init__(self, session, token: str):
        self._session = session
        self._token = token

    def _headers(self, authed: bool) -> dict:
        if authed:
            return {"X-Worker-Token": self._token}
        return {}

    async def post(self, base: str, path: str, payload: dict | None = None) -> dict:
        async with self._session.post(
            base + path, json=payload or {}, headers=self._headers(True)
        ) as resp:
            if resp.status >= 400:
                text = await resp.text()
                raise RuntimeError(f"worker call {base}{path} -> {resp.status}: {text[:300]}")
            return await resp.json()

    async def health(self, base: str) -> dict:
        async with self._session.get(base + "/health") as resp:
            if resp.status >= 400:
                raise RuntimeError(f"worker health {base} -> {resp.status}")
            return await resp.json()


@dataclass
class ResidentWorkerManager:
    transport: WorkerTransport
    workers: dict[str, str] = field(default_factory=dict)  # worker name -> base URL
    _current: str | None = field(default=None, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def _base(self, name: str) -> str:
        """Resolve a worker name to its base URL from the injected mapping."""
        try:
            return self.workers[name]
        except KeyError:
            raise KeyError(f"unknown worker: {name} (known: {list(self.workers)})")

    @property
    def resident(self) -> str | None:
        """Name of the worker whose model is currently resident (or None)."""
        return self._current

    async def ensure(self, name: str) -> None:
        """Make ``name`` resident: evict the current resident (if any), load ``name``."""
        async with self._lock:
            if self._current == name:
                return
            if self._current:
                await self.transport.post(self._base(self._current), "/evict")
            await self.transport.post(self._base(name), "/load")
            self._current = name
            logger.info("GPU swap: %s resident", name)

    async def evict_all(self) -> None:
        """Evict the current resident (if any)."""
        async with self._lock:
            if self._current:
                await self.transport.post(self._base(self._current), "/evict")
                logger.info("GPU swap: evicted %s", self._current)
                self._current = None

    async def check_health(self) -> dict:
        """Live /health probe of every worker for heartbeat metadata.

        Returns {'ltx_worker_up': bool, 'idv2v_worker_up': bool, 'warm_model': str|None}.
        """
        up = {name: False for name in self.workers}
        for name in up:
            try:
                await self.transport.health(self._base(name))
                up[name] = True
            except Exception:
                pass
        return {
            **{f"{name}_up": v for name, v in up.items()},
            "warm_model": self._current,
        }
