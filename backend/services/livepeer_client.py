"""Livepeer remote inference client.

Discovers runners through orchestrators and routes generation requests.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import aiohttp

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


class RunnerInfo:
    """Normalized runner metadata from discovery."""

    def __init__(self, runner_id: str, url: str, raw: dict) -> None:
        self.runner_id = runner_id
        self.url = url
        self.raw = raw
        self.gpu = raw.get("gpu", {})
        self.price_info = raw.get("price_info")
        self.status = raw.get("status", "ready")

    def to_dict(self) -> dict:
        return {
            "runner_id": self.runner_id,
            "url": self.url,
            "gpu": self.gpu,
            "price_info": self.price_info,
            "status": self.status,
        }


class LivepeerClient:
    """Discovers and calls LTX-Desktop runners through Livepeer orchestrators."""

    def __init__(self, signer_url: str, results_dir: Path) -> None:
        self.signer_url = signer_url
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._runners: dict[str, RunnerInfo] = {}
        self._discovery_task: asyncio.Task | None = None

    async def discover(self) -> list[RunnerInfo]:
        """Query orchestrator for available runners."""
        try:
            signer_base = self.signer_url.rstrip("/")
            async with aiohttp.ClientSession() as session:
                # Get orchestrator list from signer
                orch_url = f"{signer_base}/orchestrators"
                async with session.get(orch_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        logger.warning("Signer %s returned %s for orchestrators", self.signer_url, resp.status)
                        return []
                    orch_data = await resp.json()
                    orch_urls = orch_data if isinstance(orch_data, list) else []

                # Query each orchestrator for runners
                runners: dict[str, RunnerInfo] = {}
                for orch in orch_urls:
                    try:
                        discovery_url = f"{orch.rstrip('/')}/discovery"
                        async with session.get(
                            discovery_url,
                            params={"app": "ltx-desktop"},
                            timeout=aiohttp.ClientTimeout(total=10),
                        ) as resp:
                            if resp.status != 200:
                                continue
                            data = await resp.json()
                            for runner_raw in (data if isinstance(data, list) else []):
                                rid = runner_raw.get("runner_id", "")
                                if rid:
                                    runners[rid] = RunnerInfo(
                                        runner_id=rid,
                                        url=runner_raw.get("runner_url", ""),
                                        raw=runner_raw,
                                    )
                    except Exception:
                        logger.debug("Failed to query orchestrator %s", orch, exc_info=True)

                self._runners = runners
                logger.info("Discovered %d runners", len(self._runners))
                return list(self._runners.values())
        except Exception:
            logger.exception("Discovery failed")
            return []

    async def periodic_discovery(self, interval_s: float = 60.0) -> None:
        """Background discovery loop."""
        while True:
            await asyncio.sleep(interval_s)
            await self.discover()

    def get_runner(
        self, selected_id: str, excluded_ids: list[str]
    ) -> RunnerInfo | None:
        """Pick runner: explicit selection > first non-excluded."""
        if selected_id and selected_id in self._runners:
            return self._runners[selected_id]
        for rid, runner in self._runners.items():
            if rid not in excluded_ids:
                return runner
        return None

    async def call(
        self, runner: RunnerInfo, endpoint: str, payload: dict, timeout_s: float = 600.0
    ) -> dict:
        """Call a runner endpoint and return JSON."""
        runner_url = runner.url.rstrip("/") + endpoint
        logger.info("Calling runner %s %s", runner.runner_id, runner_url)
        async with aiohttp.ClientSession() as session:
            async with session.post(
                runner_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout_s),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Runner returned {resp.status}: {text}")
                return await resp.json()

    def save_result(self, base64_data: str, content_type: str) -> str:
        """Decode base64 and save to results directory."""
        gen_id = uuid.uuid4().hex[:12]
        ext = {"video/mp4": ".mp4", "image/png": ".png", "image/jpeg": ".jpg"}.get(content_type, ".bin")
        path = self.results_dir / f"{gen_id}{ext}"
        path.write_bytes(base64.b64decode(base64_data))
        return str(path)
