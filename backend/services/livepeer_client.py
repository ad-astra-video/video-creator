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
import ssl

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

    def __init__(self, discovery_url: str, results_dir: Path, api_key: str = "") -> None:
        self.discovery_url = discovery_url
        self.results_dir = results_dir
        self.api_key = api_key
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._runners: dict[str, RunnerInfo] = {}
        self._discovery_task: asyncio.Task | None = None
        # Self-signed test orchestrator on .8 → skip TLS verification (matches
        # the livepeer gateway SDK's ssl=False).
        self._ssl = ssl.create_default_context()
        self._ssl.check_hostname = False
        self._ssl.verify_mode = ssl.CERT_NONE

    @staticmethod
    def _ssl_ctx():
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    def _auth_headers(self) -> dict[str, str]:
        """Optional bearer token attached to outbound orchestrator/runner calls."""
        if self.api_key.strip():
            return {"Authorization": f"Bearer {self.api_key.strip()}"}
        return {}

    async def discover(self) -> list[RunnerInfo]:
        """Query the configured discovery URL directly for available runners.

        The value stored in settings is used verbatim as the discovery endpoint —
        the client does NOT append /discovery or probe /orchestrators. Supports
        both the real go-livepeer orchestrator format (discovery returns
        ``[{address, runners: [{url, gpu, app, mode, ...}]}]``) and the legacy
        flat ``[{runner_id, runner_url}]`` mock format.
        """
        try:
            url = self.discovery_url.rstrip("/")
            if not url:
                return []
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=self._ssl_ctx())) as session:
                runners: dict[str, RunnerInfo] = {}
                try:
                    async with session.get(
                        url,
                        params={"app": "ltx-desktop"},
                        timeout=aiohttp.ClientTimeout(total=10),
                        headers=self._auth_headers(),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            for runner_raw in self._parse_discovery(data):
                                rid = runner_raw.get("runner_id", "")
                                if not rid:
                                    # go-livepeer gives proxy URLs of the form
                                    # .../apps/runner_XXXXX/app → take the 2nd-to-last segment
                                    _u = (runner_raw.get("url") or "").rstrip("/")
                                    _seg = _u.split("/")
                                    rid = _seg[-2] if len(_seg) >= 2 else (_seg[-1] if _seg else "")
                                elif isinstance(rid, (list, dict)):
                                    rid = ""
                                else:
                                    rid = str(rid)
                                if rid:
                                    runners[rid] = RunnerInfo(
                                        runner_id=rid,
                                        url=runner_raw.get("runner_url", "") or runner_raw.get("url", ""),
                                        raw=runner_raw,
                                    )
                        else:
                            logger.warning("Discovery endpoint returned HTTP %d", resp.status)
                except Exception:
                    logger.exception("Discovery request failed")

                self._runners = runners
                logger.info("Discovered %d runners", len(self._runners))
                return list(self._runners.values())
        except Exception:
            logger.exception("Discovery failed")
            return []

    @staticmethod
    def _parse_discovery(data):
        """Normalize go-livepeer discovery output into a flat runner list.

        go-livepeer returns ``[{address, runners: [ {url, gpu, app, mode, ...} ]}]``.
        The legacy mock returned a flat ``[{runner_id, runner_url, ...}]`` list.
        We emit the flat shape with ``runner_url``/``url`` from the nested
        ``runners`` list so the rest of the client (and UI) sees a uniform view.
        """
        if not isinstance(data, list):
            return []
        flat = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            if "runners" in entry and isinstance(entry["runners"], list):
                for r in entry["runners"]:
                    item = dict(r)
                    item.setdefault("runner_url", item.get("url", ""))
                    flat.append(item)
            elif "runner_id" in entry or "runner_url" in entry:
                flat.append(entry)
        return flat

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
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=self._ssl_ctx())) as session:
            async with session.post(
                runner_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout_s),
                headers=self._auth_headers(),
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
