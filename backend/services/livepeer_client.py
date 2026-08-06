"""Livepeer remote inference client.

Discovers runners through orchestrators and routes generation requests.

"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import ssl
import uuid
from pathlib import Path
from typing import Any, cast

import aiohttp

logger = logging.getLogger(__name__)


class RunnerInfo:
    """Normalized runner metadata from discovery."""

    def __init__(self, runner_id: str, url: str, raw: dict[str, Any]) -> None:
        self.runner_id = runner_id
        self.url = url
        self.raw = raw
        self.gpu: Any = raw.get("gpu", {})
        self.price_info: Any = raw.get("price_info")
        self.status: str = str(raw.get("status", "ready"))
        self.label: str = str(raw.get("label", ""))
        # Capability list the runner advertises (live-runner /info or metadata).
        caps = raw.get("capabilities", [])
        if isinstance(caps, str):
            try:
                caps = json.loads(caps)
            except Exception:
                caps = []
        caps_list: list[Any] = cast(list[Any], caps if isinstance(caps, list) else [])
        self.capabilities: list[str] = [str(c) for c in caps_list]
        # Heartbeat metadata (warm model + worker up/down). The gateway sends it
        # as a JSON string; the orchestrator may pass it through as a dict.
        self.metadata: dict[str, Any] = self._parse_metadata(raw.get("metadata"))

    @staticmethod
    def _parse_metadata(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return dict(cast(dict[str, Any], value))
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return dict(cast(dict[str, Any], parsed))
                return {}
            except Exception:
                return {}
        return {}

    @property
    def warm_model(self) -> str | None:
        """The model family currently resident on this runner (from heartbeat)."""
        wm = self.metadata.get("warm_model")
        return str(wm) if wm else None

    @property
    def ltx_worker_up(self) -> bool:
        return bool(self.metadata.get("ltx_worker_up", False))

    @property
    def idv2v_worker_up(self) -> bool:
        return bool(self.metadata.get("idv2v_worker_up", False))

    def has_capability(self, capability: str) -> bool:
        """True if this runner advertises the given capability."""
        return capability in self.capabilities

    def to_dict(self) -> dict[str, Any]:
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
        self._discovery_task: asyncio.Task[Any] | None = None
        # Self-signed test orchestrator on .8 → skip TLS verification (matches
        # the livepeer gateway SDK's ssl=False).
        self._ssl = ssl.create_default_context()
        self._ssl.check_hostname = False
        self._ssl.verify_mode = ssl.CERT_NONE

    @staticmethod
    def _ssl_ctx() -> ssl.SSLContext:
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
                        params={"app": "video-creator"},
                        timeout=aiohttp.ClientTimeout(total=10),
                        headers=self._auth_headers(),
                    ) as resp:
                        if resp.status == 200:
                            data: Any = await resp.json()
                            for runner_raw in self._parse_discovery(data):
                                rid: Any = runner_raw.get("runner_id", "")
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
                    logger.error("Discovery request failed", exc_info=True)

                self._runners = runners
                logger.info("Discovered %d runners", len(self._runners))
                return list(self._runners.values())
        except Exception:
            logger.error("Discovery failed", exc_info=True)
            return []

    @staticmethod
    def _parse_discovery(data: Any) -> list[dict[str, Any]]:
        """Normalize go-livepeer discovery output into a flat runner list.

        go-livepeer returns ``[{address, runners: [ {url, gpu, app, mode, ...} ]}]``.
        The legacy mock returned a flat ``[{runner_id, runner_url, ...}]`` list.
        We emit the flat shape with ``runner_url``/``url`` from the nested
        ``runners`` list so the rest of the client (and UI) sees a uniform view.
        """
        if not isinstance(data, list):
            return []
        typed_data = cast(list[dict[str, Any]], data)
        flat: list[dict[str, Any]] = []
        for entry in typed_data:
            runners_raw: Any = entry.get("runners")
            if isinstance(runners_raw, list):
                for r in cast(list[Any], runners_raw):
                    item: dict[str, Any] = dict(r)
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

    def get_runner_with_recovery(
        self, selected_id: str, excluded_ids: list[str]
    ) -> RunnerInfo | None:
        """Pick a runner, retrying a fresh discovery pass first if none is cached.

        The background discovery loop (``periodic_discovery``) can be empty at
        startup or stale, so when no runner is known and a Discovery URL is
        configured we run one synchronous discovery pass before surfacing the
        "no available runner" error. Returns the chosen runner, or None if there
        is still none available.
        """
        runner = self.get_runner(selected_id, excluded_ids)
        if runner is None and self.discovery_url.strip():
            logger.info("No cached runner — running one discovery recovery pass")
            try:
                asyncio.run(self.discover())
            except Exception as exc:  # noqa: BLE001
                logger.warning("Discovery recovery failed: %s", exc)
            runner = self.get_runner(selected_id, excluded_ids)
        return runner

    def get_runner_for(
        self,
        selected_id: str,
        excluded_ids: list[str],
        capability: str = "restyle",
    ) -> RunnerInfo | None:
        """Pick a runner for a capability, preferring one whose model is warm.

        Selection order:
          1. explicit ``selected_id`` when it advertises the capability;
          2. the first non-excluded capability-matching runner whose heartbeat
             metadata reports its model warm;
          3. the first non-excluded capability-matching runner (fall-through).

        Falls back to the legacy ``get_runner`` spot (any runner) when no
        capability-matching runner exists, so other task types keep working.
        """
        if capability:
            def matches(r: RunnerInfo) -> bool:
                return r.has_capability(capability) and r.runner_id not in excluded_ids

            if selected_id and selected_id in self._runners:
                r = self._runners[selected_id]
                if r.has_capability(capability):
                    return r

            # Prefer warm-model capable runner (restyle warms idv2v; others warm ltx).
            want_warm = "idv2v" if capability == "restyle" else "ltx"
            for r in self._runners.values():
                if matches(r) and r.warm_model == want_warm:
                    return r
            # Fall-through: any capable runner.
            for r in self._runners.values():
                if matches(r):
                    return r
            # No capability match — fall back to any non-excluded runner.
            return self.get_runner(selected_id, excluded_ids)

        return self.get_runner(selected_id, excluded_ids)

    def get_runner_for_with_recovery(
        self,
        selected_id: str,
        excluded_ids: list[str],
        capability: str = "restyle",
    ) -> RunnerInfo | None:
        """Like ``get_runner_for``, retrying one discovery pass when none is cached."""
        runner = self.get_runner_for(selected_id, excluded_ids, capability)
        if runner is None and self.discovery_url.strip():
            logger.info("No cached %s runner — running one discovery recovery pass", capability)
            try:
                asyncio.run(self.discover())
            except Exception as exc:  # noqa: BLE001
                logger.warning("Discovery recovery failed: %s", exc)
            runner = self.get_runner_for(selected_id, excluded_ids, capability)
        return runner

    async def call(
        self, runner: RunnerInfo, endpoint: str, payload: dict[str, Any], timeout_s: float = 600.0
    ) -> dict[str, Any]:
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
