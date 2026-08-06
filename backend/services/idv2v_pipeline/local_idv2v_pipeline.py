# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportMissingTypeArgument=false
"""Local in-process ID-V2V restyle pipeline.

Wraps the portable idv2v engine (``runner.idv2v.model.ModelManager`` +
``runner.idv2v.run``) as a minimal in-process service the desktop calls when a
compatible GPU is present. Lazy-imports the heavy stack so a backend without the
idv2v deps installed still imports cleanly and the caller falls back to remote.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class LocalIdV2vPipelineUnavailable(RuntimeError):
    """Raised when the local id-v2v stack can't be loaded (deps missing)."""


class LocalIdV2vPipeline:
    """In-process id-v2v restyle service (singleton-ish, one model on the GPU).

    The engine is instance-stated (ownable/evictable): load() builds the model,
    restyle() runs one job, evict() frees VRAM when the LTX local pipeline
    needs the card. Mirrors the remote worker's ModelManager lifecycle but runs
    in this process.
    """

    def __init__(self, repo_root: Path | None = None) -> None:
        self._repo_root = repo_root
        self._model = None
        self._lock = asyncio.Lock()
        self._unavailable_reason: str | None = None

    # -- availability ------------------------------------------------------
    def available(self) -> bool:
        """True if this process has the id-v2v stack + could load the model."""
        return self._load_engine() is not None

    def _load_engine(self):
        """Return the runner.idv2v module (or None if unavailable), caching."""
        if self._unavailable_reason is not None:
            return None
        try:
            import runner.idv2v as _pkg
            _repo = self._repo_root or Path(__file__).resolve().parents[3]  # video-creator/
            # Ensure `runner` is importable if the backend didn't already add it.
            if _str_not_in_path(str(_repo)):
                sys.path.insert(0, str(_repo))
                import importlib
                _pkg = importlib.reload(_pkg)
            return _pkg
        except Exception as exc:  # noqa: BLE001 - any import failure = unavailable
            self._unavailable_reason = str(exc)
            logger.debug("Local id-v2v engine unavailable: %s", exc)
            return None

    async def ensure_loaded(self) -> None:
        """Build the ModelManager + load the model (idempotent, serialized)."""
        pkg = self._load_engine()
        if pkg is None:
            raise LocalIdV2vPipelineUnavailable(
                self._unavailable_reason or "local id-v2v engine not importable"
            )
        async with self._lock:
            if self._model is None:
                from runner.idv2v.model import ModelManager
                from runner.idv2v import config as idv2v_config
                self._model = ModelManager(device=idv2v_config.GPU_DEVICE)
                await self._model.load()
                logger.info("Local id-v2v model loaded")

    async def restyle(self, body: dict[str, Any]) -> dict[str, Any]:
        """Run one restyle job (base64 in -> base64 out), mirrors worker process_job."""
        pkg = self._load_engine()
        if pkg is None:
            raise LocalIdV2vPipelineUnavailable(
                self._unavailable_reason or "local id-v2v engine not importable"
            )
        await self.ensure_loaded()
        from runner.idv2v import run as run_mod
        # Even under the asyncio lock, model.infer is blocking GPU work — hand it
        # to a thread so the event loop stays responsive (same as the worker).
        return await asyncio.to_thread(run_mod.process_job, self._model, body)

    def evict(self) -> None:
        """Drop the model + free GPU memory (call before LTX local pipeline loads)."""
        if self._model is not None:
            self._model.evict()
            self._model = None


def _str_not_in_path(value: str) -> bool:
    norm = str(Path(value))
    for p in sys.path:
        if Path(p) == Path(norm):
            return False
    return True
