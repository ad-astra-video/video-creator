"""Restyle API orchestration handler.

Identity-preserving video restylization (id-v2v): source video + stylized first
frame image + prompt -> restyled video. Served through a compatible local GPU
when present (in-process id-v2v pipeline), else remotely via the live-runner
edge (which routes to the id-v2v worker).

Mirrors the local LTX flow: decide local mode from the runtime policy gate
(CUDA + enough VRAM), serve locally if active, else fall back to the remote
Livepeer restyle route.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import uuid
from typing import Any
from pathlib import Path
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)

from api_types import (
    RestyleRequest,
    RestyleResponse,
    RestyleVideoResponse,
)
from _routes._errors import HTTPError
from handlers.base import StateHandlerBase
from handlers.generation_handler import GenerationHandler
from handlers.video_resolution import validate_source_video_path
from runtime_config.runtime_config import RuntimeConfig
from runtime_config.runtime_policy import decide_local_idv2v_mode
from services.gpu_info.gpu_info_impl import GpuInfoImpl
from services.idv2v_pipeline import LocalIdV2vPipeline
from services.idv2v_pipeline.local_idv2v_pipeline import LocalIdV2vPipelineUnavailable
from services.interfaces import GpuInfo
from state.app_state_types import AppState


class RestyleHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        config: RuntimeConfig,
        generation_handler: GenerationHandler,
        gpu_info: "GpuInfo | None" = None,
        local_idv2v: "LocalIdV2vPipeline | None" = None,
    ) -> None:
        super().__init__(state, lock, config)
        self._generation = generation_handler
        self._gpu_info = gpu_info or GpuInfoImpl()
        self._local_idv2v = local_idv2v or LocalIdV2vPipeline()

    def run(self, req: RestyleRequest) -> RestyleResponse:
        video_path = req.video_path
        stylized_image_path = req.stylized_image_path
        prompt = req.prompt

        if not stylized_image_path:
            raise HTTPError(400, "Missing stylized_image_path parameter")

        video_file = validate_source_video_path(video_path)
        image_file = Path(stylized_image_path).resolve()
        if not image_file.is_file():
            raise HTTPError(400, f"Stylized image file not found: {stylized_image_path}")

        # Local-first: compatible GPU present + local id-v2v stack usable.
        if self._local_capable():
            try:
                return self._run_local_restyle(
                    video_file=video_file,
                    image_file=image_file,
                    prompt=prompt,
                    max_frames=req.max_frames,
                    inference_steps=req.inference_steps,
                    cfg_scale=req.cfg_scale,
                )
            except LocalIdV2vPipelineUnavailable:
                logger.info("Local id-v2v unavailable at runtime — falling back to remote")

        # Remote fallback (live-runner -> idv2v-worker).
        return self._run_livepeer_restyle(
            video_file=video_file,
            image_file=image_file,
            prompt=prompt,
            max_frames=req.max_frames,
            inference_steps=req.inference_steps,
            cfg_scale=req.cfg_scale,
        )

    def _local_capable(self) -> bool:
        """True when a compatible local GPU is present AND the engine can import."""
        try:
            mode = decide_local_idv2v_mode(
                self._gpu_info.get_cuda_available(),
                self._gpu_info.get_vram_total_gb(),
            )
        except Exception:
            return False
        if mode != "available":
            return False
        return self._local_idv2v.available()

    def _run_local_restyle(
        self,
        *,
        video_file: Path,
        image_file: Path,
        prompt: str,
        max_frames: int,
        inference_steps: int,
        cfg_scale: float,
    ) -> RestyleResponse:
        body: dict[str, Any] = {
            "source_video": base64.b64encode(video_file.read_bytes()).decode(),
            "stylized_first_frame": base64.b64encode(image_file.read_bytes()).decode(),
            "prompt": prompt,
            "max_frames": max_frames,
            "inference_steps": inference_steps,
            "cfg_scale": cfg_scale,
        }
        with self._generation.reserved_generation_start():
            generation_id = uuid.uuid4().hex[:8]
            try:
                self._generation.start_generation(generation_id)
                self._generation.update_progress("loading_model", 5, 0, 1)
                self._generation.update_progress("inference", 30, 0, 1)

                result: dict[str, Any] = asyncio.run(self._local_idv2v.restyle(body))

                self._generation.update_progress("complete", 100, 1, 1)
                output_path = self._save_local_result(result.get("output_video", ""))
                self._generation.complete_generation(output_path)
                return RestyleVideoResponse(status="complete", video_path=output_path)
            except HTTPError:
                raise
            except Exception as exc:
                self._generation.fail_generation(str(exc))
                raise HTTPError(500, f"Local restyle failed: {exc}") from exc

    def _save_local_result(self, b64: str) -> str:
        if not b64:
            raise HTTPError(500, "Local restyle returned no output")
        path = self.config.outputs_dir / f"restyle_{uuid.uuid4().hex[:8]}.mp4"
        path.write_bytes(base64.b64decode(b64))
        return str(path)

    def _run_livepeer_restyle(
        self,
        *,
        video_file: Path,
        image_file: Path,
        prompt: str,
        max_frames: int,
        inference_steps: int,
        cfg_scale: float,
    ) -> RestyleResponse:
        client = getattr(self.state, "_livepeer_client", None)
        if client is None:
            raise HTTPError(503, "Remote inference not initialized — check Discovery URL")

        settings = self.state.app_settings
        runner = client.get_runner_for_with_recovery(
            settings.livepeer_selected_runner_id,
            settings.livepeer_excluded_runner_ids,
            capability="restyle",
        )
        if runner is None:
            raise HTTPError(422, "No restyle-capable remote runner available")

        video_b64 = base64.b64encode(video_file.read_bytes()).decode()
        image_b64 = base64.b64encode(image_file.read_bytes()).decode()

        runner_payload = {
            "source_video": video_b64,
            "stylized_first_frame": image_b64,
            "prompt": prompt,
            "max_frames": max_frames,
            "inference_steps": inference_steps,
            "cfg_scale": cfg_scale,
        }

        with self._generation.reserved_generation_start():
            generation_id = uuid.uuid4().hex[:8]
            try:
                self._generation.start_api_generation(generation_id)
                self._generation.update_progress("sending_to_remote", 10, None, None)

                try:
                    result = asyncio.run(
                        client.call(
                            runner,
                            "/video-creator/v1/restyle",
                            runner_payload,
                            timeout_s=1200,
                        )
                    )
                except Exception as exc:
                    self._generation.fail_generation(str(exc))
                    raise HTTPError(500, f"Remote runner failed: {exc}") from exc

                self._generation.update_progress("downloading_result", 90, None, None)

                if "output_video" in result:
                    output_path = client.save_result(result["output_video"], "video/mp4")
                else:
                    raise HTTPError(500, "Runner returned unexpected response")

                self._generation.update_progress("complete", 100, None, None)
                self._generation.complete_generation(output_path)
                return RestyleVideoResponse(status="complete", video_path=output_path)
            except HTTPError:
                raise
            except Exception as exc:
                self._generation.fail_generation(str(exc))
                raise HTTPError(500, str(exc)) from exc
