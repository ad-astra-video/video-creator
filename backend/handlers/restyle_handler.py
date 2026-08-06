"""Restyle API orchestration handler.

Identity-preserving video restylization (id-v2v): source video + stylized first
frame image + prompt -> restyled video. Served remotely through the live-runner
edge (which routes to the id-v2v worker). A compatible local GPU path is added
in Task 8 (mirroring the local LTX flow); until then this is remote-only.

Mirrors RetakeHandler: requires the Livepeer client, picks a restyle-capable
runner (preferring a warm id-v2v model), base64-encodes the inputs, calls
/video-creator/v1/restyle, and saves the returned video.
"""

from __future__ import annotations

import asyncio
import base64
import uuid
from pathlib import Path
from threading import RLock

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
from state.app_state_types import AppState


class RestyleHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        config: RuntimeConfig,
        generation_handler: GenerationHandler,
    ) -> None:
        super().__init__(state, lock, config)
        self._generation = generation_handler

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

        # Remote inference via Livepeer (live-runner -> idv2v-worker).
        return self._run_livepeer_restyle(
            video_file=video_file,
            image_file=image_file,
            prompt=prompt,
            max_frames=req.max_frames,
            inference_steps=req.inference_steps,
            cfg_scale=req.cfg_scale,
        )

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
