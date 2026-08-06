"""ID-V2V model loader and inference pipeline (ported from the id-v2v runner).

Loads the ID-V2V model — a Wan 2.1 I2V-14B DiT + VACE-14B ControlNet video
model (from Eyeline-Labs/ID-V2V, wrapped by the `diffsynth` fork) — on GPU.

Two knobs make this fit a 32 GB RTX 5090 instead of a 96 GB card:

* INT8 QUANTIZATION of the video model: the 14B DiT and its VACE controlnet are
  quantized in-place with torchao `int8_weight_only()` (~28 GB bf16 -> ~14 GB
  int8). T5 text-encoder + VAE + tokenizer stay bf16 but live on the CPU offload
  pool.
* CPU OFFLOAD: every ModelConfig uses offload_device="cpu", pipeline uses
  enable_vram_management(vram_buffer=10). Layers move to GPU on demand.

The worker keeps the model warm between calls; the live-runner tells it to
/evict (dropping the pipe + empty_cache) before another worker needs the GPU.

Grounded in the reference pipeline:
    c:\\dev\\id-v2v\\runner\\src\\idv2v_runner\\model.py
    Eyeline-Labs/ID-V2V  src/idv2v/inference/pipeline.py
"""

import asyncio
import gc
import logging
import os
import time

import torch

from . import config

logger = logging.getLogger("video_creator.runner.idv2v.model")

# ---------------------------------------------------------------------------
# Video processing helpers — ported verbatim from the reference pipeline.py so
# clip scheduling, center-crop and frame slicing match exactly.
# ---------------------------------------------------------------------------


def center_crop_and_resize(img, width: int, height: int):
    """Center-crop + resize a PIL RGB image to (width, height) using BICUBIC."""
    from PIL import Image
    w, h = img.size
    target_aspect = height / width
    aspect = h / w
    if (h == height) and (w == width):
        return img
    if abs(aspect - target_aspect) < 1e-6:
        return img.resize((width, height), Image.BICUBIC)
    if aspect > target_aspect:  # too tall -> resize width, crop height
        new_w, new_h = width, int(aspect * width)
        resized = img.resize((new_w, new_h), Image.BICUBIC)
    else:  # too wide -> resize height, crop width
        new_h, new_w = height, int(height / aspect)
        resized = img.resize((new_w, new_h), Image.BICUBIC)
    rw, rh = resized.size
    left, top = (rw - width) // 2, (rh - height) // 2
    return resized.crop((left, top, left + width, top + height))


def compute_clip_schedule(total_frames: int, num_frames_per_clip: int):
    """Compute (start, end) frame indices for multi-clip generation.

    Regular clips advance by stride = num_frames_per_clip - 1 (1-frame overlap);
    the last clip is anchored at the end so it always has exactly
    num_frames_per_clip frames.
    """
    if total_frames <= num_frames_per_clip:
        return [(0, total_frames)]
    clips = []
    start = 0
    stride = num_frames_per_clip - 1
    while start + num_frames_per_clip < total_frames:
        clips.append((start, start + num_frames_per_clip))
        start += stride
    clips.append((total_frames - num_frames_per_clip, total_frames))
    return clips


def slice_frames(all_frames, start: int, end: int):
    """Slice frames[start:end]; pad by repeating the last frame if needed."""
    n = len(all_frames)
    if end <= n:
        return all_frames[start:end]
    result = list(all_frames[start:n])
    result += [all_frames[-1]] * (end - n)
    return result


class ModelManager:
    """Loads ID-V2V (int8 DiT+VACE) with CPU offload for a 32 GB 5090.

    Instance-stated on purpose so the worker owns/evicts its model: construct
    one per worker (or one per repo of the process), call ``load()`` to build it,
    keep it warm, and ``evict()`` to free GPU/CPU memory when another worker
    needs the card.
    """

    def __init__(self, device: str = ""):
        self.device = device or config.GPU_DEVICE
        self._pipe = None
        self._torch_dtype = torch.bfloat16

    @property
    def device_name(self) -> str:
        return self.device

    # -- lifecycle ----------------------------------------------------------
    async def load(self):
        """Build the WanVideoPipeline, quantize the video model to int8, offload."""
        if self._pipe is not None:
            return
        logger.info(
            "Loading ID-V2V on %s (quant=%s offload=%s vram_buffer=%d) ...",
            self.device, config.IDV2V_QUANT, config.IDV2V_OFFLOAD,
            config.IDV2V_VRAM_BUFFER,
        )
        start = time.time()

        try:
            import torchao
            from torchao.quantization import int8_weight_only, quantize_
            self._torchao = torchao  # keep a handle; used below
        except Exception as exc:  # pragma: no cover - quantization optional at runtime
            logger.warning("torchao not importable (%s); int8 quant disabled", exc)
            self._torchao = None

        # Load all pieces in a thread executor so we don't block the event loop
        # during the multi-minute model build.
        self._pipe = await asyncio.to_thread(self._build_pipeline)

        if config.IDV2V_QUANT == "int8" and self._torchao is not None:
            self._quantize_int8()

        self._enable_offload()

        logger.info(
            "Model loaded in %.1fs (idv2v.pth + Wan2.1 int8 + CPU offload)",
            time.time() - start,
        )

    def evict(self) -> None:
        """Drop the pipeline and free GPU/CPU memory. Safe to call when unloaded."""
        if self._pipe is not None:
            logger.info("Evicting ID-V2V pipeline (freeing GPU/CPU memory)")
        self._pipe = None
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

    def _build_pipeline(self):
        """Construct the diffsynth WanVideoPipeline (runs in a worker thread)."""
        from diffsynth.pipelines.wan_video_new_multiVace_svi import (
            ModelConfig,
            WanVideoPipeline,
        )

        # DiT + VACE come from the finetuned idv2v.pth; we skip the ~56 GB of
        # base I2V + VACE-14B weights (same fast path as the reference script).
        model_configs = [
            ModelConfig(
                model_id="Wan-AI/Wan2.1-T2V-14B",
                origin_file_pattern="google/*",           # T5 tokenizer
                offload_device="cpu",
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.1-I2V-14B-720P",
                origin_file_pattern=f"{config.WAN_MODEL_DIR}/*",
                offload_device="cpu",
            ),
        ]

        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=self._torch_dtype,
            device=self.device,
            use_usp=False,                 # single-GPU 5090 -> no USP sequence parallel
            model_configs=model_configs,
            tokenizer_config=ModelConfig(
                model_id="Wan-AI/Wan2.1-T2V-14B",
                origin_file_pattern="google/*",
                offload_device="cpu",
            ),
            local_model_path=config.WAN_MODEL_DIR,
            checkpoint_path=None,
            skip_download=True,
            redirect_common_files=False,
        )

        # Fast-load the finetuned DiT + VACE from idv2v.pth (memory-mapped).
        if os.path.isfile(config.MODEL_CHECKPOINT):
            self._load_finetuned_dit_vace(pipe, config.MODEL_CHECKPOINT)
        else:
            raise FileNotFoundError(
                f"idv2v checkpoint not found: {config.MODEL_CHECKPOINT}. "
                "Run download_models.sh first."
            )

        assert pipe.dit is not None and pipe.dit.has_image_input, "Expected I2V DiT"

        # Single-GPU (no USP): enable per-layer vram offloading.
        pipe.enable_vram_management(vram_buffer=config.IDV2V_VRAM_BUFFER)

        return pipe

    def _load_finetuned_dit_vace(self, pipe, checkpoint_path, torch_dtype=torch.bfloat16):
        """Instantiate empty DiT + VACE and load finetuned weights (mmap)."""
        from diffsynth import load_state_dict
        from diffsynth.models.utils import init_weights_on_device
        from diffsynth.models.wan_video_dit import WanModel
        from diffsynth.models.wan_video_vace import VaceWanModel

        I2V_14B_DIT_CONFIG = {
            "has_image_input": True, "patch_size": [1, 2, 2], "in_dim": 36,
            "dim": 5120, "ffn_dim": 13824, "freq_dim": 256, "text_dim": 4096,
            "out_dim": 16, "num_heads": 40, "num_layers": 40, "eps": 1e-6,
        }
        VACE_14B_CONFIG = {
            "vace_layers": (0, 5, 10, 15, 20, 25, 30, 35), "vace_in_dim": 96,
            "patch_size": (1, 2, 2), "has_image_input": False, "dim": 5120,
            "num_heads": 40, "ffn_dim": 13824, "eps": 1e-6,
        }

        logger.info("Instantiating empty DiT + VACE (int8 target) ...")
        with init_weights_on_device():
            pipe.dit = WanModel(**I2V_14B_DIT_CONFIG)
            pipe.vace = VaceWanModel(**VACE_14B_CONFIG)

        logger.info("Loading checkpoint %s ...", checkpoint_path)
        state_dict = load_state_dict(checkpoint_path)
        state_dict_vace = {k: v for k, v in state_dict.items() if "vace" in k}
        state_dict_dit = {k: v for k, v in state_dict.items() if "vace" not in k}
        pipe.dit.load_state_dict(state_dict_dit, assign=True)
        pipe.vace.load_state_dict(state_dict_vace, assign=True)
        pipe.dit = pipe.dit.to(dtype=torch_dtype)
        pipe.vace = pipe.vace.to(dtype=torch_dtype)
        logger.info(
            "Loaded DiT (%d params) + VACE (%d params)",
            len(state_dict_dit), len(state_dict_vace),
        )

    def _quantize_int8(self):
        """Quantize the video model (DiT + VACE) to int8 weights in-place."""
        if not hasattr(self, "_torchao") or self._torchao is None:
            return
        from torchao.quantization import int8_weight_only, quantize_

        q = int8_weight_only()
        logger.info("Quantizing video model (DiT + VACE) to int8 ...")
        t0 = time.time()
        quantize_(self._pipe.dit, q)
        if self._pipe.vace is not None:
            quantize_(self._pipe.vace, q)
        torch.cuda.empty_cache()
        logger.info("int8 quantization done in %.1fs", time.time() - t0)

    def _enable_offload(self):
        """Ensure pipeline is on the offloaded (CPU) execution path."""
        if config.IDV2V_OFFLOAD:
            hook = getattr(self._pipe, "enable_model_cpu_offload", None)
            if callable(hook):
                try:
                    hook()
                except Exception as exc:  # pragma: no cover - optional
                    logger.warning(
                        "model_cpu_offload hook failed (%s); using vram_management", exc
                    )

    # -- inference ----------------------------------------------------------
    def infer(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        input_image,                    # PIL stylized first frame (I2V anchor)
        condition_videos,               # list of frame-lists (VACE control videos)
        keyframes=None,                 # list of (frame_index:int, image:PIL)
        width: int = 1280,
        height: int = 720,
        num_frames: int = 81,           # frames per clip
        max_frames: int | None = None,  # cap total output frames
        num_inference_steps: int = 30,
        cfg_scale: float = 5.0,
        vace_scale: float = 1.0,
        ref_pad_num: int = -1,
        seed: int = 123,
    ):
        """Run ID-V2V clip-by-clip generation (ported from the reference).

        Returns: list of PIL RGB frames (the combined video).
        """
        pipe = self._pipe
        keyframes = keyframes or []   # [(index, PIL image), ...]
        from PIL import Image

        if max_frames is not None:
            condition_videos = [c[:max_frames] for c in condition_videos]
        total_frames = len(condition_videos[0])

        white = Image.new("RGB", (width, height), (255, 255, 255))
        black = Image.new("RGB", (width, height), (0, 0, 0))
        kf_set = {idx for idx, _ in keyframes}
        full_mask = [black if i in kf_set else white for i in range(total_frames)]

        for idx, kf_img in keyframes:
            resized = center_crop_and_resize(kf_img, width, height)
            for c in condition_videos:
                c[idx] = resized

        clip_schedule = compute_clip_schedule(total_frames, num_frames)
        logger.info("Clip schedule (%d clips): %s", len(clip_schedule), clip_schedule)

        all_clips = []
        current_input_image = input_image

        with torch.no_grad():
            for clip_idx, (frame_start, frame_end) in enumerate(clip_schedule):
                clip_seed = seed
                if clip_idx > 0:
                    splice_idx = clip_schedule[clip_idx][0] - clip_schedule[clip_idx - 1][0]
                    current_input_image = all_clips[-1][splice_idx]

                clip_end = frame_start + num_frames
                clip_conditions = [
                    slice_frames(c, frame_start, clip_end) for c in condition_videos
                ]
                clip_mask_single = slice_frames(full_mask, frame_start, clip_end)
                clip_mask = [clip_mask_single] * len(condition_videos)

                logger.info("Clip %d/%d frames=[%d,%d) seed=%d",
                            clip_idx + 1, len(clip_schedule), frame_start, frame_end,
                            clip_seed)
                generated = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    input_image=current_input_image,
                    random_ref_frame=input_image,
                    ref_pad_num=ref_pad_num,
                    vace_video=clip_conditions,
                    vace_video_mask=clip_mask,
                    seed=clip_seed,
                    num_inference_steps=num_inference_steps,
                    use_multi_control_vace=True,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    cfg_scale=cfg_scale,
                    tiled=False,
                    vace_scale=vace_scale,
                )
                all_clips.append(generated)

        if len(clip_schedule) == 1:
            combined = all_clips[0]
            if total_frames < num_frames:
                combined = combined[:total_frames]
        else:
            combined = list(all_clips[0])
            for i in range(1, len(clip_schedule)):
                overlap = clip_schedule[i - 1][1] - clip_schedule[i][0]
                combined = combined[:-overlap] + list(all_clips[i])

        logger.info("Stitched %d clips into %d frames",
                    len(clip_schedule), len(combined))
        return combined

    @property
    def is_ready(self) -> bool:
        return self._pipe is not None

    @property
    def precision(self) -> str:
        return config.IDV2V_QUANT


def health_check(model) -> dict:
    """Health payload — model status."""
    return {
        "status": "ok" if model.is_ready else "loading",
        "device": model.device,
        "model_loaded": model.is_ready,
        "precision": model.precision,
        "offload": config.IDV2V_OFFLOAD,
    }

