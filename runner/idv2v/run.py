"""Inference pipeline for the ID-V2V worker — decode inputs, run model, encode output.

Drives the ID-V2V model loaded by `model.ModelManager` (int8-quantized Wan2.1
DiT + VACE with CPU offload, suitable for a 32 GB RTX 5090).

Unlike the standalone id-v2v runner, the model is passed in explicitly (a
ModelManager instance owned by the worker) so it can be evicted/kept-warm by the
live-runner swap policy.

Ported from c:\\dev\\id-v2v\\runner\\src\\idv2v_runner\\run.py.
"""

import asyncio
import base64
import io
import logging
import os
import tempfile
import time

import numpy as np

from . import config

logger = logging.getLogger("video_creator.runner.idv2v.run")

DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
    "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
    "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


async def process_job(model, body: dict) -> dict:
    """Process a restylization job request.

    Args:
        model: a ready `ModelManager` instance (owned/evicted by the worker).
        body: parsed JSON containing source_video, stylized_first_frame, prompt,
              and parameters.

    Returns:
        dict matching the IdV2VResponse schema.
    """
    if not model.is_ready:
        raise RuntimeError("Model is not loaded yet — retry in a moment")

    start = time.time()

    prompt = body.get("prompt", "")
    max_frames = int(body.get("max_frames", 81))
    inference_steps = int(body.get("inference_steps", 30))
    cfg_scale = float(body.get("cfg_scale", 5.0))
    vace_scale = float(body.get("vace_scale", 1.0))
    width = int(body.get("width", 1280))
    height = int(body.get("height", 720))
    num_frames_per_clip = int(body.get("num_frames_per_clip", 81))
    seed = int(body.get("seed", 123))
    keyframes = body.get("keyframes", [])   # [{"frame": N, "image": "<b64>"}, ...]

    logger.info(
        "Processing restyle job: prompt=%r, max_frames=%d, steps=%d, cfg=%.1f",
        prompt, max_frames, inference_steps, cfg_scale,
    )

    source_b64 = body.get("source_video", "")
    stylized_b64 = body.get("stylized_first_frame", "")

    if not source_b64 or not stylized_b64:
        raise ValueError("source_video and stylized_first_frame are required")

    keyframe_specs = []
    for kf in keyframes:
        idx = kf.get("frame")
        img = kf.get("image")
        if not isinstance(idx, int) or idx < 1:
            raise ValueError(f"keyframe 'frame' must be int >= 1, got {idx!r}")
        if not img:
            raise ValueError(f"keyframe {idx} is missing 'image'")
        keyframe_specs.append((idx, img))

    result = await asyncio.to_thread(
        _run_pipeline,
        model, source_b64, stylized_b64, prompt,
        max_frames, inference_steps, cfg_scale, vace_scale,
        num_frames_per_clip, seed, keyframe_specs, width, height,
    )

    elapsed = time.time() - start
    logger.info("Restyle job complete in %.1fs", elapsed)

    return {
        "output_video": result["b64"],
        "frames_generated": result["frames"],
        "resolution": f"{width}x{height}",
        "processing_time_sec": round(elapsed, 2),
    }


def _run_pipeline(model, source_b64, stylized_b64, prompt, max_frames,
                  inference_steps, cfg_scale, vace_scale, num_frames_per_clip,
                  seed, keyframe_specs, width, height) -> dict:
    """Synchronous pipeline body (runs in a worker thread)."""
    tmpdir = tempfile.mkdtemp(prefix="idv2v_")

    source_path = os.path.join(tmpdir, "source.mp4")
    stylized_path = os.path.join(tmpdir, "stylized_first_frame.png")
    _write_b64(source_b64, source_path)
    _write_b64(stylized_b64, stylized_path)

    cond_frames = _segment_foreground(source_path, stylized_path, tmpdir, width, height)

    decoded_kf = []
    for idx, img_b64 in keyframe_specs:
        decoded_kf.append((idx, _decode_image(img_b64)))

    input_image = _load_anchor(stylized_path, width, height)

    frames = model.infer(
        prompt=prompt,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        input_image=input_image,
        condition_videos=[cond_frames],
        keyframes=decoded_kf,
        width=width, height=height,
        num_frames=num_frames_per_clip,
        max_frames=max_frames,
        num_inference_steps=inference_steps,
        cfg_scale=cfg_scale,
        vace_scale=vace_scale,
        seed=seed,
    )

    b64 = _encode_frames_mp4(frames)
    return {"b64": b64, "frames": len(frames)}


def _write_b64(b64str: str, path: str):
    data = base64.b64decode(b64str)
    with open(path, "wb") as f:
        f.write(data)


def _decode_image(b64str: str):
    """Decode a base64 image into a PIL RGB image."""
    from PIL import Image
    data = base64.b64decode(b64str)
    return Image.open(io.BytesIO(data)).convert("RGB")


def _load_anchor(path: str, width: int, height: int):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return img.resize((width, height))


def _read_video_frames_cv2(source_path, width, height):
    """Read a video with OpenCV into center-cropped/resized PIL RGB frames."""
    from .model import center_crop_and_resize
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(source_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
    cap.release()
    return [center_crop_and_resize(f, width, height) for f in frames]


def _segment_foreground(source_path, stylized_path, tmpdir, width, height):
    """SAM3 foreground-on-gray segmentation (VACE condition video).

    Reproduces the reference `scripts/preprocess.sh` by invoking the two
    upstream CLI modules:
      1. python -m idv2v.preprocess.sam3        (segmentation + mask cleanup)
      2. python -m idv2v.preprocess.orig_pixel  (foreground-on-gray pixels)

    Set IDV2V_SKIP_SAM3=1 to bypass and use raw source frames (relighting path).
    """
    if config.SKIP_SAM3:
        logger.info("IDV2V_SKIP_SAM3=1 — using raw source frames as condition")
        return _read_video_frames_cv2(source_path, width, height)

    try:
        import idv2v  # noqa: F401  (ensures the reference package is installed)
    except ImportError as exc:
        raise RuntimeError(
            "SAM3 foreground segmentation requires the Eyeline ID-V2V reference "
            f"package, but `import idv2v` failed: {exc}. Install the reference repo "
            "(diffsynth_studio + src) into the environment, or set IDV2V_SKIP_SAM3=1 "
            "to serve the raw-frame (relighting) path instead."
        ) from exc

    preproc_dir = os.path.join(tmpdir, "preprocessing")
    os.makedirs(preproc_dir, exist_ok=True)
    cond_video = os.path.join(preproc_dir, "orig_pixel.mp4")

    gpu_id = "0"
    model_dev = os.environ.get("GPU_DEVICE", config.GPU_DEVICE)
    if model_dev.startswith("cuda:"):
        gpu_id = model_dev.split(":", 1)[1] or "0"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_id)

    import subprocess
    import sys

    step1 = subprocess.run(
        [sys.executable, "-m", "idv2v.preprocess.sam3",
         "--video_path", source_path,
         "--sam_prompt", os.environ.get("SAM_PROMPT", config.SAM_PROMPT),
         "--output_dir", preproc_dir,
         "--model_path", config.SAM3_CKPT,
         "--joint_mask_post_proc"],
        capture_output=True, text=True, env=env,
    )
    if step1.returncode != 0:
        raise RuntimeError("SAM3 segmentation failed:\n" + (step1.stderr or step1.stdout))

    step2 = subprocess.run(
        [sys.executable, "-m", "idv2v.preprocess.orig_pixel",
         "--video_path", source_path,
         "--mask_folder", preproc_dir,
         "--mask_image_file_name", "sam3Mask_id_all.png",
         "--result_save_path", cond_video],
        capture_output=True, text=True, env=env,
    )
    if step2.returncode != 0:
        raise RuntimeError("orig_pixel (foreground-on-gray) failed:\n" + (step2.stderr or step2.stdout))

    if not os.path.isfile(cond_video):
        raise RuntimeError(f"orig_pixel did not produce {cond_video}")

    logger.info("SAM3 condition written to %s", cond_video)
    return _read_video_frames_cv2(cond_video, width, height)


def _encode_frames_mp4(frames) -> str:
    """Encode a list of PIL frames to an MP4 (H.264) and return base64."""
    import imageio.v2 as imageio

    arr = np.stack([np.asarray(f.convert("RGB")) for f in frames], axis=0)
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        imageio.mimwrite(path, arr, format="FFMPEG", fps=24, codec="libx264", quality=8)
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
