"""Configuration from environment variables for the ID-V2V worker."""

import os

# Worker identity + auth (shared with the live-runner edge).
# Auto-generated if blank at startup (see server.py).
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")

# Model paths / knobs.
MODEL_CHECKPOINT = os.environ.get("MODEL_CHECKPOINT", "/models/idv2v.pth")
# HF-cache-style repo dir holding Wan2.1-I2V-14B-720P (T5/VAE/CLIP/tokenizer +
# DiT shards). diffsynth resolves files as <parent>/Wan-AI/Wan2.1-I2V-14B-720P/<pattern>,
# so local_model_path = dirname(WAN_MODEL_DIR) = /models.
WAN_MODEL_DIR = os.environ.get("WAN_MODEL_DIR", "/models/Wan-AI/Wan2.1-I2V-14B-720P")
SAM3_CKPT = os.environ.get("SAM3_CKPT", "/models/sam3")
SAM_PROMPT = os.environ.get("SAM_PROMPT", "person")
SKIP_SAM3 = os.environ.get("IDV2V_SKIP_SAM3", "0").lower() in {"1", "true", "yes"}

# GPU + runtime knobs.
GPU_DEVICE = os.environ.get("GPU_DEVICE", "cuda:0")
GPU_NAME = os.environ.get("GPU_NAME", "RTX 5090")
GPU_VRAM_GB = float(os.environ.get("GPU_VRAM_GB", "32"))
IDV2V_QUANT = os.environ.get("IDV2V_QUANT", "int8")       # int8 | none | bf16
IDV2V_OFFLOAD = os.environ.get("IDV2V_OFFLOAD", "true").lower() in {"1", "true", "yes"}
IDV2V_VRAM_BUFFER = int(os.environ.get("IDV2V_VRAM_BUFFER", "10"))

# HTTP server.
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8992"))  # distinct from LTX worker's 8991
MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", "3000000000"))  # 3GB


def worker_token() -> str:
    """Return the worker auth token, auto-generating a stable one if blank."""
    global WORKER_TOKEN
    if WORKER_TOKEN:
        return WORKER_TOKEN
    if not WORKER_TOKEN:
        WORKER_TOKEN = os.environ.setdefault("WORKER_TOKEN", _random_token())
    return WORKER_TOKEN


def _random_token() -> str:
    import random
    import string
    return "".join(random.choices(string.ascii_letters + string.digits, k=32))
