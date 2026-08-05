# LTX Desktop Runner — GPU Inference Service (node-operator deployable image)
# ============================================================================
# Builds the LTX-Desktop GPU inference runner into a self-contained Docker
# image that node operators can launch. The runner package lives in the
# LTX-Desktop repo (`./runner/`) and reuses the same LTX-2 pip pins the desktop
# backend uses.
#
# CUDA 12.8 wheels cover BOTH target architectures, so this single image runs
# on RTX 4090 (Ada, SM89) and RTX 5090 / RTX PRO 6000 (Blackwell, SM120). The
# runner picks its VRAM mode (streaming vs full-resident) at startup.
#
# Usage (from the LTX-Desktop repo root):
#   docker build -f docker/runner.Dockerfile -t ltx-desktop-runner .
#
#   docker run --gpus all -d -p 8991:8991 \
#     -e ORCHESTRATOR_URL=https://orch:8935 \
#     -e ORCHESTRATOR_SECRET=secret \
#     -e RUNNER_URL=http://<host>:8991 \
#     -e MODEL_CHECKPOINT=/models/checkpoint \
#     -e TEXT_ENCODER_ROOT=/models/gemma \
#     --mount type=bind,source=/models,target=/models \
#     ltx-desktop-runner

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps (ffmpeg for video concat in extend/retake; curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
    git curl ca-certificates ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python venv
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Core Python deps (aiohttp, Pillow, numpy, av, requests, huggingface_hub)
COPY runner/requirements.txt /tmp/runner-requirements.txt
RUN pip install --no-cache-dir -r /tmp/runner-requirements.txt

# livepeer-gateway from the public repo, pinned to the commit that ships
# live_runner.register_runner + the live-runner header expectations the runner
# relies on (`ja/live-runner` branch HEAD). Pinning keeps the image
# reproducible; no host-side sibling repo is needed.
RUN pip install --no-cache-dir \
    "git+https://github.com/livepeer/livepeer-python-gateway.git@2f29404"

# PyTorch with CUDA 12.8 (covers Ada SM89 + Blackwell SM120)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# LTX core + pipelines from the LTX-2 repo (same rev the desktop backend pins
# in backend/requirements-remote.txt). Provides the ltx_core / ltx_pipelines
# namespaces the runner imports.
RUN pip install --no-cache-dir \
    "git+https://github.com/Lightricks/LTX-2.git@9377758131b1ffde4b7f766804590a6617bf2ab9#subdirectory=packages/ltx-core" \
    "git+https://github.com/Lightricks/LTX-2.git@9377758131b1ffde4b7f766804590a6617bf2ab9#subdirectory=packages/ltx-pipelines"

# Diffusers for image generation (Z-Image-Turbo)
RUN pip install --no-cache-dir diffusers accelerate

# Create non-root user
RUN useradd -m runner

# Runner source
WORKDIR /app
COPY runner/ ./runner/

# Models mount point
RUN mkdir -p /models && chown runner:runner /models

USER runner

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8991/ltx-desktop/v1/health || exit 1

EXPOSE 8991

ENTRYPOINT ["python", "-m", "runner"]
