# ID-V2V Worker — serves /v1/restyle (identity-preserving video restylization)
# behind the live-runner. Heavy image: torch + Wan2.1 I2V-14B DiT + VACE
# (diffsynth fork) + SAM3, int8-quantized with CPU offload, tuned for a 32GB
# RTX 5090. Does NOT register with the Orchestrator (the live-runner's job);
# it only serves the internal /health /load /evict /v1/restyle surface.
#
# Build (from the video-creator repo root):
#   docker build -f docker/idv2v-worker.Dockerfile -t video-creator-idv2v-worker

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
    git curl ca-certificates ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Worker deps: aiohttp, torch/CUDA, transformers, diffusers, xfuser, torchao int8, etc.
COPY runner/idv2v/requirements.txt /tmp/idv2v-requirements.txt
RUN pip install --no-cache-dir -r /tmp/idv2v-requirements.txt

# PyTorch with CUDA 12.8 (Blackwell SM120 for the 5090)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# diffsynth (the Wan pipeline fork) + SAM3 come from the Eyeline-Labs/ID-V2V
# reference repo, NOT pip. Clone the pinned commit and install from source.
ARG IDV2V_REF_REPO=https://github.com/Eyeline-Labs/ID-V2V
ARG IDV2V_REF_COMMIT=HEAD
RUN git clone --depth 1 "${IDV2V_REF_REPO}" /opt/idv2v-ref \
    && cd /opt/idv2v-ref \
    && (test "${IDV2V_REF_COMMIT}" = "HEAD" || git checkout "${IDV2V_REF_COMMIT}" || true) \
    && pip install --no-cache-dir ./diffsynth_studio . \
    && rm -rf /opt/idv2v-ref/.git

# Worker source
RUN useradd -m runneruser
WORKDIR /app
COPY runner/ ./runner/
RUN mkdir -p /models && chown runneruser:runneruser /models
USER runneruser

# Worker liveness probe (open, no token).
HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8992/health || exit 1

EXPOSE 8992
ENTRYPOINT ["python", "-m", "runner.idv2v"]
