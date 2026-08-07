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

# Python 3.10 via deadsnakes — Ubuntu 24.04 base doesn't ship 3.10 in the main
# repos, but the diffsynth fork's flash_attn dep only ships cp310 wheels, so we
# match the reference build's Python.
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common ca-certificates gnupg && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev \
    git curl ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# PyTorch with CUDA 12.8 (Blackwell SM120 for the 5090). Install FIRST so the
# pinned torch is authoritative and torchao/other deps later resolve against it
# (otherwise pip's ">=2.6" pulls a PyPI torch build and torchao resolves to a
# version incompatible with it).
# Pin exact cu128 versions that ship cp310 wheels (diffsynth fork requires py3.10).
# cu128 supports Blackwell SM120 (RTX 5090). torch 2.7.0+cu128 is the first
# cu128 release with cp310 wheels; use it with matching vision/audio.
RUN pip install --no-cache-dir \
    torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# Worker deps: aiohttp, transformers, diffusers, xfuser, torchao int8, etc.
COPY runner/idv2v/requirements.txt /tmp/idv2v-requirements.txt
RUN pip install --no-cache-dir -r /tmp/idv2v-requirements.txt

# diffsynth (the Wan pipeline fork) + SAM3 come from the Eyeline-Labs/ID-V2V
# reference repo, NOT pip. Clone the pinned commit and install from source.
ARG IDV2V_REF_REPO=https://github.com/Eyeline-Labs/ID-V2V
ARG IDV2V_REF_COMMIT=HEAD
# The diffsynth fork's setup.py imports pkg_resources at build time; the isolated
# pip build env lacks it, so install setuptools into the main venv and build the
# source packages with --no-build-isolation so they see the base env.
RUN pip install --no-cache-dir "setuptools<81" \
    && git clone --depth 1 "${IDV2V_REF_REPO}" /opt/idv2v-ref \
    && cd /opt/idv2v-ref \
    && (test "${IDV2V_REF_COMMIT}" = "HEAD" || git checkout "${IDV2V_REF_COMMIT}" || true) \
    && pip install --no-cache-dir --no-build-isolation ./diffsynth_studio . \
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
