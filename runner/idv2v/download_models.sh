#!/bin/bash
# Download ID-V2V model weights and dependencies for the worker container.
# Runs on first boot inside the Docker container.
#
# Targets a single RTX 5090 (32 GB): the video model (Wan 2.1 I2V-14B DiT +
# VACE) is loaded int8-quantized with CPU offload in model.py, so the full
# footprint fits in 32 GB of VRAM.
#
# Adapted from the standalone id-v2v runner scripts/download_models.sh.
# MODEL_DIR defaults to /models (shared bind-mount with the live-runner host).

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/models}"
mkdir -p "$MODEL_DIR"

echo ">>> Downloading ID-V2V model files to $MODEL_DIR (int8 / offload target: RTX 5090)"

HF_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"
if [ -z "$HF_TOKEN" ]; then
    echo "WARN: HUGGING_FACE_HUB_TOKEN not set — some models may be gated"
fi

# ID-V2V finetuned checkpoint (DiT + VACE weights). Loaded + quantized to int8
# in-process (torchao int8_weight_only), so the on-disk bf16 checkpoint is fine.
echo ">>> Downloading idv2v.pth (DiT + VACE finetuned checkpoint)"
huggingface-cli download --token "$HF_TOKEN" \
    Eyeline-Labs/ID-V2V idv2v.pth \
    --local-dir "$MODEL_DIR" || true

# SAM3 segmentation model (foreground-on-gray preprocessing)
echo ">>> Downloading SAM3"
huggingface-cli download --token "$HF_TOKEN" sam3-org/sam3 \
    --local-dir "$MODEL_DIR/sam3" || true

# Wan 2.1 I2V-14B-720P base model (T5 + VAE + CLIP tokenizer used by pipeline).
echo ">>> Downloading Wan2.1 I2V-14B-720P (T5 + VAE + tokenizer + CLIP)"
huggingface-cli download --token "$HF_TOKEN" \
    Wan-AI/Wan2.1-I2V-14B-720P \
    --local-dir "$MODEL_DIR/wan" || true

echo ">>> Model download complete"
ls -lh "$MODEL_DIR"
