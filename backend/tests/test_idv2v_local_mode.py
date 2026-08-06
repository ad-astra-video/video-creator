"""Tests for the local id-v2v compatibility gate + handler routing."""

from __future__ import annotations

from runtime_config.runtime_policy import (
    IDV2V_LOCAL_VRAM_FLOOR_GB,
    decide_local_idv2v_mode,
)


def test_decide_unsupported_without_cuda() -> None:
    assert decide_local_idv2v_mode(cuda_available=False, vram_gb=24) == "unsupported"


def test_decide_unsupported_without_vram() -> None:
    assert decide_local_idv2v_mode(cuda_available=True, vram_gb=None) == "unsupported"


def test_decide_unsupported_below_floor() -> None:
    assert (
        decide_local_idv2v_mode(cuda_available=True, vram_gb=IDV2V_LOCAL_VRAM_FLOOR_GB - 1)
        == "unsupported"
    )


def test_decide_available_at_floor() -> None:
    assert (
        decide_local_idv2v_mode(cuda_available=True, vram_gb=IDV2V_LOCAL_VRAM_FLOOR_GB)
        == "available"
    )


def test_decide_available_large_gpu() -> None:
    assert decide_local_idv2v_mode(cuda_available=True, vram_gb=32) == "available"
