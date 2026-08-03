"""Tests for forwarding the selected catalog LoRA to the remote Livepeer runner.

Covers `VideoGenerationHandler._catalog_lora_ref` — the catalog-only mapping from
an installed `LoraEntry.ref` (models_dir-relative `loras/<id>/<file>`) to a
catalog `(id, filename)` — so the remote runner never receives an arbitrary path.

Run:  python -m pytest tests/test_video_lora_remote.py -q
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402

from _routes._errors import HTTPError  # noqa: E402
from handlers.video_generation_handler import VideoGenerationHandler  # noqa: E402
from tests.fakes.services import FakeLoraCatalogProvider  # noqa: E402


def _handler_with(catalog) -> VideoGenerationHandler:
    """Build a VideoGenerationHandler with only `_catalog` set (bypasses __init__
    so we don't need to wire the full dependency graph)."""
    h = VideoGenerationHandler.__new__(VideoGenerationHandler)
    h._catalog = catalog
    return h


@pytest.fixture
def handler() -> VideoGenerationHandler:
    return _handler_with(FakeLoraCatalogProvider())  # has plain lora "cozy-felt-v1"


def test_catalog_lora_ref_happy_path(handler):
    assert handler._catalog_lora_ref("loras/cozy-felt-v1/felt.safetensors") == (
        "cozy-felt-v1",
        "felt.safetensors",
    )


def test_catalog_lora_ref_rejects_unknown_id(handler):
    with pytest.raises(HTTPError) as ei:
        handler._catalog_lora_ref("loras/nope/x.safetensors")
    assert ei.value.status_code == 400


def test_catalog_lora_ref_rejects_unknown_file(handler):
    with pytest.raises(HTTPError) as ei:
        handler._catalog_lora_ref("loras/cozy-felt-v1/other.safetensors")
    assert ei.value.status_code == 400


def test_catalog_lora_ref_rejects_non_lora_path(handler):
    with pytest.raises(HTTPError) as ei:
        handler._catalog_lora_ref("notalora/x.safetensors")
    assert ei.value.status_code == 400


def test_catalog_lora_ref_rejects_ic_lora_id(handler):
    # An IC-LoRA id is not a plain catalog lora -> 400 (not forwarded).
    with pytest.raises(HTTPError) as ei:
        handler._catalog_lora_ref("loras/ingredients-v1/i.safetensors")
    assert ei.value.status_code == 400
