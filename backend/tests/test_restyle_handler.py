"""Tests for the Restyle handler (remote id-v2v restylization via Livepeer)."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from api_types import RestyleRequest, RestyleVideoResponse
from _routes._errors import HTTPError


class FakeLivepeerClient:
    """Records the call; returns a FakeRunnerInfo with a save_result that writes a file."""

    def __init__(self, result_dir: Path):
        self.result_dir = result_dir
        self.calls: list[tuple] = []
        self.saved: list[str] = []

    def get_runner_for_with_recovery(self, selected_id, excluded_ids, capability="restyle"):
        class _Runner:
            runner_id = "runner-1"
            url = "http://runner"
        return _Runner()

    async def call(self, runner, endpoint, payload, timeout_s=600.0):
        self.calls.append((endpoint, payload, timeout_s))
        return {"output_video": base64.b64encode(b"fake-mp4-bytes").decode(), "frames_generated": 81}

    def save_result(self, base64_data: str, content_type: str) -> str:
        path = self.result_dir / "result.mp4"
        path.write_bytes(base64.b64decode(base64_data))
        self.saved.append(str(path))
        return str(path)


@pytest.fixture
def fake_client(tmp_path: Path) -> FakeLivepeerClient:
    return FakeLivepeerClient(tmp_path)


def test_restyle_dispatch_to_livepeer(test_state, tmp_path: Path, fake_client):
    """Restyle goes through LivepeerClient with the right endpoint + base64 payload."""
    # Attach the fake client to the handler state (mirrors app_factory startup).
    test_state.state._livepeer_client = fake_client  # type: ignore[attr-defined]

    video = tmp_path / "source.mp4"
    video.write_bytes(b"video-bytes")
    image = tmp_path / "style.png"
    image.write_bytes(b"image-bytes")

    req = RestyleRequest(
        video_path=str(video),
        stylized_image_path=str(image),
        prompt="make it anime",
        max_frames=81,
        inference_steps=30,
        cfg_scale=5.0,
    )

    resp = test_state.restyle.run(req)

    assert isinstance(resp, RestyleVideoResponse)
    assert resp.status == "complete"
    assert resp.video_path.endswith(".mp4")
    # Endpoint + payload shape
    assert fake_client.calls
    endpoint, payload, timeout_s = fake_client.calls[0]
    assert endpoint == "/video-creator/v1/restyle"
    assert payload["prompt"] == "make it anime"
    assert payload["source_video"] == base64.b64encode(b"video-bytes").decode()
    assert payload["stylized_first_frame"] == base64.b64encode(b"image-bytes").decode()
    assert timeout_s == 1200
    # Result saved
    assert fake_client.saved


def test_restyle_requires_livepeer_client(test_state, tmp_path: Path):
    """Without a livepeer client, restyle returns 503."""
    test_state.state._livepeer_client = None  # type: ignore[attr-defined]

    video = tmp_path / "source.mp4"
    video.write_bytes(b"v")
    image = tmp_path / "style.png"
    image.write_bytes(b"i")

    req = RestyleRequest(
        video_path=str(video),
        stylized_image_path=str(image),
        prompt="x",
    )
    with pytest.raises(HTTPError) as exc:
        test_state.restyle.run(req)
    assert exc.value.status_code == 503


def test_restyle_missing_image_path(test_state, tmp_path: Path):
    """Missing stylized_image_path -> 400."""
    video = tmp_path / "source.mp4"
    video.write_bytes(b"v")
    req = RestyleRequest(
        video_path=str(video),
        stylized_image_path="",
        prompt="x",
    )
    with pytest.raises(HTTPError) as exc:
        test_state.restyle.run(req)
    assert exc.value.status_code == 400
