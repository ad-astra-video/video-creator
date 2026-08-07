"""Tests for the Restyle handler (remote id-v2v restylization via Livepeer)."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from api_types import (
    ExtractFirstFrameRequest,
    RestyleRequest,
    RestyleVideoResponse,
)
from _routes._errors import HTTPError
from handlers.restyle_handler import DEFAULT_RESTYLE_PROMPT


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


def test_restyle_strips_default_prompt(test_state, tmp_path: Path, fake_client):
    """The UI default "restyle this video" is stripped to a blank prompt before the
    runner is called (the stylized first frame is the real style signal)."""
    test_state.state._livepeer_client = fake_client  # type: ignore[attr-defined]

    video = tmp_path / "source.mp4"
    video.write_bytes(b"video-bytes")
    image = tmp_path / "style.png"
    image.write_bytes(b"image-bytes")

    req = RestyleRequest(
        video_path=str(video),
        stylized_image_path=str(image),
        prompt=DEFAULT_RESTYLE_PROMPT,  # what the UI auto-fills
    )

    test_state.restyle.run(req)

    endpoint, payload, _ = fake_client.calls[0]
    assert endpoint == "/video-creator/v1/restyle"
    assert payload["prompt"] == ""


def test_restyle_preserves_custom_prompt(test_state, tmp_path: Path, fake_client):
    """A custom (non-default) prompt is passed through unchanged."""
    test_state.state._livepeer_client = fake_client  # type: ignore[attr-defined]

    video = tmp_path / "source.mp4"
    video.write_bytes(b"v")
    image = tmp_path / "style.png"
    image.write_bytes(b"i")

    req = RestyleRequest(
        video_path=str(video),
        stylized_image_path=str(image),
        prompt="make it look like cyberpunk anime",
    )

    test_state.restyle.run(req)

    endpoint, payload, _ = fake_client.calls[0]
    assert payload["prompt"] == "make it look like cyberpunk anime"


def test_extract_first_frame_returns_path(test_state, tmp_path: Path):
    """Extract first frame uses the fake video processor and writes a file to outputs."""
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video-bytes")
    # The fake processor returns frames only for registered videos; register it with
    # a non-empty frame so read_frame(frame_idx=0) succeeds.
    from tests.fakes.services import FakeCapture
    test_state.video_processor.register_video(str(video.resolve()), FakeCapture())

    resp = test_state.restyle.extract_first_frame(
        ExtractFirstFrameRequest(video_path=str(video))
    )

    assert resp.image_path
    # Written under the app outputs dir (not the arbitrary source path)
    assert Path(resp.image_path).is_file()
    assert Path(resp.image_path).parent == test_state.config.outputs_dir
    assert Path(resp.image_path).suffix.lower() in (".jpg", ".jpeg", ".png")


def test_extract_first_frame_missing_video(test_state, tmp_path: Path):
    """Extract first frame from a non-existent video -> 400."""
    missing = tmp_path / "nope.mp4"
    with pytest.raises(HTTPError) as exc:
        test_state.restyle.extract_first_frame(
            ExtractFirstFrameRequest(video_path=str(missing))
        )
    assert exc.value.status_code == 400
