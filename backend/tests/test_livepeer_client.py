"""Tests for warm-preference, capability-aware runner selection."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402

from services.livepeer_client import LivepeerClient, RunnerInfo  # noqa: E402


def _runner(rid, url, capabilities=None, label="", warm_model=None, workers_up=None):
    raw = {
        "runner_id": rid,
        "runner_url": url,
    }
    if capabilities is not None:
        raw["capabilities"] = capabilities
    if label:
        raw["label"] = label
    if warm_model is not None or workers_up is not None:
        meta = {}
        if warm_model is not None:
            meta["warm_model"] = warm_model
        if workers_up is not None:
            meta.update(workers_up)
        # The gateway surfaces metadata as a JSON string.
        import json
        raw["metadata"] = json.dumps(meta)
    return RunnerInfo(rid, url, raw)


@pytest.fixture
def client():
    from pathlib import Path
    return LivepeerClient(discovery_url="", results_dir=Path("."))


def _populate(client, runners):
    client._runners = {r.runner_id: r for r in runners}


def test_capability_matching_picks_restyle_runner(client):
    c = client
    a = _runner("a", "http://a", capabilities=["t2v"])
    b = _runner("b", "http://b", capabilities=["t2v", "restyle"])
    _populate(c, [a, b])
    got = c.get_runner_for("", [], capability="restyle")
    assert got is not None and got.runner_id == "b"


def test_prefers_warm_model(client):
    c = client
    cold = _runner("cold", "http://cold", capabilities=["restyle"], warm_model=None)
    warm = _runner("warm", "http://warm", capabilities=["restyle"], warm_model="idv2v")
    _populate(c, [cold, warm])
    got = c.get_runner_for("", [], capability="restyle")
    assert got is not None and got.runner_id == "warm"


def test_falls_through_when_none_warm(client):
    c = client
    a = _runner("a", "http://a", capabilities=["restyle"], warm_model=None)
    b = _runner("b", "http://b", capabilities=["restyle"], warm_model=None)
    _populate(c, [a, b])
    got = c.get_runner_for("", [], capability="restyle")
    assert got is not None and got.runner_id in ("a", "b")


def test_explicit_selection_respected_when_capable(client):
    c = client
    a = _runner("a", "http://a", capabilities=["t2v"])
    b = _runner("b", "http://b", capabilities=["restyle"])
    _populate(c, [a, b])
    got = c.get_runner_for("b", [], capability="restyle")
    assert got is not None and got.runner_id == "b"


def test_excluded_skipped_in_fallthrough(client):
    c = client
    a = _runner("a", "http://a", capabilities=["restyle"])
    b = _runner("b", "http://b", capabilities=["restyle"])
    _populate(c, [a, b])
    got = c.get_runner_for("", ["a"], capability="restyle")
    assert got is not None and got.runner_id == "b"


def test_worker_up_flags_parsed_from_metadata():
    r = _runner("x", "http://x", warm_model="ltx",
                workers_up={"ltx_worker_up": True, "idv2v_worker_up": False})
    assert r.warm_model == "ltx"
    assert r.ltx_worker_up is True
    assert r.idv2v_worker_up is False
