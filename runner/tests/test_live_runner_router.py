"""GPU-independent tests for the live-runner swap policy + routing.

Covers the ResidentWorkerManager evict-before-load invariant (only one worker
resident at a time) and the capability->worker route table. Uses an in-memory
fake transport — no aiohttp server, no GPU, no gateway SDK required.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest  # noqa: E402

from runner.live_runner.swap import ResidentWorkerManager, WorkerTransport  # noqa: E402


class InMemoryTransport(WorkerTransport):
    """Records calls; acts like a set of healthy fake workers."""

    def __init__(self):
        self.order = []       # ["load:ltx", "evict:ltx", ...]
        self.resident = None

    def _name(self, base):
        # The injected worker->base mapping points names at a base that encodes
        # the name itself ("ltx" -> "ltx", "idv2v" -> "idv2v").
        return base.rstrip("/").split("/")[-1]

    async def post(self, base, path, payload=None):
        name = self._name(base)
        kind = path.lstrip("/")  # "load" | "evict" | "restyle"...
        self.order.append(f"{kind}:{name}")
        if kind == "load":
            self.resident = name
        elif kind == "evict":
            self.resident = None
        return {}

    async def health(self, base):
        return {"status": "ok"}


@pytest.fixture
def workers_map():
    return {"ltx": "ltx", "idv2v": "idv2v"}


@pytest.fixture
def fake_workers():
    return InMemoryTransport()


def _mk(transport, workers_map):
    return ResidentWorkerManager(transport=transport, workers=workers_map)


async def _ensure_evicts_previous_resident(fake_workers, workers_map):
    w = _mk(fake_workers, workers_map)
    await w.ensure("ltx")
    await w.ensure("idv2v")
    assert fake_workers.order == ["load:ltx", "evict:ltx", "load:idv2v"]
    assert w.resident == "idv2v"


async def _ensure_same_resident_is_noop(fake_workers, workers_map):
    w = _mk(fake_workers, workers_map)
    await w.ensure("ltx")
    await w.ensure("ltx")
    assert fake_workers.order == ["load:ltx"]  # second ensure skipped load


async def _evict_all_releases(fake_workers, workers_map):
    w = _mk(fake_workers, workers_map)
    await w.ensure("ltx")
    await w.evict_all()
    assert fake_workers.order == ["load:ltx", "evict:ltx"]
    assert w.resident is None


async def _check_health_reports_workers_up(fake_workers, workers_map):
    w = _mk(fake_workers, workers_map)
    meta = await w.check_health()
    assert meta["ltx_up"] is True
    assert meta["idv2v_up"] is True


def test_resident_starts_none(fake_workers, workers_map):
    w = _mk(fake_workers, workers_map)
    assert w.resident is None


def test_ensure_evicts_previous_resident(fake_workers, workers_map):
    asyncio.run(_ensure_evicts_previous_resident(fake_workers, workers_map))


def test_ensure_same_resident_is_noop(fake_workers, workers_map):
    asyncio.run(_ensure_same_resident_is_noop(fake_workers, workers_map))


def test_evict_all_releases(fake_workers, workers_map):
    asyncio.run(_evict_all_releases(fake_workers, workers_map))


def test_check_health_reports_workers_up(fake_workers, workers_map):
    asyncio.run(_check_health_reports_workers_up(fake_workers, workers_map))


def test_routing_table_has_restyle_on_idv2v():
    from runner.live_runner.routing import ROUTES
    assert ROUTES["restyle"] == "idv2v-worker"
    # All the old LTX endpoints route to ltx-worker.
    for ep in ("t2v", "i2v", "retake", "extend", "image"):
        assert ROUTES[ep] == "ltx-worker"


if __name__ == "__main__":
    from asyncio import run

    async def _main():
        fw = InMemoryTransport()
        w = ResidentWorkerManager(transport=fw, workers={"ltx": "ltx", "idv2v": "idv2v"})
        await w.ensure("ltx")
        await w.ensure("idv2v")
        print(fw.order)
        assert fw.order == ["load:ltx", "evict:ltx", "load:idv2v"]
        print("PASS test_ensure_evicts_previous_resident")

    run(_main())
