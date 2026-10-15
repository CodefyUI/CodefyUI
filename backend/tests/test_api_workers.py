"""``GET /api/workers`` and drain/undrain (#772)."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.core import worker_protocol as wp
from app.core.auth import TOKEN_HEADER, session_token
from app.core.db import Database
from app.core.run_service import QueueLimits, RunService
from app.core.run_store import RunStore
from app.core.workers import WorkerRegistry
from app.main import app

BASE = "http://127.0.0.1:8000"


async def _noop(*_):
    return None


@pytest.fixture
async def registry():
    registry = WorkerRegistry()
    app.state.worker_registry = registry
    try:
        yield registry
    finally:
        await registry.shutdown()
        del app.state.worker_registry


def _client(token: bool = True) -> AsyncClient:
    headers = {TOKEN_HEADER: session_token()} if token else {}
    return AsyncClient(transport=ASGITransport(app=app), base_url=BASE,
                       headers=headers)


async def test_list_is_an_open_read(registry):
    registry.connect(wp.Hello(
        name="box", version="3.1.0", fingerprint="abc",
        devices=[wp.DeviceReport(value="cpu", kind="cpu", name="arm",
                                 total_bytes=8, free_bytes=2,
                                 source="psutil"),
                 wp.DeviceReport(value="mps", kind="mps")],
        mounts={"data": "/mnt"}, platform={"system": "Darwin"}),
        send=_noop, close=_noop)
    async with _client(token=False) as client:
        body = (await client.get("/api/workers")).json()
    [worker] = body["workers"]
    assert worker["name"] == worker["id"] == "box"
    assert (worker["status"], worker["draining"], worker["version"],
            worker["fingerprint_digest"]) == ("online", False, "3.1.0", "abc")
    assert worker["devices"][0]["memory"] == {
        "total_bytes": 8, "free_bytes": 2, "source": "psutil"}
    assert worker["devices"][1]["memory"] is None
    assert worker["mounts"] == {"data": "/mnt"} and worker["running"] == []


async def test_drain_and_undrain(registry):
    registry.connect(wp.Hello(name="box", version="t"), send=_noop,
                     close=_noop)
    changes = []
    registry.add_listener(lambda: changes.append(1))
    async with _client(token=False) as anonymous:
        assert (await anonymous.post("/api/workers/box/drain")
                ).status_code == 403
    async with _client() as client:
        drained = await client.post("/api/workers/box/drain")
        assert drained.status_code == 200 and drained.json()["draining"]
        assert drained.json()["status"] == "draining"
        assert not registry.get("box").accepts_runs
        assert (await client.post("/api/workers/box/undrain")
                ).json()["draining"] is False
        assert (await client.post("/api/workers/nobody/drain")
                ).status_code == 404
    assert len(changes) == 2


async def test_no_registry_is_503():
    async with _client() as client:
        assert (await client.get("/api/workers")).status_code == 503


def _devices(*values: str) -> list[wp.DeviceReport]:
    return [wp.DeviceReport(value=v, kind=v.split(":")[0]) for v in values]


async def test_every_device_carries_its_slots_and_this_server_is_listed(
        registry, tmp_path, monkeypatch):
    """#773: ``slots: {limit, taken}`` per device, for each worker and for
    this server (``local``); a same-host worker reads this server's."""
    database = Database(tmp_path / "codefyui.db")
    database.connect()
    service = RunService(RunStore(database), workers=registry,
                         limits=QueueLimits(overrides=(("far/cuda:1", 3),)))
    monkeypatch.setattr(RunService, "local_devices",
                        staticmethod(lambda: ["cpu", "cuda:0"]))
    app.state.run_service = service
    try:
        registry.connect(wp.Hello(name="far", version="t",
                                  devices=_devices("cpu", "cuda:0",
                                                   "cuda:1", "tpu")),
                         send=_noop, close=_noop)
        registry.connect(wp.Hello(name="here", version="t",
                                  devices=_devices("cuda:0")),
                         send=_noop, close=_noop, same_host=True)
        # What two running runs would hold: one on far's first card, one
        # on this machine's (shared with the same-host worker).
        service._slots.update({("far", "cuda:0"): 1, ("local", "cuda:0"): 1})
        async with _client(token=False) as client:
            body = (await client.get("/api/workers")).json()
        far, here = body["workers"]
        assert far["same_host"] is False and here["same_host"] is True
        assert [d["slots"] for d in far["devices"]] == [
            {"limit": 2, "taken": 0}, {"limit": 1, "taken": 1},
            {"limit": 3, "taken": 0}, None]
        assert here["devices"][0]["slots"] == {"limit": 1, "taken": 1}
        assert body["local"] == {"id": "local", "devices": [
            {"value": "cpu", "kind": "cpu",
             "slots": {"limit": 2, "taken": 0}},
            {"value": "cuda:0", "kind": "cuda",
             "slots": {"limit": 1, "taken": 1}}]}
        async with _client() as client:
            drained = (await client.post("/api/workers/far/drain")).json()
        assert drained["devices"][1]["slots"] == {"limit": 1, "taken": 1}
    finally:
        await service.shutdown()
        del app.state.run_service
        database.close()


async def test_without_a_run_service_slots_are_null(registry):
    registry.connect(wp.Hello(name="box", version="t",
                              devices=_devices("cpu")),
                     send=_noop, close=_noop)
    async with _client(token=False) as client:
        body = (await client.get("/api/workers")).json()
    assert body["local"] is None
    assert body["workers"][0]["devices"][0]["slots"] is None
