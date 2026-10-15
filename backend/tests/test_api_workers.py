"""``GET /api/workers`` and drain/undrain (#772)."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.core import worker_protocol as wp
from app.core.auth import TOKEN_HEADER, session_token
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
        assert not registry.get("box").accepts_runs
        assert (await client.post("/api/workers/box/undrain")
                ).json()["draining"] is False
        assert (await client.post("/api/workers/nobody/drain")
                ).status_code == 404
    assert len(changes) == 2


async def test_no_registry_is_503():
    async with _client() as client:
        assert (await client.get("/api/workers")).status_code == 503
