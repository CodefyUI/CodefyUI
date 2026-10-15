"""``GET /api/workers`` and drain/undrain (#772); ``plain_http`` (#786)."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from httpx_ws import aconnect_ws
from httpx_ws.transport import ASGIWebSocketTransport

from app.core import worker_protocol as wp
from app.core.auth import TOKEN_HEADER, session_token
from app.core.db import Database
from app.core.run_service import QueueLimits, RunService
from app.core.run_store import RunStore
from app.core.workers import WorkerRegistry
from app.main import app

from .test_ws_worker import FakeAgent, env  # noqa: F401
from .test_ws_worker_auth import enrolled, remote_peer  # noqa: F401

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
        tmp_path, monkeypatch):
    """#773: ``slots: {limit, taken}`` per device, for each worker and for
    this server (``local``); a same-host worker's shared card says
    ``shared_with: "local"`` and reads this server's slot."""
    from app.core.resources import DeviceFact

    registry = WorkerRegistry(local_devices=lambda: [
        DeviceFact("cpu", "cpu", None, "", None, None, "none"),
        DeviceFact("cuda:0", "cuda", 0, "", None, None, "none", "u-0")])
    app.state.worker_registry = registry
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
        registry.connect(wp.Hello(name="here", version="t", devices=[
            wp.DeviceReport(value="cuda:0", kind="cuda", uuid="u-0"),
            wp.DeviceReport(value="cuda:1", kind="cuda", uuid="u-9")]),
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
        assert {d["shared_with"] for d in far["devices"]} == {None}
        assert [(d["shared_with"], d["slots"]) for d in here["devices"]] == [
            ("local", {"limit": 1, "taken": 1}),
            (None, {"limit": 1, "taken": 0})]
        # Mounts (#782): this server's label and path; the same-host
        # worker, which declared none, holds this server's; the far one
        # holds nothing.
        from app.core.mounts import control_plane_mounts

        assert far["mounts"] == {}
        assert here["mounts"] == control_plane_mounts()
        assert body["local"] == {"id": "local",
                                 "mounts": control_plane_mounts(),
                                 "devices": [
            {"value": "cpu", "kind": "cpu",
             "slots": {"limit": 2, "taken": 0}},
            {"value": "cuda:0", "kind": "cuda",
             "slots": {"limit": 1, "taken": 1}}]}
        async with _client() as client:
            drained = (await client.post("/api/workers/far/drain")).json()
        assert drained["devices"][1]["slots"] == {"limit": 1, "taken": 1}
    finally:
        await service.shutdown()
        await registry.shutdown()
        del app.state.run_service
        del app.state.worker_registry
        database.close()


async def test_without_a_run_service_slots_are_null(registry):
    registry.connect(wp.Hello(name="box", version="t",
                              devices=_devices("cpu")),
                     send=_noop, close=_noop)
    async with _client(token=False) as client:
        body = (await client.get("/api/workers")).json()
    assert body["local"] is None
    assert body["workers"][0]["devices"][0]["slots"] is None


# ── the "plain HTTP" badge field (#786) ───────────────────────────────────


async def test_plain_http_is_listed_per_worker(registry):
    registry.connect(wp.Hello(name="far", version="t"), send=_noop,
                     close=_noop, plain_http=True)
    registry.connect(wp.Hello(name="near", version="t"), send=_noop,
                     close=_noop)
    async with _client(token=False) as client:
        body = (await client.get("/api/workers")).json()
    assert {w["name"]: w["plain_http"] for w in body["workers"]} == {
        "far": True, "near": False}


async def _plain_http_after_connecting(base: str, worker_token: str,
                                       registry: WorkerRegistry) -> bool:
    """Connect ``gpu-box`` through ``/ws/worker`` at *base*; its listed
    ``plain_http``."""
    async with AsyncClient(transport=ASGIWebSocketTransport(app=app),
                           base_url=base) as client, \
            aconnect_ws(wp.WORKER_WS_PATH, client,
                        headers={"X-CodefyUI-Token": worker_token}) as ws:
        await FakeAgent(ws).hello("gpu-box")
        [listed] = (await client.get("/api/workers")).json()["workers"]
        assert listed["plain_http"] == registry.get("gpu-box").plain_http
        return listed["plain_http"]


async def test_plain_ws_from_another_machine_shows_the_badge(
        enrolled, remote_peer):  # noqa: F811
    _, registry, _, worker_token = enrolled
    assert await _plain_http_after_connecting(
        "http://127.0.0.1:8000", worker_token, registry) is True


async def test_tls_from_another_machine_shows_no_badge(
        enrolled, remote_peer):  # noqa: F811
    _, registry, _, worker_token = enrolled
    assert await _plain_http_after_connecting(
        "https://127.0.0.1:8000", worker_token, registry) is False


async def test_a_loopback_peer_shows_no_badge(enrolled):  # noqa: F811
    # A same-host agent, a TLS proxy on this machine, or an SSH tunnel's
    # local end (#784): the hop across the network is not this socket.
    _, registry, _, worker_token = enrolled
    assert await _plain_http_after_connecting(
        "http://127.0.0.1:8000", worker_token, registry) is False
