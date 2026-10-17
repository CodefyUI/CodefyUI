"""``/ws/worker`` authentication (#780): worker tokens from any peer, the
session token from loopback peers only, and revocation.

The fake agent and the app wiring are ``test_ws_worker``'s; the peer
address is made non-loopback by patching ``ws_worker._peer_is_loopback``
(httpx's ASGI transport always reports 127.0.0.1).
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from httpx_ws import WebSocketDisconnect, aconnect_ws

from app.api import ws_worker
from app.core import worker_protocol as wp
from app.core.auth import TOKEN_QUERY_PARAM
from app.core.worker_store import WorkerStore
from app.core.env_fingerprint import fingerprint_digest
from app.core.workers import (
    CREDENTIAL_SESSION,
    CREDENTIAL_WORKER,
    WorkerRegistry,
)
from app.main import app

from .test_ws_worker import FakeAgent, _client, _path, _until, env  # noqa: F401


@pytest.fixture
async def enrolled(env):  # noqa: F811 -- the imported fixture
    """``env`` plus a worker store with ``gpu-box`` enrolled."""
    service, registry = env
    store = WorkerStore(app.state.db)
    app.state.worker_store = store
    created = await store.create_enrollment()
    worker_token, _ = await store.enroll(
        created["token"], name="gpu-box", platform={}, mounts={}, devices=[])
    try:
        yield service, registry, store, worker_token
    finally:
        del app.state.worker_store


@pytest.fixture
def remote_peer(monkeypatch):
    monkeypatch.setattr(ws_worker, "_peer_is_loopback", lambda ws: False)


async def test_a_worker_token_is_accepted_from_another_machine(
        enrolled, remote_peer):
    _, registry, _, worker_token = enrolled
    async with _client(token=False) as client, \
            aconnect_ws(wp.WORKER_WS_PATH, client,
                        headers={"X-CodefyUI-Token": worker_token}) as ws:
        welcome = await FakeAgent(ws).hello("gpu-box")
        assert isinstance(welcome, wp.Welcome)
        worker = registry.get("gpu-box")
        assert worker.online and worker.enrolled
        assert worker.describe()["credential"] == CREDENTIAL_WORKER


async def test_the_query_parameter_works_too(enrolled, remote_peer):
    _, registry, _, worker_token = enrolled
    async with _client(token=False) as client, \
            aconnect_ws(_path(worker_token), client) as ws:
        assert isinstance(await FakeAgent(ws).hello("gpu-box"), wp.Welcome)


async def test_the_session_token_is_refused_from_another_machine(
        enrolled, remote_peer):
    _, registry, _, _ = enrolled
    async with _client(token=False) as client:
        with pytest.raises(WebSocketDisconnect) as refused:
            async with aconnect_ws(_path(), client):
                pass
    assert refused.value.code == 4401
    assert registry.get("box") is None


async def test_the_session_token_still_works_from_this_machine(enrolled):
    _, registry, _, _ = enrolled
    async with _client(token=False) as client, \
            aconnect_ws(_path(), client) as ws:
        assert isinstance(await FakeAgent(ws).hello("box"), wp.Welcome)
        worker = registry.get("box")
        assert worker.credential == CREDENTIAL_SESSION and not worker.enrolled


@pytest.mark.parametrize("token", ["", "wrong", "abcdef.0123456789abcdef"])
async def test_an_unknown_token_is_refused(enrolled, remote_peer, token):
    async with _client(token=False) as client:
        with pytest.raises(WebSocketDisconnect) as refused:
            async with aconnect_ws(f"{wp.WORKER_WS_PATH}?"
                                   f"{TOKEN_QUERY_PARAM}={token}", client):
                pass
    assert refused.value.code == 4401


async def test_a_removed_workers_credential_is_listed_when_refused(
        enrolled, remote_peer):
    """#840: an agent left running after its worker was removed is
    refused, and the refusal is listed under its name, with why."""
    _, registry, store, worker_token = enrolled
    assert store.revoked(worker_token) is None
    assert not await store.delete_worker("ghost")
    assert await store.delete_worker("gpu-box")
    assert store.revoked(worker_token) == ("gpu-box", "removed")
    async with _client(token=False) as client:
        with pytest.raises(WebSocketDisconnect) as refused:
            async with aconnect_ws(_path(worker_token), client):
                pass
    assert refused.value.code == 4401
    [rejection] = registry.rejections()
    assert rejection["name"] == "gpu-box"
    assert "the worker was removed" in rejection["reason"]
    # Another refused token names nobody and is not listed.
    async with _client(token=False) as client:
        with pytest.raises(WebSocketDisconnect):
            async with aconnect_ws(_path("wrong"), client):
                pass
    assert len(registry.rejections()) == 1


async def test_a_rekeyed_credential_is_listed_when_refused(enrolled,
                                                           remote_peer):
    _, registry, store, worker_token = enrolled
    rekey = await store.create_enrollment(worker="gpu-box")
    await store.enroll(rekey["token"], name="gpu-box", platform={},
                       mounts={}, devices=[])
    assert store.revoked(worker_token) == ("gpu-box", "rekeyed")
    async with _client(token=False) as client:
        with pytest.raises(WebSocketDisconnect):
            async with aconnect_ws(_path(worker_token), client):
                pass
    [rejection] = registry.rejections()
    assert "replaced by a re-key" in rejection["reason"]


def test_the_store_remembers_a_bounded_number_of_revocations(monkeypatch):
    from app.core import worker_store

    monkeypatch.setattr(worker_store, "MAX_REVOKED", 2)
    store = WorkerStore(None)
    for n in range(3):
        store._remember_revoked(f"h{n}", f"w{n}", "removed")
    store._remember_revoked("", "nobody", "removed")
    assert list(store._revoked) == ["h1", "h2"]
    assert store.revoked(None) is None


async def test_a_worker_token_only_registers_its_own_name(enrolled,
                                                          remote_peer):
    _, registry, _, worker_token = enrolled
    async with _client(token=False) as client, \
            aconnect_ws(_path(worker_token), client) as ws:
        reply = await FakeAgent(ws).hello("someone-else")
    assert isinstance(reply, wp.Reject)
    assert "belongs to worker 'gpu-box'" in reply.reason
    assert registry.get("someone-else") is None


async def test_connecting_records_the_facts_and_last_seen(enrolled,
                                                          remote_peer):
    _, _, store, worker_token = enrolled
    db = app.state.db
    before = db._conn.execute(
        "SELECT last_seen_at FROM workers").fetchone()[0]
    async with _client(token=False) as client, \
            aconnect_ws(_path(worker_token), client) as ws:
        await FakeAgent(ws).hello(
            "gpu-box", devices=("cpu", "cuda:0"),
            platform={"system": "Linux"},
            fingerprint=wp.FingerprintReport(document={"k": 1}, digest="d"))
        await _until(lambda: json.loads(db._conn.execute(
            "SELECT devices_json FROM workers").fetchone()[0]) != [])
    row = db._conn.execute("SELECT * FROM workers").fetchone()
    assert [d["value"] for d in json.loads(row["devices_json"])] == [
        "cpu", "cuda:0"]
    assert json.loads(row["platform"]) == {"system": "Linux"}
    assert row["last_seen_at"] >= before
    [known] = await store.list_workers()
    assert known.name == "gpu-box"
    # Its environment (#781), shown before it reconnects after a restart.
    assert row["fingerprint_digest"] == known.fingerprint_digest \
        == fingerprint_digest({"k": 1})
    [offline] = WorkerRegistry().load_known([known])
    assert offline.describe()["fingerprint_digest"] == known.fingerprint_digest
    # Reconnected without a fingerprint: the stored digest is cleared, so
    # the panel says "not reported" rather than show a stale one.
    async with _client(token=False) as client, \
            aconnect_ws(_path(worker_token), client) as ws:
        await FakeAgent(ws).hello("gpu-box")
        await _until(lambda: db._conn.execute(
            "SELECT fingerprint_digest FROM workers").fetchone()[0] is None)


async def test_revoking_disconnects_and_refuses_reconnects(enrolled,
                                                           remote_peer):
    _, registry, _, worker_token = enrolled
    async with _client() as client, _client(token=False) as agent_client:
        async with aconnect_ws(_path(worker_token), agent_client) as ws:
            agent = FakeAgent(ws)
            await agent.hello("gpu-box")
            removed = await client.delete("/api/workers/gpu-box")
            assert removed.json()["revoked"] is True
            with pytest.raises(Exception):  # the server closed the socket
                await agent.recv(timeout=5)
        assert registry.get("gpu-box") is None
        with pytest.raises(WebSocketDisconnect) as refused:
            async with aconnect_ws(_path(worker_token), agent_client):
                pass
        assert refused.value.code == 4401


async def test_a_rekeyed_worker_loses_its_connection(enrolled, remote_peer):
    _, registry, store, worker_token = enrolled
    async with _client() as client, _client(token=False) as agent_client:
        async with aconnect_ws(_path(worker_token), agent_client) as ws:
            agent = FakeAgent(ws)
            await agent.hello("gpu-box")
            created = (await client.post("/api/workers/enrollments",
                                         json={"worker": "gpu-box"})).json()
            response = await client.post("/api/workers/enroll", json={
                "token": created["token"], "name": "gpu-box"})
            assert response.status_code == 200
            with pytest.raises(Exception):
                await agent.recv(timeout=5)
        worker = registry.get("gpu-box")
        assert worker is not None and not worker.online and worker.enrolled
    assert await store.authenticate(worker_token) is None


async def test_without_a_store_only_the_session_token_works(env,  # noqa: F811
                                                            remote_peer):
    async with _client(token=False) as client:
        with pytest.raises(WebSocketDisconnect) as refused:
            async with aconnect_ws(_path("anything"), client):
                pass
    assert refused.value.code == 4401


async def test_a_failed_last_seen_write_never_costs_the_connection(
        enrolled, remote_peer, monkeypatch, caplog):
    _, registry, store, worker_token = enrolled

    async def broken(*_: Any, **__: Any) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(store, "touch", broken)
    async with _client(token=False) as client, \
            aconnect_ws(_path(worker_token), client) as ws:
        assert isinstance(await FakeAgent(ws).hello("gpu-box"), wp.Welcome)
    await _until(lambda: not registry.get("gpu-box").online)
    assert "could not record worker gpu-box as seen" in caplog.text


# ── identity is bound to the credential (Wave 10 architecture review) ──────


async def test_the_session_token_cannot_take_an_enrolled_name(enrolled):
    _, registry, _, _ = enrolled
    registry.load_known(await enrolled[2].list_workers())
    async with _client(token=False) as client, \
            aconnect_ws(_path(), client) as ws:
        reply = await FakeAgent(ws).hello("gpu-box")
    assert isinstance(reply, wp.Reject) and "is enrolled" in reply.reason
    assert not registry.get("gpu-box").online


async def test_a_worker_token_cannot_replace_a_live_same_host_agent(enrolled):
    _, registry, _, worker_token = enrolled
    async with _client(token=False) as client, \
            aconnect_ws(_path(), client) as same_host:
        assert isinstance(await FakeAgent(same_host).hello("gpu-box"),
                          wp.Welcome)
        async with aconnect_ws(_path(worker_token), client) as ws:
            reply = await FakeAgent(ws).hello("gpu-box")
        assert isinstance(reply, wp.Reject)
        assert "another kind of credential" in reply.reason
        assert registry.get("gpu-box").credential == CREDENTIAL_SESSION


async def test_a_worker_token_may_replace_its_own_connection(enrolled,
                                                             remote_peer):
    _, registry, _, worker_token = enrolled
    async with _client(token=False) as client, \
            aconnect_ws(_path(worker_token), client) as first:
        await FakeAgent(first).hello("gpu-box")
        async with aconnect_ws(_path(worker_token), client) as second:
            assert isinstance(await FakeAgent(second).hello("gpu-box"),
                              wp.Welcome)
            assert registry.get("gpu-box").online


async def test_a_remote_peer_is_never_same_host_and_its_machine_id_is_kept(
        enrolled, remote_peer, monkeypatch):
    from app.api import ws_worker as module

    _, registry, _, worker_token = enrolled
    monkeypatch.setattr(module, "machine_identity", lambda: "m-1")
    async with _client(token=False) as client, \
            aconnect_ws(_path(worker_token), client) as ws:
        await FakeAgent(ws).hello("gpu-box", machine_id="m-1")
        assert registry.get("gpu-box").same_host is False
        await _until(lambda: app.state.db._conn.execute(
            "SELECT machine_id FROM workers").fetchone()[0] == "m-1")


async def test_a_loopback_worker_token_with_our_machine_id_is_same_host(
        enrolled, monkeypatch):
    from app.api import ws_worker as module

    _, registry, _, worker_token = enrolled
    monkeypatch.setattr(module, "machine_identity", lambda: "m-1")
    async with _client(token=False) as client, \
            aconnect_ws(_path(worker_token), client) as ws:
        await FakeAgent(ws).hello("gpu-box", machine_id="m-1")
        assert registry.get("gpu-box").same_host is True


async def test_plain_ws_from_another_machine_is_warned_about(
        enrolled, remote_peer, caplog):
    _, _, _, worker_token = enrolled
    # In the header, as the agent sends it: the test client's own request
    # log (httpx, at INFO) would otherwise print a query-string token.
    async with _client(token=False) as client, \
            aconnect_ws(wp.WORKER_WS_PATH, client,
                        headers={"X-CodefyUI-Token": worker_token}) as ws:
        await FakeAgent(ws).hello("gpu-box")
    assert "plain ws://" in caplog.text
    server_log = "\n".join(r.getMessage() for r in caplog.records
                           if r.name.startswith("app."))
    assert worker_token not in server_log


async def test_no_warning_from_this_machine(enrolled, caplog):
    _, _, _, worker_token = enrolled
    async with _client(token=False) as client, \
            aconnect_ws(_path(worker_token), client) as ws:
        await FakeAgent(ws).hello("gpu-box")
    assert "plain ws://" not in caplog.text


def _tunnel(name: str = "gpu-box", host: str = "127.0.0.1:18077"):
    """A client of SSH worker *name*'s own listener (``TunnelApp``), the
    far end of whose forward sends *host* as ``Host``."""
    from httpx import AsyncClient
    from httpx_ws.transport import ASGIWebSocketTransport

    from app.core.tunnel_listener import TunnelApp

    return AsyncClient(transport=ASGIWebSocketTransport(
        app=TunnelApp(app, name)), base_url=f"http://{host}")


async def test_an_ssh_tunnel_peer_with_another_machine_id_is_not_same_host(
        enrolled, monkeypatch):
    """#784: over ``ssh -R`` a remote agent is a loopback peer. Its machine
    id decides: not ours."""
    from httpx import AsyncClient
    from httpx_ws.transport import ASGIWebSocketTransport

    from app.api import ws_worker as module

    _, registry, _, worker_token = enrolled
    monkeypatch.setattr(module, "machine_identity", lambda: "control-plane")
    # The main port: a tunnel's ``Host`` is just a Host the guard refuses.
    forged = AsyncClient(transport=ASGIWebSocketTransport(app=app),
                         base_url="http://127.0.0.1:18077")
    async with forged as client:
        with pytest.raises(WebSocketDisconnect) as refused:
            async with aconnect_ws(_path(worker_token), client):
                pass
    assert refused.value.code == 4003
    async with _tunnel() as client, \
            aconnect_ws(_path(worker_token), client) as ws:
        await FakeAgent(ws).hello("gpu-box", machine_id="gpu-box-id")
        worker = registry.get("gpu-box")
        assert worker.same_host is False
        assert worker.credential == CREDENTIAL_WORKER
        assert worker.describe()["plain_http"] is False


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.1:8000",
                                  "localhost:8000", "evil.example"])
async def test_the_session_token_never_opens_the_worker_socket_through_a_tunnel(
        enrolled, host):
    """#784: through an SSH worker's listener the peer is ssh on this
    machine, but the client is whoever reached the forward's far end; the
    session token is refused there, whatever ``Host`` it sends."""
    _, registry, _, _ = enrolled
    async with _tunnel(host=host) as client:
        with pytest.raises(WebSocketDisconnect) as refused:
            async with aconnect_ws(_path(), client):
                pass
    assert refused.value.code == 4401
    assert registry.get("box") is None


class _Launcher:
    def __init__(self) -> None:
        self.refused: list[str] = []
        self.connected: list[str] = []

    def note_refused(self, name: str) -> None:
        self.refused.append(name)

    def worker_connected(self, name: str) -> None:
        self.connected.append(name)


async def test_a_refused_credential_on_a_tunnel_never_reaches_the_launcher(
        enrolled, monkeypatch, caplog):
    """#784: a 4401 through an SSH worker's listener is not reported to
    the launcher (any client of the listener could cause it, and acting on
    it wipes the agent's credential); neither is a Host refusal (4003) on
    the main port. The launcher acts on the agent's own log line instead.
    A tunnelled peer is not warned about plain ws:// (the hop across the
    network is ssh's)."""
    from httpx import AsyncClient
    from httpx_ws.transport import ASGIWebSocketTransport

    _, registry, _, worker_token = enrolled
    launcher = _Launcher()
    app.state.ssh_launcher = launcher
    try:
        async with _tunnel("gpu-box") as client:
            with pytest.raises(WebSocketDisconnect) as refused:
                async with aconnect_ws(_path("not-a-token"), client):
                    pass
        assert refused.value.code == 4401
        forged = AsyncClient(transport=ASGIWebSocketTransport(app=app),
                             base_url="http://127.0.0.1:18079")
        async with forged as client:
            with pytest.raises(WebSocketDisconnect) as refused:
                async with aconnect_ws(_path("not-a-token"), client):
                    pass
        assert refused.value.code == 4003
        # Not reported: the listener cannot tell the agent from another
        # client; the launcher acts on the agent's own report instead.
        assert launcher.refused == []

        monkeypatch.setattr(ws_worker, "_peer_is_loopback", lambda ws: False)
        async with _tunnel("gpu-box") as client, \
                aconnect_ws(_path(worker_token), client) as ws:
            await FakeAgent(ws).hello("gpu-box")
        assert launcher.connected == ["gpu-box"]
        assert "plain ws://" not in caplog.text
    finally:
        del app.state.ssh_launcher


async def test_the_authorisers_are_pluggable(env, monkeypatch):  # noqa: F811
    async def everyone(ws, provided, loopback):
        return ws_worker.Authorised(CREDENTIAL_SESSION) \
            if provided == "letmein" else None

    assert ws_worker.AUTHORISERS == [ws_worker.session_token_authoriser,
                                     ws_worker.worker_token_authoriser]
    monkeypatch.setattr(ws_worker, "AUTHORISERS", [everyone])
    async with _client(token=False) as client:
        async with aconnect_ws(_path("letmein"), client) as ws:
            assert isinstance(await FakeAgent(ws).hello("box"), wp.Welcome)
        with pytest.raises(WebSocketDisconnect):
            async with aconnect_ws(_path(), client):
                pass
