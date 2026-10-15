"""An SSH worker's own listener (#784, ``core.tunnel_listener``).

The ``ssh -R`` forward lands there, never on the main port, and anyone
who reaches the forward's far end chooses the ``Host`` header. So the
listener serves the agent routes only, each with a worker (or enrollment)
token, and the main port no longer treats any ``Host`` as tunnel traffic.
"""

from __future__ import annotations

import logging

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from app.core.auth import TOKEN_HEADER, session_token
from app.core.db import Database
from app.core.tunnel_listener import (
    SCOPE_KEY,
    TunnelApp,
    TunnelListener,
    agent_route,
    tunnel_of,
)
from app.core.worker_store import WorkerStore
from app.core.workers import WorkerRegistry
from app.main import app

#: Every ``Host`` a client could send through the forward, including this
#: server's own (the repro for the original hole: ``127.0.0.1:8000``).
HOSTS = ["127.0.0.1", "127.0.0.1:8000", "localhost:8000", "127.0.0.1:18000",
         "[::1]:8000", "evil.example"]


@pytest.fixture
async def workers(tmp_path):
    """A worker store and registry on ``app.state`` (the lifespan does not
    run under httpx's ASGI transport)."""
    database = Database(tmp_path / "tunnel.db")
    database.connect()
    store = WorkerStore(database)
    registry = WorkerRegistry()
    app.state.worker_store = store
    app.state.worker_registry = registry
    try:
        yield store, registry
    finally:
        await registry.shutdown()
        database.close()
        for attribute in ("worker_store", "worker_registry"):
            if hasattr(app.state, attribute):
                delattr(app.state, attribute)


def _tunnel(host: str = "127.0.0.1:8000",
            name: str = "gpu-box") -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=TunnelApp(app, name)),
                       base_url=f"http://{host}")


def _main(host: str) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app),
                       base_url=f"http://{host}")


def test_the_agent_routes_and_nothing_else():
    assert agent_route("websocket", "GET", "/ws/worker")
    assert agent_route("http", "POST", "/api/workers/enroll")
    assert agent_route("http", "GET", "/api/files/download/train.csv")
    assert agent_route("http", "GET", "/api/images/download/cat.png")
    assert agent_route("http", "GET", "/api/models/download/a/b.pt")
    assert agent_route("http", "POST", "/api/runs/r1/artifacts/upload")
    for kind, method, path in (
            ("http", "GET", "/api/auth/bootstrap"),
            ("http", "GET", "/"),
            ("http", "GET", "/index.html"),
            ("http", "GET", "/api/workers"),
            ("http", "GET", "/api/workers/enroll"),
            ("http", "POST", "/api/workers/ssh"),
            ("http", "GET", "/api/files/download/"),
            ("http", "GET", "/api/runs/r1/artifacts/upload"),
            ("http", "POST", "/api/runs"),
            ("websocket", "GET", "/ws/execution"),
            ("websocket", "GET", "/ws/worker/"),
            ("lifespan", "GET", "/ws/worker")):
        assert not agent_route(kind, method, path), (kind, method, path)


def test_only_the_listener_marks_a_request():
    assert tunnel_of({"type": "http", SCOPE_KEY: "gpu-box"}) == "gpu-box"
    assert tunnel_of({"type": "http"}) is None
    assert tunnel_of({"type": "http", SCOPE_KEY: 1}) is None
    assert tunnel_of(None) is None


@pytest.mark.parametrize("host", HOSTS)
async def test_the_session_bootstrap_does_not_exist_through_a_tunnel(host):
    """The hole this closes: through the forward, ``GET /api/auth/bootstrap``
    with ``Host: 127.0.0.1`` answered the session token."""
    async with _tunnel(host) as client:
        response = await client.get("/api/auth/bootstrap")
        assert response.status_code == 404
        assert session_token() not in response.text
        for method, path in (("GET", "/"), ("GET", "/api/workers"),
                             ("GET", "/api/runs"), ("GET", "/api/system/info"),
                             ("POST", "/api/nodes/reload"),
                             ("GET", "/api/workers/ssh")):
            answer = await client.request(
                method, path, headers={TOKEN_HEADER: session_token()})
            assert answer.status_code == 404, (method, path)
            assert session_token() not in answer.text


async def test_the_main_port_bootstrap_is_unchanged():
    """The main port's own guard: this server's hosts get the token, every
    other Host (a tunnel's old spelling included) a 421."""
    from app.core import auth

    async with _main("127.0.0.1:8000") as client:
        response = await client.get("/api/auth/bootstrap")
    assert response.status_code == 200
    assert response.json() == {"token": session_token()}
    assert not hasattr(auth, "tunnel_host_allows")
    assert not hasattr(auth, "allow_tunnel_port")
    for host in ("127.0.0.1:18000", "localhost:18000", "evil.example"):
        async with _main(host) as client:
            assert (await client.get("/api/auth/bootstrap")).status_code \
                == 421, host


@pytest.mark.parametrize("host", ["127.0.0.1:8000", "127.0.0.1:18000"])
async def test_enroll_through_a_tunnel_needs_an_enrollment_token(
        workers, host, caplog):
    store, _ = workers
    async with _tunnel(host) as client:
        refused = await client.post("/api/workers/enroll", json={
            "token": "nope", "name": "gpu-box"})
        assert refused.status_code == 401
        created = await store.create_enrollment()
        caplog.set_level(logging.WARNING)
        enrolled = await client.post("/api/workers/enroll", json={
            "token": created["token"], "name": "gpu-box"})
    assert enrolled.status_code == 200, enrolled.text
    assert enrolled.json()["worker_id"] == "gpu-box"
    # The hop across the network is ssh's: no plain-HTTP warning.
    assert "plain http://" not in caplog.text


async def test_the_worker_routes_need_a_worker_token_through_a_tunnel(
        workers):
    """Not the session token: through a tunnel even this machine's own
    token opens nothing."""
    async with _tunnel() as client:
        for headers in ({}, {TOKEN_HEADER: session_token()},
                        {TOKEN_HEADER: "forged"}):
            response = await client.get("/api/files/download/train.csv",
                                        headers=headers)
            assert response.status_code == 403
            assert "worker token" in response.json()["detail"]
            upload = await client.post("/api/runs/r1/artifacts/upload",
                                       headers=headers, content=b"x")
            assert upload.status_code == 403


async def test_each_tunnel_has_its_own_enrollment_rate_bucket(workers,
                                                              monkeypatch):
    from app.api import routes_workers

    monkeypatch.setattr(routes_workers, "enroll_limiter",
                        routes_workers.TokenBucket(burst=2, per_minute=1))
    body = {"token": "nope", "name": "b"}
    async with _tunnel(name="gpu-a") as tunnel:
        codes = [(await tunnel.post("/api/workers/enroll", json=body)
                  ).status_code for _ in range(3)]
    assert codes == [401, 401, 429]
    # Another SSH worker's listener: a bucket of its own.
    async with _tunnel(name="gpu-b") as other:
        assert (await other.post("/api/workers/enroll", json=body)
                ).status_code == 401
    # The same peer address on the main port: a bucket of its own too.
    async with _main("127.0.0.1:8000") as local:
        assert (await local.post("/api/workers/enroll", json=body)
                ).status_code == 401


async def test_a_real_listener_serves_the_agent_routes_only(workers):
    """The listener itself: a uvicorn on an ephemeral loopback port, in
    this event loop, without the app's lifespan."""
    listener = TunnelListener(app, "gpu-box")
    port = await listener.open()
    try:
        assert listener.port == port and port > 0
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}",
                                     timeout=10) as client:
            for host in HOSTS:
                response = await client.get("/api/auth/bootstrap",
                                            headers={"host": host})
                assert response.status_code == 404, host
                assert session_token() not in response.text
            enroll = await client.post("/api/workers/enroll", json={
                "token": "nope", "name": "gpu-box"})
            assert enroll.status_code == 401
    finally:
        await listener.close()
    with pytest.raises(httpx.ConnectError):
        async with httpx.AsyncClient(timeout=5) as client:
            await client.get(f"http://127.0.0.1:{port}/api/auth/bootstrap")
    await listener.close()  # twice: harmless
