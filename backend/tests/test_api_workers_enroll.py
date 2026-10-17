"""Worker enrollment over REST (#780).

``POST/GET/DELETE /api/workers/enrollments`` (session token),
``POST /api/workers/enroll`` (enrollment token only, rate-limited) and
``DELETE /api/workers/{id}`` (revoke), with a real ``Database`` and
``WorkerStore`` and an in-memory registry.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from app.api import routes_workers
from app.core import worker_store as ws_store
from app.core.auth import TOKEN_HEADER, session_token
from app.core.db import Database
from app.core.run_service import RunService
from app.core.run_store import RunStore
from app.core.worker_store import WorkerStore
from app.core.workers import STATUS_OFFLINE, WorkerRegistry
from app.main import app

BASE = "http://127.0.0.1:8000"
CPU = [{"value": "cpu", "kind": "cpu"}]


@pytest.fixture
async def env(tmp_path, monkeypatch):
    database = Database(tmp_path / "codefyui.db")
    database.connect()
    store = WorkerStore(database)
    registry = WorkerRegistry()
    monkeypatch.setattr(routes_workers, "enroll_limiter",
                        routes_workers.TokenBucket())
    app.state.db = database
    app.state.worker_store = store
    app.state.worker_registry = registry
    try:
        yield store, registry, database
    finally:
        await registry.shutdown()
        database.close()
        for attribute in ("db", "worker_store", "worker_registry",
                          "run_service"):
            if hasattr(app.state, attribute):
                delattr(app.state, attribute)


def _client(token: bool = True, base: str = BASE) -> AsyncClient:
    headers = {TOKEN_HEADER: session_token()} if token else {}
    return AsyncClient(transport=ASGITransport(app=app), base_url=base,
                       headers=headers)


async def _new_token(client: AsyncClient, **body: Any) -> dict[str, Any]:
    response = await client.post("/api/workers/enrollments", json=body)
    assert response.status_code == 200, response.text
    return response.json()


async def _enroll(token: str, name: str = "gpu-box",
                  devices: list[dict[str, Any]] | None = None) -> Any:
    async with _client(token=False) as anonymous:
        return await anonymous.post("/api/workers/enroll", json={
            "token": token, "name": name,
            "platform": {"system": "Linux"},
            "devices": CPU if devices is None else devices,
            "mounts": {}})


def _rows(database: Database, table: str) -> list[sqlite3.Row]:
    return database._conn.execute(f"SELECT * FROM {table}").fetchall()


# ── enrollment tokens ─────────────────────────────────────────────────────


async def test_a_new_token_comes_with_its_command_and_only_its_hash_is_kept(
        env):
    _, _, database = env
    async with _client() as client:
        created = await _new_token(client, label="lab box")
    token = created["token"]
    assert ws_store.ENROLLMENT_TOKEN_PATTERN.fullmatch(token)
    assert created["id"] == token.split(".")[0]
    assert created["control_url"] == BASE
    assert created["command"] == (
        f"cdui agent start --control {BASE} --token {token}")
    assert (created["uses_left"], created["state"], created["label"]) == (
        1, "pending", "lab box")
    [row] = _rows(database, "worker_enrollments")
    assert row["token_hash"] == ws_store.hash_secret(token)
    assert token not in [str(value) for value in tuple(row)]


async def test_the_command_uses_the_host_the_browser_used(env):
    async with _client(base="http://192.168.1.5:8000") as client:
        # The Host guard admits a LAN address only when the server binds a
        # wildcard (its interface IPs) or it is listed; emulate that here.
        from app.core import auth

        auth._ALLOWED_HOSTS.add("192.168.1.5:8000")
        try:
            created = await _new_token(client)
            listed = (await client.get("/api/workers/enrollments")).json()
        finally:
            auth._ALLOWED_HOSTS.discard("192.168.1.5:8000")
    assert created["command"].startswith(
        "cdui agent start --control http://192.168.1.5:8000 --token ")
    assert listed["control_url"] == "http://192.168.1.5:8000"
    assert "http://192.168.1.5:8000" in listed["other_machine_urls"]
    assert all("127.0.0.1" not in url and "localhost" not in url
               for url in listed["other_machine_urls"])
    assert listed["listening_on_loopback"] is True


@pytest.mark.parametrize("body", [{"ttl_s": 10}, {"ttl_s": 8 * 86400},
                                  {"uses": 0}, {"uses": 101},
                                  {"label": "x" * 101}])
async def test_out_of_range_requests_are_422(env, body):
    async with _client() as client:
        response = await client.post("/api/workers/enrollments", json=body)
    assert response.status_code == 422


async def test_the_token_routes_need_the_session_token(env):
    async with _client(token=False) as client:
        assert (await client.post("/api/workers/enrollments",
                                  json={})).status_code == 403
        assert (await client.get("/api/workers/enrollments")).status_code == 403
        assert (await client.delete(
            "/api/workers/enrollments/abcdef")).status_code == 403
        assert (await client.delete("/api/workers/box")).status_code == 403


async def test_no_store_is_503(env):
    del app.state.worker_store
    async with _client() as client:
        assert (await client.post("/api/workers/enrollments",
                                  json={})).status_code == 503
    assert (await _enroll("abcdef.0123456789abcdef")).status_code == 503


# ── enroll ────────────────────────────────────────────────────────────────


async def test_enroll_trades_the_token_for_a_worker_credential(env):
    store, registry, database = env
    async with _client() as client:
        created = await _new_token(client)
    response = await _enroll(created["token"])
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["worker_id"] == "gpu-box"
    worker_token = body["worker_token"]
    assert len(worker_token) >= 43
    assert await store.authenticate(worker_token) == "gpu-box"
    [row] = _rows(database, "workers")
    assert row["token_hash"] == ws_store.hash_secret(worker_token)
    assert row["name"] == row["id"] == "gpu-box"
    # Known at once, offline until it dials in, with the devices it reported.
    worker = registry.get("gpu-box")
    assert worker is not None and worker.status == STATUS_OFFLINE
    assert worker.enrolled and worker.device_values() == ["cpu"]
    described = worker.describe()
    assert (described["status"], described["connected_at"],
            described["last_heartbeat_at"], described["enrolled"]) == (
        "offline", None, None, True)


async def test_a_single_use_token_refuses_a_second_use(env):
    async with _client() as client:
        created = await _new_token(client)
        assert (await _enroll(created["token"])).status_code == 200
        second = await _enroll(created["token"], name="other-box")
        listed = (await client.get("/api/workers/enrollments")).json()
    assert second.status_code == 401
    assert "already been used" in second.json()["detail"]
    [entry] = listed["enrollments"]
    assert (entry["state"], entry["uses_left"]) == ("used", 0)
    assert "token" not in entry


async def test_a_token_with_two_uses_enrolls_two_workers(env):
    store, _, _ = env
    async with _client() as client:
        created = await _new_token(client, uses=2)
    assert (await _enroll(created["token"], "box-a")).status_code == 200
    assert (await _enroll(created["token"], "box-b")).status_code == 200
    assert (await _enroll(created["token"], "box-c")).status_code == 401
    assert [w.name for w in await store.list_workers()] == ["box-a", "box-b"]


async def test_an_expired_token_is_refused_and_listed_until_deleted(env):
    _, _, database = env
    async with _client() as client:
        created = await _new_token(client)
        database._conn.execute(
            "UPDATE worker_enrollments SET expires_at = ?",
            ("2000-01-01T00:00:00.000000Z",))
        refused = await _enroll(created["token"])
        listed = (await client.get("/api/workers/enrollments")).json()
        deleted = await client.delete(
            f"/api/workers/enrollments/{created['id']}")
        again = await client.delete(
            f"/api/workers/enrollments/{created['id']}")
        after = (await client.get("/api/workers/enrollments")).json()
    assert refused.status_code == 401 and "expired" in refused.json()["detail"]
    assert listed["enrollments"][0]["state"] == "expired"
    assert deleted.status_code == 200 and again.status_code == 404
    assert after["enrollments"] == []
    # A deleted (revoked) token is unknown.
    unknown = await _enroll(created["token"])
    assert unknown.status_code == 401
    assert "not valid" in unknown.json()["detail"]


@pytest.mark.parametrize("token, name, status", [
    ("not-an-enrollment-token", "box", 401),
    ("abcdef.0123456789abcdef", "Bad_Name", 400),
    ("abcdef.0123456789abcdef", "local", 400),
    ("abcdef.0123456789abcdef", "any", 400),
    ("abcdef.0123456789abcdef", "enroll", 400),
    ("abcdef.0123456789abcdef", "enrollments", 400),
])
async def test_bad_enrollments_are_refused(env, token, name, status):
    response = await _enroll(token, name)
    assert response.status_code == status


async def test_an_enrolled_name_needs_a_rekey_token(env):
    store, registry, database = env
    async with _client() as client:
        first = await _new_token(client)
        second = await _new_token(client)
        old = (await _enroll(first["token"])).json()["worker_token"]
        # An ordinary token cannot take the name, and spends nothing.
        taken = await _enroll(second["token"])
        assert taken.status_code == 409
        assert taken.json()["detail"] == (
            "worker gpu-box is already enrolled; remove it or create a "
            "re-key token for it")
        listed = (await client.get("/api/workers/enrollments")).json()
        assert {e["id"]: e["uses_left"] for e in listed["enrollments"]}[
            second["id"]] == 1
        # A token bound to the worker re-keys it, and names it in its command.
        rekey = await _new_token(client, worker="gpu-box")
        assert rekey["worker"] == "gpu-box"
        assert rekey["command"].endswith(f"--token {rekey['token']} "
                                         "--name gpu-box")
    new = (await _enroll(rekey["token"])).json()["worker_token"]
    assert old != new
    assert await store.authenticate(old) is None
    assert await store.authenticate(new) == "gpu-box"
    assert len(await store.list_workers()) == 1


async def test_a_rekey_token_enrolls_its_own_worker_only(env):
    async with _client() as client:
        first = await _new_token(client)
        await _enroll(first["token"])
        rekey = await _new_token(client, worker="gpu-box")
        unknown = await client.post("/api/workers/enrollments",
                                    json={"worker": "nobody"})
    assert unknown.status_code == 404
    other = await _enroll(rekey["token"], name="other-box")
    assert other.status_code == 403
    assert "re-keys worker gpu-box only" in other.json()["detail"]


async def test_a_name_held_by_a_live_same_host_agent_is_taken(env):
    _, registry, _ = env
    from app.core import worker_protocol as wp
    from app.core.workers import CREDENTIAL_SESSION

    closed = []

    async def noop(*_: Any) -> None:
        return None

    async def close() -> None:
        closed.append(True)

    registry.connect(wp.Hello(name="gpu-box", version="t"), send=noop,
                     close=close, credential=CREDENTIAL_SESSION)
    async with _client() as client:
        plain = await _new_token(client)
        assert (await _enroll(plain["token"])).status_code == 409
        rekey = await _new_token(client, worker="gpu-box")
    assert (await _enroll(rekey["token"])).status_code == 200
    # The session-token connection does not outlive the enrollment.
    await asyncio.sleep(0)
    worker = registry.get("gpu-box")
    assert closed and not worker.online and worker.enrolled


async def test_enroll_is_rate_limited_per_client(env, monkeypatch):
    monkeypatch.setattr(routes_workers, "enroll_limiter",
                        routes_workers.TokenBucket(burst=2, per_minute=1))
    statuses = [(await _enroll("abcdef.0123456789abcdef")).status_code
                for _ in range(3)]
    assert statuses == [401, 401, 429]
    limited = await _enroll("abcdef.0123456789abcdef")
    assert int(limited.headers["Retry-After"]) >= 1
    assert "too many" in limited.json()["detail"]


def test_the_token_bucket_refills_and_keeps_the_recent_keys():
    now = [0.0]
    bucket = routes_workers.TokenBucket(burst=2, per_minute=60,
                                        clock=lambda: now[0], max_keys=2)
    assert bucket.take("a") == 0.0 and bucket.take("a") == 0.0
    assert bucket.take("a") == pytest.approx(1.0)
    now[0] = 1.0
    assert bucket.take("a") == 0.0
    bucket.take("b")
    bucket.take("a")  # "a" is now the most recently used
    bucket.take("c")  # a third key evicts the least recently used: "b"
    assert list(bucket._buckets) == ["a", "c"]
    # Even when every key is still limited, the table stays bounded.
    for key in ("d", "e", "f"):
        bucket.take(key)
    assert len(bucket._buckets) == 2


async def test_no_secret_reaches_the_log(env, caplog):
    caplog.set_level(logging.DEBUG)
    async with _client() as client:
        created = await _new_token(client)
        await client.get("/api/workers/enrollments")
    body = (await _enroll(created["token"])).json()
    await _enroll(created["token"])  # refused: logged, without the token
    assert created["token"] not in caplog.text
    assert created["token"].split(".")[1] not in caplog.text
    assert body["worker_token"] not in caplog.text
    assert created["id"] in caplog.text


async def test_enrolling_over_plain_http_from_another_machine_is_warned(
        env, caplog):
    async with _client() as client:
        created = await _new_token(client)
    transport = ASGITransport(app=app, client=("10.0.0.9", 5000))
    async with AsyncClient(transport=transport, base_url=BASE) as remote:
        response = await remote.post("/api/workers/enroll", json={
            "token": created["token"], "name": "far-box"})
    assert response.status_code == 200
    assert "plain http:// from 10.0.0.9" in caplog.text
    assert response.json()["worker_token"] not in caplog.text


# ── the certificate fingerprint for the Add worker dialog (#786) ──────────


async def test_without_a_cert_file_the_dialog_gets_the_openssl_command(env):
    async with _client(base="https://127.0.0.1:8000") as client:
        listed = (await client.get("/api/workers/enrollments")).json()
        created = await _new_token(client)
    assert listed["tls_fingerprint"] is None
    assert listed["fingerprint_command"].startswith("openssl x509 ")
    assert "--ca-fingerprint" not in created["command"]


async def test_a_readable_cert_file_gives_the_fingerprint(env, tmp_path,
                                                          monkeypatch):
    import trustme

    from app.config import settings
    from app.core import tls_fingerprint

    leaf = trustme.CA().issue_cert("127.0.0.1")
    path = tmp_path / "fullchain.pem"
    path.write_bytes(b"".join(blob.bytes() for blob in leaf.cert_chain_pems))
    expected = tls_fingerprint.spki_fingerprint(
        tls_fingerprint.pem_certificates(leaf.cert_chain_pems[0].bytes())[0])
    monkeypatch.setattr(settings, "TLS_CERT_FILE", str(path))
    async with _client(base="https://127.0.0.1:8000") as client:
        listed = (await client.get("/api/workers/enrollments")).json()
        created = await _new_token(client)
    async with _client() as plain:
        over_http = await _new_token(plain)
    assert listed["tls_fingerprint"] == expected
    assert created["command"].endswith(f" --ca-fingerprint {expected}")
    # Over http the agent has no certificate to check.
    assert "--ca-fingerprint" not in over_http["command"]

    monkeypatch.setattr(settings, "TLS_CERT_FILE", str(tmp_path / "gone"))
    async with _client() as client:
        listed = (await client.get("/api/workers/enrollments")).json()
    assert listed["tls_fingerprint"] is None


async def test_the_cert_file_is_read_once_per_version_and_warned_once(
        env, tmp_path, monkeypatch, caplog):
    import os

    import trustme

    from app.config import settings
    from app.core import tls_fingerprint

    reads: list[bytes] = []
    real = tls_fingerprint.pem_certificates

    def counting(pem: bytes) -> list[bytes]:
        reads.append(pem)
        return real(pem)

    monkeypatch.setattr(routes_workers.tls_fp, "pem_certificates", counting)
    path = tmp_path / "fullchain.pem"
    path.write_bytes(b"not a certificate")
    monkeypatch.setattr(settings, "TLS_CERT_FILE", str(path))
    async with _client() as client:
        for _ in range(3):
            listed = (await client.get("/api/workers/enrollments")).json()
        assert listed["tls_fingerprint"] is None
        assert len(reads) == 1
        assert caplog.text.count("gives no certificate fingerprint") == 1

        # A new version of the file (a renewal): read again, once.
        leaf = trustme.CA().issue_cert("127.0.0.1")
        path.write_bytes(leaf.cert_chain_pems[0].bytes())
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
        for _ in range(2):
            listed = (await client.get("/api/workers/enrollments")).json()
    assert listed["tls_fingerprint"] == tls_fingerprint.spki_fingerprint(
        real(leaf.cert_chain_pems[0].bytes())[0])
    assert len(reads) == 2
    assert caplog.text.count("gives no certificate fingerprint") == 1


async def test_a_token_can_be_made_for_another_address(env):
    async with _client() as client:
        created = await _new_token(client, control_url="https://cp.lab/sub/")
        bad = await client.post("/api/workers/enrollments",
                                json={"control_url": "http://x;rm -rf /"})
    assert created["control_url"] == "https://cp.lab/sub"
    assert created["command"].startswith(
        "cdui agent start --control https://cp.lab/sub --token ")
    assert bad.status_code == 422


async def test_the_agent_command_route_uses_the_same_builder(
        env, tmp_path, monkeypatch):
    import trustme

    from app.config import settings

    leaf = trustme.CA().issue_cert("127.0.0.1")
    path = tmp_path / "fullchain.pem"
    path.write_bytes(leaf.cert_chain_pems[0].bytes())
    monkeypatch.setattr(settings, "TLS_CERT_FILE", str(path))
    token = "abcdef.0123456789abcdef"
    async with _client() as client:
        tls = (await client.post("/api/workers/agent-command", json={
            "control_url": "https://[::1]:8443", "token": token,
            "worker": "gpu-box"})).json()
        plain = (await client.post("/api/workers/agent-command", json={
            "control_url": "http://10.0.0.7:8000/", "token": token})).json()
        refused = [
            (await client.post("/api/workers/agent-command",
                               json=body)).status_code
            for body in ({"control_url": "ftp://x", "token": token},
                         {"control_url": "http://x:80$(id)", "token": token},
                         {"control_url": "http://x", "token": "w" * 43},
                         {"control_url": "http://x", "token": token,
                          "worker": "Bad Name"})]
    async with _client(token=False) as anonymous:
        unauthorised = await anonymous.post("/api/workers/agent-command", json={
            "control_url": "http://x", "token": token})
    fingerprint = routes_workers._read_tls_fingerprint()
    assert tls["command"] == (
        f"cdui agent start --control https://[::1]:8443 --token {token} "
        f"--name gpu-box --ca-fingerprint {fingerprint}")
    assert plain == {"control_url": "http://10.0.0.7:8000", "command":
                     f"cdui agent start --control http://10.0.0.7:8000 "
                     f"--token {token}"}
    assert refused == [422, 422, 422, 422]
    assert unauthorised.status_code in (401, 403)


# ── remove (revoke) ───────────────────────────────────────────────────────


async def test_remove_revokes_and_forgets(env):
    store, registry, _ = env
    async with _client() as client:
        created = await _new_token(client)
        token = (await _enroll(created["token"])).json()["worker_token"]
        removed = await client.delete("/api/workers/gpu-box")
        again = await client.delete("/api/workers/gpu-box")
        listed = (await client.get("/api/workers")).json()["workers"]
    assert removed.json() == {"id": "gpu-box", "revoked": True,
                              "failed_runs": []}
    assert again.status_code == 404
    assert listed == [] and registry.get("gpu-box") is None
    assert await store.authenticate(token) is None


async def test_remove_forgets_a_session_token_worker_too(env):
    _, registry, _ = env
    from app.core import worker_protocol as wp

    async def noop(*_: Any) -> None:
        return None

    worker, _ = registry.connect(wp.Hello(name="same-box", version="t"),
                                 send=noop, close=noop)
    registry.disconnect(worker)
    async with _client() as client:
        removed = await client.delete("/api/workers/same-box")
    assert removed.json() == {"id": "same-box", "revoked": False,
                              "failed_runs": []}
    assert registry.get("same-box") is None


# ── offline workers after a restart ───────────────────────────────────────


async def test_a_restart_knows_enrolled_workers_and_runs_queue_for_them(
        env):
    store, _, database = env
    async with _client() as client:
        created = await _new_token(client)
    await _enroll(created["token"], devices=CPU + [
        {"value": "cuda:0", "kind": "cuda", "index": 0}])

    # "Restart": a fresh registry loaded from the table, as the lifespan does.
    registry = WorkerRegistry()
    loaded = registry.load_known(await store.list_workers())
    assert [w.name for w in loaded] == ["gpu-box"]
    assert registry.load_known(await store.list_workers()) == []
    worker = registry.get("gpu-box")
    assert worker.status == STATUS_OFFLINE and not worker.online
    assert worker.device_values() == ["cpu", "cuda:0"]
    service = RunService(RunStore(database), workers=registry,
                         shutdown_grace_s=5.0)
    app.state.worker_registry = registry
    app.state.run_service = service
    try:
        async with _client() as client:
            response = await client.post("/api/runs", json={
                "graph": {"nodes": [{"id": "s", "type": "Start",
                                     "data": {"params": {}}}],
                          "edges": []},
                "options": {"target": {"worker": "gpu-box"},
                            "device": "cuda:0"}})
            assert response.status_code == 200, response.text
            run_id = response.json()["run_id"]
            deadline = time.monotonic() + 5
            while True:
                row = (await client.get(f"/api/runs/{run_id}")).json()
                if row["queue_reason"] or time.monotonic() > deadline:
                    break
                await asyncio.sleep(0.02)
        assert row["status"] == "queued"
        assert row["queue_reason"] == "waiting for worker gpu-box to reconnect"
        # Removing the worker fails what waits for it, at once.
        async with _client() as client:
            removed = (await client.delete("/api/workers/gpu-box")).json()
            row = (await client.get(f"/api/runs/{run_id}")).json()
            events = (await client.get(
                f"/api/runs/{run_id}/events")).json()["events"]
        assert removed["failed_runs"] == [run_id]
        assert (row["status"], row["error"]) == (
            "failed", "worker gpu-box was removed")
        assert events[-1]["type"] == "execution_error"
        assert events[-1]["payload"] == {"error": "worker gpu-box was removed"}
    finally:
        await service.shutdown()
        await registry.shutdown()


async def test_a_stored_device_report_that_cannot_be_read_is_left_out(env):
    store, _, database = env
    async with _client() as client:
        created = await _new_token(client)
    await _enroll(created["token"])
    database._conn.execute(
        "UPDATE workers SET devices_json = ?, platform = ?",
        ('[{"value": "cpu", "kind": "cpu"}, {"kind": 5}]', "not json"))
    registry = WorkerRegistry()
    registry.load_known(await store.list_workers())
    worker = registry.get("gpu-box")
    assert worker.device_values() == ["cpu"] and worker.platform == {}


async def test_only_post_enroll_skips_the_session_token(env):
    async with _client(token=False) as client:
        # DELETE /api/workers/enroll is the remove route for a worker named
        # "enroll" (a reserved name): it still needs the session token.
        assert (await client.delete("/api/workers/enroll")).status_code == 403
        assert (await client.put("/api/workers/enroll",
                                 json={})).status_code == 403


def test_one_loopback_rule():
    from app.core.auth import is_loopback_peer

    for address in ("127.0.0.1", "127.8.9.1", "::1", "[::1]", "localhost",
                    "::ffff:127.0.0.1", "LOCALHOST"):
        assert is_loopback_peer(address), address
    for address in ("10.0.0.1", "", None, "::ffff:10.0.0.1", "128.0.0.1"):
        assert not is_loopback_peer(address), address
