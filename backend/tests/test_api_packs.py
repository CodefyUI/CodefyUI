"""The Package Center's REST surface: ``/api/packs`` and the job long poll.

Three promises are tested here rather than in the service tests, because they
only exist once there is a request in front of the job:

* **the read is open, the writes are not.** ``GET /api/packs`` is what the
  editor polls to draw the panel, so it needs no session token; every route
  that CHANGES something needs one, and is additionally refused outright when
  the server is not bound to loopback -- installing packages is not a thing a
  stranger on the LAN gets to start.
* **the response keys are a contract.** The SPA is written against them, so a
  renamed key is a broken panel; the shape tests below are deliberately
  key-set equality rather than "contains".
* **the long poll behaves like the runs one.** It returns at once on a
  finished job, wakes on the next event rather than on a timer, and refuses
  an out-of-range ``wait`` with 422.

No test here installs anything: each one injects a scripted flow into a
``PackService`` on ``app.state`` and drives it from the test thread.
"""

from __future__ import annotations

import asyncio
import subprocess
import threading
import time

import pytest
from fastapi.routing import APIRoute
from httpx import ASGITransport, AsyncClient

from app.api import routes_packs
from app.config import settings
from app.core.auth import TOKEN_HEADER, session_token
from app.core.packs import download, flows, restart, runner, state
from app.core.packs.errors import (
    PackInstallError,
    PackInsufficientDisk,
    PackNeedsRestart,
)
from app.core.packs.service import PackService
from app.main import _AUTH_EXEMPT_PREFIXES, _prefix_exempt, app

from tests.test_packs_service import ScriptedFlow, types_of, wait_started

BASE_URL = f"http://127.0.0.1:{settings.PORT}"
SENTENCE = "sentence-embeddings"

#: Every top-level key ``GET /api/packs`` returns. Additive is a decision:
#: a new key should break this once, on purpose.
TOP_KEYS = {"packs", "active_job", "last_restart_job", "remote_install_allowed",
            "launch_mode", "gpu"}
PACK_KEYS = {"id", "title", "description", "install_mode", "status",
             "pip_ready", "usable", "depends_on", "blocked_by", "pip", "items",
             "size_bytes_total", "install_command"}
ITEM_KEYS = {"id", "kind", "repo_id", "url", "size_bytes", "license", "status"}
GPU_KEYS = {"detected_label", "recommended_variant", "installed_variant",
            "variants", "install_command"}


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """A throwaway cache root and a cold probe cache for every test."""
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path))
    state.invalidate()
    yield tmp_path
    state.invalidate()


@pytest.fixture
def flow() -> ScriptedFlow:
    return ScriptedFlow()


@pytest.fixture
async def pack_service(flow):
    """A PackService on ``app.state``, torn down like the lifespan does.

    The lifespan does not run under httpx's ASGITransport, so the service is
    installed by hand (the ``run_service`` precedent in test_api_runs.py).
    """
    service = PackService(run_flow=flow)
    previous = getattr(app.state, "pack_service", None)
    app.state.pack_service = service
    try:
        yield service
    finally:
        await service.shutdown()
        if previous is None:
            if hasattr(app.state, "pack_service"):
                delattr(app.state, "pack_service")
        else:
            app.state.pack_service = previous


@pytest.fixture
async def client(pack_service):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url=BASE_URL,
        headers={TOKEN_HEADER: session_token()},
    ) as http:
        yield http


@pytest.fixture
async def anon_client(pack_service):
    """No session token: what a browser that never bootstrapped would send."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url=BASE_URL,
    ) as http:
        yield http


async def start_install(client, pack_id=SENTENCE, **body) -> str:
    response = await client.post(f"/api/packs/{pack_id}/install", json=body)
    assert response.status_code == 202, response.text
    return response.json()["job_id"]


async def drain(client, job_id: str, *, timeout: float = 20.0
                ) -> tuple[list[dict], str]:
    """Every event of a job over HTTP, waiting for it to finish."""
    limit = 500
    events: list[dict] = []
    cursor, status = 0, "running"
    deadline = time.monotonic() + timeout
    while True:
        if time.monotonic() > deadline:
            raise AssertionError(f"job {job_id} never finished ({status})")
        response = await client.get(
            f"/api/packs/jobs/{job_id}/events",
            params={"cursor": cursor, "wait": 1.0, "limit": limit})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["job_id"] == job_id
        events.extend(body["events"])
        cursor, status = body["cursor"], body["status"]
        if status != "running" and len(body["events"]) < limit:
            return events, status


# ── GET /api/packs ────────────────────────────────────────────────────────


async def test_list_packs_is_open_get_and_has_contract_keys(anon_client):
    response = await anon_client.get("/api/packs")
    assert response.status_code == 200, response.text
    body = response.json()

    assert set(body) == TOP_KEYS
    assert body["active_job"] is None
    assert body["last_restart_job"] is None
    assert isinstance(body["remote_install_allowed"], bool)
    assert body["launch_mode"] in {"start", "dev", "unknown"}
    assert set(body["gpu"]) == GPU_KEYS
    assert body["gpu"]["recommended_variant"] in body["gpu"]["variants"]

    ids = [pack["id"] for pack in body["packs"]]
    assert ids == [SENTENCE, "word-vectors", "rag", "gpu-torch"]
    for pack in body["packs"]:
        assert set(pack) == PACK_KEYS, pack["id"]
        assert pack["status"] in {"installed", "partial", "not_installed",
                                  "installing", "needs_restart"}
        assert isinstance(pack["pip_ready"], bool)
        assert isinstance(pack["usable"], bool)
        assert all(set(spec) == {"spec"} for spec in pack["pip"])
        assert isinstance(pack["size_bytes_total"], int)
        for item in pack["items"]:
            assert set(item) == ITEM_KEYS, item
            assert item["status"] in {"present", "missing", "downloading"}


async def test_a_data_only_pack_with_nothing_downloaded_is_not_installed(
        anon_client):
    """word-vectors has no packages to probe, so "pip is ready" is vacuously
    true for it -- and must not be read as progress."""
    body = (await anon_client.get("/api/packs")).json()
    vectors = next(p for p in body["packs"] if p["id"] == "word-vectors")
    assert vectors["pip_ready"] is True
    assert vectors["status"] == "not_installed"
    assert vectors["size_bytes_total"] == 66_000_000
    assert vectors["install_command"] == "cdui packs install word-vectors"


async def test_gpu_torch_advertises_the_cli_install_command(anon_client):
    body = (await anon_client.get("/api/packs")).json()
    gpu_pack = next(p for p in body["packs"] if p["id"] == "gpu-torch")
    assert gpu_pack["install_mode"] == "restart"
    assert gpu_pack["install_command"] == body["gpu"]["install_command"]
    assert gpu_pack["install_command"].startswith("cdui install --gpu ")


async def test_rag_reports_what_blocks_it(anon_client, monkeypatch):
    monkeypatch.setattr(state, "pip_ready",
                        lambda pack: pack.pack_id != SENTENCE)
    state.invalidate()
    body = (await anon_client.get("/api/packs")).json()
    rag = next(p for p in body["packs"] if p["id"] == "rag")
    assert rag["depends_on"] == [SENTENCE]
    assert rag["blocked_by"] == [SENTENCE]


async def test_pack_status_reflects_installing_and_item_downloading(
        client, flow):
    job_id = await start_install(client, items=["bge-small-zh-v1.5"])
    flow.send({"type": "step_started", "step": "download:bge-small-zh-v1.5",
               "label": "Downloading"})
    # Wait for the step event to land rather than sleeping for it.
    await client.get(f"/api/packs/jobs/{job_id}/events",
                     params={"cursor": 1, "wait": 5})

    body = (await client.get("/api/packs")).json()
    assert body["active_job"] == {"job_id": job_id, "pack_id": SENTENCE}
    pack = next(p for p in body["packs"] if p["id"] == SENTENCE)
    assert pack["status"] == "installing"
    statuses = {item["id"]: item["status"] for item in pack["items"]}
    assert statuses["bge-small-zh-v1.5"] == "downloading"
    assert statuses["all-MiniLM-L6-v2"] == "missing"

    flow.finish()
    await drain(client, job_id)
    assert (await client.get("/api/packs")).json()["active_job"] is None


# ── auth and the remote gate ──────────────────────────────────────────────


async def test_install_requires_session_token(anon_client):
    response = await anon_client.post(f"/api/packs/{SENTENCE}/install", json={})
    assert response.status_code == 403
    assert TOKEN_HEADER in response.json()["detail"]


async def test_packs_router_is_not_auth_exempt():
    # /api/packs relies on the auth_guard middleware, exactly like /api/runs.
    # An exemption here would silently drop authentication from every
    # mutating pack route, because none of them declares a route-level
    # dependency (see test_auth_drift.py for what an exempt prefix owes).
    assert "/api/packs" not in _AUTH_EXEMPT_PREFIXES
    assert not _prefix_exempt("/api/packs")
    assert not _prefix_exempt(f"/api/packs/{SENTENCE}/install")
    assert not _prefix_exempt("/api/packs/jobs/abc/events")


async def test_remote_install_blocked_403_and_allowed_with_setting(
        client, flow, monkeypatch):
    monkeypatch.setattr(settings, "HOST", "0.0.0.0")
    monkeypatch.setattr(settings, "ALLOW_REMOTE_PACK_INSTALL", False)

    listing = (await client.get("/api/packs")).json()
    assert listing["remote_install_allowed"] is False

    refused = await client.post(f"/api/packs/{SENTENCE}/install", json={})
    assert refused.status_code == 403
    assert refused.json()["detail"] == (
        "Installing packs is only allowed from the computer that runs the "
        "server. Set CODEFYUI_ALLOW_REMOTE_PACK_INSTALL=1 to override.")

    monkeypatch.setattr(settings, "ALLOW_REMOTE_PACK_INSTALL", True)
    assert (await client.get("/api/packs")).json()["remote_install_allowed"]
    flow.script()
    job_id = await start_install(client)
    await drain(client, job_id)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
async def test_loopback_hosts_allow_installing(client, host, monkeypatch):
    monkeypatch.setattr(settings, "HOST", host)
    monkeypatch.setattr(settings, "ALLOW_REMOTE_PACK_INSTALL", False)
    assert (await client.get("/api/packs")).json()["remote_install_allowed"]


async def test_remote_gate_covers_every_mutating_packs_route(
        client, monkeypatch):
    """A future mutating route that forgets the gate fails here."""
    monkeypatch.setattr(settings, "HOST", "192.168.1.20")
    monkeypatch.setattr(settings, "ALLOW_REMOTE_PACK_INSTALL", False)

    mutating = [
        (method, route.path)
        for route in routes_packs.router.routes
        if isinstance(route, APIRoute)
        for method in sorted(route.methods)
        if method not in {"GET", "HEAD", "OPTIONS"}
    ]
    assert len(mutating) >= 3, "router walk is broken"
    for method, path in mutating:
        concrete = (path.replace("{pack_id}", SENTENCE)
                        .replace("{item_id}", "all-MiniLM-L6-v2")
                        .replace("{job_id}", "does-not-matter"))
        response = await client.request(method, concrete, json={})
        assert response.status_code == 403, f"{method} {concrete}"


# ── POST /api/packs/{id}/install ──────────────────────────────────────────


async def test_install_unknown_pack_404(client):
    response = await client.post("/api/packs/no-such-pack/install", json={})
    assert response.status_code == 404
    assert "no-such-pack" in response.json()["detail"]


async def test_install_unknown_item_400(client):
    response = await client.post(f"/api/packs/{SENTENCE}/install",
                                 json={"items": ["not-a-model"]})
    assert response.status_code == 400
    assert "not-a-model" in response.json()["detail"]


async def test_install_rejects_extra_fields_422(client):
    """The client can never hand the installer a pip spec of its own."""
    response = await client.post(f"/api/packs/{SENTENCE}/install",
                                 json={"spec": "evil-package==6.6.6"})
    assert response.status_code == 422


async def test_install_rejects_an_unknown_mode_422(client):
    response = await client.post(f"/api/packs/{SENTENCE}/install",
                                 json={"mode": "sideways"})
    assert response.status_code == 422


async def test_install_rag_blocked_until_sentence_embeddings_ready(
        client, monkeypatch):
    monkeypatch.setattr(state, "pip_ready",
                        lambda pack: pack.pack_id != SENTENCE)
    state.invalidate()

    response = await client.post("/api/packs/rag/install", json={})
    assert response.status_code == 400
    body = response.json()
    assert body["blocked_by"] == [SENTENCE]
    assert SENTENCE in body["detail"]


async def test_one_job_at_a_time_409(client, flow):
    job_id = await start_install(client)
    await wait_started(flow)

    busy = await client.post("/api/packs/word-vectors/install", json={})
    assert busy.status_code == 409
    assert busy.json()["job_id"] == job_id

    flow.finish()
    await drain(client, job_id)


async def test_insufficient_disk_is_507(client, monkeypatch):
    def _no_room(items):
        raise PackInsufficientDisk("not enough free disk space: 700 MB needed",
                                   needed=700_000_000, free=1_000_000)

    monkeypatch.setattr(download, "check_disk", _no_room)
    response = await client.post(f"/api/packs/{SENTENCE}/install",
                                 json={"items": ["all-MiniLM-L6-v2"]})
    assert response.status_code == 507
    body = response.json()
    assert body["needed"] == 700_000_000
    assert body["free"] == 1_000_000
    assert "disk" in body["detail"]


async def test_restart_mode_refused_with_command_409(client):
    gpu = await client.post("/api/packs/gpu-torch/install",
                            json={"mode": "restart", "variant": "cu128"})
    assert gpu.status_code == 409
    assert gpu.json()["command"] == "cdui install --gpu cu128"

    live = await client.post(f"/api/packs/{SENTENCE}/install",
                             json={"mode": "restart"})
    assert live.status_code == 409
    assert live.json()["command"] == f"cdui packs install {SENTENCE}"


async def test_gpu_torch_install_without_mode_is_refused_with_command_409(
        client, flow):
    """The PACK's mode decides, not the client's.

    gpu-torch has no pip specs, no probe modules and no items, so a live
    install of it would run every step successfully and change nothing -- and
    the panel would report that the GPU PyTorch install had finished.
    """
    response = await client.post("/api/packs/gpu-torch/install", json={})
    assert response.status_code == 409
    assert response.json()["command"].startswith("cdui install --gpu ")
    assert not flow.started.is_set(), "no job may have been started"
    assert (await client.get("/api/packs")).json()["active_job"] is None


async def test_gpu_pack_refusal_probes_off_the_event_loop_thread(
        client, monkeypatch):
    """Refusing gpu-torch has to NAME the wheel this machine wants.

    Answering that from a cold cache runs nvidia-smi with a five second
    timeout, and since the refusal keys off the pack's own mode it is now
    reached by every POST to gpu-torch -- so the ordinary path would stall
    the loop, not just the rare explicit `mode="restart"` request.
    """
    loop_thread = threading.get_ident()
    seen: list[int] = []

    def _detect() -> tuple[str, str]:
        seen.append(threading.get_ident())
        return ("CPU only", "cpu")

    monkeypatch.setattr(restart, "_detected", None)   # cold cache
    monkeypatch.setattr(restart, "detect_gpu", _detect)

    response = await client.post("/api/packs/gpu-torch/install", json={})

    assert response.status_code == 409
    assert response.json()["command"] == "cdui install --gpu cpu"
    assert seen, "the refusal never asked which wheel this machine wants"
    assert all(ident != loop_thread for ident in seen), (
        "the GPU probe ran on the event-loop thread")
    assert len(seen) == 1


async def test_install_rejects_an_unknown_variant_422(client):
    """``variant`` comes back as a command line to paste into a shell."""
    for bogus in ("; rm -rf /", "cu999", "", "cpu ; whoami"):
        response = await client.post(
            "/api/packs/gpu-torch/install",
            json={"mode": "restart", "variant": bogus})
        assert response.status_code == 422, bogus


async def test_install_command_for_refuses_an_unknown_variant():
    from app.core.packs.catalog import get_pack

    with pytest.raises(ValueError, match="unknown torch variant"):
        restart.install_command_for(get_pack("gpu-torch"), "; rm -rf /")
    # A known one is still fine, and a pack that ignores the variant too.
    assert restart.install_command_for(
        get_pack("gpu-torch"), "cu128") == "cdui install --gpu cu128"


async def test_install_emits_job_started_then_flow_events_then_job_done(
        client, flow):
    flow.script(
        {"type": "step_started", "step": "pip", "label": "Installing"},
        {"type": "log", "line": "collecting sentence-transformers"},
        {"type": "step_done", "step": "pip"},
    )
    job_id = await start_install(client, items=["all-MiniLM-L6-v2"])
    events, status = await drain(client, job_id)

    assert status == "done"
    assert types_of(events) == [
        "job_started", "step_started", "log", "step_done", "job_done"]
    assert events[0]["items"] == ["all-MiniLM-L6-v2"]
    cursors = [event["cursor"] for event in events]
    assert cursors == sorted(cursors) == list(range(1, len(events) + 1))
    assert all(event["ts"] for event in events)


async def test_failed_flow_ends_with_job_failed_message_and_hint(client, flow):
    flow.fail(PackInstallError("installing failed (uv exited 1)",
                               hint="No solution found"))
    job_id = await start_install(client)
    events, status = await drain(client, job_id)

    assert status == "failed"
    assert events[-1]["type"] == "job_failed"
    assert events[-1]["message"] == "installing failed (uv exited 1)"
    assert events[-1]["hint"] == "No solution found"


async def test_needs_restart_flow_ends_with_needs_restart_status_and_command(
        client, flow):
    flow.fail(PackNeedsRestart("cannot replace a package already in use",
                               command="cdui packs install gpu-torch --restart"))
    job_id = await start_install(client)
    events, status = await drain(client, job_id)

    assert status == "needs_restart"
    assert events[-1]["type"] == "needs_restart"
    assert events[-1]["command"] == "cdui packs install gpu-torch --restart"


# ── the job routes ────────────────────────────────────────────────────────


async def test_events_paginate_by_cursor_and_limit(client, flow):
    flow.script(*[{"type": "log", "line": str(n)} for n in range(4)])
    job_id = await start_install(client)
    await drain(client, job_id)

    first = (await client.get(f"/api/packs/jobs/{job_id}/events",
                              params={"cursor": 0, "limit": 2})).json()
    assert len(first["events"]) == 2
    assert first["cursor"] == first["events"][-1]["cursor"] == 2

    second = (await client.get(f"/api/packs/jobs/{job_id}/events",
                               params={"cursor": first["cursor"],
                                       "limit": 2})).json()
    assert [event["cursor"] for event in second["events"]] == [3, 4]
    assert second["status"] == "done"

    tail = (await client.get(f"/api/packs/jobs/{job_id}/events",
                             params={"cursor": 999})).json()
    assert tail["events"] == []
    assert tail["cursor"] == 999   # never moves backwards
    assert tail["status"] == "done"


async def test_events_long_poll_returns_on_terminal_without_waiting(
        client, flow):
    flow.script({"type": "log", "line": "only"})
    job_id = await start_install(client)
    events, _ = await drain(client, job_id)

    started = time.monotonic()
    response = await client.get(
        f"/api/packs/jobs/{job_id}/events",
        params={"cursor": events[-1]["cursor"], "wait": 5})
    elapsed = time.monotonic() - started

    assert response.status_code == 200
    assert response.json()["events"] == []
    assert response.json()["status"] == "done"
    assert elapsed < 2.0, "a finished job must not hold the request open"


async def test_events_long_poll_wakes_on_next_event(client, flow):
    job_id = await start_install(client)
    await wait_started(flow)

    poll = asyncio.create_task(client.get(
        f"/api/packs/jobs/{job_id}/events", params={"cursor": 1, "wait": 5}))
    await asyncio.sleep(0.05)
    assert not poll.done(), "the request must actually park"

    started = time.monotonic()
    flow.send({"type": "log", "line": "woken"})
    response = await asyncio.wait_for(poll, 10)
    elapsed = time.monotonic() - started

    assert types_of(response.json()["events"]) == ["log"]
    assert elapsed < 3.0, "the poll waited for its deadline, not for the edge"

    flow.finish()
    await drain(client, job_id)


@pytest.mark.parametrize("params", [
    {"wait": 61}, {"wait": -1}, {"cursor": -1}, {"limit": 0}, {"limit": 2001},
])
async def test_events_reject_out_of_range_params_422(client, flow, params):
    flow.script()
    job_id = await start_install(client)
    await drain(client, job_id)

    response = await client.get(f"/api/packs/jobs/{job_id}/events",
                                params=params)
    assert response.status_code == 422


async def test_events_unknown_job_404(client):
    response = await client.get("/api/packs/jobs/nope/events")
    assert response.status_code == 404
    assert "nope" in response.json()["detail"]


async def test_cancel_running_job_emits_job_cancelled_and_second_is_false(
        client, flow):
    job_id = await start_install(client)
    await wait_started(flow)

    first = await client.post(f"/api/packs/jobs/{job_id}/cancel")
    assert first.status_code == 200
    assert first.json() == {"job_id": job_id, "cancelled": True}

    events, status = await drain(client, job_id)
    assert status == "cancelled"
    assert events[-1]["type"] == "job_cancelled"

    second = await client.post(f"/api/packs/jobs/{job_id}/cancel")
    assert second.json() == {"job_id": job_id, "cancelled": False}


async def test_cancel_unknown_job_404(client):
    response = await client.post("/api/packs/jobs/nope/cancel")
    assert response.status_code == 404


# ── DELETE /api/packs/{id}/items/{item_id} ────────────────────────────────


async def test_delete_item_removes_and_refuses_while_installing(
        client, flow, monkeypatch):
    removed_calls: list[tuple[str, str]] = []

    def _fake_remove(pack, item_id):
        removed_calls.append((pack.pack_id, item_id))
        return True

    monkeypatch.setattr(flows, "remove_item", _fake_remove)

    response = await client.delete(
        f"/api/packs/{SENTENCE}/items/all-MiniLM-L6-v2")
    assert response.status_code == 200
    assert response.json() == {"pack_id": SENTENCE,
                               "item_id": "all-MiniLM-L6-v2", "removed": True}
    assert removed_calls == [(SENTENCE, "all-MiniLM-L6-v2")]

    job_id = await start_install(client)
    await wait_started(flow)
    busy = await client.delete(f"/api/packs/{SENTENCE}/items/all-MiniLM-L6-v2")
    assert busy.status_code == 409
    assert busy.json()["job_id"] == job_id
    assert removed_calls == [(SENTENCE, "all-MiniLM-L6-v2")]

    # Another pack is not blocked by this job.
    other = await client.delete("/api/packs/word-vectors/items/glove-50d")
    assert other.status_code == 200

    flow.finish()
    await drain(client, job_id)


async def test_delete_item_reports_bytes_left_behind(client, monkeypatch):
    """``removed=False`` is a real answer, not "nothing to do": on Windows a
    file another process holds open survives the delete."""
    monkeypatch.setattr(flows, "remove_item", lambda pack, item_id: False)
    response = await client.delete("/api/packs/word-vectors/items/glove-50d")
    assert response.status_code == 200
    assert response.json()["removed"] is False


async def test_delete_unknown_pack_or_item_404(client):
    assert (await client.delete("/api/packs/nope/items/x")).status_code == 404
    assert (await client.delete(
        f"/api/packs/{SENTENCE}/items/nope")).status_code == 404


# ── boot_id and the GPU facts ─────────────────────────────────────────────


async def test_health_has_boot_id_stable_within_process(test_client):
    from app.main import BOOT_ID

    first = (await test_client.get("/api/health")).json()
    second = (await test_client.get("/api/health")).json()
    assert first["boot_id"] == second["boot_id"] == BOOT_ID
    assert isinstance(BOOT_ID, str) and len(BOOT_ID) == 32


async def test_gpu_info_runs_off_the_event_loop_thread(client, monkeypatch):
    """``detect_gpu`` shells out with a five second timeout.

    On the loop that stalls every other request -- including the install long
    poll -- for as long as a wedged driver takes to answer, so the whole GPU
    probe belongs in a thread.
    """
    loop_thread = threading.get_ident()
    seen: list[int] = []

    def _probe() -> dict:
        seen.append(threading.get_ident())
        return {"detected_label": None, "recommended_variant": "cpu",
                "installed_variant": None, "variants": list(restart.VARIANTS),
                "install_command": "cdui install --gpu cpu"}

    monkeypatch.setattr(restart, "gpu_info", _probe)
    response = await client.get("/api/packs")

    assert response.status_code == 200
    assert response.json()["gpu"]["install_command"] == "cdui install --gpu cpu"
    assert seen, "the route never asked for the GPU facts"
    assert all(ident != loop_thread for ident in seen), (
        "restart.gpu_info ran on the event-loop thread")
    # Exactly one probe per request: the GPU pack's install_command is read
    # from the block above, not computed with a second call.
    assert len(seen) == 1


async def test_gpu_info_never_raises_and_mirrors_dev_py():
    import dev  # scripts/dev.py -- conftest puts scripts/ on sys.path

    assert restart.TORCH_INDEX_URLS == dev.TORCH_INDEX_URLS
    assert list(restart.VARIANTS) == [key for key in dev.TORCH_INDEX_URLS
                                      if key not in {"auto", "skip"}]
    for driver in ("580.1", "560.0", "555.99", "545.0", "530.2", "520.0",
                   "470.0", "", "not-a-version"):
        assert (restart.recommended_cu_for_driver(driver)
                == dev._recommended_cu_for_driver(driver)), driver

    info = restart.gpu_info()
    assert set(info) == GPU_KEYS
    assert info["recommended_variant"] in restart.VARIANTS
    assert info["variants"] == list(restart.VARIANTS)
    assert info["install_command"] == (
        f"cdui install --gpu {info['recommended_variant']}")


async def test_detect_gpu_matches_dev_py_on_this_machine():
    import dev

    # Both run the real probe on this box; the label and the recommended
    # wheel have to be the same answer or the panel and the CLI disagree in
    # front of the user.
    assert restart.detect_gpu() == dev.detect_gpu()


@pytest.mark.parametrize("system", ["Linux", "Windows"])
async def test_detect_gpu_matches_dev_py_with_no_gpu(monkeypatch, system):
    import dev

    # ``platform`` and ``shutil`` are the same module objects in both, so one
    # patch covers the mirror and the original -- which is the point.
    monkeypatch.setattr(restart.platform, "system", lambda: system)
    monkeypatch.setattr(restart.shutil, "which", lambda name: None)
    assert restart.detect_gpu() == dev.detect_gpu() == ("CPU only", "cpu")


async def test_detect_gpu_matches_dev_py_on_apple_silicon(monkeypatch):
    import dev

    monkeypatch.setattr(restart.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(restart.platform, "machine", lambda: "arm64")
    assert restart.detect_gpu() == dev.detect_gpu() == (
        "Apple Silicon (MPS)", "mps")


async def test_detect_gpu_probes_without_a_console_window(monkeypatch):
    """The ANSWER mirrors dev.py; the way the probe is started must not.

    dev.py runs in a console the user is already looking at. This copy runs
    inside a server ``cdui start`` detached, so nvidia-smi gets exactly what
    ``flows.verify_imports`` gives its probe: ``runner.creation_flags()`` --
    without which a console flashes over the editor the first time somebody
    opens the panel -- and no stdin to block on. ``creation_flags`` is
    replaced with a sentinel here so this passes or fails on the WIRING
    rather than on which OS the test happens to run on.
    """
    seen: dict = {}
    flags = 0x0BADF00D

    def _fake_run(argv, **kwargs):
        seen["argv"] = list(argv)
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            argv, 0, stdout="NVIDIA GeForce RTX 4080, 610.74\n", stderr="")

    monkeypatch.setattr(runner, "creation_flags", lambda: flags)
    monkeypatch.setattr(restart.platform, "system", lambda: "Windows")
    monkeypatch.setattr(restart.shutil, "which", lambda name: "nvidia-smi.exe")
    monkeypatch.setattr(restart.subprocess, "run", _fake_run)

    assert restart.detect_gpu() == (
        "NVIDIA GeForce RTX 4080 (driver 610.74)", "cu128")
    assert seen["argv"][0] == "nvidia-smi"
    assert seen["kwargs"]["creationflags"] == flags
    assert seen["kwargs"]["stdin"] is subprocess.DEVNULL


async def test_gpu_info_falls_back_when_detection_explodes(monkeypatch):
    monkeypatch.setattr(restart, "_detected", None)
    monkeypatch.setattr(restart, "detect_gpu",
                        lambda: (_ for _ in ()).throw(OSError("no smi")))
    info = restart.gpu_info()
    assert info["detected_label"] is None
    assert info["recommended_variant"] == "cpu"
    assert info["install_command"] == "cdui install --gpu cpu"


async def test_read_last_restart_is_none_without_a_file(isolated_cache):
    assert restart.read_last_restart() is None


async def test_read_last_restart_ignores_a_corrupt_file(isolated_cache):
    from app.core.packs.paths import last_restart_file

    path = last_restart_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert restart.read_last_restart() is None

    path.write_text('{"pack_id": "gpu-torch", "status": "done"}',
                    encoding="utf-8")
    assert restart.read_last_restart() == {"pack_id": "gpu-torch",
                                           "status": "done"}


async def test_restart_is_not_available_in_this_release():
    assert restart.restart_available() is False
