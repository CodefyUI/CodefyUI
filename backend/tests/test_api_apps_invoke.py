"""Tests for POST /api/apps/{slug}/invoke: 9-key envelope value rules, the
new error taxonomy rows, row-only-if-resolved run recording, pinned marker
shapes, structural RunOutputStore isolation (H1), per-slug locks with
queue-aware budgets, and snapshot-behavior immutability."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from typing import Any

import pytest

from app.config import settings
from app.core.db import Database
from app.core.node_base import BaseNode, DataType, PortDefinition
from app.core.run_output_store import RunOutputStore
from app.main import app

ENVELOPE_KEYS = {
    "status", "run_id", "graph", "app", "version",
    "device", "outputs", "error", "timing",
}

SLUG = "invoke-app"


@pytest.fixture(autouse=True)
def _graphs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.settings.GRAPHS_DIR", tmp_path)
    return tmp_path


# ── test-support nodes (test_api_graph_run.py direct-injection pattern) ──


class _SlowPassNode(BaseNode):
    """Sleeps `seconds` in the executor thread, then passes value through."""

    NODE_NAME = "_SlowPass"
    CATEGORY = "Test"
    DESCRIPTION = "Sleeps, then passes through"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    def execute(self, inputs: dict[str, Any],
                params: dict[str, Any]) -> dict[str, Any]:
        time.sleep(float(params.get("seconds", 2.0)))
        return {"value": inputs.get("value")}


class _HoldGate:
    """One held node per label: the test learns when the node has ENTERED
    ``execute`` and decides when it may RETURN.

    Same shape as the sweep tests' ``_Probe`` (#187, #378): a gate the test
    controls, never a sleep sized against wall-clock timing on a loaded
    runner. Mutated from engine worker threads, hence ``threading.Event``.
    """

    def __init__(self) -> None:
        self.gates: dict[str, tuple[threading.Event, threading.Event]] = {}

    def hold(self, label: str) -> None:
        """Register *label* to block on a gate. Call BEFORE the invoke: the
        node looks its label up on entry."""
        self.gates[label] = (threading.Event(), threading.Event())

    async def entered(self, label: str, timeout: float = 5.0) -> bool:
        """True once *label*'s node has reached its hold point.

        Waited in a THREAD: this is awaited from the test's own coroutine,
        which runs on the event loop, and the node's dispatch chain needs
        that same loop to keep turning to reach the gate at all. A direct
        ``Event.wait`` would deadlock against the thing it waits for.
        """
        entered, _release = self.gates[label]
        return await asyncio.to_thread(entered.wait, timeout)

    def release_all(self) -> None:
        for _entered, release in self.gates.values():
            release.set()


_hold = _HoldGate()


class _HeldPassNode(BaseNode):
    """Signals that it has entered ``execute``, blocks in its executor thread
    until the test releases its label, then passes the value through.

    The wait is capped so a test that fails before releasing cannot hang the
    suite; the cap is far above anything the test waits for itself.
    """

    NODE_NAME = "_HeldPass"
    CATEGORY = "Test"
    DESCRIPTION = "Blocks until released, then passes through"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    def execute(self, inputs: dict[str, Any],
                params: dict[str, Any]) -> dict[str, Any]:
        entered, release = _hold.gates[str(params["label"])]
        entered.set()
        release.wait(timeout=30.0)
        return {"value": inputs.get("value")}


class _BoomNode(BaseNode):
    """Raises on execute — drives the execution_error taxonomy row."""

    NODE_NAME = "_Boom"
    CATEGORY = "Test"
    DESCRIPTION = "Raises RuntimeError"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    def execute(self, inputs: dict[str, Any],
                params: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("boom: intentional test failure")


@pytest.fixture(autouse=True)
def _register_test_nodes():
    from app.core.node_registry import registry

    registry._nodes["_SlowPass"] = _SlowPassNode
    registry._nodes["_HeldPass"] = _HeldPassNode
    registry._nodes["_Boom"] = _BoomNode
    yield


# ── graph builders (duplicated so test modules stay independent) ─────────


def _echo_graph(name: str = "invoke-src", *, input_type: str = "string",
                required: bool = True, output_name: str = "y") -> dict:
    return {
        "name": name,
        "description": "",
        "nodes": [
            {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
             "data": {"params": {}}},
            {"id": "gi", "type": "GraphInput", "position": {"x": 200, "y": 0},
             "data": {"params": {
                 "name": "x", "type": input_type, "required": required,
                 "default": "", "description": "",
             }}},
            {"id": "out", "type": "GraphOutput", "position": {"x": 400, "y": 0},
             "data": {"params": {"name": output_name, "description": ""}}},
        ],
        "edges": [
            {"id": "t1", "source": "start", "target": "gi",
             "sourceHandle": "trigger", "targetHandle": "", "type": "trigger"},
            {"id": "d1", "source": "gi", "target": "out",
             "sourceHandle": "value", "targetHandle": "value", "type": "data"},
        ],
    }


def _chain_graph(name: str, middle_type: str,
                 middle_params: dict | None = None) -> dict:
    """Start -> GraphInput -> <middle> -> GraphOutput."""
    g = _echo_graph(name=name)
    g["nodes"].insert(2, {"id": "mid", "type": middle_type,
                          "position": {"x": 300, "y": 0},
                          "data": {"params": middle_params or {}}})
    g["edges"] = [
        {"id": "t1", "source": "start", "target": "gi",
         "sourceHandle": "trigger", "targetHandle": "", "type": "trigger"},
        {"id": "d1", "source": "gi", "target": "mid",
         "sourceHandle": "value", "targetHandle": "value", "type": "data"},
        {"id": "d2", "source": "mid", "target": "out",
         "sourceHandle": "value", "targetHandle": "value", "type": "data"},
    ]
    return g


async def _save_graph(client, graph: dict) -> None:
    resp = await client.post("/api/graph/save", json=graph)
    assert resp.status_code == 200, resp.text


async def _publish(client, slug: str, graph: dict, **overrides) -> int:
    await _save_graph(client, graph)
    payload = {"graph": graph["name"], "create": True, **overrides}
    resp = await client.post(f"/api/apps/{slug}/publish", json=payload)
    assert resp.status_code == 200, resp.text
    return resp.json()["version"]


@pytest.fixture
async def api_key(test_client, app_db) -> dict:
    resp = await test_client.post("/api/keys", json={"name": "invoke-tests"})
    assert resp.status_code == 200
    return resp.json()


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _run_rows(db: Database) -> list[dict[str, Any]]:
    def _select(conn: sqlite3.Connection) -> list[dict[str, Any]]:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM runs ORDER BY created_at").fetchall()]
    return await db.run(_select)


async def _invokes_at(client, token: str, stamps: list[str], monkeypatch,
                      slug: str = SLUG) -> list[str]:
    """One invoke per stamp, with each run's ``created_at`` pinned to it.

    Returns the run_ids in insertion order. ``_record_run`` reads the clock
    exactly once per recorded run, so the stamps line up one-for-one with
    the invokes. Call this AFTER ``_publish`` -- the publish routes read the
    same clock through the same name.
    """
    clock = iter(stamps)
    monkeypatch.setattr("app.api.routes_apps.utc_now_iso",
                        lambda: next(clock))
    ids: list[str] = []
    for n in range(len(stamps)):
        resp = await client.post(f"/api/apps/{slug}/invoke",
                                 json={"inputs": {"x": f"v{n}"}},
                                 headers=_bearer(token))
        assert resp.status_code == 200, resp.text
        ids.append(resp.json()["run_id"])
    return ids


def _cursor(row: dict[str, Any]) -> str:
    """The keyset cursor query string built from the last row of a page."""
    return f"before={row['created_at']}&before_id={row['run_id']}"


# ── envelope value rules + happy path ────────────────────────────────────


@pytest.mark.asyncio
async def test_invoke_happy_path_nine_key_envelope(
    test_client, app_db, api_key,
):
    await _publish(test_client, SLUG, _echo_graph())
    resp = await test_client.post(
        f"/api/apps/{SLUG}/invoke",
        json={"inputs": {"x": "hello"}},
        headers=_bearer(api_key["token"]),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body.keys()) == ENVELOPE_KEYS
    assert body["status"] == "ok"
    assert body["run_id"]
    assert body["graph"] == SLUG          # "the name you addressed"
    assert body["app"] == SLUG
    assert body["version"] == 1
    assert body["outputs"] == {"y": "hello"}
    assert body["error"] is None
    assert body["timing"]["total_s"] >= 0


@pytest.mark.asyncio
async def test_invoke_401_invalid_key_enveloped_with_www_authenticate(
    test_client, app_db, api_key,
):
    await _publish(test_client, SLUG, _echo_graph())
    # No Authorization header at all — the session token on test_client
    # must NOT be accepted (invoke is key-only).
    resp = await test_client.post(f"/api/apps/{SLUG}/invoke",
                                  json={"inputs": {"x": "hi"}})
    assert resp.status_code == 401
    assert resp.headers["WWW-Authenticate"] == "Bearer"
    body = resp.json()
    assert set(body.keys()) == ENVELOPE_KEYS
    assert body["error"]["code"] == "invalid_key"
    assert body["run_id"]
    assert body["graph"] == SLUG
    assert body["app"] == SLUG            # app = slug even pre-resolution
    assert body["version"] is None        # no version resolved

    # Self-diagnosing message for a pasted session token.
    from app.core.auth import session_token
    resp = await test_client.post(f"/api/apps/{SLUG}/invoke",
                                  json={"inputs": {"x": "hi"}},
                                  headers=_bearer(session_token()))
    assert resp.status_code == 401
    assert resp.json()["error"]["message"] == (
        "this endpoint takes an API key (cdui_...), "
        "not the editor session token"
    )

    # Revoked key.
    await test_client.post(f"/api/keys/{api_key['id']}/revoke")
    resp = await test_client.post(f"/api/apps/{SLUG}/invoke",
                                  json={"inputs": {"x": "hi"}},
                                  headers=_bearer(api_key["token"]))
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "invalid_key"


@pytest.mark.asyncio
async def test_invoke_404_and_409_pre_resolution_envelopes(
    test_client, app_db, api_key,
):
    key_headers = _bearer(api_key["token"])
    resp = await test_client.post("/api/apps/no-such-app/invoke",
                                  json={}, headers=key_headers)
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "app_not_found"
    assert body["graph"] == "no-such-app"
    assert body["app"] == "no-such-app"
    assert body["version"] is None

    await _publish(test_client, SLUG, _echo_graph())
    await test_client.post(f"/api/apps/{SLUG}/unpublish")
    resp = await test_client.post(f"/api/apps/{SLUG}/invoke",
                                  json={}, headers=key_headers)
    assert resp.status_code == 409
    body = resp.json()
    assert body["error"]["code"] == "app_unpublished"
    assert body["app"] == SLUG            # the APP resolved; no VERSION did
    assert body["version"] is None


# ── row-only-if-resolved ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pre_resolution_failures_write_no_runs_rows(
    test_client, app_db, api_key,
):
    await _publish(test_client, SLUG, _echo_graph())
    await _publish(test_client, "parked-app", _echo_graph(name="parked-src"))
    await test_client.post("/api/apps/parked-app/unpublish")

    # invalid_key / app_not_found / app_unpublished: never a row.
    await test_client.post(f"/api/apps/{SLUG}/invoke", json={})
    await test_client.post("/api/apps/ghost/invoke", json={},
                           headers=_bearer(api_key["token"]))
    await test_client.post("/api/apps/parked-app/invoke", json={},
                           headers=_bearer(api_key["token"]))
    assert await _run_rows(app_db) == []


@pytest.mark.asyncio
async def test_resolved_outcomes_write_exactly_one_row_each(
    test_client, app_db, api_key,
):
    await _publish(test_client, SLUG, _echo_graph())
    key_headers = _bearer(api_key["token"])

    ok = await test_client.post(f"/api/apps/{SLUG}/invoke",
                                json={"inputs": {"x": "hi"}},
                                headers=key_headers)
    assert ok.status_code == 200
    bad = await test_client.post(f"/api/apps/{SLUG}/invoke",
                                 json={"inputs": {"x": 42, "typo": 1}},
                                 headers=key_headers)
    assert bad.status_code == 422

    rows = await _run_rows(app_db)
    assert len(rows) == 2
    by_run_id = {r["run_id"]: r for r in rows}

    ok_row = by_run_id[ok.json()["run_id"]]
    assert ok_row["status"] == "ok"
    assert ok_row["version"] == 1
    assert ok_row["api_key_id"] == api_key["id"]   # never NULL in Stage 2
    assert ok_row["device"] == ok.json()["device"]
    assert ok_row["total_s"] == ok.json()["timing"]["total_s"]
    assert json.loads(ok_row["inputs_json"]) == {"x": "hi"}
    assert json.loads(ok_row["outputs_json"]) == {"y": "hi"}

    # The 422 row records the offending RAW inputs — the debugging payoff.
    bad_row = by_run_id[bad.json()["run_id"]]
    assert bad_row["status"] == "error"
    assert bad_row["error_code"] == "invalid_input"
    assert bad.json()["version"] == 1     # post-resolution error carries version
    assert json.loads(bad_row["inputs_json"]) == {"x": 42, "typo": 1}


@pytest.mark.asyncio
async def test_execution_error_row_carries_node_id_and_timings(
    test_client, app_db, api_key,
):
    await _publish(test_client, SLUG, _chain_graph("boom-pub", "_Boom"))
    resp = await test_client.post(f"/api/apps/{SLUG}/invoke",
                                  json={"inputs": {"x": "hi"}},
                                  headers=_bearer(api_key["token"]))
    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "execution_error"

    rows = await _run_rows(app_db)
    assert len(rows) == 1
    assert rows[0]["error_code"] == "execution_error"
    assert rows[0]["error_node_id"] == "mid"
    assert "boom: intentional test failure" in rows[0]["error_message"]


@pytest.mark.asyncio
async def test_per_node_timings_persisted_zero_allowed(
    test_client, app_db, api_key,
):
    await _publish(test_client, SLUG, _echo_graph())
    resp = await test_client.post(f"/api/apps/{SLUG}/invoke",
                                  json={"inputs": {"x": "hi"}},
                                  headers=_bearer(api_key["token"]))
    assert resp.status_code == 200
    # The envelope timing shape stays frozen: {total_s} only.
    assert set(resp.json()["timing"].keys()) == {"total_s"}

    rows = await _run_rows(app_db)
    timings = json.loads(rows[0]["node_timings_json"])
    assert {"gi", "out"} <= set(timings.keys())
    # No nonzero assumption: cached/skipped nodes legitimately record 0.0.
    assert all(isinstance(v, (int, float)) and v >= 0.0
               for v in timings.values())


# ── markers + redaction ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_over_cap_field_stores_pinned_truncation_marker(
    test_client, app_db, api_key, monkeypatch,
):
    monkeypatch.setattr("app.config.settings.RUN_IO_CAP_BYTES", 64)
    await _publish(test_client, SLUG, _echo_graph())
    big = "A" * 100                      # json.dumps adds 2 quote bytes
    resp = await test_client.post(f"/api/apps/{SLUG}/invoke",
                                  json={"inputs": {"x": big}},
                                  headers=_bearer(api_key["token"]))
    assert resp.status_code == 200
    assert resp.json()["outputs"] == {"y": big}   # the RESPONSE is uncapped

    rows = await _run_rows(app_db)
    stored_inputs = json.loads(rows[0]["inputs_json"])
    stored_outputs = json.loads(rows[0]["outputs_json"])
    # The EXACT pinned shape — a cross-stage contract Stage 3 switches on.
    assert stored_inputs == {"x": {"__codefyui__": "truncated", "bytes": 102}}
    assert stored_outputs == {"y": {"__codefyui__": "truncated", "bytes": 102}}


@pytest.mark.asyncio
async def test_record_io_false_stores_pinned_redaction_markers(
    test_client, app_db, api_key,
):
    await _publish(test_client, SLUG, _echo_graph(), record_io=False)
    resp = await test_client.post(f"/api/apps/{SLUG}/invoke",
                                  json={"inputs": {"x": "secret"}},
                                  headers=_bearer(api_key["token"]))
    assert resp.status_code == 200

    rows = await _run_rows(app_db)
    assert json.loads(rows[0]["inputs_json"]) == {
        "x": {"__codefyui__": "redacted"},
    }
    assert json.loads(rows[0]["outputs_json"]) == {
        "y": {"__codefyui__": "redacted"},
    }
    assert "secret" not in rows[0]["inputs_json"]


# ── best-effort INSERT + H1 isolation ────────────────────────────────────


@pytest.mark.asyncio
async def test_best_effort_insert_failure_still_returns_envelope(
    test_client, app_db, api_key, monkeypatch, caplog,
):
    import logging

    await _publish(test_client, SLUG, _echo_graph())
    real_run = Database.run

    async def _failing_run(self, fn):
        # Fail ONLY the runs INSERT (the closure is named _insert by
        # contract); resolution and key lookup proceed normally.
        if getattr(fn, "__name__", "") == "_insert":
            raise sqlite3.OperationalError("disk I/O error (injected)")
        return await real_run(self, fn)

    monkeypatch.setattr(Database, "run", _failing_run)
    with caplog.at_level(logging.ERROR, logger="app.api.routes_apps"):
        resp = await test_client.post(f"/api/apps/{SLUG}/invoke",
                                      json={"inputs": {"x": "hi"}},
                                      headers=_bearer(api_key["token"]))
    assert resp.status_code == 200                # run outcome outranks bookkeeping
    assert resp.json()["outputs"] == {"y": "hi"}
    assert "failed to record run" in caplog.text

    monkeypatch.setattr(Database, "run", real_run)
    assert await _run_rows(app_db) == []          # the row really was lost


@pytest.mark.asyncio
async def test_record_outputs_structurally_inert_on_invoke(
    test_client, app_db, api_key,
):
    # H1 regression pin: output_store=None is structural — even with
    # record_outputs=true the RunOutputStore stays empty, so the
    # unauthenticated GET /api/execution/outputs/* can never see
    # published-app data.
    store = RunOutputStore(max_runs=5)
    app.state.run_output_store = store
    await _publish(test_client, SLUG, _echo_graph())
    resp = await test_client.post(
        f"/api/apps/{SLUG}/invoke",
        json={"inputs": {"x": "hi"}, "record_outputs": True},
        headers=_bearer(api_key["token"]),
    )
    assert resp.status_code == 200                # accepted-and-ignored
    run_id = resp.json()["run_id"]
    assert await store.list_runs() == []
    listing = await test_client.get(f"/api/execution/outputs/{run_id}")
    assert listing.status_code == 404


# ── immutability (invoke-behavior pinning, deferred from PR2) ────────────


@pytest.mark.asyncio
async def test_canvas_resave_does_not_change_invoke_until_republish(
    test_client, app_db, api_key,
):
    graph = _echo_graph()
    await _publish(test_client, SLUG, graph)
    key_headers = _bearer(api_key["token"])

    resp = await test_client.post(f"/api/apps/{SLUG}/invoke",
                                  json={"inputs": {"x": "hi"}},
                                  headers=key_headers)
    assert resp.json()["outputs"] == {"y": "hi"}

    # Edit the canvas: rename the output y -> z, re-save the file.
    graph["nodes"][2]["data"]["params"]["name"] = "z"
    await _save_graph(test_client, graph)

    # Invoke still runs the SNAPSHOT (pre-edit behavior).
    resp = await test_client.post(f"/api/apps/{SLUG}/invoke",
                                  json={"inputs": {"x": "hi"}},
                                  headers=key_headers)
    assert resp.json()["outputs"] == {"y": "hi"}
    assert resp.json()["version"] == 1

    # Re-publish flips it.
    resp = await test_client.post(
        f"/api/apps/{SLUG}/publish", json={"graph": graph["name"]})
    assert resp.status_code == 200
    resp = await test_client.post(f"/api/apps/{SLUG}/invoke",
                                  json={"inputs": {"x": "hi"}},
                                  headers=key_headers)
    assert resp.json()["outputs"] == {"z": "hi"}
    assert resp.json()["version"] == 2


# ── concurrency (Decision I) ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_invokes_on_one_slug_serialize(
    test_client, app_db, api_key,
):
    await _publish(test_client, SLUG,
                   _chain_graph("slow-one", "_SlowPass", {"seconds": 0.7}))
    key_headers = _bearer(api_key["token"])
    t0 = time.monotonic()
    r1, r2 = await asyncio.gather(
        test_client.post(f"/api/apps/{SLUG}/invoke",
                         json={"inputs": {"x": "a"}}, headers=key_headers),
        test_client.post(f"/api/apps/{SLUG}/invoke",
                         json={"inputs": {"x": "b"}}, headers=key_headers),
    )
    elapsed = time.monotonic() - t0
    assert r1.status_code == 200 and r2.status_code == 200
    assert elapsed >= 1.3          # two 0.7s runs back-to-back, never overlapped
    assert len(await _run_rows(app_db)) == 2


@pytest.mark.asyncio
async def test_invokes_on_different_slugs_interleave(
    test_client, app_db, api_key,
):
    """Different slugs never queue on each other (Decision I: one lock PER
    slug).

    Proved with a gate the test controls. Earlier versions timed a
    concurrent pair against a serial baseline measured moments before and
    held the ratio under 0.85 (#157, then a cold-start fix); that still
    flaked on a loaded shared runner: the 2.6.0 release commit -- version
    numbers and nothing else in the diff -- measured interleaved=1.216s
    against serial=1.413s (0.86) on `pytest (built frontend)`. Any
    wall-clock bar on a shared runner fails some fraction of the time.

    Here the first slug's node is held open by the test. While it is held,
    an invoke of the SECOND slug has to reach its own node. Under a per-slug
    lock that happens at once; under one shared lock it cannot happen until
    the first invoke finishes, which it provably cannot while held. So the
    wait either succeeds or times out, and which one depends only on the
    shape of the lock.
    """
    _hold.hold("held-a")
    _hold.hold("held-b")
    await _publish(test_client, "para-one",
                   _chain_graph("held-one", "_HeldPass", {"label": "held-a"}))
    await _publish(test_client, "para-two",
                   _chain_graph("held-two", "_HeldPass", {"label": "held-b"}))
    key_headers = _bearer(api_key["token"])

    first = asyncio.ensure_future(test_client.post(
        "/api/apps/para-one/invoke",
        json={"inputs": {"x": "a"}}, headers=key_headers))
    second = None
    reached_second = False
    try:
        assert await _hold.entered("held-a"), \
            "the first invoke never reached its node"
        second = asyncio.ensure_future(test_client.post(
            "/api/apps/para-two/invoke",
            json={"inputs": {"x": "b"}}, headers=key_headers))
        reached_second = await _hold.entered("held-b")
    finally:
        # Whatever happened above, let both worker threads return and both
        # requests finish before the assertions below read them.
        _hold.release_all()
        settled = await asyncio.gather(
            first, *([second] if second is not None else []),
            return_exceptions=True)
    assert reached_second, (
        "the second slug did not reach its node while the first was held "
        "-- different slugs are queuing on each other")

    r1, r2 = settled
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["outputs"] == {"y": "a"}
    assert r2.json()["outputs"] == {"y": "b"}
    assert len(await _run_rows(app_db)) == 2


@pytest.mark.asyncio
async def test_queue_timeout_expires_while_queued(
    test_client, app_db, api_key,
):
    await _publish(test_client, SLUG,
                   _chain_graph("slow-hold", "_SlowPass", {"seconds": 2.0}))
    key_headers = _bearer(api_key["token"])

    first = asyncio.create_task(test_client.post(
        f"/api/apps/{SLUG}/invoke",
        json={"inputs": {"x": "a"}}, headers=key_headers))
    await asyncio.sleep(0.3)       # let the first invoke take the lock
    second = await test_client.post(
        f"/api/apps/{SLUG}/invoke",
        json={"inputs": {"x": "b"}, "timeout_s": 1}, headers=key_headers)
    # The budget covers TOTAL time including queue wait: the second call
    # dies waiting for the lock, with the queue-specific message.
    assert second.status_code == 500
    body = second.json()
    assert body["error"]["code"] == "timeout"
    assert "expired while queued" in body["error"]["message"]
    assert body["version"] == 1    # resolved before queueing -> row written

    resp1 = await first
    assert resp1.status_code == 200

    rows = await _run_rows(app_db)
    assert len(rows) == 2          # one ok row + one queue-timeout row
    assert {r["error_code"] for r in rows} == {None, "timeout"}


# ── coverage pins: 413 body-cap, route-level 422, post-timeout recovery ────


@pytest.mark.asyncio
async def test_invoke_413_body_cap_writes_no_row(
    test_client, app_db, api_key, monkeypatch,
):
    """POST over MAX_RUN_BODY_BYTES returns 413 with no runs row.

    The envelope has all 9 keys, error.code == "payload_too_large", and
    version is None (the cap fires before any version is resolved). Since
    core#265 the cap is enforced by ``core.body_limit`` for every route rather
    than by this one, but the envelope is unchanged — the per-app OpenAPI
    document types the 413 as a RunEnvelope for generated clients.

    Note the publish happens BEFORE the monkeypatch, which it did not have to
    before: the setup calls ``/api/graph/save``, and a 10-byte ceiling now
    applies to that route too. That is the coverage gap core#265 was about.
    """
    await _publish(test_client, SLUG, _echo_graph())
    monkeypatch.setattr("app.config.settings.MAX_RUN_BODY_BYTES", 10)
    key_headers = _bearer(api_key["token"])

    # Send a body larger than the tiny cap (10 bytes). The actual body
    # content is small, but Content-Length header declares the size.
    big_body = {"inputs": {"x": "hello world this is larger than 10"}}
    resp = await test_client.post(
        f"/api/apps/{SLUG}/invoke",
        json=big_body,
        headers=key_headers,
    )
    assert resp.status_code == 413
    body = resp.json()
    assert set(body.keys()) == ENVELOPE_KEYS
    assert body["error"]["code"] == "payload_too_large"
    assert body["app"] == SLUG
    assert body["version"] is None        # pre-resolution (413 happens early)
    assert "payload_too_large" in body["error"]["message"] or "max" in body["error"]["message"]

    # No rows written for pre-resolution failures.
    rows = await _run_rows(app_db)
    assert len(rows) == 0


@pytest.mark.asyncio
async def test_invoke_401_beats_the_body_cap(
    test_client, app_db, api_key, monkeypatch,
):
    """A bad key on an oversized body answers 401, not 413.

    This route checks its key BEFORE it reads the body, on purpose, and the
    body cap must not disturb that. The ordering is not hypothetical: the
    obvious way to build a global cap — Starlette's own
    RequestBodyLimitMiddleware — also wraps ``send`` and replaces whatever
    response the app produced with a plain-text 413 whenever Content-Length
    is over the limit, turning exactly this 401 into a 413. core.body_limit
    refuses only what the application actually READS, which is what keeps
    every route's own auth answer in front of it.
    """
    await _publish(test_client, SLUG, _echo_graph())
    monkeypatch.setattr("app.config.settings.MAX_RUN_BODY_BYTES", 10)

    resp = await test_client.post(
        f"/api/apps/{SLUG}/invoke",
        json={"inputs": {"x": "far larger than ten bytes"}},
        headers={"Authorization": "Bearer cdui_not_a_real_key"},
    )
    assert resp.status_code == 401, resp.text
    body = resp.json()
    assert set(body.keys()) == ENVELOPE_KEYS
    assert body["error"]["code"] == "invalid_key"
    assert resp.headers.get("WWW-Authenticate") == "Bearer"


@pytest.mark.asyncio
async def test_invoke_route_level_422_malformed_body_writes_row(
    test_client, app_db, api_key,
):
    """POST with malformed body (e.g. 'inputs' is a list instead of dict)
    fails at the _parse_run_body route level. Returns 422 envelope with
    error.code == "invalid_input", version == resolved version (not None),
    and EXACTLY ONE runs row with status "error" and error_code "invalid_input".

    This test exercises the case where the body is valid JSON and passes
    app resolution, but the 'inputs' field is not a dict — causing
    _parse_run_body to return field_errors."""
    await _publish(test_client, SLUG, _echo_graph())
    key_headers = _bearer(api_key["token"])

    # Send a body where 'inputs' is a list instead of a dict.
    # This passes JSON parsing and reaches _parse_run_body, which rejects it.
    resp = await test_client.post(
        f"/api/apps/{SLUG}/invoke",
        json={"inputs": ["not", "a", "dict"]},  # inputs must be dict, not list
        headers=key_headers,
    )
    assert resp.status_code == 422
    body = resp.json()
    assert set(body.keys()) == ENVELOPE_KEYS
    assert body["error"]["code"] == "invalid_input"
    assert body["app"] == SLUG
    assert body["version"] == 1    # resolved after app lookup -> version is set
    assert body["run_id"]

    # Exactly one row written with error status.
    rows = await _run_rows(app_db)
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert rows[0]["error_code"] == "invalid_input"
    assert rows[0]["version"] == 1
    assert rows[0]["run_id"] == body["run_id"]


@pytest.mark.asyncio
async def test_invoke_succeeds_after_queued_timeout(
    test_client, app_db, api_key,
):
    """Lock recovery after a queued timeout:
    - Invoke A: slow graph (1.5s), holds the lock
    - Invoke B: timeout_s=1 (dies waiting in queue with timeout message)
    - Await A: succeeds (lock released cleanly)
    - Invoke C: fresh invoke succeeds (lock available again)

    Proves the lock is healthy post-timeout: the lock releases in the
    finally block even when a queued request times out."""
    await _publish(test_client, SLUG,
                   _chain_graph("slow-recovery", "_SlowPass", {"seconds": 1.5}))
    key_headers = _bearer(api_key["token"])

    # Start invoke A (will hold the lock for ~1.5s)
    task_a = asyncio.create_task(test_client.post(
        f"/api/apps/{SLUG}/invoke",
        json={"inputs": {"x": "a"}}, headers=key_headers))
    await asyncio.sleep(0.2)  # let A acquire the lock

    # Invoke B with a short timeout (will queue and time out)
    resp_b = await test_client.post(
        f"/api/apps/{SLUG}/invoke",
        json={"inputs": {"x": "b"}, "timeout_s": 1},
        headers=key_headers,
    )
    assert resp_b.status_code == 500
    assert resp_b.json()["error"]["code"] == "timeout"
    assert "expired while queued" in resp_b.json()["error"]["message"]

    # Await A to complete (should succeed after ~1.5s total)
    resp_a = await task_a
    assert resp_a.status_code == 200
    assert resp_a.json()["outputs"] == {"y": "a"}

    # Invoke C fresh (lock should be free now)
    resp_c = await test_client.post(
        f"/api/apps/{SLUG}/invoke",
        json={"inputs": {"x": "c"}}, headers=key_headers)
    assert resp_c.status_code == 200
    assert resp_c.json()["outputs"] == {"y": "c"}

    # Three rows: A ok, B timeout error, C ok
    rows = await _run_rows(app_db)
    assert len(rows) == 3
    by_run_id = {r["run_id"]: r for r in rows}
    assert by_run_id[resp_a.json()["run_id"]]["status"] == "ok"
    assert by_run_id[resp_b.json()["run_id"]]["status"] == "error"
    assert by_run_id[resp_b.json()["run_id"]]["error_code"] == "timeout"
    assert by_run_id[resp_c.json()["run_id"]]["status"] == "ok"


# ── runs reads: metadata list + full detail, either-credential (Task 11) ─


@pytest.mark.asyncio
async def test_runs_list_metadata_only_newest_first(
    test_client, app_db, api_key, monkeypatch,
):
    """Ordering, field set, and the legacy ``before``-alone cursor.

    ``before`` without ``before_id`` keeps the semantics it shipped with:
    ``created_at < ?``, which skips every row sharing the cursor's
    timestamp. The clock is stepped here so the two runs land in different
    ticks and that cursor is exact -- two invokes this close together often
    share a tick, and leaving it to chance made this file fail on CI at
    random.

    The composite ``before``/``before_id`` cursor is what pages through a
    shared timestamp, in
    ``test_runs_list_pages_through_runs_sharing_one_timestamp`` and the
    three tests beside it (#372). Ordering under a genuine tie has its own
    test above, in
    ``test_two_invokes_in_one_timestamp_tick_still_list_newest_first``.
    """
    await _publish(test_client, SLUG, _echo_graph())

    stamps = iter([
        "2026-03-03T00:00:01.000000Z",
        "2026-03-03T00:00:02.000000Z",
    ])
    monkeypatch.setattr("app.api.routes_apps.utc_now_iso",
                        lambda: next(stamps))

    key_headers = _bearer(api_key["token"])
    first = await test_client.post(f"/api/apps/{SLUG}/invoke",
                                   json={"inputs": {"x": "one"}},
                                   headers=key_headers)
    second = await test_client.post(f"/api/apps/{SLUG}/invoke",
                                    json={"inputs": {"x": 2}},  # 422
                                    headers=key_headers)

    resp = await test_client.get(f"/api/apps/{SLUG}/runs")
    assert resp.status_code == 200
    rows = resp.json()
    assert [r["run_id"] for r in rows] == [
        second.json()["run_id"], first.json()["run_id"],
    ]
    for row in rows:
        assert set(row.keys()) == {
            "run_id", "version", "status", "error_code", "error_node_id",
            "device", "total_s", "api_key_id", "created_at",
        }  # metadata ONLY — no inputs/outputs/node_timings

    # limit + before cursor.
    resp = await test_client.get(f"/api/apps/{SLUG}/runs?limit=1")
    assert [r["run_id"] for r in resp.json()] == [second.json()["run_id"]]
    cursor = rows[0]["created_at"]
    resp = await test_client.get(
        f"/api/apps/{SLUG}/runs?before={cursor}")
    assert [r["run_id"] for r in resp.json()] == [first.json()["run_id"]]

    resp = await test_client.get("/api/apps/ghost/runs")
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "app_not_found"


@pytest.mark.asyncio
async def test_runs_list_tie_breaks_on_insertion_order_when_created_at_equal(
    test_client, app_db,
):
    """Rows sharing one created_at timestamp still come back newest first.

    The tie-break used to be ``run_id DESC``, chosen for determinism alone.
    It is deterministic, but ``run_id`` is a ``uuid4().hex`` -- so among rows
    that share a timestamp the order was arbitrary with respect to when they
    actually happened, and an endpoint whose whole contract is "newest first"
    could hand back the older run first. Two invokes in quick succession land
    in the same timestamp tick often enough that
    ``test_runs_list_metadata_only_newest_first`` failed on CI over it (#370).

    ``rowid`` rises with insertion, so it is both deterministic AND the real
    answer to "which happened last" -- and it is what the same query in
    ``run_store.py`` (lines 520, 695, 1174) already tie-breaks on. The
    ``before``/``before_id`` cursor added by #372 keys on these same two
    columns, so the cursor keys and the sort keys are one tuple.

    The ids below are deliberately NOT in lexicographic order: inserted
    aaa, zzz, mmm, a run_id sort would answer zzz/mmm/aaa, and only an
    insertion-order sort answers mmm/zzz/aaa.
    """
    await _publish(test_client, SLUG, _echo_graph())

    def _app_id(conn: sqlite3.Connection) -> int:
        return conn.execute(
            "SELECT id FROM apps WHERE slug = ?", (SLUG,)).fetchone()[0]

    app_id = await app_db.run(_app_id)
    same_ts = "2026-01-01T00:00:00.000000Z"

    def _seed(conn: sqlite3.Connection) -> None:
        for run_id in ("run-aaa", "run-zzz", "run-mmm"):
            conn.execute(
                "INSERT INTO runs (run_id, app_id, version, api_key_id, "
                "status, node_timings_json, inputs_json, outputs_json, "
                "created_at) VALUES (?, ?, 1, NULL, 'ok', '{}', '{}', '{}', "
                "?)",
                (run_id, app_id, same_ts),
            )

    await app_db.run(_seed)
    resp = await test_client.get(f"/api/apps/{SLUG}/runs")
    assert resp.status_code == 200
    assert [r["run_id"] for r in resp.json()] == [
        "run-mmm", "run-zzz", "run-aaa",
    ]


@pytest.mark.asyncio
async def test_two_invokes_in_one_timestamp_tick_still_list_newest_first(
    test_client, app_db, api_key, monkeypatch,
):
    """The flake in ``test_runs_list_metadata_only_newest_first``, made real.

    That test invokes twice and asserts the newer run comes back first. On a
    fast machine both rows land in the same ``created_at`` tick, and until
    #370 the tie-break was ``run_id DESC`` over two uuid4 hexes -- a coin
    flip, so CI failed it at random. Here the clock is frozen so the tie is
    guaranteed rather than hoped for: if the ordering ever goes back to
    something that is not insertion order, this fails every single time
    instead of one run in N.
    """
    await _publish(test_client, SLUG, _echo_graph())
    monkeypatch.setattr(
        "app.api.routes_apps.utc_now_iso",
        lambda: "2026-02-02T00:00:00.000000Z",
    )

    key_headers = _bearer(api_key["token"])
    first = await test_client.post(f"/api/apps/{SLUG}/invoke",
                                   json={"inputs": {"x": "one"}},
                                   headers=key_headers)
    second = await test_client.post(f"/api/apps/{SLUG}/invoke",
                                    json={"inputs": {"x": "two"}},
                                    headers=key_headers)

    rows = (await test_client.get(f"/api/apps/{SLUG}/runs")).json()
    assert {r["created_at"] for r in rows} == {"2026-02-02T00:00:00.000000Z"}, (
        "the freeze did not take -- the tie this test exists for never happened")
    assert [r["run_id"] for r in rows] == [
        second.json()["run_id"], first.json()["run_id"],
    ]


# ── keyset cursor: before + before_id (#372) ─────────────────────────────


@pytest.mark.asyncio
async def test_runs_list_pages_through_runs_sharing_one_timestamp(
    test_client, app_db, api_key, monkeypatch,
):
    """Three runs in ONE ``created_at`` tick page through one at a time.

    ``before`` alone is keyed on ``created_at`` and its predicate is
    ``created_at < ?``, so it excludes every row sharing the cursor's
    timestamp -- including the ones the client has not seen yet. Page 2 of
    this walk used to come back empty and the other two runs were
    unreachable through the API (#372). The composite cursor keys on the
    same ``(created_at, rowid)`` tuple the ORDER BY sorts on, so each run is
    visited exactly once.
    """
    await _publish(test_client, SLUG, _echo_graph())
    stamp = "2026-04-04T00:00:00.000000Z"
    ids = await _invokes_at(test_client, api_key["token"], [stamp] * 3,
                            monkeypatch)

    rows = (await test_client.get(f"/api/apps/{SLUG}/runs")).json()
    assert {r["created_at"] for r in rows} == {stamp}, (
        "the frozen clock did not take -- the tie this test exists for "
        "never happened")

    seen: list[str] = []
    query = f"/api/apps/{SLUG}/runs?limit=1"
    for page_no in (1, 2, 3):
        page = (await test_client.get(query)).json()
        assert len(page) == 1, f"page {page_no} came back as {page}"
        seen.append(page[0]["run_id"])
        query = f"/api/apps/{SLUG}/runs?limit=1&{_cursor(page[0])}"

    assert seen == list(reversed(ids))   # newest first, each run exactly once
    assert (await test_client.get(query)).json() == []


@pytest.mark.asyncio
async def test_runs_list_cursor_walks_mixed_and_shared_timestamps(
    test_client, app_db, api_key, monkeypatch,
):
    """Four runs across two ticks; one query exercises both halves of the
    predicate.

    Page 2 anchors on the older of the two runs in the newer tick. That one
    query has to keep the newer tick's other run out through
    ``created_at = :ts AND rowid < ...`` and let both runs from the older
    tick in through ``created_at < :ts``.
    """
    await _publish(test_client, SLUG, _echo_graph())
    older = "2026-05-05T00:00:00.000000Z"
    newer = "2026-05-05T00:00:01.000000Z"
    r1, r2, r3, r4 = await _invokes_at(
        test_client, api_key["token"], [older, older, newer, newer],
        monkeypatch)

    page1 = (await test_client.get(f"/api/apps/{SLUG}/runs?limit=2")).json()
    assert [r["run_id"] for r in page1] == [r4, r3]

    page2 = (await test_client.get(
        f"/api/apps/{SLUG}/runs?limit=2&{_cursor(page1[-1])}")).json()
    assert [r["run_id"] for r in page2] == [r2, r1]

    assert (await test_client.get(
        f"/api/apps/{SLUG}/runs?limit=2&{_cursor(page2[-1])}")).json() == []


@pytest.mark.asyncio
async def test_runs_list_cursor_survives_a_pruned_anchor_row(
    test_client, app_db, api_key, monkeypatch,
):
    """A cursor whose anchor row is gone falls back to ``created_at < ts``.

    Retention deletes with ``created_at < cutoff``, so it takes a timestamp
    whole. When the anchor is one of the rows that went, every row sharing
    its timestamp went with it, and ``created_at < ts`` is then the exact
    next page. The subselect resolves a missing anchor to NULL and
    ``rowid < NULL`` is NULL, so the fallback needs no branch in Python.
    """
    await _publish(test_client, SLUG, _echo_graph())
    older = "2026-06-06T00:00:00.000000Z"
    newer = "2026-06-06T00:00:01.000000Z"
    r1, r2, r3 = await _invokes_at(
        test_client, api_key["token"], [older, newer, newer], monkeypatch)

    page1 = (await test_client.get(f"/api/apps/{SLUG}/runs?limit=2")).json()
    assert [r["run_id"] for r in page1] == [r3, r2]
    cursor = _cursor(page1[-1])                   # anchored on r2

    def _prune(conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM runs WHERE created_at = ?", (newer,))

    await app_db.run(_prune)

    resp = await test_client.get(f"/api/apps/{SLUG}/runs?limit=2&{cursor}")
    assert resp.status_code == 200
    assert [r["run_id"] for r in resp.json()] == [r1]


@pytest.mark.asyncio
async def test_runs_list_ignores_a_before_id_from_another_app(
    test_client, app_db, api_key, monkeypatch,
):
    """``before_id`` resolves only inside the app being listed.

    The subselect carries ``AND app_id = :app_id``, so another app's run_id
    -- or one that never existed -- resolves to NULL and the cursor falls
    back to ``created_at < ts``. The two runs below share a timestamp, which
    is what makes the difference visible: this app's own run_id pages to the
    second run, anything else pages to nothing.
    """
    await _publish(test_client, SLUG, _echo_graph())
    other_slug = "other-app"
    await _publish(test_client, other_slug, _echo_graph(name="other-src"))
    token = api_key["token"]
    stamp = "2026-07-07T00:00:00.000000Z"
    r1, r2 = await _invokes_at(test_client, token, [stamp] * 2, monkeypatch)
    foreign = (await _invokes_at(test_client, token, [stamp], monkeypatch,
                                 slug=other_slug))[0]

    own = await test_client.get(
        f"/api/apps/{SLUG}/runs?before={stamp}&before_id={r2}")
    assert [r["run_id"] for r in own.json()] == [r1]

    cross = await test_client.get(
        f"/api/apps/{SLUG}/runs?before={stamp}&before_id={foreign}")
    assert cross.status_code == 200
    assert cross.json() == []

    unknown = await test_client.get(
        f"/api/apps/{SLUG}/runs?before={stamp}&before_id=never-ran")
    assert unknown.json() == []

    # The fallback is exactly what `before` alone answers.
    assert (await test_client.get(
        f"/api/apps/{SLUG}/runs?before={stamp}")).json() == []


@pytest.mark.asyncio
async def test_runs_list_rejects_before_id_without_before(
    test_client, app_db, api_key,
):
    """``before_id`` alone has no timestamp to anchor against -> 422.

    The management error shape this file uses everywhere, so the code is
    machine-matchable the same way the endpoint's own ``app_not_found`` is.
    """
    await _publish(test_client, SLUG, _echo_graph())
    resp = await test_client.get(f"/api/apps/{SLUG}/runs?before_id=whatever")
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "incomplete_cursor"


@pytest.mark.asyncio
async def test_run_detail_full_row_with_parsed_io(
    test_client, app_db, api_key,
):
    await _publish(test_client, SLUG, _echo_graph())
    invoke = await test_client.post(f"/api/apps/{SLUG}/invoke",
                                    json={"inputs": {"x": "hi"}},
                                    headers=_bearer(api_key["token"]))
    run_id = invoke.json()["run_id"]

    resp = await test_client.get(f"/api/apps/{SLUG}/runs/{run_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == run_id
    assert body["version"] == 1
    assert body["status"] == "ok"
    assert body["api_key_id"] == api_key["id"]
    assert body["inputs"] == {"x": "hi"}
    assert body["outputs"] == {"y": "hi"}
    assert isinstance(body["node_timings"], dict)
    assert "inputs_json" not in body        # parsed, not raw columns

    resp = await test_client.get(f"/api/apps/{SLUG}/runs/never-ran")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_runs_reads_accept_either_credential_reject_neither(
    test_client, app_db, api_key,
):
    from httpx import ASGITransport, AsyncClient

    await _publish(test_client, SLUG, _echo_graph())
    invoke = await test_client.post(f"/api/apps/{SLUG}/invoke",
                                    json={"inputs": {"x": "hi"}},
                                    headers=_bearer(api_key["token"]))
    run_id = invoke.json()["run_id"]

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url=f"http://127.0.0.1:{settings.PORT}",
    ) as anon:
        # API key alone (no session header) works.
        key_only = await anon.get(f"/api/apps/{SLUG}/runs",
                                  headers=_bearer(api_key["token"]))
        assert key_only.status_code == 200
        detail = await anon.get(f"/api/apps/{SLUG}/runs/{run_id}",
                                headers=_bearer(api_key["token"]))
        assert detail.status_code == 200
        # Neither credential -> plain {"detail": ...} 401.
        denied = await anon.get(f"/api/apps/{SLUG}/runs")
        assert denied.status_code == 401
        assert set(denied.json().keys()) == {"detail"}
        denied = await anon.get(f"/api/apps/{SLUG}/runs/{run_id}")
        assert denied.status_code == 401
    # Session token alone (test_client default headers) works.
    assert (await test_client.get(
        f"/api/apps/{SLUG}/runs")).status_code == 200


@pytest.mark.asyncio
async def test_run_detail_cross_app_isolation(
    test_client, app_db, api_key,
):
    """Verify that a run from app A cannot be accessed via app B's run-detail
    endpoint. The run must be structurally isolated by app_id, so fetching
    a valid run_id with a different app slug returns 404 with
    detail.code == 'run_not_found', indistinguishable from a nonexistent run.
    Positive control: the same run_id is accessible via app A's slug."""
    # Publish two apps.
    await _publish(test_client, SLUG, _echo_graph())
    other_slug = "other-app"
    await _publish(test_client, other_slug, _echo_graph(name="other-src"))
    key_headers = _bearer(api_key["token"])

    # Invoke app A once to create a runs row.
    invoke_resp = await test_client.post(
        f"/api/apps/{SLUG}/invoke",
        json={"inputs": {"x": "test"}},
        headers=key_headers,
    )
    assert invoke_resp.status_code == 200
    run_id = invoke_resp.json()["run_id"]

    # Attempt to access the run from app B -> 404 with run_not_found code.
    cross_app_resp = await test_client.get(
        f"/api/apps/{other_slug}/runs/{run_id}",
        headers=key_headers,
    )
    assert cross_app_resp.status_code == 404
    detail = cross_app_resp.json()["detail"]
    assert detail["code"] == "run_not_found"

    # Positive control: the same run_id is accessible via app A's slug.
    same_app_resp = await test_client.get(
        f"/api/apps/{SLUG}/runs/{run_id}",
        headers=key_headers,
    )
    assert same_app_resp.status_code == 200
    assert same_app_resp.json()["run_id"] == run_id
