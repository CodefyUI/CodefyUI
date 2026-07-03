"""Tests for POST /api/apps/{slug}/invoke: 9-key envelope value rules, the
new error taxonomy rows, row-only-if-resolved run recording, pinned marker
shapes, structural RunOutputStore isolation (H1), per-slug locks with
queue-aware budgets, and snapshot-behavior immutability."""

from __future__ import annotations

import asyncio
import json
import sqlite3
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
    await _publish(test_client, "para-one",
                   _chain_graph("slow-a", "_SlowPass", {"seconds": 0.7}))
    await _publish(test_client, "para-two",
                   _chain_graph("slow-b", "_SlowPass", {"seconds": 0.7}))
    key_headers = _bearer(api_key["token"])
    t0 = time.monotonic()
    r1, r2 = await asyncio.gather(
        test_client.post("/api/apps/para-one/invoke",
                         json={"inputs": {"x": "a"}}, headers=key_headers),
        test_client.post("/api/apps/para-two/invoke",
                         json={"inputs": {"x": "b"}}, headers=key_headers),
    )
    elapsed = time.monotonic() - t0
    assert r1.status_code == 200 and r2.status_code == 200
    assert elapsed < 1.3           # different slugs never queue on each other


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
    """POST with Content-Length > MAX_RUN_BODY_BYTES returns 413 with no
    runs row. The cap is checked against the header before the body is
    read; the envelope has all 9 keys, error.code == "payload_too_large",
    and version is None (pre-resolution failure)."""
    monkeypatch.setattr("app.config.settings.MAX_RUN_BODY_BYTES", 10)
    await _publish(test_client, SLUG, _echo_graph())
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
