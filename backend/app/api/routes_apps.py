"""Published-apps surface (Stage 2): versioned snapshots behind stable slugs.

Management routes (this file, PR2) use ``require_session_token``; the
invoke + runs + openapi routes (PR3/PR4) use the key dependencies. Every
route declares exactly ONE auth dependency — enforced by
tests/test_auth_drift.py, because ``auth_guard`` exempts the /api/apps
prefix entirely.

Management errors are plain ``{"detail": ...}`` transports whose detail
object carries a stable ``code`` (``invalid_slug``, ``graph_not_found``,
``app_not_found``, ``version_not_found``, and the five Stage-1 pre-flight
codes) — machine-matchable without the run envelope, which belongs to
execution surfaces only.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import time
from dataclasses import replace
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..config import settings
from ..core import api_contract
from ..core.api_contract import InputCoercionError
from ..core.api_keys import (
    ApiKeyResult,
    get_db,
    require_api_key,
    require_session_token,
)
from ..core.db import Database, utc_now_iso
from ..core.graph_engine import find_entry_points, validate_graph
from .routes_graph import _graph_path, _sanitize_name
from .routes_graph_run import (
    _derive_output_type,
    _parse_run_body,
    build_envelope,
    error_response,
    execute_contract_run,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/apps", tags=["apps"])

# Decision G: URL-and-DNS-safe, case-collision-free, deliberately narrower
# than the contract-name charset; independent of graph names so renaming a
# graph never breaks a published URL.
SLUG_PATTERN = re.compile(r"^[a-z][a-z0-9-]{0,63}$")


def _manage_error(
    status_code: int, code: str, message: str,
    details: list[Any] | None = None,
) -> HTTPException:
    """Management-surface error: plain ``{"detail": ...}`` transport with a
    stable ``code`` inside (publish pre-flight reuses the Stage-1 codes)."""
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message, "details": details},
    )


def _get_app_locks(request: Request) -> dict:
    """Per-slug invoke locks on app.state (503 when a fixture forgot it —
    same access rule as get_db)."""
    locks = getattr(request.app.state, "app_locks", None)
    if locks is None:
        raise HTTPException(status_code=503,
                            detail="app_locks not initialised")
    return locks


def _contract_document(
    graph_name: str,
    nodes: list[dict],
    edges: list[dict],
    contract: api_contract.Contract,
) -> dict[str, Any]:
    """The stored ``contract_json`` document (contract-endpoint shape).

    Same assembly as GET /api/graph/contract/{name}: defaults advertised
    only when the API would apply them; output types derived from the
    feeding port. ``problems`` is omitted — publish already 409'd on any.
    """
    inputs: list[dict[str, Any]] = []
    for inp in contract.inputs:
        if inp["required"] or inp["type"] == "image":
            advertised_default = None
        else:
            try:
                advertised_default = api_contract.coerce_input(
                    inp["default"], inp["type"], from_string=True,
                )
            except InputCoercionError:
                advertised_default = None
        inputs.append({
            "name": inp["name"], "type": inp["type"],
            "required": inp["required"], "default": advertised_default,
            "description": inp["description"],
        })
    outputs: list[dict[str, Any]] = []
    for out in contract.outputs:
        type_label, _problem = _derive_output_type(out["node_id"], nodes, edges)
        outputs.append({
            "name": out["name"], "type": type_label,
            "description": out["description"],
        })
    return {"graph": graph_name, "inputs": inputs, "outputs": outputs}


class PublishRequest(BaseModel):
    graph: str
    record_io: bool = True
    note: str | None = None
    create: bool = False


class PatchAppRequest(BaseModel):
    record_io: bool


class ActivateRequest(BaseModel):
    version: int


@router.post("/{slug}/publish", dependencies=[Depends(require_session_token)])
async def publish_app(slug: str, body: PublishRequest, request: Request):
    """Snapshot the saved graph as the next immutable version and activate
    it. Publish ACTIVATES IMMEDIATELY — canvas Run + pre-flight parity is
    the v1 verification story (no staging path).

    Publishing to a NONEXISTENT slug requires ``"create": true`` — a
    misspelled slug can no longer silently create a second app.
    """
    db = get_db(request)
    if SLUG_PATTERN.match(slug) is None:
        raise _manage_error(
            422, "invalid_slug",
            "slug must match ^[a-z][a-z0-9-]{0,63}$",
        )

    # Load the saved graph under the strict-name rule
    # (routes_graph_run.py:341-349): execute exactly what was named.
    if _sanitize_name(body.graph) != body.graph:
        raise _manage_error(404, "graph_not_found",
                            f"Graph '{body.graph}' not found")
    path = _graph_path(body.graph)
    if not path.exists():
        raise _manage_error(404, "graph_not_found",
                            f"Graph '{body.graph}' not found")
    try:
        graph_text = path.read_text()
        graph_data = json.loads(graph_text)
    except (ValueError, OSError):
        raise _manage_error(
            500, "graph_unreadable",
            f"Graph file for '{body.graph}' exists but is not valid JSON",
        )
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    # Stage-1 pre-flight, identical checks in identical order — you can
    # never publish a graph that POST /api/graph/run would refuse.
    contract = api_contract.derive_contract(nodes)
    if contract.problems:
        raise _manage_error(409, "invalid_contract",
                            "graph I/O contract has problems",
                            details=contract.problems)
    if not find_entry_points(nodes, edges):
        raise _manage_error(409, "no_entry_points",
                            "graph has no entry points — wire a Start node "
                            "into every GraphInput")
    wiring = api_contract.check_wiring(nodes, edges, contract)
    if wiring.untriggered:
        raise _manage_error(409, "untriggered_input",
                            "GraphInput node(s) have no incoming trigger "
                            "edge — wire Start into every GraphInput",
                            details=wiring.untriggered)
    if wiring.unreachable:
        raise _manage_error(409, "unreachable_output",
                            "GraphOutput node(s) are not reachable from any "
                            "entry point",
                            details=wiring.unreachable)
    validation_errors = validate_graph(nodes, edges)
    if validation_errors:
        raise _manage_error(409, "invalid_graph", "graph failed validation",
                            details=validation_errors)

    contract_doc = _contract_document(body.graph, nodes, edges, contract)
    now = utc_now_iso()

    def _publish(conn: sqlite3.Connection) -> dict[str, Any]:
        # BEGIN IMMEDIATE + max(version)+1 INSIDE the transaction:
        # concurrent publishes to one slug serialize safely into
        # UNIQUE(app_id, version); version-insert and active-pointer flip
        # commit atomically (Decision B — one source of truth).
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "SELECT id FROM apps WHERE slug = ?", (slug,),
            ).fetchone()
            created = row is None
            if created:
                if not body.create:
                    raise _manage_error(
                        404, "app_not_found",
                        f"app '{slug}' does not exist — pass "
                        '"create": true to create it',
                    )
                cur = conn.execute(
                    "INSERT INTO apps (slug, graph_name, active_version, "
                    "record_io, created_at, updated_at) "
                    "VALUES (?, ?, NULL, ?, ?, ?)",
                    (slug, body.graph, int(body.record_io), now, now),
                )
                app_id = cur.lastrowid
            else:
                app_id = row["id"]
            next_version = conn.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 FROM app_versions "
                "WHERE app_id = ?",
                (app_id,),
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO app_versions (app_id, version, graph_json, "
                "contract_json, source_graph_name, note, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (app_id, next_version, graph_text, json.dumps(contract_doc),
                 body.graph, body.note, now),
            )
            conn.execute(
                "UPDATE apps SET active_version = ?, graph_name = ?, "
                "record_io = ?, updated_at = ? WHERE id = ?",
                (next_version, body.graph, int(body.record_io), now, app_id),
            )
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        return {"version": next_version, "created": created}

    result = await db.run(_publish)
    return {
        "slug": slug,
        "version": result["version"],
        "active": True,
        "created": result["created"],
        "graph_name": body.graph,
        "note": body.note,
    }


@router.get("", dependencies=[Depends(require_session_token)])
async def list_apps(request: Request):
    db = get_db(request)

    def _select(conn: sqlite3.Connection) -> list[dict[str, Any]]:
        rows = conn.execute(
            "SELECT a.slug, a.graph_name, a.active_version, a.record_io, "
            "       a.created_at, a.updated_at, "
            "       (SELECT COUNT(*) FROM app_versions v "
            "        WHERE v.app_id = a.id) AS versions_count "
            "FROM apps a ORDER BY a.slug",
        ).fetchall()
        return [dict(r) for r in rows]

    rows = await db.run(_select)
    for row in rows:
        row["record_io"] = bool(row["record_io"])
    return rows


@router.get("/{slug}/versions", dependencies=[Depends(require_session_token)])
async def list_versions(slug: str, request: Request):
    db = get_db(request)

    def _select(conn: sqlite3.Connection) -> list[dict[str, Any]] | None:
        app_row = conn.execute(
            "SELECT id, active_version FROM apps WHERE slug = ?", (slug,),
        ).fetchone()
        if app_row is None:
            return None
        rows = conn.execute(
            "SELECT version, source_graph_name, note, created_at "
            "FROM app_versions WHERE app_id = ? ORDER BY version DESC",
            (app_row["id"],),
        ).fetchall()
        return [
            {**dict(r), "active": r["version"] == app_row["active_version"]}
            for r in rows
        ]

    rows = await db.run(_select)
    if rows is None:
        raise _manage_error(404, "app_not_found", f"app '{slug}' not found")
    return rows


@router.patch("/{slug}", dependencies=[Depends(require_session_token)])
async def patch_app(slug: str, body: PatchAppRequest, request: Request):
    """Recording is app state, not version state — flipping it never
    creates a version."""
    db = get_db(request)
    now = utc_now_iso()

    def _patch(conn: sqlite3.Connection) -> int:
        return conn.execute(
            "UPDATE apps SET record_io = ?, updated_at = ? WHERE slug = ?",
            (int(body.record_io), now, slug),
        ).rowcount

    if await db.run(_patch) == 0:
        raise _manage_error(404, "app_not_found", f"app '{slug}' not found")
    return {"slug": slug, "record_io": body.record_io}


@router.post("/{slug}/activate", dependencies=[Depends(require_session_token)])
async def activate_version(slug: str, body: ActivateRequest, request: Request):
    """Set ``active_version`` to ANY existing version — including from the
    unpublished state (activate-after-unpublish restores service at that
    version). Subsumes rollback; there is no separate rollback route."""
    db = get_db(request)
    now = utc_now_iso()

    def _activate(conn: sqlite3.Connection) -> None:
        conn.execute("BEGIN IMMEDIATE")
        try:
            app_row = conn.execute(
                "SELECT id FROM apps WHERE slug = ?", (slug,),
            ).fetchone()
            if app_row is None:
                raise _manage_error(404, "app_not_found",
                                    f"app '{slug}' not found")
            exists = conn.execute(
                "SELECT 1 FROM app_versions WHERE app_id = ? AND version = ?",
                (app_row["id"], body.version),
            ).fetchone()
            if exists is None:
                raise _manage_error(
                    404, "version_not_found",
                    f"app '{slug}' has no version {body.version}",
                )
            conn.execute(
                "UPDATE apps SET active_version = ?, updated_at = ? "
                "WHERE id = ?",
                (body.version, now, app_row["id"]),
            )
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise

    await db.run(_activate)
    return {"slug": slug, "active_version": body.version}


@router.post("/{slug}/unpublish",
             dependencies=[Depends(require_session_token)])
async def unpublish_app(slug: str, request: Request):
    """``active_version = NULL``; versions and runs are retained."""
    db = get_db(request)
    now = utc_now_iso()

    def _unpublish(conn: sqlite3.Connection) -> int:
        return conn.execute(
            "UPDATE apps SET active_version = NULL, updated_at = ? "
            "WHERE slug = ?",
            (now, slug),
        ).rowcount

    if await db.run(_unpublish) == 0:
        raise _manage_error(404, "app_not_found", f"app '{slug}' not found")
    return {"slug": slug, "active_version": None}


@router.delete("/{slug}", dependencies=[Depends(require_session_token)])
async def delete_app(slug: str, request: Request):
    """IRREVOCABLY removes the app, ALL its versions AND all its run
    records (FK cascade); also prunes the slug's app_locks entry."""
    db = get_db(request)

    def _delete(conn: sqlite3.Connection) -> int:
        return conn.execute(
            "DELETE FROM apps WHERE slug = ?", (slug,),
        ).rowcount

    if await db.run(_delete) == 0:
        raise _manage_error(404, "app_not_found", f"app '{slug}' not found")
    locks = getattr(request.app.state, "app_locks", None)
    if locks is not None:
        locks.pop(slug, None)
    return {"slug": slug, "deleted": True}


def _encode_io(fields: dict[str, Any], *, record_io: bool,
               cap_bytes: int) -> str:
    """Encode one runs IO column (inputs_json / outputs_json).

    Every stored field stays parseable JSON: an over-cap field becomes the
    PINNED marker ``{"__codefyui__": "truncated", "bytes": N}``; with
    ``record_io=false`` every field becomes ``{"__codefyui__":
    "redacted"}``. Never partial JSON — the marker shapes are a
    cross-stage contract Stage 3 switches on.
    """
    if not record_io:
        return json.dumps(
            {name: {"__codefyui__": "redacted"} for name in fields}
        )
    capped: dict[str, Any] = {}
    for name, value in fields.items():
        blob = json.dumps(value)
        size = len(blob.encode("utf-8"))
        if size > cap_bytes:
            capped[name] = {"__codefyui__": "truncated", "bytes": size}
        else:
            capped[name] = value
    return json.dumps(capped)


async def _record_run(
    db: Database,
    *,
    run_id: str,
    app_id: int,
    version: int,
    api_key_id: int,
    envelope: dict[str, Any],
    node_timings: dict[str, float],
    raw_inputs: dict[str, Any],
    record_io: bool,
) -> None:
    """BEST-EFFORT runs INSERT (row-only-if-resolved rule, spec 6.1).

    A failure here loses one audit row, never a run result: log at ERROR
    and return — the run outcome outranks bookkeeping. Also piggybacks
    the retention prune (rate-limited to hourly inside prune_runs).
    """
    error = envelope.get("error") or {}
    timing = envelope.get("timing") or {}
    cap = settings.RUN_IO_CAP_BYTES
    row = (
        run_id, app_id, version, api_key_id, envelope["status"],
        error.get("code"), error.get("message"), error.get("node_id"),
        envelope.get("device"), timing.get("total_s"),
        json.dumps(node_timings),
        _encode_io(raw_inputs, record_io=record_io, cap_bytes=cap),
        _encode_io(envelope.get("outputs") or {}, record_io=record_io,
                   cap_bytes=cap),
        utc_now_iso(),
    )

    def _insert(conn: sqlite3.Connection) -> None:
        # NOTE: keep this closure named `_insert` — the best-effort test
        # injects a fault by matching fn.__name__.
        conn.execute(
            "INSERT INTO runs (run_id, app_id, version, api_key_id, status, "
            "error_code, error_message, error_node_id, device, total_s, "
            "node_timings_json, inputs_json, outputs_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            row,
        )

    try:
        await db.run(_insert)
    except Exception:
        logger.error(
            "failed to record run %s for app_id=%s "
            "(run result still returned)",
            run_id, app_id, exc_info=True,
        )
        return
    try:
        await db.prune_runs(settings.RUNS_RETENTION_DAYS)
    except Exception:
        logger.error("runs retention prune failed", exc_info=True)


@router.post("/{slug}/invoke")
async def invoke_app(
    slug: str,
    request: Request,
    key_result: ApiKeyResult = Depends(require_api_key),
):
    """Execute the app's ACTIVE snapshot version.

    Key-only auth (the session token is NEVER accepted here); every
    response is the 9-key envelope with ``graph`` = ``app`` = slug; a runs
    row is written (best-effort) for every outcome that resolved to an
    app version. Body identical to Stage-1 /run — ``record_outputs`` is
    accepted-and-ignored (Decision H1).
    """
    # 1. run_id at entry — before any rejection (Stage-1 rule).
    run_id = uuid4().hex
    started = time.monotonic()

    # 2. NON-RAISING key check; THIS handler envelopes the 401 and adds
    #    WWW-Authenticate (spec Section 6.3 taxonomy row).
    if not key_result.ok:
        return error_response(
            401, run_id=run_id, graph=slug, app=slug, code="invalid_key",
            message=key_result.failure or "invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 3. Body cap against Content-Length, before reading the body
    #    (routes_graph_run step-2 pattern).
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_bytes = int(content_length)
        except ValueError:
            declared_bytes = 0
        if declared_bytes > settings.MAX_RUN_BODY_BYTES:
            return error_response(
                413, run_id=run_id, graph=slug, app=slug,
                code="payload_too_large",
                message=(
                    f"request body is {declared_bytes} bytes "
                    f"(max {settings.MAX_RUN_BODY_BYTES})"
                ),
            )

    # 4. Resolve slug -> (app, active version, record_io) in ONE SELECT
    #    join — atomic against concurrent publish (the row shows either
    #    the old or the new active version, never a mix).
    db = get_db(request)
    app_locks = _get_app_locks(request)

    def _resolve(conn: sqlite3.Connection) -> dict[str, Any] | None:
        row = conn.execute(
            "SELECT a.id AS app_id, a.active_version, a.record_io, "
            "       v.graph_json "
            "FROM apps a "
            "LEFT JOIN app_versions v "
            "  ON v.app_id = a.id AND v.version = a.active_version "
            "WHERE a.slug = ?",
            (slug,),
        ).fetchone()
        return dict(row) if row is not None else None

    resolved = await db.run(_resolve)
    if resolved is None:
        return error_response(404, run_id=run_id, graph=slug, app=slug,
                              code="app_not_found",
                              message=f"app '{slug}' not found")
    if resolved["active_version"] is None or resolved["graph_json"] is None:
        # graph_json None = belt-and-braces for a dangling active_version
        # (publish/activate both validate, so it cannot normally happen).
        return error_response(409, run_id=run_id, graph=slug, app=slug,
                              code="app_unpublished",
                              message=(
                                  f"app '{slug}' has no active version — "
                                  "publish or activate one first"
                              ))
    version = int(resolved["active_version"])
    record_io = bool(resolved["record_io"])
    api_key_id = int(key_result.key_row["id"])

    # From here the request RESOLVED to an app version: every outcome
    # records a runs row (best-effort) and carries version in the envelope.
    async def _finish(
        http_status: int,
        envelope: dict[str, Any],
        node_timings: dict[str, float],
        raw_inputs: dict[str, Any],
    ):
        envelope["app"] = slug
        envelope["version"] = version
        await _record_run(
            db, run_id=run_id, app_id=int(resolved["app_id"]),
            version=version, api_key_id=api_key_id, envelope=envelope,
            node_timings=node_timings, raw_inputs=raw_inputs,
            record_io=record_io,
        )
        if http_status == 200:
            return envelope
        return JSONResponse(status_code=http_status, content=envelope)

    snapshot = json.loads(resolved["graph_json"])
    nodes = snapshot.get("nodes", [])
    edges = snapshot.get("edges", [])

    # 5. Parse the body via the shared Stage-1 parser (enveloped 422; the
    #    row records whatever inputs parsed — for malformed bodies, {}).
    raw_body = await request.body()
    run_req, field_errors = _parse_run_body(raw_body)
    if field_errors:
        return await _finish(422, build_envelope(
            status="error", run_id=run_id, graph=slug,
            error={"code": "invalid_input", "message": "invalid request body",
                   "node_id": None, "details": field_errors},
        ), {}, run_req.inputs)

    # 6. Per-slug lock (Decision I): the timeout budget covers TOTAL
    #    request time INCLUDING queue wait. FIFO fairness is asyncio.Lock's
    #    default wake order. A client that disconnects while queued is
    #    cancelled before any envelope exists — no runs row.
    lock = app_locks.setdefault(slug, asyncio.Lock())
    remaining = max(0.001, run_req.timeout_s - (time.monotonic() - started))
    try:
        await asyncio.wait_for(lock.acquire(), timeout=remaining)
    except asyncio.TimeoutError:
        return await _finish(500, build_envelope(
            status="error", run_id=run_id, graph=slug,
            error={
                "code": "timeout",
                "message": (
                    f"run exceeded timeout_s={run_req.timeout_s} — "
                    "expired while queued behind another invoke of this app"
                ),
                "node_id": None,
                "details": None,
            },
            timing={"total_s": round(time.monotonic() - started, 3)},
        ), {}, run_req.inputs)
    try:
        # 7. Remaining-budget COPY: execute_contract_run reads
        #    run_req.timeout_s internally — without the copy, queue time
        #    would not count against the execution wait and the
        #    total-budget rule of step 6 would be broken.
        exec_req = replace(
            run_req,
            timeout_s=max(
                0.001, run_req.timeout_s - (time.monotonic() - started),
            ),
        )
        # output_store=None: Decision H1 — isolation is structural, not a
        # flag. The editor inspector store can never contain this data.
        http_status, envelope, node_timings = await execute_contract_run(
            slug, nodes, edges, exec_req, run_id, output_store=None,
        )
    finally:
        # On an execution timeout the lock releases here while the
        # shielded task's in-flight node drains — documented residual
        # overlap, mirrors Stage-1 timeout semantics (spec Section 13).
        lock.release()

    # 8. Best-effort runs INSERT, then the envelope.
    return await _finish(http_status, envelope, node_timings, run_req.inputs)
