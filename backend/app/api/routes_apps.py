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

import json
import logging
import re
import sqlite3
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ..core import api_contract
from ..core.api_contract import InputCoercionError
from ..core.api_keys import get_db, require_session_token
from ..core.db import utc_now_iso
from ..core.graph_engine import find_entry_points, validate_graph
from .routes_graph import _graph_path, _sanitize_name
from .routes_graph_run import _derive_output_type

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
