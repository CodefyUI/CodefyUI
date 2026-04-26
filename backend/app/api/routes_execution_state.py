"""REST endpoints for managing persistent node state (A2).

Lets the frontend reset persisted layer weights — both per-node ("forget the
weights for this Conv2d") and per-graph ("clear all weights for this tab").
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..core.node_state_store import NodeStateStore

router = APIRouter(prefix="/api/execution/state", tags=["execution-state"])


class ResetRequest(BaseModel):
    graph_id: str
    node_ids: list[str] | None = None


def _get_store(request: Request) -> NodeStateStore:
    store = getattr(request.app.state, "node_state_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="node_state_store not initialised")
    return store


@router.post("/reset")
async def reset_state(payload: ResetRequest, request: Request) -> dict[str, Any]:
    """Reset persisted weights.

    * If ``node_ids`` is omitted or empty, all persisted modules for the
      graph are dropped.
    * Otherwise only the named nodes are reset.
    """
    store = _get_store(request)
    if not payload.node_ids:
        evicted = store.reset_graph(payload.graph_id)
        scope = "graph"
    else:
        evicted = 0
        for nid in payload.node_ids:
            evicted += store.reset_node(payload.graph_id, nid)
        scope = "nodes"
    return {
        "graph_id": payload.graph_id,
        "scope": scope,
        "evicted": evicted,
    }


@router.get("/list")
async def list_state(request: Request, graph_id: str | None = None) -> dict[str, Any]:
    """List how many modules are persisted (overall or for one graph).

    Diagnostic endpoint — keeps the LRU bookkeeping observable without
    exposing module internals.
    """
    store = _get_store(request)
    if graph_id is None:
        return {"total": len(store)}
    items = list(store.iter_for_graph(graph_id))
    return {
        "graph_id": graph_id,
        "count": len(items),
        "node_ids": sorted({k[1] for k, _m in items}),
    }
