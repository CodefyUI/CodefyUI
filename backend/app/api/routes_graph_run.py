"""Graph-as-a-function endpoints: call a saved graph headlessly over HTTP.

POST /api/graph/run/{name}      — execute with declared inputs, return declared outputs
GET  /api/graph/contract/{name} — inspect the derived I/O contract

The contract is declared on the canvas with GraphInput / GraphOutput
nodes; the pure helpers live in ``app.core.api_contract``. This router
shares the ``/api/graph`` prefix with ``routes_graph`` (multiple routers
may share a prefix) but keeps run/serialization logic out of the CRUD
file. POST is auto-covered by the X-CodefyUI-Token middleware.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..schemas import RunEnvelope, RunError, RunTiming

router = APIRouter(prefix="/api/graph", tags=["graph-run"])


def build_envelope(
    *,
    status: str,
    run_id: str,
    graph: str,
    device: str | None = None,
    outputs: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
    timing: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Construct the one true /run envelope: all seven keys, always present.

    ``run_id`` is never null — it is assigned at request entry, before any
    rejection can happen.
    """
    return RunEnvelope(
        status=status,
        run_id=run_id,
        graph=graph,
        device=device,
        outputs=outputs,
        error=RunError(**error) if error is not None else None,
        timing=RunTiming(**timing) if timing is not None else None,
    ).model_dump()


def error_response(
    http_status: int,
    *,
    run_id: str,
    graph: str,
    code: str,
    message: str,
    device: str | None = None,
    node_id: str | None = None,
    details: list[Any] | None = None,
    timing: dict[str, float] | None = None,
) -> JSONResponse:
    """An enveloped error; the HTTP status mirrors ``error.code``."""
    return JSONResponse(
        status_code=http_status,
        content=build_envelope(
            status="error",
            run_id=run_id,
            graph=graph,
            device=device,
            outputs=None,
            error={
                "code": code,
                "message": message,
                "node_id": node_id,
                "details": details,
            },
            timing=timing,
        ),
    )
