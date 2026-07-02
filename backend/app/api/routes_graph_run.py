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

import json
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..core import api_contract
from ..core.api_contract import InputCoercionError
from ..core.node_registry import registry
from ..schemas import (
    ContractInputSchema,
    ContractOutputSchema,
    GraphContractResponse,
    RunEnvelope,
    RunError,
    RunTiming,
)
from .routes_graph import _graph_path, _sanitize_name

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


def _derive_output_type(
    node_id: str, nodes: list[dict], edges: list[dict]
) -> tuple[str, str | None]:
    """Resolve a GraphOutput's advertised type from its incoming data edge.

    Same lookup the validator does: source node class ->
    ``define_outputs_dynamic(params)`` -> port ``data_type``. Returns
    ``(type_label, problem)``; unresolvable outputs advertise ``"ANY"``
    plus a problem entry.
    """
    incoming = [
        e
        for e in edges
        if e.get("target") == node_id
        and e.get("targetHandle", "") == "value"
        and e.get("type", "data") == "data"
    ]
    if not incoming:
        return "ANY", (
            f"output node '{node_id}' has no incoming connection — type unresolved"
        )
    edge = incoming[0]
    node_map = {n.get("id"): n for n in nodes}
    src = node_map.get(edge.get("source"))
    if src is None:
        return "ANY", (
            f"output node '{node_id}': source node '{edge.get('source')}' not found"
        )
    src_cls = registry.get(src.get("type", ""))
    if src_cls is None:
        return "ANY", (
            f"output node '{node_id}': unknown source node type "
            f"'{src.get('type', '')}'"
        )
    src_data = src.get("data")
    src_params = src_data.get("params", {}) if isinstance(src_data, dict) else {}
    ports = {p.name: p for p in src_cls.define_outputs_dynamic(src_params)}
    port = ports.get(edge.get("sourceHandle", ""))
    if port is None:
        return "ANY", (
            f"output node '{node_id}': source port "
            f"'{edge.get('sourceHandle', '')}' not found"
        )
    return port.data_type.value, None


@router.get("/contract/{name}", response_model=GraphContractResponse)
async def get_contract(name: str):
    """Derived I/O contract for a saved graph.

    Problems are *reported* here (non-fatally, so users can inspect a
    half-built graph) and *enforced* with 409 only by POST /run.
    Unauthenticated GET, like /api/graph/load; 404s are plain
    ``HTTPException`` shapes matching the load endpoint.
    """
    if _sanitize_name(name) != name:
        # Strict-name rule: never silently alias to a different file.
        raise HTTPException(status_code=404, detail=f"Graph '{name}' not found")
    path = _graph_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Graph '{name}' not found")
    try:
        graph_data = json.loads(path.read_text())
    except (ValueError, OSError):
        raise HTTPException(
            status_code=500, detail=f"Graph '{name}' exists but is not valid JSON"
        )

    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    contract = api_contract.derive_contract(nodes)
    problems = list(contract.problems)

    inputs: list[ContractInputSchema] = []
    for inp in contract.inputs:
        if inp["required"] or inp["type"] == "image":
            # Never advertise a default the API will not apply. (An image
            # default is a canvas-only file path — also never applied.)
            advertised_default = None
        else:
            try:
                advertised_default = api_contract.coerce_input(
                    inp["default"], inp["type"], from_string=True
                )
            except InputCoercionError:
                advertised_default = None  # already reported in problems
        inputs.append(ContractInputSchema(
            name=inp["name"],
            type=inp["type"],
            required=inp["required"],
            default=advertised_default,
            description=inp["description"],
        ))

    outputs: list[ContractOutputSchema] = []
    for out in contract.outputs:
        type_label, problem = _derive_output_type(out["node_id"], nodes, edges)
        if problem is not None:
            problems.append(problem)
        outputs.append(ContractOutputSchema(
            name=out["name"], type=type_label, description=out["description"],
        ))

    return GraphContractResponse(
        graph=name, inputs=inputs, outputs=outputs, problems=problems,
    )
