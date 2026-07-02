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

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ..config import settings
from ..core import api_contract
from ..core.api_contract import InputCoercionError, OutputSerializationError
from ..core.device_utils import resolve_device
from ..core.execution_context import ExecutionContext
from ..core.graph_engine import (
    GraphValidationError,
    execute_graph,
    find_entry_points,
    validate_graph,
)
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

logger = logging.getLogger(__name__)

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


@dataclass
class _RunRequest:
    """Validated fields of the OPTIONAL /run body (absent body == {})."""

    inputs: dict[str, Any] = field(default_factory=dict)
    timeout_s: float = 300.0
    device: str | None = None
    record_outputs: bool = False


def _parse_run_body(raw: bytes) -> tuple[_RunRequest, list[dict[str, str]]]:
    """Manual in-handler body parse.

    Malformed JSON, a non-dict body, a non-dict ``inputs``, or wrong-typed
    ``timeout_s``/``device``/``record_outputs`` all yield enveloped 422
    field errors — FastAPI's ``RequestValidationError`` can never bypass
    the envelope. The Content-Type header is not used for dispatch.
    Unknown body fields are ignored (forward compatibility).
    """
    req = _RunRequest()
    if not raw or not raw.strip():
        return req, []
    try:
        body = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return req, [{"field": "body", "reason": "body is not valid JSON"}]
    if not isinstance(body, dict):
        return req, [{
            "field": "body",
            "reason": (
                f"body must be a JSON object, got {api_contract.json_type_name(body)}"
            ),
        }]

    errors: list[dict[str, str]] = []

    inputs = body.get("inputs", {})
    if isinstance(inputs, dict):
        req.inputs = inputs
    else:
        errors.append({
            "field": "inputs",
            "reason": f"expected object, got {api_contract.json_type_name(inputs)}",
        })

    timeout_s = body.get("timeout_s", 300)
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
        errors.append({
            "field": "timeout_s",
            "reason": f"expected number, got {api_contract.json_type_name(timeout_s)}",
        })
    elif not (1 <= timeout_s <= 3600):
        errors.append({
            "field": "timeout_s",
            "reason": f"must be between 1 and 3600, got {timeout_s}",
        })
    else:
        req.timeout_s = float(timeout_s)

    device = body.get("device")
    if device is not None and not isinstance(device, str):
        errors.append({
            "field": "device",
            "reason": f"expected string, got {api_contract.json_type_name(device)}",
        })
    else:
        req.device = device

    record_outputs = body.get("record_outputs", False)
    if not isinstance(record_outputs, bool):
        errors.append({
            "field": "record_outputs",
            "reason": (
                f"expected boolean, got {api_contract.json_type_name(record_outputs)}"
            ),
        })
    else:
        req.record_outputs = record_outputs

    return req, errors


def _retrieve_background_exception(task: asyncio.Task) -> None:
    """Consume a background run's exception.

    After a timeout (or client disconnect) the shielded task keeps running;
    without this done-callback asyncio would log 'Task exception was never
    retrieved' when it eventually fails.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.debug("background graph run ended with error after response: %s", exc)


@router.post("/run/{name}")
async def run_graph_as_function(name: str, request: Request):
    """Execute a saved graph as a named function: declared inputs in,
    declared outputs out. Every response uses the 7-key envelope."""
    # 1. run_id at request entry — every envelope carries it, including
    #    pre-flight rejections.
    run_id = uuid4().hex

    # 2. Body size cap, checked against Content-Length BEFORE reading the body.
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_bytes = int(content_length)
        except ValueError:
            declared_bytes = 0
        if declared_bytes > settings.MAX_RUN_BODY_BYTES:
            return error_response(
                413, run_id=run_id, graph=name, code="payload_too_large",
                message=(
                    f"request body is {declared_bytes} bytes "
                    f"(max {settings.MAX_RUN_BODY_BYTES})"
                ),
            )

    # 3. Strict name matching: execute exactly what was named, or nothing.
    if _sanitize_name(name) != name:
        return error_response(404, run_id=run_id, graph=name,
                              code="graph_not_found",
                              message=f"Graph '{name}' not found")
    path = _graph_path(name)
    if not path.exists():
        return error_response(404, run_id=run_id, graph=name,
                              code="graph_not_found",
                              message=f"Graph '{name}' not found")
    try:
        graph_data = json.loads(path.read_text())
    except (ValueError, OSError):
        return error_response(500, run_id=run_id, graph=name,
                              code="graph_unreadable",
                              message=(
                                  f"Graph file for '{name}' exists but is not "
                                  "valid JSON"
                              ))
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    # 4. Parse + validate the body in-handler (enveloped 422).
    raw_body = await request.body()
    run_req, field_errors = _parse_run_body(raw_body)
    if field_errors:
        return error_response(422, run_id=run_id, graph=name,
                              code="invalid_input",
                              message="invalid request body",
                              details=field_errors)

    # 5. Pre-flight on the raw graph — nobody pays for a full execution to
    #    learn about mis-wiring.
    contract = api_contract.derive_contract(nodes)
    if contract.problems:
        return error_response(409, run_id=run_id, graph=name,
                              code="invalid_contract",
                              message="graph I/O contract has problems",
                              details=contract.problems)
    if not find_entry_points(nodes, edges):
        return error_response(409, run_id=run_id, graph=name,
                              code="no_entry_points",
                              message=(
                                  "graph has no entry points — wire a Start "
                                  "node into every GraphInput"
                              ))
    wiring = api_contract.check_wiring(nodes, edges, contract)
    if wiring.untriggered:
        return error_response(409, run_id=run_id, graph=name,
                              code="untriggered_input",
                              message=(
                                  "GraphInput node(s) have no incoming trigger "
                                  "edge — wire Start into every GraphInput"
                              ),
                              details=wiring.untriggered)
    if wiring.unreachable:
        return error_response(409, run_id=run_id, graph=name,
                              code="unreachable_output",
                              message=(
                                  "GraphOutput node(s) are not reachable from "
                                  "any entry point"
                              ),
                              details=wiring.unreachable)
    validation_errors = validate_graph(nodes, edges)
    if validation_errors:
        return error_response(409, run_id=run_id, graph=name,
                              code="invalid_graph",
                              message="graph failed validation",
                              details=validation_errors)

    # 6. Inject the RAW request values; each GraphInput's execute() coerces.
    patched_nodes, input_errors = api_contract.inject_inputs(
        nodes, contract, run_req.inputs,
    )
    if input_errors:
        return error_response(422, run_id=run_id, graph=name,
                              code="invalid_input",
                              message="invalid inputs",
                              details=input_errors)

    # 7. Fresh per-request context: with persistence off the stateful mixin
    #    rebuilds modules per call, so concurrent requests share no mutable
    #    state; the app-global stores are simply not used.
    device = resolve_device(run_req.device)
    ctx = ExecutionContext(
        device=device,
        weights_persistent=False,
        node_state_store=None,
        graph_id=f"api:{name}",
    )
    output_store = None
    if run_req.record_outputs:
        # The lifespan does not run under httpx ASGITransport, so the
        # attribute may be absent — recording is then silently skipped
        # (ws_execution.py getattr precedent).
        output_store = getattr(request.app.state, "run_output_store", None)

    # 8. Launch as an INDEPENDENT task and await it under a shielded
    #    timeout. The shield means handler cancellation (client disconnect)
    #    never propagates into the run — only the timeout stops a run.
    # Minimal error capture: remember the last node that reported an error
    # so execution_error envelopes carry a node_id.
    last_error_node_id: dict[str, str | None] = {"value": None}

    async def _on_progress(
        node_id: str, status: str, data: dict[str, Any] | None
    ) -> None:
        if status == "error":
            last_error_node_id["value"] = node_id

    t0 = time.monotonic()
    task = asyncio.create_task(execute_graph(
        patched_nodes,
        edges,
        on_progress=_on_progress,
        context=ctx,
        error_mode="fail_fast",
        run_id=run_id,
        output_store=output_store,
        record_outputs=run_req.record_outputs and output_store is not None,
    ))
    task.add_done_callback(_retrieve_background_exception)
    try:
        engine_result = await asyncio.wait_for(
            asyncio.shield(task), timeout=run_req.timeout_s,
        )
    except asyncio.TimeoutError:
        # 10. Cooperative cancellation: observed at node boundaries only —
        # the node currently inside run_in_executor finishes in its thread
        # after this 500 is sent (documented limitation).
        ctx.cancel()
        return error_response(500, run_id=run_id, graph=name, code="timeout",
                              message=f"run exceeded timeout_s={run_req.timeout_s}",
                              device=device,
                              timing={"total_s": round(time.monotonic() - t0, 3)})
    except GraphValidationError as exc:
        # 9. Runtime safety net: preset expansion can invalidate a
        # pre-flight-clean graph (pruning-induced missing required input;
        # trigger-edges-into-preset-nodes dangling after expand_presets).
        return error_response(409, run_id=run_id, graph=name,
                              code="invalid_graph",
                              message="graph failed validation at runtime",
                              device=device, details=[str(exc)],
                              timing={"total_s": round(time.monotonic() - t0, 3)})
    except Exception as exc:  # noqa: BLE001 — never a raw, unenveloped 500
        # 11. asyncio.CancelledError (client disconnect) is BaseException,
        # not Exception — it passes through and the shielded run continues.
        return error_response(500, run_id=run_id, graph=name,
                              code="execution_error",
                              message=str(exc), device=device,
                              node_id=last_error_node_id["value"],
                              timing={"total_s": round(time.monotonic() - t0, 3)})
    total_s = round(time.monotonic() - t0, 3)

    # 12. Collect + serialize declared outputs.
    collected, missing = api_contract.collect_outputs(contract, engine_result)
    if missing:
        return error_response(500, run_id=run_id, graph=name,
                              code="output_not_produced",
                              message=(
                                  "declared output(s) missing from the engine "
                                  "result: " + ", ".join(missing)
                              ),
                              device=device, details=missing,
                              timing={"total_s": total_s})
    outputs_json: dict[str, Any] = {}
    serialization_errors: list[dict[str, str]] = []
    serialization_code = "unserializable_output"
    for output_name, value in collected.items():
        try:
            outputs_json[output_name] = api_contract.serialize_output(value)
        except OutputSerializationError as exc:
            serialization_errors.append(
                {"output": output_name, "reason": exc.reason}
            )
            if exc.code == "output_too_large":
                # When both kinds occur, the size violation names the code —
                # deterministic and the more actionable of the two.
                serialization_code = "output_too_large"
    if serialization_errors:
        return error_response(500, run_id=run_id, graph=name,
                              code=serialization_code,
                              message="output serialization failed",
                              device=device, details=serialization_errors,
                              timing={"total_s": total_s})

    return build_envelope(status="ok", run_id=run_id, graph=name, device=device,
                          outputs=outputs_json, error=None,
                          timing={"total_s": total_s})
