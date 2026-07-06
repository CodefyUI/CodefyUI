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
import functools
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
from .routes_graph import GraphAmbiguityError, _graph_path, _sanitize_name

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/graph", tags=["graph-run"])


def build_envelope(
    *,
    status: str,
    run_id: str,
    graph: str,
    app: str | None = None,
    version: int | None = None,
    device: str | None = None,
    outputs: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
    timing: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Construct the one true run envelope: all nine keys, always present.

    ``run_id`` is never null — it is assigned at request entry, before any
    rejection can happen. ``app``/``version`` stay None on the editor
    route; the invoke route fills them per the spec value rules.
    """
    return RunEnvelope(
        status=status,
        run_id=run_id,
        graph=graph,
        app=app,
        version=version,
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
    app: str | None = None,
    version: int | None = None,
    device: str | None = None,
    node_id: str | None = None,
    details: list[Any] | None = None,
    timing: dict[str, float] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """An enveloped error; the HTTP status mirrors ``error.code``.

    ``headers`` lets the invoke 401 carry ``WWW-Authenticate: Bearer``.
    """
    return JSONResponse(
        status_code=http_status,
        headers=headers,
        content=build_envelope(
            status="error",
            run_id=run_id,
            graph=graph,
            app=app,
            version=version,
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
    try:
        path = _graph_path(name)
    except GraphAmbiguityError as e:
        raise HTTPException(status_code=409, detail=str(e))
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


async def execute_contract_run(
    graph_label: str,
    nodes: list[dict],
    edges: list[dict],
    run_req: _RunRequest,
    run_id: str,
    output_store: Any,
) -> tuple[int, dict[str, Any], dict[str, float]]:
    """Steps 5-12 of a contract run: pre-flight, inject, execute, collect,
    serialize. Shared by the editor route below and Stage-2 invoke
    (``routes_apps`` imports it — cross-route import precedent: this
    module already imports from ``routes_graph``).

    Structured return ``(http_status, envelope_dict, node_timings)`` — no
    Response objects inside, so each route wraps it in its own transport
    shape while staying WIRE-IDENTICAL to the pre-extraction handler.

    - ``graph_label`` fills the envelope's ``graph`` key ("the name you
      addressed"): graph name on the editor route, slug on invoke.
    - ``run_req.timeout_s`` is read internally — the invoke route passes
      a COPY holding only the remaining post-queue budget (Decision I).
    - ``output_store`` is the hoisted app.state getattr; invoke passes
      None so published runs can NEVER reach the unauthenticated
      inspector store (Decision H1 — structural isolation).
    - ``node_timings`` maps node_id -> seconds (Decision F2); ``cached``/
      ``skipped`` arrive without a prior "running" and record 0.0. It
      rides the return value for runs-row persistence; the envelope
      ``timing`` stays ``{"total_s": ...}``.
    """

    def _error(
        http_status: int,
        *,
        code: str,
        message: str,
        device: str | None = None,
        node_id: str | None = None,
        details: list[Any] | None = None,
        timing: dict[str, float] | None = None,
        node_timings: dict[str, float] | None = None,
    ) -> tuple[int, dict[str, Any], dict[str, float]]:
        return (
            http_status,
            build_envelope(
                status="error", run_id=run_id, graph=graph_label,
                device=device, outputs=None,
                error={"code": code, "message": message,
                       "node_id": node_id, "details": details},
                timing=timing,
            ),
            node_timings or {},
        )

    # 5. Pre-flight on the raw graph — nobody pays for a full execution to
    #    learn about mis-wiring. On invoke this also catches STALE
    #    snapshots (e.g. a plugin-provided node type uninstalled after
    #    publish) as 409 invalid_graph instead of a raw 500.
    contract = api_contract.derive_contract(nodes)
    if contract.problems:
        return _error(409, code="invalid_contract",
                      message="graph I/O contract has problems",
                      details=contract.problems)
    if not find_entry_points(nodes, edges):
        return _error(409, code="no_entry_points",
                      message=(
                          "graph has no entry points — wire a Start "
                          "node into every GraphInput"
                      ))
    wiring = api_contract.check_wiring(nodes, edges, contract)
    if wiring.untriggered:
        return _error(409, code="untriggered_input",
                      message=(
                          "GraphInput node(s) have no incoming trigger "
                          "edge — wire Start into every GraphInput"
                      ),
                      details=wiring.untriggered)
    if wiring.unreachable:
        return _error(409, code="unreachable_output",
                      message=(
                          "GraphOutput node(s) are not reachable from "
                          "any entry point"
                      ),
                      details=wiring.unreachable)
    validation_errors = validate_graph(nodes, edges)
    if validation_errors:
        return _error(409, code="invalid_graph",
                      message="graph failed validation",
                      details=validation_errors)

    # 6. Inject the RAW request values; each GraphInput's execute() coerces.
    # inject_inputs does synchronous base64+PIL+ToTensor validation for
    # image inputs — offload to the default executor so a large/slow image
    # decode never blocks the event loop. inject_inputs is pure, so running
    # it off-thread is safe.
    patched_nodes, input_errors = await asyncio.get_running_loop().run_in_executor(
        None,
        functools.partial(api_contract.inject_inputs, nodes, contract, run_req.inputs),
    )
    if input_errors:
        return _error(422, code="invalid_input", message="invalid inputs",
                      details=input_errors)

    # 7. Fresh per-request context: with persistence off the stateful mixin
    #    rebuilds modules per call, so concurrent requests share no mutable
    #    state; the app-global stores are simply not used.
    device = resolve_device(run_req.device)
    ctx = ExecutionContext(
        device=device,
        weights_persistent=False,
        node_state_store=None,
        graph_id=f"api:{graph_label}",
    )

    # 8. Launch as an INDEPENDENT task and await it under a shielded
    #    timeout. The shield means handler cancellation (client disconnect)
    #    never propagates into the run — only the timeout stops a run.
    # Minimal error capture: remember the last node that reported an error
    # so execution_error envelopes carry a node_id. Per-node timings stamp
    # on "running" and close on completed/cached/skipped/error (F2).
    last_error_node_id: dict[str, str | None] = {"value": None}
    node_started: dict[str, float] = {}
    node_timings: dict[str, float] = {}

    async def _on_progress(
        node_id: str, status: str, data: dict[str, Any] | None
    ) -> None:
        if status == "running":
            node_started[node_id] = time.monotonic()
            return
        if status in ("completed", "cached", "skipped", "error"):
            # cached/skipped arrive WITHOUT a prior "running" (the engine
            # emits them directly): the dict.get guard records 0.0.
            started = node_started.get(node_id)
            node_timings[node_id] = (
                round(time.monotonic() - started, 3)
                if started is not None else 0.0
            )
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
        # On py3.11+ asyncio.TimeoutError IS builtins.TimeoutError, so a
        # NODE raising TimeoutError (e.g. a socket timeout) completes the
        # shielded task with that exact exception before wait_for's own
        # deadline ever fires — indistinguishable from a genuine timeout by
        # exception type alone. Disambiguate with task.done(): if the task
        # already finished, the exception came from graph execution and
        # must be reported like any other execution_error (with node_id);
        # only an incomplete task means wait_for's deadline itself expired.
        if task.done() and not task.cancelled():
            task_exc = task.exception()
            if task_exc is not None:
                return _error(
                    500, code="execution_error", message=str(task_exc),
                    device=device, node_id=last_error_node_id["value"],
                    timing={"total_s": round(time.monotonic() - t0, 3)},
                    node_timings=node_timings,
                )
        # 10. Cooperative cancellation: observed at node boundaries only —
        # the node currently inside run_in_executor finishes in its thread
        # after this 500 is sent (documented limitation).
        ctx.cancel()
        return _error(500, code="timeout",
                      message=f"run exceeded timeout_s={run_req.timeout_s}",
                      device=device,
                      timing={"total_s": round(time.monotonic() - t0, 3)},
                      node_timings=node_timings)
    except GraphValidationError as exc:
        # 9. Runtime safety net: preset expansion can invalidate a
        # pre-flight-clean graph (pruning-induced missing required input;
        # trigger-edges-into-preset-nodes dangling after expand_presets).
        return _error(409, code="invalid_graph",
                      message="graph failed validation at runtime",
                      device=device, details=[str(exc)],
                      timing={"total_s": round(time.monotonic() - t0, 3)},
                      node_timings=node_timings)
    except Exception as exc:  # noqa: BLE001 — never a raw, unenveloped 500
        # 11. asyncio.CancelledError (client disconnect) is BaseException,
        # not Exception — it passes through and the shielded run continues.
        return _error(500, code="execution_error", message=str(exc),
                      device=device, node_id=last_error_node_id["value"],
                      timing={"total_s": round(time.monotonic() - t0, 3)},
                      node_timings=node_timings)
    total_s = round(time.monotonic() - t0, 3)

    # 12. Collect + serialize declared outputs.
    collected, missing = api_contract.collect_outputs(contract, engine_result)
    if missing:
        return _error(500, code="output_not_produced",
                      message=(
                          "declared output(s) missing from the engine "
                          "result: " + ", ".join(missing)
                      ),
                      device=device, details=missing,
                      timing={"total_s": total_s}, node_timings=node_timings)
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
        return _error(500, code=serialization_code,
                      message="output serialization failed",
                      device=device, details=serialization_errors,
                      timing={"total_s": total_s}, node_timings=node_timings)

    return 200, build_envelope(
        status="ok", run_id=run_id, graph=graph_label, device=device,
        outputs=outputs_json, error=None, timing={"total_s": total_s},
    ), node_timings


@router.post("/run/{name}")
async def run_graph_as_function(name: str, request: Request):
    """Execute a saved graph as a named function: declared inputs in,
    declared outputs out. Every response uses the 9-key envelope
    (``app``/``version`` stay null on this editor route)."""
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
    try:
        path = _graph_path(name)
    except GraphAmbiguityError as e:
        return error_response(409, run_id=run_id, graph=name,
                              code="invalid_graph", message=str(e))
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

    # Steps 5-12 live in execute_contract_run. The output_store getattr is
    # hoisted HERE (editor-only surface); the invoke route passes None.
    output_store = None
    if run_req.record_outputs:
        # The lifespan does not run under httpx ASGITransport, so the
        # attribute may be absent — recording is then silently skipped
        # (ws_execution.py getattr precedent).
        output_store = getattr(request.app.state, "run_output_store", None)

    http_status, envelope, _node_timings = await execute_contract_run(
        name, nodes, edges, run_req, run_id, output_store,
    )
    if http_status != 200:
        return JSONResponse(status_code=http_status, content=envelope)
    return envelope
