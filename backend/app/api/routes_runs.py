"""REST surface for server-owned graph runs (#120).

Eight endpoints over ``RunService`` + ``RunStore``; no execution logic lives
here. The point of the surface is that a run's lifetime is no longer tied to
any connection: submit from one client, close it, and query status, events
and metrics from another.

    POST   /api/runs                     submit  -> {run_id, status}
    GET    /api/runs                     list (newest first, + queue position)
    GET    /api/runs/{id}                one row + last event cursor
    DELETE /api/runs/{id}                drop a FINISHED run and its children
    POST   /api/runs/{id}/cancel         cooperative stop
    GET    /api/runs/{id}/events         replay after a cursor, optional long poll
    GET    /api/runs/{id}/metrics        series as JSON or CSV
    GET    /api/runs/{id}/artifacts      checkpoints and other recorded files

Auth follows the house rule in ``main.py`` exactly: ``auth_guard`` requires
the session token for the two mutating routes (POST), reads are open like
every other GET in the app. This router is deliberately NOT added to
``_AUTH_EXEMPT_PREFIXES`` — the exemption exists for ``/api/apps`` and
``/api/keys``, which carry per-route dependencies instead; a run route
gaining an exemption without one would silently drop authentication (which
is what ``test_auth_drift.py`` guards for those two, and what
``test_api_runs.test_runs_routes_are_not_under_an_auth_exempt_prefix``
guards here).

Errors: 400 for a malformed submit or an unknown status filter, 404 for an
unknown run, 413 for a submit over ``MAX_RUN_BODY_BYTES``, 503 when the
service is not on ``app.state`` (the ``routes_execution_outputs._get_store``
precedent — the lifespan does not run under httpx's ASGITransport). All of
them answer ``{"detail": ...}``; the 9-key run envelope belongs to
``routes_graph_run`` / ``routes_apps``, not here.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from pydantic import BaseModel, ValidationError

from ..config import settings
from ..core.run_service import (
    RunService,
    RunServiceUnavailable,
    RunSubmitError,
    json_size,
)
from ..core.run_store import (
    RUN_STATUSES,
    STATUS_QUEUED,
    EventRecord,
    MetricRecord,
    RunRecord,
)

router = APIRouter(prefix="/api/runs", tags=["runs"])

#: Upper bound for one queue-position sweep. Deeper than this and a run's
#: exact rank stops being actionable; scanning the whole table on every poll
#: of the Runs panel to produce it never was.
_QUEUE_SCAN_LIMIT = 1000

_CSV_COLUMNS = ("run_id", "node_id", "name", "step", "value", "ts")


def _get_service(request: Request) -> RunService:
    service = getattr(request.app.state, "run_service", None)
    if service is None:
        raise HTTPException(status_code=503,
                            detail="run service not initialised")
    return service


class SubmitRunRequest(BaseModel):
    """``{graph, options, name}``.

    ``graph`` is typed only as an object here; its shape is checked by
    ``normalize_graph`` so the envelope rules live in ONE place next to the
    service that enforces them, and pydantic never has to mirror them.
    """

    graph: dict[str, Any]
    options: Any = None
    name: str | None = None


def _reject_oversized_body(request: Request) -> None:
    """413 when ``Content-Length`` declares more than ``MAX_RUN_BODY_BYTES``.

    Same setting, same comparison and same message text as the two sibling
    submit routes (``routes_graph_run`` step 2, ``routes_apps`` step 3), so a
    client that talks to more than one of them sees one behaviour. Only the
    envelope differs, and only because these routes have different envelopes:
    the sibling routes answer with the 9-key run envelope on every path, this
    router answers with ``{"detail": ...}`` on every path (see the module
    docstring). Borrowing their envelope for this one status would make
    ``POST /api/runs`` the only route in the file a client had to special-case.

    Checked against the header BEFORE anything reads the body, which is why
    ``submit_run`` parses its own body instead of declaring a pydantic
    parameter: FastAPI reads and JSON-parses a declared body before the
    endpoint (and before its dependencies) runs, so a cap placed after that
    point has already paid for the bytes it is refusing.

    Advisory, exactly like the siblings: a chunked request declares no length
    and is not caught here. The cap that matters for this route is the one on
    what gets *persisted* — every submit writes an immutable ``graph_snapshot``
    row, a queued run never reaches a terminal state, and the keep-last-N
    retention only prunes finished runs (``RUN_RETENTION_KEEP_LAST``), so
    without this nothing at all reclaims an oversized submit.
    """
    content_length = request.headers.get("content-length")
    if content_length is None:
        return
    try:
        declared_bytes = int(content_length)
    except ValueError:
        declared_bytes = 0
    if declared_bytes > settings.MAX_RUN_BODY_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"request body is {declared_bytes} bytes "
                f"(max {settings.MAX_RUN_BODY_BYTES})"
            ),
        )


async def _read_submit_body(request: Request) -> SubmitRunRequest:
    """Read and validate the submit body by hand, keeping FastAPI's 422.

    Hand-parsing is the price of capping before the read; it must not also
    cost callers the error shape they already handle. Both failure modes are
    re-raised as ``RequestValidationError`` with the same ``loc`` prefix
    FastAPI applies (``("body", ...)``) and ``include_url=False``, so a bad
    submit answers byte-for-byte what a declared pydantic body answered.
    """
    raw = await request.body()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RequestValidationError([{
            "type": "json_invalid",
            "loc": ("body", exc.pos),
            "msg": "JSON decode error",
            "input": {},
            "ctx": {"error": exc.msg},
        }], body=raw)
    try:
        return SubmitRunRequest.model_validate(payload)
    except ValidationError as exc:
        raise RequestValidationError(
            [
                {**err, "loc": ("body", *err.get("loc", ()))}
                for err in exc.errors(include_url=False)
            ],
            body=payload,
        )


async def _queue_positions(service: RunService,
                           records: list[RunRecord]) -> dict[str, int]:
    """1-based position of each QUEUED run WITHIN ITS OWN DEVICE QUEUE.

    Per ``queue_key``, not global, because that is what the scheduler is
    (#123): a CPU run behind two other CPU runs is third in line even when
    six CUDA runs were submitted before it, and a global rank would tell it
    "position 9" for a queue it is not in.

    Computed from the ROWS rather than asked of the in-memory scheduler, and
    that is deliberate. The two agree by construction — a run is enqueued in
    submit order, and ``created_at, rowid`` IS submit order — while the rows
    also cover what the scheduler cannot see: a ``queued`` orphan left by a
    process that died before ``recover_interrupted`` ran still gets an
    honest place in line instead of a null.

    Costs one extra indexed query per request, and only when a queued row is
    actually present — nothing to pay while the queues are empty, which is
    the common case, and one bounded scan when the Runs panel is polling a
    backlog.

    Past ``_QUEUE_SCAN_LIMIT`` the answer is ``None`` for everyone rather
    than a number. ``list_runs`` is NEWEST-first, so a truncated scan drops
    the OLDEST rows — the ones actually at the front — and every position
    computed from what is left would be confidently too small. "We cannot
    say" is the honest answer; a wrong rank is worse than no rank.
    """
    if not any(record.status == STATUS_QUEUED for record in records):
        return {}
    queued = await service.store.list_runs(status=STATUS_QUEUED,
                                           limit=_QUEUE_SCAN_LIMIT)
    if len(queued) >= _QUEUE_SCAN_LIMIT:
        return {}
    # list_runs is newest-first; a queue is served oldest-first.
    depth: dict[str | None, int] = {}
    positions: dict[str, int] = {}
    for record in reversed(queued):
        depth[record.queue_key] = depth.get(record.queue_key, 0) + 1
        positions[record.id] = depth[record.queue_key]
    return positions


def _run_payload(record: RunRecord, *, queue_position: int | None,
                 active: bool, last_cursor: int | None = None,
                 final_metrics: dict[str, float] | None = None) -> dict[str, Any]:
    payload = asdict(record)
    payload["queue_position"] = queue_position
    payload["active"] = active
    payload["final_metrics"] = final_metrics or {}
    if last_cursor is not None:
        payload["last_cursor"] = last_cursor
    return payload


def _event_payload(event: EventRecord) -> dict[str, Any]:
    # ``run_id`` is dropped: it is already on the envelope, and repeating it
    # on every event is pure weight on a replay of thousands.
    return {"cursor": event.cursor, "type": event.type,
            "payload": event.payload, "ts": event.ts}


def _bounded_events(events: list[EventRecord]) -> list[dict[str, Any]]:
    """Serialise events up to the response byte budget, in cursor order.

    Always yields at least one event so an oversized single event can never
    wedge a follower's cursor. Costs one size measurement per event on the
    read path; /events is polled far less often than events are produced,
    and the alternative — trusting ``limit`` alone — makes the response size
    a product of two independently-configured numbers.

    The budget is read from ``settings`` per call rather than captured in a
    module constant, so ``CODEFYUI_RUN_EVENTS_RESPONSE_CAP_BYTES`` actually
    reaches it — the previous literal here could not be configured at all.
    """
    cap_bytes = settings.RUN_EVENTS_RESPONSE_CAP_BYTES
    out: list[dict[str, Any]] = []
    total = 0
    for event in events:
        total += json_size(event.payload)
        if out and total > cap_bytes:
            break
        out.append(_event_payload(event))
    return out


async def _require_run(service: RunService, run_id: str) -> RunRecord:
    record = await service.store.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"run '{run_id}' not found")
    return record


@router.post("", openapi_extra={
    # Hand-parsing the body takes it out of the generated schema, and an
    # endpoint whose /docs page shows no request body is worse documentation
    # than the cap is worth. Re-declare the same model, inlined because
    # nothing references the component any more.
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {
                "schema": SubmitRunRequest.model_json_schema(),
            },
        },
    },
})
async def submit_run(request: Request):
    """Persist and schedule a run. Returns as soon as it is scheduled.

    Deliberately does NOT wait for the run: that is the entire point of the
    endpoint. Progress is read back through ``/events`` (long-pollable) and
    the row itself.

    ``status`` is ``running`` or ``queued`` — the latter when the run's
    device is already at its concurrency limit and the run joined that
    device's FIFO (#123). Poll the row, or long-poll ``/events``, either
    way; a queued run parks the long poll properly instead of returning
    empty pages.

    The body is capped and parsed here rather than declared as a pydantic
    parameter — see ``_reject_oversized_body`` for why the order matters.
    """
    _reject_oversized_body(request)
    body = await _read_submit_body(request)
    service = _get_service(request)
    try:
        result = await service.submit(body.graph, options=body.options,
                                      name=body.name)
    except RunSubmitError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RunServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"run_id": result.run_id, "status": result.status}


@router.get("")
async def list_runs(
    request: Request,
    status: list[str] | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """Newest first. ``?status=`` repeats (``?status=queued&status=running``).

    ``total`` is the unpaged count so a table can size itself without
    walking every page.

    Each row also carries ``final_metrics`` — the last value of every series
    the run recorded (#124). One grouped query for the whole page, so the
    Runs table can print a final loss per row without a metrics request per
    row; ``{}`` for a run that recorded nothing.
    """
    service = _get_service(request)
    # Validate the CLIENT's input here rather than catching ValueError off
    # the store call: the store raises the same exception type for a corrupt
    # `options` column, and blaming the caller with a 400 for the server's
    # own bad row would send whoever debugs it in exactly the wrong
    # direction. A corrupt row now surfaces as the 500 it is.
    if status is not None:
        unknown = sorted(set(status) - RUN_STATUSES)
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"unknown run status {unknown}; expected one of "
                       f"{sorted(RUN_STATUSES)}")
    records = await service.store.list_runs(status=status, limit=limit,
                                            offset=offset)
    total = await service.store.count_runs(status=status)
    positions = await _queue_positions(service, records)
    finals = await service.store.latest_metrics([r.id for r in records])
    return {
        "runs": [
            _run_payload(record,
                         queue_position=positions.get(record.id),
                         active=service.is_active(record.id),
                         final_metrics=finals.get(record.id))
            for record in records
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/{run_id}")
async def get_run(run_id: str, request: Request):
    """One row plus ``last_cursor`` — where an event follower should start."""
    service = _get_service(request)
    record = await _require_run(service, run_id)
    positions = await _queue_positions(service, [record])
    finals = await service.store.latest_metrics([record.id])
    return _run_payload(
        record,
        queue_position=positions.get(record.id),
        active=service.is_active(run_id),
        last_cursor=await service.store.latest_cursor(run_id),
        final_metrics=finals.get(record.id),
    )


@router.delete("/{run_id}")
async def delete_run(run_id: str, request: Request):
    """Drop a FINISHED run; its metrics, events and artifacts cascade.

    Refuses a queued or running row with a 409. That is the same policy
    ``RunStore.prune`` already encodes ("active runs are never deleted"), and
    it is checked against the SERVICE rather than the row's ``status``
    string: the row is the slower of the two: a run cancelled a millisecond
    ago is still ``running`` in SQL while its task deregisters, and a queued
    run's place in the FIFO lives only in memory. Deleting either would
    leave the scheduler holding a handle to a row that no longer exists.

    ``deleted`` is always True here — a run that was never there is a 404,
    not a quiet success, because the Runs panel showed the user a row and
    "it was already gone" is information they should see.

    Artifact FILES are left on disk, exactly as ``RunStore.delete_run`` says:
    a checkpoint is the user's, not the run log's, and this endpoint is
    history cleanup rather than a delete-my-weights button.

    Captured OUTPUTS are a different matter and DO go: ``RunOutputStore``
    keys its in-memory tensors by the same run id, so leaving them behind
    would keep a deleted run's activations resident until its LRU slot was
    reused — and ``/api/execution/outputs`` would keep serving them for a
    run that answers 404 everywhere else.
    """
    service = _get_service(request)
    await _require_run(service, run_id)
    if service.is_active(run_id) or service.is_queued(run_id):
        raise HTTPException(
            status_code=409,
            detail=f"run '{run_id}' is queued or running; cancel it first")
    deleted = await service.store.delete_run(run_id)
    if not deleted:
        # The row existed at _require_run and is gone now: another client
        # deleted it in between. Same answer as if we had lost the race by a
        # little more, which is what a 404 already means here.
        raise HTTPException(status_code=404, detail=f"run '{run_id}' not found")
    outputs = getattr(request.app.state, "run_output_store", None)
    if outputs is not None:
        await outputs.delete_run(run_id)
    return {"run_id": run_id, "deleted": True}


@router.post("/{run_id}/cancel")
async def cancel_run(run_id: str, request: Request):
    """Ask a run to stop; cooperative, so ``status`` may still say running.

    ``cancelled`` reports whether the request did anything: False for a run
    that had already finished. Both are 200 — asking twice is not an error,
    and a cancel that raced a completion is normal (the user hits Stop as
    the last epoch lands).
    """
    service = _get_service(request)
    outcome = await service.cancel(run_id)
    if outcome is None:
        raise HTTPException(status_code=404, detail=f"run '{run_id}' not found")
    return {"run_id": outcome.run_id, "status": outcome.status,
            "cancelled": outcome.cancelled}


@router.get("/{run_id}/events")
async def get_run_events(
    run_id: str,
    request: Request,
    cursor: int = Query(default=0, ge=0),
    wait: float = Query(default=0.0, ge=0.0, le=60.0),
    limit: int = Query(default=500, ge=1, le=2000),
):
    """Events strictly after *cursor*, oldest first.

    ``wait`` seconds of long polling when the tail is empty: the request
    sleeps on an in-process wake-up (never a retry loop) and returns the
    moment an event lands, the run ends, or the deadline passes — whichever
    comes first, so a finished run answers immediately regardless of
    ``wait``. The returned ``cursor`` is where to resume, and never moves
    backwards on an empty page.

    A page can come back SHORTER than ``limit`` even when more events exist:
    the response also has a byte budget. Resume from ``cursor`` — which is
    what a follower does anyway — and the next page continues.
    """
    service = _get_service(request)
    record = await _require_run(service, run_id)
    events = await service.wait_for_events(run_id, after_cursor=cursor,
                                           limit=limit, wait=wait)
    if wait > 0:
        # The status may well have changed while we were parked.
        record = await service.store.get_run(run_id) or record
    page = _bounded_events(events)
    return {
        "run_id": run_id,
        "status": record.status,
        "active": service.is_active(run_id),
        "events": page,
        # The cursor tracks what was actually RETURNED, not what was read:
        # a byte-budgeted short page must not tell a follower to skip the
        # events it did not receive.
        "cursor": page[-1]["cursor"] if page else cursor,
    }


def _metrics_csv(run_id: str, metrics: list[MetricRecord]) -> str:
    buffer = io.StringIO(newline="")
    # Explicit lineterminator: csv defaults to \r\n, and the module's own
    # newline="" contract means that would survive verbatim into the body.
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(_CSV_COLUMNS)
    for point in metrics:
        writer.writerow([
            point.run_id, point.node_id or "", point.name, point.step,
            # A non-finite value is stored as NULL (a diverged loss); an
            # EMPTY cell is what every spreadsheet reads as a gap, whereas
            # "None" would read as text and poison the column's type.
            "" if point.value is None else point.value,
            point.ts,
        ])
    return buffer.getvalue()


def _csv_filename(run_id: str) -> str:
    """A filename that cannot break out of the header.

    Run ids are our own uuid4 hex, but this endpoint's id comes off the URL,
    so the value is whitelisted rather than trusted.
    """
    safe = "".join(c for c in run_id if c.isalnum() or c in "-_")[:64]
    return f"run-{safe}-metrics.csv" if safe else "run-metrics.csv"


@router.get("/{run_id}/metrics")
async def get_run_metrics(
    run_id: str,
    request: Request,
    name: str | None = Query(default=None),
    format: str = Query(default="json", pattern="^(json|csv)$"),
):
    """Recorded scalar series, ordered ``(name, step)`` — chart order.

    ``format=csv`` returns a download for spreadsheet/pandas use; the JSON
    form additionally lists every series name so a legend can be built
    without scanning the points.
    """
    service = _get_service(request)
    await _require_run(service, run_id)
    metrics = await service.store.get_metrics(run_id, name=name)
    if format == "csv":
        return Response(
            content=_metrics_csv(run_id, metrics),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition":
                     f'attachment; filename="{_csv_filename(run_id)}"'},
        )
    return {
        "run_id": run_id,
        "names": await service.store.list_metric_names(run_id),
        "metrics": [
            {"node_id": point.node_id, "name": point.name, "step": point.step,
             "value": point.value, "ts": point.ts}
            for point in metrics
        ],
    }


@router.get("/{run_id}/artifacts")
async def get_run_artifacts(
    run_id: str,
    request: Request,
    kind: str | None = Query(default=None),
):
    """Files the run recorded — checkpoints, exports, images — oldest first.

    ``exec_run_artifacts`` rows also ride the event log as ``artifact``
    events, and until now that was the only way to reach them. Mining the
    log works for a client that watched the whole run and nothing else: the
    Runs panel opens on a run it did NOT watch, and tailing a long run's
    events to find three checkpoints means replaying thousands of node
    statuses to reconstruct a list the store can answer in one indexed read.

    ``kind`` is not validated against a whitelist — the vocabulary is open
    by design (``ARTIFACT_KIND_*`` are conveniences, not a constraint), so
    an unknown kind is an empty list rather than a 400.
    """
    service = _get_service(request)
    await _require_run(service, run_id)
    artifacts = await service.store.list_artifacts(run_id, kind=kind)
    return {
        "run_id": run_id,
        "artifacts": [
            {"id": record.id, "kind": record.kind, "path": record.path,
             "meta": record.meta, "created_at": record.created_at}
            for record in artifacts
        ],
    }
