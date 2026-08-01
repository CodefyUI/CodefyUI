"""REST surface for server-owned graph runs (#120).

Six endpoints over ``RunService`` + ``RunStore``; no execution logic lives
here. The point of the surface is that a run's lifetime is no longer tied to
any connection: submit from one client, close it, and query status, events
and metrics from another.

    POST /api/runs                       submit  -> {run_id, status}
    GET  /api/runs                       list (newest first, + queue position)
    GET  /api/runs/{id}                  one row + last event cursor
    POST /api/runs/{id}/cancel           cooperative stop
    GET  /api/runs/{id}/events           replay after a cursor, optional long poll
    GET  /api/runs/{id}/metrics          series as JSON or CSV

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
unknown run, 503 when the service is not on ``app.state`` (the
``routes_execution_outputs._get_store`` precedent — the lifespan does not
run under httpx's ASGITransport).
"""

from __future__ import annotations

import csv
import io
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel

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

#: Upper bound for one queue-position sweep. A queue deeper than this is
#: #123's problem, not a reason to scan the whole table on every poll.
_QUEUE_SCAN_LIMIT = 1000

#: Byte budget for ONE /events response. The per-event cap
#: (RUN_EVENT_PAYLOAD_CAP_BYTES) bounds a single payload; this bounds the
#: page, so ``limit`` cannot multiply the two into a response nobody asked
#: for. A short page is already normal — the cursor says where to resume —
#: so stopping early costs the client one extra round trip and nothing else.
#: Far above any honest page (ordinary events are a couple of KB).
_EVENTS_RESPONSE_CAP_BYTES = 4 * 1024 * 1024

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


async def _queue_positions(service: RunService,
                           records: list[RunRecord]) -> dict[str, int]:
    """1-based position of each QUEUED run, oldest first.

    Costs one extra query, and only when a queued row is actually present —
    which today is never (every submit starts immediately). The field exists
    from day one anyway so #123 can fill the queue without changing the
    response shape out from under a client.
    """
    if not any(record.status == STATUS_QUEUED for record in records):
        return {}
    queued = await service.store.list_runs(status=STATUS_QUEUED,
                                           limit=_QUEUE_SCAN_LIMIT)
    # list_runs is newest-first; a queue is served oldest-first.
    return {record.id: index
            for index, record in enumerate(reversed(queued), start=1)}


def _run_payload(record: RunRecord, *, queue_position: int | None,
                 active: bool, last_cursor: int | None = None) -> dict[str, Any]:
    payload = asdict(record)
    payload["queue_position"] = queue_position
    payload["active"] = active
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
    """
    out: list[dict[str, Any]] = []
    total = 0
    for event in events:
        total += json_size(event.payload)
        if out and total > _EVENTS_RESPONSE_CAP_BYTES:
            break
        out.append(_event_payload(event))
    return out


async def _require_run(service: RunService, run_id: str) -> RunRecord:
    record = await service.store.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"run '{run_id}' not found")
    return record


@router.post("")
async def submit_run(body: SubmitRunRequest, request: Request):
    """Persist and start a run. Returns as soon as it is scheduled.

    Deliberately does NOT wait for the run: that is the entire point of the
    endpoint. Progress is read back through ``/events`` (long-pollable) and
    the row itself.
    """
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
    return {
        "runs": [
            _run_payload(record,
                         queue_position=positions.get(record.id),
                         active=service.is_active(record.id))
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
    return _run_payload(
        record,
        queue_position=positions.get(record.id),
        active=service.is_active(run_id),
        last_cursor=await service.store.latest_cursor(run_id),
    )


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
