"""REST endpoints for retrieving captured per-run node outputs.

Complements the WebSocket stream by letting the frontend lazily fetch
full tensor values (or their slices) for the Teaching Inspector panel.

The ``/stats`` route is the counterpart for data too big to fetch: it
summarises a value server-side (see ``core.port_stats``) so the answer to
"what does this data look like" costs a kilobyte instead of two gigabytes.

Every read also has a query form, ``/{run_id}/value|stats|steps|grads`` with
``?node_id=&port=`` (#621), for a node id the path cannot carry: a node inside
a block runs as ``<instance>/<inner>``, and the server decodes ``%2F`` before
routing, so the path form never reaches a handler for it.

The reads themselves are ``core.output_queries``, over this process's
``RunOutputStore``. A run that executed on a worker with ``record_outputs``
(#789) keeps its captures THERE: each read is forwarded to the worker over
its ``/ws/worker`` connection (``core.worker_outputs``), which answers it
with the same functions, so the answer is the one a local run would get.
Which worker holds a run is retained in the store by run id (its retention
evicts the worker's runs as its own); after a restart of this server, the
run's record names it. A worker that is not connected answers 503, one that
does not answer in time 504, each naming the worker. A run held here never
reaches any of that: its reads are what they were before #789.
"""

from __future__ import annotations

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse

from ..config import settings
from ..core import output_queries as q
from ..core.port_stats import PortStatsCache, compute_port_stats, worker_stats_key
from ..core.run_output_store import RunOutputStore
from ..core.worker_outputs import WorkerTimeout, WorkerUnreachable

router = APIRouter(prefix="/api/execution/outputs", tags=["execution-outputs"])

#: Ceiling on stat computations running at once. The Stats tab asks for every
#: port of a node in parallel, and each one is real CPU work over a real
#: tensor, so the default executor's ~32 threads would let a few clicks
#: saturate the machine the graph is training on. An executor rather than an
#: ``asyncio.Semaphore`` because executors are not bound to an event loop —
#: the test transport runs each test on its own.
_STATS_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="port-stats")

# Kept under their old names: tests and callers import them from here.
_parse_slice = q.parse_slice
_preview_slice = q.preview_slice


def _get_store(request: Request) -> RunOutputStore:
    store = getattr(request.app.state, "run_output_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="run_output_store not initialised")
    return store


def _get_stats_cache(request: Request) -> PortStatsCache:
    """The app-wide stats LRU, created on first use if the lifespan didn't.

    ``main.lifespan`` installs one for the real server. The fallback is for
    the test transport, which never runs the lifespan (see
    ``tests/conftest.py``) — and a cache is an optimisation, so a missing one
    is worth creating rather than 503-ing over.
    """
    cache = getattr(request.app.state, "port_stats_cache", None)
    if cache is None:
        cache = PortStatsCache(max_bytes=settings.STATS_CACHE_MAX_BYTES)
        request.app.state.port_stats_cache = cache
    return cache


async def compute_stats(value: Any) -> dict[str, Any]:
    """``compute_port_stats`` on :data:`_STATS_EXECUTOR`.

    A 2 GB tensor takes about a second of real CPU work, and doing that on
    the event loop would stall every open WebSocket — including the run
    that produced the tensor.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _STATS_EXECUTOR,
        functools.partial(
            compute_port_stats,
            value,
            sample_threshold=settings.STATS_SAMPLE_THRESHOLD,
            sample_size=settings.STATS_SAMPLE_SIZE,
        ),
    )


async def _answer(read: Any) -> Any:
    """Await one ``output_queries`` read and turn its outcome into a
    response: the payload, an empty 204, or the read's HTTP error."""
    try:
        payload = await read
    except q.OutputQueryError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.detail)
    if payload is q.NO_VALUE:
        return Response(status_code=204)
    return payload


# ── worker runs (#789) ───────────────────────────────────────────────────────


async def _worker_of(request: Request, store: RunOutputStore,
                     run_id: str) -> tuple[str | None, bool]:
    """``(worker, retained)``: the worker holding *run_id*'s captures, and
    whether the store retains it; ``(None, False)`` for a run read here.

    A run held in this process is never a worker's. One the store retains
    as a worker's is its worker's. One the store evicted is no one's: it
    answers as any run not held does, and is not found again. Anything
    else is looked up once in the run's record: after a restart of this
    server the store has forgotten every worker run, but a run recorded on
    a worker with ``record_outputs`` still has its captures there (until
    that worker's own budget drops them).
    """
    if await store.has_run(run_id):
        return None, False
    holder = await store.remote_holder(run_id)
    if holder is not None:
        return holder, True
    if await store.was_evicted(run_id):
        return None, False
    service = getattr(request.app.state, "run_service", None)
    run_store = getattr(service, "store", None)
    if run_store is None:
        return None, False
    from ..core.run_store import WORKER_ANY, WORKER_LOCAL

    try:
        record = await run_store.get_run(run_id)
    except Exception:  # noqa: BLE001 -- unreadable: answered as not held
        return None, False
    if record is None or record.worker_id in (WORKER_LOCAL, WORKER_ANY) \
            or not record.options.get("record_outputs"):
        return None, False
    return record.worker_id, False


async def _from_worker(request: Request, store: RunOutputStore, worker: str,
                       retained: bool, run_id: str, op: str,
                       params: dict[str, Any]) -> Any:
    """Forward one read to *worker* and answer with what it says."""
    registry = getattr(request.app.state, "worker_registry", None)
    if registry is None:
        raise HTTPException(status_code=503, detail=(
            f"the outputs this run recorded are kept on worker {worker!r}, "
            "and this server has no worker registry"))
    cache = _get_stats_cache(request) if op == "stats" else None
    key = worker_stats_key(run_id, params.get("node_id"), params.get("port"))
    if cache is not None and retained:
        cached = cache.get(key)
        if cached is not None:
            return cached
    try:
        answer = await registry.outputs.fetch(worker, run_id, op, params)
    except WorkerUnreachable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except WorkerTimeout as exc:
        raise HTTPException(status_code=504, detail=str(exc))
    gone = answer.final and q.run_not_held(run_id, answer.status, answer.body)
    if retained and gone:
        # Its worker no longer has it (its own budget dropped it, or it was
        # deleted there): it gives its place up, and is not asked again.
        await store.forget_remote(run_id, expire=True)
    elif not retained and answer.status != 404:
        # Found on its worker after a restart of this server: retained from
        # now on, so its eviction tells the worker.
        await store.add_remote(run_id, worker)
    if answer.status == 204:
        return Response(status_code=204)
    if answer.status >= 400:
        detail = answer.body.get("detail") if isinstance(answer.body, dict) \
            else None
        raise HTTPException(status_code=answer.status,
                            detail=detail or "the worker refused the read")
    if cache is not None and answer.final:
        # Final: the run has ended on the worker, so its captures (and their
        # stats) cannot change. Dropped with the run's retention.
        cache.put(key, answer.body)
    return JSONResponse(answer.body)


# ── routes ───────────────────────────────────────────────────────────────────


@router.get("/{run_id}")
async def list_run_outputs(run_id: str, request: Request):
    store = _get_store(request)
    worker, retained = await _worker_of(request, store, run_id)
    if worker is not None:
        return await _from_worker(request, store, worker, retained, run_id,
                                  "list", {})
    return await _answer(q.list_outputs(store, run_id))


@router.delete("/{run_id}")
async def delete_run_outputs(run_id: str, request: Request):
    store = _get_store(request)
    worker, retained = await _worker_of(request, store, run_id)
    if worker is not None:
        reached = False
        try:
            response = await _from_worker(request, store, worker, retained,
                                          run_id, "delete", {})
            reached = True
            return response
        except HTTPException as exc:
            reached = exc.status_code not in (503, 504)
            raise
        finally:
            # Deleted there, or to be: this server stops retaining it and
            # never takes it back (a later read answers as for a run not
            # held). A worker that could not be reached is told when it
            # next connects.
            await store.forget_remote(run_id, expire=True)
            registry = getattr(request.app.state, "worker_registry", None)
            if not reached and registry is not None:
                registry.outputs.drop(run_id, worker)
    ok = await store.delete_run(run_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"run '{run_id}' not found")
    return {"run_id": run_id, "deleted": True}


@router.get("/{run_id}/{node_id}/__steps_index")
async def get_steps_index(
    run_id: str, node_id: str, request: Request, inner: bool = False,
):
    """List all algorithmic steps recorded for a node in a run.

    Returns a list of ``{index, name, description, scalars, tensor_keys}``
    entries ordered by step index. The frontend uses this to render the
    Steps tab without making N round-trips for individual ``__step__N__meta``
    entries.

    ``inner=true`` reads a block or preset card (#559): the steps of every
    node inside it, each entry naming the node that recorded it in
    ``node_id`` (read its tensors from there), grouped by node in the order
    the run captured them.
    """
    store = _get_store(request)
    worker, retained = await _worker_of(request, store, run_id)
    if worker is not None:
        return await _from_worker(request, store, worker, retained, run_id,
                                  "steps", {"node_id": node_id, "inner": inner})
    return await _answer(q.steps_index(store, run_id, node_id, inner))


@router.get("/{run_id}/{node_id}/__grad_index")
async def get_grad_index(
    run_id: str, node_id: str, request: Request, inner: bool = False,
):
    """List captured gradients for a node in a run.

    Returns ``[{port, kind, has_grad, health}]`` where:
      - ``kind`` is ``"port"`` (forward output gradient) or ``"weight"``
        (parameter gradient).
      - ``port`` for kind=port is the original forward port name,
        for kind=weight is the parameter name (e.g. ``"weight"``, ``"bias"``).
      - ``health`` is the dict produced by ``backward_pass.grad_health``
        (status/norm/mean/max), or ``None`` if not available.

    ``inner=true`` reads a block or preset card (#559) as
    :func:`get_steps_index` does: every node inside it, each entry naming
    its node in ``node_id``.
    """
    store = _get_store(request)
    worker, retained = await _worker_of(request, store, run_id)
    if worker is not None:
        return await _from_worker(request, store, worker, retained, run_id,
                                  "grads", {"node_id": node_id, "inner": inner})
    return await _answer(q.grad_index(store, run_id, node_id, inner))


@router.get("/{run_id}/{node_id}/{port}/stats")
async def get_output_stats(run_id: str, node_id: str, port: str, request: Request):
    """Summary statistics for one captured port, computed server-side.

    Never proportional to the input: a tensor comes back as a fixed set of
    scalars plus a 64-bin histogram (or, for label tensors, up to 64 value
    counts), which is a kilobyte or two whatever the tensor weighs.

    The compute runs on a worker thread (:func:`compute_stats`). The
    app-wide cache is keyed on the store's write serial: a node inside a
    loop overwrites one port several times in a run, and the replacement
    can land on the freed storage of the value before it — same address,
    same shape, same dtype — so anything derived from the object itself
    would happily answer for a tensor that no longer exists.

    A worker run's stats are computed on the worker and cached here once
    the run has ended there (#789).
    """
    store = _get_store(request)
    worker, retained = await _worker_of(request, store, run_id)
    if worker is not None:
        return await _from_worker(request, store, worker, retained, run_id,
                                  "stats", {"node_id": node_id, "port": port})
    return await _answer(q.output_stats(
        store, run_id, node_id, port, compute=compute_stats,
        cache=_get_stats_cache(request)))


@router.get("/{run_id}/{node_id}/{port}")
async def get_output(
    run_id: str,
    node_id: str,
    port: str,
    request: Request,
    slice: str = Query(default=""),
    max_elements: int = Query(default=4096, ge=1, le=1_000_000),
    preview: bool = Query(default=False),
):
    store = _get_store(request)
    worker, retained = await _worker_of(request, store, run_id)
    if worker is not None:
        return await _from_worker(
            request, store, worker, retained, run_id, "value",
            {"node_id": node_id, "port": port, "slice": slice,
             "max_elements": max_elements, "preview": preview})
    # A port that produced None answers an empty 204 (#602); a diverged
    # tensor's NaN/Inf leave as null (``json_safe``), see the read.
    return await _answer(q.output_value(
        store, run_id, node_id, port, slice_str=slice,
        max_elements=max_elements, preview=preview))


# ── Query form, for ids the path cannot carry (#621) ─────────────────────────
#
# The engine flattens a block before it runs, so a node inside one is captured
# as ``<instance>/<inner>`` (``<outer>/<inner instance>/<node>`` further in).
# The path routes above cannot address that id: ``%2F`` is decoded before
# routing, and ``blk%2Fmul/tensor`` matches no route. ``{node_id:path}`` would
# match, but ambiguously -- ``/run/blk/norm/stats`` is both "stats of (blk,
# norm)" and "the value of (blk/norm, stats)", and Normalize has a real output
# named ``stats``. So the id travels as a query parameter, under a second path
# segment no other route has (none has exactly two), and each route hands over
# to the path form's handler, so every answer -- 200, 204, 400, 404, 413 -- is
# the same. The client uses these only for ids containing ``/``.


@router.get("/{run_id}/value")
async def get_output_query(
    run_id: str,
    request: Request,
    node_id: str = Query(...),
    port: str = Query(...),
    slice: str = Query(default=""),
    max_elements: int = Query(default=4096, ge=1, le=1_000_000),
    preview: bool = Query(default=False),
):
    return await get_output(
        run_id, node_id, port, request,
        slice=slice, max_elements=max_elements, preview=preview,
    )


@router.get("/{run_id}/stats")
async def get_output_stats_query(
    run_id: str,
    request: Request,
    node_id: str = Query(...),
    port: str = Query(...),
):
    return await get_output_stats(run_id, node_id, port, request)


@router.get("/{run_id}/steps")
async def get_steps_index_query(
    run_id: str, request: Request, node_id: str = Query(...),
    inner: bool = False,
):
    return await get_steps_index(run_id, node_id, request, inner)


@router.get("/{run_id}/grads")
async def get_grad_index_query(
    run_id: str, request: Request, node_id: str = Query(...),
    inner: bool = False,
):
    return await get_grad_index(run_id, node_id, request, inner)
