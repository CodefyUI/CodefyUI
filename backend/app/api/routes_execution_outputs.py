"""REST endpoints for retrieving captured per-run node outputs.

Complements the WebSocket stream by letting the frontend lazily fetch
full tensor values (or their slices) for the Teaching Inspector panel.

The ``/stats`` route is the counterpart for data too big to fetch: it
summarises a value server-side (see ``core.port_stats``) so the answer to
"what does this data look like" costs a kilobyte instead of two gigabytes.
"""

from __future__ import annotations

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from ..config import settings
from ..core.port_stats import PortStatsCache, compute_port_stats
from ..core.run_output_store import RunOutputStore
from ..core.run_store import json_safe

router = APIRouter(prefix="/api/execution/outputs", tags=["execution-outputs"])

#: Ceiling on stat computations running at once. The Stats tab asks for every
#: port of a node in parallel, and each one is real CPU work over a real
#: tensor, so the default executor's ~32 threads would let a few clicks
#: saturate the machine the graph is training on. An executor rather than an
#: ``asyncio.Semaphore`` because executors are not bound to an event loop —
#: the test transport runs each test on its own.
_STATS_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="port-stats")


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


def _not_captured(what: str) -> str:
    """404 detail that names the switch the user probably left off."""
    return (
        f"{what}. Turn on Record outputs (the Rec toggle in the toolbar) and "
        "re-run the graph to capture port data."
    )


def _parse_slice(slice_str: str) -> tuple[Any, ...] | None:
    """Parse a slice string like '0,:,:,0' into an indexing tuple.

    Each comma-separated piece is either an int or a slice (``start:stop:step``).
    Returns None for empty input. Raises ``ValueError`` on malformed input.
    """
    if not slice_str:
        return None
    pieces: list[Any] = []
    for raw in slice_str.split(","):
        part = raw.strip()
        if part == ":" or part == "":
            pieces.append(slice(None))
            continue
        if ":" in part:
            bits = part.split(":")
            if len(bits) > 3:
                raise ValueError(f"bad slice piece: {part!r}")
            conv = [int(b) if b.strip() else None for b in bits]
            while len(conv) < 3:
                conv.append(None)
            pieces.append(slice(conv[0], conv[1], conv[2]))
            continue
        try:
            pieces.append(int(part))
        except ValueError as e:
            raise ValueError(f"bad slice piece: {part!r}") from e
    return tuple(pieces)


def _serialize_tensor(value: Any, slice_str: str, max_elements: int) -> dict[str, Any]:
    import torch

    tensor: torch.Tensor = value
    full_shape = list(tensor.shape)
    dtype = str(tensor.dtype)

    try:
        slicer = _parse_slice(slice_str)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"invalid slice: {e}")

    if slicer is not None:
        try:
            sliced = tensor[slicer]
        except (IndexError, TypeError) as e:
            raise HTTPException(status_code=400, detail=f"slice failed: {e}")
    else:
        sliced = tensor

    if sliced.numel() > max_elements:
        raise HTTPException(
            status_code=413,
            detail=(
                f"output has {sliced.numel()} elements (max {max_elements}); "
                "supply a narrower 'slice' parameter"
            ),
        )

    summary: dict[str, Any] = {
        "type": "tensor",
        "full_shape": full_shape,
        "dtype": dtype,
        "slice": slice_str or "",
        "sliced_shape": list(sliced.shape),
        "values": sliced.detach().cpu().tolist(),
        "truncated": False,
    }
    if sliced.numel() > 0 and sliced.is_floating_point():
        summary["min"] = round(float(sliced.min()), 6)
        summary["max"] = round(float(sliced.max()), 6)
        summary["mean"] = round(float(sliced.mean()), 6)
    elif sliced.numel() > 0 and sliced.dtype != torch.bool:
        summary["min"] = int(sliced.min())
        summary["max"] = int(sliced.max())
    return summary


def _serialize_value(value: Any, slice_str: str, max_elements: int) -> dict[str, Any]:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return _serialize_tensor(value, slice_str, max_elements)
        if isinstance(value, torch.nn.Module):
            total = sum(p.numel() for p in value.parameters())
            trainable = sum(p.numel() for p in value.parameters() if p.requires_grad)
            return {
                "type": "model",
                "class": value.__class__.__name__,
                "params": total,
                "trainable": trainable,
                "repr": repr(value)[:4000],
            }
    except ImportError:
        pass
    if isinstance(value, bool):
        return {"type": "scalar", "value": value}
    if isinstance(value, (int, float)):
        return {"type": "scalar", "value": value}
    if isinstance(value, str):
        return {"type": "string", "value": value[:4000]}
    if isinstance(value, (list, tuple)):
        out: dict[str, Any] = {
            "type": "list",
            "length": len(value),
            "repr": repr(value)[:4000],
        }
        # Include the actual values when the list is small and JSON-friendly so
        # the Inspector can render token chips, token IDs, and offset pairs
        # without a separate /list endpoint. Anything else stays repr-only.
        if len(value) <= 1024:
            primitive_types = (str, int, float, bool, type(None))
            if all(isinstance(x, primitive_types) for x in value):
                out["values"] = list(value)
            elif all(
                isinstance(x, (list, tuple))
                and len(x) == 2
                and all(isinstance(y, (int, float)) for y in x)
                for x in value
            ):
                out["values"] = [list(x) for x in value]
        return out
    return {"type": type(value).__name__, "repr": repr(value)[:4000]}


def _shape_of(value: Any) -> list[int] | None:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return list(value.shape)
    except ImportError:
        pass
    return None


def _type_label(value: Any) -> str:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return "tensor"
        if isinstance(value, torch.nn.Module):
            return "model"
    except ImportError:
        pass
    if isinstance(value, bool):
        return "scalar"
    if isinstance(value, (int, float)):
        return "scalar"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (list, tuple)):
        return "list"
    return type(value).__name__


@router.get("/{run_id}")
async def list_run_outputs(run_id: str, request: Request):
    store = _get_store(request)
    ports = await store.list_ports(run_id)
    if ports is None:
        raise HTTPException(status_code=404, detail=f"run '{run_id}' not found")
    result = []
    for node_id, port in ports:
        value = await store.get(run_id, node_id, port)
        result.append(
            {
                "node_id": node_id,
                "port": port,
                "type": _type_label(value),
                "full_shape": _shape_of(value),
            }
        )
    return result


@router.delete("/{run_id}")
async def delete_run_outputs(run_id: str, request: Request):
    store = _get_store(request)
    ok = await store.delete_run(run_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"run '{run_id}' not found")
    return {"run_id": run_id, "deleted": True}


@router.get("/{run_id}/{node_id}/__steps_index")
async def get_steps_index(run_id: str, node_id: str, request: Request):
    """List all algorithmic steps recorded for a node in a run.

    Returns a list of ``{index, name, description, scalars, tensor_keys}``
    entries ordered by step index. The frontend uses this to render the
    Steps tab without making N round-trips for individual ``__step__N__meta``
    entries.
    """
    store = _get_store(request)
    if not await store.has_run(run_id):
        raise HTTPException(status_code=404, detail=f"run '{run_id}' not found")
    ports = await store.list_ports(run_id)
    if ports is None:
        raise HTTPException(status_code=404, detail=f"run '{run_id}' not found")
    metas: dict[int, dict[str, Any]] = {}
    for nid, port in ports:
        if nid != node_id or not port.startswith("__step__"):
            continue
        if not port.endswith("__meta"):
            continue
        # port format: __step__{i}__meta
        try:
            idx = int(port[len("__step__"):-len("__meta")])
        except ValueError:
            continue
        meta = await store.get(run_id, node_id, port)
        if isinstance(meta, dict):
            metas[idx] = meta
    return [
        {"index": idx, **metas[idx]}
        for idx in sorted(metas.keys())
    ]


@router.get("/{run_id}/{node_id}/__grad_index")
async def get_grad_index(run_id: str, node_id: str, request: Request):
    """List captured gradients for a node in a run.

    Returns ``[{port, kind, has_grad, health}]`` where:
      - ``kind`` is ``"port"`` (forward output gradient) or ``"weight"``
        (parameter gradient).
      - ``port`` for kind=port is the original forward port name,
        for kind=weight is the parameter name (e.g. ``"weight"``, ``"bias"``).
      - ``health`` is the dict produced by ``backward_pass.grad_health``
        (status/norm/mean/max), or ``None`` if not available.
    """
    store = _get_store(request)
    if not await store.has_run(run_id):
        raise HTTPException(status_code=404, detail=f"run '{run_id}' not found")
    ports = await store.list_ports(run_id)
    if ports is None:
        raise HTTPException(status_code=404, detail=f"run '{run_id}' not found")

    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for nid, port in ports:
        if nid != node_id:
            continue
        if port.endswith("__grad__meta") or port.startswith("__weight_grad__") and port.endswith("__meta"):
            continue
        if port.endswith("__grad"):
            forward_port = port[:-len("__grad")]
            health = await store.get(run_id, node_id, port + "__meta")
            entries.append({
                "port": forward_port,
                "kind": "port",
                "has_grad": True,
                "health": health if isinstance(health, dict) else None,
            })
            seen.add(port)
        elif port.startswith("__weight_grad__"):
            param_name = port[len("__weight_grad__"):]
            health = await store.get(run_id, node_id, port + "__meta")
            entries.append({
                "port": param_name,
                "kind": "weight",
                "has_grad": True,
                "health": health if isinstance(health, dict) else None,
            })
            seen.add(port)
    return entries


@router.get("/{run_id}/{node_id}/{port}/stats")
async def get_output_stats(run_id: str, node_id: str, port: str, request: Request):
    """Summary statistics for one captured port, computed server-side.

    Never proportional to the input: a tensor comes back as a fixed set of
    scalars plus a 64-bin histogram (or, for label tensors, up to 64 value
    counts), which is a kilobyte or two whatever the tensor weighs.

    The compute runs on a worker thread. A 2 GB tensor takes about a second of
    real CPU work, and doing that on the event loop would stall every open
    WebSocket — including the run that produced the tensor.
    """
    store = _get_store(request)
    if not await store.has_run(run_id):
        raise HTTPException(
            status_code=404,
            detail=_not_captured(f"run '{run_id}' has no captured outputs"),
        )
    slot = await store.get_with_version(run_id, node_id, port)
    if slot is None or slot[0] is None:
        raise HTTPException(
            status_code=404,
            detail=_not_captured(
                f"nothing captured for '{node_id}.{port}' in run '{run_id}'"
            ),
        )
    value, version = slot

    cache = _get_stats_cache(request)
    # The store's write serial is part of the KEY. A node inside a loop
    # overwrites one port several times in a run, and the replacement can
    # land on the freed storage of the value before it — same address, same
    # shape, same dtype — so anything derived from the object itself would
    # happily answer for a tensor that no longer exists.
    key = (run_id, node_id, port, version)
    payload = cache.get(key)
    if payload is None:
        loop = asyncio.get_running_loop()
        stats = await loop.run_in_executor(
            _STATS_EXECUTOR,
            functools.partial(
                compute_port_stats,
                value,
                sample_threshold=settings.STATS_SAMPLE_THRESHOLD,
                sample_size=settings.STATS_SAMPLE_SIZE,
            ),
        )
        payload = {"run_id": run_id, "node_id": node_id, "port": port, **stats}
        cache.put(key, payload)
    return payload


@router.get("/{run_id}/{node_id}/{port}")
async def get_output(
    run_id: str,
    node_id: str,
    port: str,
    request: Request,
    slice: str = Query(default=""),
    max_elements: int = Query(default=4096, ge=1, le=1_000_000),
):
    store = _get_store(request)
    if not await store.has_run(run_id):
        raise HTTPException(status_code=404, detail=f"run '{run_id}' not found")
    value = await store.get(run_id, node_id, port)
    if value is None:
        raise HTTPException(
            status_code=404,
            detail=f"output '{node_id}.{port}' not found in run '{run_id}'",
        )
    payload = _serialize_value(value, slice, max_elements)
    payload["run_id"] = run_id
    payload["node_id"] = node_id
    payload["port"] = port
    # A diverged tensor's raw values/min/max/mean can be NaN/Inf. Starlette
    # renders with allow_nan=False, so one leaked value 500s the whole
    # response -- same hazard as #129, same fix: the run_store.json_safe
    # convention established there (see the /stats route above).
    return json_safe(payload)
