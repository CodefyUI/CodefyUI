"""REST surface for worker agents (#772).

    GET  /api/workers                  every known worker (open read)
    POST /api/workers/{id}/drain       no new runs on it (session token)
    POST /api/workers/{id}/undrain     take new runs again (session token)

Auth follows ``routes_runs``: reads are open, the two POSTs are covered by
``auth_guard`` in ``main.py`` (session token). The list includes workers
that disconnected since the server started (``status: disconnected``):
the registry is in memory until #780. 404 for an unknown worker, 503 when
the registry is not on ``app.state`` (the lifespan does not run under
httpx's ASGITransport).

Slots (#773): every device entry carries ``slots: {"limit", "taken"}``,
the run queue's cap for that ``(worker, device)`` slot and how many running
runs hold it; ``null`` when the run service is not on ``app.state`` or the
device report cannot be parsed. ``same_host: true`` marks a worker on this
machine (its ``hello.machine_id`` equals this server's). A device of such a
worker that is physically one of this server's (``cpu``, ``mps:0``, a CUDA
card with a matching UUID) carries ``shared_with: "local"``, otherwise
``null``: its ``slots`` ARE this server's slot for that card (the same
counter, under this server's limit), so a panel adding up usage must count
it once. The list's ``local`` field describes this server the same way
(``{"id": "local", "devices": [{"value", "kind", "slots"}]}``; ``null``
without a run service), so a Workers panel can draw this server beside its
workers.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ..core.scheduling import LOCAL_WORKER_ID, slot_for
from ..core.worker_protocol import canonical_device

router = APIRouter(prefix="/api/workers", tags=["workers"])


def _registry(request: Request) -> Any:
    registry = getattr(request.app.state, "worker_registry", None)
    if registry is None:
        raise HTTPException(503, "worker registry is not available")
    return registry


def _service(request: Request) -> Any:
    return getattr(request.app.state, "run_service", None)


def _describe(worker: Any, service: Any) -> dict[str, Any]:
    """``worker.describe()`` with each device's slot usage filled in."""
    entry = worker.describe()
    shared = getattr(worker, "shared", None) or {}
    for device in entry["devices"]:
        try:
            value = canonical_device(device["value"])
        except ValueError:
            value = None
        device["shared_with"] = (LOCAL_WORKER_ID if value in shared
                                 else None)
        device["slots"] = (None if service is None or value is None
                           else service.slot_usage(slot_for(worker, value)))
    return entry


def _describe_local(service: Any) -> dict[str, Any] | None:
    if service is None:
        return None
    return {"id": LOCAL_WORKER_ID, "devices": [{
        "value": value, "kind": value.split(":")[0],
        "slots": service.slot_usage((LOCAL_WORKER_ID, value)),
    } for value in service.local_devices()]}


@router.get("")
async def list_workers(request: Request) -> dict[str, Any]:
    registry = _registry(request)
    service = _service(request)
    return {"workers": [_describe(w, service) for w in registry.workers()],
            "local": _describe_local(service)}


def _set_draining(request: Request, worker_id: str,
                  draining: bool) -> dict[str, Any]:
    worker = _registry(request).drain(worker_id, draining)
    if worker is None:
        raise HTTPException(404, f"no worker named {worker_id!r}")
    return _describe(worker, _service(request))


@router.post("/{worker_id}/drain")
async def drain_worker(request: Request, worker_id: str) -> dict[str, Any]:
    return _set_draining(request, worker_id, True)


@router.post("/{worker_id}/undrain")
async def undrain_worker(request: Request, worker_id: str) -> dict[str, Any]:
    return _set_draining(request, worker_id, False)
