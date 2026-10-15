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
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/api/workers", tags=["workers"])


def _registry(request: Request) -> Any:
    registry = getattr(request.app.state, "worker_registry", None)
    if registry is None:
        raise HTTPException(503, "worker registry is not available")
    return registry


@router.get("")
async def list_workers(request: Request) -> dict[str, Any]:
    registry = _registry(request)
    return {"workers": [w.describe() for w in registry.workers()]}


def _set_draining(request: Request, worker_id: str,
                  draining: bool) -> dict[str, Any]:
    worker = _registry(request).drain(worker_id, draining)
    if worker is None:
        raise HTTPException(404, f"no worker named {worker_id!r}")
    return worker.describe()


@router.post("/{worker_id}/drain")
async def drain_worker(request: Request, worker_id: str) -> dict[str, Any]:
    return _set_draining(request, worker_id, True)


@router.post("/{worker_id}/undrain")
async def undrain_worker(request: Request, worker_id: str) -> dict[str, Any]:
    return _set_draining(request, worker_id, False)
