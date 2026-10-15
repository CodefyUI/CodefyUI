"""System / environment introspection endpoints (compute devices, etc.)."""
import asyncio
import re

from fastapi import APIRouter, HTTPException, Request

from ..core.device_utils import describe_accelerator
from ..core.env_fingerprint import collect_with_digest
from ..core.resources import probe_devices, with_memory

router = APIRouter(prefix="/api/system", tags=["system"])

_DIGEST = re.compile(r"^[0-9a-f]{64}$")


@router.get("/devices")
async def list_devices() -> dict:
    """Compute devices available for graph execution.

    Returns ``describe_accelerator()`` — the best-available ``default`` plus a
    ``devices`` list with human-friendly labels that distinguish NVIDIA CUDA
    from AMD ROCm and label Apple MPS. Every device dropdown (Settings, the
    graph toolbar, node params) renders this one list.

    Each device entry also carries ``"memory": {"total_bytes",
    "free_bytes", "source"}`` from ``core.resources.probe_devices`` (#767),
    or ``null`` when no probe could say (``free_bytes`` alone is ``null``
    when only torch could answer and CUDA is not initialised). Display
    only, and not cached: the figures are current as of this request.
    ``describe_accelerator()`` and the probe run together in one worker
    thread, because both make blocking driver / NVML / psutil calls that do
    not belong on the event loop.
    """
    def _describe() -> dict:
        return with_memory(describe_accelerator(), probe_devices())

    return await asyncio.to_thread(_describe)


@router.get("/fingerprint")
async def get_fingerprint() -> dict:
    """This server's environment fingerprint and its digest (#344).

    ``{"fingerprint": {...}, "digest": "<sha256 hex>"}``. The document is
    ``core.env_fingerprint.collect_fingerprint`` (versioned by its
    ``schema_version``); the digest is the sha256 of its canonical JSON and
    covers every key. It is the same document a run records when it starts,
    so comparing this digest with a run's ``fingerprint_digest`` answers
    "would a run started now execute in the same environment". No
    hostname, username or local path is included.
    """
    fingerprint, digest = await asyncio.to_thread(collect_with_digest)
    return {"fingerprint": fingerprint, "digest": digest}


@router.get("/fingerprints/{digest}")
async def get_recorded_fingerprint(digest: str, request: Request) -> dict:
    """A fingerprint some run recorded, by its digest. 404 when unknown.

    Resolves a ``fingerprint_digest`` after its runs are gone: the stored
    documents are content-addressed and retention does not prune them.
    """
    if _DIGEST.match(digest) is None:
        raise HTTPException(status_code=404,
                            detail=f"no recorded fingerprint '{digest}'")
    service = getattr(request.app.state, "run_service", None)
    if service is None:
        raise HTTPException(status_code=503,
                            detail="run service not initialised")
    fingerprint = await service.store.get_fingerprint(digest)
    if fingerprint is None:
        raise HTTPException(status_code=404,
                            detail=f"no recorded fingerprint '{digest}'")
    return {"fingerprint": fingerprint, "digest": digest}
