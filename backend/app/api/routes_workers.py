"""REST surface for worker agents (#772, enrollment #780).

    GET    /api/workers                      every known worker (open read)
    POST   /api/workers/{id}/drain           no new runs on it (session token)
    POST   /api/workers/{id}/undrain         take new runs again (session token)
    DELETE /api/workers/{id}                 revoke + disconnect + forget (token)
    POST   /api/workers/enrollments          a new enrollment token (token)
    GET    /api/workers/enrollments          the enrollment tokens (token)
    DELETE /api/workers/enrollments/{id}     revoke one (token)
    POST   /api/workers/enroll               enrollment token -> worker token
                                             (no session token; rate-limited)

Auth follows ``routes_runs``: reads are open, the writes are covered by
``auth_guard`` in ``main.py`` (session token). Two exceptions: listing the
enrollment tokens needs the session token too (route-level), and
``/api/workers/enroll`` is exempt from ``auth_guard`` (``main.py``
``_AUTH_EXEMPT_PATHS``) because the agent calling it, on another machine,
has only the enrollment token; it is rate-limited per client address
(:class:`TokenBucket`) instead. No secret is ever logged: an enrollment
token is named by its public half.

The list includes workers that disconnected since the server started
(``status: disconnected``) and enrolled workers that have not connected
since (``offline``, #780). 404 for an unknown worker, 503 when the
registry (or, for enrollment, the store) is not on ``app.state`` (the
lifespan does not run under httpx's ASGITransport).

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

import logging
import re
import threading
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..core import worker_protocol as wp
from ..core import worker_store as ws_store
from ..core.api_keys import require_session_token
from ..core.auth import allowed_hosts, bound_to_loopback
from ..core.scheduling import LOCAL_WORKER_ID, slot_for
from ..core.worker_protocol import canonical_device

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/workers", tags=["workers"])

#: The path ``auth_guard`` lets through without a session token.
ENROLL_PATH = "/api/workers/enroll"
#: Enrollment attempts per client address: a burst of this many ...
ENROLL_BURST = 10
#: ... refilled at this many per minute.
ENROLL_PER_MINUTE = 10
#: Devices an enrollment may report (a sanity cap, not a policy).
MAX_DEVICES = 256


class TokenBucket:
    """A per-key token bucket, in memory: *burst* requests at once, then
    *per_minute* a minute. Keys idle long enough to be full again are
    pruned, so the table cannot grow without bound."""

    def __init__(self, burst: int = ENROLL_BURST,
                 per_minute: float = ENROLL_PER_MINUTE,
                 clock: Any = time.monotonic, max_keys: int = 4096) -> None:
        self.burst = burst
        self.rate = per_minute / 60.0
        self._clock = clock
        self._max_keys = max_keys
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def take(self, key: str) -> float:
        """Spend one token for *key*. 0.0 when allowed, else the seconds to
        wait before the next one."""
        with self._lock:
            now = self._clock()
            tokens, at = self._buckets.get(key, (float(self.burst), now))
            tokens = min(float(self.burst), tokens + (now - at) * self.rate)
            if tokens < 1.0:
                self._buckets[key] = (tokens, now)
                return (1.0 - tokens) / self.rate
            self._buckets[key] = (tokens - 1.0, now)
            if len(self._buckets) > self._max_keys:
                self._prune(now)
            return 0.0

    def _prune(self, now: float) -> None:
        full_after = self.burst / self.rate
        for key, (_, at) in list(self._buckets.items()):
            if now - at >= full_after:
                del self._buckets[key]


enroll_limiter = TokenBucket()


def _registry(request: Request) -> Any:
    registry = getattr(request.app.state, "worker_registry", None)
    if registry is None:
        raise HTTPException(503, "worker registry is not available")
    return registry


def _service(request: Request) -> Any:
    return getattr(request.app.state, "run_service", None)


def _store(request: Request) -> Any:
    store = getattr(request.app.state, "worker_store", None)
    if store is None:
        raise HTTPException(503, "worker enrollment is not available")
    return store


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


# ── enrollment (#780) ─────────────────────────────────────────────────────


class EnrollmentRequest(BaseModel):
    ttl_s: int = Field(ws_store.DEFAULT_TTL_S, ge=ws_store.MIN_TTL_S,
                       le=ws_store.MAX_TTL_S)
    uses: int = Field(ws_store.DEFAULT_USES, ge=1, le=ws_store.MAX_USES)
    label: str | None = Field(None, max_length=100)


def public_url(request: Request) -> str:
    """This server as the browser reached it: scheme and ``Host`` (which
    the Host guard has already checked against the whitelist), plus any
    ``root_path`` a proxy mounts it under."""
    host = request.headers.get("host", "")
    root = (request.scope.get("root_path") or "").rstrip("/")
    return f"{request.url.scheme}://{host}{root}"


def agent_command(url: str, token: str) -> str:
    return f"cdui agent start --control {url} --token {token}"


_LOOPBACK_ENTRY = re.compile(r"(127\.[0-9.]+|localhost):[0-9]+")
_HOST_PORT_ENTRY = re.compile(r"[A-Za-z0-9.-]+:[0-9]+")


def other_machine_urls(scheme: str) -> list[str]:
    """Whitelisted ``host:port`` entries another machine could dial
    (hostname or IPv4, with a port, not loopback), as URLs."""
    return sorted(f"{scheme}://{entry}" for entry in allowed_hosts()
                  if _HOST_PORT_ENTRY.fullmatch(entry)
                  and not _LOOPBACK_ENTRY.fullmatch(entry))


@router.post("/enrollments")
async def create_enrollment(request: Request,
                            body: EnrollmentRequest) -> dict[str, Any]:
    """A new enrollment token and the command that uses it. The token is
    in this response only; the server keeps its hash."""
    entry = await _store(request).create_enrollment(
        ttl_s=body.ttl_s, uses=body.uses, label=body.label)
    url = public_url(request)
    logger.info("enrollment token %s created (%d use(s), expires %s)",
                entry["id"], entry["uses_left"], entry["expires_at"])
    return {**entry, "control_url": url,
            "command": agent_command(url, entry["token"])}


@router.get("/enrollments", dependencies=[Depends(require_session_token)])
async def list_enrollments(request: Request) -> dict[str, Any]:
    """The enrollment tokens (never the secrets), newest first, with the
    facts the Add worker dialog needs to offer a URL another machine can
    reach."""
    return {
        "enrollments": await _store(request).list_enrollments(),
        "control_url": public_url(request),
        "other_machine_urls": other_machine_urls(request.url.scheme),
        "listening_on_loopback": bound_to_loopback(),
    }


@router.delete("/enrollments/{enrollment_id}")
async def delete_enrollment(request: Request,
                            enrollment_id: str) -> dict[str, Any]:
    if not await _store(request).delete_enrollment(enrollment_id):
        raise HTTPException(404, f"no enrollment token {enrollment_id!r}")
    logger.info("enrollment token %s deleted", enrollment_id)
    return {"id": enrollment_id, "deleted": True}


class EnrollRequest(BaseModel):
    token: str = Field(..., max_length=64)
    name: str = Field(..., max_length=63)
    platform: dict[str, str] = Field(default_factory=dict)
    devices: list[wp.DeviceReport] = Field(default_factory=list,
                                           max_length=MAX_DEVICES)
    mounts: dict[str, str] = Field(default_factory=dict)


@router.post("/enroll")
async def enroll(request: Request, body: EnrollRequest) -> Any:
    """Trade an enrollment token for a worker token (no session token)."""
    client = request.client.host if request.client is not None else "?"
    wait = enroll_limiter.take(client)
    if wait > 0:
        logger.warning("enrollment from %s refused: too many attempts", client)
        return JSONResponse(
            status_code=429,
            content={"detail": "too many enrollment attempts; try again "
                               f"in {wait:.0f} s"},
            headers={"Retry-After": str(max(1, round(wait)))})
    name = body.name.strip().lower()
    if wp.WORKER_NAME_PATTERN.fullmatch(name) is None:
        raise HTTPException(400, f"worker name {body.name!r} must be "
                                 "lowercase letters, digits and '-', at most "
                                 "63 characters")
    if name in wp.RESERVED_WORKER_NAMES:
        raise HTTPException(400, f"worker name {name!r} is reserved")
    if not ws_store.is_enrollment_token(body.token):
        logger.warning("enrollment from %s refused: not an enrollment token",
                       client)
        raise HTTPException(401, "that is not an enrollment token (expected "
                                 "the abcdef.0123456789abcdef form the "
                                 "Workers panel generates)")
    store = _store(request)
    devices = [d.model_dump() for d in body.devices]
    try:
        worker_token, rekeyed = await store.enroll(
            body.token, name=name, platform=dict(body.platform),
            mounts=dict(body.mounts), devices=devices)
    except ws_store.EnrollmentRefused as exc:
        logger.warning("enrollment of %s from %s refused: %s", name, client,
                       exc)
        raise HTTPException(401, str(exc)) from exc
    registry = getattr(request.app.state, "worker_registry", None)
    if registry is not None:
        if rekeyed:
            # The connection that used the old credential must not outlive it.
            registry.kick(name, "the worker enrolled again; its old "
                                "credential was revoked")
        registry.load_known([ws_store.KnownWorker(
            name=name, platform=dict(body.platform), mounts=dict(body.mounts),
            devices=devices, created_at="", last_seen_at=None)])
        known = registry.get(name)
        if known is not None:
            known.enrolled = True
    logger.info("worker %s enrolled from %s", name, client)
    if request.url.scheme != "https" and not (
            client.startswith("127.") or client in ("::1", "localhost")):
        logger.warning(
            "worker %s enrolled over plain http:// from %s: its new worker "
            "token crossed the network unencrypted; put the server behind "
            "TLS on networks you do not trust", name, client)
    return {"worker_id": name, "worker_token": worker_token}


@router.delete("/{worker_id}")
async def remove_worker(request: Request, worker_id: str) -> dict[str, Any]:
    """Revoke the worker's credential, disconnect it, and forget it.

    Its running runs end ``interrupted`` (the socket closes); a run still
    queued for it keeps waiting until it is cancelled or a worker by that
    name connects again.
    """
    registry = _registry(request)
    store = getattr(request.app.state, "worker_store", None)
    revoked = False
    if store is not None:
        revoked = await store.delete_worker(worker_id)
    removed = registry.forget(
        worker_id, "the worker was removed and its credential revoked"
        if revoked else "the worker was removed")
    if removed is None and not revoked:
        raise HTTPException(404, f"no worker named {worker_id!r}")
    if revoked:
        logger.info("worker %s: credential revoked", worker_id)
    return {"id": worker_id, "revoked": revoked}
