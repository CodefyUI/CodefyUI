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
    POST   /api/workers/agent-command        the command for a token at an
                                             address (token; #786)

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
without a run service; its ``mounts`` is this server's own label and path,
#782), so a Workers panel can draw this server beside its
workers.

Handshake validation (#781): each worker entry carries ``node_catalog``
(``{"status": "match" | "mismatch" | "unknown", "hash", "missing",
"extra"}``: its node catalog against this server's, the type lists sorted)
and ``degraded_reasons`` (``node_catalog``, ``heartbeat``; ``status`` reads
``degraded`` when either applies and the worker is not draining), and
``fingerprint_digest`` is its environment's. The list's ``rejections`` field
is each worker name's last refused handshake, newest first (``[{"name",
"reason", "at"}]``, ``at`` in epoch seconds; in memory, cleared when that
name connects), so the panel can say why a worker never appeared.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..config import settings
from ..core import tls_fingerprint as tls_fp
from ..core import worker_protocol as wp
from ..core import worker_store as ws_store
from ..core.api_keys import require_session_token
from ..core.auth import allowed_hosts, bound_to_loopback, is_loopback_peer
from ..core.mounts import (
    control_plane_label,
    control_plane_mounts,
    control_plane_root,
    lends_mount,
)
from ..core.scheduling import LOCAL_WORKER_ID, slot_for
from ..core.worker_protocol import canonical_device
from ..core.workers import CREDENTIAL_SESSION

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
    *per_minute* a minute. At most *max_keys* keys are kept, least recently
    used first out, so a flood of addresses cannot grow the table without
    bound (an evicted key starts again with a full bucket)."""

    def __init__(self, burst: int = ENROLL_BURST,
                 per_minute: float = ENROLL_PER_MINUTE,
                 clock: Any = time.monotonic, max_keys: int = 4096) -> None:
        self.burst = burst
        self.rate = per_minute / 60.0
        self._clock = clock
        self._max_keys = max_keys
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()
        self._lock = threading.Lock()

    def take(self, key: str) -> float:
        """Spend one token for *key*. 0.0 when allowed, else the seconds to
        wait before the next one."""
        with self._lock:
            now = self._clock()
            tokens, at = self._buckets.get(key, (float(self.burst), now))
            tokens = min(float(self.burst), tokens + (now - at) * self.rate)
            allowed = tokens >= 1.0
            self._buckets[key] = (tokens - 1.0 if allowed else tokens, now)
            self._buckets.move_to_end(key)
            while len(self._buckets) > self._max_keys:
                self._buckets.popitem(last=False)
            return 0.0 if allowed else (1.0 - tokens) / self.rate


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
    """``worker.describe()`` with each device's slot usage filled in.

    ``mounts`` are the labels the worker declared (``--mount``, #782); a
    same-host worker that declared no directory for this server's label
    also shows this server's own path for it, which is what its runs use.
    """
    entry = worker.describe()
    label = control_plane_label()
    if lends_mount(worker, label):
        entry["mounts"][label] = str(control_plane_root())
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
    return {"id": LOCAL_WORKER_ID, "mounts": control_plane_mounts(),
            "devices": [{
        "value": value, "kind": value.split(":")[0],
        "slots": service.slot_usage((LOCAL_WORKER_ID, value)),
    } for value in service.local_devices()]}


@router.get("")
async def list_workers(request: Request) -> dict[str, Any]:
    registry = _registry(request)
    service = _service(request)
    return {"workers": [_describe(w, service) for w in registry.workers()],
            "local": _describe_local(service),
            "rejections": registry.rejections()}


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
    #: Bind the token to this known worker: a re-key token, which is the
    #: only kind that can enroll a name already taken, and only that one.
    worker: str | None = Field(None, max_length=63)
    #: The address the worker dials, as the dialog shows it; default: the
    #: address this request used (:func:`public_url`).
    control_url: str | None = Field(None, max_length=300)


#: What a control URL in a generated shell command may contain: a scheme,
#: a host (name, IPv4 or bracketed IPv6) with an optional port, and an
#: optional path, with no character a shell would interpret.
_CONTROL_URL = re.compile(
    r"https?://[A-Za-z0-9.\-]+(:[0-9]{1,5})?|https?://\[[0-9A-Fa-f:.]+\]"
    r"(:[0-9]{1,5})?", re.I)
_CONTROL_PATH = re.compile(r"(/[A-Za-z0-9._~\-]+)*/?")


def checked_control_url(url: str) -> str:
    """*url* without a trailing slash, or 422 when it is not an http(s)
    address safe to paste into a shell."""
    text = url.strip()
    match = _CONTROL_URL.match(text)
    if match is None or _CONTROL_PATH.fullmatch(text[match.end():]) is None:
        raise HTTPException(422, "control_url must be http(s)://host[:port]"
                                 "[/path] with no spaces or shell characters")
    return text.rstrip("/")


def public_url(request: Request) -> str:
    """This server as the browser reached it: scheme and ``Host`` (which
    the Host guard has already checked against the whitelist), plus any
    ``root_path`` a proxy mounts it under."""
    host = request.headers.get("host", "")
    root = (request.scope.get("root_path") or "").rstrip("/")
    return f"{request.url.scheme}://{host}{root}"


def agent_command(url: str, token: str, name: str | None = None,
                  fingerprint: str | None = None) -> str:
    """The one builder of the agent command (the dialog shows what this
    returns). *fingerprint* is pinned over ``https`` only: plain HTTP has
    no certificate."""
    command = f"cdui agent start --control {url} --token {token}"
    if name is not None:
        command += f" --name {name}"
    if fingerprint is not None and url.lower().startswith("https:"):
        command += f" --ca-fingerprint {fingerprint}"
    return command


#: ``(path, mtime_ns)`` -> the fingerprint read from it (or None). One
#: entry: a changed file (or setting) is read again; a warning is logged
#: once per version of the file.
_fingerprint_cache: dict[tuple[str, int | None], str | None] = {}


def _read_tls_fingerprint() -> str | None:
    path = settings.TLS_CERT_FILE.strip()
    if not path:
        return None
    try:
        mtime: int | None = os.stat(path).st_mtime_ns
    except OSError as exc:
        mtime, error = None, exc
    else:
        error = None
    key = (path, mtime)
    if key in _fingerprint_cache:
        return _fingerprint_cache[key]
    value = None
    if error is None:
        try:
            with open(path, "rb") as handle:
                certificates = tls_fp.pem_certificates(handle.read())
            value = tls_fp.spki_fingerprint(certificates[0])
        except (OSError, ValueError, IndexError) as exc:
            error = exc
    if error is not None:
        logger.warning("CODEFYUI_TLS_CERT_FILE %s gives no certificate "
                       "fingerprint: %s", path, error or "no certificate")
    _fingerprint_cache.clear()
    _fingerprint_cache[key] = value
    return value


async def tls_fingerprint() -> str | None:
    """The SPKI fingerprint of ``CODEFYUI_TLS_CERT_FILE``'s first (leaf)
    certificate (#786), or None when it is unset or cannot be read. Read
    in a thread, and again only when the file changes."""
    return await asyncio.to_thread(_read_tls_fingerprint)


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
    in this response only; the server keeps its hash. With ``worker`` it
    is a re-key token for that known worker, and the command names it."""
    store = _store(request)
    if body.worker is not None:
        registry = _registry(request)
        if registry.get(body.worker) is None:
            raise HTTPException(404, f"no worker named {body.worker!r}")
    url = (checked_control_url(body.control_url) if body.control_url
           else public_url(request))
    entry = await store.create_enrollment(
        ttl_s=body.ttl_s, uses=body.uses, label=body.label,
        worker=body.worker)
    logger.info("enrollment token %s created (%d use(s), expires %s%s)",
                entry["id"], entry["uses_left"], entry["expires_at"],
                f", re-keys {body.worker}" if body.worker else "")
    return {**entry, "control_url": url,
            "command": agent_command(url, entry["token"], body.worker,
                                     await tls_fingerprint())}


class AgentCommandRequest(BaseModel):
    control_url: str = Field(..., max_length=300)
    token: str = Field(..., max_length=64)
    worker: str | None = Field(None, max_length=63)


@router.post("/agent-command")
async def agent_command_for(body: AgentCommandRequest) -> dict[str, str]:
    """The agent command for an enrollment *token* at another address, for
    the Add worker dialog when its address field changes after a token was
    generated: the same builder as ``POST /enrollments`` (#786). The token
    travels in the body (a URL would be logged) and is not checked or
    stored here; it only appears in the answer."""
    if not ws_store.is_enrollment_token(body.token):
        raise HTTPException(422, "token must be an enrollment token")
    if body.worker is not None and wp.WORKER_NAME_PATTERN.fullmatch(
            body.worker) is None:
        raise HTTPException(422, "invalid worker name")
    url = checked_control_url(body.control_url)
    return {"control_url": url,
            "command": agent_command(url, body.token.strip(), body.worker,
                                     await tls_fingerprint())}


@router.get("/enrollments", dependencies=[Depends(require_session_token)])
async def list_enrollments(request: Request) -> dict[str, Any]:
    """The enrollment tokens (never the secrets), newest first, with the
    facts the Add worker dialog needs to offer a URL another machine can
    reach, and to verify the server's TLS certificate (#786):
    ``tls_fingerprint`` when ``CODEFYUI_TLS_CERT_FILE`` names a readable
    certificate, else null, and the ``openssl`` command that computes it on
    the proxy host."""
    return {
        "enrollments": await _store(request).list_enrollments(),
        "control_url": public_url(request),
        "other_machine_urls": other_machine_urls(request.url.scheme),
        "listening_on_loopback": bound_to_loopback(),
        "tls_fingerprint": await tls_fingerprint(),
        "fingerprint_command": tls_fp.OPENSSL_COMMAND,
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
    registry = getattr(request.app.state, "worker_registry", None)
    live = None if registry is None else registry.get(name)
    # A same-host agent on the session token holds the name without a row.
    held = (live is not None and live.online
            and live.credential == CREDENTIAL_SESSION)
    try:
        worker_token, rekeyed = await store.enroll(
            body.token, name=name, platform=dict(body.platform),
            mounts=dict(body.mounts), devices=devices, held_elsewhere=held)
    except ws_store.EnrollmentRefused as exc:
        logger.warning("enrollment of %s from %s refused: %s", name, client,
                       exc)
        raise HTTPException(exc.status, str(exc)) from exc
    if registry is not None:
        if rekeyed or held:
            # The connection that used the old credential (or the session
            # token, for a name now enrolled) must not outlive it.
            registry.kick(name, "the worker was re-keyed; its old "
                                "credential no longer works")
        registry.load_known([ws_store.KnownWorker(
            name=name, platform=dict(body.platform), mounts=dict(body.mounts),
            devices=devices, created_at="", last_seen_at=None)])
        known = registry.get(name)
        if known is not None:
            known.enrolled = True
    logger.info("worker %s enrolled from %s", name, client)
    if request.url.scheme != "https" and not is_loopback_peer(client):
        logger.warning(
            "worker %s enrolled over plain http:// from %s: its new worker "
            "token crossed the network unencrypted; put the server behind "
            "TLS on networks you do not trust", name, client)
    return {"worker_id": name, "worker_token": worker_token}


@router.delete("/{worker_id}")
async def remove_worker(request: Request, worker_id: str) -> dict[str, Any]:
    """Revoke the worker's credential, disconnect it, and forget it.

    Its running runs end ``interrupted`` (the socket closes), and the runs
    still queued for it fail at once: nothing would ever place them.
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
    service = _service(request)
    failed = [] if service is None else await service.fail_queued_for_worker(
        worker_id, f"worker {worker_id} was removed")
    return {"id": worker_id, "revoked": revoked, "failed_runs": failed}
