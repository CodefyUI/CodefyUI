"""``WS /ws/worker``: where worker agents connect (#772).

The server end of ``core.worker_protocol``; the agent end is ``app.agent``
(#771). A worker always dials in; this process never connects out.

Handshake checks, before ``accept``:

- Host header against the whitelist, as ``/ws/execution`` does;
- no browser: a request with an ``Origin`` header is refused (an agent
  sends none, and no page has a reason to open this socket);
- a credential (``X-CodefyUI-Token`` header, or ``?token=``), accepted by
  one of the :data:`AUTHORISERS`, tried in order:

  - the **session token**, from a loopback peer only
    (:func:`session_token_authoriser`): a same-host agent reads it from the
    file the server writes. A remote machine cannot read that file, and a
    non-loopback peer presenting it is refused anyway;
  - a **worker token** (#780), from any peer
    (:func:`worker_token_authoriser`): the long-lived credential an agent
    got from ``POST /api/workers/enroll``, looked up by hash in the
    ``workers`` table. The credential names the worker: its ``hello`` must
    use that name. Over plain ``ws://`` from another machine a warning is
    logged (the token crosses the network in the clear; TLS pinning is
    #786).

A refused handshake closes before ``accept``, which a client sees as
HTTP 403.

Then: the first frame must be ``hello`` (within :data:`HELLO_TIMEOUT_S`);
a protocol-version mismatch or a reserved or malformed name gets
``reject``. Otherwise the worker is registered (replacing an older
connection under the same name, but only one that authenticated the same
way: a session-token agent cannot take over an enrolled worker's name, nor
an enrolled worker a connected same-host agent's) and answered with ``welcome``; every later
frame is routed: ``heartbeat`` and ``run_alive`` to the registry, per-run
frames to their run's inbox. Closing the socket marks the worker
disconnected; it never cancels a run by itself (the #120/#121 rule for
``/ws/execution``) -- the worker's runs end ``interrupted`` because nothing
can report for them any more.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..core import worker_protocol as wp
from ..core.machine_identity import machine_identity, same_machine
from ..core.workers import CREDENTIAL_SESSION, CREDENTIAL_WORKER
from ..core.auth import (
    TOKEN_QUERY_PARAM,
    constant_time_equals,
    host_is_allowed,
    is_loopback_peer,
    session_token,
)

logger = logging.getLogger(__name__)

router = APIRouter()

#: Seconds a fresh connection has to say ``hello``.
HELLO_TIMEOUT_S = 10.0


def _peer_is_loopback(ws: WebSocket) -> bool:
    return is_loopback_peer(ws.client.host if ws.client is not None else None)


@dataclass(frozen=True)
class Authorised:
    """Who a handshake authenticated as."""

    #: ``workers.CREDENTIAL_SESSION`` or ``CREDENTIAL_WORKER``.
    credential: str
    #: The worker the credential belongs to, or None when it names none
    #: (the session token lets a same-host agent pick its name).
    worker: str | None = None


#: ``(ws, presented token, peer is loopback) -> Authorised | None``.
Authoriser = Callable[[WebSocket, str | None, bool],
                      Awaitable["Authorised | None"]]


async def session_token_authoriser(ws: WebSocket, provided: str | None,
                                   loopback: bool) -> Authorised | None:
    """The session token, from this machine only."""
    if loopback and constant_time_equals(provided, session_token()):
        return Authorised(CREDENTIAL_SESSION)
    return None


async def worker_token_authoriser(ws: WebSocket, provided: str | None,
                                  loopback: bool) -> Authorised | None:
    """An enrolled worker's token, from anywhere (#780)."""
    store = getattr(ws.app.state, "worker_store", None)
    if store is None or not provided:
        return None
    name = await store.authenticate(provided)
    if name is None:
        return None
    if not loopback and ws.url.scheme not in ("wss", "https"):
        logger.warning(
            "worker %s authenticated over plain ws:// from %s: its token "
            "crossed the network unencrypted; put the server behind TLS "
            "on networks you do not trust", name,
            ws.client.host if ws.client is not None else "?")
    return Authorised(CREDENTIAL_WORKER, name)


#: Tried in order; the first that accepts the handshake wins.
AUTHORISERS: list[Authoriser] = [session_token_authoriser,
                                 worker_token_authoriser]


async def authorise(ws: WebSocket) -> Authorised | None:
    provided = (ws.headers.get("x-codefyui-token")
                or ws.query_params.get(TOKEN_QUERY_PARAM))
    loopback = _peer_is_loopback(ws)
    for authoriser in AUTHORISERS:
        result = await authoriser(ws, provided, loopback)
        if result is not None:
            return result
    return None


@router.websocket(wp.WORKER_WS_PATH)
async def websocket_worker(ws: WebSocket) -> None:
    if not host_is_allowed(ws.headers.get("host", "")):
        await ws.close(code=4003, reason="Host not allowed")
        return
    if ws.headers.get("origin"):
        await ws.close(code=4003, reason="Browsers may not connect here")
        return
    auth = await authorise(ws)
    if auth is None:
        await ws.close(code=4401, reason="Missing or invalid token")
        return
    enrolled_as = auth.worker
    registry = getattr(ws.app.state, "worker_registry", None)
    if registry is None:
        await ws.close(code=1013, reason="No worker registry")
        return
    await ws.accept()
    send_lock = asyncio.Lock()

    async def send(frame: Any) -> None:
        async with send_lock:
            await ws.send_text(wp.encode_frame(frame))

    async def close() -> None:
        try:
            await ws.close(code=1000)
        except Exception:  # noqa: BLE001 -- already closed
            pass

    try:
        raw = await asyncio.wait_for(ws.receive_text(), HELLO_TIMEOUT_S)
        hello = wp.decode_agent_frame(raw)
    except (asyncio.TimeoutError, wp.ProtocolError, WebSocketDisconnect,
            RuntimeError) as exc:
        logger.warning("a worker connection said no valid hello: %s", exc)
        await close()
        return
    refusal = _refusal(hello) or _identity_refusal(hello, auth, registry)
    if refusal is not None:
        logger.warning("worker connection refused: %s", refusal)
        await send(wp.Reject(reason=refusal))
        await close()
        return

    # Same machine needs both (#773, #780): the machine id alone is
    # self-reported, so a remote agent could copy ours; the loopback peer
    # alone is not enough either, since an ``ssh -R`` tunnel or a local
    # proxy makes a remote agent look like loopback.
    worker, replaced = registry.connect(
        hello, send=send, close=close,
        same_host=_peer_is_loopback(ws)
        and same_machine(hello.machine_id, machine_identity()),
        credential=auth.credential, enrolled=enrolled_as is not None)
    store = getattr(ws.app.state, "worker_store", None)
    if enrolled_as is not None and store is not None:
        await _touch(store, hello)
    service = getattr(ws.app.state, "run_service", None)
    if service is not None:
        service.check_worker_overrides(worker)
    if replaced is not None:
        await replaced.close()
    await send(wp.Welcome(worker_id=worker.id,
                          heartbeat_s=registry.heartbeat_s,
                          grace_s=registry.grace_s))
    try:
        while True:
            raw = await ws.receive_text()
            try:
                frame = wp.decode_agent_frame(raw)
            except wp.ProtocolError as exc:
                logger.warning("worker %s sent a bad frame: %s",
                               worker.name, exc)
                continue
            if isinstance(frame, wp.Heartbeat):
                registry.heartbeat(worker, frame)
            elif isinstance(frame, wp.RunAlive):
                registry.run_alive(worker, frame)
            elif isinstance(frame, wp.Hello):
                logger.warning("worker %s said hello twice; ignored",
                               worker.name)
            else:
                registry.route(worker, frame)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        registry.disconnect(worker)
        if enrolled_as is not None and store is not None:
            await _touch(store, None, enrolled_as)


async def _touch(store: Any, hello: wp.Hello | None,
                 name: str | None = None) -> None:
    """Record an enrolled worker as seen (with its facts, on ``hello``).
    A failed write is logged; it never costs the connection."""
    try:
        if hello is None:
            await store.touch(name)
        else:
            await store.touch(
                hello.name, platform=dict(hello.platform),
                mounts=dict(hello.mounts),
                devices=[d.model_dump() for d in hello.devices],
                machine_id=hello.machine_id)
    except Exception:  # noqa: BLE001
        logger.warning("could not record worker %s as seen",
                       name or (hello.name if hello else "?"), exc_info=True)


def _identity_refusal(hello: wp.Hello, auth: Authorised,
                      registry: Any) -> str | None:
    """Why *auth* may not register as ``hello.name``, or None.

    A worker token names its worker, so the ``hello`` must use that name.
    And a connection may replace a live one under the same name only if it
    authenticated the same way: the session token cannot take over an
    enrolled worker's name, nor a worker token a live same-host agent's.
    """
    if auth.worker is not None and hello.name != auth.worker:
        return (f"this worker token belongs to worker {auth.worker!r}, not "
                f"{hello.name!r}; start the agent with --name {auth.worker}, "
                f"or enroll {hello.name!r} with a new token")
    existing = registry.get(hello.name)
    if existing is None:
        return None
    if auth.credential == CREDENTIAL_SESSION and existing.enrolled:
        return (f"worker {hello.name!r} is enrolled; it connects with its "
                "worker token, not the session token. Pick another --name")
    if existing.online and existing.credential not in (None,
                                                       auth.credential):
        return (f"a worker named {hello.name!r} is connected with another "
                "kind of credential; stop it first")
    return None


def _refusal(frame: Any) -> str | None:
    """Why a first frame cannot register a worker, or None."""
    if not isinstance(frame, wp.Hello):
        return f"expected hello, got {frame.type}"
    if frame.protocol_version != wp.PROTOCOL_VERSION:
        return (f"protocol version {frame.protocol_version} is not this "
                f"server's {wp.PROTOCOL_VERSION}; update the agent")
    if wp.WORKER_NAME_PATTERN.fullmatch(frame.name) is None:
        return (f"worker name {frame.name!r} must be lowercase letters, "
                "digits and '-', at most 63 characters")
    if frame.name in wp.RESERVED_WORKER_NAMES:
        return f"worker name {frame.name!r} is reserved"
    return None
