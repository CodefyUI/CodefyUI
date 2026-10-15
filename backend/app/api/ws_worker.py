"""``WS /ws/worker``: where worker agents connect (#772).

The server end of ``core.worker_protocol``; the agent end is ``app.agent``
(#771). A worker always dials in; this process never connects out.

Handshake checks, before ``accept``:

- Host header against the whitelist, as ``/ws/execution`` does, or the
  remote end of an SSH worker's reverse tunnel (``127.0.0.1:<port>``,
  ``auth.tunnel_host_allows``, #784);
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
    logged (the token crosses the network in the clear), and the worker
    is listed with ``plain_http: true`` (#786, :func:`plain_from_another_machine`).

A refused handshake closes before ``accept``, which a client sees as
HTTP 403.

Then: the first frame must be ``hello`` (within :data:`HELLO_TIMEOUT_S`).
What it must satisfy is decided in ``core.worker_handshake`` (#781): the
same protocol version and the same CodefyUI version as this server (a
mismatch is refused naming both), a well-formed unreserved name, and a
name this credential may use. A refusal is a ``reject`` and a close, and
is remembered per name for the Workers panel
(``WorkerRegistry.record_rejection``). Otherwise the worker is registered
(replacing an older connection under the same name, but only one that
authenticated the same way: a session-token agent cannot take over an
enrolled worker's name, nor an enrolled worker a connected same-host
agent's), its node catalog is compared with this server's (a difference
makes it ``degraded``), its fingerprint is recorded (on its ``workers``
row and in ``exec_run_fingerprints``), and it is answered with
``welcome`` carrying this server's ``capabilities``; every later
frame is routed: ``heartbeat`` and ``run_alive`` to the registry, per-run
frames to their run's inbox (pausing the read while one run's inbox is
far behind, ``WorkerRegistry.backpressure``). Closing the socket marks the
worker disconnected; it never cancels a run by itself (the #120/#121 rule
for ``/ws/execution``). The worker's runs wait ``disconnected`` for it to
reconnect and replay what was not acknowledged, and end ``interrupted``
only after the resume grace (#783, ``core.workers``).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..core import worker_handshake as handshake
from ..core import worker_protocol as wp
from ..core.machine_identity import machine_identity, same_machine
from ..core.version import get_version
from ..core.workers import CREDENTIAL_SESSION, CREDENTIAL_WORKER
from ..core.auth import (
    TOKEN_QUERY_PARAM,
    constant_time_equals,
    host_is_allowed,
    is_loopback_peer,
    session_token,
    tunnel_host_allows,
    tunnel_port,
)

logger = logging.getLogger(__name__)

router = APIRouter()

#: Seconds a fresh connection has to say ``hello``.
HELLO_TIMEOUT_S = 10.0


def _peer_is_loopback(ws: WebSocket) -> bool:
    return is_loopback_peer(ws.client.host if ws.client is not None else None)


def plain_from_another_machine(ws: WebSocket) -> bool:
    """The connection is unencrypted and comes from another machine (#786).

    The scheme is the one this process was reached with: ``wss`` only when
    uvicorn says so (it rewrites the scheme from ``X-Forwarded-Proto`` for
    the proxies ``--forwarded-allow-ips`` trusts; this code never reads the
    header). A loopback peer is exempt: a same-host agent, a TLS proxy on
    this machine that forwards no headers, or the local end of an SSH
    tunnel (#784), whose hop across the network is encrypted by ssh.
    """
    if tunnel_port(ws.headers.get("host")) is not None:
        # Through an SSH tunnel to a server bound to one LAN address: the
        # peer is that address, but the network hop is ssh's.
        return False
    return not _peer_is_loopback(ws) and ws.url.scheme not in ("wss",
                                                               "https")


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
    if plain_from_another_machine(ws):
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
    host = ws.headers.get("host", "")
    if not host_is_allowed(host) and not tunnel_host_allows(
            host, wp.WORKER_WS_PATH):
        await ws.close(code=4003, reason="Host not allowed")
        return
    if ws.headers.get("origin"):
        await ws.close(code=4003, reason="Browsers may not connect here")
        return
    auth = await authorise(ws)
    if auth is None:
        _note_tunnel_refusal(ws, host)
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

    peer = ws.client.host if ws.client is not None else None

    async def refuse(name: str | None, reason: str) -> None:
        logger.warning("worker connection refused: %s", reason)
        if name is not None:
            registry.record_rejection(name, reason, peer=peer)
        await send(wp.Reject(reason=reason))
        await close()

    try:
        raw = await asyncio.wait_for(ws.receive_text(), HELLO_TIMEOUT_S)
    except (asyncio.TimeoutError, WebSocketDisconnect, RuntimeError) as exc:
        logger.warning("a worker connection said no hello: %s", exc)
        await close()
        return
    # The version pair first, off the raw JSON: an agent of another release
    # may send a hello this build cannot validate, and must still hear why.
    early = handshake.raw_refusal(raw, control_version=get_version())
    if early is not None:
        await refuse(*early)
        return
    try:
        hello = wp.decode_agent_frame(raw)
    except wp.ProtocolError as exc:
        logger.warning("a worker connection said no valid hello: %s", exc)
        await close()
        return
    refusal = (handshake.refusal(hello, control_version=get_version())
               or handshake.identity_refusal(
                   hello, credential=auth.credential, enrolled_as=enrolled_as,
                   registry=registry))
    if refusal is not None:
        await refuse(hello.name if isinstance(hello, wp.Hello) else None,
                     refusal)
        return
    if not hello.node_catalog_hash:
        logger.warning("worker %s reported no node catalog (its facts helper "
                       "failed; see the agent's log): its node types are not "
                       "checked", hello.name)

    # ``welcome`` goes out BEFORE the worker is registered (#783): from
    # ``connect`` on, executors send through this socket (a stop that waited
    # for the worker, an ack), and nothing may precede ``welcome``. The
    # worker's id is its name (``WorkerRegistry.connect``).
    await send(wp.Welcome(worker_id=hello.name,
                          heartbeat_s=registry.heartbeat_s,
                          grace_s=registry.grace_s,
                          resume_grace_s=registry.resume_grace_s,
                          capabilities=list(wp.CAPABILITIES)))
    # Same machine needs both (#773, #780): the machine id alone is
    # self-reported, so a remote agent could copy ours; the loopback peer
    # alone is not enough either, since an ``ssh -R`` tunnel or a local
    # proxy makes a remote agent look like loopback.
    worker, replaced = registry.connect(
        hello, send=send, close=close,
        same_host=_peer_is_loopback(ws)
        and same_machine(hello.machine_id, machine_identity()),
        credential=auth.credential, enrolled=enrolled_as is not None,
        plain_http=plain_from_another_machine(ws))
    store = getattr(ws.app.state, "worker_store", None)
    if enrolled_as is not None and store is not None:
        await _touch(store, hello, fingerprint_digest=worker.fingerprint)
    launcher = getattr(ws.app.state, "ssh_launcher", None)
    if launcher is not None:
        launcher.worker_connected(worker.name)
    service = getattr(ws.app.state, "run_service", None)
    if service is not None:
        service.check_worker_overrides(worker)
        await _record_fingerprint(service, worker)
    if replaced is not None:
        await replaced.close()
    try:
        while True:
            raw = await ws.receive_text()
            # Any frame says the worker is alive (#783).
            registry.touch(worker)
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
                await registry.backpressure(worker, frame)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        registry.disconnect(worker)
        if enrolled_as is not None and store is not None:
            await _touch(store, None, enrolled_as)


async def _record_fingerprint(service: Any, worker: Any) -> None:
    """Store the worker's fingerprint document, so its digest resolves
    (``GET /api/system/fingerprints/{digest}``) before any run records it.
    A failed write is logged; it never costs the connection."""
    if worker.fingerprint_document is None or worker.fingerprint is None:
        return
    try:
        await service.store.record_fingerprint(worker.fingerprint_document,
                                               worker.fingerprint)
    except Exception:  # noqa: BLE001
        logger.warning("could not record worker %s's fingerprint",
                       worker.name, exc_info=True)


def _note_tunnel_refusal(ws: WebSocket, host: str) -> None:
    """A credential refused through an SSH worker's tunnel (#784): tell
    the launcher, which restarts that agent with ``cdui agent forget``.
    Only this refusal (an unknown or revoked token, 4401) counts; a Host
    or registry refusal never does."""
    port = tunnel_port(host)
    launcher = getattr(ws.app.state, "ssh_launcher", None)
    if port is not None and launcher is not None:
        launcher.note_refused(port)


async def _touch(store: Any, hello: wp.Hello | None,
                 name: str | None = None, *,
                 fingerprint_digest: str | None = None) -> None:
    # On ``hello`` the digest is always written: None clears a stale one.
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
                machine_id=hello.machine_id,
                fingerprint_digest=fingerprint_digest)
    except Exception:  # noqa: BLE001
        logger.warning("could not record worker %s as seen",
                       name or (hello.name if hello else "?"), exc_info=True)

