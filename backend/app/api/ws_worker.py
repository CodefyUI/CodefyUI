"""``WS /ws/worker``: where worker agents connect (#772).

The server end of ``core.worker_protocol``; the agent end is ``app.agent``
(#771). A worker always dials in; this process never connects out.

Handshake checks, before ``accept``:

- Host header against the whitelist, as ``/ws/execution`` does; a
  connection through an SSH worker's own listener (#784,
  ``core.tunnel_listener``) skips that check, since whoever reaches the
  forward's far end chooses the ``Host``, and the session token is never
  accepted there;
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
HTTP 403; a refused credential is answered with HTTP 401 instead
(:func:`refuse_token`), so the agent can tell the two apart.

Then: the first frame must be ``hello`` (within :data:`HELLO_TIMEOUT_S`).
What it must satisfy is decided in ``core.worker_handshake`` (#781): the
same protocol version and the same CodefyUI version as this server (a
mismatch is refused naming both), a well-formed unreserved name, and a
name this credential may use. A refusal is a ``reject`` and a close, and
is remembered per name for the Workers panel
(``WorkerRegistry.record_rejection``). A one-shot agent (#790) is also
refused when another agent holds its name while connected or with a run in
flight (``worker_handshake.one_shot_refusal``), and, started with
``--run-id``, when that run does not wait for it
(``RunService.one_shot_run_refusal``). A one-shot agent started with
``--instance`` (#845) is registered as ``<name>.<instance>`` under its
credential's name, and every check above applies to that id; a run that
instance took before this server restarted can be taken back only by the
agent process that took it (``worker_handshake.same_agent``), as for any
one-shot run. An instance never writes its credential's ``workers`` row:
the credential's own agent may be connected, and what that row records
(its devices, its one-shot flag) is that agent's.
Otherwise the worker is registered
(replacing an older connection under the same name, but only one that
authenticated the same way: a session-token agent cannot take over an
enrolled worker's name, nor an enrolled worker a connected same-host
agent's), its node catalog is compared with this server's (a difference
makes it ``degraded``), its fingerprint is recorded (on its ``workers``
row and in ``exec_run_fingerprints``), and it is answered with
``welcome`` carrying this server's ``capabilities``; every later
frame is routed: ``heartbeat`` and ``run_alive`` to the registry,
``output_answer`` (a read of a run's recorded outputs, #789) to the reads in
flight (``WorkerRegistry.outputs``), per-run frames to their run's inbox (pausing the read while one run's inbox is
far behind, ``WorkerRegistry.backpressure``). Closing the socket marks the
worker disconnected; it never cancels a run by itself (the #120/#121 rule
for ``/ws/execution``). The worker's runs wait ``disconnected`` for it to
reconnect and replay what was not acknowledged, and end ``interrupted``
only after the resume grace (#783, ``core.workers``).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
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
)
from ..core.tunnel_listener import tunnel_of
from ..core.worker_store import hash_secret

logger = logging.getLogger(__name__)

router = APIRouter()

#: Seconds a fresh connection has to say ``hello``.
HELLO_TIMEOUT_S = 10.0

#: Seconds a same-host worker's handshake waits for this server's device
#: list (#842) before it is registered anyway: well inside the agent's
#: 10 s wait for ``welcome`` (``agent.connection.WELCOME_TIMEOUT_S``).
SAME_HOST_PROBE_WAIT_S = 4.0


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
    if tunnel_of(ws.scope) is not None:
        # Through an SSH worker's listener: the network hop is ssh's.
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


async def refuse_token(ws: WebSocket, *, revoked: bool = False) -> None:
    """Refuse a missing or invalid credential so the agent can tell it from
    every other refusal: HTTP 401 where the server can answer a handshake
    with a response of its own (uvicorn's ``websocket.http.response``),
    else a close with 4401 (seen as HTTP 403, like the other refusals).
    The agent reads only a 401 as "my token was refused", which is what
    makes an SSH worker forget its credential (#784). A *revoked* credential
    (one this server revoked) says so in ``X-CodefyUI-Refusal``, on which
    the agent stops (#840)."""
    if "websocket.http.response" in ws.scope.get("extensions", {}):
        from fastapi.responses import JSONResponse

        if revoked:
            await ws.send_denial_response(JSONResponse(
                status_code=401,
                content={"detail": "This worker's credential was revoked"},
                headers={wp.REFUSAL_HEADER: wp.REFUSAL_REVOKED}))
            return
        await ws.send_denial_response(JSONResponse(
            status_code=401,
            content={"detail": "Missing or invalid token"}))
        return
    await ws.close(code=4401, reason="Missing or invalid token")


#: Why a revoked credential was refused, by ``WorkerStore.revoked``'s why.
REVOKED_REASONS = {
    "removed": "its credential was revoked: the worker was removed in the "
               "Workers panel. Stop that agent, or enroll it again with a "
               "new token (cdui agent forget <address> first)",
    "rekeyed": "its credential was replaced by a re-key; only the agent "
               "that enrolled with the re-key token can connect",
}


#: A revoked credential presented again within this many seconds is not
#: logged or listed again (#840): an agent that keeps dialling would
#: otherwise log a warning every few seconds.
REVOKED_NOTE_INTERVAL_S = 3600.0
MAX_REVOKED_NOTED = 256
#: ``(name, credential hash) -> when (monotonic)`` it was last noted.
_revoked_noted: OrderedDict[tuple[str, str], float] = OrderedDict()


def _note_revoked(ws: WebSocket, provided: str | None) -> bool:
    """Whether the refused *provided* is a credential this server revoked
    (a Remove, a re-key). Such a refusal is logged and listed under its
    worker's name in the Workers panel, so an agent left running after its
    worker was removed does not go unseen (#840); both at most once per
    worker and credential per :data:`REVOKED_NOTE_INTERVAL_S`. Any other
    refused token names no worker and is not listed."""
    store = getattr(ws.app.state, "worker_store", None)
    revoked = getattr(store, "revoked", None)
    found = None if revoked is None else revoked(provided)
    if found is None:
        return False
    name, why = found
    key = (name, hash_secret(provided or ""))
    now = time.monotonic()
    last = _revoked_noted.get(key)
    if last is not None and now - last < REVOKED_NOTE_INTERVAL_S:
        return True
    _revoked_noted.pop(key, None)
    _revoked_noted[key] = now
    while len(_revoked_noted) > MAX_REVOKED_NOTED:
        _revoked_noted.popitem(last=False)
    reason = REVOKED_REASONS.get(why, "its credential was revoked")
    logger.warning("worker connection refused: %s presented a revoked "
                   "credential (%s)", name, why)
    registry = getattr(ws.app.state, "worker_registry", None)
    if registry is not None:
        registry.record_rejection(
            name, reason,
            peer=ws.client.host if ws.client is not None else None)
    return True


def presented_token(ws: WebSocket) -> str | None:
    """The token the handshake carries (header first, then the query)."""
    return (ws.headers.get("x-codefyui-token")
            or ws.query_params.get(TOKEN_QUERY_PARAM))


async def authorise(ws: WebSocket,
                    provided: str | None) -> Authorised | None:
    # Through an SSH worker's listener the peer is ssh on this machine, but
    # the client is anyone who can reach the forward's far end: never the
    # session token there (#784).
    loopback = _peer_is_loopback(ws) and tunnel_of(ws.scope) is None
    for authoriser in AUTHORISERS:
        result = await authoriser(ws, provided, loopback)
        if result is not None:
            return result
    return None


@router.websocket(wp.WORKER_WS_PATH)
async def websocket_worker(ws: WebSocket) -> None:
    host = ws.headers.get("host", "")
    if tunnel_of(ws.scope) is None and not host_is_allowed(host):
        await ws.close(code=4003, reason="Host not allowed")
        return
    if ws.headers.get("origin"):
        await ws.close(code=4003, reason="Browsers may not connect here")
        return
    provided = presented_token(ws)
    auth = await authorise(ws, provided)
    if auth is None:
        # Not reported to the SSH launcher, even through a tunnel: this
        # side cannot tell the agent from another client of the listener.
        # The launcher acts on the agent's own report instead
        # (``ssh_launch.TOKEN_REFUSED``).
        await refuse_token(ws, revoked=_note_revoked(ws, provided))
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
    refusal = handshake.refusal(hello, control_version=get_version())
    if refusal is None and hello.instance is not None:
        # One of several one-shot agents on this credential (#845): it is
        # registered, and from here on known, as ``<name>.<instance>``.
        hello = hello.model_copy(update={"name": wp.instance_worker_id(
            hello.name, hello.instance)})
    refusal = (refusal
               or handshake.identity_refusal(
                   hello, credential=auth.credential, enrolled_as=enrolled_as,
                   registry=registry))
    if refusal is None:
        refusal = await _one_shot_run_refusal(ws, registry, hello)
    if refusal is not None:
        await refuse(hello.name if isinstance(hello, wp.Hello) else None,
                     refusal)
        return
    if not hello.node_catalog_hash:
        logger.warning("worker %s reported no node catalog (its facts helper "
                       "failed; see the agent's log): its node types are not "
                       "checked", hello.name)

    # Same machine needs both (#773, #780): the machine id alone is
    # self-reported, so a remote agent could copy ours; the loopback peer
    # alone is not enough either, since an ``ssh -R`` tunnel or a local
    # proxy makes a remote agent look like loopback.
    same_host = (_peer_is_loopback(ws)
                 and same_machine(hello.machine_id, machine_identity()))
    probe = getattr(registry, "probe_local_devices", None)
    if same_host and probe is not None:
        # Its cards may be this server's: list ours (once, in a thread) so
        # they are matched at ``connect`` (#773, #842). Bounded well inside
        # the agent's wait for ``welcome``; a slower probe goes on in the
        # background and the index is redone when it lands. Here, not
        # between ``welcome`` and ``connect``: an agent that has its
        # ``welcome`` is registered.
        try:
            await asyncio.wait_for(asyncio.shield(probe()),
                                   SAME_HOST_PROBE_WAIT_S)
        except asyncio.TimeoutError:
            logger.warning("listing this server's devices takes longer than "
                           "%.0f s; worker %s is registered meanwhile and its "
                           "cards are matched when the list lands",
                           SAME_HOST_PROBE_WAIT_S, hello.name)

    # ``welcome`` goes out BEFORE the worker is registered (#783): from
    # ``connect`` on, executors send through this socket (a stop that waited
    # for the worker, an ack), and nothing may precede ``welcome``. The
    # worker's id is its name (``WorkerRegistry.connect``).
    await send(wp.Welcome(worker_id=hello.name,
                          heartbeat_s=registry.heartbeat_s,
                          grace_s=registry.grace_s,
                          resume_grace_s=registry.resume_grace_s,
                          capabilities=list(wp.CAPABILITIES)))
    # Checked again with no await before ``connect`` (#790): the check above
    # awaited (a run lookup, ``welcome``), and two agents taking one
    # one-shot name at once would both have passed it, the second then
    # replacing the first and moving its run.
    holder = registry.holder(hello.name)
    late = (handshake.one_shot_refusal(hello, holder)
            if holder is not None else None)
    if late is not None:
        await refuse(hello.name, late)
        return
    worker, replaced = registry.connect(
        hello, send=send, close=close, same_host=same_host,
        credential=auth.credential,
        # An instance (#845) is not the enrolled worker itself: its
        # credential is revoked by removing the worker it belongs to.
        enrolled=enrolled_as is not None and hello.instance is None,
        plain_http=plain_from_another_machine(ws))
    store = getattr(ws.app.state, "worker_store", None)
    # The credential's row is its own agent's (#845): an instance, which
    # registered under ``<name>.<instance>``, never writes it.
    record = (store if enrolled_as is not None and hello.instance is None
              else None)
    if record is not None:
        await _touch(record, hello, fingerprint_digest=worker.fingerprint)
    launcher = getattr(ws.app.state, "ssh_launcher", None)
    if launcher is not None:
        launcher.worker_connected(worker.name)
    service = getattr(ws.app.state, "run_service", None)
    if service is not None:
        service.check_worker_overrides(worker)
        await _record_fingerprint(service, worker)
    if replaced is not None:
        await replaced.close()
    # Recorded outputs this server stopped retaining while it was away
    # (#789): it deletes them now.
    registry.outputs.flush_drops(worker.name)
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
            elif isinstance(frame, wp.OutputAnswer):
                # A read of a run's recorded outputs (#789): no seq.
                registry.outputs.deliver(worker.name, frame)
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
        if record is not None:
            await _touch(record, None, enrolled_as)


async def _one_shot_run_refusal(ws: WebSocket, registry: Any,
                                hello: wp.Hello) -> str | None:
    """A one-shot agent started with ``--run-id`` (#790) is refused unless
    that run waits for it (or already runs on it): it would otherwise wait
    for a run that is never placed there. Checked on the agent's first
    connection only; its reconnects (the same ``instance_id``) carry on
    with the run it took."""
    if not hello.one_shot or hello.one_shot_run_id is None \
            or hello.one_shot_taken is not None:
        # An agent that already took its run is carrying on with it (a
        # reconnect, perhaps after this server restarted), not asking.
        return None
    existing = registry.holder(hello.name)
    if existing is not None and handshake.same_agent(hello, existing):
        return None
    service = getattr(ws.app.state, "run_service", None)
    if service is None:
        return None
    return await service.one_shot_run_refusal(hello.one_shot_run_id,
                                              hello.name)


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


async def _touch(store: Any, hello: wp.Hello | None,
                 name: str | None = None, *,
                 fingerprint_digest: str | None = None) -> None:
    # On ``hello`` the digest is always written: None clears a stale one.
    """Record an enrolled worker as seen (with its facts and whether it
    is one-shot, #845, on ``hello``). A failed write is logged; it never
    costs the connection."""
    try:
        if hello is None:
            await store.touch(name)
        else:
            await store.touch(
                hello.name, platform=dict(hello.platform),
                mounts=dict(hello.mounts),
                devices=[d.model_dump() for d in hello.devices],
                machine_id=hello.machine_id,
                fingerprint_digest=fingerprint_digest,
                one_shot=hello.one_shot)
    except Exception:  # noqa: BLE001
        logger.warning("could not record worker %s as seen",
                       name or (hello.name if hello else "?"), exc_info=True)

