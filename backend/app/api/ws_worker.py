"""``WS /ws/worker``: where worker agents connect (#772).

The server end of ``core.worker_protocol``; the agent end is ``app.agent``
(#771). A worker always dials in; this process never connects out.

Handshake checks, before ``accept``:

- Host header against the whitelist, as ``/ws/execution`` does;
- no browser: a request with an ``Origin`` header is refused (an agent
  sends none, and no page has a reason to open this socket);
- loopback peers only in this wave: enrollment for remote workers is #780;
- the session token (``?token=`` or ``X-CodefyUI-Token``), which a
  same-host agent reads from the file the server writes.

A refused handshake closes before ``accept``, which a client sees as
HTTP 403.

Then: the first frame must be ``hello`` (within :data:`HELLO_TIMEOUT_S`);
a protocol-version mismatch or a reserved or malformed name gets
``reject``. Otherwise the worker is registered (replacing an older
connection under the same name) and answered with ``welcome``; every later
frame is routed: ``heartbeat`` and ``run_alive`` to the registry, per-run
frames to their run's inbox. Closing the socket marks the worker
disconnected; it never cancels a run by itself (the #120/#121 rule for
``/ws/execution``) -- the worker's runs end ``interrupted`` because nothing
can report for them any more.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..core import worker_protocol as wp
from ..core.auth import (
    TOKEN_QUERY_PARAM,
    constant_time_equals,
    host_is_allowed,
    session_token,
)
from ..core.run_service import TARGET_WORKER_PATTERN
from ..core.workers import RESERVED_NAMES

logger = logging.getLogger(__name__)

router = APIRouter()

#: Seconds a fresh connection has to say ``hello``.
HELLO_TIMEOUT_S = 10.0
#: Peers allowed before enrollment exists (#780).
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def _peer_is_loopback(ws: WebSocket) -> bool:
    client = ws.client
    host = client.host if client is not None else ""
    return host in LOOPBACK_HOSTS or host.startswith("127.") \
        or host == "::ffff:127.0.0.1"


@router.websocket(wp.WORKER_WS_PATH)
async def websocket_worker(ws: WebSocket) -> None:
    if not host_is_allowed(ws.headers.get("host", "")):
        await ws.close(code=4003, reason="Host not allowed")
        return
    if ws.headers.get("origin"):
        await ws.close(code=4003, reason="Browsers may not connect here")
        return
    if not _peer_is_loopback(ws):
        await ws.close(code=4003, reason="Only same-host workers until "
                                          "enrollment exists")
        return
    provided = (ws.headers.get("x-codefyui-token")
                or ws.query_params.get(TOKEN_QUERY_PARAM))
    if not constant_time_equals(provided, session_token()):
        await ws.close(code=4401, reason="Missing or invalid session token")
        return
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
    refusal = _refusal(hello)
    if refusal is not None:
        logger.warning("worker connection refused: %s", refusal)
        await send(wp.Reject(reason=refusal))
        await close()
        return

    worker, replaced = registry.connect(hello, send=send, close=close)
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
                await registry.route(worker, frame)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        registry.disconnect(worker)


def _refusal(frame: Any) -> str | None:
    """Why a first frame cannot register a worker, or None."""
    if not isinstance(frame, wp.Hello):
        return f"expected hello, got {frame.type}"
    if frame.protocol_version != wp.PROTOCOL_VERSION:
        return (f"protocol version {frame.protocol_version} is not this "
                f"server's {wp.PROTOCOL_VERSION}; update the agent")
    if TARGET_WORKER_PATTERN.fullmatch(frame.name) is None:
        return (f"worker name {frame.name!r} must be lowercase letters, "
                "digits and '-', at most 63 characters")
    if frame.name in RESERVED_NAMES:
        return f"worker name {frame.name!r} is reserved"
    return None
