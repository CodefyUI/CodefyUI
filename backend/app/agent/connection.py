"""The agent's WebSocket to the control plane's ``/ws/worker`` (#771).

The agent always dials out; it never listens. One connection at a time,
re-dialled with the backoff the frontend's execution socket uses (1 s,
doubling, capped at 30 s, back to 1 s after a ``welcome``).

Per connection:

1. send ``hello``; wait for ``welcome`` (``reject`` ends the agent: a refused
   token or version does not get better by retrying);
2. if runs survived a previous connection, send ``run_alive``;
3. send a ``heartbeat`` every ``welcome.heartbeat_s`` seconds, with a fresh
   device reading and the running run ids;
4. dispatch ``assign`` / ``cancel`` / ``drain`` to the :class:`Runner`.

Runs do not depend on the connection. While it is down, their subprocesses
keep running and the frames they produce are dropped (logged): until #783
adds a spool, the control plane has already marked those runs
``interrupted`` by then. ``drain`` refuses new assignments until the next
``welcome``.

Frames from every run's reader thread and from the heartbeat go out through
:meth:`AgentConnection.send_frame`, one at a time.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import platform
from typing import Any, Callable
from urllib.parse import quote, urlsplit, urlunsplit

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from ..core import worker_protocol as wp
from ..core.machine_identity import machine_identity
from .runner import Runner

logger = logging.getLogger(__name__)

#: Reconnect backoff, seconds: first wait and ceiling (the frontend's).
BACKOFF_MIN_S = 1.0
BACKOFF_MAX_S = 30.0
#: How long the control plane has to answer ``hello``.
WELCOME_TIMEOUT_S = 10.0


def _encode_or_warn(frame: Any) -> str:
    """*frame* as JSON; a frame that cannot be encoded becomes a ``warning``
    for its run with the same ``seq`` (a ``run_finished`` keeps its status
    and drops only what failed), so nothing vanishes without a trace."""
    try:
        return wp.encode_frame(frame)
    except (TypeError, ValueError, RecursionError) as exc:
        logger.warning("a %s frame could not be encoded: %s",
                       getattr(frame, "type", "?"), exc)
        if isinstance(frame, wp.RunFinished):
            return wp.encode_frame(frame.model_copy(update={
                "peak_memory": None,
                "error": (frame.error or "") + f" (part of this report "
                         f"could not be encoded: {exc})"}))
        run_id, seq = getattr(frame, "run_id", None), getattr(frame, "seq", None)
        if run_id is None or seq is None:
            raise
        return wp.encode_frame(wp.RunWarning(
            run_id=run_id, seq=seq, kind="unserialisable_frame",
            detail=f"a {frame.type} frame could not be encoded: {exc}"))


def _http_status(exc: BaseException) -> int | None:
    """The HTTP status a refused WebSocket handshake answered with."""
    response = getattr(exc, "response", None)  # websockets >= 14
    status = getattr(response, "status_code", None)
    if status is None:
        status = getattr(exc, "status_code", None)  # legacy client
    return status if isinstance(status, int) else None


class AgentRejected(RuntimeError):
    """The control plane refused this agent (``reject``); do not retry."""


def worker_ws_url(control_url: str, token: str | None) -> str:
    """``http(s)://host:port[/prefix]`` -> ``ws(s)://host:port[/prefix]/ws/worker?token=...``."""
    parts = urlsplit(control_url.strip())
    scheme = {"http": "ws", "https": "wss", "ws": "ws", "wss": "wss"}.get(
        parts.scheme.lower())
    if scheme is None or not parts.netloc:
        raise ValueError(f"control URL must be http(s)://host:port, got "
                         f"{control_url!r}")
    path = parts.path.rstrip("/") + wp.WORKER_WS_PATH
    query = f"token={quote(token, safe='')}" if token else ""
    return urlunsplit((scheme, parts.netloc, path, query, ""))


#: The header the token travels in (``core.auth.TOKEN_HEADER``). Not the
#: ``?token=`` query: uvicorn logs a WebSocket's path WITH its query string
#: at INFO, and a worker token is a long-lived credential (#780).
TOKEN_HEADER = "X-CodefyUI-Token"


def _headers_kwarg() -> str:
    """``websockets.connect``'s name for extra request headers:
    ``additional_headers`` from websockets 14, ``extra_headers`` before."""
    try:
        params = inspect.signature(websockets.connect).parameters
    except (TypeError, ValueError):  # pragma: no cover - C-level callable
        return "additional_headers"
    return ("additional_headers" if "additional_headers" in params
            else "extra_headers")


def next_backoff(current: float) -> float:
    """The wait after *current*: doubled, capped at :data:`BACKOFF_MAX_S`."""
    return min(max(current, BACKOFF_MIN_S) * 2, BACKOFF_MAX_S)


def platform_facts() -> dict[str, str]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }


class _NoToken(Exception):
    """No token to dial with; logged where it is raised."""


class AgentConnection:
    """Keeps one ``/ws/worker`` connection up and serves it.

    *control_url* and *token* may be callables, read at every dial: in
    same-host mode they come from files ``cdui start`` rewrites (the
    server's address and its per-process session token), so a restarted
    server is found again without restarting the agent.
    """

    def __init__(
        self, *, runner: Runner, name: str, version: str,
        control_url: str | Callable[[], str],
        token: str | None | Callable[[], str | None],
        probe_devices: Callable[[], list[wp.DeviceReport]],
        max_message_bytes: int | None = None,
        backoff_min_s: float = BACKOFF_MIN_S,
        token_source: str = "--token",
    ) -> None:
        self.runner = runner
        runner.send = self.send_frame
        self.name = name
        self.version = version
        self._control_url = control_url
        self._token = token
        self._probe_devices = probe_devices
        self.max_message_bytes = max_message_bytes
        self.backoff_min_s = backoff_min_s
        #: Where the token comes from, for error messages (a file path).
        self.token_source = token_source
        self.heartbeat_s = wp.DEFAULT_HEARTBEAT_S
        self.worker_id: str | None = None
        self._ws: Any = None
        self._send_lock = asyncio.Lock()

    # ── sending ───────────────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        return self._ws is not None

    async def send_frame(self, frame: Any) -> bool:
        """Send *frame* on the live connection. False when it was dropped."""
        async with self._send_lock:
            ws = self._ws
            if ws is None:
                logger.debug("not connected; dropped a %s frame",
                             getattr(frame, "type", "?"))
                return False
            try:
                await ws.send(_encode_or_warn(frame))
                return True
            except ConnectionClosed:
                logger.debug("connection closed; dropped a %s frame",
                             getattr(frame, "type", "?"))
                return False

    async def _device_reports(self) -> list[wp.DeviceReport]:
        try:
            return await asyncio.to_thread(self._probe_devices)
        except Exception:  # noqa: BLE001 -- a heartbeat without devices beats none
            logger.warning("device probe failed", exc_info=True)
            return []

    # ── the loop ──────────────────────────────────────────────────────────

    def _resolve(self, value: Any) -> Any:
        return value() if callable(value) else value

    async def run(self, stop: asyncio.Event) -> int:
        """Connect, serve, reconnect until *stop* is set. Returns an exit code.

        0 after *stop*; 2 when the control plane rejected this agent.
        Running runs are stopped before this returns, either way.
        """
        backoff = self.backoff_min_s
        code = 0
        while not stop.is_set():
            welcomed = False
            try:
                token = self._resolve(self._token)
                if not token:
                    logger.error("no connection token: %s", self.token_source)
                    raise _NoToken
                url = worker_ws_url(self._resolve(self._control_url), None)
                async with websockets.connect(
                        url, max_size=self.max_message_bytes,
                        open_timeout=WELCOME_TIMEOUT_S,
                        **{_headers_kwarg(): {TOKEN_HEADER: token}}) as ws:
                    welcomed = await self._serve(ws, stop)
            except AgentRejected as exc:
                logger.error("the control plane rejected this agent: %s", exc)
                code = 2
                break
            except _NoToken:
                pass
            except (OSError, WebSocketException, asyncio.TimeoutError,
                    wp.ProtocolError, ValueError) as exc:
                if _http_status(exc) in (401, 403):
                    logger.error(
                        "the control plane refused the token (HTTP %s); "
                        "token from %s", _http_status(exc), self.token_source)
                else:
                    logger.warning(
                        "connection to the control plane failed: %s", exc)
            if stop.is_set():
                break
            if welcomed:
                backoff = self.backoff_min_s
            logger.info("reconnecting in %.0f s", backoff)
            try:
                await asyncio.wait_for(stop.wait(), timeout=backoff)
            except asyncio.TimeoutError:
                pass
            backoff = min(next_backoff(backoff), BACKOFF_MAX_S)
        await self.runner.shutdown()
        return code

    async def _handshake(self, ws: Any) -> wp.Welcome:
        hello = wp.Hello(
            name=self.name, version=self.version,
            devices=await self._device_reports(), platform=platform_facts(),
            machine_id=await asyncio.to_thread(machine_identity))
        await ws.send(wp.encode_frame(hello))
        raw = await asyncio.wait_for(ws.recv(), timeout=WELCOME_TIMEOUT_S)
        frame = wp.decode_control_frame(raw)
        if isinstance(frame, wp.Reject):
            raise AgentRejected(frame.reason)
        if not isinstance(frame, wp.Welcome):
            raise wp.ProtocolError(f"expected welcome, got {frame.type!r}")
        if frame.protocol_version != wp.PROTOCOL_VERSION:
            raise AgentRejected(
                f"protocol version {frame.protocol_version} on the control "
                f"plane, {wp.PROTOCOL_VERSION} here; update so both match")
        return frame

    async def _serve(self, ws: Any, stop: asyncio.Event) -> bool:
        """Serve one connection until it closes or *stop*. True once welcomed."""
        welcome = await self._handshake(ws)
        self.worker_id = welcome.worker_id
        self.heartbeat_s = welcome.heartbeat_s
        self.runner.default_grace_s = welcome.grace_s
        self.runner.draining = False
        logger.info("connected as %s (worker id %s)", self.name,
                    welcome.worker_id)
        self._ws = ws
        tasks = [
            asyncio.create_task(self._heartbeats(), name="agent-heartbeat"),
            asyncio.create_task(self._receive(ws), name="agent-receive"),
            asyncio.create_task(stop.wait(), name="agent-stop"),
        ]
        try:
            if self.runner.runs:
                await self.send_frame(
                    wp.RunAlive(runs=self.runner.alive_entries()))
            done, _ = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED)
            if stop.is_set():
                # Stop the runs while the socket is still up, so their
                # run_finished frames reach the control plane.
                await self.runner.shutdown()
            for task in done:
                if not task.cancelled() and task.exception() is not None:
                    raise task.exception()  # type: ignore[misc]
        finally:
            self._ws = None
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        return True

    async def _heartbeats(self) -> None:
        while True:
            await self.send_frame(wp.Heartbeat(
                devices=await self._device_reports(),
                running=self.runner.running_ids()))
            await asyncio.sleep(self.heartbeat_s)

    async def _receive(self, ws: Any) -> None:
        try:
            async for raw in ws:
                try:
                    frame = wp.decode_control_frame(raw)
                except wp.ProtocolError as exc:
                    logger.warning("ignoring a bad frame: %s", exc)
                    continue
                try:
                    await self._dispatch(frame)
                except AgentRejected:
                    raise
                except Exception:  # noqa: BLE001 -- one bad frame, not the socket
                    logger.exception("handling a %s frame failed", frame.type)
        except ConnectionClosed:
            pass
        logger.warning("the control plane closed the connection")

    async def _dispatch(self, frame: Any) -> None:
        if isinstance(frame, wp.Assign):
            await self.runner.assign(frame)
        elif isinstance(frame, wp.Cancel):
            if not await self.runner.cancel(frame.run_id, frame.grace_s):
                logger.info("cancel for unknown run %s", frame.run_id)
        elif isinstance(frame, wp.Drain):
            logger.info("draining: no new runs until the next connection")
            self.runner.draining = True
        elif isinstance(frame, wp.Reject):
            raise AgentRejected(frame.reason)
        else:
            logger.debug("ignoring %s frame", frame.type)
