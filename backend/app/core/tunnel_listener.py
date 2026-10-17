"""The SSH workers' own listener (#784): where an ``ssh -R`` forward lands.

An SSH worker's agent reaches this server through a reverse forward, and
everyone who can reach the forward's remote end can send whatever they
like through it, ``Host`` header included. So the forward never lands on
the main port. Each SSH worker's launch (``core.ssh_launch``) opens a
listener of its own (:class:`TunnelListener`: a Unix socket in a private
directory on POSIX, an ephemeral loopback port on Windows) and ``-R``
points there. That listener serves the agent routes and nothing else
(:func:`agent_route`, checked in :class:`TunnelApp` and only there):

- ``WS /ws/worker``, with a worker token (the session token is refused
  here, whatever the peer address);
- ``POST /api/workers/enroll``, with an enrollment token;
- the transfer backend's three input downloads and
  ``POST /api/runs/{id}/artifacts/upload`` (#785), with a worker token.

Every other path answers 404, before the application sees it: the session
bootstrap, the UI and the rest of the API do not exist on this socket.

Requests that came this way carry :data:`SCOPE_KEY` (the SSH worker's
name) in their ASGI scope, set here and nowhere else, which is how the
application tells them apart (:func:`tunnel_of`). Nothing a client sends
decides it: the main port treats every ``Host`` the same as before #784.

The listener runs a second ``uvicorn.Server`` in this process and event
loop, without the application's lifespan (it is the same app, already
started) and without signal handlers of its own (the main server's stay).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import socket
import sys
import tempfile
from typing import Any

logger = logging.getLogger(__name__)

#: The ASGI scope key naming the SSH worker a request came through.
SCOPE_KEY = "codefyui.ssh_tunnel"
#: Where the listener binds. Loopback only: ssh connects from this machine.
BIND_HOST = "127.0.0.1"
#: How long :meth:`TunnelListener.close` lets open connections finish.
CLOSE_TIMEOUT_S = 5.0

WORKER_WS_PATH = "/ws/worker"
ENROLL_PATH = "/api/workers/enroll"


def tunnel_of(scope: Any) -> str | None:
    """The SSH worker whose listener *scope* arrived on, else None."""
    value = scope.get(SCOPE_KEY) if isinstance(scope, dict) else None
    return value if isinstance(value, str) else None


def worker_token_route(method: str, path: str) -> bool:
    """The transfer backend's routes an agent uses (#785): the three input
    downloads (GET) and the output upload (POST). Through a tunnel they
    need a worker token (``main.host_guard`` checks it); on the main port
    the downloads are open reads like every GET."""
    from .transfer import DOWNLOAD_ROUTES, UPLOAD_PATH_PATTERN

    if method == "GET":
        return any(path.startswith(route) and len(path) > len(route)
                   for route in DOWNLOAD_ROUTES.values())
    return method == "POST" and UPLOAD_PATH_PATTERN.fullmatch(path) is not None


def agent_route(scope_type: str, method: str, path: str) -> bool:
    """Whether a tunnel listener serves *method* *path* at all."""
    if scope_type == "websocket":
        return path == WORKER_WS_PATH
    if scope_type != "http":
        return False
    if method == "POST" and path == ENROLL_PATH:
        return True
    return worker_token_route(method, path)


class TunnelApp:
    """ASGI: *app* with only the agent routes, every request marked with
    the SSH worker *name* it came through."""

    def __init__(self, app: Any, name: str) -> None:
        self.app = app
        self.name = name

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        kind = scope.get("type")
        if kind == "lifespan":
            # The application is already started; this listener has no
            # lifespan of its own (uvicorn runs it with ``lifespan="off"``).
            return
        path = scope.get("path", "")
        if not agent_route(kind, scope.get("method", "GET"), path):
            await _not_found(kind, receive, send)
            return
        await self.app({**scope, SCOPE_KEY: self.name}, receive, send)


async def _not_found(kind: str | None, receive: Any, send: Any) -> None:
    if kind == "websocket":
        await receive()  # websocket.connect
        await send({"type": "websocket.close", "code": 4404,
                    "reason": "Not found"})
        return
    body = b'{"detail":"Not Found"}'
    await send({"type": "http.response.start", "status": 404,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


class TunnelListener:
    """One listener for one SSH worker: :meth:`open`, read :attr:`target`
    (``-R``'s right-hand side), :meth:`close`.

    On POSIX it is a Unix socket in a private temporary directory (mode
    0700, from ``tempfile.mkdtemp``): only this account can connect, so no
    other local user can reach the agent routes, spend the worker's
    enrollment attempts or speak to it as the agent. Windows (no
    ``AF_UNIX`` server in uvicorn's loop there, and ``ssh.exe`` forwarding
    to a socket is not something to rely on) gets an ephemeral port on
    127.0.0.1 instead.
    """

    def __init__(self, app: Any, name: str, *,
                 use_unix: bool | None = None) -> None:
        self.app = TunnelApp(app, name)
        self.name = name
        self.use_unix = (use_unix if use_unix is not None
                         else sys.platform != "win32"
                         and hasattr(socket, "AF_UNIX"))
        #: ``-R``'s local end: a socket path, or ``127.0.0.1:<port>``. Set
        #: only once the server has started; None before and after.
        self.target: str | None = None
        self.port: int | None = None
        self.path: str | None = None
        self._dir: str | None = None
        self._server: Any = None
        self._task: asyncio.Task | None = None

    @property
    def alive(self) -> bool:
        """Started, and still serving."""
        return (self.target is not None and self._task is not None
                and not self._task.done())

    def _bind(self) -> socket.socket:
        if self.use_unix:
            base = tempfile.gettempdir()
            if len(base) > 60:  # keep the path well inside ``sun_path``
                base = "/tmp"
            self._dir = tempfile.mkdtemp(prefix="cdui-ssh-", dir=base)
            path = os.path.join(self._dir, "s")
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.bind(path)
                os.chmod(path, 0o600)
            except OSError:
                sock.close()
                self._cleanup_dir()
                raise
            self.path = path
            return sock
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((BIND_HOST, 0))
        except OSError:
            sock.close()
            raise
        return sock

    async def open(self) -> str:
        import uvicorn

        from ..config import settings

        class _Server(uvicorn.Server):
            # The main server owns SIGINT / SIGTERM.
            @contextlib.contextmanager
            def capture_signals(self):  # type: ignore[override]
                yield

        sock = self._bind()
        try:
            sock.listen(64)
            sock.setblocking(False)
            config = uvicorn.Config(
                self.app, lifespan="off", log_config=None, access_log=False,
                proxy_headers=False, server_header=False,
                # The main port's own cap (``cdui start`` hands uvicorn
                # ``--ws-max-size`` from the same setting).
                ws_max_size=settings.WS_MAX_MESSAGE_BYTES,
                timeout_graceful_shutdown=CLOSE_TIMEOUT_S)
            server = _Server(config)
            task = asyncio.get_running_loop().create_task(
                server.serve(sockets=[sock]), name=f"ssh-tunnel-{self.name}")
            while not server.started and not task.done():
                await asyncio.sleep(0.01)
            if task.done():
                task.result()  # raises what stopped it
                raise OSError(f"the SSH tunnel listener for {self.name} "
                              "did not start")
        except BaseException:
            sock.close()
            self._cleanup_dir()
            raise
        self._server, self._task = server, task
        if self.path is not None:
            self.target = self.path
        else:
            self.port = sock.getsockname()[1]
            self.target = f"{BIND_HOST}:{self.port}"
        logger.info("SSH worker %s: tunnel listener on %s", self.name,
                    self.target)
        return self.target

    def _cleanup_dir(self) -> None:
        if self._dir is not None:
            shutil.rmtree(self._dir, ignore_errors=True)
        self._dir = None

    async def close(self) -> None:
        server, task = self._server, self._task
        self._server = self._task = None
        self.target = self.port = self.path = None
        if server is not None and task is not None:
            server.should_exit = True
            try:
                await asyncio.wait_for(asyncio.shield(task),
                                       CLOSE_TIMEOUT_S * 2)
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                server.force_exit = True
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            logger.info("SSH worker %s: tunnel listener closed", self.name)
        self._cleanup_dir()
