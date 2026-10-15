"""The agent through a Unix socket (#784): ``cdui agent start --uds PATH``.

An SSH worker's agent dials the forward's socket (in a directory only its
account can open), never a loopback TCP port another user of the host
could bind first. Every connection goes there: the enrollment, the worker
socket and the transfer backend's requests; the ``--control`` URL still
names the control plane.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import socketserver
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import websockets

from app.agent import cli
from app.agent import enroll as agent_enroll
from app.agent.connection import AgentConnection
from app.agent.runner import Runner
from app.agent.transfer import InputCache, TransferClient

from .test_connection import _cpu

pytestmark = pytest.mark.skipif(sys.platform == "win32",
                                reason="Unix-domain sockets")

ENROLL_TOKEN = "abcdef.0123456789abcdef"


@pytest.fixture
def sock_dir():
    # Short: ``sun_path`` holds about 100 bytes.
    path = Path(tempfile.mkdtemp(prefix="cdui-uds-", dir="/tmp"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


class UnixControlPlane:
    """An HTTP server on a Unix socket: enroll (POST) and a download (GET)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.requests: list[dict[str, Any]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self, status: int, payload: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                outer.requests.append({
                    "method": "POST", "path": self.path,
                    "host": self.headers.get("Host"),
                    "body": json.loads(self.rfile.read(length))})
                self._answer(200, json.dumps({
                    "worker_id": "gpu-box",
                    "worker_token": "w" * 43}).encode())

            def do_GET(self) -> None:  # noqa: N802
                outer.requests.append({
                    "method": "GET", "path": self.path,
                    "host": self.headers.get("Host"),
                    "token": self.headers.get("X-CodefyUI-Token")})
                self._answer(200, b"a,b\n1,2\n")

            def address_string(self) -> str:
                return "unix"

            def log_message(self, *_: Any) -> None:
                pass

        self.server = socketserver.ThreadingUnixStreamServer(str(path),
                                                             Handler)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)

    def __enter__(self) -> "UnixControlPlane":
        self.thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.server.shutdown()
        self.server.server_close()


def test_enroll_goes_to_the_socket(sock_dir):
    with UnixControlPlane(sock_dir / "sock") as cp:
        worker_id, token = agent_enroll.enroll(
            "http://127.0.0.1:18000", ENROLL_TOKEN, name="gpu-box",
            platform={}, devices=[], uds=str(sock_dir / "sock"))
    assert (worker_id, token) == ("gpu-box", "w" * 43)
    [request] = cp.requests
    assert request["path"] == "/api/workers/enroll"
    # The URL still names the control plane.
    assert request["host"] == "127.0.0.1:18000"


def test_enroll_through_the_socket_ignores_a_proxy(sock_dir, monkeypatch):
    """``http_proxy`` on the GPU host must not take the request: it would
    get an absolute URL (and a Proxy-Authorization) instead of the socket
    getting the request."""
    for name in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.setenv(name, "http://user:secret@127.0.0.1:9")
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    with UnixControlPlane(sock_dir / "sock") as cp:
        worker_id, _ = agent_enroll.enroll(
            "http://cp-abcd1234.invalid:18000", ENROLL_TOKEN, name="gpu-box",
            platform={}, devices=[], uds=str(sock_dir / "sock"))
    assert worker_id == "gpu-box"
    [request] = cp.requests
    assert request["path"] == "/api/workers/enroll"
    assert request["host"] == "cp-abcd1234.invalid:18000"


def test_enroll_to_a_missing_socket_is_an_enroll_error(sock_dir):
    with pytest.raises(agent_enroll.EnrollError, match="could not reach"):
        agent_enroll.enroll("http://127.0.0.1:18000", ENROLL_TOKEN,
                            name="gpu-box", platform={}, devices=[],
                            uds=str(sock_dir / "missing"), timeout=2)


def test_the_transfer_client_goes_to_the_socket(sock_dir, tmp_path):
    client = TransferClient(base_url="http://127.0.0.1:18000",
                            token="tok", cache=InputCache(tmp_path / "cache"),
                            uds=str(sock_dir / "sock"))
    with UnixControlPlane(sock_dir / "sock") as cp, client.session() as http:
        response = http.get(client._url("/api/files/download/train.csv"),
                            headers=client._headers())
    assert response.status_code == 200 and response.content == b"a,b\n1,2\n"
    [request] = cp.requests
    assert request["token"] == "tok"
    assert request["host"] == "127.0.0.1:18000"


async def test_the_worker_socket_is_dialled_through_the_unix_socket(
        sock_dir, tmp_path):
    from websockets.datastructures import Headers
    from websockets.http11 import Response

    seen: list[tuple[str, str | None, str | None]] = []

    def record(connection, request):
        seen.append((request.path, request.headers.get("X-CodefyUI-Token"),
                     request.headers.get("Host")))
        return Response(403, "Forbidden", Headers(), b"no")

    server = await websockets.unix_serve(lambda ws: None,
                                         str(sock_dir / "sock"),
                                         process_request=record)
    runner = Runner(runs_dir=tmp_path / "runs", offered_devices={"cpu"})
    connection = AgentConnection(
        runner=runner, name="gpu-box", version="9.9.9",
        control_url="http://127.0.0.1:18000", token="w" * 43,
        probe_devices=_cpu, backoff_min_s=0.05, uds=str(sock_dir / "sock"))
    assert runner.transfer is not None and runner.transfer.uds == str(
        sock_dir / "sock")
    stop = asyncio.Event()
    task = asyncio.create_task(connection.run(stop))
    try:
        deadline = time.monotonic() + 10
        while not seen and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
    finally:
        stop.set()
        await asyncio.wait_for(task, 30)
        server.close()
        await server.wait_closed()
    assert seen and seen[0] == ("/ws/worker", "w" * 43, "127.0.0.1:18000")


@pytest.fixture
def user_data(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "_REPO_ROOT", tmp_path / "no-repo")
    monkeypatch.delenv(cli.TOKEN_ENV, raising=False)
    monkeypatch.setenv("CODEFYUI_LANG", "en")
    monkeypatch.setattr(cli, "_interactive", lambda: False)
    monkeypatch.setattr(cli, "_probe_facts", lambda: [SimpleNamespace(
        value="cpu", kind="cpu", index=None, name="CPU", total_bytes=None,
        free_bytes=None, source="none", uuid=None)])
    return tmp_path


@pytest.mark.parametrize("control", [None, "https://cp.example:8443"])
def test_uds_needs_an_http_control_url(user_data, control, capsys):
    argv = ["start", "--foreground", "--uds", "/tmp/x.sock",
            "--token", "w" * 43]
    if control:
        argv += ["--control", control]
    assert cli.main(argv) == 2
    assert "--uds needs an http:// --control URL" in capsys.readouterr().err


def test_uds_reaches_the_enrollment_and_the_agent(user_data, sock_dir,
                                                  monkeypatch, capsys):
    started: list[Any] = []

    def fake_run_agent(args, control, token, facts, token_source=None,
                       mounts=None, trust=None):
        started.append((args.uds, control, token))
        return 0

    monkeypatch.setattr(cli, "run_agent", fake_run_agent)
    monkeypatch.setattr(cli, "_setup_logging", lambda daemonized: None)
    sock = str(sock_dir / "sock")
    with UnixControlPlane(sock_dir / "sock") as cp:
        code = cli.main(["start", "--foreground", "--uds", sock,
                         "--control", "http://cp-abcd1234.invalid:18000",
                         "--name", "gpu-box", "--token", ENROLL_TOKEN])
    assert code == 0
    assert [r["path"] for r in cp.requests] == ["/api/workers/enroll"]
    assert started == [(sock, "http://cp-abcd1234.invalid:18000", "w" * 43)]
    # ssh carries the socket across the network: no plain-HTTP warning.
    assert "plain HTTP" not in capsys.readouterr().err


def test_the_background_agent_keeps_the_socket(user_data, monkeypatch):
    launched: list[list[str]] = []

    class FakePopen:
        pid = 4321

        def __init__(self, argv, **_: Any) -> None:
            launched.append(argv)

        def poll(self) -> None:
            return None

    monkeypatch.setattr(cli.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(cli, "START_SETTLE_S", 0.0)
    args = cli.build_parser().parse_args([
        "start", "--uds", "/tmp/x.sock", "--name", "gpu-box"])
    assert cli._start_background(args, "http://127.0.0.1:18000",
                                 "w" * 43) == 0
    [argv] = launched
    assert argv[argv.index("--uds") + 1] == "/tmp/x.sock"
