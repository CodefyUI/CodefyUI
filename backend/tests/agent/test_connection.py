"""``app.agent.connection`` against a fake ``/ws/worker`` server (#771).

The fake control plane is a ``websockets`` server on a free loopback port
that speaks ``core.worker_protocol``: it records every frame the agent
sends and lets a test push frames back.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
import websockets

from app.agent.connection import (
    BACKOFF_MAX_S,
    AgentConnection,
    next_backoff,
    worker_ws_url,
)
from app.agent.facts import AgentFacts
from app.agent.runner import Runner
from app.core import worker_protocol as wp
from app.core.machine_identity import machine_identity

from . import _helpers as h


class FakeControlPlane:
    """Answers ``hello`` with ``welcome`` (or ``reject``) and records frames."""

    def __init__(self, *, reject: str | None = None,
                 heartbeat_s: float = 0.2, grace_s: float = 5.0) -> None:
        self.reject = reject
        self.heartbeat_s = heartbeat_s
        self.grace_s = grace_s
        self.frames: list[tuple[int, Any]] = []
        self.paths: list[str] = []
        self.tokens: list[str | None] = []
        self.sockets: list[Any] = []
        self._changed = asyncio.Event()
        self.server: Any = None

    @property
    def url(self) -> str:
        port = self.server.sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{port}"

    async def __aenter__(self) -> "FakeControlPlane":
        self.server = await websockets.serve(self._handle, "127.0.0.1", 0)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.server.close()
        await self.server.wait_closed()

    def _record(self, number: int, frame: Any) -> None:
        self.frames.append((number, frame))
        self._changed.set()

    async def _handle(self, ws: Any, *_: Any) -> None:
        request = getattr(ws, "request", None)
        self.paths.append(request.path if request is not None else ws.path)
        headers = request.headers if request is not None else ws.request_headers
        self.tokens.append(headers.get("X-CodefyUI-Token"))
        number = len(self.sockets)
        self.sockets.append(ws)
        try:
            self._record(number, wp.decode_agent_frame(await ws.recv()))
            if self.reject is not None:
                await ws.send(wp.encode_frame(wp.Reject(reason=self.reject)))
                return
            await ws.send(wp.encode_frame(wp.Welcome(
                worker_id=f"w-{number}", heartbeat_s=self.heartbeat_s,
                grace_s=self.grace_s)))
            async for raw in ws:
                self._record(number, wp.decode_agent_frame(raw))
        except websockets.exceptions.ConnectionClosed:
            pass

    async def send(self, frame: Any, connection: int = -1) -> None:
        await self.sockets[connection].send(wp.encode_frame(frame))

    async def send_raw(self, raw: str, connection: int = -1) -> None:
        await self.sockets[connection].send(raw)

    def of_type(self, kind: type, connection: int | None = None) -> list[Any]:
        return [f for n, f in self.frames if isinstance(f, kind)
                and (connection is None or n == connection)]

    async def wait_for(self, predicate: Any, timeout: float = 60.0) -> Any:
        deadline = time.monotonic() + timeout
        while True:
            for number, frame in self.frames:
                if predicate(number, frame):
                    return frame
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(
                    f"no matching frame in {timeout}s; got {self.frames}")
            self._changed.clear()
            try:
                await asyncio.wait_for(self._changed.wait(), remaining)
            except asyncio.TimeoutError:
                pass


def _cpu() -> list[wp.DeviceReport]:
    return [wp.DeviceReport(value="cpu", kind="cpu", name="test",
                            total_bytes=1, free_bytes=1, source="psutil")]


class Agent:
    """An ``AgentConnection`` serving in a task, stopped on exit."""

    def __init__(self, tmp_path: Any, control: str, facts: Any = None,
                 **runner_kw: Any) -> None:
        runner_kw.setdefault("offered_devices", {"cpu"})
        self.runner = Runner(runs_dir=tmp_path / "agent-runs", **runner_kw)
        self.connection = AgentConnection(
            runner=self.runner, name="test-box", version="9.9.9",
            control_url=lambda: control, token="tok en/1",
            probe_devices=_cpu, backoff_min_s=0.05, facts=facts)
        self.stop = asyncio.Event()
        self.task: asyncio.Task | None = None

    async def __aenter__(self) -> "Agent":
        self.task = asyncio.create_task(self.connection.run(self.stop))
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.stop.set()
        await asyncio.wait_for(self.task, 30)


# ── pure helpers ──────────────────────────────────────────────────────────


def test_worker_ws_url():
    assert worker_ws_url("http://127.0.0.1:8000", "a b") == (
        "ws://127.0.0.1:8000/ws/worker?token=a%20b")
    assert worker_ws_url("https://cp.example/base/", None) == (
        "wss://cp.example/base/ws/worker")
    with pytest.raises(ValueError):
        worker_ws_url("127.0.0.1:8000", "t")


def test_backoff_doubles_to_thirty_seconds():
    waits = [1.0]
    for _ in range(8):
        waits.append(next_backoff(waits[-1]))
    assert waits[:6] == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0]
    assert waits[-1] == BACKOFF_MAX_S


async def test_send_frame_while_disconnected_is_dropped(tmp_path):
    agent = Agent(tmp_path, "http://127.0.0.1:1")
    assert await agent.connection.send_frame(wp.Heartbeat()) is False


# ── against the fake control plane ───────────────────────────────────────


async def test_hello_and_heartbeats_carry_the_mounts_that_exist(tmp_path):
    """``--mount`` labels reach the control plane in ``hello`` (#782); a
    saved mount whose directory is missing is left out, and offered by the
    next heartbeat once it appears."""
    nas = tmp_path / "nas"
    nas.mkdir()
    later = tmp_path / "later"
    async with FakeControlPlane() as cp, Agent(
            tmp_path, cp.url,
            mounts={"data": str(nas), "scratch": str(later)}) as agent:
        agent.connection.heartbeat_s = 0.05
        hello = await cp.wait_for(lambda n, f: isinstance(f, wp.Hello))
        assert hello.mounts == {"data": str(nas)}
        later.mkdir()
        beat = await cp.wait_for(lambda n, f: isinstance(f, wp.Heartbeat)
                                 and "scratch" in (f.mounts or {}))
        assert beat.mounts == {"data": str(nas), "scratch": str(later)}


async def test_hello_heartbeats_and_a_run_match_in_process(tmp_path):
    """Acceptance: connect, hello, heartbeats, a tiny CPU graph in a child,
    and event frames byte-identical to InProcessExecutor's."""
    async with FakeControlPlane() as cp, Agent(tmp_path, cp.url):
        hello = await cp.wait_for(lambda n, f: isinstance(f, wp.Hello))
        assert hello.protocol_version == wp.PROTOCOL_VERSION
        assert hello.name == "test-box" and hello.version == "9.9.9"
        assert [d.value for d in hello.devices] == ["cpu"]
        assert set(hello.platform) == {"system", "release", "machine",
                                       "python"}
        # A digest of this machine's id, never the raw id (#773).
        assert hello.machine_id == machine_identity()
        # The token travels in a header, never in the logged query (#780).
        assert cp.paths[0] == wp.WORKER_WS_PATH
        assert cp.tokens[0] == "tok en/1"
        await cp.wait_for(lambda n, f: len(cp.of_type(wp.Heartbeat)) >= 2)

        await cp.send(wp.Assign(run_id="run1", graph=h.tensor_graph(),
                                options=h.TENSOR_OPTIONS))
        finished = await cp.wait_for(
            lambda n, f: isinstance(f, wp.RunFinished), timeout=120)
        assert finished.status == "succeeded"
        run_frames = [f for _, f in cp.frames
                      if getattr(f, "run_id", None) == "run1"]
        assert run_frames[0].type == "run_started"
        assert [f.seq for f in run_frames] == list(
            range(1, len(run_frames) + 1))
        worker = [h.event_bytes(f.event_type, f.payload) for f in run_frames
                  if isinstance(f, wp.Event)]
        assert worker == await h.in_process_events(h.tensor_graph(),
                                                   h.TENSOR_OPTIONS)
        # A heartbeat during the run named it.
        assert any("run1" in beat.running
                   for beat in cp.of_type(wp.Heartbeat))


async def test_hello_carries_the_facts_and_capabilities(tmp_path):
    """#781: the fingerprint and node catalog the helper collected, and
    this build's capabilities; none of them without facts."""
    facts = AgentFacts(fingerprint={"a": 1}, fingerprint_digest="d",
                       node_catalog_hash="h", node_types=["Start", "x:Y"])
    async with FakeControlPlane() as cp, Agent(tmp_path, cp.url,
                                               facts=facts):
        hello = await cp.wait_for(lambda n, f: isinstance(f, wp.Hello))
        assert hello.fingerprint == wp.FingerprintReport(document={"a": 1},
                                                         digest="d")
        assert (hello.node_catalog_hash, hello.node_types) == (
            "h", ["Start", "x:Y"])
        assert hello.capabilities == list(wp.CAPABILITIES)
    async with FakeControlPlane() as cp, Agent(tmp_path, cp.url):
        hello = await cp.wait_for(lambda n, f: isinstance(f, wp.Hello))
        assert hello.fingerprint is None and hello.node_catalog_hash is None
        assert hello.node_types == []


async def test_reject_stops_the_agent(tmp_path):
    async with FakeControlPlane(reject="bad token") as cp:
        agent = Agent(tmp_path, cp.url)
        code = await asyncio.wait_for(agent.connection.run(agent.stop), 15)
        assert code == 2
        assert len(cp.of_type(wp.Hello)) == 1  # no retry after a reject


async def test_reconnect_reports_runs_alive_and_cancel_still_works(tmp_path):
    async with FakeControlPlane() as cp, Agent(
            tmp_path, cp.url, child_entry=h.deaf_child, kill_after_s=1.0):
        await cp.wait_for(lambda n, f: isinstance(f, wp.Hello))
        await cp.send(wp.Assign(run_id="live1", graph={"nodes": []}))
        await cp.wait_for(lambda n, f: isinstance(f, wp.Event))
        await cp.sockets[0].close()  # the network drops

        alive = await cp.wait_for(
            lambda n, f: n == 1 and isinstance(f, wp.RunAlive), timeout=15)
        assert [(e.run_id, e.last_seq) for e in alive.runs] == [("live1", 2)]
        assert len(cp.of_type(wp.Hello)) == 2

        await cp.send(wp.Cancel(run_id="live1", grace_s=0.2))
        finished = await cp.wait_for(
            lambda n, f: n == 1 and isinstance(f, wp.RunFinished), timeout=15)
        assert finished.status == "interrupted" and finished.seq == 3


async def test_drain_refuses_assignments_and_bad_frames_are_ignored(tmp_path):
    async with FakeControlPlane() as cp, Agent(
            tmp_path, cp.url, child_entry=h.quick_child):
        await cp.wait_for(lambda n, f: isinstance(f, wp.Hello))
        await cp.send_raw("this is not a frame")
        await cp.send_raw('{"type": "assign"}')
        await cp.send(wp.Assign(run_id="ok1", graph={"nodes": []}))
        ok = await cp.wait_for(lambda n, f: isinstance(f, wp.RunFinished))
        assert ok.run_id == "ok1" and ok.status == "succeeded"

        await cp.send(wp.Drain())
        await cp.send(wp.Assign(run_id="no1", graph={"nodes": []}))
        refused = await cp.wait_for(
            lambda n, f: isinstance(f, wp.RunFinished) and f.run_id == "no1")
        assert refused.status == "failed" and "draining" in refused.error


async def test_stop_while_connected_reports_running_runs(tmp_path):
    async with FakeControlPlane() as cp:
        agent = Agent(tmp_path, cp.url, child_entry=h.deaf_child,
                      kill_after_s=1.0)
        async with agent:
            await cp.wait_for(lambda n, f: isinstance(f, wp.Hello))
            await cp.send(wp.Assign(run_id="stop1", graph={"nodes": []}))
            await cp.wait_for(lambda n, f: isinstance(f, wp.Event))
            agent.runner.shutdown_grace_s = 0.2
        finished = await cp.wait_for(
            lambda n, f: isinstance(f, wp.RunFinished), timeout=5)
        assert finished.status == "interrupted"
        assert agent.task.result() == 0


def test_an_unencodable_frame_becomes_a_warning_with_its_seq():
    from app.agent.connection import _encode_or_warn

    loop: dict = {}
    loop["self"] = loop
    raw = _encode_or_warn(wp.Event(run_id="r1", seq=4, event_type="x",
                                   payload=loop))
    frame = wp.decode_agent_frame(raw)
    assert isinstance(frame, wp.RunWarning)
    assert (frame.seq, frame.kind) == (4, "unserialisable_frame")
    finished = wp.decode_agent_frame(_encode_or_warn(wp.RunFinished(
        run_id="r1", seq=5, status="succeeded", peak_memory={"x": loop})))
    assert finished.status == "succeeded" and finished.peak_memory is None


async def test_a_refused_token_is_logged_as_an_error_and_retried(
        tmp_path, caplog):
    from websockets.datastructures import Headers
    from websockets.http11 import Response

    def refuse(connection, request):
        return Response(403, "Forbidden", Headers(), b"bad token")

    server = await websockets.serve(lambda ws: None, "127.0.0.1", 0,
                                    process_request=refuse)
    try:
        port = server.sockets[0].getsockname()[1]
        agent = Agent(tmp_path, f"http://127.0.0.1:{port}")
        agent.connection.token_source = "the session token file /x/session.token"
        with caplog.at_level("ERROR", logger="app.agent.connection"):
            async with agent:
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline and len(
                        [r for r in caplog.records
                         if r.levelname == "ERROR"]) < 2:
                    await asyncio.sleep(0.05)
        errors = [r.getMessage() for r in caplog.records
                  if r.levelname == "ERROR"]
        assert len(errors) >= 2  # backed off and tried again
        assert "HTTP 403" in errors[0] and "/x/session.token" in errors[0]
    finally:
        server.close()
        await server.wait_closed()


async def test_no_token_is_logged_and_retried(tmp_path, caplog):
    agent = Agent(tmp_path, "http://127.0.0.1:1")
    agent.connection._token = lambda: None
    with caplog.at_level("ERROR", logger="app.agent.connection"):
        async with agent:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and len(caplog.records) < 2:
                await asyncio.sleep(0.05)
    assert sum("no connection token" in r.getMessage()
               for r in caplog.records) >= 2


async def test_cancel_grace_falls_back_to_assign_then_welcome(tmp_path):
    async with FakeControlPlane(grace_s=0.3) as cp, Agent(
            tmp_path, cp.url, child_entry=h.deaf_child, kill_after_s=1.0):
        await cp.wait_for(lambda n, f: isinstance(f, wp.Hello))
        await cp.send(wp.Assign(run_id="g1", graph={"nodes": []}))
        await cp.send(wp.Assign(run_id="g2", graph={"nodes": []},
                                grace_s=60))
        await cp.wait_for(lambda n, f: isinstance(f, wp.Event)
                          and f.run_id == "g2")
        started = time.monotonic()
        await cp.send(wp.Cancel(run_id="g1"))  # welcome's 0.3 s
        await cp.wait_for(lambda n, f: isinstance(f, wp.RunFinished)
                          and f.run_id == "g1", timeout=10)
        assert time.monotonic() - started < 5
        await cp.send(wp.Cancel(run_id="g2", grace_s=0.1))  # cancel wins
        finished = await cp.wait_for(
            lambda n, f: isinstance(f, wp.RunFinished) and f.run_id == "g2",
            timeout=10)
        assert finished.status == "interrupted"

