"""WebSocket protocol v2: attach / detach / cancel over the Run Service (#121).

The headline behaviour of the whole run-service wave is one sentence: *close
the tab, the run survives; reopen, re-attach, see everything you missed.*
That splits into four things this module pins down:

1. The legacy ``execute`` message still works, unchanged, and now runs on the
   interactive lane with the canvas's own cache / module store / dirty-node
   hint (``test_ws_execution.py`` covers the resulting frames; here we check
   the run ROW and the parity of what reaches the engine).
2. ``attach`` replays ``exec_run_events`` from a cursor and then live-tails
   with NO GAP and NO DUPLICATE across the boundary — including when the
   subscriber queue overflows and the pump has to fall back to the store.
3. ``detach``, and closing the socket, unsubscribe and NOTHING ELSE.
4. ``cancel`` is the only thing that stops a run.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest
from httpx import AsyncClient
from httpx_ws import aconnect_ws
from httpx_ws.transport import ASGIWebSocketTransport

from app.config import settings
from app.core.auth import TOKEN_QUERY_PARAM, session_token
from app.core.db import Database
from app.core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition
from app.core.node_registry import registry
from app.core.node_state_store import NodeStateStore
from app.core.run_output_store import RunOutputStore
from app.core.run_service import LANE_INTERACTIVE, RunService
from app.core.run_store import RunRecord, RunStore
from app.main import app

_BASE_URL = f"http://127.0.0.1:{settings.PORT}"
_WS_PATH = f"/ws/execution?{TOKEN_QUERY_PARAM}={session_token()}"

#: Nothing here waits on wall-clock; this is only a deadlock guard.
_RECV_TIMEOUT = 20.0


# ── test nodes ────────────────────────────────────────────────────────────


class _WsSlowNode(BaseNode):
    """Blocks a worker thread — the "run outlives the socket" probe."""

    NODE_NAME = "_WsSlow"
    CATEGORY = "Test"
    DESCRIPTION = "Sleeps, then passes through"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [ParamDefinition(name="seconds", param_type=ParamType.FLOAT,
                                default=0.6)]

    def execute(self, inputs: dict[str, Any],
                params: dict[str, Any]) -> dict[str, Any]:
        time.sleep(float(params.get("seconds", 0.6)))
        return {"value": inputs.get("value")}


class _WsChattyNode(BaseNode):
    """Emits many progress events, so a replay has something to overlap."""

    NODE_NAME = "_WsChatty"
    CATEGORY = "Test"
    DESCRIPTION = "Emits N epoch progress events"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [ParamDefinition(name="count", param_type=ParamType.INT,
                                default=40)]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any],
                progress_callback=None) -> dict[str, Any]:
        for epoch in range(1, int(params.get("count", 40)) + 1):
            if progress_callback:
                progress_callback({"event": "epoch", "epoch": epoch,
                                   "loss": 1.0 / epoch})
        return {"value": inputs.get("value")}


class _WsContextProbeNode(BaseNode):
    """Reports the ExecutionContext it was handed. The canvas-parity probe."""

    NODE_NAME = "_WsContextProbe"
    CATEGORY = "Test"
    DESCRIPTION = "Logs the per-run feature flags it can see"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any],
                context=None) -> dict[str, Any]:
        return {
            "value": {
                "execution_id": getattr(context, "execution_id", None),
                "device": getattr(context, "device", None),
                "verbose": getattr(context, "verbose", None),
                "graph_id": getattr(context, "graph_id", None),
                "weights_persistent": getattr(context, "weights_persistent", None),
                "backward_mode": getattr(context, "backward_mode", None),
                "auto_backward": getattr(context, "auto_backward", None),
                "has_state_store": getattr(context, "node_state_store", None) is not None,
            }
        }


_TEST_NODES = {
    "_WsSlow": _WsSlowNode,
    "_WsChatty": _WsChattyNode,
    "_WsContextProbe": _WsContextProbeNode,
}


@pytest.fixture(autouse=True)
def _register_test_nodes():
    registry._nodes.update(_TEST_NODES)
    yield
    for name in _TEST_NODES:
        registry._nodes.pop(name, None)


# ── fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
async def run_env(request, tmp_path):
    """Database + RunService + the two canvas stores on ``app.state``.

    Mirrors ``main.lifespan``, including its teardown ORDER (drain the
    service, then close the database). Parametrize indirectly to change the
    service's construction, e.g.::

        @pytest.mark.parametrize("run_env", [{"subscriber_queue_size": 1}],
                                 indirect=True)
    """
    kwargs: dict[str, Any] = getattr(request, "param", None) or {}
    database = Database(tmp_path / "codefyui.db")
    database.connect()
    output_store = RunOutputStore(max_runs=20)
    node_state_store = NodeStateStore(max_modules=8)
    service = RunService(RunStore(database), output_store=output_store,
                         **kwargs)
    app.state.db = database
    app.state.run_output_store = output_store
    app.state.node_state_store = node_state_store
    app.state.run_service = service
    try:
        yield service
    finally:
        await service.shutdown()
        database.close()
        for attribute in ("db", "run_service", "run_output_store",
                          "node_state_store"):
            if hasattr(app.state, attribute):
                delattr(app.state, attribute)


def _client() -> AsyncClient:
    """A SEPARATE connection to the same app — a different browser tab."""
    return AsyncClient(transport=ASGIWebSocketTransport(app=app),
                       base_url=_BASE_URL)


def _graph(middle: str | None = None,
           params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Start -> _TestSource -> [middle] -> Print."""
    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {"id": "src", "type": "_TestSource", "data": {"params": {"val": "hi"}}},
        {"id": "print", "type": "Print", "data": {"params": {"label": "out"}}},
    ]
    edges = [{"id": "et", "source": "start", "target": "src",
              "sourceHandle": "trigger", "type": "trigger"}]
    if middle is None:
        edges.append({"id": "e1", "source": "src", "target": "print",
                      "sourceHandle": "value", "targetHandle": "value"})
    else:
        nodes.insert(2, {"id": "mid", "type": middle,
                         "data": {"params": params or {}}})
        edges += [
            {"id": "e1", "source": "src", "target": "mid",
             "sourceHandle": "value", "targetHandle": "value"},
            {"id": "e2", "source": "mid", "target": "print",
             "sourceHandle": "value", "targetHandle": "value"},
        ]
    return {"nodes": nodes, "edges": edges}


# ── helpers ───────────────────────────────────────────────────────────────


async def _recv(ws) -> dict[str, Any]:
    return json.loads(await asyncio.wait_for(ws.receive_text(), _RECV_TIMEOUT))


async def _recv_until(ws, *types: str) -> list[dict[str, Any]]:
    """Drain frames until one of *types* arrives (inclusive)."""
    frames: list[dict[str, Any]] = []
    while True:
        frame = await _recv(ws)
        frames.append(frame)
        if frame["type"] in types:
            return frames


async def _execute(ws, graph: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """Send the legacy execute message; return the ``attached`` frame.

    Drains rather than asserting on the very next frame: a socket that was
    already watching a run keeps receiving ITS frames until the new attach
    replaces the pump, so the acknowledgement is not necessarily first. That
    interleaving is exactly why every frame carries ``run_id``.
    """
    await ws.send_text(json.dumps({"action": "execute", **graph, **extra}))
    return (await _recv_until(ws, "attached", "execution_error"))[-1]


async def _await_terminal(store: RunStore, run_id: str,
                          timeout: float = 20.0) -> RunRecord:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = await store.get_run(run_id)
        assert record is not None
        if record.finished_at is not None:
            return record
        await asyncio.sleep(0.02)
    raise AssertionError(f"run {run_id} did not finish within {timeout}s")


def _cursors(frames: list[dict[str, Any]]) -> list[int]:
    """Cursors of the log frames, ignoring transport-only frames."""
    return [f["cursor"] for f in frames if "cursor" in f]


def _assert_gapless(frames: list[dict[str, Any]], *, start_after: int) -> None:
    """Cursors are exactly ``start_after+1 .. N``, in order, each once.

    This single assertion is the whole replay/live contract: strictly
    increasing rules out reordering, consecutive rules out a gap, and
    "each once" rules out the duplicate that a naive replay-then-subscribe
    would produce for every event that landed during the replay.
    """
    cursors = _cursors(frames)
    assert cursors, "no log frames at all"
    expected = list(range(start_after + 1, start_after + 1 + len(cursors)))
    assert cursors == expected, f"cursors {cursors} != {expected}"


# ── 1. the legacy execute path ────────────────────────────────────────────


async def test_execute_submits_on_the_interactive_lane_and_attaches(run_env):
    """The pre-v2 message shape starts a real, persisted, interactive run."""
    async with _client() as client:
        async with aconnect_ws(_WS_PATH, client) as ws:
            attached = await _execute(ws, _graph())
            run_id = attached["run_id"]
            assert attached["cursor"] == 0
            frames = await _recv_until(ws, "execution_complete")

    assert frames[0]["type"] == "execution_start"
    assert frames[0]["run_id"] == run_id
    _assert_gapless(frames, start_after=0)

    record = await run_env.store.get_run(run_id)
    assert record is not None
    assert record.status == "succeeded"
    assert record.options["lane"] == LANE_INTERACTIVE


async def test_execute_persists_the_canvas_flags_and_passes_them_to_the_engine(
    run_env,
):
    """Canvas parity: every A1-A3 flag reaches the ExecutionContext.

    The regression this guards is the whole risk of #121 — #120's
    server-owned runs are deliberately isolated (``weights_persistent``
    False, no module store), and routing the canvas through them without an
    opt-in would silently break "keep training from where it left off",
    verbose step traces and gradient capture all at once.
    """
    async with _client() as client:
        async with aconnect_ws(_WS_PATH, client) as ws:
            attached = await _execute(
                ws, _graph("_WsContextProbe"),
                verbose_mode=True, graph_id="canvas-graph-1",
                weights_persistent=True, backward_mode=True,
                auto_backward=True, record_outputs=True,
            )
            run_id = attached["run_id"]
            await _recv_until(ws, "execution_complete")

    # ``record_outputs`` -> RunOutputStore is itself part of the teaching
    # loop (it is what the Inspector reads), so asserting through it covers
    # two parity claims at once.
    captured = await app.state.run_output_store.get(run_id, "mid", "value")
    assert captured is not None, "record_outputs did not reach the run"
    assert captured["verbose"] is True
    assert captured["graph_id"] == "canvas-graph-1"
    assert captured["weights_persistent"] is True
    assert captured["backward_mode"] is True
    assert captured["auto_backward"] is True
    assert captured["has_state_store"] is True, \
        "the canvas's NodeStateStore did not reach the run"
    # ONE id: the context's execution_id IS the run row's id, so captured
    # outputs stay findable through /api/execution/outputs/{run_id}.
    assert captured["execution_id"] == run_id

    record = await run_env.store.get_run(run_id)
    assert record is not None
    assert record.options["verbose"] is True
    assert record.options["graph_id"] == "canvas-graph-1"
    assert record.options["weights_persistent"] is True
    assert record.options["record_outputs"] is True


async def test_the_socket_cache_survives_across_executes(run_env):
    """Second Run on the same socket serves unchanged nodes from cache.

    The ExecutionCache is per SOCKET, exactly as it was when the socket
    owned the run — losing it would turn every Run click into a full
    re-execution of the graph.
    """
    async with _client() as client:
        async with aconnect_ws(_WS_PATH, client) as ws:
            await _execute(ws, _graph())
            first = await _recv_until(ws, "execution_complete")
            await _execute(ws, _graph())
            second = await _recv_until(ws, "execution_complete")

    def statuses(frames, node_id):
        return [f["status"] for f in frames
                if f["type"] == "node_status" and f.get("node_id") == node_id]

    assert "completed" in statuses(first, "src")
    assert "cached" in statuses(second, "src"), \
        "the socket's ExecutionCache did not survive the first run"


async def test_changed_nodes_forces_a_re_execution(run_env):
    """The dirty-node hint still bypasses the cache for the edited subtree."""
    async with _client() as client:
        async with aconnect_ws(_WS_PATH, client) as ws:
            await _execute(ws, _graph())
            await _recv_until(ws, "execution_complete")
            await _execute(ws, _graph(), changed_nodes=["src"])
            second = await _recv_until(ws, "execution_complete")

    src = [f["status"] for f in second
           if f["type"] == "node_status" and f.get("node_id") == "src"]
    assert "cached" not in src
    assert "completed" in src


async def test_clear_cache_still_works(run_env):
    async with _client() as client:
        async with aconnect_ws(_WS_PATH, client) as ws:
            await ws.send_text(json.dumps({"action": "clear_cache"}))
            assert (await _recv(ws))["type"] == "cache_cleared"


async def test_a_rejected_submit_reports_execution_error(run_env):
    """A malformed envelope reads like any other failure on the canvas."""
    async with _client() as client:
        async with aconnect_ws(_WS_PATH, client) as ws:
            await ws.send_text(json.dumps({"action": "execute", "nodes": []}))
            frame = await _recv(ws)
            assert frame["type"] == "execution_error"
            assert "empty" in frame["error"]


async def test_a_broken_submit_does_not_take_the_socket_down(run_env,
                                                             monkeypatch):
    """A sick database fails the RUN, not the connection.

    ``submit`` now runs inline on the receive loop, where the pre-v2 handler
    did its work inside a task and could not take the socket with it. An
    escaping exception would close the connection with a 1011 and leave the
    canvas spinning with no explanation.
    """
    async def exploding_create_run(*args, **kwargs):
        raise RuntimeError("disk is on fire")

    monkeypatch.setattr(run_env.store, "create_run", exploding_create_run)

    async with _client() as client:
        async with aconnect_ws(_WS_PATH, client) as ws:
            await ws.send_text(json.dumps({"action": "execute", **_graph()}))
            frame = await _recv(ws)
            assert frame["type"] == "execution_error"
            assert "disk is on fire" in frame["error"]
            # ...and the socket is still usable afterwards.
            await ws.send_text(json.dumps({"action": "clear_cache"}))
            assert (await _recv(ws))["type"] == "cache_cleared"


async def test_a_failing_action_does_not_take_the_socket_down(run_env,
                                                              monkeypatch):
    """The same guard for every other action, not just execute."""
    async def exploding_get_run(*args, **kwargs):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(run_env.store, "get_run", exploding_get_run)

    async with _client() as client:
        async with aconnect_ws(_WS_PATH, client) as ws:
            await ws.send_text(json.dumps({"action": "attach",
                                           "run_id": "whatever"}))
            frame = await _recv(ws)
            assert frame["type"] == "error"
            assert "store unavailable" in frame["error"]
            await ws.send_text(json.dumps({"action": "clear_cache"}))
            assert (await _recv(ws))["type"] == "cache_cleared"


# ── 2. attach / replay ────────────────────────────────────────────────────


async def test_attach_replays_the_whole_log_of_a_finished_run(run_env):
    """Reopen a tab after the run ended: the history is all still there."""
    async with _client() as client:
        async with aconnect_ws(_WS_PATH, client) as ws:
            attached = await _execute(ws, _graph())
            run_id = attached["run_id"]
            live = await _recv_until(ws, "execution_complete")

    await _await_terminal(run_env.store, run_id)

    async with _client() as client:
        async with aconnect_ws(_WS_PATH, client) as ws:
            await ws.send_text(json.dumps({"action": "attach",
                                           "run_id": run_id}))
            ack = await _recv(ws)
            assert ack == {"type": "attached", "run_id": run_id,
                           "cursor": 0, "status": "succeeded"}
            replayed = await _recv_until(ws, "execution_complete")

    _assert_gapless(replayed, start_after=0)
    # Byte-identical to what the live viewer saw: replayed history and live
    # progress are the same frames, which is what makes re-attach invisible.
    assert replayed == live


async def test_attach_from_a_cursor_sends_only_what_came_after(run_env):
    """A reconnect resumes instead of replaying history already rendered."""
    async with _client() as client:
        async with aconnect_ws(_WS_PATH, client) as ws:
            attached = await _execute(ws, _graph())
            run_id = attached["run_id"]
            full = await _recv_until(ws, "execution_complete")

    await _await_terminal(run_env.store, run_id)
    midpoint = full[len(full) // 2]["cursor"]

    async with _client() as client:
        async with aconnect_ws(_WS_PATH, client) as ws:
            await ws.send_text(json.dumps({"action": "attach",
                                           "run_id": run_id,
                                           "cursor": midpoint}))
            assert (await _recv(ws))["cursor"] == midpoint
            tail = await _recv_until(ws, "execution_complete")

    _assert_gapless(tail, start_after=midpoint)
    assert tail == [f for f in full if f["cursor"] > midpoint]


async def test_attaching_mid_run_crosses_the_replay_boundary_cleanly(run_env):
    """The gap/duplicate test: attach WHILE a chatty run is in flight."""
    async with _client() as client:
        async with aconnect_ws(_WS_PATH, client) as ws:
            attached = await _execute(ws, _graph("_WsChatty", {"count": 60}))
            run_id = attached["run_id"]
            # Let the run get well underway so the attach below has real
            # history to replay AND live events still arriving.
            await _recv_until(ws, "execution_start")
            await asyncio.sleep(0.05)

            async with _client() as watcher_client:
                async with aconnect_ws(_WS_PATH, watcher_client) as watcher:
                    await watcher.send_text(json.dumps({"action": "attach",
                                                        "run_id": run_id}))
                    assert (await _recv(watcher))["type"] == "attached"
                    seen = await _recv_until(watcher, "execution_complete")

            await _recv_until(ws, "execution_complete")

    _assert_gapless(seen, start_after=0)
    stored = await run_env.store.get_events(run_id)
    assert _cursors(seen) == [event.cursor for event in stored]


@pytest.mark.parametrize("run_env", [{"subscriber_queue_size": 1}],
                         indirect=True)
async def test_a_lagging_subscriber_loses_nothing(run_env):
    """Overflowing the fan-out queue must not lose or duplicate an event.

    A depth-1 subscription drops nearly everything: the run emits ~130
    events and the pump can hold one at a time, so the store re-read is
    doing almost all the work here. The delivered stream must STILL be the
    complete log, once each, in order — the drop-tolerance contract
    ``RunSubscription`` documents, exercised end to end.
    """
    async with _client() as client:
        async with aconnect_ws(_WS_PATH, client) as ws:
            attached = await _execute(ws, _graph("_WsChatty", {"count": 120}))
            run_id = attached["run_id"]
            frames = await _recv_until(ws, "execution_complete")

    _assert_gapless(frames, start_after=0)
    stored = await run_env.store.get_events(run_id)
    assert len(stored) > 120, "the run was not chatty enough to overflow"
    assert _cursors(frames) == [event.cursor for event in stored]


async def test_attach_rejects_a_missing_or_unknown_run(run_env):
    async with _client() as client:
        async with aconnect_ws(_WS_PATH, client) as ws:
            await ws.send_text(json.dumps({"action": "attach"}))
            assert "run_id" in (await _recv(ws))["error"]

            await ws.send_text(json.dumps({"action": "attach",
                                           "run_id": "nope"}))
            assert "not found" in (await _recv(ws))["error"]

            await ws.send_text(json.dumps({"action": "attach",
                                           "run_id": "x", "cursor": -1}))
            assert "cursor" in (await _recv(ws))["error"]


async def test_attach_refuses_a_cursor_past_the_end_of_the_log(run_env):
    """Accepting one would attach successfully and deliver NOTHING.

    Indistinguishable from a healthy attach to a quiet run — the worst
    failure mode for a client whose only job is following along. An
    off-by-one, or a cursor kept from a different run, should be loud.
    """
    async with _client() as client:
        async with aconnect_ws(_WS_PATH, client) as ws:
            attached = await _execute(ws, _graph())
            run_id = attached["run_id"]
            frames = await _recv_until(ws, "execution_complete")
    latest = frames[-1]["cursor"]

    async with _client() as client:
        async with aconnect_ws(_WS_PATH, client) as ws:
            # The last cursor is legal: nothing to replay, then live-tail.
            await ws.send_text(json.dumps({"action": "attach",
                                           "run_id": run_id,
                                           "cursor": latest}))
            assert (await _recv(ws))["type"] == "attached"

            await ws.send_text(json.dumps({"action": "attach",
                                           "run_id": run_id,
                                           "cursor": latest + 1}))
            frame = await _recv(ws)
            assert frame["type"] == "error"
            assert f"latest cursor {latest}" in frame["error"]


async def test_a_broken_store_tells_the_client_instead_of_stalling(
    run_env, monkeypatch,
):
    """A pump that dies mid-replay must not leave the client on "Running".

    The socket is still open and nothing more is coming; without a frame
    saying so, the canvas waits forever for a run that is, from its point of
    view, silently gone.
    """
    async with _client() as client:
        async with aconnect_ws(_WS_PATH, client) as ws:
            attached = await _execute(ws, _graph())
            run_id = attached["run_id"]
            await _recv_until(ws, "execution_complete")
    await _await_terminal(run_env.store, run_id)

    async def exploding_get_events(*args, **kwargs):
        raise RuntimeError("events table is gone")

    monkeypatch.setattr(run_env.store, "get_events", exploding_get_events)

    async with _client() as client:
        async with aconnect_ws(_WS_PATH, client) as ws:
            await ws.send_text(json.dumps({"action": "attach",
                                           "run_id": run_id}))
            assert (await _recv(ws))["type"] == "attached"
            frame = await _recv(ws)
            assert frame["type"] == "error"
            assert "event stream" in frame["error"]


async def test_a_malformed_message_does_not_kill_the_socket(run_env):
    async with _client() as client:
        async with aconnect_ws(_WS_PATH, client) as ws:
            await ws.send_text("{not json")
            assert (await _recv(ws))["type"] == "error"
            await ws.send_text(json.dumps(["not", "an", "object"]))
            assert (await _recv(ws))["type"] == "error"
            # ...and the socket is still usable.
            await ws.send_text(json.dumps({"action": "clear_cache"}))
            assert (await _recv(ws))["type"] == "cache_cleared"


# ── 3. detach never cancels ───────────────────────────────────────────────


async def test_closing_the_socket_never_cancels_the_run(run_env):
    """THE headline behaviour of the wave.

    Before #120/#121 this exact sequence killed the run: the handler owned
    the task and cancelled it on WebSocketDisconnect. Now the tab is gone
    and the training is not.
    """
    async with _client() as client:
        async with aconnect_ws(_WS_PATH, client) as ws:
            attached = await _execute(ws, _graph("_WsSlow", {"seconds": 0.6}))
            run_id = attached["run_id"]
            await _recv_until(ws, "execution_start")
        # socket closed here, mid-run

    assert run_env.is_active(run_id), \
        "the run was already over — this test proves nothing"
    record = await _await_terminal(run_env.store, run_id)
    assert record.status == "succeeded"

    # ...and a new tab can pick the whole thing up afterwards.
    async with _client() as client:
        async with aconnect_ws(_WS_PATH, client) as ws:
            await ws.send_text(json.dumps({"action": "attach",
                                           "run_id": run_id}))
            assert (await _recv(ws))["type"] == "attached"
            replayed = await _recv_until(ws, "execution_complete")
    _assert_gapless(replayed, start_after=0)


async def test_detach_unsubscribes_without_cancelling(run_env):
    """``detach`` stops the frames, not the run."""
    async with _client() as client:
        async with aconnect_ws(_WS_PATH, client) as ws:
            attached = await _execute(ws, _graph("_WsSlow", {"seconds": 0.6}))
            run_id = attached["run_id"]
            await _recv_until(ws, "execution_start")

            await ws.send_text(json.dumps({"action": "detach"}))
            frame = (await _recv_until(ws, "detached"))[-1]
            assert frame == {"type": "detached", "run_id": run_id}

            assert run_env.is_active(run_id), "detach cancelled the run"
            record = await _await_terminal(run_env.store, run_id)
            assert record.status == "succeeded"

            # Nothing further arrives on this socket for the detached run,
            # but the socket itself is alive.
            await ws.send_text(json.dumps({"action": "clear_cache"}))
            assert (await _recv(ws))["type"] == "cache_cleared"


async def test_a_second_execute_detaches_rather_than_cancels(run_env):
    """Submitting again does not kill the run this socket was watching."""
    async with _client() as client:
        async with aconnect_ws(_WS_PATH, client) as ws:
            first = await _execute(ws, _graph("_WsSlow", {"seconds": 0.6}))
            await _recv_until(ws, "execution_start")
            second = await _execute(ws, _graph())
            assert second["run_id"] != first["run_id"]
            await _recv_until(ws, "execution_complete")

            record = await _await_terminal(run_env.store, first["run_id"])
            assert record.status == "succeeded", \
                "the first run was cancelled by the second execute"


# ── 4. cancel is the only stop ────────────────────────────────────────────


async def test_cancel_action_stops_the_attached_run(run_env):
    async with _client() as client:
        async with aconnect_ws(_WS_PATH, client) as ws:
            attached = await _execute(ws, _graph("_WsSlow", {"seconds": 0.6}))
            run_id = attached["run_id"]
            await _recv_until(ws, "execution_start")

            await ws.send_text(json.dumps({"action": "cancel",
                                           "run_id": run_id}))
            frames = await _recv_until(ws, "cancel_ack")
            assert frames[-1] == {"type": "cancel_ack", "run_id": run_id,
                                  "status": "running", "cancelled": True}
            stopped = (await _recv_until(ws, "execution_stopped"))[-1]
            assert stopped["reason"] == "cancelled"

    record = await _await_terminal(run_env.store, run_id)
    assert record.status == "cancelled"


async def test_cancel_defaults_to_the_attached_run(run_env):
    async with _client() as client:
        async with aconnect_ws(_WS_PATH, client) as ws:
            attached = await _execute(ws, _graph("_WsSlow", {"seconds": 0.6}))
            await _recv_until(ws, "execution_start")
            await ws.send_text(json.dumps({"action": "cancel"}))
            ack = (await _recv_until(ws, "cancel_ack"))[-1]
            assert ack["run_id"] == attached["run_id"]
            assert ack["cancelled"] is True

    # Cancellation is COOPERATIVE: the sleeping node finishes first, so the
    # row turns terminal after the acknowledgement, not with it.
    record = await _await_terminal(run_env.store, attached["run_id"])
    assert record.status == "cancelled"


async def test_the_legacy_stop_action_still_cancels(run_env):
    """A prebuilt frontend from before v2 sends ``stop``; it must work."""
    async with _client() as client:
        async with aconnect_ws(_WS_PATH, client) as ws:
            attached = await _execute(ws, _graph("_WsSlow", {"seconds": 0.6}))
            await _recv_until(ws, "execution_start")
            await ws.send_text(json.dumps({"action": "stop"}))
            ack = (await _recv_until(ws, "cancel_ack"))[-1]
            assert ack["cancelled"] is True

    record = await _await_terminal(run_env.store, attached["run_id"])
    assert record.status == "cancelled"


async def test_cancel_with_nothing_attached_unsticks_the_ui(run_env):
    """Stop on an idle canvas answers instead of leaving it spinning."""
    async with _client() as client:
        async with aconnect_ws(_WS_PATH, client) as ws:
            await ws.send_text(json.dumps({"action": "cancel"}))
            frame = await _recv(ws)
            assert frame == {"type": "execution_stopped",
                             "reason": "not_running"}

            await ws.send_text(json.dumps({"action": "cancel",
                                           "run_id": "ghost"}))
            frame = await _recv(ws)
            assert frame["type"] == "execution_stopped"
            assert frame["reason"] == "not_running"


async def test_cancelling_a_finished_run_invents_no_second_terminal_frame(
    run_env,
):
    """A late Stop must not rewrite "completed" as "cancelled" in the UI."""
    async with _client() as client:
        async with aconnect_ws(_WS_PATH, client) as ws:
            attached = await _execute(ws, _graph())
            run_id = attached["run_id"]
            await _recv_until(ws, "execution_complete")

            await ws.send_text(json.dumps({"action": "cancel",
                                           "run_id": run_id}))
            frame = await _recv(ws)
            assert frame["type"] == "cancel_ack"
            assert frame["cancelled"] is False
            assert frame["status"] == "succeeded"

    record = await run_env.store.get_run(run_id)
    assert record is not None and record.status == "succeeded"
