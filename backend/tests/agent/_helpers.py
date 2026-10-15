"""Shared pieces for the agent tests (#771).

The ``*_child`` functions are run-subprocess entry points with
``child.main``'s signature, handed to ``Runner(child_entry=...)``. They live
at module level in a module that imports nothing heavy, because a *spawn*
child imports this module to find them: the cheap ones never load torch,
which is what keeps the subprocess tests fast on every OS.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any

from app.agent import child
from app.core import worker_protocol as wp

#: Printed by ``crash_child`` before it dies, to find in ``stderr_tail``.
CRASH_MARKER = "agent-test: about to crash"


def node(node_id: str, node_type: str, **params: Any) -> dict[str, Any]:
    return {"id": node_id, "type": node_type, "data": {"params": params}}


def wire(source: str, source_handle: str, target: str,
         target_handle: str) -> dict[str, Any]:
    return {"id": f"e-{source}-{target}", "source": source,
            "sourceHandle": source_handle, "target": target,
            "targetHandle": target_handle}


def trigger(source: str, target: str) -> dict[str, Any]:
    return {"id": f"t-{source}-{target}", "source": source, "target": target,
            "sourceHandle": "trigger", "type": "trigger"}


def tensor_graph() -> dict[str, Any]:
    """Start -> TensorCreate -> ScalarMultiply -> Print (#765's tensor
    chain): seeded, it runs serially, so its events come in one order."""
    return {
        "nodes": [
            node("start", "Start"),
            node("x", "TensorCreate", shape="2,3", fill="ones", value=0,
                 requires_grad=False),
            node("scale", "ScalarMultiply", scalar=2.5),
            node("show", "Print", label="tensor"),
        ],
        "edges": [
            trigger("start", "x"),
            wire("x", "tensor", "scale", "tensor"),
            wire("scale", "tensor", "show", "value"),
        ],
    }


TENSOR_OPTIONS = {"seed": 0}


def event_bytes(event_type: str, payload: Any) -> str:
    """One event as comparable bytes: the JSON ``encode_frame`` writes."""
    return json.dumps({"event_type": event_type, "payload": payload},
                      allow_nan=True, separators=(",", ":"),
                      ensure_ascii=False)


async def in_process_events(graph: dict[str, Any],
                            options: dict[str, Any]) -> list[str]:
    """The events ``InProcessExecutor`` emits for *graph*, as bytes."""
    from app.agent.child import build_context
    from app.core.executors import (
        ExecutionRequest,
        InProcessExecutor,
        RunSinks,
    )

    emitted: list[str] = []

    async def emit(event_type: str, payload: Any) -> None:
        emitted.append(event_bytes(event_type, payload))

    async def ignore(*_: Any) -> None:
        return None

    outcome = await InProcessExecutor().run(ExecutionRequest(
        run_id="parity", graph=child._normalised_graph(graph),
        options=dict(options), context=build_context("parity", options, "cpu"),
        session=None, output_store=None, record_outputs=False,
        sinks=RunSinks(emit=emit, record_metrics=ignore,
                       record_artifact=ignore, warn=ignore),
        explicit_metric_nodes=set()))
    assert outcome.status == "succeeded", outcome
    return emitted


class FrameSink:
    """A ``Runner.send`` that keeps every frame and wakes waiters."""

    def __init__(self) -> None:
        self.frames: list[Any] = []
        self._changed = asyncio.Event()

    async def __call__(self, frame: Any) -> None:
        self.frames.append(frame)
        self._changed.set()

    def of(self, run_id: str) -> list[Any]:
        return [f for f in self.frames if getattr(f, "run_id", None) == run_id]

    def finished(self, run_id: str) -> Any:
        for frame in self.of(run_id):
            if isinstance(frame, wp.RunFinished):
                return frame
        return None

    async def wait_for(self, predicate: Any, timeout: float = 60.0) -> Any:
        deadline = time.monotonic() + timeout
        while True:
            for frame in self.frames:
                if predicate(frame):
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


# ── run-subprocess entry points ─────────────────────────────────────────


def _running(events_out: Any, node_id: str) -> None:
    events_out.send((child.MSG_EVENT, "node_status",
                     {"node_id": node_id, "status": "running"}))


def crash_child(spec: dict[str, Any], events_out: Any, control_in: Any) -> None:
    """Start a node, say something on stderr, die with exit code 3."""
    child._redirect_output(spec.get("log_path"))
    _running(events_out, "boom")
    print(CRASH_MARKER, file=sys.stderr, flush=True)
    os._exit(3)


def crash_on_cancel_child(spec: dict[str, Any], events_out: Any,
                          control_in: Any) -> None:
    """Start a node; on ``cancel``, crash with exit code 7 instead of stopping."""
    child._redirect_output(spec.get("log_path"))
    _running(events_out, "fragile")
    control_in.recv()
    print(CRASH_MARKER, file=sys.stderr, flush=True)
    os._exit(7)


def quick_child(spec: dict[str, Any], events_out: Any, control_in: Any) -> None:
    """Report success at once."""
    events_out.send((child.MSG_FINISHED, {"status": "succeeded",
                                          "error": None, "peak_memory": None}))


def lingering_child(spec: dict[str, Any], events_out: Any,
                    control_in: Any) -> None:
    """Report success, then stay alive (pipe open) until terminated."""
    quick_child(spec, events_out, control_in)
    while True:
        time.sleep(0.05)


def deaf_child(spec: dict[str, Any], events_out: Any, control_in: Any) -> None:
    """Start a node and never stop: ignores ``cancel`` (terminate path)."""
    _running(events_out, "deaf")
    while True:
        time.sleep(0.05)


def env_child(spec: dict[str, Any], events_out: Any, control_in: Any) -> None:
    """Report the environment and device the run would execute with."""
    child.apply_environment(spec.get("env") or {})
    names = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
             "ROCR_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CODEFYUI_DB_PATH",
             "CODEFYUI_AGENT_TOKEN")
    events_out.send((child.MSG_EVENT, "env", {
        "env": {name: os.environ.get(name) for name in names},
        "device": spec.get("device"),
        "torch_loaded": "torch" in sys.modules,
    }))
    quick_child(spec, events_out, control_in)


def slow_graph_child(spec: dict[str, Any], events_out: Any,
                     control_in: Any) -> None:
    """``child.main`` with one registered node that runs until cancelled.

    A real engine run, cancelled the real way (``should_stop()``), with no
    node discovery: the registry holds only this node.
    """
    child.apply_environment(spec.get("env") or {})
    child._redirect_output(spec.get("log_path"))
    from app.core.node_base import BaseNode, DataType, PortDefinition
    from app.core.node_registry import registry

    class AgentSlowNode(BaseNode):
        NODE_NAME = "_AgentSlow"
        CATEGORY = "Test"

        @classmethod
        def define_inputs(cls) -> list:
            return []

        @classmethod
        def define_outputs(cls) -> list:
            return [PortDefinition(name="value", data_type=DataType.ANY)]

        def execute(self, inputs: dict, params: dict, *,
                    context: Any = None) -> dict:
            while context is not None and not context.should_stop():
                time.sleep(0.02)
            return {"value": 0}

    from app.nodes.control.start_node import StartNode

    registry.register(StartNode)
    registry.register(AgentSlowNode)
    child.run_assignment(spec, child._safe_sender(events_out), control_in,
                         discover=False)


def slow_graph() -> dict[str, Any]:
    return {"nodes": [node("start", "Start"), node("slow", "_AgentSlow")],
            "edges": [trigger("start", "slow")]}
