"""A device the run asked for and does not get is a run warning (#839).

``resolve_device`` still falls back (exported scripts and saved graphs rely
on it); ``core.device_fallback`` only reports it, and ``InProcessExecutor``
sends each report as a ``run_warning`` of kind ``device_fallback`` before
the graph executes. A run that meets no fallback emits nothing new.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.core import device_fallback
from app.core.device_fallback import (
    WARNING_DEVICE_FALLBACK,
    device_fallbacks,
    substituted,
)
from app.core.execution_context import ExecutionContext


def _graph(*nodes: dict[str, Any]) -> dict[str, Any]:
    return {"nodes": list(nodes), "edges": [], "presets": [],
            "subgraphs": []}


def _node(node_id: str, node_type: str, **params: Any) -> dict[str, Any]:
    return {"id": node_id, "type": node_type, "data": {"params": params}}


@pytest.fixture
def cpu_only(monkeypatch):
    """A box with no accelerator: ``cuda``/``mps`` resolve to ``cpu``."""
    def resolve(requested: str) -> str:
        kind = requested.split(":")[0]
        return "cpu" if kind in ("cuda", "mps") else requested

    monkeypatch.setattr(device_fallback, "resolve_device", resolve)


@pytest.mark.parametrize("requested,resolved,expected", [
    ("cuda", "cpu", True),
    ("cuda:0", "cpu", True),
    ("mps", "cpu", True),
    ("cuda:3", "cuda:0", True),
    ("cuda", "cuda:1", False),
    ("cuda", "cuda", False),
    ("cuda:1", "cuda:1", False),
    ("mps", "mps:0", False),
    ("mps:0", "mps", False),
    ("mps:1", "mps", True),
    ("cpu", "cpu", False),
    (" CUDA:0 ", "cuda:0", False),
])
def test_substituted(requested, resolved, expected):
    assert substituted(requested, resolved) is expected


def test_the_runs_own_device_falls_back(cpu_only):
    found = device_fallbacks(_graph(), "cuda", "cpu")
    assert found == [{
        "detail": ("The run asked for device cuda, which this server does "
                   "not have; it runs on cpu."),
        "node_id": None}]


@pytest.mark.parametrize("requested", ["auto", "cpu", "", None, "cuda"])
def test_a_run_on_the_device_it_asked_for_is_not_reported(requested):
    run_device = "cuda:0" if requested == "cuda" else "cpu"
    assert device_fallbacks(_graph(), requested, run_device) == []


def test_check_run_false_leaves_the_runs_device_alone(cpu_only):
    """A worker's run subprocess sees its cards renumbered (``cuda:1`` is
    its ``cuda:0``); submit checked the set against the worker."""
    assert device_fallbacks(_graph(), "cuda:1", "cuda:0",
                            check_run=False) == []


def test_a_pinned_node_falls_back(cpu_only):
    graph = _graph(_node("train", "TrainingLoop", device="cuda:0"))
    assert device_fallbacks(graph, "cpu", "cpu", where="this worker") == [{
        "detail": ("Node train (TrainingLoop) asks for device cuda:0, which "
                   "this worker does not have; it runs on cpu."),
        "node_id": "train"}]


def test_auto_and_non_device_values_are_not_reported(cpu_only):
    graph = _graph(
        _node("a", "TrainingLoop", device="auto"),
        _node("b", "TrainingLoop", device=""),
        _node("c", "TrainingLoop", device="/dev/ttyUSB0"),
        _node("d", "TrainingLoop", device=3),
        _node("e", "TrainingLoop", device="cpu"),
        # Not a declared param: a stray key does not steer anything.
        _node("f", "Print", device="cuda"),
        _node("g", "NoSuchNodeType", device="cuda"),
        {"id": "h", "type": "TrainingLoop", "data": "broken"},
    )
    assert device_fallbacks(graph, "cpu", "cpu") == []


def test_a_node_whose_params_cannot_be_read_is_skipped(monkeypatch,
                                                         cpu_only):
    class Broken:
        @classmethod
        def define_params(cls):
            raise RuntimeError("broken plugin")

    from app.core.node_registry import registry

    real_get = registry.get
    monkeypatch.setattr(registry, "get", lambda name: Broken
                        if name == "BrokenNode" else real_get(name))
    graph = _graph(_node("x", "BrokenNode", device="cuda"))
    assert device_fallbacks(graph, "cpu", "cpu") == []


def test_one_resolution_per_device_string(monkeypatch):
    calls: list[str] = []

    def resolve(requested: str) -> str:
        calls.append(requested)
        return "cpu"

    monkeypatch.setattr(device_fallback, "resolve_device", resolve)
    graph = _graph(_node("a", "TrainingLoop", device="cuda"),
                   _node("b", "TrainingLoop", device="cuda"))
    found = device_fallbacks(graph, "cpu", "cpu")
    assert [entry["node_id"] for entry in found] == ["a", "b"]
    assert calls == ["cuda"]


# ── InProcessExecutor sends them as run_warning ──────────────────────────


async def _execute(graph: dict[str, Any], options: dict[str, Any],
                   device: str, *, exclusive: bool = False
                   ) -> list[tuple[str, Any]]:
    from app.core.executors import ExecutionRequest, InProcessExecutor, RunSinks

    seen: list[tuple[str, Any]] = []

    async def emit(event_type, payload):
        seen.append((event_type, payload))

    async def warn(signal):
        seen.append(("run_warning", signal))

    async def ignore(*_: Any) -> None:
        return None

    outcome = await InProcessExecutor(exclusive_process=exclusive).run(
        ExecutionRequest(
            run_id="r", graph=graph, options=options,
            context=ExecutionContext(device=device), session=None,
            output_store=None, record_outputs=False,
            sinks=RunSinks(emit=emit, record_metrics=ignore,
                           record_artifact=ignore, warn=warn),
            explicit_metric_nodes=set()))
    # An empty graph fails ("no entry points"): what matters here is what
    # was said before the engine looked at it.
    assert outcome.status == "failed", outcome
    return seen


async def test_the_executor_warns_before_the_graph_runs(cpu_only):
    seen = await _execute(_graph(), {"device": "cuda"}, "cpu")
    assert seen[0][0] == "run_warning"
    signal = seen[0][1]
    assert signal.kind == WARNING_DEVICE_FALLBACK
    assert signal.node_id is None
    assert "asked for device cuda" in signal.detail
    assert sum(1 for kind, _ in seen if kind == "run_warning") == 1


async def test_a_run_without_a_fallback_emits_no_warning():
    seen = await _execute(_graph(), {"device": "cpu"}, "cpu")
    assert all(kind != "run_warning" for kind, _ in seen)


async def test_a_worker_subprocess_names_the_worker(cpu_only):
    graph = _graph(_node("train", "TrainingLoop", device="mps"))
    from app.core.executors import InProcessExecutor

    executor = InProcessExecutor(exclusive_process=True)
    warned: list[Any] = []

    class _Sinks:
        async def warn(self, signal):
            warned.append(signal)

    class _Request:
        run_id = "r"
        options = {"device": "cuda:1"}
        context = ExecutionContext(device="cuda:0")
        sinks = _Sinks()

    _Request.graph = graph
    await executor._warn_device_fallbacks(_Request())
    # The run's own device is not judged there (renumbered); the node is.
    assert [s.node_id for s in warned] == ["train"]
    assert "this worker does not have" in warned[0].detail


async def test_a_failing_check_never_fails_the_run(monkeypatch):
    def boom(*_: Any, **__: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(device_fallback, "device_fallbacks", boom)
    seen = await _execute(_graph(), {"device": "cuda"}, "cpu")
    assert all(kind != "run_warning" for kind, _ in seen)
