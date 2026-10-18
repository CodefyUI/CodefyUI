"""A device the run asked for and does not get is a run warning (#839).

``resolve_device`` still falls back (exported scripts and saved graphs rely
on it); ``core.device_fallback`` only reports it, and ``InProcessExecutor``
sends each report as a ``run_warning`` of kind ``device_fallback`` from
``execute_graph``'s ``on_expanded``, before any node runs. A run that meets
no fallback emits nothing new.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from app.core import device_fallback, device_utils
from app.core.device_fallback import (
    WARNING_DEVICE_FALLBACK,
    device_fallbacks,
    substituted,
)
from app.core.execution_context import ExecutionContext


def _node(node_id: str, node_type: str, **params: Any) -> dict[str, Any]:
    return {"id": node_id, "type": node_type, "data": {"params": params}}


def _graph(*nodes: dict[str, Any], edges=None) -> dict[str, Any]:
    return {"nodes": list(nodes), "edges": list(edges or []), "presets": [],
            "subgraphs": []}


def _trigger(source: str, target: str) -> dict[str, Any]:
    return {"id": f"t-{source}-{target}", "source": source, "target": target,
            "sourceHandle": "trigger", "type": "trigger"}


@pytest.fixture
def cpu_only(monkeypatch):
    """A box with no accelerator: ``cuda``/``mps`` resolve to ``cpu``."""
    def resolve(requested: str) -> str:
        kind = requested.split(":")[0]
        return "cpu" if kind in ("cuda", "mps") else requested

    monkeypatch.setattr(device_utils, "resolve_device", resolve)


@pytest.fixture
def one_card(monkeypatch):
    """A run subprocess that sees one CUDA card: every CUDA name is cuda:0."""
    def resolve(requested: str) -> str:
        kind = requested.split(":")[0].strip()
        if kind == "cuda":
            return "cuda:0"
        return "cpu" if kind == "mps" else requested

    monkeypatch.setattr(device_utils, "resolve_device", resolve)


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
    # Malformed spellings never come back as written (review item 9).
    ("cuda:abc", "cuda:0", True),
    ("cuda: 0", "cuda:0", True),
    ("cuda:", "cuda:0", True),
])
def test_substituted(requested, resolved, expected):
    assert substituted(requested, resolved) is expected


def test_the_runs_own_device_falls_back():
    assert device_fallbacks([], "cuda", "cpu") == [{
        "detail": ("The run asked for device cuda, which this server does "
                   "not have; it runs on cpu."),
        "node_id": None, "code": "device_fallback_run",
        "params": {"requested": "cuda", "device": "cpu", "where": "server"}}]


@pytest.mark.parametrize("requested", ["auto", "cpu", "", None, "cuda"])
def test_a_run_on_the_device_it_asked_for_is_not_reported(requested):
    run_device = "cuda:0" if requested == "cuda" else "cpu"
    assert device_fallbacks([], requested, run_device) == []


def test_check_run_false_leaves_the_runs_device_alone():
    assert device_fallbacks([], "cuda:1", "cuda:0", check_run=False) == []


def test_a_pinned_node_falls_back_on_the_server(cpu_only):
    nodes = [_node("train", "TrainingLoop", device="cuda:0")]
    assert device_fallbacks(nodes, "cpu", "cpu") == [{
        "detail": ("Node train (TrainingLoop) is set to device cuda:0, which "
                   "this server does not have; when it executes, it runs on "
                   "cpu."),
        "node_id": "train", "code": "device_fallback_node",
        "params": {"node": "train", "type": "TrainingLoop",
                   "requested": "cuda:0", "device": "cpu",
                   "where": "server"}}]


def test_a_malformed_node_device_is_reported(one_card):
    """Review item 9: ``cuda:abc`` lands on cuda:0 like any wrong name."""
    nodes = [_node("a", "TrainingLoop", device="cuda:abc"),
             _node("b", "TrainingLoop", device="cuda: 0")]
    found = device_fallbacks(nodes, "cpu", "cpu")
    assert [entry["node_id"] for entry in found] == ["a", "b"]
    assert "is set to device cuda:abc" in found[0]["detail"]


def test_a_spelling_the_engine_ignores_is_not_reported(one_card):
    """Second review, item 5: the check asks ``node_target_device`` itself,
    so it agrees with the engine. ``cuda : 0`` is not a device to it (the
    run's device stands), so nothing falls back."""
    nodes = [_node("a", "TrainingLoop", device="cuda : 0")]
    assert device_fallbacks(nodes, "cpu", "cpu") == []


# ── in-run names on a worker (review item 1) ─────────────────────────────


def test_a_worker_node_is_described_in_the_runs_card_names(one_card):
    """Set ``[cuda:1]``: the run's only card is cuda:0 (the worker's cuda:1).
    A node pinned to cuda:1 names a card the run does not have."""
    nodes = [_node("train", "TrainingLoop", device="cuda:1")]
    found = device_fallbacks(nodes, "cuda:1", "cuda:0", where="this worker",
                             check_run=False, run_cards={"cuda:0": "cuda:1"})
    assert found == [{
        "detail": ("Node train (TrainingLoop) is set to device cuda:1; this "
                   "run has 1 card (cuda:0 = the worker's cuda:1), so when "
                   "it executes, it runs on cuda:0."),
        "node_id": "train", "code": "device_fallback_node_in_run",
        "params": {"node": "train", "type": "TrainingLoop",
                   "requested": "cuda:1", "device": "cuda:0",
                   "cards": [["cuda:0", "cuda:1"]]}}]


def test_a_worker_node_on_an_in_run_name_that_exists_is_quiet(one_card):
    nodes = [_node("train", "TrainingLoop", device="cuda:0")]
    assert device_fallbacks(nodes, "cuda:1", "cuda:0", where="this worker",
                            check_run=False,
                            run_cards={"cuda:0": "cuda:1"}) == []


def test_several_cards_are_all_named(monkeypatch):
    monkeypatch.setattr(device_utils, "resolve_device", lambda r: "cuda:0")
    nodes = [_node("train", "TrainingLoop", device="cuda:5")]
    found = device_fallbacks(
        nodes, "cuda:2", "cuda:0", where="this worker", check_run=False,
        run_cards={"cuda:0": "cuda:2", "cuda:1": "cuda:0"})
    assert ("this run has 2 cards (cuda:0 = the worker's cuda:2, cuda:1 = "
            "the worker's cuda:0), so when it executes, it runs on "
            "cuda:0.") in found[0]["detail"]


def test_a_cpu_run_on_a_worker_sees_no_card(cpu_only):
    nodes = [_node("train", "TrainingLoop", device="cuda:0"),
             _node("eval", "TrainingLoop", device="mps")]
    found = device_fallbacks(nodes, "cpu", "cpu", where="this worker",
                             check_run=False, run_cards={})
    assert [entry["detail"] for entry in found] == [
        ("Node train (TrainingLoop) is set to device cuda:0; this run has no "
         "CUDA card, so when it executes, it runs on cpu."),
        ("Node eval (TrainingLoop) is set to device mps, which this worker "
         "does not have; when it executes, it runs on cpu.")]


# ── guards ────────────────────────────────────────────────────────────────


def test_auto_and_non_device_values_are_not_reported(cpu_only):
    nodes = [
        _node("a", "TrainingLoop", device="auto"),
        _node("b", "TrainingLoop", device=""),
        _node("c", "TrainingLoop", device="/dev/ttyUSB0"),
        _node("d", "TrainingLoop", device=3),
        _node("e", "TrainingLoop", device="cpu"),
        _node("f", "Print", device="cuda"),
        _node("g", "NoSuchNodeType", device="cuda"),
        {"id": "h", "type": "TrainingLoop", "data": "broken"},
    ]
    assert device_fallbacks(nodes, "cpu", "cpu") == []


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
    assert device_fallbacks([_node("x", "BrokenNode", device="cuda")],
                            "cpu", "cpu") == []


def test_one_resolution_per_device_string(monkeypatch):
    calls: list[str] = []

    def resolve(requested: str) -> str:
        calls.append(requested)
        return "cpu"

    monkeypatch.setattr(device_utils, "resolve_device", resolve)
    nodes = [_node("a", "TrainingLoop", device="cuda"),
             _node("b", "TrainingLoop", device="cuda")]
    found = device_fallbacks(nodes, "cpu", "cpu")
    assert [entry["node_id"] for entry in found] == ["a", "b"]
    assert calls == ["cuda"]


def test_the_check_does_not_log_the_fallback_a_second_time(caplog):
    """Review item 6: the node logs its fallback; the pre-check must not."""
    caplog.set_level(logging.WARNING, logger=device_utils.logger.name)
    device_fallbacks([_node("a", "TrainingLoop", device="cuda:abc")],
                     "cpu", "cpu")
    assert [r for r in caplog.records
            if r.name == device_utils.logger.name] == []
    # The node's own resolution still logs.
    device_utils.resolve_device("cuda:abc")
    assert [r for r in caplog.records if r.name == device_utils.logger.name]


# ── InProcessExecutor sends them from on_expanded ────────────────────────


async def _execute(graph: dict[str, Any], options: dict[str, Any],
                   device: str, *, exclusive: bool = False,
                   worker_run: bool = False,
                   run_cards=None,
                   warning_sink_fails: bool = False) -> list[tuple[str, Any]]:
    from app.core.executors import ExecutionRequest, InProcessExecutor, RunSinks

    seen: list[tuple[str, Any]] = []

    async def emit(event_type, payload):
        if warning_sink_fails and event_type == "run_warning":
            raise RuntimeError("sink down")
        seen.append((event_type, payload))

    async def ignore(*_: Any) -> None:
        return None

    await InProcessExecutor(exclusive_process=exclusive,
                            worker_run=worker_run,
                            run_cards=run_cards).run(
        ExecutionRequest(
            run_id="r", graph=graph, options=options,
            context=ExecutionContext(device=device), session=None,
            output_store=None, record_outputs=False,
            sinks=RunSinks(emit=emit, record_metrics=ignore,
                           record_artifact=ignore, warn=ignore),
            explicit_metric_nodes=set()))
    return seen


@pytest.fixture
def probe():
    from app.core.node_base import (
        BaseNode,
        DataType,
        ParamDefinition,
        ParamType,
        PortDefinition,
    )
    from app.core.node_registry import registry
    from app.nodes.control.start_node import StartNode

    class DeviceProbeNode(BaseNode):
        NODE_NAME = "_DeviceProbe"
        CATEGORY = "Test"

        @classmethod
        def define_inputs(cls) -> list:
            return []

        @classmethod
        def define_outputs(cls) -> list:
            return [PortDefinition(name="value", data_type=DataType.ANY)]

        @classmethod
        def define_params(cls) -> list:
            return [ParamDefinition(
                name="device", param_type=ParamType.SELECT, default="auto",
                options=list(device_utils.DEVICE_PARAM_OPTIONS))]

        def execute(self, inputs: dict, params: dict, *,
                    context: Any = None) -> dict:
            return {"value": 0}

    registry.register(StartNode)
    registry.register(DeviceProbeNode)
    yield
    registry._nodes.pop("_DeviceProbe", None)


def _probe_graph(*devices: str) -> dict[str, Any]:
    nodes = [_node("start", "Start")]
    edges = []
    for i, device in enumerate(devices):
        node_id = "train" if i == 0 else f"train{i}"
        nodes.append(_node(node_id, "_DeviceProbe", device=device))
        edges.append(_trigger("start", node_id))
    return _graph(*nodes, edges=edges)


def _warnings(seen):
    return [SimpleNamespace(**payload) for kind, payload in seen
            if kind == "run_warning"
            and payload["kind"] == WARNING_DEVICE_FALLBACK]


async def test_the_executor_warns_before_any_node_runs(probe, cpu_only):
    seen = await _execute(_probe_graph("cuda:0"), {"device": "cuda"},
                          "cpu")
    warnings = _warnings(seen)
    assert [w.node_id for w in warnings] == [None, "train"]
    first_node = next(i for i, (kind, _) in enumerate(seen)
                      if kind == "node_status")
    last_warning = max(i for i, (kind, _) in enumerate(seen)
                       if kind == "run_warning")
    assert last_warning < first_node
    assert [w.code for w in warnings] == ["device_fallback_run",
                                          "device_fallback_node"]


async def test_a_run_without_a_fallback_emits_no_warning(probe):
    seen = await _execute(_probe_graph("cpu"), {"device": "cpu"},
                          "cpu")
    assert _warnings(seen) == []


async def test_a_graph_that_does_not_prepare_warns_about_nothing(probe, cpu_only):
    """Review item 4: no entry point, so nothing runs and nothing is said."""
    graph = _graph(_node("train", "_DeviceProbe", device="cuda"))
    seen = await _execute(graph, {"device": "cuda"}, "cpu")
    assert _warnings(seen) == []


async def test_a_bypassed_or_unreachable_node_warns_about_nothing(probe, cpu_only):
    bypassed = _node("off", "_DeviceProbe", device="cuda")
    bypassed["data"]["bypassed"] = True
    graph = _graph(_node("start", "Start"), bypassed,
                   _node("island", "_DeviceProbe", device="cuda"),
                   _node("train", "_DeviceProbe", device="cpu"),
                   edges=[_trigger("start", "train"),
                          _trigger("start", "off")])
    seen = await _execute(graph, {"device": "cpu"}, "cpu")
    assert _warnings(seen) == []


async def test_the_graph_is_expanded_once(probe, monkeypatch, cpu_only):
    """Review item 5: the check reads execute_graph's own expansion."""
    from app.core import graph_engine

    calls = []
    real = graph_engine.prepare_executable_graph

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(graph_engine, "prepare_executable_graph", counting)
    seen = await _execute(_probe_graph("cuda"), {"device": "cpu"},
                          "cpu")
    assert len(_warnings(seen)) == 1
    assert len(calls) == 1


async def test_a_worker_subprocess_uses_in_run_names(probe, one_card):
    seen = await _execute(_probe_graph("cuda:1"),
                          {"device": "cuda:1"}, "cuda:0", exclusive=True,
                          worker_run=True,
                          run_cards={"cuda:0": "cuda:1"})
    (warning,) = _warnings(seen)
    assert warning.node_id == "train"
    assert "this run has 1 card (cuda:0 = the worker's cuda:1)" \
        in warning.detail


async def test_a_failing_report_never_fails_the_run(probe, cpu_only, caplog):
    """Review item 7: a sink that raises is logged; the run carries on to
    its own outcome."""
    seen = await _execute(_probe_graph("cuda"), {"device": "cpu"},
                          "cpu", warning_sink_fails=True)
    assert any(kind == "node_status" for kind, _ in seen)
    assert "could not report device fallbacks" in caplog.text


async def test_a_failing_check_never_fails_the_run(probe, monkeypatch):
    def boom(*_: Any, **__: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(device_fallback, "device_fallbacks", boom)
    seen = await _execute(_probe_graph("cpu"), {"device": "cuda"},
                          "cpu")
    assert _warnings(seen) == []
    assert any(kind == "node_status" for kind, _ in seen)


async def test_an_exclusive_process_alone_is_not_a_worker_run(probe,
                                                              cpu_only):
    """Second review, item 6: ``exclusive_process`` is about peak memory;
    whether the run is a worker's is its own flag."""
    seen = await _execute(_probe_graph("cuda"), {"device": "cuda"}, "cpu",
                          exclusive=True)
    details = [w.detail for w in _warnings(seen)]
    assert details[0].startswith("The run asked for device cuda")
    assert "which this server does not have" in details[1]
    seen = await _execute(_probe_graph("cuda"), {"device": "cuda"}, "cpu",
                          worker_run=True, run_cards={})
    details = [w.detail for w in _warnings(seen)]
    assert len(details) == 1 and "this run has no CUDA card" in details[0]


async def test_a_cached_node_is_described_as_its_setting(probe, cpu_only):
    """Second review, item 7: a canvas run may serve a node from its cache,
    and that node never executes. The sentence is about the setting and
    what happens when it executes, true either way."""
    from app.core.cache import ExecutionCache

    class _Session:
        cache = ExecutionCache()
        changed_nodes: list[str] = []

    from app.core.executors import ExecutionRequest, InProcessExecutor, RunSinks

    warned: list[Any] = []

    async def emit(event_type, payload):
        if event_type == "run_warning":
            warned.append(SimpleNamespace(**payload))

    async def ignore(*_: Any) -> None:
        return None

    for _ in range(2):  # the second run serves the node from the cache
        await InProcessExecutor().run(ExecutionRequest(
            run_id="r", graph=_probe_graph("cuda"), options={"device": "cpu"},
            context=ExecutionContext(device="cpu"), session=_Session(),
            output_store=None, record_outputs=False,
            sinks=RunSinks(emit=emit, record_metrics=ignore,
                           record_artifact=ignore, warn=ignore),
            explicit_metric_nodes=set()))
    assert len(warned) == 2
    assert all("is set to device cuda" in w.detail
               and "when it executes, it runs on cpu" in w.detail
               for w in warned)
