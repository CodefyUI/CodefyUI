"""core#260 -- a container's terminal status must say what it actually did.

``_emit_preset_aware`` rolls every internal node of a preset (or a subgraph
instance) up into ONE status for the box the canvas draws. Until this file
existed it rolled ``completed``, ``cached`` and ``skipped`` into the same
counter and then emitted ``completed`` whenever that counter filled -- so a
preset whose every internal node was a cache hit was indistinguishable, from
outside, from one that trained for ten minutes. That is what made #253
(a second Run doing zero training) invisible rather than merely wrong.

The tests here follow the methodology that found #253: they COUNT real
``execute()`` calls rather than trust the status plumbing, and then assert
what the status plumbing said about those calls. A test that only compared
statuses to statuses could pass on a build where nothing ran at all.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

from app.core.cache import ExecutionCache
from app.core.graph_engine import build_preset_fallback, execute_graph
from app.core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition
from app.core.node_registry import registry

#: node name -> how many times its ``execute()`` really ran, this test.
EXECUTIONS: Counter = Counter()


class _CountingScaleNode(BaseNode):
    """Cacheable, deterministic, and it counts how often it truly ran.

    Deliberately a test node rather than the real ``ScalarMultiply``: the
    claim under test is "nothing inside the box executed", and the only way
    to state that without depending on the status being fixed is to count
    the calls at the source.
    """

    NODE_NAME = "_CountingScale"
    CATEGORY = "Test"
    DESCRIPTION = "Multiply by a scalar and record that it happened"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="tensor", data_type=DataType.TENSOR)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="tensor", data_type=DataType.TENSOR)]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [ParamDefinition(name="scalar", param_type=ParamType.FLOAT, default=1.0)]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        EXECUTIONS[self.NODE_NAME] += 1
        return {"tensor": inputs["tensor"] * float(params.get("scalar", 1.0))}


class _UncacheableProbeNode(_CountingScaleNode):
    """Same shape, but the cache may never serve it -- so it always runs.

    This is the "some internal really did work" half of the mixed case. A
    node that opts out of the cache is not exotic: ``PythonScript`` (#256)
    and ``TrainingLoop`` (#253) both do it, for the same reason.
    """

    NODE_NAME = "_UncacheableProbe"
    DESCRIPTION = "Never served from cache; records every run"
    cacheable = False


class _AlwaysFailsNode(BaseNode):
    """Raises wherever it is placed -- as a top-level root or inside the box.

    Its one input is OPTIONAL so the same class can stand in for either
    position: a root with nothing wired in, or the second node of the
    preset below.
    """

    NODE_NAME = "_AlwaysFails"
    CATEGORY = "Test"
    DESCRIPTION = "Raises on every execution"
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, optional=True),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="tensor", data_type=DataType.TENSOR)]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [ParamDefinition(name="scalar", param_type=ParamType.FLOAT, default=1.0)]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("this node always fails")


_TEST_NODES = {
    "_CountingScale": _CountingScaleNode,
    "_UncacheableProbe": _UncacheableProbeNode,
    "_AlwaysFails": _AlwaysFailsNode,
}


@pytest.fixture(autouse=True)
def _register_test_nodes():
    registry._nodes.update(_TEST_NODES)
    EXECUTIONS.clear()
    yield
    for name in _TEST_NODES:
        registry._nodes.pop(name, None)
    EXECUTIONS.clear()


# ── graph fixtures ──────────────────────────────────────────────────────
#
# start --trigger--> src (TensorCreate)
#   src.tensor --data--> box.value  (exposed input -> first internal node)
#
# Two internal nodes, never one: the container only emits a terminal status
# once EVERY internal has reported, so a single-node preset would satisfy
# any roll-up rule vacuously.


def _preset(second_type: str, name: str = "_StatusProbePipeline") -> dict:
    return {
        "preset_name": name,
        "category": "Test",
        "description": "",
        "tags": [],
        "nodes": [
            {"id": "first", "type": "_CountingScale", "params": {"scalar": 2.0}},
            {"id": "second", "type": second_type, "params": {"scalar": 3.0}},
        ],
        "edges": [
            {"source": "first", "target": "second",
             "sourceHandle": "tensor", "targetHandle": "tensor"},
        ],
        "exposed_inputs": [
            {"name": "value", "internal_node": "first", "internal_port": "tensor",
             "data_type": "TENSOR", "description": ""},
        ],
        "exposed_outputs": [],
        "exposed_params": [],
    }


def _graph(source_type: str = "TensorCreate") -> tuple[list[dict], list[dict]]:
    source_params = (
        {"shape": "2,2", "fill": "full", "value": 5.0}
        if source_type == "TensorCreate" else {}
    )
    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
         "data": {"params": {}}},
        {"id": "src", "type": source_type, "position": {"x": 1, "y": 0},
         "data": {"params": source_params}},
        {"id": "box", "type": "preset:_StatusProbePipeline",
         "position": {"x": 2, "y": 0}, "data": {"params": {}}},
    ]
    edges = [
        {"id": "t1", "source": "start", "target": "src",
         "sourceHandle": "trigger", "targetHandle": "__trigger",
         "type": "trigger"},
        {"id": "e1", "source": "src", "target": "box",
         "sourceHandle": "tensor", "targetHandle": "value", "type": "data"},
    ]
    return nodes, edges


async def _run(nodes, edges, preset, cache=None, **kwargs) -> list[tuple[str, str]]:
    """Execute once and return every (node_id, status) the canvas would see."""
    seen: list[tuple[str, str]] = []

    async def on_progress(node_id, status, data):
        if status != "progress":
            seen.append((node_id, status))

    await execute_graph(
        nodes, edges, on_progress=on_progress, cache=cache,
        preset_fallback=build_preset_fallback([preset]), **kwargs,
    )
    return seen


def _statuses_for(seen: list[tuple[str, str]], node_id: str) -> list[str]:
    return [status for nid, status in seen if nid == node_id]


# ── the bug #260 filed ──────────────────────────────────────────────────


async def test_a_preset_whose_internals_were_all_cached_reports_cached():
    """Two runs, one cache. The second one did nothing and must say so.

    This is the exact shape measured on the shipped C2-5 MLP-MNIST example:
    run 1 trains, run 2 executes nothing, and the preset used to report
    ``completed`` both times.
    """
    preset = _preset("_CountingScale")
    nodes, edges = _graph()
    cache = ExecutionCache()

    first = await _run(nodes, edges, preset, cache=cache)
    assert EXECUTIONS["_CountingScale"] == 2, "both internals should have run"
    assert _statuses_for(first, "box")[-1] == "completed", first

    second = await _run(nodes, edges, preset, cache=cache)
    # Counted, not inferred: the second run really executed nothing.
    assert EXECUTIONS["_CountingScale"] == 2, "run 2 must not have re-executed"
    assert _statuses_for(second, "src") == ["cached"], second
    assert _statuses_for(second, "box")[-1] == "cached", second
    # And exactly one terminal status, not a 'cached' plus a stray 'completed'.
    box_terminals = [
        s for s in _statuses_for(second, "box")
        if s in ("completed", "cached", "skipped")
    ]
    assert box_terminals == ["cached"], second


async def test_the_internal_node_ids_still_never_surface():
    """The roll-up is still a roll-up: only the box the canvas drew reports."""
    preset = _preset("_CountingScale")
    nodes, edges = _graph()
    cache = ExecutionCache()

    await _run(nodes, edges, preset, cache=cache)
    seen = await _run(nodes, edges, preset, cache=cache)
    assert not any(nid.startswith("box__") for nid, _ in seen), seen


# ── the case that must NOT change ───────────────────────────────────────


async def test_a_mixed_preset_still_reports_completed():
    """One cache hit plus one real execution is a box that DID work.

    Demoting this to ``cached`` would trade one lie for another: the user
    would be told nothing ran when half of it did.
    """
    preset = _preset("_UncacheableProbe")
    nodes, edges = _graph()
    cache = ExecutionCache()

    await _run(nodes, edges, preset, cache=cache)
    assert EXECUTIONS["_CountingScale"] == 1
    assert EXECUTIONS["_UncacheableProbe"] == 1

    second = await _run(nodes, edges, preset, cache=cache)
    # The cacheable half was served from cache; the other half really ran.
    assert EXECUTIONS["_CountingScale"] == 1, "the cacheable internal re-ran"
    assert EXECUTIONS["_UncacheableProbe"] == 2, "the live internal did not run"
    assert _statuses_for(second, "box")[-1] == "completed", second


async def test_a_first_run_with_an_empty_cache_still_reports_completed():
    """No cache at all: nothing changes for the ordinary case."""
    preset = _preset("_CountingScale")
    nodes, edges = _graph()

    seen = await _run(nodes, edges, preset)
    assert EXECUTIONS["_CountingScale"] == 2
    assert _statuses_for(seen, "box") == ["running", "completed"], seen


# ── the third state: nothing was even attempted ─────────────────────────


async def test_a_preset_passed_over_by_an_upstream_failure_reports_skipped():
    """Every internal ``skipped`` -> the box is ``skipped``, not ``completed``.

    ``skipped`` is emitted in exactly one place -- an upstream node failed --
    so a box of nothing but skips was never attempted. A failure INSIDE the
    box settles it as ``error`` instead (see the test below), which is why
    this shape needs the failure to sit OUTSIDE.
    """
    preset = _preset("_CountingScale")
    nodes, edges = _graph(source_type="_AlwaysFails")

    seen = await _run(nodes, edges, preset, error_mode="continue")
    assert EXECUTIONS["_CountingScale"] == 0, "nothing inside the box may run"
    assert _statuses_for(seen, "box")[-1] == "skipped", seen
    assert "completed" not in _statuses_for(seen, "box"), seen


async def test_an_internal_failure_still_settles_the_box_as_error():
    """The pre-existing precedence is untouched: error wins immediately.

    An errored internal never increments the done count, so the terminal
    branch this PR changed is not even reached -- pinned here because the
    new rank table would be a plausible place to break it.
    """
    preset = _preset("_AlwaysFails")
    nodes, edges = _graph()

    seen = await _run(nodes, edges, preset, error_mode="continue")
    box = _statuses_for(seen, "box")
    assert "error" in box, seen
    assert not any(s in ("completed", "cached", "skipped") for s in box), seen
