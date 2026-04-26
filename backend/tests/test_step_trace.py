"""Tests for the verbose / step-trace pipeline (A1)."""

from __future__ import annotations

import pytest
import torch

from app.core.execution_context import ExecutionContext
from app.core.graph_engine import execute_graph
from app.core.run_output_store import RunOutputStore
from app.core.step_trace import Step, StepRecorder


# ── StepRecorder unit tests ─────────────────────────────────────────


def test_recorder_starts_empty():
    r = StepRecorder()
    assert len(r) == 0
    assert r.steps == []


def test_recorder_records_tensors_and_scalars():
    r = StepRecorder()
    a = torch.zeros(2)
    b = torch.ones(3)
    r.record("step_a", "first step", scalars={"alpha": 0.5}, x=a, y=b)

    assert len(r) == 1
    s = r.steps[0]
    assert isinstance(s, Step)
    assert s.name == "step_a"
    assert s.description == "first step"
    assert s.scalars == {"alpha": 0.5}
    assert set(s.tensors.keys()) == {"x", "y"}
    assert torch.equal(s.tensors["x"], a)


def test_recorder_preserves_order():
    r = StepRecorder()
    r.record("first", "")
    r.record("second", "")
    r.record("third", "")
    assert [s.name for s in r.steps] == ["first", "second", "third"]


# ── Verbose mode end-to-end via execute_graph ───────────────────────


@pytest.mark.asyncio
async def test_attention_emits_steps_when_verbose():
    """Running MultiHeadAttention with verbose=True should populate __steps__
    entries in the RunOutputStore via graph_engine's expansion logic."""

    store = RunOutputStore(max_runs=5)
    nodes = [
        {
            "id": "start",
            "type": "Start",
            "position": {"x": 0, "y": 0},
            "data": {"params": {}},
        },
        {
            "id": "q",
            "type": "TensorInput",
            "position": {"x": 100, "y": -60},
            "data": {
                "params": {
                    "shape": "2,1,8",
                    "dtype": "float32",
                    "value_mode": "random",
                    "seed": 0,
                }
            },
        },
        {
            "id": "k",
            "type": "TensorInput",
            "position": {"x": 100, "y": 0},
            "data": {
                "params": {
                    "shape": "2,1,8",
                    "dtype": "float32",
                    "value_mode": "random",
                    "seed": 1,
                }
            },
        },
        {
            "id": "v",
            "type": "TensorInput",
            "position": {"x": 100, "y": 60},
            "data": {
                "params": {
                    "shape": "2,1,8",
                    "dtype": "float32",
                    "value_mode": "random",
                    "seed": 2,
                }
            },
        },
        {
            "id": "att",
            "type": "MultiHeadAttention",
            "position": {"x": 300, "y": 0},
            "data": {"params": {"embed_dim": 8, "num_heads": 2}},
        },
    ]
    edges = [
        # All three TensorInput nodes need a trigger so they enter the executable subgraph.
        {"id": "etq", "source": "start", "target": "q", "sourceHandle": "trigger", "type": "trigger"},
        {"id": "etk", "source": "start", "target": "k", "sourceHandle": "trigger", "type": "trigger"},
        {"id": "etv", "source": "start", "target": "v", "sourceHandle": "trigger", "type": "trigger"},
        {"id": "eq", "source": "q", "target": "att", "sourceHandle": "tensor", "targetHandle": "query"},
        {"id": "ek", "source": "k", "target": "att", "sourceHandle": "tensor", "targetHandle": "key"},
        {"id": "ev", "source": "v", "target": "att", "sourceHandle": "tensor", "targetHandle": "value"},
    ]

    ctx = ExecutionContext(verbose=True)
    await execute_graph(
        nodes,
        edges,
        context=ctx,
        run_id="run-verbose",
        output_store=store,
        record_outputs=True,
    )

    ports = await store.list_ports("run-verbose")
    assert ports is not None
    att_ports = {port for nid, port in ports if nid == "att"}

    # Forward outputs still captured.
    assert "output" in att_ports
    assert "weights" in att_ports

    # Each verbose step contributes one __step__N__meta plus its tensors.
    step_metas = [p for p in att_ports if p.startswith("__step__") and p.endswith("__meta")]
    assert len(step_metas) >= 4, f"expected >=4 step metas, got {sorted(att_ports)}"

    # Spot-check that compute_qkv exposed Q, K, V tensor entries.
    qkv_tensors = {p for p in att_ports if p.startswith("__step__0__") and not p.endswith("__meta")}
    assert qkv_tensors == {"__step__0__Q", "__step__0__K", "__step__0__V"}


@pytest.mark.asyncio
async def test_attention_emits_no_steps_when_not_verbose():
    """Default (verbose=False) path should not emit any __step__* entries."""

    store = RunOutputStore(max_runs=5)
    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0}, "data": {"params": {}}},
        {
            "id": "q",
            "type": "TensorInput",
            "position": {"x": 100, "y": -60},
            "data": {"params": {"shape": "2,1,8", "dtype": "float32", "value_mode": "random", "seed": 0}},
        },
        {
            "id": "k",
            "type": "TensorInput",
            "position": {"x": 100, "y": 0},
            "data": {"params": {"shape": "2,1,8", "dtype": "float32", "value_mode": "random", "seed": 1}},
        },
        {
            "id": "v",
            "type": "TensorInput",
            "position": {"x": 100, "y": 60},
            "data": {"params": {"shape": "2,1,8", "dtype": "float32", "value_mode": "random", "seed": 2}},
        },
        {
            "id": "att",
            "type": "MultiHeadAttention",
            "position": {"x": 300, "y": 0},
            "data": {"params": {"embed_dim": 8, "num_heads": 2}},
        },
    ]
    edges = [
        # All three TensorInput nodes need a trigger so they enter the executable subgraph.
        {"id": "etq", "source": "start", "target": "q", "sourceHandle": "trigger", "type": "trigger"},
        {"id": "etk", "source": "start", "target": "k", "sourceHandle": "trigger", "type": "trigger"},
        {"id": "etv", "source": "start", "target": "v", "sourceHandle": "trigger", "type": "trigger"},
        {"id": "eq", "source": "q", "target": "att", "sourceHandle": "tensor", "targetHandle": "query"},
        {"id": "ek", "source": "k", "target": "att", "sourceHandle": "tensor", "targetHandle": "key"},
        {"id": "ev", "source": "v", "target": "att", "sourceHandle": "tensor", "targetHandle": "value"},
    ]

    ctx = ExecutionContext(verbose=False)
    await execute_graph(
        nodes,
        edges,
        context=ctx,
        run_id="run-quiet",
        output_store=store,
        record_outputs=True,
    )

    ports = await store.list_ports("run-quiet")
    att_ports = {port for nid, port in (ports or []) if nid == "att"}
    assert "output" in att_ports
    assert "weights" in att_ports
    assert not any(p.startswith("__step__") for p in att_ports), (
        f"verbose=False should produce no __step__* ports, got {sorted(att_ports)}"
    )
