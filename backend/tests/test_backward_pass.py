"""Tests for the backward / gradient inspector pipeline (A3)."""

from __future__ import annotations

import pytest
import torch

from app.core.backward_pass import (
    attach_retain_grad,
    grad_health,
    select_backward_target,
)
from app.core.execution_context import ExecutionContext
from app.core.graph_engine import execute_graph
from app.core.node_state_store import NodeStateStore
from app.core.run_output_store import RunOutputStore


# ── Health classifier ───────────────────────────────────────────────


def test_grad_health_classifies_vanishing():
    g = torch.zeros(10) + 1e-12
    assert grad_health(g)["status"] == "vanishing"


def test_grad_health_classifies_exploding():
    g = torch.full((10,), 1e3)
    assert grad_health(g)["status"] == "exploding"


def test_grad_health_classifies_healthy():
    g = torch.full((10,), 0.1)
    assert grad_health(g)["status"] == "healthy"


# ── attach_retain_grad ──────────────────────────────────────────────


def test_attach_retain_grad_records_floating_tensors_only():
    sink: dict[tuple[str, str], torch.Tensor] = {}
    f = torch.randn(3, requires_grad=True)
    i = torch.tensor([1, 2, 3])
    attach_retain_grad({"x": f, "y": i}, sink, "n1", "")
    assert ("n1", "x") in sink
    assert ("n1", "y") not in sink


def test_attach_retain_grad_walks_nested_dicts():
    sink: dict[tuple[str, str], torch.Tensor] = {}
    a = torch.randn(2, requires_grad=True)
    attach_retain_grad({"a": a, "__steps__": [1, 2]}, sink, "n1", "")
    # __steps__ should be skipped due to leading-underscore filter.
    assert list(sink.keys()) == [("n1", "a")]


# ── select_backward_target ──────────────────────────────────────────


def test_select_target_prefers_backward_once():
    t = torch.randn(2, requires_grad=True)
    nodes = [
        {"id": "x", "type": "Conv2d"},
        {"id": "bwd", "type": "BackwardOnce"},
    ]
    outputs = {"bwd": {"tensor": t}}
    result = select_backward_target(nodes, outputs, auto_backward=True)
    assert result is not None
    loss, label = result
    assert label.startswith("BackwardOnce(")


def test_select_target_skips_when_training_loop_present_and_no_backward_once():
    nodes = [
        {"id": "tl", "type": "TrainingLoop"},
        {"id": "loss", "type": "Loss"},
    ]
    outputs = {"tl": {"trained_model": torch.nn.Linear(2, 2)}}
    result = select_backward_target(nodes, outputs, auto_backward=True)
    assert result is None


def test_select_target_uses_largest_floating_when_auto_backward():
    a = torch.randn(2, 4, requires_grad=True)
    b = torch.randn(8, requires_grad=True)
    nodes = [
        {"id": "n1", "type": "Linear"},
        {"id": "n2", "type": "Linear"},
    ]
    outputs = {
        "n1": {"tensor": a},      # numel = 8
        "n2": {"tensor": b},      # numel = 8
    }
    # Tie-break: first-seen wins; just ensure something is returned.
    result = select_backward_target(nodes, outputs, auto_backward=True)
    assert result is not None


def test_select_target_returns_none_when_auto_off_and_no_marker():
    nodes = [{"id": "n1", "type": "Conv2d"}]
    outputs = {"n1": {"tensor": torch.randn(2, requires_grad=True)}}
    result = select_backward_target(nodes, outputs, auto_backward=False)
    assert result is None


# ── End-to-end via execute_graph ────────────────────────────────────


@pytest.mark.asyncio
async def test_e2e_backward_captures_weight_and_port_grads():
    """TensorInput → Conv2d → Mean → BackwardOnce should populate
    `tensor__grad`, `__weight_grad__weight`, and `__weight_grad__bias`
    in the run output store."""

    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0}, "data": {"params": {}}},
        {
            "id": "x",
            "type": "TensorInput",
            "position": {"x": 100, "y": 0},
            "data": {
                "params": {
                    "shape": "1,1,4,4",
                    "dtype": "float32",
                    "value_mode": "random",
                    "seed": 0,
                }
            },
        },
        {
            "id": "conv",
            "type": "Conv2d",
            "position": {"x": 300, "y": 0},
            "data": {"params": {"in_channels": 1, "out_channels": 4, "kernel_size": 3, "stride": 1, "padding": 1}},
        },
        {
            "id": "mean",
            "type": "Mean",
            "position": {"x": 500, "y": 0},
            "data": {"params": {}},
        },
        {
            "id": "bwd",
            "type": "BackwardOnce",
            "position": {"x": 700, "y": 0},
            "data": {"params": {}},
        },
    ]
    edges = [
        {"id": "et", "source": "start", "target": "x", "sourceHandle": "trigger", "type": "trigger"},
        {"id": "e1", "source": "x", "target": "conv", "sourceHandle": "tensor", "targetHandle": "tensor"},
        {"id": "e2", "source": "conv", "target": "mean", "sourceHandle": "tensor", "targetHandle": "tensor"},
        {"id": "e3", "source": "mean", "target": "bwd", "sourceHandle": "tensor", "targetHandle": "tensor"},
    ]

    store = RunOutputStore(max_runs=5)
    nss = NodeStateStore()
    ctx = ExecutionContext(
        graph_id="g-bwd",
        weights_persistent=True,
        node_state_store=nss,
        backward_mode=True,
    )

    await execute_graph(
        nodes,
        edges,
        context=ctx,
        run_id="r-bwd",
        output_store=store,
        record_outputs=True,
    )

    ports = await store.list_ports("r-bwd")
    conv_ports = {p for nid, p in (ports or []) if nid == "conv"}

    # Weight gradients captured.
    assert "__weight_grad__weight" in conv_ports
    assert "__weight_grad__bias" in conv_ports
    # Health metadata captured.
    assert "__weight_grad__weight__meta" in conv_ports

    # Per-port forward gradient captured.
    assert "tensor__grad" in conv_ports

    weight_grad = await store.get("r-bwd", "conv", "__weight_grad__weight")
    assert isinstance(weight_grad, torch.Tensor)
    assert weight_grad.shape == (4, 1, 3, 3)


@pytest.mark.asyncio
async def test_no_backward_when_backward_mode_off():
    """Without backward_mode, no __grad ports are written."""
    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0}, "data": {"params": {}}},
        {
            "id": "x",
            "type": "TensorInput",
            "position": {"x": 100, "y": 0},
            "data": {"params": {"shape": "1,1,4,4", "dtype": "float32", "value_mode": "random", "seed": 0}},
        },
        {
            "id": "conv",
            "type": "Conv2d",
            "position": {"x": 300, "y": 0},
            "data": {"params": {"in_channels": 1, "out_channels": 4, "kernel_size": 3, "stride": 1, "padding": 1}},
        },
    ]
    edges = [
        {"id": "et", "source": "start", "target": "x", "sourceHandle": "trigger", "type": "trigger"},
        {"id": "e1", "source": "x", "target": "conv", "sourceHandle": "tensor", "targetHandle": "tensor"},
    ]

    store = RunOutputStore(max_runs=5)
    ctx = ExecutionContext(
        graph_id="g-no-bwd",
        weights_persistent=True,
        node_state_store=NodeStateStore(),
        backward_mode=False,
    )
    await execute_graph(nodes, edges, context=ctx, run_id="r-no-bwd", output_store=store, record_outputs=True)

    ports = await store.list_ports("r-no-bwd") or []
    conv_ports = {p for nid, p in ports if nid == "conv"}
    assert not any(p.endswith("__grad") for p in conv_ports)
    assert not any(p.startswith("__weight_grad__") for p in conv_ports)
