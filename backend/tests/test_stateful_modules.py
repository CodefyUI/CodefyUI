"""Integration tests for StatefulModuleMixin layer nodes (A2)."""

from __future__ import annotations

import pytest
import torch

from app.core.execution_context import ExecutionContext
from app.core.graph_engine import execute_graph
from app.core.node_state_store import NodeStateStore
from app.core.run_output_store import RunOutputStore
from app.nodes.cnn.conv2d_node import Conv2dNode
from app.nodes.utility.linear_node import LinearNode


# ── Direct (no graph_engine) tests on a single node ─────────────────


def _make_ctx(persistent: bool = True, graph_id: str = "g1", node_id: str = "n1") -> ExecutionContext:
    return ExecutionContext(
        graph_id=graph_id,
        weights_persistent=persistent,
        node_state_store=NodeStateStore(),
        current_node_id=node_id,
    )


def test_conv2d_returns_same_module_across_calls_when_persistent():
    node = Conv2dNode()
    ctx = _make_ctx()
    params = {"in_channels": 1, "out_channels": 4, "kernel_size": 3, "stride": 1, "padding": 1}
    x = torch.randn(1, 1, 8, 8)
    a = node.get_or_build_module(ctx, params)
    b = node.get_or_build_module(ctx, params)
    assert a is b
    # Output is reproducible across calls (weights didn't change between fetches).
    out1 = node.execute({"tensor": x}, params, context=ctx)["tensor"]
    out2 = node.execute({"tensor": x}, params, context=ctx)["tensor"]
    assert torch.equal(out1, out2)


def test_conv2d_rebuilds_when_weights_persistent_is_false():
    node = Conv2dNode()
    ctx_off = _make_ctx(persistent=False)
    params = {"in_channels": 1, "out_channels": 4, "kernel_size": 3, "stride": 1, "padding": 1}
    a = node.get_or_build_module(ctx_off, params)
    b = node.get_or_build_module(ctx_off, params)
    assert a is not b


def test_conv2d_rebuilds_on_structural_change():
    node = Conv2dNode()
    ctx = _make_ctx()
    params_a = {"in_channels": 1, "out_channels": 4, "kernel_size": 3, "stride": 1, "padding": 1}
    params_b = {"in_channels": 1, "out_channels": 8, "kernel_size": 3, "stride": 1, "padding": 1}
    a = node.get_or_build_module(ctx, params_a)
    b = node.get_or_build_module(ctx, params_b)
    assert a is not b
    assert b.out_channels == 8


def test_node_without_context_falls_back_to_fresh_module():
    """Direct .execute() without a context (e.g. CLI) keeps stateless behaviour."""
    node = Conv2dNode()
    params = {"in_channels": 1, "out_channels": 4, "kernel_size": 3, "stride": 1, "padding": 1}
    x = torch.randn(1, 1, 8, 8)
    out_a = node.execute({"tensor": x}, params)["tensor"]
    out_b = node.execute({"tensor": x}, params)["tensor"]
    # Without persistence, fresh modules → different random init → outputs differ.
    assert not torch.equal(out_a, out_b)


def test_linear_persists_across_calls():
    node = LinearNode()
    ctx = _make_ctx(node_id="lin")
    params = {"in_features": 4, "out_features": 8}
    x = torch.randn(2, 4)
    a = node.execute({"tensor": x}, params, context=ctx)["tensor"]
    b = node.execute({"tensor": x}, params, context=ctx)["tensor"]
    assert torch.equal(a, b)


# ── End-to-end via execute_graph (cache + ws-level integration) ─────


@pytest.mark.asyncio
async def test_persisted_conv2d_is_stable_across_runs_in_graph_engine():
    """Two execute_graph calls with persistent weights should produce identical outputs."""

    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0}, "data": {"params": {}}},
        {
            "id": "x",
            "type": "TensorInput",
            "position": {"x": 100, "y": 0},
            "data": {
                "params": {
                    "shape": "1,1,8,8",
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
    ]
    edges = [
        {"id": "et", "source": "start", "target": "x", "sourceHandle": "trigger", "type": "trigger"},
        {"id": "ex", "source": "x", "target": "conv", "sourceHandle": "tensor", "targetHandle": "tensor"},
    ]

    store_a = RunOutputStore(max_runs=5)
    store_b = RunOutputStore(max_runs=5)
    nss = NodeStateStore()

    ctx_a = ExecutionContext(
        graph_id="g-stateful",
        weights_persistent=True,
        node_state_store=nss,
    )
    out_a = await execute_graph(nodes, edges, context=ctx_a, run_id="r1", output_store=store_a, record_outputs=True)

    ctx_b = ExecutionContext(
        graph_id="g-stateful",
        weights_persistent=True,
        node_state_store=nss,
    )
    out_b = await execute_graph(nodes, edges, context=ctx_b, run_id="r2", output_store=store_b, record_outputs=True)

    t_a = out_a["conv"]["tensor"]
    t_b = out_b["conv"]["tensor"]
    assert torch.allclose(t_a, t_b), "persistent weights should produce identical Conv2d output"


@pytest.mark.asyncio
async def test_reset_then_run_changes_weights():
    """After reset_graph, the next run should produce different weights / output."""
    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0}, "data": {"params": {}}},
        {
            "id": "x",
            "type": "TensorInput",
            "position": {"x": 100, "y": 0},
            "data": {
                "params": {
                    "shape": "1,1,8,8",
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
    ]
    edges = [
        {"id": "et", "source": "start", "target": "x", "sourceHandle": "trigger", "type": "trigger"},
        {"id": "ex", "source": "x", "target": "conv", "sourceHandle": "tensor", "targetHandle": "tensor"},
    ]

    nss = NodeStateStore()

    ctx_a = ExecutionContext(graph_id="g-reset", weights_persistent=True, node_state_store=nss)
    out_a = await execute_graph(nodes, edges, context=ctx_a)

    nss.reset_graph("g-reset")

    ctx_b = ExecutionContext(graph_id="g-reset", weights_persistent=True, node_state_store=nss)
    out_b = await execute_graph(nodes, edges, context=ctx_b)

    t_a = out_a["conv"]["tensor"]
    t_b = out_b["conv"]["tensor"]
    assert not torch.allclose(t_a, t_b), "after reset, fresh init should change the output"


@pytest.mark.asyncio
async def test_stateful_node_skips_cache():
    """ExecutionCache must not cache stateful node output, otherwise drift breaks correctness."""
    from app.core.cache import ExecutionCache

    nodes = [
        {"id": "start", "type": "Start", "position": {"x": 0, "y": 0}, "data": {"params": {}}},
        {
            "id": "x",
            "type": "TensorInput",
            "position": {"x": 100, "y": 0},
            "data": {
                "params": {
                    "shape": "1,1,8,8",
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
    ]
    edges = [
        {"id": "et", "source": "start", "target": "x", "sourceHandle": "trigger", "type": "trigger"},
        {"id": "ex", "source": "x", "target": "conv", "sourceHandle": "tensor", "targetHandle": "tensor"},
    ]

    cache = ExecutionCache()
    ctx = ExecutionContext(
        graph_id="g-cache",
        weights_persistent=True,
        node_state_store=NodeStateStore(),
    )
    await execute_graph(nodes, edges, context=ctx, cache=cache)
    # The cache should have entries for cacheable nodes (Start, TensorInput) but
    # NOT Conv2d. Easiest check: cache size <= 2 and all hits exclude conv.
    assert len(cache) <= 2
