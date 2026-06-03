"""Tests for SplitNode (param-driven port count)."""

from __future__ import annotations

import torch

from app.core.graph_engine import validate_graph
from app.nodes.tensor_ops.split_node import MAX_CHUNKS, SplitNode


def _run(tensor, **params):
    return SplitNode().execute({"tensor": tensor}, params)


# ── Static schema (palette template) ──


def test_node_metadata():
    assert SplitNode.NODE_NAME == "Split"
    out_names = [p.name for p in SplitNode.define_outputs()]
    # Baseline schema matches the default `chunks=2` so the palette renders
    # two ports for a freshly-dragged node.
    assert out_names == ["chunk_0", "chunk_1"]


def test_chunks_param_has_min_max_in_define_params():
    params = {p.name: p for p in SplitNode.define_params()}
    assert "chunks" in params
    assert params["chunks"].min_value == 1
    assert params["chunks"].max_value == MAX_CHUNKS


# ── Dynamic ports ──


def test_dynamic_outputs_match_chunks_param():
    out_names = [p.name for p in SplitNode.define_outputs_dynamic({"chunks": 4})]
    assert out_names == ["chunk_0", "chunk_1", "chunk_2", "chunk_3"]


def test_dynamic_outputs_defaults_to_two_when_no_params():
    out_names_none = [p.name for p in SplitNode.define_outputs_dynamic(None)]
    out_names_empty = [p.name for p in SplitNode.define_outputs_dynamic({})]
    assert out_names_none == ["chunk_0", "chunk_1"]
    assert out_names_empty == ["chunk_0", "chunk_1"]


def test_dynamic_outputs_clamps_to_one_when_below_one():
    out_names = [p.name for p in SplitNode.define_outputs_dynamic({"chunks": 0})]
    assert out_names == ["chunk_0"]


def test_dynamic_outputs_clamps_to_max():
    out_names = [p.name for p in SplitNode.define_outputs_dynamic({"chunks": 99})]
    assert len(out_names) == MAX_CHUNKS
    assert out_names[0] == "chunk_0"
    assert out_names[-1] == f"chunk_{MAX_CHUNKS - 1}"


def test_dynamic_outputs_tolerates_non_int_chunks():
    out_names = [p.name for p in SplitNode.define_outputs_dynamic({"chunks": "three"})]
    # Falls back to default 2 when value can't be coerced to int.
    assert out_names == ["chunk_0", "chunk_1"]


# ── Execution ──


def test_split_into_two_along_dim_zero():
    x = torch.arange(8).reshape(4, 2)
    res = _run(x, chunks=2, dim=0)
    assert set(res.keys()) == {"chunk_0", "chunk_1"}
    assert res["chunk_0"].shape == (2, 2)
    assert res["chunk_1"].shape == (2, 2)
    assert torch.equal(torch.cat([res["chunk_0"], res["chunk_1"]], dim=0), x)


def test_split_along_dim_one():
    x = torch.arange(8).reshape(2, 4)
    res = _run(x, chunks=2, dim=1)
    assert res["chunk_0"].shape == (2, 2)
    assert res["chunk_1"].shape == (2, 2)


def test_split_into_four_chunks_returns_all_keys():
    x = torch.arange(8)
    res = _run(x, chunks=4, dim=0)
    assert set(res.keys()) == {"chunk_0", "chunk_1", "chunk_2", "chunk_3"}
    for v in res.values():
        assert v.shape == (2,)


def test_split_three_chunks_unbalanced_tail():
    # torch.chunk(size=8, chunks=3) → [3, 3, 2]
    x = torch.arange(8)
    res = _run(x, chunks=3, dim=0)
    assert set(res.keys()) == {"chunk_0", "chunk_1", "chunk_2"}
    assert res["chunk_0"].shape == (3,)
    assert res["chunk_1"].shape == (3,)
    assert res["chunk_2"].shape == (2,)


def test_default_two_chunks_along_dim_zero():
    x = torch.zeros(6)
    res = SplitNode().execute({"tensor": x}, {})
    assert res["chunk_0"].shape == (3,)
    assert res["chunk_1"].shape == (3,)


# ── Validator integration ──


def _split_then_print(chunks: int, source_handle: str) -> tuple[list[dict], list[dict]]:
    """Build a tiny graph: Start → TensorInput → Split → Print.

    The Print pulls from `source_handle` on the Split node so we can probe
    whether the validator accepts a given chunk port for a given `chunks`
    setting.
    """
    nodes = [
        {"id": "s", "type": "Start", "data": {"params": {}}},
        {"id": "t", "type": "TensorInput", "data": {"params": {"shape": "6", "value_mode": "zeros"}}},
        {"id": "sp", "type": "Split", "data": {"params": {"chunks": chunks, "dim": 0}}},
        {"id": "p", "type": "Print", "data": {"params": {}}},
    ]
    edges = [
        {"id": "e1", "source": "s", "target": "t", "sourceHandle": "trigger", "type": "trigger"},
        {"id": "e2", "source": "t", "target": "sp", "sourceHandle": "tensor", "targetHandle": "tensor"},
        {"id": "e3", "source": "sp", "target": "p", "sourceHandle": source_handle, "targetHandle": "value"},
    ]
    return nodes, edges


def test_validator_accepts_chunk_2_when_chunks_is_three():
    nodes, edges = _split_then_print(chunks=3, source_handle="chunk_2")
    errors = validate_graph(nodes, edges)
    assert not any("Invalid output port" in e for e in errors), errors


def test_validator_rejects_chunk_3_when_chunks_is_three():
    nodes, edges = _split_then_print(chunks=3, source_handle="chunk_3")
    errors = validate_graph(nodes, edges)
    assert any("Invalid output port 'chunk_3' on Split" in e for e in errors), errors


def test_validator_accepts_chunk_0_when_chunks_is_one():
    nodes, edges = _split_then_print(chunks=1, source_handle="chunk_0")
    errors = validate_graph(nodes, edges)
    assert not any("Invalid output port" in e for e in errors), errors


def test_validator_rejects_chunk_1_when_chunks_is_one():
    nodes, edges = _split_then_print(chunks=1, source_handle="chunk_1")
    errors = validate_graph(nodes, edges)
    assert any("Invalid output port 'chunk_1' on Split" in e for e in errors), errors
