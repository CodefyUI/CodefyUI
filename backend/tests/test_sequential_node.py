"""Tests for SequentialModelNode (graph-based model builder)."""

from __future__ import annotations

import json

import pytest
import torch

from app.nodes.utility.sequential_node import SequentialModelNode


def test_node_metadata():
    assert SequentialModelNode.NODE_NAME == "SequentialModel"
    assert SequentialModelNode.CATEGORY == "Training"


def test_default_layers_param_is_valid_json():
    params = SequentialModelNode.define_params()
    layers_param = [p for p in params if p.name == "layers"][0]
    parsed = json.loads(layers_param.default)
    assert "nodes" in parsed
    assert "edges" in parsed


def test_build_simple_linear_model():
    spec = {
        "version": 2,
        "nodes": [
            {"id": "in", "type": "Input", "ports": [{"id": "p_x", "name": "x"}]},
            {"id": "l1", "type": "Linear", "params": {"in_features": 4, "out_features": 8}},
            {"id": "out", "type": "Output", "ports": [{"id": "p_y", "name": "y"}]},
        ],
        "edges": [
            {"id": "e1", "source": "in", "sourceHandle": "p_x", "target": "l1"},
            {"id": "e2", "source": "l1", "target": "out", "targetHandle": "p_y"},
        ],
    }
    res = SequentialModelNode().execute({}, {"layers": json.dumps(spec)})
    model = res["model"]
    assert callable(model)
    x = torch.randn(2, 4)
    y = model(x)
    assert y.shape == (2, 8)


def test_build_default_cnn_runs_on_28x28():
    """The default config should successfully build a model."""
    node = SequentialModelNode()
    params = node.define_params()
    layers_default = [p for p in params if p.name == "layers"][0].default
    res = node.execute({}, {"layers": layers_default})
    model = res["model"]
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    assert y.shape == (1, 10)


def test_unknown_layer_type_raises():
    spec = {
        "version": 2,
        "nodes": [
            {"id": "in", "type": "Input", "ports": [{"id": "p_x", "name": "x"}]},
            {"id": "bogus", "type": "BogusLayer", "params": {}},
            {"id": "out", "type": "Output", "ports": [{"id": "p_y", "name": "y"}]},
        ],
        "edges": [
            {"id": "e1", "source": "in", "sourceHandle": "p_x", "target": "bogus"},
            {"id": "e2", "source": "bogus", "target": "out", "targetHandle": "p_y"},
        ],
    }
    with pytest.raises((ValueError, KeyError)):
        SequentialModelNode().execute({}, {"layers": json.dumps(spec)})


# ── #688: the TransformerDecoder layer is causal ────────────────────────────


def _decoder_lm(**decoder_params):
    """Embedding -> TransformerDecoder -> Linear, the next-token LM from #688."""
    spec = {
        "version": 2,
        "nodes": [
            {"id": "in", "type": "Input", "ports": [{"id": "p_x", "name": "x"}]},
            {"id": "emb", "type": "Embedding", "params": {"num_embeddings": 10, "embedding_dim": 16}},
            {"id": "dec", "type": "TransformerDecoder", "params": {
                "d_model": 16, "nhead": 2, "num_layers": 2, "dim_feedforward": 32, **decoder_params}},
            {"id": "head", "type": "Linear", "params": {"in_features": 16, "out_features": 10}},
            {"id": "out", "type": "Output", "ports": [{"id": "p_y", "name": "y"}]},
        ],
        "edges": [
            {"id": "e1", "source": "in", "sourceHandle": "p_x", "target": "emb"},
            {"id": "e2", "source": "emb", "target": "dec"},
            {"id": "e3", "source": "dec", "target": "head"},
            {"id": "e4", "source": "head", "target": "out", "targetHandle": "p_y"},
        ],
    }
    torch.manual_seed(0)
    return SequentialModelNode().execute({}, {"layers": json.dumps(spec)})["model"].eval()


def _change_at(model, position):
    """How much each earlier position's output moves when token ``position`` changes."""
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    changed = ids.clone()
    changed[0, position] = 9
    with torch.no_grad():
        return (model(ids) - model(changed))[0, :position].abs().max().item()


@pytest.mark.parametrize("position", [1, 5, 7])
def test_a_later_token_never_changes_an_earlier_output(position):
    assert _change_at(_decoder_lm(), position) == 0.0


def test_causal_false_keeps_the_old_full_attention():
    assert _change_at(_decoder_lm(causal=False), 5) > 1e-3


def test_causal_mask_handles_unbatched_input():
    from app.nodes.utility.sequential_modules import TransformerDecoderBlock

    torch.manual_seed(0)
    block = TransformerDecoderBlock(d_model=8, nhead=2, dim_feedforward=16).eval()
    x = torch.randn(5, 8)  # unbatched [T, D]
    y = x.clone()
    y[4] += 1.0
    with torch.no_grad():
        torch.testing.assert_close(block(x)[:4], block(y)[:4])
