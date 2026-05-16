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
