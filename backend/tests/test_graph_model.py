# backend/tests/test_graph_model.py

import pytest
import torch

from app.nodes.utility.graph_model import build_graph_model


def _spec(nodes, edges):
    return {"version": 2, "nodes": nodes, "edges": edges}


def test_linear_dag_single_layer():
    spec = _spec(
        nodes=[
            {"id": "in", "type": "Input", "ports": [{"id": "p_in", "name": "x"}]},
            {"id": "lin", "type": "Linear", "params": {"in_features": 4, "out_features": 2}},
            {"id": "out", "type": "Output", "ports": [{"id": "p_out", "name": "y"}]},
        ],
        edges=[
            {"id": "e1", "source": "in", "sourceHandle": "p_in", "target": "lin", "targetHandle": None},
            {"id": "e2", "source": "lin", "sourceHandle": None, "target": "out", "targetHandle": "p_out"},
        ],
    )

    model = build_graph_model(spec)
    x = torch.randn(3, 4)
    y = model(x)

    assert isinstance(y, torch.Tensor)
    assert y.shape == (3, 2)


def test_residual_block_with_add():
    spec = _spec(
        nodes=[
            {"id": "in", "type": "Input", "ports": [{"id": "p_in", "name": "x"}]},
            {"id": "lin1", "type": "Linear", "params": {"in_features": 4, "out_features": 4}},
            {"id": "relu", "type": "ReLU"},
            {"id": "lin2", "type": "Linear", "params": {"in_features": 4, "out_features": 4}},
            {"id": "add", "type": "Add"},
            {"id": "out", "type": "Output", "ports": [{"id": "p_out", "name": "y"}]},
        ],
        edges=[
            {"id": "e1", "source": "in", "sourceHandle": "p_in", "target": "lin1"},
            {"id": "e2", "source": "lin1", "target": "relu"},
            {"id": "e3", "source": "relu", "target": "lin2"},
            {"id": "e4", "source": "lin2", "target": "add"},
            {"id": "e5", "source": "in", "sourceHandle": "p_in", "target": "add"},  # skip
            {"id": "e6", "source": "add", "target": "out", "targetHandle": "p_out"},
        ],
    )
    model = build_graph_model(spec)
    x = torch.randn(2, 4)
    y = model(x)
    assert y.shape == (2, 4)
    # Skip means at least one parameter pathway exists; verify gradient flows
    y.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_multi_input():
    spec = _spec(
        nodes=[
            {"id": "in", "type": "Input", "ports": [
                {"id": "p_a", "name": "a"},
                {"id": "p_b", "name": "b"},
            ]},
            {"id": "add", "type": "Add"},
            {"id": "out", "type": "Output", "ports": [{"id": "p_y", "name": "y"}]},
        ],
        edges=[
            {"id": "e1", "source": "in", "sourceHandle": "p_a", "target": "add"},
            {"id": "e2", "source": "in", "sourceHandle": "p_b", "target": "add"},
            {"id": "e3", "source": "add", "target": "out", "targetHandle": "p_y"},
        ],
    )
    model = build_graph_model(spec)
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([10.0, 20.0])
    y = model(a=a, b=b)
    assert torch.allclose(y, torch.tensor([11.0, 22.0]))


def test_multi_output_returns_dict():
    spec = _spec(
        nodes=[
            {"id": "in", "type": "Input", "ports": [{"id": "p_x", "name": "x"}]},
            {"id": "lin", "type": "Linear", "params": {"in_features": 4, "out_features": 2}},
            {"id": "relu", "type": "ReLU"},
            {"id": "out", "type": "Output", "ports": [
                {"id": "p_raw", "name": "raw"},
                {"id": "p_act", "name": "activated"},
            ]},
        ],
        edges=[
            {"id": "e1", "source": "in", "sourceHandle": "p_x", "target": "lin"},
            {"id": "e2", "source": "lin", "target": "relu"},
            {"id": "e3", "source": "lin", "target": "out", "targetHandle": "p_raw"},
            {"id": "e4", "source": "relu", "target": "out", "targetHandle": "p_act"},
        ],
    )
    model = build_graph_model(spec)
    x = torch.randn(3, 4)
    out = model(x=x)
    assert isinstance(out, dict)
    assert set(out.keys()) == {"raw", "activated"}
    assert out["raw"].shape == (3, 2)
    assert out["activated"].shape == (3, 2)


def test_unet_skip_with_concat():
    spec = _spec(
        nodes=[
            {"id": "in", "type": "Input", "ports": [{"id": "p_x", "name": "x"}]},
            {"id": "down1", "type": "Conv2d", "params": {"in_channels": 1, "out_channels": 4, "kernel_size": 3, "padding": 1}},
            {"id": "pool", "type": "MaxPool2d", "params": {"kernel_size": 2, "stride": 2}},
            {"id": "deep", "type": "Conv2d", "params": {"in_channels": 4, "out_channels": 4, "kernel_size": 3, "padding": 1}},
            {"id": "up", "type": "ConvTranspose2d", "params": {"in_channels": 4, "out_channels": 4, "kernel_size": 2, "stride": 2}},
            {"id": "concat", "type": "Concat", "params": {"dim": 1}},
            {"id": "final", "type": "Conv2d", "params": {"in_channels": 8, "out_channels": 1, "kernel_size": 1}},
            {"id": "out", "type": "Output", "ports": [{"id": "p_y", "name": "y"}]},
        ],
        edges=[
            {"id": "e1", "source": "in", "sourceHandle": "p_x", "target": "down1"},
            {"id": "e2", "source": "down1", "target": "pool"},
            {"id": "e3", "source": "pool", "target": "deep"},
            {"id": "e4", "source": "deep", "target": "up"},
            {"id": "e5", "source": "up", "target": "concat"},
            {"id": "e6", "source": "down1", "target": "concat"},  # skip
            {"id": "e7", "source": "concat", "target": "final"},
            {"id": "e8", "source": "final", "target": "out", "targetHandle": "p_y"},
        ],
    )
    model = build_graph_model(spec)
    x = torch.randn(2, 1, 16, 16)
    y = model(x)
    assert y.shape == (2, 1, 16, 16)


def test_validation_cycle():
    spec = _spec(
        nodes=[
            {"id": "in", "type": "Input", "ports": [{"id": "p_x", "name": "x"}]},
            {"id": "a", "type": "Linear", "params": {"in_features": 4, "out_features": 4}},
            {"id": "b", "type": "Linear", "params": {"in_features": 4, "out_features": 4}},
            {"id": "out", "type": "Output", "ports": [{"id": "p_y", "name": "y"}]},
        ],
        edges=[
            {"id": "e1", "source": "in", "sourceHandle": "p_x", "target": "a"},
            {"id": "e2", "source": "a", "target": "b"},
            {"id": "e3", "source": "b", "target": "a"},  # cycle
            {"id": "e4", "source": "b", "target": "out", "targetHandle": "p_y"},
        ],
    )
    with pytest.raises(ValueError, match="cycle"):
        build_graph_model(spec)


def test_validation_no_input():
    spec = _spec(
        nodes=[
            {"id": "out", "type": "Output", "ports": [{"id": "p_y", "name": "y"}]},
        ],
        edges=[],
    )
    with pytest.raises(ValueError, match="exactly one Input"):
        build_graph_model(spec)


def test_validation_no_output():
    spec = _spec(
        nodes=[
            {"id": "in", "type": "Input", "ports": [{"id": "p_x", "name": "x"}]},
        ],
        edges=[],
    )
    with pytest.raises(ValueError, match="exactly one Output"):
        build_graph_model(spec)


def test_validation_duplicate_input_port_names():
    spec = _spec(
        nodes=[
            {"id": "in", "type": "Input", "ports": [
                {"id": "p1", "name": "x"},
                {"id": "p2", "name": "x"},
            ]},
            {"id": "out", "type": "Output", "ports": [{"id": "p_y", "name": "y"}]},
        ],
        edges=[],
    )
    with pytest.raises(ValueError, match="Input port names must be unique"):
        build_graph_model(spec)


def test_validation_unconnected_output_port():
    spec = _spec(
        nodes=[
            {"id": "in", "type": "Input", "ports": [{"id": "p_x", "name": "x"}]},
            {"id": "lin", "type": "Linear", "params": {"in_features": 4, "out_features": 2}},
            {"id": "out", "type": "Output", "ports": [{"id": "p_y", "name": "y"}]},
        ],
        edges=[
            {"id": "e1", "source": "in", "sourceHandle": "p_x", "target": "lin"},
            # no edge into output
        ],
    )
    with pytest.raises(ValueError, match="Output port 'y' must have exactly one incoming edge"):
        build_graph_model(spec)


def test_validation_plain_layer_multi_input():
    spec = _spec(
        nodes=[
            {"id": "in", "type": "Input", "ports": [
                {"id": "p_a", "name": "a"},
                {"id": "p_b", "name": "b"},
            ]},
            {"id": "lin", "type": "Linear", "params": {"in_features": 4, "out_features": 2}},
            {"id": "out", "type": "Output", "ports": [{"id": "p_y", "name": "y"}]},
        ],
        edges=[
            {"id": "e1", "source": "in", "sourceHandle": "p_a", "target": "lin"},
            {"id": "e2", "source": "in", "sourceHandle": "p_b", "target": "lin"},  # plain layer can't take 2 inputs
            {"id": "e3", "source": "lin", "target": "out", "targetHandle": "p_y"},
        ],
    )
    with pytest.raises(ValueError, match="must have exactly 1 incoming edge"):
        build_graph_model(spec)


def test_sequential_node_executes_v2_spec():
    from app.nodes.utility.sequential_node import SequentialModelNode
    import json as _json

    spec = _spec(
        nodes=[
            {"id": "in", "type": "Input", "ports": [{"id": "p_x", "name": "x"}]},
            {"id": "lin", "type": "Linear", "params": {"in_features": 4, "out_features": 2}},
            {"id": "out", "type": "Output", "ports": [{"id": "p_y", "name": "y"}]},
        ],
        edges=[
            {"id": "e1", "source": "in", "sourceHandle": "p_x", "target": "lin"},
            {"id": "e2", "source": "lin", "target": "out", "targetHandle": "p_y"},
        ],
    )
    node = SequentialModelNode()
    result = node.execute(inputs={}, params={"layers": _json.dumps(spec)})
    model = result["model"]
    y = model(torch.randn(3, 4))
    assert y.shape == (3, 2)


# ── #694: an activation must not rewrite a value another consumer reads ──


def _pre_activation_skip(activation: str) -> dict:
    """``Input -> <activation> -> Add`` plus ``Input -> Add``."""
    return _spec(
        nodes=[
            {"id": "in", "type": "Input", "ports": [{"id": "p_x", "name": "x"}]},
            {"id": "act", "type": activation},
            {"id": "add", "type": "Add"},
            {"id": "out", "type": "Output", "ports": [{"id": "p_y", "name": "y"}]},
        ],
        edges=[
            {"id": "e1", "source": "in", "sourceHandle": "p_x", "target": "act"},
            {"id": "e2", "source": "act", "target": "add"},
            {"id": "e3", "source": "in", "sourceHandle": "p_x", "target": "add"},
            {"id": "e4", "source": "add", "target": "out", "targetHandle": "p_y"},
        ],
    )


@pytest.mark.parametrize(
    "activation, reference",
    [
        ("ReLU", torch.nn.functional.relu),
        ("LeakyReLU", torch.nn.functional.leaky_relu),
        ("ELU", torch.nn.functional.elu),
        ("SiLU", torch.nn.functional.silu),
        ("Mish", torch.nn.functional.mish),
        ("SELU", torch.nn.functional.selu),
        ("Hardswish", torch.nn.functional.hardswish),
    ],
)
def test_a_skip_taken_before_the_activation_keeps_the_raw_value(
    activation, reference,
):
    """The issue's repro: ``relu(x) + x``, and the caller's ``x`` unchanged.

    Measured before the fix for ReLU: ``[0, 4, 0]`` for ``[-1, 2, -3]``,
    and the input came back as ``[0, 2, 0]``.
    """
    model = build_graph_model(_pre_activation_skip(activation))
    x = torch.tensor([-1.0, 2.0, -3.0, 0.5])
    before = x.clone()

    y = model(x)

    torch.testing.assert_close(y, reference(before) + before)
    torch.testing.assert_close(x, before)


def test_a_residual_block_with_a_pre_activation_identity_matches_pytorch():
    """``conv -> bn -> (ReLU -> conv) + bn``, forward and gradients.

    The reference runs the graph model's own layers in plain PyTorch, so
    the only difference left is how the graph hands values around. Before
    the fix the identity path came out fully ReLU'd (max |diff| 2.15 at
    this scale) and training ran without an autograd error.
    """
    spec = _spec(
        nodes=[
            {"id": "in", "type": "Input", "ports": [{"id": "p_x", "name": "x"}]},
            {"id": "c1", "type": "Conv2d",
             "params": {"in_channels": 2, "out_channels": 4, "kernel_size": 3,
                        "padding": 1}},
            {"id": "bn", "type": "BatchNorm2d", "params": {"num_features": 4}},
            {"id": "relu", "type": "ReLU"},
            {"id": "c2", "type": "Conv2d",
             "params": {"in_channels": 4, "out_channels": 4, "kernel_size": 3,
                        "padding": 1}},
            {"id": "add", "type": "Add"},
            {"id": "out", "type": "Output", "ports": [{"id": "p_y", "name": "y"}]},
        ],
        edges=[
            {"id": "e1", "source": "in", "sourceHandle": "p_x", "target": "c1"},
            {"id": "e2", "source": "c1", "target": "bn"},
            {"id": "e3", "source": "bn", "target": "relu"},
            {"id": "e4", "source": "relu", "target": "c2"},
            {"id": "e5", "source": "c2", "target": "add"},
            {"id": "e6", "source": "bn", "target": "add"},  # identity
            {"id": "e7", "source": "add", "target": "out", "targetHandle": "p_y"},
        ],
    )
    torch.manual_seed(0)
    model = build_graph_model(spec)
    layers = model.layers
    x = torch.randn(3, 2, 5, 5)

    def reference(inp):
        h = layers["bn"](layers["c1"](inp))
        return layers["c2"](torch.relu(h)) + h

    model.eval()  # BN running stats stay put between the two passes
    expected = reference(x)
    expected.square().sum().backward()
    expected_grads = {name: p.grad.clone() for name, p in model.named_parameters()}
    model.zero_grad()

    got = model(x)
    got.square().sum().backward()

    torch.testing.assert_close(got, expected)
    for name, p in model.named_parameters():
        torch.testing.assert_close(p.grad, expected_grads[name], msg=name)
