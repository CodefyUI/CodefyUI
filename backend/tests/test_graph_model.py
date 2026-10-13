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


# ── #703: a runtime error names the layer that raised it ────────────────


def _mlp_spec(fc2_in: int):
    return _spec(
        nodes=[
            {"id": "in", "type": "Input", "ports": [{"id": "p_x", "name": "x"}]},
            {"id": "flat", "type": "Flatten"},
            {"id": "fc1", "type": "Linear", "params": {"in_features": 784, "out_features": 1568}},
            {"id": "act", "type": "ReLU"},
            {"id": "fc2", "type": "Linear", "params": {"in_features": fc2_in, "out_features": 10}},
            {"id": "out", "type": "Output", "ports": [{"id": "p_y", "name": "y"}]},
        ],
        edges=[
            {"id": "e1", "source": "in", "sourceHandle": "p_x", "target": "flat"},
            {"id": "e2", "source": "flat", "target": "fc1"},
            {"id": "e3", "source": "fc1", "target": "act"},
            {"id": "e4", "source": "act", "target": "fc2"},
            {"id": "e5", "source": "fc2", "target": "out", "targetHandle": "p_y"},
        ],
    )


def test_a_shape_mismatch_names_the_layer_id_and_type():
    # The issue's repro: the last Linear has in_features=999 while the
    # tensor reaching it has 1568 features.
    model = build_graph_model(_mlp_spec(fc2_in=999))
    with pytest.raises(RuntimeError) as info:
        model(torch.randn(4, 1, 28, 28))
    message = str(info.value)
    assert message.startswith('layer "fc2" (Linear, in_features=999')
    # torch's own wording is kept after the prefix, shapes included.
    assert "1568" in message and "999" in message
    assert isinstance(info.value.__cause__, RuntimeError)
    assert "layer" not in str(info.value.__cause__)


def test_a_correct_model_is_unaffected_by_the_error_wrapper():
    model = build_graph_model(_mlp_spec(fc2_in=1568))
    assert model(torch.randn(4, 1, 28, 28)).shape == (4, 10)


def test_a_merge_shape_mismatch_names_the_merge_node():
    spec = _spec(
        nodes=[
            {"id": "in", "type": "Input", "ports": [{"id": "p_x", "name": "x"}]},
            {"id": "a", "type": "Linear", "params": {"in_features": 4, "out_features": 3}},
            {"id": "b", "type": "Linear", "params": {"in_features": 4, "out_features": 5}},
            {"id": "sum", "type": "Add"},
            {"id": "out", "type": "Output", "ports": [{"id": "p_y", "name": "y"}]},
        ],
        edges=[
            {"id": "e1", "source": "in", "sourceHandle": "p_x", "target": "a"},
            {"id": "e2", "source": "in", "sourceHandle": "p_x", "target": "b"},
            {"id": "e3", "source": "a", "target": "sum"},
            {"id": "e4", "source": "b", "target": "sum"},
            {"id": "e5", "source": "sum", "target": "out", "targetHandle": "p_y"},
        ],
    )
    with pytest.raises(RuntimeError, match=r'^layer "sum" \(Add\): '):
        build_graph_model(spec)(torch.randn(2, 4))


def test_a_wrapper_layer_is_named_by_the_type_the_user_picked():
    spec = _spec(
        nodes=[
            {"id": "in", "type": "Input", "ports": [{"id": "p_x", "name": "x"}]},
            {"id": "rnn", "type": "LSTM", "params": {"input_size": 8, "hidden_size": 4}},
            {"id": "out", "type": "Output", "ports": [{"id": "p_y", "name": "y"}]},
        ],
        edges=[
            {"id": "e1", "source": "in", "sourceHandle": "p_x", "target": "rnn"},
            {"id": "e2", "source": "rnn", "target": "out", "targetHandle": "p_y"},
        ],
    )
    with pytest.raises(Exception, match=r'^layer "rnn" \(LSTM'):
        build_graph_model(spec)(torch.randn(2, 5, 3))


def test_an_out_of_memory_error_passes_through_unchanged(monkeypatch):
    # The engine recognises OOM by type and message and adds its own hints;
    # a prefix in front would be harmless, a different class would not.
    model = build_graph_model(_mlp_spec(fc2_in=1568))
    oom = torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    def boom(_x):
        raise oom

    monkeypatch.setattr(model.layers["fc1"], "forward", boom)
    with pytest.raises(torch.OutOfMemoryError) as info:
        model(torch.randn(1, 1, 28, 28))
    assert info.value is oom


def test_the_error_reaches_the_user_through_the_sequential_model_node():
    import json

    from app.nodes.utility.sequential_node import SequentialModelNode

    model = SequentialModelNode().execute(
        {}, {"layers": json.dumps(_mlp_spec(fc2_in=999))})["model"]
    with pytest.raises(RuntimeError, match=r'layer "fc2" \(Linear, in_features=999'):
        model(torch.randn(2, 1, 28, 28))


# ── #703: the layer options a CNN needs ──────────────────────────────────


def _single_layer(layer: dict):
    return build_graph_model(_spec(
        nodes=[
            {"id": "in", "type": "Input", "ports": [{"id": "p_x", "name": "x"}]},
            {"id": "l", **layer},
            {"id": "out", "type": "Output", "ports": [{"id": "p_y", "name": "y"}]},
        ],
        edges=[
            {"id": "e1", "source": "in", "sourceHandle": "p_x", "target": "l"},
            {"id": "e2", "source": "l", "target": "out", "targetHandle": "p_y"},
        ],
    ))


def test_a_depthwise_conv_builds_from_the_layer_spec():
    model = _single_layer({"type": "Conv2d", "params": {
        "in_channels": 8, "out_channels": 8, "kernel_size": 3, "padding": 2,
        "groups": 8, "dilation": 2, "bias": False}})
    conv = model.layers["l"]
    assert conv.groups == 8 and conv.dilation == (2, 2) and conv.bias is None
    # One 3x3 filter per channel: 8 * 1 * 3 * 3 weights.
    assert conv.weight.shape == (8, 1, 3, 3)
    x = torch.randn(2, 8, 10, 10)
    torch.testing.assert_close(
        model(x), torch.nn.functional.conv2d(x, conv.weight, None, padding=2, dilation=2, groups=8))


def test_leaky_relu_takes_its_negative_slope():
    model = _single_layer({"type": "LeakyReLU", "params": {"negative_slope": 0.2}})
    x = torch.tensor([-1.0, 2.0])
    torch.testing.assert_close(model(x), torch.tensor([-0.2, 2.0]))


def test_softmax_takes_its_dim():
    model = _single_layer({"type": "Softmax", "params": {"dim": 0}})
    x = torch.randn(3, 4)
    torch.testing.assert_close(model(x), torch.softmax(x, dim=0))


def test_activations_without_params_build_as_before():
    leaky = _single_layer({"type": "LeakyReLU"}).layers["l"]
    soft = _single_layer({"type": "Softmax"}).layers["l"]
    assert leaky.negative_slope == 0.01
    assert soft.dim == -1


def test_dropout2d_zeroes_whole_channels():
    model = _single_layer({"type": "Dropout2d", "params": {"p": 0.5}})
    torch.manual_seed(0)
    y = model.train()(torch.ones(4, 16, 5, 5))
    per_channel = y.flatten(2)
    # Every channel is either all zero or all scaled by 1 / (1 - p).
    for channel in per_channel.reshape(-1, 25):
        assert torch.all(channel == 0) or torch.allclose(channel, torch.full((25,), 2.0))
    assert (per_channel.sum(-1) == 0).any()


# ── operand order of Subtract / Concat / Stack follows the target handle ──


def _two_branch_merge(merge_type, edge_order, handles, params=None):
    """relu(x) and sigmoid(x) into one merge; edges listed in *edge_order*.

    *handles* maps the branch id to the merge input it is wired to (None for
    an old spec without handles).
    """
    branch_edges = {
        b: {"id": f"e_{b}", "source": b, "sourceHandle": None, "target": "m",
            "targetHandle": handles[b]}
        for b in ("relu", "sig")
    }
    return _spec(
        nodes=[
            {"id": "in", "type": "Input", "ports": [{"id": "p_in", "name": "x"}]},
            {"id": "relu", "type": "ReLU"},
            {"id": "sig", "type": "Sigmoid"},
            {"id": "m", "type": merge_type, "params": params or {}},
            {"id": "out", "type": "Output", "ports": [{"id": "p_out", "name": "y"}]},
        ],
        edges=[
            {"id": "e1", "source": "in", "sourceHandle": "p_in", "target": "relu"},
            {"id": "e2", "source": "in", "sourceHandle": "p_in", "target": "sig"},
            *[branch_edges[b] for b in edge_order],
            {"id": "e9", "source": "m", "target": "out", "targetHandle": "p_out"},
        ],
    )


@pytest.mark.parametrize("edge_order", [("relu", "sig"), ("sig", "relu")])
def test_subtract_takes_operands_in_handle_order_whatever_the_edge_order(edge_order):
    x = torch.tensor([[2.0]])
    expected = torch.relu(x) - torch.sigmoid(x)  # 1.1192
    spec = _two_branch_merge("Subtract", edge_order, {"relu": "a", "sig": "b"})
    assert torch.allclose(build_graph_model(spec)(x), expected)


def test_subtract_handles_swapped_swap_the_sign():
    x = torch.tensor([[2.0]])
    spec = _two_branch_merge("Subtract", ("relu", "sig"), {"relu": "b", "sig": "a"})
    assert torch.allclose(build_graph_model(spec)(x), torch.sigmoid(x) - torch.relu(x))


@pytest.mark.parametrize("merge_type", ["Concat", "Stack"])
@pytest.mark.parametrize("edge_order", [("relu", "sig"), ("sig", "relu")])
def test_concat_and_stack_take_operands_in_handle_order(merge_type, edge_order):
    x = torch.randn(2, 3)
    spec = _two_branch_merge(
        merge_type, edge_order, {"relu": "in1", "sig": "in0"}, params={"dim": 1},
    )
    op = torch.cat if merge_type == "Concat" else torch.stack
    assert torch.allclose(build_graph_model(spec)(x), op([torch.sigmoid(x), torch.relu(x)], dim=1))


def test_indexed_handles_order_numerically_past_nine():
    """in10 comes after in2, which a string sort would get wrong."""
    from app.nodes.utility.graph_model import _order_operands

    edges_in = [("c", None, "in10"), ("a", None, "in0"), ("b", None, "in2")]
    ordered = _order_operands({"id": "cat", "type": "Concat"}, edges_in)
    assert [src for src, _sh, _th in ordered] == ["a", "b", "c"]


@pytest.mark.parametrize("edge_order", [("relu", "sig"), ("sig", "relu")])
def test_old_spec_without_handles_keeps_edge_list_order(edge_order):
    """Saved models from before the handles existed build as they always did."""
    x = torch.tensor([[2.0]])
    spec = _two_branch_merge("Subtract", edge_order, {"relu": None, "sig": None})
    first, second = (torch.relu, torch.sigmoid) if edge_order[0] == "relu" else (torch.sigmoid, torch.relu)
    assert torch.allclose(build_graph_model(spec)(x), first(x) - second(x))


def test_two_edges_into_one_merge_handle_name_the_layer():
    spec = _two_branch_merge("Subtract", ("relu", "sig"), {"relu": "a", "sig": "a"})
    with pytest.raises(ValueError, match=r'Subtract layer "m": two edges go into input "a"'):
        build_graph_model(spec)
