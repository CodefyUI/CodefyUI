"""Input dtypes, param spellings and hidden state in the layer and tensor-op
nodes (2026-10-13 functional-correctness audit).

Each test pins one observed failure: the number or message the node gave
before, and the one it gives now.
"""

from __future__ import annotations

import asyncio
import base64

import pytest
import torch
import torch.nn.functional as F

from app.core.execution_context import ExecutionContext
from app.core.graph_engine import execute_graph
from app.core.node_state_store import NodeStateStore
from app.nodes.cnn.activation_node import ActivationNode
from app.nodes.cnn.avgpool2d_node import AvgPool2dNode
from app.nodes.cnn.batchnorm_node import BatchNormNode
from app.nodes.cnn.conv2d_explicit_node import Conv2dExplicitNode
from app.nodes.cnn.conv2d_node import Conv2dNode
from app.nodes.cnn.dropout_node import DropoutNode
from app.nodes.cnn.maxpool2d_node import MaxPool2dNode
from app.nodes.normalization.batchnorm1d_node import BatchNorm1dNode
from app.nodes.normalization.instancenorm2d_node import InstanceNorm2dNode
from app.nodes.normalization.layernorm_node import LayerNormNode
from app.nodes.rnn.lstm_node import LSTMNode
from app.nodes.tensor_ops.add_node import AddNode
from app.nodes.tensor_ops.mean_node import MeanNode
from app.nodes.tensor_ops.softmax_node import SoftmaxNode
from app.nodes.tensor_ops.split_node import SplitNode
from app.nodes.tensor_ops.tensor_create_node import TensorCreateNode
from app.nodes.tensor_ops.tensor_stats_node import TensorStatsNode
from app.nodes.tensor_ops.unsqueeze_node import UnsqueezeNode
from app.nodes.transformer.attention_node import MultiHeadAttentionNode
from app.nodes.transformer.moe_layer_node import MoELayerNode
from app.nodes.utility.concat_node import ConcatNode
from app.nodes.utility.embedding_node import EmbeddingNode
from app.nodes.utility.linear_node import LinearNode
from app.nodes.utility.python_script_node import PythonScriptNode
from app.nodes.utility.scatter_plot_2d_node import ScatterPlot2DNode
from app.nodes.utility.visualize_node import VisualizeNode


def _ctx(node_id: str = "n", store: NodeStateStore | None = None, device: str = "cpu"):
    return ExecutionContext(
        graph_id="g",
        weights_persistent=True,
        node_state_store=store if store is not None else NodeStateStore(),
        current_node_id=node_id,
        device=device,
    )


def _node(node_id: str, node_type: str, params: dict) -> dict:
    return {"id": node_id, "type": node_type, "data": {"params": params}}


def _wire(source: str, source_port: str, target: str, target_port: str) -> dict:
    return {
        "id": f"{source}-{target}-{target_port}", "source": source,
        "sourceHandle": source_port, "target": target, "targetHandle": target_port,
    }


# -- PythonScript: in-place edits of an input -----------------------------------


def test_python_script_refuses_an_in_place_edit_that_leaks_into_a_sibling():
    """TensorCreate(ones) fed a PythonScript doing ``x += 1`` and a
    TensorStats. Before: the upstream output became [2, 2, 2] and the
    sibling's mean was 2.0 or 1.0 depending on scheduling."""
    code = "def run(inputs, params):\n    x = inputs['in1']\n    x += 1\n    return {'out1': x}\n"
    nodes = [
        _node("start", "Start", {}),
        _node("tc", "TensorCreate", {"shape": "3", "fill": "ones"}),
        _node("py", "PythonScript", {"code": code}),
        _node("st", "TensorStats", {}),
    ]
    edges = [
        {"id": "t", "source": "start", "target": "tc", "sourceHandle": "trigger", "type": "trigger"},
        _wire("tc", "tensor", "py", "in1"),
        _wire("tc", "tensor", "st", "tensor"),
    ]
    with pytest.raises(RuntimeError, match=r"changed the input tensor `in1` in place.*x = x \+ 1"):
        asyncio.run(execute_graph(nodes, edges, context=ExecutionContext(device="cpu", seed=1)))


def test_python_script_refuses_an_in_place_edit_inside_a_list_input():
    code = "def run(inputs, params):\n    inputs['in1'][1].mul_(10)\n    return 0\n"
    tensors = [torch.ones(2), torch.ones(2)]
    with pytest.raises(RuntimeError, match=r"`in1`\[1\] in place"):
        PythonScriptNode().execute({"in1": tensors}, {"code": code})


def test_python_script_still_accepts_a_new_tensor_and_a_clone():
    code = (
        "def run(inputs, params):\n"
        "    x = inputs['in1']\n"
        "    y = x.clone()\n"
        "    y += 1\n"
        "    return {'out1': x + 1, 'out2': y}\n"
    )
    src = torch.ones(3)
    out = PythonScriptNode().execute({"in1": src}, {"code": code, "output_ports": 2})
    assert out["out1"].tolist() == [2.0, 2.0, 2.0]
    assert out["out2"].tolist() == [2.0, 2.0, 2.0]
    assert src.tolist() == [1.0, 1.0, 1.0]


# -- Conv2dExplicit on integer images ---------------------------------------------


def test_conv2d_explicit_convolves_uint8_in_float():
    """Before: Sharpen3x3 on uint8 arange(25) wrapped mod 256 ([250, 253, 255, 1, 8])."""
    x = torch.arange(25, dtype=torch.uint8).reshape(1, 1, 5, 5)
    out = Conv2dExplicitNode().execute({"tensor": x}, {"preset": "Sharpen3x3"})["tensor"]
    kernel = torch.tensor([[0., -1., 0.], [-1., 5., -1.], [0., -1., 0.]]).reshape(1, 1, 3, 3)
    assert out.dtype == torch.float32
    assert torch.equal(out, F.conv2d(x.float(), kernel, padding=1))
    assert out[0, 0, 0].tolist() == [-6.0, -3.0, -1.0, 1.0, 8.0]


def test_conv2d_explicit_box_blur_on_int64_is_not_truncated():
    """Before: a Custom 1/9 box blur on an int64 input returned 0, not 4.0."""
    x = torch.arange(9, dtype=torch.int64).reshape(1, 1, 3, 3)
    params = {"preset": "Custom", "kernel_size": 3, "weights": [[1 / 9] * 3] * 3, "padding": 0}
    out = Conv2dExplicitNode().execute({"tensor": x}, params)["tensor"]
    assert out.item() == pytest.approx(4.0)


@pytest.mark.parametrize("text", [
    "[[0, -1, 0], [-1, 5, -1], [0, -1, 0]]",
    "0 -1 0\n-1 5 -1\n0 -1 0",
])
def test_conv2d_explicit_reads_a_weights_matrix_typed_as_text(text):
    """Before: a JSON matrix string failed with "`weights` has 1 elements"."""
    x = torch.rand(1, 2, 5, 5)
    by_text = Conv2dExplicitNode().execute(
        {"tensor": x}, {"preset": "Custom", "kernel_size": 3, "weights": text})["tensor"]
    by_preset = Conv2dExplicitNode().execute({"tensor": x}, {"preset": "Sharpen3x3"})["tensor"]
    assert torch.equal(by_text, by_preset)


def test_conv2d_explicit_names_a_non_numeric_weight():
    with pytest.raises(ValueError, match="Conv2dExplicit: `weights` must be a matrix of numbers; 'x'"):
        Conv2dExplicitNode().execute(
            {"tensor": torch.rand(1, 1, 3, 3)},
            {"preset": "Custom", "kernel_size": 1, "weights": "x"})


# -- TensorCreate full -----------------------------------------------------------


def test_tensor_create_full_with_a_whole_number_is_float():
    """The canvas saves 2.0 as 2. Before: int64, and requires_grad failed."""
    out = TensorCreateNode().execute({}, {"shape": "2", "fill": "full", "value": 3})["tensor"]
    assert out.dtype == torch.float32
    assert out.tolist() == [3.0, 3.0]
    grad = TensorCreateNode().execute(
        {}, {"shape": "2", "fill": "full", "value": 3, "requires_grad": True})["tensor"]
    assert grad.requires_grad and grad.dtype == torch.float32


# -- float64 into module-backed layers on CPU ---------------------------------------


@pytest.mark.parametrize("node_cls, params, shape", [
    (LinearNode, {"in_features": 4, "out_features": 3}, (2, 4)),
    (Conv2dNode, {"in_channels": 1, "out_channels": 2}, (1, 1, 5, 5)),
    (BatchNormNode, {"num_features": 2}, (2, 2, 3, 3)),
    (LayerNormNode, {"normalized_shape": "4"}, (2, 4)),
    (LSTMNode, {"input_size": 4, "hidden_size": 3}, (2, 5, 4)),
])
def test_float64_input_runs_on_cpu_as_it_does_on_mps(node_cls, params, shape):
    """Before: ``mat1 and mat2 must have the same dtype, but got Double and Float``
    on CPU, while MPS downcast the input and ran."""
    x = torch.randn(*shape, dtype=torch.float64)
    out = node_cls().execute({"tensor": x}, params, context=_ctx())
    first = out.get("tensor", out.get("output"))
    assert first.dtype == torch.float32


def test_float64_query_key_value_run_in_attention():
    x = torch.randn(2, 3, 8, dtype=torch.float64)
    out = MultiHeadAttentionNode().execute(
        {"query": x, "key": x, "value": x},
        {"embed_dim": 8, "num_heads": 2, "batch_first": True}, context=_ctx())
    assert out["output"].dtype == torch.float32


def test_integer_input_to_a_layer_names_the_node_and_port():
    with pytest.raises(ValueError, match="Linear: the `tensor` input is int64 but this layer needs floating-point"):
        LinearNode().execute(
            {"tensor": torch.ones(2, 4, dtype=torch.int64)},
            {"in_features": 4, "out_features": 3}, context=_ctx())


def test_embedding_names_float_indices():
    with pytest.raises(ValueError, match="Embedding: the `tensor` input must hold integer token ids, got float32"):
        EmbeddingNode().execute(
            {"tensor": torch.tensor([0.0, 1.0])},
            {"num_embeddings": 4, "embedding_dim": 2}, context=_ctx())


# -- Activation prelu ----------------------------------------------------------------


def test_prelu_output_does_not_require_grad():
    """Before: nn.PReLU()(x) carried requires_grad=True out of a pure op."""
    x = torch.tensor([-2.0, 3.0])
    out = ActivationNode().execute({"tensor": x}, {"function": "prelu"})["tensor"]
    assert out.tolist() == [-0.5, 3.0]
    assert out.requires_grad is False


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_prelu_runs_on_mps():
    """Before: ``Placeholder storage has not been allocated on MPS device!``."""
    x = torch.tensor([-2.0, 3.0], device="mps")
    out = ActivationNode().execute({"tensor": x}, {"function": "prelu"})["tensor"]
    assert out.cpu().tolist() == [-0.5, 3.0]


# -- Add and Mean on integer tensors -----------------------------------------------------


def test_add_integer_tensors_with_the_float_default_alpha():
    """Before: ``For integral input tensors, argument alpha must not be a floating point number``."""
    a, b = torch.tensor([1, 2]), torch.tensor([3, 4])
    assert AddNode().execute({"tensor_a": a, "tensor_b": b}, {"alpha": 1.0})["tensor"].tolist() == [4, 6]
    assert AddNode().execute({"tensor_a": a, "tensor_b": b}, {"alpha": 2.0})["tensor"].tolist() == [7, 10]
    half = AddNode().execute({"tensor_a": a, "tensor_b": b}, {"alpha": 0.5})["tensor"]
    assert half.tolist() == [2.5, 4.0]


@pytest.mark.parametrize("tensor, expected", [
    (torch.tensor([1, 0, 1, 1]), 0.75),
    (torch.tensor([True, False, True, True]), 0.75),
])
def test_mean_of_integer_and_bool_tensors(tensor, expected):
    """Before: ``mean(): could not infer output dtype``."""
    out = MeanNode().execute({"tensor": tensor}, {"dim": "-1"})["tensor"]
    assert out.item() == pytest.approx(expected)


@pytest.mark.parametrize("dim, shape", [
    (-1, (2, 3, 4)),
    ("(2,3)", (2, 3)),
    ("2 3", (2, 3)),
    ("", ()),
])
def test_mean_dim_spellings(dim, shape):
    """Before: an int dim raised ``'int' object has no attribute 'split'``;
    blank and "(2,3)" raised ``invalid literal for int()``."""
    x = torch.rand(2, 3, 4, 5)
    out = MeanNode().execute({"tensor": x}, {"dim": dim})["tensor"]
    assert tuple(out.shape) == shape


# -- BatchNorm hidden state ------------------------------------------------------------


@pytest.mark.parametrize("node_cls, shape", [
    (BatchNormNode, (3, 2, 4, 4)),
    (BatchNorm1dNode, (5, 2)),
])
def test_batchnorm_keeps_no_running_statistics(node_cls, shape):
    """Before: running_mean drifted on every canvas run with persistence on
    (``num_batches_tracked`` 2 after two runs)."""
    store = NodeStateStore()
    node = node_cls()
    x = torch.randn(*shape) * 3 + 2
    params = {"num_features": 2}
    first = node.execute({"tensor": x}, params, context=_ctx("bn", store))["tensor"]
    second = node.execute({"tensor": x}, params, context=_ctx("bn", store))["tensor"]
    module = node.get_or_build_module(_ctx("bn", store), params)
    assert module.running_mean is None and module.num_batches_tracked is None
    reference = F.batch_norm(x, None, None, module.weight, module.bias, training=True)
    assert torch.allclose(first, reference) and torch.equal(first, second)


@pytest.mark.parametrize("node_cls, shape", [
    (BatchNormNode, (1, 2, 1, 1)),
    (BatchNorm1dNode, (1, 2)),
])
def test_batchnorm_names_a_single_value_per_channel(node_cls, shape):
    """Before: torch's ``Expected more than 1 value per channel when training``."""
    with pytest.raises(ValueError, match="one value per channel"):
        node_cls().execute({"tensor": torch.randn(*shape)}, {"num_features": 2}, context=_ctx())


# -- Visualize / ScatterPlot2D ---------------------------------------------------------------


def _is_png(image: str) -> bool:
    return base64.b64decode(image)[:4] == b"\x89PNG"


def test_visualize_tiles_an_n_h_w_batch():
    """Before: ``Invalid shape (10, 28, 28) for image data``."""
    out = VisualizeNode().execute({"data": torch.rand(10, 28, 28)}, {"plot_type": "image"})
    assert _is_png(out["image"])


def test_visualize_and_scatter_accept_bfloat16():
    """Before: ``Got unsupported ScalarType BFloat16``."""
    assert _is_png(VisualizeNode().execute(
        {"data": torch.randn(5, dtype=torch.bfloat16)}, {"plot_type": "line"})["image"])
    assert _is_png(ScatterPlot2DNode().execute(
        {"points": torch.randn(5, 2, dtype=torch.bfloat16)}, {})["image"])


# -- param coercion -------------------------------------------------------------------------


def test_conv2d_reads_string_ints_and_a_false_word():
    """Before: kernel_size="3" -> ``empty(): argument 'size' failed to unpack``;
    bias="false" silently kept the bias."""
    module = Conv2dNode().build_module(
        {"in_channels": 4, "out_channels": 2, "kernel_size": "3", "bias": "false"})
    assert module.kernel_size == (3, 3)
    assert module.bias is None


def test_pools_dropout_and_dim_params_read_numeric_strings():
    x = torch.rand(2, 4, 9, 9)
    assert torch.equal(
        MaxPool2dNode().execute({"tensor": x}, {"kernel_size": "3", "stride": 2.0})["tensor"],
        F.max_pool2d(x, 3, 2))
    assert torch.equal(
        AvgPool2dNode().execute({"tensor": x}, {"kernel_size": "3", "stride": 2, "padding": 0})["tensor"],
        F.avg_pool2d(x, 3, 2))
    assert DropoutNode().execute({"tensor": torch.ones(4)}, {"p": "0"})["tensor"].tolist() == [1.0] * 4
    assert torch.equal(SoftmaxNode().execute({"tensor": x}, {"dim": "1"})["tensor"], F.softmax(x, dim=1))
    assert len(SplitNode().execute({"tensor": x}, {"chunks": 2, "dim": "0"})) == 2
    assert UnsqueezeNode().execute({"tensor": x}, {"dim": "0"})["tensor"].shape == (1, 2, 4, 9, 9)
    assert ConcatNode().execute({"tensor_a": x, "tensor_b": x}, {"dim": "1"})["tensor"].shape == (2, 8, 9, 9)


def test_maxpool2d_names_a_2d_input_and_a_bad_kernel():
    with pytest.raises(ValueError, match=r"MaxPool2d: the input must be \(C, H, W\).*Unsqueeze"):
        MaxPool2dNode().execute({"tensor": torch.rand(8, 8)}, {})
    with pytest.raises(ValueError, match="MaxPool2d: kernel_size must be at least 1"):
        MaxPool2dNode().execute({"tensor": torch.rand(1, 8, 8)}, {"kernel_size": 0})


@pytest.mark.parametrize("shape", [5, "(4, 5)", "4 5"])
def test_layernorm_normalized_shape_spellings(shape):
    """Before: an int raised ``'int' object has no attribute 'split'`` and
    "(4, 5)" raised ``invalid literal for int()``."""
    x = torch.rand(2, 3, 4, 5)
    out = LayerNormNode().execute({"tensor": x}, {"normalized_shape": shape}, context=_ctx())
    assert out["tensor"].shape == x.shape


# -- TensorStats precision -------------------------------------------------------------------


def test_tensor_stats_keeps_large_int64_values_exact():
    """Before: max 16777216.0 and sum 16777216.0 for [16777217, 1]."""
    out = TensorStatsNode().execute({"tensor": torch.tensor([16777217, 1])}, {})
    assert out["max"] == 16777217.0
    assert out["sum"] == 16777218.0


def test_tensor_stats_float16_sum_does_not_overflow():
    """Before: 70000 float16 ones reported ``sum inf``."""
    out = TensorStatsNode().execute({"tensor": torch.ones(70000, dtype=torch.float16)}, {})
    assert out["sum"] == 70000.0
    assert out["mean"] == 1.0


# -- MoELayer and InstanceNorm2d ------------------------------------------------------------


_MOE = {"num_experts": 3, "top_k": 2, "hidden_dim": 6, "expert_hidden_dim": 8, "seed": 1}


def test_moe_returns_the_rank_it_was_given():
    """Before: a [T, H] input came back as [1, T, H]."""
    x = torch.randn(4, 6)
    out = MoELayerNode().execute({"x": x}, _MOE)
    batched = MoELayerNode().execute({"x": x.unsqueeze(0)}, _MOE)
    assert out["output"].shape == (4, 6)
    assert out["routing_weights"].shape == (4, 2)
    assert torch.equal(out["output"], batched["output"][0])


def test_moe_names_a_one_dimensional_input():
    with pytest.raises(ValueError, match=r"MoELayer: `x` must be \[B, T, H\] or \[T, H\]"):
        MoELayerNode().execute({"x": torch.randn(6)}, _MOE)


def test_instancenorm_checks_num_features_without_affine():
    """Before: num_features=7 on a 3-channel input ran."""
    with pytest.raises(ValueError, match="InstanceNorm2d: the input has 3 channels but num_features is 7"):
        InstanceNorm2dNode().execute(
            {"tensor": torch.rand(2, 3, 4, 4)}, {"num_features": 7, "affine": False}, context=_ctx())
