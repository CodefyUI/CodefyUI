"""Tests for Conv2dKernelNode (pure kernel-tensor producer)."""

from __future__ import annotations

import pytest
import torch

from app.nodes.cnn.conv2d_kernel_node import (
    CUSTOM_OPTION,
    MAX_KERNEL_SIZE,
    PRESET_OPTIONS,
    PRESETS_3X3,
    Conv2dKernelNode,
)


def _run(**params):
    # Conv2dKernel has no inputs — it's a pure producer.
    return Conv2dKernelNode().execute({}, params)


# ── Schema ──


def test_node_metadata():
    assert Conv2dKernelNode.NODE_NAME == "Conv2dKernel"
    assert Conv2dKernelNode.CATEGORY == "CNN"
    assert Conv2dKernelNode.define_inputs() == []
    out_names = [p.name for p in Conv2dKernelNode.define_outputs()]
    assert out_names == ["tensor"]


def test_preset_param_options_include_three_presets_and_custom():
    params = {p.name: p for p in Conv2dKernelNode.define_params()}
    assert params["preset"].options == [
        "EdgeDetection3x3",
        "Sharpen3x3",
        "VerticalEdge3x3",
        "Custom",
    ]
    assert PRESET_OPTIONS[-1] == CUSTOM_OPTION


def test_kernel_size_and_weights_are_visible_only_when_custom():
    params = {p.name: p for p in Conv2dKernelNode.define_params()}
    assert params["kernel_size"].visible_when == {"preset": "Custom"}
    assert params["weights"].visible_when == {"preset": "Custom"}
    # The preset selector itself is always visible.
    assert params["preset"].visible_when is None


def test_kernel_size_param_bounds():
    params = {p.name: p for p in Conv2dKernelNode.define_params()}
    assert params["kernel_size"].min_value == 1
    assert params["kernel_size"].max_value == MAX_KERNEL_SIZE
    assert params["kernel_size"].default == 3


def test_no_stride_or_padding_params():
    # Conv2dKernel emits a kernel — convolution-time options like stride
    # and padding belong on the downstream conv node, not here.
    names = {p.name for p in Conv2dKernelNode.define_params()}
    assert "stride" not in names
    assert "padding" not in names


def test_weights_default_is_identity_3x3():
    params = {p.name: p for p in Conv2dKernelNode.define_params()}
    assert params["weights"].default == [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ]


# ── Preset kernel data ──


def test_edge_detection_preset_values():
    assert PRESETS_3X3["EdgeDetection3x3"] == [
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1],
    ]


def test_sharpen_preset_values():
    assert PRESETS_3X3["Sharpen3x3"] == [
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0],
    ]


def test_vertical_edge_preset_values():
    assert PRESETS_3X3["VerticalEdge3x3"] == [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1],
    ]


# ── Execution ──


def test_emits_edge_detection_preset_as_3x3_tensor():
    res = _run(preset="EdgeDetection3x3")
    t = res["tensor"]
    assert isinstance(t, torch.Tensor)
    assert t.shape == (3, 3)
    expected = torch.tensor([
        [-1.0, -1.0, -1.0],
        [-1.0,  8.0, -1.0],
        [-1.0, -1.0, -1.0],
    ])
    assert torch.equal(t, expected)


def test_emits_sharpen_preset_as_3x3_tensor():
    res = _run(preset="Sharpen3x3")
    expected = torch.tensor([
        [ 0.0, -1.0,  0.0],
        [-1.0,  5.0, -1.0],
        [ 0.0, -1.0,  0.0],
    ])
    assert torch.equal(res["tensor"], expected)


def test_emits_vertical_edge_preset_as_3x3_tensor():
    res = _run(preset="VerticalEdge3x3")
    expected = torch.tensor([
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
    ])
    assert torch.equal(res["tensor"], expected)


def test_emits_custom_3x3_identity_when_explicitly_provided():
    identity = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    res = _run(preset="Custom", kernel_size=3, weights=identity)
    expected = torch.tensor(identity)
    assert torch.equal(res["tensor"], expected)


def test_emits_custom_5x5_box_blur():
    avg = [[1.0 / 25.0] * 5 for _ in range(5)]
    res = _run(preset="Custom", kernel_size=5, weights=avg)
    assert res["tensor"].shape == (5, 5)
    assert torch.allclose(res["tensor"], torch.full((5, 5), 1.0 / 25.0))


def test_emits_custom_2x2():
    weights = [[1.0, 2.0], [3.0, 4.0]]
    res = _run(preset="Custom", kernel_size=2, weights=weights)
    expected = torch.tensor(weights)
    assert torch.equal(res["tensor"], expected)


def test_output_dtype_is_float32():
    res = _run(preset="EdgeDetection3x3")
    assert res["tensor"].dtype == torch.float32


# ── Errors ──


def test_custom_rejects_when_weights_size_mismatches_kernel_size():
    with pytest.raises(ValueError, match=r"weights.*9 elements.*kernel_size=5.*25"):
        _run(preset="Custom", kernel_size=5, weights=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def test_custom_rejects_when_weights_missing():
    with pytest.raises(ValueError, match=r"requires `weights` to be set"):
        _run(preset="Custom", kernel_size=3, weights=None)


def test_unknown_preset_rejected():
    with pytest.raises(ValueError, match=r"Unknown preset"):
        _run(preset="MysteryKernel")
