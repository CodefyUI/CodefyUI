"""Tests for Conv2dExplicitNode — convolution with a kernel picked in the node.

The kernel used to arrive on a second input port from a ``Conv2dKernel``
producer. Those two nodes are now one, so the tests are too: the preset /
Custom-matrix half came over from ``test_conv2d_kernel_node.py``, and the
tests that only described the removed port (4D kernel shapes, a non-tensor
kernel) went with it. What a caller can express changed; what the convolution
computes did not, and the numeric tests below are unchanged from both files.
"""

from __future__ import annotations

import pytest
import torch

from app.nodes.cnn.conv2d_explicit_node import (
    CUSTOM_OPTION,
    MAX_KERNEL_SIZE,
    MIN_KERNEL_SIZE,
    PRESET_OPTIONS,
    PRESETS_3X3,
    Conv2dExplicitNode,
)


def _run(tensor, **params):
    return Conv2dExplicitNode().execute({"tensor": tensor}, params)


def _custom(weights, **params):
    """Run with a hand-authored kernel matrix."""
    k = len(weights)
    return _run(params.pop("tensor"), preset=CUSTOM_OPTION, kernel_size=k,
                weights=weights, **params)


# ── Schema ──


def test_node_metadata():
    assert Conv2dExplicitNode.NODE_NAME == "Conv2dExplicit"
    assert Conv2dExplicitNode.CATEGORY == "CNN"
    in_names = [p.name for p in Conv2dExplicitNode.define_inputs()]
    out_names = [p.name for p in Conv2dExplicitNode.define_outputs()]
    # One input. The kernel is a parameter now, not a wire.
    assert in_names == ["tensor"]
    assert out_names == ["tensor"]


def test_params_are_the_kernel_choice_then_the_conv_trim():
    names = [p.name for p in Conv2dExplicitNode.define_params()]
    # Order is the reading order on the node card: what filter, then how it
    # is swept.
    assert names == ["preset", "kernel_size", "weights", "stride", "padding"]


def test_preset_options_are_the_three_builtins_plus_custom():
    preset = next(p for p in Conv2dExplicitNode.define_params() if p.name == "preset")
    assert preset.options == PRESET_OPTIONS
    assert preset.options == [
        "EdgeDetection3x3", "Sharpen3x3", "VerticalEdge3x3", CUSTOM_OPTION,
    ]
    assert preset.default == "EdgeDetection3x3"


def test_kernel_size_and_weights_are_visible_only_when_custom():
    params = {p.name: p for p in Conv2dExplicitNode.define_params()}
    assert params["kernel_size"].visible_when == {"preset": CUSTOM_OPTION}
    assert params["weights"].visible_when == {"preset": CUSTOM_OPTION}
    # stride/padding apply to every preset, so they are always on show.
    assert params["stride"].visible_when is None
    assert params["padding"].visible_when is None


def test_kernel_size_param_bounds():
    ks = next(p for p in Conv2dExplicitNode.define_params() if p.name == "kernel_size")
    assert ks.min_value == MIN_KERNEL_SIZE
    assert ks.max_value == MAX_KERNEL_SIZE
    assert ks.default == 3


def test_weights_default_is_identity_3x3():
    w = next(p for p in Conv2dExplicitNode.define_params() if p.name == "weights")
    assert w.default == [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]


def test_no_in_or_out_channel_params():
    # Depthwise: the channel count comes from the input, never from a param.
    names = {p.name for p in Conv2dExplicitNode.define_params()}
    assert "in_channels" not in names
    assert "out_channels" not in names


# ── Preset values ──


def test_edge_detection_preset_values():
    assert PRESETS_3X3["EdgeDetection3x3"] == [
        [-1.0, -1.0, -1.0],
        [-1.0,  8.0, -1.0],
        [-1.0, -1.0, -1.0],
    ]


def test_sharpen_preset_values():
    assert PRESETS_3X3["Sharpen3x3"] == [
        [ 0.0, -1.0,  0.0],
        [-1.0,  5.0, -1.0],
        [ 0.0, -1.0,  0.0],
    ]


def test_vertical_edge_preset_values():
    assert PRESETS_3X3["VerticalEdge3x3"] == [
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
    ]


# ── Execution: presets ──


def test_edge_detection_preset_detects_an_isolated_pixel():
    x = torch.zeros(1, 1, 5, 5)
    x[0, 0, 2, 2] = 1.0
    res = _run(x, preset="EdgeDetection3x3", padding=1)
    out = res["tensor"]
    assert out.shape == (1, 1, 5, 5)
    # Centre sees +8 from itself; each neighbour sees -1 from the centre.
    assert out[0, 0, 2, 2].item() == pytest.approx(8.0)
    assert out[0, 0, 1, 2].item() == pytest.approx(-1.0)


def test_uniform_input_with_edge_detection_is_zero_in_interior():
    # The Laplacian sums to 0, so a flat region has no response. Only the
    # border reacts, and only because zero padding invents an edge there.
    x = torch.full((1, 1, 6, 6), 3.0)
    out = _run(x, preset="EdgeDetection3x3", padding=1)["tensor"]
    assert torch.allclose(out[0, 0, 1:-1, 1:-1], torch.zeros(4, 4), atol=1e-5)


def test_sharpen_preset_leaves_a_flat_region_alone():
    # Sharpen sums to 1: a flat region passes through at its own value.
    x = torch.full((1, 1, 6, 6), 2.0)
    out = _run(x, preset="Sharpen3x3", padding=1)["tensor"]
    assert torch.allclose(out[0, 0, 1:-1, 1:-1], torch.full((4, 4), 2.0), atol=1e-5)


def test_vertical_edge_preset_responds_to_a_vertical_boundary():
    # Left half 0, right half 1. Prewitt-X fires on the boundary column.
    x = torch.zeros(1, 1, 5, 6)
    x[0, 0, :, 3:] = 1.0
    out = _run(x, preset="VerticalEdge3x3", padding=1)["tensor"]
    assert out[0, 0, 2, 2].item() == pytest.approx(3.0)
    # Well inside either flat half there is nothing to respond to.
    assert out[0, 0, 2, 0].item() == pytest.approx(0.0)


def test_unknown_preset_rejected():
    with pytest.raises(ValueError, match="Unknown preset"):
        _run(torch.zeros(1, 1, 5, 5), preset="Nope")


# ── Execution: custom kernels ──


def test_custom_identity_passes_the_image_through():
    identity = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    x = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5)
    out = _custom(identity, tensor=x, padding=1)["tensor"]
    assert torch.allclose(out, x)


def test_custom_5x5_box_blur():
    box = [[1.0 / 25] * 5 for _ in range(5)]
    x = torch.full((1, 1, 7, 7), 4.0)
    out = _custom(box, tensor=x, padding=2)["tensor"]
    # The interior averages 25 identical values back to the same number.
    assert out[0, 0, 3, 3].item() == pytest.approx(4.0)


def test_custom_2x2_is_allowed():
    # Even sizes are legal; only non-square grids are not expressible.
    out = _custom([[1.0, 1.0], [1.0, 1.0]], tensor=torch.ones(1, 1, 4, 4),
                  padding=0)["tensor"]
    assert out.shape == (1, 1, 3, 3)
    assert torch.allclose(out, torch.full((1, 1, 3, 3), 4.0))


def test_custom_rejects_when_weights_size_mismatches_kernel_size():
    # A grid the user resized but never re-filled. Padding or truncating it
    # would convolve with a filter they did not write.
    with pytest.raises(ValueError, match=r"9 elements but kernel_size=5"):
        _run(torch.zeros(1, 1, 5, 5), preset=CUSTOM_OPTION, kernel_size=5,
             weights=[[0.0] * 3 for _ in range(3)])


def test_custom_rejects_when_weights_missing():
    with pytest.raises(ValueError, match="requires `weights`"):
        _run(torch.zeros(1, 1, 5, 5), preset=CUSTOM_OPTION, kernel_size=3)


# ── Execution: depthwise behaviour ──


def test_the_same_kernel_is_applied_to_every_channel():
    # 3 channels, each uniform. A Laplacian on a flat region is 0 everywhere
    # inside, whatever the channel's value -- and the channel count survives.
    x = torch.stack([torch.full((1, 6, 6), float(c)) for c in range(3)], dim=1)
    out = _run(x, preset="EdgeDetection3x3", padding=1)["tensor"]
    assert out.shape == (1, 3, 6, 6)
    assert torch.allclose(out[0, :, 1:-1, 1:-1], torch.zeros(3, 4, 4), atol=1e-5)


def test_channels_stay_independent():
    identity = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    x = torch.stack([
        torch.full((1, 5, 5), 7.0),
        torch.full((1, 5, 5), 9.0),
    ], dim=1)
    out = _custom(identity, tensor=x, padding=1)["tensor"]
    # Grouped conv, not a sum across channels: 7 stays 7 and 9 stays 9.
    assert torch.allclose(out[0, 0], torch.full((5, 5), 7.0))
    assert torch.allclose(out[0, 1], torch.full((5, 5), 9.0))


# ── Execution: shape and dtype ──


def test_stride_2_halves_spatial_dims():
    out = _run(torch.zeros(1, 1, 8, 8), preset="EdgeDetection3x3",
               stride=2, padding=1)["tensor"]
    assert out.shape == (1, 1, 4, 4)


def test_preserves_input_dtype():
    x = torch.zeros(1, 1, 5, 5, dtype=torch.float64)
    out = _run(x, preset="EdgeDetection3x3", padding=1)["tensor"]
    assert out.dtype == torch.float64


def test_output_stays_on_the_inputs_device():
    x = torch.zeros(1, 1, 5, 5)
    out = _run(x, preset="EdgeDetection3x3", padding=1)["tensor"]
    # The kernel is built on the input's device rather than on the CPU and
    # moved, so a node that conjures its own tensor does not reintroduce the
    # mismatch the engine aligns away for wires.
    assert out.device == x.device


# ── Input validation ──


def test_rejects_3d_input_with_helpful_message():
    with pytest.raises(ValueError, match=r"must be 4D"):
        _run(torch.zeros(1, 5, 5), preset="EdgeDetection3x3")


def test_rejects_non_tensor_input():
    with pytest.raises(ValueError, match=r"must be a torch.Tensor"):
        _run([[1, 2], [3, 4]], preset="EdgeDetection3x3")
