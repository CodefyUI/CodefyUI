"""Tests for Conv2dExplicitNode (conv with kernel via input port)."""

from __future__ import annotations

import pytest
import torch

from app.nodes.cnn.conv2d_explicit_node import Conv2dExplicitNode


def _run(tensor, kernel, **params):
    return Conv2dExplicitNode().execute({"tensor": tensor, "kernel": kernel}, params)


# ── Schema ──


def test_node_metadata():
    assert Conv2dExplicitNode.NODE_NAME == "Conv2dExplicit"
    assert Conv2dExplicitNode.CATEGORY == "CNN"
    in_names = [p.name for p in Conv2dExplicitNode.define_inputs()]
    out_names = [p.name for p in Conv2dExplicitNode.define_outputs()]
    assert in_names == ["tensor", "kernel"]
    assert out_names == ["tensor"]


def test_param_names_are_just_stride_and_padding():
    names = [p.name for p in Conv2dExplicitNode.define_params()]
    assert names == ["stride", "padding"]


def test_no_kernel_size_or_in_out_channels_params():
    # All kernel-shape info is derived from the kernel input tensor.
    names = {p.name for p in Conv2dExplicitNode.define_params()}
    assert "kernel_size" not in names
    assert "in_channels" not in names
    assert "out_channels" not in names


# ── Execution: bare 2D kernel ──


def test_identity_2d_kernel_passes_image_through():
    identity = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    x = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5)
    res = _run(x, identity, stride=1, padding=1)
    assert torch.allclose(res["tensor"], x)


def test_laplacian_2d_kernel_detects_isolated_pixel():
    laplacian = torch.tensor([
        [-1.0, -1.0, -1.0],
        [-1.0,  8.0, -1.0],
        [-1.0, -1.0, -1.0],
    ])
    x = torch.zeros(1, 1, 5, 5)
    x[0, 0, 2, 2] = 1.0
    res = _run(x, laplacian, stride=1, padding=1)
    y = res["tensor"]
    assert y[0, 0, 2, 2].item() == pytest.approx(8.0)
    assert y[0, 0, 1, 1].item() == pytest.approx(-1.0)


def test_uniform_input_with_laplacian_is_zero_in_interior():
    # padding=0 keeps the kernel window away from zero-padding artefacts.
    laplacian = torch.tensor([
        [-1.0, -1.0, -1.0],
        [-1.0,  8.0, -1.0],
        [-1.0, -1.0, -1.0],
    ])
    x = torch.ones(1, 1, 5, 5)
    res = _run(x, laplacian, stride=1, padding=0)
    assert res["tensor"].shape == (1, 1, 3, 3)
    assert torch.allclose(res["tensor"], torch.zeros(1, 1, 3, 3))


def test_2d_kernel_rejects_non_square():
    x = torch.zeros(1, 1, 5, 5)
    k = torch.zeros(2, 3)
    with pytest.raises(ValueError, match=r"square"):
        _run(x, k)


# ── Execution: 4D kernel forms ──


def test_4d_kernel_with_out_channel_1_is_broadcast():
    # (1, 1, k, k) → same effective behaviour as a (k, k) kernel.
    k = torch.tensor([
        [-1.0, -1.0, -1.0],
        [-1.0,  8.0, -1.0],
        [-1.0, -1.0, -1.0],
    ]).reshape(1, 1, 3, 3)
    x = torch.zeros(1, 1, 5, 5)
    x[0, 0, 2, 2] = 1.0
    res = _run(x, k, padding=1)
    assert res["tensor"][0, 0, 2, 2].item() == pytest.approx(8.0)


def test_4d_kernel_with_out_channel_matching_input_channels():
    # (C, 1, k, k) — one distinct kernel per input channel.
    # ch0: identity. ch1: 3x zero (kills the channel).
    identity = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    zero = [[0.0] * 3 for _ in range(3)]
    k = torch.tensor([identity, zero]).reshape(2, 1, 3, 3)
    x = torch.stack([
        torch.full((1, 5, 5), 7.0),
        torch.full((1, 5, 5), 9.0),
    ], dim=1)
    res = _run(x, k, padding=1)
    assert res["tensor"].shape == (1, 2, 5, 5)
    # Channel 0 passed through as identity → still 7.
    assert torch.allclose(res["tensor"][0, 0], torch.full((5, 5), 7.0))
    # Channel 1 zeroed out by the zero kernel.
    assert torch.allclose(res["tensor"][0, 1], torch.zeros(5, 5))


def test_4d_kernel_rejects_mismatched_out_channels():
    x = torch.zeros(1, 3, 5, 5)
    # (2, 1, 3, 3) doesn't match C=3 and isn't 1 (broadcast) — invalid.
    k = torch.zeros(2, 1, 3, 3)
    with pytest.raises(ValueError, match=r"out-channel.*1.*C=3"):
        _run(x, k, padding=1)


def test_4d_kernel_rejects_inner_channel_not_one():
    # (1, 3, 3, 3) — looks like a full Conv2d weight tensor for a multi-channel
    # convolution. We only support depthwise here.
    x = torch.zeros(1, 3, 5, 5)
    k = torch.zeros(1, 3, 3, 3)
    with pytest.raises(ValueError, match=r"inner channel must be 1"):
        _run(x, k, padding=1)


def test_3d_kernel_rejected():
    x = torch.zeros(1, 1, 5, 5)
    k = torch.zeros(3, 3, 3)
    with pytest.raises(ValueError, match=r"2D \(k, k\) or 4D"):
        _run(x, k, padding=1)


# ── Execution: multi-channel depthwise ──


def test_depthwise_applies_same_2d_kernel_per_channel():
    # 3 channels uniformly 0/1/2. Laplacian on a flat region → 0.
    laplacian = torch.tensor([
        [-1.0, -1.0, -1.0],
        [-1.0,  8.0, -1.0],
        [-1.0, -1.0, -1.0],
    ])
    x = torch.stack([
        torch.zeros(1, 5, 5),
        torch.ones(1, 5, 5),
        torch.full((1, 5, 5), 2.0),
    ], dim=1)
    res = _run(x, laplacian, padding=0)
    assert res["tensor"].shape == (1, 3, 3, 3)
    assert torch.allclose(res["tensor"], torch.zeros(1, 3, 3, 3))


# ── Validation: input shapes ──


def test_rejects_3d_input_with_helpful_message():
    x = torch.zeros(1, 5, 5)
    k = torch.zeros(3, 3)
    with pytest.raises(ValueError, match=r"4D \(N, C, H, W\).*Unsqueeze"):
        _run(x, k)


def test_rejects_non_tensor_input():
    k = torch.zeros(3, 3)
    with pytest.raises(ValueError, match=r"`tensor` input must be a torch.Tensor"):
        Conv2dExplicitNode().execute({"tensor": [[1, 2], [3, 4]], "kernel": k}, {})


def test_rejects_non_tensor_kernel():
    x = torch.zeros(1, 1, 5, 5)
    with pytest.raises(ValueError, match=r"`kernel` input must be a torch.Tensor"):
        Conv2dExplicitNode().execute({"tensor": x, "kernel": [[1, 0], [0, 1]]}, {})


# ── Stride / dtype ──


def test_stride_2_halves_spatial_dims():
    k = torch.ones(3, 3)
    x = torch.ones(1, 1, 8, 8)
    res = _run(x, k, stride=2, padding=1)
    # (8 + 2 - 3) // 2 + 1 = 4
    assert res["tensor"].shape == (1, 1, 4, 4)


def test_preserves_input_dtype():
    k = torch.ones(3, 3)
    x = torch.ones(1, 1, 5, 5, dtype=torch.float64)
    res = _run(x, k, padding=1)
    assert res["tensor"].dtype == torch.float64
