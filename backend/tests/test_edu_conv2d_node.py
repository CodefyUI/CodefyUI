"""Tests for EduConv2dNode (chapter pack I3 — CNN / im2col convolution)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from cdui_plugins.deep.nodes.edu_conv2d_node import EduConv2dNode


class _Ctx:
    """Minimal stand-in for ExecutionContext to toggle verbose mode."""

    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose


def _run(x, weight=None, bias=None, *, context=None, **params):
    inputs: dict = {"x": x}
    if weight is not None:
        inputs["weight"] = weight
    if bias is not None:
        inputs["bias"] = bias
    return EduConv2dNode().execute(inputs, dict(params), context=context)


def test_node_metadata():
    assert EduConv2dNode.NODE_NAME == "Edu-Conv2d"
    assert EduConv2dNode.CATEGORY == "CNN"
    out_names = [p.name for p in EduConv2dNode.define_outputs()]
    assert out_names == ["y", "cols", "weight"]


def test_matches_f_conv2d_basic():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 7, 7)
    weight = torch.randn(4, 3, 3, 3)
    bias = torch.randn(4)
    res = _run(x, weight, bias, stride=1, padding=0)
    expected = F.conv2d(x, weight, bias, stride=1, padding=0)
    assert torch.allclose(res["y"], expected, atol=1e-5)


def test_matches_f_conv2d_with_stride_and_padding():
    torch.manual_seed(1)
    x = torch.randn(2, 3, 9, 8)
    weight = torch.randn(5, 3, 3, 3)
    bias = torch.randn(5)
    res = _run(x, weight, bias, stride=2, padding=1)
    expected = F.conv2d(x, weight, bias, stride=2, padding=1)
    assert torch.allclose(res["y"], expected, atol=1e-5)


def test_matches_f_conv2d_no_bias_nonsquare_kernel():
    torch.manual_seed(2)
    x = torch.randn(1, 2, 10, 12)
    weight = torch.randn(3, 2, 2, 4)  # kH=2, kW=4
    res = _run(x, weight, stride=2, padding=1)
    expected = F.conv2d(x, weight, None, stride=2, padding=1)
    assert torch.allclose(res["y"], expected, atol=1e-5)


def test_returned_weight_is_the_one_used():
    torch.manual_seed(3)
    x = torch.randn(1, 2, 5, 5)
    weight = torch.randn(3, 2, 3, 3)
    res = _run(x, weight)
    assert torch.equal(res["weight"], weight)


def test_random_weight_when_absent_uses_seed():
    x = torch.randn(1, 3, 6, 6)
    a = _run(x, out_channels=4, kernel_size=3, seed=7)
    b = _run(x, out_channels=4, kernel_size=3, seed=7)
    c = _run(x, out_channels=4, kernel_size=3, seed=8)
    # Same seed → identical weights; the conv output is deterministic too.
    assert torch.equal(a["weight"], b["weight"])
    assert a["weight"].shape == (4, 3, 3, 3)
    assert torch.allclose(a["y"], b["y"])
    assert not torch.equal(a["weight"], c["weight"])


def test_random_weight_path_still_matches_f_conv2d():
    x = torch.randn(2, 3, 8, 8)
    res = _run(x, out_channels=4, kernel_size=3, stride=1, padding=1, seed=5)
    w = res["weight"]
    expected = F.conv2d(x, w, None, stride=1, padding=1)
    assert torch.allclose(res["y"], expected, atol=1e-5)


def test_cols_shape_is_im2col_matrix():
    x = torch.randn(2, 3, 7, 7)
    weight = torch.randn(4, 3, 3, 3)
    res = _run(x, weight, stride=1, padding=0)
    # Hout = Wout = (7 - 3) + 1 = 5 → L = 25; Cin*kH*kW = 27.
    assert res["cols"].shape == (2, 3 * 3 * 3, 25)


def test_promotes_3d_input_to_batch_one():
    x = torch.randn(3, 6, 6)  # [Cin, H, W]
    weight = torch.randn(4, 3, 3, 3)
    res = _run(x, weight)
    assert res["y"].shape[0] == 1
    # Equivalent to F.conv2d on the batched version.
    expected = F.conv2d(x.unsqueeze(0), weight, None)
    assert torch.allclose(res["y"], expected, atol=1e-5)


def test_output_shape_formula():
    x = torch.randn(2, 3, 16, 16)
    weight = torch.randn(7, 3, 5, 5)
    res = _run(x, weight, stride=3, padding=2)
    Hout = (16 + 2 * 2 - 5) // 3 + 1
    Wout = (16 + 2 * 2 - 5) // 3 + 1
    assert res["y"].shape == (2, 7, Hout, Wout)


def test_rejects_non_4d_input():
    x = torch.randn(2, 5)  # 2-D, not promotable
    with pytest.raises(ValueError, match="shape"):
        _run(x, torch.randn(2, 2, 3, 3))


def test_rejects_channel_mismatch():
    x = torch.randn(1, 3, 6, 6)
    weight = torch.randn(4, 2, 3, 3)  # weight Cin=2 but x Cin=3
    with pytest.raises(ValueError, match="Cin"):
        _run(x, weight)


def test_rejects_kernel_larger_than_input():
    x = torch.randn(1, 1, 3, 3)
    weight = torch.randn(2, 1, 5, 5)  # 5 > 3, no padding → Hout < 1
    with pytest.raises(ValueError, match=">= 1"):
        _run(x, weight, stride=1, padding=0)


def test_rejects_bad_bias_shape():
    x = torch.randn(1, 2, 5, 5)
    weight = torch.randn(3, 2, 3, 3)
    bias = torch.randn(5)  # should be [3]
    with pytest.raises(ValueError, match="bias"):
        _run(x, weight, bias)


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        EduConv2dNode().execute({}, {})


def test_verbose_records_steps():
    x = torch.randn(1, 2, 6, 6)
    weight = torch.randn(3, 2, 3, 3)
    res = _run(x, weight, context=_Ctx(verbose=True))
    steps = res["__steps__"]
    assert len(steps) == 4
    names = [s.name for s in steps]
    assert names == ["im2col", "weight_matrix", "matmul", "feature_map"]
    # The im2col step carries the cols tensor and shape scalars.
    im2col = steps[0]
    assert "cols" in im2col.tensors
    assert im2col.scalars["Cin"] == 2.0
    assert im2col.scalars["kH"] == 3.0
    # The feature_map step carries y.
    assert "y" in steps[3].tensors


def test_non_verbose_has_no_steps():
    x = torch.randn(1, 2, 6, 6)
    weight = torch.randn(3, 2, 3, 3)
    res = _run(x, weight, context=_Ctx(verbose=False))
    assert "__steps__" not in res
    # And no context at all behaves the same.
    res2 = _run(x, weight)
    assert "__steps__" not in res2
