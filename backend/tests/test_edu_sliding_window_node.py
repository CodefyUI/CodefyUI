"""Tests for EduSlidingWindowNode (chapter pack foundations, lesson I1-2)."""

from __future__ import annotations

import pytest
import torch

from cdui_plugins.foundations.nodes.edu_sliding_window_node import EduSlidingWindowNode


class _Ctx:
    """Minimal stand-in for ExecutionContext exposing only ``.verbose``."""

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose


def _run(image, *, kernel=None, context=None, **params):
    p = {"kernel_preset": "edge", "stride": 1, "padding": 0}
    p.update(params)
    inputs = {"image": image}
    if kernel is not None:
        inputs["kernel"] = kernel
    return EduSlidingWindowNode().execute(inputs, p, context=context)


def _ramp(h, w):
    """h*w ramp image where pixel (i, j) = i*w + j."""
    return torch.arange(h * w, dtype=torch.float32).reshape(h, w)


def test_node_metadata():
    assert EduSlidingWindowNode.NODE_NAME == "Edu-SlidingWindow"
    assert EduSlidingWindowNode.CATEGORY == "Vision"
    out_names = [p.name for p in EduSlidingWindowNode.define_outputs()]
    assert out_names == ["feature_map", "kernel", "windows"]


def test_identity_preset_is_center_crop():
    # identity 3x3 with padding 0 picks the centre pixel of each window, so the
    # valid-region output is exactly the input with its 1-pixel border removed.
    img = _ramp(5, 5)
    res = _run(img, kernel_preset="identity")
    expected = img[1:-1, 1:-1]
    assert res["feature_map"].shape == expected.shape
    assert torch.allclose(res["feature_map"], expected, atol=1e-6)
    # The kernel echoed out is the 3x3 identity.
    expected_kernel = torch.zeros(3, 3)
    expected_kernel[1, 1] = 1.0
    assert torch.allclose(res["kernel"], expected_kernel, atol=1e-6)


def test_blur_mean_hand_computed():
    # 4x4 ramp, 3x3 mean (blur) kernel, padding 0, stride 1 -> 2x2 output.
    # Hand-computed window means (see module docstring):
    #   (0,0)=5, (0,1)=6, (1,0)=9, (1,1)=10
    img = _ramp(4, 4)
    res = _run(img, kernel_preset="blur")
    fm = res["feature_map"]
    assert fm.shape == (2, 2)
    assert fm[0, 0].item() == pytest.approx(5.0)
    assert fm[0, 1].item() == pytest.approx(6.0)
    assert fm[1, 0].item() == pytest.approx(9.0)
    assert fm[1, 1].item() == pytest.approx(10.0)


def test_windows_shape_and_content():
    img = _ramp(4, 4)
    res = _run(img, kernel_preset="blur")
    # 2x2 outputs => 4 stacked 3x3 receptive fields.
    assert res["windows"].shape == (4, 3, 3)
    # First window (row-major flat index 0) is the top-left 3x3 patch.
    assert torch.allclose(res["windows"][0], img[0:3, 0:3], atol=1e-6)
    # Last window is the bottom-right 3x3 patch.
    assert torch.allclose(res["windows"][3], img[1:4, 1:4], atol=1e-6)


def test_kernel_input_overrides_preset():
    img = _ramp(4, 4)
    # A 2x2 sum kernel of ones: each output cell = sum of a 2x2 window.
    k = torch.ones(2, 2)
    res = _run(img, kernel=k, kernel_preset="edge")
    fm = res["feature_map"]
    # 4x4, 2x2 kernel, padding 0, stride 1 -> 3x3 output.
    assert fm.shape == (3, 3)
    # Top-left window {0,1,4,5} sums to 10.
    assert fm[0, 0].item() == pytest.approx(10.0)
    # Echoed kernel is the override, not the preset.
    assert torch.allclose(res["kernel"], k, atol=1e-6)


def test_three_dim_image_auto_reduced_to_channel_zero():
    # [C, H, W]: channel 0 is a 4x4 ramp, channel 1 is garbage. Only ch0 used.
    ch0 = _ramp(4, 4)
    ch1 = torch.full((4, 4), 999.0)
    img = torch.stack([ch0, ch1], dim=0)  # [2, 4, 4]
    res = _run(img, kernel_preset="blur")
    assert res["feature_map"].shape == (2, 2)
    assert res["feature_map"][0, 0].item() == pytest.approx(5.0)


def test_stride_changes_output_shape():
    img = _ramp(5, 5)
    # 5x5, 3x3 kernel, padding 0: stride 1 -> (5-3)//1+1 = 3; stride 2 -> 2.
    res1 = _run(img, kernel_preset="identity", stride=1)
    res2 = _run(img, kernel_preset="identity", stride=2)
    assert res1["feature_map"].shape == (3, 3)
    assert res2["feature_map"].shape == (2, 2)


def test_padding_changes_output_shape():
    img = _ramp(5, 5)
    # padding 0 -> 3x3; padding 1 keeps spatial size -> (5+2-3)+1 = 5x5.
    res0 = _run(img, kernel_preset="edge", padding=0)
    res1 = _run(img, kernel_preset="edge", padding=1)
    assert res0["feature_map"].shape == (3, 3)
    assert res1["feature_map"].shape == (5, 5)


def test_rejects_one_dim_image():
    with pytest.raises(ValueError, match="2-D"):
        _run(torch.arange(10, dtype=torch.float32))


def test_rejects_kernel_bigger_than_image():
    img = _ramp(3, 3)
    big = torch.ones(5, 5)
    with pytest.raises(ValueError, match="does not fit"):
        _run(img, kernel=big)


def test_rejects_bad_stride():
    with pytest.raises(ValueError, match="stride"):
        _run(_ramp(5, 5), stride=0)


def test_verbose_emits_steps():
    img = _ramp(4, 4)
    res = _run(img, kernel_preset="blur", context=_Ctx(verbose=True))
    assert "__steps__" in res
    steps = res["__steps__"]
    assert isinstance(steps, list)
    assert len(steps) > 0
    names = [s.name for s in steps]
    # First step describes the kernel, last assembles the feature map.
    assert names[0] == "kernel"
    assert names[-1] == "feature_map"
    # At least one per-position step recording the receptive field + product.
    pos_steps = [s for s in steps if s.name.startswith("pos_r")]
    assert pos_steps
    assert "receptive_field" in pos_steps[0].tensors
    assert "product" in pos_steps[0].tensors
    assert "value" in pos_steps[0].scalars


def test_verbose_caps_position_steps():
    # 8x8 image, 3x3 kernel, padding 0, stride 1 -> 6x6 = 36 positions, far
    # more than the cap of 9. Per-position steps must be capped at 9.
    img = _ramp(8, 8)
    res = _run(img, kernel_preset="edge", context=_Ctx(verbose=True))
    pos_steps = [s for s in res["__steps__"] if s.name.startswith("pos_r")]
    assert len(pos_steps) == 9


def test_non_verbose_has_no_steps():
    img = _ramp(4, 4)
    # Explicit non-verbose context.
    res = _run(img, kernel_preset="blur", context=_Ctx(verbose=False))
    assert "__steps__" not in res
    # And no context at all.
    res2 = _run(img, kernel_preset="blur", context=None)
    assert "__steps__" not in res2
