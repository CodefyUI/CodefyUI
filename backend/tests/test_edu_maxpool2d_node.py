"""Tests for EduMaxPool2dNode (chapter pack I3, lesson I3-1: max pooling)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from cdui_plugins.deep.nodes.edu_maxpool2d_node import EduMaxPool2dNode


class _VerboseCtx:
    """Minimal stand-in for ExecutionContext with verbose step-trace on."""

    verbose = True
    weights_persistent = False
    node_state_store = None


def _run(x, *, context=None, **params):
    p = {"kernel_size": 2, "stride": 0, "padding": 0}
    p.update(params)
    return EduMaxPool2dNode().execute({"x": x}, p, context=context)


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def test_node_metadata():
    assert EduMaxPool2dNode.NODE_NAME == "Edu-MaxPool2d"
    assert EduMaxPool2dNode.CATEGORY == "CNN"
    out_names = [p.name for p in EduMaxPool2dNode.define_outputs()]
    assert out_names == ["y", "argmax"]


def test_input_and_param_names():
    in_names = [p.name for p in EduMaxPool2dNode.define_inputs()]
    assert in_names == ["x"]
    param_names = {p.name for p in EduMaxPool2dNode.define_params()}
    assert param_names == {"kernel_size", "stride", "padding"}


# --------------------------------------------------------------------------- #
# Equivalence to F.max_pool2d
# --------------------------------------------------------------------------- #
def test_matches_max_pool2d_default_stride():
    """Default stride (0 → kernel_size) must reproduce F.max_pool2d exactly."""
    torch.manual_seed(0)
    x = torch.randn(3, 5, 8, 8)
    res = _run(x, kernel_size=2, stride=0, padding=0)
    expected = F.max_pool2d(x, kernel_size=2)  # stride defaults to kernel_size
    assert res["y"].shape == expected.shape
    assert torch.allclose(res["y"], expected)


def test_matches_max_pool2d_explicit_stride_and_padding():
    """Explicit stride + padding must reproduce F.max_pool2d (incl. indices)."""
    torch.manual_seed(1)
    x = torch.randn(2, 4, 9, 7)
    res = _run(x, kernel_size=3, stride=2, padding=1)
    expected, expected_idx = F.max_pool2d(
        x, kernel_size=3, stride=2, padding=1, return_indices=True
    )
    assert torch.allclose(res["y"], expected)
    assert torch.equal(res["argmax"], expected_idx)


def test_argmax_indices_point_to_the_max():
    """argmax indexes the flattened H×W plane; gathering there returns y."""
    torch.manual_seed(2)
    x = torch.randn(1, 2, 6, 6)
    res = _run(x, kernel_size=2, stride=2)
    y, argmax = res["y"], res["argmax"]
    N, C, H, W = x.shape
    flat = x.reshape(N, C, H * W)
    gathered = flat.gather(2, argmax.reshape(N, C, -1)).reshape_as(y)
    assert torch.allclose(gathered, y)


# --------------------------------------------------------------------------- #
# Hand-computed 4x4 -> 2x2
# --------------------------------------------------------------------------- #
def test_hand_computed_4x4_to_2x2():
    """A known 4×4 input with 2×2 non-overlapping windows."""
    x = torch.tensor(
        [
            [1.0, 2.0, 5.0, 6.0],
            [3.0, 4.0, 7.0, 8.0],
            [9.0, 10.0, 13.0, 14.0],
            [11.0, 12.0, 15.0, 16.0],
        ]
    ).reshape(1, 1, 4, 4)
    res = _run(x, kernel_size=2, stride=2)
    # Each 2×2 block's max: TL=4, TR=8, BL=12, BR=16.
    expected = torch.tensor([[4.0, 8.0], [12.0, 16.0]]).reshape(1, 1, 2, 2)
    assert res["y"].shape == (1, 1, 2, 2)
    assert torch.equal(res["y"], expected)
    # The maxima sit at the bottom-right of each block: flat input indices
    # 5 (=row1,col1), 7 (=row1,col3), 13 (=row3,col1), 15 (=row3,col3).
    expected_idx = torch.tensor([[5, 7], [13, 15]]).reshape(1, 1, 2, 2)
    assert torch.equal(res["argmax"], expected_idx)


# --------------------------------------------------------------------------- #
# [C, H, W] gets a batch dim
# --------------------------------------------------------------------------- #
def test_chw_input_gains_batch_dim():
    """A bare [C, H, W] is promoted to [1, C, H, W]."""
    torch.manual_seed(3)
    x = torch.randn(3, 8, 8)  # [C, H, W]
    res = _run(x, kernel_size=2)
    assert res["y"].shape == (1, 3, 4, 4)
    # Same numbers as pooling the explicitly-batched version.
    expected = F.max_pool2d(x.unsqueeze(0), kernel_size=2)
    assert torch.allclose(res["y"], expected)


# --------------------------------------------------------------------------- #
# Downsample shape formula
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "H,W,k,s,p",
    [
        (8, 8, 2, 0, 0),   # default stride
        (9, 7, 3, 2, 1),   # padded, overlapping-ish
        (10, 10, 2, 2, 0),
        (5, 5, 2, 1, 0),   # stride 1 → overlapping windows
        (7, 9, 3, 3, 0),
    ],
)
def test_downsample_shape_formula(H, W, k, s, p):
    torch.manual_seed(4)
    x = torch.randn(2, 3, H, W)
    stride = k if s == 0 else s
    expected_h = math.floor((H + 2 * p - k) / stride) + 1
    expected_w = math.floor((W + 2 * p - k) / stride) + 1
    res = _run(x, kernel_size=k, stride=s, padding=p)
    assert res["y"].shape == (2, 3, expected_h, expected_w)


def test_stride_defaults_to_kernel_size():
    """stride=0 must behave identically to stride=kernel_size."""
    torch.manual_seed(5)
    x = torch.randn(1, 2, 8, 8)
    a = _run(x, kernel_size=2, stride=0)
    b = _run(x, kernel_size=2, stride=2)
    assert torch.equal(a["y"], b["y"])


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_invalid_shape_raises():
    with pytest.raises(ValueError, match="shape"):
        _run(torch.zeros(8, 8))  # 2-D, not [C,H,W] or [N,C,H,W]


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        EduMaxPool2dNode().execute({}, {"kernel_size": 2, "stride": 0, "padding": 0})


def test_kernel_too_large_raises():
    """Window bigger than the input → output dim < 1 → ValueError."""
    with pytest.raises(ValueError, match=">= 1"):
        _run(torch.zeros(1, 1, 3, 3), kernel_size=5, stride=5, padding=0)


def test_kernel_below_one_raises():
    with pytest.raises(ValueError, match="kernel_size"):
        _run(torch.zeros(1, 1, 4, 4), kernel_size=0)


def test_excessive_padding_raises():
    """padding must not exceed kernel_size // 2 (mirrors F.max_pool2d)."""
    with pytest.raises(ValueError, match="padding"):
        _run(torch.zeros(1, 1, 6, 6), kernel_size=2, padding=3)


# --------------------------------------------------------------------------- #
# Verbose step trace
# --------------------------------------------------------------------------- #
def test_no_steps_without_verbose_context():
    x = torch.randn(1, 1, 4, 4, generator=torch.Generator().manual_seed(6))
    res = _run(x, kernel_size=2)
    assert "__steps__" not in res


def test_steps_emitted_when_verbose():
    x = torch.randn(1, 2, 4, 4, generator=torch.Generator().manual_seed(7))
    res = _run(x, kernel_size=2, context=_VerboseCtx())
    assert "__steps__" in res
    step_names = [s.name for s in res["__steps__"]]
    # A handful of sampled windows followed by the downsample summary.
    assert "downsample" in step_names
    assert any(name.startswith("window_") for name in step_names)


def test_verbose_window_steps_capped_at_six():
    """A large map has many windows; per-window steps stay bounded (~6)."""
    x = torch.randn(1, 1, 16, 16, generator=torch.Generator().manual_seed(8))
    res = _run(x, kernel_size=2, context=_VerboseCtx())
    window_steps = [s for s in res["__steps__"] if s.name.startswith("window_")]
    assert 1 <= len(window_steps) <= 6


def test_verbose_window_records_value_and_max():
    """Each window step carries the window tensor and a scalar max == y cell."""
    x = torch.tensor(
        [
            [1.0, 2.0, 5.0, 6.0],
            [3.0, 4.0, 7.0, 8.0],
            [9.0, 10.0, 13.0, 14.0],
            [11.0, 12.0, 15.0, 16.0],
        ]
    ).reshape(1, 1, 4, 4)
    res = _run(x, kernel_size=2, stride=2, context=_VerboseCtx())
    window_steps = [s for s in res["__steps__"] if s.name.startswith("window_")]
    # First sampled window is the top-left block; its max is 4.0.
    first = window_steps[0]
    assert "window" in first.tensors
    assert first.scalars["max"] == pytest.approx(4.0)
    # The recorded window tensor's own max equals the kept value.
    assert float(first.tensors["window"].max().item()) == pytest.approx(
        first.scalars["max"]
    )


def test_downsample_step_carries_before_after_shapes():
    x = torch.randn(1, 1, 8, 8, generator=torch.Generator().manual_seed(9))
    res = _run(x, kernel_size=2, stride=2, context=_VerboseCtx())
    downsample = next(s for s in res["__steps__"] if s.name == "downsample")
    assert downsample.scalars["H"] == 8.0
    assert downsample.scalars["W"] == 8.0
    assert downsample.scalars["Hout"] == 4.0
    assert downsample.scalars["Wout"] == 4.0
    assert "y" in downsample.tensors
