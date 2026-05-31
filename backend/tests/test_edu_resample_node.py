"""Tests for EduResampleNode (lesson I3-2: U-Net down/up sampling)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from cdui_plugins.deep.nodes.edu_resample_node import EduResampleNode


class _Ctx:
    """Minimal stand-in for ExecutionContext exposing only ``.verbose``."""

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose


def _run(x, *, skip=None, context=None, **params):
    base = {"direction": "down", "factor": 2, "mode": "nearest"}
    base.update(params)
    inputs: dict = {"x": x}
    if skip is not None:
        inputs["skip"] = skip
    return EduResampleNode().execute(inputs, base, context=context)


def test_node_metadata():
    assert EduResampleNode.NODE_NAME == "Edu-Resample"
    assert EduResampleNode.CATEGORY == "Diffusion"
    out_names = [p.name for p in EduResampleNode.define_outputs()]
    assert out_names == ["y", "shape"]


def test_down_factor_2_halves_spatial_dims():
    x = torch.randn(2, 3, 16, 16)
    res = _run(x, direction="down", factor=2, mode="avgpool")
    assert res["y"].shape == (2, 3, 8, 8)
    # shape output mirrors the resampled dims.
    assert res["shape"].tolist() == [2, 3, 8, 8]


def test_up_factor_2_doubles_spatial_dims():
    x = torch.randn(2, 3, 8, 8)
    res = _run(x, direction="up", factor=2, mode="nearest")
    assert res["y"].shape == (2, 3, 16, 16)
    assert res["shape"].tolist() == [2, 3, 16, 16]


def test_nearest_up_matches_interpolate():
    x = torch.randn(1, 4, 5, 7)
    res = _run(x, direction="up", factor=2, mode="nearest")
    expected = F.interpolate(x, scale_factor=2, mode="nearest")
    assert torch.equal(res["y"], expected)


def test_avgpool_down_matches_avg_pool2d():
    x = torch.randn(1, 4, 8, 8)
    res = _run(x, direction="down", factor=2, mode="avgpool")
    expected = F.avg_pool2d(x, 2)
    assert torch.allclose(res["y"], expected)


def test_skip_concat_doubles_channels():
    # x: [1, 3, 8, 8] up by 2 → [1, 3, 16, 16]; skip is same resolution.
    x = torch.randn(1, 3, 8, 8)
    skip = torch.randn(1, 3, 16, 16)
    res = _run(x, skip=skip, direction="up", factor=2, mode="nearest")
    assert res["y"].shape == (1, 6, 16, 16)
    assert res["shape"].tolist() == [1, 6, 16, 16]
    # The concatenated half is the skip tensor verbatim.
    assert torch.equal(res["y"][:, 3:], skip)


def test_skip_ignored_when_direction_down():
    """A skip on the encoder side has nowhere to splice in — it's a no-op."""
    x = torch.randn(1, 3, 16, 16)
    skip = torch.randn(1, 3, 8, 8)
    res = _run(x, skip=skip, direction="down", factor=2, mode="avgpool")
    assert res["y"].shape == (1, 3, 8, 8)  # channels unchanged


def test_promotes_3d_input_to_batch_one():
    x = torch.randn(3, 8, 8)
    res = _run(x, direction="down", factor=2, mode="avgpool")
    assert res["y"].shape == (1, 3, 4, 4)


def test_rejects_invalid_factor():
    with pytest.raises(ValueError, match="factor"):
        _run(torch.randn(1, 3, 8, 8), factor=1)


def test_rejects_mismatched_skip():
    x = torch.randn(1, 3, 8, 8)
    bad_skip = torch.randn(1, 3, 8, 8)  # up→16, but skip is 8 → mismatch
    with pytest.raises(ValueError, match="skip"):
        _run(x, skip=bad_skip, direction="up", factor=2, mode="nearest")


def test_rejects_non_4d_input():
    with pytest.raises(ValueError, match="shape"):
        _run(torch.randn(8))


def test_verbose_emits_resample_step():
    x = torch.randn(1, 3, 8, 8)
    res = _run(x, direction="down", factor=2, mode="avgpool", context=_Ctx(verbose=True))
    assert "__steps__" in res
    step_names = [s.name for s in res["__steps__"]]
    assert "resample" in step_names
    resample = next(s for s in res["__steps__"] if s.name == "resample")
    assert resample.scalars["H_before"] == 8.0
    assert resample.scalars["H_after"] == 4.0


def test_verbose_emits_skip_concat_step():
    x = torch.randn(1, 3, 8, 8)
    skip = torch.randn(1, 3, 16, 16)
    res = _run(
        x, skip=skip, direction="up", factor=2, mode="nearest", context=_Ctx(verbose=True)
    )
    step_names = [s.name for s in res["__steps__"]]
    assert step_names == ["resample", "skip_concat"]
    concat = next(s for s in res["__steps__"] if s.name == "skip_concat")
    assert concat.scalars["channels_before"] == 3.0
    assert concat.scalars["channels_after"] == 6.0


def test_non_verbose_has_no_steps():
    x = torch.randn(1, 3, 8, 8)
    # No context at all.
    res = _run(x, direction="down", factor=2, mode="avgpool")
    assert "__steps__" not in res
    # Explicit non-verbose context.
    res2 = _run(x, direction="down", factor=2, mode="avgpool", context=_Ctx(verbose=False))
    assert "__steps__" not in res2


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        EduResampleNode().execute({}, {"direction": "down", "factor": 2, "mode": "nearest"})
