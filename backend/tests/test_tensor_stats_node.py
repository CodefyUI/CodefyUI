"""Tests for TensorStatsNode."""

from __future__ import annotations

import math

import pytest
import torch

from app.nodes.tensor_ops.tensor_stats_node import TensorStatsNode


def _run(tensor, **params):
    return TensorStatsNode().execute({"tensor": tensor}, params)


def test_node_metadata():
    assert TensorStatsNode.NODE_NAME == "TensorStats"
    assert TensorStatsNode.CATEGORY == "Tensor Operations"
    out_names = [p.name for p in TensorStatsNode.define_outputs()]
    for name in ("tensor", "shape", "numel", "min", "max", "mean", "sum", "std", "report"):
        assert name in out_names
    assert [p.name for p in TensorStatsNode.define_params()] == ["precision"]


def test_stats_on_a_known_tensor():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    res = _run(x)
    assert res["shape"] == "(2, 2)"
    assert res["numel"] == 4
    assert res["min"] == 1.0
    assert res["max"] == 4.0
    assert res["mean"] == 2.5
    assert res["sum"] == 10.0
    # Population std of 1,2,3,4: sqrt(1.25). unbiased=True would give sqrt(5/3).
    assert math.isclose(res["std"], math.sqrt(1.25), rel_tol=1e-6)


def test_scalar_ports_are_python_floats_not_tensors():
    res = _run(torch.tensor([2.0, 4.0]))
    for name in ("min", "max", "mean", "sum", "std"):
        assert isinstance(res[name], float), name
    assert isinstance(res["numel"], int)


def test_input_passes_through_unchanged():
    x = torch.randn(3, 5)
    res = _run(x)
    assert res["tensor"] is x


def test_report_goes_to_the_log():
    res = _run(torch.tensor([1.0, 2.0, 3.0]))
    assert res["__log__"] == res["report"]
    assert "shape=(3,)" in res["report"]
    assert "mean=2.000000" in res["report"]


def test_precision_only_touches_the_report():
    x = torch.tensor([1.0, 2.0])
    loose = _run(x, precision=2)
    tight = _run(x, precision=8)
    assert "mean=1.50\n" in loose["report"] or "mean=1.50 " in loose["report"]
    assert "mean=1.50000000" in tight["report"]
    assert loose["mean"] == tight["mean"] == 1.5


def test_integer_input_is_promoted():
    res = _run(torch.tensor([1, 2, 3, 4]))
    assert res["mean"] == 2.5
    assert res["sum"] == 10.0


def test_bool_input_is_promoted():
    res = _run(torch.tensor([True, False, True, True]))
    assert res["sum"] == 3.0
    assert res["mean"] == 0.75


def test_plain_list_is_accepted():
    res = _run([[1.0, 3.0], [5.0, 7.0]])
    assert res["shape"] == "(2, 2)"
    assert res["mean"] == 4.0


def test_single_element_has_zero_std():
    res = _run(torch.tensor([7.0]))
    assert res["std"] == 0.0
    assert res["min"] == res["max"] == 7.0


def test_empty_tensor_is_refused_with_its_shape():
    with pytest.raises(ValueError, match=r"empty.*\(0, 3\)"):
        _run(torch.zeros(0, 3))


def test_missing_input_is_refused():
    with pytest.raises(ValueError, match="tensor"):
        TensorStatsNode().execute({}, {})


def test_float32_mean_matches_torch_on_the_same_dtype():
    # The number a student sees must be the one a Mean node would print on
    # the same edge, so the float32 path must not silently widen to float64.
    x = torch.rand(1, 2, 48, 48)
    res = _run(x)
    assert res["mean"] == float(x.mean())
    assert res["sum"] == float(x.sum())
