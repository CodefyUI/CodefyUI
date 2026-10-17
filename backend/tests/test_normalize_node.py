"""Tests for NormalizeNode."""

from __future__ import annotations

import pytest
import torch

from app.nodes.data.normalize_node import NormalizeNode


def _run(tensor, **params):
    p = {"mode": "zscore", "axis": 0}
    p.update(params)
    return NormalizeNode().execute({"tensor": tensor}, p)


def test_node_metadata():
    assert NormalizeNode.NODE_NAME == "Normalize"
    assert NormalizeNode.CATEGORY == "Data"
    out_names = [p.name for p in NormalizeNode.define_outputs()]
    assert out_names == ["tensor", "stats"]


def test_zscore_per_column():
    """Each column should have mean 0, std 1 after zscore (axis=0)."""
    x = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    res = _run(x, mode="zscore", axis=0)
    means = res["tensor"].mean(dim=0)
    stds = res["tensor"].std(dim=0, unbiased=False)
    assert torch.allclose(means, torch.zeros(2), atol=1e-6)
    assert torch.allclose(stds, torch.ones(2), atol=1e-5)


def test_minmax_per_column_range_zero_one():
    x = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    res = _run(x, mode="minmax", axis=0)
    assert res["tensor"].min(dim=0).values.allclose(torch.zeros(2))
    assert res["tensor"].max(dim=0).values.allclose(torch.ones(2))


def test_unit_norm_each_row_has_norm_one():
    x = torch.tensor([[3.0, 4.0], [1.0, 0.0], [0.0, 5.0]])
    res = _run(x, mode="unit_norm", axis=1)
    norms = res["tensor"].norm(dim=1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-6)


def test_zscore_axis_1():
    """axis=1 means normalize across each row (per-sample)."""
    x = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    res = _run(x, mode="zscore", axis=1)
    means = res["tensor"].mean(dim=1)
    assert torch.allclose(means, torch.zeros(2), atol=1e-6)


def test_stats_returned_for_zscore():
    x = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    res = _run(x, mode="zscore", axis=0)
    assert "mean" in res["stats"]
    assert "std" in res["stats"]


def test_stats_returned_for_minmax():
    x = torch.tensor([[1.0, 10.0], [2.0, 20.0]])
    res = _run(x, mode="minmax", axis=0)
    assert "min" in res["stats"]
    assert "max" in res["stats"]


def test_zero_variance_column_handled():
    """A constant column produces std=0; should not divide by zero."""
    x = torch.tensor([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    res = _run(x, mode="zscore", axis=0)
    # The constant column should become zero (mean subtracted, no divide).
    assert torch.allclose(res["tensor"][:, 0], torch.zeros(3), atol=1e-6)


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="mode"):
        _run(torch.zeros(3, 2), mode="not-a-mode")


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        NormalizeNode().execute({}, {"mode": "zscore", "axis": 0})


def test_dtype_is_float32():
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    res = _run(x)
    assert res["tensor"].dtype == torch.float32


# ── #713: apply training statistics to test data; stats port type ──

_TRAIN = torch.tensor([[1.0, 10.0], [2.0, 30.0], [3.0, 20.0], [6.0, 40.0]])
_TEST = torch.tensor([[4.0, 25.0], [0.0, 50.0]])


@pytest.mark.parametrize("mode", ["zscore", "minmax", "unit_norm"])
def test_wired_stats_reproduce_the_training_transform_on_test_rows(mode):
    fit = _run(_TRAIN, mode=mode, axis=0)
    applied = NormalizeNode().execute(
        {"tensor": _TEST, "stats": fit["stats"]}, {"mode": "zscore", "axis": 0}
    )
    if mode == "zscore":
        ref = (_TEST - _TRAIN.mean(0)) / _TRAIN.std(0, unbiased=False)
    elif mode == "minmax":
        mn, mx = _TRAIN.min(0).values, _TRAIN.max(0).values
        ref = (_TEST - mn) / (mx - mn)
    else:
        ref = _TEST / _TRAIN.norm(dim=0)
    assert torch.allclose(applied["tensor"], ref, atol=1e-6)
    # The same stats applied to the training rows give the fitted output back.
    again = NormalizeNode().execute({"tensor": _TRAIN, "stats": fit["stats"]}, {"axis": 0})
    assert torch.allclose(again["tensor"], fit["tensor"], atol=1e-6)


@pytest.mark.parametrize("mode", ["zscore", "minmax", "unit_norm"])
def test_wired_stats_follow_the_device_of_the_tensor(mode):
    """The stats are lists, and turning them back into tensors put them on
    the CPU, so on a GPU run (test rows on cuda:0) Iris-Sklearn-KNN and
    Tabular-Iris-Pipeline died with "found at least two devices" (found in
    the #808 sweep). The meta device stands in for the GPU, which CI does not
    have: an op mixing meta and CPU tensors refuses the same way."""
    fit = _run(_TRAIN, mode=mode, axis=0)
    applied = NormalizeNode().execute(
        {"tensor": _TEST.to("meta"), "stats": fit["stats"]}, {"axis": 0}
    )
    assert applied["tensor"].device.type == "meta"


def test_wired_stats_do_not_refit_on_the_test_rows():
    fit = _run(_TRAIN, mode="zscore", axis=0)
    applied = NormalizeNode().execute(
        {"tensor": _TEST, "stats": fit["stats"]}, {"mode": "zscore", "axis": 0}
    )
    # A refit on the two test rows would give each column mean 0 exactly.
    assert not torch.allclose(applied["tensor"].mean(0), torch.zeros(2), atol=1e-3)


def test_single_column_stats_round_trip():
    train = torch.tensor([[1.0], [3.0]])
    fit = _run(train, mode="zscore", axis=0)
    applied = NormalizeNode().execute({"tensor": torch.tensor([[5.0]]), "stats": fit["stats"]}, {"axis": 0})
    assert applied["tensor"].item() == pytest.approx(3.0)


def test_stats_with_the_wrong_column_count_are_refused():
    fit = _run(_TRAIN, mode="zscore", axis=0)
    with pytest.raises(ValueError, match="needs 3"):
        NormalizeNode().execute({"tensor": torch.zeros(2, 3), "stats": fit["stats"]}, {"axis": 0})


def test_stats_port_type_accepts_the_dict_it_carries():
    from app.core.node_base import DataType

    out = {p.name: p for p in NormalizeNode.define_outputs()}["stats"]
    inp = {p.name: p for p in NormalizeNode.define_inputs()}["stats"]
    assert isinstance(_run(_TRAIN)["stats"], dict)
    assert out.data_type == DataType.ANY
    assert inp.data_type == DataType.ANY and inp.optional


# ── 2026-10-13 audit: per-row (axis=1) stats, axis checks ──

_TRAIN_ROWS = torch.tensor([[1.0, 2.0, 3.0, 4.0], [10.0, 0.0, 5.0, 5.0], [2.0, 2.0, 8.0, 0.0]])
_TEST_ROWS = torch.tensor([[9.0, 8.0, 6.0, -3.0], [1.0, 1.0, 1.0, 2.0]])


@pytest.mark.parametrize("mode", ["zscore", "minmax", "unit_norm"])
def test_axis1_stats_normalise_each_test_row_by_its_own_statistics(mode):
    fit = _run(_TRAIN_ROWS, mode=mode, axis=1)
    applied = NormalizeNode().execute(
        {"tensor": _TEST_ROWS, "stats": fit["stats"]}, {"mode": "zscore", "axis": 1}
    )
    # The per-row recipe on the test rows themselves: training row i's
    # statistics say nothing about test row i.
    ref = _run(_TEST_ROWS, mode=mode, axis=1)["tensor"]
    assert torch.allclose(applied["tensor"], ref, atol=1e-6)


def test_axis1_stats_accept_a_test_split_of_another_size():
    fit = _run(_TRAIN_ROWS, mode="unit_norm", axis=1)
    applied = NormalizeNode().execute(
        {"tensor": _TEST_ROWS, "stats": fit["stats"]}, {"axis": 1}
    )
    assert torch.allclose(applied["tensor"].norm(dim=1), torch.ones(2), atol=1e-6)


def test_stats_fitted_on_another_axis_are_refused():
    fit = _run(_TRAIN, mode="zscore", axis=0)
    with pytest.raises(ValueError, match=r"Normalize: the `stats` input was fitted with axis=0"):
        NormalizeNode().execute({"tensor": _TRAIN, "stats": fit["stats"]}, {"axis": 1})


def test_axis0_stats_keep_their_keys_and_numbers():
    fit = _run(_TRAIN, mode="zscore", axis=0)
    assert sorted(fit["stats"]) == ["mean", "std"]
    assert fit["stats"]["mean"] == pytest.approx(_TRAIN.mean(0).tolist())


def test_axis1_stats_record_their_axis():
    assert _run(_TRAIN_ROWS, mode="minmax", axis=1)["stats"]["axis"] == 1


def test_axis1_stats_wired_into_an_axis0_normalize_are_refused():
    fit = _run(_TRAIN_ROWS, mode="zscore", axis=1)
    with pytest.raises(ValueError, match=r"fitted with axis=1, but this node has axis=0"):
        NormalizeNode().execute({"tensor": _TEST_ROWS, "stats": fit["stats"]}, {"axis": 0})


def test_an_out_of_range_axis_names_the_node():
    with pytest.raises(ValueError, match=r"Normalize: axis=2 is out of range for a 2-D tensor"):
        _run(_TRAIN, axis=2)
