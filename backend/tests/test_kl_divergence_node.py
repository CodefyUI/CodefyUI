"""Tests for KLDivergenceNode."""

from __future__ import annotations

import math

import pytest
import torch

from app.nodes.rl.kl_divergence_node import KLDivergenceNode


def _run(p, q, **params):
    pp = {"input_kind": "probs", "reduction": "batchmean"}
    pp.update(params)
    return KLDivergenceNode().execute({"p": p, "q": q}, pp)


def test_node_metadata():
    assert KLDivergenceNode.NODE_NAME == "KLDivergence"
    assert KLDivergenceNode.CATEGORY == "RL"
    out_names = [p.name for p in KLDivergenceNode.define_outputs()]
    assert "kl" in out_names


def test_identical_distributions_give_zero():
    p = torch.tensor([[0.5, 0.5]])
    q = torch.tensor([[0.5, 0.5]])
    res = _run(p, q)
    assert abs(float(res["kl"])) < 1e-6


def test_known_value():
    """KL([1,0] || [0.5,0.5]) = log(2) ≈ 0.693."""
    p = torch.tensor([[1.0, 0.0]])
    q = torch.tensor([[0.5, 0.5]])
    res = _run(p, q)
    assert abs(float(res["kl"]) - math.log(2.0)) < 1e-4


def test_kl_is_nonnegative_random():
    torch.manual_seed(0)
    logits_p = torch.randn(8, 5)
    logits_q = torch.randn(8, 5)
    p = torch.softmax(logits_p, dim=-1)
    q = torch.softmax(logits_q, dim=-1)
    res = _run(p, q)
    assert float(res["kl"]) >= 0.0


def test_logits_input():
    """Same KL value whether you pass probs or logits."""
    logits_p = torch.tensor([[2.0, 0.0]])
    logits_q = torch.tensor([[0.0, 2.0]])
    p_probs = torch.softmax(logits_p, dim=-1)
    q_probs = torch.softmax(logits_q, dim=-1)

    a = _run(p_probs, q_probs, input_kind="probs")
    b = _run(logits_p, logits_q, input_kind="logits")
    assert abs(float(a["kl"]) - float(b["kl"])) < 1e-5


def test_reduction_none_gives_per_sample():
    p = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    q = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    res = _run(p, q, reduction="none")
    assert res["kl"].shape == (2,)
    # First sample: log(2). Second sample: 0.
    assert abs(float(res["kl"][0]) - math.log(2.0)) < 1e-4
    assert abs(float(res["kl"][1])) < 1e-5


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        _run(torch.zeros(3, 5), torch.zeros(3, 4))


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        KLDivergenceNode().execute(
            {"p": torch.zeros(3, 5)},
            {"input_kind": "probs", "reduction": "batchmean"},
        )


# ── #707: reductions aggregate per-sample KL, whatever the input rank ──

_P = [0.5, 0.5]
_Q = [0.9, 0.1]
# KL(p || q) by hand: 0.5·ln(0.5/0.9) + 0.5·ln(0.5/0.1)
_KL_PQ = 0.5 * math.log(0.5 / 0.9) + 0.5 * math.log(0.5 / 0.1)


def test_one_dimensional_input_is_one_sample_with_batchmean():
    res = _run(torch.tensor(_P), torch.tensor(_Q))
    assert float(res["kl"]) == pytest.approx(_KL_PQ, abs=1e-5)
    assert float(res["kl"]) == pytest.approx(0.5108, abs=1e-4)


def test_mean_equals_batchmean_without_a_torch_warning(recwarn):
    p = torch.tensor([_P, [0.2, 0.8], [0.7, 0.3]])
    q = torch.tensor([_Q, [0.5, 0.5], [0.1, 0.9]])
    hand = [
        sum(pi * math.log(pi / qi) for pi, qi in zip(prow, qrow))
        for prow, qrow in zip(p.tolist(), q.tolist())
    ]
    mean = float(_run(p, q, reduction="mean")["kl"])
    batchmean = float(_run(p, q, reduction="batchmean")["kl"])
    assert mean == pytest.approx(sum(hand) / 3, abs=1e-5)
    assert batchmean == pytest.approx(mean, abs=1e-7)
    assert not [w for w in recwarn if "reduction" in str(w.message)]


def test_sum_and_none_on_a_batch_match_the_hand_computed_kl():
    p = torch.tensor([_P, [1.0, 0.0]])
    q = torch.tensor([_Q, [0.5, 0.5]])
    expected = [_KL_PQ, math.log(2.0)]
    none = _run(p, q, reduction="none")["kl"]
    assert none.tolist() == pytest.approx(expected, abs=1e-5)
    assert float(_run(p, q, reduction="sum")["kl"]) == pytest.approx(sum(expected), abs=1e-5)


def test_leading_axes_are_flattened_into_samples_for_the_mean():
    p = torch.tensor([[_P, [1.0, 0.0]], [[0.5, 0.5], _P]])
    q = torch.tensor([[_Q, [0.5, 0.5]], [[0.5, 0.5], _Q]])
    expected = (_KL_PQ + math.log(2.0) + 0.0 + _KL_PQ) / 4
    assert float(_run(p, q)["kl"]) == pytest.approx(expected, abs=1e-5)
