"""Tests for LogisticRegressionNode (sklearn wrapper)."""

from __future__ import annotations

import math
import warnings

import pytest
import sklearn
import sklearn.linear_model
import torch

from app.nodes.classical.logistic_regression_node import LogisticRegressionNode

# scikit-learn announces a removal a release or two ahead with a FutureWarning
# (sometimes a DeprecationWarning). Failing on one keeps the node from passing
# an argument that is about to go, on whichever version CI resolves: 1.7.x on
# Python 3.10, the newest release on 3.11 and 3.12.
pytestmark = [
    pytest.mark.filterwarnings("error::FutureWarning"),
    pytest.mark.filterwarnings("error::DeprecationWarning"),
]


def _run(x_train, y_train, x_query, **params):
    p = {"C": 1.0, "max_iter": 200, "penalty": "l2"}
    p.update(params)
    return LogisticRegressionNode().execute(
        {"x_train": x_train, "y_train": y_train, "x_query": x_query}, p
    )


def _signal_and_noise():
    """400 rows: 3 features that set the label, then 9 of pure noise.

    Label noise keeps the two classes overlapping, so the unpenalised fit has
    finite coefficients.
    """
    g = torch.Generator().manual_seed(0)
    signal = torch.randn(400, 3, generator=g)
    noise = torch.randn(400, 9, generator=g)
    logits = signal @ torch.tensor([2.0, -1.5, 1.0]) + 0.5 * torch.randn(400, generator=g)
    y = ["a" if v > 0 else "b" for v in logits.tolist()]
    return torch.cat([signal, noise], dim=1), y


def test_node_metadata():
    assert LogisticRegressionNode.NODE_NAME == "LogisticRegression"
    assert LogisticRegressionNode.CATEGORY == "Classical"
    out_names = [p.name for p in LogisticRegressionNode.define_outputs()]
    assert set(out_names) >= {"predictions", "probabilities", "classes", "coef"}


def test_separates_two_clusters():
    """Two well-separated clusters → near-perfect classification."""
    torch.manual_seed(0)
    a = torch.randn(50, 2) + torch.tensor([5.0, 5.0])
    b = torch.randn(50, 2) + torch.tensor([-5.0, -5.0])
    x = torch.cat([a, b], dim=0)
    y = ["a"] * 50 + ["b"] * 50
    res = _run(x, y, x)
    correct = sum(1 for i, p in enumerate(res["predictions"]) if p == y[i])
    assert correct >= 95


def test_probabilities_sum_to_one():
    torch.manual_seed(1)
    x = torch.randn(40, 3)
    y = ["a"] * 20 + ["b"] * 20
    res = _run(x, y, x)
    sums = res["probabilities"].sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_classes_match_columns():
    torch.manual_seed(2)
    x = torch.randn(30, 2)
    y = ["a"] * 10 + ["b"] * 10 + ["c"] * 10
    res = _run(x, y, x)
    assert res["classes"] == ["a", "b", "c"]
    assert res["probabilities"].shape == (30, 3)


def test_works_with_integer_labels():
    x = torch.randn(20, 2, generator=torch.Generator().manual_seed(0))
    y = torch.tensor([0] * 10 + [1] * 10)
    res = _run(x, y, x)
    # Predictions get stringified for consistency
    assert all(isinstance(p, str) for p in res["predictions"])


def test_label_count_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        _run(torch.zeros(5, 2), ["a"] * 3, torch.zeros(1, 2))


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        LogisticRegressionNode().execute(
            {"x_train": torch.zeros(5, 2)},
            {"C": 1.0, "max_iter": 200, "penalty": "l2"},
        )


def test_single_class_raises():
    with pytest.raises(ValueError, match="class"):
        _run(torch.randn(10, 2), ["a"] * 10, torch.randn(2, 2))


@pytest.mark.parametrize("penalty", ["l2", "l1", "none"])
def test_each_penalty_fits_three_classes(penalty):
    """Each option fits on the installed scikit-learn with a solver that takes
    more than two classes, and without a deprecation warning."""
    g = torch.Generator().manual_seed(0)
    centres = torch.tensor([[2.0, 0.0], [-1.0, 1.7], [-1.0, -1.7]])
    x = torch.cat([torch.randn(30, 2, generator=g) + c for c in centres])
    y = ["a"] * 30 + ["b"] * 30 + ["c"] * 30
    res = _run(x, y, x, penalty=penalty, C=0.5)
    assert res["classes"] == ["a", "b", "c"]
    assert res["coef"].shape == (3, 2)
    assert sum(p == t for p, t in zip(res["predictions"], y)) >= 80


def test_l1_zeroes_the_noise_features_and_l2_does_not():
    """Before scikit-learn 1.8, l1_ratio only counts with penalty='elasticnet',
    so l1_ratio=1 there quietly fits an L2 model. The Python 3.10 CI job
    resolves 1.7.x and would catch that here."""
    x, y = _signal_and_noise()
    l1 = _run(x, y, x, penalty="l1", C=0.05)["coef"][0]
    l2 = _run(x, y, x, penalty="l2", C=0.05)["coef"][0]
    assert (l1[3:] == 0).sum() >= 6
    assert (l1[:3] != 0).all()
    assert (l2 != 0).all()


def test_none_matches_a_very_large_C_fit():
    """none leaves the node's C unused: it fits what l2 fits as C grows."""
    x, y = _signal_and_noise()
    none = _run(x, y, x, penalty="none", C=0.05)["coef"]
    huge_c = _run(x, y, x, penalty="l2", C=1e10)["coef"]
    small_c = _run(x, y, x, penalty="l2", C=0.05)["coef"]
    assert torch.allclose(none, huge_c, rtol=1e-4, atol=1e-5)
    assert none.norm() > 3 * small_c.norm()


def test_l2_matches_scikit_learns_default():
    """l2 is scikit-learn's default penalty; the node fits exactly that model."""
    x, y = _signal_and_noise()
    res = _run(x, y, x, penalty="l2", C=0.05)
    ref = sklearn.linear_model.LogisticRegression(C=0.05, max_iter=200).fit(x.numpy(), y)
    assert torch.allclose(res["coef"], torch.from_numpy(ref.coef_).float(), rtol=0, atol=1e-6)


@pytest.mark.filterwarnings("error:Setting penalty=None:UserWarning")
def test_none_does_not_warn_that_C_is_ignored():
    """scikit-learn warns that C is ignored when penalty=None comes with a C
    other than 1.0, and 1.8.x does so for C=inf too, the newer spelling of
    none. The node never uses C for none, so the warning says nothing about
    the graph."""
    x, y = _signal_and_noise()
    _run(x, y, x, penalty="none", C=0.05)


def test_an_unknown_penalty_raises():
    """A value outside the three options must not quietly fit an L2 model."""
    with pytest.raises(ValueError, match="penalty must be l2, l1 or none"):
        _run(torch.randn(4, 2), ["a", "b", "a", "b"], torch.randn(1, 2), penalty="elasticnet")


@pytest.mark.parametrize("penalty", ["l2", "l1", "none"])
@pytest.mark.parametrize("C", [0.0, -1.0, math.nan])
def test_a_C_that_is_not_positive_raises(penalty, C):
    """For every option and every scikit-learn version. From 1.8 the none
    spelling hands scikit-learn C=inf instead, so its own check never sees
    the node's C."""
    with pytest.raises(ValueError, match="C must be greater than 0"):
        _run(torch.randn(4, 2), ["a", "b", "a", "b"], torch.randn(1, 2), penalty=penalty, C=C)


@pytest.mark.parametrize(("penalty", "filtered"), [("l2", False), ("l1", False), ("none", True)])
def test_only_none_filters_warnings_around_the_fit(monkeypatch, penalty, filtered):
    """Python's warning filters are process-wide and nodes run in worker
    threads, so the node filters "C is ignored" only around the fit that can
    raise it. A stand-in fit raises a copy of that message to see."""
    real = sklearn.linear_model.LogisticRegression

    class Probe(real):
        def fit(self, X, y):
            warnings.warn("Setting penalty=None will ignore the C (probe)", UserWarning)
            return super().fit(X, y)

    monkeypatch.setattr(sklearn.linear_model, "LogisticRegression", Probe)
    x, y = _signal_and_noise()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _run(x, y, x, penalty=penalty)
    probes = [w for w in caught if "(probe)" in str(w.message)]
    assert (not probes) is filtered


class _Constructed(Exception):
    """Raised by the stand-in estimator to stop ``execute`` once it is built."""


#: What the node passes to LogisticRegression (besides max_iter) on
#: scikit-learn before 1.8, where ``penalty`` is the only spelling ...
_PENALTY_SPELLING = {
    "l2": {"C": 0.5, "penalty": "l2", "solver": "lbfgs"},
    "l1": {"C": 0.5, "penalty": "l1", "solver": "saga"},
    "none": {"C": 0.5, "penalty": None, "solver": "lbfgs"},
}
#: ... and from 1.8, which deprecates ``penalty`` for ``l1_ratio`` and ``C``.
_L1_RATIO_SPELLING = {
    "l2": {"C": 0.5, "l1_ratio": 0.0, "solver": "lbfgs"},
    "l1": {"C": 0.5, "l1_ratio": 1.0, "solver": "saga"},
    "none": {"C": math.inf, "solver": "lbfgs"},
}
#: Versions either side of 1.8, with the spelling each must get.
_SPELLING_BY_VERSION = {
    "1.3.0": _PENALTY_SPELLING,
    "1.7.2": _PENALTY_SPELLING,
    "1.8.0rc1": _L1_RATIO_SPELLING,
    "1.9.1": _L1_RATIO_SPELLING,
    "1.10.dev0": _L1_RATIO_SPELLING,
}


@pytest.mark.parametrize("penalty", ["l2", "l1", "none"])
@pytest.mark.parametrize("version", list(_SPELLING_BY_VERSION))
def test_the_spelling_follows_the_installed_version(monkeypatch, version, penalty):
    """The fits above run the installed version's spelling for real; this
    covers both on every CI job by faking the version."""
    seen = {}

    def build(**kwargs):
        seen.update(kwargs)
        raise _Constructed

    monkeypatch.setattr(sklearn, "__version__", version)
    monkeypatch.setattr(sklearn.linear_model, "LogisticRegression", build)
    with pytest.raises(_Constructed):
        _run(torch.randn(4, 2), ["a", "b", "a", "b"], torch.randn(1, 2), penalty=penalty, C=0.5)
    assert seen == {**_SPELLING_BY_VERSION[version][penalty], "max_iter": 200}
