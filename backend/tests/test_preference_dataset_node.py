"""Tests for PreferenceDatasetNode.

The planted shortcut is what makes I5-3's reward-hacking demonstration
reproducible rather than anecdotal, so the asymmetry between the two splits
is asserted directly.
"""

from __future__ import annotations

import pytest
import torch

from app.nodes.rl.preference_dataset_node import PreferenceDatasetNode


def _run(**params):
    p = {"n_pairs": 512, "holdout_pairs": 256, "feature_dim": 16,
         "signal_dims": 8, "shortcut_strength": 3.0, "seed": 0}
    p.update(params)
    return PreferenceDatasetNode().execute({}, p)


def test_node_metadata():
    assert PreferenceDatasetNode.NODE_NAME == "PreferenceDataset"
    assert PreferenceDatasetNode.CATEGORY == "RL"
    out_names = [p.name for p in PreferenceDatasetNode.define_outputs()]
    for expected in ("train_w", "train_l", "holdout_w", "holdout_l", "feature_dim"):
        assert expected in out_names


def test_shapes_match_the_params():
    out = _run()
    assert out["train_w"].shape == (512, 16)
    assert out["train_l"].shape == (512, 16)
    assert out["holdout_w"].shape == (256, 16)
    assert out["holdout_l"].shape == (256, 16)
    assert out["feature_dim"] == 16


def test_pairs_are_aligned():
    out = _run(n_pairs=32, holdout_pairs=16)
    assert out["train_w"].shape == out["train_l"].shape
    assert out["holdout_w"].shape == out["holdout_l"].shape


def test_same_seed_is_reproducible():
    a, b = _run(seed=7), _run(seed=7)
    assert torch.equal(a["train_w"], b["train_w"])
    assert torch.equal(a["holdout_w"], b["holdout_w"])


def test_different_seeds_give_different_data():
    a, b = _run(seed=0), _run(seed=1)
    assert not torch.equal(a["train_w"], b["train_w"])


def test_shortcut_is_planted_in_train_and_absent_from_holdout():
    """The last coordinate tracks quality in train, and is noise in holdout.
    This asymmetry IS the reward-hacking mechanism."""
    out = _run(shortcut_strength=3.0)
    train_gap = (out["train_w"][:, -1] - out["train_l"][:, -1]).mean()
    holdout_gap = (out["holdout_w"][:, -1] - out["holdout_l"][:, -1]).mean()
    assert float(train_gap) > 1.0        # the winner is loudly marked
    assert abs(float(holdout_gap)) < 0.3  # no such marking survives


def test_shortcut_strength_zero_removes_the_marking():
    """The control run: with no shortcut, train looks like holdout."""
    out = _run(shortcut_strength=0.0)
    train_gap = (out["train_w"][:, -1] - out["train_l"][:, -1]).mean()
    assert abs(float(train_gap)) < 0.3


def test_stronger_shortcut_marks_the_winner_harder():
    weak = _run(shortcut_strength=1.0)
    loud = _run(shortcut_strength=6.0)
    weak_gap = float((weak["train_w"][:, -1] - weak["train_l"][:, -1]).mean())
    loud_gap = float((loud["train_w"][:, -1] - loud["train_l"][:, -1]).mean())
    assert loud_gap > weak_gap


def test_the_shortcut_coordinate_is_the_only_one_treated_differently():
    """A signal coordinate separates the pair by the same amount in BOTH
    splits -- the two splits are the same problem. Only the last coordinate
    is manufactured in one split and not the other, so any train/holdout gap
    a model shows is attributable to the shortcut alone."""
    out = _run(shortcut_strength=3.0)

    def gap(split, col):
        return float((out[f"{split}_w"][:, col] - out[f"{split}_l"][:, col]).mean())

    signal_train, signal_holdout = gap("train", 0), gap("holdout", 0)
    assert abs(signal_train - signal_holdout) < 0.25

    shortcut_train, shortcut_holdout = gap("train", -1), gap("holdout", -1)
    assert abs(shortcut_train - shortcut_holdout) > 1.0


def test_labels_follow_the_same_quality_direction_in_both_splits():
    """One fixed weight vector defines "quality" for both splits, so the mean
    (winner - loser) vector over the signal coordinates must point the same
    way in each. If it did not, the holdout would be a different problem and
    the train/holdout gap I5-3 measures would prove nothing."""
    out = _run(shortcut_strength=3.0, seed=3)
    k = 8
    train_dir = (out["train_w"][:, :k] - out["train_l"][:, :k]).mean(dim=0)
    holdout_dir = (out["holdout_w"][:, :k] - out["holdout_l"][:, :k]).mean(dim=0)
    cosine = torch.nn.functional.cosine_similarity(train_dir, holdout_dir, dim=0)
    assert float(cosine) > 0.95


def test_signal_dims_must_be_smaller_than_feature_dim():
    """The last coordinate is reserved for the shortcut."""
    with pytest.raises(ValueError, match="signal_dims"):
        _run(feature_dim=8, signal_dims=8)


def test_small_configurations_still_work():
    out = _run(n_pairs=4, holdout_pairs=2, feature_dim=4, signal_dims=2)
    assert out["train_w"].shape == (4, 4)
    assert out["holdout_w"].shape == (2, 4)


def test_outputs_are_float_tensors():
    out = _run(n_pairs=8, holdout_pairs=4)
    for key in ("train_w", "train_l", "holdout_w", "holdout_l"):
        assert out[key].dtype == torch.float32
        assert torch.isfinite(out[key]).all()
