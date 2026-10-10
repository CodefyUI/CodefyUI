"""Tests for BradleyTerryLossNode and BradleyTerryTrainNode.

Shift invariance is I5-3's first experiment and the train/holdout gap is its
second, so both are pinned to the numbers the chapter prints.
"""

from __future__ import annotations

import asyncio

import pytest
import torch
import torch.nn as nn

from app.core.graph_engine import execute_graph
from app.nodes.rl.bradley_terry_node import BradleyTerryLossNode, BradleyTerryTrainNode
from app.nodes.rl.preference_dataset_node import PreferenceDatasetNode


def _loss(reward_w, reward_l):
    return BradleyTerryLossNode().execute(
        {"reward_w": torch.as_tensor(reward_w, dtype=torch.float32),
         "reward_l": torch.as_tensor(reward_l, dtype=torch.float32)}, {})


# ── BradleyTerryLoss ────────────────────────────────────────────────────

def test_loss_node_metadata():
    assert BradleyTerryLossNode.NODE_NAME == "BradleyTerryLoss"
    assert BradleyTerryLossNode.CATEGORY == "RL"
    out_names = [p.name for p in BradleyTerryLossNode.define_outputs()]
    for expected in ("loss", "preference_prob", "score_diff", "accuracy"):
        assert expected in out_names


def test_reproduces_i5_3_worked_pair():
    out = _loss([1.82], [-0.44])
    assert float(out["score_diff"]) == pytest.approx(2.26, abs=1e-4)
    assert float(out["preference_prob"]) == pytest.approx(0.905510, abs=1e-5)


def test_only_the_difference_matters():
    """I5-3 experiment 1: add 100 to both scores, nothing moves. This is why
    RLHF never needs an absolute human score."""
    base = _loss([1.82], [-0.44])
    shifted = _loss([101.82], [99.56])
    assert float(shifted["score_diff"]) == pytest.approx(float(base["score_diff"]), abs=1e-4)
    assert float(shifted["preference_prob"]) == pytest.approx(
        float(base["preference_prob"]), abs=1e-5)
    assert float(shifted["loss"]) == pytest.approx(float(base["loss"]), abs=1e-5)


def test_equal_scores_are_a_coin_flip():
    out = _loss([2.0], [2.0])
    assert float(out["preference_prob"]) == pytest.approx(0.5)


def test_probability_saturates_as_the_gap_widens():
    near = float(_loss([1.0], [0.0])["preference_prob"])
    far = float(_loss([5.0], [0.0])["preference_prob"])
    assert far > near
    assert far < 1.0


def test_loss_falls_as_the_model_gets_the_pair_more_right():
    wrong = float(_loss([0.0], [3.0])["loss"])
    right = float(_loss([3.0], [0.0])["loss"])
    assert right < wrong


def test_accuracy_counts_correctly_ordered_pairs():
    out = _loss([1.0, 0.0, 2.0, 5.0], [0.0, 1.0, 1.0, 4.0])
    assert out["accuracy"] == pytest.approx(0.75)


def test_loss_is_stable_at_a_huge_gap():
    """softplus(-x), not -log(sigmoid(x)): no overflow, no NaN."""
    out = _loss([500.0], [-500.0])
    assert torch.isfinite(out["loss"])
    assert float(out["loss"]) == pytest.approx(0.0, abs=1e-6)


def test_mismatched_halves_are_rejected():
    with pytest.raises(ValueError, match="two halves"):
        _loss([1.0, 2.0], [1.0])


def test_loss_accepts_plain_lists():
    out = BradleyTerryLossNode().execute({"reward_w": [1.0], "reward_l": [0.0]}, {})
    assert float(out["preference_prob"]) > 0.5


# ── BradleyTerryTrain ───────────────────────────────────────────────────

def _dataset(**params):
    p = {"n_pairs": 512, "holdout_pairs": 256, "feature_dim": 16,
         "signal_dims": 8, "shortcut_strength": 3.0, "seed": 0}
    p.update(params)
    return PreferenceDatasetNode().execute({}, p)


def _train(data, **params):
    p = {"epochs": 60, "hidden_dim": 32, "lr": 0.01, "seed": 0}
    p.update(params)
    return BradleyTerryTrainNode().execute(
        {k: data[k] for k in ("train_w", "train_l", "holdout_w", "holdout_l")}, p)


def test_train_node_metadata():
    assert BradleyTerryTrainNode.NODE_NAME == "BradleyTerryTrain"
    assert BradleyTerryTrainNode.CATEGORY == "RL"
    out_names = [p.name for p in BradleyTerryTrainNode.define_outputs()]
    for expected in ("model", "losses", "train_accuracy", "holdout_accuracy",
                     "best_holdout_epoch", "report"):
        assert expected in out_names


def test_train_is_not_cacheable():
    """It owns fitted weights; a cache hit would hand back a flat curve."""
    assert BradleyTerryTrainNode.cacheable is False


def test_returns_a_fitted_module_and_per_epoch_curves():
    out = _train(_dataset(), epochs=5)
    assert isinstance(out["model"], nn.Module)
    for key in ("losses", "train_accuracy", "holdout_accuracy"):
        assert out[key].shape == (5,)


def test_reproduces_i5_3_control_run():
    """shortcut_strength 0: the reward model generalises."""
    out = _train(_dataset(shortcut_strength=0.0))
    assert float(out["train_accuracy"][-1]) == pytest.approx(1.0000, abs=1e-4)
    assert float(out["holdout_accuracy"][-1]) == pytest.approx(0.9609, abs=1e-3)


def test_reproduces_i5_3_hacked_run():
    """shortcut_strength 3: training accuracy is unchanged, holdout collapses."""
    out = _train(_dataset(shortcut_strength=3.0))
    assert float(out["train_accuracy"][-1]) == pytest.approx(1.0000, abs=1e-4)
    assert float(out["holdout_accuracy"][-1]) == pytest.approx(0.7773, abs=1e-3)


def test_the_training_score_cannot_see_the_shortcut():
    """The whole lesson: both runs look identical from the training set."""
    clean = _train(_dataset(shortcut_strength=0.0))
    hacked = _train(_dataset(shortcut_strength=3.0))
    assert float(clean["train_accuracy"][-1]) == pytest.approx(
        float(hacked["train_accuracy"][-1]), abs=1e-3)
    assert float(clean["holdout_accuracy"][-1]) - float(hacked["holdout_accuracy"][-1]) > 0.1


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_the_gap_holds_across_seeds(seed):
    """Measured across 5 seeds the two holdout bands never overlap."""
    clean = _train(_dataset(shortcut_strength=0.0, seed=seed), seed=seed)
    hacked = _train(_dataset(shortcut_strength=3.0, seed=seed), seed=seed)
    assert float(clean["holdout_accuracy"][-1]) > float(hacked["holdout_accuracy"][-1])


def test_training_loss_goes_down():
    out = _train(_dataset(), epochs=40)
    assert float(out["losses"][-1]) < float(out["losses"][0])


def test_report_has_one_line_per_epoch():
    out = _train(_dataset(n_pairs=32, holdout_pairs=16), epochs=6)
    assert len(out["report"].splitlines()) == 6


def test_holdout_is_optional():
    """Without a holdout split the accuracy column is NaN, not a crash."""
    data = _dataset(n_pairs=32, holdout_pairs=16)
    out = BradleyTerryTrainNode().execute(
        {"train_w": data["train_w"], "train_l": data["train_l"]},
        {"epochs": 3, "hidden_dim": 8, "lr": 0.01, "seed": 0})
    assert torch.isnan(out["holdout_accuracy"]).all()
    assert out["best_holdout_epoch"] == 0


def test_same_seed_is_reproducible():
    data = _dataset(n_pairs=64, holdout_pairs=32)
    a = _train(data, epochs=10)
    b = _train(data, epochs=10)
    assert a["losses"].tolist() == pytest.approx(b["losses"].tolist())


def test_mismatched_train_halves_are_rejected():
    with pytest.raises(ValueError, match="paired"):
        BradleyTerryTrainNode().execute(
            {"train_w": torch.randn(4, 8), "train_l": torch.randn(3, 8)},
            {"epochs": 2, "hidden_dim": 4, "lr": 0.01, "seed": 0})


def test_best_holdout_epoch_is_within_range():
    out = _train(_dataset(n_pairs=64, holdout_pairs=32), epochs=8)
    assert 1 <= out["best_holdout_epoch"] <= 8


# ── the seed alone decides the weights ────────────────────────────────────


def _draw_beside_every_linear(monkeypatch) -> list[torch.Tensor]:
    """Another node drawing from the global RNG in the middle of the build.

    A deterministic stand-in for the engine running a level's nodes on
    worker threads (see ``test_reward_model_node.py``, which has the same
    helper for the same reason).
    """
    drawn: list[torch.Tensor] = []
    original = nn.Linear.reset_parameters

    def interleaved(self):
        drawn.append(torch.rand(1))
        return original(self)

    monkeypatch.setattr(nn.Linear, "reset_parameters", interleaved)
    return drawn


def test_same_seed_trains_the_same_curve_while_another_node_draws(monkeypatch):
    """Two arms on one level of an unseeded run reseeded each other.

    Each called ``torch.manual_seed`` and then built its network from the
    process-global generator, so a neighbour's draw in between changed the
    weights, and with them every curve the node returns.
    """
    data = _dataset(shortcut_strength=3.0)
    alone = _train(data)

    _draw_beside_every_linear(monkeypatch)
    beside = _train(data)

    for key in ("losses", "train_accuracy", "holdout_accuracy"):
        assert torch.equal(alone[key], beside[key]), key
    assert alone["best_holdout_epoch"] == beside["best_holdout_epoch"]


def test_the_weights_are_the_ones_the_seeded_constructor_gave():
    """Bit for bit what ``torch.manual_seed(seed)`` + the constructor gave,
    so the I5-3 numbers pinned above were read off the same network."""
    torch.manual_seed(7)
    expected = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1))

    got = _train(_dataset(), epochs=0, seed=7)["model"]

    for a, b in zip(expected.state_dict().values(), got.state_dict().values()):
        assert torch.equal(a, b)


def test_training_never_reseeds_the_global_rng(monkeypatch):
    """The process-global generator belongs to the run, not to this node."""
    torch.manual_seed(1234)
    first = torch.rand(1)
    torch.manual_seed(1234)
    drawn = _draw_beside_every_linear(monkeypatch)

    _train(_dataset(), epochs=2)

    assert torch.equal(drawn[0], first), (
        "the node reseeded the global RNG: a draw made beside it came from "
        "another stream than the run's")


def _two_arm_graph() -> tuple[list[dict], list[dict]]:
    """A shortcut arm and its control, side by side on one level."""
    def node(id_, type_, **params):
        return {"id": id_, "type": type_, "position": {"x": 0, "y": 0},
                "data": {"params": params}}

    nodes, edges = [node("start", "Start")], []
    for arm, strength in (("hack", 3.0), ("ctrl", 0.0)):
        edges.append({"id": f"t-{arm}", "source": "start", "target": f"pairs-{arm}",
                      "sourceHandle": "trigger", "type": "trigger"})
        nodes += [
            node(f"pairs-{arm}", "PreferenceDataset", n_pairs=512,
                 holdout_pairs=256, feature_dim=16, signal_dims=8,
                 shortcut_strength=strength, seed=0),
            node(f"rm-{arm}", "BradleyTerryTrain", epochs=60, hidden_dim=32,
                 lr=0.01, seed=0),
        ]
        edges += [
            {"id": f"{arm}-{port}", "source": f"pairs-{arm}",
             "sourceHandle": port, "target": f"rm-{arm}", "targetHandle": port}
            for port in ("train_w", "train_l", "holdout_w", "holdout_l")
        ]
    return nodes, edges


def test_two_arms_in_one_unseeded_graph_score_the_same_every_run(
        registry_with_nodes):
    """A SMOKE test; the deterministic guarantee is the monkeypatched one.

    Whether a given run overlaps the two builds is up to the scheduler, so
    this passing proves little -- but before the fix it failed in a few
    runs out of ten, with the shortcut arm ending anywhere from 0.75 to
    0.78 on holdout.
    """
    nodes, edges = _two_arm_graph()
    for _ in range(10):
        results = asyncio.run(execute_graph(nodes, edges, error_mode="fail_fast"))
        hack = results["rm-hack"]["holdout_accuracy"]
        ctrl = results["rm-ctrl"]["holdout_accuracy"]
        assert float(hack[-1]) == pytest.approx(0.7773, abs=1e-3)
        assert float(ctrl[-1]) == pytest.approx(0.9609, abs=1e-3)
