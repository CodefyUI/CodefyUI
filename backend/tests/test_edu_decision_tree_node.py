"""Tests for EduDecisionTreeNode."""

from __future__ import annotations

import json

import pytest
import torch

from cdui_plugins.foundations.nodes.edu_decision_tree_node import EduDecisionTreeNode


class _Ctx:
    """Minimal stand-in for ExecutionContext with a verbose flag."""

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose


def _run(x_train, y_train, x_query, *, context=None, **params):
    p = {"max_depth": 3, "min_samples_split": 2, "criterion": "gini"}
    p.update(params)
    return EduDecisionTreeNode().execute(
        {"x_train": x_train, "y_train": y_train, "x_query": x_query},
        p,
        context=context,
    )


def test_node_metadata():
    assert EduDecisionTreeNode.NODE_NAME == "Edu-DecisionTree"
    assert EduDecisionTreeNode.CATEGORY == "Classical"
    out_names = [p.name for p in EduDecisionTreeNode.define_outputs()]
    assert out_names == ["predictions", "tree", "node_count"]


def test_separable_one_feature_root_split():
    """x < 0 -> class 0, x > 0 -> class 1: root splits feature 0 near 0."""
    x_train = torch.tensor(
        [[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]]
    )
    y_train = torch.tensor([0, 0, 0, 1, 1, 1])
    x_query = torch.tensor([[-1.5], [-0.5], [0.5], [2.5]])
    res = _run(x_train, y_train, x_query)

    tree = res["tree"]
    assert tree.get("leaf") is None  # root is internal
    assert tree["feature"] == 0
    # Midpoint between -1 and 1 is 0.0.
    assert abs(tree["threshold"]) < 1.0
    assert res["predictions"].tolist() == [0, 0, 1, 1]


def test_two_blocks_full_train_accuracy():
    """Two axis-aligned blocks separable with depth-2 splits -> 100% train acc."""
    # Class 0 lower-left & lower-right, class 1 upper band — needs >=2 splits.
    x_train = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 5.0],
        [1.0, 5.0],
        [5.0, 0.0],
        [5.0, 1.0],
    ])
    # XOR-ish: class 1 when exactly one coordinate is "large".
    y_train = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    res = _run(x_train, y_train, x_train, max_depth=3)
    assert res["predictions"].tolist() == y_train.tolist()


def test_two_d_needs_depth_two_full_train_accuracy():
    """A 2-D dataset whose boundary needs two nested axis-aligned splits.

    Region rule:
        feature 0 <= 2  -> class 0   (left band)
        feature 0  > 2 and feature 1 <= 2 -> class 1  (lower-right)
        feature 0  > 2 and feature 1  > 2 -> class 2  (upper-right)
    A single split cannot separate all three classes; depth >= 2 can, and
    a greedy tree fits it exactly because each split has positive gain.
    """
    x_train = torch.tensor([
        [0.0, 0.0], [1.0, 4.0], [0.0, 5.0],   # left band -> 0
        [4.0, 0.0], [5.0, 1.0], [4.0, 0.5],   # lower-right -> 1
        [4.0, 5.0], [5.0, 4.0], [4.5, 5.0],   # upper-right -> 2
    ])
    y_train = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    res = _run(x_train, y_train, x_train, max_depth=2)
    assert res["predictions"].tolist() == y_train.tolist()
    # Tree must actually be at least depth 2 to separate three classes.
    node = res["tree"]
    assert node.get("leaf") is None
    deeper = node["left"] if node["left"].get("leaf") is None else node["right"]
    assert deeper.get("leaf") is None  # a grandchild split exists


def test_tree_is_json_serializable_and_well_shaped():
    x_train = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
    y_train = torch.tensor([0, 0, 1, 1])
    res = _run(x_train, y_train, x_train)
    tree = res["tree"]

    # JSON round-trips cleanly.
    dumped = json.dumps(tree)
    assert isinstance(dumped, str)

    # Internal-node shape.
    for key in ("feature", "threshold", "impurity", "gain", "n_samples", "left", "right"):
        assert key in tree
    assert isinstance(tree["feature"], int)
    assert isinstance(tree["threshold"], float)
    assert isinstance(tree["n_samples"], int)

    # Walk to a leaf and check its shape.
    leaf = tree["left"]
    while not leaf.get("leaf"):
        leaf = leaf["left"]
    assert leaf["leaf"] is True
    assert isinstance(leaf["prediction"], int)
    assert isinstance(leaf["n_samples"], int)
    assert isinstance(leaf["class_counts"], dict)
    # class_counts keys are strings (JSON-safe), values are ints.
    for k, v in leaf["class_counts"].items():
        assert isinstance(k, str)
        assert isinstance(v, int)


def test_max_depth_respected():
    """A deep, separable dataset capped at max_depth=1 yields a single split."""
    # 4 well-separated clusters on feature 0 -> would normally build a deep tree.
    x_train = torch.tensor(
        [[0.0], [1.0], [10.0], [11.0], [20.0], [21.0], [30.0], [31.0]]
    )
    y_train = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    res = _run(x_train, y_train, x_train, max_depth=1)
    tree = res["tree"]
    # Root is one split; both children must be leaves (no depth beyond 1).
    assert tree.get("leaf") is None
    assert tree["left"].get("leaf") is True
    assert tree["right"].get("leaf") is True
    # node_count = root + 2 leaves = 3.
    assert int(res["node_count"].item()) == 3


def test_shape_mismatch_raises():
    # x_query feature count differs from x_train.
    with pytest.raises(ValueError):
        _run(
            torch.zeros(4, 2),
            torch.tensor([0, 1, 0, 1]),
            torch.zeros(2, 3),
        )


def test_label_count_mismatch_raises():
    with pytest.raises(ValueError, match="mismatch"):
        _run(
            torch.zeros(4, 2),
            torch.tensor([0, 1, 0]),
            torch.zeros(1, 2),
        )


def test_empty_training_set_raises():
    with pytest.raises(ValueError, match="empty"):
        _run(torch.zeros(0, 2), torch.zeros(0, dtype=torch.long), torch.zeros(1, 2))


def test_bad_params_raise():
    with pytest.raises(ValueError, match="max_depth"):
        _run(torch.tensor([[0.0], [1.0]]), torch.tensor([0, 1]), torch.tensor([[0.5]]), max_depth=0)
    with pytest.raises(ValueError, match="min_samples_split"):
        _run(
            torch.tensor([[0.0], [1.0]]),
            torch.tensor([0, 1]),
            torch.tensor([[0.5]]),
            min_samples_split=1,
        )


def test_entropy_criterion_also_works():
    x_train = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
    y_train = torch.tensor([0, 0, 1, 1])
    res = _run(x_train, y_train, x_train, criterion="entropy")
    assert res["predictions"].tolist() == [0, 0, 1, 1]


def test_verbose_emits_steps_non_verbose_does_not():
    x_train = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
    y_train = torch.tensor([0, 0, 1, 1])

    quiet = _run(x_train, y_train, x_train)
    assert "__steps__" not in quiet

    loud = _run(x_train, y_train, x_train, context=_Ctx(verbose=True))
    steps = loud["__steps__"]
    assert len(steps) >= 1
    names = [s.name for s in steps]
    # At least one per-split step plus the final "tree" summary.
    assert any(n.startswith("split_d") for n in names)
    assert names[-1] == "tree"
    # The per-split step carries the documented scalars.
    split_step = next(s for s in steps if s.name.startswith("split_d"))
    for key in ("feature", "threshold", "impurity_before", "impurity_after", "gain", "n_samples"):
        assert key in split_step.scalars
    assert set(steps[-1].scalars) == {"node_count", "depth"}


def test_determinism_repeated_runs_match():
    x_train = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y_train = torch.tensor([0, 1, 1, 0])
    a = _run(x_train, y_train, x_train, max_depth=2)
    b = _run(x_train, y_train, x_train, max_depth=2)
    assert a["predictions"].tolist() == b["predictions"].tolist()
    assert json.dumps(a["tree"]) == json.dumps(b["tree"])
