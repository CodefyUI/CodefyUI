"""TensorDataset's class numbering across a train and a test split.

Each TensorDataset numbers the labels it is given. A test split rarely holds
every class, so numbering it on its own gives a class a different index than
training did, and a perfect model scores 0% with no error. The ``classes``
input takes the training node's order instead.
"""

from __future__ import annotations

import pytest
import torch

from app.nodes.data.tensor_dataset_node import TensorDatasetNode
from app.nodes.data.train_test_split_node import TrainTestSplitNode


def _build(features, targets, classes=None):
    inputs = {"features": features, "targets": targets}
    if classes is not None:
        inputs["classes"] = classes
    return TensorDatasetNode().execute(inputs, {"target_type": "class"})


def test_classes_is_an_optional_list_input():
    from app.core.node_base import DataType

    port = {p.name: p for p in TensorDatasetNode.define_inputs()}["classes"]
    assert port.data_type == DataType.LIST and port.optional


def test_a_test_split_missing_a_class_uses_the_training_indices():
    labels = ["a"] * 10 + ["b"] * 3 + ["c"] * 10
    x = torch.arange(len(labels), dtype=torch.float32).unsqueeze(1)
    split = TrainTestSplitNode().execute(
        {"features": x, "labels": labels}, {"test_size": 0.2, "seed": 0, "stratify": False}
    )
    assert sorted(set(split["y_test"])) == ["b", "c"]  # 'a' is missing
    train = _build(split["x_train"], split["y_train"])
    test = _build(split["x_test"], split["y_test"], classes=train["classes"])
    assert test["classes"] == train["classes"] == ["a", "b", "c"]
    targets = [int(t) for _, t in test["dataset"]]
    assert targets == [train["classes"].index(label) for label in split["y_test"]]


def test_without_the_classes_input_the_numbering_is_unchanged():
    out = _build(torch.zeros(3, 1), ["c", "b", "c"])
    assert out["classes"] == ["b", "c"]
    assert [int(t) for _, t in out["dataset"]] == [1, 0, 1]


def test_a_label_the_training_split_never_had_names_the_node():
    with pytest.raises(ValueError, match=r"TensorDataset: label\(s\) 'z' are not in the `classes` input"):
        _build(torch.zeros(2, 1), ["a", "z"], classes=["a", "b"])


def test_numeric_labels_match_numeric_classes():
    train = _build(torch.zeros(3, 1), [0.0, 1.0, 2.0])
    test = _build(torch.zeros(2, 1), [2.0, 2.0], classes=train["classes"])
    assert [int(t) for _, t in test["dataset"]] == [2, 2]


def test_integer_tensor_targets_are_checked_against_the_classes():
    out = _build(torch.zeros(2, 1), torch.tensor([0, 1]), classes=["x", "y", "z"])
    assert out["classes"] == ["x", "y", "z"]
    with pytest.raises(ValueError, match="TensorDataset: the targets are class indices up to 3"):
        _build(torch.zeros(2, 1), torch.tensor([0, 3]), classes=["x", "y"])


def test_a_classes_input_with_a_repeated_label_is_refused():
    with pytest.raises(ValueError, match="repeats a label"):
        _build(torch.zeros(1, 1), ["a"], classes=["a", "a"])
