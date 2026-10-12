"""Labels picked as a column, or held as floats, still score.

ColumnSelector picks a label column as ``[N, 1]`` and a CSV column of 0/1
can be float. Read raw, the first became one-element lists (``'[1.0]'``)
and the second ``'1.0'`` against CSVReader's ``'1'``, so Accuracy read 0
with no error even for perfect predictions.
"""

from __future__ import annotations

import pytest
import torch

from app.nodes.classical.accuracy_node import AccuracyNode
from app.nodes.classical.knn_node import KNNNode
from app.nodes.classical.logistic_regression_node import LogisticRegressionNode
from app.nodes.data.train_test_split_node import TrainTestSplitNode


def _two_clusters():
    features = torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1], [0.05, 0.05],
         [5.0, 5.0], [5.1, 5.0], [5.0, 5.1], [5.1, 5.1], [5.05, 5.05]])
    labels = torch.tensor([[0.0]] * 5 + [[1.0]] * 5)  # ColumnSelector's [N, 1]
    return features, labels


def test_a_label_column_splits_and_scores():
    features, labels = _two_clusters()
    split = TrainTestSplitNode().execute(
        {"features": features, "labels": labels},
        {"test_size": 0.4, "seed": 0, "stratify": True})
    assert all(not isinstance(y, list) for y in split["y_train"] + split["y_test"])

    for node in (KNNNode(), LogisticRegressionNode()):
        out = node.execute(
            {"x_train": split["x_train"], "y_train": split["y_train"],
             "x_query": split["x_test"]},
            {"n_neighbors": 3})
        assert set(out["predictions"]) <= {"0", "1"}, out["predictions"]
        acc = AccuracyNode().execute(
            {"predictions": out["predictions"], "labels": split["y_test"]}, {})
        assert acc["accuracy"] == 1.0


@pytest.mark.parametrize("labels", [
    ["0", "1", "1"],
    torch.tensor([0, 1, 1]),
    torch.tensor([0.0, 1.0, 1.0]),
    torch.tensor([[0.0], [1.0], [1.0]]),
])
def test_accuracy_compares_classes_not_their_spelling(labels):
    acc = AccuracyNode().execute({"predictions": ["0", "1.0", "1"], "labels": labels}, {})
    assert acc["accuracy"] == 1.0


def test_accuracy_refuses_probabilities_for_predictions():
    with pytest.raises(ValueError, match=r"shape \(3, 2\)"):
        AccuracyNode().execute(
            {"predictions": torch.rand(3, 2), "labels": ["0", "1", "1"]}, {})


def test_non_integral_labels_pass_through():
    acc = AccuracyNode().execute(
        {"predictions": torch.tensor([0.5, 1.5]), "labels": ["0.5", "1.5"]}, {})
    assert acc["accuracy"] == 1.0
