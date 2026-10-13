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


# ── 2026-10-13 audit: one class order for every classifier, and bools ───
# LogisticRegression, MLP, SVM, RF and DT fit on label strings, so 12
# integer classes sorted as '0', '1', '10', '11', '2', ... while KNN kept
# numeric order: the probabilities columns meant different classes per node.


def _twelve_classes():
    torch.manual_seed(0)
    labels = torch.arange(12).repeat_interleave(6)
    features = torch.randn(72, 3) * 0.1 + labels.float().unsqueeze(1)
    return features, labels


def _classifiers():
    from app.nodes.classical.decision_tree_classifier_node import (
        DecisionTreeClassifierNode,
    )
    from app.nodes.classical.mlp_classifier_node import MLPClassifierNode
    from app.nodes.classical.random_forest_classifier_node import (
        RandomForestClassifierNode,
    )
    from app.nodes.classical.svm_classifier_node import SVMClassifierNode

    return [
        (KNNNode, {}),
        (LogisticRegressionNode, {"max_iter": 500}),
        (MLPClassifierNode, {"max_iter": 50}),
        (SVMClassifierNode, {}),
        (RandomForestClassifierNode, {"n_estimators": 5}),
        (DecisionTreeClassifierNode, {}),
    ]


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("as_text", [False, True], ids=["tensor", "csv_strings"])
def test_every_classifier_orders_numeric_classes_numerically(as_text):
    from app.core.param_defaults import fill_missing_params

    features, labels = _twelve_classes()
    y = [str(int(v)) for v in labels] if as_text else labels
    expected = [str(i) for i in range(12)]
    for node_cls, params in _classifiers():
        out = node_cls().execute(
            {"x_train": features, "y_train": y, "x_query": features[:6]},
            fill_missing_params(node_cls, params))
        assert out["classes"] == expected, node_cls.NODE_NAME
        assert all(p in expected for p in out["predictions"]), node_cls.NODE_NAME


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_probability_columns_follow_the_classes_output():
    from app.core.param_defaults import fill_missing_params

    features, labels = _twelve_classes()
    out = LogisticRegressionNode().execute(
        {"x_train": features, "y_train": labels, "x_query": features},
        fill_missing_params(LogisticRegressionNode, {"max_iter": 500}))
    picked = [out["classes"][i] for i in out["probabilities"].argmax(dim=1).tolist()]
    assert picked == out["predictions"]


def test_text_labels_keep_text_order():
    from app.nodes.classical._labels import class_order

    assert class_order(["b", "a", "10", "2"]) == ["10", "2", "a", "b"]
    assert class_order(["10", "2", "-1", "0.5"]) == ["-1", "0.5", "2", "10"]


def test_predictions_keep_the_label_spelling():
    features, labels = _two_clusters()
    y = ["cat"] * 5 + ["dog"] * 5
    out = KNNNode().execute(
        {"x_train": features, "y_train": y, "x_query": features},
        {"n_neighbors": 1})
    assert out["predictions"] == y
    assert out["classes"] == ["cat", "dog"]


def test_accuracy_treats_true_as_1():
    acc = AccuracyNode().execute(
        {"predictions": [True, False, True], "labels": [1, 0, 0]}, {})
    assert acc["accuracy"] == pytest.approx(2 / 3)


def test_accuracy_treats_a_bool_tensor_as_0_and_1():
    acc = AccuracyNode().execute(
        {"predictions": torch.tensor([True, False]), "labels": ["1", "0"]}, {})
    assert acc["accuracy"] == 1.0
