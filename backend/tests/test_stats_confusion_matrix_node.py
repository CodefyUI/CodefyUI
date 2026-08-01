"""Stats-ConfusionMatrix: sklearn parity across every normalize mode.

scikit-learn is already a backend dependency (the classical-ML nodes use it)
and its `confusion_matrix` is the definition everyone compares against, so it
is the reference here — including its 0/0 -> 0 resolution, which is the part a
hand-written implementation usually gets wrong.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from sklearn.metrics import confusion_matrix

from cdui_plugins.stats.nodes.stats_confusion_matrix_node import (
    StatsConfusionMatrixNode,
)

TRUE = ["cat", "cat", "dog", "dog", "dog", "bird", "bird"]
PRED = ["cat", "dog", "dog", "dog", "cat", "bird", "cat"]


def _run(predictions, labels, params=None):
    return StatsConfusionMatrixNode().execute(
        {"predictions": predictions, "labels": labels}, params or {}
    )


def test_node_metadata():
    assert StatsConfusionMatrixNode.NODE_NAME == "Stats-ConfusionMatrix"
    outputs = StatsConfusionMatrixNode.define_outputs()
    assert [p.name for p in outputs] == [
        "matrix", "columns", "row_labels", "accuracy", "chart",
    ]
    assert next(p for p in outputs if p.name == "chart").media == "chart"


@pytest.mark.parametrize("normalize", ["none", "true", "pred", "all"])
def test_every_normalize_mode_matches_sklearn(normalize):
    result = _run(PRED, TRUE, {"normalize": normalize})
    expected = confusion_matrix(
        TRUE, PRED, normalize=None if normalize == "none" else normalize
    )
    assert result["matrix"].numpy() == pytest.approx(expected, rel=1e-6, abs=1e-9)


def test_classes_are_sorted_and_label_both_axes_like_sklearn():
    result = _run(PRED, TRUE)
    assert result["columns"] == ["bird", "cat", "dog"]
    assert result["row_labels"] == result["columns"]


def test_rows_are_the_true_class_and_columns_the_prediction():
    """The one orientation mistake that silently transposes every rate."""
    result = _run(["dog"], ["cat"])
    matrix = result["matrix"].numpy()
    row = result["row_labels"].index("cat")
    col = result["columns"].index("dog")
    assert matrix[row, col] == 1.0
    assert matrix[col, row] == 0.0


def test_normalize_true_puts_recall_on_the_diagonal():
    result = _run(PRED, TRUE, {"normalize": "true"})
    matrix = result["matrix"].numpy()
    # dog: 2 of 3 dogs predicted dog.
    dog = result["row_labels"].index("dog")
    assert matrix[dog, dog] == pytest.approx(2.0 / 3.0)
    assert matrix.sum(axis=1) == pytest.approx(np.ones(3))


def test_accuracy_is_the_diagonal_share_of_the_raw_counts():
    result = _run(PRED, TRUE)
    assert result["accuracy"] == pytest.approx(4.0 / 7.0)


def test_accuracy_ignores_the_normalize_setting():
    """Normalizing changes the picture, not how often the model was right."""
    plain = _run(PRED, TRUE)["accuracy"]
    for mode in ("true", "pred", "all"):
        assert _run(PRED, TRUE, {"normalize": mode})["accuracy"] == pytest.approx(plain)


def test_explicit_class_names_fix_the_order_and_keep_absent_classes():
    result = _run(["a", "a"], ["a", "b"], {"class_names": "b, a, z"})
    assert result["columns"] == ["b", "a", "z"]
    assert result["matrix"].shape == (3, 3)
    # `z` never appears, so its row and column are all zeros — not dropped.
    assert result["matrix"].numpy()[2].tolist() == [0.0, 0.0, 0.0]


def test_a_class_missing_from_class_names_is_ignored_and_reported():
    result = _run(["a", "zzz"], ["a", "a"], {"class_names": "a"})
    assert result["matrix"].numpy().tolist() == [[1.0]]
    assert "1 sample(s) ignored" in result["chart"]["note"]


def test_integer_class_indices_sort_numerically_not_lexicographically():
    """Sorted as strings, class 10 would land before class 2."""
    labels = [str(v) for v in (2, 10, 2, 10)]
    result = _run(labels, labels)
    assert result["columns"] == ["2", "10"]


def test_tensor_class_indices_line_up_with_string_labels():
    """An argmax gives floats; CSVReader gives strings. They must still match."""
    result = _run(torch.tensor([0.0, 1.0, 1.0]), ["0", "1", "1"])
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["columns"] == ["0", "1"]


def test_a_2d_score_matrix_is_reduced_by_argmax():
    scores = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
    result = _run(scores, ["0", "1", "1"])
    assert result["accuracy"] == pytest.approx(2.0 / 3.0)


def test_chart_is_a_heatmap_labelled_true_versus_predicted():
    chart = _run(PRED, TRUE)["chart"]
    assert chart["kind"] == "heatmap"
    assert (chart["x_label"], chart["y_label"]) == ("predicted", "true")
    assert chart["row_labels"] == ["bird", "cat", "dog"]
    assert chart["matrix"] == confusion_matrix(TRUE, PRED).tolist()


def test_normalized_charts_are_pinned_to_zero_one_and_raw_ones_are_not():
    normalized = _run(PRED, TRUE, {"normalize": "true"})["chart"]
    assert (normalized["vmin"], normalized["vmax"]) == (0.0, 1.0)
    raw = _run(PRED, TRUE)["chart"]
    assert raw["vmin"] == 0.0 and raw["vmax"] == 2.0


def test_chart_payload_is_json_safe_when_a_class_has_no_samples():
    """0/0 must land on 0, as sklearn's nan_to_num does — NaN is not JSON."""
    result = _run(["a", "a"], ["a", "a"], {"class_names": "a,b", "normalize": "true"})
    assert result["matrix"].numpy().tolist() == [[1.0, 0.0], [0.0, 0.0]]
    expected = confusion_matrix(["a", "a"], ["a", "a"], labels=["a", "b"],
                                normalize="true")
    assert result["matrix"].numpy() == pytest.approx(expected)
    json.dumps(result["chart"], allow_nan=False)


# ── boundaries ───────────────────────────────────────────────────────────────

def test_a_perfect_classifier_is_purely_diagonal():
    result = _run(TRUE, TRUE)
    matrix = result["matrix"].numpy()
    assert result["accuracy"] == pytest.approx(1.0)
    assert (matrix - np.diag(np.diag(matrix)) == 0).all()


def test_a_single_class_produces_a_one_by_one_matrix():
    result = _run(["a"], ["a"])
    assert result["matrix"].numpy().tolist() == [[1.0]]


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="must be per-sample and aligned"):
        _run(["a", "b"], ["a"])


def test_empty_inputs_without_class_names_are_rejected():
    with pytest.raises(ValueError, match="no classes to report"):
        _run([], [])


def test_empty_inputs_with_class_names_give_an_all_zero_matrix():
    result = _run([], [], {"class_names": "a,b"})
    assert result["matrix"].numpy().tolist() == [[0.0, 0.0], [0.0, 0.0]]
    assert result["accuracy"] == 0.0


def test_unknown_normalize_is_rejected():
    with pytest.raises(ValueError, match="unknown normalize"):
        _run(PRED, TRUE, {"normalize": "row"})


def test_missing_inputs_are_clear_errors():
    with pytest.raises(ValueError, match="requires a `predictions` input"):
        StatsConfusionMatrixNode().execute({"labels": ["a"]}, {})
    with pytest.raises(ValueError, match="requires a `labels` input"):
        StatsConfusionMatrixNode().execute({"predictions": ["a"]}, {})


def test_output_is_float32():
    assert _run(PRED, TRUE)["matrix"].dtype == torch.float32
