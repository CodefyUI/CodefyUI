"""Fixes from the 2026-10-13 functional-correctness audit of the edu,
foundations, deep and rl plugin packs.

Each test pins a number or a message the audit found wrong:

* the edu classifiers read an ``[N, 1]`` label tensor as the strings
  ``'[0.0]'`` (Accuracy 0 on a perfect fit) and sorted integer classes as
  text (0, 1, 10, 2, ...), shuffling the probability columns;
* Edu-TokenEmbedding's ordinal mode wrapped row ids, so distinct tokens
  shared one vector;
* Edu-PolicyGradient is covered in test_edu_policy_gradient_node.py;
* smaller ones: TrainAndEvaluate on MPS, FilterRows rounding float64/int64
  to float32, Edu-KNN voting on "tensor(0)", raw torch errors for 1-D
  features and width mismatches, ColumnStats in float32, Edu-FFN drawing
  from the global RNG, BOOL params read with ``bool()``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from app.nodes.classical.accuracy_node import AccuracyNode
from cdui_plugins.deep.nodes.edu_cross_attention_node import EduCrossAttentionNode
from cdui_plugins.deep.nodes.edu_patchify_node import EduPatchifyNode
from cdui_plugins.deep.nodes.edu_resblock_node import EduResBlockNode
from cdui_plugins.deep.nodes.edu_self_attention_node import EduSelfAttentionNode
from cdui_plugins.edu.nodes.advanced_classifier_node import AdvancedClassifierNode
from cdui_plugins.edu.nodes.classifier_node import ClassifierNode
from cdui_plugins.edu.nodes.edu_filter_rows_node import EduFilterRowsNode
from cdui_plugins.edu.nodes.edu_sentence_embedding_node import (
    _EMBED_CACHE,
    EduSentenceEmbeddingNode,
)
from cdui_plugins.edu.nodes.edu_sliding_window_2d_node import EduSlidingWindow2DNode
from cdui_plugins.edu.nodes.ffn_layer_node import FFNLayerNode
from cdui_plugins.edu.nodes.train_and_evaluate_node import TrainAndEvaluateNode
from cdui_plugins.foundations.nodes.edu_column_stats_node import EduColumnStatsNode
from cdui_plugins.foundations.nodes.edu_ffn_node import EduFFNNode
from cdui_plugins.foundations.nodes.edu_knn_node import EduKNNNode
from cdui_plugins.foundations.nodes.edu_linear_regression_node import EduLinearRegressionNode
from cdui_plugins.foundations.nodes.edu_logistic_regression_node import (
    EduLogisticRegressionNode,
)
from cdui_plugins.foundations.nodes.edu_token_embedding_node import EduTokenEmbeddingNode


def _blobs(n_per_class: int, n_classes: int, seed: int = 0) -> tuple[torch.Tensor, list[int]]:
    """Well-separated 2-D blobs: class c sits around (3c, 3c)."""
    gen = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    for c in range(n_classes):
        xs.append(torch.randn(n_per_class, 2, generator=gen) * 0.1 + 3.0 * c)
        ys.extend([c] * n_per_class)
    return torch.cat(xs), ys


def _accuracy(predictions, truth) -> float:
    return AccuracyNode().execute({"predictions": predictions, "labels": truth}, {})["accuracy"]


def _stack(in_features: int = 2) -> nn.Module:
    first = FFNLayerNode().execute({}, {"in_features": in_features, "out_features": 8})["model"]
    return first


# -- HIGH: edu classifiers and [N, 1] / integer labels -----------------------


@pytest.mark.parametrize("kind", ["knn", "logistic", "linear"])
def test_classifier_reads_a_label_column_tensor(kind):
    x, y = _blobs(10, 2)
    y_col = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    out = ClassifierNode().execute(
        {"x_train": x, "y_train": y_col, "x_query": x}, {"kind": kind, "n_neighbors": 1})
    assert out["classes"] == ["0", "1"]
    assert _accuracy(out["predictions"], y_col) == 1.0


@pytest.mark.parametrize("kind", ["SVM", "Decision Tree", "Random Forest"])
def test_advanced_classifier_reads_a_label_column_tensor(kind):
    x, y = _blobs(10, 2)
    y_col = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    out = AdvancedClassifierNode().execute(
        {"x_train": x, "y_train": y_col, "x_query": x}, {"kind": kind})
    assert out["classes"] == ["0", "1"]
    assert _accuracy(out["predictions"], y_col) == 1.0
    assert out["model"].predict(x.numpy()) == out["predictions"]


def test_train_and_evaluate_reads_a_label_column_tensor():
    x, y = _blobs(10, 2)
    y_col = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    out = TrainAndEvaluateNode().execute(
        {"model": _stack(), "x_train": x, "y_train": y_col, "x_query": x},
        {"epochs": 60, "lr": 0.05})
    assert out["classes"] == ["0", "1"]
    assert _accuracy(out["predictions"], y_col) == 1.0


def test_integer_classes_sort_by_value_so_probability_columns_line_up():
    x, y = _blobs(6, 11)
    out = ClassifierNode().execute(
        {"x_train": x, "y_train": y, "x_query": x}, {"kind": "knn", "n_neighbors": 3})
    assert out["classes"] == [str(c) for c in range(11)]
    # Column j of probabilities is classes[j]: the argmax names the prediction.
    argmax = out["probabilities"].argmax(dim=1).tolist()
    assert [out["classes"][j] for j in argmax] == out["predictions"]
    assert _accuracy(out["predictions"], y) == 1.0

    adv = AdvancedClassifierNode().execute(
        {"x_train": x, "y_train": y, "x_query": x}, {"kind": "Decision Tree"})
    assert adv["classes"] == [str(c) for c in range(11)]
    trained = TrainAndEvaluateNode().execute(
        {"model": _stack(), "x_train": x, "y_train": y, "x_query": x}, {"epochs": 2})
    assert trained["classes"] == [str(c) for c in range(11)]


def test_text_labels_keep_their_text_order():
    x, _ = _blobs(4, 3)
    y = ["cat"] * 4 + ["ant"] * 4 + ["bee"] * 4
    out = ClassifierNode().execute({"x_train": x, "y_train": y, "x_query": x}, {"kind": "knn"})
    assert out["classes"] == ["ant", "bee", "cat"]
    assert out["predictions"] == y


def test_classifier_errors_name_the_node_and_param():
    x, y = _blobs(4, 2)
    with pytest.raises(ValueError, match="Classifier: `x_query` has 1 features"):
        ClassifierNode().execute({"x_train": x, "y_train": y, "x_query": x[:, :1]}, {})
    with pytest.raises(ValueError, match="Classifier: `n_neighbors`"):
        ClassifierNode().execute({"x_train": x, "y_train": y, "x_query": x}, {"n_neighbors": None})
    with pytest.raises(ValueError, match="AdvancedClassifier: unknown `kernel`"):
        AdvancedClassifierNode().execute(
            {"x_train": x, "y_train": y, "x_query": x}, {"kind": "SVM", "kernel": "sigmoidx"})


def test_train_and_evaluate_names_width_and_length_mismatches():
    x, y = _blobs(5, 2)
    x3 = torch.cat([x, x[:, :1]], dim=1)
    with pytest.raises(ValueError, match="in_features=2; set that FFNLayer's `in_features` to 3"):
        TrainAndEvaluateNode().execute(
            {"model": _stack(2), "x_train": x3, "y_train": y, "x_query": x3}, {"epochs": 1})
    with pytest.raises(ValueError, match="TrainAndEvaluate: features and labels length"):
        TrainAndEvaluateNode().execute(
            {"model": _stack(2), "x_train": x, "y_train": y[:-2], "x_query": x}, {"epochs": 1})


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_train_and_evaluate_model_predicts_a_cpu_grid_after_an_mps_run():
    x, y = _blobs(10, 2)
    out = TrainAndEvaluateNode().execute(
        {"model": _stack(), "x_train": x.to("mps"), "y_train": y, "x_query": x.to("mps")},
        {"epochs": 30})
    grid = np.zeros((3, 2), dtype="float32")
    assert len(out["model"].predict(grid)) == 3


# -- HIGH: Edu-TokenEmbedding ordinal overflow -------------------------------


def test_ordinal_mode_refuses_more_unique_tokens_than_rows():
    tokens = [f"t{i}" for i in range(6)]
    with pytest.raises(ValueError, match="6 unique tokens and vocab_size is 4"):
        EduTokenEmbeddingNode().execute({"tokens": tokens}, {"mode": "ordinal", "vocab_size": 4})


def test_ordinal_mode_at_exactly_vocab_size_gives_distinct_rows():
    tokens = [f"t{i}" for i in range(4)] + ["t0"]
    out = EduTokenEmbeddingNode().execute(
        {"tokens": tokens}, {"mode": "ordinal", "vocab_size": 4})
    emb = out["embeddings"]
    assert out["vocab"] == ["t0", "t1", "t2", "t3"]
    assert torch.equal(emb[0], emb[4])
    assert len({tuple(r.tolist()) for r in emb[:4]}) == 4


def test_token_embedding_refuses_length_mismatch_and_fractional_ids():
    with pytest.raises(ValueError, match="`tokens` has 3 entries but `token_ids` has 2"):
        EduTokenEmbeddingNode().execute({"tokens": ["a", "b", "c"], "token_ids": [1, 2]}, {})
    with pytest.raises(ValueError, match="Edu-TokenEmbedding: `token_ids` holds 1.7"):
        EduTokenEmbeddingNode().execute({"token_ids": [1.7]}, {})
    out = EduTokenEmbeddingNode().execute({"token_ids": [1.0, torch.tensor(2)]}, {})
    assert out["vocab"] == ["<id:1>", "<id:2>"]


# -- MEDIUM: FilterRows keeps the table's precision ---------------------------


def test_filter_rows_compares_float64_without_rounding():
    table = torch.tensor([[90.0], [80.0], [80.000001], [0.1]], dtype=torch.float64)
    out = EduFilterRowsNode().execute({"table": table}, {"op": ">", "threshold": 80})
    assert out["count"] == 2
    assert out["tensor"].dtype == torch.float64
    assert out["tensor"][:, 0].tolist() == [90.0, 80.000001]


def test_filter_rows_compares_large_integers_exactly():
    table = torch.tensor([[16777216], [16777217]], dtype=torch.int64)
    out = EduFilterRowsNode().execute({"table": table}, {"op": ">", "threshold": 16777216})
    assert out["count"] == 1
    assert out["tensor"].tolist() == [[16777217]]


def test_filter_rows_float32_results_are_unchanged():
    table = torch.tensor([[0.1], [0.2], [80.0]], dtype=torch.float32)
    out = EduFilterRowsNode().execute({"table": table}, {"op": "==", "threshold": 0.1})
    assert out["count"] == 1
    assert out["tensor"].dtype == torch.float32


def test_filter_rows_threshold_error_names_the_param():
    with pytest.raises(ValueError, match="FilterRows: `threshold` must be a number"):
        EduFilterRowsNode().execute({"table": torch.zeros(2, 1)}, {"threshold": "abc"})


# -- MEDIUM: Edu-KNN tensor labels, shapes ------------------------------------


def test_knn_votes_on_tensor_labels_as_their_values():
    x = torch.tensor([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0], [6.0, 6.0]])
    for y in (torch.tensor([0, 0, 1, 1]), torch.tensor([[0.0], [0.0], [1.0], [1.0]])):
        out = EduKNNNode().execute(
            {"x_train": x, "y_train": y, "x_query": x}, {"k": 1})
        assert out["predictions"] == ["0", "0", "1", "1"]
        assert out["train_labels"] == [0, 0, 1, 1]


@pytest.mark.parametrize("node, name", [
    (EduKNNNode, "Edu-KNN"),
    (EduLinearRegressionNode, "Edu-LinearRegression"),
    (EduLogisticRegressionNode, "Edu-LogisticRegression"),
])
def test_one_dimensional_features_and_width_mismatch_name_the_node(node, name):
    y = [0, 1, 0]
    with pytest.raises(ValueError, match=f"{name}: `x_train` and `x_query` must be \\[rows, features\\]"):
        node().execute({"x_train": torch.tensor([1.0, 2.0, 3.0]), "y_train": y,
                        "x_query": torch.tensor([1.0])}, {})
    with pytest.raises(ValueError, match=f"{name}: `x_query` has 2 features but `x_train` has 3"):
        node().execute({"x_train": torch.rand(3, 3), "y_train": y,
                        "x_query": torch.rand(2, 2)}, {})


# -- LOW: Edu-LinearRegression -------------------------------------------------


def test_linear_regression_single_training_row_runs():
    out = EduLinearRegressionNode().execute(
        {"x_train": torch.tensor([[1.0, 2.0]]), "y_train": torch.tensor([[3.0]]),
         "x_query": torch.tensor([[1.0, 2.0]])}, {})
    assert out["predictions"].shape == (1,)
    assert float(out["predictions"][0]) == pytest.approx(3.0, abs=1e-4)


def test_linear_regression_collinear_columns_take_the_minimum_norm_solution():
    """Columns [x0, x0, x1]: float32 solve() did not raise and gave w = [0, 1.59, -2.05]."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 3)).astype(np.float32)
    y = (x @ [1.5, -2, 0.5] + 3 + rng.normal(scale=0.1, size=50)).astype(np.float32)
    x2 = np.c_[x[:, 0], x[:, 0], x[:, 1]].astype(np.float32)
    out = EduLinearRegressionNode().execute(
        {"x_train": torch.tensor(x2), "y_train": torch.tensor(y), "x_query": torch.tensor(x2[:3])},
        {"method": "closed_form"})
    w = out["weights"].tolist()
    # sklearn LinearRegression on the same data: [0.7952, 0.7952, -2.0549].
    assert w == pytest.approx([0.7952, 0.7952, -2.0549], abs=1e-3)


def test_linear_regression_divergence_names_the_learning_rate():
    x = torch.tensor([[100.0], [200.0], [300.0]])
    with pytest.raises(ValueError, match="diverged .* `lr`=0.01"):
        EduLinearRegressionNode().execute(
            {"x_train": x, "y_train": torch.tensor([1.0, 2.0, 3.0]), "x_query": x},
            {"method": "gradient_descent", "lr": 0.01, "epochs": 100})


# -- LOW: Edu-LogisticRegression class order ------------------------------------


def test_logistic_regression_sorts_numeric_classes_by_value():
    x, y = _blobs(5, 3)
    labels = [{0: 2, 1: 9, 2: 10}[v] for v in y]
    out = EduLogisticRegressionNode().execute(
        {"x_train": x, "y_train": labels, "x_query": x}, {"epochs": 300})
    assert out["classes"] == ["2", "9", "10"]
    assert out["predictions"] == [str(v) for v in labels]


# -- LOW: Edu-ColumnStats ----------------------------------------------------------


def test_column_stats_keeps_float64_precision():
    table = torch.tensor([[1e7 + 1], [1e7 + 2], [1e7 + 3]], dtype=torch.float64)
    out = EduColumnStatsNode().execute({"table": table}, {})
    assert float(out["stds"][0]) == pytest.approx(math.sqrt(2 / 3), rel=1e-9)
    assert float(out["means"][0]) == 1e7 + 2


def test_column_stats_single_row_sample_std_is_nan():
    out = EduColumnStatsNode().execute(
        {"table": torch.tensor([[1.0, 2.0]])}, {"unbiased": True})
    assert all(math.isnan(v) for v in out["stds"].tolist())


def test_column_stats_reads_the_string_false_as_false():
    table = torch.tensor([[1.0], [3.0]])
    out = EduColumnStatsNode().execute({"table": table}, {"unbiased": "false"})
    assert float(out["stds"][0]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="Edu-ColumnStats: `unbiased` must be true or false"):
        EduColumnStatsNode().execute({"table": table}, {"unbiased": "maybe"})


# -- LOW: Edu-FFN and the global RNG -------------------------------------------------


def test_ffn_build_leaves_the_global_rng_alone_and_keeps_its_weights():
    torch.manual_seed(0)
    expected = torch.rand(3)
    torch.manual_seed(0)
    module = EduFFNNode().build_module({"embed_dim": 4, "hidden_dim": 6, "seed": 7})
    assert torch.equal(torch.rand(3), expected)

    gen = torch.Generator().manual_seed(7)
    w1 = torch.randn((6, 4), generator=gen) * (1.0 / 4 ** 0.5)
    w2 = torch.randn((4, 6), generator=gen) * (1.0 / 6 ** 0.5)
    assert torch.equal(module.fc1.weight, w1)
    assert torch.equal(module.fc2.weight, w2)
    assert torch.equal(module.fc1.bias, torch.zeros(6))


# -- LOW: deep pack shapes and BOOL params ---------------------------------------------


def test_cross_attention_names_a_batch_mismatch():
    with pytest.raises(ValueError, match="query has batch size 2 but context has 4"):
        EduCrossAttentionNode().execute(
            {"query": torch.rand(3, 2, 8), "context": torch.rand(5, 4, 8)},
            {"embed_dim": 8, "num_heads": 2})


@pytest.mark.parametrize("shape", [(3, 32), (2, 1, 32)])
def test_resblock_names_a_bad_time_emb(shape):
    with pytest.raises(ValueError, match="EduResBlock: `time_emb` must be"):
        EduResBlockNode().execute(
            {"tensor": torch.rand(2, 8, 4, 4), "time_emb": torch.rand(*shape)},
            {"in_channels": 8, "out_channels": 8, "groups": 4, "time_emb_dim": 32})


def test_bool_params_read_the_string_false_as_false():
    x = torch.rand(4, 8)
    as_bool = EduSelfAttentionNode().execute({"tensor": x}, {"embed_dim": 8, "causal": False})
    as_text = EduSelfAttentionNode().execute({"tensor": x}, {"embed_dim": 8, "causal": "false"})
    assert torch.equal(as_bool["weights"], as_text["weights"])
    assert float(as_text["weights"].detach()[0, 1]) > 0  # not causally masked

    img = torch.rand(1, 3, 8, 8)
    flat = EduPatchifyNode().execute({"image": img}, {"patch_size": 4, "flatten": "false"})
    assert flat["tokens"].dim() == 5


# -- LOW: SentenceEmbedding cache, SlidingWindow2D messages -------------------------


def test_sentence_embedding_returns_a_copy_of_the_cached_vector():
    key = ("minishlab/potion-multilingual-128M", "audit-copy-check")
    _EMBED_CACHE[key] = torch.tensor([1.0, 2.0])
    try:
        node = EduSentenceEmbeddingNode()
        first = node.execute({"text": "audit-copy-check"}, {"model": "multilingual"})["embedding"]
        first.mul_(0)
        again = node.execute({"text": "audit-copy-check"}, {"model": "multilingual"})["embedding"]
        assert again.tolist() == [1.0, 2.0]
    finally:
        _EMBED_CACHE.pop(key, None)


def test_sliding_window_names_the_node_when_the_kernel_does_not_fit():
    with pytest.raises(ValueError, match="SlidingWindow2D: the 3x3 kernel does not fit a 2x2"):
        EduSlidingWindow2DNode().execute({"image": torch.rand(1, 2, 2)}, {"preset": "Blur3x3"})
