"""Tests for EmbeddingScatterNode."""

from __future__ import annotations

import numpy as np
import torch

from app.nodes.llm.embedding_scatter_node import EmbeddingScatterNode


def _run(embeddings, labels=None, **params):
    p = {"method": "PCA", "perplexity": 5.0, "seed": 42}
    p.update(params)
    inputs = {"embeddings": embeddings}
    if labels is not None:
        inputs["labels"] = labels
    return EmbeddingScatterNode().execute(inputs, p)


def test_node_metadata():
    assert EmbeddingScatterNode.NODE_NAME == "EmbeddingScatter"
    assert EmbeddingScatterNode.CATEGORY == "LLM"


def test_pca_basic_shape():
    emb = torch.randn(20, 8)
    res = _run(emb, ["w" + str(i) for i in range(20)])
    assert res["points_2d"].shape == (20, 2)
    assert len(res["labels"]) == 20
    # Normalised: max abs coord should be ≤ 1 (and ≈ 1 for non-degenerate input).
    assert torch.max(torch.abs(res["points_2d"])).item() <= 1.0 + 1e-6


def test_pca_preserves_three_clusters():
    """Three well-separated 8D Gaussian clusters → 2D projection still clusters."""
    rng = np.random.default_rng(7)
    n_per = 15
    centres = np.array([
        [10, 0, 0, 0, 0, 0, 0, 0],
        [0, 10, 0, 0, 0, 0, 0, 0],
        [0, 0, 10, 0, 0, 0, 0, 0],
    ], dtype=np.float32)
    pts = np.concatenate([
        c + rng.standard_normal(size=(n_per, 8)).astype(np.float32) * 0.3
        for c in centres
    ], axis=0)

    res = _run(torch.from_numpy(pts))
    coords = res["points_2d"].numpy()

    from sklearn.metrics import silhouette_score

    cluster_labels = np.repeat([0, 1, 2], n_per)
    score = silhouette_score(coords, cluster_labels)
    assert score > 0.5, f"silhouette {score} too low — clusters lost in projection"


def test_tsne_runs_on_small_input_via_perplexity_clamp():
    emb = torch.randn(8, 16)
    res = _run(emb, method="t-SNE", perplexity=20.0)  # > N-1, will be clamped
    assert res["points_2d"].shape == (8, 2)


def test_tsne_falls_back_to_pca_for_very_small_input():
    emb = torch.randn(3, 4)
    res = _run(emb, method="t-SNE")
    # No exception, output still 2D.
    assert res["points_2d"].shape == (3, 2)


def test_empty_input_returns_empty_2d_tensor():
    emb = torch.zeros((0, 4))
    res = _run(emb)
    assert res["points_2d"].shape == (0, 2)
    assert res["labels"] == []


def test_labels_truncated_to_match_points():
    emb = torch.randn(3, 4)
    res = _run(emb, labels=["a", "b", "c", "d", "e"])
    assert res["labels"] == ["a", "b", "c"]


def test_pca_seed_is_deterministic():
    emb = torch.randn(20, 8)
    a = _run(emb, seed=123)["points_2d"]
    b = _run(emb, seed=123)["points_2d"]
    assert torch.allclose(a, b, atol=1e-5)


def test_one_dimensional_input_is_padded():
    """When D < 2 PCA can only emit 1 component — pad to 2D so the contract holds."""
    emb = torch.tensor([[1.0], [2.0], [3.0]])
    res = _run(emb)
    assert res["points_2d"].shape == (3, 2)
