"""Tests for TimestepEmbeddingNode."""

from __future__ import annotations

import pytest
import torch

from app.nodes.diffusion.timestep_embedding_node import TimestepEmbeddingNode


def _run(timestep, **params):
    p = {"embed_dim": 32, "max_period": 10000, "seed": 42}
    p.update(params)
    return TimestepEmbeddingNode().execute({"timestep": timestep}, p)


def test_node_metadata():
    assert TimestepEmbeddingNode.NODE_NAME == "TimestepEmbedding"
    assert TimestepEmbeddingNode.CATEGORY == "Diffusion"
    out_names = [p.name for p in TimestepEmbeddingNode.define_outputs()]
    assert out_names == ["embedding"]


def test_scalar_timestep_returns_batch_one():
    """Single int timestep → [1, embed_dim] embedding."""
    res = _run(timestep=10, embed_dim=32)
    assert res["embedding"].shape == (1, 32)


def test_scalar_tensor_timestep():
    """0-d tensor timestep treated as a single timestep."""
    t = torch.tensor(10)
    res = _run(timestep=t, embed_dim=32)
    assert res["embedding"].shape == (1, 32)


def test_1d_tensor_timestep_batches():
    """[B] tensor of timesteps → [B, embed_dim]."""
    t = torch.tensor([0, 50, 100])
    res = _run(timestep=t, embed_dim=32)
    assert res["embedding"].shape == (3, 32)


def test_deterministic_given_seed():
    a = _run(timestep=10, seed=42)
    b = _run(timestep=10, seed=42)
    assert torch.allclose(a["embedding"], b["embedding"])


def test_different_timesteps_give_different_embeddings():
    a = _run(timestep=10, seed=42)
    b = _run(timestep=20, seed=42)
    assert not torch.allclose(a["embedding"], b["embedding"])


def test_different_seeds_give_different_embeddings():
    """Same t, different seed → different output (because of MLP weights)."""
    a = _run(timestep=10, seed=1)
    b = _run(timestep=10, seed=2)
    assert not torch.allclose(a["embedding"], b["embedding"])


def test_embed_dim_must_be_even():
    """sin/cos halves require even dim."""
    with pytest.raises(ValueError, match="even"):
        _run(timestep=10, embed_dim=33)


def test_consistent_across_batch():
    """Same timestep value at two batch positions should give identical embeddings."""
    t = torch.tensor([5, 5, 5])
    res = _run(timestep=t, embed_dim=16)
    assert torch.allclose(res["embedding"][0], res["embedding"][1])
    assert torch.allclose(res["embedding"][0], res["embedding"][2])


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        TimestepEmbeddingNode().execute({}, {"embed_dim": 32, "max_period": 10000, "seed": 42})


def test_negative_or_zero_embed_dim_raises():
    with pytest.raises(ValueError, match="embed_dim"):
        _run(timestep=10, embed_dim=0)
