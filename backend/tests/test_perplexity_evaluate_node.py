"""Tests for PerplexityEvaluateNode."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from app.nodes.llm.causal_lm_model_node import CausalLMModelNode
from app.nodes.llm.perplexity_evaluate_node import PerplexityEvaluateNode

TINY = {
    "vocab_size": 256, "d_model": 32, "n_layers": 2, "n_heads": 2,
    "d_ff": 64, "max_seq_len": 32, "seed": 3,
}


def tiny_model():
    return CausalLMModelNode().execute({}, TINY)["model"]


def packed_dataset(num_samples=6, seq_len=8, with_ignore=False):
    generator = torch.Generator().manual_seed(0)
    inputs = torch.randint(0, TINY["vocab_size"], (num_samples, seq_len),
                           dtype=torch.int64, generator=generator)
    labels = torch.roll(inputs, shifts=-1, dims=1)
    if with_ignore:
        labels[:, -2:] = -100
    return TensorDataset(inputs, labels)


def manual_reference(model, dataset, batch_size, max_batches=0):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = model.eval()
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for index, (x, y) in enumerate(loader):
            if max_batches and index >= max_batches:
                break
            logits = model(x)
            loss_sum += float(F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                y.reshape(-1), ignore_index=-100, reduction="sum",
            ).item())
            count += int((y.reshape(-1) != -100).sum().item())
    return loss_sum / count, count


def test_node_metadata():
    assert PerplexityEvaluateNode.NODE_NAME == "PerplexityEvaluate"
    assert PerplexityEvaluateNode.CATEGORY == "LLM"
    assert PerplexityEvaluateNode.cacheable is False


def test_matches_manual_token_weighted_cross_entropy():
    model = tiny_model()
    dataset = packed_dataset(num_samples=6, seq_len=8)
    result = PerplexityEvaluateNode().execute(
        {"model": model, "dataset": dataset},
        {"batch_size": 4, "device": "cpu", "precision": "fp32"},
    )
    expected_loss, expected_tokens = manual_reference(model, dataset, batch_size=4)
    assert result["tokens"] == expected_tokens == 6 * 8
    assert result["val_loss"] == pytest.approx(expected_loss, rel=1e-6)
    assert result["perplexity"] == pytest.approx(math.exp(expected_loss), rel=1e-6)


def test_ignore_index_positions_are_excluded():
    model = tiny_model()
    dataset = packed_dataset(num_samples=4, seq_len=8, with_ignore=True)
    result = PerplexityEvaluateNode().execute(
        {"model": model, "dataset": dataset},
        {"batch_size": 4, "device": "cpu", "precision": "fp32"},
    )
    expected_loss, expected_tokens = manual_reference(model, dataset, batch_size=4)
    assert result["tokens"] == expected_tokens == 4 * 6  # two ignored per row
    assert result["val_loss"] == pytest.approx(expected_loss, rel=1e-6)


def test_max_batches_caps_the_evaluation():
    model = tiny_model()
    dataset = packed_dataset(num_samples=6, seq_len=8)
    result = PerplexityEvaluateNode().execute(
        {"model": model, "dataset": dataset},
        {"batch_size": 4, "max_batches": 1, "device": "cpu", "precision": "fp32"},
    )
    expected_loss, expected_tokens = manual_reference(
        model, dataset, batch_size=4, max_batches=1,
    )
    assert result["tokens"] == expected_tokens == 4 * 8
    assert result["val_loss"] == pytest.approx(expected_loss, rel=1e-6)


def test_all_ignored_labels_raise_an_actionable_error():
    model = tiny_model()
    inputs = torch.zeros((2, 8), dtype=torch.int64)
    labels = torch.full((2, 8), -100, dtype=torch.int64)
    with pytest.raises(RuntimeError, match="zero tokens"):
        PerplexityEvaluateNode().execute(
            {"model": model, "dataset": TensorDataset(inputs, labels)},
            {"device": "cpu", "precision": "fp32"},
        )


def test_missing_inputs_are_refused():
    with pytest.raises(ValueError, match="model"):
        PerplexityEvaluateNode().execute({"dataset": packed_dataset()}, {})
    with pytest.raises(ValueError, match="dataset"):
        PerplexityEvaluateNode().execute({"model": tiny_model()}, {})
