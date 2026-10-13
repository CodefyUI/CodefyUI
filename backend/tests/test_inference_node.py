"""Tests for InferenceNode."""

from __future__ import annotations

import torch
import torch.nn as nn

from app.nodes.io.inference_node import InferenceNode


def test_node_metadata():
    assert InferenceNode.NODE_NAME == "Inference"
    assert InferenceNode.CATEGORY == "IO"


def test_runs_forward_pass():
    model = nn.Linear(8, 4)
    x = torch.randn(2, 8)
    res = InferenceNode().execute({"model": model, "input": x}, {"device": "cpu"})
    assert res["output"].shape == (2, 4)


def test_passes_through_model():
    model = nn.Linear(4, 2)
    x = torch.zeros(1, 4)
    res = InferenceNode().execute({"model": model, "input": x}, {"device": "cpu"})
    assert res["model"] is model


def test_no_gradients_required():
    model = nn.Linear(4, 2)
    x = torch.randn(1, 4)
    res = InferenceNode().execute({"model": model, "input": x}, {"device": "cpu"})
    # In inference mode, output should not have grad_fn from autograd
    assert res["output"].requires_grad is False


def test_falls_back_to_cpu_when_cuda_unavailable():
    """Even if device='cuda' is requested, falls back gracefully."""
    model = nn.Linear(4, 2)
    x = torch.zeros(1, 4)
    # This should not error even if CUDA isn't available
    res = InferenceNode().execute({"model": model, "input": x}, {"device": "cuda"})
    assert res["output"].shape == (1, 2)


def test_sets_model_to_eval_mode():
    """Inference should put model in eval mode."""
    model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5))
    x = torch.randn(1, 4)
    InferenceNode().execute({"model": model, "input": x}, {"device": "cpu"})
    # After inference, model should be in eval mode
    assert not model.training


# -- 2026-10-13 audit: input dtype -------------------------------------------


def test_float64_input_is_cast_to_the_model_dtype_on_cpu():
    """CPU used to fail with "mat1 and mat2 must have the same dtype"."""
    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    x = torch.randn(5, 4, dtype=torch.float64)
    out = InferenceNode().execute({"model": model, "input": x}, {"device": "cpu"})["output"]
    assert out.dtype == torch.float32
    with torch.no_grad():
        assert torch.allclose(out, model(x.float()))


def test_integer_input_still_reaches_an_embedding_model():
    model = nn.Sequential(nn.Embedding(10, 4), nn.Flatten(), nn.Linear(8, 2))
    idx = torch.tensor([[1, 2], [3, 4]])
    out = InferenceNode().execute({"model": model, "input": idx}, {"device": "cpu"})["output"]
    assert out.shape == (2, 2)


def test_integer_input_into_a_float_layer_gets_a_node_named_error():
    import pytest

    with pytest.raises(ValueError, match=r"^Inference: the 'input' tensor is torch.int64"):
        InferenceNode().execute(
            {"model": nn.Linear(4, 2), "input": torch.ones(2, 4, dtype=torch.int64)},
            {"device": "cpu"})
