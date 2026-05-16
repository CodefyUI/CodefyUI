"""Tests for AccuracyNode."""

from __future__ import annotations

import pytest
import torch

from app.nodes.classical.accuracy_node import AccuracyNode


def _run(preds, labels):
    return AccuracyNode().execute({"predictions": preds, "labels": labels}, {})


def test_perfect_score():
    out = _run(["0", "1", "0", "1"], ["0", "1", "0", "1"])
    assert out["accuracy"] == 1.0
    assert out["correct"] == 4
    assert out["total"] == 4


def test_half_score():
    out = _run(["0", "0", "1", "1"], ["0", "1", "0", "1"])
    assert out["accuracy"] == 0.5
    assert out["correct"] == 2
    assert out["total"] == 4


def test_zero_score():
    out = _run(["0", "0"], ["1", "1"])
    assert out["accuracy"] == 0.0


def test_string_int_normalization():
    out = _run([0, 1, 0], ["0", "1", "0"])
    assert out["accuracy"] == 1.0


def test_tensor_predictions_accepted():
    out = _run(torch.tensor([0, 1, 0]), ["0", "1", "0"])
    assert out["accuracy"] == 1.0


def test_empty_inputs():
    out = _run([], [])
    assert out["accuracy"] == 0.0
    assert out["total"] == 0


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        _run(["0"], ["0", "1"])
