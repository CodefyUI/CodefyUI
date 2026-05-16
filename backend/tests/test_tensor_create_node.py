"""Tests for TensorCreateNode."""

from __future__ import annotations

import pytest
import torch

from app.nodes.tensor_ops.tensor_create_node import TensorCreateNode


def _run(**params):
    return TensorCreateNode().execute({}, params)


def test_node_metadata():
    assert TensorCreateNode.NODE_NAME == "TensorCreate"
    assert TensorCreateNode.CATEGORY == "Tensor Operations"
    assert TensorCreateNode.define_inputs() == []


def test_zeros_default():
    res = _run(shape="2,3", fill="zeros")
    assert res["tensor"].shape == (2, 3)
    assert torch.all(res["tensor"] == 0)


def test_ones():
    res = _run(shape="2,3", fill="ones")
    assert res["tensor"].shape == (2, 3)
    assert torch.all(res["tensor"] == 1)


def test_randn_returns_correct_shape():
    res = _run(shape="4,5", fill="randn")
    assert res["tensor"].shape == (4, 5)


def test_rand_in_unit_interval():
    res = _run(shape="10,10", fill="rand")
    assert (res["tensor"] >= 0).all()
    assert (res["tensor"] < 1).all()


def test_full_uses_value():
    res = _run(shape="2,2", fill="full", value=7.5)
    assert torch.all(res["tensor"] == 7.5)


def test_arange_returns_sequence():
    res = _run(shape="5", fill="arange")
    assert torch.equal(res["tensor"], torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]))


def test_requires_grad_flag():
    res = _run(shape="3", fill="zeros", requires_grad=True)
    assert res["tensor"].requires_grad


def test_no_requires_grad_default():
    res = _run(shape="3", fill="zeros")
    assert not res["tensor"].requires_grad


def test_unknown_fill_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        _run(shape="2", fill="bogus")


def test_4d_image_shape():
    res = _run(shape="1,3,224,224", fill="zeros")
    assert res["tensor"].shape == (1, 3, 224, 224)
