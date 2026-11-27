"""FreezeParameters: requires_grad by name prefix, in place, with a report."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from app.nodes.training.freeze_parameters_node import FreezeParametersNode, select_parameters


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 3))


def _flags(model: nn.Module) -> dict[str, bool]:
    return {name: p.requires_grad for name, p in model.named_parameters()}


def _run(model, **params):
    return FreezeParametersNode().execute({"model": model}, {"mode": "freeze", "match": "", "invert": False, **params})


def test_metadata():
    assert FreezeParametersNode.NODE_NAME == "FreezeParameters"
    assert FreezeParametersNode.CATEGORY == "Training"
    assert FreezeParametersNode.cacheable is False
    assert [p.name for p in FreezeParametersNode.define_params()] == ["mode", "match", "invert"]


def test_empty_match_freezes_everything_and_returns_the_same_object():
    model = _model()
    result = _run(model)
    assert result["model"] is model
    assert not any(_flags(model).values())
    assert result["report"].startswith("frozen: 6 tensors, 139 parameters; trainable: 0 tensors, 0 parameters")


def test_a_prefix_freezes_only_the_matching_parameters():
    model = _model()
    _run(model, match="0., 2.")
    assert _flags(model) == {"0.weight": False, "0.bias": False, "2.weight": False,
                             "2.bias": False, "4.weight": True, "4.bias": True}


def test_invert_keeps_only_the_head_trainable():
    model = _model()
    result = _run(model, match="4.", invert=True)
    assert [n for n, flag in _flags(model).items() if flag] == ["4.weight", "4.bias"]
    assert "trainable: 2 tensors, 27 parameters" in result["report"]


def test_unfreeze_turns_requires_grad_back_on():
    model = _model()
    _run(model)
    result = _run(model, mode="unfreeze", match="4.")
    assert [n for n, flag in _flags(model).items() if flag] == ["4.weight", "4.bias"]
    assert result["report"].startswith("frozen: 4 tensors")


def test_a_prefix_that_matches_nothing_is_named_in_the_report():
    model = _model()
    result = _run(model, match="backbone.")
    assert all(_flags(model).values())
    assert result["report"].endswith("no parameter starts with: backbone.")


def test_frozen_weights_stay_fixed_through_a_training_step():
    model = _model()
    _run(model, match="4.", invert=True)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    loss = model(torch.randn(16, 4)).pow(2).mean()
    loss.backward()
    optimizer.step()
    after = dict(model.named_parameters())
    assert all(torch.equal(before[n], after[n]) for n in ("0.weight", "0.bias", "2.weight", "2.bias"))
    assert not torch.equal(before["4.weight"], after["4.weight"])


@pytest.mark.parametrize("match, invert, expected", [
    ("", False, ["a.w", "a.b", "b.w"]),
    ("", True, []),
    ("a.", False, ["a.w", "a.b"]),
    ("a.", True, ["b.w"]),
    (" a. , ,b.w ", False, ["a.w", "a.b", "b.w"]),
    (None, False, ["a.w", "a.b", "b.w"]),
])
def test_select_parameters(match, invert, expected):
    assert select_parameters(["a.w", "a.b", "b.w"], match, invert) == expected


def test_a_value_that_is_not_a_model_is_refused():
    with pytest.raises(ValueError, match="needs a model"):
        _run(torch.zeros(3))
