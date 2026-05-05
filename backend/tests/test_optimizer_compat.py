"""Regression tests for OptimizerNode + TrainingLoop optimizer rebind.

Covers two PyTorch 2.11+ compatibility breaks reported by users:

1. OptimizerNode: Rprop does not accept ``weight_decay`` at all, so passing
   the default-shaped kwargs (``lr=..., weight_decay=...``) blew up at
   create time. We silently drop the kwarg when it's the default 0.0 and
   raise a clear error when it isn't.

2. TrainingLoop: Adam's defaults dict in PyTorch 2.11 contains
   ``decoupled_weight_decay`` (used internally to switch between Adam and
   AdamW behaviour). AdamW (an Adam subclass) rejects that key as a
   public kwarg. The naive rebind ``optimizer_cls(model.parameters(),
   **optimizer.defaults)`` therefore raised
   ``AdamW.__init__() got an unexpected keyword argument 'decoupled_weight_decay'``
   for any graph that selected AdamW (notably the Train Mini-GPT on MNIST
   example, via the Training Pipeline preset).
"""

from __future__ import annotations

import inspect

import pytest
import torch
import torch.optim as optim

from app.nodes.training.optimizer_node import OptimizerNode

# Mirrors OptimizerNode's SELECT options so a future addition to one side
# without the other shows up as a test failure.
ALL_OPTIMIZER_TYPES = [
    "Adam", "SGD", "AdamW", "RMSprop", "Adagrad",
    "RAdam", "NAdam", "Rprop", "ASGD",
]


class _TinyModel(torch.nn.Module):
    """Smallest possible model with trainable params."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(2, 2)


@pytest.mark.parametrize("opt_type", ALL_OPTIMIZER_TYPES)
def test_optimizer_node_creates_with_default_weight_decay(opt_type: str) -> None:
    """Every optimizer the SELECT exposes must construct cleanly with the
    default ``weight_decay=0.0`` — including Rprop, which doesn't even have
    a ``weight_decay`` parameter (we silently drop the kwarg in that case).
    """
    node = OptimizerNode()
    out = node.execute(
        {"model": _TinyModel()},
        {"type": opt_type, "lr": 0.001, "weight_decay": 0.0},
    )
    assert out["optimizer"].__class__.__name__ == opt_type


def test_optimizer_node_rprop_rejects_nonzero_weight_decay() -> None:
    """A clear ValueError is far better than a confusing
    ``Rprop.__init__() got an unexpected keyword argument 'weight_decay'``.
    """
    node = OptimizerNode()
    with pytest.raises(ValueError, match="weight_decay"):
        node.execute(
            {"model": _TinyModel()},
            {"type": "Rprop", "lr": 0.001, "weight_decay": 0.1},
        )


@pytest.mark.parametrize("opt_type", ALL_OPTIMIZER_TYPES)
def test_optimizer_defaults_round_trip(opt_type: str) -> None:
    """Simulates the rebind in TrainingLoopNode: take ``optimizer.defaults``,
    filter to keys the constructor accepts, re-instantiate against a
    fresh model. Must succeed for every optimizer type — this is the
    invariant that broke on AdamW with the unfiltered ``**defaults``.
    """
    node = OptimizerNode()
    opt1 = node.execute(
        {"model": _TinyModel()},
        {"type": opt_type, "lr": 0.001, "weight_decay": 0.0},
    )["optimizer"]

    cls = type(opt1)
    accepted = set(inspect.signature(cls.__init__).parameters)
    filtered = {k: v for k, v in opt1.defaults.items() if k in accepted}

    opt2 = cls(_TinyModel().parameters(), **filtered)
    assert opt2.__class__ is cls


def test_adamw_round_trip_via_naive_defaults_would_fail() -> None:
    """Documents the upstream behaviour we're working around: a naive
    ``cls(params, **defaults)`` (no signature filtering) raises on AdamW
    in PyTorch 2.11+. If this assertion ever flips, the
    ``decoupled_weight_decay`` workaround in TrainingLoopNode and the
    ``inspect.signature`` filter in this test can be revisited.
    """
    opt1 = optim.AdamW(_TinyModel().parameters(), lr=0.001, weight_decay=0.0)
    if "decoupled_weight_decay" not in opt1.defaults:
        pytest.skip("PyTorch < 2.11; the workaround isn't needed here.")
    with pytest.raises(TypeError, match="decoupled_weight_decay"):
        optim.AdamW(_TinyModel().parameters(), **opt1.defaults)
