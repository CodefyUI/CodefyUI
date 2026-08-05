"""EvaluateModelNode: device resolution (#204).

Companion to test_device_utils.py / test_device_index.py, which already
exhaustively cover resolve_node_device / resolve_device themselves. This
file is about the NODE'S WIRING to those functions -- #204 was a wiring bug
(the node called a different, context-blind function), not a bug in device
resolution itself.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from app.core import device_utils
from app.core.execution_context import ExecutionContext
from app.nodes.training.evaluate_model_node import EvaluateModelNode


def _dataset(n: int = 16, features: int = 4, classes: int = 3) -> TensorDataset:
    x = torch.randn(n, features)
    y = torch.randint(0, classes, (n,))
    return TensorDataset(x, y)


def _fresh_model(features: int = 4, classes: int = 3) -> nn.Module:
    return nn.Linear(features, classes)


# ── 1a (#204): device follows the run, not a hardcoded default ─────────


def test_device_param_offers_the_run_level_vocabulary_and_defaults_to_auto():
    """Same options, same default as TrainingLoop.device -- "auto" means
    "follow the run-level device", the behaviour users expect."""
    device_param = next(
        p for p in EvaluateModelNode.define_params() if p.name == "device")
    assert device_param.default == "auto"
    assert device_param.options == ["auto", "cpu", "cuda", "mps"]


def test_execute_resolves_device_through_resolve_node_device(monkeypatch):
    """#204: must resolve through resolve_node_device (context-aware,
    understands "auto") rather than the old
    resolve_device(str(params.get("device", "cpu"))) -- the latter never
    saw the context at all, so a graph submitted with {"device": "cuda"}
    trained on the GPU and then silently evaluated on the CPU.

    The spy delegates to the real implementation rather than replacing it,
    so this proves both the wiring (right function, right arguments) and
    that real resolution still runs -- not just that a mock was poked.
    """
    real = device_utils.resolve_node_device
    seen: dict[str, object] = {}

    def spy(value, context):
        seen["value"] = value
        seen["context"] = context
        return real(value, context)

    monkeypatch.setattr(device_utils, "resolve_node_device", spy)

    ctx = ExecutionContext(device="cpu")
    EvaluateModelNode().execute(
        {"model": _fresh_model(), "dataset": _dataset()}, {}, context=ctx)

    assert seen == {"value": None, "context": ctx}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_left_at_its_default_it_follows_a_real_cuda_run_end_to_end():
    """The acceptance criterion, on real hardware: a graph submitted with
    {"device": "cuda"} evaluates on cuda with no second setting touched.

    ``model.to(device)`` mutates in place and returns ``self``, so the
    caller's own ``model`` reference is checked directly afterwards --
    no probe needed for this one.
    """
    model = _fresh_model()
    EvaluateModelNode().execute(
        {"model": model, "dataset": _dataset()}, {},
        context=ExecutionContext(device="cuda"))
    assert next(model.parameters()).device.type == "cuda"


def test_an_explicit_device_still_overrides_the_run_device():
    """"auto" is the only value that defers to the context; an explicit
    "cpu" wins even when the run itself asked for something else."""
    model = _fresh_model()
    EvaluateModelNode().execute(
        {"model": model, "dataset": _dataset()},
        {"device": "cpu"},
        context=ExecutionContext(device="cuda"))
    assert next(model.parameters()).device.type == "cpu"


def test_with_no_context_it_defaults_to_cpu_like_the_cli_runner():
    """resolve_node_device handles context=None safely (falls back to cpu
    via context_device), so the CLI runner path -- which never builds an
    ExecutionContext -- is unaffected by this change."""
    model = _fresh_model()
    EvaluateModelNode().execute({"model": model, "dataset": _dataset()}, {})
    assert next(model.parameters()).device.type == "cpu"
