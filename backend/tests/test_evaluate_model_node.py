"""EvaluateModelNode: device resolution (#204) and precision (#193 item 1).

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
from app.core.amp import PRECISIONS
from app.core.execution_context import ExecutionContext
from app.nodes.training.evaluate_model_node import EvaluateModelNode


def _dataset(n: int = 16, features: int = 4, classes: int = 3) -> TensorDataset:
    x = torch.randn(n, features)
    y = torch.randint(0, classes, (n,))
    return TensorDataset(x, y)


def _fresh_model(features: int = 4, classes: int = 3) -> nn.Module:
    return nn.Linear(features, classes)


class _DtypeProbe(nn.Module):
    """A tiny model that records the dtype its forward pass actually ran
    in, so a test can prove autocast was (or wasn't) applied without
    inspecting torch internals."""

    def __init__(self, in_features: int = 4, out_features: int = 3):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.seen_dtype: torch.dtype | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        self.seen_dtype = out.dtype
        return out


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


def test_left_at_its_default_it_reaches_the_model_without_real_hardware(monkeypatch):
    """The test above is skipif'd away on CPU-only CI, and its two
    unconditional neighbours (explicit-override, no-context) pass against
    the pre-#204 code too -- they are regression pins, not proof of the
    fix. On CI the only thing pinning "left at its default follows the
    run device" is the spy test, which only proves resolve_node_device
    was CALLED with the right arguments, not that whatever it RETURNS
    actually reaches next(model.parameters()).device. This closes that
    gap without a GPU.

    A real run's context.device is already resolve_device-d by the time a
    node sees it (resolve_node_device's own docstring). Monkeypatching
    resolve_device to map "cuda" -> torch's "meta" device -- real, always
    available, needs no driver -- simulates exactly that, so the eventual
    model.to(device) call below is genuine rather than mocked. Every OTHER
    input passes through unchanged: the mock must not swallow the
    difference between the two code paths under test, only stand in for
    real cuda hardware. (A `lambda requested: "meta"` unconditionally,
    tried first, made this test pass against the pre-#204 code too --
    "cpu", the OLD hardcoded default, hit the same mock and also came
    back "meta". That is exactly the "test that would pass against
    unmodified code" trap; caught by actually running it against the
    pre-fix source before trusting it.)

    The node param is left empty ({}), i.e. at its "auto" default, so
    this exercises the SAME auto -> context.device path as the CUDA test
    above; resolve_node_device's "auto" branch is a direct, unvalidated
    passthrough of context.device (see resolve_node_device's docstring),
    so it never calls resolve_device itself -- the monkeypatch here is
    standing in for the run-submission step that populates context.device
    in production, not for anything this node calls directly.

    Meta tensors carry no data, so real evaluation (which reads back a
    scalar via .item()) would raise; this test is about the device MOVE
    (model = model.to(device)), which happens before the batch loop, so
    the context is pre-cancelled -- the same idiom test_cancellation.py
    uses -- and returns cleanly without ever touching tensor data.
    """
    from app.core import device_utils

    real_resolve_device = device_utils.resolve_device
    monkeypatch.setattr(
        device_utils, "resolve_device",
        lambda requested: "meta" if requested == "cuda" else real_resolve_device(requested),
    )
    context = ExecutionContext(device=device_utils.resolve_device("cuda"))
    context.cancel()

    model = _fresh_model()
    EvaluateModelNode().execute(
        {"model": model, "dataset": _dataset()}, {}, context=context)
    assert next(model.parameters()).device.type == "meta"


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


# ── 1b (#193 item 1): precision, mirroring TrainingLoop.precision ──────


def test_precision_param_matches_training_loops_vocabulary_and_default():
    p = next(p for p in EvaluateModelNode.define_params() if p.name == "precision")
    assert p.default == "fp32"
    assert p.options == list(PRECISIONS)
    assert p.advanced is True


def test_precision_fp32_default_leaves_the_forward_pass_dtype_unchanged():
    probe = _DtypeProbe()
    EvaluateModelNode().execute(
        {"model": probe, "dataset": _dataset()}, {"device": "cpu"})
    assert probe.seen_dtype == torch.float32


def test_precision_bf16_runs_the_forward_pass_in_bf16_but_keeps_params_fp32():
    """Parameters stay fp32 regardless of precision -- there is no
    gradient here for a lower precision to make numerically UNSTABLE the
    way it can mid-training. That is NOT the same as accuracy-free: the
    forward pass itself runs in bf16, so a lower-precision logit can still
    flip an argmax on a near-tie and shift the reported number. fp32 (the
    default) is the accuracy to report; this test only pins that
    parameters are untouched, not that the two precisions agree on
    accuracy."""
    probe = _DtypeProbe()
    EvaluateModelNode().execute(
        {"model": probe, "dataset": _dataset()},
        {"device": "cpu", "precision": "bf16"})
    assert probe.seen_dtype == torch.bfloat16
    assert probe.linear.weight.dtype == torch.float32


# ── 1d (#202): step, so several EvaluateModel nodes in one graph don't
# overwrite each other's point on the eval_accuracy chart ──────────────


class _RecordingContext:
    """Collects log_metric calls; should_stop always False. Mirrors the
    stub of the same name in test_training_loop_node.py."""

    def __init__(self):
        self.current_node_id = "eval"
        self.metrics: list[tuple[str, float, int]] = []

    def should_stop(self):
        return False

    def log_metric(self, name, value, step, node_id=None):
        self.metrics.append((name, value, step))

    def can_record_artifacts(self):
        return False


def test_step_param_defaults_to_1_matching_the_prior_hardcoded_behaviour():
    p = next(p for p in EvaluateModelNode.define_params() if p.name == "step")
    assert p.default == 1


def test_eval_accuracy_is_logged_at_the_default_step_1():
    """Regression pin: leaving `step` unset must behave exactly like the
    previous hardcoded `context.log_metric("eval_accuracy", accuracy, 1)`."""
    ctx = _RecordingContext()
    result = EvaluateModelNode().execute(
        {"model": _fresh_model(), "dataset": _dataset()}, {}, context=ctx)
    assert ctx.metrics == [("eval_accuracy", result["accuracy"], 1)]


def test_two_evaluate_model_nodes_with_different_steps_do_not_collide():
    """Acceptance: several EvaluateModel nodes in one graph no longer
    overwrite each other -- e.g. a before/after-fine-tuning comparison,
    each logged at its own step on the shared eval_accuracy series."""
    ctx = _RecordingContext()
    EvaluateModelNode().execute(
        {"model": _fresh_model(), "dataset": _dataset()},
        {"step": 1}, context=ctx)
    EvaluateModelNode().execute(
        {"model": _fresh_model(), "dataset": _dataset()},
        {"step": 2}, context=ctx)
    steps = [step for name, _, step in ctx.metrics if name == "eval_accuracy"]
    assert steps == [1, 2]
