"""Gradient accumulation is arithmetic, so it is tested as arithmetic (#135).

The claim ``accumulate_steps=4`` at batch 8 makes is a strong one: the
weights it produces are the weights batch 32 would have produced. Asserting
that the loop divides by N would not test that claim -- it would test the
implementation against itself. So the reference here is built by hand:
compute the full-batch gradient with plain torch, apply one SGD step to the
known initial weights, and require ``TrainingLoopNode`` to land on exactly
that.

``test_the_reference_has_teeth`` exists to keep the tolerance honest. It
asserts the reference does NOT match the weights an implementation that
forgot to divide would produce, so a tolerance loose enough to accept both
fails the suite rather than passing it quietly.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from app.nodes.training.training_loop_node import TrainingLoopNode

LR = 0.1
SAMPLES = 32
FEATURES = 4
OUTPUTS = 3


def _fixture() -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    """A fixed model and a fixed regression dataset. No randomness at all.

    ``manual_seed`` would be enough for the model, but a hand-written
    dataset makes the reference computation below independent of torch's
    RNG stream -- so this test cannot start passing or failing because some
    other test drew a different number of randoms first.
    """
    torch.manual_seed(1234)
    model = nn.Linear(FEATURES, OUTPUTS)
    x = torch.linspace(-2.0, 2.0, SAMPLES * FEATURES).reshape(SAMPLES, FEATURES)
    y = torch.linspace(1.0, -1.0, SAMPLES * OUTPUTS).reshape(SAMPLES, OUTPUTS)
    return model, x, y


def _full_batch_gradient(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """The gradient of the mean loss over ALL of *x*, computed by hand."""
    reference = copy.deepcopy(model)
    reference.zero_grad()
    nn.MSELoss()(reference(x), y).backward()
    return {name: p.grad.detach().clone()
            for name, p in reference.named_parameters()}


def _sgd_step(
    model: nn.Module, gradient: dict[str, torch.Tensor], lr: float = LR,
) -> dict[str, torch.Tensor]:
    """Where plain SGD would put the weights after one step on *gradient*."""
    return {name: p.detach() - lr * gradient[name]
            for name, p in model.named_parameters()}


def _train(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, *,
    batch_size: int, params: dict,
) -> dict:
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y),
        batch_size=batch_size, shuffle=False,
    )
    return TrainingLoopNode().execute(
        {
            "model": model,
            "dataloader": loader,
            "optimizer": torch.optim.SGD(model.parameters(), lr=LR),
            "loss_fn": nn.MSELoss(),
        },
        {"epochs": 1, "device": "cpu", **params},
    )


def test_accumulate_four_at_batch_eight_equals_one_batch_of_thirty_two():
    model, x, y = _fixture()
    expected = _sgd_step(model, _full_batch_gradient(model, x, y))

    trained = _train(copy.deepcopy(model), x, y,
                     batch_size=8, params={"accumulate_steps": 4})

    for name, param in trained["model"].named_parameters():
        assert torch.allclose(param.detach(), expected[name], atol=1e-6), name
    # One optimizer step over four batches, and the node says so.
    assert trained["metrics"]["total_steps"] == 1
    assert trained["metrics"]["total_batches"] == 4
    assert trained["metrics"]["accumulate_steps"] == 4


def test_the_unaccumulated_full_batch_lands_on_the_same_reference():
    """The other side of the equivalence, against the SAME hand-built value.

    Without this the first test could be satisfied by a loop that is
    consistently wrong in the same way as the reference.
    """
    model, x, y = _fixture()
    expected = _sgd_step(model, _full_batch_gradient(model, x, y))

    trained = _train(copy.deepcopy(model), x, y,
                     batch_size=SAMPLES, params={})

    for name, param in trained["model"].named_parameters():
        assert torch.allclose(param.detach(), expected[name], atol=1e-6), name
    assert trained["metrics"]["total_steps"] == 1


def test_the_reference_has_teeth():
    """A missing ``/ accumulate_steps`` must not pass the tolerance above.

    Four un-divided micro-batch gradients sum to four times the mean
    gradient, so the weights would move four times as far. If ``atol``
    were ever loosened to the point of accepting that, this fails.
    """
    model, x, y = _fixture()
    gradient = _full_batch_gradient(model, x, y)
    correct = _sgd_step(model, gradient)
    undivided = _sgd_step(model, {k: v * 4 for k, v in gradient.items()})

    assert not all(
        torch.allclose(correct[name], undivided[name], atol=1e-6)
        for name in correct
    )


def test_accumulation_changes_nothing_when_the_batches_are_the_same_size():
    """``accumulate_steps=1`` is byte-for-byte the pre-#135 loop."""
    model, x, y = _fixture()
    without = _train(copy.deepcopy(model), x, y, batch_size=8, params={})
    with_explicit_one = _train(copy.deepcopy(model), x, y, batch_size=8,
                               params={"accumulate_steps": 1})

    left = dict(without["model"].named_parameters())
    for name, param in with_explicit_one["model"].named_parameters():
        assert torch.equal(param.detach(), left[name].detach()), name
    assert without["metrics"]["total_steps"] == 4


def test_a_partial_final_window_is_still_applied():
    """16 samples at batch 4, accumulated 3 deep: a window of 3 and one of 1.

    The tail must be stepped rather than dropped -- otherwise an epoch whose
    batch count is not a multiple of ``accumulate_steps`` silently trains on
    less data than it read.
    """
    model, x, y = _fixture()
    trained = _train(copy.deepcopy(model), x[:16], y[:16],
                     batch_size=4, params={"accumulate_steps": 3})

    assert trained["metrics"]["total_batches"] == 4
    assert trained["metrics"]["total_steps"] == 2


def test_gradient_clipping_bounds_the_accumulated_gradient_not_each_batch():
    """Clipping happens once per STEP, over the whole accumulated gradient.

    With a clip threshold far below the true norm, one SGD step moves the
    weights by exactly ``lr * clip`` in total. Clipping each micro-batch
    instead would let four clipped gradients sum to ``4 * clip`` before the
    step -- four times the distance -- which is what the upper bound here
    rules out.
    """
    clip = 0.01
    model, x, y = _fixture()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}

    trained = _train(copy.deepcopy(model), x, y, batch_size=8,
                     params={"accumulate_steps": 4, "grad_clip_norm": clip})

    moved = torch.cat([
        (p.detach() - before[name]).flatten()
        for name, p in trained["model"].named_parameters()
    ])
    assert torch.linalg.vector_norm(moved).item() == pytest.approx(
        LR * clip, rel=1e-4)


def test_max_steps_counts_optimizer_steps_not_batches():
    """``max_steps=2`` at ``accumulate_steps=4`` means eight forward passes."""
    model, x, y = _fixture()
    trained = _train(copy.deepcopy(model), x, y, batch_size=4,
                     params={"accumulate_steps": 4, "max_steps": 2,
                             "epochs": 5})

    assert trained["metrics"]["total_steps"] == 2
    assert trained["metrics"]["total_batches"] == 8
    assert trained["metrics"]["stopped_at_max_steps"] is True


@pytest.mark.parametrize("max_steps", [1, 2, 3, 4, 5])
def test_max_steps_is_never_overshot_by_a_tail_window(max_steps):
    """The budget holds even when the epoch does not divide by the window.

    Ten batches accumulated four deep gives windows of 4, 4 and a TAIL of
    2. The tail is a real optimizer step, so it has to be counted against
    ``max_steps`` and has to be refused once the budget is spent --
    otherwise a run told to take three steps takes four, and reads an
    extra epoch's first batch to discover it.
    """
    model, x, y = _fixture()  # 32 samples
    extra_x = torch.cat([x, x[:8]])
    extra_y = torch.cat([y, y[:8]])  # 40 samples -> 10 batches of 4
    trained = _train(copy.deepcopy(model), extra_x, extra_y, batch_size=4,
                     params={"accumulate_steps": 4, "max_steps": max_steps,
                             "epochs": 5})

    assert trained["metrics"]["total_steps"] == max_steps
    assert trained["metrics"]["stopped_at_max_steps"] is True


def test_an_interrupted_window_leaves_no_gradient_behind():
    """A discarded window is CLEARED, not merely left un-stepped.

    The model is persisted between runs, so a full model's worth of
    ``.grad`` would stay resident on the device until some later run's
    first epoch happened to zero it -- in the one feature whose whole
    subject is bounding memory.
    """
    class _StopAfter:
        def __init__(self, batches: int) -> None:
            self.seen = 0
            self.batches = batches

        def should_stop(self) -> bool:
            self.seen += 1
            return self.seen > self.batches

    model, x, y = _fixture()
    model = copy.deepcopy(model)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=4, shuffle=False)
    TrainingLoopNode().execute(
        {
            "model": model,
            "dataloader": loader,
            "optimizer": torch.optim.SGD(model.parameters(), lr=LR),
            "loss_fn": nn.MSELoss(),
        },
        {"epochs": 1, "device": "cpu", "accumulate_steps": 8},
        context=_StopAfter(3),
    )

    for name, param in model.named_parameters():
        assert param.grad is None or torch.count_nonzero(param.grad) == 0, name


def test_the_reported_loss_does_not_depend_on_accumulation():
    """The recorded loss is the UNDIVIDED one this batch actually measured.

    If the ``/ accumulate_steps`` leaked into the reported number, the same
    run at ``accumulate_steps=4`` would draw a chart four times lower than
    at 1, and the two would look like different experiments.

    One batch per epoch, deliberately: with several batches the two runs
    step at different points and their later losses legitimately diverge,
    which would leave nothing exact to compare.
    """
    model, x, y = _fixture()
    plain = _train(copy.deepcopy(model), x, y, batch_size=SAMPLES,
                   params={"epochs": 1})
    accumulated = _train(copy.deepcopy(model), x, y, batch_size=SAMPLES,
                         params={"epochs": 1, "accumulate_steps": 4})

    assert accumulated["losses"][0].item() == pytest.approx(
        plain["losses"][0].item(), rel=1e-6)
