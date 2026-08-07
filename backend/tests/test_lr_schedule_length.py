"""A learning-rate schedule that does not fit the run must SAY so (#205, #244).

`TrainingLoop` steps the scheduler once per epoch, so every cycle-length
parameter on `LRScheduler` is an epoch count -- including the two PyTorch
itself counts optimizer steps for. Nothing reconciles the units, and nothing
ever failed when they disagreed: the loss curve looks plausible and the
accuracy is merely worse, which is the size of gap people then go hunting for
in their augmentation or their architecture.

Two shapes of that, and the second is the one that matters most:

* `CosineAnnealingLR.T_max` different from `TrainingLoop.epochs` (#205) --
  requires the user to have SET something wrong.
* `OneCycleLR.total_steps` left at the node's default of 1000 (#244) --
  requires the user to have changed NOTHING. Pick OneCycleLR from the
  dropdown, press Run, and the schedule silently does not work.

`test_the_default_one_cycle_schedule_really_does_never_anneal` and
`test_a_short_cosine_really_does_raise_the_lr_back_up` are here so the rest
of the file is testing a warning about something real, not a warning about a
belief. Everything else asserts the advisory reaches a surface a user can
actually read, and that it never fails a run to do it -- a truncated schedule
is a legitimate choice (#244 option 4 was rejected on exactly that ground).
"""

from __future__ import annotations

import logging

import pytest
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as sched_module

from app.core.execution_context import ExecutionContext, WarningSignal
from app.nodes.training.training_loop_node import (
    SCHEDULE_NOTE_PREFIX,
    SCHEDULE_WARNING_KIND,
    TrainingLoopNode,
    _schedule_length_note,
)


def _loader(n=8, in_features=4, out_classes=2, batch_size=4):
    X = torch.randn(n, in_features)
    y = torch.randint(0, out_classes, (n,))
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y), batch_size=batch_size)


def _train(scheduler_factory, epochs, *, context=None, lr=0.1):
    """One tiny run with the scheduler *scheduler_factory* builds.

    Returns the node's result. The scheduler is built over the same
    optimizer the loop is handed, which is what `LRScheduler` does.
    """
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    return TrainingLoopNode().execute(
        {
            "model": model,
            "dataloader": _loader(),
            "optimizer": optimizer,
            "loss_fn": nn.CrossEntropyLoss(),
            "lr_scheduler": scheduler_factory(optimizer),
        },
        {"epochs": epochs, "device": "cpu"},
        context=context,
    )


def _one_cycle(total_steps, max_lr=0.1):
    return lambda opt: sched_module.OneCycleLR(
        opt, max_lr=max_lr, total_steps=total_steps)


def _cosine(t_max):
    return lambda opt: sched_module.CosineAnnealingLR(opt, T_max=t_max)


def _warnings_on(context: ExecutionContext) -> list[WarningSignal]:
    signals, _dropped = context.outbox.drain()
    return [s for s in signals if isinstance(s, WarningSignal)]


# ── the traps are real ────────────────────────────────────────────────────


def test_the_default_one_cycle_schedule_really_does_never_anneal():
    """#244's claim, measured: at the node's default the LR only ever rises.

    `total_steps=1000` over a realistic 5 epochs traverses 0.5% of the
    cycle. One-cycle's entire benefit is the anneal -- the high-LR phase
    explores, the low-LR phase converges -- and it never arrives.
    """
    res = _train(_one_cycle(1000), epochs=5)
    history = res["metrics"]["lr_history"]

    assert history == sorted(history), "the LR should still be climbing"
    assert max(history) < 0.1 * 0.05, (
        f"the LR barely left the floor: {history}")

    # The same schedule told the truth about the run length does anneal, all
    # the way to zero. This is the accuracy the default silently forfeits.
    matched = _train(_one_cycle(5), epochs=5)["metrics"]["lr_history"]
    assert max(matched) > 0.5 * 0.1, "the warm-up should reach near max_lr"
    assert matched[-1] < matched[0], "the anneal should come back down"


def test_a_short_cosine_really_does_raise_the_lr_back_up():
    """#205's "worse in a different way": past T_max the cosine turns up."""
    history = _train(_cosine(3), epochs=6)["metrics"]["lr_history"]
    bottom = history.index(min(history))
    assert bottom < len(history) - 1, "the minimum should not be the last epoch"
    assert history[-1] > history[bottom], (
        f"the tail should be climbing again: {history}")


def test_one_cycle_shorter_than_the_run_takes_the_run_down():
    """The other direction is not silent at all -- it raises mid-run.

    Which is exactly why the advisory is worth emitting BEFORE the epochs
    that reach it, rather than letting a user discover it forty minutes in.
    """
    with pytest.raises(ValueError, match="Tried to step"):
        _train(_one_cycle(2), epochs=5)


# ── what the note says ────────────────────────────────────────────────────


def test_the_default_one_cycle_total_steps_is_reported():
    """The #244 case: the user changed nothing and gets told anyway."""
    note = _schedule_length_note(
        _one_cycle(1000)(torch.optim.SGD([nn.Parameter(torch.zeros(2))], lr=0.1)),
        epochs=20,
    )
    assert note is not None
    assert "total_steps=1000" in note and "epochs=20" in note
    # The consequence, in the words the failure does not give the user.
    assert "never anneal" in note.lower()
    # And the correction, spelled out as an epoch count -- the unit is the
    # whole trap, so naming the number without the unit would not fix it.
    assert "Set LRScheduler.total_steps to 20" in note
    assert "not the batch count" in note


def test_a_matching_one_cycle_says_nothing():
    opt = torch.optim.SGD([nn.Parameter(torch.zeros(2))], lr=0.1)
    assert _schedule_length_note(_one_cycle(20)(opt), epochs=20) is None


def test_a_one_cycle_shorter_than_the_run_names_the_crash_it_will_cause():
    opt = torch.optim.SGD([nn.Parameter(torch.zeros(2))], lr=0.1)
    note = _schedule_length_note(_one_cycle(4)(opt), epochs=20)
    assert note is not None
    assert "Tried to step 5 times" in note


@pytest.mark.parametrize(
    "t_max, epochs, expected",
    [
        (200, 100, "never reaches its minimum"),
        (50, 100, "RISING learning rate"),
    ],
)
def test_a_cosine_that_disagrees_with_epochs_names_its_own_consequence(
        t_max, epochs, expected):
    """The two directions fail differently, so they read differently."""
    opt = torch.optim.SGD([nn.Parameter(torch.zeros(2))], lr=0.1)
    note = _schedule_length_note(_cosine(t_max)(opt), epochs=epochs)
    assert note is not None
    assert expected in note
    assert f"T_max={t_max}" in note and f"epochs={epochs}" in note


def test_a_matching_cosine_says_nothing():
    opt = torch.optim.SGD([nn.Parameter(torch.zeros(2))], lr=0.1)
    assert _schedule_length_note(_cosine(100)(opt), epochs=100) is None


# ── warm restarts are the inverted case, not an exempt one ────────────────


def test_warm_restarts_are_silent_when_the_cycle_is_shorter_than_the_run():
    """T_0 < epochs is the CORRECT setting here, and must not be nagged at.

    This is the subtlety that stops "T_max should equal epochs" being a
    universal rule: `CosineAnnealingWarmRestarts` reuses the node's `T_max`
    as `T_0`, the length of the first cycle.
    """
    opt = torch.optim.SGD([nn.Parameter(torch.zeros(2))], lr=0.1)
    sched = sched_module.CosineAnnealingWarmRestarts(opt, T_0=10)
    assert _schedule_length_note(sched, epochs=100) is None


@pytest.mark.parametrize("t_0", [50, 80])
def test_warm_restarts_warn_the_other_way_round(t_0):
    """T_0 >= epochs means no restart ever happens -- the opposite advice."""
    opt = torch.optim.SGD([nn.Parameter(torch.zeros(2))], lr=0.1)
    sched = sched_module.CosineAnnealingWarmRestarts(opt, T_0=t_0)
    note = _schedule_length_note(sched, epochs=50)
    assert note is not None
    assert "no restart" in note.lower()
    # It must never repeat the CosineAnnealingLR advice, which would be
    # actively wrong here: equality is the thing to avoid, not to aim for.
    assert "SMALLER" in note
    assert "Set LRScheduler.T_max to 50" not in note


# ── schedulers that provably never fire ───────────────────────────────────


def test_a_step_lr_whose_drop_never_arrives_is_reported():
    opt = torch.optim.SGD([nn.Parameter(torch.zeros(2))], lr=0.1)
    note = _schedule_length_note(sched_module.StepLR(opt, step_size=50),
                                 epochs=20)
    assert note is not None
    assert "constant learning rate" in note


def test_a_step_lr_that_does_fire_says_nothing():
    """A period merely longer than ideal is tuning, not a fault."""
    opt = torch.optim.SGD([nn.Parameter(torch.zeros(2))], lr=0.1)
    assert _schedule_length_note(
        sched_module.StepLR(opt, step_size=19), epochs=20) is None


def test_a_multi_step_lr_whose_first_milestone_is_out_of_reach_is_reported():
    opt = torch.optim.SGD([nn.Parameter(torch.zeros(2))], lr=0.1)
    note = _schedule_length_note(
        sched_module.MultiStepLR(opt, milestones=[30, 60, 90, 120]), epochs=20)
    assert note is not None
    assert "30" in note and "constant learning rate" in note


def test_a_multi_step_lr_that_reaches_its_first_milestone_says_nothing():
    opt = torch.optim.SGD([nn.Parameter(torch.zeros(2))], lr=0.1)
    assert _schedule_length_note(
        sched_module.MultiStepLR(opt, milestones=[10, 20, 30, 40]),
        epochs=20) is None


@pytest.mark.parametrize("factory", [
    lambda opt: sched_module.ExponentialLR(opt, gamma=0.9),
    lambda opt: sched_module.ReduceLROnPlateau(opt, factor=0.1),
])
def test_schedulers_with_no_length_are_never_reported(factory):
    """Neither has an epoch-coupled length, so neither can disagree."""
    opt = torch.optim.SGD([nn.Parameter(torch.zeros(2))], lr=0.1)
    assert _schedule_length_note(factory(opt), epochs=20) is None


# ── the note must never break anything ────────────────────────────────────


def test_no_scheduler_means_no_note():
    assert _schedule_length_note(None, epochs=20) is None


@pytest.mark.parametrize("epochs", [0, -1, None, "twenty"])
def test_an_unusable_epoch_count_is_declined_rather_than_guessed_at(epochs):
    opt = torch.optim.SGD([nn.Parameter(torch.zeros(2))], lr=0.1)
    assert _schedule_length_note(_one_cycle(1000)(opt), epochs=epochs) is None


def test_a_scheduler_carrying_something_odd_is_declined_rather_than_guessed():
    """A plugin scheduler may hold a tensor under these names. Say nothing."""
    opt = torch.optim.SGD([nn.Parameter(torch.zeros(2))], lr=0.1)
    sched = _cosine(50)(opt)
    sched.T_max = torch.tensor(50)
    assert _schedule_length_note(sched, epochs=20) is None


def test_an_object_that_is_not_a_scheduler_at_all_is_declined():
    assert _schedule_length_note(object(), epochs=20) is None


# ── the surfaces a user can actually read ─────────────────────────────────


def test_the_note_reaches_the_canvas_log_through_the_result():
    """`__log__` is the only result key the canvas Log tab renders.

    Marked with the node's own prefix, because `LogEntry.type` has no
    warning severity -- an advisory and a `Print` node's output arrive in
    the same colour.
    """
    res = _train(_one_cycle(1000), epochs=5)
    assert "__log__" in res, "nothing would appear in the canvas log"
    assert res["__log__"].startswith(SCHEDULE_NOTE_PREFIX)
    assert "never anneal" in res["__log__"].lower()


def test_a_schedule_that_fits_leaves_the_canvas_log_alone():
    assert "__log__" not in _train(_one_cycle(5), epochs=5)


def test_the_note_reaches_the_durable_run_log_as_a_warning():
    """`log_warning` is what the Runs panel reads back, live, WHILE the run
    is still going -- the only surface that gets this before the epochs it
    is about have already been spent."""
    context = ExecutionContext()
    _train(_one_cycle(1000), epochs=5, context=context)

    warnings = _warnings_on(context)
    assert len(warnings) == 1, f"expected one warning, got {warnings}"
    assert warnings[0].kind == SCHEDULE_WARNING_KIND
    assert "never anneal" in warnings[0].detail.lower()


def test_a_schedule_that_fits_produces_no_run_warning():
    context = ExecutionContext()
    _train(_one_cycle(5), epochs=5, context=context)
    assert _warnings_on(context) == []


def test_the_note_reaches_the_server_log_too(caplog):
    """The only channel `run_graph.py` and an exported script have at all."""
    with caplog.at_level(logging.WARNING,
                         logger="app.nodes.training.training_loop_node"):
        _train(_one_cycle(1000), epochs=5)
    assert any("never anneal" in r.message.lower() for r in caplog.records)


def test_the_warning_is_emitted_before_the_schedule_can_fail_the_run():
    """Ordering is the point: the crash arrives at epoch 3, the note at 0."""
    context = ExecutionContext()
    with pytest.raises(ValueError, match="Tried to step"):
        _train(_one_cycle(2), epochs=5, context=context)
    warnings = _warnings_on(context)
    assert len(warnings) == 1
    assert "Tried to step 3 times" in warnings[0].detail


# ── advisory, never fatal ─────────────────────────────────────────────────


def test_a_mismatched_schedule_still_trains_the_full_run():
    """#244 option 4 ("refuse to run") was rejected: a truncated schedule is
    a legitimate choice, and refusing is too aggressive for a teaching tool.
    """
    res = _train(_cosine(200), epochs=4)
    assert res["losses"].shape == (4,)
    assert res["metrics"]["total_epochs_run"] == 4
    # The schedule itself is untouched -- the advisory does not "fix" it
    # behind the user's back, which would change what a saved graph does.
    assert res["metrics"]["lr_history"] == sorted(
        res["metrics"]["lr_history"], reverse=True)


def test_a_run_with_no_context_still_gets_the_note():
    """The REST contract runner and an exported script pass no context at
    all, so the durable channel is simply absent there. The other two
    surfaces must still work rather than the whole check being skipped."""
    res = _train(_one_cycle(1000), epochs=2, context=None)
    assert res["metrics"]["total_epochs_run"] == 2
    assert "__log__" in res


def test_a_durable_channel_that_fails_does_not_fail_the_run():
    """An advisory must not be able to take down the run it is about."""
    class _Hostile(ExecutionContext):
        def log_warning(self, kind, detail, node_id=None):
            raise RuntimeError("the outbox is on fire")

    res = _train(_one_cycle(1000), epochs=2, context=_Hostile())
    assert res["metrics"]["total_epochs_run"] == 2
    # The surface that did work still carries it.
    assert "never anneal" in res["__log__"].lower()


# ── the length is the WHOLE run's, not this leg's ─────────────────────────


def test_a_resumed_run_is_measured_against_the_absolute_epoch_count():
    """`epochs` is the absolute target and the scheduler is fast-forwarded
    to `start_epoch`, so the schedule still spans the whole run. Comparing
    against the remaining epochs instead would warn about every resume."""
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    context = ExecutionContext()
    res = TrainingLoopNode().execute(
        {
            "model": model,
            "dataloader": _loader(),
            "optimizer": optimizer,
            "loss_fn": nn.CrossEntropyLoss(),
            "lr_scheduler": _cosine(6)(optimizer),
            "start_epoch": 4,
        },
        {"epochs": 6, "device": "cpu"},
        context=context,
    )
    assert res["metrics"]["total_epochs_run"] == 2
    assert _warnings_on(context) == [], "T_max=6 matches epochs=6"
    assert "__log__" not in res
