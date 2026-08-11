"""A learning-rate schedule that does not fit the run must SAY so (#205, #244).

At the default `scheduler_step=epoch`, `TrainingLoop` steps the scheduler once
per epoch, so every cycle-length parameter on `LRScheduler` is an epoch count
-- including the two PyTorch itself counts optimizer steps for. Nothing
reconciles the units, and nothing ever failed when they disagreed: the loss
curve looks plausible and the accuracy is merely worse, which is the size of
gap people then go hunting for in their augmentation or their architecture.

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

**Which clock (#308).** #303 added `scheduler_step=optimizer_step`, and this
check kept comparing against the epoch count regardless -- so the exact
configuration #303's own description recommends (OneCycleLR with
total_steps = max_steps) got a warning telling the user to set total_steps to
the epoch count, every run. Wrong advice is worse than none: it is
indistinguishable from right advice, and following it breaks a working
schedule. The second half of this file (from "the other clock") is the
mode-aware behaviour, plus the warmup_* families (#297) whose SequentialLR
matched no branch here at all and so escaped the check entirely in BOTH modes.
"""

from __future__ import annotations

import logging

import pytest
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as sched_module

from app.core.execution_context import ExecutionContext, WarningSignal
from app.nodes.training.lr_scheduler_node import LRSchedulerNode
from app.nodes.training.training_loop_node import (
    SCHEDULE_NOTE_PREFIX,
    SCHEDULE_WARNING_KIND,
    TrainingLoopNode,
    _optimizer_step_budget,
    _schedule_length_note,
)


def _loader(n=8, in_features=4, out_classes=2, batch_size=4):
    X = torch.randn(n, in_features)
    y = torch.randint(0, out_classes, (n,))
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y), batch_size=batch_size)


def _train(scheduler_factory, epochs, *, context=None, lr=0.1, **params):
    """One tiny run with the scheduler *scheduler_factory* builds.

    Returns the node's result. The scheduler is built over the same
    optimizer the loop is handed, which is what `LRScheduler` does. The
    default loader is 8 samples at batch_size 4 -- 2 batches, and so 2
    optimizer steps, per epoch.
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
        {"epochs": epochs, "device": "cpu", **params},
        context=context,
    )


def _opt():
    return torch.optim.SGD([nn.Parameter(torch.zeros(2))], lr=0.1)


def _trace(optimizer, scheduler, steps):
    """The LR before each of *steps* scheduler steps.

    `optimizer.step()` first, in that order, because that is what the loop
    does -- and doing it the other way round makes torch warn twice about
    exactly this, which would be noise in the suite rather than a finding.
    """
    history = []
    for _ in range(steps):
        history.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return history


def _warmup(kind, warmup_steps, total_steps):
    """A warmup family built by the real `LRScheduler` node (#297).

    Built through the node rather than by hand so the composed
    `SequentialLR` this file checks is the object users actually get.
    """
    def build(optimizer):
        return LRSchedulerNode().execute(
            {"optimizer": optimizer},
            {"type": kind, "warmup_steps": warmup_steps,
             "total_steps": total_steps},
        )["scheduler"]
    return build


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


# ══ the other clock: scheduler_step=optimizer_step (#308) ═════════════════
#
# Everything above measures the schedule against `epochs`, which is right
# only while the loop steps the scheduler once per epoch. #303 added the
# other mode and this check did not learn about it.


def _step_note(scheduler, budget):
    """The note a per-step run with a *budget*-step budget would emit."""
    return _schedule_length_note(
        scheduler, epochs=5, per_step=True, step_budget=budget)


# ── the run's own step budget ─────────────────────────────────────────────


def test_the_step_budget_is_the_epoch_count_times_the_steps_in_an_epoch():
    """No max_steps: the loader's length is what makes the budget knowable."""
    assert _optimizer_step_budget(
        epochs=5, start_epoch=0, batches_per_epoch=2,
        accumulate_steps=1, max_steps=0) == 10


def test_a_binding_max_steps_is_the_budget():
    """The configuration #303 recommends: epochs high, max_steps the real end."""
    assert _optimizer_step_budget(
        epochs=100, start_epoch=0, batches_per_epoch=2,
        accumulate_steps=1, max_steps=6) == 6


def test_a_max_steps_the_run_never_reaches_does_not_become_the_budget():
    """max_steps is a cap, not a promise -- 5 epochs of 2 steps is 10, not 900."""
    assert _optimizer_step_budget(
        epochs=5, start_epoch=0, batches_per_epoch=2,
        accumulate_steps=1, max_steps=900) == 10


def test_accumulation_divides_the_budget_and_rounds_UP():
    """The loop applies an epoch's short tail window as a step of its own.

    So 5 batches at accumulate_steps=2 is THREE optimizer steps per epoch,
    not two -- see the "tail of a partial accumulation window" comment in
    the loop. Rounding down here would under-count every epoch and warn
    about schedules that fit.
    """
    assert _optimizer_step_budget(
        epochs=4, start_epoch=0, batches_per_epoch=5,
        accumulate_steps=2, max_steps=0) == 12


def test_a_resumed_run_is_still_measured_against_the_whole_run():
    """Same rule as the epoch-mode check: the schedule spans the whole run."""
    assert _optimizer_step_budget(
        epochs=5, start_epoch=3, batches_per_epoch=2,
        accumulate_steps=1, max_steps=0) == 10


def test_a_dataloader_with_no_length_and_no_max_steps_is_unknowable():
    """An IterableDataset has no len(): the steps per epoch cannot be known
    BEFORE the run, and an advisory that guesses is worse than none."""
    assert _optimizer_step_budget(
        epochs=5, start_epoch=0, batches_per_epoch=None,
        accumulate_steps=1, max_steps=0) is None


def test_a_dataloader_with_no_length_but_a_declared_budget_is_knowable():
    assert _optimizer_step_budget(
        epochs=5, start_epoch=0, batches_per_epoch=None,
        accumulate_steps=1, max_steps=1500) == 1500


def test_an_empty_loader_is_declined_rather_than_guessed_at():
    assert _optimizer_step_budget(
        epochs=5, start_epoch=0, batches_per_epoch=0,
        accumulate_steps=1, max_steps=0) is None


@pytest.mark.parametrize("epochs", [0, -1, None, "five"])
def test_an_unusable_epoch_count_yields_no_budget(epochs):
    assert _optimizer_step_budget(
        epochs=epochs, start_epoch=0, batches_per_epoch=2,
        accumulate_steps=1, max_steps=0) is None


# ── the note names the clock it is actually on ─────────────────────────────


def test_the_configuration_303_recommends_is_not_warned_about():
    """#308's headline bug: total_steps = the step budget is CORRECT."""
    assert _step_note(_one_cycle(1500)(_opt()), 1500) is None


def test_the_same_scheduler_is_judged_differently_in_the_two_modes():
    """The scheduler object is identical; only the clock changed.

    This is the whole of #308 in one assertion -- the advisory was reading
    the schedule and not the mode.
    """
    scheduler = _one_cycle(10)(_opt())
    assert _schedule_length_note(scheduler, epochs=5) is not None
    assert _schedule_length_note(
        scheduler, epochs=5, per_step=True, step_budget=10) is None


def test_a_per_step_mismatch_is_named_in_steps_and_never_in_epochs():
    """The unit IS the advice; naming the wrong one is the #308 bug."""
    note = _step_note(_one_cycle(1000)(_opt()), 10)
    assert note is not None
    assert "optimizer step" in note
    assert "Set LRScheduler.total_steps to 10" in note
    # The two epoch-mode phrasings must not survive into this mode: the
    # first is the wrong number, the second the wrong unit.
    assert "Set LRScheduler.total_steps to 5" not in note
    assert "once per EPOCH" not in note
    assert "TrainingLoop.epochs" not in note


def test_a_per_step_one_cycle_shorter_than_the_budget_names_the_crash():
    note = _step_note(_one_cycle(4)(_opt()), 20)
    assert note is not None
    assert "Tried to step 5 times" in note
    assert "optimizer step 5" in note


@pytest.mark.parametrize("t_max, budget, expected", [
    (200, 100, "never reaches its minimum"),
    (50, 100, "RISING learning rate"),
])
def test_a_per_step_cosine_reports_the_same_two_consequences(
        t_max, budget, expected):
    note = _step_note(_cosine(t_max)(_opt()), budget)
    assert note is not None
    assert expected in note
    assert f"T_max={t_max}" in note
    assert f"Set LRScheduler.T_max to {budget}" in note
    assert "optimizer steps" in note


def test_a_per_step_cosine_that_matches_the_budget_says_nothing():
    assert _step_note(_cosine(100)(_opt()), 100) is None


def test_a_per_step_step_lr_whose_drop_never_arrives_is_reported():
    note = _step_note(sched_module.StepLR(_opt(), step_size=500), 100)
    assert note is not None
    assert "constant learning rate" in note
    assert "optimizer steps" in note


def test_a_per_step_step_lr_that_does_fire_says_nothing():
    assert _step_note(sched_module.StepLR(_opt(), step_size=99), 100) is None


def test_a_per_step_multi_step_lr_out_of_reach_is_reported_in_steps():
    note = _step_note(
        sched_module.MultiStepLR(_opt(), milestones=[300, 600]), 100)
    assert note is not None
    assert "optimizer step 300" in note


def test_a_per_step_warm_restart_is_still_the_inverted_case():
    """T_0 < the budget stays the right setting; only >= is reported."""
    assert _step_note(
        sched_module.CosineAnnealingWarmRestarts(_opt(), T_0=10), 100) is None
    note = _step_note(
        sched_module.CosineAnnealingWarmRestarts(_opt(), T_0=100), 100)
    assert note is not None
    assert "no restart" in note.lower()
    assert "SMALLER" in note


@pytest.mark.parametrize("budget", [None, 0, -1, "twenty"])
def test_an_unusable_step_budget_says_nothing_rather_than_guessing(budget):
    """The #308 instruction: no advisory beats a wrong one. A budget that
    cannot be known before the run (an IterableDataset with no max_steps)
    arrives here as None."""
    assert _step_note(_one_cycle(1000)(_opt()), budget) is None


# ── end to end, through the loop that picks the clock ──────────────────────


def test_a_per_step_run_whose_schedule_fits_is_completely_silent():
    """2 batches x 5 epochs = 10 steps, and the schedule spans 10.

    Before #308 this run -- a correct one -- was told every time to set
    total_steps to 5.
    """
    context = ExecutionContext()
    res = _train(_one_cycle(10), epochs=5, context=context,
                 scheduler_step="optimizer_step")
    assert _warnings_on(context) == []
    assert "__log__" not in res
    # And the schedule really did run its whole course per-step.
    assert res["metrics"]["total_steps"] == 10


def test_a_per_step_run_with_a_step_budget_is_measured_against_it():
    """epochs=100 with max_steps=6 and total_steps=6: #303's own example."""
    context = ExecutionContext()
    res = _train(_one_cycle(6), epochs=100, context=context,
                 scheduler_step="optimizer_step", max_steps=6)
    assert _warnings_on(context) == []
    assert res["metrics"]["stopped_at_max_steps"] is True
    assert res["metrics"]["total_steps"] == 6


def test_a_per_step_run_with_the_default_total_steps_is_still_reported():
    """Mode-awareness is not an exemption -- the #244 trap still bites here,
    it just has a different right answer."""
    context = ExecutionContext()
    res = _train(_one_cycle(1000), epochs=5, context=context,
                 scheduler_step="optimizer_step")
    warnings = _warnings_on(context)
    assert len(warnings) == 1
    assert warnings[0].kind == SCHEDULE_WARNING_KIND
    assert "Set LRScheduler.total_steps to 10" in warnings[0].detail
    assert "__log__" in res


def test_an_epoch_mode_run_is_untouched_by_the_new_mode():
    """The historical default keeps its historical words, exactly."""
    context = ExecutionContext()
    _train(_one_cycle(1000), epochs=5, context=context)
    detail = _warnings_on(context)[0].detail
    assert "Set LRScheduler.total_steps to 5" in detail
    assert "once per EPOCH" in detail


def test_a_per_step_run_over_an_unmeasurable_loader_says_nothing():
    """No len() on the loader and no max_steps: the budget is unknowable
    before the run, so #308 asks for silence rather than a guess."""
    class _Unmeasurable(torch.utils.data.IterableDataset):
        def __iter__(self):
            for _ in range(2):
                yield torch.randn(4), torch.randint(0, 2, (1,)).squeeze()

    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    context = ExecutionContext()
    res = TrainingLoopNode().execute(
        {
            "model": model,
            "dataloader": torch.utils.data.DataLoader(
                _Unmeasurable(), batch_size=2),
            "optimizer": optimizer,
            "loss_fn": nn.CrossEntropyLoss(),
            "lr_scheduler": _one_cycle(1000)(optimizer),
        },
        {"epochs": 2, "device": "cpu", "scheduler_step": "optimizer_step"},
        context=context,
    )
    assert _warnings_on(context) == []
    assert "__log__" not in res


# ══ the warmup families escaped the check entirely (#308) ══════════════════
#
# warmup_cosine / warmup_linear / constant_with_warmup (#297) build a
# SequentialLR, which matched none of the isinstance branches above -- so
# the ONE family whose whole point is a step-denominated total was the one
# family never checked, in either mode.


# ── the traps are real ────────────────────────────────────────────────────


def test_a_warmup_cosine_stepped_past_its_total_really_does_turn_back_up():
    """Same shape as the bare cosine: past T_max the curve rises again."""
    optimizer = _opt()
    history = _trace(optimizer, _warmup("warmup_cosine", 3, 10)(optimizer), 16)

    bottom = history.index(min(history))
    assert bottom < len(history) - 1
    assert history[-1] > history[bottom], f"the tail should climb: {history}"


def test_a_warmup_linear_stepped_past_its_total_trains_at_nothing():
    """The other family's tail does not rise -- it clamps at ~0, which is
    compute spent without learning."""
    optimizer = _opt()
    history = _trace(optimizer, _warmup("warmup_linear", 3, 10)(optimizer), 16)

    assert max(history) > 0.05, "the ramp should reach the optimizer's LR"
    assert all(lr < 1e-6 for lr in history[10:]), (
        f"the tail should be flat at zero: {history}")


def test_a_ramp_that_never_finishes_really_does_leave_the_lr_low():
    """warmup_steps=100 on a 5-step run: the LR never gets near where the
    user set it, which is the epoch-mode trap of the default."""
    optimizer = _opt()
    _trace(optimizer, _warmup("warmup_cosine", 100, 1000)(optimizer), 5)
    assert optimizer.param_groups[0]["lr"] < 0.1 * 0.1


# ── and now they are checked, in both modes ───────────────────────────────


@pytest.mark.parametrize("kind", ["warmup_cosine", "warmup_linear"])
def test_a_warmup_schedule_matching_the_step_budget_says_nothing(kind):
    assert _step_note(_warmup(kind, 100, 1500)(_opt()), 1500) is None


@pytest.mark.parametrize("kind", ["warmup_cosine", "warmup_linear"])
def test_a_warmup_schedule_longer_than_the_step_budget_is_reported(kind):
    note = _step_note(_warmup(kind, 100, 1500)(_opt()), 500)
    assert note is not None
    assert "1500" in note and "500" in note
    assert "optimizer steps" in note
    assert "Set LRScheduler.total_steps to 500" in note


def test_a_warmup_cosine_shorter_than_the_step_budget_names_the_rise():
    note = _step_note(_warmup("warmup_cosine", 100, 500)(_opt()), 1500)
    assert note is not None
    assert "RISING" in note


def test_a_warmup_linear_shorter_than_the_step_budget_names_the_flat_tail():
    note = _step_note(_warmup("warmup_linear", 100, 500)(_opt()), 1500)
    assert note is not None
    assert "RISING" not in note, "LinearLR clamps, it does not turn back up"
    assert "1000" in note, "the wasted tail should be quantified"


def test_the_default_warmup_family_is_reported_in_epoch_mode_too():
    """Pick warmup_cosine, change nothing, leave scheduler_step alone: 100
    warmup steps against 5 epochs means the ramp never finishes."""
    note = _schedule_length_note(_warmup("warmup_cosine", 100, 1000)(_opt()),
                                epochs=5)
    assert note is not None
    assert "warmup_steps" in note
    assert "once per EPOCH" in note
    # And the fix offered here is the mode, not just a smaller number: "a few
    # percent" of a 5-epoch budget is below warmup_steps' minimum of 1.
    assert "scheduler_step to optimizer_step" in note


def test_a_constant_with_warmup_longer_than_its_ramp_says_nothing():
    """Its tail holds the LR forever, so a longer run is just a longer hold
    -- there is no length to disagree about, and nagging about the CORRECT
    configuration is the #308 mistake in a new place."""
    assert _step_note(
        _warmup("constant_with_warmup", 100, 1500)(_opt()), 1500) is None
    assert _step_note(
        _warmup("constant_with_warmup", 100, 1500)(_opt()), 99999) is None


def test_a_constant_with_warmup_whose_ramp_never_finishes_is_reported():
    note = _step_note(_warmup("constant_with_warmup", 100, 1500)(_opt()), 50)
    assert note is not None
    assert "ramp never finishes" in note


def test_a_sequential_schedule_with_no_recoverable_length_is_declined():
    """A hand-built (or plugin) SequentialLR whose tail declares no length
    gets no advisory: the composed total is not knowable, so there is
    nothing honest to compare."""
    optimizer = _opt()
    scheduler = sched_module.SequentialLR(
        optimizer,
        schedulers=[
            sched_module.LinearLR(optimizer, start_factor=0.1, total_iters=3),
            sched_module.ExponentialLR(optimizer, gamma=0.9),
        ],
        milestones=[3],
    )
    assert _step_note(scheduler, 1500) is None
    assert _schedule_length_note(scheduler, epochs=1500) is None


def test_a_warmup_family_note_reaches_the_run_through_the_loop():
    """End to end: the family that escaped the check now reaches a surface."""
    context = ExecutionContext()
    res = _train(_warmup("warmup_cosine", 100, 1000), epochs=5,
                 context=context, scheduler_step="optimizer_step")
    warnings = _warnings_on(context)
    assert len(warnings) == 1
    assert warnings[0].kind == SCHEDULE_WARNING_KIND
    assert "__log__" in res
    assert res["__log__"].startswith(SCHEDULE_NOTE_PREFIX)
