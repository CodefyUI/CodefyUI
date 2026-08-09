"""Regression tests for #149 - a discarded LR schedule must say so.

#149 was filed as "LRScheduler wired after CheckpointLoader resumes from
already-decayed base_lrs". That mechanism does not reproduce, and the
issue carries the measurements saying so: ``LRScheduler.__init__`` reads
``group.setdefault("initial_lr", ...)``, and ``initial_lr`` survives an
``Optimizer.state_dict()`` round trip, so a scheduler built on a restored
optimizer still starts from the original learning rate in either wiring
order (both measured to 1.4e-17 of the closed-form cosine). The one thing
that could have broken it -- a checkpoint with no ``initial_lr`` at all --
this product does not write, and the ``initial_lrs`` payload key added
since makes CodefyUI's contract independent of torch's choice anyway (see
``test_training_resume.py``'s "honest base_lrs" section).

What IS reachable in that wiring is a different loss, and it is the one
this file covers. ``CheckpointLoader.lr_scheduler`` is an optional input.
Leave it unwired and a checkpoint that carries ``scheduler_state_dict``
has nothing to be restored INTO, so the state is dropped and the schedule
is instead reconstructed by replaying ``start_epoch`` steps from
``base_lrs``. For a closed-form schedule that replay is exact -- which is
why this never surfaced as a wrong number. For a metric-driven one it
cannot be: measured on a ``ReduceLROnPlateau`` resumed from an 8-epoch
checkpoint whose last five epochs were a plateau, the restored-state path
came back with ``best=0.8, num_bad_epochs=5`` and the discard path with
``best=inf, num_bad_epochs=0`` -- the decay that was one epoch away
postponed indefinitely.

Both halves used to be invisible: the loader logged the discard at INFO
and the fast-forward's refusal went to ``logger.warning``, i.e. the server
console. Neither reaches a user watching the canvas. They now go through
``core.advisories``, the same three surfaces #252 built for the
schedule-length notes -- server log, durable run log, and the node's
``__log__`` (the canvas Log tab).

Advisory, never fatal: resuming onto a reset schedule is still a resume,
and refusing to run is too aggressive for a teaching tool. That is the
same call #244 made for the schedule-length notes.
"""

from __future__ import annotations

import logging

import pytest
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as sched_module

from app.config import settings
from app.core.advisories import emit_advisory, join_notes
from app.core.checkpoints import write_checkpoint
from app.core.execution_context import ExecutionContext, WarningSignal
from app.nodes.io.checkpoint_node import (
    CHECKPOINT_NOTE_PREFIX,
    SCHEDULER_STATE_DISCARDED_KIND,
    CheckpointLoaderNode,
)
from app.nodes.training.training_loop_node import (
    SCHEDULE_NOTE_PREFIX,
    SCHEDULE_RESUME_WARNING_KIND,
    SCHEDULE_WARNING_KIND,
    TrainingLoopNode,
    _prepare_scheduler,
)


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    d = tmp_path / "data" / "models"
    d.mkdir(parents=True)
    monkeypatch.setattr(settings, "MODELS_DIR", d)
    return d


def _model():
    torch.manual_seed(0)
    return nn.Linear(4, 2)


def _warnings_on(context: ExecutionContext) -> list[WarningSignal]:
    signals, _dropped = context.outbox.drain()
    return [s for s in signals if isinstance(s, WarningSignal)]


def _loader(n=8, batch_size=4):
    x = torch.randn(n, 4)
    y = torch.randint(0, 2, (n,))
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=batch_size)


def _write_checkpoint_with_schedule(name: str, *, epoch: int = 4):
    """A checkpoint that carries a scheduler's position, as #118 writes it."""
    import warnings

    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = sched_module.StepLR(optimizer, step_size=2, gamma=0.5)
    with warnings.catch_warnings():
        # Stepping without a preceding optimizer.step() is what building
        # the fixture requires; torch's advice is for real training loops.
        warnings.filterwarnings("ignore", message="Detected call of")
        for _ in range(epoch):
            scheduler.step()
    write_checkpoint(name, model, optimizer, epoch=epoch,
                     lr_scheduler=scheduler)
    return model, optimizer


def _load(name, *, lr_scheduler=None, context=None):
    model = _model()
    return CheckpointLoaderNode().execute(
        {
            "model": model,
            "optimizer": torch.optim.SGD(model.parameters(), lr=0.1),
            "lr_scheduler": lr_scheduler,
        },
        {"path": name, "device": "cpu"},
        context=context,
    )


# ─────────────────────────────────────────────────────────────────────────
# 1. CheckpointLoader: the state it drops is state it HAS
# ─────────────────────────────────────────────────────────────────────────

def test_discarding_a_stored_schedule_reaches_the_canvas_log(models_dir):
    _write_checkpoint_with_schedule("with_sched.pt")
    result = _load("with_sched.pt")

    note = result.get("__log__")
    assert note is not None, (
        "the checkpoint holds a schedule position that is being thrown "
        "away; the canvas Log tab is the only place the user would see it")
    assert note.startswith(CHECKPOINT_NOTE_PREFIX), note
    assert "lr_scheduler" in note and "StepLR" in note, (
        "the note has to name the port to wire and the schedule that was "
        f"stored, or it is not actionable: {note!r}")


def test_discarding_a_stored_schedule_reaches_the_durable_run_log(models_dir):
    _write_checkpoint_with_schedule("with_sched.pt")
    context = ExecutionContext(graph_id="test-149", device="cpu")
    _load("with_sched.pt", context=context)

    warnings = _warnings_on(context)
    assert [w.kind for w in warnings] == [SCHEDULER_STATE_DISCARDED_KIND], (
        f"expected exactly one discarded-schedule warning, got {warnings}")


def test_discarding_a_stored_schedule_reaches_the_server_log(models_dir, caplog):
    _write_checkpoint_with_schedule("with_sched.pt")
    with caplog.at_level(logging.WARNING):
        _load("with_sched.pt")

    assert any(r.levelno >= logging.WARNING and "lr_scheduler" in r.getMessage()
               for r in caplog.records), (
        "an INFO line is what this was before #149; the server log is the "
        "only channel an exported script or run_graph.py has at all")


def test_a_wired_scheduler_restores_the_state_and_says_nothing(models_dir):
    """The correct wiring must stay quiet, or the advisory is noise."""
    _write_checkpoint_with_schedule("with_sched.pt", epoch=4)
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    live = sched_module.StepLR(optimizer, step_size=2, gamma=0.5)

    context = ExecutionContext(graph_id="test-149", device="cpu")
    result = CheckpointLoaderNode().execute(
        {"model": model, "optimizer": optimizer, "lr_scheduler": live},
        {"path": "with_sched.pt", "device": "cpu"},
        context=context,
    )

    assert "__log__" not in result, result.get("__log__")
    assert _warnings_on(context) == []
    assert live.last_epoch == 4, (
        "the whole point of wiring it is that the position is restored "
        f"exactly; last_epoch={live.last_epoch}")


def test_a_checkpoint_with_no_schedule_says_nothing(models_dir):
    """Nothing was lost, so there is nothing to report.

    A checkpoint written before #118, or by a graph that had no scheduler,
    is the ordinary case -- warning about it would train users to ignore
    the warning that matters.
    """
    model = _model()
    write_checkpoint("plain.pt", model,
                     torch.optim.SGD(model.parameters(), lr=0.1), epoch=2)

    context = ExecutionContext(graph_id="test-149", device="cpu")
    result = _load("plain.pt", context=context)

    assert "__log__" not in result
    assert _warnings_on(context) == []


def test_the_advisory_does_not_disturb_what_the_loader_returns(models_dir):
    """``__log__`` is additive: every declared output is still there."""
    _write_checkpoint_with_schedule("with_sched.pt", epoch=4)
    result = _load("with_sched.pt")

    for port in CheckpointLoaderNode.define_outputs():
        assert port.name in result, port.name
    assert result["epoch"] == 4


# ─────────────────────────────────────────────────────────────────────────
# 2. TrainingLoop: the schedule it could not put back where it was
# ─────────────────────────────────────────────────────────────────────────

def test_a_plateau_schedule_that_cannot_be_replayed_is_reported():
    """The measured half of #149, at the point the information is lost."""
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = sched_module.ReduceLROnPlateau(optimizer, factor=0.5)
    context = ExecutionContext(graph_id="test-149", device="cpu")

    prepared, note = _prepare_scheduler(scheduler, optimizer, 8, "kept", context)

    assert prepared is scheduler
    assert note is not None and note.startswith(SCHEDULE_NOTE_PREFIX), note
    assert "ReduceLROnPlateau" in note and "CheckpointLoader" in note, (
        f"the note must name the schedule and the fix: {note!r}")
    assert [w.kind for w in _warnings_on(context)] == [
        SCHEDULE_RESUME_WARNING_KIND]


def test_a_closed_form_schedule_that_replays_exactly_says_nothing():
    """StepLR's replay IS the original trajectory, so there is no advisory."""
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = sched_module.StepLR(optimizer, step_size=2, gamma=0.5)
    context = ExecutionContext(graph_id="test-149", device="cpu")

    _prepared, note = _prepare_scheduler(scheduler, optimizer, 4, "kept", context)

    assert note is None
    assert _warnings_on(context) == []
    assert scheduler.last_epoch == 4


def test_a_restored_schedule_says_nothing_even_for_a_plateau():
    """Wired through the loader, a plateau resumes exactly. No advisory.

    This is the pair to the first test in this section: the SAME scheduler
    type, the difference being only that its state came back, which is
    exactly what the advisory tells the user to do.
    """
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = sched_module.ReduceLROnPlateau(optimizer, factor=0.5)
    scheduler.load_state_dict(
        {**scheduler.state_dict(), "last_epoch": 8, "best": 0.8,
         "num_bad_epochs": 5})
    context = ExecutionContext(graph_id="test-149", device="cpu")

    _prepared, note = _prepare_scheduler(scheduler, optimizer, 8, "kept", context)

    assert note is None
    assert _warnings_on(context) == []
    assert scheduler.num_bad_epochs == 5, (
        "the plateau history has to survive untouched -- that is the thing "
        "the unwired path loses")


def test_a_run_that_cannot_replay_still_trains_and_still_reports():
    """Advisory, not fatal: the run completes and the note rides out on it."""
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    context = ExecutionContext(graph_id="test-149", device="cpu")

    result = TrainingLoopNode().execute(
        {
            "model": model,
            "dataloader": _loader(),
            "optimizer": optimizer,
            "loss_fn": nn.CrossEntropyLoss(),
            "lr_scheduler": sched_module.ReduceLROnPlateau(optimizer, factor=0.5),
            "start_epoch": 8,
        },
        {"epochs": 10, "device": "cpu"},
        context=context,
    )

    assert len(result["losses"]) == 2, "epochs 9 and 10 must actually run"
    assert result["__log__"].startswith(SCHEDULE_NOTE_PREFIX)
    assert "ReduceLROnPlateau" in result["__log__"]


def test_both_schedule_notes_stack_rather_than_one_winning():
    """A resumed run can be wrong in two ways at once.

    ``__log__`` is a single string, so before #149 a second note would have
    silently replaced the first. Here the schedule cannot be replayed AND
    its length disagrees with the run.
    """
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    context = ExecutionContext(graph_id="test-149", device="cpu")

    # StepLR whose only drop is far past the end of the run, so the
    # schedule-LENGTH note fires. Armed after construction so it raises on
    # the replay only -- the other branch _fast_forward_scheduler declines
    # on, and the one that leaves the run trainable.
    scheduler = _UnreplayableStepLR(optimizer, step_size=50)
    scheduler.armed = True

    result = TrainingLoopNode().execute(
        {
            "model": model,
            "dataloader": _loader(),
            "optimizer": optimizer,
            "loss_fn": nn.CrossEntropyLoss(),
            "lr_scheduler": scheduler,
            "start_epoch": 8,
        },
        {"epochs": 10, "device": "cpu"},
        context=context,
    )

    note = result["__log__"]
    assert "could not be repositioned" in note, note
    assert "never happens" in note or "constant learning rate" in note, note
    assert {w.kind for w in _warnings_on(context)} == {
        SCHEDULE_RESUME_WARNING_KIND, SCHEDULE_WARNING_KIND}


class _UnreplayableStepLR(sched_module.StepLR):
    """Raises once, the first time it is stepped after being armed.

    Stands in for a plugin scheduler whose replay blows up. Arming happens
    after construction because torch steps a scheduler once inside
    ``__init__``, and raising there would be a different bug entirely.
    """

    armed = False

    def get_lr(self):
        if self.armed:
            self.armed = False
            raise RuntimeError("cannot replay this schedule")
        return super().get_lr()


# ─────────────────────────────────────────────────────────────────────────
# 3. The shared emitter
# ─────────────────────────────────────────────────────────────────────────

def test_emit_advisory_reaches_all_three_surfaces(caplog):
    context = ExecutionContext(graph_id="test-149", device="cpu")
    with caplog.at_level(logging.WARNING):
        line = emit_advisory("something is off", kind="k", prefix="[X] ",
                             context=context)

    assert line == "[X] something is off"
    assert any("something is off" in r.getMessage() for r in caplog.records)
    warnings = _warnings_on(context)
    assert [(w.kind, w.detail) for w in warnings] == [("k", "something is off")]


def test_emit_advisory_survives_a_broken_durable_channel():
    """A run must never fail because it tried to warn about itself."""

    class _Broken:
        def log_warning(self, kind, detail):
            raise RuntimeError("outbox is gone")

    assert emit_advisory("note", kind="k", prefix="[X] ",
                         context=_Broken()) == "[X] note"


def test_emit_advisory_without_a_context_still_returns_the_line():
    assert emit_advisory("note", kind="k", prefix="[X] ") == "[X] note"


def test_join_notes_stacks_and_drops_the_empties():
    assert join_notes(None, None) is None
    assert join_notes("a") == "a"
    assert join_notes("a", None, "b") == "a\n\nb"
