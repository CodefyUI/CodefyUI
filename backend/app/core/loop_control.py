"""The cooperative-stop toolkit for long-running nodes (#122).

Before #122, Stop was a lie for anything with an inner loop: the engine
checked ``context.cancelled`` only BETWEEN nodes, so a ``TrainingLoop`` with
a 40-minute epoch acknowledged the click and then kept training. This module
is the three things a long-loop node needs to make Stop mean something,
kept in one place so ``TrainingLoop``, ``DiffusionTrainingLoop``,
``EvaluateModel``, ``DDPMSampler`` and ``Map`` cannot each invent their own:

``ProgressThrottle``
    Rate-limit per-iteration progress so a 5000-batch epoch reports twice a
    second rather than 5000 times.

``save_interrupt_checkpoint``
    Write the partial training state where ``CheckpointLoader`` can find it
    and register it on the run.

``interrupted_result``
    Build the ``__interrupted__`` marker the engine turns into an
    ``interrupted`` node status.

The pattern, in full::

    should_stop = stop_checker(context)
    throttle = ProgressThrottle(progress_callback)
    stopped_at = None

    for batch_index, batch in enumerate(loader):
        if should_stop():
            stopped_at = batch_index
            break
        ...
        throttle.emit({"event": EVENT_BATCH, "epoch": epoch + 1,
                       "batch": batch_index + 1, "total_batches": total,
                       "loss": loss_value})

    result = {...}                      # whatever the node managed to produce
    if stopped_at is not None:
        path = save_interrupt_checkpoint(
            context, model, optimizer, epoch=completed_epochs,
            batch=stopped_at)
        result.update(interrupted_result(
            epoch=completed_epochs, batch=stopped_at, checkpoint_path=path))
    return result

Breaking out and handling it AFTER the loop, rather than returning from
inside, is what lets the node finish its own bookkeeping (loss tensors,
metrics, the checkpoint) exactly once and in one place.

Note what is NOT here: raising. A stopped node returns its partial outputs
and says so. The engine raises ``CancellationError`` at the next node
boundary, which is where the run's outcome is decided; a node that raised
would look like a failure and lose the partial results this whole exercise
exists to keep.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from .execution_context import INTERRUPTED_KEY

logger = logging.getLogger(__name__)

#: Floor between two per-iteration progress frames, in seconds. #121
#: measured one durable event at 0.68 ms mean, so 2/s costs about 0.14% of a
#: second of wall clock -- three orders of magnitude below the work being
#: reported on, and slow enough that the browser is not repainting a chart
#: per batch either.
PROGRESS_MIN_INTERVAL_S = 0.5

#: Progress ``event`` discriminator for a per-iteration frame. Distinct from
#: ``"epoch"`` (a completed epoch, with its arrays) because the two mean
#: different things to every consumer: a batch frame is LIVENESS, throttled
#: by wall clock, and must not be mined for metric points -- several frames
#: legitimately share one step. See ``run_service.scalar_metrics``.
EVENT_BATCH = "batch"


def _never_stop() -> bool:
    return False


def stop_checker(context: Any) -> Callable[[], bool]:
    """A zero-argument ``should_stop`` for *context*. Never None.

    Lets a loop write ``if stop(): ...`` without re-testing the context on
    every iteration, and keeps a node runnable with ``context=None`` -- which
    is how the export runner, most unit tests and ``invoke_node`` on a node
    that does not declare ``context`` all call it.
    """
    checker = getattr(context, "should_stop", None)
    return checker if callable(checker) else _never_stop


def loader_length(loader: Any) -> int | None:
    """``len(loader)`` when it has one, else None.

    A ``DataLoader`` over an ``IterableDataset`` has no length, and neither
    does a hand-rolled generator someone wired into the port. "Unknown" is a
    perfectly good thing for a progress bar to be told; raising is not.
    """
    try:
        return len(loader)
    except (TypeError, AttributeError, NotImplementedError):
        return None


class ProgressThrottle:
    """Rate-limit a ``progress_callback``. Cheap enough for an inner loop.

    One ``time.monotonic()`` per call on the reject path and nothing else --
    no lock, no allocation, no formatting. Callers should still build the
    payload lazily where that is easy, but they do not have to: the cost of
    the dict is nothing next to a training step.

    The FIRST call always emits, so a long first batch shows movement
    immediately instead of after the interval. A ``None`` callback (the
    export runner, a direct unit-test call) makes every ``emit`` a no-op.
    """

    __slots__ = ("_callback", "_min_interval", "_last", "emitted")

    def __init__(
        self,
        callback: Callable[[dict[str, Any]], Any] | None,
        *,
        min_interval_s: float | None = None,
    ) -> None:
        # Read at CONSTRUCTION, not baked into the signature default: a
        # default argument is evaluated at import time, which would make
        # ``PROGRESS_MIN_INTERVAL_S`` unpatchable and every node's throttle
        # untestable without sleeping through the real interval.
        if min_interval_s is None:
            min_interval_s = PROGRESS_MIN_INTERVAL_S
        self._callback = callback
        self._min_interval = max(0.0, float(min_interval_s))
        self._last: float | None = None
        #: How many frames actually went out, for a node that wants to say
        #: so in its logs. Public because it is an observation, not state
        #: anything reads back.
        self.emitted = 0

    def emit(self, payload: dict[str, Any]) -> bool:
        """Deliver *payload* unless one went out less than the interval ago."""
        if self._callback is None:
            return False
        now = time.monotonic()
        if self._last is not None and now - self._last < self._min_interval:
            return False
        self._last = now
        self.emitted += 1
        self._callback(payload)
        return True


def interrupted_result(
    *,
    epoch: int | None = None,
    batch: int | None = None,
    checkpoint_path: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """The ``__interrupted__`` marker to merge into a node's return value.

    Keys whose value is None are omitted rather than stored as null, so the
    marker of a node with no epochs (``EvaluateModel``) does not claim to
    have stopped at epoch "none".
    """
    detail: dict[str, Any] = {
        k: v for k, v in
        (("epoch", epoch), ("batch", batch),
         ("checkpoint_path", checkpoint_path))
        if v is not None
    }
    detail.update(extra)
    return {INTERRUPTED_KEY: detail}


def save_interrupt_checkpoint(
    context: Any,
    model: Any,
    optimizer: Any,
    *,
    epoch: int,
    batch: int,
    losses: Any = None,
    lr_scheduler: Any = None,
    node_id: str | None = None,
) -> str | None:
    """Persist where an interrupted training node got to. Never raises.

    Writes an ORDINARY checkpoint (see ``core.checkpoints``) under
    ``MODELS_DIR/interrupted/`` and registers it on the run as an
    ``exec_run_artifacts`` row of kind ``checkpoint``, with the epoch and
    batch it stopped at in ``meta``. Returns the path, or None when nothing
    was written.

    **Nothing is written when the run cannot record the row.** The artifact
    row is the only index of these files, so a run without a durable signal
    consumer (the REST contract runner, an exported script, a bare
    ``execute_graph``) would leave a model-sized orphan on every timeout.
    ``context.can_record_artifacts()`` is the check; see the "No row, no
    file" section of ``core.checkpoints`` for the full reasoning.

    **What ``epoch`` must be.** The number of epochs that are COMPLETE, i.e.
    exactly what ``CheckpointLoader.epoch`` should hand to
    ``TrainingLoop.start_epoch``. Stopping part-way through epoch 3 (0-based)
    therefore stores ``epoch=3``: the resumed run re-runs epoch 3 from its
    first batch, on top of the weights the partial epoch already produced.
    That is the honest resume -- the partial epoch's updates are in the
    weights and its loss average is not, so replaying it continues the loss
    curve instead of double-counting a fraction of an epoch.

    **``batch`` is provenance, not a resume point.** It records where the
    stop landed so a user can see how much of the epoch was lost; nothing
    reads it back. Batch-precise resumption is NOT supported and would need
    more than an index -- the dataloader's sampler state, and for a shuffled
    loader its generator too. The resumed run replays the partial epoch from
    its first batch.

    **Why it cannot raise.** This runs while the user is stopping a run, on
    the way out of a node that has already decided to return partial
    results. A failed checkpoint write is worth a warning; it is not worth
    turning a clean interruption into a failed run.
    """
    if context is None:
        return None

    resolved_node_id = node_id or getattr(context, "current_node_id", "") or "node"
    run_id = getattr(context, "execution_id", "") or "run"

    can_record = getattr(context, "can_record_artifacts", None)
    if not (callable(can_record) and can_record()):
        logger.info(
            "Node %s was interrupted, but this run records no artifacts, so "
            "no interrupt checkpoint was written -- the file would have had "
            "no row referencing it and nothing would ever clean it up.",
            resolved_node_id,
        )
        return None

    from .checkpoints import interrupt_checkpoint_path, write_checkpoint

    try:
        target = interrupt_checkpoint_path(
            run_id, resolved_node_id, epoch=epoch, batch=batch)
        write_checkpoint(
            target, model, optimizer, epoch=epoch, losses=losses,
            lr_scheduler=lr_scheduler, resolve=False,
        )
    except Exception:  # noqa: BLE001 - see the docstring
        logger.warning(
            "could not write the interrupt checkpoint for node %s of run %s; "
            "the run still stops cleanly, but its partial training is lost",
            resolved_node_id, run_id, exc_info=True,
        )
        return None

    # Queued LAST, and the node returns immediately after -- see
    # ``ArtifactSignal``'s tail-safety obligation. Under drop-oldest that is
    # what makes this signal safe from a burst of per-batch progress.
    log_artifact = getattr(context, "log_artifact", None)
    if callable(log_artifact):
        log_artifact(
            "checkpoint",
            str(target),
            {"reason": "interrupted", "epoch": int(epoch), "batch": int(batch)},
            resolved_node_id,
        )
    logger.info(
        "Interrupt checkpoint for node %s written to %s (epoch=%d, batch=%d)",
        resolved_node_id, target, epoch, batch,
    )
    return str(target)
