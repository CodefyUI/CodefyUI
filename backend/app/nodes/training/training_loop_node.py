import logging
from pathlib import Path
from typing import Any

from ...core.amp import PRECISIONS
from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition

logger = logging.getLogger(__name__)


def _coerce_start_epoch(value: Any) -> int:
    """Read an epoch index off the ``start_epoch`` SCALAR port.

    ``CheckpointLoader.epoch`` hands back whatever was stored in the
    checkpoint, which is normally a plain ``int`` but may be a 0-dim tensor
    (older checkpoints written by hand) or a float. Anything unusable degrades
    to 0 -- "start from the beginning" -- with a warning, never an exception.
    """
    if value is None:
        return 0
    if hasattr(value, "item") and not isinstance(value, (int, float, bool)):
        try:
            value = value.item()
        except Exception:  # noqa: BLE001 - any odd object means "no offset"
            logger.warning("start_epoch %r is not readable as a number; starting from 0", value)
            return 0
    try:
        epoch = int(value)
    except (TypeError, ValueError):
        logger.warning("start_epoch %r is not a number; starting from 0", value)
        return 0
    if epoch < 0:
        logger.warning("start_epoch %d is negative; starting from 0", epoch)
        return 0
    return epoch


def _tracked_params(optimizer: Any) -> list[Any]:
    """Every parameter the optimizer currently owns, in param-group order."""
    return [p for group in optimizer.param_groups for p in group["params"]]


def _sync_optimizer_state_device(optimizer: Any) -> None:
    """Move stranded optimizer state onto the device of its parameter.

    ``model.to(device)`` swaps each ``Parameter``'s storage in place but leaves
    the optimizer's momentum / moment buffers wherever they were, which blows
    up on the first ``optimizer.step()`` with a device mismatch. The repair is
    a ``state_dict()`` round-trip rather than a manual ``.to()`` sweep, because
    ``Optimizer.load_state_dict`` applies torch's own per-key device policy --
    notably it keeps the 0-dim ``step`` counter on the CPU for non-fused,
    non-capturable optimizers, which a blanket ``.to(device)`` would break.

    The scan deliberately ignores 0-dim tensors for the same reason: those are
    the scalar step counters that are *supposed* to stay on the CPU.
    """
    import torch

    stranded = any(
        isinstance(value, torch.Tensor)
        and value.dim() > 0
        and value.device != param.device
        for param, state in optimizer.state.items()
        # A state dict loaded with unmatched parameter ids leaves plain ints as
        # keys; those carry no device to compare against.
        if isinstance(param, torch.Tensor)
        for value in state.values()
    )
    if stranded:
        optimizer.load_state_dict(optimizer.state_dict())


def _prepare_optimizer(optimizer: Any, model: Any) -> tuple[Any, str]:
    """Bind ``optimizer`` to ``model``'s current parameters, keeping its state.

    #118: this used to unconditionally rebuild the optimizer from
    ``type(optimizer)(model.parameters(), **optimizer.defaults)``, which threw
    away exactly the state a resume needs (Adam's ``exp_avg``/``exp_avg_sq``
    and step count, SGD's momentum buffer). The rebuild existed to re-bind the
    optimizer to parameters that ``model.to(device)`` had replaced -- a much
    smaller problem than the one it caused.

    The resume rule, in order:

    ``reused``
        Every parameter the optimizer tracks *is* (object identity) the
        corresponding parameter of the model, in order. The optimizer is used
        untouched, so all of its state survives. Only stranded state tensors
        are moved (see :func:`_sync_optimizer_state_device`).

    ``rebound``
        The optimizer tracks a different set of parameter *objects* but a
        structurally identical one (same count, same shapes, same dtypes) --
        e.g. a torch build where a device move allocates fresh ``Parameter``
        objects, or a graph that feeds a differently-constructed instance of
        the same architecture. Each group's ``params`` list is re-pointed at
        the model's parameters positionally and the optimizer's own
        ``state_dict()`` is loaded straight back, which is how torch re-keys
        per-parameter state onto new parameter objects. Hyperparameters,
        including per-group overrides, are preserved.

    ``rebuilt``
        The optimizer's parameters do not line up with the model's at all
        (different count, or different shapes), so its state cannot be
        re-keyed onto this model. Usually that means an optimizer built for
        a *different* model -- but it is also exactly what a deliberately
        PARTIAL optimizer looks like (one built over an unfrozen head only),
        and the rebuild widens it to every parameter. A fresh optimizer of
        the same class is constructed over ``model.parameters()``, its state
        is discarded, and a warning spelling both cases out is logged.

    Note that the check is on the *parameters*, not on the module object:
    ``nn.Module.to()`` mutates in place and returns ``self``, so the module is
    always the same object after a device move while its parameters may or may
    not be, depending on the torch version. Comparing parameters is the check
    that actually detects a stale binding.
    """
    model_params = list(model.parameters())
    tracked = _tracked_params(optimizer)

    if len(tracked) == len(model_params) and all(
        a is b for a, b in zip(tracked, model_params)
    ):
        _sync_optimizer_state_device(optimizer)
        return optimizer, "reused"

    structurally_same = len(tracked) == len(model_params) and all(
        a.shape == b.shape and a.dtype == b.dtype
        for a, b in zip(tracked, model_params)
    )
    if structurally_same:
        saved = optimizer.state_dict()
        offset = 0
        for group in optimizer.param_groups:
            count = len(group["params"])
            group["params"] = model_params[offset:offset + count]
            offset += count
        # Re-keys `state` from positional ids onto the parameters just bound,
        # and casts each state tensor to that parameter's device/dtype.
        optimizer.load_state_dict(saved)
        return optimizer, "rebound"

    # Filter ``defaults`` to keys the constructor actually accepts. PyTorch
    # 2.11 added ``decoupled_weight_decay`` to Adam's defaults, but AdamW
    # (an Adam subclass) re-routes that internally and rejects it as a
    # public kwarg -- a naive ``**defaults`` round-trip raises
    # ``AdamW.__init__() got an unexpected keyword argument
    # 'decoupled_weight_decay'``. Filtering by signature keeps the rebuild
    # robust against future upstream churn.
    import inspect

    optimizer_cls = type(optimizer)
    accepted = set(inspect.signature(optimizer_cls.__init__).parameters)
    optimizer_kwargs = {k: v for k, v in optimizer.defaults.items() if k in accepted}
    logger.warning(
        "Optimizer %s tracks %d parameter tensors that do not line up with the "
        "model's %d (count or shape), so its state cannot be re-keyed onto this "
        "model. Constructing a fresh %s over model.parameters() and discarding any "
        "optimizer state. Note this also widens a deliberately partial optimizer "
        "(e.g. one built over an unfrozen head only) to every parameter.",
        optimizer_cls.__name__, len(tracked), len(model_params), optimizer_cls.__name__,
    )
    return optimizer_cls(model.parameters(), **optimizer_kwargs), "rebuilt"


def _fast_forward_scheduler(lr_scheduler: Any, optimizer: Any, start_epoch: int) -> bool:
    """Advance a never-stepped scheduler to where epoch ``start_epoch`` begins.

    #118 asked for ``last_epoch=start_epoch - 1`` at construction time. The
    observable end state here is the same (``last_epoch == start_epoch``) but
    the mechanism is a replay of ``start_epoch`` ``step()`` calls from the
    scheduler's ``base_lrs``, because reconstructing is measurably wrong:

    * ``StepLR``/``ExponentialLR``/``CosineAnnealingLR`` derive the next LR
      from the optimizer's **current** ``lr``, and on a resume that value has
      already been overwritten with the decayed LR out of the checkpoint. A
      ``last_epoch=k`` rebuild therefore decays an already-decayed LR (with
      lr0=0.1, gamma=0.1, step_size=2 and start_epoch=2 it yields 0.001 where
      the straight run is at 0.01).
    * a rebuild leaves ``_step_count == 1``, which the issue explicitly calls
      out as the thing that must not be dropped, and which changes
      ``CosineAnnealingLR``'s closed-form/recursive branch.
    * a rebuild has to guess constructor kwargs out of ``state_dict()``, which
      does not round-trip for every scheduler type.

    The replay performs exactly the same float operations in the same order as
    the straight run, so the resumed LR trajectory is bit-identical to it.

    Returns True when the scheduler was advanced. ``ReduceLROnPlateau`` is
    metric-driven and cannot be replayed without the original loss history, so
    it is left alone with a warning. A scheduler that raises mid-replay is
    likewise left where it is with a warning: an unusable schedule must not
    take the training run down with it.
    """
    import re
    import warnings

    import torch.optim.lr_scheduler as sched_module

    if isinstance(lr_scheduler, sched_module.ReduceLROnPlateau):
        logger.warning(
            "ReduceLROnPlateau is driven by the validation metric and cannot be "
            "replayed from start_epoch=%d. Wire the scheduler through "
            "CheckpointSaver/CheckpointLoader to restore its state exactly.",
            start_epoch,
        )
        return False

    base_lrs = getattr(lr_scheduler, "base_lrs", None)
    if base_lrs and len(base_lrs) == len(optimizer.param_groups):
        # Replay from the schedule's own starting point, not from whatever LR
        # the restored optimizer state left in the param groups.
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr

    with warnings.catch_warnings():
        # The replay calls step() before this scheduler instance has seen an
        # optimizer.step(), which torch warns about twice over. Both are
        # expected here and only here -- so suppress those two messages by
        # name. A blanket UserWarning filter would also swallow whatever the
        # schedule itself (or a plugin's custom scheduler) has to say, which
        # the user needs to see.
        for prefix in (
            "Detected call of `lr_scheduler.step()` before `optimizer.step()`",
            "Seems like `optimizer.step()` has been overridden",
        ):
            warnings.filterwarnings(
                "ignore", message=re.escape(prefix), category=UserWarning
            )
        for completed in range(start_epoch):
            try:
                lr_scheduler.step()
            except Exception:  # noqa: BLE001 - a broken schedule must not fail the run
                logger.warning(
                    "%s raised while replaying epoch %d of %d; leaving it at "
                    "last_epoch=%s and training on. Wire the scheduler through "
                    "CheckpointSaver/CheckpointLoader to restore it exactly.",
                    type(lr_scheduler).__name__, completed + 1, start_epoch,
                    getattr(lr_scheduler, "last_epoch", "?"),
                    exc_info=True,
                )
                return False
    return True


def _prepare_scheduler(
    lr_scheduler: Any, optimizer: Any, start_epoch: int, optimizer_mode: str
) -> Any:
    """Attach the scheduler to ``optimizer`` and position it at ``start_epoch``.

    Three situations, in order of preference:

    1. the scheduler already carries restored state (``last_epoch > 0``, which
       is what ``CheckpointLoader`` produces when the checkpoint holds a
       ``scheduler_state_dict``) -- it is left exactly as it is, which is the
       only way to resume a ``ReduceLROnPlateau`` or a hand-built schedule
       faithfully;
    2. the scheduler is fresh and ``start_epoch > 0`` -- it is fast-forwarded
       (see :func:`_fast_forward_scheduler`);
    3. neither -- nothing to do.

    Independently of the above, a scheduler pointing at a *replaced* optimizer
    is re-pointed at the live one. Re-pointing rather than reconstructing keeps
    ``last_epoch``/``_step_count``/``base_lrs`` and works for scheduler types
    whose constructor arguments cannot be recovered from ``state_dict()``.
    """
    if lr_scheduler is None:
        return None

    name = type(lr_scheduler).__name__
    if getattr(lr_scheduler, "optimizer", optimizer) is not optimizer:
        lr_scheduler.optimizer = optimizer
        if optimizer_mode == "rebuilt":
            logger.warning(
                "The optimizer had to be rebuilt, so %s is now driving an optimizer "
                "it was not created for; its schedule may not line up.", name,
            )
        else:
            logger.info("Re-pointed %s at the optimizer being trained", name)

    last_epoch = getattr(lr_scheduler, "last_epoch", 0)
    if last_epoch > 0:
        if last_epoch != start_epoch:
            # The desync this issue exists to eliminate: the schedule and the
            # epoch counter disagree about where the run is. Trust the
            # scheduler's own state (it is the only faithful record for a
            # metric-driven or hand-built schedule) but say so loudly.
            logger.warning(
                "%s is at last_epoch=%d but training resumes at epoch %d. The "
                "learning rate will follow the scheduler, not start_epoch. This "
                "usually means the checkpoint's scheduler state and its epoch were "
                "written at different times.",
                name, last_epoch, start_epoch,
            )
        else:
            logger.info("%s resumes at last_epoch=%d (restored state)", name, last_epoch)
    elif start_epoch > 0 and _fast_forward_scheduler(lr_scheduler, optimizer, start_epoch):
        logger.info("Fast-forwarded %s to last_epoch=%d", name, lr_scheduler.last_epoch)

    return lr_scheduler


class TrainingLoopNode(BaseNode):
    """Train a model, optionally resuming a previous run.

    **Indexing basis (#118).** Two things are counted differently and the
    difference is deliberate:

    * *Epoch numbers are absolute* -- they index the whole training run, not
      this call. ``epochs`` is the absolute target, the loop runs
      ``range(start_epoch, epochs)``, and the epoch numbers in the log lines,
      the ``epoch`` field of progress events and ``metrics["best_epoch"]`` are
      all the absolute one. Resuming at 2 with ``epochs=4`` reports "3/4" and
      "4/4".
    * *Arrays are per call* -- the ``losses`` and ``val_losses`` outputs, the
      ``losses`` field of progress events and ``metrics["lr_history"]`` cover
      only the epochs **this** call ran. A resumed leg's ``losses`` therefore
      holds the remaining epochs alone, not the whole history; the earlier
      epochs live in the checkpoint's ``losses``. Map array index to absolute
      epoch with ``metrics["start_epoch"]``, which progress events also carry
      as ``start_epoch`` whenever it is non-zero.

    **Stopping (#122).** ``context.should_stop()`` is polled once per batch,
    training and validation alike, so Stop lands within one batch instead of
    at the end of the run. On a stop the node does NOT raise: it writes an
    interrupt checkpoint (``MODELS_DIR/interrupted/``, registered as an
    ``exec_run_artifacts`` row of kind ``checkpoint``), returns the partial
    ``losses``/``val_losses``/``metrics`` it has, and marks the result
    interrupted so the engine reports an ``interrupted`` node status.
    Resuming is the ordinary #118 path -- ``CheckpointLoader.epoch`` into
    ``start_epoch`` -- and the checkpoint stores the number of COMPLETE
    epochs, so a run stopped part-way through epoch 3 resumes by re-running
    epoch 3 over the weights that partial epoch already produced.

    One consequence worth knowing: an interruption during VALIDATION leaves
    ``val_losses`` one entry shorter than ``losses``, because that epoch's
    training finished and its validation did not.

    **Periodic checkpointing (#203).** ``should_stop()`` and the interrupt
    checkpoint above cover every way a RUN can end early -- a Stop click, a
    cancelled browser tab, the submitting process exiting. None of them
    cover the SERVER PROCESS dying outright (SIGKILL, an OOM kill, a
    restart): that path never reaches the interrupt checkpoint, because it
    only runs on the way OUT of a node that has already decided to return.
    (NOT power loss -- ``write_checkpoint`` deliberately skips fsync, so a
    rename can be visible with unflushed content; a process death survives
    because the OS and its page cache do, a power cut does not. See
    ``core.checkpoints``'s "Durability" section.) ``checkpoint_every``
    (epochs, ``0`` = disabled, matching ``early_stopping_patience``'s own
    convention and every other opt-in knob on this node) closes the
    process-death gap: every Nth COMPLETE epoch, on a run that is training
    normally, an ordinary checkpoint is written under ``MODELS_DIR/periodic/``
    and registered exactly like the interrupt checkpoint (an
    ``exec_run_artifacts`` row of kind ``checkpoint``, same
    ``can_record_artifacts()`` gating, same atomic write). Resuming from one
    needs no new concepts -- the same ``CheckpointLoader.epoch ->
    start_epoch`` wiring.

    Cost is real and NOT bounded while the run is active: each checkpoint is
    written synchronously on the training thread and costs roughly model
    size plus optimizer state (often several times the model's own size) --
    a 200-epoch ResNet-18 at ``checkpoint_every=1`` is on the order of
    18 GB before the run ends. Retention (``RunStore.prune``) is what
    bounds the file count, but only once a run's row ages out of the
    keep-last-N window, which structurally cannot happen while the run is
    still running -- see ``core.checkpoints``' "No row, no file" section.
    Each write's duration and this run's cumulative periodic-checkpoint
    bytes are logged (INFO), so the cost is visible rather than only felt
    as "training got slower". A run left ``interrupted`` -- including by a
    server restart retiring an abandoned one -- keeps its periodic
    checkpoint files even once retention removes its row; see
    ``RunStore.prune``'s docstring for why.

    **Fitting a bigger run on one card (#135).** Two independent levers,
    both off by default:

    * ``precision`` -- ``bf16`` runs the forward pass and the loss under
      ``autocast``, roughly halving activation memory on Ampere and newer
      with no scaler and no numerical babysitting. ``fp16`` does the same
      for older cards and adds a ``GradScaler``. See ``core.amp`` for what
      each device can honour and what happens when it cannot.
    * ``accumulate_steps`` -- run N micro-batches, dividing each one's loss
      by N, and step the optimizer once. Gradients are then mathematically
      the same as one batch N times larger (exactly, when the loss is a
      mean and every micro-batch is full), so a batch size the card cannot
      hold becomes a batch size it can.

    The two compose, and both interact with the rest of the loop in ways
    worth stating: gradient clipping happens at STEP time over the whole
    accumulated gradient (never per micro-batch, which would clip a
    quarter of a gradient to the full-gradient threshold); ``max_steps``
    counts OPTIMIZER steps, so it means the same thing at any
    ``accumulate_steps``; and ``train_loss_batch`` keeps being stepped by
    micro-batch, because that is what it measures.

    **Metrics (#122).** Per epoch, stepped by the ABSOLUTE epoch number:
    ``train_loss``, ``lr``, ``val_loss`` when a validation loader is wired,
    and ``patience_counter``/``best_epoch`` when early stopping is on. Under
    the ``batch_metrics`` option, ``train_loss_batch`` per batch, stepped by
    the global batch index. Note the rename: the series the run service used
    to INFER from the epoch progress payload was called ``loss``; it is
    ``train_loss`` now, which is the same number under a name that lines up
    with ``val_loss``.

    **val_accuracy (#202).** Recorded alongside ``val_loss``, under the same
    gate (a validation loader must be wired) plus one more: ``loss_fn`` must
    be a classification loss (``nn.CrossEntropyLoss`` or ``nn.NLLLoss`` --
    the only two in this repo's loss map whose targets are integer class
    indices against ``[B, C]`` logits, which is the shape ``argmax(dim=1)``
    assumes). A regression or BCE run therefore never gets a val_accuracy
    series. ``monitor`` (default ``val_loss``) can point early stopping at
    it instead; requesting ``val_accuracy`` without either gate above
    holding falls back to ``val_loss`` with a warning rather than
    comparing against a value that was never computed.
    """

    NODE_NAME = "TrainingLoop"
    CATEGORY = "Training"
    DESCRIPTION = (
        "Run a training loop with optional validation, early stopping, "
        "learning rate scheduling, and gradient clipping."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL, description="Model to train"),
            PortDefinition(name="dataloader", data_type=DataType.DATALOADER, description="Training data loader"),
            PortDefinition(name="optimizer", data_type=DataType.OPTIMIZER, description="Optimizer for parameter updates"),
            PortDefinition(name="loss_fn", data_type=DataType.LOSS_FN, description="Loss function"),
            PortDefinition(name="val_dataloader", data_type=DataType.DATALOADER, description="Validation data loader", optional=True),
            PortDefinition(name="lr_scheduler", data_type=DataType.ANY, description="Learning rate scheduler", optional=True),
            PortDefinition(
                name="start_epoch",
                data_type=DataType.SCALAR,
                description=(
                    "Resume from this epoch (0-based). Wire CheckpointLoader.epoch here; "
                    "the loop then runs epochs start_epoch..epochs-1"
                ),
                optional=True,
            ),
            PortDefinition(
                name="grad_scaler_state",
                data_type=DataType.ANY,
                description=(
                    "fp16 loss-scale state to resume from. Wire "
                    "CheckpointLoader.grad_scaler_state here; ignored unless "
                    "precision is fp16"
                ),
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL, description="Trained model (best if early stopping)"),
            PortDefinition(name="losses", data_type=DataType.TENSOR, description="Training loss per epoch"),
            PortDefinition(name="val_losses", data_type=DataType.TENSOR, description="Validation loss per epoch (empty if no val_dataloader)"),
            PortDefinition(name="metrics", data_type=DataType.ANY, description="Training metrics dict (final_loss, best_epoch, lr_history, etc.)"),
            PortDefinition(
                name="grad_scaler_state",
                data_type=DataType.ANY,
                description=(
                    "fp16 loss-scale state to store in a checkpoint "
                    "(None unless precision is fp16)"
                ),
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="epochs", param_type=ParamType.INT, default=5, description="Number of training epochs", min_value=1),
            ParamDefinition(
                name="device",
                param_type=ParamType.SELECT,
                default="auto",
                description="Device to train on ('auto' follows the global device)",
                options=["auto", "cpu", "cuda", "mps"],
            ),
            ParamDefinition(
                name="early_stopping_patience",
                param_type=ParamType.INT,
                default=0,
                description="Stop if the monitored metric doesn't improve for N epochs (0 = disabled)",
                min_value=0,
            ),
            ParamDefinition(
                name="monitor",
                param_type=ParamType.SELECT,
                default="val_loss",
                description=(
                    "Metric early stopping watches. val_loss: lower is "
                    "better (the default). val_accuracy: higher is "
                    "better -- only recorded for a classification loss "
                    "(CrossEntropyLoss/NLLLoss) with a val_dataloader "
                    "wired; falls back to val_loss with a warning when "
                    "neither holds, rather than watching a value that "
                    "was never computed."
                ),
                options=["val_loss", "val_accuracy"],
            ),
            ParamDefinition(
                name="checkpoint_every",
                param_type=ParamType.INT,
                default=0,
                description=(
                    "Save a checkpoint every N completed epochs, so a "
                    "server crash loses at most N epochs instead of the "
                    "whole run. Independent of CheckpointSaver and resumes "
                    "the same way: wire CheckpointLoader.epoch to "
                    "start_epoch. Each checkpoint is roughly model size "
                    "plus optimizer state (often several times the "
                    "model's own size), written synchronously on the "
                    "training thread; a low N on a large model over a "
                    "long run can use many GB before the run ends, since "
                    "nothing bounds this while the run is still active "
                    "(0 = disabled)"
                ),
                min_value=0,
            ),
            ParamDefinition(
                name="grad_clip_norm",
                param_type=ParamType.FLOAT,
                default=0.0,
                description="Max gradient norm for clipping (0 = disabled)",
                min_value=0.0,
            ),
            ParamDefinition(
                name="batch_metrics",
                param_type=ParamType.BOOL,
                default=False,
                description=(
                    "Also record the loss of every batch as the "
                    "'train_loss_batch' series (off by default: one row per "
                    "batch is a lot of rows)"
                ),
            ),
            ParamDefinition(
                name="precision",
                param_type=ParamType.SELECT,
                default="fp32",
                description=(
                    "Mixed precision. bf16 roughly halves activation memory "
                    "on Ampere and newer with no other change; fp16 does the "
                    "same on older cards and adds a loss scaler. val_accuracy "
                    "floats at this same precision rather than being forced "
                    "to fp32, so it stays comparable with val_loss -- but a "
                    "lower-precision logit can flip an argmax on a near-tie "
                    "and shift the reported accuracy slightly. A device "
                    "that cannot honour the choice falls back to fp32 and "
                    "says so."
                ),
                options=list(PRECISIONS),
                advanced=True,
            ),
            ParamDefinition(
                name="accumulate_steps",
                param_type=ParamType.INT,
                default=1,
                description=(
                    "Run this many batches before each optimizer step, "
                    "dividing each one's loss by the same number. 4 at "
                    "batch_size 8 gives the gradients of batch_size 32 "
                    "while only ever holding 8 in memory (1 = off)."
                ),
                min_value=1,
                advanced=True,
            ),
            ParamDefinition(
                name="max_steps",
                param_type=ParamType.INT,
                default=0,
                description=(
                    "Stop after this many optimizer steps in total, whatever "
                    "epochs says. Counts optimizer steps, not batches, so it "
                    "means the same thing at any accumulate_steps "
                    "(0 = no limit)"
                ),
                min_value=0,
                advanced=True,
            ),
            ParamDefinition(
                name="log_interval",
                param_type=ParamType.INT,
                default=1,
                description=(
                    "Record every Nth batch when batch metrics are on. Raise "
                    "it to thin out a long run's chart."
                ),
                min_value=1,
                visible_when={"batch_metrics": True},
                advanced=True,
            ),
            ParamDefinition(
                name="deterministic",
                param_type=ParamType.BOOL,
                default=False,
                description=(
                    "Ask torch for deterministic kernels. Ops with no "
                    "deterministic implementation warn instead of failing."
                ),
                advanced=True,
            ),
            ParamDefinition(
                name="tensorboard",
                param_type=ParamType.BOOL,
                default=False,
                description=(
                    "Also write the metrics as TensorBoard event files, "
                    "under the run's own folder. Open them with "
                    "`tensorboard --logdir <path>`; the path is listed as an "
                    "artifact of the run."
                ),
                advanced=True,
            ),
        ]

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        progress_callback: Any | None = None,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        """Own the TensorBoard writer's lifetime; train in ``_run_training``.

        The split exists for one reason: ``close_and_register`` must run on
        the exception path too. "No row, no file" (``core.checkpoints``, and
        ``open_run_writer``'s own docstring) means a directory nothing
        references is litter no retention sweep can find -- and a run that
        raises mid-training is *exactly* when the partial loss curve matters
        most. Registering it here means a crashed run's curves are listed as
        an artifact and reachable from the Runs panel, instead of sitting on
        disk with nothing pointing at them.

        Queueing it in a ``finally`` also makes it genuinely LAST, after
        ``save_interrupt_checkpoint``, which is what
        ``close_and_register``'s tail-safety docstring always claimed and
        did not previously get: the outbox drops the OLDEST item, so an
        artifact row queued before a burst of per-batch progress frames is
        the one that disappears.
        """
        from ...core.tensorboard import close_and_register, open_run_writer

        # #136: the same series, in a second place. Opened BEFORE any
        # training so the "no row, no file" check happens before the
        # directory is created, not after a full run has filled it.
        #
        # It also sits before this method's input unpacking and parameter
        # coercion, and that ordering is deliberate -- core#136 re-review,
        # N-3. The cost is that an ``execute`` which dies while reading its
        # own inputs now leaves a 40-byte header-only directory AND an
        # artifact row, where it previously left nothing. The gain is that
        # every failure AFTER that point -- ``model.to(device)`` running out
        # of memory, an optimizer that rejects the model's parameters, a
        # first batch that raises -- leaves a directory that is INDEXED
        # rather than orphaned. Those failures already created the directory
        # before this change; what they did not create was the row pointing
        # at it, so they were litter no retention sweep could find. Trading
        # "nothing" for "a row" on a narrow window (graph validation already
        # rejects unconnected required inputs) to convert real orphans into
        # listed artifacts is the right way round. Please do not "fix" this
        # by moving the open below the input unpacking.
        tb_writer = (open_run_writer(context)
                     if bool(params.get("tensorboard", False)) else None)
        try:
            result = self._run_training(
                inputs, params, progress_callback,
                context=context, tb_writer=tb_writer,
            )
        finally:
            # Total by contract (see close_and_register): raising in here
            # would REPLACE whatever killed the run with an error about
            # logging, which is the one thing instrumentation must never do.
            tb_logdir = close_and_register(tb_writer, context)

        if tb_logdir is not None and isinstance(result.get("metrics"), dict):
            result["metrics"]["tensorboard_logdir"] = tb_logdir
        return result

    def _run_training(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        progress_callback: Any | None = None,
        *,
        context: Any = None,
        tb_writer: Any = None,
    ) -> dict[str, Any]:
        import torch
        import torch.nn as nn

        from ...core.amp import AmpPolicy
        from ...core.device_utils import resolve_node_device, to_device
        from ...core.loop_control import (
            EVENT_BATCH,
            ProgressThrottle,
            interrupted_result,
            loader_length,
            save_interrupt_checkpoint,
            save_periodic_checkpoint,
            stop_checker,
        )
        from ...core.seeding import apply_determinism

        model = inputs["model"]
        dataloader = inputs["dataloader"]
        optimizer = inputs["optimizer"]
        loss_fn = inputs["loss_fn"]
        # CrossEntropyLoss / NLLLoss are the only two losses in this repo's
        # map (loss_node.py's LOSS_TYPES) whose targets are integer class
        # indices against [B, C] logits -- the shape argmax(dim=1) assumes.
        # Every other loss (BCE*, the regression losses) is not
        # argmax-shaped, so val_accuracy is only ever computed, recorded,
        # or monitored by early stopping when this holds. Getting this
        # wrong in the permissive direction -- e.g. admitting
        # BCEWithLogitsLoss -- either crashes on a shape mismatch or
        # silently broadcasts into a meaningless number, which is worse
        # than not shipping the feature at all.
        is_classification_loss = isinstance(loss_fn, (nn.CrossEntropyLoss, nn.NLLLoss))
        val_dataloader = inputs.get("val_dataloader")
        lr_scheduler = inputs.get("lr_scheduler")
        start_epoch = _coerce_start_epoch(inputs.get("start_epoch"))

        epochs = params.get("epochs", 5)
        device = resolve_node_device(params.get("device"), context)
        patience = params.get("early_stopping_patience", 0)
        monitor = params.get("monitor", "val_loss")
        checkpoint_every = int(params.get("checkpoint_every", 0) or 0)
        grad_clip = params.get("grad_clip_norm", 0.0)
        batch_metrics = bool(params.get("batch_metrics", False))
        max_steps = int(params.get("max_steps", 0) or 0)
        log_interval = max(1, int(params.get("log_interval", 1) or 1))
        accumulate_steps = max(1, int(params.get("accumulate_steps", 1) or 1))

        # Built once for the whole call, so the fallback warning is logged
        # once rather than per batch, and so an fp16 run's loss scale is one
        # continuous trajectory instead of restarting every epoch.
        policy = AmpPolicy.for_device(params.get("precision"), device)
        policy.load_state_dict(inputs.get("grad_scaler_state"))

        # ORed with the run option rather than overriding it: the run-level
        # switch is "make this whole run reproducible" and a node must not
        # be able to opt out of it. Turning it on here covers the case the
        # run option cannot reach — a graph exported to Python, or a canvas
        # run whose submitter did not ask.
        #
        # Safe to latch from inside a node because ``execute_graph`` wraps
        # every run in ``deterministic_scope`` and hands the process setting
        # back afterwards. Without that, opening someone else's saved graph
        # would have made every LATER run in the server deterministic.
        if bool(params.get("deterministic", False)) or getattr(
                context, "deterministic", False):
            apply_determinism(True)

        def record_metric(name: str, value: Any, step: int) -> None:
            """One point, to the run's metric store and to TensorBoard.

            One call site for both so the two can never drift into
            disagreeing about what was recorded -- which would be the worst
            possible failure for something whose only job is to say what
            happened.
            """
            if context is not None:
                context.log_metric(name, value, step)
            if tb_writer is not None:
                tb_writer.add_scalar(name, value, step)

        should_stop = stop_checker(context)
        throttle = ProgressThrottle(progress_callback)
        total_batches = loader_length(dataloader)
        total_val_batches = loader_length(val_dataloader)

        model = to_device(model, device)
        loss_fn = to_device(loss_fn, device)

        # #118: keep the incoming optimizer (and therefore its state) whenever
        # it is usable for this model -- see _prepare_optimizer for the rule.
        optimizer, optimizer_mode = _prepare_optimizer(optimizer, model)
        lr_scheduler = _prepare_scheduler(lr_scheduler, optimizer, start_epoch, optimizer_mode)

        if start_epoch >= epochs:
            logger.warning(
                "start_epoch=%d is at or past epochs=%d; no epoch will run.",
                start_epoch, epochs,
            )

        # Training config for frontend display
        param_count = sum(p.numel() for p in model.parameters())
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        training_config = {
            "model_class": model.__class__.__name__,
            "params": param_count,
            "trainable": trainable_count,
            "optimizer": optimizer.__class__.__name__,
            "lr": optimizer.param_groups[0].get("lr", "N/A"),
            "loss_fn": loss_fn.__class__.__name__,
            "epochs": epochs,
            "start_epoch": start_epoch,
            "device": device,
            "batch_size": getattr(dataloader, "batch_size", "N/A"),
            "early_stopping": patience > 0,
            "patience": patience,
            "grad_clip_norm": grad_clip if grad_clip > 0 else "disabled",
            "has_validation": val_dataloader is not None,
            "has_lr_scheduler": lr_scheduler is not None,
            # Both the request and what the device could honour, because
            # "I asked for bf16" and "this ran in bf16" are different
            # statements and only one of them is visible from the loss curve.
            "precision": policy.precision,
            "precision_requested": policy.requested,
            "accumulate_steps": accumulate_steps,
        }
        batch_size = getattr(dataloader, "batch_size", None)
        if accumulate_steps > 1 and isinstance(batch_size, int):
            training_config["effective_batch_size"] = batch_size * accumulate_steps
        if policy.fell_back:
            logger.warning(
                "precision=%s was requested but %s runs this in %s; the run "
                "continues at the lower memory saving.",
                policy.requested, device, policy.precision,
            )

        if progress_callback:
            progress_callback({"event": "config", "config": training_config})

        epoch_losses: list[float] = []
        val_epoch_losses: list[float] = []
        val_epoch_accuracies: list[float] = []
        lr_history: list[float] = []

        # #203: running total of what checkpoint_every has written so far
        # THIS call, logged alongside each new periodic checkpoint. Not a
        # limit -- retention (RunStore.prune) cannot reach a still-active
        # run, so nothing here bounds it; this only makes the cost visible.
        # See checkpoint_every's own description for the size/cost warning.
        periodic_checkpoint_bytes = 0

        # Early stopping state.
        #
        # `monitor` picks which scalar is watched: val_loss (lower is
        # better, the default -- unchanged from before this param existed)
        # or val_accuracy (higher is better). val_accuracy only exists when
        # BOTH gates above hold (a val_dataloader is wired AND loss_fn is
        # classification -- see is_classification_loss); asking to monitor
        # it without either means there is no series to watch. This
        # codebase's convention for an unusable request is to degrade with
        # a loud warning rather than fail the run (precision falling back
        # to fp32, a scheduler replay that cannot proceed, ...), so the
        # same rule applies here: fall back to val_loss, which itself falls
        # further back to avg_train_loss via the pre-existing rule just
        # below when there is no val_dataloader at all. Resolved once, here
        # -- not per epoch -- so the warning (if any) is logged once.
        #
        # A THIRD reason joins the two above: `monitor` itself might not be
        # one of the two values the param declares at all. The editor never
        # sends anything else, but a hand-built graph.json or CLI
        # invocation is not guaranteed to -- and a typo ("accuracy" instead
        # of "val_accuracy") must not silently monitor loss with no signal,
        # the same way the other two reasons are not silent.
        monitor_is_accuracy = monitor == "val_accuracy"
        monitor_fallback_reason: str | None = None
        if patience > 0:
            if monitor not in ("val_loss", "val_accuracy"):
                monitor_fallback_reason = (
                    f"{monitor!r} is not a recognised monitor value "
                    "(expected 'val_loss' or 'val_accuracy')"
                )
            elif monitor_is_accuracy and val_dataloader is None:
                monitor_fallback_reason = "no val_dataloader is wired"
            elif monitor_is_accuracy and not is_classification_loss:
                monitor_fallback_reason = (
                    "loss_fn is not a classification loss (CrossEntropyLoss "
                    "or NLLLoss)"
                )
        if monitor_fallback_reason is not None:
            logger.warning(
                "monitor=%r was requested for early stopping, but %s. "
                "Falling back to monitor=val_loss.",
                monitor, monitor_fallback_reason,
            )
            monitor_is_accuracy = False
        best_monitor_value = float("-inf") if monitor_is_accuracy else float("inf")
        best_epoch = 0
        best_state_dict = None
        patience_counter = 0
        # Set True the epoch early stopping actually breaks the loop on.
        # NOT derived from best_state_dict/patience_counter at exit time
        # (which is what metrics["stopped_early"] used to do): the
        # monitor=val_accuracy degenerate path below can leave
        # best_state_dict at None on every epoch (nothing ever counts as
        # an improvement when there is nothing to compare), which made
        # that derivation silently say False for a run that plainly did
        # stop early.
        early_stopping_triggered = False
        # Logged once, the first epoch it happens, not every epoch of a
        # persistently-empty val_dataloader.
        monitor_value_missing_warned = False

        # Where a stop landed: the 0-based index of the batch that never ran,
        # and which loop it was in. None until someone presses Stop.
        stopped_at: dict[str, Any] | None = None
        global_batch = 0
        # Optimizer steps that actually moved the weights. Equal to
        # ``global_batch`` in the default configuration, and the honest
        # denominator once accumulation (several batches per step) or fp16
        # (a step skipped on an overflowing gradient) is in play.
        optimizer_steps = 0
        # Set when ``max_steps`` ends the run. Distinct from ``stopped_at``:
        # that means "a human pressed Stop and these results are partial",
        # this means "the configured budget was spent and the run is done".
        step_budget_reached = False

        def apply_gradients() -> bool:
            """Clip, step, and clear -- one accumulation window's update.

            Clipping happens HERE rather than after each micro-batch's
            backward, because the thing that must be bounded is the
            gradient the optimizer actually uses. Clipping a quarter of a
            gradient against the whole-gradient threshold would leave the
            sum unbounded and the individual contributions distorted.

            The unscale before it is what makes the norm meaningful under
            fp16: the gradients are still multiplied by the loss scale at
            this point, so a clip measured on them would compare a number
            around 65536x too large against the threshold.
            """
            if grad_clip > 0:
                policy.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            applied = policy.step(optimizer)
            optimizer.zero_grad()
            return applied

        # ``epoch`` is the ABSOLUTE epoch index for the whole training run, not
        # an offset into this call: a resume at start_epoch=2 with epochs=4 runs
        # epochs 2 and 3, and every epoch number it reports (logs, progress
        # events, best_epoch) is that absolute number.
        for epoch in range(start_epoch, epochs):
            # ── Training phase ──
            model.train()
            running_loss = 0.0
            batch_count = 0
            # Micro-batches whose gradients are sitting in ``.grad`` waiting
            # for a step. Reset per epoch: an accumulation window never
            # spans an epoch boundary, so an epoch's last partial window is
            # applied as a step of its own rather than being carried into
            # the next epoch's first window.
            pending = 0
            optimizer.zero_grad()

            for batch_index, batch_data in enumerate(dataloader):
                # #122: once per batch, before any work. A plain
                # threading.Event read -- cheap enough to sit in the hot loop,
                # which is the whole point of doing it here rather than at the
                # node boundary the engine already checks.
                if should_stop():
                    stopped_at = {"phase": "train", "batch": batch_index}
                    break

                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    data, targets = batch_data
                    data = to_device(data, device)
                    targets = to_device(targets, device)
                else:
                    data = to_device(batch_data, device) if hasattr(batch_data, "to") else batch_data
                    targets = None

                # The forward pass AND the loss go under autocast: torch
                # picks the right dtype per op, and keeping the loss inside
                # is what makes reductions that need float32 (softmax,
                # cross-entropy) run in float32 rather than in bf16.
                with policy.autocast():
                    outputs = model(data)

                    if targets is not None:
                        loss = loss_fn(outputs, targets)
                    else:
                        loss = loss_fn(outputs)

                # Divided by the WINDOW size, not by how many micro-batches
                # this window ended up with. Every micro-batch then
                # contributes exactly 1/N of the update, which is what makes
                # N windows of B equal to one batch of N*B. A final short
                # window therefore takes a proportionally smaller step, which
                # is the honest treatment of a smaller sample -- normalising
                # it back up would give a full-sized update computed from
                # fewer examples.
                policy.scale(loss / accumulate_steps).backward()
                pending += 1

                # The UNSCALED, undivided loss: what this batch actually
                # measured. The division is a detail of how the gradient is
                # assembled and must not leak into the reported curve, or
                # the same run at a different accumulate_steps would draw a
                # different chart.
                batch_loss = loss.item()
                running_loss += batch_loss
                batch_count += 1
                global_batch += 1

                if pending >= accumulate_steps:
                    if apply_gradients():
                        optimizer_steps += 1
                    pending = 0

                if batch_metrics and global_batch % log_interval == 0:
                    # A series of its own, stepped by GLOBAL batch index --
                    # not "train_loss at a finer resolution", which would put
                    # two different x-axes in one series.
                    #
                    # ``log_interval`` thins the series without touching the
                    # step numbers, so a chart drawn at interval 10 lines up
                    # with one drawn at interval 1 -- it simply has fewer
                    # points. Keying off ``global_batch`` rather than
                    # ``batch_index`` keeps the spacing even across an epoch
                    # boundary whose batch count is not a multiple of it.
                    record_metric("train_loss_batch", batch_loss,
                                  global_batch)
                throttle.emit({
                    "event": EVENT_BATCH,
                    "epoch": epoch + 1,
                    "batch": batch_index + 1,
                    "total_batches": total_batches,
                    "loss": round(batch_loss, 6),
                })

                # A step budget, checked AFTER the step so ``max_steps=1``
                # means one step actually happened. It ends the epoch early,
                # and the epoch-level bookkeeping below still runs on the
                # partial epoch -- an epoch that trained on half its batches
                # has a real average loss, and dropping it would leave the
                # chart with nothing to show for the work.
                #
                # Counted in OPTIMIZER steps, so "stop after 100 steps"
                # means the same amount of learning whether the batches are
                # accumulated or not. At the default accumulate_steps=1 the
                # two counters are the same number.
                if max_steps and optimizer_steps >= max_steps:
                    step_budget_reached = True
                    break

            # The tail of a partial accumulation window. Applied rather than
            # discarded: those micro-batches already paid for their forward
            # and backward passes, and dropping their gradient would make an
            # epoch whose batch count is not a multiple of accumulate_steps
            # quietly train on less data than it read.
            #
            # NOT applied when the run is ending, for either reason. A stop
            # means "these results are partial", and stepping on the way out
            # would move the weights once more after the user asked for the
            # run to end. A spent ``max_steps`` budget is the same statement
            # in a different voice -- the tail would be an (N+1)th optimizer
            # step on a run that was told to take N, which is exactly what
            # the parameter exists to prevent.
            #
            # Either way the accumulated gradient is CLEARED rather than
            # left behind: the model is persisted in the node state store
            # between runs, so a full model's worth of ``.grad`` would stay
            # resident until some later run's first epoch zeroed it.
            if pending:
                if stopped_at is None and not step_budget_reached:
                    if apply_gradients():
                        optimizer_steps += 1
                    # Re-checked here and not only inside the batch loop:
                    # a budget reached BY the tail step must end the run now,
                    # rather than after the next epoch has read one more
                    # batch to discover it.
                    if max_steps and optimizer_steps >= max_steps:
                        step_budget_reached = True
                else:
                    optimizer.zero_grad()
                pending = 0

            if stopped_at is not None:
                break

            avg_train_loss = running_loss / max(batch_count, 1)
            epoch_losses.append(avg_train_loss)

            current_lr = optimizer.param_groups[0].get("lr", 0)
            lr_history.append(current_lr)

            # ── Validation phase ──
            avg_val_loss = None
            avg_val_accuracy = None
            if val_dataloader is not None:
                model.eval()
                val_running_loss = 0.0
                val_batch_count = 0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch_index, batch_data in enumerate(val_dataloader):
                        if should_stop():
                            stopped_at = {"phase": "val", "batch": batch_index}
                            break

                        if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                            data, targets = batch_data
                            data = to_device(data, device)
                            targets = to_device(targets, device)
                        else:
                            data = to_device(batch_data, device) if hasattr(batch_data, "to") else batch_data
                            targets = None

                        # Validation runs under the SAME autocast as
                        # training. Two reasons, and the second is the one
                        # that bites: the memory saving is largest exactly
                        # here (no gradients to keep, so activations are
                        # the whole cost), and a validation loss computed
                        # in a different precision from the training loss
                        # is not comparable with it -- which is the only
                        # thing anyone ever does with it.
                        with policy.autocast():
                            outputs = model(data)
                            if targets is not None:
                                loss = loss_fn(outputs, targets)
                            else:
                                loss = loss_fn(outputs)
                        val_running_loss += loss.item()
                        val_batch_count += 1

                        # Accuracy from this SAME forward pass -- no second
                        # model call. Same argmax-vs-label comparison as
                        # EvaluateModel (evaluate_model_node.py:163-165),
                        # deliberately at whatever precision ``outputs`` was
                        # already produced at (the policy above) rather than
                        # forced to fp32: the val_loss next to it is not
                        # fp32-forced either, and the two must stay
                        # comparable (see the ``precision`` param
                        # description for the consequence -- a lower-
                        # precision logit can flip an argmax on a near-tie).
                        #
                        # Classification-only: see is_classification_loss
                        # above. A regression/BCE loss's outputs are not
                        # [B, C] class logits, so argmax(dim=1) over them is
                        # not "the predicted class" -- it is a shape that
                        # happens to run (or, for a [B, 1] regression head,
                        # a broadcasted comparison that happens NOT to
                        # crash) and a number with no meaning either way.
                        if is_classification_loss and targets is not None:
                            pred = outputs.argmax(dim=1)
                            val_correct += int((pred == targets).sum().item())
                            val_total += int(targets.numel())

                        throttle.emit({
                            "event": EVENT_BATCH,
                            "epoch": epoch + 1,
                            "batch": batch_index + 1,
                            "total_batches": total_val_batches,
                            "loss": round(val_running_loss
                                          / max(val_batch_count, 1), 6),
                            "phase": "val",
                        })

                if stopped_at is not None:
                    # This epoch's TRAINING is already banked in
                    # ``epoch_losses``; only its validation is missing. The
                    # resume point is therefore the next epoch.
                    break

                avg_val_loss = val_running_loss / max(val_batch_count, 1)
                val_epoch_losses.append(avg_val_loss)
                if is_classification_loss and val_total > 0:
                    avg_val_accuracy = val_correct / val_total
                    val_epoch_accuracies.append(avg_val_accuracy)

            # ── LR Scheduler step ──
            if lr_scheduler is not None:
                # ReduceLROnPlateau needs a metric
                if hasattr(lr_scheduler, "step"):
                    import torch.optim.lr_scheduler as sched_module
                    if isinstance(lr_scheduler, sched_module.ReduceLROnPlateau):
                        metric = avg_val_loss if avg_val_loss is not None else avg_train_loss
                        lr_scheduler.step(metric)
                    else:
                        lr_scheduler.step()

            # ── Early stopping check ──
            stopped_early = False
            if patience > 0:
                if monitor_is_accuracy:
                    monitor_value = avg_val_accuracy
                else:
                    monitor_value = avg_val_loss if avg_val_loss is not None else avg_train_loss
                # monitor_value can only be None here if monitor_is_accuracy
                # but this SPECIFIC epoch produced no accuracy points (e.g.
                # an empty val_dataloader) despite the run-level gates
                # passing. Loud, like the two run-level fallbacks above:
                # silently spending patience on a comparison that never
                # happened would be the same mistake in a smaller dose.
                if (monitor_is_accuracy and monitor_value is None
                        and not monitor_value_missing_warned):
                    logger.warning(
                        "monitor=val_accuracy but epoch %d produced no "
                        "val_accuracy points (e.g. an empty val_dataloader); "
                        "treating it as no improvement for early stopping.",
                        epoch + 1,
                    )
                    monitor_value_missing_warned = True
                improved = monitor_value is not None and (
                    monitor_value > best_monitor_value if monitor_is_accuracy
                    else monitor_value < best_monitor_value
                )
                if improved:
                    best_monitor_value = monitor_value
                    best_epoch = epoch + 1
                    best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    stopped_early = True
                    early_stopping_triggered = True

            logger.info(
                "Epoch %d/%d - Train Loss: %.4f%s%s",
                epoch + 1, epochs, avg_train_loss,
                f" - Val Loss: {avg_val_loss:.4f}" if avg_val_loss is not None else "",
                f" - LR: {current_lr:.6f}" if lr_scheduler else "",
            )

            # #122: the durable series, logged BEFORE the progress frame so
            # the run service knows this node reports its own metrics and
            # stops inferring duplicates from the payload below.
            #
            # Because inference stands down per NODE, this list has to cover
            # everything ``scalar_metrics`` used to mine out of the epoch
            # payload, or a series would silently disappear. That payload
            # yields lr, loss, val_loss and -- when early stopping is on --
            # patience_counter and best_epoch. All of them are here; the one
            # deliberate change is that ``loss`` is now ``train_loss``, which
            # names the same number and lines up with ``val_loss``.
            record_metric("train_loss", avg_train_loss, epoch + 1)
            if avg_val_loss is not None:
                record_metric("val_loss", avg_val_loss, epoch + 1)
            if avg_val_accuracy is not None:
                record_metric("val_accuracy", avg_val_accuracy, epoch + 1)
            record_metric("lr", current_lr, epoch + 1)
            if patience > 0:
                record_metric("patience_counter", patience_counter,
                              epoch + 1)
                record_metric("best_epoch", best_epoch, epoch + 1)

            if progress_callback:
                progress_data = {
                    "event": "epoch",
                    "epoch": epoch + 1,
                    "total_epochs": epochs,
                    "loss": round(avg_train_loss, 6),
                    "losses": [round(l, 6) for l in epoch_losses],
                    "lr": current_lr,
                }
                if avg_val_loss is not None:
                    progress_data["val_loss"] = round(avg_val_loss, 6)
                    progress_data["val_losses"] = [round(l, 6) for l in val_epoch_losses]
                if avg_val_accuracy is not None:
                    progress_data["val_accuracy"] = round(avg_val_accuracy, 6)
                if patience > 0:
                    progress_data["patience_counter"] = patience_counter
                    progress_data["best_epoch"] = best_epoch
                if start_epoch > 0:
                    # ``losses`` only covers the epochs THIS call ran, so a
                    # consumer plotting it against absolute epochs needs the
                    # offset. Omitted entirely when there is none, so a normal
                    # run's payload is byte-for-byte what it was before #118.
                    progress_data["start_epoch"] = start_epoch
                progress_callback(progress_data)

            # #203: a periodic checkpoint, taken on a HEALTHY run so the
            # server process itself -- the one failure nothing else in this
            # stack survives -- is no longer a single point of failure for
            # hours of completed epochs. Keyed off the ABSOLUTE epoch
            # number, so a resumed run keeps hitting the same multiples it
            # would have hit uninterrupted, exactly like every other
            # epoch-indexed decision in this loop. Placed AFTER the epoch is
            # fully banked (loss, LR, early-stopping bookkeeping, progress)
            # so the checkpoint's own `epoch`/`losses` line up with what was
            # just reported, and BEFORE the early-stopping/step-budget/
            # should_stop exits below so the run's very last epoch is
            # covered too when it happens to land on a multiple.
            if checkpoint_every and (epoch + 1) % checkpoint_every == 0:
                periodic_path = save_periodic_checkpoint(
                    context, model, optimizer,
                    epoch=epoch + 1,
                    losses=torch.tensor(epoch_losses, dtype=torch.float32),
                    lr_scheduler=lr_scheduler,
                    scaler_state=policy.state_dict(),
                )
                # Cumulative, not per-call: retention cannot reach a
                # still-active run (see the class docstring's "Periodic
                # checkpointing" section), so this total can only grow for
                # as long as the run keeps training -- exactly the cost
                # checkpoint_every's own description warns about.
                if periodic_path is not None:
                    try:
                        periodic_checkpoint_bytes += Path(periodic_path).stat().st_size
                        logger.info(
                            "checkpoint_every: %.1f MB written to periodic "
                            "checkpoints so far this run",
                            periodic_checkpoint_bytes / (1024 * 1024),
                        )
                    except OSError:
                        pass  # already logged by save_periodic_checkpoint itself

            if stopped_early:
                logger.info("Early stopping triggered at epoch %d (best epoch: %d)", epoch + 1, best_epoch)
                break

            # The step budget ends the whole run, not just the epoch it was
            # reached in. Checked here, alongside early stopping, because it
            # is the same kind of event: a configured reason to finish, and
            # a normal completion rather than an interruption.
            if step_budget_reached:
                logger.info(
                    "max_steps=%d reached at epoch %d; stopping",
                    max_steps, epoch + 1)
                break

            # #122: a stop that arrives between epochs. Checked AFTER early
            # stopping so a run that finished on its own terms is reported as
            # finished, not interrupted.
            if should_stop():
                stopped_at = {"phase": "epoch", "batch": 0}
                break

        # Restore best model if early stopping was used and found a best.
        # NOT on an interruption: the point of stopping is to resume, and
        # resuming means continuing from where training actually is, not
        # rewinding the weights to an earlier epoch behind the user's back.
        if best_state_dict is not None and stopped_at is None:
            model.load_state_dict(best_state_dict)
            logger.info("Restored best model from epoch %d", best_epoch)

        losses_tensor = torch.tensor(epoch_losses, dtype=torch.float32)
        val_losses_tensor = torch.tensor(val_epoch_losses, dtype=torch.float32) if val_epoch_losses else torch.tensor([], dtype=torch.float32)

        # Complete epochs, which is exactly what CheckpointLoader.epoch has
        # to hand back to start_epoch. A mid-epoch stop does not count the
        # partial epoch (its weight updates are in the model, its loss
        # average is not), so the resumed run re-runs it from batch 0.
        completed_epochs = start_epoch + len(epoch_losses)

        checkpoint_path: str | None = None
        if stopped_at is not None:
            # The epoch the stop landed IN, 1-based. Only the ``train`` phase
            # stops inside an epoch that has not been counted yet; ``val`` and
            # ``epoch`` both stop after one was banked.
            stopped_in_epoch = completed_epochs + (
                1 if stopped_at["phase"] == "train" else 0)
            logger.info(
                "Training interrupted during %s of epoch %d/%d at batch %d",
                stopped_at["phase"], stopped_in_epoch, epochs,
                stopped_at["batch"],
            )
            checkpoint_path = save_interrupt_checkpoint(
                context, model, optimizer,
                epoch=completed_epochs, batch=stopped_at["batch"],
                losses=losses_tensor, lr_scheduler=lr_scheduler,
                scaler_state=policy.state_dict(),
            )

        # Epoch numbers are absolute; counts are for this call only.
        # ``best_epoch`` keeps its pre-#118 value when start_epoch is 0.
        metrics = {
            "final_train_loss": epoch_losses[-1] if epoch_losses else 0.0,
            "final_val_loss": val_epoch_losses[-1] if val_epoch_losses else None,
            "final_val_accuracy": val_epoch_accuracies[-1] if val_epoch_accuracies else None,
            "best_epoch": best_epoch if patience > 0 else completed_epochs,
            "total_epochs_run": len(epoch_losses),
            "start_epoch": start_epoch,
            "last_epoch": completed_epochs,
            # #202 review: was `best_state_dict is not None and
            # patience_counter >= patience`, an indirect proxy that
            # silently read False whenever best_state_dict never got set
            # -- which the monitor=val_accuracy degenerate path (an empty
            # val_dataloader; see early_stopping_triggered's own comment)
            # made newly possible even though the loop DID exit early.
            # early_stopping_triggered is set at the exact point the break
            # actually happens, so it cannot drift from what the loop did.
            "stopped_early": early_stopping_triggered,
            "lr_history": lr_history,
            "interrupted": stopped_at is not None,
            # How much training actually happened, and whether the budget is
            # why it ended. ``total_steps`` is reported unconditionally
            # because "how many optimizer steps was that?" is the question
            # ``max_steps`` makes people ask. It counts OPTIMIZER steps;
            # ``total_batches`` counts the forward passes that fed them, and
            # the two differ exactly when accumulation or an fp16 skip is in
            # play.
            "total_steps": optimizer_steps,
            "total_batches": global_batch,
            "stopped_at_max_steps": step_budget_reached,
            "accumulate_steps": accumulate_steps,
            "precision": policy.precision,
        }
        if policy.fell_back:
            metrics["precision_requested"] = policy.requested
        if patience > 0:
            # Only meaningful when early stopping actually ran -- with
            # patience=0, monitor is not consulted at all, so reporting it
            # would imply a comparison that never happened.
            metrics["monitor"] = "val_accuracy" if monitor_is_accuracy else "val_loss"
            if monitor_fallback_reason is not None:
                metrics["monitor_requested"] = monitor
        # ``tensorboard_logdir`` is added by ``execute`` after the writer is
        # closed and registered, which happens in a ``finally`` outside this
        # method so a run that raises still gets its artifact row.
        if policy.uses_scaler:
            # Zero is a meaningful reading here ("the loss scale never
            # overflowed"), so it is reported whenever there was a scaler
            # to do the skipping rather than only when it did.
            metrics["skipped_steps"] = policy.skipped_steps
        if stopped_at is not None:
            metrics["interrupt_checkpoint"] = checkpoint_path

        result: dict[str, Any] = {
            "model": model,
            "losses": losses_tensor,
            "val_losses": val_losses_tensor,
            "metrics": metrics,
            # None on every fp32 and bf16 run, which is what CheckpointSaver
            # expects to see for "there is no loss scale to store".
            "grad_scaler_state": policy.state_dict(),
        }
        if stopped_at is not None:
            result.update(interrupted_result(
                epoch=completed_epochs, batch=stopped_at["batch"],
                checkpoint_path=checkpoint_path, phase=stopped_at["phase"],
            ))
        return result
