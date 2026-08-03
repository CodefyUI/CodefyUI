import logging
from typing import Any

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

    **Metrics (#122).** Per epoch, stepped by the ABSOLUTE epoch number:
    ``train_loss``, ``lr``, ``val_loss`` when a validation loader is wired,
    and ``patience_counter``/``best_epoch`` when early stopping is on. Under
    the ``batch_metrics`` option, ``train_loss_batch`` per batch, stepped by
    the global batch index. Note the rename: the series the run service used
    to INFER from the epoch progress payload was called ``loss``; it is
    ``train_loss`` now, which is the same number under a name that lines up
    with ``val_loss``.
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
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL, description="Trained model (best if early stopping)"),
            PortDefinition(name="losses", data_type=DataType.TENSOR, description="Training loss per epoch"),
            PortDefinition(name="val_losses", data_type=DataType.TENSOR, description="Validation loss per epoch (empty if no val_dataloader)"),
            PortDefinition(name="metrics", data_type=DataType.ANY, description="Training metrics dict (final_loss, best_epoch, lr_history, etc.)"),
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
                description="Stop if val loss doesn't improve for N epochs (0 = disabled)",
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
                name="max_steps",
                param_type=ParamType.INT,
                default=0,
                description=(
                    "Stop after this many optimizer steps in total, whatever "
                    "epochs says (0 = no limit)"
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
        ]

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        progress_callback: Any | None = None,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        import torch

        from ...core.device_utils import resolve_node_device, to_device
        from ...core.loop_control import (
            EVENT_BATCH,
            ProgressThrottle,
            interrupted_result,
            loader_length,
            save_interrupt_checkpoint,
            stop_checker,
        )
        from ...core.seeding import apply_determinism

        model = inputs["model"]
        dataloader = inputs["dataloader"]
        optimizer = inputs["optimizer"]
        loss_fn = inputs["loss_fn"]
        val_dataloader = inputs.get("val_dataloader")
        lr_scheduler = inputs.get("lr_scheduler")
        start_epoch = _coerce_start_epoch(inputs.get("start_epoch"))

        epochs = params.get("epochs", 5)
        device = resolve_node_device(params.get("device"), context)
        patience = params.get("early_stopping_patience", 0)
        grad_clip = params.get("grad_clip_norm", 0.0)
        batch_metrics = bool(params.get("batch_metrics", False))
        max_steps = int(params.get("max_steps", 0) or 0)
        log_interval = max(1, int(params.get("log_interval", 1) or 1))

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
        }

        if progress_callback:
            progress_callback({"event": "config", "config": training_config})

        epoch_losses: list[float] = []
        val_epoch_losses: list[float] = []
        lr_history: list[float] = []

        # Early stopping state
        best_val_loss = float("inf")
        best_epoch = 0
        best_state_dict = None
        patience_counter = 0

        # Where a stop landed: the 0-based index of the batch that never ran,
        # and which loop it was in. None until someone presses Stop.
        stopped_at: dict[str, Any] | None = None
        global_batch = 0
        # Set when ``max_steps`` ends the run. Distinct from ``stopped_at``:
        # that means "a human pressed Stop and these results are partial",
        # this means "the configured budget was spent and the run is done".
        step_budget_reached = False

        # ``epoch`` is the ABSOLUTE epoch index for the whole training run, not
        # an offset into this call: a resume at start_epoch=2 with epochs=4 runs
        # epochs 2 and 3, and every epoch number it reports (logs, progress
        # events, best_epoch) is that absolute number.
        for epoch in range(start_epoch, epochs):
            # ── Training phase ──
            model.train()
            running_loss = 0.0
            batch_count = 0

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

                optimizer.zero_grad()
                outputs = model(data)

                if targets is not None:
                    loss = loss_fn(outputs, targets)
                else:
                    loss = loss_fn(outputs)

                loss.backward()

                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                optimizer.step()
                batch_loss = loss.item()
                running_loss += batch_loss
                batch_count += 1
                global_batch += 1

                if (batch_metrics and context is not None
                        and global_batch % log_interval == 0):
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
                    context.log_metric("train_loss_batch", batch_loss,
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
                if max_steps and global_batch >= max_steps:
                    step_budget_reached = True
                    break

            if stopped_at is not None:
                break

            avg_train_loss = running_loss / max(batch_count, 1)
            epoch_losses.append(avg_train_loss)

            current_lr = optimizer.param_groups[0].get("lr", 0)
            lr_history.append(current_lr)

            # ── Validation phase ──
            avg_val_loss = None
            if val_dataloader is not None:
                model.eval()
                val_running_loss = 0.0
                val_batch_count = 0

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

                        outputs = model(data)
                        if targets is not None:
                            loss = loss_fn(outputs, targets)
                        else:
                            loss = loss_fn(outputs)
                        val_running_loss += loss.item()
                        val_batch_count += 1
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
                monitor_loss = avg_val_loss if avg_val_loss is not None else avg_train_loss
                if monitor_loss < best_val_loss:
                    best_val_loss = monitor_loss
                    best_epoch = epoch + 1
                    best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    stopped_early = True

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
            if context is not None:
                context.log_metric("train_loss", avg_train_loss, epoch + 1)
                if avg_val_loss is not None:
                    context.log_metric("val_loss", avg_val_loss, epoch + 1)
                context.log_metric("lr", current_lr, epoch + 1)
                if patience > 0:
                    context.log_metric("patience_counter", patience_counter,
                                       epoch + 1)
                    context.log_metric("best_epoch", best_epoch, epoch + 1)

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
            )

        # Epoch numbers are absolute; counts are for this call only.
        # ``best_epoch`` keeps its pre-#118 value when start_epoch is 0.
        metrics = {
            "final_train_loss": epoch_losses[-1] if epoch_losses else 0.0,
            "final_val_loss": val_epoch_losses[-1] if val_epoch_losses else None,
            "best_epoch": best_epoch if patience > 0 else completed_epochs,
            "total_epochs_run": len(epoch_losses),
            "start_epoch": start_epoch,
            "last_epoch": completed_epochs,
            "stopped_early": best_state_dict is not None and patience_counter >= patience,
            "lr_history": lr_history,
            "interrupted": stopped_at is not None,
            # How much training actually happened, and whether the budget is
            # why it ended. ``total_steps`` is reported unconditionally
            # because "how many optimizer steps was that?" is the question
            # ``max_steps`` makes people ask.
            "total_steps": global_batch,
            "stopped_at_max_steps": step_budget_reached,
        }
        if stopped_at is not None:
            metrics["interrupt_checkpoint"] = checkpoint_path

        result: dict[str, Any] = {
            "model": model,
            "losses": losses_tensor,
            "val_losses": val_losses_tensor,
            "metrics": metrics,
        }
        if stopped_at is not None:
            result.update(interrupted_result(
                epoch=completed_epochs, batch=stopped_at["batch"],
                checkpoint_path=checkpoint_path, phase=stopped_at["phase"],
            ))
        return result
