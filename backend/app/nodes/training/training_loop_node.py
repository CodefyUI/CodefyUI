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
        The optimizer belongs to a *different* model (different parameter
        count or shapes), so its state is meaningless here. A fresh optimizer
        of the same class is constructed over ``model.parameters()`` and a
        warning is logged.

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
        "Optimizer %s tracks %d parameter tensors that do not match the model's %d "
        "in count or shape, so it was built for a different model: constructing a "
        "fresh one and discarding any optimizer state.",
        optimizer_cls.__name__, len(tracked), len(model_params),
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
    it is left alone with a warning.
    """
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
        # The first replayed step() precedes any optimizer.step() on this
        # scheduler instance, which torch warns about. Here it is expected.
        warnings.simplefilter("ignore", UserWarning)
        for _ in range(start_epoch):
            lr_scheduler.step()
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

    if getattr(lr_scheduler, "last_epoch", 0) > 0:
        logger.info("%s resumes at last_epoch=%d (restored state)", name, lr_scheduler.last_epoch)
    elif start_epoch > 0 and _fast_forward_scheduler(lr_scheduler, optimizer, start_epoch):
        logger.info("Fast-forwarded %s to last_epoch=%d", name, lr_scheduler.last_epoch)

    return lr_scheduler


class TrainingLoopNode(BaseNode):
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

        # ``epoch`` is the ABSOLUTE epoch index for the whole training run, not
        # an offset into this call: a resume at start_epoch=2 with epochs=4 runs
        # epochs 2 and 3, and every epoch number it reports (logs, progress
        # events, best_epoch) is that absolute number.
        for epoch in range(start_epoch, epochs):
            # ── Training phase ──
            model.train()
            running_loss = 0.0
            batch_count = 0

            for batch_data in dataloader:
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
                running_loss += loss.item()
                batch_count += 1

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
                    for batch_data in val_dataloader:
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

        # Restore best model if early stopping was used and found a best
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
            logger.info("Restored best model from epoch %d", best_epoch)

        losses_tensor = torch.tensor(epoch_losses, dtype=torch.float32)
        val_losses_tensor = torch.tensor(val_epoch_losses, dtype=torch.float32) if val_epoch_losses else torch.tensor([], dtype=torch.float32)

        # Epoch numbers are absolute; counts are for this call only.
        # ``best_epoch`` keeps its pre-#118 value when start_epoch is 0.
        metrics = {
            "final_train_loss": epoch_losses[-1] if epoch_losses else 0.0,
            "final_val_loss": val_epoch_losses[-1] if val_epoch_losses else None,
            "best_epoch": best_epoch if patience > 0 else start_epoch + len(epoch_losses),
            "total_epochs_run": len(epoch_losses),
            "start_epoch": start_epoch,
            "last_epoch": start_epoch + len(epoch_losses),
            "stopped_early": best_state_dict is not None and patience_counter >= patience,
            "lr_history": lr_history,
        }

        return {
            "model": model,
            "losses": losses_tensor,
            "val_losses": val_losses_tensor,
            "metrics": metrics,
        }
