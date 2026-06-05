import logging
from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition

logger = logging.getLogger(__name__)


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

        epochs = params.get("epochs", 5)
        device = resolve_node_device(params.get("device"), context)
        patience = params.get("early_stopping_patience", 0)
        grad_clip = params.get("grad_clip_norm", 0.0)

        model = to_device(model, device)
        loss_fn = to_device(loss_fn, device)

        # Recreate the optimizer so its param list, momentum buffers and state are
        # freshly bound to the moved model parameters. The Optimizer node runs before
        # TrainingLoop with the model still on CPU, so any optimizer created there
        # holds stale references / state. Preserving the original class + defaults
        # keeps the user's choices (Adam/SGD/..., lr, weight_decay, etc.) intact.
        #
        # Filter ``defaults`` to keys the constructor actually accepts. PyTorch
        # 2.11 added ``decoupled_weight_decay`` to Adam's defaults, but AdamW
        # (an Adam subclass) re-routes that internally and rejects it as a
        # public kwarg — a naive ``**defaults`` round-trip raises
        # ``AdamW.__init__() got an unexpected keyword argument 'decoupled_weight_decay'``.
        # Filtering by signature keeps the rebind robust against future
        # upstream churn.
        import inspect
        optimizer_cls = type(optimizer)
        accepted = set(inspect.signature(optimizer_cls.__init__).parameters)
        optimizer_kwargs = {
            k: v for k, v in optimizer.defaults.items() if k in accepted
        }
        optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)

        # If an LR scheduler was passed in, rebind it to the new optimizer using
        # the same class and non-optimizer init arguments.
        if lr_scheduler is not None:
            scheduler_cls = type(lr_scheduler)
            # Only keep primitive/scalar state to re-init the scheduler cleanly
            scheduler_init_kwargs: dict[str, Any] = {}
            for key, value in lr_scheduler.state_dict().items():
                if key in ("base_lrs", "_last_lr", "last_epoch", "_step_count", "verbose", "_get_lr_called_within_step"):
                    continue
                if isinstance(value, (int, float, str, bool, list, tuple)):
                    scheduler_init_kwargs[key] = value
            try:
                lr_scheduler = scheduler_cls(optimizer, **scheduler_init_kwargs)
            except TypeError:
                # Couldn't reconstruct — fall back to manual optimizer attachment.
                lr_scheduler.optimizer = optimizer

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

        for epoch in range(epochs):
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

        metrics = {
            "final_train_loss": epoch_losses[-1] if epoch_losses else 0.0,
            "final_val_loss": val_epoch_losses[-1] if val_epoch_losses else None,
            "best_epoch": best_epoch if patience > 0 else len(epoch_losses),
            "total_epochs_run": len(epoch_losses),
            "stopped_early": best_state_dict is not None and patience_counter >= patience,
            "lr_history": lr_history,
        }

        return {
            "model": model,
            "losses": losses_tensor,
            "val_losses": val_losses_tensor,
            "metrics": metrics,
        }
