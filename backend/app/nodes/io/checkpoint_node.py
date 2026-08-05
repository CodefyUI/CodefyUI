"""Save / load a full training checkpoint.

The payload format, the path rules and the writer itself live in
``core.checkpoints`` since #122 -- the interrupt path of every long-running
training node writes the same files, and one format needs one definition.
Read that module's docstring for the key table; these two nodes are the
graph-facing wrapper around it.
"""

import logging
from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition

logger = logging.getLogger(__name__)


class CheckpointSaverNode(BaseNode):
    NODE_NAME = "CheckpointSaver"
    CATEGORY = "IO"
    DESCRIPTION = "Save a full training checkpoint (model + optimizer + LR schedule + epoch + loss) for resuming training later"

    # The write to disk IS this node's output. A cache hit returns the
    # recorded {"path": ..., "model": ...} without calling execute() again,
    # which is correct for a pure node and wrong here: if the user deleted
    # the checkpoint (or it never ran with Rec off), a cache hit would hand
    # back a path that no longer points at anything (#143).
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL, description="Trained model"),
            PortDefinition(name="optimizer", data_type=DataType.OPTIMIZER, description="Optimizer state"),
            PortDefinition(name="losses", data_type=DataType.TENSOR, description="Loss history", optional=True),
            PortDefinition(
                name="lr_scheduler",
                data_type=DataType.ANY,
                description="LR scheduler whose position in the schedule to store",
                optional=True,
            ),
            PortDefinition(
                name="grad_scaler_state",
                data_type=DataType.ANY,
                description=(
                    "fp16 loss-scale state to store. Wire "
                    "TrainingLoop.grad_scaler_state here; leave unconnected "
                    "for fp32 and bf16 runs"
                ),
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="path", data_type=DataType.STRING, description="Path to the saved checkpoint"),
            PortDefinition(name="model", data_type=DataType.MODEL, description="Pass-through model"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="path",
                param_type=ParamType.STRING,
                default="checkpoint.pt",
                description="Output checkpoint file path",
            ),
            ParamDefinition(
                name="epoch",
                param_type=ParamType.INT,
                default=0,
                description="Current epoch number to store in checkpoint",
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        from ...core.checkpoints import write_checkpoint

        model = inputs["model"]
        target = write_checkpoint(
            params.get("path", "checkpoint.pt"),
            model,
            inputs["optimizer"],
            epoch=params.get("epoch", 0),
            losses=inputs.get("losses"),
            lr_scheduler=inputs.get("lr_scheduler"),
            scaler_state=inputs.get("grad_scaler_state"),
        )
        return {"path": str(target), "model": model}


class CheckpointLoaderNode(BaseNode):
    NODE_NAME = "CheckpointLoader"
    CATEGORY = "IO"
    DESCRIPTION = "Load a training checkpoint to resume training (restores model + optimizer + LR schedule + epoch)"

    # #144: cacheable again -- cache_fingerprint() below folds the resolved
    # checkpoint file's (size, mtime, and for small files a content hash)
    # into the cache key. Saving a newer checkpoint to the same path
    # between runs is the whole point of resumable training; the
    # fingerprint changes with the resave, so the cache correctly misses
    # instead of restoring a stale epoch.
    cacheable = True

    @classmethod
    def cache_fingerprint(cls, params: dict[str, Any]) -> Any:
        from ...core.cache_fingerprint import path_fingerprint
        from ...core.checkpoints import resolve_checkpoint_path

        path = str(params.get("path", "") or "")
        if not path:
            return None
        try:
            resolved = resolve_checkpoint_path(path)
        except Exception:
            return None
        return path_fingerprint(resolved)

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL, description="Model architecture to load weights into"),
            PortDefinition(name="optimizer", data_type=DataType.OPTIMIZER, description="Optimizer to restore state into"),
            PortDefinition(
                name="lr_scheduler",
                data_type=DataType.ANY,
                description="LR scheduler to restore its position in the schedule into",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL, description="Model with restored weights"),
            PortDefinition(name="optimizer", data_type=DataType.OPTIMIZER, description="Optimizer with restored state"),
            PortDefinition(name="epoch", data_type=DataType.SCALAR, description="Epoch number from checkpoint (wire to TrainingLoop.start_epoch)"),
            PortDefinition(name="losses", data_type=DataType.TENSOR, description="Loss history from checkpoint"),
            PortDefinition(name="lr_scheduler", data_type=DataType.ANY, description="LR scheduler with restored state (None if none was wired in)"),
            PortDefinition(
                name="grad_scaler_state",
                data_type=DataType.ANY,
                description=(
                    "fp16 loss-scale state from the checkpoint, or None. "
                    "Wire to TrainingLoop.grad_scaler_state to resume an "
                    "fp16 run at the scale it reached"
                ),
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="path",
                param_type=ParamType.MODEL_FILE,
                default="",
                description="Path to the checkpoint file",
            ),
            ParamDefinition(
                name="device",
                param_type=ParamType.SELECT,
                default="auto",
                description="Device to load onto ('auto' follows the global device)",
                options=["auto", "cpu", "cuda", "mps"],
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any], *, context: Any = None) -> dict[str, Any]:
        import torch

        from ...core.checkpoints import resolve_checkpoint_path
        from ...core.device_utils import is_mps_device, resolve_node_device, to_device

        model = inputs["model"]
        optimizer = inputs["optimizer"]
        lr_scheduler = inputs.get("lr_scheduler")
        path = params.get("path", "checkpoint.pt")
        device = resolve_node_device(params.get("device"), context)

        # Same rules as the saver: relative to MODELS_DIR, never outside the
        # project data directory.
        try:
            p = resolve_checkpoint_path(path)
        except ValueError:
            raise ValueError(
                "Checkpoint file path must be within the project data directory"
            ) from None

        if not p.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {p}")

        # MPS can't receive float64 via map_location, so stage doubles on CPU
        # and let to_device downcast them on the way to the device.
        load_device = "cpu" if is_mps_device(device) else device
        checkpoint = torch.load(str(p), map_location=load_device, weights_only=True)

        model.load_state_dict(checkpoint["model_state_dict"])
        model = to_device(model, device)

        # Re-bind optimizer to device-mapped parameters before restoring state
        for param_group in optimizer.param_groups:
            param_group["params"] = list(model.parameters())

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Move optimizer state tensors (momentum buffers, etc.) to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = to_device(v, device)

        # LR schedule (#118). Restored AFTER the optimizer on purpose: the
        # optimizer state dict carries the learning rate that was live when the
        # checkpoint was taken, and ``LRScheduler.load_state_dict`` does not
        # touch ``param_groups``, so this order leaves both consistent.
        #
        # Every branch below is a no-op-with-a-log rather than an error, so a
        # checkpoint written before this key existed still loads (and so does a
        # newer checkpoint read by a graph that has no scheduler wired in).
        scheduler_state = checkpoint.get("scheduler_state_dict")
        if lr_scheduler is not None and scheduler_state is not None:
            saved_class = checkpoint.get("scheduler_class")
            live_class = type(lr_scheduler).__name__
            if saved_class and saved_class != live_class:
                logger.warning(
                    "Checkpoint holds %s state but a %s is wired in; leaving the "
                    "scheduler untouched rather than restoring incompatible state.",
                    saved_class, live_class,
                )
            else:
                lr_scheduler.load_state_dict(scheduler_state)
                logger.info(
                    "Restored %s at last_epoch=%s",
                    live_class, getattr(lr_scheduler, "last_epoch", "?"),
                )
        elif lr_scheduler is not None:
            logger.info(
                "Checkpoint %s holds no scheduler state (written before it was "
                "recorded); the LR schedule will be fast-forwarded from start_epoch "
                "instead.", p,
            )
        elif scheduler_state is not None:
            logger.info(
                "Checkpoint %s holds scheduler state but no scheduler is wired into "
                "this loader; it will be ignored.", p,
            )

        epoch = checkpoint.get("epoch", 0)
        losses = checkpoint.get("losses", torch.tensor([]))

        logger.info("Loaded checkpoint from %s (epoch=%d)", p, epoch)

        return {
            "model": model,
            "optimizer": optimizer,
            "epoch": epoch,
            "losses": losses,
            "lr_scheduler": lr_scheduler,
            # Absent from every checkpoint written before #135 and from
            # every fp32/bf16 run since, so None is the ordinary answer
            # rather than a failure -- ``TrainingLoop`` treats it as "start
            # from a fresh loss scale".
            "grad_scaler_state": checkpoint.get("scaler_state_dict"),
        }
