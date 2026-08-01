"""Save / load a full training checkpoint.

Checkpoint payload format
-------------------------
A checkpoint is a plain dict written with ``torch.save`` and read back with
``torch.load(..., weights_only=True)``. Every key is optional on read; the
loader must degrade gracefully when one is missing so that checkpoints written
by an older CodefyUI keep working.

===========================  ============================================
key                          contents
===========================  ============================================
``epoch``                    int, the epoch the checkpoint was taken at
``model_state_dict``         ``model.state_dict()``
``optimizer_state_dict``     ``optimizer.state_dict()``
``losses``                   optional tensor of per-epoch training losses
``scheduler_state_dict``     optional ``lr_scheduler.state_dict()`` (#118)
``scheduler_class``          optional class name guarding the restore (#118)
===========================  ============================================

**The format is append-only.** New keys are added with ``.get()`` on the read
side and never made mandatory, so a newer loader reads an older checkpoint and
an older loader ignores what it does not know. Everything stored must survive
``weights_only=True`` unpickling, i.e. tensors and plain Python containers --
no arbitrary objects.
"""

import logging
from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition

logger = logging.getLogger(__name__)


class CheckpointSaverNode(BaseNode):
    NODE_NAME = "CheckpointSaver"
    CATEGORY = "IO"
    DESCRIPTION = "Save a full training checkpoint (model + optimizer + LR schedule + epoch + loss) for resuming training later"

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
        import torch
        from pathlib import Path

        from ...config import settings

        model = inputs["model"]
        optimizer = inputs["optimizer"]
        losses = inputs.get("losses")
        lr_scheduler = inputs.get("lr_scheduler")
        path = params.get("path", "checkpoint.pt")
        epoch = params.get("epoch", 0)

        p = Path(path)
        if not p.is_absolute():
            p = settings.MODELS_DIR / p
        p = p.resolve()

        # Restrict writes to project data directory
        data_root = settings.MODELS_DIR.parent.resolve()
        if not p.is_relative_to(data_root):
            raise ValueError("Output path must be within the project data directory")

        p.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if losses is not None:
            checkpoint["losses"] = losses
        if lr_scheduler is not None:
            # #118: without this the LR schedule restarts from scratch on
            # resume. The class name is stored alongside so the loader can
            # refuse to splice a StepLR state into a CosineAnnealingLR.
            #
            # The port is DataType.ANY, so anything at all can be wired into
            # it. Say what is wrong before writing the file rather than
            # crashing half way through serialization.
            if not hasattr(lr_scheduler, "state_dict"):
                raise ValueError(
                    f"The lr_scheduler input is a {type(lr_scheduler).__name__}, "
                    "which has no state_dict(); connect an LRScheduler node's "
                    "'scheduler' output (or a CheckpointLoader's) to this port."
                )
            checkpoint["scheduler_state_dict"] = lr_scheduler.state_dict()
            checkpoint["scheduler_class"] = type(lr_scheduler).__name__

        torch.save(checkpoint, str(p))
        logger.info(
            "Saved checkpoint to %s (epoch=%d, scheduler=%s)",
            p, epoch, checkpoint.get("scheduler_class", "none"),
        )

        return {"path": str(p), "model": model}


class CheckpointLoaderNode(BaseNode):
    NODE_NAME = "CheckpointLoader"
    CATEGORY = "IO"
    DESCRIPTION = "Load a training checkpoint to resume training (restores model + optimizer + LR schedule + epoch)"

    # The cache key hashes `path`, never the checkpoint behind it. Saving a
    # newer checkpoint to the same path between runs is the whole point of
    # resumable training, so a cache hit would restore a stale epoch.
    cacheable = False

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
        from pathlib import Path

        from ...config import settings

        from ...core.device_utils import is_mps_device, resolve_node_device, to_device

        model = inputs["model"]
        optimizer = inputs["optimizer"]
        lr_scheduler = inputs.get("lr_scheduler")
        path = params.get("path", "checkpoint.pt")
        device = resolve_node_device(params.get("device"), context)

        p = Path(path)
        if not p.is_absolute():
            p = settings.MODELS_DIR / p
        p = p.resolve()

        # Restrict reads to project data directory
        data_root = settings.MODELS_DIR.parent.resolve()
        if not p.is_relative_to(data_root):
            raise ValueError("Checkpoint file path must be within the project data directory")

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
        }
