import logging
from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition

logger = logging.getLogger(__name__)


class CheckpointSaverNode(BaseNode):
    NODE_NAME = "CheckpointSaver"
    CATEGORY = "IO"
    DESCRIPTION = "Save a full training checkpoint (model + optimizer + epoch + loss) for resuming training later"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL, description="Trained model"),
            PortDefinition(name="optimizer", data_type=DataType.OPTIMIZER, description="Optimizer state"),
            PortDefinition(name="losses", data_type=DataType.TENSOR, description="Loss history", optional=True),
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

        torch.save(checkpoint, str(p))
        logger.info("Saved checkpoint to %s (epoch=%d)", p, epoch)

        return {"path": str(p), "model": model}


class CheckpointLoaderNode(BaseNode):
    NODE_NAME = "CheckpointLoader"
    CATEGORY = "IO"
    DESCRIPTION = "Load a training checkpoint to resume training (restores model + optimizer + epoch)"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL, description="Model architecture to load weights into"),
            PortDefinition(name="optimizer", data_type=DataType.OPTIMIZER, description="Optimizer to restore state into"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL, description="Model with restored weights"),
            PortDefinition(name="optimizer", data_type=DataType.OPTIMIZER, description="Optimizer with restored state"),
            PortDefinition(name="epoch", data_type=DataType.SCALAR, description="Epoch number from checkpoint"),
            PortDefinition(name="losses", data_type=DataType.TENSOR, description="Loss history from checkpoint"),
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

        epoch = checkpoint.get("epoch", 0)
        losses = checkpoint.get("losses", torch.tensor([]))

        logger.info("Loaded checkpoint from %s (epoch=%d)", p, epoch)

        return {
            "model": model,
            "optimizer": optimizer,
            "epoch": epoch,
            "losses": losses,
        }
