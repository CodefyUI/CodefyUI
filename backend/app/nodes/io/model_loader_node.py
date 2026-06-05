import logging
from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition

logger = logging.getLogger(__name__)


class ModelLoaderNode(BaseNode):
    NODE_NAME = "ModelLoader"
    CATEGORY = "IO"
    DESCRIPTION = "Load model weights from a .pt/.pth file into a model, or load a full saved model"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="model",
                data_type=DataType.MODEL,
                description="Model architecture to load weights into (required for state_dict mode)",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL, description="Model with loaded weights"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="path",
                param_type=ParamType.MODEL_FILE,
                default="",
                description="Path to the weights file (.pt, .pth, .safetensors)",
            ),
            ParamDefinition(
                name="load_mode",
                param_type=ParamType.SELECT,
                default="state_dict",
                description="Load mode: state_dict (requires model input) or full_model",
                options=["state_dict", "full_model"],
            ),
            ParamDefinition(
                name="device",
                param_type=ParamType.SELECT,
                default="auto",
                description="Device to load weights onto ('auto' follows the global device)",
                options=["auto", "cpu", "cuda", "mps"],
            ),
            ParamDefinition(
                name="strict",
                param_type=ParamType.BOOL,
                default=True,
                description="Whether to strictly enforce that the keys in state_dict match (state_dict mode only)",
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any], *, context: Any = None) -> dict[str, Any]:
        import torch
        from pathlib import Path

        from ...config import settings

        from ...core.device_utils import is_mps_device, resolve_node_device, to_device

        path = params.get("path", "model_weights.pt")
        load_mode = params.get("load_mode", "state_dict")
        device = resolve_node_device(params.get("device"), context)
        strict = params.get("strict", True)

        # MPS can't receive float64 via map_location, so stage doubles on CPU
        # and let to_device downcast them on the way to the device.
        load_device = "cpu" if is_mps_device(device) else device

        p = Path(path)
        if not p.is_absolute():
            p = settings.MODELS_DIR / p
        p = p.resolve()

        # Restrict reads to project data directory
        data_root = settings.MODELS_DIR.parent.resolve()
        if not p.is_relative_to(data_root):
            raise ValueError("Weights file path must be within the project data directory")

        if not p.exists():
            raise FileNotFoundError(f"Weights file not found: {p}")

        is_safetensors = p.suffix == ".safetensors"

        if load_mode == "state_dict":
            model = inputs.get("model")
            if model is None:
                raise ValueError(
                    "state_dict mode requires a model input. "
                    "Connect a SequentialModel or other model node, or use full_model mode."
                )
            if is_safetensors:
                from safetensors.torch import load_file
                state_dict = load_file(str(p), device=load_device)
            else:
                state_dict = torch.load(str(p), map_location=load_device, weights_only=True)
            model.load_state_dict(state_dict, strict=strict)
            model = to_device(model, device)
            param_count = sum(p_.numel() for p_ in model.parameters())
            logger.info("Loaded state_dict from %s (%s parameters, strict=%s)", p, f"{param_count:,}", strict)
        else:
            if is_safetensors:
                raise ValueError("safetensors format only supports state_dict mode, not full_model")
            logger.warning(
                "full_model mode uses weights_only=True for safety. "
                "If loading fails, re-save the model as state_dict."
            )
            model = torch.load(str(p), map_location=load_device, weights_only=True)
            model = to_device(model, device)
            logger.info("Loaded full model from %s", p)

        return {"model": model}
