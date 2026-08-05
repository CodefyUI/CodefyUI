import logging
from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition

logger = logging.getLogger(__name__)


class ModelSaverNode(BaseNode):
    NODE_NAME = "ModelSaver"
    CATEGORY = "IO"
    DESCRIPTION = "Save model weights (state_dict) to a .pt/.pth/.safetensors file"

    # The write to disk IS this node's output. A cache hit returns the
    # recorded {"path": ..., "model": ...} without calling execute() again,
    # which is correct for a pure node and wrong here: if the user deleted
    # the file (or it never ran with Rec off), a cache hit would hand back a
    # path that no longer points at anything (#143).
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL, description="Trained model to save"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="path", data_type=DataType.STRING, description="Path to the saved file"),
            PortDefinition(name="model", data_type=DataType.MODEL, description="Pass-through model (for chaining)"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="path",
                param_type=ParamType.STRING,
                default="model_weights.pt",
                description="Output file path (.pt, .pth, or .safetensors)",
            ),
            ParamDefinition(
                name="save_mode",
                param_type=ParamType.SELECT,
                default="state_dict",
                description="Save mode: state_dict (recommended) or full model",
                options=["state_dict", "full_model"],
            ),
            ParamDefinition(
                name="format",
                param_type=ParamType.SELECT,
                default="pytorch",
                description="File format: pytorch (.pt/.pth) or safetensors (.safetensors)",
                options=["pytorch", "safetensors"],
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch
        from pathlib import Path

        from ...config import settings

        model = inputs["model"]
        path = params.get("path", "model_weights.pt")
        save_mode = params.get("save_mode", "state_dict")
        fmt = params.get("format", "pytorch")

        p = Path(path)
        if not p.is_absolute():
            p = settings.MODELS_DIR / p
        p = p.resolve()

        # Restrict writes to project data directory
        data_root = settings.MODELS_DIR.parent.resolve()
        if not p.is_relative_to(data_root):
            raise ValueError("Output path must be within the project data directory")

        if fmt == "safetensors":
            if save_mode == "full_model":
                raise ValueError("safetensors format only supports state_dict mode, not full_model")
            if p.suffix not in (".safetensors",):
                p = p.with_suffix(".safetensors")

        p.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "safetensors":
            from safetensors.torch import save_file
            save_file(model.state_dict(), str(p))
            param_count = sum(p_.numel() for p_ in model.parameters())
            logger.info("Saved safetensors to %s (%s parameters)", p, f"{param_count:,}")
        elif save_mode == "state_dict":
            torch.save(model.state_dict(), str(p))
            param_count = sum(p_.numel() for p_ in model.parameters())
            logger.info("Saved state_dict to %s (%s parameters)", p, f"{param_count:,}")
        else:
            torch.save(model, str(p))
            logger.info("Saved full model to %s", p)

        return {"path": str(p), "model": model}
