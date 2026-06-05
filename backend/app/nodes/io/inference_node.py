import logging
from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition

logger = logging.getLogger(__name__)


class InferenceNode(BaseNode):
    NODE_NAME = "Inference"
    CATEGORY = "IO"
    DESCRIPTION = "Run inference (forward pass) on a trained model. Sets model to eval mode and disables gradients."

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL, description="Trained model"),
            PortDefinition(name="input", data_type=DataType.TENSOR, description="Input tensor"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="output", data_type=DataType.TENSOR, description="Model prediction"),
            PortDefinition(name="model", data_type=DataType.MODEL, description="Pass-through model"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="device",
                param_type=ParamType.SELECT,
                default="auto",
                description="Device to run inference on ('auto' follows the global device)",
                options=["auto", "cpu", "cuda", "mps"],
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any], *, context: Any = None) -> dict[str, Any]:
        import torch

        from ...core.device_utils import resolve_node_device, to_device

        model = inputs["model"]
        input_tensor = inputs["input"]
        device = resolve_node_device(params.get("device"), context)

        # to_device downcasts float64 → float32 when targeting MPS (MPS has no
        # float64), so a TensorInput(dtype=float64) feeding inference won't crash.
        model = to_device(model, device)
        input_tensor = to_device(input_tensor, device)

        model.eval()
        with torch.no_grad():
            output = model(input_tensor)

        logger.info("Inference complete — input %s → output %s", list(input_tensor.shape), list(output.shape))

        return {"output": output, "model": model}
