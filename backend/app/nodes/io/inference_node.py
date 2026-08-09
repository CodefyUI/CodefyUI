import logging
from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition

logger = logging.getLogger(__name__)


class InferenceNode(BaseNode):
    NODE_NAME = "Inference"
    CATEGORY = "IO"
    DESCRIPTION = "Run inference (forward pass) on a trained model. Sets model to eval mode and disables gradients."

    # #254. Two reasons, either one sufficient. (1) ``model.eval()`` is a
    # permanent flip on a module this node was handed, and ``to_device``
    # moves it; on a cache hit neither happens while the node reports
    # success. (2) ``output`` is a function of the module's WEIGHTS, which
    # the cache key does not describe -- it keys on how the module was
    # BUILT. Measured with a training loop mutating the same module in the
    # same graph, three runs against one ExecutionCache: 1 / 0 / 0 real
    # execute() calls while the module's true output moved
    # 0.0937 -> -0.0309 -> 0.2178. Runs 2 and 3 served run 1's answer for a
    # network that no longer existed.
    #
    # It also hands the module onward on its ``model`` output, which is the
    # line drawn in ``test_cache_live_handle_nodes.py``: a node that passes
    # a live MODEL or OPTIMIZER on cannot be replayed. (``DecisionBoundary``
    # also reads a model but does not hand one out; it is protected by the
    # engine propagating its upstream's non-cacheability, and rendering a
    # plot is the kind of thing the cache exists for.)
    cacheable = False

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
