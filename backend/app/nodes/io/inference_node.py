import logging
from typing import Any

from ...core.device_utils import DEVICE_PARAM_DESCRIPTION, DEVICE_PARAM_OPTIONS
from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition

logger = logging.getLogger(__name__)


def _floating_param_dtype(model: Any) -> Any:
    """The dtype of the model's first floating parameter, or ``None``."""
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return None
    for param in parameters():
        if param.is_floating_point():
            return param.dtype
    return None


class InferenceNode(BaseNode):
    NODE_NAME = "Inference"
    CATEGORY = "IO"
    DESCRIPTION = "Forward pass on a model in eval mode, gradients off"
    DETAILS = (
        "eval() and the move to the selected device both apply to the model that "
        "was wired in, and stay in effect after the node returns. Floating input "
        "is cast to the model's dtype (float64 data works on every device); "
        "integer input is passed as is, for models that start with an Embedding."
    )

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
                options=list(DEVICE_PARAM_OPTIONS),
                description=DEVICE_PARAM_DESCRIPTION,
                advanced=True,
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

        # A float64 tensor (a CSV read by NumPy, TensorInput dtype=float64)
        # into a float32 model failed on every device but MPS with "mat1 and
        # mat2 must have the same dtype". Only floating input is cast, and
        # only to a floating parameter dtype: integer input is left alone so
        # an Embedding-style model still receives its indices.
        param_dtype = _floating_param_dtype(model)
        if (param_dtype is not None and isinstance(input_tensor, torch.Tensor)
                and input_tensor.is_floating_point() and input_tensor.dtype != param_dtype):
            input_tensor = input_tensor.to(param_dtype)

        model.eval()
        with torch.no_grad():
            try:
                output = model(input_tensor)
            except RuntimeError as exc:
                if (param_dtype is not None and isinstance(input_tensor, torch.Tensor)
                        and not input_tensor.is_floating_point()
                        and "type" in str(exc)):
                    raise ValueError(
                        f"Inference: the 'input' tensor is {input_tensor.dtype} "
                        f"but the model computes in {param_dtype}, and its "
                        f"first layer does not take integer input ({exc}). "
                        f"Convert the input to float before this node (for "
                        f"example TensorInput dtype=float32); only models that "
                        f"start with an Embedding take integer indices."
                    ) from exc
                raise

        logger.info("Inference complete — input %s → output %s", list(input_tensor.shape), list(output.shape))

        return {"output": output, "model": model}
