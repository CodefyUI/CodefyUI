"""BackwardOnce — explicit marker node for the gradient inspector (A3).

Insert between any forward tensor and a downstream sink to tell graph_engine
"call ``loss = input.sum(); loss.backward()`` here". The node itself is a
pass-through — it doesn't transform the tensor — it just lets the user mark
*where* the backward pass should kick off when ``BackwardMode`` is enabled.

Behaviour without ``BackwardMode``: pure pass-through with no side effects.
"""

from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, PortDefinition


class BackwardOnceNode(BaseNode):
    NODE_NAME = "BackwardOnce"
    CATEGORY = "Training"
    DESCRIPTION = (
        "Marks a tensor as the target of an autograd backward pass for the "
        "Backward inspector. Runs only when Backward mode is enabled in the toolbar. "
        "Backward target: $\\mathcal{L} = \\sum(\\text{input})$ (synthetic scalar)."
    )

    cacheable = False  # Forward outputs depend on requires_grad state.

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Tensor to differentiate"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Pass-through of the input tensor"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return []

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        tensor = inputs["tensor"]
        return {"tensor": tensor}
