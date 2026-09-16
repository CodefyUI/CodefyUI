from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class MeanNode(BaseNode):
    NODE_NAME = "Mean"
    CATEGORY = "Tensor Operations"
    DESCRIPTION = "Average a tensor over one or more dimensions"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Input tensor"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Reduced tensor"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="dim",
                param_type=ParamType.STRING,
                default="-1",
                description="Dimension(s) to reduce as comma-separated ints (e.g. '-1' or '2,3')",
            ),
            ParamDefinition(name="keepdim", param_type=ParamType.BOOL, default=False, description="Whether to keep the reduced dimension"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        tensor = inputs["tensor"]
        dim_str = params.get("dim", "-1")
        dims = [int(d.strip()) for d in dim_str.split(",")]
        keepdim = params.get("keepdim", False)

        if len(dims) == 1:
            return {"tensor": tensor.mean(dim=dims[0], keepdim=keepdim)}
        return {"tensor": tensor.mean(dim=dims, keepdim=keepdim)}
