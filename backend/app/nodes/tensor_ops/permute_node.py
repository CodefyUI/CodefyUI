from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class PermuteNode(BaseNode):
    NODE_NAME = "Permute"
    CATEGORY = "Tensor Operations"
    DESCRIPTION = "Reorder a tensor's dimensions"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Input tensor"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Permuted tensor"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="dims",
                param_type=ParamType.STRING,
                default="0,2,1",
                description="New dimension order as comma-separated ints (e.g. '0,2,1')",
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        tensor = inputs["tensor"]
        dims_str = params.get("dims", "0,2,1")
        dims = [int(d.strip()) for d in dims_str.split(",")]
        return {"tensor": tensor.permute(*dims)}
