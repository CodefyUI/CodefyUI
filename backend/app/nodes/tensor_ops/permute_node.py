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
        dims_str = str(params.get("dims", "0,2,1"))
        # "1,0", "1, 0", "(1, 0)", "[1,0]" and "1 0" all mean the same order.
        parts = dims_str.strip().strip("()[]").replace(",", " ").split()
        try:
            dims = [int(d) for d in parts]
        except ValueError:
            raise ValueError(
                f"Permute: dims must be integers separated by commas, like '1,0'; "
                f"got {dims_str!r}."
            ) from None
        if len(dims) != tensor.dim():
            raise ValueError(
                f"Permute: the tensor has {tensor.dim()} dims {tuple(tensor.shape)} but "
                f"dims lists {len(dims)} ({dims_str!r}); give one entry per dim, "
                f"e.g. '1,0' to transpose a 2-D tensor."
            )
        return {"tensor": tensor.permute(*dims)}
