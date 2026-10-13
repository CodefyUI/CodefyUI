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
                description="Dimension(s) to reduce as comma-separated ints (e.g. '-1' or '2,3'); blank averages every element",
            ),
            ParamDefinition(name="keepdim", param_type=ParamType.BOOL, default=False, description="Whether to keep the reduced dimension"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch

        from ...core.param_values import bool_param, parse_int_sequence

        tensor = inputs["tensor"]
        # "-1", -1, "2,3", "(2, 3)" and "2 3" all name dimensions; a blank
        # dim averages over every element.
        dims = parse_int_sequence(params.get("dim", "-1"), name="dim", node="Mean")
        keepdim = bool_param(params, "keepdim", node="Mean", default=False)

        # torch has no mean for integer or bool tensors; promote them to
        # float32. Floating input keeps its dtype.
        if isinstance(tensor, torch.Tensor) and not (
            tensor.is_floating_point() or tensor.is_complex()
        ):
            tensor = tensor.to(torch.float32)

        if dims is None:
            if keepdim:
                return {"tensor": tensor.mean(dim=tuple(range(tensor.dim())), keepdim=True)}
            return {"tensor": tensor.mean()}
        if len(dims) == 1:
            return {"tensor": tensor.mean(dim=dims[0], keepdim=keepdim)}
        return {"tensor": tensor.mean(dim=list(dims), keepdim=keepdim)}
