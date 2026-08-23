from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class ReduceNode(BaseNode):
    NODE_NAME = "Reduce"
    CATEGORY = "Data Flow"
    DESCRIPTION = (
        "Aggregate a list of values into a single result. "
        "Supports sum, mean, min, max, concat (tensors), and first/last selection."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="items", data_type=DataType.LIST, description="List of items to aggregate"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="result", data_type=DataType.ANY, description="Aggregated result"),
            PortDefinition(name="count", data_type=DataType.SCALAR, description="Number of items aggregated"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="operation",
                param_type=ParamType.SELECT,
                default="mean",
                description="Aggregation operation",
                options=["sum", "mean", "min", "max", "concat", "stack", "first", "last"],
            ),
            ParamDefinition(
                name="dim",
                param_type=ParamType.INT,
                default=0,
                description="Dimension for concat/stack operations",
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch

        items = inputs.get("items", [])
        if not isinstance(items, (list, tuple)):
            raise ValueError(f"Reduce expects a list input, got {type(items).__name__}")
        if len(items) == 0:
            raise ValueError("Reduce received an empty list")

        op = params.get("operation", "mean")
        dim = params.get("dim", 0)
        count = float(len(items))

        if op == "first":
            return {"result": items[0], "count": count}
        if op == "last":
            return {"result": items[-1], "count": count}

        # For tensor operations, try to convert items to tensors.
        #
        # The scalars are built on the device the tensor items are already
        # on. The engine aligns what arrives on the wire, but it cannot see a
        # tensor this node makes itself -- so a list mixing `2.0` with real
        # tensors would reach `torch.stack` half on the accelerator and half
        # on the host, which is a raise on CUDA and a segfault on MPS.
        item_device = next(
            (item.device for item in items if isinstance(item, torch.Tensor)), None)
        tensors = []
        for item in items:
            if isinstance(item, torch.Tensor):
                tensors.append(item)
            elif isinstance(item, (int, float)):
                tensors.append(
                    torch.tensor(item, dtype=torch.float32, device=item_device))
            else:
                raise ValueError(
                    f"Reduce '{op}' requires numeric/tensor items, got {type(item).__name__}"
                )

        if op == "sum":
            result = torch.stack(tensors).sum(dim=0)
        elif op == "mean":
            result = torch.stack(tensors).mean(dim=0)
        elif op == "min":
            result = torch.stack(tensors).min(dim=0).values
        elif op == "max":
            result = torch.stack(tensors).max(dim=0).values
        elif op == "concat":
            result = torch.cat(tensors, dim=dim)
        elif op == "stack":
            result = torch.stack(tensors, dim=dim)
        else:
            raise ValueError(f"Unknown reduce operation: {op}")

        return {"result": result, "count": count}
