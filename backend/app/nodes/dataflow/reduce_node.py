from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class ReduceNode(BaseNode):
    NODE_NAME = "Reduce"
    CATEGORY = "Data Flow"
    DESCRIPTION = "Aggregate a list to one value: sum, mean, min, max"
    DETAILS = (
        "operation covers sum, mean, min, max, concat, stack, first and last; dim "
        "applies to concat and stack only. Everything except first and last needs "
        "numeric or tensor items, and an empty list raises."
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
        try:
            dim = int(params.get("dim", 0))
        except (TypeError, ValueError):
            raise ValueError(
                f"Reduce: `dim` must be a whole number, got {params.get('dim')!r}."
            ) from None
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

        def _combine(fn: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                return fn(tensors, *args, **kwargs)
            except (RuntimeError, IndexError) as exc:
                shapes = [list(t.shape) for t in tensors]
                raise ValueError(
                    f"Reduce '{op}': the items cannot be combined (shapes "
                    f"{shapes}, dim={dim}). sum, mean, min, max and stack need "
                    "every item to have the same shape; concat needs them to "
                    f"match except along dim. ({exc})"
                ) from exc

        if op == "sum":
            result = _combine(torch.stack).sum(dim=0)
        elif op == "mean":
            stacked = _combine(torch.stack)
            # torch.mean is defined for floating types only; an integer
            # tensor (e.g. class labels) raises "could not infer output
            # dtype". The mean of integers is a float, as with Python ints.
            if not (stacked.is_floating_point() or stacked.is_complex()):
                stacked = stacked.to(torch.float32)
            result = stacked.mean(dim=0)
        elif op == "min":
            result = _combine(torch.stack).min(dim=0).values
        elif op == "max":
            result = _combine(torch.stack).max(dim=0).values
        elif op == "concat":
            result = _combine(torch.cat, dim=dim)
        elif op == "stack":
            result = _combine(torch.stack, dim=dim)
        else:
            raise ValueError(f"Unknown reduce operation: {op}")

        return {"result": result, "count": count}
