from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


def parse_shape(shape_str: Any) -> tuple[int, ...]:
    """``"1,3,224,224"`` as a tuple of positive ints, or a ValueError naming
    the ``shape`` param and the format it expects (#682)."""
    try:
        shape = tuple(int(s.strip()) for s in str(shape_str).split(","))
    except ValueError:
        shape = ()
    if not shape or any(dim <= 0 for dim in shape):
        raise ValueError(
            "shape must be comma-separated positive integers "
            f"(e.g. 1,3,224,224), got {shape_str!r}"
        )
    return shape


class TensorCreateNode(BaseNode):
    NODE_NAME = "TensorCreate"
    CATEGORY = "Tensor Operations"
    DESCRIPTION = "Tensor of a given shape: zeros, ones, random, constant"
    DETAILS = (
        "The `arange` fill uses only the first entry of `shape` and returns a 1-D "
        "float tensor."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Created tensor"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="shape",
                param_type=ParamType.STRING,
                default="1,3,224,224",
                description="Tensor shape as comma-separated ints (e.g. '1,3,224,224')",
            ),
            ParamDefinition(
                name="fill",
                param_type=ParamType.SELECT,
                default="zeros",
                description="Fill method",
                options=["zeros", "ones", "randn", "rand", "full", "arange"],
            ),
            ParamDefinition(name="value", param_type=ParamType.FLOAT, default=0.0, description="Fill value (for 'full' mode)"),
            ParamDefinition(name="requires_grad", param_type=ParamType.BOOL, default=False, description="Whether the tensor requires gradient"),
        ]

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        import torch

        from ...core.device_utils import context_device, to_device

        shape = parse_shape(params.get("shape", "1,3,224,224"))
        fill = params.get("fill", "zeros")
        value = params.get("value", 0.0)
        requires_grad = params.get("requires_grad", False)

        creators = {
            "zeros": lambda: torch.zeros(*shape),
            "ones": lambda: torch.ones(*shape),
            "randn": lambda: torch.randn(*shape),
            "rand": lambda: torch.rand(*shape),
            "full": lambda: torch.full(shape, value),
            "arange": lambda: torch.arange(shape[0]).float(),
        }

        create_fn = creators.get(fill)
        if create_fn is None:
            raise ValueError(f"Unsupported fill method: {fill}")

        # Build on CPU then move to the run's global device — keeps the leaf
        # tensor on-device so requires_grad_ marks an on-device autograd leaf.
        tensor = to_device(create_fn(), context_device(context))
        if requires_grad:
            tensor = tensor.requires_grad_(True)
        return {"tensor": tensor}
