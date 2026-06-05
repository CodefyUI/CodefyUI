from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class TensorInputNode(BaseNode):
    NODE_NAME = "TensorInput"
    CATEGORY = "Data"
    DESCRIPTION = "Teaching entry point — inline tensor editor with explicit values, random, zeros, ones, or arange. Seed-reproducible."

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Output tensor"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="shape",
                param_type=ParamType.STRING,
                default="1,4,4",
                description="Tensor shape as comma-separated ints (e.g. '1,4,4')",
            ),
            ParamDefinition(
                name="dtype",
                param_type=ParamType.SELECT,
                default="float32",
                description="Data type",
                options=["float32", "float64", "int64", "int32", "bool"],
            ),
            ParamDefinition(
                name="value_mode",
                param_type=ParamType.SELECT,
                default="random",
                description="How to fill the tensor",
                options=["explicit", "random", "zeros", "ones", "arange"],
            ),
            ParamDefinition(
                name="values",
                param_type=ParamType.TENSOR_GRID,
                default=None,
                description="Nested list of values (used when value_mode=explicit)",
            ),
            ParamDefinition(
                name="seed",
                param_type=ParamType.INT,
                default=0,
                description="Seed for reproducible random (used when value_mode=random)",
            ),
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

        shape_str = params.get("shape", "1,4,4")
        shape = tuple(int(s.strip()) for s in shape_str.split(",") if s.strip())
        if not shape:
            raise ValueError("shape must have at least one dimension")

        dtype_name = params.get("dtype", "float32")
        dtype = {
            "float32": torch.float32,
            "float64": torch.float64,
            "int64": torch.int64,
            "int32": torch.int32,
            "bool": torch.bool,
        }.get(dtype_name)
        if dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype_name}")

        mode = params.get("value_mode", "random")
        seed = int(params.get("seed", 0) or 0)

        numel = 1
        for d in shape:
            numel *= d

        if mode == "explicit":
            values = params.get("values")
            if values is None:
                raise ValueError("value_mode=explicit requires 'values' to be set")
            flat = _flatten(values)
            if len(flat) != numel:
                raise ValueError(
                    f"values has {len(flat)} elements but shape {list(shape)} requires {numel}"
                )
            if dtype == torch.bool:
                flat = [bool(v) for v in flat]
            tensor = torch.tensor(flat, dtype=dtype).reshape(shape)
        elif mode == "random":
            generator = torch.Generator().manual_seed(seed)
            if dtype.is_floating_point:
                tensor = torch.randn(*shape, generator=generator, dtype=dtype)
            elif dtype == torch.bool:
                tensor = torch.randint(0, 2, shape, generator=generator).bool()
            else:
                tensor = torch.randint(0, 10, shape, generator=generator, dtype=dtype)
        elif mode == "zeros":
            tensor = torch.zeros(*shape, dtype=dtype)
        elif mode == "ones":
            tensor = torch.ones(*shape, dtype=dtype)
        elif mode == "arange":
            tensor = torch.arange(numel, dtype=dtype).reshape(shape)
        else:
            raise ValueError(f"Unsupported value_mode: {mode}")

        # Move to the run's global device. Seeded RNG above runs on the CPU
        # generator for reproducibility, so we generate on CPU then move —
        # this also downcasts float64→float32 for MPS via to_device.
        tensor = to_device(tensor, context_device(context))

        # When the graph runs in backward mode, opt floating tensors into
        # autograd so .grad becomes available after the post-forward
        # backward pass executed by graph_engine.
        if (
            context is not None
            and getattr(context, "backward_mode", False)
            and tensor.is_floating_point()
        ):
            tensor.requires_grad_(True)

        return {"tensor": tensor}


def _flatten(x: Any) -> list:
    if isinstance(x, (list, tuple)):
        out: list = []
        for v in x:
            out.extend(_flatten(v))
        return out
    return [x]
