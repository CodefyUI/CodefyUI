from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition

MAX_CHUNKS = 32


def _resolve_chunks(params: dict[str, Any] | None) -> int:
    raw = (params or {}).get("chunks", 2)
    try:
        n = int(raw)
    except (TypeError, ValueError):
        n = 2
    return max(1, min(MAX_CHUNKS, n))


class SplitNode(BaseNode):
    NODE_NAME = "Split"
    CATEGORY = "Tensor Operations"
    DESCRIPTION = (
        "Split a tensor into N chunks along a dimension. The number of "
        "output ports (chunk_0, chunk_1, ...) follows the `chunks` param."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Input tensor to split"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        # Baseline schema for the palette template — matches the default
        # `chunks=2`. The live port count for a placed node comes from
        # `define_outputs_dynamic` below.
        return [
            PortDefinition(name="chunk_0", data_type=DataType.TENSOR, description="First chunk"),
            PortDefinition(name="chunk_1", data_type=DataType.TENSOR, description="Second chunk"),
        ]

    @classmethod
    def define_outputs_dynamic(
        cls,
        params: dict[str, Any] | None = None,
    ) -> list[PortDefinition]:
        n = _resolve_chunks(params)
        return [
            PortDefinition(
                name=f"chunk_{i}",
                data_type=DataType.TENSOR,
                description=f"Chunk {i} of {n}",
            )
            for i in range(n)
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="chunks",
                param_type=ParamType.INT,
                default=2,
                description=f"Number of chunks to split into (1..{MAX_CHUNKS}); also drives output port count",
                min_value=1,
                max_value=MAX_CHUNKS,
            ),
            ParamDefinition(name="dim", param_type=ParamType.INT, default=0, description="Dimension along which to split"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch

        tensor = inputs["tensor"]
        chunks = _resolve_chunks(params)
        dim = params.get("dim", 0)
        parts = torch.chunk(tensor, chunks, dim=dim)
        return {f"chunk_{i}": part for i, part in enumerate(parts)}
