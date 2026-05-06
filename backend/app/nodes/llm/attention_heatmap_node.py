"""AttentionHeatmapNode — pure visualisation passthrough for attention weights.

Computation-wise this node does nothing: it forwards its ``weights`` input
to its ``weights`` output unchanged. Its real job is to give the frontend
a stable handle to anchor a heatmap viz on.

Why a separate node? Two reasons:

1. The Transformer/MultiHeadAttention node (production-style, large dims)
   already emits ``weights`` but has no inline viz. Rather than extend
   that node, students can drop an AttentionHeatmap downstream and get
   the heatmap for free.
2. Sometimes you want to compare the attention pattern at multiple points
   in a graph (before vs after a mask, this layer vs that layer). A
   passthrough viz lets you "tap" the wire without changing the data.

Optionally selects a single head when given ``[H, seq, seq]`` input via
``head_index``; ``-1`` keeps the full per-head tensor for grid display.
"""

from __future__ import annotations

from typing import Any

import torch

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)


class AttentionHeatmapNode(BaseNode):
    NODE_NAME = "AttentionHeatmap"
    CATEGORY = "LLM"
    DESCRIPTION = (
        "Pure visualisation node: passes attention weights through unchanged "
        "while exposing them to a heatmap viz. Use it to tap the `weights` "
        "output of any attention node (toy or production) without changing "
        "the downstream graph."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="weights",
                data_type=DataType.TENSOR,
                description="Attention weights — accepts [seq, seq], [H, seq, seq], or [B, H, seq, seq].",
            ),
            PortDefinition(
                name="labels",
                data_type=DataType.LIST,
                description="Optional token labels to annotate heatmap axes.",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="weights",
                data_type=DataType.TENSOR,
                description="Pass-through of the input weights (or a single head, if `head_index` is set).",
            ),
            PortDefinition(
                name="labels",
                data_type=DataType.LIST,
                description="Pass-through of the input labels (empty list when not provided).",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="head_index",
                param_type=ParamType.INT,
                default=-1,
                description=(
                    "For per-head weights ([H,seq,seq] or [B,H,seq,seq]), pick a single "
                    "head to display. -1 keeps all heads for a side-by-side grid."
                ),
            ),
            ParamDefinition(
                name="colormap",
                param_type=ParamType.SELECT,
                default="viridis",
                options=["viridis", "blues", "RdBu"],
                description="Colour map for the heatmap viz (frontend-only; back end ignores).",
            ),
        ]

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        progress_callback: Any | None = None,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        weights = inputs.get("weights")
        if weights is None:
            raise ValueError("AttentionHeatmap requires a `weights` input.")
        if not isinstance(weights, torch.Tensor):
            weights = torch.as_tensor(weights, dtype=torch.float32)

        head_index = int(params.get("head_index", -1))

        # Slice a single head when meaningful. We treat dim 0 as the head axis
        # for 3D input ([H, seq, seq]) and dim 1 for 4D input ([B, H, seq, seq]).
        if head_index >= 0:
            if weights.ndim == 3:
                if head_index >= weights.shape[0]:
                    raise ValueError(
                        f"head_index={head_index} out of range for {weights.shape[0]} heads."
                    )
                weights = weights[head_index]
            elif weights.ndim == 4:
                if head_index >= weights.shape[1]:
                    raise ValueError(
                        f"head_index={head_index} out of range for {weights.shape[1]} heads."
                    )
                weights = weights[:, head_index]
            # 2D input: head_index is meaningless, silently keep all of it.

        labels = list(inputs.get("labels") or [])
        return {"weights": weights, "labels": labels}
