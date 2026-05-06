"""AttentionMaskNode — generate attention masks (causal or padding).

Self-attention defaults to seeing every position. Two common scenarios need
to block some of those connections:

1. **Causal** (decoder-style, GPT-like): position i can only attend to
   positions ≤ i. The mask is the strictly upper-triangular part of the
   square matrix — True everywhere a query is forbidden from peeking.
2. **Padding**: when sequences in a batch were padded to a common length,
   the padding columns must be blocked so attention doesn't bleed into
   meaningless slots.

Convention: the output is a boolean tensor of shape ``[seq, seq]`` where
``True`` means *blocked*. Downstream attention nodes consume it via
``scores.masked_fill(mask, -inf)``.
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


class AttentionMaskNode(BaseNode):
    NODE_NAME = "AttentionMask"
    CATEGORY = "LLM"
    DESCRIPTION = (
        "Generate a boolean attention mask (True = blocked). `causal` blocks "
        "future positions for GPT-style decoders; `padding` blocks the columns "
        "matching the pad token so attention doesn't bleed into padding slots."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tensor",
                data_type=DataType.TENSOR,
                description="Sequence tensor — seq length is taken from dim 0. Optional if `tokens` is connected.",
                optional=True,
            ),
            PortDefinition(
                name="tokens",
                data_type=DataType.LIST,
                description="Token list — seq length is len(tokens). Required for padding mode.",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="mask",
                data_type=DataType.TENSOR,
                description="Boolean mask of shape [seq, seq]. True at [i, j] means query i may NOT attend to key j.",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="mode",
                param_type=ParamType.SELECT,
                default="causal",
                options=["causal", "padding"],
                description=(
                    "causal: block strictly future positions (decoder-style). "
                    "padding: block columns matching `pad_token`."
                ),
            ),
            ParamDefinition(
                name="pad_token",
                param_type=ParamType.STRING,
                default="<pad>",
                description="Token string treated as padding. Used in padding mode only.",
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
        mode = str(params.get("mode", "causal"))
        pad_token = str(params.get("pad_token", "<pad>"))

        tensor = inputs.get("tensor")
        tokens = inputs.get("tokens")

        if tensor is None and tokens is None:
            raise ValueError("AttentionMask requires either `tensor` or `tokens` input.")

        if mode == "causal":
            seq = self._infer_seq_len(tensor, tokens)
            # Upper-triangular True (diagonal=1 means strictly above the main diagonal)
            # so each position can still attend to itself.
            mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
            return {"mask": mask}

        if mode == "padding":
            if tokens is None:
                raise ValueError(
                    "AttentionMask `padding` mode requires `tokens` input to know which positions are padding."
                )
            tok_list = [str(t) for t in tokens]
            seq = len(tok_list)
            is_pad = torch.tensor([t == pad_token for t in tok_list], dtype=torch.bool)
            # Block whole columns where the key is padding — every query is
            # forbidden from attending to a pad slot.
            mask = is_pad.unsqueeze(0).expand(seq, seq).clone()
            return {"mask": mask}

        raise ValueError(f"Unknown AttentionMask mode: {mode!r}")

    @staticmethod
    def _infer_seq_len(tensor: Any, tokens: Any) -> int:
        if tokens is not None:
            return len(list(tokens))
        if isinstance(tensor, torch.Tensor):
            return int(tensor.shape[0])
        # Best-effort fallback for array-like input.
        t = torch.as_tensor(tensor)
        return int(t.shape[0])
