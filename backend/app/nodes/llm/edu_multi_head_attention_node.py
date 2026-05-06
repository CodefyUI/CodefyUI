"""EduMultiHeadAttentionNode — toy multi-head attention with per-head heatmaps.

Each "head" projects the input into a smaller subspace, runs scaled
dot-product attention there, and the heads' outputs are concatenated and
mixed with a final linear layer:

    Q_h, K_h, V_h = W_q^h(x), W_k^h(x), W_v^h(x)        # for h = 1..H
    A_h = softmax(Q_h K_h^T / sqrt(d_h))                 # [seq, seq] per head
    O_h = A_h V_h
    O   = W_o( concat(O_1, …, O_H) )

This educational variant exposes ``weights`` as ``[H, seq, seq]`` (or
``[batch, H, seq, seq]`` when batched) so the heatmap viz can render H
small panels side-by-side — students can see different heads picking up
different relations (e.g. one head attending to the previous token, one to
the verb's subject).

Implementation note: instead of running H separate Linear layers, we use
one big Linear per role and reshape — this is the standard, efficient
formulation and matches what ``nn.MultiheadAttention`` does internally.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)
from ...core.stateful_module import StatefulModuleMixin
from ...core.step_trace import StepRecorder


class _MultiHeadProjections(nn.Module):
    """Q/K/V/O linear projections, all heads packed into the embed_dim slot."""

    def __init__(self, embed_dim: int, num_heads: int, seed: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        gen = torch.Generator()
        gen.manual_seed(int(seed))
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)
        scale = 1.0 / math.sqrt(embed_dim)
        with torch.no_grad():
            for layer in (self.W_q, self.W_k, self.W_v, self.W_o):
                layer.weight.copy_(
                    torch.randn(layer.weight.shape, generator=gen) * scale
                )


class EduMultiHeadAttentionNode(StatefulModuleMixin, BaseNode):
    NODE_NAME = "EduMultiHeadAttention"
    CATEGORY = "LLM"
    DESCRIPTION = (
        "Toy multi-head self-attention. Splits embed_dim across num_heads, runs "
        "scaled dot-product attention per head, then mixes with W_o. Outputs "
        "[H, seq, seq] weights so each head's attention pattern can be "
        "visualised side-by-side."
    )

    structural_params = ("embed_dim", "num_heads", "seed")

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tensor",
                data_type=DataType.TENSOR,
                description="Input embeddings of shape [seq, D] or [seq, batch, D].",
            ),
            PortDefinition(
                name="mask",
                data_type=DataType.TENSOR,
                description="Optional [seq, seq] boolean mask (True = blocked). Combined with `causal` via OR.",
                optional=True,
            ),
            PortDefinition(
                name="labels",
                data_type=DataType.LIST,
                description="Optional token labels for heatmap viz axes.",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="output",
                data_type=DataType.TENSOR,
                description="Attention output, same shape as input.",
            ),
            PortDefinition(
                name="weights",
                data_type=DataType.TENSOR,
                description="Per-head weights — [H, seq, seq] for 2D input, [batch, H, seq, seq] for 3D input.",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="embed_dim",
                param_type=ParamType.INT,
                default=8,
                min_value=1,
                description="Token dimension. Must be divisible by num_heads.",
            ),
            ParamDefinition(
                name="num_heads",
                param_type=ParamType.INT,
                default=2,
                min_value=1,
                description="Number of parallel attention heads. Must divide embed_dim evenly.",
            ),
            ParamDefinition(
                name="causal",
                param_type=ParamType.BOOL,
                default=False,
                description="Block future positions per head (decoder-style).",
            ),
            ParamDefinition(
                name="seed",
                param_type=ParamType.INT,
                default=42,
                description="Seed for the W_q/W_k/W_v/W_o initialisation.",
            ),
        ]

    def build_module(self, params: dict[str, Any]) -> nn.Module:
        embed_dim = int(params.get("embed_dim", 8))
        num_heads = int(params.get("num_heads", 2))
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"EduMultiHeadAttention: embed_dim={embed_dim} must be divisible by num_heads={num_heads}."
            )
        return _MultiHeadProjections(
            embed_dim=embed_dim,
            num_heads=num_heads,
            seed=int(params.get("seed", 42)),
        )

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        progress_callback: Any | None = None,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        x = inputs.get("tensor")
        if x is None:
            raise ValueError("EduMultiHeadAttention requires a `tensor` input.")
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.float()

        embed_dim = int(params.get("embed_dim", 8))
        num_heads = int(params.get("num_heads", 2))
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"EduMultiHeadAttention: embed_dim={embed_dim} must be divisible by num_heads={num_heads}."
            )
        head_dim = embed_dim // num_heads

        if x.shape[-1] != embed_dim:
            raise ValueError(
                f"EduMultiHeadAttention: input last dim {x.shape[-1]} does not match embed_dim={embed_dim}."
            )

        causal = bool(params.get("causal", False))

        # Normalise to [B, seq, D].
        if x.ndim == 2:
            seq, _ = x.shape
            x_bf = x.unsqueeze(0)
            squeeze_out = True
        elif x.ndim == 3:
            seq, _, _ = x.shape
            x_bf = x.transpose(0, 1)  # [batch, seq, D]
            squeeze_out = False
        else:
            raise ValueError(
                f"EduMultiHeadAttention expects [seq, D] or [seq, batch, D]; got shape {tuple(x.shape)}"
            )

        B = x_bf.shape[0]

        module = self.get_or_build_module(context, params)

        def _split_heads(t: torch.Tensor) -> torch.Tensor:
            # [B, seq, D] → [B, H, seq, head_dim]
            return t.view(B, seq, num_heads, head_dim).transpose(1, 2)

        Q = _split_heads(module.W_q(x_bf))
        K = _split_heads(module.W_k(x_bf))
        V = _split_heads(module.W_v(x_bf))

        # scores [B, H, seq, seq]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)

        combined_mask = self._build_mask(seq, causal, inputs.get("mask"))
        if combined_mask is not None:
            # broadcast [seq, seq] over [B, H]
            scores = scores.masked_fill(
                combined_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)

        # Apply weights to V, then concat heads back together.
        attended = torch.matmul(weights, V)  # [B, H, seq, head_dim]
        attended = attended.transpose(1, 2).contiguous().view(B, seq, embed_dim)
        out_bf = module.W_o(attended)  # [B, seq, D]

        if squeeze_out:
            output = out_bf.squeeze(0)
            weights_out = weights.squeeze(0)  # [H, seq, seq]
        else:
            output = out_bf.transpose(0, 1)  # [seq, batch, D]
            weights_out = weights  # [batch, H, seq, seq]

        verbose = context is not None and getattr(context, "verbose", False)
        if verbose:
            recorder = StepRecorder()
            recorder.record(
                "input",
                f"Input embeddings of shape {tuple(x.shape)}; embed_dim={embed_dim}, num_heads={num_heads}, head_dim={head_dim}.",
                tensor=x,
            )
            recorder.record(
                "compute_qkv",
                "Project Q, K, V and split each into H heads of size head_dim.",
                Q=Q.squeeze(0) if squeeze_out else Q,
                K=K.squeeze(0) if squeeze_out else K,
                V=V.squeeze(0) if squeeze_out else V,
            )
            recorder.record(
                "scaled_scores",
                "Per-head scores: $S_h = Q_h K_h^T / \\sqrt{d_h}$.",
                scalars={"head_dim": float(head_dim), "sqrt_dh": float(math.sqrt(head_dim))},
                scores=scores.squeeze(0) if squeeze_out else scores,
            )
            recorder.record(
                "softmax_weights",
                "Per-head softmax — each head learns a different attention pattern.",
                weights=weights_out,
            )
            recorder.record(
                "concat_and_project",
                "Concatenate head outputs and mix with $W_o$.",
                output=output,
            )
            return {
                "output": output,
                "weights": weights_out,
                "__steps__": recorder.steps,
            }

        return {"output": output, "weights": weights_out}

    @staticmethod
    def _build_mask(seq: int, causal: bool, explicit_mask: Any) -> torch.Tensor | None:
        causal_mask = None
        if causal:
            causal_mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)

        ext_mask = None
        if explicit_mask is not None:
            ext_mask = explicit_mask
            if not isinstance(ext_mask, torch.Tensor):
                ext_mask = torch.as_tensor(ext_mask, dtype=torch.bool)
            if ext_mask.dtype != torch.bool:
                ext_mask = ext_mask.bool()
            if ext_mask.shape != (seq, seq):
                raise ValueError(
                    f"EduMultiHeadAttention: mask shape {tuple(ext_mask.shape)} doesn't match seq_len={seq}."
                )

        if causal_mask is None and ext_mask is None:
            return None
        if causal_mask is None:
            return ext_mask
        if ext_mask is None:
            return causal_mask
        return causal_mask | ext_mask
