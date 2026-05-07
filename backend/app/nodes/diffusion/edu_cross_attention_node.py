"""EduCrossAttentionNode — multi-head attention with separate Q and K/V sources.

Where :class:`EduSelfAttentionNode` projects all three Q/K/V from a
single input, cross-attention takes Q from one tensor (e.g. image
latent) and K/V from another (e.g. text embedding). This is the
mechanism that lets Stable Diffusion's U-Net condition image generation
on a text prompt: each image position's query attends over the prompt's
token embeddings, fetching whichever ones it needs to denoise itself.

The math is the same scaled dot-product attention as self-attention; only
the input wiring changes:

    Q = W_q(query)        # [Q_seq, *, D]    — image latent
    K = W_k(context)      # [K_seq, *, D]    — text embedding
    V = W_v(context)      # [K_seq, *, D]
    A = softmax(Q K^T / sqrt(d_h))            # [H, Q_seq, K_seq]
    O = W_o( concat(heads of A V) )           # [Q_seq, *, D]

The crucial observation for teaching: Q and K can have different
sequence lengths. The output shape mirrors Q (one row per query
position), and the attention matrix is rectangular.
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


class _CrossAttentionProjections(nn.Module):
    """W_q on query input, W_k/W_v on context input, W_o on concatenated heads."""

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


class EduCrossAttentionNode(StatefulModuleMixin, BaseNode):
    NODE_NAME = "EduCrossAttention"
    CATEGORY = "Diffusion"
    DESCRIPTION = (
        "Multi-head cross-attention. $Q$ comes from `query`, $K$ and $V$ from "
        "`context` — they may have different sequence lengths. Outputs a "
        "rectangular [H, Q_seq, K_seq] attention map showing how each "
        "query position attended to each context token."
    )

    structural_params = ("embed_dim", "num_heads", "seed")

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="query",
                data_type=DataType.TENSOR,
                description="Query tensor [Q_seq, D] or [Q_seq, batch, D].",
            ),
            PortDefinition(
                name="context",
                data_type=DataType.TENSOR,
                description="Key/value source [K_seq, D] or [K_seq, batch, D]. May differ in seq length from query.",
            ),
            PortDefinition(
                name="mask",
                data_type=DataType.TENSOR,
                description="Optional [Q_seq, K_seq] boolean mask (True = blocked).",
                optional=True,
            ),
            PortDefinition(
                name="q_labels",
                data_type=DataType.LIST,
                description="Optional row labels for the heatmap viz (one per query position).",
                optional=True,
            ),
            PortDefinition(
                name="k_labels",
                data_type=DataType.LIST,
                description="Optional column labels for the heatmap viz (one per context position).",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="output",
                data_type=DataType.TENSOR,
                description="Attention output, same shape as query.",
            ),
            PortDefinition(
                name="weights",
                data_type=DataType.TENSOR,
                description="Cross-attention weights — [H, Q_seq, K_seq] or [batch, H, Q_seq, K_seq].",
            ),
            PortDefinition(
                name="q_labels",
                data_type=DataType.LIST,
                description="Pass-through of the query labels for downstream viz.",
            ),
            PortDefinition(
                name="k_labels",
                data_type=DataType.LIST,
                description="Pass-through of the context labels for downstream viz.",
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
                description="Token dimension. Both query and context must use this same dim.",
            ),
            ParamDefinition(
                name="num_heads",
                param_type=ParamType.INT,
                default=2,
                min_value=1,
                description="Number of parallel attention heads. Must divide embed_dim.",
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
                f"EduCrossAttention: embed_dim={embed_dim} must be divisible by num_heads={num_heads}."
            )
        return _CrossAttentionProjections(
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
        q_in = inputs.get("query")
        c_in = inputs.get("context")
        if q_in is None or c_in is None:
            raise ValueError("EduCrossAttention requires both `query` and `context` inputs.")
        if not isinstance(q_in, torch.Tensor):
            q_in = torch.as_tensor(q_in, dtype=torch.float32)
        if not isinstance(c_in, torch.Tensor):
            c_in = torch.as_tensor(c_in, dtype=torch.float32)
        q_in = q_in.float()
        c_in = c_in.float()

        embed_dim = int(params.get("embed_dim", 8))
        num_heads = int(params.get("num_heads", 2))
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"EduCrossAttention: embed_dim={embed_dim} must be divisible by num_heads={num_heads}."
            )
        head_dim = embed_dim // num_heads

        if q_in.shape[-1] != embed_dim:
            raise ValueError(
                f"EduCrossAttention: query last dim {q_in.shape[-1]} does not match embed_dim={embed_dim}."
            )
        if c_in.shape[-1] != embed_dim:
            raise ValueError(
                f"EduCrossAttention: context last dim {c_in.shape[-1]} does not match embed_dim={embed_dim}."
            )

        # Normalise query and context to batch-first [B, seq, D].
        if q_in.ndim == 2 and c_in.ndim == 2:
            q_bf = q_in.unsqueeze(0)  # [1, Q_seq, D]
            c_bf = c_in.unsqueeze(0)
            squeeze_out = True
        elif q_in.ndim == 3 and c_in.ndim == 3:
            q_bf = q_in.transpose(0, 1)  # [B, Q_seq, D]
            c_bf = c_in.transpose(0, 1)
            squeeze_out = False
        else:
            raise ValueError(
                "EduCrossAttention: query and context must both be 2D or both be 3D, "
                f"got query.shape={tuple(q_in.shape)}, context.shape={tuple(c_in.shape)}."
            )
        B = q_bf.shape[0]
        Q_seq = q_bf.shape[1]
        K_seq = c_bf.shape[1]

        module = self.get_or_build_module(context, params)

        def _split_heads(t: torch.Tensor, seq: int) -> torch.Tensor:
            return t.view(B, seq, num_heads, head_dim).transpose(1, 2)

        Q = _split_heads(module.W_q(q_bf), Q_seq)  # [B, H, Q_seq, head_dim]
        K = _split_heads(module.W_k(c_bf), K_seq)  # [B, H, K_seq, head_dim]
        V = _split_heads(module.W_v(c_bf), K_seq)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)  # [B, H, Q_seq, K_seq]

        ext_mask = inputs.get("mask")
        if ext_mask is not None:
            if not isinstance(ext_mask, torch.Tensor):
                ext_mask = torch.as_tensor(ext_mask, dtype=torch.bool)
            if ext_mask.dtype != torch.bool:
                ext_mask = ext_mask.bool()
            if ext_mask.shape != (Q_seq, K_seq):
                raise ValueError(
                    f"EduCrossAttention: mask shape {tuple(ext_mask.shape)} doesn't match (Q_seq={Q_seq}, K_seq={K_seq})."
                )
            scores = scores.masked_fill(ext_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)

        attended = torch.matmul(weights, V)  # [B, H, Q_seq, head_dim]
        attended = attended.transpose(1, 2).contiguous().view(B, Q_seq, embed_dim)
        out_bf = module.W_o(attended)

        if squeeze_out:
            output = out_bf.squeeze(0)
            weights_out = weights.squeeze(0)  # [H, Q_seq, K_seq]
        else:
            output = out_bf.transpose(0, 1)  # [Q_seq, batch, D]
            weights_out = weights  # [B, H, Q_seq, K_seq]

        q_labels = list(inputs.get("q_labels") or [])
        k_labels = list(inputs.get("k_labels") or [])

        verbose = context is not None and getattr(context, "verbose", False)
        if verbose:
            recorder = StepRecorder()
            recorder.record(
                "input",
                f"Query [Q_seq={Q_seq}, B={B}, D={embed_dim}], context [K_seq={K_seq}, B={B}, D={embed_dim}]",
                query=q_in,
                context=c_in,
            )
            recorder.record(
                "compute_qkv",
                "Project Q from query, K and V from context — note the asymmetric source.",
                Q=Q.squeeze(0) if squeeze_out else Q,
                K=K.squeeze(0) if squeeze_out else K,
                V=V.squeeze(0) if squeeze_out else V,
            )
            recorder.record(
                "scaled_scores",
                "Scores: $S_h = Q_h K_h^T / \\sqrt{d_h}$ — rectangular [Q_seq × K_seq].",
                scalars={"head_dim": float(head_dim)},
                scores=scores.squeeze(0) if squeeze_out else scores,
            )
            recorder.record(
                "softmax_weights",
                "Per-head softmax over the K-axis.",
                weights=weights_out,
            )
            recorder.record(
                "concat_and_project",
                "Concatenate heads, apply W_o.",
                output=output,
            )
            return {
                "output": output,
                "weights": weights_out,
                "q_labels": q_labels,
                "k_labels": k_labels,
                "__steps__": recorder.steps,
            }

        return {
            "output": output,
            "weights": weights_out,
            "q_labels": q_labels,
            "k_labels": k_labels,
        }
