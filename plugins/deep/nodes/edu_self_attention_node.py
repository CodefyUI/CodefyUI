"""EduSelfAttentionNode — single-head, hand-written scaled dot-product attention.

This is the educational counterpart to the Transformer/MultiHeadAttention
node. Where that node wraps ``nn.MultiheadAttention`` (production-sized,
opaque), this one writes out the textbook formula step by step:

    Q, K, V = W_q @ x, W_k @ x, W_v @ x          # three projections
    S       = Q @ K.T / sqrt(d_k)                  # scaled scores
    S       = S.masked_fill(mask, -inf)            # optional masking
    A       = softmax(S / temperature, dim=-1)     # attention weights
    O       = A @ V                                # weighted sum of values

with tiny defaults (``embed_dim=8``) so the resulting [seq, seq] attention
matrix can be rendered as a heatmap directly. The optional ``labels`` input
lets the heatmap viz show real token names on its axes.

Causal attention is supported via the ``causal`` flag (decoder-style: a
position cannot attend to anything to its right). External masks coming from
``AttentionMask`` are also honoured. Both are combined with logical OR — a
position is blocked if either source says so.

Inputs may be 2D ``[seq, D]`` or 3D ``[seq, batch, D]`` (transformer
convention). Output shape mirrors the input; the ``weights`` output is
``[seq, seq]`` for 2D input and ``[batch, seq, seq]`` for 3D input.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)
from app.core.stateful_module import StatefulModuleMixin
from app.core.step_trace import StepRecorder


class _SelfAttentionProjections(nn.Module):
    """Three W_q / W_k / W_v linear projections, seeded for reproducibility."""

    def __init__(self, embed_dim: int, seed: int) -> None:
        super().__init__()
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        scale = 1.0 / math.sqrt(embed_dim)
        with torch.no_grad():
            for layer in (self.W_q, self.W_k, self.W_v):
                layer.weight.copy_(
                    torch.randn(layer.weight.shape, generator=gen) * scale
                )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.W_q(x), self.W_k(x), self.W_v(x)


class EduSelfAttentionNode(StatefulModuleMixin, BaseNode):
    NODE_NAME = "Edu-SelfAttention"
    CATEGORY = "LLM"
    DESCRIPTION = (
        "Hand-written single-head self-attention: Q,K,V = three Linear projections; "
        "scores = Q@K^T/√d; weights = softmax(scores); output = weights@V. "
        "Outputs the [seq, seq] weight matrix for direct heatmap visualisation."
    )

    structural_params = ("embed_dim", "seed")

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
                description="Optional token labels for heatmap viz axes. Pass-through to the viz layer.",
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
                description="Attention weights — [seq, seq] for 2D input, [batch, seq, seq] for 3D input.",
            ),
            PortDefinition(
                name="labels",
                data_type=DataType.LIST,
                description="Pass-through of the optional `labels` input — surfaces token names to the heatmap viz.",
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
                description="Token dimension. Must match the upstream tensor's last dim.",
            ),
            ParamDefinition(
                name="causal",
                param_type=ParamType.BOOL,
                default=False,
                description="Block positions from attending to anything to their right (decoder-style / GPT).",
            ),
            ParamDefinition(
                name="temperature",
                param_type=ParamType.FLOAT,
                default=1.0,
                min_value=0.01,
                description="Divide scores by this before softmax. <1 sharpens, >1 flattens the distribution.",
            ),
            ParamDefinition(
                name="seed",
                param_type=ParamType.INT,
                default=42,
                description="Seed for the W_q/W_k/W_v initialisation. Same seed → same weights.",
            ),
        ]

    def build_module(self, params: dict[str, Any]) -> nn.Module:
        return _SelfAttentionProjections(
            embed_dim=int(params.get("embed_dim", 8)),
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
            raise ValueError("EduSelfAttention requires a `tensor` input.")
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.float()

        embed_dim = int(params.get("embed_dim", 8))
        if x.shape[-1] != embed_dim:
            raise ValueError(
                f"EduSelfAttention: input last dim {x.shape[-1]} does not match embed_dim={embed_dim}."
            )

        causal = bool(params.get("causal", False))
        temperature = max(1e-6, float(params.get("temperature", 1.0)))

        # Normalise to batch-first [B, seq, D] so we can use a single matmul path.
        if x.ndim == 2:
            seq, d = x.shape
            x_bf = x.unsqueeze(0)  # [1, seq, D]
            squeeze_out = True
        elif x.ndim == 3:
            seq, batch, d = x.shape
            x_bf = x.transpose(0, 1)  # [batch, seq, D]
            squeeze_out = False
        else:
            raise ValueError(
                f"EduSelfAttention expects [seq, D] or [seq, batch, D]; got shape {tuple(x.shape)}"
            )

        module = self.get_or_build_module(context, params)
        Q, K, V = module(x_bf)  # each [B, seq, D]

        # scores[b, i, j] = (Q[b, i] · K[b, j]) / sqrt(d)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)

        # Combine causal + explicit mask into a single boolean. True = blocked.
        combined_mask = self._build_mask(seq, causal, inputs.get("mask"))
        if combined_mask is not None:
            # Broadcast mask [seq, seq] over batch dim.
            scores = scores.masked_fill(combined_mask.unsqueeze(0), float("-inf"))

        weights = F.softmax(scores / temperature, dim=-1)
        # Numerical safety: rows that are entirely masked become NaN under
        # softmax. Replace with zeros so downstream consumers don't crash.
        weights = torch.nan_to_num(weights, nan=0.0)

        out_bf = torch.matmul(weights, V)  # [B, seq, D]

        if squeeze_out:
            output = out_bf.squeeze(0)  # [seq, D]
            weights_out = weights.squeeze(0)  # [seq, seq]
        else:
            output = out_bf.transpose(0, 1)  # [seq, batch, D]
            weights_out = weights  # [batch, seq, seq]

        labels_out = list(inputs.get("labels") or [])

        verbose = context is not None and getattr(context, "verbose", False)
        if verbose:
            recorder = StepRecorder()
            recorder.record(
                "input",
                f"Input embeddings of shape {tuple(x.shape)}.",
                tensor=x,
            )
            recorder.record(
                "compute_qkv",
                "Project the input three times: $Q = xW_q$, $K = xW_k$, $V = xW_v$.",
                Q=Q.squeeze(0) if squeeze_out else Q,
                K=K.squeeze(0) if squeeze_out else K,
                V=V.squeeze(0) if squeeze_out else V,
            )
            recorder.record(
                "scaled_scores",
                "Compute attention scores: $S = QK^T / \\sqrt{d}$.",
                scalars={"d": float(d), "sqrt_d": float(math.sqrt(d))},
                scores=scores.squeeze(0) if squeeze_out else scores,
            )
            if combined_mask is not None:
                recorder.record(
                    "mask",
                    f"Apply mask (causal={causal}, explicit_mask={inputs.get('mask') is not None}).",
                    mask=combined_mask,
                )
            recorder.record(
                "softmax_weights",
                f"Normalise with softmax / temperature ({temperature}). Each row sums to 1.",
                weights=weights_out,
            )
            recorder.record(
                "weighted_sum",
                "Weighted sum of value vectors: $O = AV$.",
                output=output,
            )
            return {
                "output": output,
                "weights": weights_out,
                "labels": labels_out,
                "__steps__": recorder.steps,
            }

        return {"output": output, "weights": weights_out, "labels": labels_out}

    @staticmethod
    def _build_mask(seq: int, causal: bool, explicit_mask: Any) -> torch.Tensor | None:
        """Combine causal + explicit masks. Returns None if no masking needed."""
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
                    f"EduSelfAttention: mask shape {tuple(ext_mask.shape)} doesn't match seq_len={seq}."
                )

        if causal_mask is None and ext_mask is None:
            return None
        if causal_mask is None:
            return ext_mask
        if ext_mask is None:
            return causal_mask
        return causal_mask | ext_mask
