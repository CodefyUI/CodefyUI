import math
from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition
from ...core.stateful_module import StatefulModuleMixin


class MultiHeadAttentionNode(StatefulModuleMixin, BaseNode):
    NODE_NAME = "MultiHeadAttention"
    CATEGORY = "Transformer"
    DESCRIPTION = (
        "Apply multi-head attention mechanism (wraps nn.MultiheadAttention). "
        "Core: $\\text{Attention}(Q,K,V)=\\text{softmax}(\\frac{QK^T}{\\sqrt{d_k}})V$"
    )

    structural_params = ("embed_dim", "num_heads")

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="query", data_type=DataType.TENSOR, description="Query tensor (seq_len, batch, embed_dim)"),
            PortDefinition(name="key", data_type=DataType.TENSOR, description="Key tensor (seq_len, batch, embed_dim)"),
            PortDefinition(name="value", data_type=DataType.TENSOR, description="Value tensor (seq_len, batch, embed_dim)"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="output", data_type=DataType.TENSOR, description="Attention output tensor"),
            PortDefinition(name="weights", data_type=DataType.TENSOR, description="Attention weight tensor"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="embed_dim", param_type=ParamType.INT, default=512, description="Total dimension of the model"),
            ParamDefinition(name="num_heads", param_type=ParamType.INT, default=8, description="Number of parallel attention heads"),
        ]

    def build_module(self, params: dict[str, Any]) -> Any:
        import torch.nn as nn
        return nn.MultiheadAttention(
            embed_dim=params.get("embed_dim", 512),
            num_heads=params.get("num_heads", 8),
        )

    def execute(self, inputs: dict[str, Any], params: dict[str, Any], *, context: Any = None) -> dict[str, Any]:
        import torch
        import torch.nn.functional as F

        query = inputs["query"]
        key = inputs["key"]
        value = inputs["value"]
        embed_dim = params.get("embed_dim", 512)
        num_heads = params.get("num_heads", 8)

        mha = self.get_or_build_module(context, params)
        output, weights = mha(query, key, value)
        result: dict[str, Any] = {"output": output, "weights": weights}

        if context is not None and getattr(context, "verbose", False):
            from ...core.step_trace import StepRecorder
            recorder = StepRecorder()
            d_k = embed_dim // max(num_heads, 1)
            # Transpose to (batch, seq, embed) for textbook-style display.
            q_b = query.transpose(0, 1) if query.dim() == 3 else query
            k_b = key.transpose(0, 1) if key.dim() == 3 else key
            v_b = value.transpose(0, 1) if value.dim() == 3 else value
            recorder.record(
                "inputs_qkv",
                "Receive Q, K, V (shown as batch-first for clarity).",
                Q=q_b, K=k_b, V=v_b,
            )
            scores = torch.matmul(q_b, k_b.transpose(-2, -1)) / math.sqrt(d_k)
            recorder.record(
                "scaled_scores",
                "Compute attention scores: $S = QK^T / \\sqrt{d_k}$.",
                scalars={"d_k": float(d_k)},
                scores=scores,
            )
            attn_simple = F.softmax(scores, dim=-1)
            recorder.record(
                "softmax_weights",
                "Normalise with softmax: $A = \\text{softmax}(S)$ (rows sum to 1).",
                weights=attn_simple,
            )
            attended = torch.matmul(attn_simple, v_b)
            recorder.record(
                "attended_output",
                "Weighted sum of V: $O = AV$ (each output is a convex combination of value rows).",
                output=attended,
            )
            result["__steps__"] = recorder.steps

        return result
