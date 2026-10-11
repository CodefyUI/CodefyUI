import math
from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition
from ...core.stateful_module import StatefulModuleMixin


class MultiHeadAttentionNode(StatefulModuleMixin, BaseNode):
    NODE_NAME = "MultiHeadAttention"
    CATEGORY = "Transformer"
    DESCRIPTION = "Attention over query, key and value, plus its weights"
    DETAILS = (
        "The layer is PyTorch's `nn.MultiheadAttention`. "
        "$\\text{Attention}(Q,K,V)=\\text{softmax}(\\frac{QK^T}{\\sqrt{d_k}})V$, with "
        "`embed_dim` split across `num_heads`. Inputs are (seq, batch, embed) "
        "unless `batch_first` puts the batch first, and the returned weights are "
        "averaged over the heads."
    )

    structural_params = ("embed_dim", "num_heads", "batch_first")

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="query", data_type=DataType.TENSOR, description="Query tensor — (seq, batch, embed) by default, (batch, seq, embed) when batch_first is on"),
            PortDefinition(name="key", data_type=DataType.TENSOR, description="Key tensor — same layout as query"),
            PortDefinition(name="value", data_type=DataType.TENSOR, description="Value tensor — same layout as query"),
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
            ParamDefinition(
                name="batch_first",
                param_type=ParamType.BOOL,
                default=False,
                description=(
                    "If True, input/output shape is (batch, seq, embed) — the same layout the "
                    "RNN nodes use by default. Off means torch's transformer default, "
                    "(seq, batch, embed). Existing graphs were built against that, which is why "
                    "it stays the default here."
                ),
            ),
        ]

    def build_module(self, params: dict[str, Any]) -> Any:
        import torch.nn as nn
        return nn.MultiheadAttention(
            embed_dim=params.get("embed_dim", 512),
            num_heads=params.get("num_heads", 8),
            batch_first=bool(params.get("batch_first", False)),
        )

    def execute(self, inputs: dict[str, Any], params: dict[str, Any], *, context: Any = None) -> dict[str, Any]:
        query = inputs["query"]
        key = inputs["key"]
        value = inputs["value"]

        mha = self.get_or_build_module(context, params)
        output, weights = mha(query, key, value)
        result: dict[str, Any] = {"output": output, "weights": weights}

        if context is not None and getattr(context, "verbose", False):
            result["__steps__"] = _trace_steps(mha, query, key, value, params)

        return result


def _trace_steps(mha: Any, query: Any, key: Any, value: Any, params: dict[str, Any]) -> list:
    """Recompute ``mha``'s forward pass step by step, from its own weights.

    Every tensor is shown batch-first, ``(batch, seq, ...)``, whatever the
    node's layout. The head-averaged weights and the final output equal the
    node's ``weights`` and ``output`` up to that transpose.
    """
    import torch
    import torch.nn.functional as F

    from ...core.step_trace import StepRecorder

    unbatched = query.dim() == 2
    seq_first = not unbatched and not bool(params.get("batch_first", False))

    def batch_first(t: Any) -> Any:
        if unbatched:
            return t.unsqueeze(0)
        return t.transpose(0, 1) if seq_first else t

    def shown(t: Any) -> Any:
        return t.squeeze(0) if unbatched else t

    num_heads = mha.num_heads
    d_k = mha.head_dim
    recorder = StepRecorder()
    with torch.no_grad():
        q_b, k_b, v_b = batch_first(query), batch_first(key), batch_first(value)
        recorder.record(
            "inputs_qkv",
            "Receive Q, K, V (shown as batch-first for clarity).",
            Q=shown(q_b), K=shown(k_b), V=shown(v_b),
        )

        w_q, w_k, w_v = mha.in_proj_weight.chunk(3)
        b_q, b_k, b_v = (
            mha.in_proj_bias.chunk(3) if mha.in_proj_bias is not None else (None, None, None)
        )
        q_p, k_p, v_p = F.linear(q_b, w_q, b_q), F.linear(k_b, w_k, b_k), F.linear(v_b, w_v, b_v)
        recorder.record(
            "project_qkv",
            "Project with the layer's learned weights: $Q' = QW_Q$, $K' = KW_K$, $V' = VW_V$.",
            Q=shown(q_p), K=shown(k_p), V=shown(v_p),
        )

        def split(t: Any) -> Any:  # (B, L, E) -> (B, H, L, d_k)
            return t.reshape(t.shape[0], t.shape[1], num_heads, d_k).transpose(1, 2)

        q_h, k_h, v_h = split(q_p), split(k_p), split(v_p)
        recorder.record(
            "split_heads",
            f"Split the {mha.embed_dim} features into {num_heads} heads of $d_k = {d_k}$ each.",
            scalars={"num_heads": float(num_heads), "d_k": float(d_k)},
            Q=shown(q_h), K=shown(k_h), V=shown(v_h),
        )

        scores = torch.matmul(q_h, k_h.transpose(-2, -1)) / math.sqrt(d_k)
        recorder.record(
            "scaled_scores",
            "Compute each head's attention scores: $S_h = Q_h K_h^T / \\sqrt{d_k}$.",
            scalars={"d_k": float(d_k)},
            scores=shown(scores),
        )

        head_weights = F.softmax(scores, dim=-1)
        recorder.record(
            "softmax_weights",
            "Normalise each head with softmax: $A_h = \\text{softmax}(S_h)$ (rows sum to 1). "
            "`weights` is the average over heads, the node's `weights` output.",
            head_weights=shown(head_weights),
            weights=shown(head_weights.mean(dim=1)),
        )

        heads = torch.matmul(head_weights, v_h)
        concat = heads.transpose(1, 2).reshape(heads.shape[0], heads.shape[2], mha.embed_dim)
        recorder.record(
            "concat_heads",
            "Weighted sum of each head's values, $O_h = A_h V_h$, then the heads side by side.",
            output=shown(concat),
        )

        output = mha.out_proj(concat)
        recorder.record(
            "attended_output",
            "Mix the heads with the output projection: $O = \\text{concat}(O_h) W_O$. "
            "This is the node's `output`.",
            output=shown(output),
        )
    return recorder.steps
