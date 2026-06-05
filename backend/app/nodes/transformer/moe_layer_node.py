"""MoELayerNode — Mixture-of-Experts feed-forward layer.

Replaces the FFN block in a Transformer with N expert FFNs plus a gating
network. For each token, the gate picks the top-k experts by softmax score;
the layer's output is the weighted sum of those k experts' outputs.

This is the structure used in Switch Transformer, Mixtral, and DeepSeek-MoE.
The teaching version here is intentionally tiny — students see the gating
math and per-expert routing without GPU-cluster boilerplate.

Outputs:

- ``output``: [B, T, H], same shape as input.
- ``routing_weights``: [B, T, k], the softmax-normalised weights over the
  selected experts (one row per token sums to 1).
- ``expert_indices``: [B, T, k], the integer expert indices each token
  was routed to. Lets a viz draw "which expert handled which token."
"""

from __future__ import annotations

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


class _ExpertFFN(nn.Module):
    """Single expert: SwiGLU-style 2-layer FFN."""

    def __init__(self, hidden_dim: int, expert_hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, expert_hidden_dim)
        self.fc2 = nn.Linear(expert_hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)))


class _MoELayer(nn.Module):
    def __init__(self, num_experts: int, top_k: int, hidden_dim: int, expert_hidden_dim: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = max(1, min(top_k, num_experts))
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [_ExpertFFN(hidden_dim, expert_hidden_dim) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, T, H]
        b, t, h = x.shape
        # Gating logits → top-k softmax. We softmax *only* over the selected
        # experts so the routing weights sum to 1 per token (the standard
        # Switch / Mixtral convention).
        gate_logits = self.gate(x)  # [B, T, N]
        topk_vals, topk_idx = torch.topk(gate_logits, k=self.top_k, dim=-1)  # both [B, T, k]
        topk_weights = F.softmax(topk_vals, dim=-1)  # [B, T, k]

        # For each (b, t, k) slot, run x[b, t] through experts[topk_idx[b, t, k]]
        # and weight it by topk_weights[b, t, k]. We do this densely (small N
        # for teaching), then mask + sum — clear over fast.
        flat_x = x.reshape(b * t, h)  # [B*T, H]
        flat_idx = topk_idx.reshape(b * t, self.top_k)  # [B*T, k]
        flat_w = topk_weights.reshape(b * t, self.top_k)  # [B*T, k]

        out = torch.zeros_like(flat_x)
        for k in range(self.top_k):
            expert_id_for_slot = flat_idx[:, k]  # [B*T]
            weights_for_slot = flat_w[:, k].unsqueeze(-1)  # [B*T, 1]
            # Group by expert and run the matching tokens through that expert.
            for e in range(self.num_experts):
                mask = expert_id_for_slot == e
                if not mask.any():
                    continue
                tokens = flat_x[mask]
                expert_out = self.experts[e](tokens)
                out[mask] = out[mask] + weights_for_slot[mask] * expert_out

        return out.reshape(b, t, h), topk_weights, topk_idx


class MoELayerNode(BaseNode):
    NODE_NAME = "MoELayer"
    CATEGORY = "Transformer"
    DESCRIPTION = (
        "Mixture-of-Experts FFN. For each token, a gate picks the top-k "
        "experts by softmax score; the layer output is the weighted sum of "
        "those experts' outputs. The architecture used in Switch / Mixtral / "
        "DeepSeek-MoE."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="x", data_type=DataType.TENSOR, description="Input hidden states [B, T, H]."),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="output", data_type=DataType.TENSOR, description="MoE output, same shape as input [B, T, H]."),
            PortDefinition(name="routing_weights", data_type=DataType.TENSOR, description="Per-token softmax weights over top-k experts [B, T, k]."),
            PortDefinition(name="expert_indices", data_type=DataType.TENSOR, description="Per-token top-k expert ids [B, T, k]."),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="num_experts", param_type=ParamType.INT, default=4, min_value=1, description="Number of expert FFNs."),
            ParamDefinition(name="top_k", param_type=ParamType.INT, default=2, min_value=1, description="Experts per token (clamped to num_experts)."),
            ParamDefinition(name="hidden_dim", param_type=ParamType.INT, default=128, min_value=1, description="Token hidden size H."),
            ParamDefinition(name="expert_hidden_dim", param_type=ParamType.INT, default=256, min_value=1, description="Inner width of each expert FFN."),
            ParamDefinition(name="seed", param_type=ParamType.INT, default=42, description="Init seed for reproducibility."),
        ]

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        progress_callback: Any | None = None,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        x = inputs.get("x")
        if x is None:
            raise ValueError("MoELayer requires an `x` input tensor.")
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.float()
        if x.dim() == 2:
            x = x.unsqueeze(0)  # treat [T, H] as [1, T, H]

        num_experts = max(1, int(params.get("num_experts", 4)))
        top_k = max(1, int(params.get("top_k", 2)))
        hidden_dim = int(params.get("hidden_dim", 128))
        expert_hidden_dim = int(params.get("expert_hidden_dim", 256))
        seed = int(params.get("seed", 42))

        gen_state = torch.random.get_rng_state()
        try:
            torch.manual_seed(seed)
            layer = _MoELayer(num_experts, top_k, hidden_dim, expert_hidden_dim)
        finally:
            torch.random.set_rng_state(gen_state)

        # The layer is built fresh on CPU; move it to the input's device so the
        # gate/expert weights match x under the global device setting.
        from ...core.device_utils import to_device
        layer = to_device(layer, x.device)

        with torch.no_grad():
            out, routing_w, idx = layer(x)

        return {
            "output": out,
            "routing_weights": routing_w,
            "expert_indices": idx,
        }
