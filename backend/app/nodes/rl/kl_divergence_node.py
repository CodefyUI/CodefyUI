"""KLDivergenceNode — KL(p || q) for the RLHF "stay close to ref" term.

In PPO-RLHF, the policy update is regularised by KL divergence against the
reference (frozen) policy:

    L = E[ r(x) − β · KL(π_policy(x) || π_ref(x)) ]

This node computes that KL term given two probability (or logit) tensors
of the same shape ``[..., V]``. ``reduction`` controls whether the output
is per-sample or scalar (matching the PyTorch convention).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)


class KLDivergenceNode(BaseNode):
    NODE_NAME = "KLDivergence"
    CATEGORY = "RL"
    DESCRIPTION = (
        "KL(p || q) divergence — the regularisation term in RLHF that keeps "
        "the policy close to the reference. Accepts probabilities or logits; "
        "returns scalar (default) or per-sample."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="p", data_type=DataType.TENSOR, description="Policy distribution: [..., V] probs or logits."),
            PortDefinition(name="q", data_type=DataType.TENSOR, description="Reference distribution: [..., V] probs or logits."),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="kl", data_type=DataType.TENSOR, description="KL divergence — scalar or per-sample, depending on reduction."),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="input_kind",
                param_type=ParamType.SELECT,
                default="probs",
                options=["probs", "logits"],
                description="Whether p and q are already probabilities or pre-softmax logits.",
            ),
            ParamDefinition(
                name="reduction",
                param_type=ParamType.SELECT,
                default="batchmean",
                options=["batchmean", "sum", "mean", "none"],
                description="How to aggregate per-sample KL. batchmean = sum / batch_size (the RLHF default).",
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
        p = inputs.get("p")
        q = inputs.get("q")
        if p is None or q is None:
            raise ValueError("KLDivergence requires `p` and `q` inputs.")

        if not isinstance(p, torch.Tensor):
            p = torch.as_tensor(p, dtype=torch.float32)
        if not isinstance(q, torch.Tensor):
            q = torch.as_tensor(q, dtype=torch.float32)
        p = p.float()
        q = q.float()

        if p.shape != q.shape:
            raise ValueError(
                f"KLDivergence: `p` and `q` must have the same shape, got "
                f"{tuple(p.shape)} vs {tuple(q.shape)}."
            )

        kind = str(params.get("input_kind", "probs"))
        reduction = str(params.get("reduction", "batchmean"))

        # F.kl_div expects target = p (probs) and input = log q.
        if kind == "logits":
            log_q = F.log_softmax(q, dim=-1)
            p_probs = F.softmax(p, dim=-1)
        else:
            eps = 1e-12
            log_q = torch.log(q.clamp_min(eps))
            p_probs = p

        # KL(p || q) = Σ p_i (log p_i − log q_i). F.kl_div with log_target=False
        # computes target * (log target − input) per element.
        if reduction == "none":
            # Element-wise KL contribution; reduce over last dim to get per-sample.
            kl_elem = F.kl_div(log_q, p_probs, reduction="none", log_target=False)
            kl = kl_elem.sum(dim=-1)
            # Flatten leading batch dims into [B] when multi-axis.
            kl = kl.reshape(-1) if kl.dim() > 1 else kl
        else:
            kl = F.kl_div(log_q, p_probs, reduction=reduction, log_target=False)

        return {"kl": kl}
