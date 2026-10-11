"""KLDivergenceNode — KL(p || q) for the RLHF "stay close to ref" term.

In PPO-RLHF, the policy update is regularised by KL divergence against the
reference (frozen) policy:

    L = E[ r(x) − β · KL(π_policy(x) || π_ref(x)) ]

This node computes that KL term given two probability (or logit) tensors
of the same shape ``[..., V]``. ``reduction`` aggregates the per-sample KL
(the sum over the last axis): ``batchmean`` and ``mean`` average it over
samples, ``sum`` totals it and ``none`` returns it per sample.
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
    DESCRIPTION = "KL(p || q) from probabilities or logits"
    DETAILS = (
        "reduction aggregates the per-sample KL, the sum over the last axis: "
        "batchmean (the RLHF default) and mean both average it over samples, sum "
        "adds it up and none returns one value per sample. A 1-D input is one "
        "sample. "
        "KL is not symmetric: p is the policy and q the frozen reference it is "
        "held near."
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
                description="How to aggregate per-sample KL. batchmean and mean = average over samples (the RLHF default); sum = total; none = one value per sample.",
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
        # computes target * (log target − input) per element; summing the last
        # axis gives one KL per sample. A 1-D input is a single sample, and any
        # leading axes are flattened into the sample axis.
        kl_elem = F.kl_div(log_q, p_probs, reduction="none", log_target=False)
        per_sample = kl_elem.sum(dim=-1).reshape(-1)

        if reduction == "none":
            kl = per_sample
        elif reduction == "sum":
            kl = per_sample.sum()
        elif reduction in ("batchmean", "mean"):
            kl = per_sample.mean()
        else:
            raise ValueError(f"KLDivergence: unknown reduction {reduction!r}")

        return {"kl": kl}
