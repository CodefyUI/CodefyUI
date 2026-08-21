"""PPOClipObjectiveNode -- the clipped surrogate, with both branches exposed.

C5-2's whole contribution is one line:

    L = min( r * A,  clip(r, 1-eps, 1+eps) * A )

and the reason a lesson needs a node for it rather than a formula on a slide
is that the interesting part is *which branch won*. Emitting only the final
value hides exactly what the chapter is trying to show, so this node returns
the unclipped term, the clipped term, the final minimum, and a boolean mask of
which samples were truncated. A student can put all four next to each other
and read the safety belt engaging.

The min is the part everyone gets wrong on first reading, so it is worth
stating plainly. For a positive advantage the objective rises with r until
1+eps and is flat after: pushing the probability further stops paying. For a
negative advantage it rises as r *falls* until 1-eps and is flat below: pushing
it down further stops paying too. Two directions, one expression, because
taking the smaller of the two terms picks the flat branch in both cases.

Note what the clip does NOT do: it does not clamp r, and it does not forbid an
update. The policy is free to produce r = 1.4; it simply gains nothing by it,
so the gradient stops pointing that way. The incentive is removed, not the
action.

``ratio`` can be supplied directly -- which is what a worked example wants, so
the three rows from the textbook can be typed in and checked -- or computed
from new and old log-probabilities, which is what a real training step has.
"""

from __future__ import annotations

from typing import Any

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)


class PPOClipObjectiveNode(BaseNode):
    NODE_NAME = "PPOClipObjective"
    CATEGORY = "RL"
    DESCRIPTION = (
        "PPO's clipped surrogate objective: min(r*A, clip(r, 1-eps, 1+eps)*A). Returns the "
        "unclipped term, the clipped term, the final objective and a mask of which samples "
        "were truncated, so you can see the safety belt engage. Clipping flattens the "
        "OBJECTIVE, it does not clamp the ratio -- it removes the incentive to step further, "
        "it does not forbid the step. Give it `ratio` directly, or new/old log-probs."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="advantages",
                data_type=DataType.TENSOR,
                description="[N] advantage per sample. Sign decides push-up vs push-down.",
            ),
            PortDefinition(
                name="ratio",
                data_type=DataType.TENSOR,
                description=(
                    "[N] probability ratio pi_new/pi_old. Supply this directly to reproduce a "
                    "worked example; otherwise leave it and give the two log-prob inputs."
                ),
                optional=True,
            ),
            PortDefinition(
                name="log_probs_new",
                data_type=DataType.TENSOR,
                description="[N] log pi_new(a|s). Used with log_probs_old when `ratio` is absent.",
                optional=True,
            ),
            PortDefinition(
                name="log_probs_old",
                data_type=DataType.TENSOR,
                description="[N] log pi_old(a|s).",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="objective", data_type=DataType.TENSOR, description="[N] min(unclipped, clipped) per sample."),
            PortDefinition(name="unclipped", data_type=DataType.TENSOR, description="[N] r * A, before any truncation."),
            PortDefinition(name="clipped", data_type=DataType.TENSOR, description="[N] clip(r, 1-eps, 1+eps) * A."),
            PortDefinition(name="ratio", data_type=DataType.TENSOR, description="[N] the ratio actually used."),
            PortDefinition(name="was_clipped", data_type=DataType.TENSOR, description="[N] 1 where the clipped branch won, 0 where it did not."),
            PortDefinition(name="loss", data_type=DataType.TENSOR, description="Scalar -mean(objective): maximise the objective by minimising this."),
            PortDefinition(name="clip_fraction", data_type=DataType.SCALAR, description="Fraction of samples that were truncated -- a standard PPO health metric."),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="epsilon",
                param_type=ParamType.FLOAT,
                default=0.2,
                min_value=0.0,
                description=(
                    "Clip radius. 0.1-0.2 is the usual range: too large and one lucky sample "
                    "can bolt, too small and learning crawls."
                ),
            ),
        ]

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        import torch

        advantages = inputs["advantages"]
        if not isinstance(advantages, torch.Tensor):
            advantages = torch.as_tensor(advantages, dtype=torch.float32)
        advantages = advantages.float().flatten()

        ratio = inputs.get("ratio")
        if ratio is None:
            lp_new = inputs.get("log_probs_new")
            lp_old = inputs.get("log_probs_old")
            if lp_new is None or lp_old is None:
                raise ValueError(
                    "PPOClipObjective: give either `ratio`, or both `log_probs_new` and "
                    "`log_probs_old` so the ratio can be computed as exp(new - old)."
                )
            lp_new = lp_new.flatten()
            lp_old = lp_old.flatten().to(lp_new.device)
            ratio = torch.exp(lp_new - lp_old)
        else:
            if not isinstance(ratio, torch.Tensor):
                ratio = torch.as_tensor(ratio, dtype=torch.float32)
            ratio = ratio.float().flatten()

        ratio = ratio.to(advantages.device)
        if ratio.shape != advantages.shape:
            raise ValueError(
                f"PPOClipObjective: ratio has shape {tuple(ratio.shape)} but advantages has "
                f"{tuple(advantages.shape)} -- they are per-sample and must line up."
            )

        eps = float(params.get("epsilon", 0.2))
        if eps < 0:
            raise ValueError(f"PPOClipObjective: epsilon must be >= 0, got {eps}.")

        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advantages
        objective = torch.minimum(unclipped, clipped)

        # "Was clipped" means the clipped branch actually decided the value --
        # not merely that the ratio left the interval. When the advantage is
        # zero both branches agree and nothing was truncated.
        was_clipped = (clipped < unclipped).to(torch.float32)

        loss = -objective.mean()
        result: dict[str, Any] = {
            "objective": objective,
            "unclipped": unclipped,
            "clipped": clipped,
            "ratio": ratio,
            "was_clipped": was_clipped,
            "loss": loss,
            "clip_fraction": float(was_clipped.mean()) if was_clipped.numel() else 0.0,
        }

        if context is not None and getattr(context, "verbose", False):
            from ...core.step_trace import StepRecorder

            recorder = StepRecorder()
            recorder.record(
                "ratio",
                "Probability ratio $r = \\pi_{new}/\\pi_{old}$. 1 means no change.",
                ratio=ratio,
            )
            recorder.record(
                "two_branches",
                f"Unclipped $rA$ against clipped $\\mathrm{{clip}}(r, {1-eps:.2f}, {1+eps:.2f})A$.",
                unclipped=unclipped,
                clipped=clipped,
            )
            recorder.record(
                "minimum",
                "Take the smaller. Past the interval the objective is flat, so stepping further pays nothing.",
                objective=objective,
                was_clipped=was_clipped,
            )
            result["__steps__"] = recorder.steps

        return result
