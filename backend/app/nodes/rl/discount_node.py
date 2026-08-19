"""DiscountNode -- fold a reward sequence into discounted returns.

C5-1 defines the return as

    G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

and then spends a section on what gamma buys: it is the dial that decides how
far the agent can see, and the reason an agent prefers the short path to the
goal without anyone telling it to. Both of those are claims a lesson should be
able to *show*, and showing them needs one node that turns a reward sequence
into the G it implies.

The fold runs backwards, which is the only sensible way to do it: G_t depends
on G_{t+1}, so one right-to-left pass computes every step's return in linear
time. Written forwards it is a quadratic mess that also obscures the recurrence

    G_t = r_t + gamma * G_{t+1}

which is the form worth recognising, because it is the same recurrence value
functions satisfy.

``episode_ids`` matters more than it looks. Rewards from a batch of rollouts
arrive concatenated, and discounting across the boundary would let one
episode's ending leak into the previous episode's returns. Given the ids, the
fold resets at each boundary; without them the whole tensor is treated as one
episode, which is correct only when it is one.
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


class DiscountNode(BaseNode):
    NODE_NAME = "Discount"
    CATEGORY = "RL"
    DESCRIPTION = (
        "Turn a reward sequence into discounted returns: G_t = r_t + gamma * G_{t+1}, "
        "folded backwards. gamma near 0 sees only the next step; near 1 carries a "
        "terminal reward all the way back to the first action. Connect episode_ids so "
        "the fold restarts at each episode boundary instead of discounting across it."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="rewards",
                data_type=DataType.TENSOR,
                description="[N] reward at each step.",
            ),
            PortDefinition(
                name="episode_ids",
                data_type=DataType.TENSOR,
                description=(
                    "[N] which episode each step belongs to. Optional; without it the "
                    "whole tensor is folded as a single episode."
                ),
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="returns",
                data_type=DataType.TENSOR,
                description="[N] discounted return G_t from each step onward.",
            ),
            PortDefinition(
                name="first_return",
                data_type=DataType.SCALAR,
                description="G at the first step of the first episode -- the number a lesson usually quotes.",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="gamma",
                param_type=ParamType.FLOAT,
                default=0.9,
                min_value=0.0,
                max_value=1.0,
                description=(
                    "Discount factor. 0 = only the immediate reward counts; 1 = no discount "
                    "at all. LLM-style tasks, where the reward lands only on the last step, "
                    "need it close to 1 or the earlier actions receive nothing."
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

        rewards = inputs["rewards"]
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.as_tensor(rewards, dtype=torch.float32)
        rewards = rewards.float().flatten()

        gamma = float(params.get("gamma", 0.9))
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"Discount: gamma must be between 0 and 1, got {gamma}.")

        ids = inputs.get("episode_ids")
        if ids is not None:
            if not isinstance(ids, torch.Tensor):
                ids = torch.as_tensor(ids)
            # Same-tensor comparisons below would survive a device mismatch,
            # but keeping the two aligned makes the intent explicit.
            ids = ids.flatten().to(rewards.device)
            if ids.shape[0] != rewards.shape[0]:
                raise ValueError(
                    f"Discount: episode_ids has {ids.shape[0]} entries but rewards has "
                    f"{rewards.shape[0]} -- they index the same steps and must match."
                )

        returns = torch.zeros_like(rewards)
        running = 0.0
        for t in range(rewards.shape[0] - 1, -1, -1):
            # A boundary resets the fold: the next step belongs to a different
            # episode, so its return is not this step's future.
            if ids is not None and t + 1 < rewards.shape[0] and ids[t + 1] != ids[t]:
                running = 0.0
            running = float(rewards[t]) + gamma * running
            returns[t] = running

        result: dict[str, Any] = {
            "returns": returns,
            "first_return": float(returns[0]) if returns.numel() else 0.0,
        }

        if context is not None and getattr(context, "verbose", False):
            from ...core.step_trace import StepRecorder

            recorder = StepRecorder()
            recorder.record(
                "rewards",
                "The reward at each step, before discounting.",
                rewards=rewards,
            )
            recorder.record(
                "backward_fold",
                "Folded right to left: $G_t = r_t + \\gamma G_{t+1}$.",
                scalars={"gamma": gamma},
                returns=returns,
            )
            result["__steps__"] = recorder.steps

        return result
