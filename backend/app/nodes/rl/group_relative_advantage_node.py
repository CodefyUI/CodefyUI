"""GroupRelativeAdvantageNode -- GRPO's one change to PPO, on its own.

Put C5-4's algorithm next to C5-2's and only step 3 differs. PPO trains a
critic to estimate "what this state is normally worth" and subtracts it; GRPO
samples the same prompt K times and subtracts the mean of that group. Steps 4
and 5 -- the clip, the KL -- are untouched.

So the node is deliberately small. It takes K rewards and returns K
advantages, and the arithmetic is one line:

    A_i = r_i - mean(r)

That line is the entire reason a model can be deleted from the training setup.
Worth dwelling on: the baseline here is *computed*, not *estimated*. A critic
is a network that can be wrong, that needs its own optimiser, and whose error
shows up as noise in every advantage it produces. A mean cannot be wrong about
the sample it is the mean of.

Two properties are worth checking on sight and are cheap to assert:

  * the advantages sum to zero, because subtracting the mean is what that
    means -- the fastest way to catch a mis-wired baseline;
  * if every sample in a group scores the same, every advantage is zero and
    the group teaches nothing. That is not a bug, it is the constraint GRPO
    runs under: the task has to be one the policy sometimes-but-not-always
    gets right, and K has to be big enough for the mean to be worth trusting.

``normalize`` divides by the group's standard deviation, which C5-4 mentions
as optional. It is off by default because dividing makes the numbers stop
matching a hand calculation, and matching a hand calculation is most of the
point at this size.
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


class GroupRelativeAdvantageNode(BaseNode):
    NODE_NAME = "GroupRelativeAdvantage"
    CATEGORY = "RL"
    DESCRIPTION = (
        "GRPO's baseline: sample the same prompt K times, then A_i = r_i - mean(r). This one "
        "line is what replaces PPO's critic -- the baseline is COMPUTED from the group, not "
        "estimated by a network that can be wrong. Advantages always sum to zero (a quick "
        "wiring check). A group where every sample scores the same yields all-zero advantages "
        "and teaches nothing, which is why the task must be one the policy sometimes gets right."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="rewards",
                data_type=DataType.TENSOR,
                description="[K] one reward per sample in the group.",
            ),
            PortDefinition(
                name="group_ids",
                data_type=DataType.TENSOR,
                description=(
                    "[K] which group each sample belongs to, when several prompts are batched "
                    "together. Optional; without it everything is one group."
                ),
                optional=True,
            ),
            PortDefinition(
                name="expand_index",
                data_type=DataType.TENSOR,
                description=(
                    "[N] which entry of `rewards` each downstream sample belongs to -- "
                    "PolicyRollout's `episode_ids`. Real GRPO scores a whole response and then "
                    "gives EVERY token in it that response's advantage; this input is that "
                    "broadcast, and without it the per-episode advantages cannot be wired into "
                    "a per-step objective."
                ),
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="advantages", data_type=DataType.TENSOR, description="[K] r_i minus the group mean."),
            PortDefinition(name="baseline", data_type=DataType.TENSOR, description="[K] the group mean used for each sample."),
            PortDefinition(name="group_mean", data_type=DataType.SCALAR, description="Mean of the first group -- the number a lesson quotes."),
            PortDefinition(name="group_std", data_type=DataType.SCALAR, description="Standard deviation of the first group; 0 means no signal."),
            PortDefinition(name="report", data_type=DataType.STRING, description="Per-sample reward, baseline and advantage as text."),
            PortDefinition(
                name="advantages_expanded",
                data_type=DataType.TENSOR,
                description=(
                    "[N] the advantages spread back over the samples named by `expand_index`. "
                    "Equals `advantages` when that input is absent."
                ),
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="normalize",
                param_type=ParamType.BOOL,
                default=False,
                description=(
                    "Also divide by the group standard deviation. Off by default so the "
                    "numbers still match a hand calculation."
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
        if rewards.numel() == 0:
            raise ValueError("GroupRelativeAdvantage: `rewards` is empty -- nothing to compare.")

        ids = inputs.get("group_ids")
        if ids is None:
            ids = torch.zeros_like(rewards, dtype=torch.long)
        else:
            if not isinstance(ids, torch.Tensor):
                ids = torch.as_tensor(ids)
            ids = ids.flatten().long().to(rewards.device)
            if ids.shape[0] != rewards.shape[0]:
                raise ValueError(
                    f"GroupRelativeAdvantage: group_ids has {ids.shape[0]} entries but rewards "
                    f"has {rewards.shape[0]} -- they index the same samples."
                )

        normalize = bool(params.get("normalize", False))
        baseline = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        for g in ids.unique():
            mask = ids == g
            group = rewards[mask]
            mean = group.mean()
            baseline[mask] = mean
            adv = group - mean
            if normalize:
                # Unbiased std needs 2+ samples, and a constant group has std 0;
                # either way there is nothing to scale by, so leave it alone
                # rather than dividing by an epsilon and inventing signal.
                std = group.std(unbiased=False)
                if std > 1e-8:
                    adv = adv / std
            advantages[mask] = adv

        first = ids == ids[0]
        first_group = rewards[first]

        lines = [
            f"sample {i}: reward {float(rewards[i]):+.4f}  baseline {float(baseline[i]):+.4f}  "
            f"advantage {float(advantages[i]):+.4f}"
            for i in range(rewards.numel())
        ]
        lines.append(f"advantages sum to {float(advantages.sum()):+.6f} (should be 0)")

        expand = inputs.get("expand_index")
        if expand is None:
            expanded = advantages
        else:
            if not isinstance(expand, torch.Tensor):
                expand = torch.as_tensor(expand)
            expand = expand.flatten().long().to(advantages.device)
            if expand.numel() and (int(expand.min()) < 0 or int(expand.max()) >= rewards.numel()):
                raise ValueError(
                    f"GroupRelativeAdvantage: expand_index refers to entry "
                    f"{int(expand.max())} but only {rewards.numel()} rewards were given."
                )
            expanded = advantages[expand]

        result: dict[str, Any] = {
            "advantages": advantages,
            "advantages_expanded": expanded,
            "baseline": baseline,
            "group_mean": float(first_group.mean()),
            "group_std": float(first_group.std(unbiased=False)),
            "report": "\n".join(lines),
        }

        if context is not None and getattr(context, "verbose", False):
            from ...core.step_trace import StepRecorder

            recorder = StepRecorder()
            recorder.record(
                "rewards",
                "One reward per sample in the group.",
                rewards=rewards,
            )
            recorder.record(
                "baseline",
                "The group mean. No critic estimated this -- it is the average of the numbers above.",
                scalars={"group_mean": result["group_mean"]},
                baseline=baseline,
            )
            recorder.record(
                "advantages",
                "$A_i = r_i - \\bar r$. Positive gets pushed up, negative gets pushed down, and they sum to zero.",
                advantages=advantages,
            )
            result["__steps__"] = recorder.steps

        return result
