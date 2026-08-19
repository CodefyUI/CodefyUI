"""PolicyRolloutNode -- run a policy in an environment and hand back the trajectories.

This is the node the RL category was missing. Every algorithm in C5 starts
with the same sentence -- "use the current policy to collect a batch of
trajectories" -- and until now nothing on the canvas could do it. The four
chapter examples all faked this step with a ``TensorInput``, which meant the
one thing that makes RL *reinforcement* learning, the loop through an
environment, was the one thing a student never saw run.

It returns a batch of ``episodes`` rollouts, flattened into the four tensors
every downstream RL node wants:

    states     [N, state_dim]  the observation at each step
    actions    [N]             the action taken there
    rewards    [N]             the reward that came back
    logits     [N, action_dim] the policy's scores at that state

plus the per-episode bookkeeping (``returns``, ``episode_lengths``,
``episode_ids``) that group-based methods need to tell one rollout from
another. ``logits`` is what makes the batch usable by
``Edu-PolicyGradient`` and ``PPOClipObjective`` -- an advantage is meaningless
without the policy's own scores to weight.

**Sampling, not argmax.** Actions are drawn from ``softmax(logits)``, so two
rollouts of the same policy differ. That is the exploration C5-1 describes, and
it is also the entire basis of GRPO: sample the same start K times and the
spread between the results is the baseline.

**One seed, reproducible batch.** The seed fixes the whole batch, so a lesson
can quote a number and a student can reproduce it. Change the seed and you get
a different batch from the same policy, which is the point of the
``exploration`` demonstration.

The policy is any module mapping ``[B, state_dim]`` to action scores. A bare
``Linear`` is the tabular policy the textbook draws as arrows; a ``PPO``
actor-critic works too -- if it returns a tuple, the first element is taken as
the action distribution and probabilities are converted back to logits.
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


class PolicyRolloutNode(BaseNode):
    NODE_NAME = "PolicyRollout"
    CATEGORY = "RL"
    DESCRIPTION = (
        "Run a policy in an environment for N episodes and return the trajectories: "
        "states, actions, rewards and the policy's logits at each step, plus per-episode "
        "returns and lengths. Actions are SAMPLED from softmax(logits), so repeated "
        "rollouts of the same policy differ -- that spread is what GRPO's group baseline "
        "is built from. Feeds Edu-PolicyGradient, PPOClipObjective and GroupRelativeAdvantage."
    )

    #: Rolls a stochastic policy through a stateful env; a cached batch would
    #: silently be the previous run's experience.
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="model",
                data_type=DataType.MODEL,
                description="The policy. Maps [B, state_dim] to action scores over action_dim actions.",
            ),
            PortDefinition(
                name="env",
                data_type=DataType.ANY,
                description="Environment exposing reset() and step(action), e.g. from GridWorldEnv.",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="states", data_type=DataType.TENSOR, description="[N, state_dim] observation at each step."),
            PortDefinition(name="actions", data_type=DataType.TENSOR, description="[N] action taken at each step (int64)."),
            PortDefinition(name="rewards", data_type=DataType.TENSOR, description="[N] reward received at each step."),
            PortDefinition(name="logits", data_type=DataType.TENSOR, description="[N, action_dim] the policy's scores at each state."),
            PortDefinition(
                name="log_probs",
                data_type=DataType.TENSOR,
                description=(
                    "[N] log pi(a_t | s_t) for the action actually taken, under the behaviour "
                    "policy. This is PPO's `log_probs_old`: the record of what the policy "
                    "thought at sampling time, which the ratio is measured against."
                ),
            ),
            PortDefinition(name="returns", data_type=DataType.TENSOR, description="[episodes] undiscounted total reward per episode."),
            PortDefinition(name="episode_lengths", data_type=DataType.TENSOR, description="[episodes] number of steps in each episode."),
            PortDefinition(name="episode_ids", data_type=DataType.TENSOR, description="[N] which episode each step belongs to."),
            PortDefinition(name="success_rate", data_type=DataType.SCALAR, description="Fraction of episodes that ended at the goal."),
            PortDefinition(name="report", data_type=DataType.STRING, description="Per-episode summary as text -- connect a Print."),
            PortDefinition(
                name="trajectory",
                data_type=DataType.STRING,
                description=(
                    "The FIRST episode step by step: state, action, reward. This is the table "
                    "that makes a sparse reward visible -- T actions, one non-zero reward, and "
                    "no indication of which action earned it."
                ),
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="episodes",
                param_type=ParamType.INT,
                default=1,
                min_value=1,
                description=(
                    "How many rollouts to collect. 1 gives a single trajectory to read; "
                    "set it to K for a GRPO group."
                ),
            ),
            ParamDefinition(
                name="temperature",
                param_type=ParamType.FLOAT,
                default=1.0,
                min_value=0.01,
                description=(
                    "Divides the logits before sampling. Low = exploit (nearly always the "
                    "top action), high = explore. This is C5-1's exploration dial."
                ),
            ),
            ParamDefinition(
                name="seed",
                param_type=ParamType.INT,
                default=0,
                description="Fixes the whole batch, so a quoted number is reproducible.",
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

        model = inputs["model"]
        env = inputs["env"]

        for name in ("reset", "step"):
            if not hasattr(env, name):
                raise ValueError(
                    f"PolicyRollout: the `env` input has no {name}() -- it does not look like "
                    "an environment. GridWorldEnv's `env` output is what belongs here."
                )

        episodes = int(params.get("episodes", 1))
        temperature = float(params.get("temperature", 1.0))
        if temperature <= 0:
            raise ValueError(f"PolicyRollout: temperature must be > 0, got {temperature}.")
        gen = torch.Generator().manual_seed(int(params.get("seed", 0)))

        was_training = getattr(model, "training", False)
        if hasattr(model, "eval"):
            model.eval()

        # The environment is a plain object that builds CPU tensors; the policy
        # may live on cuda/mps. Bridge the two here rather than making every
        # environment device-aware, and record everything back on CPU so no
        # downstream node has to know where the policy happened to sit.
        try:
            model_device = next(model.parameters()).device
        except (StopIteration, AttributeError, TypeError):
            model_device = torch.device("cpu")

        s_all, a_all, r_all, l_all, ep_all, lp_all = [], [], [], [], [], []
        returns, lengths, outcomes = [], [], []

        try:
            for ep in range(episodes):
                obs = env.reset()
                done = False
                total, steps = 0.0, 0
                outcome = "timeout"
                while not done:
                    with torch.no_grad():
                        out = model(obs.unsqueeze(0).to(model_device))
                    # A PPO-style actor-critic returns (probs, value); take the
                    # distribution and put it back on a logit scale so every
                    # downstream node sees one consistent kind of number.
                    if isinstance(out, tuple):
                        out = out[0]
                        logits = torch.log(out.clamp_min(1e-12))
                    else:
                        logits = out
                    logits = logits.squeeze(0).detach().cpu()

                    probs = torch.softmax(logits / temperature, dim=-1)
                    action = int(torch.multinomial(probs, 1, generator=gen).item())

                    s_all.append(obs)
                    a_all.append(action)
                    l_all.append(logits)
                    ep_all.append(ep)
                    # Recorded at sampling time on purpose: once the policy is
                    # updated this number is no longer recoverable, and it is
                    # exactly the denominator of PPO's ratio.
                    lp_all.append(torch.log(probs[action].clamp_min(1e-12)))

                    obs, reward, done, info = env.step(action)
                    r_all.append(float(reward))
                    total += float(reward)
                    steps += 1
                    outcome = info.get("outcome", outcome) if isinstance(info, dict) else outcome

                returns.append(total)
                lengths.append(steps)
                outcomes.append(outcome)
        finally:
            if was_training and hasattr(model, "train"):
                model.train()

        lines = [
            f"episode {i}: {lengths[i]:3d} steps, return {returns[i]:+.3f}, ended at {outcomes[i]}"
            for i in range(episodes)
        ]
        n_goal = sum(1 for o in outcomes if o == "goal")

        # The first episode, step by step. States are reported as the index of
        # the hot entry, which is what a one-hot observation means and is the
        # only reading that stays env-agnostic -- naming the cells or the
        # directions would bake gridworld into a node that does not know it.
        first_len = lengths[0] if lengths else 0
        traj = ["step | state | action | reward"]
        for t in range(first_len):
            obs = s_all[t]
            state = int(obs.argmax()) if obs.ndim == 1 else t
            traj.append(f"{t:4d} | {state:5d} | {a_all[t]:6d} | {r_all[t]:+.3f}")
        traj.append(f"episode 0 ended at {outcomes[0] if outcomes else 'n/a'}")

        return {
            "states": torch.stack(s_all),
            "actions": torch.tensor(a_all, dtype=torch.long),
            "rewards": torch.tensor(r_all, dtype=torch.float32),
            "logits": torch.stack(l_all),
            "log_probs": torch.stack(lp_all),
            "returns": torch.tensor(returns, dtype=torch.float32),
            "episode_lengths": torch.tensor(lengths, dtype=torch.long),
            "episode_ids": torch.tensor(ep_all, dtype=torch.long),
            "success_rate": n_goal / episodes,
            "report": "\n".join(lines),
            "trajectory": "\n".join(traj),
        }
