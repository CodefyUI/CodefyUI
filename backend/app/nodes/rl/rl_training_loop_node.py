"""RLTrainingLoopNode -- run the collect-and-update loop of an RL algorithm.

The RL category had every part of an agent except the loop: networks
(``DQN``, ``PPO``), environments (``EnvWrapper``, ``GridWorldEnv``), one
batch of experience (``PolicyRollout``) and the per-batch math
(``Discount``, ``PPOClipObjective``). Training an agent needs hundreds of
alternations of "act in the env" and "update the network", which a graph
that runs each node once cannot express. This node runs that alternation
for one of three algorithms:

``reinforce`` (Williams 1992)
    Play ``episodes_per_update`` episodes (1 by default) with the current
    policy, compute the discounted return G_t of every step, and step the
    optimizer on ``-sum_t log pi(a_t|s_t) * (G_t - b)`` averaged over the
    episodes, minus ``entropy_coef`` times the policy entropy.
    ``baseline=mean`` subtracts the batch's mean return; ``normalize_returns``
    also divides by its standard deviation. With one episode per update and
    a reward that arrives only at the end (GridWorld), normalising inside a
    single episode gives every early step a negative weight, so use several
    episodes per update or turn ``normalize_returns`` off there.

``dqn`` (Mnih et al. 2013, with the 2015 target network)
    Act epsilon-greedily on the model's Q-values, store every transition in
    a replay buffer, and after ``learning_starts`` env steps take one
    gradient step per env step on the Huber loss between Q(s, a) and
    ``r + gamma * max_a' Q_target(s', a')``. The target network is a copy of
    the model refreshed every ``target_update_every`` env steps;
    ``double_dqn`` picks a' with the online network (van Hasselt 2016).

``ppo`` (Schulman et al. 2017)
    Collect ``steps_per_update`` env steps (episodes may span two rollouts),
    estimate advantages with GAE(gamma, lambda) from the critic, then run
    ``ppo_epochs`` passes of shuffled minibatches over the clipped surrogate
    plus ``value_coef`` times the value loss minus ``entropy_coef`` times the
    entropy.

**One budget for all three: ``episodes``.** The run ends after that many
complete episodes, or earlier when the mean return of the last 20 episodes
reaches ``target_return``. A PPO rollout that completes the last episode
ends there and the steps collected so far form the final update, so
``episode_returns`` always has one entry per episode played.

**Models.** The output of the model decides what it can be used for. A
plain tensor is read as action logits (REINFORCE) or Q-values (DQN). A tuple
``(policy, value)`` is an actor-critic, which PPO requires; the ``PPO``
node's model returns probabilities there, and a first element that is a
probability distribution (non-negative, rows summing to 1) is read as one,
anything else as logits. PPO refuses a model without a value head, and DQN
refuses an actor-critic, each with a message naming the node to use.

**Environments.** Anything with ``reset()`` and ``step(action)``:
``GridWorldEnv`` (one-hot tensor observations, ``(obs, reward, done, info)``
steps) or a Gymnasium env from ``EnvWrapper`` (``(obs, info)`` resets,
``(obs, reward, terminated, truncated, info)`` steps). A Gymnasium env with
a ``Discrete`` observation space is one-hot encoded. Only discrete action
spaces are supported; a continuous one is refused with the env's id named.
Termination and truncation are kept apart: a time-limit truncation (or
GridWorld's ``timeout``) still bootstraps from the value of the last state.

**Reproducible.** Every random draw (action sampling, epsilon, replay
sampling, minibatch shuffles) comes from a local ``torch.Generator`` seeded
from ``seed``, and a Gymnasium env is reset with ``seed`` once at the start,
so the same model weights and seed give the same ``episode_returns`` on CPU.
The global torch RNG is never touched (#737).
"""

from __future__ import annotations

import math
import time
from typing import Any

from ...core.device_utils import DEVICE_PARAM_DESCRIPTION, DEVICE_PARAM_OPTIONS
from ...core.loop_control import ProgressThrottle
from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)

NODE = "RLTrainingLoop"

ALGORITHMS = ("reinforce", "ppo", "dqn")

#: How many recent episodes ``mean_return`` and ``target_return`` average.
RETURN_WINDOW = 20


# ── environment adapter ─────────────────────────────────────────────────


def _env_name(env: Any) -> str:
    spec = getattr(env, "spec", None)
    env_id = getattr(spec, "id", None)
    if env_id:
        return str(env_id)
    return type(getattr(env, "unwrapped", env)).__name__.lstrip("_")


class _Env:
    """Both env APIs behind one ``reset()`` / ``step()`` pair.

    ``reset()`` returns a float32 CPU tensor; ``step()`` returns
    ``(obs, reward, terminated, truncated)``.
    """

    def __init__(self, env: Any, seed: int) -> None:
        for name in ("reset", "step"):
            if not callable(getattr(env, name, None)):
                raise ValueError(
                    f"{NODE}: the `env` input has no {name}() and does not look "
                    "like an environment. Connect the `env` output of "
                    "GridWorldEnv or EnvWrapper.")
        self.env = env
        self.name = _env_name(env)
        self._seed = seed
        self._seeded = False
        self.n_actions = self._action_count()
        self._obs_onehot = self._discrete_observations()

    def _action_count(self) -> int | None:
        space = getattr(self.env, "action_space", None)
        if space is not None:
            n = getattr(space, "n", None)
            if n is None:
                raise ValueError(
                    f"{NODE}: env {self.name} has a {type(space).__name__} action "
                    "space. RLTrainingLoop supports discrete action spaces only "
                    "(REINFORCE, DQN and PPO here all pick one of N actions); "
                    "choose an env with a Discrete action space such as "
                    "CartPole-v1.")
            return int(n)
        n = getattr(self.env, "n_actions", None)
        return int(n) if n is not None else None

    def _discrete_observations(self) -> int | None:
        space = getattr(self.env, "observation_space", None)
        if space is not None and type(space).__name__ == "Discrete":
            return int(space.n)
        return None

    def _observation(self, obs: Any, where: str):
        import numpy as np
        import torch

        if self._obs_onehot is not None:
            vec = torch.zeros(self._obs_onehot, dtype=torch.float32)
            vec[int(obs)] = 1.0
            return vec
        if isinstance(obs, torch.Tensor):
            return obs.detach().to("cpu", torch.float32).reshape(-1)
        try:
            array = np.asarray(obs, dtype=np.float32)
        except (TypeError, ValueError):
            array = None
        if array is None or array.dtype == object or array.ndim == 0:
            raise ValueError(
                f"{NODE}: env {self.name}'s {where} gave an observation of type "
                f"{type(obs).__name__}, which is not an array of numbers.")
        return torch.from_numpy(array.reshape(-1).copy())

    def reset(self):
        if not self._seeded and self._accepts_seed():
            out = self.env.reset(seed=self._seed)
        else:
            out = self.env.reset()
        self._seeded = True
        if isinstance(out, tuple):
            if len(out) != 2 or not isinstance(out[1], dict):
                raise ValueError(
                    f"{NODE}: env {self.name}'s reset() returned a "
                    f"{len(out)}-tuple; expected obs or (obs, info).")
            out = out[0]
        return self._observation(out, "reset()")

    def _accepts_seed(self) -> bool:
        import inspect

        try:
            params = inspect.signature(self.env.reset).parameters
        except (TypeError, ValueError):
            return False
        return "seed" in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())

    def step(self, action: int):
        out = self.env.step(action)
        if not isinstance(out, tuple) or len(out) not in (4, 5):
            raise ValueError(
                f"{NODE}: env {self.name}'s step() must return (obs, reward, "
                "done, info) or (obs, reward, terminated, truncated, info).")
        if len(out) == 5:
            obs, reward, terminated, truncated, _info = out
            terminated, truncated = bool(terminated), bool(truncated)
        else:
            obs, reward, done, info = out
            # GridWorld names a time-limit ending "timeout": that one is a
            # truncation, the goal and a trap are real terminal states.
            outcome = info.get("outcome") if isinstance(info, dict) else None
            truncated = bool(done) and outcome == "timeout"
            terminated = bool(done) and not truncated
        return self._observation(obs, "step()"), float(reward), terminated, truncated


# ── model adapter ───────────────────────────────────────────────────────


def _policy_logits(head: Any):
    """Logits from a policy head that may be logits or probabilities."""
    import torch

    if head.numel() and bool((head >= 0).all()) and bool(
            torch.allclose(head.sum(-1), torch.ones_like(head.sum(-1)), atol=1e-4)):
        return torch.log(head.clamp_min(1e-12))
    return head


def _forward(model: Any, obs: Any):
    """``(policy_logits_or_q, value_or_None)`` for a batch of observations."""
    out = model(obs)
    if isinstance(out, (tuple, list)):
        if len(out) < 2:
            raise ValueError(
                f"{NODE}: the model returned a {len(out)}-tuple; an actor-critic "
                "returns (policy, value).")
        return out[0], out[1].reshape(out[1].shape[0], -1)[:, 0]
    return out, None


def _check_model(model: Any, env: _Env, obs: Any, algorithm: str, device: Any) -> None:
    import torch
    import torch.nn as nn

    first = next((m for m in model.modules()
                  if not list(m.children()) and list(m.parameters(recurse=False))),
                 None)
    if isinstance(first, nn.Linear) and first.in_features != obs.shape[0]:
        raise ValueError(
            f"{NODE}: env {env.name} gives observations of {obs.shape[0]} "
            f"features but the model expects {first.in_features}. Set the "
            f"model's state_dim (its input size) to {obs.shape[0]}.")
    with torch.no_grad():
        out = model(obs.unsqueeze(0).to(device))
    is_pair = isinstance(out, (tuple, list))
    head = out[0] if is_pair else out
    if algorithm == "ppo" and not is_pair:
        raise ValueError(
            f"{NODE}: algorithm 'ppo' needs an actor-critic model that returns "
            "(policy, value), such as the PPO node's. This model returns one "
            "tensor; connect a PPO node, or set algorithm to 'reinforce', which "
            "needs only the policy.")
    if algorithm == "dqn" and is_pair:
        raise ValueError(
            f"{NODE}: algorithm 'dqn' needs a model that returns one Q-value "
            "per action, such as the DQN node's. This model returns "
            "(policy, value), an actor-critic; use algorithm 'ppo' or "
            "'reinforce' with it.")
    if not isinstance(head, torch.Tensor) or head.ndim != 2:
        raise ValueError(
            f"{NODE}: the model must map [B, state_dim] to [B, action_dim] "
            f"scores; it returned {type(head).__name__} of shape "
            f"{tuple(getattr(head, 'shape', ()))}.")
    if env.n_actions is not None and head.shape[1] != env.n_actions:
        raise ValueError(
            f"{NODE}: env {env.name} has {env.n_actions} actions but the model "
            f"outputs {head.shape[1]} scores. Set the model's action_dim to "
            f"{env.n_actions}.")


# ── the node ────────────────────────────────────────────────────────────


def _algo_param(name, ptype, default, description, algorithms, **extra):
    return ParamDefinition(
        name=name, param_type=ptype, default=default, description=description,
        visible_when={"algorithm": list(algorithms)}, **extra)


class RLTrainingLoopNode(BaseNode):
    NODE_NAME = NODE
    CATEGORY = "RL"
    DESCRIPTION = "Train an agent in an env: REINFORCE, DQN or PPO"
    DETAILS = (
        "Runs the act-then-update loop for `episodes` episodes, or until the "
        "mean return of the last 20 episodes reaches target_return. "
        "REINFORCE takes one policy-gradient step every episodes_per_update "
        "episodes, DQN one "
        "replay-buffer step per env step with a target network, and PPO "
        "collects steps_per_update steps and optimises the clipped objective "
        "with GAE advantages. REINFORCE and DQN take a model that returns one "
        "score per action (DQN or SequentialModel); PPO needs an actor-critic "
        "that returns (policy, value), such as the PPO node's. The env is "
        "GridWorldEnv or a Gymnasium env from EnvWrapper with discrete "
        "actions. All randomness comes from seed, so a rerun with the same "
        "weights repeats. Small policies acting one step at a time usually "
        "run fastest on cpu. The live loss chart follows the run, and "
        "mean_return (plus solved_at_episode once reached) is recorded as a "
        "run metric, so a sweep can rank by it."
    )

    #: Trains the model in place and steps a stateful env.
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="model", data_type=DataType.MODEL,
                description=(
                    "The agent. Q-values or logits per action for REINFORCE and "
                    "DQN; (policy, value) for PPO.")),
            PortDefinition(
                name="env", data_type=DataType.ANY,
                description="Environment from GridWorldEnv or EnvWrapper (discrete actions)."),
            PortDefinition(
                name="optimizer", data_type=DataType.OPTIMIZER,
                description="Optimizer over the model's parameters (its learning rate is used)."),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL, description="The trained model."),
            PortDefinition(
                name="episode_returns", data_type=DataType.TENSOR,
                description="[episodes] undiscounted return of each episode played."),
            PortDefinition(
                name="losses", data_type=DataType.TENSOR,
                description=(
                    "One loss per update: per policy-gradient step (REINFORCE), per gradient "
                    "step (DQN) or per rollout (PPO, averaged over its minibatches).")),
            PortDefinition(
                name="metrics", data_type=DataType.ANY,
                description=(
                    "Dict: episodes, env_steps, updates, mean_return_last_20, "
                    "best_return, solved_at_episode (None if never), wall_time.")),
            PortDefinition(name="report", data_type=DataType.STRING, description="A short summary of the run."),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="algorithm", param_type=ParamType.SELECT, default="reinforce",
                options=list(ALGORITHMS),
                description="reinforce (Williams 1992), dqn (Mnih et al. 2013) or ppo (Schulman et al. 2017)."),
            ParamDefinition(
                name="episodes", param_type=ParamType.INT, default=500, min_value=1,
                description="Training budget in complete episodes."),
            ParamDefinition(
                name="gamma", param_type=ParamType.FLOAT, default=0.99,
                min_value=0.0, max_value=1.0, description="Discount factor."),
            ParamDefinition(
                name="max_steps_per_episode", param_type=ParamType.INT, default=0, min_value=0,
                description="Cut an episode after this many steps (a truncation). 0 keeps the env's own limit."),
            ParamDefinition(
                name="target_return", param_type=ParamType.FLOAT, default=0.0,
                description=(
                    "Stop once the mean return of the last 20 episodes reaches "
                    "this. 0 never stops early. CartPole-v1 counts as solved at 475.")),
            ParamDefinition(
                name="seed", param_type=ParamType.INT, default=0,
                description="Seeds the env reset, action sampling, replay and minibatch draws."),
            ParamDefinition(
                name="grad_clip_norm", param_type=ParamType.FLOAT, default=0.0, min_value=0.0,
                description="Clip the gradient norm before each step. 0 turns clipping off.",
                advanced=True),
            ParamDefinition(
                name="device", param_type=ParamType.SELECT, default="auto",
                options=list(DEVICE_PARAM_OPTIONS),
                description=DEVICE_PARAM_DESCRIPTION,
                advanced=True),
            ParamDefinition(
                name="log_interval", param_type=ParamType.INT, default=1, min_value=1,
                description="Report progress and record metrics every Nth episode.",
                advanced=True),
            # REINFORCE
            _algo_param(
                "baseline", ParamType.SELECT, "mean",
                "Subtracted from the returns: none, or the mean return of the update's batch.",
                ["reinforce"], options=["none", "mean"]),
            _algo_param(
                "normalize_returns", ParamType.BOOL, True,
                "Subtract the mean and divide by the standard deviation of the batch's returns.",
                ["reinforce"]),
            _algo_param(
                "episodes_per_update", ParamType.INT, 1,
                "Episodes per policy-gradient step. Several give a steadier "
                "gradient and a fairer baseline when the reward comes only at the end.",
                ["reinforce"], min_value=1),
            # PPO
            _algo_param(
                "steps_per_update", ParamType.INT, 512,
                "Env steps collected per PPO rollout.", ["ppo"], min_value=1),
            _algo_param(
                "ppo_epochs", ParamType.INT, 4,
                "Passes over each rollout.", ["ppo"], min_value=1),
            _algo_param(
                "minibatch_size", ParamType.INT, 64,
                "Steps per PPO gradient step.", ["ppo"], min_value=1),
            _algo_param(
                "clip_eps", ParamType.FLOAT, 0.2,
                "Ratio clip range epsilon.", ["ppo"], min_value=0.0),
            _algo_param(
                "gae_lambda", ParamType.FLOAT, 0.95,
                "GAE lambda: 1 is the Monte Carlo return, 0 one-step TD.",
                ["ppo"], min_value=0.0, max_value=1.0),
            _algo_param(
                "value_coef", ParamType.FLOAT, 0.5,
                "Weight of the value loss.", ["ppo"], min_value=0.0),
            _algo_param(
                "entropy_coef", ParamType.FLOAT, 0.01,
                "Weight of the entropy bonus, which keeps the policy exploring.",
                ["reinforce", "ppo"], min_value=0.0),
            # DQN
            _algo_param(
                "replay_size", ParamType.INT, 50000,
                "Replay buffer capacity in transitions.", ["dqn"], min_value=1),
            _algo_param(
                "batch_size", ParamType.INT, 64,
                "Transitions per gradient step.", ["dqn"], min_value=1),
            _algo_param(
                "epsilon_start", ParamType.FLOAT, 1.0,
                "Exploration rate at the first episode.", ["dqn"], min_value=0.0, max_value=1.0),
            _algo_param(
                "epsilon_end", ParamType.FLOAT, 0.05,
                "Exploration rate after the decay.", ["dqn"], min_value=0.0, max_value=1.0),
            _algo_param(
                "epsilon_decay_episodes", ParamType.INT, 100,
                "Episodes over which epsilon falls linearly to epsilon_end.", ["dqn"], min_value=1),
            _algo_param(
                "target_update_every", ParamType.INT, 500,
                "Copy the model into the target network every N env steps.", ["dqn"], min_value=1),
            _algo_param(
                "learning_starts", ParamType.INT, 1000,
                "Env steps of experience to collect before the first update.", ["dqn"], min_value=0),
            _algo_param(
                "double_dqn", ParamType.BOOL, True,
                "Pick the next action with the model and score it with the target network.",
                ["dqn"]),
        ]

    # ------------------------------------------------------------------

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        progress_callback: Any | None = None,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        import torch

        from ...core.device_utils import resolve_node_device, to_device
        from ...core.loop_control import interrupted_result, stop_checker
        from ..training.training_loop_node import _prepare_optimizer

        model = inputs.get("model")
        optimizer = inputs.get("optimizer")
        if model is None or not callable(getattr(model, "parameters", None)):
            raise ValueError(f"{NODE}: the `model` input must be a torch module.")
        if optimizer is None or not hasattr(optimizer, "step"):
            raise ValueError(f"{NODE}: the `optimizer` input must be an optimizer (from the Optimizer node).")

        algorithm = str(params.get("algorithm", "reinforce") or "reinforce").lower()
        if algorithm not in ALGORITHMS:
            raise ValueError(f"{NODE}: algorithm must be one of {', '.join(ALGORITHMS)}, got {algorithm!r}.")

        cfg = _Config(params)
        env = _Env(inputs.get("env"), cfg.seed)

        device = resolve_node_device(params.get("device"), context)
        model = to_device(model, device)
        optimizer, _mode = _prepare_optimizer(optimizer, model)
        model.train()

        first_obs = env.reset()
        _check_model(model, env, first_obs, algorithm, device)

        run = _Run(
            model=model, optimizer=optimizer, env=env, cfg=cfg, device=device,
            first_obs=first_obs, should_stop=stop_checker(context),
            progress_callback=progress_callback, context=context,
            gen=torch.Generator().manual_seed(cfg.seed),
        )
        {"reinforce": run.reinforce, "dqn": run.dqn, "ppo": run.ppo}[algorithm]()
        result = run.result(algorithm)
        if run.interrupted:
            result.update(interrupted_result(epoch=len(run.returns), phase="episode"))
        return result


class _Config:
    """The params, coerced once."""

    def __init__(self, p: dict[str, Any]) -> None:
        def num(name, default, cast=float):
            raw = p.get(name, default)
            if raw is None or raw == "":
                raw = default
            try:
                return cast(float(raw)) if cast is int else cast(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{NODE}: {name} must be a number, got {raw!r}.") from exc

        def flag(name, default):
            raw = p.get(name, default)
            if isinstance(raw, str):
                return raw.strip().lower() in ("1", "true", "yes", "on")
            return bool(raw)

        self.episodes = max(1, num("episodes", 500, int))
        self.gamma = num("gamma", 0.99)
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"{NODE}: gamma must be between 0 and 1, got {self.gamma}.")
        self.max_steps = max(0, num("max_steps_per_episode", 0, int))
        self.target_return = num("target_return", 0.0)
        self.seed = num("seed", 0, int)
        self.grad_clip = max(0.0, num("grad_clip_norm", 0.0))
        self.log_interval = max(1, num("log_interval", 1, int))
        self.baseline = str(p.get("baseline", "mean") or "mean").lower()
        if self.baseline not in ("none", "mean"):
            raise ValueError(f"{NODE}: baseline must be 'none' or 'mean', got {self.baseline!r}.")
        self.normalize_returns = flag("normalize_returns", True)
        self.episodes_per_update = max(1, num("episodes_per_update", 1, int))
        self.steps_per_update = max(1, num("steps_per_update", 512, int))
        self.ppo_epochs = max(1, num("ppo_epochs", 4, int))
        self.minibatch_size = max(1, num("minibatch_size", 64, int))
        self.clip_eps = num("clip_eps", 0.2)
        self.gae_lambda = num("gae_lambda", 0.95)
        self.value_coef = num("value_coef", 0.5)
        self.entropy_coef = num("entropy_coef", 0.01)
        self.replay_size = max(1, num("replay_size", 50000, int))
        self.batch_size = max(1, num("batch_size", 64, int))
        self.epsilon_start = num("epsilon_start", 1.0)
        self.epsilon_end = num("epsilon_end", 0.05)
        self.epsilon_decay = max(1, num("epsilon_decay_episodes", 100, int))
        self.target_update_every = max(1, num("target_update_every", 500, int))
        self.learning_starts = max(0, num("learning_starts", 1000, int))
        self.double_dqn = flag("double_dqn", True)


class _Run:
    """One training run: the shared bookkeeping and the three algorithms."""

    def __init__(self, *, model, optimizer, env, cfg, device, first_obs,
                 should_stop, progress_callback, context, gen) -> None:
        self.model = model
        self.optimizer = optimizer
        self.env = env
        self.cfg = cfg
        self.device = device
        self._pending_obs = first_obs
        self.should_stop = should_stop
        self.progress_callback = progress_callback
        self.throttle = ProgressThrottle(progress_callback)
        self.context = context
        self.gen = gen
        self.returns: list[float] = []
        self.losses: list[float] = []
        self.episode_losses: list[float] = []
        self._loss_mark = 0
        self._batch_start = 0
        self.env_steps = 0
        self.updates = 0
        self.solved_at: int | None = None
        self.interrupted = False
        self.started = time.monotonic()

    # ── shared pieces ──────────────────────────────────────────────────

    def reset(self):
        if self._pending_obs is not None:
            obs, self._pending_obs = self._pending_obs, None
            return obs
        return self.env.reset()

    def step(self, action: int, steps_in_episode: int):
        obs, reward, terminated, truncated = self.env.step(action)
        self.env_steps += 1
        if self.cfg.max_steps and steps_in_episode + 1 >= self.cfg.max_steps:
            truncated = truncated or not terminated
        return obs, reward, terminated, truncated

    def optimize(self, loss, *, record: bool = True) -> float:
        """One optimizer step. *record* books it as an update in ``losses``."""
        import torch

        self.optimizer.zero_grad()
        loss.backward()
        if self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        value = float(loss.detach())
        if record:
            self.updates += 1
            self.losses.append(value)
        return value

    def mean_return(self) -> float:
        window = self.returns[-RETURN_WINDOW:]
        return sum(window) / len(window) if window else 0.0

    def finish_episode(self, total: float) -> bool:
        """Book one episode; True when the run should end."""
        self.returns.append(total)
        n = len(self.returns)
        new_losses = self.losses[self._loss_mark:]
        self._loss_mark = len(self.losses)
        if new_losses:
            episode_loss = sum(new_losses) / len(new_losses)
        else:
            episode_loss = self.episode_losses[-1] if self.episode_losses else 0.0
        self.episode_losses.append(episode_loss)
        mean = self.mean_return()
        target = self.cfg.target_return
        if (self.solved_at is None and target != 0 and n >= RETURN_WINDOW
                and mean >= target):
            self.solved_at = n
            if self.context is not None:
                self.context.log_metric("solved_at_episode", n, n)
        done = self.solved_at is not None or n >= self.cfg.episodes
        if n % self.cfg.log_interval == 0 or done:
            self.report_progress(n, total, mean, episode_loss)
        if not done and self.should_stop():
            self.interrupted = True
            return True
        return done

    def report_progress(self, n: int, total: float, mean: float, loss: float) -> None:
        if self.context is not None:
            self.context.log_metric("episode_return", total, n)
            self.context.log_metric("mean_return", mean, n)
        if self.progress_callback:
            # Rate-limited like TrainingLoop's batch frames: an RL run plays
            # hundreds of episodes a second on CartPole, and each frame
            # carries the whole per-episode loss list. The last one always
            # goes out so the chart ends on the final episode.
            final = n >= self.cfg.episodes or self.solved_at is not None
            emit = self.progress_callback if final else self.throttle.emit
            emit({
                "event": "epoch",
                "epoch": n,
                "total_epochs": self.cfg.episodes,
                "loss": round(loss, 6),
                "losses": [round(v, 6) for v in self.episode_losses],
                "episode_return": round(total, 6),
                "mean_return": round(mean, 6),
            })

    def stop_requested(self) -> bool:
        if self.should_stop():
            self.interrupted = True
            return True
        return False

    def sample(self, logits) -> int:
        import torch

        probs = torch.softmax(logits.detach().float().cpu(), dim=-1)
        return int(torch.multinomial(probs, 1, generator=self.gen).item())

    def discounted(self, rewards: list[float]) -> list[float]:
        out = [0.0] * len(rewards)
        running = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            running = rewards[t] + self.cfg.gamma * running
            out[t] = running
        return out

    # ── REINFORCE ──────────────────────────────────────────────────────

    def reinforce(self) -> None:
        import torch

        cfg = self.cfg
        while True:
            # One update per ``episodes_per_update`` episodes; the baseline
            # and the normalisation are taken over the whole batch, so with
            # several episodes a successful one keeps its positive weight
            # even when its reward arrives only at the end.
            states, actions, returns = [], [], []
            finished = False
            for _ in range(cfg.episodes_per_update):
                obs = self.reset()
                rewards = []
                done = False
                while not done:
                    if self.stop_requested():
                        return
                    with torch.no_grad():
                        head, _ = _forward(self.model, obs.unsqueeze(0).to(self.device))
                    action = self.sample(_policy_logits(head)[0])
                    states.append(obs)
                    actions.append(action)
                    obs, reward, terminated, truncated = self.step(action, len(rewards))
                    rewards.append(reward)
                    done = terminated or truncated
                returns.extend(self.discounted(rewards))
                if self.finish_episode(sum(rewards)):
                    finished = True
                    break
            n_episodes = max(1, len(self.returns) - self._batch_start)
            self._batch_start = len(self.returns)

            g = torch.tensor(returns, dtype=torch.float32)
            if cfg.baseline == "mean" or cfg.normalize_returns:
                g = g - g.mean()
            if cfg.normalize_returns and len(g) > 1:
                g = g / (g.std() + 1e-8)
            batch = torch.stack(states).to(self.device)
            head, _ = _forward(self.model, batch)
            log_probs = torch.log_softmax(_policy_logits(head), dim=-1)
            taken = log_probs.gather(
                1, torch.tensor(actions, device=self.device).unsqueeze(1)).squeeze(1)
            # sum over steps, mean over episodes: Williams' estimator.
            loss = -(taken * g.to(self.device)).sum() / n_episodes
            if cfg.entropy_coef:
                # Per step, on the same sum-over-steps scale as the gradient.
                entropy = -(log_probs.exp() * log_probs).sum() / n_episodes
                loss = loss - cfg.entropy_coef * entropy
            self.optimize(loss)
            if finished or self.interrupted:
                return

    # ── DQN ────────────────────────────────────────────────────────────

    def dqn(self) -> None:
        import copy

        import torch
        import torch.nn.functional as F

        cfg = self.cfg
        target = copy.deepcopy(self.model)
        target.eval()
        for p in target.parameters():
            p.requires_grad_(False)

        dim = self._pending_obs.shape[0]
        cap = cfg.replay_size
        buf_s = torch.zeros(cap, dim)
        buf_a = torch.zeros(cap, dtype=torch.long)
        buf_r = torch.zeros(cap)
        buf_s2 = torch.zeros(cap, dim)
        buf_d = torch.zeros(cap)
        size = pos = 0

        while True:
            episode = len(self.returns)
            frac = min(1.0, episode / cfg.epsilon_decay)
            epsilon = cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)
            obs = self.reset()
            total, steps, done = 0.0, 0, False
            while not done:
                if self.stop_requested():
                    return
                if float(torch.rand((), generator=self.gen)) < epsilon:
                    n_actions = self.env.n_actions or self._n_outputs(obs)
                    action = int(torch.randint(n_actions, (), generator=self.gen))
                else:
                    with torch.no_grad():
                        q, _ = _forward(self.model, obs.unsqueeze(0).to(self.device))
                    action = int(q[0].argmax())
                next_obs, reward, terminated, truncated = self.step(action, steps)
                buf_s[pos], buf_a[pos], buf_r[pos] = obs, action, reward
                buf_s2[pos], buf_d[pos] = next_obs, float(terminated)
                pos = (pos + 1) % cap
                size = min(size + 1, cap)
                obs = next_obs
                total += reward
                steps += 1
                done = terminated or truncated

                if self.env_steps > cfg.learning_starts and size >= cfg.batch_size:
                    idx = torch.randint(size, (cfg.batch_size,), generator=self.gen)
                    s = buf_s[idx].to(self.device)
                    a = buf_a[idx].to(self.device)
                    r = buf_r[idx].to(self.device)
                    s2 = buf_s2[idx].to(self.device)
                    d = buf_d[idx].to(self.device)
                    with torch.no_grad():
                        q_next_target = target(s2)
                        if cfg.double_dqn:
                            a2 = self.model(s2).argmax(dim=1, keepdim=True)
                            q_next = q_next_target.gather(1, a2).squeeze(1)
                        else:
                            q_next = q_next_target.max(dim=1).values
                        y = r + cfg.gamma * (1.0 - d) * q_next
                    q = self.model(s).gather(1, a.unsqueeze(1)).squeeze(1)
                    self.optimize(F.smooth_l1_loss(q, y))
                if self.env_steps % cfg.target_update_every == 0:
                    target.load_state_dict(self.model.state_dict())
            if self.finish_episode(total):
                return

    def _n_outputs(self, obs) -> int:
        import torch

        with torch.no_grad():
            q, _ = _forward(self.model, obs.unsqueeze(0).to(self.device))
        return int(q.shape[1])

    # ── PPO ────────────────────────────────────────────────────────────

    def ppo(self) -> None:
        import torch

        cfg = self.cfg
        obs = self.reset()
        total, steps = 0.0, 0
        finished = False
        while not finished:
            states, actions, logps, values, rewards = [], [], [], [], []
            dones = []
            for _ in range(cfg.steps_per_update):
                if self.stop_requested():
                    return
                with torch.no_grad():
                    head, value = _forward(self.model, obs.unsqueeze(0).to(self.device))
                logits = torch.log_softmax(_policy_logits(head)[0].float().cpu(), dim=-1)
                action = self.sample(logits)
                next_obs, reward, terminated, truncated = self.step(action, steps)
                states.append(obs)
                actions.append(action)
                logps.append(float(logits[action]))
                values.append(float(value[0]))
                total += reward
                steps += 1
                boot = 0.0
                if truncated and not terminated:
                    # A time limit is not a terminal state: credit the value
                    # of where the agent was cut off.
                    with torch.no_grad():
                        _, v_cut = _forward(self.model, next_obs.unsqueeze(0).to(self.device))
                    boot = float(v_cut[0])
                rewards.append(reward + cfg.gamma * boot)
                dones.append(terminated or truncated)
                if terminated or truncated:
                    if self.finish_episode(total):
                        finished = True
                        break
                    obs = self.reset()
                    total, steps = 0.0, 0
                else:
                    obs = next_obs
            if not states:
                return

            # GAE over the rollout, bootstrapping the unfinished tail.
            last_value = 0.0
            if not dones[-1]:
                with torch.no_grad():
                    _, v_last = _forward(self.model, obs.unsqueeze(0).to(self.device))
                last_value = float(v_last[0])
            n = len(states)
            adv = [0.0] * n
            running = 0.0
            for t in range(n - 1, -1, -1):
                next_v = last_value if t == n - 1 else values[t + 1]
                nonterminal = 0.0 if dones[t] else 1.0
                delta = rewards[t] + cfg.gamma * next_v * nonterminal - values[t]
                running = delta + cfg.gamma * cfg.gae_lambda * nonterminal * running
                adv[t] = running
            adv_t = torch.tensor(adv, dtype=torch.float32)
            ret_t = adv_t + torch.tensor(values, dtype=torch.float32)
            s_t = torch.stack(states)
            a_t = torch.tensor(actions, dtype=torch.long)
            lp_t = torch.tensor(logps, dtype=torch.float32)

            minibatch_losses = []
            for _epoch in range(cfg.ppo_epochs):
                order = torch.randperm(n, generator=self.gen)
                for start in range(0, n, cfg.minibatch_size):
                    idx = order[start:start + cfg.minibatch_size]
                    mb_adv = adv_t[idx]
                    if len(idx) > 1:
                        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    head, value = _forward(self.model, s_t[idx].to(self.device))
                    log_probs = torch.log_softmax(_policy_logits(head), dim=-1)
                    new_lp = log_probs.gather(1, a_t[idx].to(self.device).unsqueeze(1)).squeeze(1)
                    ratio = torch.exp(new_lp - lp_t[idx].to(self.device))
                    mb_adv = mb_adv.to(self.device)
                    surrogate = torch.min(
                        ratio * mb_adv,
                        torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * mb_adv)
                    value_loss = (value - ret_t[idx].to(self.device)).pow(2).mean()
                    entropy = -(log_probs.exp() * log_probs).sum(-1).mean()
                    loss = (-surrogate.mean() + cfg.value_coef * value_loss
                            - cfg.entropy_coef * entropy)
                    minibatch_losses.append(self.optimize(loss, record=False))
            # One entry per PPO update in ``losses``: the minibatch mean.
            self.losses.append(sum(minibatch_losses) / len(minibatch_losses))
            self.updates += 1

    # ── result ─────────────────────────────────────────────────────────

    def result(self, algorithm: str) -> dict[str, Any]:
        import torch

        wall = time.monotonic() - self.started
        n = len(self.returns)
        mean = self.mean_return()
        best = max(self.returns) if self.returns else None
        metrics = {
            "algorithm": algorithm,
            "episodes": n,
            "env_steps": self.env_steps,
            "updates": self.updates,
            "mean_return_last_20": mean,
            "best_return": best,
            "solved_at_episode": self.solved_at,
            "wall_time": wall,
            "interrupted": self.interrupted,
        }
        if self.context is not None and n and n % self.cfg.log_interval != 0:
            self.context.log_metric("mean_return", mean, n)
        lines = [
            f"{algorithm} on {self.env.name}: {n} episodes, {self.env_steps} env steps, "
            f"{self.updates} updates in {wall:.1f} s.",
            f"Mean return of the last {min(n, RETURN_WINDOW)} episodes: {mean:.2f}"
            + (f"; best episode {best:.2f}." if best is not None else "."),
        ]
        if self.solved_at is not None:
            lines.append(
                f"Reached target_return {self.cfg.target_return:g} at episode {self.solved_at}.")
        elif self.cfg.target_return != 0:
            lines.append(f"Did not reach target_return {self.cfg.target_return:g}.")
        if self.interrupted:
            lines.append("Stopped on request.")
        if not math.isfinite(mean):
            lines.append("The returns are not finite; lower the learning rate.")
        return {
            "model": self.model,
            "episode_returns": torch.tensor(self.returns, dtype=torch.float32),
            "losses": torch.tensor(self.losses, dtype=torch.float32),
            "metrics": metrics,
            "report": "\n".join(lines),
        }
