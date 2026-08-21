"""GridWorldEnvNode -- the textbook gridworld, with no gym dependency.

The RL category had exactly one environment, ``EnvWrapper``, and it reaches
for gymnasium and hands back CartPole. That is a fine control benchmark and a
poor teaching one: its state is four continuous floats, its reward is +1 every
step until failure, and nothing about it can be checked by hand.

Gridworld is the environment RL courses actually teach on, for reasons that
all matter here. The state space is small enough to enumerate, so ``V(s)`` and
``Q(s, a)`` can be written out per cell. The reward is sparse and terminal,
which is the structure that makes credit assignment hard and is therefore the
thing worth showing. And every number a lesson quotes -- a return, a discount,
an advantage -- is small enough that a student can recompute it on paper and
catch a wiring mistake.

Layout, defaults chosen to match the canonical picture: the agent starts at
the top-left, the goal sits at the bottom-right, and traps sit wherever
``traps`` says. Reaching the goal ends the episode with ``goal_reward``;
stepping into a trap ends it with ``trap_reward``; every other step pays
``step_reward`` (0 by default, which is what makes the reward sparse). Walking
into a wall costs a step and leaves the agent where it was.

**The environment is a plain object, not a gym.Env.** It exposes ``reset()``
and ``step(action)`` and nothing else, because those two are the whole of the
C5-1 loop and anything more is surface a lesson has to explain away.

State is returned as a one-hot vector over cells, so a policy can be a single
``Linear`` layer and still be exactly expressive enough -- one weight per
(cell, action) pair, which is the tabular policy the textbook draws as arrows
on the grid.
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


class _GridWorld:
    """A size x size grid. Start top-left, goal bottom-right, traps in between."""

    def __init__(
        self,
        size: int,
        traps: list[tuple[int, int]],
        step_reward: float,
        goal_reward: float,
        trap_reward: float,
        max_steps: int,
    ) -> None:
        self.size = size
        self.traps = set(traps)
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.trap_reward = trap_reward
        self.max_steps = max_steps
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.n_states = size * size
        self.n_actions = 4  # 0=up 1=right 2=down 3=left
        self._pos = self.start
        self._steps = 0

    # ── observation ────────────────────────────────────────────────────
    def _one_hot(self, pos: tuple[int, int]):
        import torch

        v = torch.zeros(self.n_states, dtype=torch.float32)
        v[pos[0] * self.size + pos[1]] = 1.0
        return v

    # ── gym-shaped surface, minus gym ──────────────────────────────────
    def reset(self):
        self._pos = self.start
        self._steps = 0
        return self._one_hot(self._pos)

    def step(self, action: int):
        """Return (observation, reward, done, info)."""
        r, c = self._pos
        if action == 0:
            r -= 1
        elif action == 1:
            c += 1
        elif action == 2:
            r += 1
        elif action == 3:
            c -= 1
        else:
            raise ValueError(f"GridWorld: action must be 0-3, got {action!r}")

        # A wall is not a special case with its own reward -- bumping into one
        # simply costs the step, which is what makes "the long way round" cost
        # something without needing a penalty the student has to reason about.
        if 0 <= r < self.size and 0 <= c < self.size:
            self._pos = (r, c)

        self._steps += 1

        if self._pos == self.goal:
            return self._one_hot(self._pos), self.goal_reward, True, {"outcome": "goal"}
        if self._pos in self.traps:
            return self._one_hot(self._pos), self.trap_reward, True, {"outcome": "trap"}
        if self._steps >= self.max_steps:
            return self._one_hot(self._pos), self.step_reward, True, {"outcome": "timeout"}
        return self._one_hot(self._pos), self.step_reward, False, {"outcome": "step"}

    def describe(self) -> str:
        rows = []
        for r in range(self.size):
            cells = []
            for c in range(self.size):
                if (r, c) == self.start:
                    cells.append("S")
                elif (r, c) == self.goal:
                    cells.append("G")
                elif (r, c) in self.traps:
                    cells.append("X")
                else:
                    cells.append(".")
            rows.append(" ".join(cells))
        return "\n".join(rows)


def _parse_traps(spec: str, size: int) -> list[tuple[int, int]]:
    """Parse ``"1,1; 2,3"`` into [(1, 1), (2, 3)], rejecting bad cells here.

    Failing on this node with the offending pair named beats failing later
    inside a rollout with an index error nobody can trace back to a param.
    """
    traps: list[tuple[int, int]] = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 2:
            raise ValueError(
                f"GridWorldEnv: trap {chunk!r} must be 'row,col' -- separate several with ';'."
            )
        try:
            r, c = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise ValueError(f"GridWorldEnv: trap {chunk!r} must be two integers.") from exc
        if not (0 <= r < size and 0 <= c < size):
            raise ValueError(
                f"GridWorldEnv: trap ({r}, {c}) is outside a {size}x{size} grid."
            )
        if (r, c) == (0, 0):
            raise ValueError("GridWorldEnv: a trap on the start cell ends every episode at step 0.")
        if (r, c) == (size - 1, size - 1):
            raise ValueError("GridWorldEnv: a trap on the goal cell makes the goal unreachable.")
        traps.append((r, c))
    return traps


class GridWorldEnvNode(BaseNode):
    NODE_NAME = "GridWorldEnv"
    CATEGORY = "RL"
    DESCRIPTION = (
        "The textbook gridworld, with no gym dependency. Agent starts top-left, "
        "goal is bottom-right, traps sit where you put them. Reward is sparse and "
        "terminal by default (0 per step, +1 at the goal, -1 in a trap), which is "
        "the structure that makes credit assignment hard. State is a one-hot over "
        "cells, so a single Linear layer is an exactly-expressive tabular policy. "
        "Pair with PolicyRollout."
    )

    #: Builds a fresh environment object each run. Nothing here owns weights,
    #: but an env carries position state, so handing a cached one to a later
    #: run would start it mid-episode.
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="env",
                data_type=DataType.ANY,
                description="The environment. Exposes reset() and step(action); connect to PolicyRollout.",
            ),
            PortDefinition(
                name="state_dim",
                data_type=DataType.SCALAR,
                description="Observation size (size * size). Set your policy's input dim to this.",
            ),
            PortDefinition(
                name="action_dim",
                data_type=DataType.SCALAR,
                description="Number of actions (always 4: up, right, down, left).",
            ),
            PortDefinition(
                name="layout",
                data_type=DataType.STRING,
                description="The grid drawn as text (S start, G goal, X trap) -- connect a Print to see it.",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="size",
                param_type=ParamType.INT,
                default=4,
                min_value=2,
                description="Grid edge length. 4 gives the 4x4 grid the textbook draws.",
            ),
            ParamDefinition(
                name="traps",
                param_type=ParamType.STRING,
                default="1,1",
                description=(
                    "Trap cells as 'row,col', several separated by ';' (e.g. '1,1; 2,3'). "
                    "Leave empty for no traps."
                ),
            ),
            ParamDefinition(
                name="step_reward",
                param_type=ParamType.FLOAT,
                default=0.0,
                description=(
                    "Reward for a step that ends nothing. 0 keeps the reward sparse "
                    "-- the whole episode pays out only at the end."
                ),
            ),
            ParamDefinition(
                name="goal_reward",
                param_type=ParamType.FLOAT,
                default=1.0,
                description="Reward for reaching the goal (ends the episode).",
            ),
            ParamDefinition(
                name="trap_reward",
                param_type=ParamType.FLOAT,
                default=-1.0,
                description="Reward for stepping into a trap (ends the episode).",
            ),
            ParamDefinition(
                name="max_steps",
                param_type=ParamType.INT,
                default=30,
                min_value=1,
                description="Step cap. Hitting it ends the episode without reaching the goal.",
            ),
        ]

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        size = int(params.get("size", 4))
        if size < 2:
            raise ValueError(f"GridWorldEnv: size must be >= 2, got {size}.")

        traps = _parse_traps(str(params.get("traps", "") or ""), size)

        env = _GridWorld(
            size=size,
            traps=traps,
            step_reward=float(params.get("step_reward", 0.0)),
            goal_reward=float(params.get("goal_reward", 1.0)),
            trap_reward=float(params.get("trap_reward", -1.0)),
            max_steps=int(params.get("max_steps", 30)),
        )

        return {
            "env": env,
            "state_dim": env.n_states,
            "action_dim": env.n_actions,
            "layout": env.describe(),
        }
