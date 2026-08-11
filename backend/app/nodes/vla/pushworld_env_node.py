"""PushWorldEnv node (#311) -- a language-conditioned 2D push benchmark.

The environment side of the VLA epic (#309): an agent disc, colored pucks,
colored ring targets, and an instruction of the form "push the {color} puck
to the {color} target". With distractors present, neither the puck nor the
target is inferable from pixels alone -- the policy must READ the text, and
the swapped-instruction ablation in VLARollout can measure that it does.

This is PushT's spirit (the Diffusion Policy / LeRobot benchmark) at
zero-dependency scale: pure torch, no pygame/pymunk/gymnasium, seeded and
deterministic, ~4 ms per episode including rendering, so demonstration
collection and closed-loop evaluation both run inside a node without any
simulator install. Contact dynamics are deliberately simple (circle-overlap
resolution rather than rigid-body physics): the research questions this
canvas targets -- language grounding, head paradigms, chunking, data
mixtures -- live above the contact model.

Everything is drawn as anti-aliased distance-field masks on a cached
meshgrid, layered painting, ``(3, S, S)`` float [0,1] -- no renderer
dependency either.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition

logger = logging.getLogger(__name__)

#: Named colors an episode samples from. Six entries: up to four pucks and
#: two targets can be on screen with every color distinct.
PALETTE: dict[str, tuple[float, float, float]] = {
    "red": (0.86, 0.20, 0.18),
    "green": (0.00, 0.71, 0.31),
    "blue": (0.15, 0.43, 1.00),
    "yellow": (0.94, 0.78, 0.00),
    "magenta": (0.83, 0.21, 0.51),
    "cyan": (0.16, 0.63, 0.60),
}
_COLOR_NAMES = list(PALETTE)

#: Entity radii in arena units ([0,1]^2 arena). Chosen so a 96px render
#: keeps every entity several pixels wide and the agent can slip between
#: two pucks: agent < puck < target ring.
R_AGENT = 0.055
R_PUCK = 0.070
R_TARGET = 0.095
#: Largest per-step displacement, arena units -- actions are [-1,1]^2 times
#: this many arena units per step.
MAX_SPEED = 0.05
#: The goal puck's center must come this close to the target's center.
SUCCESS_DIST = 0.05
#: Keep-out from the walls at placement time.
_MARGIN = 0.13


class PushWorldEpisode:
    """One live episode: reset state, step dynamics, render frames.

    Positions live in [0,1]^2. All randomness comes from the constructor
    seed, so (seed, config) -> identical episode, which is what makes
    demonstrations and evaluations reproducible per seed.
    """

    def __init__(self, image_size: int, n_distractors: int, max_steps: int,
                 seed: int):
        self.image_size = image_size
        self.n_distractors = n_distractors
        self.max_steps = max_steps
        self.gen = torch.Generator().manual_seed(seed)
        axis = torch.linspace(0.0, 1.0, image_size)
        self._grid_y, self._grid_x = torch.meshgrid(axis, axis, indexing="ij")
        self._reset()

    # -- setup ----------------------------------------------------------

    def _rand(self, n: int) -> torch.Tensor:
        return torch.rand(n, generator=self.gen)

    def _place(self, existing: list[torch.Tensor], min_sep: float) -> torch.Tensor:
        for _ in range(200):
            p = _MARGIN + self._rand(2) * (1 - 2 * _MARGIN)
            if all(torch.linalg.vector_norm(p - q) >= min_sep for q in existing):
                return p
        return p  # crowded fallback; the radii keep it playable

    def _reset(self) -> None:
        n_pucks = 1 + self.n_distractors
        n_targets = 2 if self.n_distractors > 0 else 1
        order = torch.randperm(len(_COLOR_NAMES), generator=self.gen).tolist()
        self.puck_colors = [_COLOR_NAMES[i] for i in order[:n_pucks]]
        order2 = torch.randperm(len(_COLOR_NAMES), generator=self.gen).tolist()
        self.target_colors = [_COLOR_NAMES[i] for i in order2[:n_targets]]
        placed: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []
        for _ in range(n_targets):
            p = self._place(placed, 2.6 * R_TARGET)
            self.targets.append(p)
            placed.append(p)
        self.pucks: list[torch.Tensor] = []
        for _ in range(n_pucks):
            p = self._place(placed, 2.4 * R_TARGET)
            self.pucks.append(p)
            placed.append(p)
        self.agent = self._place(placed, R_AGENT + R_PUCK + 0.02)
        # The goal is always puck 0 -- colors are freshly shuffled per
        # episode, so "which color is the goal" still varies uniformly.
        self.goal_puck = 0
        self.goal_target = int(self._rand(1).item() * n_targets) % n_targets
        self.instruction = (
            f"push the {self.puck_colors[0]} puck to the "
            f"{self.target_colors[self.goal_target]} target")
        self.steps = 0
        self.success = False

    # -- dynamics -------------------------------------------------------

    def step(self, action: torch.Tensor) -> bool:
        """Advance one step; returns True when the episode is over."""
        move = torch.clamp(action.detach().float().view(2), -1.0, 1.0) * MAX_SPEED
        self.agent = torch.clamp(self.agent + move, R_AGENT, 1 - R_AGENT)
        for i, puck in enumerate(self.pucks):
            offset = puck - self.agent
            dist = torch.linalg.vector_norm(offset)
            min_dist = R_AGENT + R_PUCK
            if dist < min_dist:
                normal = offset / (dist + 1e-8)
                self.pucks[i] = torch.clamp(
                    self.agent + normal * min_dist, R_PUCK, 1 - R_PUCK)
        # Puck-puck separation, single pass -- enough at these densities.
        for i in range(len(self.pucks)):
            for j in range(i + 1, len(self.pucks)):
                offset = self.pucks[j] - self.pucks[i]
                dist = torch.linalg.vector_norm(offset)
                if dist < 2 * R_PUCK:
                    normal = offset / (dist + 1e-8)
                    shift = (2 * R_PUCK - dist) / 2
                    self.pucks[i] = torch.clamp(
                        self.pucks[i] - normal * shift, R_PUCK, 1 - R_PUCK)
                    self.pucks[j] = torch.clamp(
                        self.pucks[j] + normal * shift, R_PUCK, 1 - R_PUCK)
        self.steps += 1
        goal_dist = torch.linalg.vector_norm(
            self.pucks[self.goal_puck] - self.targets[self.goal_target])
        if goal_dist < SUCCESS_DIST:
            self.success = True
        return self.success or self.steps >= self.max_steps

    # -- rendering ------------------------------------------------------

    def _circle(self, center: torch.Tensor, radius: float) -> torch.Tensor:
        aa = 1.5 / self.image_size
        d = torch.sqrt((self._grid_x - center[0]) ** 2
                       + (self._grid_y - center[1]) ** 2)
        return torch.clamp((radius - d) / aa, 0.0, 1.0)

    def _ring(self, center: torch.Tensor, radius: float, width: float) -> torch.Tensor:
        aa = 1.5 / self.image_size
        d = torch.sqrt((self._grid_x - center[0]) ** 2
                       + (self._grid_y - center[1]) ** 2)
        return torch.clamp((width - torch.abs(d - radius)) / aa, 0.0, 1.0)

    def render(self) -> torch.Tensor:
        """Current frame as ``(3, S, S)`` float [0,1]."""
        canvas = torch.full((3, self.image_size, self.image_size), 0.09)

        def paint(base: torch.Tensor, mask: torch.Tensor,
                  rgb: tuple[float, float, float]) -> torch.Tensor:
            color = torch.tensor(rgb).view(3, 1, 1)
            m = mask.unsqueeze(0)
            return base * (1 - m) + color * m

        for target, name in zip(self.targets, self.target_colors):
            canvas = paint(canvas, self._ring(target, R_TARGET, 0.016), PALETTE[name])
        for puck, name in zip(self.pucks, self.puck_colors):
            canvas = paint(canvas, self._circle(puck, R_PUCK), PALETTE[name])
        canvas = paint(canvas, self._circle(self.agent, R_AGENT), (0.92, 0.92, 0.92))
        return canvas


def _rotate_toward(u: torch.Tensor, v: torch.Tensor, max_angle: float) -> torch.Tensor:
    """Rotate unit vector *u* toward unit vector *v* by at most *max_angle*."""
    cross = u[0] * v[1] - u[1] * v[0]
    dot = torch.clamp(u[0] * v[0] + u[1] * v[1], -1.0, 1.0)
    angle = torch.atan2(cross, dot)
    step = torch.clamp(angle, -max_angle, max_angle)
    c, s = torch.cos(step), torch.sin(step)
    return torch.stack([c * u[0] - s * u[1], s * u[0] + c * u[1]])


def scripted_expert_action(episode: PushWorldEpisode) -> torch.Tensor:
    """The demonstration policy: get behind the goal puck, push through it.

    Three regimes, recomputed every step so the expert self-corrects:
    aligned behind the puck -> push toward the target through the puck's
    center; near the puck on the wrong side -> orbit around it (never plow
    through); far away -> walk to the point behind it. Validated at 100/100
    episodes, mean 23 steps, on the default configuration.
    """
    puck = episode.pucks[episode.goal_puck]
    target = episode.targets[episode.goal_target]
    to_target = target - puck
    dist_target = torch.linalg.vector_norm(to_target)
    push_dir = to_target / (dist_target + 1e-8)
    rel = episode.agent - puck
    dist_agent = torch.linalg.vector_norm(rel)
    behind = -push_dir
    gap = R_AGENT + R_PUCK + 0.012
    aligned = (rel / (dist_agent + 1e-8) * behind).sum() > 0.85
    if aligned and dist_agent < gap + 0.05:
        goal = puck + push_dir * 0.03
    elif dist_agent < gap + 0.06:
        u = rel / (dist_agent + 1e-8)
        goal = puck + _rotate_toward(u, behind, 0.5) * gap
    else:
        goal = puck + behind * gap
    delta = goal - episode.agent
    if torch.linalg.vector_norm(delta) < 1e-6:
        return torch.zeros(2)
    return torch.clamp(delta / MAX_SPEED, -1.0, 1.0)


class PushWorldFactory:
    """The env handle on the wire: immutable config, fresh episode per seed.

    Consumers (PushWorldDemos, VLARollout) call :meth:`episode` with their
    own seed streams, so the factory itself carries no mutable state.
    """

    def __init__(self, image_size: int, n_distractors: int, max_steps: int):
        self.image_size = image_size
        self.n_distractors = n_distractors
        self.max_steps = max_steps
        self.action_dim = 2

    def episode(self, seed: int, max_steps: int | None = None) -> PushWorldEpisode:
        return PushWorldEpisode(
            image_size=self.image_size,
            n_distractors=self.n_distractors,
            max_steps=self.max_steps if max_steps is None else max_steps,
            seed=seed,
        )


class PushWorldEnvNode(BaseNode):
    NODE_NAME = "PushWorldEnv"
    CATEGORY = "VLA"
    DESCRIPTION = (
        "A language-conditioned 2D push environment (PushT's spirit, pure "
        "torch): an agent disc, colored pucks, colored ring targets, and an "
        "instruction naming which puck goes to which target. With "
        "distractors, pixels alone cannot identify the goal - the policy "
        "must read the instruction. Feed the env handle to PushWorldDemos "
        "for demonstrations and VLARollout for closed-loop evaluation."
    )

    # The output is a live handle consumers construct episodes from. It is
    # immutable config and rebuilding costs microseconds, so there is
    # nothing for a cache to buy -- and an object identity crossing runs
    # out of ExecutionCache is exactly the #253/#254 shape of surprise.
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
                description=(
                    "Environment handle (episode factory) for "
                    "PushWorldDemos and VLARollout"
                ),
            ),
            PortDefinition(
                name="image_size",
                data_type=DataType.SCALAR,
                description="Render size in pixels - match VLAModel's image_size",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="image_size",
                param_type=ParamType.INT,
                default=96,
                min_value=32,
                max_value=256,
                description=(
                    "Rendered frame size in pixels (square). 96 matches the "
                    "PushT convention and keeps a 4080-class GPU fed."
                ),
            ),
            ParamDefinition(
                name="n_distractors",
                param_type=ParamType.INT,
                default=1,
                min_value=0,
                max_value=3,
                description=(
                    "Extra pucks beside the goal puck. At 0 there is one "
                    "puck and one target and language is decoration; from 1 "
                    "up there are two targets and 2+ pucks, so the "
                    "instruction is the ONLY way to know the goal."
                ),
            ),
            ParamDefinition(
                name="max_steps",
                param_type=ParamType.INT,
                default=60,
                min_value=10,
                max_value=1000,
                description=(
                    "Steps before an episode times out. The scripted expert "
                    "needs ~23 on average; policies are slower - VLARollout "
                    "can raise its own budget per evaluation."
                ),
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        image_size = int(params.get("image_size", 96) or 96)
        n_distractors = max(0, min(3, int(params.get("n_distractors", 1) or 0)))
        max_steps = max(10, int(params.get("max_steps", 60) or 60))
        factory = PushWorldFactory(image_size, n_distractors, max_steps)
        sample = factory.episode(seed=0)
        logger.info(
            "PushWorldEnv: %dpx, %d distractor(s), %d max steps",
            image_size, n_distractors, max_steps)
        return {
            "env": factory,
            "image_size": image_size,
            "__log__": (
                f"PushWorld {image_size}px, {n_distractors} distractor(s), "
                f"{1 + n_distractors} puck(s) / "
                f"{2 if n_distractors else 1} target(s), max {max_steps} "
                f"steps. Example instruction: \"{sample.instruction}\"."
            ),
        }
