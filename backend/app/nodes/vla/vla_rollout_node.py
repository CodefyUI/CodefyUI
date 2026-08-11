"""VLARollout node (#312) -- closed-loop evaluation, with the receipts.

Rolls a trained VLAModel policy through fresh PushWorld episodes and
reports what a VLA paper reports: closed-loop SUCCESS RATE, plus the
rollout video (wire ``frames`` into VideoWrite). Two design points carry
the science:

**Receding horizon is a measured knob, not a detail.** The policy predicts
a chunk of H actions; ``execute_k`` says how many to execute before
re-planning from a fresh observation. Prototype measurement, same trained
policy: execute_k=2 -> 46%, 4 -> 34%, 8 (full chunk, open loop) -> 20%.
The default is 2; raising it toward H is the cheap way to STUDY the
open-loop compounding-error effect rather than suffer it.

**instruction_mode=swapped is the language-grounding ablation.** It hands
the policy a deliberately mismatched instruction (a distractor puck's
color). A policy that reads pixels only scores the same as normal mode; a
policy that reads language collapses (46% -> 2% in the prototype). The gap
IS the evidence that the instruction is load-bearing.

Evaluation episodes come from a seed stream offset far from the
demonstration nodes' defaults, so a default-wired graph never evaluates on
initial states it trained on.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition
from .pushworld_demos_node import encode_instruction
from .pushworld_env_node import PushWorldFactory

logger = logging.getLogger(__name__)

INSTRUCTION_MODES = ["normal", "swapped"]

#: Rollout seeds start here (plus the node's seed param), keeping them
#: disjoint from PushWorldDemos' training stream (seed+0..) and holdout
#: stream (seed+1_000_000..) at default settings.
_ROLLOUT_SEED_OFFSET = 2_000_000

#: Border tint painted onto recorded frames after the episode resolves.
_TINT_SUCCESS = (40, 200, 60)
_TINT_FAILURE = (210, 60, 50)
_TINT_WIDTH = 2


def _tint_border(frames: list[torch.Tensor], success: bool) -> None:
    color = torch.tensor(
        _TINT_SUCCESS if success else _TINT_FAILURE, dtype=torch.uint8)
    for frame in frames:
        frame[:, :_TINT_WIDTH, :] = color[:, None, None]
        frame[:, -_TINT_WIDTH:, :] = color[:, None, None]
        frame[:, :, :_TINT_WIDTH] = color[:, None, None]
        frame[:, :, -_TINT_WIDTH:] = color[:, None, None]


class VLARolloutNode(BaseNode):
    NODE_NAME = "VLARollout"
    CATEGORY = "VLA"
    DESCRIPTION = (
        "Closed-loop evaluation of a VLAModel in PushWorld: fresh episodes, "
        "receding-horizon execution (predict a chunk, execute execute_k, "
        "re-plan), success rate + per-episode metrics + a rollout video "
        "tensor for VideoWrite (green/red border per outcome). "
        "instruction_mode=swapped is the language ablation: a policy that "
        "actually reads the instruction collapses when it lies."
    )

    # Consumes two live handles (model, env factory) and its outputs
    # describe THIS run's policy weights -- nothing a cache key can see.
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="model",
                data_type=DataType.MODEL,
                description="Trained VLAModel (needs predict_action)",
            ),
            PortDefinition(
                name="env",
                data_type=DataType.ANY,
                description="PushWorldEnv handle",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="success_rate",
                data_type=DataType.SCALAR,
                description="Episodes solved / episodes run, in [0, 1]",
            ),
            PortDefinition(
                name="avg_steps",
                data_type=DataType.SCALAR,
                description=(
                    "Mean episode length (timeouts count at the full "
                    "budget, so lower is strictly better)"
                ),
            ),
            PortDefinition(
                name="frames",
                data_type=DataType.TENSOR,
                description=(
                    "Recorded episodes, uint8 (T,3,S,S), border green/red "
                    "by outcome - wire into VideoWrite"
                ),
            ),
            PortDefinition(
                name="report",
                data_type=DataType.STRING,
                description="Per-episode outcomes as text",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="episodes",
                param_type=ParamType.INT,
                default=20,
                min_value=1,
                max_value=500,
                description="Evaluation episodes (fresh seeds, disjoint from training)",
            ),
            ParamDefinition(
                name="execute_k",
                param_type=ParamType.INT,
                default=2,
                min_value=1,
                max_value=64,
                description=(
                    "Actions executed per predicted chunk before re-planning "
                    "(receding horizon). Measured on one trained policy: "
                    "2 -> 46%, 4 -> 34%, full chunk 8 -> 20% - raise toward "
                    "the chunk size to study open-loop compounding error."
                ),
            ),
            ParamDefinition(
                name="max_steps",
                param_type=ParamType.INT,
                default=100,
                min_value=10,
                max_value=1000,
                description=(
                    "Per-episode step budget (overrides the env's). Learned "
                    "policies are slower than the scripted expert; a tight "
                    "budget scores control errors as timeouts."
                ),
            ),
            ParamDefinition(
                name="instruction_mode",
                param_type=ParamType.SELECT,
                default="normal",
                options=INSTRUCTION_MODES,
                description=(
                    "normal: the episode's true instruction. swapped: name a "
                    "DISTRACTOR puck instead - the language-grounding "
                    "ablation. A pixels-only policy scores the same either "
                    "way; a language-reading one collapses under swapped."
                ),
            ),
            ParamDefinition(
                name="record_episodes",
                param_type=ParamType.INT,
                default=4,
                min_value=0,
                max_value=32,
                description="First N episodes recorded into frames (0 = none)",
            ),
            ParamDefinition(
                name="seed",
                param_type=ParamType.INT,
                default=0,
                min_value=0,
                description=(
                    "Evaluation seed. Episode seeds run from an offset "
                    "stream disjoint from PushWorldDemos' defaults."
                ),
                advanced=True,
            ),
            ParamDefinition(
                name="device",
                param_type=ParamType.SELECT,
                default="auto",
                options=["auto", "cpu", "cuda", "mps"],
                description="auto follows the run's device",
                advanced=True,
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
        from ...core.device_utils import resolve_node_device
        from ...core.loop_control import ProgressThrottle, interrupted_result, stop_checker

        model = inputs["model"]
        factory = inputs["env"]
        if not isinstance(factory, PushWorldFactory):
            raise TypeError(
                "VLARollout: the env input must come from PushWorldEnv "
                f"(got {type(factory).__name__}).")
        if not hasattr(model, "predict_action"):
            raise TypeError(
                "VLARollout: the model input must be a VLAModel (it has no "
                "predict_action).")

        episodes = max(1, int(params.get("episodes", 20) or 20))
        execute_k = max(1, int(params.get("execute_k", 2) or 2))
        max_steps = max(10, int(params.get("max_steps", 100) or 100))
        mode = str(params.get("instruction_mode", "normal") or "normal")
        record_episodes = max(0, int(params.get("record_episodes", 4) or 0))
        seed = max(0, int(params.get("seed", 0) or 0))
        if mode == "swapped" and factory.n_distractors < 1:
            raise ValueError(
                "instruction_mode=swapped needs at least one distractor "
                "puck to name - set PushWorldEnv.n_distractors >= 1.")

        device = resolve_node_device(params.get("device"), context)
        model_device = next(model.parameters()).device
        model.to(device)

        should_stop = stop_checker(context)
        throttle = ProgressThrottle(progress_callback)
        max_text_len = int(getattr(model, "max_text_len", 48))

        successes = 0
        total_steps = 0
        completed = 0
        stopped_at: int | None = None
        recorded: list[torch.Tensor] = []
        lines: list[str] = []
        try:
            for index in range(episodes):
                if should_stop():
                    stopped_at = index
                    break
                episode = factory.episode(
                    seed=seed + _ROLLOUT_SEED_OFFSET + index,
                    max_steps=max_steps)
                instruction = episode.instruction
                if mode == "swapped":
                    wrong_color = episode.puck_colors[1]
                    instruction = (
                        f"push the {wrong_color} puck to the "
                        f"{episode.target_colors[episode.goal_target]} target")
                tokens = encode_instruction(
                    instruction, max_text_len).unsqueeze(0).to(device)

                episode_frames: list[torch.Tensor] = []
                done = False
                while not done:
                    frame = episode.render()
                    if index < record_episodes:
                        episode_frames.append(
                            (frame.clamp(0, 1) * 255).round().to(torch.uint8))
                    observation = frame.unsqueeze(0).to(device)
                    chunk = model.predict_action(observation, tokens)[0]
                    for k in range(min(execute_k, chunk.shape[0])):
                        done = episode.step(chunk[k].float().cpu())
                        if done:
                            break
                if index < record_episodes and episode_frames:
                    _tint_border(episode_frames, episode.success)
                    recorded.extend(episode_frames)

                completed += 1
                successes += int(episode.success)
                total_steps += episode.steps
                lines.append(
                    f"episode {index + 1}: "
                    f"{'success' if episode.success else 'timeout'} in "
                    f"{episode.steps} steps -- \"{instruction}\"")
                if context is not None:
                    context.log_metric(
                        "success_rate", successes / completed, completed)
                    context.log_metric(
                        "episode_steps", float(episode.steps), completed)
                throttle.emit({
                    "event": "progress",
                    "episode": completed,
                    "total_episodes": episodes,
                    "success_rate": round(successes / completed, 4),
                })
        finally:
            model.to(model_device)

        if completed == 0:
            raise RuntimeError("VLARollout: stopped before any episode ran.")

        success_rate = successes / completed
        avg_steps = total_steps / completed
        frames = (
            torch.stack(recorded) if recorded
            else torch.zeros(0, 3, factory.image_size, factory.image_size,
                             dtype=torch.uint8))
        note = (
            f"{mode} instructions: {successes}/{completed} solved "
            f"(success_rate {success_rate:.2f}), avg {avg_steps:.1f} steps, "
            f"execute_k {execute_k}, {frames.shape[0]} recorded frames."
        )
        logger.info("VLARollout: %s", note)
        result: dict[str, Any] = {
            "success_rate": success_rate,
            "avg_steps": avg_steps,
            "frames": frames,
            "report": "\n".join(lines),
            "__log__": note,
        }
        if stopped_at is not None:
            result.update(interrupted_result(episode=stopped_at))
        return result
