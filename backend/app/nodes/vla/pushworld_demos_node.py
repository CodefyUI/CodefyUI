"""PushWorldDemos node (#311) -- scripted-expert demonstrations for BC.

Rolls PushWorld episodes with the scripted expert and packages every step
as a behavior-cloning sample. Two design choices carry the closed-loop
result and are worth knowing before turning knobs:

**demo_noise is DART, not decoration.** With ``demo_noise > 0`` the action
EXECUTED in the environment is perturbed while the RECORDED label stays the
expert's. Pure on-trajectory demonstrations contain no recovery states, so
a cloned policy that drifts a pixel off the expert manifold has never seen
its situation and compounds the error -- measured here: 4% closed-loop
success trained clean vs 24% with noise 0.5 on an otherwise identical
prototype, before any architecture change. Perturbed states with corrective
labels are what closed-loop control learns from.

**Actions appear in the sample's data AND target.** A sample is
``((image, instruction_tokens, action_chunk), action_chunk)`` -- the chunk
rides along inside ``data`` so a flow-matching model can noise it INSIDE
``forward`` while the TrainingLoop contract (``outputs = model(data);
loss = loss_fn(outputs, targets)``) stays untouched. A regression model
simply ignores ``data[2]``.

Instructions are UTF-8 BYTES zero-padded to ``TEXT_LEN`` -- at this scale a
byte embedding (256 rows) is the whole language stack, and BPE would buy
nothing but a 50k-row table. Images are stored uint8 and converted per
``__getitem__``, which is 4x less resident memory than float storage
(~28 KB/sample at 96px; the default 600 episodes come to ~15k samples,
~0.4 GB).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch.utils.data import Dataset

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition
from .pushworld_env_node import PushWorldFactory, scripted_expert_action

logger = logging.getLogger(__name__)

#: Instruction encoding length, bytes. The template maxes out at 40
#: characters of ASCII; 48 leaves room without wasting attention tokens.
TEXT_LEN = 48

#: Seed offset separating the holdout episode stream from training's, so
#: the two sets can never share an episode at any `episodes` value.
_HOLDOUT_SEED_OFFSET = 1_000_000


def encode_instruction(text: str, length: int = TEXT_LEN) -> torch.Tensor:
    """UTF-8 bytes, zero-padded/truncated to *length*, as int64 (length,)."""
    raw = text.encode("utf-8")[:length]
    out = torch.zeros(length, dtype=torch.long)
    out[: len(raw)] = torch.tensor(list(raw), dtype=torch.long)
    return out


class PushWorldDemoDataset(Dataset):
    """BC samples over uint8 frames; float conversion happens per item."""

    def __init__(self, images_u8: torch.Tensor, tokens: torch.Tensor,
                 chunks: torch.Tensor):
        self.images_u8 = images_u8
        self.tokens = tokens
        self.chunks = chunks

    def __len__(self) -> int:
        return self.images_u8.shape[0]

    def __getitem__(self, index: int):
        image = self.images_u8[index].float() / 255.0
        chunk = self.chunks[index]
        return (image, self.tokens[index], chunk), chunk


def _collect(
    factory: PushWorldFactory,
    episodes: int,
    chunk: int,
    demo_noise: float,
    seed0: int,
    noise_gen: torch.Generator,
    video_episodes: int = 0,
    should_stop=None,
    on_episode=None,
) -> tuple[list, list, list, list, int | None]:
    """Roll episodes; returns (images_u8, tokens, chunks, video, stopped_at)."""
    images: list[torch.Tensor] = []
    tokens: list[torch.Tensor] = []
    chunks: list[torch.Tensor] = []
    video: list[torch.Tensor] = []
    stopped_at: int | None = None
    for index in range(episodes):
        if should_stop is not None and should_stop():
            stopped_at = index
            break
        episode = factory.episode(seed=seed0 + index)
        frames: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        done = False
        while not done:
            frames.append(episode.render())
            action = scripted_expert_action(episode)
            actions.append(action)
            executed = action
            if demo_noise > 0 and torch.rand(1, generator=noise_gen).item() < 0.5:
                executed = torch.clamp(
                    action + torch.randn(2, generator=noise_gen) * demo_noise,
                    -1.0, 1.0)
            done = episode.step(executed)
        instruction = encode_instruction(episode.instruction)
        total = len(actions)
        for t in range(total):
            step_chunk = [actions[min(t + k, total - 1)] for k in range(chunk)]
            images.append((frames[t].clamp(0, 1) * 255).round().to(torch.uint8))
            tokens.append(instruction)
            chunks.append(torch.stack(step_chunk))
        if index < video_episodes:
            video.extend(
                (f.clamp(0, 1) * 255).round().to(torch.uint8) for f in frames)
        if on_episode is not None:
            on_episode(index + 1)
    return images, tokens, chunks, video, stopped_at


class PushWorldDemosNode(BaseNode):
    NODE_NAME = "PushWorldDemos"
    CATEGORY = "VLA"
    DESCRIPTION = (
        "Roll the scripted expert through PushWorld episodes and emit "
        "behavior-cloning samples ((image, instruction bytes, action "
        "chunk), action chunk) plus a held-out split and a demo video "
        "tensor for VideoWrite. demo_noise executes DART-style perturbed "
        "actions while recording the expert's - the recovery data "
        "closed-loop control needs."
    )

    # Consumes the live env handle and owns a multi-second collection pass
    # whose products (three stacked tensors per split) no fingerprint
    # describes -- the DataMixDataset shape of the rule.
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
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
                name="dataset",
                data_type=DataType.DATASET,
                description=(
                    "Training samples ((image f32 (3,S,S), instruction "
                    "int64 (48,), chunk f32 (H,2)), chunk) - feed DataLoader"
                ),
            ),
            PortDefinition(
                name="holdout",
                data_type=DataType.DATASET,
                description=(
                    "Held-out episodes from a disjoint seed stream, for "
                    "VLAActionEval / a val DataLoader"
                ),
            ),
            PortDefinition(
                name="num_samples",
                data_type=DataType.SCALAR,
                description="Training samples emitted",
            ),
            PortDefinition(
                name="demo_video",
                data_type=DataType.TENSOR,
                description=(
                    "First episodes' frames, uint8 (T,3,S,S) - wire into "
                    "VideoWrite to watch the (noise-perturbed) expert"
                ),
            ),
            PortDefinition(
                name="instruction_example",
                data_type=DataType.STRING,
                description="First episode's instruction, for a quick sanity read",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="episodes",
                param_type=ParamType.INT,
                default=600,
                min_value=1,
                max_value=20000,
                description=(
                    "Training episodes to roll (~25 samples each; 600 is "
                    "~15k samples / ~0.4 GB resident at 96px)"
                ),
            ),
            ParamDefinition(
                name="chunk",
                param_type=ParamType.INT,
                default=8,
                min_value=1,
                max_value=64,
                description=(
                    "Actions per sample (the action-chunk horizon H; SmolVLA "
                    "uses 50 at robot scale). Must match VLAModel's chunk. "
                    "Past the episode end the last action repeats."
                ),
            ),
            ParamDefinition(
                name="demo_noise",
                param_type=ParamType.FLOAT,
                default=0.5,
                min_value=0.0,
                max_value=2.0,
                description=(
                    "DART perturbation scale: with probability 1/2 per step, "
                    "N(0, noise) is added to the EXECUTED action while the "
                    "expert's stays the label. 0 disables - measured to "
                    "collapse closed-loop success (4% vs 24% at 0.5 in the "
                    "prototype); keep it on unless studying exactly that."
                ),
            ),
            ParamDefinition(
                name="holdout_episodes",
                param_type=ParamType.INT,
                default=60,
                min_value=0,
                max_value=2000,
                description="Held-out episodes (disjoint seeds), 0 = empty holdout",
            ),
            ParamDefinition(
                name="video_episodes",
                param_type=ParamType.INT,
                default=4,
                min_value=0,
                max_value=16,
                description="Episodes recorded into demo_video (0 = none)",
                advanced=True,
            ),
            ParamDefinition(
                name="seed",
                param_type=ParamType.INT,
                default=0,
                min_value=0,
                description="Base seed - same seed and params reproduce the datasets exactly",
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
        from ...core.loop_control import ProgressThrottle, interrupted_result, stop_checker

        factory = inputs["env"]
        if not isinstance(factory, PushWorldFactory):
            raise TypeError(
                "PushWorldDemos: the env input must come from PushWorldEnv "
                f"(got {type(factory).__name__}).")

        episodes = max(1, int(params.get("episodes", 600) or 600))
        chunk = max(1, min(64, int(params.get("chunk", 8) or 8)))
        demo_noise = max(0.0, float(params.get("demo_noise", 0.5) or 0.0))
        holdout_episodes = max(0, int(params.get("holdout_episodes", 60) or 0))
        video_episodes = max(0, min(16, int(params.get("video_episodes", 4) or 0)))
        seed = max(0, int(params.get("seed", 0) or 0))

        should_stop = stop_checker(context)
        throttle = ProgressThrottle(progress_callback)
        noise_gen = torch.Generator().manual_seed(seed + 991)

        def report(done_episodes: int) -> None:
            throttle.emit({
                "event": "progress",
                "phase": "collect",
                "episode": done_episodes,
                "total_episodes": episodes + holdout_episodes,
            })

        images, tokens, chunks, video, stopped_at = _collect(
            factory, episodes, chunk, demo_noise, seed, noise_gen,
            video_episodes=video_episodes, should_stop=should_stop,
            on_episode=report)

        holdout_images: list = []
        holdout_tokens: list = []
        holdout_chunks: list = []
        if stopped_at is None and holdout_episodes:
            holdout_images, holdout_tokens, holdout_chunks, _, hold_stop = _collect(
                factory, holdout_episodes, chunk, demo_noise,
                seed + _HOLDOUT_SEED_OFFSET, noise_gen,
                should_stop=should_stop,
                on_episode=lambda n: report(episodes + n))
            if hold_stop is not None:
                stopped_at = episodes + hold_stop

        def build(imgs: list, toks: list, chks: list) -> PushWorldDemoDataset:
            if not imgs:
                size = factory.image_size
                return PushWorldDemoDataset(
                    torch.zeros(0, 3, size, size, dtype=torch.uint8),
                    torch.zeros(0, TEXT_LEN, dtype=torch.long),
                    torch.zeros(0, chunk, 2))
            return PushWorldDemoDataset(
                torch.stack(imgs), torch.stack(toks), torch.stack(chks))

        dataset = build(images, tokens, chunks)
        holdout = build(holdout_images, holdout_tokens, holdout_chunks)
        if len(dataset) == 0:
            raise RuntimeError(
                "PushWorldDemos: stopped before any episode finished - "
                "no dataset to hand downstream.")

        instruction_example = ""
        if tokens:
            raw = bytes(b for b in tokens[0].tolist() if b)
            instruction_example = raw.decode("utf-8", "replace")

        demo_video = (
            torch.stack(video) if video
            else torch.zeros(0, 3, factory.image_size, factory.image_size,
                             dtype=torch.uint8))

        note = (
            f"{len(dataset):,} samples from "
            f"{stopped_at if stopped_at is not None else episodes} episodes "
            f"(chunk {chunk}, demo_noise {demo_noise:g}), "
            f"{len(holdout):,} holdout samples, "
            f"{demo_video.shape[0]} video frames."
        )
        logger.info("PushWorldDemos: %s", note)
        result: dict[str, Any] = {
            "dataset": dataset,
            "holdout": holdout,
            "num_samples": len(dataset),
            "demo_video": demo_video,
            "instruction_example": instruction_example,
            "__log__": note,
        }
        if stopped_at is not None:
            # Partial data is still usable data, but the run must say so
            # rather than pass it off as the requested collection.
            result.update(interrupted_result(episode=stopped_at))
        return result
