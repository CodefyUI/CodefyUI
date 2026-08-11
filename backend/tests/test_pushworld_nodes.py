"""PushWorldEnv + PushWorldDemos (#311): determinism, expert quality, and
the BC sample contract the VLA training stack builds on."""

import re

import pytest
import torch

from app.nodes.vla.pushworld_demos_node import (
    TEXT_LEN,
    PushWorldDemosNode,
    encode_instruction,
)
from app.nodes.vla.pushworld_env_node import (
    PushWorldEnvNode,
    PushWorldFactory,
    scripted_expert_action,
)

_SIZE = 48  # small render keeps the suite fast; dynamics are size-free


def _factory(**overrides) -> PushWorldFactory:
    config = {"image_size": _SIZE, "n_distractors": 1, "max_steps": 60}
    config.update(overrides)
    return PushWorldFactory(**config)


# ── environment ─────────────────────────────────────────────────────────


def test_same_seed_reproduces_the_episode_exactly():
    a = _factory().episode(seed=7)
    b = _factory().episode(seed=7)
    assert a.instruction == b.instruction
    assert torch.equal(a.render(), b.render())
    done_a = a.step(torch.tensor([1.0, 0.5]))
    done_b = b.step(torch.tensor([1.0, 0.5]))
    assert done_a == done_b
    assert torch.equal(a.render(), b.render())


def test_different_seeds_differ():
    a = _factory().episode(seed=1)
    b = _factory().episode(seed=2)
    assert not torch.equal(a.render(), b.render())


def test_instruction_matches_the_template():
    episode = _factory().episode(seed=3)
    assert re.fullmatch(
        r"push the \w+ puck to the \w+ target", episode.instruction)


def test_distractors_control_entity_counts():
    lone = _factory(n_distractors=0).episode(seed=0)
    assert len(lone.pucks) == 1 and len(lone.targets) == 1
    crowded = _factory(n_distractors=3).episode(seed=0)
    assert len(crowded.pucks) == 4 and len(crowded.targets) == 2
    # colors distinct within each entity class
    assert len(set(crowded.puck_colors)) == 4
    assert len(set(crowded.target_colors)) == 2


def test_render_shape_and_range():
    frame = _factory().episode(seed=0).render()
    assert frame.shape == (3, _SIZE, _SIZE)
    assert 0.0 <= frame.min() and frame.max() <= 1.0


def test_episode_times_out_at_max_steps():
    episode = _factory(max_steps=10).episode(seed=5)
    done = False
    for _ in range(10):
        assert not done
        done = episode.step(torch.zeros(2))  # standing still cannot succeed
    assert done and not episode.success


def test_scripted_expert_succeeds_on_effectively_every_episode():
    successes = 0
    steps = []
    for seed in range(100):
        episode = _factory().episode(seed=1000 + seed)
        done = False
        while not done:
            done = episode.step(scripted_expert_action(episode))
        if episode.success:
            successes += 1
            steps.append(episode.steps)
    assert successes >= 99
    assert sum(steps) / len(steps) < 40


# ── demos node ──────────────────────────────────────────────────────────


def _demos(**param_overrides):
    params = {
        "episodes": 6, "chunk": 4, "demo_noise": 0.5,
        "holdout_episodes": 3, "video_episodes": 2, "seed": 0,
    }
    params.update(param_overrides)
    factory = _factory()
    return PushWorldDemosNode().execute({"env": factory}, params)


def test_sample_contract_shapes_dtypes_and_target_identity():
    result = _demos()
    dataset = result["dataset"]
    (image, tokens, chunk), target = dataset[0]
    assert image.shape == (3, _SIZE, _SIZE) and image.dtype == torch.float32
    assert 0.0 <= image.min() and image.max() <= 1.0
    assert tokens.shape == (TEXT_LEN,) and tokens.dtype == torch.int64
    assert chunk.shape == (4, 2) and chunk.dtype == torch.float32
    # the chunk rides in data AND target, and they are the same actions
    assert torch.equal(chunk, target)
    assert result["num_samples"] == len(dataset) > 0


def test_chunk_pads_by_repeating_the_last_action():
    result = _demos(demo_noise=0.0)
    dataset = result["dataset"]
    (_, _, chunk), _ = dataset[len(dataset) - 1]  # last step of last episode
    assert torch.equal(chunk[0], chunk[1])
    assert torch.equal(chunk[0], chunk[3])


def test_datasets_are_deterministic_per_seed():
    a = _demos()["dataset"]
    b = _demos()["dataset"]
    assert torch.equal(a.images_u8, b.images_u8)
    assert torch.equal(a.tokens, b.tokens)
    assert torch.equal(a.chunks, b.chunks)
    c = _demos(seed=1)["dataset"]
    assert not torch.equal(a.images_u8[:16], c.images_u8[:16])


def test_holdout_is_disjoint_from_training():
    result = _demos()
    train, holdout = result["dataset"], result["holdout"]
    assert len(holdout) > 0
    # disjoint seed streams -> the first frames cannot coincide
    assert not torch.equal(train.images_u8[0], holdout.images_u8[0])


def test_demo_video_and_instruction_example():
    result = _demos()
    video = result["demo_video"]
    assert video.dtype == torch.uint8
    assert video.dim() == 4 and video.shape[1:] == (3, _SIZE, _SIZE)
    assert video.shape[0] > 0
    assert re.fullmatch(
        r"push the \w+ puck to the \w+ target", result["instruction_example"])


def test_default_collate_batches_the_nested_samples():
    dataset = _demos()["dataset"]
    loader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=False)
    (images, tokens, chunks), targets = next(iter(loader))
    assert images.shape == (5, 3, _SIZE, _SIZE)
    assert tokens.shape == (5, TEXT_LEN)
    assert chunks.shape == (5, 4, 2)
    assert targets.shape == (5, 4, 2)


def test_stop_mid_collection_returns_partial_and_says_so():
    class _StopAfterFirstCheck:
        def __init__(self):
            self.calls = 0

        def should_stop(self):
            self.calls += 1
            return self.calls > 2

    result = PushWorldDemosNode().execute(
        {"env": _factory()},
        {"episodes": 50, "chunk": 4, "demo_noise": 0.0,
         "holdout_episodes": 0, "video_episodes": 0, "seed": 0},
        context=_StopAfterFirstCheck(),
    )
    assert "__interrupted__" in result
    assert 0 < len(result["dataset"]) < 50 * 60


def test_wrong_env_input_is_a_clear_error():
    with pytest.raises(TypeError, match="PushWorldEnv"):
        PushWorldDemosNode().execute({"env": object()}, {"episodes": 1})


def test_encode_instruction_pads_and_truncates():
    short = encode_instruction("abc")
    assert short.shape == (TEXT_LEN,)
    assert short[:3].tolist() == [97, 98, 99]
    assert short[3:].sum() == 0
    long = encode_instruction("x" * 100)
    assert long.shape == (TEXT_LEN,) and (long != 0).all()


# ── env node ────────────────────────────────────────────────────────────


def test_env_node_outputs_factory_and_size():
    result = PushWorldEnvNode().execute(
        {}, {"image_size": _SIZE, "n_distractors": 1, "max_steps": 60})
    assert isinstance(result["env"], PushWorldFactory)
    assert result["image_size"] == _SIZE
    assert "push the" in result["__log__"]
