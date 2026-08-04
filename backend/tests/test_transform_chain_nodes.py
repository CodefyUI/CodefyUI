"""Transform-chain nodes (core#136): parity with torchvision, and ordering.

The load-bearing test here is PARITY. A node that accepts ``size`` and
``padding`` and then quietly builds something slightly different from
``transforms.RandomCrop(size, padding=padding)`` still runs, still trains,
and still produces a plausible loss curve -- it just does not reproduce the
baseline anyone is comparing against. So every step node is checked by
seeding the global RNG, running torchvision's own transform, re-seeding to
the same value, running the node's, and demanding the two tensors are
IDENTICAL. That catches a wrong keyword, a wrong default and a wrong
argument order in one assertion, and it is the only check that can.

The chain tests are about ORDER, which is the other thing a composable
pipeline can silently get wrong.
"""

from __future__ import annotations

import pickle

import pytest
import torch
from PIL import Image
from torchvision import transforms as T

from app.core.node_base import DataType
from app.nodes.data.transforms._base import (
    TransformStepNode,
    compose,
    pipeline_is_random,
)
from app.nodes.data.transforms.color_jitter_node import ColorJitterNode
from app.nodes.data.transforms.compose_transform_node import ComposeTransformNode
from app.nodes.data.transforms.normalize_transform_node import (
    PRESETS,
    NormalizeTransformNode,
)
from app.nodes.data.transforms.rand_augment_node import RandAugmentNode
from app.nodes.data.transforms.random_crop_node import RandomCropNode
from app.nodes.data.transforms.random_horizontal_flip_node import (
    RandomHorizontalFlipNode,
)
from app.nodes.data.transforms.random_rotation_node import RandomRotationNode
from app.nodes.data.transforms.resize_transform_node import ResizeTransformNode
from app.nodes.data.transforms.to_tensor_transform_node import (
    ToTensorTransformNode,
)

STEP_NODES = [
    ResizeTransformNode,
    ToTensorTransformNode,
    NormalizeTransformNode,
    RandomCropNode,
    RandomHorizontalFlipNode,
    RandomRotationNode,
    ColorJitterNode,
    RandAugmentNode,
]

PARITY_SEED = 20260801


def _image(size: int = 32, seed: int = 7) -> Image.Image:
    """A deterministic, non-uniform RGB image.

    Non-uniform matters: a flat grey image survives a horizontal flip, a
    crop and a rotation unchanged, so a parity test on one would pass
    against a transform that does nothing at all.
    """
    generator = torch.Generator().manual_seed(seed)
    pixels = torch.randint(0, 256, (size, size, 3), generator=generator,
                           dtype=torch.uint8)
    return Image.fromarray(pixels.numpy(), mode="RGB")


def _build(node_cls, params: dict | None = None):
    """The transform a node contributes, on its own."""
    result = node_cls().execute({}, params or {})
    pipeline = result["transform"]
    assert isinstance(pipeline, T.Compose)
    assert len(pipeline.transforms) == 1
    return pipeline.transforms[0]


def _seeded(transform, sample):
    """Apply *transform* from a fixed RNG state, as a tensor."""
    torch.manual_seed(PARITY_SEED)
    out = transform(sample)
    return out if isinstance(out, torch.Tensor) else T.ToTensor()(out)


def _assert_parity(node_cls, params, reference, sample=None):
    sample = _image() if sample is None else sample
    theirs = _seeded(reference, sample)
    ours = _seeded(_build(node_cls, params), sample)
    assert torch.equal(theirs, ours), (
        f"{node_cls.NODE_NAME} diverged from torchvision on seed "
        f"{PARITY_SEED}")


# ── shape of the port contract ───────────────────────────────────────────


@pytest.mark.parametrize("node_cls", STEP_NODES)
def test_every_step_node_has_the_same_chain_ports(node_cls):
    inputs = node_cls.define_inputs()
    outputs = node_cls.define_outputs()
    assert [p.name for p in inputs] == ["transform"]
    assert inputs[0].data_type is DataType.TRANSFORM
    assert inputs[0].optional is True
    assert [p.name for p in outputs] == ["transform"]
    assert outputs[0].data_type is DataType.TRANSFORM


@pytest.mark.parametrize("node_cls", STEP_NODES)
def test_every_step_node_is_registered_under_data(node_cls):
    assert node_cls.CATEGORY == "Data"
    assert node_cls.NODE_NAME
    assert node_cls.DESCRIPTION


def test_the_base_class_is_not_a_registrable_node():
    """An empty NODE_NAME is what keeps the registry from picking it up."""
    assert TransformStepNode.NODE_NAME == ""


# ── parity with torchvision, one test per transform ──────────────────────


def test_resize_matches_torchvision():
    _assert_parity(
        ResizeTransformNode, {"size": 16},
        T.Resize((16, 16), interpolation=T.InterpolationMode.BILINEAR))


def test_resize_honours_the_interpolation_choice():
    _assert_parity(
        ResizeTransformNode, {"size": 16, "interpolation": "nearest"},
        T.Resize((16, 16), interpolation=T.InterpolationMode.NEAREST))
    # And the modes are genuinely different, so the test above is not
    # comparing two identical pipelines.
    nearest = _seeded(_build(ResizeTransformNode,
                             {"size": 16, "interpolation": "nearest"}), _image())
    bicubic = _seeded(_build(ResizeTransformNode,
                             {"size": 16, "interpolation": "bicubic"}), _image())
    assert not torch.equal(nearest, bicubic)


def test_resize_produces_a_square_not_a_shortest_side():
    """The bare-int form of Resize would leave a 64x32 image 128x64."""
    tall = Image.new("RGB", (64, 32))
    out = _build(ResizeTransformNode, {"size": 24})(tall)
    assert out.size == (24, 24)


def test_to_tensor_matches_torchvision():
    _assert_parity(ToTensorTransformNode, {}, T.ToTensor())


def test_random_crop_matches_torchvision():
    _assert_parity(
        RandomCropNode, {"size": 32, "padding": 4},
        T.RandomCrop(32, padding=4, padding_mode="constant"))


def test_random_crop_honours_padding_mode():
    _assert_parity(
        RandomCropNode, {"size": 32, "padding": 4, "padding_mode": "reflect"},
        T.RandomCrop(32, padding=4, padding_mode="reflect"))


def test_random_horizontal_flip_matches_torchvision():
    _assert_parity(RandomHorizontalFlipNode, {"p": 0.5},
                   T.RandomHorizontalFlip(p=0.5))


def test_random_horizontal_flip_p_one_is_a_real_flip():
    """p is wired to the probability, not swallowed."""
    image = _image()
    flipped = _build(RandomHorizontalFlipNode, {"p": 1.0})(image)
    never = _build(RandomHorizontalFlipNode, {"p": 0.0})(image)
    assert torch.equal(T.ToTensor()(flipped),
                       torch.flip(T.ToTensor()(image), dims=[2]))
    assert torch.equal(T.ToTensor()(never), T.ToTensor()(image))


def test_random_rotation_matches_torchvision():
    _assert_parity(RandomRotationNode, {"degrees": 15.0},
                   T.RandomRotation(15.0, expand=False, fill=0.0))


def test_random_rotation_expand_matches_torchvision():
    _assert_parity(RandomRotationNode,
                   {"degrees": 30.0, "expand": True, "fill": 0.0},
                   T.RandomRotation(30.0, expand=True, fill=0.0))


def test_random_rotation_range_is_symmetric_around_zero():
    """``degrees`` is the half-width, so a negative reading is the same."""
    _assert_parity(RandomRotationNode, {"degrees": -15.0},
                   T.RandomRotation(15.0, expand=False, fill=0.0))


def test_color_jitter_matches_torchvision():
    _assert_parity(
        ColorJitterNode,
        {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1},
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1))


def test_color_jitter_each_channel_is_wired_to_its_own_parameter():
    """Four parameters that all reached the right keyword.

    Turning exactly one of them on has to differ from turning none on, and
    from turning a different one on -- which is what a copy-paste error in
    the ``T.ColorJitter(...)`` call would break.
    """
    image = _image()
    outputs = {}
    for name in ("brightness", "contrast", "saturation", "hue"):
        params = {"brightness": 0.0, "contrast": 0.0,
                  "saturation": 0.0, "hue": 0.0, name: 0.3}
        outputs[name] = _seeded(_build(ColorJitterNode, params), image)
    baseline = _seeded(_build(ColorJitterNode, {
        "brightness": 0.0, "contrast": 0.0, "saturation": 0.0, "hue": 0.0,
    }), image)
    for name, out in outputs.items():
        assert not torch.equal(out, baseline), name
    seen = list(outputs.values())
    for index, first in enumerate(seen):
        for second in seen[index + 1:]:
            assert not torch.equal(first, second)


def test_rand_augment_matches_torchvision():
    _assert_parity(
        RandAugmentNode, {"num_ops": 2, "magnitude": 9},
        T.RandAugment(num_ops=2, magnitude=9, num_magnitude_bins=31))


def test_rand_augment_num_ops_zero_is_the_identity():
    image = _image()
    out = _build(RandAugmentNode, {"num_ops": 0, "magnitude": 9})(image)
    assert torch.equal(T.ToTensor()(out), T.ToTensor()(image))


def test_rand_augment_rejects_a_magnitude_past_the_bin_count():
    with pytest.raises(ValueError, match="num_magnitude_bins"):
        RandAugmentNode().execute(
            {}, {"num_ops": 2, "magnitude": 31, "num_magnitude_bins": 31})


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_normalize_presets_match_torchvision(preset):
    mean, std = PRESETS[preset]
    sample = torch.rand(3, 8, 8)
    _assert_parity(NormalizeTransformNode, {"preset": preset},
                   T.Normalize(mean, std), sample=sample)


def test_normalize_custom_parses_a_comma_separated_list():
    sample = torch.rand(3, 8, 8)
    _assert_parity(
        NormalizeTransformNode,
        {"preset": "Custom", "mean": "0.1, 0.2, 0.3", "std": "0.4,0.5,0.6"},
        T.Normalize((0.1, 0.2, 0.3), (0.4, 0.5, 0.6)), sample=sample)


def test_normalize_half_preset_reproduces_the_old_hardcoded_behaviour():
    """The migration promise: a chain can rebuild a pre-#136 graph exactly."""
    sample = torch.rand(1, 8, 8)
    _assert_parity(NormalizeTransformNode, {"preset": "Half"},
                   T.Normalize((0.5,), (0.5,)), sample=sample)


def test_normalize_single_value_broadcasts_over_three_channels():
    out = _build(NormalizeTransformNode, {"preset": "Half"})(torch.ones(3, 4, 4))
    assert torch.equal(out, torch.ones(3, 4, 4))


def test_normalize_rejects_mismatched_channel_counts():
    with pytest.raises(ValueError, match="same number of channels"):
        NormalizeTransformNode().execute(
            {}, {"preset": "Custom", "mean": "0.1, 0.2", "std": "0.4"})


def test_normalize_rejects_a_zero_std():
    with pytest.raises(ValueError, match="std cannot contain 0"):
        NormalizeTransformNode().execute(
            {}, {"preset": "Custom", "mean": "0.1", "std": "0"})


def test_normalize_custom_needs_both_numbers():
    with pytest.raises(ValueError, match="needs both mean and std"):
        NormalizeTransformNode().execute(
            {}, {"preset": "Custom", "mean": "", "std": ""})


def test_imagenet_preset_carries_the_published_statistics():
    """A regression guard on the numbers themselves.

    Every torchvision pretrained checkpoint was trained against these; a
    typo here silently shifts the input distribution of a whole class of
    graphs and shows up only as slightly worse accuracy.
    """
    assert PRESETS["ImageNet"] == (
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


# ── chaining and order ───────────────────────────────────────────────────


def test_chaining_appends_in_edge_order():
    chain = RandomCropNode().execute({}, {"size": 32, "padding": 4})
    chain = RandomHorizontalFlipNode().execute(
        {"transform": chain["transform"]}, {"p": 0.5})
    chain = ToTensorTransformNode().execute({"transform": chain["transform"]}, {})
    chain = NormalizeTransformNode().execute(
        {"transform": chain["transform"]}, {"preset": "CIFAR-10"})

    steps = chain["transform"].transforms
    assert [type(step).__name__ for step in steps] == [
        "RandomCrop", "RandomHorizontalFlip", "ToTensor", "Normalize"]


def test_a_chain_is_one_flat_compose_not_a_nest():
    """Splicing, not nesting: ``len(pipeline.transforms)`` means what it says."""
    chain = ResizeTransformNode().execute({}, {"size": 8})
    for _ in range(3):
        chain = ResizeTransformNode().execute(
            {"transform": chain["transform"]}, {"size": 8})
    assert len(chain["transform"].transforms) == 4
    assert not any(isinstance(step, T.Compose)
                   for step in chain["transform"].transforms)


def test_the_whole_chain_runs_end_to_end():
    chain = RandomCropNode().execute({}, {"size": 32, "padding": 4})
    chain = ToTensorTransformNode().execute({"transform": chain["transform"]}, {})
    chain = NormalizeTransformNode().execute(
        {"transform": chain["transform"]}, {"preset": "CIFAR-10"})
    out = chain["transform"](_image())
    assert out.shape == (3, 32, 32)
    assert out.dtype is torch.float32


def test_compose_joins_chains_in_port_order():
    left = ResizeTransformNode().execute({}, {"size": 8})["transform"]
    right = ToTensorTransformNode().execute({}, {})["transform"]
    joined = ComposeTransformNode().execute(
        {"step_1": left, "step_2": right}, {"steps": 2})["transform"]
    assert [type(step).__name__ for step in joined.transforms] == [
        "Resize", "ToTensor"]

    swapped = ComposeTransformNode().execute(
        {"step_1": right, "step_2": left}, {"steps": 2})["transform"]
    assert [type(step).__name__ for step in swapped.transforms] == [
        "ToTensor", "Resize"]


def test_compose_skips_an_unwired_port_without_shifting_the_rest():
    first = ResizeTransformNode().execute({}, {"size": 8})["transform"]
    third = ToTensorTransformNode().execute({}, {})["transform"]
    joined = ComposeTransformNode().execute(
        {"step_1": first, "step_3": third}, {"steps": 3})["transform"]
    assert [type(step).__name__ for step in joined.transforms] == [
        "Resize", "ToTensor"]


def test_compose_ports_follow_the_steps_param():
    for count, expected in ((2, 2), (5, 5), (99, 8), (0, 2), ("bogus", 2)):
        ports = ComposeTransformNode.define_inputs_dynamic({"steps": count})
        assert [p.name for p in ports] == [
            f"step_{i}" for i in range(1, expected + 1)]
        assert all(p.data_type is DataType.TRANSFORM for p in ports)
        assert all(p.optional for p in ports)


def test_compose_static_inputs_are_the_default_params():
    assert ComposeTransformNode.define_inputs() == \
        ComposeTransformNode.define_inputs_dynamic({"steps": 2})


def test_compose_of_chains_flattens_both():
    left = RandomCropNode().execute({}, {"size": 32, "padding": 4})["transform"]
    left = RandomHorizontalFlipNode().execute(
        {"transform": left}, {"p": 0.5})["transform"]
    right = ToTensorTransformNode().execute({}, {})["transform"]
    right = NormalizeTransformNode().execute(
        {"transform": right}, {"preset": "Half"})["transform"]
    joined = ComposeTransformNode().execute(
        {"step_1": left, "step_2": right}, {"steps": 2})["transform"]
    assert [type(step).__name__ for step in joined.transforms] == [
        "RandomCrop", "RandomHorizontalFlip", "ToTensor", "Normalize"]


# ── helpers ──────────────────────────────────────────────────────────────


def test_compose_drops_none_and_splices_lists():
    pipeline = compose(None, [T.ToTensor(), None], T.Compose([T.ToTensor()]))
    assert len(pipeline.transforms) == 2


def test_pipeline_is_random_sees_through_a_compose():
    deterministic = compose(T.ToTensor(), T.Normalize((0.5,), (0.5,)))
    augmenting = compose(T.RandomHorizontalFlip(), T.ToTensor())
    assert pipeline_is_random(deterministic) is False
    assert pipeline_is_random(augmenting) is True
    assert pipeline_is_random(None) is False


def test_every_random_step_node_is_recognised_as_random():
    """The set that decides whether the seeding wrapper is installed.

    A node added here later whose class name is not in
    ``RANDOM_TRANSFORM_NAMES`` would silently lose its isolated stream, so
    the two lists are checked against each other rather than trusted.
    """
    for node_cls, params in (
        (RandomCropNode, {}),
        (RandomHorizontalFlipNode, {}),
        (RandomRotationNode, {}),
        (ColorJitterNode, {}),
        (RandAugmentNode, {}),
    ):
        assert pipeline_is_random(
            node_cls().execute({}, params)["transform"]), node_cls.NODE_NAME
    for node_cls, params in (
        (ResizeTransformNode, {}),
        (ToTensorTransformNode, {}),
        (NormalizeTransformNode, {}),
    ):
        assert not pipeline_is_random(
            node_cls().execute({}, params)["transform"]), node_cls.NODE_NAME


def test_a_built_chain_survives_pickling():
    """DataLoader workers under 'spawn' pickle the dataset, transform included."""
    chain = RandomCropNode().execute({}, {"size": 32, "padding": 4})
    chain = ToTensorTransformNode().execute({"transform": chain["transform"]}, {})
    restored = pickle.loads(pickle.dumps(chain["transform"]))
    assert restored(_image()).shape == (3, 32, 32)
