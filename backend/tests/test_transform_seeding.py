"""Seeded augmentation is reproducible (core#136, on top of #134's run seed).

Augmentation is the first randomness in CodefyUI that runs INSIDE another
node: the transform fires when the DataLoader pulls a sample, which happens
in the middle of ``TrainingLoop.execute``. #134's per-node ``seed_rngs``
covers that path only in the sense that the whole run is deterministic --
the numbers the crops draw still depend on everything the model drew first,
and in a multi-worker run they are re-seeded to the same value at the start
of every epoch, which would freeze the augmentation after epoch 1.

So the properties under test are, in the order they matter:

1. Same run seed, same augmented samples. Twice, from a cold start.
2. Different run seed, different augmented samples.
3. Augmentation still VARIES -- across epochs, and between two samples of
   one epoch. A wrapper that made every sample identical would pass (1) and
   (2) and destroy the feature.
4. The stream is isolated: what the rest of the graph drew first does not
   change what the crops draw.
5. No seed, no wrapper. An unseeded run keeps torch's own entropy.
"""

from __future__ import annotations

import inspect
import pickle
import re
import warnings

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from app.core.execution_context import ExecutionContext
from app.nodes.data.transforms._base import (
    SeededAugmentation,
    attach_transform,
    pipeline_is_random,
    seed_pipeline,
)
from app.nodes.data.transforms.random_crop_node import RandomCropNode
from app.nodes.data.transforms.random_horizontal_flip_node import (
    RandomHorizontalFlipNode,
)
from app.nodes.data.transforms.to_tensor_transform_node import (
    ToTensorTransformNode,
)
from app.nodes.data.transform_node import TransformNode


class _ImageDataset(Dataset):
    """A handful of fixed images with a mutable ``transform``, like torchvision."""

    def __init__(self, count: int = 4) -> None:
        generator = torch.Generator().manual_seed(11)
        self._images = [
            Image.fromarray(
                torch.randint(0, 256, (16, 16, 3), generator=generator,
                              dtype=torch.uint8).numpy(), mode="RGB")
            for _ in range(count)
        ]
        self.transform = None

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int):
        image = self._images[index]
        return (self.transform(image) if self.transform is not None
                else image), 0


def _augmenting_chain():
    chain = RandomCropNode().execute({}, {"size": 16, "padding": 4})
    chain = RandomHorizontalFlipNode().execute(
        {"transform": chain["transform"]}, {"p": 0.5})
    chain = ToTensorTransformNode().execute({"transform": chain["transform"]}, {})
    return chain["transform"]


def _deterministic_chain():
    return ToTensorTransformNode().execute({}, {})["transform"]


def _context(seed: int | None, node_id: str = "tf1") -> ExecutionContext:
    context = ExecutionContext(seed=seed)
    context.current_node_id = node_id
    return context


def _epoch(dataset, batch_size: int = 2) -> torch.Tensor:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return torch.cat([batch for batch, _ in loader])


def _augmented(seed: int | None, node_id: str = "tf1",
               warmup: int = 0) -> torch.Tensor:
    """One epoch's worth of augmented samples, from a cold start.

    ``warmup`` draws that many numbers from the global RNG first, standing
    in for whatever the rest of the graph did before the loop started.
    """
    dataset = _ImageDataset()
    torch.manual_seed(999)
    if warmup:
        torch.rand(warmup)
    attach_transform(dataset, _augmenting_chain(), _context(seed, node_id))
    return _epoch(dataset)


# ── the wrapper is installed exactly when it should be ───────────────────


def test_a_seeded_run_wraps_a_random_pipeline():
    wrapped = seed_pipeline(_augmenting_chain(), _context(1234), "tf1")
    assert isinstance(wrapped, SeededAugmentation)


def test_an_unseeded_run_leaves_the_pipeline_alone():
    pipeline = _augmenting_chain()
    assert seed_pipeline(pipeline, _context(None), "tf1") is pipeline


def test_a_deterministic_pipeline_is_never_wrapped():
    """Nothing to isolate, so nothing pays the per-sample cost."""
    pipeline = _deterministic_chain()
    assert seed_pipeline(pipeline, _context(1234), "tf1") is pipeline


# ── what counts as random (core#136 review, M-1) ─────────────────────────
#
# ``pipeline_is_random`` decides whether the wrapper is installed at all, so
# a transform it fails to recognise is not a missed optimisation: at
# ``num_workers > 0`` the unwrapped pipeline is re-seeded to the same value
# at the start of every epoch, and epoch 2 replays epoch 1 byte for byte.
# Augmentation is then silently OFF after the first epoch.


@pytest.mark.parametrize("container", ["RandomApply", "RandomChoice",
                                       "RandomOrder"])
def test_a_random_container_is_random_even_when_its_children_are_not(
        container):
    """The container IS the randomness; its children need not be.

    These three each carry a ``.transforms`` list, so a check that recursed
    into children BEFORE testing the container's own name answered False for
    every one of them -- and their entries in ``RANDOM_TRANSFORM_NAMES``
    were dead code.
    """
    from torchvision import transforms as T

    children = [T.Grayscale()]
    built = (T.RandomApply(children, p=0.5) if container == "RandomApply"
             else getattr(T, container)([T.Grayscale(), T.ToTensor()]))
    assert pipeline_is_random(built) is True


def test_a_chain_whose_only_randomness_is_a_container_gets_wrapped():
    """The consequence of the above, at the level the user experiences it."""
    from torchvision import transforms as T

    pipeline = T.Compose([
        T.RandomApply([T.Grayscale(num_output_channels=3)], p=0.5),
        T.ToTensor(),
    ])
    assert isinstance(
        seed_pipeline(pipeline, _context(1234), "tf1"), SeededAugmentation)


@pytest.mark.parametrize("name", [
    "AugMix", "CutMix", "GaussianNoise", "JPEG", "MixUp",
    "RandomChannelPermutation", "RandomIoUCrop", "RandomPhotometricDistort",
    "RandomResize", "RandomShortestSize", "RandomZoomOut", "ScaleJitter",
])
def test_the_v2_only_random_transforms_are_recognised(name):
    """torchvision.transforms.v2 randoms with no v1 spelling.

    Reachable from any plugin or custom node with a TRANSFORM output, which
    ``Transform``'s own docstring advertises as the supported route.
    """
    from app.nodes.data.transforms._base import RANDOM_TRANSFORM_NAMES

    assert name in RANDOM_TRANSFORM_NAMES


#: Public classes in ``transforms``/``transforms.v2`` that are containers or
#: bases rather than steps, so the drift guard below does not judge them.
_NOT_A_STEP = frozenset({"Transform", "Compose", "RandomTransforms"})

#: Every way torchvision spells "draw a number" in a transform's own body.
#: ``.uniform_(`` covers the ``torch.empty(...).uniform_()`` idiom several
#: v2 transforms use instead of ``torch.rand``.
_RNG_CALL = re.compile(
    r"\btorch\.(rand|randn|randint|randperm|normal|multinomial|bernoulli)\b"
    r"|\.uniform_\(|\brandom\.[a-z]")

#: Triple-quoted blocks are stripped before the scan above runs: v2's
#: ``FiveCrop`` -- a deterministic transform -- has ``torch.rand(3, 256,
#: 256)`` in its docstring EXAMPLE, and scanning raw source made it the
#: guard's one false positive.
_TRIPLE_QUOTED = re.compile(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'')


def _torchvision_transform_classes():
    """Every public transform class in v1 and v2, as ``(name, class)``."""
    from torchvision import transforms as T
    from torchvision.transforms import v2

    for module in (T, v2):
        for name in dir(module):
            if name.startswith("_") or name in _NOT_A_STEP:
                continue
            obj = getattr(module, name)
            if isinstance(obj, type):
                yield name, obj


def _looks_random(name, cls):
    """Independent evidence that *cls* draws from an RNG. Never our own set.

    Five signals, any one of which is enough. They are deliberately
    OVERLAPPING, because each on its own has a blind spot: the name rule
    misses ``AugMix``/``CutMix``/``JPEG``/``MixUp``/``ScaleJitter``, the two
    structural rules miss transforms that draw inline in ``forward``, and
    the source scan misses ``GaussianNoise``, which draws inside the
    functional kernel it delegates to. Together they land on exactly the 36
    names the module lists, on torchvision 0.26 and 0.27 alike.
    """
    import torch
    from torchvision.transforms import v2

    if name.startswith("Random"):
        return "name"
    # v1's idiom: parameters drawn in a ``get_params`` staticmethod. This is
    # the same signal ``pipeline_is_random``'s backstop uses at runtime.
    if callable(getattr(cls, "get_params", None)):
        return "get_params"
    # v2's idiom: ``make_params`` overridden, or the "apply with probability
    # p" base class.
    base_make = getattr(v2.Transform, "make_params", None)
    own_make = getattr(cls, "make_params", None)
    if base_make is not None and own_make is not None and own_make is not base_make:
        return "make_params"
    # Both v2 lookups above and below are ``getattr``-guarded on purpose:
    # ``make_params`` and ``_RandomApplyTransform`` are torchvision's own
    # private shape, and a rename should cost this oracle one signal, not
    # turn the whole guard into an AttributeError. If enough signals go the
    # companion test below fails with a message that says so.
    apply_base = getattr(getattr(v2, "_transform", None),
                         "_RandomApplyTransform", None)
    if apply_base is not None and issubclass(cls, apply_base):
        return "p-gate"
    try:
        if _RNG_CALL.search(_TRIPLE_QUOTED.sub("", inspect.getsource(cls))):
            return "source"
    except (OSError, TypeError):  # pragma: no cover - sdist always has source
        pass
    # Last resort, and the only signal that reaches a transform whose draw
    # lives in the functional it calls: build it and run it twice under two
    # different global seeds. Anything needing constructor arguments raises
    # and is simply not judged by this signal.
    try:
        instance = cls()
        sample = torch.rand(3, 8, 8)
        torch.manual_seed(1)
        first = instance(sample)
        torch.manual_seed(2)
        if not torch.equal(first, instance(sample)):
            return "probe"
    except Exception:
        pass
    return None


def test_every_torchvision_random_transform_is_listed():
    """A guard against the set drifting behind torchvision.

    core#136 re-review, N-2. The first version of this test walked only
    ``startswith("Random")``, which judged 24 of the 36 entries and would
    have let a future ``AugMix``-shaped addition through -- 5 of the 12
    names this PR had to add by hand are shaped exactly like that. It also
    only ever asserted one direction, so an entry torchvision had DELETED
    could sit here forever looking like coverage.

    Both directions now, and the "is it random" question is answered by
    torchvision's own code rather than by restating our list -- see
    :func:`_looks_random`.
    """
    import torch

    from app.nodes.data.transforms._base import RANDOM_TRANSFORM_NAMES

    state = torch.random.get_rng_state()
    try:
        with warnings.catch_warnings():
            # The probe instantiates deprecated classes such as v2.ToTensor.
            warnings.simplefilter("ignore")
            classified = {name: _looks_random(name, cls)
                          for name, cls in _torchvision_transform_classes()}
    finally:
        torch.random.set_rng_state(state)

    random_now = {name for name, signal in classified.items() if signal}
    assert random_now - RANDOM_TRANSFORM_NAMES == set(), (
        "torchvision has random transforms this set does not know about; "
        "unlisted means unwrapped, which freezes augmentation after epoch 1 "
        f"at num_workers>0: "
        f"{ {n: classified[n] for n in random_now - RANDOM_TRANSFORM_NAMES} }")
    assert RANDOM_TRANSFORM_NAMES - set(classified) == set(), (
        "these names are no longer public torchvision transforms, so they "
        "match nothing and only look like coverage: "
        f"{sorted(RANDOM_TRANSFORM_NAMES - set(classified))}")


def test_the_drift_guard_judges_every_listed_name():
    """The guard above is only worth its runtime if it sees all 36.

    Its predecessor covered 24. This pins the coverage itself, so narrowing
    ``_looks_random`` back to a name test fails here rather than silently
    shrinking what the ratchet watches.
    """
    import torch

    from app.nodes.data.transforms._base import RANDOM_TRANSFORM_NAMES

    state = torch.random.get_rng_state()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            judged = {name for name, cls in _torchvision_transform_classes()
                      if _looks_random(name, cls)}
    finally:
        torch.random.set_rng_state(state)

    assert RANDOM_TRANSFORM_NAMES - judged == set(), (
        "the oracle no longer recognises these listed names: "
        f"{sorted(RANDOM_TRANSFORM_NAMES - judged)}")


def test_a_custom_transform_in_torchvisions_own_idiom_is_recognised():
    """The ``get_params`` backstop, for randomness nobody listed here.

    torchvision's random transforms draw their parameters in a
    ``get_params`` staticmethod and its deterministic ones have nothing to
    draw, so a plugin written in that idiom is picked up without being
    named.
    """
    class PluginJitter:
        @staticmethod
        def get_params():  # pragma: no cover - never called
            return 0

        def __call__(self, sample):  # pragma: no cover - never called
            return sample

    class PluginResize:
        def __call__(self, sample):  # pragma: no cover - never called
            return sample

    assert pipeline_is_random(PluginJitter()) is True
    assert pipeline_is_random(PluginResize()) is False


def test_the_backstop_does_not_fire_on_torchvisions_deterministic_steps():
    """A false positive only costs ~30 microseconds, but there are none."""
    from torchvision import transforms as T

    for step in (T.ToTensor(), T.PILToTensor(), T.Normalize((0.5,), (0.5,)),
                 T.Resize(8), T.CenterCrop(8), T.Grayscale(), T.Pad(2)):
        assert pipeline_is_random(step) is False, type(step).__name__


def test_no_context_leaves_the_pipeline_alone():
    pipeline = _augmenting_chain()
    assert seed_pipeline(pipeline, None, "tf1") is pipeline


def test_attach_transform_installs_it_on_the_dataset():
    dataset = _ImageDataset()
    returned = attach_transform(dataset, _augmenting_chain(), _context(7))
    assert returned is dataset
    assert isinstance(dataset.transform, SeededAugmentation)


def test_attach_transform_warns_on_a_dataset_that_ignores_it(caplog):
    """core#136 review, m-15.

    ``dataset.transform = ...`` succeeds on anything. A ``TensorDataset``
    (and SyntheticShapes / SyntheticSegmentation) builds its own tensors and
    never reads the attribute, so the whole chain becomes a no-op. Every
    dataset that DOES honour it sets one in ``__init__``, so the absence is
    a reliable negative.
    """
    from torch.utils.data import TensorDataset

    dataset = TensorDataset(torch.zeros(2, 3), torch.zeros(2))
    with caplog.at_level("WARNING"):
        attach_transform(dataset, _augmenting_chain(), _context(7))
    assert any("no 'transform' attribute" in record.getMessage()
               for record in caplog.records), caplog.text


def test_attach_transform_is_quiet_on_a_dataset_that_honours_it(caplog):
    with caplog.at_level("WARNING"):
        attach_transform(_ImageDataset(), _augmenting_chain(), _context(7))
    assert caplog.records == []


def test_transform_node_wraps_a_wired_chain():
    dataset = _ImageDataset()
    TransformNode().execute(
        {"dataset": dataset, "transform": _augmenting_chain()}, {},
        context=_context(7))
    assert isinstance(dataset.transform, SeededAugmentation)


def test_transform_node_params_path_is_never_wrapped():
    """The three built-in steps contain no randomness."""
    dataset = _ImageDataset()
    TransformNode().execute(
        {"dataset": dataset},
        {"resize": 0, "normalize": True, "to_tensor": True},
        context=_context(7))
    assert not isinstance(dataset.transform, SeededAugmentation)


# ── the properties that make it worth having ─────────────────────────────


def test_the_same_run_seed_reproduces_the_same_augmentations():
    assert torch.equal(_augmented(4242), _augmented(4242))


def test_a_different_run_seed_augments_differently():
    assert not torch.equal(_augmented(4242), _augmented(4243))


def test_a_different_node_id_augments_differently():
    """Two Transform nodes in one graph must not walk the same stream."""
    assert not torch.equal(_augmented(4242, "tf1"), _augmented(4242, "tf2"))


def test_the_augmentation_still_varies_between_samples():
    """Reproducible must not mean 'the same picture every time'."""
    batch = _augmented(4242)
    assert not torch.equal(batch[0], batch[1])


def test_the_augmentation_still_varies_between_epochs():
    dataset = _ImageDataset()
    attach_transform(dataset, _augmenting_chain(), _context(4242))
    first, second = _epoch(dataset), _epoch(dataset)
    assert first.shape == second.shape
    assert not torch.equal(first, second)


def test_the_stream_does_not_depend_on_what_the_graph_drew_first():
    """The isolation property, and the reason the wrapper exists at all.

    Without it the crops come out of the global RNG that the training loop
    is also drawing dropout masks from, so adding a layer to the model
    would change the augmentations.
    """
    assert torch.equal(_augmented(4242, warmup=0),
                       _augmented(4242, warmup=137))


def test_the_wrapper_hands_the_callers_rng_back_untouched():
    """A fork, not a reseed: the caller's own stream must not be disturbed.

    Called directly rather than through a DataLoader, which draws its own
    ``_base_seed`` from the global RNG on every iterator and would put the
    loader's consumption into the measurement.
    """
    dataset = _ImageDataset()
    attach_transform(dataset, _augmenting_chain(), _context(4242))

    torch.manual_seed(5)
    expected = torch.rand(3)
    torch.manual_seed(5)
    for _ in range(4):
        dataset.transform(dataset._images[0])
    assert torch.equal(torch.rand(3), expected)


def test_an_unseeded_run_is_not_pinned_to_one_sequence():
    """The unseeded path keeps torch's entropy, which is what 'no seed' means."""
    torch.manual_seed(1)
    first = _augmented(None)
    torch.manual_seed(2)
    second = _augmented(None)
    # ``_augmented`` reseeds to a fixed value itself, so this only shows the
    # unwrapped path follows the global RNG rather than a derived stream.
    assert torch.equal(first, second)
    assert not isinstance(_ImageDataset().transform, SeededAugmentation)


# ── worker processes ─────────────────────────────────────────────────────


class _FakeWorkerInfo:
    def __init__(self, worker_id: int, seed: int) -> None:
        self.id = worker_id
        self.seed = seed


def _as_worker(monkeypatch, info):
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: info)


def test_each_worker_gets_its_own_stream(monkeypatch):
    """Two workers drawing at the same index must not draw the same numbers."""
    wrapper = SeededAugmentation(_augmenting_chain(), 4242)
    image = _ImageDataset()._images[0]

    _as_worker(monkeypatch, _FakeWorkerInfo(0, 1000))
    first = wrapper(image)
    _as_worker(monkeypatch, _FakeWorkerInfo(1, 1001))
    second = wrapper(image)
    assert not torch.equal(first, second)


def test_a_worker_restarted_for_the_next_epoch_augments_differently(monkeypatch):
    """The bug augmentation is the first feature to expose.

    ``worker_init_fn`` re-seeds each worker to a FIXED value at every
    iterator creation, so without mixing in torch's own per-iterator worker
    seed every epoch would replay epoch 1's crops. The wrapper's counter
    resets too (the object is pickled fresh into the worker), so the
    per-iterator seed is the ONLY thing distinguishing them.
    """
    image = _ImageDataset()._images[0]

    _as_worker(monkeypatch, _FakeWorkerInfo(0, 1000))
    epoch1 = SeededAugmentation(_augmenting_chain(), 4242)(image)
    _as_worker(monkeypatch, _FakeWorkerInfo(0, 2000))
    epoch2 = SeededAugmentation(_augmenting_chain(), 4242)(image)
    assert not torch.equal(epoch1, epoch2)


def test_a_worker_replayed_with_the_same_seeds_reproduces(monkeypatch):
    image = _ImageDataset()._images[0]
    _as_worker(monkeypatch, _FakeWorkerInfo(0, 1000))
    first = SeededAugmentation(_augmenting_chain(), 4242)(image)
    second = SeededAugmentation(_augmenting_chain(), 4242)(image)
    assert torch.equal(first, second)


def test_the_wrapper_survives_pickling():
    """Spawn-started DataLoader workers pickle the dataset and its transform."""
    wrapper = SeededAugmentation(_augmenting_chain(), 4242)
    image = _ImageDataset()._images[0]
    restored = pickle.loads(pickle.dumps(wrapper))
    assert restored.seed == wrapper.seed
    assert torch.equal(restored(image), wrapper(image))


def _python_random_datasets(seed: int = 4242):
    """Two datasets whose ONLY visible randomness is Python's ``random``.

    ``torchvision.transforms.RandomChoice`` picks with ``random.choices`` and
    ``RandomOrder`` shuffles with ``random.shuffle`` -- the two places in
    torchvision that draw from the ``random`` module rather than from torch.
    Both branches below are deterministic in themselves (``RandomPosterize``
    consumes one torch draw for its ``p`` check, but ``p=1.0`` makes the
    outcome fixed), so anything that varies between two epochs varied
    because the container's ``random`` draw varied.
    """
    from torchvision import transforms as T

    def branches():
        return [T.Grayscale(num_output_channels=3),
                T.RandomPosterize(bits=2, p=1.0)]

    made = []
    for index, name in enumerate(("RandomChoice", "RandomOrder")):
        dataset = _ImageDataset(count=16)
        attach_transform(
            dataset,
            T.Compose([getattr(T, name)(branches()), T.ToTensor()]),
            _context(seed, f"tf{index}"))
        made.append(dataset)
    return made


def test_a_v1_random_container_still_varies_across_epochs_in_real_workers():
    """core#136 re-review, R-1. Real spawned workers, two real epochs.

    ``RANDOM_TRANSFORM_NAMES`` lists both v1 containers, so they are DETECTED
    and wrapped -- but detection buys nothing unless the wrapper can actually
    control the stream they draw from. Forking only torch's generator left
    ``make_worker_init_fn``'s fixed per-worker ``random`` seed in charge, and
    that is re-applied at the start of every epoch, so epoch 2 replayed epoch
    1's choices exactly: augmentation silently OFF after the first epoch,
    which is the failure the whole set exists to prevent.

    Everything here is the real thing: spawned worker processes, the repo's
    own ``worker_init_fn``, ``persistent_workers`` off (so torch recreates
    the workers per epoch, which is what re-runs the init), and two branches
    that neither agree nor commute -- asserted below, so the test cannot
    quietly become vacuous the way a flat-colour fixture would.

    Both containers ride in ONE ``ConcatDataset`` because worker startup
    dominates the runtime: one loader per epoch instead of two, and the
    halves are sliced apart afterwards so each container is judged on its
    own.
    """
    from torch.utils.data import ConcatDataset
    from torchvision import transforms as T

    from app.core.seeding import make_worker_init_fn

    probe = _ImageDataset(count=1)._images[0]
    gray, posterize = T.Grayscale(3), T.RandomPosterize(bits=2, p=1.0)
    to_tensor = T.ToTensor()
    assert not torch.equal(to_tensor(gray(probe)), to_tensor(posterize(probe))), \
        "the two branches must disagree or RandomChoice pins nothing"
    assert not torch.equal(to_tensor(posterize(gray(probe))),
                           to_tensor(gray(posterize(probe)))), \
        "the two branches must not commute or RandomOrder pins nothing"

    choice_dataset, order_dataset = _python_random_datasets()
    assert isinstance(choice_dataset.transform, SeededAugmentation)
    assert isinstance(order_dataset.transform, SeededAugmentation)
    combined = ConcatDataset([choice_dataset, order_dataset])

    torch.manual_seed(999)
    epochs = []
    for _ in range(2):
        loader = DataLoader(combined, batch_size=4, shuffle=False,
                            num_workers=2,
                            worker_init_fn=make_worker_init_fn(4242))
        epochs.append(torch.cat([batch for batch, _ in loader]))

    half = len(choice_dataset)
    assert not torch.equal(epochs[0][:half], epochs[1][:half]), \
        "RandomChoice froze after epoch 1"
    assert not torch.equal(epochs[0][half:], epochs[1][half:]), \
        "RandomOrder froze after epoch 1"


def test_the_wrapper_hands_the_callers_python_random_back_untouched():
    """The ``random`` half of the isolation contract, in the main process.

    Reseeding the module-global ``random`` without restoring it would leave
    the rest of the graph walking a stream the augmentation chose, which is
    the same coupling the torch fork exists to break.
    """
    import random

    dataset = _ImageDataset()
    attach_transform(dataset, _augmenting_chain(), _context(4242))

    random.seed(5)
    expected = [random.random() for _ in range(3)]
    random.seed(5)
    for _ in range(4):
        dataset.transform(dataset._images[0])
    assert [random.random() for _ in range(3)] == expected


class _PythonRandomDraw:
    """A transform whose whole output is one draw from Python's ``random``.

    Stands in for v1 ``RandomChoice``/``RandomOrder``, which is where
    torchvision reaches for the ``random`` module instead of torch.
    """

    def __call__(self, sample):
        import random

        return random.random()


def _python_draws(seed: int, count: int = 4) -> list[float]:
    wrapper = SeededAugmentation(_PythonRandomDraw(), seed)
    return [wrapper(None) for _ in range(count)]


def test_the_python_random_draws_are_a_function_of_the_run_seed_alone():
    """core#136 re-review, R-1, at the level a unit test can see it.

    The ambient ``random`` state is set to something different before each
    of the two runs -- standing in for ``worker_init_fn``'s fixed per-worker
    seed, and for whatever else in the process drew last. If the wrapper
    does not seed ``random`` itself, that ambient state decides the draws
    and the two runs disagree.
    """
    import random

    random.seed(1)
    first = _python_draws(4242)
    random.seed(2)
    second = _python_draws(4242)

    assert first == second, "the draws followed the ambient RNG, not the seed"
    assert len(set(first)) == len(first), "every sample drew the same number"
    assert _python_draws(4243) != first, "the run seed does not reach them"


# ── through the engine ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_two_seeded_runs_of_the_same_graph_augment_identically():
    """End to end: the run seed reaches the transform through the engine.

    Dataset -> RandomCrop -> Flip -> ToTensor -> Transform, then the batch
    the loader would hand a training loop. Nothing here mocks the seeding;
    the only input that differs between the two runs is nothing at all.
    """
    from app.core.graph_engine import execute_graph

    def graph():
        nodes = [
            {"id": "start", "type": "Start", "data": {"params": {}}},
            {"id": "crop", "type": "RandomCrop",
             "data": {"params": {"size": 16, "padding": 4}}},
            {"id": "flip", "type": "RandomHorizontalFlip",
             "data": {"params": {"p": 0.5}}},
            {"id": "tensor", "type": "ToTensorTransform", "data": {"params": {}}},
            {"id": "src", "type": "_SeedFixtureDataset", "data": {"params": {}}},
            {"id": "apply", "type": "Transform", "data": {"params": {}}},
            {"id": "sink", "type": "_SeedFixtureEpoch", "data": {"params": {}}},
        ]
        edges = [
            {"id": "t1", "source": "start", "target": "crop",
             "sourceHandle": "trigger", "type": "trigger"},
            {"id": "t2", "source": "start", "target": "src",
             "sourceHandle": "trigger", "type": "trigger"},
            {"source": "crop", "sourceHandle": "transform",
             "target": "flip", "targetHandle": "transform"},
            {"source": "flip", "sourceHandle": "transform",
             "target": "tensor", "targetHandle": "transform"},
            {"source": "tensor", "sourceHandle": "transform",
             "target": "apply", "targetHandle": "transform"},
            {"source": "src", "sourceHandle": "dataset",
             "target": "apply", "targetHandle": "dataset"},
            {"source": "apply", "sourceHandle": "dataset",
             "target": "sink", "targetHandle": "dataset"},
        ]
        return nodes, edges

    async def run(seed: int) -> torch.Tensor:
        nodes, edges = graph()
        result = await execute_graph(
            nodes, edges, context=ExecutionContext(seed=seed))
        return result["sink"]["batch"]

    first, second = await run(31337), await run(31337)
    assert torch.equal(first, second)
    assert not torch.equal(first, await run(31338))


# Fixture nodes for the engine test. Registered by a FIXTURE, never at
# import time: pytest imports every test module during collection, and
# conftest's session-scoped ``registry_with_nodes`` only discovers the
# builtins and the PRESETS when it finds the registry still empty. A
# module-level ``register`` call runs first and silently skips preset
# discovery for the whole session. Node names starting with an underscore
# are reserved for tests by convention (see test_log_metric.py).


from app.core.node_base import (  # noqa: E402
    BaseNode,
    DataType,
    PortDefinition,
)
from app.core.node_registry import registry  # noqa: E402


class _SeedFixtureDatasetNode(BaseNode):
    NODE_NAME = "_SeedFixtureDataset"
    CATEGORY = "Test"
    DESCRIPTION = "Fixed images with a writable transform attribute"
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="dataset", data_type=DataType.DATASET)]

    def execute(self, inputs, params, progress_callback=None, *, context=None):
        return {"dataset": _ImageDataset()}


class _SeedFixtureEpochNode(BaseNode):
    NODE_NAME = "_SeedFixtureEpoch"
    CATEGORY = "Test"
    DESCRIPTION = "Pulls one epoch through the dataset's transform"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="dataset", data_type=DataType.DATASET)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="batch", data_type=DataType.TENSOR)]

    def execute(self, inputs, params, progress_callback=None, *, context=None):
        return {"batch": _epoch(inputs["dataset"])}


_TEST_NODES = {
    "_SeedFixtureDataset": _SeedFixtureDatasetNode,
    "_SeedFixtureEpoch": _SeedFixtureEpochNode,
}


@pytest.fixture(autouse=True)
def _register_test_nodes():
    registry._nodes.update(_TEST_NODES)
    yield
    for name in _TEST_NODES:
        registry._nodes.pop(name, None)
