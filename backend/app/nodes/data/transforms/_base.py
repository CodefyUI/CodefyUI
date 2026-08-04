"""The chaining contract shared by every transform node, plus the seeding.

What travels on a TRANSFORM edge
--------------------------------
A ``torchvision.transforms.Compose``, always -- never a bare callable and
never a list. That single decision is what makes the port type useful rather
than decorative: anything holding one can assign it straight to a dataset's
``transform`` attribute, an exported script gets a real torchvision object,
and :func:`compose` can flatten two of them into one without inspecting
types it does not own.

How a chain is built
--------------------
Every step node takes an OPTIONAL ``transform`` input and returns its own
step appended to it. Wiring ``RandomCrop -> RandomHorizontalFlip ->
ToTensorTransform -> NormalizeTransform`` therefore produces exactly
``Compose([RandomCrop, RandomHorizontalFlip, ToTensor, Normalize])``, in the
order the edges are drawn. ``ComposeTransform`` exists for the other shape --
several chains built in parallel and merged in a fixed order.

Order is the user's responsibility and torchvision's rules still apply: the
geometric and photometric steps want a PIL image, ``ToTensor`` converts, and
``Normalize`` needs a tensor. A chain that gets this wrong fails inside the
DataLoader with torchvision's own message, which names the step.

Reproducibility
---------------
Augmentation is the first randomness in CodefyUI that runs INSIDE another
node: the transform is applied when the DataLoader pulls a sample, which is
in the middle of ``TrainingLoop.execute`` (or in a worker process). #134
seeds the global RNGs per node and serialises a seeded run, so that path is
already deterministic -- but the numbers it draws depend on everything the
training loop drew before it, so changing the dropout rate would change the
crops. #134's stated goal is that a node's randomness depend on the run seed
and its own identity and on nothing else, and :class:`SeededAugmentation` is
what gives the transform chain that property.

It also fixes something augmentation is the first feature to expose. A
seeded multi-worker run installs a ``worker_init_fn`` that seeds each worker
to a FIXED value, and torch re-runs it at every epoch (workers are recreated
unless ``persistent_workers`` is on) -- so every epoch would apply the SAME
crops and flips, quietly turning augmentation off after epoch 1. The wrapper
mixes in ``get_worker_info().seed``, which torch draws afresh from the
loader's own generator for each iterator, so the stream advances per epoch
and is still a function of the run seed alone.
"""

from __future__ import annotations

import logging
from typing import Any

from ....core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    PortDefinition,
)

logger = logging.getLogger(__name__)

#: Port descriptions, written once so nine nodes cannot drift apart.
CHAIN_INPUT_DESC = (
    "Steps to run before this one. Leave unwired to start a new chain."
)
CHAIN_OUTPUT_DESC = "The chain so far, with this step appended"

#: torchvision transform classes that draw from the RNG. Used only to decide
#: whether wrapping a pipeline in :class:`SeededAugmentation` would buy
#: anything -- a deterministic pipeline is left alone so the per-sample cost
#: is paid only where there is randomness to isolate.
#:
#: Matched by CLASS NAME rather than by identity because torchvision moved
#: these between ``transforms`` and ``transforms.v2`` and both spellings are
#: in use. A random transform this set does not know about is not a
#: correctness problem: it falls back to the global RNG, which the engine
#: seeds per node, so a seeded single-process run still reproduces. It only
#: loses the isolation.
RANDOM_TRANSFORM_NAMES = frozenset({
    "AutoAugment",
    "ColorJitter",
    "ElasticTransform",
    "GaussianBlur",
    "RandAugment",
    "RandomAdjustSharpness",
    "RandomAffine",
    "RandomApply",
    "RandomAutocontrast",
    "RandomChoice",
    "RandomCrop",
    "RandomEqualize",
    "RandomErasing",
    "RandomGrayscale",
    "RandomHorizontalFlip",
    "RandomInvert",
    "RandomOrder",
    "RandomPerspective",
    "RandomPosterize",
    "RandomResizedCrop",
    "RandomRotation",
    "RandomSolarize",
    "RandomVerticalFlip",
    "TrivialAugmentWide",
})


def compose(*parts: Any) -> Any:
    """Flatten *parts* into one ``Compose``, preserving order.

    ``None`` parts are dropped (that is what an unwired optional input
    looks like), a ``Compose`` is spliced in rather than nested, and a
    list/tuple is spliced too. Everything else is appended as one step.

    Splicing rather than nesting matters for what the user sees: a chain of
    six steps prints as one ``Compose`` of six, not as six ``Compose``es
    inside each other, and ``len(pipeline.transforms)`` means what it looks
    like it means.
    """
    from torchvision import transforms as T

    steps: list[Any] = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, T.Compose):
            steps.extend(part.transforms)
        elif isinstance(part, (list, tuple)):
            steps.extend(item for item in part if item is not None)
        else:
            steps.append(part)
    return T.Compose(steps)


def pipeline_is_random(pipeline: Any) -> bool:
    """Does *pipeline* contain a step that draws from the RNG?

    Recurses into ``Compose`` because a merged chain is one level deep at
    most, and into the wrapper below so re-checking an already-wrapped
    pipeline answers the same question.
    """
    if pipeline is None:
        return False
    if isinstance(pipeline, SeededAugmentation):
        return True
    steps = getattr(pipeline, "transforms", None)
    if isinstance(steps, (list, tuple)):
        return any(pipeline_is_random(step) for step in steps)
    return type(pipeline).__name__ in RANDOM_TRANSFORM_NAMES


class SeededAugmentation:
    """Runs *transform* on a stream derived from the run seed and a label.

    One instance wraps one dataset's pipeline. Each call takes a child seed
    of ``(base seed, stream, call index)``, applies the transform under
    :func:`torch.random.fork_rng`, and hands the caller's RNG state back
    untouched -- so the model's dropout masks and the augmentation's crops
    no longer draw from the same sequence and neither can shift the other.

    ``devices=[]`` on the fork is deliberate: only the CPU generator is
    saved and restored, and only the CPU generator is reseeded
    (``torch.default_generator``, not ``torch.manual_seed``, which would
    also reseed every CUDA device). torchvision's random transforms are all
    CPU-side, and reseeding a CUDA generator per sample would both cost a
    synchronisation and corrupt the model's own GPU randomness.

    **Per-stream counters, not one counter.** ``num_workers > 0`` pickles a
    copy of this object into every worker, so a single counter would restart
    at 0 in each of them and in each epoch. The stream key carries the
    worker id and the per-iterator seed torch assigns it, which makes the
    streams disjoint across workers and fresh across epochs while staying a
    pure function of the run seed. In the main process there is one stream
    and its counter simply keeps going, which gives epoch 2 different
    augmentations from epoch 1 for the same reason.

    Costs about 6 microseconds per sample on top of a transform that
    typically costs 80, and is only installed when the pipeline actually
    contains randomness and the run actually has a seed.
    """

    __slots__ = ("transform", "seed", "_counters")

    def __init__(self, transform: Any, seed: int) -> None:
        self.transform = transform
        self.seed = int(seed)
        self._counters: dict[str, int] = {}

    def _next_label(self) -> str:
        import torch

        info = torch.utils.data.get_worker_info()
        if info is None:
            stream = "main"
        else:
            # ``info.seed`` is torch's own per-worker, per-iterator seed,
            # drawn from the DataLoader's generator -- which the DataLoader
            # node seeds from the run seed. So it varies per epoch and is
            # still reproducible, which is exactly the pair we need.
            stream = f"w{info.id}.{info.seed}"
        index = self._counters.get(stream, 0)
        self._counters[stream] = index + 1
        return f"{stream}:{index}"

    def __call__(self, sample: Any) -> Any:
        import torch

        from ....core.seeding import SEED_SPACE, derive_seed

        child = derive_seed(self.seed, self._next_label())
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.default_generator.manual_seed(child % SEED_SPACE)
            return self.transform(sample)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"SeededAugmentation(seed={self.seed}, {self.transform!r})"

    # ``Compose``-shaped so ``pipeline_is_random`` and any caller that
    # introspects a pipeline sees through the wrapper.
    @property
    def transforms(self) -> list[Any]:
        return [self.transform]


def seed_pipeline(
    pipeline: Any,
    context: Any,
    label: str,
) -> Any:
    """Wrap *pipeline* for reproducibility, or return it unchanged.

    Unchanged when there is no run seed (an unseeded run keeps torch's own
    entropy, which is what "no seed" means everywhere else in the run path)
    or when nothing in the pipeline is random (nothing to isolate, so
    nothing to pay for).
    """
    if pipeline is None or context is None:
        return pipeline
    if not pipeline_is_random(pipeline):
        return pipeline
    derive = getattr(context, "derive_seed", None)
    if not callable(derive):
        return pipeline
    seed = derive(f"transform:{label}")
    if seed is None:
        return pipeline
    return SeededAugmentation(pipeline, seed)


def seeded_for_node(pipeline: Any, context: Any) -> Any:
    """:func:`seed_pipeline` labelled with the executing node's id.

    What a node that CONSTRUCTS a dataset calls, so the pipeline can go
    into the constructor the way torchvision expects rather than being
    written over the top afterwards.
    """
    label = getattr(context, "current_node_id", "") or "transform"
    return seed_pipeline(pipeline, context, label)


def attach_transform(
    dataset: Any,
    pipeline: Any,
    context: Any = None,
    label: str = "",
) -> Any:
    """Install *pipeline* on an EXISTING *dataset*, seeded when asked for.

    Returns the dataset, which is mutated in place -- the same contract
    ``TransformNode`` has always had, and what torchvision datasets expect
    (``transform`` is a public, writable attribute on all of them). A node
    that builds the dataset itself should pass :func:`seeded_for_node` to
    the constructor instead.
    """
    if pipeline is None:
        return dataset
    node_id = label or getattr(context, "current_node_id", "") or "transform"
    dataset.transform = seed_pipeline(pipeline, context, node_id)
    return dataset


class TransformStepNode(BaseNode):
    """One step of a chain: takes the steps before it, appends its own.

    Subclasses implement :meth:`build` and declare their own params. The
    ports are identical for all of them by design -- a chain is only
    readable if every link looks the same.
    """

    # Empty, so the registry's discovery skips this base class.
    NODE_NAME = ""
    CATEGORY = "Data"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="transform",
                data_type=DataType.TRANSFORM,
                description=CHAIN_INPUT_DESC,
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="transform",
                data_type=DataType.TRANSFORM,
                description=CHAIN_OUTPUT_DESC,
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return []

    def build(self, params: dict[str, Any]) -> Any:
        """The torchvision step this node contributes. Subclass hook."""
        raise NotImplementedError

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        progress_callback: Any | None = None,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        return {"transform": compose(inputs.get("transform"),
                                     self.build(params))}
