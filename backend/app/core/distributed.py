"""Data-parallel training across the cards of one worker run (#792).

``TrainingLoop``'s ``strategy`` param picks how it trains. ``single`` (the
default) is the one-process loop every graph has always run. ``ddp`` trains
one model across several GPUs with ``torch.nn.parallel.DistributedDataParallel``,
and only does so in the one place that can host it: a run on a worker agent
whose device set has two or more CUDA cards. There the run subprocess
(``agent.child``) starts one process per card, a *rank* (``agent.ddp``);
everywhere else, the canvas, a run on this server, a worker run on one card,
an exported script, the loop trains in one process and says so
(:data:`FALLBACK_NOTE`).

This module holds the parts both sides need without importing torch at
module level: which nodes of a graph train with ``ddp``
(:func:`ddp_loop_ids`), the graph a rank other than 0 executes
(:func:`rank_graph`), and the process-wide :class:`RankState` a rank installs
before its graph runs. :class:`DataParallel` is what ``TrainingLoop`` uses
inside a rank: the DDP wrap, the ``DistributedSampler`` shards, the stop flag
every rank agrees on, and the all-reduced sums its metrics are computed from.

How a ``ddp`` run executes, in short. Rank 0 executes the whole graph and is
the only rank whose events, metrics and artifacts reach the control plane.
The other ranks execute only what the ``ddp`` loops need (their ancestors and
the loops themselves), with no signal consumer, so the checkpoint and
TensorBoard writers, which ask ``context.can_record_artifacts()`` first, write
nothing there; their warnings do reach the run, once each. They wait to
start until rank 0 reaches its first ``ddp`` loop (:meth:`DataParallel.enter`),
so a dataset that downloads on first use is downloaded once, by rank 0.
Nodes downstream of a loop run on rank 0 alone, on the trained model it
returns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

STRATEGY_PARAM = "strategy"
STRATEGY_SINGLE = "single"
STRATEGY_DDP = "ddp"
STRATEGIES = (STRATEGY_SINGLE, STRATEGY_DDP)

#: The node type whose ``strategy`` param this module reads.
TRAINING_LOOP = "TrainingLoop"

#: The fewest CUDA cards a worker run needs for its ``ddp`` loops to train
#: across them. An explicit plan (``agent.ddp.plan_for``) may use fewer.
MIN_DEVICES = 2

#: ``WarningSignal.kind`` of the note a ``ddp`` loop gives when it trains in
#: one process.
FALLBACK_WARNING_KIND = "ddp_not_available"

#: Prefix of that note in the node's ``__log__`` (the canvas Log tab).
NOTE_PREFIX = "[TrainingLoop] "

FALLBACK_NOTE = (
    "strategy=ddp trains across the CUDA cards of a run on a worker agent "
    "whose device set has two or more cards, and only for a TrainingLoop on "
    "the top level of the graph. This TrainingLoop is not in such a run, so "
    "it trains in one process on its own device, as with strategy=single."
)


class StrategyError(ValueError):
    """A graph whose ``ddp`` loops cannot run as one distributed run."""


# ── graphs ────────────────────────────────────────────────────────────────


def node_params(node: dict[str, Any]) -> dict[str, Any]:
    data = node.get("data")
    params = data.get("params") if isinstance(data, dict) else None
    return params if isinstance(params, dict) else {}


def _bypassed(node: dict[str, Any]) -> bool:
    data = node.get("data")
    return bool(data.get("bypassed")) if isinstance(data, dict) else False


def is_ddp_loop(node: dict[str, Any]) -> bool:
    """A ``TrainingLoop`` whose ``strategy`` is ``ddp``."""
    return (node.get("type") == TRAINING_LOOP
            and str(node_params(node).get(STRATEGY_PARAM) or "").strip().lower()
            == STRATEGY_DDP)


def ddp_loop_ids(graph: dict[str, Any]) -> list[str]:
    """The ids of the top-level, not bypassed ``ddp`` loops of *graph*, in
    node order. A loop inside a block or a preset is not one: it trains in
    one process (:data:`FALLBACK_NOTE`)."""
    return [str(node.get("id")) for node in graph.get("nodes") or []
            if isinstance(node, dict) and is_ddp_loop(node)
            and not _bypassed(node)]


def _edges(graph: dict[str, Any]) -> list[dict[str, Any]]:
    return [edge for edge in graph.get("edges") or []
            if isinstance(edge, dict) and edge.get("source") is not None
            and edge.get("target") is not None]


def ancestors(graph: dict[str, Any], targets: list[str]) -> set[str]:
    """*targets* and every node with a path into one of them, over data and
    trigger edges alike (a ``Start`` node is an ancestor of what it
    triggers)."""
    into: dict[str, list[str]] = {}
    for edge in _edges(graph):
        into.setdefault(str(edge["target"]), []).append(str(edge["source"]))
    seen: set[str] = set()
    frontier = list(targets)
    while frontier:
        node = frontier.pop()
        if node in seen:
            continue
        seen.add(node)
        frontier.extend(into.get(node, ()))
    return seen


def check_loops_ordered(graph: dict[str, Any], loop_ids: list[str]) -> None:
    """Refuse two ``ddp`` loops neither of which feeds the other.

    The ranks must enter the loops in one order, because every collective
    of one loop pairs with the same collective on every other rank. Two
    loops on separate branches could run concurrently, in a different order
    on each rank, and deadlock. A chain (one loop's model into the next
    one's) is fine.
    """
    for position, first in enumerate(loop_ids):
        for second in loop_ids[position + 1:]:
            if first in ancestors(graph, [second]) \
                    or second in ancestors(graph, [first]):
                continue
            raise StrategyError(
                f"TrainingLoop {first!r} and {second!r} both use "
                "strategy=ddp, and neither one feeds the other, so their "
                "ranks could run them in different orders. Connect them in "
                "sequence (one loop's model into the next), or set one of "
                "them to strategy=single.")


def rank_graph(graph: dict[str, Any], loop_ids: list[str]) -> dict[str, Any]:
    """The graph a rank other than 0 executes: *loop_ids* and their
    ancestors, with the edges between them. Block definitions and presets
    are kept whole (an ancestor may be a block)."""
    keep = ancestors(graph, loop_ids)
    pruned = dict(graph)
    pruned["nodes"] = [node for node in graph.get("nodes") or []
                       if isinstance(node, dict)
                       and str(node.get("id")) in keep]
    pruned["edges"] = [edge for edge in _edges(graph)
                       if str(edge["source"]) in keep
                       and str(edge["target"]) in keep]
    return pruned


# ── the rank's process-wide state ─────────────────────────────────────────


@dataclass
class RankState:
    """What a rank process knows about its place in the run. One per
    process, installed before its graph executes (:func:`install`)."""

    rank: int
    world_size: int
    #: This rank's device, as the run subprocess numbers its cards.
    device: str
    #: The backend of the default process group (``nccl`` or ``gloo``).
    backend: str
    #: The node ids of the ``ddp`` loops; another loop trains alone.
    loop_ids: frozenset[str]
    #: A ``gloo`` group for the small CPU collectives (the stop flag, the
    #: metric sums); None when the default group is ``gloo`` already.
    group: Any = None
    #: Rank 0's: tells the other ranks they may start their graph.
    release: Callable[[], None] | None = None
    released: bool = field(default=False)

    @property
    def is_main(self) -> bool:
        return self.rank == 0


_state: RankState | None = None


def install(state: RankState | None) -> None:
    """Make *state* this process's rank state (None clears it)."""
    global _state
    _state = state


def current() -> RankState | None:
    """This process's rank state, or None outside a ``ddp`` rank."""
    return _state


def requested_strategy(params: dict[str, Any]) -> str:
    """The ``strategy`` param, validated. Missing means ``single``."""
    value = str(params.get(STRATEGY_PARAM) or STRATEGY_SINGLE).strip().lower()
    if value not in STRATEGIES:
        raise ValueError(
            f"TrainingLoop: strategy {params.get(STRATEGY_PARAM)!r} is not "
            f"one of {', '.join(STRATEGIES)}.")
    return value


def data_parallel_for(params: dict[str, Any], context: Any, *,
                      log: logging.Logger | None = None
                      ) -> tuple["DataParallel | None", str | None]:
    """``(DataParallel, None)`` for a ``ddp`` loop inside a rank,
    ``(None, note)`` for a ``ddp`` loop anywhere else (the note is already
    delivered, see ``core.advisories``), ``(None, None)`` for ``single``."""
    if requested_strategy(params) != STRATEGY_DDP:
        return None, None
    state = current()
    node_id = getattr(context, "current_node_id", "") or ""
    if state is not None and node_id in state.loop_ids:
        return DataParallel(state), None
    from .advisories import emit_advisory

    note = emit_advisory(FALLBACK_NOTE, kind=FALLBACK_WARNING_KIND,
                         prefix=NOTE_PREFIX, context=context,
                         logger=log or logger)
    return None, note


# ── inside a rank ─────────────────────────────────────────────────────────


class DataParallel:
    """One ``ddp`` loop's view of its rank. Every method that communicates
    is called by every rank at the same point of the loop, which is what
    keeps their collectives paired."""

    def __init__(self, state: RankState) -> None:
        self.state = state

    @property
    def is_main(self) -> bool:
        return self.state.is_main

    @property
    def device(self) -> str:
        return self.state.device

    def enter(self) -> None:
        """Rank 0, on its first ``ddp`` loop: let the other ranks start
        their graph, now that everything upstream ran here once."""
        state = self.state
        if state.released:
            return
        state.released = True
        if state.release is not None:
            state.release()

    def config(self) -> dict[str, Any]:
        """What the loop's config frame and ``metrics`` add."""
        return {"strategy": STRATEGY_DDP,
                "world_size": self.state.world_size,
                "backend": self.state.backend}

    def wrap(self, model: Any) -> Any:
        """*model* (already on this rank's device) in a
        ``DistributedDataParallel``. Its constructor broadcasts rank 0's
        parameters and buffers, so every rank starts from the same
        weights."""
        import torch
        from torch.nn.parallel import DistributedDataParallel

        device = torch.device(self.device)
        if device.type == "cuda":
            index = device.index if device.index is not None \
                else torch.cuda.current_device()
            return DistributedDataParallel(model, device_ids=[index],
                                           output_device=index)
        return DistributedDataParallel(model)

    def _broadcast(self, value: int) -> int:
        import torch.distributed as dist

        box = [value]
        dist.broadcast_object_list(box, src=0, group=self.state.group)
        return int(box[0])

    def shard(self, loader: Any, port: str) -> Any:
        """*loader* rebuilt over a ``DistributedSampler``: this rank's
        ``1/world_size`` of the dataset, at the loader's own batch size.

        Shuffling follows the loader (a ``RandomSampler`` shuffles), with
        one seed every rank agrees on (rank 0's, from the loader's own
        generator when it has one, so a seeded run shuffles the same way
        every time). ``drop_last`` stays off on the sampler: it pads the
        last shard with repeated samples so every rank runs the same number
        of batches, which their collectives need (the stop flag is agreed
        once per batch, in validation too). :meth:`real_samples` says how
        many of a rank's samples are its own, and :meth:`trim` cuts the
        repeated ones out of a validation batch so they count nowhere.
        """
        import torch
        from torch.utils.data import (
            DataLoader,
            IterableDataset,
            RandomSampler,
            SequentialSampler,
        )
        from torch.utils.data.distributed import DistributedSampler

        if not isinstance(loader, DataLoader):
            raise ValueError(
                f"TrainingLoop: strategy=ddp needs a DataLoader on {port}, "
                f"got {type(loader).__name__}.")
        if isinstance(loader.dataset, IterableDataset):
            raise ValueError(
                f"TrainingLoop: strategy=ddp splits the dataset on {port} "
                "between the ranks by index, and an iterable dataset has no "
                "index. Use a map-style dataset, or strategy=single.")
        if loader.batch_size is None or not isinstance(
                loader.sampler, (RandomSampler, SequentialSampler)):
            raise ValueError(
                f"TrainingLoop: strategy=ddp rebuilds the DataLoader on "
                f"{port} with a DistributedSampler, which replaces a custom "
                "sampler or batch sampler. Use the DataLoader node's own "
                "shuffle, or strategy=single.")
        shuffle = isinstance(loader.sampler, RandomSampler)
        generator = getattr(loader, "generator", None)
        seed = int(generator.initial_seed()) if generator is not None \
            else int(torch.initial_seed())
        seed = self._broadcast(seed % (2 ** 63))
        sampler = DistributedSampler(
            loader.dataset, num_replicas=self.state.world_size,
            rank=self.state.rank, shuffle=shuffle, seed=seed, drop_last=False)
        kwargs: dict[str, Any] = {
            "batch_size": loader.batch_size,
            "sampler": sampler,
            "num_workers": loader.num_workers,
            "collate_fn": loader.collate_fn,
            "pin_memory": loader.pin_memory,
            "drop_last": loader.drop_last,
            "timeout": loader.timeout,
            "worker_init_fn": loader.worker_init_fn,
            "generator": generator,
        }
        if loader.num_workers > 0:
            kwargs["multiprocessing_context"] = loader.multiprocessing_context
            kwargs["persistent_workers"] = loader.persistent_workers
            kwargs["prefetch_factor"] = loader.prefetch_factor
        return DataLoader(loader.dataset, **kwargs)

    def real_samples(self, loader: Any) -> int:
        """How many samples of this rank's shard of *loader* (from
        :meth:`shard`) are the dataset's own, the rest being the padding
        ``DistributedSampler`` repeats at the end of the shard.

        The sampler lays the (shuffled) indices out in order, pads them to
        a multiple of the world size, and gives rank *r* every *N*-th one
        from position *r*; a position at or past the dataset's length is
        padding, and those are the last ones of each shard.
        """
        size = len(loader.dataset)
        rank, world = self.state.rank, self.state.world_size
        return max(0, -(-(size - rank) // world))

    @staticmethod
    def trim(data: Any, targets: Any, remaining: int) -> tuple[Any, Any, int]:
        """*data* and *targets* (a batch, on any device) cut to their first
        *remaining* samples, and how many samples the cut batch holds.

        The batch size is the first dimension of the first tensor found in
        *data* (then *targets*); tensors, and tensors inside lists, tuples
        and dicts, are cut along it. A batch with no tensor in it cannot be
        cut: it is returned whole and counted as one sample, so it is only
        ever skipped when the shard is exhausted at one sample per batch.
        """
        import torch

        size = _batch_size(data)
        if size is None:
            size = _batch_size(targets)
        if size is None:
            return data, targets, 1 if remaining > 0 else 0
        if remaining >= size:
            return data, targets, size
        keep = max(0, remaining)

        def cut(value: Any) -> Any:
            if isinstance(value, torch.Tensor) and value.dim() > 0 \
                    and value.shape[0] == size:
                return value[:keep]
            if isinstance(value, list):
                return [cut(item) for item in value]
            if isinstance(value, tuple):
                return tuple(cut(item) for item in value)
            if isinstance(value, dict):
                return {key: cut(item) for key, item in value.items()}
            return value

        return cut(data), cut(targets), keep

    @staticmethod
    def set_epoch(loader: Any, epoch: int) -> None:
        """A new shuffle for *epoch*, the same on every rank."""
        sampler = getattr(loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    def stop_checker(self, local: Callable[[], bool]) -> Callable[[], bool]:
        """A stop check every rank answers alike: True on all of them as
        soon as one rank's own check is. Called once per batch on every
        rank, so a Stop ends every rank's loop at the same batch."""
        import torch
        import torch.distributed as dist

        group = self.state.group

        def should_stop() -> bool:
            flag = torch.tensor([1 if local() else 0], dtype=torch.int32)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=group)
            return bool(flag.item())

        return should_stop

    def sums(self, *values: Any) -> list[float]:
        """Each of *values* (numbers or one-element tensors) summed over the
        ranks, in float64 on the CPU."""
        import torch
        import torch.distributed as dist

        local = torch.tensor([float(value) for value in values],
                             dtype=torch.float64)
        dist.all_reduce(local, op=dist.ReduceOp.SUM, group=self.state.group)
        return [float(total) for total in local.tolist()]

    def mean(self, value: Any) -> float:
        """*value* averaged over the ranks."""
        (total,) = self.sums(value)
        return total / self.state.world_size


def _batch_size(value: Any) -> int | None:
    """The first dimension of the first tensor in *value*, or None."""
    import torch

    if isinstance(value, torch.Tensor):
        return int(value.shape[0]) if value.dim() > 0 else None
    if isinstance(value, (list, tuple)):
        for item in value:
            size = _batch_size(item)
            if size is not None:
                return size
    elif isinstance(value, dict):
        for item in value.values():
            size = _batch_size(item)
            if size is not None:
                return size
    return None
