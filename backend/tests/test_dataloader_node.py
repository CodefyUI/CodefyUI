"""Tests for DataLoaderNode."""

from __future__ import annotations

import pickle
import random

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from app.core.seeding import make_worker_init_fn
from app.nodes.data.dataloader_node import DataLoaderNode


def _dataset(n=16, in_features=4):
    X = torch.randn(n, in_features)
    y = torch.randint(0, 2, (n,))
    return TensorDataset(X, y)


def test_node_metadata():
    assert DataLoaderNode.NODE_NAME == "DataLoader"
    assert DataLoaderNode.CATEGORY == "Data"


def test_wraps_dataset_in_dataloader():
    ds = _dataset()
    res = DataLoaderNode().execute(
        {"dataset": ds},
        {"batch_size": 4, "shuffle": False, "num_workers": 0},
    )
    assert isinstance(res["dataloader"], DataLoader)


def test_batch_size_param():
    ds = _dataset(n=16)
    res = DataLoaderNode().execute({"dataset": ds}, {"batch_size": 8})
    loader = res["dataloader"]
    assert loader.batch_size == 8
    first_batch = next(iter(loader))
    assert first_batch[0].shape[0] == 8


def test_shuffle_false_yields_deterministic_order():
    ds = _dataset()
    res = DataLoaderNode().execute(
        {"dataset": ds},
        {"batch_size": 4, "shuffle": False, "num_workers": 0},
    )
    batches_1 = [b[1].tolist() for b in res["dataloader"]]
    res2 = DataLoaderNode().execute(
        {"dataset": ds},
        {"batch_size": 4, "shuffle": False, "num_workers": 0},
    )
    batches_2 = [b[1].tolist() for b in res2["dataloader"]]
    assert batches_1 == batches_2


def test_default_batch_size_is_32():
    ds = _dataset(n=64)
    res = DataLoaderNode().execute({"dataset": ds}, {})
    assert res["dataloader"].batch_size == 32


# ── full parameterization + seeding (core#134) ────────────────────────────


def _loader(params: dict, context=None, dataset=None):
    return DataLoaderNode().execute(
        {"dataset": dataset if dataset is not None else _dataset()},
        params, context=context)["dataloader"]


def test_pin_memory_and_drop_last_reach_the_loader():
    loader = _loader({"batch_size": 4, "pin_memory": True, "drop_last": True})
    assert loader.pin_memory is True
    assert loader.drop_last is True


def test_drop_last_actually_drops_the_short_batch():
    """Observable, not just an attribute: 10 samples at batch 4 is 2, not 3."""
    dataset = _dataset(n=10)
    assert len(list(_loader({"batch_size": 4, "shuffle": False},
                            dataset=dataset))) == 3
    assert len(list(_loader({"batch_size": 4, "shuffle": False,
                             "drop_last": True}, dataset=dataset))) == 2


def test_prefetch_factor_is_dropped_when_there_are_no_workers():
    """torch rejects any prefetch_factor at num_workers=0.

    The node's default is 2, so forwarding it unconditionally would break
    every single-process loader in the gallery.
    """
    loader = _loader({"num_workers": 0, "prefetch_factor": 4})
    assert loader.num_workers == 0
    assert loader.prefetch_factor is None


def test_persistent_workers_without_workers_is_an_explicit_error():
    with pytest.raises(ValueError, match="requires num_workers > 0"):
        _loader({"num_workers": 0, "persistent_workers": True})


def test_defaults_reproduce_the_pre_change_loader():
    """A saved graph carrying only the three original keys is unaffected."""
    built = _loader({"batch_size": 8, "shuffle": False, "num_workers": 0})
    reference = DataLoader(_dataset(), batch_size=8, shuffle=False,
                           num_workers=0)

    assert built.batch_size == reference.batch_size
    assert built.pin_memory == reference.pin_memory
    assert built.drop_last == reference.drop_last
    assert built.num_workers == reference.num_workers
    assert built.prefetch_factor == reference.prefetch_factor


class _Ctx:
    """The slice of ExecutionContext a DataLoader node actually touches."""

    def __init__(self, seed=None, node_id="loader-1"):
        from app.core.execution_context import ExecutionContext

        self._inner = ExecutionContext(seed=seed)
        self._inner.current_node_id = node_id
        self.current_node_id = node_id

    def make_generator(self, label):
        return self._inner.make_generator(label)

    def derive_seed(self, label):
        return self._inner.derive_seed(label)


def _order(context, dataset):
    return [int(y) for _, y in
            _loader({"batch_size": 1, "shuffle": True}, context=context,
                    dataset=dataset)]


def test_seeded_context_fixes_the_shuffle_order():
    """Same seed, same epoch order; different seed, different order.

    The generator is what makes this true: without one the RandomSampler
    draws from the global RNG when the loop iterates it, long after this
    node ran.
    """
    dataset = _dataset(n=12)

    first = _order(_Ctx(seed=7), dataset)
    second = _order(_Ctx(seed=7), dataset)
    other = _order(_Ctx(seed=8), dataset)

    assert first == second
    assert first != other


def test_two_loaders_in_one_run_do_not_share_a_stream():
    """Distinct node ids give distinct orders under one run seed.

    Otherwise a train loader and a val loader would walk their samples in
    lockstep, which is a correlation nobody asked for.
    """
    dataset = _dataset(n=12)
    train = _order(_Ctx(seed=7, node_id="train-loader"), dataset)
    val = _order(_Ctx(seed=7, node_id="val-loader"), dataset)

    assert train != val


def test_no_seed_means_no_generator():
    """An unseeded run keeps torch's own default sampling."""
    assert _loader({"shuffle": True}, context=_Ctx(seed=None)).generator is None
    assert _loader({"shuffle": True}, context=None).generator is None


def test_seeded_loader_carries_a_worker_init_fn_only_when_it_has_workers():
    assert _loader({"num_workers": 0}, context=_Ctx(seed=3)).worker_init_fn is None
    assert _loader({"num_workers": 2}, context=_Ctx(seed=3)).worker_init_fn is not None


# ── multi-worker loaders must actually RUN (#188 review, C1) ─────────────


class _RandomTagDataset(torch.utils.data.Dataset):
    """Each item carries a draw from ``random`` — the stream torch does NOT
    reseed per worker. Module level so ``spawn`` can pickle it.
    """

    def __init__(self, n: int = 8) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int):
        return torch.tensor([float(index)]), torch.tensor(random.random())


def test_the_worker_init_fn_can_be_pickled():
    """``spawn`` PICKLES this argument, and a closure cannot be pickled.

    Windows and macOS spawn by default, and Python 3.14+ moves non-Mac POSIX
    to ``forkserver``, which pickles too. A closure here raised
    ``Can't pickle local object`` at ITERATION time — inside TrainingLoop,
    naming neither the seeding nor the DataLoader node. Cheap, direct
    regression guard for the shape of the callable.
    """
    init = make_worker_init_fn(11)
    revived = pickle.loads(pickle.dumps(init))

    revived(0)
    first = random.random()
    init(0)
    assert random.random() == first, "the revived callable seeds differently"


def test_a_seeded_multi_worker_loader_iterates_and_seeds_its_workers():
    """The end-to-end proof: real worker processes, real batches.

    The shipped test asserted only that ``worker_init_fn is not None`` and
    never iterated — precisely the "accepted but not applied" failure the
    brief's ``label_smoothing`` criterion exists to catch. This one starts
    the workers, so it fails outright if the callable cannot cross the
    process boundary.
    """
    dataset = _RandomTagDataset(8)

    def _tags(seed):
        loader = _loader(
            {"batch_size": 2, "shuffle": False, "num_workers": 2},
            context=_Ctx(seed=seed), dataset=dataset)
        return [round(float(tag), 12) for _, batch in loader for tag in batch]

    first = _tags(21)
    again = _tags(21)
    other = _tags(22)

    assert len(first) == 8, "the loader produced no batches"
    # Same seed -> the workers drew the same augmentations.
    assert first == again
    # A different run seed -> genuinely different ones.
    assert first != other
    # And the two workers did not walk the same stream: with per-worker
    # seeding the 8 values are distinct rather than two repeated halves.
    assert first[:2] != first[4:6]
