"""Tests for DataLoaderNode."""

from __future__ import annotations

import contextlib
import pickle
import random

import numpy as np
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
    """Each item carries a draw from ``random``, tagged with the worker that
    produced it. Module level so ``spawn`` can pickle it.

    The worker id is reported by the dataset rather than inferred from the
    batch index: torch hands batches out round-robin today, and a test that
    assumed it compared worker 0 with itself (#188 re-review, D4).
    """

    def __init__(self, n: int = 8) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int):
        info = torch.utils.data.get_worker_info()
        worker_id = -1 if info is None else info.id
        # float64 so the draw survives the process boundary exactly; the
        # default float32 rounds it and the comparison below is exact.
        return (torch.tensor([worker_id]),
                torch.tensor(random.random(), dtype=torch.float64))


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


def _tagged_draws(seed, dataset, node_id="loader-1"):
    """Run a real 2-worker loader; return ``{worker id: [draws in order]}``.

    Grouped by the worker that reported each item, so the comparison below
    is genuinely across processes.
    """
    loader = _loader(
        {"batch_size": 2, "shuffle": False, "num_workers": 2},
        context=_Ctx(seed=seed, node_id=node_id), dataset=dataset)
    by_worker: dict[int, list[float]] = {}
    for workers, values in loader:
        for worker_id, value in zip(workers.tolist(), values.tolist()):
            by_worker.setdefault(int(worker_id[0]), []).append(float(value))
    return by_worker


@contextlib.contextmanager
def _global_rngs_restored():
    """Put ``random``, numpy and torch back exactly as they were.

    ``seed_rngs`` seeds all THREE process-global RNGs, so a helper that
    saves only ``random``'s state hands every test that runs afterwards a
    numpy and a torch stream pinned to a derived constant (#190). Nothing
    in the suite depends on that today — the point is that nothing should
    be able to start, because the symptom would surface as an unrelated
    test going flaky depending on collection order, which is close to the
    most expensive class of bug to chase.
    """
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    try:
        yield
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])


def _predicted_draws(seed, node_id, worker_id, count):
    """What ``worker_init_fn`` promises worker *worker_id* will draw.

    Computed from the derivation alone — (run seed, this loader's node id,
    this worker) — which is the property the function exists for. torch's
    own per-worker seeding would produce a different stream, so this is
    what tells "our seeding took effect" apart from "torch seeded it".

    Seeds the globals to get there, and hands them back; see
    :func:`_global_rngs_restored`.
    """
    from app.core.execution_context import ExecutionContext
    from app.core.seeding import derive_seed, seed_rngs

    base = ExecutionContext(seed=seed).derive_seed(f"dataloader:{node_id}")
    with _global_rngs_restored():
        seed_rngs(derive_seed(base, f"worker:{worker_id}"))
        return [random.random() for _ in range(count)]


def test_predicted_draws_leaves_no_global_rng_pinned():
    """The measuring stick must not move what it measures (#190).

    Asserted by DRAWING either side rather than by comparing saved state
    blobs: a pinned RNG is only a problem because of what the next caller
    gets out of it, and two consecutive calls to a pinned generator return
    the same number.

    Compared PER RNG, and that is not cosmetic. A single tuple comparison
    passes as soon as any ONE component moves — and ``random`` always moves,
    because it is the one the old code did restore, so its stream advances
    across the two calls. The tuple version of this test was green against
    a helper that pinned both numpy and torch; only the per-generator
    version fails, and it names which one.
    """
    def _after_a_call() -> dict[str, float]:
        _predicted_draws(21, "loader-1", 0, 4)
        return {
            "random": random.random(),
            "numpy": float(np.random.random()),
            "torch": float(torch.rand(1)),
        }

    first, second = _after_a_call(), _after_a_call()
    pinned = sorted(name for name in first if first[name] == second[name])

    assert not pinned, (
        f"still pinned to the derived worker seed: {pinned} — the same "
        f"number came back from a fresh draw ({first} then {second})")


def test_a_seeded_multi_worker_loader_iterates_and_seeds_its_workers():
    """The end-to-end proof: real worker processes, real batches.

    The shipped test asserted only that ``worker_init_fn is not None`` and
    never iterated — precisely the "accepted but not applied" failure the
    brief's ``label_smoothing`` criterion exists to catch. This one starts
    the workers, so it fails outright if the callable cannot cross the
    process boundary.

    Its distinctness guard used to compare worker 0 with itself, which left
    two real bugs green (#188 re-review, D4): dropping ``worker_init_fn``
    entirely, and handing every worker the SAME seed — the exact "my four
    workers produced four identical crops" failure the function exists to
    prevent. The three assertions below are each aimed at one of those.
    """
    dataset = _RandomTagDataset(8)

    first = _tagged_draws(21, dataset)
    again = _tagged_draws(21, dataset)
    other = _tagged_draws(22, dataset)

    assert sorted(first) == [0, 1], (
        f"expected two worker processes, saw {sorted(first)}")
    assert sum(len(v) for v in first.values()) == 8, "no batches were produced"

    # Same seed -> the workers drew the same augmentations; a different run
    # seed -> genuinely different ones.
    assert first == again
    assert first != other

    # The two workers did not walk the same stream. Across PROCESSES, not
    # across two draws of one process: this is what a same-seed-for-every-
    # worker bug breaks, and the old guard did not look at it.
    assert first[0] != first[1]
    everything = [v for values in first.values() for v in values]
    assert len(set(everything)) == 8, f"duplicate draws across workers: {first}"

    # And the streams are OURS — derived from (run seed, node id, worker id)
    # rather than from torch's own base_seed. Without this, deleting
    # ``worker_init_fn`` leaves every assertion above satisfied, because
    # torch seeds its workers from the generator we already seeded.
    for worker_id, values in first.items():
        assert values == _predicted_draws(21, "loader-1", worker_id,
                                          len(values)), (
            f"worker {worker_id} did not draw from its derived seed")
