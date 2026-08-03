"""Run-scoped randomness: one seed in, a reproducible run out.

The problem this solves
----------------------
Before #134 a run had a ``seed`` option that reached exactly one call site --
``torch.manual_seed`` and friends, once, immediately before the graph
started. That is enough for a *single-threaded* graph whose nodes draw from
the global RNG in a fixed order, and it is not enough for anything else:

* the engine executes independent nodes CONCURRENTLY (``max_workers``
  defaults to 4), so which node draws which numbers depends on thread
  scheduling -- the same seed gives a different model every time;
* a ``DataLoader`` with ``shuffle=True`` builds a ``RandomSampler`` that
  draws from the global RNG *when it is iterated*, which is inside some
  other node's execute();
* worker subprocesses get torch's own per-worker seeding, derived from the
  parent's global RNG state at iterator-creation time -- again, whatever
  the interleaving happened to leave there.

So this module does two things. :func:`derive_seed` turns (run seed, label)
into a stable child seed, which lets every node be seeded INDEPENDENTLY of
what any other node drew; and :func:`seed_rngs` applies one of those child
seeds to the global RNGs. ``graph_engine`` calls the pair before every node
invocation, and serialises node execution while a seed is set -- see
``execute_graph``. The result is the property the issue actually asks for:
two runs with the same seed produce bitwise-identical loss curves on CPU.

Nothing here is active unless a seed was requested. An unseeded run keeps
torch's own default entropy and the engine keeps its parallelism.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import logging
import os
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)

#: Seeds are drawn from, and derived into, the 32-bit space. ``numpy``'s
#: legacy ``RandomState`` refuses anything wider, and it is the narrowest of
#: the three RNGs we set -- so it sets the vocabulary for all of them.
SEED_SPACE = 2 ** 32

#: Largest accepted seed. Mirrors :data:`SEED_SPACE`; ``run_service`` imports
#: this rather than keeping its own copy.
MAX_SEED = SEED_SPACE - 1

#: cuBLAS needs this to make its reduction order reproducible, and it is read
#: ONCE when the CUDA context is created -- setting it later has no effect.
#: Harmless on a CPU-only box, which is why it is set unconditionally with
#: determinism rather than gated on ``torch.cuda.is_available()``.
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def derive_seed(run_seed: int, label: str) -> int:
    """A stable child seed for *label* under *run_seed*.

    Stable across processes and interpreter restarts, which rules out
    ``hash()`` (PYTHONHASHSEED randomises it per process, so the "same" seed
    would give different node seeds on the next run -- the exact bug this
    function exists to prevent). BLAKE2b is used as a plain deterministic
    mixer, not for any security property.

    Distinct labels give unrelated streams, so two nodes seeded from one run
    do not walk the same sequence -- a real hazard if a graph has two
    identically-shaped ``Linear`` layers that would otherwise get identical
    initial weights.
    """
    digest = hashlib.blake2b(
        f"{int(run_seed)}\x00{label}".encode("utf-8"), digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big") % SEED_SPACE


def seed_rngs(seed: int | None) -> None:
    """Seed ``random``, ``numpy`` and ``torch`` (plus CUDA) from *seed*.

    PROCESS-GLOBAL and best effort: a missing or unhappy backend is logged
    at debug level rather than failing a six-hour training run. ``None`` is
    a no-op, which is what "no seed requested" means everywhere in the run
    path.
    """
    if seed is None:
        return
    value = int(seed) % SEED_SPACE
    import random

    random.seed(value)
    try:
        import numpy as np

        np.random.seed(value)
    except Exception:  # pragma: no cover - numpy always present in practice
        logger.debug("numpy seeding skipped", exc_info=True)
    try:
        import torch

        torch.manual_seed(value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(value)
    except Exception:  # pragma: no cover - torch always present in practice
        logger.debug("torch seeding skipped", exc_info=True)


def apply_determinism(enabled: bool) -> None:
    """Ask torch for deterministic kernels. ``warn_only=True``, deliberately.

    Strict mode raises on any op with no deterministic implementation, which
    on a teaching canvas means a graph that ran yesterday suddenly dies with
    a stack trace from inside cuDNN. ``warn_only`` keeps the run alive and
    tells the user which op could not comply, which is the honest answer:
    "everything reproducible was made reproducible".

    Idempotent, and a no-op when *enabled* is false. Turning it back off is
    :func:`deterministic_scope`'s job, not this function's -- it does not
    know what the setting was before it was called.
    """
    if not enabled:
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)
    try:
        import torch

        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
    except Exception:  # pragma: no cover - torch always present in practice
        logger.debug("deterministic algorithms not applied", exc_info=True)


#: Nesting depth and the state to put back at depth 0. Guarded by a lock
#: because a server drives its runs from the event loop while
#: ``run_graph.py`` and the device smoke script drive theirs from their own
#: thread -- the counter is protected rather than assumed single-threaded.
_DETERMINISM_LOCK = threading.Lock()
_DETERMINISM_DEPTH = 0
_DETERMINISM_BASELINE: tuple[bool, bool, bool, str | None] | None = None


def determinism_depth() -> int:
    """How many :func:`deterministic_scope` scopes are open. For tests."""
    return _DETERMINISM_DEPTH


@contextlib.contextmanager
def deterministic_scope(enabled: bool):
    """Bound determinism to the RUNS, instead of latching the whole process.

    ``torch.use_deterministic_algorithms`` is process-wide with no per-run
    variant, so without this a single deterministic run silently made every
    later run in that server deterministic until restart -- and because
    ``TrainingLoopNode`` ORs in its own ``deterministic`` param, merely
    OPENING someone else's saved graph could do it. The setting is a
    property of a run; it now dies with the runs.

    Entered UNCONDITIONALLY by the engine, not only when determinism was
    requested, precisely so a node that turns it on mid-run is scoped too.

    REFCOUNTED, not save-and-restore per scope (#188 re-review, D1). Two
    OVERLAPPING runs with different values used to latch it on for the life
    of the process: A recorded "was off" and set it on, B then recorded
    "was on", A restored off, and B put its poisoned reading back -- on,
    forever, for every later run. The cpu queue admits two runs by default
    and neither of them has to be seeded, so that is an ordinary Tuesday and
    not a corner. The baseline is therefore captured by the FIRST scope to
    open and put back by the LAST one to close, which is the only reading of
    a process-wide setting that composes.

    The residual, stated rather than pretended away: while runs overlap, a
    run that did not ask for determinism can be dragged into a neighbour's.
    That is inherent in a process-global torch setting -- the alternative is
    serialising every run in the server.

    Be precise about how far it reaches, because the obvious guess is wrong.
    It is NOT bounded by the last run that asked for determinism: refcounting
    holds the baseline until the depth returns to zero, so an unbroken chain
    of overlapping plain runs keeps the setting on long after the run that
    asked for it finished. The real bound is process idleness -- determinism
    lifts once nothing at all is executing -- and every headless invoke opens
    a scope too, so it counts toward that chain. A run that needs the
    guarantee in both directions takes a seed, which also makes it run alone
    (see ``run_service._RunExclusion``).

    Restores ``CUBLAS_WORKSPACE_CONFIG`` as well. Clearing it does not
    un-apply anything in this process (cuBLAS reads it once, at CUDA context
    creation), but it stops the value leaking into every subprocess spawned
    afterwards -- a DataLoader worker among them.
    """
    global _DETERMINISM_DEPTH, _DETERMINISM_BASELINE
    try:
        import torch
    except Exception:  # pragma: no cover - torch always present in practice
        yield
        return

    with _DETERMINISM_LOCK:
        if _DETERMINISM_DEPTH == 0:
            _DETERMINISM_BASELINE = (
                torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
                "CUBLAS_WORKSPACE_CONFIG" in os.environ,
                os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            )
        _DETERMINISM_DEPTH += 1
    try:
        apply_determinism(enabled)
        yield
    finally:
        with _DETERMINISM_LOCK:
            _DETERMINISM_DEPTH -= 1
            baseline = (_DETERMINISM_BASELINE
                        if _DETERMINISM_DEPTH == 0 else None)
            if baseline is not None:
                _DETERMINISM_BASELINE = None
        if baseline is not None:
            previous, previous_warn, had_cublas, previous_cublas = baseline
            try:
                torch.use_deterministic_algorithms(previous,
                                                   warn_only=previous_warn)
            except Exception:  # pragma: no cover - defensive
                logger.debug("determinism not restored", exc_info=True)
            if had_cublas:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_cublas or ""
            else:
                os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


def make_generator(seed: int | None) -> Any | None:
    """A CPU ``torch.Generator`` seeded from *seed*, or ``None``.

    Handed to ``DataLoader(generator=...)`` so its ``RandomSampler`` draws
    from a stream nobody else touches -- the shuffle order then depends only
    on the run seed and the node's identity, not on how much randomness the
    rest of the graph consumed first.

    CPU generator even for a CUDA run: this is what torch's own samplers
    use, and a device generator would make the shuffle order depend on the
    accelerator.
    """
    if seed is None:
        return None
    try:
        import torch

        generator = torch.Generator()
        generator.manual_seed(int(seed) % SEED_SPACE)
        return generator
    except Exception:  # pragma: no cover - torch always present in practice
        logger.debug("generator not created", exc_info=True)
        return None


def seed_worker(base_seed: int, worker_id: int) -> None:
    """Seed one DataLoader worker process. MODULE LEVEL, and that is the point.

    ``DataLoader`` hands ``worker_init_fn`` to each worker, and under the
    **spawn** start method -- the default on Windows and on macOS since 3.8,
    and where Python 3.14+ is taking non-Mac POSIX via ``forkserver`` -- that
    argument is PICKLED. A closure cannot be pickled:
    ``AttributeError: Can't pickle local object``. So the callable handed to
    torch has to be a module-level function with its base seed bound by
    :func:`functools.partial`, which pickles by reference.

    This crashed at ITERATION time, inside ``TrainingLoop``, naming neither
    seeding nor the DataLoader node -- so it is worth the indirection.
    """
    seed_rngs(derive_seed(base_seed, f"worker:{worker_id}"))


def make_worker_init_fn(seed: int | None) -> Callable[[int], None] | None:
    """A ``worker_init_fn`` giving each DataLoader worker its own stream.

    ``None`` when unseeded, which is also ``DataLoader``'s default.

    torch does seed all three RNGs per worker on the pinned version, from
    ``base_seed + worker_id`` -- what it does NOT do is make that base a
    function of anything we control. It draws it from the loader's
    ``generator``, or from the process-global RNG when there is none, at
    ITERATION time: inside the training loop, after every other node has
    drawn. Overriding it with ``derive_seed(seed, "worker:N")`` makes each
    worker's stream a function of the run seed and this loader's identity
    and nothing else -- so worker count changes what is drawn, but
    re-running with the same settings does not.

    The result must survive ``pickle``; see :func:`seed_worker`.
    """
    if seed is None:
        return None
    return functools.partial(seed_worker, int(seed) % SEED_SPACE)
