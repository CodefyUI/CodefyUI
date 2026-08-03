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

import hashlib
import logging
import os
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

    Idempotent, and a no-op when *enabled* is false -- it never turns
    determinism back OFF, because it does not own the process-wide setting
    and another run may have asked for it.
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


def make_worker_init_fn(seed: int | None) -> Callable[[int], None] | None:
    """A ``worker_init_fn`` giving each DataLoader worker its own stream.

    ``None`` when unseeded, which is also ``DataLoader``'s default.

    torch already reseeds ``torch``'s RNG per worker; it does NOT touch
    ``random`` or ``numpy``, and a dataset whose ``__getitem__`` augments
    with either would hand every worker the SAME augmentations (the classic
    "my four workers produced four identical crops" bug). Seeding all three
    from ``derive_seed(seed, "worker:N")`` closes that and keeps the streams
    a function of the run seed alone -- so worker count changes what is
    drawn, but re-running with the same settings does not.
    """
    if seed is None:
        return None
    base = int(seed) % SEED_SPACE

    def _init(worker_id: int) -> None:
        seed_rngs(derive_seed(base, f"worker:{worker_id}"))

    return _init
