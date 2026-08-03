"""How many bytes a cached value actually occupies (core#135).

The three in-memory stores -- ``ExecutionCache``, ``RunOutputStore`` and
``NodeStateStore`` -- used to be bounded by ENTRY COUNT, which on a single
32 GB card is a limit in name only: 256 cached node outputs is either 40 MB
of scalars or 200 GB of feature maps, and the count cannot tell the two
apart. This module is the measuring stick that lets them be bounded by
bytes instead.

What is counted
---------------
Tensor payload, and the containers holding it:

* ``torch.Tensor`` -- its **storage**, not ``element_size() * nelement()``.
  For an ordinary contiguous tensor the two are identical, which is the case
  the acceptance criterion names. They differ for a VIEW: ``big[0:1]``
  reports a few kilobytes by the element formula while keeping the whole
  parent storage alive, and a cache bounded by the first number would happily
  hold gigabytes it believes are megabytes. Storage size is what the
  allocator cannot reuse, so storage size is what a memory budget must count.
* ``torch.nn.Module`` -- parameters and buffers, deduplicated (a tied
  embedding/decoder pair is one allocation, not two). Gradients and
  optimizer state are counted when present but are usually absent at the
  moment a module is stored; see the caveat below.
* ``numpy.ndarray`` -- ``.nbytes``.
* ``bytes``/``bytearray``/``memoryview`` -- their length; ``str`` -- its real
  allocation via ``sys.getsizeof``, because a base64 PNG on an IMAGE port is
  megabytes of genuine residency that must not read as zero.
* ``list``/``tuple``/``set``/``dict`` -- walked through, keys included. A
  dict of lists of tensors is the ordinary shape of a node's outputs, not an
  edge case.

Everything else measures 0. That is deliberate: this is a budget for the
payload that makes a machine run out of memory, not an ``sys.getsizeof``
crawl of every Python object in reach. A sklearn estimator, a DataLoader, an
int -- none of them is what fills a card.

Sharing, and the two ways it is handled
---------------------------------------
Within ONE measurement, a storage reached twice is counted once: dedup is
keyed on ``(device type, device index, storage data_ptr, storage nbytes)``,
so ``{"a": t, "b": t.view(-1)}`` measures one tensor, not two.

ACROSS measurements it is not, and cannot be: two cache entries holding the
same tensor each report its full size, so the store's running total
over-counts. That direction is the safe one -- the budget evicts slightly
earlier than strictly necessary -- and the alternative (a process-wide
refcounted ledger of every storage every store has ever seen) buys accuracy
nobody can observe at a cost everybody pays on every put.

Caveats, stated rather than hidden
----------------------------------
* Sizes are measured ONCE, when a value is stored. A module that later grows
  a gradient for every parameter (i.e. any module that gets trained) is
  still carried at its parameters-only size until it is stored again. The
  alternative is re-walking every stored module on every insert, which is
  O(total parameters) per cache write.
* CUDA and CPU bytes are added together. The stores hold whatever the run
  produced and a single number is what the budget compares; ``/api/health``
  reports the same total the eviction policy uses, so the two never
  disagree.
* A tensor whose storage cannot be read (meta, fake, sparse, or a subclass
  that overrides it) falls back to ``element_size() * nelement()``.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

#: How deep the walk goes before it stops descending. A cached value is a
#: node's output dict, not a recursive data structure; anything past this is
#: either pathological or not tensor payload. Cycles are handled separately
#: (by identity), so this only bounds honest-but-deep nesting.
MAX_WALK_DEPTH = 24

#: Cap on how many objects one measurement visits. A list of a million
#: Python ints is not tensor payload, and walking it to discover that would
#: put a visible pause on the hot path of every cache write.
MAX_WALK_ITEMS = 100_000


def _tensor_bytes(tensor: Any, seen_storage: set[tuple]) -> int:
    """Bytes of *tensor*'s storage, or 0 if that storage was already counted."""
    try:
        storage = tensor.untyped_storage()
        nbytes = int(storage.nbytes())
        device = tensor.device
        key = (device.type, device.index, int(storage.data_ptr()), nbytes)
    except Exception:  # noqa: BLE001 - meta/fake/sparse tensors have no storage
        try:
            return int(tensor.element_size()) * int(tensor.nelement())
        except Exception:  # noqa: BLE001 - not tensor-shaped after all
            return 0
    if key in seen_storage:
        return 0
    seen_storage.add(key)
    return nbytes


def value_bytes(value: Any) -> int:
    """Resident bytes of *value*, walking containers. Never raises.

    Iterative rather than recursive so a pathological nesting depth cannot
    raise ``RecursionError`` from inside a cache write; bounded by
    :data:`MAX_WALK_DEPTH` and :data:`MAX_WALK_ITEMS` so it cannot become
    the slow part of one either.

    "Never raises" is enforced here rather than merely intended, because of
    WHERE this runs: inside ``ExecutionCache.put``, which the engine calls
    from inside the node's own ``try``. An exception escaping a measurement
    would be caught as if the NODE had failed -- so a node that computed
    its result perfectly well would be reported as an error, and in
    ``fail_fast`` it would take the run down. The walk touches arbitrary
    user objects (a custom node's return value, a plugin's dataclass, a
    wrapper whose ``nbytes`` is a property that raises), so "no object in
    this repo triggers it" is not a guarantee about anything.

    A measurement that gives up returns 0, not the partial total: the walk
    is recursive, so the count lives in the frames the exception unwound.
    Under-counting keeps a value that should have been evicted, which costs
    memory; an exception loses a run. That is the trade being made, and it
    is worth knowing which way it errs before trusting a budget.
    """
    try:
        return _walk(value)
    except Exception:  # noqa: BLE001 - see above; a budget must not fail a run
        logger.debug("byte measurement failed; treating it as 0",
                     exc_info=True)
        return 0


def _walk(value: Any) -> int:
    """The measurement itself. See :func:`value_bytes` for the contract."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        torch = None  # type: ignore[assignment]

    total = 0
    visited: set[int] = set()
    seen_storage: set[tuple] = set()
    stack: list[tuple[Any, int]] = [(value, 0)]
    items = 0

    while stack:
        obj, depth = stack.pop()
        items += 1
        if items > MAX_WALK_ITEMS:
            logger.debug(
                "byte measurement stopped after %d objects; the total is a "
                "lower bound", MAX_WALK_ITEMS)
            break
        if obj is None or isinstance(obj, (bool, int, float, complex)):
            continue

        if torch is not None and isinstance(obj, torch.Tensor):
            total += _tensor_bytes(obj, seen_storage)
            continue

        if torch is not None and isinstance(obj, torch.nn.Module):
            if id(obj) in visited:
                continue
            visited.add(id(obj))
            for tensor in obj.parameters(recurse=True):
                total += _tensor_bytes(tensor, seen_storage)
                grad = tensor.grad
                if grad is not None:
                    total += _tensor_bytes(grad, seen_storage)
            for tensor in obj.buffers(recurse=True):
                total += _tensor_bytes(tensor, seen_storage)
            continue

        if isinstance(obj, (bytes, bytearray)):
            total += len(obj)
            continue
        if isinstance(obj, memoryview):
            total += obj.nbytes
            continue
        if isinstance(obj, str):
            total += sys.getsizeof(obj)
            continue

        # Guarded because ``nbytes`` may be a PROPERTY, and a property can
        # run arbitrary code: a lazily-loaded array that hits the disk, a
        # wrapper that raises for an unmaterialised value. A bare
        # ``getattr`` propagates everything that is not an AttributeError,
        # which is the one thing this function promises not to do.
        try:
            nbytes = getattr(obj, "nbytes", None)
        except Exception:  # noqa: BLE001 - an unreadable size measures 0
            nbytes = None
        if isinstance(nbytes, int) and type(obj).__module__.startswith("numpy"):
            total += nbytes
            continue

        if depth >= MAX_WALK_DEPTH:
            continue
        if isinstance(obj, dict):
            if id(obj) in visited:
                continue
            visited.add(id(obj))
            for key, item in obj.items():
                stack.append((key, depth + 1))
                stack.append((item, depth + 1))
            continue
        if isinstance(obj, (list, tuple, set, frozenset)):
            if id(obj) in visited:
                continue
            visited.add(id(obj))
            for item in obj:
                stack.append((item, depth + 1))
            continue

    return total


def format_bytes(count: int) -> str:
    """``1536`` -> ``"1.5 KB"``. For log lines and hint text only."""
    size = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size) < 1024.0 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"  # pragma: no cover - unreachable, loop returns
