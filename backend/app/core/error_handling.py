"""Error recovery types for graph execution."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ErrorMode(str, Enum):
    FAIL_FAST = "fail_fast"
    CONTINUE = "continue"
    RETRY = "retry"


@dataclass
class NodeError:
    """Represents a failed node execution."""

    node_id: str
    error: str
    traceback: str | None = None


def is_node_error(value: Any) -> bool:
    return isinstance(value, NodeError)


# ── out of memory (core#135) ──────────────────────────────────────────────
#
# A CUDA OOM used to surface as a raw
# ``torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB``
# with a paragraph of allocator statistics after it and no statement at all
# about what to change. It is also the single most common way a real
# training graph fails, and the one failure where the user's next action is
# entirely predictable: make the batch smaller, or make each batch cheaper.
# So it gets a type of its own, with the three levers this release actually
# provides written into the message.


#: The levers, in the order worth trying. Formatted into every
#: :class:`NodeOOMError` message.
OOM_HINTS = (
    "reduce batch_size on the DataLoader node",
    "set TrainingLoop's precision to bf16 (roughly halves activation memory "
    "on Ampere and newer)",
    "raise TrainingLoop's accumulate_steps and lower batch_size by the same "
    "factor, which keeps the effective batch identical",
    "make the model smaller, or shorten the sequence / shrink the image",
)


class NodeOOMError(RuntimeError):
    """A node ran out of accelerator memory.

    Carries the node it happened in, the device it happened on, the
    allocator's own numbers at the moment of failure, and the hints above.
    ``str(exc)`` is the whole thing, formatted for the error panel, because
    that is the only part of it most users will ever see.

    **Deliberately not retried, and deliberately not auto-shrunk.** Re-running
    the same allocation gets the same answer, and quietly halving the batch
    behind the user's back changes the numbers their run produces without
    telling them -- the same run would then mean two different things
    depending on how much VRAM happened to be free. The run fails, says what
    to change, and the user changes it.
    """

    def __init__(
        self,
        message: str,
        *,
        node_id: str = "",
        node_type: str = "",
        device: str = "",
        digest: str = "",
        hints: tuple[str, ...] = OOM_HINTS,
    ) -> None:
        self.node_id = node_id
        self.node_type = node_type
        self.device = device
        self.digest = digest
        self.hints = tuple(hints)
        super().__init__(message)

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        *,
        node_id: str = "",
        node_type: str = "",
        device: str = "",
        digest: str = "",
    ) -> "NodeOOMError":
        """Build the user-facing error for an underlying allocator failure."""
        where = f"Node {node_type or 'unknown'}"
        if node_id:
            where += f" ({node_id})"
        on = f" on {device}" if device else ""
        lines = [
            f"{where} ran out of memory{on}.",
            "",
            "What to try, cheapest first:",
        ]
        lines.extend(f"  - {hint}" for hint in OOM_HINTS)
        if digest:
            lines.extend(["", digest])
        lines.extend(["", f"Original error: {exc}"])
        error = cls(
            "\n".join(lines),
            node_id=node_id, node_type=node_type, device=device, digest=digest,
        )
        error.__cause__ = exc
        return error


def is_out_of_memory(exc: BaseException) -> bool:
    """True for an accelerator out-of-memory failure, whatever raised it.

    Three shapes, because torch has three:

    * ``torch.OutOfMemoryError`` (aliased as ``torch.cuda.OutOfMemoryError``),
      which is what a CUDA allocation failure raises on any modern build;
    * a plain ``RuntimeError`` whose message says so -- MPS reports
      ``"MPS backend out of memory"`` this way, and so do older torch builds
      and several out-of-tree backends;
    * a ``MemoryError``, which is what the CPU allocator raises.

    The message match is deliberately narrow (``out of memory`` /
    ``CUDA out of memory``) so an unrelated RuntimeError is never dressed up
    as an OOM with hints that would send the user in the wrong direction.
    """
    if isinstance(exc, MemoryError):
        return True
    try:
        import torch

        oom_type = getattr(torch, "OutOfMemoryError", None)
        if oom_type is not None and isinstance(exc, oom_type):
            return True
    except Exception:  # noqa: BLE001 - classification must never raise
        pass
    if not isinstance(exc, RuntimeError):
        return False
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text


def current_cuda_device() -> str:
    """``"cuda:N"`` for the device torch is pointed at, or ``""``.

    For the caller who has to report an out-of-memory failure without a
    run context to read the device off.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return ""
        return f"cuda:{int(torch.cuda.current_device())}"
    except Exception:  # noqa: BLE001 - no torch, no driver, no device
        return ""


def memory_digest(device: str = "") -> str:
    """One paragraph of allocator numbers for *device*, or "" when unknown.

    Read BEFORE any recovery runs, so it describes the failure rather than
    the cleanup. Compact by design: ``torch.cuda.memory_summary()`` is a
    60-line table that no error panel can show, so the four numbers that
    answer "was it this run's fault?" are pulled out of the same allocator
    statistics -- how much this process holds, how much of that is actually
    in use, the peak, and how much the card has in total.
    """
    kind = (device or "").split(":", 1)[0]
    if kind != "cuda":
        return ""
    try:
        import torch

        if not torch.cuda.is_available():
            return ""
        index = torch.device(device).index
        if index is None:
            index = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(index)
        reserved = torch.cuda.memory_reserved(index)
        peak = torch.cuda.max_memory_allocated(index)
        total = torch.cuda.get_device_properties(index).total_memory
    except Exception:  # noqa: BLE001 - a digest is never worth an exception
        logger.debug("could not read CUDA memory statistics", exc_info=True)
        return ""

    def _gb(value: int) -> str:
        return f"{value / (1024 ** 3):.2f} GiB"

    return (
        f"CUDA memory on {device}: {_gb(allocated)} held by live tensors, "
        f"{_gb(reserved)} reserved by the caching allocator, "
        f"{_gb(peak)} peak this process, {_gb(total)} on the card."
    )


def release_cached_memory(device: str = "") -> None:
    """Hand the caching allocator's free blocks back. Never raises.

    Worth doing after an OOM even though it frees nothing the run was
    using: the allocator holds RESERVED-but-unused blocks that another
    process (or the next run, at a different shape) cannot touch, and a
    fragmented reserve is a common reason the second attempt fails where
    the first nearly fitted.

    It cannot free what is still referenced, and immediately after an
    exception that includes every tensor held alive by the traceback still
    on the stack. So this is one half of the recovery; the caller dropping
    that node's cached outputs is the other.
    """
    kind = (device or "cuda").split(":", 1)[0]
    if kind != "cuda":
        return
    try:
        import torch

        if not torch.cuda.is_available():
            return
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - recovery must not raise over recovery
        logger.debug("could not empty the CUDA cache", exc_info=True)
