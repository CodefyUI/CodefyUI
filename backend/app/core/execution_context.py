"""Execution context for tracking and cancelling graph runs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from .node_state_store import NodeStateStore


@dataclass
class ExecutionContext:
    """Shared context for a single graph execution run.

    Carries cancellation, parallelism settings, plus per-run feature flags
    (verbose step trace, weight persistence, backward gradient capture)
    that nodes consult to opt into educational behaviour.
    """

    execution_id: str = field(default_factory=lambda: str(uuid4()))
    max_workers: int = 4
    _cancel_event: asyncio.Event = field(default_factory=asyncio.Event)

    # A1: verbose step-trace mode. Instrumented nodes emit __steps__ when True.
    verbose: bool = False

    # A2: per-node weight persistence.
    graph_id: str = ""
    weights_persistent: bool = True
    node_state_store: "NodeStateStore | None" = None

    # A3: backward-pass gradient capture.
    backward_mode: bool = False
    auto_backward: bool = False

    # Mutated per-node by graph_engine before each execute() call so that
    # StatefulModuleMixin.get_or_build_module knows which node it is in.
    current_node_id: str = ""

    # Populated during a run when backward_mode is True. Maps
    # (node_id, port) -> tensor reference whose .grad we want to capture
    # after the backward pass. graph_engine writes here; capture_grads reads.
    grad_targets: dict[tuple[str, str], Any] = field(default_factory=dict)

    def cancel(self) -> None:
        """Signal cancellation."""
        self._cancel_event.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set()


class CancellationError(Exception):
    """Raised when a graph execution is cancelled."""
