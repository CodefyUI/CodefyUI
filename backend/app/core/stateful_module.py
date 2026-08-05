"""Mixin that lets a layer node persist its ``nn.Module`` across runs.

Use as the first base class on a node alongside :class:`BaseNode`:

    class Conv2dNode(StatefulModuleMixin, BaseNode):
        structural_params = ("in_channels", "out_channels", "kernel_size", "stride", "padding")

        def build_module(self, params):
            import torch.nn as nn
            return nn.Conv2d(**{k: params[k] for k in self.structural_params})

        def execute(self, inputs, params, *, context=None):
            module = self.get_or_build_module(context, params)
            return {"tensor": module(inputs["tensor"])}

When ``context.weights_persistent`` is False (or ``context`` itself is None,
e.g. for the legacy CLI runner) the mixin transparently rebuilds the module
on every call — preserving the original stateless behaviour.

Stateful nodes opt out of :class:`ExecutionCache` via ``cacheable = False``:
the cache assumes ``f(params, upstream) -> output`` is deterministic, which
breaks once the module's internal weights drift.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    import torch

    from .execution_context import ExecutionContext

logger = logging.getLogger(__name__)


class StatefulModuleMixin:
    """Mixin for nodes that wrap a single ``nn.Module`` with persistent weights."""

    # Tuple of param names whose values define the module's *shape*. When any
    # of these change, the persisted module is dropped and a fresh one is
    # built (old weights would have incompatible shapes anyway).
    structural_params: ClassVar[tuple[str, ...]] = ()

    # The cache assumes the same params produce the same outputs — false
    # for trainable layers whose weights change between runs.
    cacheable: ClassVar[bool] = False

    def build_module(self, params: dict[str, Any]) -> "torch.nn.Module":
        """Build a fresh ``nn.Module``. Subclasses must implement."""
        raise NotImplementedError

    # ── helpers ─────────────────────────────────────────────────────

    def _normalise_for_hash(self, params: dict[str, Any]) -> dict[str, Any]:
        """Hook for subclasses to normalise param values before hashing.

        Default: pass through. Override when a sentinel value (e.g.
        ``padding_idx=-1`` meaning ``None``) needs collapsing so two
        equivalent configurations don't get separate hashes.
        """
        return dict(params)

    def _structure_hash(self, params: dict[str, Any]) -> str:
        keys = self.structural_params or tuple(sorted(params.keys()))
        normalised = self._normalise_for_hash({k: params.get(k) for k in keys})
        payload = json.dumps(normalised, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def get_or_build_module(
        self,
        context: "ExecutionContext | None",
        params: dict[str, Any],
    ) -> "torch.nn.Module":
        """Look up the persisted module via NodeStateStore, building if missing.

        Falls back to a fresh build when persistence isn't wired up — keeps
        stateless behaviour for tests / CLI / direct ``execute()`` calls.

        The returned module is moved to ``context.device`` (the global run
        device), so every layer node runs on the chosen accelerator without
        each node needing its own device handling. Moving is idempotent and,
        for a persisted module, carries its trained weights across a
        device switch.

        When storing this module pushes the store past a limit and some
        OTHER node's weights are discarded to make room, the loss is
        reported on the run's event stream (#135 review). A server log line
        is not somewhere a user watching the canvas will look, and the thing
        being lost is training -- the next run silently restarts that node
        from a fresh initialisation.
        """
        if (
            context is None
            or not context.weights_persistent
            or context.node_state_store is None
            or not context.current_node_id
        ):
            module = self.build_module(params)
        else:
            h = self._structure_hash(params)
            module = context.node_state_store.get_or_create(
                context.graph_id,
                context.current_node_id,
                h,
                lambda: self.build_module(params),
                on_evict=lambda keys, reason: _report_eviction(
                    context, keys, reason),
            )

        if context is not None and getattr(context, "device", None):
            from .device_utils import to_device

            module = to_device(module, context.device)
        return module


#: How many evicted node ids a single warning names before it stops
#: listing them. A budget that evicts twenty nodes at once has a message
#: nobody reads; the count still tells the whole story.
_MAX_NAMED_EVICTIONS = 3


def _report_eviction(
    context: "ExecutionContext",
    keys: list[tuple[str, str, str]],
    reason: str,
) -> None:
    """Put a weight-discarding eviction where the user will see it.

    Onto the run's durable event stream, which the Runs panel reads back.
    The canvas does not render it -- see ``WarningSignal``. The store has
    already logged it at WARNING; this is the half that survives past the
    server console.

    Never raises: an eviction notice must not be able to fail the node that
    happened to trigger it.
    """
    if not keys:
        return
    try:
        names = [key[1] for key in keys]
        shown = ", ".join(names[:_MAX_NAMED_EVICTIONS])
        if len(names) > _MAX_NAMED_EVICTIONS:
            shown += f" and {len(names) - _MAX_NAMED_EVICTIONS} more"
        limit = ("byte budget (CODEFYUI_NODE_STATE_STORE_MAX_MB)"
                 if reason == "bytes" else
                 "module count limit")
        context.log_warning(
            "node_state_evicted",
            f"The persisted weights of {len(names)} node(s) were discarded "
            f"to stay inside the node state store's {limit}: {shown}. "
            f"Those nodes will start from a fresh initialisation on the "
            f"next run.",
            # The VICTIM, not context.current_node_id (log_warning's
            # default): current_node_id names whichever node's build
            # TRIGGERED the eviction, which is not the node that lost its
            # weights -- and the store is shared across graphs, so the
            # trigger can even be a different tab's node entirely. Only
            # one id fits in this field; when several were evicted at
            # once, the first is the one this warning points at (the rest
            # are still named in the detail text above).
            node_id=keys[0][1],
        )
    except Exception:  # noqa: BLE001 - a notice must not fail the node
        logger.debug("could not report a node state eviction", exc_info=True)
