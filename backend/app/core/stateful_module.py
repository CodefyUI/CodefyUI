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
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    import torch

    from .execution_context import ExecutionContext


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
        """
        if (
            context is None
            or not context.weights_persistent
            or context.node_state_store is None
            or not context.current_node_id
        ):
            return self.build_module(params)

        h = self._structure_hash(params)
        return context.node_state_store.get_or_create(
            context.graph_id,
            context.current_node_id,
            h,
            lambda: self.build_module(params),
        )
