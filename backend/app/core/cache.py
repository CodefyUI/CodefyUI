"""Node-level execution cache with LRU eviction."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import Any


class ExecutionCache:
    """Cache node outputs keyed by a hash of (type, params, upstream keys).

    Uses LRU eviction when max_entries is reached.
    """

    def __init__(self, max_entries: int = 256) -> None:
        self._store: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._max_entries = max_entries

    @staticmethod
    def compute_key(
        node_type: str,
        params: dict[str, Any],
        upstream_keys: list[str],
        device: str = "cpu",
    ) -> str:
        """Compute a deterministic SHA-256 cache key.

        ``device`` is part of the key: a device-aware source node (e.g.
        TensorCreate) produces tensors on the run's global device, so a CPU
        result must not be served for a later MPS run.
        """
        payload = json.dumps(
            {
                "type": node_type,
                "params": params,
                "upstream": sorted(upstream_keys),
                "device": device,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        return None

    def put(self, key: str, outputs: dict[str, Any]) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        else:
            if len(self._store) >= self._max_entries:
                self._store.popitem(last=False)
        self._store[key] = outputs

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
