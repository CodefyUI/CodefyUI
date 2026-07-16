"""Headless CodefyUI runtime bootstrap.

The server, CLI tooling, and exported Python runners all need the same built-in,
custom, plugin, and preset discovery rules.  Keep the exported runner on the
existing central ``rediscover_all`` path rather than growing a second partial
registry bootstrap.
"""

from __future__ import annotations

from ..config import settings
from .node_registry import registry
from .plugin_loader import plugins_builtin_root, plugins_user_root, rediscover_all
from .preset_registry import preset_registry


def initialize_runtime() -> dict[str, int]:
    """Reset and discover every executable node and preset source."""

    return rediscover_all(
        registry,
        preset_registry,
        nodes_dir=settings.NODES_DIR,
        custom_nodes_dir=settings.CUSTOM_NODES_DIR,
        presets_dir=settings.PRESETS_DIR,
        builtin_root=plugins_builtin_root(),
        user_root=plugins_user_root(),
    )
