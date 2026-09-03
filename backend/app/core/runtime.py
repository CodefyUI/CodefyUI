"""Headless CodefyUI runtime bootstrap.

The server, CLI tooling, and exported Python runners all need the same
built-in, custom, plugin, and preset discovery rules. Keep the exported
runner on the existing central re-discovery path -- one call, whose seven
arguments live in :func:`~app.core.plugins.reload.rediscover_now` -- rather
than growing a second partial registry bootstrap.
"""

from __future__ import annotations

from .plugins.reload import rediscover_now


def initialize_runtime() -> dict[str, int]:
    """Reset and discover every executable node and preset source."""

    return rediscover_now()
