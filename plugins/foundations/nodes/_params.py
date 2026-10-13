"""Reading BOOL params the way the canvas means them.

``bool("false")`` is True, so a BOOL param that arrives as the string
``"false"`` (hand-edited graph JSON, an API call, a sweep axis) would switch
the option on. :func:`param_bool` reads the usual spellings and refuses
anything else with a message naming the node and the param.
"""

from __future__ import annotations

from typing import Any

_TRUE = frozenset({"true", "1", "yes", "on"})
_FALSE = frozenset({"false", "0", "no", "off", ""})


def param_bool(params: dict[str, Any], name: str, default: bool, node: str) -> bool:
    raw = (params or {}).get(name, default)
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)) and raw in (0, 1):
        return bool(raw)
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
    raise ValueError(f"{node}: `{name}` must be true or false, got {raw!r}.")
