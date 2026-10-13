"""Layer-size params shared by the RL network constructors (DQN, PPO).

A size reaching the node from a hand-edited graph, a CLI ``--param`` or a
generated script can be a string (``"4"``) or out of range; handed straight
to ``nn.Linear`` the first is a raw torch ``TypeError`` and a zero or a
negative size is either a shape error deep in the first forward pass or a
network with no weights at all.
"""

from __future__ import annotations

from typing import Any


def positive_int(params: dict[str, Any], name: str, default: int, node: str) -> int:
    """``params[name]`` as an int >= 1, or a ValueError naming *node* and *name*.

    Missing, ``None`` and ``""`` take *default*. A whole-number string or
    float (``"4"``, ``4.0``) is accepted; anything else is refused.
    """
    raw = params.get(name, default)
    if raw is None or raw == "":
        raw = default
    value: int | None = None
    if not isinstance(raw, bool):
        try:
            value = int(raw)
        except (TypeError, ValueError):
            try:
                number = float(raw)
            except (TypeError, ValueError):
                number = None
            if number is not None and number.is_integer():
                value = int(number)
        else:
            if isinstance(raw, float) and not raw.is_integer():
                value = None
    if value is None:
        raise ValueError(
            f"{node}: {name} must be a whole number, got {raw!r}.")
    if value < 1:
        raise ValueError(
            f"{node}: {name} must be at least 1, got {value}. Set it to the "
            f"size of the layer it describes.")
    return value
