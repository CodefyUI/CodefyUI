"""Parsers for parameters whose value is a small sequence of numbers.

``ParamType`` has no tuple or list member, and adding one would mean a new
editor widget on every surface that renders a param. The cheaper contract —
and the one every deep-learning config file already uses — is a STRING the
node parses: ``"0.9, 0.999"`` for Adam's ``betas``, ``"1, 5, 2"`` for
``CrossEntropyLoss``'s per-class ``weight``.

Parsing lives here rather than in each node so that the accepted spellings
and, more importantly, the ERROR MESSAGES are identical everywhere. A typo
in a hyperparameter should say which parameter, what it got and what shape
it wanted — not surface as a ``ValueError`` from inside torch three frames
into the training loop.
"""

from __future__ import annotations

from typing import Any, Sequence

__all__ = [
    "bool_param",
    "float_param",
    "int_param",
    "is_blank",
    "parse_float_sequence",
    "parse_int_sequence",
]

#: Brackets a user may reasonably paste around a sequence, stripped before
#: splitting. ``(0.9, 0.999)`` is how torch's own docs write betas, and
#: rejecting the spelling the documentation uses would be perverse.
_BRACKETS = "()[]{}"


def is_blank(raw: Any) -> bool:
    """True for the values that mean "not set".

    ``None`` and the empty/whitespace string. Deliberately NOT ``0`` or
    ``0.0``, which are legitimate values for most numeric parameters.
    """
    if raw is None:
        return True
    return isinstance(raw, str) and not raw.strip()


def parse_float_sequence(
    raw: Any,
    *,
    name: str,
    length: int | None = None,
) -> tuple[float, ...] | None:
    """Parse *raw* into a tuple of floats, or ``None`` when unset.

    Accepts a string (``"0.9, 0.999"``, ``"(0.9 0.999)"``, ``"1;2;3"``), an
    existing list/tuple of numbers (which is what a graph JSON round-trip
    produces if a client ever sends one), or a lone number. *length*, when
    given, is enforced.

    Raises ``ValueError`` naming *name* on anything else. Booleans are
    rejected on purpose: ``True`` is an ``int`` in Python, and silently
    reading it as ``1.0`` turns a wiring mistake into a plausible-looking
    hyperparameter.
    """
    if is_blank(raw):
        return None

    values: Sequence[Any]
    if isinstance(raw, str):
        text = raw.strip().strip(_BRACKETS)
        for separator in (",", ";"):
            text = text.replace(separator, " ")
        values = text.split()
    elif isinstance(raw, (list, tuple)):
        values = raw
    elif isinstance(raw, bool):
        raise ValueError(
            f"{name} must be a number or a list of numbers, got {raw!r}")
    elif isinstance(raw, (int, float)):
        values = [raw]
    else:
        raise ValueError(
            f"{name} must be a number or a comma-separated list of numbers, "
            f"got {type(raw).__name__}")

    parsed: list[float] = []
    for item in values:
        if isinstance(item, bool):
            raise ValueError(f"{name} must contain numbers, got {item!r}")
        try:
            parsed.append(float(item))
        except (TypeError, ValueError):
            raise ValueError(
                f"{name} must be a comma-separated list of numbers; "
                f"{item!r} is not a number") from None

    if not parsed:
        return None
    if length is not None and len(parsed) != length:
        raise ValueError(
            f"{name} must have exactly {length} values, got {len(parsed)} "
            f"({', '.join(str(v) for v in parsed)})")
    return tuple(parsed)


# -- single-value params ----------------------------------------------------
#
# A graph's params are JSON: the canvas writes an INT as a number, but a CLI
# run, a preset, a hand-edited graph or an exported script can carry "3",
# 3.0 or "false". Read straight into torch those surface as messages from
# deep inside it (``empty(): argument 'size' failed to unpack``), and
# ``bool("false")`` is True. These helpers accept the honest spellings and
# raise a message naming the node and the param for everything else.

_TRUE_WORDS = frozenset({"true", "1", "yes", "on"})
_FALSE_WORDS = frozenset({"false", "0", "no", "off"})


def _as_whole_number(raw: Any) -> int | None:
    """*raw* as an int when it is one (``3``, ``3.0``, ``"3"``), else None.

    Booleans are refused: ``True`` is an ``int`` in Python, and reading it
    as 1 hides a wiring mistake.
    """
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, (float, str)):
        try:
            as_float = float(raw)
        except ValueError:
            return None
        if as_float.is_integer():
            return int(as_float)
    return None


def _check_range(
    value: float,
    *,
    name: str,
    node: str,
    min_value: float | None,
    max_value: float | None,
) -> None:
    if min_value is not None and value < min_value:
        raise ValueError(
            f"{node}: {name} must be at least {min_value}, got {value}.")
    if max_value is not None and value > max_value:
        raise ValueError(
            f"{node}: {name} must be at most {max_value}, got {value}.")


def int_param(
    params: dict[str, Any],
    name: str,
    *,
    node: str,
    default: int,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    """*name* from *params* as an int; *default* when unset or blank.

    Accepts an int, a whole float (``2.0``) and a string of either
    (``"3"``, ``" 3.0 "``). A fractional value, a boolean, or a value
    outside ``[min_value, max_value]`` raises a message naming *node*.
    """
    raw = params.get(name)
    if is_blank(raw):
        return default
    value = _as_whole_number(raw)
    if value is None:
        raise ValueError(
            f"{node}: {name} must be a whole number, got {raw!r}.")
    _check_range(value, name=name, node=node,
                 min_value=min_value, max_value=max_value)
    return value


def float_param(
    params: dict[str, Any],
    name: str,
    *,
    node: str,
    default: float,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    """*name* from *params* as a float; *default* when unset or blank.

    Accepts a number or a numeric string; a boolean is an error.
    """
    raw = params.get(name)
    if is_blank(raw):
        return float(default)
    value: float | None = None
    if not isinstance(raw, bool) and isinstance(raw, (int, float, str)):
        try:
            value = float(raw)
        except ValueError:
            value = None
    if value is None:
        raise ValueError(f"{node}: {name} must be a number, got {raw!r}.")
    _check_range(value, name=name, node=node,
                 min_value=min_value, max_value=max_value)
    return value


def bool_param(
    params: dict[str, Any],
    name: str,
    *,
    node: str,
    default: bool,
) -> bool:
    """*name* from *params* as a bool; *default* when unset or blank.

    Accepts ``True``/``False``, ``1``/``0`` and the strings ``"true"``,
    ``"false"``, ``"1"``, ``"0"``, ``"yes"``, ``"no"``, ``"on"``, ``"off"``
    in any case. A string is never passed to ``bool()``, which reads
    ``"false"`` as True.
    """
    raw = params.get(name)
    if is_blank(raw):
        return bool(default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)) and raw in (0, 1):
        return bool(raw)
    if isinstance(raw, str):
        word = raw.strip().lower()
        if word in _TRUE_WORDS:
            return True
        if word in _FALSE_WORDS:
            return False
    raise ValueError(f"{node}: {name} must be true or false, got {raw!r}.")


def parse_int_sequence(
    raw: Any,
    *,
    name: str,
    node: str,
) -> tuple[int, ...] | None:
    """Parse *raw* into a tuple of ints, or ``None`` when unset or blank.

    Accepts a lone int (``-1``), a list/tuple of ints, or a string in any
    of the spellings ``"2,3"``, ``"(2, 3)"``, ``"[2 3]"``, ``"2 3"``. A
    whole float counts as an int; anything else raises a message naming
    *node* and *name*.
    """
    if is_blank(raw):
        return None
    items: Sequence[Any]
    if isinstance(raw, str):
        text = raw.strip().strip(_BRACKETS)
        for separator in (",", ";"):
            text = text.replace(separator, " ")
        items = text.split()
    elif isinstance(raw, (list, tuple)):
        items = raw
    else:
        items = [raw]

    parsed: list[int] = []
    for item in items:
        value = _as_whole_number(item)
        if value is None:
            raise ValueError(
                f"{node}: {name} must be whole numbers separated by commas "
                f"(e.g. 2,3); {item!r} is not one.")
        parsed.append(value)
    return tuple(parsed) if parsed else None
