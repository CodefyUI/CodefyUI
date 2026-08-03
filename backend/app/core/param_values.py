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

__all__ = ["is_blank", "parse_float_sequence"]

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
