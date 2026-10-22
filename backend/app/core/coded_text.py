"""Immutable parameters for coded messages (#841).

A queue reason (``scheduling.QueueReason``) and a worker message
(``workers.WorkerReason``) carry a code, its parameters and the English
text. Several are module constants shared by every run, so their
parameters must not be mutable: :func:`freeze` turns maps into read-only
``MappingProxyType`` views and lists into tuples, all the way down, and
:func:`thaw` gives back plain ``dict`` / ``list`` for JSON.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


def freeze(value: Any) -> Any:
    """*value* with every map read-only and every list a tuple."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze(item)
                                 for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    return value


def thaw(value: Any) -> Any:
    """*value* as plain ``dict`` and ``list``, a new copy, for JSON."""
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw(item) for item in value]
    return value
