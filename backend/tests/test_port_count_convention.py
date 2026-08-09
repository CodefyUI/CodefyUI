"""One convention for every param that decides how many ports a node has.

Three nodes vary their port count from a param -- ``ComposeTransform``
(``steps``), ``PythonScript`` (``input_ports`` / ``output_ports``) and
``Split`` (``chunks``) -- and each used to spell its own ``int(raw)``. The
canvas reads all three with ``parseInt(String(raw), 10)``
(``frontend/src/utils/index.ts``'s ``clampCount``), which takes a NUMERIC
PREFIX where ``int()`` demands the whole string. The two therefore disagreed
on anything with a tail: the canvas drew ports the engine then rejected as
invalid input ports (``graph_engine`` port validation).

Only reachable through hand-edited or externally-generated graph JSON -- the
INT widget cannot produce a float-string -- which is why this was a nit
(#197) rather than a bug.

The expected column below is ``clampCount``'s answer, computed by hand from
its source, not from the backend. That is the whole point of the file: it is
the frontend's behaviour written down where a Python test can check it.
"""

from __future__ import annotations

import pytest

from app.core.node_base import resolve_count_param
from app.nodes.data.transforms.compose_transform_node import (
    DEFAULT_STEPS,
    MAX_STEPS,
    MIN_STEPS,
    _step_count,
)
from app.nodes.tensor_ops.split_node import MAX_CHUNKS, _resolve_chunks
from app.nodes.utility.python_script_node import MAX_PORTS, resolve_port_count

# raw value -> parseInt(String(raw), 10), or None where parseInt gives NaN
# and clampCount falls back. Verified against the JS semantics, not derived
# from the Python implementation.
PARSE_INT = [
    # the cases that used to diverge
    ("5.7", 5),        # int("5.7") raised -> fell back to the default
    ("6 chains", 6),   # ditto
    ("3px", 3),
    ("1e3", 1),        # int(float(raw)) would say 1000 -- the wrong fix
    ("0x5", 0),        # parseInt base 10 stops at the x
    # the cases that already agreed, kept so the fix cannot break them
    ("5", 5),
    (5, 5),
    (5.7, 5),
    (-2.7, -3),        # Math.floor, not truncation
    ("  6  ", 6),
    ("+3", 3),
    ("-4", -4),
    ("99", 99),
    ("0", 0),
    ("", None),
    ("abc", None),
    (".5", None),
    ("Infinity", None),
    (None, None),
    (float("nan"), None),
    (float("inf"), None),
    (True, None),      # String(true) is "true", which parseInt rejects
]


def _clamp_count(parsed: int | None, fallback: int, maximum: int) -> int:
    """``clampCount`` itself, transcribed. The oracle for the table above."""
    return max(1, min(maximum, fallback if parsed is None else parsed))


@pytest.mark.parametrize("raw,parsed", PARSE_INT)
def test_python_script_agrees_with_the_canvas(raw, parsed):
    assert resolve_port_count({"input_ports": raw}, "input_ports") == (
        _clamp_count(parsed, 1, MAX_PORTS))


@pytest.mark.parametrize("raw,parsed", PARSE_INT)
def test_split_agrees_with_the_canvas(raw, parsed):
    assert _resolve_chunks({"chunks": raw}) == _clamp_count(parsed, 2, MAX_CHUNKS)


@pytest.mark.parametrize("raw,parsed", PARSE_INT)
def test_compose_transform_agrees_with_the_canvas(raw, parsed):
    # `resolveDynamicInputs` wraps clampCount in a second
    # `Math.max(COMPOSE_MIN_STEPS, ...)`, because composing one chain is what
    # a plain edge already does.
    expected = max(MIN_STEPS,
                   _clamp_count(parsed, DEFAULT_STEPS, MAX_STEPS))
    assert _step_count({"steps": raw}) == expected


def test_the_float_string_that_started_this():
    """`{"steps": "5.7"}` drew five handles on the canvas and validated two,
    so an edge into step_3 was refused as an invalid input port."""
    assert _step_count({"steps": "5.7"}) == 5


def test_a_missing_param_takes_the_default():
    assert _step_count({}) == DEFAULT_STEPS
    assert _step_count(None) == DEFAULT_STEPS
    assert resolve_port_count(None, "input_ports") == 1
    assert _resolve_chunks(None) == 2


def test_out_of_range_values_clamp_rather_than_raise():
    assert _step_count({"steps": 99}) == MAX_STEPS
    assert _step_count({"steps": 0}) == MIN_STEPS
    assert _step_count({"steps": -5}) == MIN_STEPS
    assert resolve_port_count({"input_ports": 99}, "input_ports") == MAX_PORTS
    assert resolve_port_count({"input_ports": 0}, "input_ports") == 1
    assert _resolve_chunks({"chunks": 999}) == MAX_CHUNKS
    assert _resolve_chunks({"chunks": -1}) == 1


def test_unicode_digits_are_not_counted():
    """Python's ``\\d`` matches Devanagari; JS ``parseInt`` does not. Written
    as escapes so a cp950/cp1252 console cannot mangle the test id."""
    assert resolve_count_param({"n": "\u0967\u0968"}, "n",
                               default=4, minimum=1, maximum=8) == 4


def test_the_helper_clamps_to_the_callers_own_floor():
    """ComposeTransform's floor is 2, the other two nodes' is 1."""
    assert resolve_count_param({"n": 1}, "n",
                               default=2, minimum=2, maximum=8) == 2
    assert resolve_count_param({"n": 1}, "n",
                               default=1, minimum=1, maximum=8) == 1
