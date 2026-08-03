"""Tests for the shared numeric-sequence param parser (core#134).

``betas``, ``weight`` and ``pos_weight`` are STRING params because
``ParamType`` has no tuple member. That makes the parser the boundary
between a user's typing and torch's constructor, so its edge cases are worth
pinning: a wrong answer here becomes a silently mis-tuned run.
"""

from __future__ import annotations

import pytest

from app.core.param_values import is_blank, parse_float_sequence


@pytest.mark.parametrize("raw", [None, "", "   ", "\t\n"])
def test_blank_values_mean_not_set(raw):
    assert is_blank(raw) is True
    assert parse_float_sequence(raw, name="x") is None


@pytest.mark.parametrize("raw", [0, 0.0, False, "0"])
def test_zero_is_a_value_not_a_blank(raw):
    """``0`` is a legitimate weight. Treating it as "unset" would lose it."""
    assert is_blank(raw) is False


@pytest.mark.parametrize("raw,expected", [
    ("0.9, 0.999", (0.9, 0.999)),
    ("(0.9, 0.999)", (0.9, 0.999)),
    ("[0.9 0.999]", (0.9, 0.999)),
    ("0.9;0.999", (0.9, 0.999)),
    ("  0.9 ,0.999  ", (0.9, 0.999)),
    ([0.9, 0.999], (0.9, 0.999)),
    ((1, 5), (1.0, 5.0)),
    ("3", (3.0,)),
    (3, (3.0,)),
    (2.5, (2.5,)),
])
def test_accepted_spellings(raw, expected):
    assert parse_float_sequence(raw, name="x") == expected


def test_length_is_enforced_when_asked():
    assert parse_float_sequence("1, 2", name="betas", length=2) == (1.0, 2.0)
    with pytest.raises(ValueError, match="exactly 2 values"):
        parse_float_sequence("1, 2, 3", name="betas", length=2)


def test_the_error_names_the_parameter_and_the_offending_token():
    with pytest.raises(ValueError, match="weight.*'five'.*not a number"):
        parse_float_sequence("1, five", name="weight")


def test_booleans_are_rejected_rather_than_read_as_one():
    """``True`` is an ``int`` in Python. Reading it as 1.0 hides a miswiring."""
    with pytest.raises(ValueError, match="must be a number"):
        parse_float_sequence(True, name="x")
    with pytest.raises(ValueError, match="must contain numbers"):
        parse_float_sequence([1, True], name="x")


def test_unsupported_types_are_rejected_by_name():
    with pytest.raises(ValueError, match="got dict"):
        parse_float_sequence({"a": 1}, name="x")
