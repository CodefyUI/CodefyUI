"""Tests for the shared numeric-sequence param parser (core#134).

``betas``, ``weight`` and ``pos_weight`` are STRING params because
``ParamType`` has no tuple member. That makes the parser the boundary
between a user's typing and torch's constructor, so its edge cases are worth
pinning: a wrong answer here becomes a silently mis-tuned run.
"""

from __future__ import annotations

import pytest

from app.core.param_values import (
    bool_param,
    float_param,
    int_param,
    is_blank,
    parse_float_sequence,
    parse_int_sequence,
)


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


# -- single-value helpers (2026-10-13 audit) ---------------------------------


@pytest.mark.parametrize("raw, expected", [(3, 3), (3.0, 3), ("3", 3), (" 3.0 ", 3), (None, 7), ("", 7)])
def test_int_param_accepts_whole_numbers_in_any_spelling(raw, expected):
    assert int_param({"k": raw}, "k", node="N", default=7) == expected


@pytest.mark.parametrize("raw", [2.5, "abc", True, [3]])
def test_int_param_names_the_node_and_param(raw):
    with pytest.raises(ValueError, match="N: k must be a whole number"):
        int_param({"k": raw}, "k", node="N", default=1)


def test_int_param_enforces_the_minimum():
    with pytest.raises(ValueError, match="N: k must be at least 1, got 0"):
        int_param({"k": 0}, "k", node="N", default=1, min_value=1)


@pytest.mark.parametrize("raw, expected", [(0.5, 0.5), (2, 2.0), ("0.25", 0.25), (None, 1.0)])
def test_float_param(raw, expected):
    assert float_param({"p": raw}, "p", node="N", default=1.0) == expected


def test_float_param_refuses_a_boolean():
    with pytest.raises(ValueError, match="N: p must be a number, got True"):
        float_param({"p": True}, "p", node="N", default=1.0)


@pytest.mark.parametrize("raw, expected", [
    (True, True), (False, False), (1, True), (0, False),
    ("true", True), ("False", False), ("1", True), ("0", False), (None, True),
])
def test_bool_param_reads_words_not_truthiness(raw, expected):
    assert bool_param({"b": raw}, "b", node="N", default=True) is expected


@pytest.mark.parametrize("raw", ["maybe", 2, [True]])
def test_bool_param_refuses_anything_else(raw):
    with pytest.raises(ValueError, match="N: b must be true or false"):
        bool_param({"b": raw}, "b", node="N", default=True)


@pytest.mark.parametrize("raw, expected", [
    (-1, (-1,)), ("2,3", (2, 3)), ("(2, 3)", (2, 3)), ("2 3", (2, 3)),
    ([2, 3], (2, 3)), ("", None), (None, None),
])
def test_parse_int_sequence_spellings(raw, expected):
    assert parse_int_sequence(raw, name="dim", node="N") == expected


def test_parse_int_sequence_names_a_bad_item():
    with pytest.raises(ValueError, match="N: dim must be whole numbers.*'x' is not one"):
        parse_int_sequence("1,x", name="dim", node="N")
