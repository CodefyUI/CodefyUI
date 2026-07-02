"""Tests for app.core.api_contract — pure graph-as-a-function contract helpers."""

from __future__ import annotations

import pytest

from app.core.api_contract import InputCoercionError, coerce_input, json_type_name


def _tiny_png_base64() -> str:
    import base64
    import io

    from PIL import Image

    img = Image.new("RGB", (4, 2), color=(255, 0, 0))  # width=4, height=2
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ── coerce_input: strict API table (from_string=False) ──────────────────


def test_string_passes_through():
    assert coerce_input("hi", "string") == "hi"


def test_string_rejects_non_strings():
    for bad in (3, 2.5, True, None, {"a": 1}, [1]):
        with pytest.raises(InputCoercionError):
            coerce_input(bad, "string")


def test_rejection_names_expected_vs_received():
    with pytest.raises(InputCoercionError, match="expected string, got number"):
        coerce_input(3, "string")
    with pytest.raises(InputCoercionError, match="expected number, got string"):
        coerce_input("3", "number")
    with pytest.raises(InputCoercionError, match="expected boolean, got number"):
        coerce_input(1, "boolean")


def test_number_accepts_int_and_float_as_float():
    assert coerce_input(3, "number") == 3.0
    assert isinstance(coerce_input(3, "number"), float)
    assert coerce_input(2.5, "number") == 2.5


def test_number_rejects_string_bool_null():
    for bad in ("3", True, False, None, [1]):
        with pytest.raises(InputCoercionError):
            coerce_input(bad, "number")


def test_integer_accepts_int_and_integral_float():
    assert coerce_input(3, "integer") == 3
    assert coerce_input(3.0, "integer") == 3
    assert isinstance(coerce_input(3.0, "integer"), int)


def test_integer_rejects_fractional_string_bool():
    for bad in (3.5, "3", True, None):
        with pytest.raises(InputCoercionError):
            coerce_input(bad, "integer")


def test_boolean_accepts_only_bool():
    assert coerce_input(True, "boolean") is True
    assert coerce_input(False, "boolean") is False
    for bad in (0, 1, "true", None):
        with pytest.raises(InputCoercionError):
            coerce_input(bad, "boolean")


def test_json_accepts_any_json_value():
    for val in ({"a": 1}, [1, 2], "s", 3, 2.5, True, None):
        assert coerce_input(val, "json") == val


def test_unknown_declared_type_raises():
    with pytest.raises(InputCoercionError, match="unknown input type"):
        coerce_input("x", "tensor")


# ── coerce_input: from_string=True (canvas `default` path) ──────────────


def test_from_string_parses_number_integer_boolean_json():
    assert coerce_input("2.5", "number", from_string=True) == 2.5
    assert coerce_input("3", "integer", from_string=True) == 3
    assert coerce_input("3.0", "integer", from_string=True) == 3
    assert coerce_input("true", "boolean", from_string=True) is True
    assert coerce_input("False", "boolean", from_string=True) is False
    assert coerce_input('{"k": [1, 2]}', "json", from_string=True) == {"k": [1, 2]}


def test_from_string_string_type_stays_verbatim():
    assert coerce_input("3.5", "string", from_string=True) == "3.5"


def test_from_string_bad_parses_raise():
    with pytest.raises(InputCoercionError):
        coerce_input("abc", "number", from_string=True)
    with pytest.raises(InputCoercionError):
        coerce_input("3.5", "integer", from_string=True)
    with pytest.raises(InputCoercionError):
        coerce_input("yes", "boolean", from_string=True)
    with pytest.raises(InputCoercionError):
        coerce_input("{not json", "json", from_string=True)


def test_from_string_false_never_parses_strings():
    with pytest.raises(InputCoercionError):
        coerce_input("2.5", "number")


def test_from_string_non_string_value_uses_strict_table():
    # A hand-edited graph JSON may hold a non-string default; strict rules apply.
    assert coerce_input(5, "number", from_string=True) == 5.0
    with pytest.raises(InputCoercionError):
        coerce_input(True, "number", from_string=True)


# ── coerce_input: image ─────────────────────────────────────────────────


def test_image_decodes_to_chw_float_tensor():
    import torch

    tensor = coerce_input(_tiny_png_base64(), "image")
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 2, 4)  # (C, H, W)
    assert tensor.dtype == torch.float32
    assert float(tensor.min()) >= 0.0 and float(tensor.max()) <= 1.0


def test_image_accepts_data_uri_prefix():
    import torch

    tensor = coerce_input("data:image/png;base64," + _tiny_png_base64(), "image")
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 2, 4)


def test_image_rejects_non_string_and_garbage():
    with pytest.raises(InputCoercionError, match="expected base64 image string"):
        coerce_input(123, "image")
    with pytest.raises(InputCoercionError, match="not valid base64"):
        coerce_input("!!!not-base64!!!", "image")
    with pytest.raises(InputCoercionError, match="does not decode to an image"):
        coerce_input("aGVsbG8gd29ybGQ=", "image")  # base64("hello world")


# ── json_type_name ───────────────────────────────────────────────────────


def test_json_type_name_labels():
    assert json_type_name(None) == "null"
    assert json_type_name(True) == "boolean"
    assert json_type_name(3) == "number"
    assert json_type_name(2.5) == "number"
    assert json_type_name("s") == "string"
    assert json_type_name([1]) == "array"
    assert json_type_name({}) == "object"
