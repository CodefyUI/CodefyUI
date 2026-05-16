"""Tests for SwitchNode."""

from __future__ import annotations

import torch

from app.nodes.dataflow.switch_node import SwitchNode


def _run(**inputs):
    return SwitchNode().execute(inputs, {})


def test_node_metadata():
    assert SwitchNode.NODE_NAME == "Switch"
    assert SwitchNode.CATEGORY == "Data Flow"


def test_selector_zero_picks_input_0():
    res = _run(selector=0, input_0="a", input_1="b")
    assert res["output"] == "a"


def test_selector_one_picks_input_1():
    res = _run(selector=1, input_0="a", input_1="b")
    assert res["output"] == "b"


def test_selector_two_picks_input_2():
    res = _run(selector=2, input_0="a", input_1="b", input_2="c")
    assert res["output"] == "c"


def test_selector_index_out_of_range_falls_back_to_zero():
    res = _run(selector=5, input_0="default")
    assert res["output"] == "default"


def test_selector_tensor_scalar_extracted():
    res = _run(selector=torch.tensor(1), input_0="a", input_1="b")
    assert res["output"] == "b"


def test_selector_float_truncated_to_int():
    res = _run(selector=1.7, input_0="a", input_1="b", input_2="c")
    assert res["output"] == "b"


def test_unconnected_input_falls_back_to_zero():
    res = _run(selector=2, input_0="default", input_1="other")
    # input_2 not provided -> falls back to input_0
    assert res["output"] == "default"
