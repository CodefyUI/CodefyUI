"""Tests for RowSelectorNode."""

from __future__ import annotations

import pytest
import torch

from app.nodes.data.row_selector_node import RowSelectorNode

_X = torch.arange(12, dtype=torch.float32).view(6, 2)
_LABELS = ["cat", "dog", "cat", "dog", "cat", "dog"]


def _run(tensor=_X, *, labels=None, **params):
    p = {"indices": "0", "names": ""}
    p.update(params)
    inputs: dict = {"tensor": tensor}
    if labels is not None:
        inputs["labels"] = labels
    return RowSelectorNode().execute(inputs, p)["tensor"]


def test_node_metadata():
    assert RowSelectorNode.NODE_NAME == "RowSelector"
    assert RowSelectorNode.CATEGORY == "Data"


def test_select_by_indices():
    assert _run(indices="0,2").tolist() == [[0.0, 1.0], [4.0, 5.0]]


def test_out_of_range_index_is_refused():
    with pytest.raises(ValueError, match="out-of-range value 6"):
        _run(indices="6")


@pytest.mark.parametrize("bad", ["a", "1.0"])
def test_indices_that_are_not_whole_numbers_name_the_node(bad):
    with pytest.raises(ValueError, match=r"RowSelector: indices has"):
        _run(indices=bad)


def test_a_name_selects_every_row_with_that_label():
    assert _run(labels=_LABELS, names="cat").tolist() == [[0.0, 1.0], [4.0, 5.0], [8.0, 9.0]]


def test_several_names_come_out_in_the_order_given():
    out = _run(labels=_LABELS, names="dog,cat")
    assert out[:, 0].tolist() == [2.0, 6.0, 10.0, 0.0, 4.0, 8.0]


@pytest.mark.parametrize("labels", [_LABELS[:2], _LABELS + ["cat"]])
def test_labels_that_do_not_describe_the_tensor_are_refused(labels):
    with pytest.raises(ValueError, match=r"RowSelector: the `labels` input has \d+ labels but the tensor has 6 rows"):
        _run(labels=labels, names="cat")


def test_tensor_labels_are_accepted():
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    assert _run(labels=labels, names="1")[:, 0].tolist() == [2.0, 6.0, 10.0]


def test_labels_are_stripped_like_column_names():
    labels = [" cat", "dog ", "cat", "dog", "cat", "dog"]
    assert _run(labels=labels, names="cat,dog").shape == (6, 2)


def test_a_name_not_in_the_labels_is_refused():
    with pytest.raises(ValueError, match=r"names \['bird'\] not in labels"):
        _run(labels=_LABELS, names="bird")


def test_names_without_labels_is_refused():
    with pytest.raises(ValueError, match="no `labels` input"):
        _run(names="cat")
