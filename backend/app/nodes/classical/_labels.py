"""Class labels as the classical nodes take them: one per row, integers integral.

Labels reach these nodes as a list of strings from CSVReader, or as a tensor
from anywhere else -- ColumnSelector picks a label column as ``[N, 1]``, and a
float column holds ``0.0``/``1.0``. Read raw, ``[N, 1].tolist()`` is a list of
one-element lists, whose string form ``'[1.0]'`` never equals a prediction,
and ``1.0`` prints as ``'1.0'`` against CSVReader's ``'1'``: either way
Accuracy reports 0 with no error. Every node that takes labels goes through
:func:`class_labels` instead.
"""

from __future__ import annotations

import re
from typing import Any

import torch

#: An integer written as a float, e.g. ``'1.0'`` from a node that stringified
#: a float label before this module existed.
_INTEGRAL_FLOAT_TEXT = re.compile(r"-?\d+\.0+")


def class_labels(value: Any, where: str) -> list[Any]:
    """One label per row, with integral floats turned into ints.

    A ``[N, 1]`` tensor is read as ``N`` labels; any other shape with more
    than one dimension is refused, naming *where* (``"KNN y_train"``) and the
    shape. Strings and non-integral numbers pass through unchanged, so
    numeric labels keep their numeric order.
    """
    if isinstance(value, torch.Tensor):
        if value.dim() == 2 and value.shape[1] == 1:
            value = value.reshape(-1)
        elif value.dim() == 0:
            raise ValueError(f"{where}: expected one label per row, got a single value.")
        elif value.dim() > 1:
            raise ValueError(
                f"{where}: expected one label per row, got a tensor of shape "
                f"{tuple(value.shape)}. Labels are a list or a 1-D tensor; pick "
                "one column (ColumnSelector) or use CSVReader's `labels` output."
            )
        items = value.tolist()
    else:
        items = list(value)
    return [_canonical(v) for v in items]


def label_strings(value: Any, where: str) -> list[str]:
    """:func:`class_labels` as strings, the form predictions are compared in.

    ``'1.0'`` reads as ``'1'`` too, so a prediction spelled as a float still
    matches CSVReader's label.
    """
    out = []
    for v in class_labels(value, where):
        text = str(v)
        if _INTEGRAL_FLOAT_TEXT.fullmatch(text):
            text = text.split(".")[0]
        out.append(text)
    return out


def class_order(names: Any) -> list[str]:
    """The distinct label strings in column order.

    Numeric order when every label parses as a number, so 12 integer classes
    run 0, 1, 2, ..., 11 (the order TensorDataset already uses); text order
    otherwise. Sorting the strings directly would put '10' before '2'.
    """
    distinct = set(names)
    try:
        return sorted(distinct, key=lambda text: (float(text), text))
    except ValueError:
        return sorted(distinct)


def encode_labels(value: Any, where: str) -> tuple[list[int], list[str]]:
    """``(codes, classes)``: each row's index into ``classes``.

    Every classifier fits on the codes and maps sklearn's ``classes_`` and
    predictions back through ``classes``, so the ``probabilities`` columns
    follow :func:`class_order` on every node. The label strings are
    ``str()`` of :func:`class_labels`, the form predictions are returned in.
    """
    names = [str(v) for v in class_labels(value, where)]
    classes = class_order(names)
    index = {name: i for i, name in enumerate(classes)}
    return [index[name] for name in names], classes


def _canonical(v: Any) -> Any:
    if isinstance(v, list) and len(v) == 1:
        v = v[0]
    # bool before the numeric cases: True must read as 1, or predictions
    # [True, False] never match labels [1, 0] once both are strings.
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):  # numpy scalar
        return _canonical(v.item())
    return v
