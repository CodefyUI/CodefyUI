"""Class labels for the edu classifiers, read the way the core classical nodes read them.

ColumnSelector hands a label column over as an ``[N, 1]`` tensor. Read raw,
``.tolist()`` makes each label the string ``'[0.0]'``, which never equals a
prediction, so Accuracy reads 0 with no error. The core helper
:func:`app.nodes.classical._labels.label_strings` flattens ``[N, 1]`` and
spells ``1.0`` as ``'1'``; this module adds the class order: numeric labels
sort by value (``0, 1, 2, ..., 10``), anything else as text. A text sort
would put ``10`` before ``2`` and shuffle the probability columns.
"""

from __future__ import annotations

from typing import Any

from app.nodes.classical._labels import label_strings


def _as_number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def read_labels(y_train: Any, node: str) -> tuple[list[str], list[str], list[int]]:
    """Return ``(labels, classes, y_index)`` for *y_train*.

    ``labels`` holds one string per row; ``classes`` is the sorted distinct
    labels (by value when every label is a number); ``y_index[i]`` is the
    position of ``labels[i]`` in ``classes``. Fitting an estimator on
    ``y_index`` keeps its class order equal to ``classes``.
    """
    labels = label_strings(y_train, f"{node} y_train")
    distinct = set(labels)
    numbers = {lab: _as_number(lab) for lab in distinct}
    if distinct and all(v is not None for v in numbers.values()):
        classes = sorted(distinct, key=lambda lab: (numbers[lab], lab))
    else:
        classes = sorted(distinct)
    position = {c: i for i, c in enumerate(classes)}
    return labels, classes, [position[lab] for lab in labels]
