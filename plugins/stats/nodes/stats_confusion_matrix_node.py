"""StatsConfusionMatrixNode — true-vs-predicted counts, and the heatmap of them.

Matches ``sklearn.metrics.confusion_matrix``: rows are the true class, columns
the predicted one, classes sort ascending unless named explicitly, and
``normalize`` takes the same four values with the same meanings —

===========  ================================================================
``none``     raw counts
``true``     each row divided by its own total: per-class recall on the diagonal
``pred``     each column divided by its own total: per-class precision
``all``      divided by the grand total: the joint distribution
===========  ================================================================

A row (or column) that sums to zero would make that division 0/0. sklearn
resolves it with ``np.nan_to_num``, so an absent class reads as a band of
zeros rather than a band of NaN, and this node does the same — which also
keeps the chart payload JSON-safe.

Predictions and labels may each arrive as a list of strings (``CSVReader``'s
``labels``), a 1D tensor of class indices (an ``argmax``), or a 2D tensor of
logits / one-hots, whose row argmax is taken. Everything is compared as a
string, so an integer-coded prediction still lines up with a string label of
the same class.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.core.node_base import (
    MEDIA_CHART,
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)

from ._stats_core import add_note, as_labels, heatmap_chart, parse_names, to_tensor

_NORMALIZE = ("none", "true", "pred", "all")


def _sorted_classes(labels: list[str]) -> list[str]:
    """sklearn's class order: sorted, numerically where the labels are numbers.

    Plain lexicographic sorting puts class 10 before class 2, which silently
    permutes the rows of any matrix built over integer-coded classes.
    """
    unique = sorted(set(labels))
    try:
        return sorted(unique, key=float)
    except ValueError:
        return unique


class StatsConfusionMatrixNode(BaseNode):
    NODE_NAME = "Stats-ConfusionMatrix"
    CATEGORY = "Data"
    DESCRIPTION = (
        "Confusion matrix of predictions against true labels — rows are the "
        "true class, columns the predicted one. Normalize by true row "
        "(recall), predicted column (precision) or the grand total. Renders as "
        "a heatmap in the Results panel and the node's detail view."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="predictions",
                data_type=DataType.ANY,
                description=(
                    "Predicted class per sample: a list of labels, a 1D tensor of "
                    "class indices, or a 2D [samples, classes] score matrix."
                ),
            ),
            PortDefinition(
                name="labels",
                data_type=DataType.ANY,
                description="True class per sample, in the same forms as `predictions`.",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="matrix",
                data_type=DataType.TENSOR,
                description="float32 [classes, classes]; rows true, columns predicted.",
            ),
            PortDefinition(
                name="columns",
                data_type=DataType.LIST,
                description="Predicted-class names, matching the matrix columns.",
            ),
            PortDefinition(
                name="row_labels",
                data_type=DataType.LIST,
                description="True-class names, matching the matrix rows.",
            ),
            PortDefinition(
                name="accuracy",
                data_type=DataType.SCALAR,
                description="Fraction of samples on the diagonal, from the raw counts.",
            ),
            PortDefinition(
                name="chart",
                data_type=DataType.ANY,
                description="Heatmap spec for Stats-ChartView; also rendered directly.",
                media=MEDIA_CHART,
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="normalize",
                param_type=ParamType.SELECT,
                default="none",
                options=list(_NORMALIZE),
                description=(
                    "`none` counts; `true` divides each row (recall on the "
                    "diagonal); `pred` divides each column (precision); `all` "
                    "divides by the total."
                ),
            ),
            ParamDefinition(
                name="class_names",
                param_type=ParamType.STRING,
                default="",
                description=(
                    "Comma-separated class order. Empty means every class seen in "
                    "either input, sorted. Naming classes explicitly also keeps a "
                    "class with no samples in the matrix."
                ),
            ),
            ParamDefinition(
                name="title",
                param_type=ParamType.STRING,
                default="Confusion matrix",
                description="Chart title.",
            ),
        ]

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        progress_callback: Any | None = None,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        predictions = as_labels(
            inputs.get("predictions"), node=self.NODE_NAME, port="predictions"
        )
        truth = as_labels(inputs.get("labels"), node=self.NODE_NAME, port="labels")
        if len(predictions) != len(truth):
            raise ValueError(
                f"{self.NODE_NAME}: `predictions` has {len(predictions)} entries but "
                f"`labels` has {len(truth)} — they must be per-sample and aligned."
            )

        normalize = str(params.get("normalize", "none") or "none")
        if normalize not in _NORMALIZE:
            raise ValueError(
                f"{self.NODE_NAME}: unknown normalize {normalize!r}; "
                f"expected one of {list(_NORMALIZE)}."
            )

        named = parse_names(params.get("class_names", ""))
        classes = named or _sorted_classes(truth + predictions)
        if not classes:
            raise ValueError(
                f"{self.NODE_NAME}: no classes to report — both inputs are empty "
                "and no class_names were given."
            )
        index = {name: i for i, name in enumerate(classes)}

        counts = np.zeros((len(classes), len(classes)), dtype=np.float64)
        ignored = 0
        for actual, predicted in zip(truth, predictions):
            row = index.get(actual)
            col = index.get(predicted)
            if row is None or col is None:
                # Only reachable with an explicit class_names list that omits a
                # class present in the data. Counting it into some other cell
                # would corrupt every rate derived from the matrix.
                ignored += 1
                continue
            counts[row, col] += 1.0

        total = float(counts.sum())
        accuracy = float(np.trace(counts) / total) if total else 0.0

        matrix = counts
        if normalize == "true":
            matrix = _divide(counts, counts.sum(axis=1, keepdims=True))
        elif normalize == "pred":
            matrix = _divide(counts, counts.sum(axis=0, keepdims=True))
        elif normalize == "all":
            matrix = _divide(counts, np.array([[total]]))

        chart = heatmap_chart(
            matrix,
            row_labels=classes,
            col_labels=classes,
            colormap="blues",
            title=str(params.get("title", "Confusion matrix") or "Confusion matrix"),
            x_label="predicted",
            y_label="true",
            vmin=0.0,
            vmax=None if normalize == "none" else 1.0,
        )
        if ignored:
            # add_note, never `chart["note"] = ...`: heatmap_chart may already
            # have recorded a downsampling notice, and overwriting it would
            # hide the truncation the cap exists to disclose.
            add_note(
                chart,
                f"{ignored} sample(s) ignored — their class is not in class_names",
            )

        return {
            "matrix": to_tensor(matrix),
            "columns": list(classes),
            "row_labels": list(classes),
            "accuracy": accuracy,
            "chart": chart,
        }


def _divide(counts: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Normalise, resolving 0/0 to 0 the way sklearn's nan_to_num does.

    ``np.divide`` with a ``where`` mask leaves the untouched cells at whatever
    ``out`` was initialised to, so starting from zeros gives exactly sklearn's
    answer without ever evaluating the invalid division (and so without the
    RuntimeWarning that would come with it).
    """
    out = np.zeros_like(counts)
    denom = np.broadcast_to(denominator, counts.shape)
    np.divide(counts, denom, out=out, where=denom != 0)
    return out
