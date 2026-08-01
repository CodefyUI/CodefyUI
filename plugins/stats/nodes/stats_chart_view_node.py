"""StatsChartViewNode — render a chart payload, or a plain table, as a chart.

Declares ``media=MEDIA_CHART`` on its output, so the spec rides the #117
``output_kind`` channel to the browser and is drawn by the app's own SVG
plotting components — crisp, themed and hoverable, rather than a
server-rendered PNG.

It takes either input:

* a ``chart`` spec from ``Stats-Histogram`` / ``Stats-Correlation`` /
  ``Stats-ConfusionMatrix``, to retitle or redraw as a different kind; or
* a ``table`` (plus the usual ``columns`` / ``row_labels``), so the required
  "CSV -> GroupBy -> Chart" shape needs no chart-payload adapter node in
  between. Anything producing the pack's table contract can be charted.

``kind`` conversions are best-effort and never fatal: a heatmap asked to
become a bar chart stays a heatmap and says so in the chart's note, because
there is no honest way to flatten a matrix into bars and a failed run would
lose the picture entirely.
"""

from __future__ import annotations

from typing import Any

from app.core.node_base import (
    MEDIA_CHART,
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)

from ._stats_core import (
    CHART_KINDS,
    add_note,
    as_matrix,
    bar_chart,
    heatmap_chart,
    line_chart,
    parse_names,
    resolve_columns,
    scatter_chart,
)

_KINDS = ("auto",) + CHART_KINDS


def _points_of(spec: dict[str, Any]) -> list[tuple[float, float, str]]:
    """Flatten whatever a spec carries into (x, y, label) triples."""
    if "bars" in spec:
        return [
            (float(i), float(bar.get("value", 0.0)), str(bar.get("label", i)))
            for i, bar in enumerate(spec["bars"])
        ]
    if "series" in spec:
        out: list[tuple[float, float, str]] = []
        for series in spec["series"]:
            name = str(series.get("name", ""))
            for x, y in series.get("points", []):
                out.append((float(x), float(y), name))
        return out
    if "points" in spec:
        return [
            (float(p.get("x", 0.0)), float(p.get("y", 0.0)), str(p.get("label", "")))
            for p in spec["points"]
        ]
    return []


def convert_chart(spec: dict[str, Any], kind: str) -> dict[str, Any]:
    """Redraw ``spec`` as ``kind``, or hand it back untouched with a note."""
    current = str(spec.get("kind", ""))
    if kind == current:
        return spec

    title = str(spec.get("title", ""))
    x_label = str(spec.get("x_label", ""))
    y_label = str(spec.get("y_label", ""))

    # A matrix has no faithful 1D reading, and nothing 1D can be made into a
    # matrix, so those two directions are the ones that refuse.
    if current == "heatmap" or kind == "heatmap":
        out = dict(spec)
        add_note(
            out, f"cannot redraw a {current} chart as {kind}; showing it as {current}"
        )
        return out

    triples = _points_of(spec)
    if kind == "bar":
        converted = bar_chart(
            [label or f"{x:g}" for x, _y, label in triples],
            [y for _x, y, _label in triples],
            title=title, x_label=x_label, y_label=y_label,
        )
    elif kind == "line":
        converted = line_chart(
            [("", [x for x, _y, _l in triples], [y for _x, y, _l in triples])],
            title=title, x_label=x_label, y_label=y_label,
        )
    else:
        converted = scatter_chart(
            [x for x, _y, _l in triples],
            [y for _x, y, _l in triples],
            labels=[label for _x, _y, label in triples],
            title=title, x_label=x_label, y_label=y_label,
        )
    if spec.get("note"):
        # The source's notice comes FIRST, then whatever the rebuild added.
        carried = converted.pop("note", "")
        add_note(converted, str(spec["note"]))
        if carried:
            add_note(converted, carried)
    return converted


class StatsChartViewNode(BaseNode):
    NODE_NAME = "Stats-ChartView"
    CATEGORY = "Data"
    DESCRIPTION = (
        "Draw a chart in the Results panel: bar, line, scatter or heatmap. "
        "Takes a chart payload from another Stats node, or any table plus its "
        "column names and row labels."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="chart",
                data_type=DataType.ANY,
                description="Chart payload from Stats-Histogram / Correlation / ConfusionMatrix.",
                optional=True,
            ),
            PortDefinition(
                name="table",
                data_type=DataType.TENSOR,
                description="A table to chart, when no `chart` payload is connected.",
                optional=True,
            ),
            PortDefinition(
                name="columns",
                data_type=DataType.LIST,
                description="Optional column names for `table`.",
                optional=True,
            ),
            PortDefinition(
                name="row_labels",
                data_type=DataType.LIST,
                description="Optional row labels for `table` — the category axis of a bar chart.",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="chart",
                data_type=DataType.ANY,
                description="The rendered chart spec (also drawn in the Results panel).",
                media=MEDIA_CHART,
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="kind",
                param_type=ParamType.SELECT,
                default="auto",
                options=list(_KINDS),
                description=(
                    "`auto` keeps a payload's own kind, and picks bar or line for a "
                    "table depending on whether it has row labels."
                ),
            ),
            ParamDefinition(
                name="title",
                param_type=ParamType.STRING,
                default="",
                description="Chart title. Empty keeps the incoming payload's title.",
            ),
            ParamDefinition(
                name="columns_filter",
                param_type=ParamType.STRING,
                default="",
                description=(
                    "Which table columns to chart, by name or index, comma-separated. "
                    "Empty charts them all. A scatter uses the first two."
                ),
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
        kind = str(params.get("kind", "auto") or "auto")
        if kind not in _KINDS:
            raise ValueError(
                f"{self.NODE_NAME}: unknown kind {kind!r}; expected one of {list(_KINDS)}."
            )
        title = str(params.get("title", "") or "").strip()

        payload = inputs.get("chart")
        if isinstance(payload, dict) and payload:
            if "kind" not in payload:
                raise ValueError(
                    f"{self.NODE_NAME}: the `chart` input is a dict but carries no "
                    "`kind` — it is not a chart payload."
                )
            spec = dict(payload) if kind == "auto" else convert_chart(dict(payload), kind)
        elif inputs.get("table") is not None:
            spec = self._chart_from_table(inputs, params, kind, title)
        else:
            raise ValueError(
                f"{self.NODE_NAME}: connect either a `chart` payload or a `table`."
            )

        if title:
            spec["title"] = title
        return {"chart": spec}

    def _chart_from_table(
        self, inputs: dict[str, Any], params: dict[str, Any], kind: str, title: str
    ) -> dict[str, Any]:
        table = as_matrix(inputs.get("table"), node=self.NODE_NAME)
        names = resolve_columns(inputs.get("columns"), table.shape[1])
        raw_rows = inputs.get("row_labels")
        row_labels = (
            [str(r) for r in raw_rows] if isinstance(raw_rows, (list, tuple)) else []
        )

        selection = [
            self._resolve_column(token, names)
            for token in parse_names(params.get("columns_filter"))
        ] or list(range(len(names)))

        if kind == "auto":
            # Row labels mean the rows are categories, and categories are bars.
            # Without them the row index is a continuous axis, so: a line.
            kind = "bar" if row_labels else "line"

        labels = list(row_labels) or [str(i) for i in range(table.shape[0])]

        if kind == "heatmap":
            return heatmap_chart(
                table[:, selection],
                row_labels=labels,
                col_labels=[names[c] for c in selection],
                title=title,
            )
        if kind == "scatter":
            if len(selection) < 2:
                raise ValueError(
                    f"{self.NODE_NAME}: a scatter needs two columns for x and y; "
                    f"only {len(selection)} selected."
                )
            x, y = selection[0], selection[1]
            return scatter_chart(
                table[:, x], table[:, y], labels=labels,
                title=title, x_label=names[x], y_label=names[y],
            )
        if kind == "line":
            return line_chart(
                [
                    (names[c], list(range(table.shape[0])), table[:, c])
                    for c in selection
                ],
                title=title, x_label="row", y_label="value",
            )
        column = selection[0]
        spec = bar_chart(
            labels, table[:, column],
            title=title, x_label="group", y_label=names[column],
        )
        if len(selection) > 1:
            add_note(
                spec,
                f"charting {names[column]!r}; a bar chart shows one column — set "
                "columns_filter to pick another",
            )
        return spec

    def _resolve_column(self, token: str, names: list[str]) -> int:
        if token in names:
            return names.index(token)
        try:
            index = int(token)
        except ValueError:
            raise ValueError(
                f"{self.NODE_NAME}: `columns_filter` names unknown column "
                f"{token!r}; the table has {names}."
            ) from None
        if not 0 <= index < len(names):
            raise ValueError(
                f"{self.NODE_NAME}: `columns_filter` index {index} is out of range "
                f"for a table with {len(names)} column(s)."
            )
        return index
