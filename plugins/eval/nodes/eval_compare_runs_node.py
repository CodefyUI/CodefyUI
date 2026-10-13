"""EvalCompareRunsNode -- two prediction tables, item by item, aligned by id.

Rows are matched on the key column and never by position: a run that
shuffled its rows, or batched them differently, compares exactly as one that
did not. What would make the comparison meaningless is refused, naming the
first offenders, instead of being compared anyway:

* a key that appears twice in either table (which row is "item 7"?);
* key sets that differ (the runs did not score the same items);
* the same key with different input hashes (the items were not the same
  inputs -- different templates, a different file);
* the same key with different labels (different ground truth).

An item is compared only when both runs answered it (``ok`` on both sides);
the rest stay in the diff table with their statuses and empty comparison
cells, and are counted as skipped.
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

from ._eval_core import (
    STATUS_OK,
    first_duplicates,
    format_records,
    heatmap_spec,
    make_table,
    numeric_projection,
    producer_label,
    require_table,
    table_records,
)

DIFF_COLUMNS = [
    "id", "answer_a", "answer_b", "label", "agree", "a_correct", "b_correct",
    "status_a", "status_b",
]
#: The numeric columns of the diff, for the stats split table.
DIFF_NUMERIC = ["agree", "a_correct", "b_correct"]
OUTCOMES = ["both_correct", "only_a_correct", "only_b_correct", "both_wrong"]
#: Past this many distinct answers the agreement matrix stops being readable.
MAX_MATRIX_CLASSES = 20


def _ids(records: list[dict[str, Any]], key: str) -> list[str]:
    return [str(r.get(key)) for r in records]


class EvalCompareRunsNode(BaseNode):
    NODE_NAME = "Eval-CompareRuns"
    CATEGORY = "Training"
    DESCRIPTION = "Item-by-item A/B diff of two prediction tables, by id"
    DETAILS = (
        "Matches rows on `key_column`, never by position. Refuses duplicate "
        "keys, key sets that differ, the same key with different input hashes "
        "(turn `require_same_input` off to compare two prompts on purpose) and "
        "the same key with different labels, naming the first offenders. An "
        "item is compared when both runs answered it; the others stay in the "
        "diff with their statuses. Outputs the per-item diff, an outcome table "
        "(both correct, only A, only B, both wrong) for Stats-ChartView, the "
        "agreement rate, and an agreement matrix of A's answers against B's."
    )

    align_inputs = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="table_a",
                data_type=DataType.ANY,
                description="Prediction table of run A.",
            ),
            PortDefinition(
                name="table_b",
                data_type=DataType.ANY,
                description="Prediction table of run B.",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="diff",
                data_type=DataType.ANY,
                description="Per-item record table: id, answer_a, answer_b, label, agree, a_correct, b_correct, statuses.",
            ),
            PortDefinition(
                name="diff_table",
                data_type=DataType.TENSOR,
                description="[items, 3] agree / a_correct / b_correct as 1 or 0; NaN where not compared.",
            ),
            PortDefinition(
                name="diff_columns",
                data_type=DataType.LIST,
                description="Column names of `diff_table`.",
            ),
            PortDefinition(
                name="diff_ids",
                data_type=DataType.LIST,
                description="Item ids, the row labels of `diff_table`.",
            ),
            PortDefinition(
                name="summary_table",
                data_type=DataType.TENSOR,
                description="[4, 2] count and share of both correct, only A, only B, both wrong.",
            ),
            PortDefinition(
                name="summary_columns",
                data_type=DataType.LIST,
                description="Column names of `summary_table`: count, share.",
            ),
            PortDefinition(
                name="summary_rows",
                data_type=DataType.LIST,
                description="Row labels of `summary_table`.",
            ),
            PortDefinition(
                name="agreement",
                data_type=DataType.SCALAR,
                description="Share of compared items where A and B gave the same answer.",
            ),
            PortDefinition(
                name="summary",
                data_type=DataType.ANY,
                description="Agreement, outcome counts, accuracies and both runs' provenance.",
            ),
            PortDefinition(
                name="chart",
                data_type=DataType.ANY,
                description="Agreement matrix: A's answers (rows) against B's (columns).",
                media=MEDIA_CHART,
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="key_column",
                param_type=ParamType.STRING,
                default="id",
                description="Column the two tables are matched on.",
            ),
            ParamDefinition(
                name="require_same_input",
                param_type=ParamType.BOOL,
                default=True,
                description="Refuse items whose input hashes differ between the runs.",
            ),
            ParamDefinition(
                name="max_rows",
                param_type=ParamType.INT,
                default=20,
                min_value=0,
                max_value=1000,
                description="Diff rows printed in the log, disagreements first. 0 prints all.",
                advanced=True,
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
        node = self.NODE_NAME
        key = str(params.get("key_column", "id") or "id").strip()
        needed = [key, "answer", "label", "status"]
        table_a = require_table(inputs.get("table_a"), node=node, port="table_a", columns=needed)
        table_b = require_table(inputs.get("table_b"), node=node, port="table_b", columns=needed)
        rows_a = table_records(table_a)
        rows_b = table_records(table_b)
        ids_a = _ids(rows_a, key)
        ids_b = _ids(rows_b, key)

        for port, ids in (("table_a", ids_a), ("table_b", ids_b)):
            dupes = first_duplicates(ids)
            if dupes:
                raise ValueError(
                    f"{node}: {port} has duplicate {key} values {dupes}. Every item "
                    f"must appear once -- rows are matched by {key}, never by position."
                )
        set_a, set_b = set(ids_a), set(ids_b)
        only_a = [i for i in ids_a if i not in set_b]
        only_b = [i for i in ids_b if i not in set_a]
        if only_a or only_b:
            raise ValueError(
                f"{node}: the two runs did not score the same items: "
                f"{len(only_a)} {key}(s) only in table_a (first: {only_a[:5]}), "
                f"{len(only_b)} only in table_b (first: {only_b[:5]})."
            )

        by_b = dict(zip(ids_b, rows_b))
        if bool(params.get("require_same_input", True)):
            differ = [
                i for i, r in zip(ids_a, rows_a)
                if r.get("input_hash") is not None
                and by_b[i].get("input_hash") is not None
                and r.get("input_hash") != by_b[i].get("input_hash")
            ]
            if differ:
                raise ValueError(
                    f"{node}: {len(differ)} item(s) were given different inputs in the "
                    f"two runs (first: {differ[:5]}). Turn `require_same_input` off to "
                    "compare runs that used different prompts on purpose."
                )
        label_differ = [
            i for i, r in zip(ids_a, rows_a)
            if r.get("label") is not None and by_b[i].get("label") is not None
            and r.get("label") != by_b[i].get("label")
        ]
        if label_differ:
            raise ValueError(
                f"{node}: {len(label_differ)} item(s) carry different labels in the two "
                f"runs (first: {label_differ[:5]}) -- they were not scored against the "
                "same ground truth."
            )

        diff_rows: list[list[Any]] = []
        counts = dict.fromkeys(OUTCOMES, 0)
        compared = agree = unlabelled = 0
        correct_a = correct_b = 0
        pairs: list[tuple[str, str]] = []
        for item_id, ra in zip(ids_a, rows_a):
            rb = by_b[item_id]
            label = ra.get("label") if ra.get("label") is not None else rb.get("label")
            both_ok = ra.get("status") == STATUS_OK and rb.get("status") == STATUS_OK
            same = a_ok = b_ok = None
            if both_ok:
                compared += 1
                same = int(ra.get("answer") == rb.get("answer"))
                agree += same
                pairs.append((str(ra.get("answer")), str(rb.get("answer"))))
                if label is None:
                    unlabelled += 1
                else:
                    a_ok = int(ra.get("answer") == label)
                    b_ok = int(rb.get("answer") == label)
                    correct_a += a_ok
                    correct_b += b_ok
                    outcome = (
                        "both_correct" if a_ok and b_ok else
                        "only_a_correct" if a_ok else
                        "only_b_correct" if b_ok else "both_wrong"
                    )
                    counts[outcome] += 1
            diff_rows.append([
                item_id, ra.get("answer"), rb.get("answer"), label, same, a_ok, b_ok,
                ra.get("status"), rb.get("status"),
            ])

        scored = compared - unlabelled
        agreement = agree / compared if compared else 0.0
        meta_a = table_a.get("meta") or {}
        meta_b = table_b.get("meta") or {}
        summary = {
            "key_column": key,
            "items": len(ids_a),
            "compared": compared,
            "skipped": len(ids_a) - compared,
            "unlabelled": unlabelled,
            "agreement": agreement,
            "agree": agree,
            "disagree": compared - agree,
            "outcomes": dict(counts),
            "accuracy_a": correct_a / scored if scored else None,
            "accuracy_b": correct_b / scored if scored else None,
            "run_a": {"label": producer_label(meta_a), "producer": meta_a.get("producer"),
                      "dataset": meta_a.get("dataset")},
            "run_b": {"label": producer_label(meta_b), "producer": meta_b.get("producer"),
                      "dataset": meta_b.get("dataset")},
        }
        diff = make_table(DIFF_COLUMNS, diff_rows, meta={"summary": summary})

        summary_rows = [
            [counts[o], (counts[o] / scored) if scored else 0.0] for o in OUTCOMES
        ]
        chart = self._chart(pairs, summary)

        # Disagreements first, so a capped log shows the interesting rows.
        ordered = sorted(diff_rows, key=lambda r: (r[4] != 0, r[4] is None))
        text = (
            f"A = {summary['run_a']['label']}  B = {summary['run_b']['label']}\n"
            f"{compared} of {len(ids_a)} item(s) compared; agreement {agreement:.4f} "
            f"({agree} agree, {compared - agree} disagree)\n"
            + "  ".join(f"{o} {counts[o]}" for o in OUTCOMES)
            + "\n"
            + format_records(
                DIFF_COLUMNS[:7], [r[:7] for r in ordered],
                max_rows=int(params.get("max_rows", 20) or 0),
            )
        )
        return {
            "diff": diff,
            "diff_table": numeric_projection([r[4:7] for r in diff_rows], DIFF_NUMERIC),
            "diff_columns": list(DIFF_NUMERIC),
            "diff_ids": list(ids_a),
            "summary_table": numeric_projection(summary_rows, ["count", "share"]),
            "summary_columns": ["count", "share"],
            "summary_rows": list(OUTCOMES),
            "agreement": float(agreement),
            "summary": summary,
            "chart": chart,
            "__log__": text,
        }

    def _chart(self, pairs: list[tuple[str, str]], summary: dict[str, Any]) -> dict[str, Any]:
        title = f"A ({summary['run_a']['label']}) vs B ({summary['run_b']['label']})"
        classes = sorted({a for a, _ in pairs} | {b for _, b in pairs})
        if pairs and len(classes) <= MAX_MATRIX_CLASSES:
            index = {c: i for i, c in enumerate(classes)}
            matrix = [[0.0] * len(classes) for _ in classes]
            for a, b in pairs:
                matrix[index[a]][index[b]] += 1.0
            return heatmap_spec(
                matrix, classes, classes, title=title,
                x_label="answer of B", y_label="answer of A",
            )
        spec = {
            "kind": "bar",
            "title": title,
            "x_label": "outcome",
            "y_label": "items",
            "bars": [
                {"label": o, "value": float(summary["outcomes"][o])} for o in OUTCOMES
            ],
        }
        if pairs:
            spec["note"] = (
                f"{len(classes)} distinct answers -- too many for an agreement "
                "matrix, so the outcome counts are shown instead"
            )
        return spec
