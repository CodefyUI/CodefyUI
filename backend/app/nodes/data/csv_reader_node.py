"""CSVReaderNode — load tabular data from a CSV file.

Entry point for the C1-2 (tabular data) lesson in the textbook. Reads a
CSV via pandas, separates numeric feature columns from a string label
column, and emits both as tensor + label-list.

Three params shape the output:

* ``target_column`` — when set, its values become the ``labels`` LIST
  output (always stringified for downstream classifier nodes that
  expect string class labels). Other columns become the tensor.
* ``include_columns`` — comma-separated whitelist of feature columns.
  Empty means "all numeric columns except the target".
* ``skip_header`` — almost always True; turn off only for headerless
  CSVs (then columns are auto-named 0, 1, 2, ...).

Why this exists when ``HuggingFaceDataset`` already loads tabular data:
HuggingFace requires an internet round-trip + dataset publishing; for
classroom use, students need to point at a local CSV they wrote
themselves. The node is intentionally minimal — students can pre-clean
their CSV in pandas before pointing this node at it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)


class CSVReaderNode(BaseNode):
    NODE_NAME = "CSVReader"
    CATEGORY = "Data"
    DESCRIPTION = (
        "Load a CSV into a feature tensor + label list. Numeric columns "
        "(filtered by `include_columns` if set) become a [N, F] float32 "
        "tensor; the column named in `target_column` becomes a list of "
        "string labels for downstream classifier nodes."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tensor",
                data_type=DataType.TENSOR,
                description="Float32 [N, F] tensor of feature columns.",
            ),
            PortDefinition(
                name="labels",
                data_type=DataType.LIST,
                description="String labels from `target_column` (empty list when no target column set).",
            ),
            PortDefinition(
                name="columns",
                data_type=DataType.LIST,
                description="Names of the feature columns in tensor order.",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="path",
                param_type=ParamType.STRING,
                default="data/samples/iris.csv",
                description=(
                    "Path to the CSV file (absolute or relative to the backend "
                    "working dir; in project mode, relative paths resolve "
                    "inside the project directory)."
                ),
            ),
            ParamDefinition(
                name="target_column",
                param_type=ParamType.STRING,
                default="",
                description="Optional name of the label column. Leave blank for unsupervised loading.",
            ),
            ParamDefinition(
                name="include_columns",
                param_type=ParamType.STRING,
                default="",
                description=(
                    "Optional comma-separated list of feature columns to keep. "
                    "Empty means 'all numeric columns except the target'."
                ),
            ),
            ParamDefinition(
                name="skip_header",
                param_type=ParamType.BOOL,
                default=True,
                description="True when the first row is a header. False generates 0-indexed column names.",
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
        import pandas as pd

        from ...config import settings

        path_str = str(params.get("path", "")).strip()
        if not path_str:
            raise ValueError("CSVReader requires a non-empty `path` param.")
        path = Path(path_str)
        if settings.PROJECT_DIR is not None and not path.is_absolute():
            # The bundled sample is special-cased to the install (cwd stays
            # backend/ even in project mode) so demos keep working (spec 7.2).
            if path_str.replace("\\", "/") == "data/samples/iris.csv":
                path = path.resolve()
            else:
                proj = settings.PROJECT_DIR.resolve()
                resolved = (proj / path).resolve()
                if not resolved.is_relative_to(proj):
                    raise ValueError(
                        f"CSVReader: path {path_str!r} escapes the project directory"
                    )
                path = resolved
        if not path.exists():
            raise FileNotFoundError(f"CSVReader: file not found at {path}")

        skip_header = bool(params.get("skip_header", True))
        df = pd.read_csv(path) if skip_header else pd.read_csv(path, header=None)
        # When no header, name columns 0, 1, 2 ...
        if not skip_header:
            df.columns = [str(i) for i in range(len(df.columns))]

        target_column = str(params.get("target_column", "")).strip()
        if target_column and target_column not in df.columns:
            raise ValueError(
                f"CSVReader: target_column={target_column!r} is not in the file. "
                f"Available columns: {list(df.columns)}"
            )

        include_raw = str(params.get("include_columns", "")).strip()
        if include_raw:
            requested = [c.strip() for c in include_raw.split(",") if c.strip()]
            missing = [c for c in requested if c not in df.columns]
            if missing:
                raise ValueError(
                    f"CSVReader: include_columns references unknown columns: {missing}. "
                    f"Available: {list(df.columns)}"
                )
            feature_cols = requested
        else:
            feature_cols = [c for c in df.columns if c != target_column]
            # Drop non-numeric features when no whitelist given. Students who
            # need a one-hot or text feature should run pre-processing first.
            # Empty DataFrame: dtype inference returns object for every column,
            # which would drop them all — keep them so the column metadata
            # survives a header-only CSV.
            if len(df) > 0:
                feature_cols = [
                    c for c in feature_cols
                    if pd.api.types.is_numeric_dtype(df[c])
                ]

        if feature_cols:
            features = df[feature_cols].to_numpy(dtype="float32")
            tensor = torch.from_numpy(features)
        else:
            tensor = torch.zeros((len(df), 0), dtype=torch.float32)

        labels: list[str] = []
        if target_column:
            labels = [str(v) for v in df[target_column].tolist()]

        return {
            "tensor": tensor,
            "labels": labels,
            "columns": list(feature_cols),
        }
