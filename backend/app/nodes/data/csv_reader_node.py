"""CSVReaderNode — load tabular data from a CSV file.

Entry point for the C1-2 (tabular data) lesson in the textbook. Reads a
CSV via pandas, separates numeric feature columns from a string label
column, and emits both as tensor + label-list.

Three params shape the output:

* ``target_column`` — when set, its values become the ``labels`` LIST
  output, stringified for downstream classifier nodes that expect string
  class labels, or as floats when ``target_type`` is ``number`` (#712).
  Other columns become the tensor.
* ``include_columns`` — comma-separated whitelist of feature columns.
  Empty means "all numeric columns except the target".
* ``skip_header`` — almost always True; turn off only for headerless
  CSVs (then columns are auto-named 0, 1, 2, ...).

A row with no target value is refused by default, naming the rows, because
pandas reads the gap as NaN and ``str(nan)`` would become a ``'nan'`` class.
``drop_missing_target`` drops those rows instead and logs which ones.
Integral labels render as ints (``'1'``, not ``'1.0'``) even when pandas
read the column as float. NaN in a feature column is kept and logged as a
warning naming the column.

Why this exists when ``HuggingFaceDataset`` already loads tabular data:
HuggingFace requires an internet round-trip + dataset publishing; for
classroom use, students need to point at a local CSV they wrote
themselves. The node is intentionally minimal — students can pre-clean
their CSV in pandas before pointing this node at it.
"""

from __future__ import annotations

import logging
import re
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

logger = logging.getLogger(__name__)

# Row numbers quoted in a missing-target message before it says "and N more".
_MAX_NAMED_ROWS = 10

_UNNAMED = re.compile(r"Unnamed: \d+")


def _row_list(rows: list[int]) -> str:
    shown = ", ".join(str(r) for r in rows[:_MAX_NAMED_ROWS])
    if len(rows) > _MAX_NAMED_ROWS:
        shown += f" and {len(rows) - _MAX_NAMED_ROWS} more"
    return shown


def _label_strings(values: Any) -> list[str]:
    """Stringify a target column, keeping integer classes integral.

    One missing value promotes an int column to float64, and a column
    written as ``0.0, 1.0`` reads as float too; either way the classes are
    integers, so render them as ``'0'``/``'1'``.
    """
    import pandas as pd

    if pd.api.types.is_float_dtype(values) and all(float(v).is_integer() for v in values):
        return [str(int(v)) for v in values]
    return [str(v) for v in values.tolist()]


def _target_numbers(values: Any, column: str) -> list[float]:
    """A regression target as floats, refusing any cell that is not a number.

    Missing values never reach here: they stop the run or are dropped
    earlier, under ``drop_missing_target``.
    """
    import pandas as pd

    numbers = pd.to_numeric(values, errors="coerce")
    bad = numbers.isna()
    if bad.any():
        rows = [int(i) + 1 for i in values.index[bad]]
        sample = ", ".join(repr(str(v)) for v in values[bad].tolist()[:3])
        raise ValueError(
            f"CSVReader: target_type='number' but target_column={column!r} "
            f"is not a number in data rows {_row_list(rows)} ({sample}). "
            f"Use target_type='class' for class labels."
        )
    return [float(v) for v in numbers.tolist()]


def _is_unnamed(column: Any) -> bool:
    """True for the name pandas gives a column whose header cell is blank.

    ``df.to_csv(path)`` writes the row index as a first column with an empty
    header, and a trailing comma on every line adds an empty last column;
    pandas names both ``Unnamed: k``.
    """
    return bool(_UNNAMED.fullmatch(str(column)))


def _first_non_number(values: Any) -> tuple[int, str] | None:
    """(1-based data row, cell text) of the first cell that is not a number."""
    import pandas as pd

    numbers = pd.to_numeric(values, errors="coerce")
    bad = numbers.isna() & values.notna()
    if not bad.any():
        return None
    first = values.index[bad][0]
    return int(first) + 1, str(values.loc[first])


def _warn(context: Any, detail: str) -> None:
    logger.warning("%s", detail)
    log_warning = getattr(context, "log_warning", None)
    if callable(log_warning):
        log_warning("csv_reader", detail)


class CSVReaderNode(BaseNode):
    NODE_NAME = "CSVReader"
    CATEGORY = "Data"
    DESCRIPTION = "Load a CSV into features, labels and column names"
    DETAILS = (
        "Numeric columns become the [N, F] float32 tensor; `include_columns` "
        "narrows that set, and without it non-numeric columns are dropped. The "
        "column named in `target_column` becomes the string label list, or a "
        "list of floats when `target_type` is `number`; a row "
        "with no target value stops the run, naming the data rows, unless "
        "`drop_missing_target` is on, which drops them. Integral labels read as "
        "'1', not '1.0'. A `number` target that is not a number stops the run, "
        "naming the data rows. NaN in a feature column is kept and logged as a "
        "warning. Without `include_columns`, a column with a blank header (a "
        "saved row index, or the empty column a trailing comma makes) is left "
        "out, and so is a column with any text cell, such as '?' or '1,000'; "
        "both are named in a warning. Turn "
        "`skip_header` off for a headerless file, whose columns are then named 0, "
        "1, 2 and so on."
    )

    # #144: cacheable again -- cache_fingerprint() below folds the resolved
    # file's (size, mtime, and for small files a content hash) into the
    # cache key, so an edited CSV busts the cache instead of the key
    # surviving unchanged because only the `path` string is hashed.
    cacheable = True

    @staticmethod
    def _resolve_path(path_str: str) -> Path:
        """Same resolution ``execute()`` uses -- may raise ValueError for a
        path that escapes the project directory, exactly as execute() does.
        """
        from ...config import settings
        from ...core.data_paths import working_dir_file

        path = Path(path_str)
        # A bare filename is what the DATA_FILE upload dropdown produces —
        # resolve it against DATA_FILES_DIR so picking a file never depends on
        # the working directory. Checked before the project-mode branch, and
        # only when the value has no directory component, so existing values
        # like "data/samples/iris.csv" keep their old resolution untouched.
        # Living here rather than in execute() matters: cache_fingerprint()
        # calls this too, so an uploaded file gets fingerprinted by its real
        # contents instead of silently falling back to None.
        if not path.is_absolute() and path.parent == Path("."):
            if (local := working_dir_file(path)) is not None:
                return local
            candidate = settings.DATA_FILES_DIR / path.name
            if candidate.is_file():
                return candidate
        if settings.PROJECT_DIR is not None and not path.is_absolute():
            # The bundled sample is special-cased to the install (cwd stays
            # backend/ even in project mode) so demos keep working (spec 7.2).
            if path_str.replace("\\", "/") == "data/samples/iris.csv":
                return path.resolve()
            proj = settings.PROJECT_DIR.resolve()
            resolved = (proj / path).resolve()
            if not resolved.is_relative_to(proj):
                raise ValueError(
                    f"CSVReader: path {path_str!r} escapes the project directory"
                )
            return resolved
        return path

    @classmethod
    def cache_fingerprint(cls, params: dict[str, Any]) -> Any:
        from ...core.cache_fingerprint import path_fingerprint

        path_str = str(params.get("path", "") or "").strip()
        if not path_str:
            return None
        try:
            resolved = cls._resolve_path(path_str)
        except Exception:
            return None
        return path_fingerprint(resolved)

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
                description=(
                    "Values of `target_column`: strings, or floats when "
                    "target_type is number (empty list when no target column set)."
                ),
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
                param_type=ParamType.DATA_FILE,
                default="",
                description=(
                    "CSV file to read. Pick one you uploaded from the dropdown, "
                    "or type a path (absolute, or relative to the backend "
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
                name="target_type",
                param_type=ParamType.SELECT,
                default="class",
                options=["class", "number"],
                description=(
                    "class: the target becomes string class labels. number: "
                    "the target becomes a list of floats, for regression."
                ),
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
                name="drop_missing_target",
                param_type=ParamType.BOOL,
                default=False,
                description=(
                    "Drop rows whose target value is empty and log which ones. "
                    "Off: such a row stops the run with the row numbers."
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

        path_str = str(params.get("path", "")).strip()
        if not path_str:
            raise ValueError(
                "CSVReader has no file selected. Pick one from the `path` "
                "dropdown, or use the upload button next to it to add a CSV."
            )
        path = self._resolve_path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"CSVReader: file not found at {path}")

        skip_header = bool(params.get("skip_header", True))
        df = pd.read_csv(path) if skip_header else pd.read_csv(path, header=None)
        # When no header, name columns 0, 1, 2 ...
        if not skip_header:
            df.columns = [str(i) for i in range(len(df.columns))]
        else:
            # "a, b ,y" names its columns " b " and "y"; every name typed into
            # target_column, include_columns or ColumnSelector is stripped, so
            # strip the header too (unless that would merge two columns).
            stripped = [str(c).strip() for c in df.columns]
            if len(set(stripped)) == len(stripped):
                df.columns = stripped

        target_column = str(params.get("target_column", "")).strip()
        if target_column and target_column not in df.columns:
            raise ValueError(
                f"CSVReader: target_column={target_column!r} is not in the file. "
                f"Available columns: {list(df.columns)}"
            )

        if target_column:
            missing_mask = df[target_column].isna()
            if missing_mask.any():
                # 1-based data rows: the first row after the header is row 1.
                rows = [int(i) + 1 for i in df.index[missing_mask]]
                if not bool(params.get("drop_missing_target", False)):
                    first = "the first row after the header" if skip_header else "the first line"
                    raise ValueError(
                        f"CSVReader: target_column={target_column!r} is empty in "
                        f"{len(rows)} row(s): data rows {_row_list(rows)} (row 1 is "
                        f"{first}). Fill them in, or turn on drop_missing_target "
                        f"to skip those rows."
                    )
                df = df.loc[~missing_mask]
                _warn(
                    context,
                    f"CSVReader: dropped {len(rows)} row(s) with no "
                    f"{target_column!r} value: data rows {_row_list(rows)}.",
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
            if target_column and target_column in requested:
                # The label would be a feature too, and any classifier on top
                # scores near 1.0 for the wrong reason.
                raise ValueError(
                    f"CSVReader: include_columns lists the target column "
                    f"{target_column!r}; the label cannot also be a feature. "
                    f"Remove it from include_columns."
                )
            feature_cols = requested
            for c in feature_cols:
                if len(df) > 0 and not pd.api.types.is_numeric_dtype(df[c]):
                    found = _first_non_number(df[c])
                    if found is not None:
                        row, cell = found
                        raise ValueError(
                            f"CSVReader: include_columns lists {c!r}, but its "
                            f"value {cell!r} in data row {row} is not a number. "
                            f"Fix or remove that cell (a missing value can be "
                            f"left empty), or drop {c!r} from include_columns."
                        )
        else:
            feature_cols = [c for c in df.columns if c != target_column]
            if skip_header:
                # A blank header cell is a saved row index or the empty column
                # a trailing comma makes, never a feature: an index column on
                # a file sorted by class predicts the class by itself.
                unnamed = [c for c in feature_cols if _is_unnamed(c)]
                if unnamed:
                    feature_cols = [c for c in feature_cols if c not in unnamed]
                    _warn(
                        context,
                        f"CSVReader: left out column(s) with a blank header: "
                        f"{', '.join(repr(c) for c in unnamed)} (a saved row "
                        f"index or a trailing comma). Name one in include_columns "
                        f"to keep it as a feature.",
                    )
            # Drop non-numeric features when no whitelist given. Students who
            # need a one-hot or text feature should run pre-processing first.
            # Empty DataFrame: dtype inference returns object for every column,
            # which would drop them all — keep them so the column metadata
            # survives a header-only CSV.
            if len(df) > 0:
                dropped = [
                    c for c in feature_cols
                    if not pd.api.types.is_numeric_dtype(df[c])
                ]
                if dropped:
                    feature_cols = [c for c in feature_cols if c not in dropped]
                    described = []
                    for c in dropped:
                        found = _first_non_number(df[c])
                        if found is None:
                            described.append(repr(c))
                        else:
                            described.append(f"{c!r} (data row {found[0]}: {found[1]!r})")
                    _warn(
                        context,
                        f"CSVReader: left out non-numeric column(s): "
                        f"{', '.join(described)}. A number column with one text "
                        f"cell, such as '?' or '1,000', is read as text; fix "
                        f"that cell to keep the column.",
                    )

        if feature_cols:
            nan_counts = df[feature_cols].isna().sum()
            nan_cols = [f"{c} ({int(n)})" for c, n in nan_counts.items() if n > 0]
            if nan_cols:
                _warn(
                    context,
                    f"CSVReader: feature columns with missing values (count), "
                    f"kept as NaN: {', '.join(nan_cols)}. Normalize and most "
                    f"models turn a NaN column into all NaN.",
                )
            features = df[feature_cols].to_numpy(dtype="float32")
            tensor = torch.from_numpy(features)
        else:
            tensor = torch.zeros((len(df), 0), dtype=torch.float32)

        labels: list[Any] = []
        if target_column:
            if str(params.get("target_type", "class")) == "number":
                labels = _target_numbers(df[target_column], target_column)
            else:
                labels = _label_strings(df[target_column])

        return {
            "tensor": tensor,
            "labels": labels,
            "columns": list(feature_cols),
        }
