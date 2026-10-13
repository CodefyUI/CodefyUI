import logging
from typing import TYPE_CHECKING, Any

from ...core.advisories import emit_advisory
from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition

if TYPE_CHECKING:
    # `pathlib` is imported lazily inside the methods below (startup cost);
    # this makes the "Path" annotations resolvable to a type checker without
    # paying that cost at import time.
    from pathlib import Path

logger = logging.getLogger(__name__)

#: Prefix on the canvas-log line this node's advisory carries.
_LOG_PREFIX = "[WARN] FileReader: "


class FileReaderNode(BaseNode):
    NODE_NAME = "FileReader"
    CATEGORY = "IO"
    DESCRIPTION = "Text or CSV file to a string, numeric CSV to a tensor"
    DETAILS = (
        "In csv mode the numeric rows become a 2D float tensor, and a single "
        "non-numeric cell or a row of a different length leaves that output "
        "empty, with a run-log line naming the row; in text mode it stays empty. A "
        "relative path resolves under the graphs directory, or under assets/data "
        "in a project, and files outside the project data directories are refused."
    )

    # #144: cacheable again -- cache_fingerprint() below folds the resolved
    # file's (size, mtime, and for small files a content hash) into the
    # cache key, so an edited file busts the cache instead of the key
    # surviving unchanged because only the `path` string is hashed.
    cacheable = True

    @staticmethod
    def _resolve_path(path: str) -> "Path":
        from pathlib import Path

        from ...config import settings

        p = Path(path)
        if not p.is_absolute():
            if settings.PROJECT_DIR is not None:
                # Project mode: relative paths live under assets/data (spec 7.2).
                p = settings.PROJECT_DIR / "assets" / "data" / p
            else:
                p = settings.GRAPHS_DIR / p
        return p.resolve()

    @classmethod
    def cache_fingerprint(cls, params: dict[str, Any]) -> Any:
        from ...core.cache_fingerprint import path_fingerprint

        path = str(params.get("path", "") or "")
        if not path:
            return None
        try:
            resolved = cls._resolve_path(path)
        except Exception:
            return None
        return path_fingerprint(resolved)

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="text", data_type=DataType.STRING, description="Raw file contents as string"),
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Numeric data as tensor (for CSV files)"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="path",
                param_type=ParamType.STRING,
                default="",
                description="Path to the file",
            ),
            ParamDefinition(
                name="mode",
                param_type=ParamType.SELECT,
                default="text",
                description="How to read the file",
                options=["text", "csv"],
            ),
            ParamDefinition(
                name="encoding",
                param_type=ParamType.STRING,
                default="utf-8",
                description="Text encoding",
            ),
            ParamDefinition(
                name="csv_header",
                param_type=ParamType.BOOL,
                default=True,
                description="Whether the CSV has a header row (skipped when loading to tensor)",
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any], *, context: Any = None) -> dict[str, Any]:
        import torch

        path = params.get("path", "")
        mode = params.get("mode", "text")
        encoding = params.get("encoding", "utf-8")
        csv_header = params.get("csv_header", True)

        from ...config import settings

        if not path:
            raise ValueError("File path is required")

        p = self._resolve_path(path)

        # Restrict file reading to project data directories
        allowed_roots = [
            settings.GRAPHS_DIR.resolve(),
            settings.MODELS_DIR.resolve(),
            settings.EXAMPLES_DIR.resolve(),
        ]
        if settings.PROJECT_DIR is not None:
            # The whole assets/ tree is readable in project mode.
            allowed_roots.append((settings.PROJECT_DIR / "assets").resolve())
        if not any(p.is_relative_to(root) for root in allowed_roots):
            raise ValueError(
                "File path must be within the project data directories "
                "(graphs, models, or examples)"
            )

        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")

        text = p.read_text(encoding=encoding)

        tensor = torch.tensor([])
        if mode == "csv":
            import csv
            import io

            reader = csv.reader(io.StringIO(text))
            # (line number in the file, cells): the advisory below names the
            # line a user would look at in an editor.
            rows = [(reader.line_num, row) for row in reader]
            if csv_header and rows:
                rows = rows[1:]  # skip header
            rows = [(line, row) for line, row in rows if row]
            problem = _csv_tensor_problem(rows, p.name)
            if problem is None:
                data = [[float(cell) for cell in row] for _, row in rows]
                tensor = torch.tensor(data, dtype=torch.float32)
                logger.info("Loaded CSV with shape %s", tensor.shape)
            else:
                log_line = emit_advisory(
                    problem + " The tensor output is empty; the text output is unaffected.",
                    kind="file_reader_csv_not_numeric", prefix=_LOG_PREFIX,
                    context=context, logger=logger)
                return {"text": text, "tensor": tensor, "__log__": log_line}

        return {"text": text, "tensor": tensor}


def _csv_tensor_problem(rows: list[tuple[int, list[str]]], file_name: str) -> str | None:
    """Why *rows* cannot become a 2D float tensor, or ``None`` if they can.

    Ragged rows used to be reported as "non-numeric data", which sent the user
    looking for text in a file that had none.
    """
    if not rows:
        return None
    first_line, first_row = rows[0]
    width = len(first_row)
    for line, row in rows:
        if len(row) != width:
            return (
                f"{file_name} line {line} has {len(row)} column(s), but line "
                f"{first_line} has {width}; every row must have the same "
                f"number of columns to form a 2D tensor."
            )
        for column, cell in enumerate(row, start=1):
            try:
                float(cell)
            except ValueError:
                return (
                    f"{file_name} line {line}, column {column} is {cell!r}, "
                    f"which is not a number; csv mode builds a tensor only "
                    f"when every cell is numeric."
                )
    return None
