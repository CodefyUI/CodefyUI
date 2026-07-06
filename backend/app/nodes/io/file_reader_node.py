import logging
from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition

logger = logging.getLogger(__name__)


class FileReaderNode(BaseNode):
    NODE_NAME = "FileReader"
    CATEGORY = "IO"
    DESCRIPTION = "Read a text or CSV file and output its contents as a string or as a tensor (for numeric CSV)"

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

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        from pathlib import Path

        import torch

        path = params.get("path", "")
        mode = params.get("mode", "text")
        encoding = params.get("encoding", "utf-8")
        csv_header = params.get("csv_header", True)

        from ...config import settings

        if not path:
            raise ValueError("File path is required")

        p = Path(path)
        if not p.is_absolute():
            if settings.PROJECT_DIR is not None:
                # Project mode: relative paths live under assets/data (spec 7.2).
                p = settings.PROJECT_DIR / "assets" / "data" / p
            else:
                p = settings.GRAPHS_DIR / p
        p = p.resolve()

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
            rows = list(reader)
            if csv_header and rows:
                rows = rows[1:]  # skip header
            try:
                data = [[float(cell) for cell in row] for row in rows if row]
                tensor = torch.tensor(data, dtype=torch.float32)
                logger.info("Loaded CSV with shape %s", tensor.shape)
            except ValueError:
                logger.warning("CSV contains non-numeric data, tensor output will be empty")

        return {"text": text, "tensor": tensor}
