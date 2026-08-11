"""TextCorpusDatasetNode — 把文字語料載成「每列一段文字」的 DATASET。

`HuggingFaceDataset` 是影像分類形狀（image/label 欄位），沒有節點能把
TinyStories、wikitext 這類文字語料帶進圖裡。這顆載入 HuggingFace Hub 的文字
資料集（或本機文字檔，一行一筆），輸出原始字串列的 DATASET，接
LMTokenizedDataset 編碼打包後就是 LM 訓練資料。
"""

from __future__ import annotations

from typing import Any

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)


class TextCorpusDatasetNode(BaseNode):
    NODE_NAME = "TextCorpusDataset"
    CATEGORY = "LLM"
    DESCRIPTION = (
        "Load a plain-text corpus as a dataset of raw text rows — from a "
        "HuggingFace Hub text dataset (e.g. roneneldan/TinyStories, wikitext) "
        "or a local text file with one sample per line. Feed the rows into "
        "LMTokenizedDataset to build packed LM training data."
    )

    # Network + on-disk HF cache: neither the remote revision nor the cached
    # files are visible to the cache key (same reasoning, same wording risk
    # as HuggingFaceDataset — a hit would pin the graph to whatever the
    # first run happened to fetch).
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="dataset",
                data_type=DataType.DATASET,
                description="Rows of raw text (torch.utils.data.Dataset of str)",
            ),
            PortDefinition(
                name="num_rows",
                data_type=DataType.SCALAR,
                description="Number of text rows loaded",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="source",
                param_type=ParamType.SELECT,
                default="huggingface",
                options=["huggingface", "local_file"],
                description="Where the corpus comes from",
            ),
            ParamDefinition(
                name="dataset_name",
                param_type=ParamType.STRING,
                default="roneneldan/TinyStories",
                description="HuggingFace Hub repo id (e.g. roneneldan/TinyStories, wikitext)",
                visible_when={"source": "huggingface"},
            ),
            ParamDefinition(
                name="subset",
                param_type=ParamType.STRING,
                default="",
                description="Config name for multi-config datasets (empty = none; wikitext needs e.g. wikitext-103-raw-v1)",
                visible_when={"source": "huggingface"},
            ),
            ParamDefinition(
                name="split",
                param_type=ParamType.STRING,
                default="train",
                description="Split: train/validation/test, or HF slice syntax (train[:10000])",
                visible_when={"source": "huggingface"},
            ),
            ParamDefinition(
                name="text_column",
                param_type=ParamType.STRING,
                default="text",
                description="Column holding the text",
                visible_when={"source": "huggingface"},
            ),
            ParamDefinition(
                name="local_file",
                param_type=ParamType.DATA_FILE,
                default="",
                description="Local text file, one sample per line (blank lines skipped)",
                visible_when={"source": "local_file"},
            ),
            ParamDefinition(
                name="max_rows",
                param_type=ParamType.INT,
                default=0,
                min_value=0,
                description="Keep only the first N rows (0 = all) — the cheap way to budget a pilot run",
            ),
            ParamDefinition(
                name="cache_dir",
                param_type=ParamType.STRING,
                default="",
                description="Override the HuggingFace cache directory (empty = HF default)",
                visible_when={"source": "huggingface"},
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
        source = str(params.get("source", "huggingface"))
        max_rows = max(0, int(params.get("max_rows", 0) or 0))

        if source == "local_file":
            dataset = self._load_local(str(params.get("local_file", "")), max_rows)
        else:
            dataset = self._load_huggingface(params, max_rows)
        return {"dataset": dataset, "num_rows": len(dataset)}

    # -- local file --------------------------------------------------------
    @staticmethod
    def _load_local(path_str: str, max_rows: int):
        from pathlib import Path

        from ...config import settings
        from ..data._hf_adapter import LocalTextListDataset

        if not path_str.strip():
            raise ValueError(
                "TextCorpusDataset with source='local_file' needs the "
                "local_file parameter (upload a .txt through the file picker)"
            )
        path = Path(path_str)
        # A bare filename is what the DATA_FILE upload dropdown produces —
        # resolve it against DATA_FILES_DIR (same rule as CSVReader).
        if not path.is_absolute() and path.parent == Path("."):
            candidate = settings.DATA_FILES_DIR / path.name
            if candidate.is_file():
                path = candidate
        if not path.is_file():
            raise RuntimeError(f"Text file not found: {path_str}")
        rows: list[str] = []
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                rows.append(stripped)
                if max_rows and len(rows) >= max_rows:
                    break
        if not rows:
            raise RuntimeError(f"Text file has no non-empty lines: {path_str}")
        return LocalTextListDataset(rows)

    # -- HuggingFace -------------------------------------------------------
    @staticmethod
    def _load_huggingface(params: dict[str, Any], max_rows: int):
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise RuntimeError(
                "TextCorpusDataset requires the 'datasets' package. "
                "Install with: pip install datasets"
            ) from error

        from ..data._hf_adapter import HFTorchTextDataset

        dataset_name = str(params.get("dataset_name", "roneneldan/TinyStories"))
        subset = str(params.get("subset", "") or "") or None
        split = str(params.get("split", "train") or "train")
        text_column = str(params.get("text_column", "text"))
        cache_dir = str(params.get("cache_dir", "") or "") or None

        try:
            # trust_remote_code=False: same dataset-script RCE stance as
            # HuggingFaceDataset.
            ds = load_dataset(
                dataset_name,
                subset,
                split=split,
                cache_dir=cache_dir,
                trust_remote_code=False,
            )
        except Exception as error:
            name = type(error).__name__
            message = str(error)
            looks_like_auth = (
                "GatedRepoError" in name
                or "RepositoryNotFoundError" in name
                or "401" in message
                or "unauthorized" in message.lower()
            )
            if looks_like_auth:
                raise RuntimeError(
                    "HuggingFace authentication required to load "
                    f"'{dataset_name}'. Set the HF_TOKEN environment variable "
                    "to a token with read access."
                ) from error
            raise

        available = list(ds.features.keys()) if hasattr(ds, "features") else None
        if available is not None and text_column not in available:
            raise RuntimeError(
                f"Column '{text_column}' not found in dataset "
                f"'{dataset_name}'. Available columns: {available}"
            )
        if max_rows and len(ds) > max_rows:
            ds = ds.select(range(max_rows))
        return HFTorchTextDataset(ds, text_column=text_column)
