"""TextCorpusDataset node (#290) -- raw text rows, from the Hub or from a file.

The first node of the LM training chain::

    TextCorpusDataset -> LMTokenizedDataset -> DataLoader -> TrainingLoop
                              ^
                        LMTokenizer

**What comes out of the ``dataset`` port is NOT what other dataset nodes emit.**
``ImageFolderDataset`` and ``HuggingFaceDataset`` yield ``(data, target)``
pairs, which is what ``DataLoader`` and ``TrainingLoop`` consume. This one
yields plain strings, one per document, because a corpus has no targets yet --
the targets are the next tokens, and they only exist after tokenization. Wiring
this straight into ``DataLoader`` therefore fails; ``LMTokenizedDataset`` goes
in between.

**Why it lives in the LLM category rather than Data.** It only makes sense as
the head of the chain above, and the seven nodes of the LLM epic (#292) are
easier to find in one palette section than split across two.

Two sources, one port. ``huggingface`` streams a published corpus (the default,
``roneneldan/TinyStories``, is small and readable enough that a learner can see
what the model is imitating); ``local_file`` reads a ``.txt`` the learner wrote
or uploaded, as one document or one document per line.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)

logger = logging.getLogger(__name__)

#: The two things the ``source`` param may say.
SOURCES = ["huggingface", "local_file"]


class TextRowDataset(Dataset):
    """A list of strings as a map-style torch Dataset.

    Module scope, not a closure inside the node, so ``torch.save`` and a
    DataLoader worker process can name the class (#283).
    """

    def __init__(self, rows: list[str]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> str:
        # Plain list indexing, so an out-of-range index raises IndexError --
        # which is also what makes ``for row in dataset`` terminate.
        return self.rows[idx]


def _resolve_corpus_path(path_str: str) -> Path:
    """Where a ``local_path`` param points, under the same rules ``CSVReader``
    uses for its DATA_FILE param.

    Two cases beyond "use it as typed":

    1. A BARE FILENAME is what the DATA_FILE upload dropdown produces, so it
       resolves against ``DATA_FILES_DIR`` -- picking a file must not depend on
       the server's working directory.
    2. In PROJECT MODE a relative path resolves inside the project directory and
       may not escape it, so a graph shared between machines cannot read
       arbitrary files on the one that opens it.

    Deliberately NOT importing ``CSVReaderNode._resolve_path``: that one also
    special-cases the bundled ``data/samples/iris.csv`` and puts "CSVReader" in
    its error message, neither of which belongs here. The shared rule is small
    enough that a copy with this comment is cheaper than a third party module
    both nodes have to agree with; if a fourth DATA_FILE reader appears, extract
    it then.
    """
    from ...config import settings

    path = Path(path_str)
    if not path.is_absolute() and path.parent == Path("."):
        candidate = settings.DATA_FILES_DIR / path.name
        if candidate.is_file():
            return candidate
    if settings.PROJECT_DIR is not None and not path.is_absolute():
        project = settings.PROJECT_DIR.resolve()
        resolved = (project / path).resolve()
        if not resolved.is_relative_to(project):
            raise ValueError(
                f"TextCorpusDataset: local_path {path_str!r} escapes the "
                f"project directory. Use a path inside the project, or upload "
                f"the file and pick it from the dropdown.")
        return resolved
    return path


def _max_rows(params: dict[str, Any]) -> int:
    """The ``max_rows`` cap, floored at 0 (= no cap).

    The ``or 0`` idiom is safe here precisely because 0 is the disabled value:
    a falsy ``max_rows`` and an absent one mean the same thing.
    """
    raw = params.get("max_rows", 0) or 0
    try:
        return max(0, int(raw))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"TextCorpusDataset: max_rows must be a whole number (0 = every "
            f"row), got {params.get('max_rows')!r}."
        ) from exc


class TextCorpusDatasetNode(BaseNode):
    NODE_NAME = "TextCorpusDataset"
    CATEGORY = "LLM"
    DESCRIPTION = (
        "Load a text corpus as rows of raw text -- either a dataset from the "
        "HuggingFace Hub or a .txt file you uploaded. This is the raw material "
        "for language-model training, so it has no targets yet: wire it into "
        "LMTokenizedDataset (not straight into DataLoader), which cuts the "
        "text into next-token training blocks."
    )

    # Same reasoning as HuggingFaceDataset: the huggingface source hits the
    # network and the on-disk HF cache, and neither the remote revision nor the
    # cached files are visible to the cache key -- so a hit would pin the graph
    # to whatever the first run happened to fetch. The local_file source COULD
    # carry a path fingerprint (#144's pattern), but ``cacheable`` is one flag
    # for the whole node and the fingerprint hook cannot say "only in this
    # mode"; re-reading a text file is cheap, and the cost of getting this
    # wrong for the network mode is a stale corpus nobody can explain.
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
                description=(
                    "Rows of raw text, one string per document -- NOT "
                    "(data, target) pairs. Feed it to LMTokenizedDataset; "
                    "DataLoader cannot batch it as it stands."
                ),
            ),
            PortDefinition(
                name="num_rows",
                data_type=DataType.SCALAR,
                description="How many documents were loaded.",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="source",
                param_type=ParamType.SELECT,
                default="huggingface",
                options=list(SOURCES),
                description=(
                    "Where the text comes from: a published dataset on the "
                    "HuggingFace Hub, or a .txt file on this machine."
                ),
            ),
            ParamDefinition(
                name="dataset_name",
                param_type=ParamType.STRING,
                default="roneneldan/TinyStories",
                description=(
                    "HuggingFace Hub repo id. TinyStories is ~2M simple "
                    "children's stories -- small enough to train on and "
                    "readable enough to judge the output by eye."
                ),
                visible_when={"source": "huggingface"},
            ),
            ParamDefinition(
                name="subset",
                param_type=ParamType.STRING,
                default="",
                description=(
                    "Config name for multi-config datasets, e.g. 'wikitext-"
                    "103-raw-v1' (empty = the dataset's default)."
                ),
                visible_when={"source": "huggingface"},
            ),
            ParamDefinition(
                name="split",
                param_type=ParamType.STRING,
                default="train",
                description=(
                    "Which split to load: train/test/validation, or HF slice "
                    "syntax such as train[:5000]."
                ),
                visible_when={"source": "huggingface"},
            ),
            ParamDefinition(
                name="text_column",
                param_type=ParamType.STRING,
                default="text",
                description=(
                    "Column holding the document text. 'text' is the "
                    "convention; the error message lists the real ones if it "
                    "is wrong."
                ),
                visible_when={"source": "huggingface"},
            ),
            ParamDefinition(
                name="cache_dir",
                param_type=ParamType.STRING,
                default="",
                description=(
                    "Override where HuggingFace keeps its download cache "
                    "(empty = the HF default, usually ~/.cache/huggingface)."
                ),
                visible_when={"source": "huggingface"},
                advanced=True,
            ),
            ParamDefinition(
                name="local_path",
                param_type=ParamType.DATA_FILE,
                default="",
                description=(
                    "Text file to read. Pick one you uploaded from the "
                    "dropdown, or type a path (absolute, or relative to the "
                    "backend working dir; in project mode, relative paths "
                    "resolve inside the project directory)."
                ),
                visible_when={"source": "local_file"},
            ),
            ParamDefinition(
                name="split_lines",
                param_type=ParamType.BOOL,
                default=False,
                description=(
                    "Treat every line as a separate document instead of "
                    "reading the whole file as one. Turn it on for a file of "
                    "one sentence/example per line; leave it off for prose, "
                    "where a paragraph break is not a document break."
                ),
                visible_when={"source": "local_file"},
            ),
            ParamDefinition(
                name="max_rows",
                param_type=ParamType.INT,
                default=0,
                min_value=0,
                description=(
                    "Keep at most this many documents (0 = all). For the Hub "
                    "source this becomes split slicing, so the rest is never "
                    "downloaded -- the fast way to try a graph on a big "
                    "corpus."
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
        source = str(params.get("source", "huggingface") or "huggingface")
        if source not in SOURCES:
            raise ValueError(
                f"TextCorpusDataset: unknown source {source!r}; set the "
                f"`source` param to one of {SOURCES}.")

        limit = _max_rows(params)
        if source == "local_file":
            dataset, num_rows = self._load_local(params, limit)
        else:
            dataset, num_rows = self._load_hub(params, limit)

        logger.info(
            "TextCorpusDataset loaded %d text rows from %s", num_rows, source)
        return {"dataset": dataset, "num_rows": num_rows}

    # ── local_file ──────────────────────────────────────────────────────

    @staticmethod
    def _load_local(
        params: dict[str, Any], limit: int,
    ) -> tuple[Dataset, int]:
        path_str = str(params.get("local_path", "") or "").strip()
        if not path_str:
            raise ValueError(
                "TextCorpusDataset has no file selected. Pick one from the "
                "`local_path` dropdown, upload a .txt with the button next to "
                "it, or switch `source` to huggingface.")
        path = _resolve_corpus_path(path_str)
        if not path.is_file():
            raise FileNotFoundError(
                f"TextCorpusDataset: no file at {path}. Check `local_path`.")

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            # Reported rather than read with errors="replace": silently
            # substituting U+FFFD would train the model on mojibake and the
            # only symptom would be strange generations much later.
            raise ValueError(
                f"TextCorpusDataset: {path} is not valid UTF-8 "
                f"(byte {exc.start} of the file). Re-save it as UTF-8 -- a "
                f"corpus read with the wrong encoding trains the model on "
                f"mojibake."
            ) from exc

        if bool(params.get("split_lines", False)):
            # Whitespace-only lines are dropped, not kept as empty documents:
            # an empty row contributes nothing but an EOS token to the packed
            # stream, which teaches the model that documents end for no reason.
            rows = [line for line in text.splitlines() if line.strip()]
        else:
            rows = [text] if text.strip() else []

        if limit:
            rows = rows[:limit]
        return TextRowDataset(rows), len(rows)

    # ── huggingface ─────────────────────────────────────────────────────

    @staticmethod
    def _load_hub(
        params: dict[str, Any], limit: int,
    ) -> tuple[Dataset, int]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "TextCorpusDataset requires the 'datasets' package for the "
                "huggingface source. Install with: pip install datasets "
                "(or `uv sync` from backend/)."
            ) from exc

        from ..data._hf_adapter import HFTorchTextDataset

        dataset_name = str(
            params.get("dataset_name", "roneneldan/TinyStories")
            or "roneneldan/TinyStories")
        subset = str(params.get("subset", "") or "") or None
        split = str(params.get("split", "train") or "train")
        text_column = str(params.get("text_column", "text") or "text")
        cache_dir = str(params.get("cache_dir", "") or "") or None

        # Slice in the SPLIT, not after loading: the point of max_rows is not
        # downloading the other two million stories. Skipped when the user
        # already wrote their own slice -- "train[:3][:2]" is not valid HF
        # syntax, so the explicit slice wins and the cap is applied to the rows
        # that come back instead (below).
        if limit and "[" not in split:
            split = f"{split}[:{limit}]"

        try:
            # trust_remote_code=False blocks the HF dataset-script RCE class,
            # for the same reason HuggingFaceDataset passes it: without it,
            # ``datasets`` < 2.16 silently executes a script.py bundled in the
            # remote repo on first load.
            ds = load_dataset(
                dataset_name,
                subset,
                split=split,
                cache_dir=cache_dir,
                trust_remote_code=False,
            )
        except Exception as exc:
            err_name = type(exc).__name__
            err_msg = str(exc)
            looks_like_auth = (
                "GatedRepoError" in err_name
                or "RepositoryNotFoundError" in err_name
                or "401" in err_msg
                or "unauthorized" in err_msg.lower()
            )
            if looks_like_auth:
                raise RuntimeError(
                    f"HuggingFace authentication required to load "
                    f"'{dataset_name}'. Set the HF_TOKEN environment variable "
                    f"to a token with read access. See "
                    f"https://huggingface.co/docs/hub/security-tokens"
                ) from exc
            # Anything else keeps its own type and message: a misspelled split
            # or a missing config is not an auth problem and must not be
            # reported as one.
            raise

        available = list(ds.features.keys()) if hasattr(ds, "features") else None
        if available is not None and text_column not in available:
            raise RuntimeError(
                f"Column '{text_column}' not found in dataset "
                f"'{dataset_name}'. Available columns: {available}")

        wrapped = HFTorchTextDataset(ds, text_column=text_column)
        num_rows = len(wrapped)
        if limit and num_rows > limit:
            # Only reachable when the split carried its own slice, so the cap
            # could not go into the split string.
            wrapped = TextRowDataset([wrapped[i] for i in range(limit)])
            num_rows = limit
        return wrapped, num_rows
