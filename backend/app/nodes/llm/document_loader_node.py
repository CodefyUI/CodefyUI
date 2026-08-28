"""DocumentLoader -- plain-text documents in, ``{text, source}`` dicts out.

The first node of the RAG chain::

    DocumentLoader -> TextChunker -> TextEmbedding -> VectorStore

**Why a dict and not just a string.** Retrieval-augmented generation is only
trustworthy if the answer can be traced back to the passage it came from, and
a bare list of strings loses that the moment the chunker cuts the text up.
Carrying ``source`` alongside ``text`` from the very first node means every
chunk, every vector and every retrieved passage can still name the file it
came from -- so the generator can cite it and a learner can go read it.

**Why it is deliberately narrow.** Only ``.txt`` and ``.md``, no PDF, no HTML,
no DOCX. Each of those needs a parser with its own failure modes (scanned
pages, encodings, tables that flatten into nonsense), and none of them teach
anything about retrieval. A folder of plain text is the shortest path from
"press Run" to seeing why chunking and embedding matter; a learner with a PDF
can convert it once, outside the graph.

**Why it needs no pack.** Every other RAG node needs something downloaded --
``TextEmbedding`` wants the sentence-embeddings pack, the generator wants a
model. This one is stdlib file reading, and it ships with the corpus it reads
(``backend/data/samples/rag``), so the head of the chain always runs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)

logger = logging.getLogger(__name__)

#: The two things the ``source`` param may say.
SOURCES = ["directory", "uploaded_file"]

#: What counts as a plain-text document. Lower-cased before comparison, so
#: ``NOTES.MD`` (Windows, or an export tool that shouts) is read too.
TEXT_SUFFIXES = {".txt", ".md"}

#: The bundled corpus. Also the ``directory`` default, so a freshly dropped
#: node loads something real before anyone types a path.
DEFAULT_DIRECTORY = "data/samples/rag"

#: The CodefyUI ``backend/`` folder -- this file is ``backend/app/nodes/llm/``.
#: A relative path falls back to here so the bundled samples are found no
#: matter which directory the server was started from (``cdui start`` run from
#: a home directory is normal, and the shipped RAG examples must still run).
_BACKEND_DIR = Path(__file__).resolve().parents[3]


def _resolve_document_path(path_str: str, *, kind: str) -> Path:
    """Where a ``directory`` / ``file`` param points.

    Four rules, in order:

    1. An ABSOLUTE path is used exactly as typed.
    2. A BARE FILENAME is what the DATA_FILE upload dropdown produces, so it
       resolves against ``DATA_FILES_DIR`` when something is actually there --
       picking an uploaded file must not depend on the server's working
       directory.
    3. In PROJECT MODE a relative path resolves inside the project directory
       and may not escape it, so a graph shared between machines does not
       reach outside the project by accident. (It is not a sandbox: an
       ABSOLUTE path is honoured as typed, here and in every other reader
       node in this project, because a learner pointing at a folder of their
       own notes is the ordinary case.) The exception is a path under
       ``data/samples/``: that names the INSTALL, not the project, and
       ``CSVReader`` already carves out the same hole for ``iris.csv`` so its
       demo keeps working in project mode. The exemption is a prefix test, so
       it also requires that no component is ``..`` -- otherwise
       ``data/samples/../../elsewhere`` would wear the prefix and land
       anywhere.
    4. Otherwise: the backend working directory first, then ``backend/``.

    *kind* is the param name, and only exists so the project-escape error can
    say which field to fix.
    """
    from ...config import settings

    path = Path(path_str)
    if path.is_absolute():
        return path

    if path.parent == Path("."):
        candidate = settings.DATA_FILES_DIR / path.name
        if candidate.exists():
            return candidate

    # Backslashes normalised first: a Windows editor may well have written
    # ``data\samples\rag`` into the param, and that is the same bundled
    # corpus as the POSIX spelling.
    normalised = path_str.replace("\\", "/")
    # The prefix is not enough on its own: ``data/samples/../../elsewhere``
    # starts with it and ends outside the install, so the exemption also
    # requires that the path only ever goes downwards.
    is_bundled_sample = (
        normalised.startswith("data/samples/")
        and ".." not in Path(normalised).parts)

    if settings.PROJECT_DIR is not None and not is_bundled_sample:
        project = settings.PROJECT_DIR.resolve()
        resolved = (project / path).resolve()
        if not resolved.is_relative_to(project):
            raise ValueError(
                f"DocumentLoader: {kind} {path_str!r} escapes the project "
                f"directory. Use a path inside the project, or upload the "
                f"file and pick it from the dropdown.")
        return resolved

    from_cwd = Path.cwd() / path
    if from_cwd.exists():
        return from_cwd
    from_backend = _BACKEND_DIR / path
    if from_backend.exists():
        return from_backend
    # Neither exists: hand back the working-directory reading, which is the
    # primary rule and therefore the path the "not found" error should name.
    return from_cwd


def _max_docs(params: dict[str, Any]) -> int:
    """The ``max_docs`` cap, floored at 0 (= no cap).

    The ``or 0`` idiom is safe precisely because 0 is the disabled value: an
    absent ``max_docs`` and a falsy one mean the same thing.
    """
    raw = params.get("max_docs", 0) or 0
    try:
        return max(0, int(raw))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"DocumentLoader: max_docs must be a whole number (0 = every "
            f"document), got {params.get('max_docs')!r}."
        ) from exc


def _read_document(path: Path) -> str | None:
    """UTF-8 text of *path*, or None when there is nothing worth keeping.

    A whitespace-only file returns None rather than an empty document.
    Empty documents are worse than absent ones downstream: the chunker emits
    a zero-length chunk, the encoder turns it into a vector that sits near
    nothing in particular, and retrieval then offers it as a citation with no
    text under it.

    ``utf-8-sig`` rather than ``utf-8``: Notepad and several Windows export
    tools write a byte-order mark, which is a legal UTF-8 file whose first
    character is an invisible U+FEFF. Read as plain utf-8 that character
    survives into the first chunk, into its embedding, and into the first
    citation the model is shown -- a corruption nobody can see. The codec
    eats a BOM when there is one and behaves exactly like utf-8 when there
    is not.
    """
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        # Named and refused rather than read with errors="replace": silently
        # substituting U+FFFD would embed mojibake, and the only symptom
        # would be a retrieval result nobody can explain.
        raise ValueError(
            f"DocumentLoader: {path} is not valid UTF-8 (byte {exc.start} of "
            f"the file). Re-save it as UTF-8 -- text read with the wrong "
            f"encoding embeds as mojibake."
        ) from exc
    return text if text.strip() else None


class DocumentLoaderNode(BaseNode):
    NODE_NAME = "DocumentLoader"
    CATEGORY = "LLM"
    DESCRIPTION = (
        "Read plain-text documents (.txt and .md) from a folder or one "
        "uploaded file. Each document comes out as {text, source} so later "
        "nodes can cite where an answer came from. The bundled "
        "data/samples/rag folder holds five short bilingual notes about "
        "CodefyUI and ML basics, so the RAG examples run with no setup. "
        "First node of the RAG chain: DocumentLoader -> TextChunker -> "
        "TextEmbedding -> VectorStore."
    )

    # Stated rather than inherited: every other node in the RAG chain needs a
    # pack, so "this one does not" is the fact worth writing down.
    REQUIRES_PACK = None

    # Cacheable with a fingerprint (#144's pattern): re-reading a folder of
    # notes on every run is pure waste, and ``cache_fingerprint`` below folds
    # the files' own (size, mtime, content hash) into the key so an EDITED
    # note busts the cache instead of the key surviving unchanged because
    # only the path string was hashed.
    cacheable = True

    @classmethod
    def cache_fingerprint(cls, params: dict[str, Any]) -> Any:
        from ...core.cache_fingerprint import directory_fingerprint, path_fingerprint

        source = str(params.get("source", "directory") or "directory")
        if source == "uploaded_file":
            path_str = str(params.get("file", "") or "").strip()
            kind = "file"
        else:
            path_str = str(
                params.get("directory", DEFAULT_DIRECTORY) or "").strip()
            kind = "directory"
        if not path_str:
            return None
        try:
            resolved = _resolve_document_path(path_str, kind=kind)
        except Exception:
            # A fingerprint answers "did the input change", never "is the
            # input valid" -- execute() raises the real error.
            return None
        if source == "uploaded_file":
            return path_fingerprint(resolved)
        # Recursive even when ``recursive`` is off. A change below an
        # unread subfolder then busts a key it did not have to, which costs
        # one re-read; the opposite mistake serves a stale answer.
        return directory_fingerprint(resolved)

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="documents",
                data_type=DataType.LIST,
                description="list of {text, source} dicts",
            ),
            PortDefinition(
                name="texts",
                data_type=DataType.LIST,
                description="the same texts as plain strings",
            ),
            PortDefinition(
                name="count",
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
                default="directory",
                options=list(SOURCES),
                description=(
                    "Where the documents come from: a folder, or one file "
                    "you uploaded."
                ),
            ),
            ParamDefinition(
                name="directory",
                param_type=ParamType.STRING,
                default=DEFAULT_DIRECTORY,
                description=(
                    "Folder of .txt/.md files. Relative paths resolve against "
                    "the backend working directory, then against the CodefyUI "
                    "backend folder (so the bundled samples work from any "
                    "directory); in project mode a relative path must stay "
                    "inside the project."
                ),
                visible_when={"source": "directory"},
            ),
            ParamDefinition(
                name="recursive",
                param_type=ParamType.BOOL,
                default=False,
                description="Also read subfolders.",
                visible_when={"source": "directory"},
                advanced=True,
            ),
            ParamDefinition(
                name="file",
                param_type=ParamType.DATA_FILE,
                default="",
                description="A .txt you uploaded with the button.",
                visible_when={"source": "uploaded_file"},
            ),
            ParamDefinition(
                name="max_docs",
                param_type=ParamType.INT,
                default=0,
                min_value=0,
                description=(
                    "Keep at most this many documents, in name order "
                    "(0 = all)."
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
        source = str(params.get("source", "directory") or "directory")
        if source not in SOURCES:
            raise ValueError(
                f"DocumentLoader: unknown source {source!r}; set the `source` "
                f"param to one of {SOURCES}.")

        limit = _max_docs(params)
        if source == "uploaded_file":
            documents, where = self._load_file(params, limit)
        else:
            documents, where = self._load_directory(params, limit)

        texts = [doc["text"] for doc in documents]
        total_chars = sum(len(text) for text in texts)
        noun = "document" if len(documents) == 1 else "documents"
        logger.info(
            "DocumentLoader loaded %d %s (%d chars) from %s",
            len(documents), noun, total_chars, where)
        return {
            "documents": documents,
            "texts": texts,
            "count": len(documents),
            "__log__": (
                f"loaded {len(documents)} {noun} ({total_chars:,} chars) "
                f"from {where}"
            ),
        }

    # -- directory -------------------------------------------------------

    @staticmethod
    def _load_directory(
        params: dict[str, Any], limit: int,
    ) -> tuple[list[dict[str, str]], str]:
        directory = str(
            params.get("directory", DEFAULT_DIRECTORY) or "").strip()
        if not directory:
            raise ValueError(
                "DocumentLoader has no folder set. Put a path in the "
                "`directory` param (the bundled corpus is "
                f"{DEFAULT_DIRECTORY}), or switch `source` to uploaded_file.")
        root = _resolve_document_path(directory, kind="directory")
        if not root.is_dir():
            raise FileNotFoundError(
                f"DocumentLoader: no folder at {root}. Check `directory`.")

        walk = root.rglob if bool(params.get("recursive", False)) else root.glob
        paths = sorted(
            p for p in walk("*")
            if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES
        )
        if not paths:
            raise FileNotFoundError(
                f"DocumentLoader found no .txt or .md files in {root}")

        documents: list[dict[str, str]] = []
        for path in paths:
            text = _read_document(path)
            if text is None:
                continue
            documents.append(
                # POSIX on purpose: the source string is what a citation
                # prints and what a saved run records, so it must read the
                # same on Windows as everywhere else.
                {"text": text, "source": path.relative_to(root).as_posix()})
            # Stop as soon as the cap is met rather than reading the whole
            # folder and slicing -- max_docs is how a learner tries a graph
            # against a big folder cheaply.
            if limit and len(documents) >= limit:
                break
        if not documents:
            # Files were found and every one of them was blank. Returning
            # zero documents instead would push the failure two nodes
            # downstream, where VectorStore reports an empty embedding
            # matrix -- true, and about the wrong folder. The same exception
            # class as its sibling above, because it is the same outcome for
            # the learner: there is nothing here to read.
            raise FileNotFoundError(
                f"DocumentLoader found only blank files in {root}: every "
                ".txt and .md there is empty or whitespace, so there is "
                "nothing to chunk.")
        return documents, directory

    # -- uploaded_file ---------------------------------------------------

    @staticmethod
    def _load_file(
        params: dict[str, Any], limit: int,
    ) -> tuple[list[dict[str, str]], str]:
        file_str = str(params.get("file", "") or "").strip()
        if not file_str:
            raise ValueError(
                "DocumentLoader has no file selected. Pick one from the "
                "`file` dropdown, upload a .txt with the button next to it, "
                "or switch `source` to directory.")
        path = _resolve_document_path(file_str, kind="file")
        if not path.is_file():
            raise FileNotFoundError(
                f"DocumentLoader: no file at {path}. Check `file`.")

        text = _read_document(path)
        documents = [] if text is None else [{"text": text, "source": path.name}]
        if limit:
            documents = documents[:limit]
        return documents, path.name
