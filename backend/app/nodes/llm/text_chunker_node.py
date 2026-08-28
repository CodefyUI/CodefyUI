"""TextChunker -- whole documents in, embeddable chunks with offsets out.

The second node of the RAG chain::

    DocumentLoader -> TextChunker -> TextEmbedding -> VectorStore

**Why chunk at all.** An encoder has a token cap and a prompt has a budget,
so a five-page document cannot be one vector -- but the real reason is
narrower than either limit. One vector for a whole file is the AVERAGE of
everything in it, and the average of five pages is close to nothing in
particular: the question "how do I install a plugin?" scores about the same
against every file in the corpus. Retrieval only works when the unit being
compared is small enough to be ABOUT one thing, so the paragraph that
answers the question can win.

**Why offsets travel with every chunk.** ``start_char`` and ``end_char``
index the document the chunk came from, and the node guarantees
``text[start_char:end_char] == chunk``. That is what makes a citation
checkable rather than decorative: a learner can open the file and find the
passage, and a later node can widen a retrieved chunk back into its
surrounding context without re-running the chunker. Keeping that invariant
true is most of this module -- a chunk is always a SLICE of the original
text, never a string assembled from pieces, and the three places that could
break it (stripping a window's whitespace edges, packing sentences with the
separators between them, merging a short tail into its predecessor) all move
the offsets rather than rebuild the text.

**Why three strategies and not one.** ``characters`` is the only one that
works everywhere: Chinese has no spaces and no reliable word boundary, so a
fixed character window is both the simplest rule and the language-neutral
one. ``sentences`` and ``paragraphs`` buy something a window cannot -- a
chunk that starts and ends where the author meant it to -- at the cost of
being punctuation-dependent, and both still pack up to ``chunk_size`` so the
cap holds no matter which is chosen. A single piece longer than the cap
falls back to character windows, because a strategy that promises a maximum
has to keep it.

**Why overlap belongs to ``characters`` alone.** A window boundary lands
wherever the arithmetic puts it, which is usually mid-sentence, so the
overlap exists to make sure the cut sentence survives whole in the next
chunk. The packing strategies cut on boundaries the author wrote, so there
is nothing to repair, and duplicating a paragraph into two chunks would just
return the same passage twice from one search. The param is therefore hidden
for those strategies -- and, because a hidden param is still present in the
saved graph, its "must be smaller than chunk_size" rule is only enforced
where the param is actually read.

**What the Inspector shows.** No special-casing here: ``chunks`` is a list
of strings, so ``output_entries`` embeds the values inline while there are
at most 256 of them and reports length only beyond that, and ``metadata`` is
a list of dicts, which never embeds. That is the intended split -- the chunk
texts are what a learner wants to read, and they stay small enough to ride
the event stream at teaching scale.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)
from ...core.step_trace import StepRecorder

logger = logging.getLogger(__name__)

#: The three things the ``strategy`` param may say.
STRATEGIES = ["characters", "sentences", "paragraphs"]

#: End of a sentence: the ASCII terminators plus their full-width CJK twins,
#: consumed by a lookbehind so the punctuation stays with the sentence it
#: closed, and any whitespace after it is swallowed as the separator.
#: Chinese writes 。！？ and no spaces at all, which is exactly the case a
#: whitespace-based splitter gets wrong.
_SENTENCE_END = re.compile(r"(?<=[.!?。！？])\s*")

#: A blank line -- one newline, optional whitespace, another newline. Greedy
#: ``\s*`` so a run of blank lines is one separator rather than several
#: separators with empty paragraphs between them.
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


def _integer(params: dict[str, Any], name: str, default: int, *,
             minimum: int) -> int:
    """A whole-number param, clamped to *minimum*, defaulted on null/empty.

    Clamped rather than refused, like ``TextEmbedding``'s twin: the editor
    already bounds these with min/max, so an out-of-range value arrived from
    a hand-edited graph or an exported script, and chunking at the smallest
    workable size is a better answer than refusing to chunk. A value that is
    not a number at all does raise -- there is nothing to clamp.

    *minimum* is what the ALGORITHM needs, not what the editor asks for --
    ``chunk_size`` is floored at 1 though its widget starts at 20. Flooring
    it at 20 would be the one clamp that lies: this node's whole promise is
    that ``chunk_size`` is the cap, and a script asking for 10 would get
    20-character chunks back. The editor's min_value is a pedagogical bound
    and belongs in the editor.
    """
    raw = params.get(name, default)
    if raw is None or raw == "":
        raw = default
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"TextChunker: {name} must be a whole number, got {raw!r}."
        ) from exc
    return max(minimum, value)


def _strip_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    """``(start, end)`` with whitespace edges trimmed, or None if empty.

    The offset-preserving form of ``str.strip()``: it moves the boundaries
    instead of producing a new string, which is what keeps
    ``text[start:end] == chunk`` true after a window opens or closes in the
    middle of a run of spaces. Same predicate ``strip()`` uses, so the two
    always agree about what counts as whitespace.
    """
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return None if start >= end else (start, end)


def _split_spans(text: str, pattern: re.Pattern[str]) -> list[tuple[int, int]]:
    """Spans of the pieces *pattern* separates *text* into.

    ``re.split(pattern, text)`` gives the same pieces and no offsets, and
    offsets are the point. Walking the separator matches keeps both halves
    of the answer, and it handles the zero-width match a lookbehind
    separator produces (``"A.B"`` splits between the ``.`` and the ``B``
    with nothing between them) the same way ``re.split`` does.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in pattern.finditer(text):
        spans.append((cursor, match.start()))
        cursor = match.end()
    spans.append((cursor, len(text)))
    return spans


def _character_spans(text: str, start: int, end: int, *,
                     size: int, step: int) -> list[tuple[int, int]]:
    """Fixed-size windows over ``text[start:end]``, stripped and non-empty.

    The walk stops as soon as a window reaches *end* rather than stepping
    once more: a final window that begins inside its predecessor and ends at
    the same place is a duplicate of text already emitted, and it would be
    embedded, stored and retrieved a second time.
    """
    spans: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        window = _strip_span(text, cursor, min(cursor + size, end))
        if window is not None:
            spans.append(window)
        if cursor + size >= end:
            break
        cursor += step
    return spans


def _packed_spans(text: str, pieces: list[tuple[int, int]], *,
                  size: int) -> list[tuple[int, int]]:
    """Greedily fill chunks of at most *size* characters from *pieces*.

    A chunk grows by absorbing whole pieces until the next one would push it
    past the cap; the chunk is then the SLICE from the first piece's start to
    the last piece's end, separators included. Measuring the candidate as
    ``end - chunk_start`` rather than as a sum of piece lengths is what makes
    that honest -- the blank line between two packed paragraphs is in the
    chunk, so it has to be in the count.
    """
    spans: list[tuple[int, int]] = []
    current: tuple[int, int] | None = None
    for piece in pieces:
        piece_start, piece_end = piece
        if piece_end - piece_start > size:
            # One sentence longer than the whole budget. Nothing can pack it,
            # so it is windowed instead -- with no overlap, since the only
            # neighbours it would share text with are windows of itself.
            if current is not None:
                spans.append(current)
                current = None
            spans.extend(
                _character_spans(text, piece_start, piece_end,
                                 size=size, step=size))
        elif current is None:
            current = piece
        elif piece_end - current[0] <= size:
            current = (current[0], piece_end)
        else:
            spans.append(current)
            current = piece
    if current is not None:
        spans.append(current)
    return spans


def _merge_short_tail(spans: list[tuple[int, int]],
                      minimum: int) -> list[tuple[int, int]]:
    """Fold a too-short LAST chunk into the one before it.

    The last window of a document is whatever is left over, which is
    routinely a few characters. Embedded on its own it becomes a vector with
    almost no meaning in it, and a search that returns it shows the learner a
    citation with three words under it. Joining it to its predecessor can
    push that chunk past ``chunk_size``; that is the trade, and it is the
    right way round, because the cap is about what an encoder can read and a
    little over is far cheaper than a chunk that says nothing.

    Only the tail, and only when there is something to merge into: a whole
    document shorter than *minimum* stays as its single chunk rather than
    disappearing.
    """
    if len(spans) >= 2 and spans[-1][1] - spans[-1][0] < minimum:
        tail = spans.pop()
        spans[-1] = (spans[-1][0], max(spans[-1][1], tail[1]))
    return spans


def _document_spans(text: str, *, strategy: str, chunk_size: int,
                    chunk_overlap: int,
                    min_chunk_chars: int) -> list[tuple[int, int]]:
    """Every chunk of one document, as spans into *text*."""
    if strategy == "characters":
        # ``execute`` has already refused an overlap at or above chunk_size,
        # so the step is at least 1 by the time it gets here. The floor is
        # belt and braces for a future caller: a step of 0 is not a wrong
        # answer, it is a window that never advances, i.e. a hung run.
        spans = _character_spans(text, 0, len(text), size=chunk_size,
                                 step=max(1, chunk_size - chunk_overlap))
    else:
        pattern = (_SENTENCE_END if strategy == "sentences"
                   else _PARAGRAPH_BREAK)
        # Pieces are stripped BEFORE packing, so a packed chunk's own edges
        # are always non-whitespace and need no second trim.
        pieces = [
            span for span in (
                _strip_span(text, start, end)
                for start, end in _split_spans(text, pattern)
            ) if span is not None
        ]
        spans = _packed_spans(text, pieces, size=chunk_size)
    return _merge_short_tail(spans, min_chunk_chars)


def _document_entry(item: Any, index: int) -> tuple[str, str]:
    """One element of the ``documents`` input, as ``(text, source)``.

    A dict is what ``DocumentLoader`` produces and what every later node
    reads; a bare string is what a hand-built list or a ``Split`` node
    produces, and it carries no name, so it is called ``doc-<position>``.
    Numbering from 0 to match ``chunk_index``, so both numbers in a citation
    count the same way.
    """
    fallback = f"doc-{index}"
    if isinstance(item, dict):
        if "text" not in item:
            raise ValueError(
                "TextChunker: a dict on the `documents` input must carry a "
                "'text' key holding the document; this one has "
                f"{sorted(str(key) for key in item)}.")
        return str(item["text"]), str(item.get("source") or fallback)
    return str(item), fallback


class TextChunkerNode(BaseNode):
    NODE_NAME = "TextChunker"
    CATEGORY = "LLM"
    DESCRIPTION = (
        "Cut documents into overlapping chunks small enough to embed and to "
        "fit in a prompt. Retrieval works on chunks, not whole files: a "
        "question should pull back the paragraph that answers it, not a "
        "5-page document. characters is language-neutral and works for "
        "Chinese, which has no spaces; sentences and paragraphs keep natural "
        "boundaries and pack them up to chunk_size."
    )

    # Stated rather than inherited, like DocumentLoader's: this and the
    # loader are the two nodes of the RAG chain that need nothing
    # downloaded, and that is worth writing down next to the ones that do.
    REQUIRES_PACK = None

    # A pure function of its inputs -- same documents and same params, same
    # chunks, every time. No files are read (that is DocumentLoader's job,
    # and its fingerprint already busts this node's key through the upstream
    # hash), so there is no external state to fingerprint here.
    cacheable = True

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="documents",
                data_type=DataType.LIST,
                description=(
                    "{text, source} dicts from DocumentLoader, or plain "
                    "strings"
                ),
                optional=True,
            ),
            PortDefinition(
                name="text",
                data_type=DataType.STRING,
                description="A single document; source is reported as 'text'",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="chunks",
                data_type=DataType.LIST,
                description=(
                    "The chunk texts, one string each; wire to "
                    "TextEmbedding.texts"
                ),
            ),
            PortDefinition(
                name="metadata",
                data_type=DataType.LIST,
                description=(
                    "One {source, chunk_index, start_char, end_char} dict per "
                    "chunk, in the same order; wire to VectorStore.metadata"
                ),
            ),
            PortDefinition(
                name="count",
                data_type=DataType.SCALAR,
                description="How many chunks came out.",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="strategy",
                param_type=ParamType.SELECT,
                default="characters",
                options=list(STRATEGIES),
                description=(
                    "characters: fixed-size character windows. sentences: "
                    "split on . ! ? and CJK 。！？ then pack. paragraphs: "
                    "split on blank lines then pack."
                ),
            ),
            ParamDefinition(
                name="chunk_size",
                param_type=ParamType.INT,
                default=400,
                min_value=20,
                max_value=20000,
                description=(
                    "Characters per chunk (the cap for every strategy)."
                ),
            ),
            ParamDefinition(
                name="chunk_overlap",
                param_type=ParamType.INT,
                default=80,
                min_value=0,
                max_value=5000,
                description=(
                    "Characters shared by consecutive chunks so a sentence "
                    "cut in half still appears whole in one of them; must be "
                    "smaller than chunk_size."
                ),
                visible_when={"strategy": "characters"},
            ),
            ParamDefinition(
                name="min_chunk_chars",
                param_type=ParamType.INT,
                default=40,
                min_value=1,
                description=(
                    "A trailing chunk shorter than this is merged into the "
                    "previous one."
                ),
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
        strategy = str(params.get("strategy") or "characters")
        if strategy not in STRATEGIES:
            raise ValueError(
                f"TextChunker: unknown strategy {strategy!r}; set the "
                f"`strategy` param to one of {STRATEGIES}.")

        chunk_size = _integer(params, "chunk_size", 400, minimum=1)
        chunk_overlap = _integer(params, "chunk_overlap", 80, minimum=0)
        min_chunk_chars = _integer(params, "min_chunk_chars", 40, minimum=1)
        # Only where the param is visible -- see the module docstring. A
        # graph switched to sentences still carries whatever overlap the
        # canvas wrote onto the node, and failing on a value nothing reads
        # would name a field the learner cannot even see.
        if strategy == "characters" and chunk_overlap >= chunk_size:
            raise ValueError(
                f"TextChunker: chunk_overlap ({chunk_overlap}) must be "
                f"smaller than chunk_size ({chunk_size}); at or above it the "
                f"window never advances and the same text repeats forever.")

        documents = self._resolve_documents(inputs)

        chunks: list[str] = []
        metadata: list[dict[str, Any]] = []
        for text, source in documents:
            spans = _document_spans(
                text, strategy=strategy, chunk_size=chunk_size,
                chunk_overlap=chunk_overlap, min_chunk_chars=min_chunk_chars)
            for index, (start, end) in enumerate(spans):
                chunks.append(text[start:end])
                metadata.append({
                    "source": source,
                    # Within the DOCUMENT, not within the run: (source,
                    # chunk_index) is an address a citation can print, while
                    # a running counter would only repeat the list index.
                    "chunk_index": index,
                    "start_char": start,
                    "end_char": end,
                })

        total_chars = sum(len(text) for text, _ in documents)
        mean_chars = (sum(len(chunk) for chunk in chunks) / len(chunks)
                      if chunks else 0.0)
        note = (
            f"{len(chunks)} {'chunk' if len(chunks) == 1 else 'chunks'} from "
            f"{len(documents)} "
            f"{'document' if len(documents) == 1 else 'documents'} "
            f"(mean {mean_chars:.0f} chars)"
        )

        result: dict[str, Any] = {
            "chunks": chunks,
            "metadata": metadata,
            "count": len(chunks),
            "__log__": note,
        }

        if context is not None and getattr(context, "verbose", False):
            recorder = StepRecorder()
            recorder.record(
                "input",
                f"{len(documents)} document(s), {total_chars:,} characters, "
                f"to cut with the {strategy} strategy.",
                scalars={"documents": float(len(documents)),
                         "total_chars": float(total_chars)},
            )
            recorder.record(
                "chunk",
                f"Cut to at most {chunk_size} characters each: "
                f"{len(chunks)} chunk(s), averaging {mean_chars:.0f} "
                f"characters.",
                scalars={"count": float(len(chunks)),
                         "mean_chars": float(mean_chars)},
            )
            result["__steps__"] = recorder.steps

        logger.info("TextChunker: %s", note)
        return result

    @staticmethod
    def _resolve_documents(inputs: dict[str, Any]) -> list[tuple[str, str]]:
        """Every document to cut, as ``(text, source)`` pairs.

        The two inputs ADD UP rather than one winning, which is the opposite
        of ``TextEmbedding``'s rule and for the reason that rule gives:
        picking one there would drop half of what was connected. Here
        nothing is dropped -- chunking is per-document, every chunk carries
        its own source, and a corpus plus one typed note is a perfectly
        ordinary thing to want.

        A CONNECTED but empty list is not the "nothing wired" error: the
        wiring is right and the upstream node found no files, which
        ``DocumentLoader`` has already reported in its own words.
        """
        listed = inputs.get("documents")
        single = inputs.get("text")
        if listed is None and single is None:
            raise ValueError("TextChunker needs documents or text.")

        documents: list[tuple[str, str]] = []
        if listed is not None:
            if isinstance(listed, str):
                # ``list("hello")`` is five one-character documents, and
                # every layer below would accept that: five chunks come
                # back, the offsets are right, nothing raises.
                raise ValueError(
                    "TextChunker: the `documents` input carries a single "
                    "string, not a list of documents. Wire it into the "
                    "`text` input instead.")
            documents.extend(
                _document_entry(item, index)
                for index, item in enumerate(listed))
        if single is not None:
            documents.append((str(single), "text"))
        return documents
