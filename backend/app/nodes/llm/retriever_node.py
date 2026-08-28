"""Retriever -- the question in, the chunks that might answer it out.

The fifth node of the RAG chain::

    ... -> TextEmbedding -> VectorStore -> Retriever -> PromptBuilder

**What it actually does.** One matrix multiply. The question is already a
vector by the time it arrives, every chunk is already a row of the index, and
the whole of "retrieval" is scoring the query against every row and keeping
the best few -- the same kernel ``CosineSimilarity`` runs on word vectors.
There is no model here and nothing to train, which is the thing worth
learning: RAG buys its knowledge with an index, not with weights.

**Why the scores are outputs and not just plumbing.** A retriever always
returns something. Ask a corpus about Napoleon and it will hand back its
three least-unrelated paragraphs with a straight face, and the generator
downstream will write a confident answer out of them. The score is the only
signal that this happened -- a top hit near 0.3 means the corpus does not
contain the answer -- so ``scores`` and ``indices`` are ports a learner can
read, and every hit that CLEARS ``min_score`` is listed with its score in the
node's log. The ones the floor dropped are not: they are still on the
``scores`` port, and the log would otherwise read as a list of chunks that
went into the prompt when they did not.

**Why ``min_score`` filters the texts and not the numbers.** Dropping a
weak hit from ``contexts`` keeps it out of the prompt, which is the point.
Dropping it from ``scores`` too would hide exactly what the learner needs in
order to choose a threshold: what the runners-up scored, and how far under
the line they fell.

**Why ``min_score = 0`` is no floor rather than a floor at zero.** The
param's help says "0 keeps everything", and the default has to keep that
promise: cosine runs from -1, so a floor AT zero would silently drop the
anticorrelated hits -- and a corpus whose best answer scores -0.1 is the
single most instructive thing this node can show a learner. Every other
value is read literally, negatives included: somebody who types -0.5 means
-0.5, not "off".

**Why only the first query's chunks.** ``contexts`` and ``sources`` are
single-valued LIST ports -- one list of strings, not one per query row -- so
a query tensor with several rows can only put one row's texts on the wire.
The first is taken, ``scores``/``indices`` still cover all Q, and the node
says so in its log rather than letting the mismatch pass silently.
"""

from __future__ import annotations

import logging
from typing import Any

from ...core.advisories import emit_advisory, join_notes
from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)
from ...core.step_trace import StepRecorder
from ._vector_index import VectorIndex

logger = logging.getLogger(__name__)

#: What the ``top_k`` widget offers, and what a value from anywhere else is
#: clamped into. 50 rather than N because the node cannot know N until it
#: has an index, and ``VectorIndex.search`` clamps to the corpus anyway.
TOP_K_DEFAULT = 3
TOP_K_MIN = 1
TOP_K_MAX = 50

#: ``min_score``'s bounds: the cosine range. A dot-metric index can score
#: outside it, and a floor of 1.0 there is simply a strict one.
MIN_SCORE_DEFAULT = 0.0
MIN_SCORE_MIN = -1.0
MIN_SCORE_MAX = 1.0

#: The largest N -- the corpus size, the second axis of the ``[Q, N]`` score
#: matrix -- the verbose trace will record the whole matrix for. Past this it
#: is a payload rather than a picture: a step tensor is kept in the
#: ``RunOutputStore``, which is byte-budgeted and hands the value over only
#: when the Inspector asks for it, and 4096 float32 scores per query row is
#: already 16 KB of numbers nobody is going to read one by one.
MAX_TRACE_SCORES = 4096

#: Advisory delivery (``core.advisories``): the Log tab has no severity of
#: its own, so the prefix is what distinguishes this from a Print node's
#: output, and the kind is the token a client may branch on.
RETRIEVER_NOTE_PREFIX = "[Retriever] "
MULTI_QUERY_WARNING_KIND = "retriever_multi_query"


def _check_index(value: Any) -> VectorIndex:
    """Enforce that the ``index`` port really carries an index.

    The ANY port that lets ``VectorStore`` hand over a whole object is also
    a port the canvas will let anything into, so the guard is this node's
    job -- the precedent ``TextGenerate._check_tokenizer`` set for its
    duck-typed tokenizer port. Without it the first symptom is an
    ``AttributeError`` on ``.search`` from inside a node the learner did not
    write, which says nothing about the edge that caused it.

    Naming the TYPE that arrived is most of the message's value: ``list``
    usually means the chunks were wired here instead of the index, and
    ``Tensor`` means the embeddings went straight past VectorStore.
    """
    if isinstance(value, VectorIndex):
        return value
    if value is None:
        raise ValueError(
            "Retriever has no index. Wire VectorStore.index into the `index` "
            "input -- there is nothing to search without one.")
    raise ValueError(
        f"Retriever: the index input must come from a VectorStore node "
        f"(got {type(value).__name__})")


def _top_k(params: dict[str, Any]) -> int:
    """How many chunks to bring back, clamped to the widget's own bounds.

    The same split ``TextChunker._integer`` makes, and for its reasons. A
    number OUT OF RANGE is clamped: the INT widget already bounds this
    param, so such a value reached the node from a hand-edited graph or a
    generated script, and returning three chunks beats failing a run that
    has already paid for the embeddings. A value that is not a number at
    all is REFUSED -- there is nothing to clamp, and quietly answering with
    the default would hide a param the caller believes they set. Missing,
    null and empty all mean "not set" and take the default.
    """
    raw = params.get("top_k", TOP_K_DEFAULT)
    if raw is None or raw == "":
        raw = TOP_K_DEFAULT
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Retriever: top_k must be a whole number, got {raw!r}."
        ) from exc
    return max(TOP_K_MIN, min(TOP_K_MAX, value))


def _min_score(params: dict[str, Any]) -> float:
    """The score floor, clamped to the cosine range. See :func:`_top_k`.

    0.0 -- the default -- is not a floor at all but the absence of one; see
    :func:`_below_floor`.
    """
    raw = params.get("min_score", MIN_SCORE_DEFAULT)
    if raw is None or raw == "":
        raw = MIN_SCORE_DEFAULT
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Retriever: min_score must be a number, got {raw!r}."
        ) from exc
    return max(MIN_SCORE_MIN, min(MIN_SCORE_MAX, value))


def _below_floor(score: float, min_score: float) -> bool:
    """Whether *score* is under the floor -- and 0.0 is not a floor.

    The param's help says "0 keeps everything", and that has to be true:
    the default must not quietly drop the anticorrelated hits a learner is
    entitled to see (a top hit at -0.1 is the corpus saying it does not
    contain the answer, which is the lesson). Any value the learner
    actually typed IS a literal floor, negative ones included -- somebody
    who sets -0.5 means -0.5, not "off".
    """
    return min_score != 0.0 and score < min_score


def _source_of(meta: dict[str, Any]) -> str:
    """The file a chunk came from, or ``?``.

    ``?`` rather than an empty string or a skipped entry, because
    ``sources`` is read positionally: ``PromptBuilder`` pairs it with
    ``contexts`` by index, so a citation that cannot be made still has to
    occupy its slot.
    """
    source = meta.get("source")
    return str(source) if source else "?"


def _hit_label(meta: dict[str, Any]) -> str:
    """``source#chunk_index`` -- an address a learner can go and read.

    The pair is what ``TextChunker`` guarantees is unique, and printing both
    is what separates "the answer came from this file" from "the answer came
    from this passage of this file".
    """
    source = _source_of(meta)
    chunk_index = meta.get("chunk_index")
    if chunk_index is None or isinstance(chunk_index, bool):
        return source
    return f"{source}#{chunk_index}"


class RetrieverNode(BaseNode):
    NODE_NAME = "Retriever"
    CATEGORY = "LLM"
    DESCRIPTION = (
        "Find the chunks most similar to a question. Scores every stored "
        "chunk against the query embedding (one matrix multiply, the same "
        "kernel as CosineSimilarity), keeps the top_k, drops anything under "
        "min_score, and hands the chunk texts to PromptBuilder. Watch the "
        "scores: a top hit near 0.3 means the corpus probably does not "
        "contain the answer."
    )

    # Nothing downloaded: the vectors on both sides already exist, and the
    # search is arithmetic over them.
    REQUIRES_PACK = None

    # Pure: the index is immutable, this node only reads it, and the same
    # question against the same index scores the same way every time.
    cacheable = True

    # The index holds CPU rows and ``VectorIndex.scores`` coerces the query
    # to meet them, so aligning the inputs onto the run device would be undone
    # one line later. Same reasoning as ``TrainTestSplit``'s.
    align_inputs = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="query",
                data_type=DataType.TENSOR,
                description=(
                    "[1, D] (or [D]) from TextEmbedding on the question; must "
                    "use the same model as the index"
                ),
            ),
            PortDefinition(
                name="index",
                data_type=DataType.ANY,
                description="from VectorStore",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="contexts",
                data_type=DataType.LIST,
                description=(
                    "top_k chunk texts, best first (first query row when "
                    "several were embedded)"
                ),
            ),
            PortDefinition(
                name="sources",
                data_type=DataType.LIST,
                description=(
                    "source label per context from the chunk metadata ('?' "
                    "when none)"
                ),
            ),
            PortDefinition(
                name="scores",
                data_type=DataType.TENSOR,
                description="[Q, k] similarity scores",
            ),
            PortDefinition(
                name="indices",
                data_type=DataType.LIST,
                description="[Q][k] chunk indices into the index",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="top_k",
                param_type=ParamType.INT,
                default=TOP_K_DEFAULT,
                min_value=TOP_K_MIN,
                max_value=TOP_K_MAX,
                description="How many chunks to bring back.",
            ),
            ParamDefinition(
                name="min_score",
                param_type=ParamType.FLOAT,
                default=MIN_SCORE_DEFAULT,
                min_value=MIN_SCORE_MIN,
                max_value=MIN_SCORE_MAX,
                description=(
                    "Drop hits scoring below this (cosine range -1..1). 0 "
                    "keeps everything; 0.3-0.5 is a sensible floor for "
                    "e5/MiniLM"
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
        index = _check_index(inputs.get("index"))
        query = inputs.get("query")
        top_k = _top_k(params)
        min_score = _min_score(params)

        # Coercion, the dimension check and the metric all live in the index
        # (see ``_vector_index``), so this node never has an opinion about
        # what a query looks like. ``search`` split into its two halves: the
        # whole [Q, N] is what the verbose trace shows, and taking it from
        # the SAME matrix multiply the outputs came from is what stops the
        # trace and the results below it from disagreeing.
        matrix = index.scores(query)
        scores, positions = index.top_hits(matrix, top_k)
        indices = positions.tolist()

        queries = int(scores.shape[0])
        # Row 0 only: see the module docstring. An empty query matrix has no
        # row 0, and answering "nothing" beats an IndexError from a node the
        # learner did not write.
        best_scores = scores[0].tolist() if queries else []
        best_indices = indices[0] if indices else []

        contexts: list[str] = []
        sources: list[str] = []
        hits: list[str] = []
        for score, position in zip(best_scores, best_indices):
            # ``continue`` rather than ``break`` though the row is sorted:
            # the filter is a statement about each hit, and it should not
            # quietly depend on an ordering guarantee made two modules away.
            if _below_floor(score, min_score):
                continue
            meta = index.metadata[position]
            contexts.append(index.chunks[position])
            sources.append(_source_of(meta))
            hits.append(f"[{score:.2f}] {_hit_label(meta)}")

        if hits:
            note = "\n".join(hits)
        elif best_scores:
            note = (
                f"no chunk reached min_score {min_score:.2f} (best: "
                f"{best_scores[0]:.2f}) -- the corpus may not contain the "
                "answer, or the question was embedded with a different model"
            )
        else:
            note = ("the query holds no rows, so nothing was searched -- "
                    "check what is wired into the query input")
        warning = None
        if queries > 1:
            warning = emit_advisory(
                f"the query holds {queries} rows: contexts reflect the first "
                "query only; scores/indices cover all Q",
                kind=MULTI_QUERY_WARNING_KIND,
                prefix=RETRIEVER_NOTE_PREFIX,
                context=context,
                logger=logger,
            )

        result: dict[str, Any] = {
            "contexts": contexts,
            "sources": sources,
            "scores": scores,
            "indices": indices,
            # The one result key the canvas Log tab renders; dunder keys are
            # filtered out of recorded outputs and port summaries.
            "__log__": join_notes(note, warning),
        }

        if context is not None and getattr(context, "verbose", False):
            result["__steps__"] = self._trace(
                index, matrix, queries=queries, top_k=top_k,
                min_score=min_score, kept=len(contexts))

        logger.info("Retriever: %d of %d chunk(s) kept for the top query",
                    len(contexts), len(best_indices))
        return result

    @staticmethod
    def _trace(
        index: VectorIndex,
        matrix: Any,
        *,
        queries: int,
        top_k: int,
        min_score: float,
        kept: int,
    ) -> list[Any]:
        """The three steps the Teaching Inspector shows for one search."""
        recorder = StepRecorder()
        size = len(index)
        recorder.record(
            "query",
            f"The question as {queries} row(s) of {index.dim} numbers -- the "
            f"same {index.dim} dimensions every chunk in the index was "
            "embedded into.",
            scalars={"Q": float(queries), "D": float(index.dim)},
        )
        if size <= MAX_TRACE_SCORES:
            # The matrix ``execute`` already had, not a second matmul over
            # the same vectors: the losing columns are precisely what makes
            # a winning score readable, and a trace showing numbers the
            # outputs were not taken from would be worse than no trace.
            recorder.record(
                "scores",
                f"{index.metric}(query, chunk) for all {size} chunks -- one "
                f"[{queries}, {size}] matrix, computed as a single matrix "
                "multiply.",
                scores=matrix,
            )
        else:
            recorder.record(
                "scores",
                f"{index.metric}(query, chunk) for all {size} chunks. The "
                f"[{queries}, {size}] matrix is too large to show here (the "
                f"trace stops at {MAX_TRACE_SCORES}); the kept scores are on "
                "the scores output.",
                scalars={"N": float(size)},
            )
        recorder.record(
            "top_k",
            f"Keep the {top_k} highest-scoring chunks, then drop anything "
            f"under min_score {min_score:.2f}: {kept} chunk(s) go to the "
            "prompt.",
            scalars={"top_k": float(top_k), "kept": float(kept),
                     "min_score": min_score},
        )
        return recorder.steps
