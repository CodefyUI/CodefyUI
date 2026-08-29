"""Tests for RetrieverNode -- the search half of the RAG chain.

The index is built from three 2-D directions whose similarities to a query
pointing east are known exactly (1.0, 0.7071, 0.0), so every assertion here
is about ORDER, FILTERING and LABELLING rather than about floating-point
retrieval quality. Real corpora are ``test_rag_examples.py``'s job.

Two things these tests watch particularly closely, because both are ways a
retriever can look right and be useless: that ``contexts`` and ``sources``
stay aligned index for index (a citation under the wrong passage is worse
than no citation), and that ``min_score`` filters the CONTEXTS without
quietly editing the ``scores`` tensor a learner is meant to read.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from app.core.node_base import DataType, ParamType
from app.nodes.llm._vector_index import build_index
from app.nodes.llm.retriever_node import RetrieverNode

_DEFAULTS = {p.name: p.default for p in RetrieverNode.define_params()}

#: north, north-east, east -- deliberately NOT in similarity order for a
#: query pointing east, so a result of [2, 1, 0] proves a sort happened.
_VECTORS = torch.tensor([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
_CHUNKS = [
    "CodefyUI is a node-graph teaching tool.",
    "A node is a box that computes; an edge carries a tensor.",
    "Embeddings turn text into vectors so retrieval can compare them.",
]
_METADATA = [
    {"source": "01-what-is-codefyui.md", "chunk_index": 0},
    {"source": "02-nodes-and-edges.md", "chunk_index": 3},
    {"source": "04-embeddings-and-rag.md", "chunk_index": 1},
]
_EAST = torch.tensor([[1.0, 0.0]])


@dataclass
class FakeContext:
    """The one attribute this node reads off an ExecutionContext."""

    verbose: bool = False


def _index(metadata=_METADATA, metric: str = "cosine"):
    return build_index(_VECTORS, _CHUNKS, metadata, metric=metric,
                       normalize=True)


def _run(*, inputs=None, context=None, **params) -> dict:
    p = dict(_DEFAULTS)
    p.update(params)
    return RetrieverNode().execute(dict(inputs or {}), p, context=context)


def test_node_metadata():
    assert RetrieverNode.NODE_NAME == "Retriever"
    assert RetrieverNode.CATEGORY == "LLM"
    # Nothing downloaded: searching is one matrix multiply over vectors that
    # already exist.
    assert RetrieverNode.REQUIRES_PACK is None
    # Pure: the index is immutable and the same question scores it the same
    # way every time.
    assert RetrieverNode.cacheable is True
    # The index holds CPU rows and coerces the query to meet them, so moving
    # the query onto the run device on the way in would be undone one line
    # later.
    assert RetrieverNode.align_inputs is False

    inputs = {p.name: p for p in RetrieverNode.define_inputs()}
    assert list(inputs) == ["query", "index"]
    assert inputs["query"].data_type == DataType.TENSOR
    # ANY, matching VectorStore.index: what travels is a VectorIndex object.
    assert inputs["index"].data_type == DataType.ANY
    assert not any(port.optional for port in inputs.values())

    outputs = {p.name: p for p in RetrieverNode.define_outputs()}
    assert list(outputs) == ["contexts", "sources", "scores", "indices"]
    assert outputs["contexts"].data_type == DataType.LIST
    assert outputs["sources"].data_type == DataType.LIST
    assert outputs["scores"].data_type == DataType.TENSOR
    assert outputs["indices"].data_type == DataType.LIST

    params = {p.name: p for p in RetrieverNode.define_params()}
    assert list(params) == ["top_k", "min_score"]
    assert params["top_k"].param_type == ParamType.INT
    assert params["top_k"].default == 3
    assert params["top_k"].min_value == 1
    assert params["top_k"].max_value == 50
    assert params["min_score"].param_type == ParamType.FLOAT
    assert params["min_score"].default == 0.0
    assert params["min_score"].min_value == -1.0
    assert params["min_score"].max_value == 1.0


def test_returns_top_k_best_first_with_sources():
    result = _run(inputs={"query": _EAST, "index": _index()}, top_k=2)

    # east (1.0) then north-east (0.7071); north (0.0) did not make the cut.
    assert result["contexts"] == [_CHUNKS[2], _CHUNKS[1]]
    # Aligned index for index with contexts -- this is the pairing every
    # citation downstream depends on.
    assert result["sources"] == ["04-embeddings-and-rag.md",
                                 "02-nodes-and-edges.md"]
    assert result["indices"] == [[2, 1]]
    assert result["scores"].shape == (1, 2)
    assert result["scores"][0, 0].item() == pytest.approx(1.0, abs=1e-6)
    assert result["scores"][0, 1].item() == pytest.approx(0.70710678, abs=1e-6)

    # More chunks than the corpus holds: everything comes back rather than
    # torch.topk raising, which is what a default top_k of 3 over a
    # two-chunk corpus would otherwise do to a beginner's first graph.
    everything = _run(inputs={"query": _EAST, "index": _index()}, top_k=50)
    assert len(everything["contexts"]) == 3
    assert everything["indices"] == [[2, 1, 0]]

    # Empty means "not set" and takes the default, the same reading
    # TextChunker's integers give it.
    defaulted = _run(inputs={"query": _EAST, "index": _index()}, top_k="")
    assert len(defaulted["contexts"]) == 3


def test_min_score_filters_contexts_but_not_scores_tensor():
    result = _run(inputs={"query": _EAST, "index": _index()},
                  top_k=3, min_score=0.5)

    # north scores 0.0 and is dropped from what the generator will read.
    assert result["contexts"] == [_CHUNKS[2], _CHUNKS[1]]
    assert result["sources"] == ["04-embeddings-and-rag.md",
                                 "02-nodes-and-edges.md"]

    # The numbers are NOT filtered: scores and indices are the diagnostic
    # outputs, and a learner tuning min_score has to be able to see what the
    # threshold just excluded.
    assert result["scores"].shape == (1, 3)
    assert result["indices"] == [[2, 1, 0]]

    strict = _run(inputs={"query": _EAST, "index": _index()},
                  top_k=3, min_score=0.99)
    assert strict["contexts"] == [_CHUNKS[2]]

    # A threshold nothing clears is a real and instructive outcome -- the
    # corpus does not contain the answer -- so it is an empty result with an
    # explanation, not an error.
    all_dropped = _run(inputs={"query": torch.tensor([[-1.0, 0.0]]),
                               "index": _index()}, top_k=3, min_score=0.5)
    assert all_dropped["contexts"] == []
    assert all_dropped["sources"] == []
    assert all_dropped["scores"].shape == (1, 3)
    assert "min_score" in all_dropped["__log__"]

    # Empty means "not set": no floor at all, so everything comes back.
    defaulted = _run(inputs={"query": _EAST, "index": _index()},
                     top_k=3, min_score="")
    assert len(defaulted["contexts"]) == 3


def test_a_param_that_is_not_a_number_is_refused():
    """Out of range is clamped; not a number at all is an error.

    The split ``TextChunker._integer`` makes, and these two helpers now make
    the same one. A value the INT widget cannot produce came from a
    hand-edited graph or a generated script: 500 still means "as many as
    you have", so it is clamped, but ``"three"`` means the caller believes
    they set something, and answering with the default would hide that from
    them for the whole run.
    """
    with pytest.raises(ValueError) as top_k_case:
        _run(inputs={"query": _EAST, "index": _index()}, top_k="three")
    assert str(top_k_case.value) == (
        "Retriever: top_k must be a whole number, got 'three'.")

    with pytest.raises(ValueError) as min_score_case:
        _run(inputs={"query": _EAST, "index": _index()}, min_score="high")
    assert str(min_score_case.value) == (
        "Retriever: min_score must be a number, got 'high'.")

    # A NUMBER outside the widget's bounds is still clamped, not refused.
    everything = _run(inputs={"query": _EAST, "index": _index()}, top_k=500)
    assert len(everything["contexts"]) == 3
    clamped = _run(inputs={"query": _EAST, "index": _index()},
                   top_k=3, min_score=9.0)
    # Clamped to 1.0, which only the exact match clears.
    assert clamped["contexts"] == [_CHUNKS[2]]


def test_min_score_zero_is_no_floor_at_all():
    """The default keeps an ANTICORRELATED hit; -0.5 drops it.

    ``min_score``'s help says "0 keeps everything", and cosine runs from
    -1, so a floor AT zero would make that sentence false -- and would hide
    the most instructive result this node produces: a corpus whose best
    match is negative is the corpus saying it does not contain the answer.
    Any value a learner actually types is read literally instead.
    """
    west = torch.tensor([[-1.0, 0.0]])

    kept = _run(inputs={"query": west, "index": _index()}, top_k=3)
    # east scores -1.0 against a query pointing west, and it is still on
    # the wire at the default.
    assert kept["contexts"][-1] == _CHUNKS[2]
    assert kept["scores"][0, -1].item() == pytest.approx(-1.0, abs=1e-6)
    assert len(kept["contexts"]) == 3

    floored = _run(inputs={"query": west, "index": _index()}, top_k=3,
                   min_score=-0.5)
    # north-east is at -0.7071 and east at -1.0; only north (0.0) survives.
    assert floored["contexts"] == [_CHUNKS[0]]
    # The numbers are still all there -- filtering the texts, never the
    # diagnostic tensor.
    assert floored["scores"].shape == (1, 3)


def test_an_empty_query_matrix_answers_nothing():
    # Not reachable from TextEmbedding, which refuses an empty texts list,
    # but perfectly reachable from a PythonScript or a Split that produced
    # nothing. An IndexError on row 0 of a [0, D] query would be raised from
    # a node the learner did not write and would name nothing they can fix.
    result = _run(inputs={"query": torch.zeros(0, 2), "index": _index()})

    assert result["contexts"] == []
    assert result["sources"] == []
    assert result["indices"] == []
    assert "no rows" in result["__log__"]


def test_wrong_index_type_names_vector_store():
    with pytest.raises(ValueError) as excinfo:
        _run(inputs={"query": _EAST, "index": [0.1, 0.2, 0.3]})

    # The mistake is an edge: something that is not an index got wired into
    # the index port, and the message says which node produces one.
    assert str(excinfo.value) == (
        "Retriever: the index input must come from a VectorStore node "
        "(got list)"
    )

    # A tensor is the likeliest wrong thing to wire here -- the embeddings
    # themselves, straight past VectorStore -- and it is named too.
    with pytest.raises(ValueError) as tensor_case:
        _run(inputs={"query": _EAST, "index": _VECTORS})
    assert "(got Tensor)" in str(tensor_case.value)


def test_an_unwired_query_names_the_port_and_the_node_to_wire():
    """``None`` on the query port is a missing edge, not a torch problem.

    ``torch.as_tensor(None)`` raises "Could not infer dtype of NoneType",
    which is true, comes from a module the learner never opened, and names
    nothing they can draw.
    """
    expected = (
        "Retriever: the query input must be the embedding of the question "
        "(a tensor) -- connect TextEmbedding.embeddings"
    )

    with pytest.raises(ValueError) as missing:
        _run(inputs={"query": None, "index": _index()})
    assert str(missing.value) == expected

    # Something wired that no tensor can be made of -- a Print node's text,
    # a list of chunk strings -- is the same mistake from the other side.
    with pytest.raises(ValueError) as text_case:
        _run(inputs={"query": "what is a node?", "index": _index()})
    assert str(text_case.value) == expected


def test_dimension_mismatch_message():
    with pytest.raises(ValueError) as excinfo:
        _run(inputs={"query": torch.zeros(1, 3), "index": _index()})

    # Straight from VectorIndex.scores, unwrapped: the two encoders
    # disagreeing is the single most common way a RAG graph goes wrong, and
    # the message is mandated word for word.
    assert str(excinfo.value) == (
        "query dimension 3 does not match the index dimension 2 -- embed the "
        "question with the same model the documents were embedded with"
    )


def test_1d_query_is_accepted():
    flat = _run(inputs={"query": torch.tensor([1.0, 0.0]), "index": _index()},
                top_k=2)
    batched = _run(inputs={"query": _EAST, "index": _index()}, top_k=2)

    # A [D] question is one question, so it comes back as one row of answers.
    assert flat["scores"].shape == (1, 2)
    assert flat["contexts"] == batched["contexts"]
    assert flat["indices"] == batched["indices"]


def test_multi_query_uses_row_zero_and_logs():
    two = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    result = _run(inputs={"query": two, "index": _index()}, top_k=2)

    # Row 0 is east, row 1 is north, and they disagree about which chunk
    # wins -- so a node that silently used row 1 would still look plausible.
    assert result["scores"].shape == (2, 2)
    assert result["indices"] == [[2, 1], [0, 1]]
    # contexts/sources are single-valued ports, so they can only carry one
    # query's answer, and the log says which.
    assert result["contexts"] == [_CHUNKS[2], _CHUNKS[1]]
    assert ("contexts reflect the first query only; scores/indices cover "
            "all Q") in result["__log__"]

    # One query is the normal case and says nothing about it.
    single = _run(inputs={"query": _EAST, "index": _index()}, top_k=2)
    assert "first query only" not in single["__log__"]


def test_step_trace_when_verbose():
    result = _run(inputs={"query": _EAST, "index": _index()},
                  top_k=2, context=FakeContext(verbose=True))

    assert [s.name for s in result["__steps__"]] == ["query", "scores",
                                                     "top_k"]
    steps = {s.name: s for s in result["__steps__"]}
    assert steps["query"].scalars["Q"] == 1.0
    assert steps["query"].scalars["D"] == 2.0
    # The whole [Q, N] matrix, not the [Q, k] survivors: the chunks that
    # lost are what make a winning score mean anything.
    matrix = steps["scores"].tensors["scores"]
    assert matrix.shape == (1, 3)
    assert matrix[0].tolist() == pytest.approx([0.0, 0.70710678, 1.0],
                                               abs=1e-6)
    assert steps["top_k"].scalars["top_k"] == 2.0
    assert steps["top_k"].scalars["kept"] == 2.0

    quiet = _run(inputs={"query": _EAST, "index": _index()}, top_k=2)
    assert "__steps__" not in quiet


def test_verbose_trace_reuses_the_matrix_the_search_computed():
    """ONE matrix multiply, whether or not the trace is on.

    The trace shows the whole ``[Q, N]`` and the outputs show its winners,
    and those have to be the same numbers: a second ``scores`` call is both
    a second matmul over the corpus and a set of numbers that is merely
    expected to agree with the results printed under it.
    """
    index = _index()
    computed: list[object] = []
    real = index.scores

    def counted(query):
        computed.append(query)
        return real(query)

    index.scores = counted

    result = _run(inputs={"query": _EAST, "index": index}, top_k=2,
                  context=FakeContext(verbose=True))

    assert len(computed) == 1
    step = {s.name: s for s in result["__steps__"]}["scores"]
    assert step.tensors["scores"].shape == (1, 3)


def test_step_trace_omits_the_matrix_for_a_large_corpus():
    big = build_index(torch.zeros(5000, 2), [f"c{i}" for i in range(5000)],
                      None, metric="cosine", normalize=True)

    result = _run(inputs={"query": _EAST, "index": big},
                  context=FakeContext(verbose=True))

    # The step still happens -- the trace should not go quiet on the runs
    # where it is most tempting to wonder what the retriever did -- but a
    # [1, 5000] float matrix is a payload, not a picture, so only the
    # summary numbers travel.
    step = {s.name: s for s in result["__steps__"]}["scores"]
    assert step.tensors == {}
    assert "5000" in step.description


def test_log_lists_source_and_chunk_index():
    result = _run(inputs={"query": _EAST, "index": _index()}, top_k=3)

    log = result["__log__"]
    # One line per hit: the score a learner should be sceptical about, then
    # the address they can go and read.
    assert "[0.71] 02-nodes-and-edges.md#3" in log
    assert log.splitlines()[0] == "[1.00] 04-embeddings-and-rag.md#1"
    assert log.splitlines()[2] == "[0.00] 01-what-is-codefyui.md#0"

    # No metadata at all is the normal state of an index built from an
    # unwired metadata port, and a citation still has to print something.
    bare = _run(inputs={"query": _EAST, "index": _index(metadata=None)},
                top_k=1)
    assert bare["sources"] == ["?"]
    assert bare["__log__"] == "[1.00] ?"
