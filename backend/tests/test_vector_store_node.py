"""Tests for VectorStoreNode -- the node that packs the RAG "database".

The node itself is four lines of glue over ``build_index``, so these tests
are about the two things glue can get wrong: the PORT CONTRACT the canvas
reads (an ANY index port, a SCALAR size and dim, an optional metadata
input) and the PARAMS reaching ``build_index`` intact. The matrix
arithmetic and the mandated error texts belong to ``test_vector_index.py``
and are only spot-checked here, at the boundary a learner actually wires.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from app.core.node_base import DataType, ParamType
from app.core.output_entries import _summarize_single
from app.nodes.llm._vector_index import VectorIndex
from app.nodes.llm.vector_store_node import VectorStoreNode

#: Every default the node ships, read off the node itself so a test names
#: only the params it actually changes.
_DEFAULTS = {p.name: p.default for p in VectorStoreNode.define_params()}

#: Deliberately NOT unit length: whether the node normalised is half of what
#: these tests are about, and rows that were already unit-length could not
#: tell "normalised" from "left alone".
_RAW = torch.tensor([[3.0, 0.0], [0.0, 4.0], [1.0, 1.0]])
_CHUNKS = ["east", "north", "north-east"]


@dataclass
class FakeContext:
    """The one attribute this node reads off an ExecutionContext."""

    verbose: bool = False


def _run(*, inputs=None, context=None, **params) -> dict:
    p = dict(_DEFAULTS)
    p.update(params)
    return VectorStoreNode().execute(dict(inputs or {}), p, context=context)


def test_node_metadata_and_index_port_is_any():
    assert VectorStoreNode.NODE_NAME == "VectorStore"
    assert VectorStoreNode.CATEGORY == "LLM"
    # Nothing downloaded: the index is arithmetic over vectors the encoder
    # already produced.
    assert VectorStoreNode.REQUIRES_PACK is None
    # Pure -- same embeddings, same params, same index, and no file is
    # written -- so a re-run rebuilds it from the cached embeddings.
    assert VectorStoreNode.cacheable is True
    # The index is a host-side object whose rows live on the CPU by
    # construction. Aligning the inputs would move the embedding matrix onto
    # the run device on the way in, for ``build_index`` to put straight back.
    assert VectorStoreNode.align_inputs is False

    inputs = {p.name: p for p in VectorStoreNode.define_inputs()}
    assert list(inputs) == ["embeddings", "chunks", "metadata"]
    assert inputs["embeddings"].data_type == DataType.TENSOR
    assert inputs["chunks"].data_type == DataType.LIST
    assert inputs["metadata"].data_type == DataType.LIST
    # The two that make an index are required; the labels on it are not.
    assert inputs["embeddings"].optional is False
    assert inputs["chunks"].optional is False
    assert inputs["metadata"].optional is True

    outputs = {p.name: p for p in VectorStoreNode.define_outputs()}
    assert list(outputs) == ["index", "size", "dim"]
    # ANY, not TENSOR: what travels the wire is a VectorIndex object. Typing
    # it TENSOR would let the canvas connect it to every node that takes a
    # tensor, and each of those would fail inside torch rather than here.
    assert outputs["index"].data_type == DataType.ANY
    assert outputs["size"].data_type == DataType.SCALAR
    assert outputs["dim"].data_type == DataType.SCALAR

    params = {p.name: p for p in VectorStoreNode.define_params()}
    assert list(params) == ["metric", "normalize"]
    assert params["metric"].param_type == ParamType.SELECT
    assert params["metric"].default == "cosine"
    assert params["metric"].options == ["cosine", "dot"]
    assert params["normalize"].param_type == ParamType.BOOL
    assert params["normalize"].default is True


def test_builds_index_with_unit_rows_for_cosine():
    result = _run(inputs={"embeddings": _RAW, "chunks": _CHUNKS})

    index = result["index"]
    assert isinstance(index, VectorIndex)
    assert index.metric == "cosine"
    assert index.normalized is True
    assert index.chunks == _CHUNKS
    # Stored unit-length, which is what makes a later search a single matrix
    # multiply instead of a multiply plus two normalisations.
    assert torch.allclose(index.vectors.norm(dim=1), torch.ones(3))
    # The rows still point where they did -- normalising is a rescale, not a
    # rotation.
    assert torch.allclose(index.vectors[0], torch.tensor([1.0, 0.0]))

    assert result["size"] == 3
    assert result["dim"] == 2
    assert result["__log__"] == "index of 3 chunks x 2 dims (cosine)"


def test_dot_metric_keeps_raw_rows():
    result = _run(inputs={"embeddings": _RAW, "chunks": _CHUNKS}, metric="dot")

    index = result["index"]
    assert index.metric == "dot"
    # normalize defaults to True and is documented as ignored here: scaling
    # every row to unit length is precisely the information dot was chosen to
    # keep, so obeying both would make dot a slower spelling of cosine.
    assert index.normalized is False
    assert torch.allclose(index.vectors, _RAW)
    assert result["__log__"] == "index of 3 chunks x 2 dims (dot)"


def test_length_mismatch_names_both_counts():
    with pytest.raises(ValueError) as excinfo:
        _run(inputs={"embeddings": torch.zeros(3, 2), "chunks": ["only", "two"]})

    # The node adds nothing to build_index's mandated wording: the fix is an
    # edge, and naming it is the whole point of the message.
    assert str(excinfo.value) == (
        "VectorStore got 3 embeddings but 2 chunks; both must come from "
        "the same TextChunker output"
    )


def test_metadata_optional_and_length_checked():
    # Unwired: every chunk still gets a dict, so Retriever can read
    # metadata[i] without checking whether there is one.
    without = _run(inputs={"embeddings": _RAW, "chunks": _CHUNKS})
    assert without["index"].metadata == [{}, {}, {}]

    supplied = [
        {"source": "one.md", "chunk_index": 0},
        {"source": "one.md", "chunk_index": 1},
        {"source": "two.md", "chunk_index": 0},
    ]
    with_meta = _run(
        inputs={"embeddings": _RAW, "chunks": _CHUNKS, "metadata": supplied})
    assert with_meta["index"].metadata == supplied

    with pytest.raises(ValueError) as excinfo:
        _run(inputs={"embeddings": _RAW, "chunks": _CHUNKS,
                     "metadata": supplied[:1]})

    message = str(excinfo.value)
    assert "3 chunks" in message
    assert "1 metadata" in message
    assert "TextChunker" in message


def test_empty_input_raises():
    # The encoder produced nothing. Reported as "nothing to index" rather
    # than as a length mismatch, because 0 embeddings and 0 chunks agree
    # perfectly and are still not an index.
    with pytest.raises(ValueError, match="nothing to index"):
        _run(inputs={"embeddings": torch.zeros(0, 384), "chunks": []})


def test_index_repr_is_summarisable():
    index = _run(inputs={"embeddings": _RAW, "chunks": _CHUNKS})["index"]

    # An ANY port goes through the generic branch of the WS summariser, so
    # the Inspector shows a repr and nothing else. That repr has to be worth
    # reading -- the dataclass default would spend its 200 characters on the
    # first few floats of the matrix.
    summary = _summarize_single(index)
    assert summary["type"] == "VectorIndex"
    assert summary["repr"] == "VectorIndex(size=3, dim=2, metric=cosine)"


def test_step_trace_when_verbose():
    result = _run(inputs={"embeddings": _RAW, "chunks": _CHUNKS},
                  context=FakeContext(verbose=True))

    assert [s.name for s in result["__steps__"]] == ["build"]
    step = result["__steps__"][0]
    assert step.scalars["N"] == 3.0
    assert step.scalars["D"] == 2.0

    quiet = _run(inputs={"embeddings": _RAW, "chunks": _CHUNKS})
    assert "__steps__" not in quiet
