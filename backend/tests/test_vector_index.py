"""Tests for VectorIndex -- the object VectorStore builds and Retriever searches."""

from __future__ import annotations

import pytest
import torch

from app.core.output_entries import _summarize_single
from app.nodes.llm._vector_index import VectorIndex, build_index

# Three 2-D unit-ish directions, deliberately NOT in similarity order for a
# query pointing east: north, north-east, east. A search that returns
# [2, 1, 0] has actually sorted; one that returns [0, 1, 2] has only echoed
# the storage order back.
_NORTH_NE_EAST = torch.tensor([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
_EAST = torch.tensor([[1.0, 0.0]])


def _small_index(metric: str = "cosine", normalize: bool = True) -> VectorIndex:
    return build_index(
        _NORTH_NE_EAST,
        ["north", "north-east", "east"],
        None,
        metric=metric,
        normalize=normalize,
    )


def test_search_returns_nearest_first():
    index = _small_index()
    scores, indices = index.search(_EAST, 3)

    assert scores.shape == (1, 3)
    assert indices.shape == (1, 3)
    assert indices.dtype == torch.int64
    # east (cos 1.0), north-east (cos 0.7071), north (cos 0.0).
    assert indices[0].tolist() == [2, 1, 0]
    assert scores[0, 0].item() == pytest.approx(1.0, abs=1e-6)
    assert scores[0, 1].item() == pytest.approx(0.70710678, abs=1e-6)
    assert scores[0, 2].item() == pytest.approx(0.0, abs=1e-6)
    # Descending, which is what "best first" means downstream.
    assert scores[0, 0] >= scores[0, 1] >= scores[0, 2]


def test_dimension_mismatch_error_mentions_same_model():
    index = build_index(torch.eye(2, 4), ["a", "b"], None, metric="cosine", normalize=True)

    with pytest.raises(ValueError) as excinfo:
        index.search(torch.zeros(1, 3), 1)

    # The whole string, not three substrings: this message is mandated word
    # for word, and a substring check still passes after its actionable half
    # -- the two sides have to come from one encoder -- has been reworded
    # around the two words the assertion happens to name.
    assert str(excinfo.value) == (
        "query dimension 3 does not match the index dimension 4 -- embed the "
        "question with the same model the documents were embedded with"
    )

    # The check lives in ``scores``, which is why it reads the same whether
    # the caller asked for the best few or for the whole matrix.
    with pytest.raises(ValueError) as direct:
        index.scores(torch.zeros(1, 3))
    assert str(direct.value) == str(excinfo.value)


def test_scores_returns_full_matrix():
    index = _small_index()

    matrix = index.scores(_EAST)

    # [Q, N], not [Q, k]: this is what Retriever's verbose trace shows, so a
    # learner can see the chunks that did NOT win alongside the ones that did.
    assert matrix.shape == (1, 3)
    assert matrix[0].tolist() == pytest.approx([0.0, 0.70710678, 1.0], abs=1e-6)

    # ``search`` is topk over exactly this matrix, so the two must agree --
    # a trace showing different numbers from the results below it would be
    # worse than no trace.
    scores, indices = index.search(_EAST, 2)
    assert torch.allclose(scores, matrix[:, indices[0]], atol=1e-6)

    # One row per query, in the order the queries arrived.
    both = index.scores(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    assert both.shape == (2, 3)
    assert both[0, 2].item() == pytest.approx(1.0, abs=1e-6)
    assert both[1, 0].item() == pytest.approx(1.0, abs=1e-6)

    # Same coercion as search: a [D] question is one row of answers.
    assert index.scores(torch.tensor([1.0, 0.0])).shape == (1, 3)


def test_a_query_that_is_not_a_tensor_names_the_wiring():
    """``None`` and a string are the same mistake: no embedding arrived.

    ``torch.as_tensor`` raises "Could not infer dtype of NoneType" for the
    first and of str for the second -- both true, both from a module the
    learner never opened, and neither naming the edge that is missing.
    """
    index = _small_index()
    expected = (
        "Retriever: the query input must be the embedding of the question "
        "(a tensor) -- connect TextEmbedding.embeddings"
    )

    with pytest.raises(ValueError) as nothing:
        index.scores(None)
    assert str(nothing.value) == expected

    with pytest.raises(ValueError) as text:
        index.scores("what is a node?")
    assert str(text.value) == expected

    # And through ``search``, because that is the call Retriever makes when
    # it is not tracing.
    with pytest.raises(ValueError) as searched:
        index.search(None, 3)
    assert str(searched.value) == expected

    # A list of numbers is still a perfectly good query: the guard is about
    # what cannot become a tensor, not about what is not already one.
    assert index.scores([[1.0, 0.0]]).shape == (1, 3)


def test_top_k_is_clamped_to_size():
    index = _small_index()

    # More than the index holds: give back everything rather than raising
    # from torch.topk, which is what a default top_k of 3 over a two-chunk
    # corpus would otherwise do.
    scores, indices = index.search(_EAST, 10)
    assert scores.shape == (1, 3)
    assert indices.shape == (1, 3)

    # Below one: still one row of results, never an empty tensor.
    assert index.search(_EAST, 0)[0].shape == (1, 1)
    assert index.search(_EAST, -5)[0].shape == (1, 1)


def test_dot_metric_skips_normalisation():
    long_and_short = torch.tensor([[3.0, 0.0], [0.0, 1.0]])
    # normalize=True is what the VectorStore param defaults to; for dot it is
    # documented as ignored, because scaling the rows is exactly the
    # information the dot metric was chosen to keep.
    index = build_index(long_and_short, ["long", "short"], None, metric="dot", normalize=True)

    assert index.normalized is False
    assert torch.allclose(index.vectors, long_and_short)

    scores, indices = index.search(_EAST, 2)
    assert indices[0, 0].item() == 0
    # Raw product, not a cosine: 3.0 rather than 1.0.
    assert scores[0, 0].item() == pytest.approx(3.0)
    assert scores[0, 1].item() == pytest.approx(0.0)


def test_one_dimensional_query_is_accepted():
    index = _small_index()

    flat_scores, flat_indices = index.search(torch.tensor([1.0, 0.0]), 2)
    batched_scores, batched_indices = index.search(_EAST, 2)

    # A [D] query is one question, so it comes back as a [1, k] answer.
    assert flat_scores.shape == (1, 2)
    assert flat_indices.shape == (1, 2)
    assert torch.equal(flat_indices, batched_indices)
    assert torch.allclose(flat_scores, batched_scores)


def test_build_index_length_mismatch_names_both_counts():
    with pytest.raises(ValueError) as excinfo:
        build_index(torch.zeros(37, 4), ["chunk"] * 36, None, metric="cosine", normalize=True)

    # Mandated word for word, like the dimension message above, and pinned
    # the same way: the counts alone do not tell the learner which of their
    # two edges to go and look at.
    assert str(excinfo.value) == (
        "VectorStore got 37 embeddings but 36 chunks; both must come from "
        "the same TextChunker output"
    )


def test_build_index_fills_missing_metadata():
    no_metadata = _small_index()
    assert no_metadata.metadata == [{}, {}, {}]

    # An empty list is the same statement as no list at all -- an optional
    # port that was never wired can arrive either way.
    empty_list = build_index(
        _NORTH_NE_EAST, ["a", "b", "c"], [], metric="cosine", normalize=True
    )
    assert empty_list.metadata == [{}, {}, {}]

    supplied = build_index(
        _NORTH_NE_EAST,
        ["a", "b", "c"],
        [{"source": "one.md", "chunk_index": 0}, {}, {"source": "two.md"}],
        metric="cosine",
        normalize=True,
    )
    assert supplied.metadata[0] == {"source": "one.md", "chunk_index": 0}
    assert supplied.metadata[2]["source"] == "two.md"


def test_build_index_rejects_wrong_length_metadata():
    with pytest.raises(ValueError) as excinfo:
        build_index(
            _NORTH_NE_EAST,
            ["a", "b", "c"],
            [{"source": "one.md"}],
            metric="cosine",
            normalize=True,
        )

    message = str(excinfo.value)
    assert "3 chunks" in message
    assert "1 metadata" in message


def test_build_index_rejects_empty():
    with pytest.raises(ValueError, match="nothing to index"):
        build_index(torch.zeros(0, 4), [], None, metric="cosine", normalize=True)


def test_build_index_rejects_unknown_metric():
    # "must be one of", not "cosine": the metric being REJECTED could itself
    # be spelled "cosine-ish" and satisfy the loose pattern, and what the
    # message has to carry is the list of the two that work.
    with pytest.raises(ValueError, match="must be one of"):
        build_index(_NORTH_NE_EAST, ["a", "b", "c"], None, metric="euclidean", normalize=True)


def test_build_index_coerces_non_dict_metadata():
    # Metadata is decoration -- Retriever falls back to "?" for a missing
    # source -- so a list that is the right LENGTH but holds the wrong thing
    # (a bare source string, a None from a hand-built graph) loses the label
    # rather than failing a run that has already paid for the embeddings.
    index = build_index(
        _NORTH_NE_EAST,
        ["a", "b", "c"],
        ["one.md", None, {"source": "three.md"}],
        metric="cosine",
        normalize=True,
    )

    assert index.metadata == [{}, {}, {"source": "three.md"}]


def test_build_index_accepts_a_single_vector():
    # A one-chunk corpus arrives as [D], not [1, D], from anything that
    # squeezed on the way -- the index is still an index.
    index = build_index(torch.tensor([3.0, 4.0]), ["only"], None, metric="cosine", normalize=True)

    assert len(index) == 1
    assert index.dim == 2
    assert index.vectors.shape == (1, 2)
    assert torch.allclose(index.vectors, torch.tensor([[0.6, 0.8]]))


def test_save_and_load_round_trip_including_cjk_chunks(tmp_path):
    chunks = ["第一段：什麼是節點？", "plain ascii chunk", "混合 mixed 內容"]
    metadata = [
        {"source": "第一章.md", "chunk_index": 0, "start_char": 0, "end_char": 10},
        {"source": "b.md", "chunk_index": 1},
        {},
    ]
    index = build_index(torch.eye(3), chunks, metadata, metric="cosine", normalize=True)

    path = tmp_path / "corpus.npz"
    index.save(path)

    # The exact name asked for: np.savez appends ".npz" to a path that lacks
    # it, and a save handed "corpus" must not land as "corpus.npz.npz".
    assert path.exists()

    loaded = VectorIndex.load(path)
    assert loaded.chunks == chunks
    assert loaded.metadata == metadata
    assert loaded.metric == "cosine"
    assert loaded.normalized is True
    assert loaded.vectors.dtype == torch.float32
    assert torch.allclose(loaded.vectors, index.vectors)
    assert len(loaded) == 3
    assert loaded.dim == 3

    # The point of persisting it: the same question gets the same answer.
    before = index.search(torch.tensor([1.0, 0.0, 0.0]), 2)
    after = loaded.search(torch.tensor([1.0, 0.0, 0.0]), 2)
    assert torch.equal(before[1], after[1])
    assert torch.allclose(before[0], after[0])


def test_save_and_load_round_trip_keeps_a_dot_index_unnormalised(tmp_path):
    index = build_index(
        torch.tensor([[3.0, 0.0], [0.0, 1.0]]), ["a", "b"], None, metric="dot", normalize=True
    )

    path = tmp_path / "dot.npz"
    index.save(path)
    loaded = VectorIndex.load(path)

    assert loaded.metric == "dot"
    assert loaded.normalized is False
    assert torch.allclose(loaded.vectors, torch.tensor([[3.0, 0.0], [0.0, 1.0]]))


def test_load_rejects_a_file_whose_lists_disagree(tmp_path):
    index = build_index(torch.eye(3), ["a", "b", "c"], None, metric="cosine", normalize=True)
    # Three rows, two texts -- what a truncated write, a hand-edited archive
    # or a file from some other tool looks like from in here. Unchecked it
    # loads happily and raises IndexError deep inside Retriever, which points
    # the learner at the wrong node entirely.
    index.chunks = ["a", "b"]
    path = tmp_path / "short.npz"
    index.save(path)

    with pytest.raises(ValueError) as excinfo:
        VectorIndex.load(path)

    message = str(excinfo.value)
    assert "3 vectors" in message
    assert "2 chunks" in message
    # Only the list that is WRONG is named: the metadata is three entries
    # long and agrees with the vectors, and inviting the reader to check a
    # count that is already right is how a message wastes its one chance.
    assert "metadata" not in message

    # Metadata is checked the same way and for the same reason: the class
    # promises metadata[i] is always safe to index, and load is the one
    # constructor that could hand out an object where it is not.
    thin = build_index(torch.eye(2), ["a", "b"], None, metric="cosine", normalize=True)
    thin.metadata = [{"source": "a.md"}]
    thin_path = tmp_path / "thin.npz"
    thin.save(thin_path)

    with pytest.raises(ValueError) as thin_case:
        VectorIndex.load(thin_path)

    # Singular, because there is one of them.
    assert "1 metadata entry --" in str(thin_case.value)
    assert "chunk" not in str(thin_case.value)

    # Both wrong at once still reads as a sentence.
    both = build_index(torch.eye(3), ["a", "b", "c"], None, metric="cosine",
                       normalize=True)
    both.chunks = ["a"]
    both.metadata = [{}, {}]
    both_path = tmp_path / "both.npz"
    both.save(both_path)

    with pytest.raises(ValueError) as both_case:
        VectorIndex.load(both_path)
    assert "3 vectors but 1 chunk and 2 metadata entries --" in str(
        both_case.value)


def test_load_rejects_an_unknown_metric(tmp_path):
    index = build_index(torch.eye(2), ["a", "b"], None, metric="cosine", normalize=True)
    index.metric = "euclidean"
    path = tmp_path / "odd.npz"
    index.save(path)

    # The same guard build_index applies, at the other door into the class:
    # an unknown metric would otherwise fall through to the dot branch and
    # silently answer a question nobody asked.
    with pytest.raises(ValueError, match="must be one of"):
        VectorIndex.load(path)


def test_two_indexes_compare_by_identity_and_stay_hashable():
    """A dataclass with a tensor field must not generate ``__eq__``.

    The generated one compares the fields as a tuple, and ``tensor ==
    tensor`` is an elementwise tensor whose truth value RAISES -- so a
    caller doing nothing more exotic than "did this port's value change?"
    would get a RuntimeError out of a comparison. Defining ``__eq__`` also
    sets ``__hash__`` to None, which would keep the index out of any dict or
    set that wanted to hold one.
    """
    index = _small_index()
    twin = _small_index()

    # Same contents, different objects: not equal, and no exception on the
    # way to that answer.
    assert index != twin
    assert index == index
    assert bool(index != twin) is True

    # Hashable, so it can be a dict key or live in a set.
    assert hash(index) == hash(index)
    assert len({index, twin, index}) == 2


def test_repr_is_short_and_summarisable():
    index = _small_index()

    assert repr(index) == "VectorIndex(size=3, dim=2, metric=cosine)"
    # The WS summariser truncates at 200 chars; a repr that dumped the matrix
    # would arrive at the Inspector as a sawn-off wall of floats.
    assert len(repr(index)) < 200

    summary = _summarize_single(index)
    assert summary["type"] == "VectorIndex"
    assert summary["repr"].startswith("VectorIndex(")
