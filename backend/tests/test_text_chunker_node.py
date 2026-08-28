"""Tests for TextChunkerNode -- the second node of the RAG chain.

No pack, no network, no filesystem: chunking is pure string arithmetic, so
every test states a document literally and asserts on exact chunk texts and
exact offsets rather than on lengths alone.

The invariant nearly every test re-checks is ``text[start:end] == chunk``.
It is the whole reason the node reports offsets at all -- a citation that
says "notes/one.md, characters 400-800" is only worth printing if those
numbers still select the passage after the chunk was stripped, packed with
its neighbours, or merged with a short tail. Each of those three steps is a
separate opportunity to shift an offset by the width of a space, and none of
them would show up in the chunk texts themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.core.node_base import DataType
from app.nodes.llm.text_chunker_node import TextChunkerNode, _character_spans

#: Every default the node ships, read off the node itself so a test names
#: only the params it actually changes.
_DEFAULTS = {p.name: p.default for p in TextChunkerNode.define_params()}


@dataclass
class FakeContext:
    """The one attribute this node reads off an ExecutionContext."""

    verbose: bool = False


def _run(*, inputs=None, context=None, **params) -> dict:
    p = dict(_DEFAULTS)
    p.update(params)
    return TextChunkerNode().execute(dict(inputs or {}), p, context=context)


def _assert_offsets_select_the_chunks(text: str, result: dict) -> None:
    """The invariant, for a run over a single document."""
    for chunk, meta in zip(result["chunks"], result["metadata"]):
        assert text[meta["start_char"]:meta["end_char"]] == chunk, meta
        # Offsets that select the chunk are not enough on their own: a chunk
        # that still carries its whitespace edges makes every downstream
        # citation start on a blank line.
        assert chunk == chunk.strip(), repr(chunk)


# -- metadata ------------------------------------------------------------


def test_node_metadata():
    assert TextChunkerNode.NODE_NAME == "TextChunker"
    assert TextChunkerNode.CATEGORY == "LLM"
    # Pure string work: the chunker is the one node between DocumentLoader
    # and TextEmbedding that needs nothing downloaded.
    assert TextChunkerNode.REQUIRES_PACK is None
    assert TextChunkerNode.cacheable is True

    inputs = {p.name: p for p in TextChunkerNode.define_inputs()}
    assert list(inputs) == ["documents", "text"]
    assert inputs["documents"].data_type == DataType.LIST
    assert inputs["text"].data_type == DataType.STRING
    # Both optional, because either one alone is a complete input.
    assert all(port.optional for port in inputs.values())

    outputs = {p.name: p for p in TextChunkerNode.define_outputs()}
    assert list(outputs) == ["chunks", "metadata", "count"]
    assert outputs["chunks"].data_type == DataType.LIST
    assert outputs["metadata"].data_type == DataType.LIST
    assert outputs["count"].data_type == DataType.SCALAR

    params = {p.name: p for p in TextChunkerNode.define_params()}
    assert list(params) == [
        "strategy", "chunk_size", "chunk_overlap", "min_chunk_chars"]
    assert params["strategy"].default == "characters"
    assert params["strategy"].options == [
        "characters", "sentences", "paragraphs"]
    assert params["chunk_size"].default == 400
    assert params["chunk_size"].min_value == 20
    assert params["chunk_size"].max_value == 20000
    assert params["chunk_overlap"].default == 80
    assert params["chunk_overlap"].min_value == 0
    assert params["chunk_overlap"].max_value == 5000
    # Overlap is a character-window idea; the packing strategies have no
    # room for it, so the editor hides it rather than showing a knob that
    # does nothing.
    assert params["chunk_overlap"].visible_when == {"strategy": "characters"}
    assert params["min_chunk_chars"].default == 40
    assert params["min_chunk_chars"].min_value == 1
    assert params["min_chunk_chars"].advanced is True


# -- characters ----------------------------------------------------------


def test_characters_windows_with_overlap_and_offsets():
    text = "".join(str(i % 10) for i in range(250))

    result = _run(
        inputs={"text": text},
        strategy="characters", chunk_size=100, chunk_overlap=20,
        min_chunk_chars=1)

    # Step is chunk_size - chunk_overlap = 80, and the walk stops as soon as
    # a window reaches the end: a fourth window at 240 would be wholly
    # contained in the third and would embed the same passage twice.
    assert [m["start_char"] for m in result["metadata"]] == [0, 80, 160]
    assert [m["end_char"] for m in result["metadata"]] == [100, 180, 250]
    assert [len(c) for c in result["chunks"]] == [100, 100, 90]
    assert result["count"] == 3
    _assert_offsets_select_the_chunks(text, result)

    # What the overlap is FOR: the last 20 characters of one chunk are the
    # first 20 of the next, so a sentence cut by the first boundary survives
    # whole in the second chunk.
    chunks = result["chunks"]
    assert chunks[0][-20:] == chunks[1][:20]
    assert chunks[1][-20:] == chunks[2][:20]


def test_a_window_that_strips_to_its_predecessor_is_dropped():
    """One passage, one chunk -- however much whitespace surrounds it.

    Two characters in the middle of seventy blanks: with a 50-character
    window and a 25-character step, windows 0-50 and 25-72 both strip down
    to the same ``xy``. Emitted twice it is embedded twice, stored twice and
    returned twice by one search, which reads to the learner as the corpus
    saying the same thing in two places.
    """
    text = " " * 30 + "xy" + " " * 40

    # The unit that produces the duplicate, checked directly: at the node's
    # default min_chunk_chars the tail merge folds the second copy back into
    # the first, which would make an end-to-end assertion pass either way.
    assert _character_spans(text, 0, len(text), size=50, step=25) == [(30, 32)]

    result = _run(inputs={"text": text}, strategy="characters",
                  chunk_size=50, chunk_overlap=25, min_chunk_chars=1)

    assert result["chunks"] == ["xy"]
    assert result["count"] == 1
    assert [m["start_char"] for m in result["metadata"]] == [30]
    _assert_offsets_select_the_chunks(text, result)

    # Not a blanket de-duplicator: the same TEXT at two different places is
    # two chunks, because the offsets a citation prints differ.
    twice = "ab" + " " * 60 + "ab"
    assert _character_spans(twice, 0, len(twice), size=50, step=50) == [
        (0, 2), (62, 64)]


def test_overlap_must_be_smaller_than_chunk_size():
    """Equal is already broken: the window never advances."""
    with pytest.raises(ValueError) as excinfo:
        _run(inputs={"text": "a" * 500},
             strategy="characters", chunk_size=100, chunk_overlap=100)

    message = str(excinfo.value)
    assert "chunk_overlap" in message
    assert "chunk_size" in message

    # The param is hidden for the packing strategies, and a hidden param is
    # still PRESENT -- the canvas wrote the default onto the node and
    # switching strategy does not clear it. So a leftover overlap must not
    # fail a run that never looks at it.
    packed = _run(inputs={"text": "One. Two. Three."},
                  strategy="sentences", chunk_size=100, chunk_overlap=100,
                  min_chunk_chars=1)
    assert packed["count"] >= 1


def test_short_tail_is_merged():
    text = "".join(str(i % 10) for i in range(105))

    merged = _run(
        inputs={"text": text},
        strategy="characters", chunk_size=50, chunk_overlap=0,
        min_chunk_chars=20)

    # The third window is five characters long. On its own it embeds to a
    # vector with almost no meaning in it, so it joins the chunk before it
    # -- which is allowed to exceed chunk_size, and does.
    assert merged["chunks"] == [text[0:50], text[50:105]]
    assert [m["start_char"] for m in merged["metadata"]] == [0, 50]
    assert [m["end_char"] for m in merged["metadata"]] == [50, 105]
    assert [m["chunk_index"] for m in merged["metadata"]] == [0, 1]
    _assert_offsets_select_the_chunks(text, merged)

    # The control: the same windows, with the merge threshold turned down.
    kept = _run(
        inputs={"text": text},
        strategy="characters", chunk_size=50, chunk_overlap=0,
        min_chunk_chars=1)
    assert kept["chunks"] == [text[0:50], text[50:100], text[100:105]]


# -- sentences -----------------------------------------------------------


def test_sentences_strategy_keeps_boundaries_for_cjk_punctuation():
    # Three sentences, no spaces anywhere -- which is exactly why a
    # whitespace-based splitter is useless for Chinese and this one splits
    # on the full-width punctuation instead. 16, 17 and 11 characters.
    text = (
        "檢索是以塊為單位而不是整份檔案。"
        "一個問題應該撈回能回答它的那一段！"
        "而不是整份五頁的文件。"
    )

    one_each = _run(
        inputs={"text": text},
        strategy="sentences", chunk_size=20, min_chunk_chars=1)

    assert one_each["chunks"] == [
        "檢索是以塊為單位而不是整份檔案。",
        "一個問題應該撈回能回答它的那一段！",
        "而不是整份五頁的文件。",
    ]
    # Every chunk ends on a real sentence boundary and none begins with the
    # punctuation that closed the one before it.
    for chunk in one_each["chunks"]:
        assert chunk[-1] in "。！"
        assert chunk[0] not in "。！"
    _assert_offsets_select_the_chunks(text, one_each)

    # Sentences are PACKED, not emitted one per chunk: with room for two,
    # two travel together.
    packed = _run(
        inputs={"text": text},
        strategy="sentences", chunk_size=35, min_chunk_chars=1)

    assert packed["chunks"] == [
        "檢索是以塊為單位而不是整份檔案。一個問題應該撈回能回答它的那一段！",
        "而不是整份五頁的文件。",
    ]
    _assert_offsets_select_the_chunks(text, packed)


def test_paragraphs_strategy_packs_paragraphs():
    text = (
        "Alpha paragraph.\n\n"
        "Bravo paragraph.\n\n"
        "Charlie paragraph is the long one here."
    )

    result = _run(
        inputs={"text": text},
        strategy="paragraphs", chunk_size=40, min_chunk_chars=1)

    # Two paragraphs fit under 40 characters together; the third does not
    # join them. The blank line between the packed pair survives inside the
    # chunk, because the chunk IS the slice of the original document.
    assert result["chunks"] == [
        "Alpha paragraph.\n\nBravo paragraph.",
        "Charlie paragraph is the long one here.",
    ]
    assert [m["start_char"] for m in result["metadata"]] == [0, 36]
    _assert_offsets_select_the_chunks(text, result)


def test_oversized_sentence_falls_back_to_character_windows():
    """One runaway sentence must not produce one runaway chunk.

    chunk_size is a cap for every strategy, so a sentence longer than the
    cap is windowed like the characters strategy would do it -- and without
    overlap, since the neighbours it would share text with are windows of
    the same sentence.
    """
    text = "Short one. " + "y" * 250 + ". Tail."

    result = _run(
        inputs={"text": text},
        strategy="sentences", chunk_size=100, min_chunk_chars=1)

    chunks = result["chunks"]
    assert chunks[0] == "Short one."
    assert chunks[-1] == "Tail."
    assert [len(c) for c in chunks] == [10, 100, 100, 51, 5]
    assert all(len(c) <= 100 for c in chunks)
    # The three windows tile the long sentence exactly once: no overlap, no
    # gap, nothing dropped.
    assert "".join(chunks[1:4]) == "y" * 250 + "."
    _assert_offsets_select_the_chunks(text, result)


# -- metadata and sources ------------------------------------------------


def test_metadata_carries_source_and_index():
    documents = [
        {"text": "A" * 120, "source": "notes/one.md"},
        {"text": "B" * 60, "source": "two.txt"},
    ]

    result = _run(
        inputs={"documents": documents},
        strategy="characters", chunk_size=50, chunk_overlap=0,
        min_chunk_chars=1)

    assert result["count"] == 5
    assert [m["source"] for m in result["metadata"]] == [
        "notes/one.md", "notes/one.md", "notes/one.md", "two.txt", "two.txt"]
    # chunk_index counts WITHIN its document, so (source, chunk_index) is an
    # address a citation can print; the position in the flat list is already
    # the list index and would say nothing new.
    assert [m["chunk_index"] for m in result["metadata"]] == [0, 1, 2, 0, 1]
    # Offsets are into the document they came from, not into a concatenation
    # of all of them.
    assert [m["start_char"] for m in result["metadata"]] == [0, 50, 100, 0, 50]
    assert [m["end_char"] for m in result["metadata"]] == [50, 100, 120, 50, 60]
    assert all(
        set(m) == {"source", "chunk_index", "start_char", "end_char"}
        for m in result["metadata"])

    for doc in documents:
        rows = [(c, m) for c, m in zip(result["chunks"], result["metadata"])
                if m["source"] == doc["source"]]
        for chunk, meta in rows:
            assert doc["text"][meta["start_char"]:meta["end_char"]] == chunk


def test_plain_string_documents_and_text_input():
    """Neither input requires the dict shape DocumentLoader produces."""
    plain = _run(
        inputs={"documents": ["first document", "second document"]},
        strategy="characters", chunk_size=100, chunk_overlap=0,
        min_chunk_chars=1)

    assert plain["chunks"] == ["first document", "second document"]
    # A bare string carries no source, so it is named by its position in the
    # list -- 0-based, like chunk_index.
    assert [m["source"] for m in plain["metadata"]] == ["doc-0", "doc-1"]

    single = _run(
        inputs={"text": "a single typed document"},
        strategy="characters", chunk_size=100, chunk_overlap=0,
        min_chunk_chars=1)

    assert single["chunks"] == ["a single typed document"]
    assert [m["source"] for m in single["metadata"]] == ["text"]

    # Wired together the two inputs ADD UP rather than one winning: chunking
    # is per-document and every chunk names its own source, so there is
    # nothing ambiguous about a corpus plus one extra note.
    both = _run(
        inputs={"documents": ["from the list"], "text": "typed in the box"},
        strategy="characters", chunk_size=100, chunk_overlap=0,
        min_chunk_chars=1)

    assert both["chunks"] == ["from the list", "typed in the box"]
    assert [m["source"] for m in both["metadata"]] == ["doc-0", "text"]


# -- whitespace ----------------------------------------------------------


def test_offsets_survive_stripping():
    """Windows land mid-whitespace; the offsets have to move with the trim."""
    text = "  " + "a" * 40 + "     " + "b" * 40 + "  "

    result = _run(
        inputs={"text": text},
        strategy="characters", chunk_size=45, chunk_overlap=0,
        min_chunk_chars=1)

    assert result["chunks"] == ["a" * 40, "b" * 40]
    # Not [0, 45]: both windows opened on whitespace, so both starts moved.
    assert [m["start_char"] for m in result["metadata"]] == [2, 47]
    assert [m["end_char"] for m in result["metadata"]] == [42, 87]
    _assert_offsets_select_the_chunks(text, result)

    # A window with nothing but whitespace in it is dropped, not emitted as
    # an empty chunk: an empty chunk embeds to a vector near nothing in
    # particular and then turns up in retrieval with no text under it.
    gap = "x" * 10 + " " * 30 + "y" * 10
    holes = _run(
        inputs={"text": gap},
        strategy="characters", chunk_size=20, chunk_overlap=0,
        min_chunk_chars=1)

    assert holes["chunks"] == ["x" * 10, "y" * 10]
    assert [m["start_char"] for m in holes["metadata"]] == [0, 40]
    _assert_offsets_select_the_chunks(gap, holes)


# -- trace and errors ----------------------------------------------------


def test_step_trace_when_verbose():
    documents = [{"text": "A" * 120, "source": "one.md"},
                 {"text": "B" * 80, "source": "two.md"}]

    result = _run(
        inputs={"documents": documents},
        context=FakeContext(verbose=True),
        strategy="characters", chunk_size=50, chunk_overlap=0,
        min_chunk_chars=1)

    assert [s.name for s in result["__steps__"]] == ["input", "chunk"]
    steps = {s.name: s for s in result["__steps__"]}
    assert steps["input"].scalars["documents"] == 2.0
    assert steps["input"].scalars["total_chars"] == 200.0
    # 120 -> 50/50/20 and 80 -> 50/30: five chunks, 200 characters.
    assert steps["chunk"].scalars["count"] == 5.0
    assert steps["chunk"].scalars["mean_chars"] == pytest.approx(40.0)

    quiet = _run(inputs={"documents": documents})
    assert "__steps__" not in quiet


def test_neither_input_is_an_error():
    with pytest.raises(ValueError) as excinfo:
        _run(inputs={})

    assert str(excinfo.value) == "TextChunker needs documents or text."

    # A CONNECTED but empty list is a different thing: the wiring is right
    # and the upstream node produced nothing, which is not this error.
    empty = _run(inputs={"documents": []})
    assert empty["chunks"] == []
    assert empty["metadata"] == []
    assert empty["count"] == 0


def test_a_dict_without_text_says_which_keys_it_had():
    """The likeliest hand-built document: the right shape, the wrong key.

    ``{"content": ...}`` and ``{"body": ...}`` are what a PythonScript node
    or an exported script produces, and ``str(item)`` of a dict would chunk
    the literal ``{'content': 'hello'}`` -- a run that succeeds and embeds
    punctuation.
    """
    with pytest.raises(ValueError) as excinfo:
        _run(inputs={"documents": [{"content": "hello", "source": "a.md"}]})

    assert str(excinfo.value) == (
        "TextChunker: a dict on the `documents` input must carry a 'text' "
        "key holding the document; this one has ['content', 'source'].")


def test_a_bare_string_on_the_documents_port_names_the_other_port():
    """``list("hello")`` is five one-character documents.

    Every layer below would accept that: five chunks come back, the offsets
    are right, nothing raises -- and the learner is looking at a corpus of
    single letters wondering why retrieval is broken.
    """
    with pytest.raises(ValueError) as excinfo:
        _run(inputs={"documents": "one long document"})

    assert str(excinfo.value) == (
        "TextChunker: the `documents` input carries a single string, not a "
        "list of documents. Wire it into the `text` input instead.")


def test_unknown_strategy_lists_the_three():
    """The strategy decides what the chunks ARE, so it is refused rather
    than defaulted -- and the message names the three that work."""
    with pytest.raises(ValueError) as excinfo:
        _run(inputs={"text": "a paragraph"}, strategy="semantic")

    assert str(excinfo.value) == (
        "TextChunker: unknown strategy 'semantic'; set the `strategy` param "
        "to one of ['characters', 'sentences', 'paragraphs'].")


def test_log_counts_chunks_and_documents_and_says_the_mean():
    """The one line the canvas Log tab shows, including its singulars.

    "37 chunks from 5 documents (mean 312 chars)" is what a learner reads to
    decide whether chunk_size is doing what they think, so both counts and
    the mean are pinned -- and so is the grammar, because "1 chunks from 1
    documents" is the kind of thing that ships.
    """
    text = "".join(str(i % 10) for i in range(250))

    many = _run(inputs={"text": text}, strategy="characters",
                chunk_size=100, chunk_overlap=20, min_chunk_chars=1)
    # Three windows of 100, 100 and 90 characters: mean 96.67, rounded.
    assert many["__log__"] == "3 chunks from 1 document (mean 97 chars)"

    one = _run(inputs={"text": "a short note"})
    assert one["__log__"] == "1 chunk from 1 document (mean 12 chars)"

    # 10 characters and 12: the mean is over CHUNKS, not documents.
    pair = _run(inputs={"documents": ["first note", "second draft"]})
    assert pair["__log__"] == "2 chunks from 2 documents (mean 11 chars)"
