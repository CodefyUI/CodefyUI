"""Tests for TextEmbeddingNode.

One node with one job -- texts in, one dense vector each out -- so what is
worth pinning is the seams around it: which of the two inputs wins, how a
string becomes several texts, that the labels never drift out of step with
the rows, and that a stop keeps the rows it already paid for.

Nothing here downloads anything. The ``fake_sentence_transformers`` fixture
from ``conftest`` installs a deterministic stand-in for the library AND
tells the packs bridge the pack is present; both halves are needed, or a
test stops at the gate before it reaches the node's own logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pytest
import torch

from app.core.node_base import DataType
from app.core.packs import PackMissingError, parse_requirement
from app.core.packs.catalog import get_item, get_pack
from app.nodes.llm import _packs_bridge as bridge
from app.nodes.llm._sentence_models import (
    DEFAULT_SENTENCE_MODEL,
    SENTENCE_MODELS,
    SENTENCE_PACK,
)
from app.nodes.llm.text_embedding_node import TextEmbeddingNode

#: Row width of the fake encoder's embeddings (``conftest._FAKE_EMBED_DIM``).
FAKE_DIM = 32

#: A second model id, spelled as the SELECT spells it. Used where a test
#: needs a SECOND encoder instance: the loader's cache is keyed by
#: ``(repo, device)``, so two runs against one model share one object.
BGE = "BAAI/bge-small-zh-v1.5"

#: Every default the node ships, read off the node itself so a test names
#: only the params it actually changes -- and so a changed default cannot
#: quietly stop being the thing under test.
_DEFAULTS = {p.name: p.default for p in TextEmbeddingNode.define_params()}


def _never_stop() -> bool:
    return False


@dataclass
class FakeContext:
    """The three attributes this node reads off an ExecutionContext."""

    device: str = "cpu"
    verbose: bool = False
    should_stop: Callable[[], bool] = _never_stop


def _run(*, inputs=None, context=None, progress_callback=None, **params):
    p = dict(_DEFAULTS)
    p.update(params)
    return TextEmbeddingNode().execute(
        dict(inputs or {}), p, progress_callback, context=context)


@pytest.fixture
def encoders(fake_sentence_transformers, monkeypatch) -> list[Any]:
    """Every fake encoder the node loads, in creation order.

    The instance is what the interesting assertions are about -- which
    strings reached ``encode``, which token cap was set on it, which device
    it was loaded onto -- and the node hands it out to nobody. Recorded by
    subclassing rather than by reading ``_sentence_models._CACHE``, so the
    test does not depend on the loader's private cache staying a dict.
    """
    created: list[Any] = []
    base = fake_sentence_transformers.SentenceTransformer

    class Recording(base):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(
        fake_sentence_transformers, "SentenceTransformer", Recording)
    return created


# -- the node's shape ------------------------------------------------------


def test_node_metadata_and_requires_pack():
    assert TextEmbeddingNode.NODE_NAME == "TextEmbedding"
    assert TextEmbeddingNode.CATEGORY == "LLM"
    # The WHOLE node needs the pack, unlike WordVector where only some of the
    # options do: there is no offline backend here to fall back to.
    assert TextEmbeddingNode.REQUIRES_PACK == SENTENCE_PACK
    assert TextEmbeddingNode.REQUIRES_PACK == "sentence-embeddings"

    inputs = {p.name: p for p in TextEmbeddingNode.define_inputs()}
    assert list(inputs) == ["texts", "text"]
    assert inputs["texts"].data_type is DataType.LIST
    assert inputs["text"].data_type is DataType.STRING
    # Both optional: the node runs on its own param with neither connected.
    assert all(port.optional for port in inputs.values())

    outputs = {p.name: p for p in TextEmbeddingNode.define_outputs()}
    assert list(outputs) == ["embeddings", "labels"]
    assert outputs["embeddings"].data_type is DataType.TENSOR
    assert outputs["labels"].data_type is DataType.LIST

    model = {p.name: p for p in TextEmbeddingNode.define_params()}["model"]
    assert model.default == DEFAULT_SENTENCE_MODEL
    assert model.options == list(SENTENCE_MODELS)

    # The SELECT and its pack map are one statement, so they are read as one:
    # a missing entry offers an option to an install that cannot run it, and
    # an entry naming an item the catalog does not have greys one out for
    # good. Both are silent in the editor, which is why the ids are resolved
    # against the catalog rather than eyeballed.
    option_packs = model.option_packs or {}
    assert set(option_packs) == set(model.options)
    for repo_id, item_id in SENTENCE_MODELS.items():
        assert option_packs[repo_id] == f"{SENTENCE_PACK}:{item_id}"
        pack_id, parsed = parse_requirement(option_packs[repo_id])
        assert parsed == item_id, f"{repo_id} gates on a whole pack"
        # KeyError here means the value names a pack or an item that is not
        # in the catalog -- a typo nothing else would report.
        get_item(get_pack(pack_id), parsed)


# -- what gets embedded ----------------------------------------------------


def test_list_input_gives_one_row_per_text(fake_sentence_transformers):
    res = _run(inputs={"texts": ["one", "two", "three"]})

    assert res["embeddings"].shape == (3, FAKE_DIM)
    assert res["embeddings"].dtype == torch.float32
    assert res["labels"] == ["one", "two", "three"]
    # Gated on THIS model, not on the pack in general: the pack ships four
    # and a learner who downloaded one of the others has not got this one.
    assert (SENTENCE_PACK, SENTENCE_MODELS[DEFAULT_SENTENCE_MODEL]) in \
        fake_sentence_transformers.required
    # The marker is for a stop, and a run that finished must not carry it.
    assert "__interrupted__" not in res

    note = res["__log__"]
    assert note.startswith(
        f"embedded 3 texts with {DEFAULT_SENTENCE_MODEL} (D={FAKE_DIM}) in ")
    assert note.endswith("s on cpu")

    # Anything on a LIST port becomes a string; a number does not silently
    # embed as nothing.
    coerced = _run(inputs={"texts": ["a", 7]})
    assert coerced["labels"] == ["a", "7"]


def test_string_input_splits_lines_by_default_and_not_when_disabled(
        fake_sentence_transformers):
    text = "first line\n\n  second line  \nthird"

    split = _run(inputs={"text": text})
    assert split["labels"] == ["first line", "second line", "third"]
    assert split["embeddings"].shape == (3, FAKE_DIM)

    whole = _run(inputs={"text": text}, split_lines=False)
    assert whole["embeddings"].shape == (1, FAKE_DIM)
    assert whole["labels"] == [text]


def test_param_fallback_when_nothing_connected(fake_sentence_transformers):
    """Neither input connected: the node still runs, on its own param."""
    res = _run()
    assert res["labels"] == ["Machine learning finds patterns in data."]

    # The param goes through the same splitter the input does.
    assert _run(text="alpha\nbeta")["labels"] == ["alpha", "beta"]

    with pytest.raises(ValueError) as caught:
        _run(text="   \n  ")
    assert "nothing to embed" in str(caught.value)


def test_an_empty_texts_list_blames_the_upstream_node():
    """A connected-but-empty list is not "you forgot to connect something".

    No pack fixture: like the both-inputs case this is decided before
    anything is loaded. "connect texts or text" would send a learner to
    check wiring that is already right -- the chunker or filter feeding this
    node is what produced nothing.
    """
    with pytest.raises(ValueError) as caught:
        _run(inputs={"texts": []})

    assert str(caught.value) == (
        "TextEmbedding received an empty texts list - the upstream node "
        "produced no texts.")


def test_both_inputs_connected_is_an_error():
    """Two inputs saying different things is a graph bug, not a merge.

    No pack fixture: this must be decided before anything is loaded, so a
    learner who wired both sees the wiring mistake rather than a download
    prompt for a pack they would not have needed either way.
    """
    with pytest.raises(ValueError) as caught:
        _run(inputs={"texts": ["a"], "text": "b"})

    assert str(caught.value) == (
        "TextEmbedding: connect either texts or text, not both.")


def test_dict_elements_use_their_text_field(fake_sentence_transformers,
                                            encoders):
    """A retrieval chain passes chunks around as dicts, not bare strings.

    Stringifying the whole dict would embed the words "source" and "page"
    into every vector.
    """
    res = _run(inputs={"texts": [
        {"text": "chunk one", "source": "a.md"},
        {"text": "chunk two", "source": "b.md"},
    ]})

    assert res["labels"] == ["chunk one", "chunk two"]
    assert encoders[0].calls == [["chunk one", "chunk two"]]

    with pytest.raises(ValueError) as caught:
        _run(inputs={"texts": [{"page": 3}]})
    assert "'text' key" in str(caught.value)


# -- the vectors and the labels beside them --------------------------------


def test_labels_are_truncated_to_label_chars(fake_sentence_transformers,
                                             encoders):
    long_text = "x" * 200

    res = _run(inputs={"texts": [long_text]}, label_chars=12)

    assert res["labels"] == ["x" * 12]
    # Display only: the model still saw the whole text, so shortening a
    # label cannot change the vector beside it.
    assert encoders[0].calls == [[long_text]]


def test_normalize_gives_unit_rows(fake_sentence_transformers):
    unit = _run(inputs={"texts": ["alpha", "beta"]}, normalize=True)
    norms = torch.norm(unit["embeddings"], dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    raw = _run(inputs={"texts": ["alpha", "beta"]}, normalize=False)
    raw_norms = torch.norm(raw["embeddings"], dim=1)
    assert not torch.allclose(raw_norms, torch.ones_like(raw_norms), atol=1e-3)


def test_prefix_reaches_the_model(fake_sentence_transformers, encoders):
    """multilingual-e5 is trained with ``query: `` / ``passage: ``."""
    res = _run(inputs={"texts": ["what is a vector?"]}, prefix="query: ")

    assert encoders[0].calls == [["query: what is a vector?"]]
    # The label is what the learner typed, not the prompt engineering.
    assert res["labels"] == ["what is a vector?"]


def test_max_seq_length_reaches_the_model(fake_sentence_transformers,
                                          encoders):
    """The cap is the node's to set, and 0 leaves the model's own alone."""
    _run(inputs={"texts": ["alpha"]}, max_seq_length=64)
    assert encoders[0].max_seq_length == 64

    # A different model, because the loader caches by ``(repo, device)`` and
    # the object above already has a cap set on it.
    _run(inputs={"texts": ["alpha"]}, model=BGE, max_seq_length=0)
    assert encoders[1].max_seq_length == 128  # the fake's shipped default


def test_out_of_range_integers_are_clamped_to_the_declared_bounds(
        fake_sentence_transformers, encoders):
    """The editor's ``max_value`` binds a graph that came from elsewhere.

    Nothing the editor produces can be out of range; a hand-edited JSON or
    an exported script can, and ``max_seq_length: 99999`` would otherwise be
    written straight onto the shared encoder.
    """
    _run(inputs={"texts": ["alpha"]}, max_seq_length=99999)
    assert encoders[0].max_seq_length == 8192

    # The floor still holds: a batch of 0 is one text per forward pass.
    _run(inputs={"texts": ["bravo", "charlie"]}, batch_size=0)
    assert encoders[0].calls[-2:] == [["bravo"], ["charlie"]]

    # And the label cap, whose ceiling is display-only but still a ceiling.
    res = _run(inputs={"texts": ["x" * 400]}, label_chars=9999)
    assert res["labels"] == ["x" * 200]


def test_a_non_numeric_integer_param_raises(fake_sentence_transformers):
    """Nothing to clamp is a different thing from out of range."""
    with pytest.raises(ValueError, match="batch_size"):
        _run(inputs={"texts": ["alpha"]}, batch_size="abc")


# -- the pack gate ---------------------------------------------------------


def test_missing_pack_error_names_package_center(monkeypatch):
    """The gate's message reaches the learner unwrapped, suffix and all.

    The editor routes on the ``(pack=<id>)`` suffix to offer the download,
    so a node that re-raised this as its own error would cost the learner
    the button.
    """
    def refuse(pack_id, item_id=None):
        raise PackMissingError(
            pack_id,
            f"Model '{item_id}' from the Sentence embeddings pack is not "
            "downloaded. Open Package Center to download it")

    monkeypatch.setattr(bridge, "require_pack", refuse)

    with pytest.raises(PackMissingError) as caught:
        _run(inputs={"texts": ["hello"]})

    message = str(caught.value)
    assert "Package Center" in message
    assert message.endswith(f"(pack={SENTENCE_PACK})")
    assert caught.value.pack_id == SENTENCE_PACK


# -- progress, stop, trace, device -----------------------------------------


@pytest.mark.parametrize("stop_after", [0, 1],
                         ids=["before-the-first-batch", "after-one-batch"])
def test_progress_frames_and_cooperative_stop(fake_sentence_transformers,
                                              stop_after):
    """A stop keeps the rows already paid for, and the labels naming them.

    The zero-row case is the one worth pinning: an encode stopped before its
    first batch never ran a forward pass to learn the width from, so the
    tensor is ``(0, 0)`` -- and a labels list still holding four names would
    be read downstream against a tensor with none.
    """
    frames: list[dict] = []
    asked = {"count": 0}

    def should_stop():
        asked["count"] += 1
        return asked["count"] > stop_after

    res = _run(
        inputs={"texts": ["one", "two", "three", "four"]},
        batch_size=1,
        context=FakeContext(should_stop=should_stop),
        progress_callback=frames.append,
    )

    assert res["embeddings"].shape == (stop_after, FAKE_DIM if stop_after else 0)
    assert res["labels"] == ["one", "two", "three", "four"][:stop_after]
    assert len(res["labels"]) == res["embeddings"].shape[0]
    assert res["__interrupted__"]["batch"] == stop_after
    assert res["__interrupted__"]["texts"] == stop_after

    # The Log tab is where a learner reads what a run did, and "embedded 0
    # texts" with nothing after it reads as a finished run that found
    # nothing. The batch number is 1-based, like the progress frames'.
    assert res["__log__"].endswith(
        f" (stopped before batch {stop_after + 1})"), res["__log__"]

    # One frame per batch that ran -- ``ProgressThrottle`` always emits the
    # first, and nothing here runs long enough to reach a second.
    assert len(frames) == stop_after
    for frame in frames:
        # Without ``total_batches`` the canvas has a count and no bar.
        assert frame["total_batches"] == 4
        assert frame["batch"] == 1


def test_step_trace_when_verbose(fake_sentence_transformers):
    res = _run(inputs={"texts": ["alpha", "bravo charlie"]},
               context=FakeContext(verbose=True), normalize=True)

    assert [s.name for s in res["__steps__"]] == [
        "input_texts", "encode", "normalize"]
    steps = {s.name: s for s in res["__steps__"]}
    assert steps["input_texts"].scalars["count"] == 2.0
    assert steps["input_texts"].scalars["mean_chars"] == pytest.approx(9.0)
    assert steps["encode"].scalars["dim"] == float(FAKE_DIM)
    assert steps["encode"].tensors["embeddings"].shape == (2, FAKE_DIM)

    # The normalisation happens INSIDE the encode call, so both steps hold
    # the same rows. The step that claims it therefore has to prove it with
    # a number rather than describe a second pass that never runs.
    assert "normalised" in steps["encode"].description
    assert steps["normalize"].scalars["row_length"] == pytest.approx(
        1.0, abs=1e-5)

    # The normalise step describes a thing that happened; with the knob off
    # it must not claim otherwise.
    off = _run(inputs={"texts": ["alpha"]}, normalize=False,
               context=FakeContext(verbose=True))
    assert [s.name for s in off["__steps__"]] == ["input_texts", "encode"]

    quiet = _run(inputs={"texts": ["alpha"]}, context=FakeContext())
    assert "__steps__" not in quiet


def test_output_lands_on_context_device(fake_sentence_transformers, encoders):
    """The rows rejoin the run, and the encode happens where it was told."""
    res = _run(inputs={"texts": ["alpha"]}, context=FakeContext(device="cpu"))

    assert res["embeddings"].device.type == "cpu"
    # ``device="auto"`` follows the context rather than guessing, which is
    # what puts the model and the run's other tensors on one device.
    assert encoders[0].init_kwargs["device"] == "cpu"
