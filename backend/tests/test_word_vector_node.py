"""Tests for WordVectorNode.

Three backends, one node. ``demo-16d`` is a table that ships inline,
``glove-50d`` is a table that comes out of the Package Center's cache, and
the four sentence-transformer repos are an encoder that has no table at all
-- so the tests below are grouped by which of those three a behaviour
belongs to, and the ones that must hold for every backend (option order,
pack requirements) sit at the end.

Nothing here downloads anything. The GloVe tests build a four-word gzip in
``tmp_path`` and point the packs bridge at it; the sentence tests use the
``fake_sentence_transformers`` fixture from ``conftest``.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from app.core.packs import PackMissingError, parse_requirement
from app.core.packs.catalog import get_item, get_pack
from app.nodes.llm import _glove, word_vector_node
from app.nodes.llm import _packs_bridge as bridge
from app.nodes.llm._demo_vectors import DIM as DEMO_DIM
from app.nodes.llm._sentence_models import (
    SENTENCE_MODELS,
    SENTENCE_PACK,
    option_packs_for_models,
)
from app.nodes.llm.word_vector_node import WordVectorNode, _load_backend

#: One of the four sentence backends, spelled as the SELECT spells it.
MINI = "sentence-transformers/all-MiniLM-L6-v2"

#: Row width of the fake encoder's embeddings (``conftest._FAKE_EMBED_DIM``).
FAKE_DIM = 32


def _run(words=None, *, context=None, progress_callback=None, **params):
    p = {
        "backend": "demo-16d",
        "words": "king queen man woman",
        "normalize": False,
        "keep_oov": False,
    }
    p.update(params)
    inputs = {"tokens": words} if words is not None else {}
    return WordVectorNode().execute(
        inputs, p, progress_callback, context=context)


@pytest.fixture(autouse=True)
def _no_table_survives_a_test():
    """Start and end every test with an empty backend cache.

    ``_load_backend`` is process-wide and ``lru_cache``d, so a GloVe table
    read from one test's ``tmp_path`` would otherwise be handed to the next
    test after that directory is gone -- and a test that patched the bridge
    would be answered from a cache filled before the patch.
    """
    _load_backend.cache_clear()
    yield
    _load_backend.cache_clear()


def _write_glove_gz(path: Path, words: list[str]) -> Path:
    """A tiny GloVe table: one line per word, ``GLOVE_DIM`` numbers each.

    Every value says which cell it came from, so a transposed or repeated
    row cannot pass unnoticed.
    """
    lines = [
        f"{word} " + " ".join(f"{row + column / 100:.4f}"
                              for column in range(_glove.GLOVE_DIM))
        for row, word in enumerate(words)
    ]
    path.write_bytes(gzip.compress(("\n".join(lines) + "\n").encode("utf-8")))
    return path


@pytest.fixture
def glove_pack(tmp_path, monkeypatch) -> Path:
    """Pretend the word-vectors pack is downloaded, with a four-word table."""
    gz = _write_glove_gz(tmp_path / _glove.GLOVE_50D_ASSET,
                         ["king", "queen", "man", "woman"])
    monkeypatch.setattr(bridge, "asset_path", lambda pack_id, filename: gz)
    return gz


# -- the node itself -------------------------------------------------------


def test_node_metadata():
    assert WordVectorNode.NODE_NAME == "WordVector"
    assert WordVectorNode.CATEGORY == "LLM"
    assert [p.name for p in WordVectorNode.define_outputs()] == ["embeddings", "labels"]


# -- demo-16d (the table that ships inline) --------------------------------


def test_demo_backend_returns_correct_shape():
    res = _run(["king", "queen", "man", "woman"])
    assert res["embeddings"].shape == (4, DEMO_DIM)
    assert res["labels"] == ["king", "queen", "man", "woman"]
    assert res["embeddings"].dtype == torch.float32


def test_oov_words_are_dropped_by_default():
    res = _run(["king", "asdfqwerty", "queen"])
    assert res["labels"] == ["king", "queen"]
    assert res["embeddings"].shape == (2, DEMO_DIM)


def test_keep_oov_emits_zero_rows():
    res = _run(["king", "asdfqwerty", "queen"], keep_oov=True)
    assert res["labels"] == ["king", "asdfqwerty", "queen"]
    assert res["embeddings"].shape == (3, DEMO_DIM)
    assert torch.all(res["embeddings"][1] == 0)


def test_words_param_used_when_no_input_connected():
    res = WordVectorNode().execute(
        {},
        {
            "backend": "demo-16d",
            "words": "cat, dog, fish",
            "normalize": False,
            "keep_oov": False,
        },
    )
    assert res["labels"] == ["cat", "dog", "fish"]


def test_case_insensitive_lookup():
    res = _run(["KING", "Queen", "MAN"])
    assert res["labels"] == ["king", "queen", "man"]


def test_empty_input_returns_empty_tensor():
    res = _run([])
    assert res["embeddings"].shape == (0, DEMO_DIM)
    assert res["labels"] == []


def test_king_minus_man_plus_woman_is_close_to_queen():
    """Canonical demo: in the demo-16d basis, the analogy holds exactly."""
    res = _run(["king", "queen", "man", "woman"])
    e = res["embeddings"]
    king, queen, man, woman = e[0], e[1], e[2], e[3]
    analogy = king - man + woman
    diff = torch.norm(analogy - queen).item()
    # Vectors are sparse hand-built tuples; the analogy is exact (zero diff).
    assert diff < 1e-5


def test_normalize_makes_unit_rows():
    res = _run(["king", "queen", "cat"], normalize=True)
    norms = torch.norm(res["embeddings"], dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_normalize_leaves_zero_rows_alone():
    res = _run(["king", "asdfqwerty"], keep_oov=True, normalize=True)
    # Row 0 (king) is unit; row 1 (zero vector) stays zero.
    norms = torch.norm(res["embeddings"], dim=1)
    assert abs(norms[0].item() - 1.0) < 1e-6
    assert norms[1].item() == 0.0


# -- glove-50d (the table the Package Center downloads) --------------------


def test_glove_backend_without_pack_names_package_center(monkeypatch):
    """The one thing a learner without the pack must be told: where to get it.

    Raised, never wrapped: the frontend routes on the ``(pack=<id>)`` suffix
    to offer the download, and a node that re-raised this as its own error
    would cost the learner the button.
    """
    monkeypatch.setattr(bridge, "asset_path", lambda pack_id, filename: None)

    with pytest.raises(PackMissingError) as caught:
        _run(["king"], backend="glove-50d")

    message = str(caught.value)
    assert "Package Center" in message
    assert "word-vectors" in message
    assert message.endswith("(pack=word-vectors)")
    assert caught.value.pack_id == "word-vectors"


def test_glove_backend_reads_the_converted_table(glove_pack):
    res = _run(["king", "queen", "man", "woman"], backend="glove-50d")

    assert res["labels"] == ["king", "queen", "man", "woman"]
    assert res["embeddings"].shape == (4, _glove.GLOVE_DIM)
    assert res["embeddings"].dtype == torch.float32
    # The npz the loader converted on the way sits beside the download.
    assert _glove.npz_path_for(glove_pack).is_file()


def test_glove_lookup_is_case_insensitive_and_drops_oov(glove_pack):
    """The table path is one path: what demo-16d does with a word it does
    not have, glove-50d does too."""
    res = _run(["KING", "asdfqwerty", "Woman"], backend="glove-50d")

    assert res["labels"] == ["king", "woman"]
    assert res["embeddings"].shape == (2, _glove.GLOVE_DIM)

    kept = _run(["KING", "asdfqwerty"], backend="glove-50d", keep_oov=True)
    assert kept["labels"] == ["king", "asdfqwerty"]
    assert torch.all(kept["embeddings"][1] == 0)


# -- names saved graphs may still carry ------------------------------------


@pytest.mark.parametrize("retired, replacement", [
    ("glove-100d", "'glove-50d'"),
    ("minilm-sentence-384d", f"'{MINI}'"),
])
def test_retired_backend_names_point_at_the_replacement(retired, replacement):
    """A graph saved against a preview build names a backend that is gone.

    ``ValueError`` rather than ``PackMissingError``: nothing a learner can
    install fixes a name that no longer exists, so the message has to name
    the option they should pick instead.
    """
    with pytest.raises(ValueError) as caught:
        _run(["king"], backend=retired)

    message = str(caught.value)
    assert retired in message
    assert replacement in message


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown WordVector backend"):
        _run(["king"], backend="not-a-real-backend")


# -- the sentence encoders -------------------------------------------------


def test_sentence_backend_embeds_every_word_without_oov(
        fake_sentence_transformers):
    """An encoder has no vocabulary, so nothing can fall out of it.

    ``asdfqwerty`` is not a word in any table here; the sentence backend
    still returns a row for it, and the labels still line up with the rows.
    """
    res = _run(["king", "asdfqwerty", "queen"], backend=MINI)

    assert res["labels"] == ["king", "asdfqwerty", "queen"]
    assert res["embeddings"].shape == (3, FAKE_DIM)
    assert res["embeddings"].dtype == torch.float32
    # Gated on THIS model, not on the pack in general: the pack ships four
    # and a learner who downloaded one of the others has not got this one.
    assert (SENTENCE_PACK, SENTENCE_MODELS[MINI]) in \
        fake_sentence_transformers.required

    unit = _run(["king", "queen"], backend=MINI, normalize=True)
    norms = torch.norm(unit["embeddings"], dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_sentence_backend_ignores_keep_oov(fake_sentence_transformers):
    """``keep_oov`` is a table backend's knob and a no-op here."""
    dropped = _run(["king", "asdfqwerty"], backend=MINI, keep_oov=False)
    kept = _run(["king", "asdfqwerty"], backend=MINI, keep_oov=True)

    assert dropped["labels"] == kept["labels"] == ["king", "asdfqwerty"]
    assert torch.equal(dropped["embeddings"], kept["embeddings"])
    # And no zero row was substituted for the word no table would have.
    assert torch.any(kept["embeddings"][1] != 0)


def test_sentence_backend_without_pack_is_friendly(monkeypatch):
    """The gate's message reaches the learner unwrapped, suffix and all."""
    def refuse(pack_id, item_id=None):
        raise PackMissingError(
            pack_id,
            f"Model '{item_id}' from the Sentence embeddings pack is not "
            "downloaded. Open Package Center to download it")

    monkeypatch.setattr(bridge, "require_pack", refuse)

    with pytest.raises(PackMissingError) as caught:
        _run(["king"], backend=MINI)

    message = str(caught.value)
    assert "Package Center" in message
    assert message.endswith(f"(pack={SENTENCE_PACK})")


@pytest.mark.parametrize("stop_after", [0, 1],
                         ids=["before-the-first-batch", "after-one-batch"])
def test_sentence_backend_honours_stop(fake_sentence_transformers, monkeypatch,
                                       stop_after):
    """Stop keeps the rows already paid for, and the labels that name them.

    The zero-row case is the one worth pinning: an encode that stopped
    before its first batch has no width to report, so the tensor is
    ``(0, 0)`` -- and a labels list that still held four names would be read
    downstream against a tensor with none.
    """
    monkeypatch.setattr(word_vector_node, "SENTENCE_BATCH_SIZE", 2)
    asked = {"count": 0}

    def should_stop():
        asked["count"] += 1
        return asked["count"] > stop_after

    res = _run(["king", "queen", "man", "woman"], backend=MINI,
               context=SimpleNamespace(should_stop=should_stop))

    rows = stop_after * 2
    assert res["embeddings"].shape[0] == rows
    assert res["labels"] == ["king", "queen"][:rows]
    assert len(res["labels"]) == res["embeddings"].shape[0]
    assert "__interrupted__" in res
    assert res["__interrupted__"]["batch"] == stop_after


def test_sentence_backend_finishing_is_not_interrupted(
        fake_sentence_transformers):
    """The marker is for a stop, and an unstopped run must not carry it."""
    res = _run(["king", "queen"], backend=MINI,
               context=SimpleNamespace(should_stop=lambda: False))

    assert "__interrupted__" not in res
    assert res["embeddings"].shape == (2, FAKE_DIM)


# -- every backend ---------------------------------------------------------


def _backend_param():
    return {p.name: p for p in WordVectorNode.define_params()}["backend"]


def test_backend_options_and_option_packs_agree():
    """The SELECT and its pack map are one statement, so they are read as one.

    An option with no entry is offered to everyone; an entry naming an item
    the catalog does not have would grey an option out for good. Both are
    silent failures in the editor, which is why the ids are resolved against
    the catalog here rather than eyeballed.
    """
    param = _backend_param()

    assert param.options == ["demo-16d", "glove-50d", *SENTENCE_MODELS]
    assert param.default == "demo-16d"
    # The default works offline, so the NODE needs no pack; only its options do.
    assert WordVectorNode.REQUIRES_PACK is None

    option_packs = param.option_packs or {}
    assert "demo-16d" not in option_packs
    assert set(option_packs) == set(param.options) - {"demo-16d"}
    assert option_packs["glove-50d"] == "word-vectors:glove-50d"
    assert all(option_packs[repo_id] == requirement
               for repo_id, requirement in option_packs_for_models().items())

    for option, requirement in option_packs.items():
        pack_id, item_id = parse_requirement(requirement)
        assert item_id is not None, f"{option} gates on a whole pack"
        # KeyError here means the value names a pack or an item that is not
        # in the catalog -- a typo nothing else would report.
        get_item(get_pack(pack_id), item_id)
