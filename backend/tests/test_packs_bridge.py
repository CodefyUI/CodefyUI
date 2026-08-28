"""The one seam LLM nodes reach ``app.core.packs`` through, plus the offline
stand-in for sentence-transformers that every later node test runs against.

Two things are being pinned here.

The BRIDGE exists so a node module never imports ``app.core.packs`` at import
time. Two consequences are tested: a node module still imports (and reports
"nothing is available") in an install where the packs package is missing
entirely, and in a normal install every call lands on the real function with
its arguments unchanged -- a bridge that quietly dropped ``item_id`` would
gate a SELECT option on "any of the four models" instead of the one chosen.

The FAKE lets the sentence-embedding nodes be tested with no download, no
torch and no network. Its contract is what those tests will lean on:
deterministic rows, unit rows when asked to normalise, and texts that share
words or characters coming out closer than texts that share neither.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from app.core import packs
from app.nodes.llm import _packs_bridge as bridge


# ── the bridge ───────────────────────────────────────────────────────────


def test_bridge_reports_nothing_available_when_packs_package_is_missing(
        monkeypatch):
    """A stripped install must still be able to IMPORT and RUN a node module.

    ``None`` in ``sys.modules`` is exactly what a failed import leaves behind,
    and it makes the bridge's lazy ``from ...core.packs import ...`` raise
    ``ModuleNotFoundError`` the way a missing package would.

    The real functions are replaced with ones that answer YES first, so this
    cannot pass merely because the machine running it has no packs installed
    -- every assertion below is False/None only if the import never happened.
    """
    monkeypatch.setattr(packs, "pack_available", lambda *args: True)
    monkeypatch.setattr(packs, "require_pack", lambda *args: None)
    monkeypatch.setattr(packs, "model_dir", lambda *args: Path("models"))
    monkeypatch.setattr(packs, "asset_path", lambda *args: Path("assets"))
    monkeypatch.setitem(sys.modules, "app.core.packs", None)

    assert bridge.pack_available("rag") is False
    assert bridge.pack_available("sentence-embeddings", "bge-small-zh-v1.5") is False
    assert bridge.model_dir("Qwen/Qwen2.5-0.5B-Instruct") is None
    assert bridge.asset_path("word-vectors", "glove-wiki-gigaword-50.gz") is None

    # Same shape as PackMissingError, because the message is what the editor
    # reads the pack id back off: a bare RuntimeError here would show the
    # learner a stack trace with no install button.
    with pytest.raises(bridge.PacksUnavailableError) as caught:
        bridge.require_pack("rag")

    assert isinstance(caught.value, RuntimeError)
    assert str(caught.value).endswith("(pack=rag)")
    assert caught.value.pack_id == "rag"


def test_bridge_delegates_to_packs(monkeypatch):
    """Every argument reaches the real function untouched, ``item_id``
    included -- the bridge adds no policy of its own."""
    calls: list[tuple[str, tuple, dict]] = []

    def recorder(name: str, result):
        def record(*args, **kwargs):
            calls.append((name, args, kwargs))
            return result
        return record

    monkeypatch.setattr(packs, "pack_available", recorder("pack_available", True))
    monkeypatch.setattr(packs, "require_pack", recorder("require_pack", None))
    monkeypatch.setattr(packs, "model_dir", recorder("model_dir", Path("models")))
    monkeypatch.setattr(packs, "asset_path", recorder("asset_path", Path("assets")))

    assert bridge.pack_available("rag") is True
    assert bridge.pack_available("sentence-embeddings", "bge-small-zh-v1.5") is True
    assert bridge.require_pack("sentence-embeddings", "multilingual-e5-small") is None
    assert bridge.model_dir("BAAI/bge-small-zh-v1.5") == Path("models")
    assert bridge.asset_path("word-vectors", "glove-wiki-gigaword-50.gz") == Path("assets")

    assert calls == [
        ("pack_available", ("rag", None), {}),
        ("pack_available", ("sentence-embeddings", "bge-small-zh-v1.5"), {}),
        ("require_pack", ("sentence-embeddings", "multilingual-e5-small"), {}),
        ("model_dir", ("BAAI/bge-small-zh-v1.5",), {}),
        ("asset_path", ("word-vectors", "glove-wiki-gigaword-50.gz"), {}),
    ]


def test_requirement_formats_pack_and_item():
    """``requirement`` writes what ``parse_requirement`` reads: a node author
    should never have to hand-format an ``option_packs`` value."""
    assert bridge.requirement("a") == "a"
    assert bridge.requirement("a", "b") == "a:b"

    assert packs.parse_requirement(bridge.requirement("word-vectors")) == (
        "word-vectors", None)
    assert packs.parse_requirement(
        bridge.requirement("sentence-embeddings", "bge-small-zh-v1.5")) == (
        "sentence-embeddings", "bge-small-zh-v1.5")


# ── the offline sentence-transformers fake ───────────────────────────────


def test_fake_sentence_transformers_fixture_pretends_the_pack_is_installed(
        fake_sentence_transformers, tmp_path):
    """The fixture answers for the pack as well as for the library: a node
    test should not have to write a sentinel to get past the gate."""
    assert sys.modules["sentence_transformers"] is fake_sentence_transformers

    assert bridge.pack_available("sentence-embeddings") is True
    assert bridge.pack_available("sentence-embeddings", "all-MiniLM-L6-v2") is True
    assert bridge.require_pack("sentence-embeddings", "all-MiniLM-L6-v2") is None

    # An id that cannot be in the catalog: the answers above are the
    # fixture's, not a machine that happens to have the pack installed.
    assert bridge.pack_available("no-such-pack", "no-such-model") is True
    assert bridge.require_pack("no-such-pack") is None

    model_path = bridge.model_dir("sentence-transformers/all-MiniLM-L6-v2")
    assert model_path == tmp_path / "model"
    assert (model_path / "config.json").is_file()


def test_fake_sentence_transformers_fixture_is_deterministic_and_normalises(
        fake_sentence_transformers):
    model = fake_sentence_transformers.SentenceTransformer(
        "/models/all-MiniLM-L6-v2", device="cpu", local_files_only=True)

    assert model.init_kwargs == {"device": "cpu", "local_files_only": True}
    assert model.max_seq_length == 128
    model.max_seq_length = 64
    assert model.max_seq_length == 64

    # Deterministic: no PYTHONHASHSEED dependency, no per-instance state.
    texts = ["the neural network learns", "the neural network learns"]
    first = model.encode(texts, batch_size=2)
    second = model.encode(texts, batch_size=2)

    assert first.shape == (2, 32)
    assert first.dtype == np.float32
    assert np.array_equal(first[0], first[1])
    assert np.array_equal(first, second)

    unit = model.encode(texts, normalize_embeddings=True)
    assert np.allclose(np.linalg.norm(unit, axis=1), 1.0)
    assert not np.allclose(np.linalg.norm(first, axis=1), 1.0), (
        "unnormalised rows should not already be unit length")

    # A blank line in a corpus is not an error, and a NaN row would poison
    # every similarity computed against it.
    blank = model.encode([""], normalize_embeddings=True)
    assert not np.isnan(blank).any()
    assert np.array_equal(blank, np.zeros((1, 32), dtype=np.float32))

    # Shared words pull two texts together; unrelated ones stay apart. Node
    # tests assert on relative similarity (nearest neighbour, ranking), so
    # the fake has to model at least that much.
    english = ["the neural network learns embeddings",
               "the neural network trains slowly",
               "bright orange sunset over calm water"]
    related, sibling, unrelated = model.encode(
        english, normalize_embeddings=True)
    assert float(related @ sibling) > float(related @ unrelated)

    # Same again with characters instead of whitespace tokens: Chinese text
    # is one "word" to ``str.split``, so the character bag is what makes
    # these two sentences neighbours at all.
    chinese = ["深度學習很有趣", "深度學習很困難", "明天的天氣晴朗"]
    zh_a, zh_b, zh_other = model.encode(chinese, normalize_embeddings=True)
    assert float(zh_a @ zh_b) > float(zh_a @ zh_other)

    assert model.calls == [texts, texts, texts, [""], english, chinese]
