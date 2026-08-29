"""The shared sentence-transformer loader: what it refuses, what it caches,
and what it hands back.

Everything here runs offline against the ``fake_sentence_transformers``
fixture, so this file also pins the two promises that make that possible.

The GATE runs before anything heavy. A model id that is not in the catalog
is rejected before the packs layer is asked about it, a pack that is not
installed stops the run with the message the Package Center wrote, and a
snapshot that was never downloaded is named in the error rather than
silently downloaded mid-run. Each rung of that ladder is a separate test
because each one is a different thing going wrong for the learner, and the
frontend routes them by the ``(pack=<id>)`` suffix.

The CACHE is process-wide and shared by nodes running on different worker
threads, so its bounds are behaviour, not an implementation detail: a
learner switching models in the editor must not accumulate half a gigabyte
per switch, and two nodes asking for the same model must not load it twice.
"""

from __future__ import annotations

import sys
import threading
import time

import numpy as np
import pytest

from app.core.loop_control import EVENT_BATCH, ProgressThrottle
from app.core.packs import PackMissingError, parse_requirement
from app.core.packs.catalog import get_pack
from app.nodes.llm import _packs_bridge as bridge
from app.nodes.llm import _sentence_models as sentence_models

#: Row width of the fake's embeddings (``conftest._FAKE_EMBED_DIM``). Pinned
#: here so a shape assertion reads as "one row per text", not as a magic
#: number smuggled in from another file.
FAKE_DIM = 32

MINI = "sentence-transformers/all-MiniLM-L6-v2"
BGE = "BAAI/bge-small-zh-v1.5"


@pytest.fixture(autouse=True)
def no_model_survives_a_test():
    """Start and end every test with an empty model cache.

    The cache is module state, so without this a test that loaded a model
    would decide what the NEXT test's constructor-call count means. The
    fixture also happens to exercise ``clear_model_cache``'s idempotence on
    every single test.
    """
    sentence_models.clear_model_cache()
    yield
    sentence_models.clear_model_cache()


def counting_transformer(module, loads, delay=0.0):
    """Install a constructor-counting subclass of the fake, and return it.

    A subclass rather than a lambda: the fake's ``__init__`` is strict on
    purpose (see ``test_packs_bridge``), and a wrapper that accepted
    anything would let a loader that calls the real library wrongly pass
    here. Only the first positional argument is accepted positionally, so
    ``SentenceTransformer(path, "cpu")`` still fails the way it would
    upstream.

    *delay* stretches the load into a window a second thread can be caught
    inside, which is the only way to observe a lock that is doing its job.
    """
    real = module.SentenceTransformer

    class Counting(real):
        def __init__(self, path, **kwargs):
            loads.append((path, kwargs))
            if delay:
                time.sleep(delay)
            super().__init__(path, **kwargs)

    module.SentenceTransformer = Counting
    return Counting


# ── the registry ─────────────────────────────────────────────────────────


def test_model_ids_match_the_catalog():
    """The loader's model table and the Package Center's allowlist describe
    the same four downloads.

    A drift here is invisible until a learner picks the option: the node
    would gate on an item id the catalog has never heard of, which
    ``pack_available`` answers False for, so the model would report itself
    as un-downloadable however many times it was downloaded.
    """
    pack = get_pack(sentence_models.SENTENCE_PACK)
    catalog_items = {item.repo_id: item.item_id
                     for item in pack.items if item.kind == "hf"}

    assert sentence_models.SENTENCE_MODELS == catalog_items
    assert sentence_models.DEFAULT_SENTENCE_MODEL in sentence_models.SENTENCE_MODELS

    options = sentence_models.option_packs_for_models()
    assert set(options) == set(sentence_models.SENTENCE_MODELS)
    for repo_id, requirement in options.items():
        assert requirement == f"sentence-embeddings:{sentence_models.SENTENCE_MODELS[repo_id]}"
        # What a node writes must be what the packs layer reads back.
        assert parse_requirement(requirement) == (
            sentence_models.SENTENCE_PACK,
            sentence_models.SENTENCE_MODELS[repo_id])


# ── the gating ladder ────────────────────────────────────────────────────


def test_unknown_repo_raises_before_touching_packs(monkeypatch):
    """An id that is not in the table is a graph/authoring bug, not a
    missing download -- so it must not reach the packs layer, which would
    report it as something a learner could install."""
    def explode(*args, **kwargs):
        raise AssertionError("the packs layer was asked about an unknown id")

    monkeypatch.setattr(bridge, "require_pack", explode)
    monkeypatch.setattr(bridge, "model_dir", explode)

    with pytest.raises(ValueError) as caught:
        sentence_models.load_sentence_model("no-such/model", "cpu")

    message = str(caught.value)
    assert "no-such/model" in message
    # The learner is told what they CAN pick, not just that they were wrong.
    for repo_id in sentence_models.SENTENCE_MODELS:
        assert repo_id in message
    assert not isinstance(caught.value, PackMissingError)


def test_missing_pack_raises_pack_missing_error_naming_package_center(
        monkeypatch):
    """``require_pack`` already writes the sentence a learner reads; the
    loader must let it through untouched.

    Wrapping it would either lose the ``(pack=<id>)`` suffix the editor
    parses, or duplicate the Package Center wording in a second place where
    it would drift.
    """
    original = PackMissingError(
        sentence_models.SENTENCE_PACK,
        "Model 'all-MiniLM-L6-v2' from the Sentence embeddings pack is not "
        "installed. Open Package Center (toolbar > Settings > Optional "
        "packs) to download it; graph runs never download")

    def refuse(pack_id, item_id=None):
        raise original

    monkeypatch.setattr(bridge, "require_pack", refuse)
    monkeypatch.setattr(bridge, "model_dir",
                        lambda repo_id: pytest.fail("gate was not honoured"))

    with pytest.raises(PackMissingError) as caught:
        sentence_models.load_sentence_model(MINI, "cpu")

    assert caught.value is original
    assert str(caught.value).endswith("(pack=sentence-embeddings)")


def test_missing_model_dir_names_the_model_and_keeps_suffix(
        fake_sentence_transformers, monkeypatch):
    """The cached probe says the snapshot is there and it is not.

    In PRODUCTION this rung is not what refuses a model nobody downloaded:
    ``require_pack(pack, item)`` asks about that exact item and raises
    first. What it cannot see is a change since the last probe --
    ``pack_available`` reads ``state.probe_all()``, memoised for the whole
    process, while ``model_dir`` re-checks the sentinel and the bytes now.
    A cache cleaned out by hand, an uninstall or a half-finished download
    lands here, and has to read as the same actionable sentence rather than
    as ``SentenceTransformer`` failing on a missing directory.

    The fixture patches both, so the test reaches the rung by making
    ``model_dir`` answer None while the gate keeps saying yes -- which is
    exactly the disagreement being described.
    """
    monkeypatch.setattr(bridge, "model_dir", lambda repo_id: None)

    with pytest.raises(PackMissingError) as caught:
        sentence_models.load_sentence_model(BGE, "cpu")

    message = str(caught.value)
    assert BGE in message
    assert "Package Center" in message
    assert "never download" in message
    assert message.endswith("(pack=sentence-embeddings)")
    assert caught.value.pack_id == sentence_models.SENTENCE_PACK

    # Gated on the model the caller picked, not merely on the pack.
    assert fake_sentence_transformers.required == [
        (sentence_models.SENTENCE_PACK, "bge-small-zh-v1.5")]


def test_import_error_is_reported_as_pack_problem(
        fake_sentence_transformers, monkeypatch):
    """A broken install must not reach the learner as a traceback.

    ``None`` in ``sys.modules`` is what a failed import leaves behind, and
    it is also the shape of the real failure: the pack's sentinel says
    installed, its site-packages say otherwise.
    """
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    with pytest.raises(PackMissingError) as caught:
        sentence_models.load_sentence_model(MINI, "cpu")

    message = str(caught.value)
    assert "sentence_transformers" in message
    assert "reinstall" in message.lower()
    assert message.endswith("(pack=sentence-embeddings)")


# ── the model cache ──────────────────────────────────────────────────────


def test_model_is_loaded_from_the_local_snapshot_only(
        fake_sentence_transformers, tmp_path):
    """The one guarantee a graph run makes about pack contents: it opens
    what the Package Center downloaded, and it never fetches."""
    loads: list = []
    counting_transformer(fake_sentence_transformers, loads)

    model = sentence_models.load_sentence_model(MINI, "cpu")

    assert loads == [(str(tmp_path / "model"),
                      {"device": "cpu",
                       "local_files_only": True,
                       "trust_remote_code": False})]
    assert model.init_kwargs["cache_folder"] is None
    assert model.init_kwargs["token"] is None


def test_cache_is_keyed_by_repo_and_device_and_bounded_to_two(
        fake_sentence_transformers):
    """Two models stay resident; the third evicts the least recently used.

    Two because that is what a learner actually does -- compare an English
    model against a multilingual one -- and because four resident models
    would be over a gigabyte of RAM held after the run finished. The key
    carries the DEVICE as well: the same weights on cuda and on cpu are two
    different objects, and handing back the cpu one for a cuda request
    would put the embeddings on the wrong device.
    """
    loads: list = []
    counting_transformer(fake_sentence_transformers, loads)

    first = sentence_models.load_sentence_model(MINI, "cpu")
    assert len(loads) == 1
    assert sentence_models.load_sentence_model(MINI, "cpu") is first
    assert len(loads) == 1, "the same model on the same device loaded twice"

    # Same weights, different device -> a different entry.
    on_cuda = sentence_models.load_sentence_model(MINI, "cuda")
    assert len(loads) == 2

    # Third distinct key: the least recently used entry (MINI/cpu) goes.
    sentence_models.load_sentence_model(BGE, "cpu")
    assert len(loads) == 3

    # MINI/cuda was still resident...
    assert sentence_models.load_sentence_model(MINI, "cuda") is on_cuda
    assert len(loads) == 3

    # ...and MINI/cpu was not: it has to be loaded again.
    reloaded = sentence_models.load_sentence_model(MINI, "cpu")
    assert len(loads) == 4
    assert reloaded is not first

    # Least RECENTLY USED, not first in: the hit above moved MINI/cuda to
    # the back, so the load that just evicted something evicted BGE/cpu
    # rather than the entry a plain insertion-ordered cache would have
    # dropped. Nothing else in this test can tell the two policies apart.
    assert sentence_models.load_sentence_model(MINI, "cuda") is on_cuda
    assert len(loads) == 4

    # The fixture's teardown leans on this, and it runs whether or not any
    # model was ever loaded.
    sentence_models.clear_model_cache()
    sentence_models.clear_model_cache()
    sentence_models.load_sentence_model(MINI, "cpu")
    assert len(loads) == 5


def test_concurrent_loads_of_the_same_model_construct_it_once(
        fake_sentence_transformers):
    """The lock, observed rather than assumed.

    Nodes run on worker threads (``MAX_PARALLEL_NODES = 4``), so a graph
    with two embedding nodes really can have two threads inside this
    function at the same moment. Without the lock they would both miss the
    cache and both load the model -- twice the wait and twice the RAM, with
    one of the two copies immediately orphaned by the other's write.

    The barrier makes all four threads arrive together instead of hoping
    they do, and the delay in the constructor keeps the first one inside
    the critical section while the rest queue up.
    """
    loads: list = []
    counting_transformer(fake_sentence_transformers, loads, delay=0.05)

    workers = 4
    ready = threading.Barrier(workers)
    results: list = []
    errors: list = []

    def load_once():
        try:
            ready.wait(timeout=10)
            results.append(sentence_models.load_sentence_model(MINI, "cpu"))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=load_once) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert errors == []
    assert [thread.is_alive() for thread in threads] == [False] * workers, (
        "a load never returned -- the lock is not being released")
    assert len(loads) == 1, (
        f"{workers} threads asked for one model and it was loaded "
        f"{len(loads)} times")
    assert len(results) == workers
    assert all(model is results[0] for model in results), (
        "the threads were handed different objects for one cache key")


def test_max_seq_length_is_applied_on_every_call(fake_sentence_transformers):
    """The cap is a per-NODE choice on a per-PROCESS object.

    Two TextEmbedding nodes can share one cached model with different token
    caps, so applying the cap only on load would silently give the second
    node the first node's truncation.
    """
    default = fake_sentence_transformers.SentenceTransformer("/x").max_seq_length

    model = sentence_models.load_sentence_model(MINI, "cpu", max_seq_length=64)
    assert model.max_seq_length == 64

    # 0 means "leave the cached model's cap alone" -- not "restore the
    # shipped default". ("The model's own default" is the NODE PARAM's
    # wording, and it is only true of a cold load.)
    again = sentence_models.load_sentence_model(MINI, "cpu")
    assert again is model
    assert again.max_seq_length == 64

    # A cache HIT still gets the caller's cap.
    third = sentence_models.load_sentence_model(MINI, "cpu", max_seq_length=256)
    assert third is model
    assert third.max_seq_length == 256

    assert default != 256, "the fake's default would hide a no-op assignment"


# ── the encode loop ──────────────────────────────────────────────────────


def test_encode_in_batches_emits_progress_and_honours_stop(
        fake_sentence_transformers):
    """Stop is checked at the top of each batch, and what was already
    embedded comes back rather than being thrown away."""
    model = fake_sentence_transformers.SentenceTransformer("/x")
    frames: list[dict] = []
    stop_after = iter([False, True, True])

    rows, stopped_at = sentence_models.encode_in_batches(
        model, ["one", "two", "three"],
        batch_size=1,
        normalize=False,
        prefix="",
        progress=ProgressThrottle(frames.append, min_interval_s=0.0),
        should_stop=lambda: next(stop_after))

    assert stopped_at == 1
    assert model.calls == [["one"]], "a stopped loop kept embedding"
    assert rows.shape == (1, FAKE_DIM), "the finished batch was discarded"

    assert frames == [{"event": EVENT_BATCH, "batch": 1, "total_batches": 3,
                       "text": "Embedding 1/3"}]

    # A run that finishes, with batches and texts deliberately different
    # numbers: a frame counting texts as batches would report 5 of 5 while
    # two thirds of the work was still to come.
    frames.clear()
    sentence_models.encode_in_batches(
        model, ["a", "b", "c", "d", "e"],
        batch_size=2,
        normalize=False,
        prefix="",
        progress=ProgressThrottle(frames.append, min_interval_s=0.0),
        should_stop=lambda: False)

    assert frames == [
        {"event": EVENT_BATCH, "batch": 1, "total_batches": 3,
         "text": "Embedding 2/5"},
        {"event": EVENT_BATCH, "batch": 2, "total_batches": 3,
         "text": "Embedding 4/5"},
        # The short final batch counts the texts it held, not the batch size.
        {"event": EVENT_BATCH, "batch": 3, "total_batches": 3,
         "text": "Embedding 5/5"},
    ]


def test_a_bare_string_is_rejected_instead_of_embedded_per_character(
        fake_sentence_transformers):
    """``list("hello")`` is five one-character texts.

    Nothing downstream would object: five rows come back, the shapes are
    right, the run succeeds. Only the embeddings are nonsense. A caller who
    meant one text has to hear about it here, where the mistake is.
    """
    model = fake_sentence_transformers.SentenceTransformer("/x")

    with pytest.raises(ValueError) as caught:
        sentence_models.encode_in_batches(
            model, "hello",
            batch_size=2, normalize=False, prefix="",
            progress=None, should_stop=lambda: False)

    assert "sequence of strings" in str(caught.value)
    assert model.calls == [], "characters reached the model"


def test_prefix_is_prepended_to_every_text(fake_sentence_transformers):
    """multilingual-e5 was trained with ``query: ``/``passage: ``; a prefix
    applied to only the first row of a batch would quietly weaken every
    other row's retrieval score."""
    model = fake_sentence_transformers.SentenceTransformer("/x")

    sentence_models.encode_in_batches(
        model, ["a", "b", "c"],
        batch_size=2, normalize=False, prefix="query: ",
        progress=None, should_stop=lambda: False)

    assert model.calls == [["query: a", "query: b"], ["query: c"]]


def test_encode_returns_float32_and_normalised_rows_when_asked(
        fake_sentence_transformers):
    """One float32 row per text, in input order, across batch boundaries."""
    model = fake_sentence_transformers.SentenceTransformer("/x")
    texts = ["alpha", "beta", "gamma", "delta", "epsilon"]

    rows, stopped_at = sentence_models.encode_in_batches(
        model, texts,
        batch_size=2, normalize=True, prefix="",
        progress=None, should_stop=lambda: False)

    assert stopped_at is None
    assert rows.dtype == np.float32
    assert rows.shape == (len(texts), FAKE_DIM)
    assert np.allclose(np.linalg.norm(rows, axis=1), 1.0)

    # Order survives the split into batches.
    one_by_one = np.stack(
        [model.encode([text], normalize_embeddings=True)[0] for text in texts])
    assert np.allclose(rows, one_by_one)

    plain, _ = sentence_models.encode_in_batches(
        model, texts, batch_size=8, normalize=False, prefix="",
        progress=None, should_stop=lambda: False)
    assert plain.dtype == np.float32
    assert not np.allclose(np.linalg.norm(plain, axis=1), 1.0)

    # Nothing to embed is not an error, and the empty result is still a
    # 2-D float32 array so a caller can concatenate or stack it.
    empty, empty_stop = sentence_models.encode_in_batches(
        model, [], batch_size=4, normalize=True, prefix="",
        progress=None, should_stop=lambda: True)
    assert empty.shape == (0, 0)
    assert empty.dtype == np.float32
    assert empty_stop is None
