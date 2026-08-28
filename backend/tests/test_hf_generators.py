"""The shared causal-LM loader: what it refuses, what it caches, what it
hands back.

Everything here runs offline against the ``fake_transformers`` fixture, so
this file also pins the two promises that make that possible.

The GATE runs before anything heavy. A model id that is not in the catalog
is rejected before the packs layer is asked about it and so is a dtype name
nothing recognises (both are authoring bugs, and no download fixes either),
a pack that is not installed stops the run with the message the Package
Center wrote, and a snapshot that was never downloaded is named in the
error rather than silently downloaded mid-run. Each rung is a separate test
because each one is a different thing going wrong for the learner, and the
frontend routes them by the ``(pack=<id>)`` suffix.

The CACHE holds exactly ONE generator, and that bound is behaviour rather
than an implementation detail: these weights are a gigabyte, and a learner
flipping between two devices or two precisions in the editor must not
accumulate one resident copy per flip.
"""

from __future__ import annotations

import sys
import threading
import time

import pytest
import torch

from app.core.packs import PackMissingError, parse_requirement
from app.core.packs.catalog import get_pack
from app.nodes.llm import _hf_generators as generators
from app.nodes.llm import _packs_bridge as bridge

QWEN = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(autouse=True)
def no_generator_survives_a_test():
    """Start and end every test with an empty generator cache.

    The cache is module state, so without this a test that loaded a model
    would decide what the NEXT test's load count means. It also exercises
    ``clear_generator_cache``'s idempotence on every single test.
    """
    generators.clear_generator_cache()
    yield
    generators.clear_generator_cache()


# ── the registry ─────────────────────────────────────────────────────────


def test_generator_ids_match_the_catalog():
    """The loader's model table and the Package Center's allowlist describe
    the same download.

    A drift here is invisible until a learner picks the option: the node
    would gate on an item id the catalog has never heard of, which
    ``pack_available`` answers False for, so the model would report itself
    as un-downloadable however many times it was downloaded.
    """
    pack = get_pack(generators.RAG_PACK)
    catalog_items = {item.repo_id: item.item_id
                     for item in pack.items if item.kind == "hf"}

    assert generators.GENERATOR_MODELS == catalog_items
    assert generators.DEFAULT_GENERATOR in generators.GENERATOR_MODELS

    options = generators.option_packs_for_generators()
    assert set(options) == set(generators.GENERATOR_MODELS)
    for repo_id, requirement in options.items():
        item_id = generators.GENERATOR_MODELS[repo_id]
        assert requirement == f"rag:{item_id}"
        # What a node writes must be what the packs layer reads back.
        assert parse_requirement(requirement) == (generators.RAG_PACK, item_id)


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
        generators.load_causal_lm("no-such/model", "cpu", "auto")

    message = str(caught.value)
    assert "no-such/model" in message
    # The learner is told what they CAN pick, not just that they were wrong.
    for repo_id in generators.GENERATOR_MODELS:
        assert repo_id in message
    assert not isinstance(caught.value, PackMissingError)


def test_unknown_dtype_raises_before_touching_packs(monkeypatch):
    """The second authoring bug, checked on the same rung as the first.

    A ``dtype`` name nothing recognises came from a hand-edited graph or a
    generated script, and no download fixes it -- so it must not reach the
    packs layer, which would send the learner to Package Center to install
    a gigabyte that will not help.
    """
    def explode(*args, **kwargs):
        raise AssertionError("the packs layer was asked about a bad dtype")

    monkeypatch.setattr(bridge, "require_pack", explode)
    monkeypatch.setattr(bridge, "model_dir", explode)

    with pytest.raises(ValueError) as caught:
        generators.load_causal_lm(QWEN, "cpu", "int8")

    message = str(caught.value)
    assert "int8" in message
    # Named, so a graph carrying a dead option says what the live ones are.
    for name in generators.DTYPES:
        assert name in message
    assert not isinstance(caught.value, PackMissingError)


def test_require_pack_error_reaches_the_learner_untouched(monkeypatch):
    """``require_pack`` already writes the sentence a learner reads.

    Wrapping it would either lose the ``(pack=<id>)`` suffix the editor
    parses to offer the download, or duplicate the Package Center wording in
    a second place where it would drift.
    """
    original = PackMissingError(
        generators.RAG_PACK,
        "Model 'qwen2.5-0.5b-instruct' from the RAG stack pack is not "
        "installed. Open Package Center to download it")

    def refuse(pack_id, item_id=None):
        raise original

    monkeypatch.setattr(bridge, "require_pack", refuse)
    monkeypatch.setattr(bridge, "model_dir",
                        lambda repo_id: pytest.fail("gate was not honoured"))

    with pytest.raises(PackMissingError) as caught:
        generators.load_causal_lm(QWEN, "cpu", "auto")

    assert caught.value is original
    assert str(caught.value).endswith("(pack=rag)")


def test_missing_pack_error_names_package_center(
        fake_transformers, monkeypatch):
    """The cached probe says the snapshot is there and it is not.

    In PRODUCTION this rung is not what refuses a model nobody downloaded:
    ``require_pack(pack, item)`` asks about that exact item and raises
    first. What it cannot see is a change since the last probe, so a cache
    cleaned out by hand, an uninstall or a half-finished download lands
    here -- and has to read as the same actionable sentence rather than as
    ``from_pretrained`` failing on a missing directory.
    """
    monkeypatch.setattr(bridge, "model_dir", lambda repo_id: None)

    with pytest.raises(PackMissingError) as caught:
        generators.load_causal_lm(QWEN, "cpu", "auto")

    message = str(caught.value)
    assert QWEN in message
    assert "Package Center" in message
    assert "RAG stack" in message
    assert "never download" in message
    assert message.endswith("(pack=rag)")
    assert caught.value.pack_id == generators.RAG_PACK

    # Gated on the model the caller picked, not merely on the pack.
    assert fake_transformers.required == [
        (generators.RAG_PACK, "qwen2.5-0.5b-instruct")]
    assert fake_transformers.model_loads == []


def test_import_error_is_reported_as_a_pack_problem(
        fake_transformers, monkeypatch):
    """A broken install must not reach the learner as a traceback.

    ``None`` in ``sys.modules`` is what a failed import leaves behind, and
    it is also the shape of the real failure: the pack's sentinel says
    installed, its site-packages say otherwise. The message has to name the
    SENTENCE-EMBEDDINGS pack, because that is where transformers comes
    from -- the rag pack merely depends on it, so "reinstall the rag pack"
    would send the learner to the wrong button.
    """
    monkeypatch.setitem(sys.modules, "transformers", None)

    with pytest.raises(PackMissingError) as caught:
        generators.load_causal_lm(QWEN, "cpu", "auto")

    message = str(caught.value)
    assert "transformers" in message
    assert "sentence-embeddings" in message
    assert "reinstall" in message.lower()
    assert message.endswith("(pack=rag)")


# ── the precision rules ──────────────────────────────────────────────────


def test_resolve_dtype_rules(monkeypatch):
    """auto is half precision on CUDA and float32 everywhere else."""
    assert generators.resolve_dtype("auto", "cpu") is torch.float32
    assert generators.resolve_dtype("auto", "mps") is torch.float32
    # A hand-edited graph or an old saved node can carry a missing value;
    # that is "auto", not a crash.
    assert generators.resolve_dtype("", "cpu") is torch.float32
    assert generators.resolve_dtype(None, "cpu") is torch.float32

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert generators.resolve_dtype("auto", "cuda") is torch.bfloat16
    # ``cuda:1`` is still cuda.
    assert generators.resolve_dtype("auto", "cuda:1") is torch.bfloat16

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert generators.resolve_dtype("auto", "cuda") is torch.float16

    def no_driver():
        raise RuntimeError("cannot talk to the driver")

    # A capability nobody can read is an absent one, not a failed run.
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", no_driver)
    assert generators.resolve_dtype("auto", "cuda") is torch.float16

    # An explicit name maps straight through, on any device -- including the
    # combinations auto would never pick.
    assert generators.resolve_dtype("float32", "cuda") is torch.float32
    assert generators.resolve_dtype("float16", "cpu") is torch.float16
    assert generators.resolve_dtype("bfloat16", "cpu") is torch.bfloat16
    assert generators.resolve_dtype("  BFloat16 ", "cpu") is torch.bfloat16

    with pytest.raises(ValueError) as caught:
        generators.resolve_dtype("int8", "cpu")
    assert "int8" in str(caught.value)
    # Named, so a graph carrying a dead option says what the live ones are.
    for name in generators.DTYPES:
        assert name in str(caught.value)


# ── the loaded model ─────────────────────────────────────────────────────


def test_model_is_loaded_from_the_local_snapshot_and_cast_afterwards(
        fake_transformers):
    """The one guarantee a graph run makes about pack contents: it opens
    what the Package Center downloaded, and it never fetches.

    The cast is checked here too, because the keyword that would have done
    it during the load (``torch_dtype`` in transformers 4, ``dtype`` in 5)
    is exactly what this loader avoids depending on.
    """
    tokenizer, model = generators.load_causal_lm(QWEN, "cpu", "float32")

    snapshot = str(fake_transformers.model_path)
    assert fake_transformers.tokenizer_loads == [
        (snapshot, {"local_files_only": True})]
    assert fake_transformers.model_loads == [
        (snapshot, {"local_files_only": True})]
    assert tokenizer is fake_transformers.tokenizers[0]
    assert model is fake_transformers.models[0]

    assert model.to_calls == [((), {"device": "cpu", "dtype": torch.float32})]
    assert model.eval_calls == 1
    assert model.training is False


def test_generator_cache_holds_one_model(fake_transformers):
    """One entry, keyed by ``(repo, device, resolved dtype)``.

    One rather than the encoder cache's two because these weights are a
    gigabyte: a second resident entry could only ever be the same model on
    a second device or in a second precision, and switching either is a
    deliberate act worth paying a reload for.
    """
    loads = fake_transformers.model_loads

    first = generators.load_causal_lm(QWEN, "cpu", "float32")
    assert len(loads) == 1
    assert generators.load_causal_lm(QWEN, "cpu", "float32") is first
    assert len(loads) == 1, "the same model on the same device loaded twice"

    # The key carries the RESOLVED precision, so ``auto`` on a CPU and an
    # explicit float32 are one entry rather than two identical gigabytes.
    assert generators.load_causal_lm(QWEN, "cpu", "auto") is first
    assert len(loads) == 1

    # A second precision is a second key -- and it evicts the first.
    half = generators.load_causal_lm(QWEN, "cpu", "float16")
    assert len(loads) == 2
    assert half[1] is not first[1]

    reloaded = generators.load_causal_lm(QWEN, "cpu", "float32")
    assert len(loads) == 3, "two generators stayed resident"
    assert reloaded[1] is not first[1]

    # The fixture's teardown leans on this, and it runs whether or not any
    # model was ever loaded.
    generators.clear_generator_cache()
    generators.clear_generator_cache()
    generators.load_causal_lm(QWEN, "cpu", "float32")
    assert len(loads) == 4


def test_concurrent_loads_of_the_same_model_construct_it_once(
        fake_transformers, monkeypatch):
    """The lock, observed rather than assumed.

    Nodes run on worker threads (``MAX_PARALLEL_NODES = 4``), so a graph
    with two generator nodes really can have two threads inside this
    function at the same moment. Without the lock they would both miss the
    cache and both load a gigabyte of weights -- twice the wait and twice
    the RAM, with one of the two copies immediately orphaned by the other's
    write into a cache that holds exactly one.

    The barrier makes all four threads arrive together instead of hoping
    they do, and the delay inside ``from_pretrained`` keeps the first one
    in the critical section while the rest queue up.
    """
    loaded = fake_transformers.AutoModelForCausalLM.from_pretrained

    def slow_from_pretrained(path, **kwargs):
        time.sleep(0.05)
        return loaded(path, **kwargs)

    monkeypatch.setattr(fake_transformers.AutoModelForCausalLM,
                        "from_pretrained", slow_from_pretrained)

    workers = 4
    ready = threading.Barrier(workers)
    results: list = []
    errors: list = []

    def load_once():
        try:
            ready.wait(timeout=10)
            results.append(generators.load_causal_lm(QWEN, "cpu", "float32"))
        except Exception as exc:  # noqa: BLE001 -- reported by the assert
            errors.append(exc)

    threads = [threading.Thread(target=load_once) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert errors == []
    assert [thread.is_alive() for thread in threads] == [False] * workers, (
        "a load never returned -- the lock is not being released")
    loads = fake_transformers.model_loads
    assert len(loads) == 1, (
        f"{workers} threads asked for one generator and it was loaded "
        f"{len(loads)} times")
    assert len(results) == workers
    tokenizer, model = results[0]
    assert all(entry[0] is tokenizer and entry[1] is model
               for entry in results), (
        "the threads were handed different objects for one cache key")


# ── the stop tokens ──────────────────────────────────────────────────────


class _Config:
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id


class _Model:
    def __init__(self, eos_token_id):
        self.generation_config = _Config(eos_token_id)


class _Tokenizer:
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id


def test_stop_ids_reads_the_tokenizer_and_the_generation_config():
    """Both sources, because an instruction-tuned model needs both.

    The tokenizer's ``eos_token_id`` is the base model's end-of-text; the
    chat template ends a turn with a different token, and the only place
    that is written down is the generation config -- where it may be one id
    or a list. Reading either alone ends a run at ``max_new_tokens`` with
    the model talking to itself in both voices.
    """
    assert generators.stop_ids(_Tokenizer(7), _Model([7, 11])) == {7, 11}
    assert generators.stop_ids(_Tokenizer(7), _Model(11)) == {7, 11}

    # None-safe at every rung: no eos, no config, a null id in the config.
    assert generators.stop_ids(_Tokenizer(None), _Model(None)) == set()
    assert generators.stop_ids(_Tokenizer(3), object()) == {3}
    assert generators.stop_ids(object(), _Model([5])) == {5}

    # ``bool`` is an ``int`` subclass, so an unset flag left in a config
    # would otherwise become "stop on token 1".
    assert generators.stop_ids(_Tokenizer(True), _Model(False)) == set()
