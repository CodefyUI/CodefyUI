"""Which optional packs are usable right now -- and what a node does when one
is not.

Two audiences, one answer. The Package Center lists every pack with a state
badge, which has to be cheap enough to compute on every poll: no importing
sentence-transformers to find out whether sentence-transformers is there.
And a node that needs a pack has to fail with a message the editor can turn
into an install button, rather than an ImportError traceback.

A downloaded item is judged by its SENTINEL, not by the presence of a
directory: an interrupted 400 MB snapshot leaves a directory that looks
exactly like a finished one. The sentinel is written last, so it is the only
honest record that the bytes arrived -- and it is re-checked against the
catalog, so a pack that changed which revision it wants does not report the
old download as the new one.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import shutil
import sys

import pytest
import torch

from app.core import packs
from app.core.asset_cache import cache_dir
from app.core.packs import PackMissingError, state
from app.core.packs.catalog import Pack, get_item, get_pack, iter_packs
from app.core.packs.paths import asset_dir, hf_cache_dir, sentinel_path


@pytest.fixture(autouse=True)
def user_data_dir(tmp_path, monkeypatch):
    """Every test in this file gets a throwaway cache root AND a cold probe
    cache -- the module-level cache would otherwise carry one test's answer
    (or the developer's real machine) into the next."""
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path))
    clear = getattr(cache_dir, "cache_clear", None)
    if clear is not None:
        clear()
    state.invalidate()
    yield tmp_path
    if clear is not None:
        clear()
    state.invalidate()


def _hf_sentinel(pack_id: str, item_id: str, *, repo_id: str, revision: str,
                 snapshot_dir) -> None:
    state.write_sentinel(pack_id, item_id, {
        "schema": 1, "pack_id": pack_id, "item_id": item_id, "kind": "hf",
        "repo_id": repo_id, "revision": revision,
        "snapshot_dir": str(snapshot_dir), "bytes": 1, "at": "2026-08-28T00:00:00Z",
    })


def _install_hf(pack_id: str, item_id: str):
    """Make one Hugging Face item look downloaded. Returns the snapshot."""
    item = get_item(get_pack(pack_id), item_id)
    snapshot = (hf_cache_dir()
                / f"models--{item.repo_id.replace('/', '--')}" / "abc123")
    snapshot.mkdir(parents=True)
    _hf_sentinel(pack_id, item_id, repo_id=item.repo_id,
                 revision=item.revision, snapshot_dir=snapshot)
    return snapshot


def _install_qwen():
    """Make the rag pack's one model look downloaded. Returns the snapshot."""
    return _install_hf("rag", "qwen2.5-0.5b-instruct")


def _asset_sentinel(pack_id: str, item_id: str, *, url, path, sha256):
    state.write_sentinel(pack_id, item_id, {
        "schema": 1, "pack_id": pack_id, "item_id": item_id, "kind": "asset",
        "url": url, "path": str(path), "bytes": 30, "sha256": sha256,
        "at": "2026-08-28T00:00:00Z",
    })


def _install_glove():
    """Make the word-vectors asset look downloaded. Returns the file path.

    The digest recorded is the CATALOG's, which is what a real install
    writes: ``download._finish_asset`` stores the digest it computed off the
    bytes that arrived, and that download only succeeded because it matched.
    """
    item = get_item(get_pack("word-vectors"), "glove-50d")
    path = asset_dir() / item.filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not really a word vector table")
    _asset_sentinel("word-vectors", item.item_id, url=item.url, path=path,
                    sha256=item.sha256)
    return path


# ── probing pip packages ─────────────────────────────────────────────────


def test_pip_ready_uses_find_spec(monkeypatch):
    """``find_spec``, never ``import``: asking whether transformers is there
    must not cost the two seconds and half a gigabyte of importing it."""
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name: None if name == "sentence_transformers" else object())

    assert not state.pip_ready(get_pack("sentence-embeddings"))
    assert state.pip_ready(get_pack("rag"))


def test_pip_ready_is_true_when_nothing_to_probe():
    """word-vectors is pure data -- no packages to import, so nothing can be
    missing. It must not read as "not ready" forever."""
    assert get_pack("word-vectors").probe_modules == ()
    assert state.pip_ready(get_pack("word-vectors"))


def test_a_probe_that_raises_reads_as_missing(monkeypatch):
    """``find_spec`` imports parent packages, so a half-installed dependency
    can raise anything at all. A status poll must survive that."""
    def explode(name):
        raise ValueError("no parent package")

    monkeypatch.setattr(importlib.util, "find_spec", explode)

    assert not state.pip_ready(get_pack("sentence-embeddings"))


# ── judging a downloaded item ────────────────────────────────────────────


def test_item_present_requires_sentinel_and_directory(user_data_dir):
    pack, item = get_pack("rag"), get_item(get_pack("rag"), "qwen2.5-0.5b-instruct")

    missing = state.item_state(pack, item)
    assert not missing.present
    assert missing.snapshot_dir is None
    assert missing.sentinel == sentinel_path(pack.pack_id, item.item_id)

    snapshot = _install_qwen()
    assert state.item_state(pack, item).present
    assert state.item_state(pack, item).snapshot_dir == snapshot

    shutil.rmtree(snapshot)
    gone = state.item_state(pack, item)
    assert not gone.present, "a sentinel outlived the bytes it vouched for"
    assert gone.snapshot_dir is None


def test_corrupt_sentinel_reads_as_missing(user_data_dir):
    """Both ways a sentinel can be garbage. The second one is not JSON at all:
    ``read_text`` raises ``UnicodeDecodeError`` on it, which is a ValueError
    rather than an OSError -- and one corrupt byte in one sentinel must cost
    that item alone, not every pack query in the process."""
    pack, item = get_pack("rag"), get_item(get_pack("rag"), "qwen2.5-0.5b-instruct")
    _install_qwen()
    sentinel = sentinel_path(pack.pack_id, item.item_id)

    sentinel.write_text("{ truncated", encoding="utf-8")
    assert state.read_sentinel(sentinel) is None
    assert not state.item_state(pack, item).present

    sentinel.write_bytes(b"\xff\xfe\x00garbage")
    assert state.read_sentinel(sentinel) is None
    assert not state.item_state(pack, item).present
    assert state.probe_all()["rag"].pack_id == "rag"


def test_sentinel_for_the_wrong_revision_reads_as_missing(user_data_dir):
    """The catalog is allowed to move to a new revision. Yesterday's download
    is then not this pack's model, however complete it is on disk."""
    pack, item = get_pack("rag"), get_item(get_pack("rag"), "qwen2.5-0.5b-instruct")
    snapshot = _install_qwen()

    _hf_sentinel(pack.pack_id, item.item_id, repo_id="somebody/else",
                 revision=item.revision, snapshot_dir=snapshot)
    assert not state.item_state(pack, item).present

    _hf_sentinel(pack.pack_id, item.item_id, repo_id=item.repo_id,
                 revision="v-from-the-future", snapshot_dir=snapshot)
    assert not state.item_state(pack, item).present


def test_asset_item_points_at_the_file(user_data_dir):
    pack, item = get_pack("word-vectors"), get_item(get_pack("word-vectors"),
                                                    "glove-50d")

    assert not state.item_state(pack, item).present

    path = _install_glove()
    present = state.item_state(pack, item)

    assert present.present
    assert present.snapshot_dir == path

    path.unlink()
    assert not state.item_state(pack, item).present


def test_asset_sentinel_for_the_wrong_url_reads_as_missing(user_data_dir):
    """An asset is re-checked against the catalog, exactly as an hf item is.

    The catalog is allowed to move an asset to a new URL -- a rehosted
    mirror, a new upstream release. Yesterday's bytes are then not this
    item's file, however present they are on disk, and the file itself
    cannot say so: unlike a hub snapshot it carries no record of where it
    came from except this sentinel.
    """
    pack, item = get_pack("word-vectors"), get_item(get_pack("word-vectors"),
                                                    "glove-50d")
    path = _install_glove()
    assert state.item_state(pack, item).present

    _asset_sentinel(pack.pack_id, item.item_id, url="https://elsewhere/x.gz",
                    path=path, sha256=item.sha256)

    assert not state.item_state(pack, item).present


def test_asset_sentinel_for_the_wrong_digest_reads_as_missing(user_data_dir):
    """A digest the catalog now records and the download does not match is
    the same fact: those are not the bytes this item names.

    An install done BEFORE a digest was recorded is unaffected -- it wrote
    the digest it computed off the bytes that arrived, which is the one the
    catalog then records -- so adding a digest does not invalidate anything
    already on disk.
    """
    pack, item = get_pack("word-vectors"), get_item(get_pack("word-vectors"),
                                                    "glove-50d")
    path = _install_glove()

    _asset_sentinel(pack.pack_id, item.item_id, url=item.url, path=path,
                    sha256="0" * 64)

    assert not state.item_state(pack, item).present


# ── writing and removing sentinels ───────────────────────────────────────


def test_write_sentinel_is_atomic_and_readable(user_data_dir):
    path = state.write_sentinel("rag", "qwen2.5-0.5b-instruct", {"schema": 1})

    assert path == sentinel_path("rag", "qwen2.5-0.5b-instruct")
    assert state.read_sentinel(path) == {"schema": 1}
    assert json.loads(path.read_text(encoding="utf-8")) == {"schema": 1}
    assert list(path.parent.iterdir()) == [path], "a .tmp file was left behind"


def test_write_sentinel_uses_lf_endings(user_data_dir):
    """One line ending on every platform.

    ``write_text`` with no ``newline=`` translates ``json.dumps``'s ``\\n``
    to ``os.linesep``, so the same sentinel is written differently on
    Windows and on Linux. It round-trips either way -- this is cosmetic --
    but ``write_constraints_file`` already pins LF for exactly this reason
    and two files in one feature should not disagree about it.

    RED ON WINDOWS ONLY. A green run on Linux proves nothing: there
    ``os.linesep`` is already ``\\n``.
    """
    path = state.write_sentinel("rag", "qwen2.5-0.5b-instruct",
                                {"schema": 1, "path": "x"})

    assert b"\r\n" not in path.read_bytes()


def test_remove_sentinel_reports_whether_there_was_one(user_data_dir):
    state.write_sentinel("rag", "qwen2.5-0.5b-instruct", {"schema": 1})

    assert state.remove_sentinel("rag", "qwen2.5-0.5b-instruct") is True
    assert state.remove_sentinel("rag", "qwen2.5-0.5b-instruct") is False
    assert state.read_sentinel(sentinel_path("rag", "qwen2.5-0.5b-instruct")) is None


def test_read_sentinel_of_a_missing_file_is_none(user_data_dir):
    assert state.read_sentinel(user_data_dir / "nope.json") is None


def test_read_sentinel_rejects_json_that_is_not_an_object(user_data_dir):
    path = user_data_dir / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    assert state.read_sentinel(path) is None


@pytest.mark.parametrize("bad_id", [
    pytest.param("..", id="parent"),
    pytest.param(".", id="dot"),
    pytest.param("../../evil", id="traversal"),
    pytest.param("a/b", id="slash"),
    pytest.param("a\\b", id="backslash"),
    pytest.param("", id="empty"),
    # A `$`-anchored pattern used with `.match()` accepts a TRAILING NEWLINE:
    # "rag\n" and "..\n" both pass `^[A-Za-z0-9._-]+$`, and "..\n" is not
    # equal to ".." either, so it walks past the explicit check as well.
    pytest.param("rag\n", id="trailing-newline"),
    pytest.param("..\n", id="parent-with-newline"),
])
def test_sentinel_ids_may_not_walk_out_of_the_state_directory(bad_id, user_data_dir):
    """``sentinel_path`` interpolates both ids straight into a filename. The
    ids only ever come from the catalog today -- this is the guard for the
    day one arrives from a request body instead."""
    with pytest.raises(ValueError):
        state.write_sentinel(bad_id, "item", {})
    with pytest.raises(ValueError):
        state.write_sentinel("pack", bad_id, {})
    with pytest.raises(ValueError):
        state.remove_sentinel(bad_id, "item")


# ── the torch build ──────────────────────────────────────────────────────


@pytest.mark.parametrize("version, expected", [
    ("2.11.0+cu128", "cu128"),
    ("2.6.0+cpu", "cpu"),
    ("2.5.1+rocm6.2", "rocm6.2"),
])
def test_torch_variant_parses_local_tag(monkeypatch, version, expected):
    monkeypatch.setattr(torch, "__version__", version)

    assert state.torch_variant() == expected


def test_torch_variant_of_an_untagged_build_depends_on_the_platform(monkeypatch):
    """PyPI's default wheel is the CPU build on Windows and Linux and carries
    no tag. On macOS the same untagged wheel is the MPS build, so "no tag"
    there says nothing about acceleration and the answer is "unknown"."""
    monkeypatch.setattr(torch, "__version__", "2.6.0")

    monkeypatch.setattr(sys, "platform", "win32")
    assert state.torch_variant() == "cpu"

    monkeypatch.setattr(sys, "platform", "linux")
    assert state.torch_variant() == "cpu"

    monkeypatch.setattr(sys, "platform", "darwin")
    assert state.torch_variant() is None


def test_gpu_torch_installed_follows_torch_variant(monkeypatch):
    """"Installed" for this pack does not mean "torch is importable" -- torch
    always is. It means the accelerated build is the one in place."""
    monkeypatch.setattr(torch, "__version__", "2.11.0+cu128")
    cuda = state.pack_state(get_pack("gpu-torch"))

    assert cuda.installed
    assert cuda.pip_ready
    assert cuda.items == ()

    monkeypatch.setattr(torch, "__version__", "2.6.0+cpu")
    assert not state.pack_state(get_pack("gpu-torch")).installed

    monkeypatch.setattr(torch, "__version__", "2.5.1+rocm6.2")
    assert state.pack_state(get_pack("gpu-torch")).installed


# ── whole-pack state ─────────────────────────────────────────────────────


def test_pack_is_installed_only_when_packages_and_items_are_both_there(
        monkeypatch, user_data_dir):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    pack = get_pack("rag")

    assert not state.pack_state(pack).installed, "no model downloaded yet"

    _install_qwen()
    ready = state.pack_state(pack)
    assert ready.installed
    assert ready.pack_id == "rag"
    assert [item.item_id for item in ready.items] == ["qwen2.5-0.5b-instruct"]

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert not state.pack_state(pack).installed, "model present, package gone"


def test_installed_needs_every_model_but_usable_needs_one(monkeypatch,
                                                          user_data_dir):
    """The four embedding models are alternatives, not parts of one thing.

    A learner who downloaded the 90 MB English model and skipped the three
    multilingual ones has a working TextEmbedding node -- so the pack is
    USABLE. It is not INSTALLED, and the Package Center still shows three
    models to download, which is the honest thing to show.
    """
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    pack = get_pack("sentence-embeddings")

    nothing = state.pack_state(pack)
    assert not nothing.installed
    assert not nothing.usable

    _install_hf("sentence-embeddings", "all-MiniLM-L6-v2")
    partial = state.pack_state(pack)
    assert partial.usable, "one downloaded model is enough to run a node"
    assert not partial.installed, "three of the four are still missing"

    for item_id in ("paraphrase-multilingual-MiniLM-L12-v2",
                    "bge-small-zh-v1.5", "multilingual-e5-small"):
        _install_hf("sentence-embeddings", item_id)
    assert state.pack_state(pack).installed

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    without_packages = state.pack_state(pack)
    assert not without_packages.usable, "models alone cannot run the node"
    assert not without_packages.installed


def test_a_pack_with_nothing_to_download_is_usable_once_its_packages_are_there(
        monkeypatch):
    """Nothing to download means nothing to wait for. No shipped pack is
    pip-only today (gpu-torch has no items but answers to the torch build
    instead), so the rule is pinned against a pack built here."""
    pip_only = Pack(pack_id="pip-only", title="Pip only", description="",
                    pip=("something",), probe_modules=("something",), items=(),
                    depends_on=(), install_mode="live")

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    assert state.pack_state(pip_only).usable
    assert state.pack_state(pip_only).installed

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert not state.pack_state(pip_only).usable


def test_gpu_torch_usable_follows_torch_variant(monkeypatch):
    """gpu-torch has no items, so the general "any item present" rule would
    call it usable on a CPU-only box. It is not: for this pack alone, usable
    means the accelerated build is the one in place, exactly like installed."""
    monkeypatch.setattr(torch, "__version__", "2.6.0+cpu")
    cpu = state.pack_state(get_pack("gpu-torch"))

    assert not cpu.usable
    assert not cpu.installed

    monkeypatch.setattr(torch, "__version__", "2.11.0+cu128")
    assert state.pack_state(get_pack("gpu-torch")).usable


def test_blocked_by_lists_unready_dependencies(monkeypatch):
    """RAG retrieves before it generates. With the embedder's packages
    missing, the UI has to say "install that one first" rather than offer an
    install that cannot work."""
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    assert state.pack_state(get_pack("rag")).blocked_by == ("sentence-embeddings",)
    assert state.pack_state(get_pack("sentence-embeddings")).blocked_by == ()

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    assert state.pack_state(get_pack("rag")).blocked_by == ()


def test_probe_all_covers_the_whole_catalog():
    probed = state.probe_all()

    assert set(probed) == {pack.pack_id for pack in iter_packs()}
    assert probed["rag"].pack_id == "rag"


# ── caching ──────────────────────────────────────────────────────────────


def test_probe_all_is_cached_until_invalidate(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name: calls.append(name) or None)

    first = state.probe_all()
    per_probe = len(calls)
    assert per_probe > 0, "nothing was probed at all"

    assert state.probe_all()["rag"] == first["rag"]
    assert len(calls) == per_probe, "the second poll re-probed"

    state.invalidate()
    state.probe_all()
    assert len(calls) == 2 * per_probe, "invalidate did not drop the cache"


def test_the_cache_cannot_be_edited_by_a_caller():
    """Handing out the live dict would let one route's bookkeeping become the
    next route's answer. Both ways out of ``probe_all`` have to hand back a
    copy -- the answer it just computed, and the one it had already."""
    state.probe_all()["rag"] = "nonsense"  # the freshly computed answer
    state.probe_all()["rag"] = "nonsense"  # the one served from the cache

    assert state.probe_all()["rag"].pack_id == "rag"


def test_invalidate_flushes_the_import_system(monkeypatch):
    """A pack installed a second ago is on disk but not in the import
    system's cached directory listings; without this, "installed" packages
    keep reading as missing until the server restarts."""
    flushed: list[bool] = []
    monkeypatch.setattr(importlib, "invalidate_caches",
                        lambda: flushed.append(True))

    state.invalidate()

    assert flushed == [True]


def test_an_invalidate_during_a_probe_is_not_cached(monkeypatch):
    """An install finishing mid-poll must not have its result overwritten by
    the answer the poll started computing before it."""
    def invalidating_probe(name):
        state.invalidate()
        return None

    monkeypatch.setattr(importlib.util, "find_spec", invalidating_probe)
    state.probe_all()

    calls: list[str] = []
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name: calls.append(name) or object())
    state.probe_all()

    assert calls, "the answer computed across an invalidate was cached anyway"


# ── what a node sees ─────────────────────────────────────────────────────


def test_require_pack_raises_pack_missing_error_with_id_suffix(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    state.invalidate()

    with pytest.raises(PackMissingError) as caught:
        packs.require_pack("rag")

    message = str(caught.value)
    assert caught.value.pack_id == "rag"
    assert message.endswith("(pack=rag)")
    assert message.startswith("RAG stack is not installed.")
    assert "Package Center" in message
    assert "graph runs never download" in message


def test_pack_available_without_item_is_true_when_any_item_present(
        monkeypatch, user_data_dir):
    """A node that takes whichever embedding model the user chose only needs
    ONE of the four to be there."""
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    state.invalidate()
    assert not packs.pack_available("sentence-embeddings")

    _install_hf("sentence-embeddings", "bge-small-zh-v1.5")
    state.invalidate()

    assert packs.pack_available("sentence-embeddings")
    assert not state.probe_all()["sentence-embeddings"].installed


def test_pack_available_with_item_checks_that_item(monkeypatch, user_data_dir):
    """A SELECT option naming one model has to be gated on THAT model. Being
    able to run English embeddings says nothing about the Chinese one."""
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    _install_hf("sentence-embeddings", "all-MiniLM-L6-v2")
    state.invalidate()

    assert packs.pack_available("sentence-embeddings", "all-MiniLM-L6-v2")
    assert not packs.pack_available("sentence-embeddings", "bge-small-zh-v1.5")
    assert not packs.pack_available("sentence-embeddings", "no-such-model")
    assert not packs.pack_available("no-such-pack", "all-MiniLM-L6-v2")

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    state.invalidate()
    assert not packs.pack_available("sentence-embeddings", "all-MiniLM-L6-v2"), (
        "the model is downloaded but sentence-transformers is gone")


def test_require_pack_with_item_names_the_model_and_keeps_suffix(
        monkeypatch, user_data_dir):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    _install_hf("sentence-embeddings", "all-MiniLM-L6-v2")
    state.invalidate()

    assert packs.require_pack("sentence-embeddings", "all-MiniLM-L6-v2") is None

    with pytest.raises(PackMissingError) as caught:
        packs.require_pack("sentence-embeddings", "bge-small-zh-v1.5")

    message = str(caught.value)
    assert caught.value.pack_id == "sentence-embeddings"
    assert message.endswith("(pack=sentence-embeddings)")
    assert message.startswith(
        "Model 'bge-small-zh-v1.5' from the Sentence embeddings pack "
        "is not downloaded.")
    assert "Package Center" in message
    assert "graph runs never download" in message


def test_require_pack_with_an_unknown_item_still_names_the_pack(
        monkeypatch, user_data_dir):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    state.invalidate()

    with pytest.raises(PackMissingError) as caught:
        packs.require_pack("sentence-embeddings", "no-such-model")

    assert caught.value.pack_id == "sentence-embeddings"
    assert str(caught.value).endswith("(pack=sentence-embeddings)")


@pytest.mark.parametrize("value, expected", [
    ("rag", ("rag", None)),
    ("sentence-embeddings", ("sentence-embeddings", None)),
    ("sentence-embeddings:all-MiniLM-L6-v2",
     ("sentence-embeddings", "all-MiniLM-L6-v2")),
])
def test_parse_requirement_shapes(value, expected):
    assert packs.parse_requirement(value) == expected


@pytest.mark.parametrize("value", [
    pytest.param("", id="empty"),
    pytest.param(":", id="both-empty"),
    pytest.param(":item", id="no-pack"),
    pytest.param("pack:", id="no-item"),
    pytest.param("pack:item:extra", id="two-colons"),
    pytest.param("  ", id="whitespace"),
])
def test_parse_requirement_rejects_malformed_values(value):
    """``option_packs`` values are written by node authors by hand. A typo has
    to fail loudly at the point it is read, not silently gate on a pack id
    that is the empty string."""
    with pytest.raises(ValueError):
        packs.parse_requirement(value)


def test_an_unknown_pack_id_is_never_available(monkeypatch):
    assert not packs.pack_available("no-such-pack")

    with pytest.raises(PackMissingError) as caught:
        packs.require_pack("no-such-pack")

    assert caught.value.pack_id == "no-such-pack"


def test_require_pack_returns_quietly_once_the_pack_is_there(user_data_dir):
    _install_glove()
    state.invalidate()

    assert packs.pack_available("word-vectors")
    assert packs.require_pack("word-vectors") is None


def test_model_dir_and_asset_path_read_sentinels(user_data_dir):
    assert packs.model_dir("Qwen/Qwen2.5-0.5B-Instruct") is None
    assert packs.asset_path("word-vectors", "glove-wiki-gigaword-50.gz") is None

    snapshot = _install_qwen()
    glove = _install_glove()

    assert packs.model_dir("Qwen/Qwen2.5-0.5B-Instruct") == snapshot
    assert packs.asset_path("word-vectors", "glove-wiki-gigaword-50.gz") == glove


def test_model_dir_and_asset_path_do_not_guess(user_data_dir):
    """A node that gets a path back may open it without checking. Nothing
    that was not actually downloaded may come back as a path."""
    _install_qwen()

    assert packs.model_dir("nobody/never-published") is None
    assert packs.asset_path("word-vectors", "not-the-file.gz") is None
    assert packs.asset_path("no-such-pack", "glove-wiki-gigaword-50.gz") is None
    assert packs.asset_path("rag", "glove-wiki-gigaword-50.gz") is None


def test_model_dir_ignores_a_stale_download(user_data_dir):
    snapshot = _install_qwen()
    shutil.rmtree(snapshot)

    assert packs.model_dir("Qwen/Qwen2.5-0.5B-Instruct") is None
