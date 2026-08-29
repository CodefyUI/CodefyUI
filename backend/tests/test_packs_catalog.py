"""The Package Center allowlist, and the paths its state lives at.

The catalog is pure data, and data drifts. The pip spec here has to stay
equal to the ``llm-sentence`` extra in ``backend/pyproject.toml`` -- one is
what the in-app installer runs, the other is what a source install gets, and
nothing notices when the two describe different versions. Every declared
dependency has to name a pack that exists. And every path has to keep
honouring ``CODEFYUI_USER_DATA_DIR``, or a dev clone starts writing model
snapshots into the OS-wide cache it was isolated from.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.config import _user_data_root
from app.core.asset_cache import cache_dir
from app.core.packs import PackMissingError
from app.core.packs.catalog import (
    CATALOG,
    PACK_IDS,
    PYPROJECT_EXTRA_FOR_PACK,
    ModelItem,
    Pack,
    _validate,
    find_pack,
    get_item,
    get_pack,
    iter_packs,
)
from app.core.packs.paths import (
    asset_dir,
    control_dir,
    hf_cache_dir,
    job_log_dir,
    last_restart_file,
    pending_restart_file,
    sentinel_dir,
    sentinel_path,
)
from app.core.plugin_loader import tomllib  # stdlib on 3.11+, tomli on 3.10

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

#: Every path helper that takes no argument, so one test can call them all.
_PATH_HELPERS = (
    hf_cache_dir,
    asset_dir,
    sentinel_dir,
    control_dir,
    pending_restart_file,
    last_restart_file,
    job_log_dir,
)


# ── catalog ──────────────────────────────────────────────────────────────


def test_pip_specs_mirror_pyproject_extra():
    """The installer and `pip install -e .[llm-sentence]` must agree.

    They are two spellings of one decision: which sentence-transformers a
    CodefyUI install is tested against. A version bump applied to only one
    of them means the Package Center installs something the test suite has
    never run.
    """
    extras = tomllib.loads(
        _PYPROJECT.read_text(encoding="utf-8"))["project"]["optional-dependencies"]

    assert tuple(extras["llm-sentence"]) == get_pack("sentence-embeddings").pip

    for pack_id, extra_name in PYPROJECT_EXTRA_FOR_PACK.items():
        assert extra_name in extras, f"pyproject has no '{extra_name}' extra"
        assert tuple(extras[extra_name]) == get_pack(pack_id).pip


def test_pack_ids_are_unique_and_known():
    ids = [pack.pack_id for pack in iter_packs()]

    assert ids == ["sentence-embeddings", "word-vectors", "rag", "gpu-torch"]
    assert len(set(ids)) == len(ids)
    assert PACK_IDS == frozenset(ids)


def test_every_dependency_resolves():
    for pack in iter_packs():
        for dep in pack.depends_on:
            assert find_pack(dep) is not None, (
                f"pack '{pack.pack_id}' depends on unknown pack '{dep}'")


def test_rag_depends_on_sentence_embeddings():
    """RAG retrieves before it generates, so the embedder is not optional."""
    assert get_pack("rag").depends_on == ("sentence-embeddings",)


def test_install_modes_are_valid():
    assert {pack.pack_id for pack in iter_packs()
            if pack.install_mode == "restart"} == {"gpu-torch"}
    assert all(pack.install_mode in {"live", "restart"} for pack in iter_packs())


def test_hf_items_have_repo_ids_and_asset_items_have_urls():
    for pack in iter_packs():
        for item in pack.items:
            assert item.kind in {"hf", "asset"}
            if item.kind == "hf":
                assert item.repo_id, f"{item.item_id} is an hf item with no repo_id"
                assert item.revision
            else:
                assert item.url, f"{item.item_id} is an asset item with no url"
                assert item.filename, f"{item.item_id} is an asset item with no filename"
            assert item.approx_bytes > 0
            assert item.license


def test_every_asset_item_records_a_digest():
    """No asset SHIPS unverified.

    An ``hf`` item is content-addressed by the hub, which checks it. A plain
    HTTPS asset has nothing behind it but this string: without a digest the
    installer has to fall back to ``allow_unverified=True`` and write whatever
    arrives into the cache, which is a very long way to spell "trust the
    network". Recording one is a two-minute job -- run

        CODEFYUI_PACK_NETWORK_TESTS=1 pytest tests/test_packs_network.py -q

    and the GloVe test prints the digest of anything that has none -- so this
    is a gate rather than an aspiration.
    """
    for pack in iter_packs():
        for item in pack.items:
            if item.kind != "asset":
                continue
            assert item.sha256, (
                f"asset item {item.item_id!r} in pack {pack.pack_id!r} has no "
                f"sha256; download it once and record the digest it reports")
            assert len(item.sha256) == 64, item.sha256
            assert item.sha256 == item.sha256.lower().strip()
            assert all(character in "0123456789abcdef"
                       for character in item.sha256), item.sha256


def test_item_ids_unique_within_pack():
    for pack in iter_packs():
        ids = [item.item_id for item in pack.items]
        assert len(set(ids)) == len(ids), f"duplicate item id in '{pack.pack_id}'"


def test_get_pack_and_get_item_raise_on_unknown_ids():
    with pytest.raises(KeyError):
        get_pack("no-such-pack")
    assert find_pack("no-such-pack") is None

    embeddings = get_pack("sentence-embeddings")
    assert get_item(embeddings, "all-MiniLM-L6-v2").repo_id == (
        "sentence-transformers/all-MiniLM-L6-v2")
    with pytest.raises(KeyError):
        get_item(embeddings, "no-such-item")


# ── the validator (it runs at import time, so it must be right) ──────────


def _item(item_id: str = "m", **overrides) -> ModelItem:
    fields = dict(item_id=item_id, kind="hf", approx_bytes=1,
                  license="MIT", repo_id="org/model")
    fields.update(overrides)
    return ModelItem(**fields)


def _pack(pack_id: str = "a", **overrides) -> Pack:
    fields = dict(pack_id=pack_id, title=pack_id, description="", pip=(),
                  probe_modules=(), items=(), depends_on=(),
                  install_mode="live")
    fields.update(overrides)
    return Pack(**fields)


def test_validate_accepts_the_shipped_catalog():
    """The helpers above must build packs the validator would accept, or
    every rejection below could be passing for the wrong reason."""
    _validate(CATALOG)
    _validate((_pack(),))


def test_validate_rejects_dependency_cycle():
    cycle = (_pack("a", depends_on=("b",)), _pack("b", depends_on=("a",)))

    with pytest.raises(ValueError, match="cycle"):
        _validate(cycle)


@pytest.mark.parametrize("packs", [
    pytest.param((_pack("a"), _pack("a")), id="duplicate-pack-id"),
    pytest.param((_pack("a", depends_on=("a",)),), id="self-dependency"),
    pytest.param((_pack("a", depends_on=("ghost",)),), id="unknown-dependency"),
    pytest.param((_pack(install_mode="sometimes"),), id="bad-install-mode"),
    pytest.param((_pack(items=(_item(kind="wat"),)),), id="unknown-item-kind"),
    pytest.param((_pack(items=(_item(repo_id=None),)),), id="hf-without-repo-id"),
    pytest.param((_pack(items=(_item(kind="asset", repo_id=None,
                                     filename="f.gz"),)),), id="asset-without-url"),
    pytest.param((_pack(items=(_item(kind="asset", repo_id=None,
                                     url="https://x/f.gz"),)),),
                 id="asset-without-filename"),
    # Distinct repo ids on purpose: the repo-id-uniqueness rule runs FIRST,
    # so two items sharing ``_item``'s default repo would be rejected for
    # that instead and this case would never reach the rule it is named for.
    pytest.param((_pack(items=(_item("dup"),
                               _item("dup", repo_id="org/other"))),),
                 id="duplicate-item-id"),
    pytest.param((_pack("a", items=(_item("m"),)),
                  _pack("b", items=(_item("n"),))),
                 id="repo-id-shared-by-two-packs"),
    pytest.param((_pack("a", items=(_item("m"), _item("n"))),),
                 id="repo-id-shared-inside-one-pack"),
])
def test_validate_rejects_malformed_packs(packs):
    with pytest.raises(ValueError):
        _validate(packs)


def test_repo_ids_are_unique_across_the_whole_catalog():
    """Two items may not name the same Hugging Face repo.

    Removing an item deletes the repo FOLDER, which is where the bytes
    actually are -- one directory shared by every revision of one repo. If
    two items pointed at the same repo, uninstalling one model would delete
    the other pack's model out from under it, and the other pack would go on
    reporting itself installed because its sentinel is still there.
    """
    repo_ids = [item.repo_id for pack in CATALOG for item in pack.items
                if item.kind == "hf"]

    assert len(repo_ids) == len(set(repo_ids)), sorted(repo_ids)


def test_pack_missing_error_names_the_pack_in_its_message():
    """The frontend reads the id back off the message to offer an install."""
    err = PackMissingError("word-vectors", "WordVector needs the GloVe table")

    assert err.pack_id == "word-vectors"
    assert str(err).endswith("(pack=word-vectors)")


# ── paths ────────────────────────────────────────────────────────────────


def _clear_path_caches() -> None:
    """Neither user-directory helper memoizes today. If one ever starts, a
    test that pins the env var would otherwise read -- and leave behind --
    a root belonging to some other test."""
    for fn in (cache_dir, _user_data_root):
        clear = getattr(fn, "cache_clear", None)
        if clear is not None:
            clear()


@pytest.fixture
def user_data_dir(tmp_path, monkeypatch):
    """Point every user-directory helper at a throwaway root for one test."""
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path))
    _clear_path_caches()
    yield tmp_path
    _clear_path_caches()


def test_paths_honour_user_data_dir(user_data_dir):
    """Dev-mode isolation is inherited, never re-implemented: these helpers
    delegate to ``cache_dir()`` / ``_user_data_root()`` rather than reading
    the environment variable themselves."""
    paths = [helper() for helper in _PATH_HELPERS]
    paths.append(sentinel_path("rag", "qwen2.5-0.5b-instruct"))

    for path in paths:
        assert path.is_relative_to(user_data_dir), f"{path} escapes the data dir"

    assert sentinel_path("rag", "qwen2.5-0.5b-instruct") == (
        sentinel_dir() / "rag__qwen2.5-0.5b-instruct.json")
    assert pending_restart_file().parent == control_dir()
    assert last_restart_file().parent == control_dir()
    assert job_log_dir().parent == control_dir()
    assert hf_cache_dir().parent == asset_dir()


def test_path_helpers_do_not_create_directories(user_data_dir):
    """Computing a path and committing to it on disk are separate acts; the
    caller mkdirs when it actually has something to write."""
    for path in (hf_cache_dir(), sentinel_dir(), control_dir(), job_log_dir()):
        assert not path.exists(), f"{path} was created just by asking for it"

    # The one exception, pinned rather than quietly omitted: ``asset_dir()``
    # IS the cache root, and ``asset_cache.cache_dir()`` mkdirs it as it
    # answers. The day that helper goes lazy this fails, which is the
    # signal to move ``asset_dir()`` up into the loop -- the guarantee
    # above would then hold for every path in this module.
    assert asset_dir().exists(), (
        "asset_cache.cache_dir() no longer creates the cache root")


def test_paths_never_touch_hf_home(user_data_dir, monkeypatch):
    """``HF_HOME`` is the whole machine's Hugging Face cache. CodefyUI keeps
    its snapshots in its own cache dir and leaves that variable to its
    owner -- neither reading it nor, worse, setting it for the process."""
    monkeypatch.delenv("HF_HOME", raising=False)

    for helper in _PATH_HELPERS:
        helper()
    sentinel_path("sentence-embeddings", "all-MiniLM-L6-v2")

    assert "HF_HOME" not in os.environ
