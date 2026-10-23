"""Offline pack bundles: ``cdui packs export`` and ``install --from``.

Every bundle here is built in ``tmp_path`` from items "installed" by writing
the files and the sentinel a real download leaves. Nothing reaches the
network or a package manager: ``runner.run_argv`` (the ``pip download``) and
``runner.run_pip`` (the install) are replaced wherever a test gets that far.

What these pin:

* a round trip puts the bytes where a download would have, with the same
  sentinel, so ``state`` reports the item present;
* the catalog stays the allowlist: a manifest naming a pack, an item, a
  revision or a file the catalog does not know is refused before anything is
  copied, and so is a path that leaves the bundle;
* a damaged file fails the install with nothing half-written;
* the packages install with ``--no-index --find-links <wheels>`` under the
  same constraints file a live install uses, and only from wheels built for
  this platform.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.core.auth import TOKEN_HEADER, session_token
from app.core.packs import bundle, download, flows, runner, state
from app.core.packs.catalog import ModelItem, Pack, get_pack
from app.core.packs.errors import (
    BundleInvalid,
    PackCancelled,
    PackInstallError,
    PackInsufficientDisk,
    PackNeedsRestart,
)
from app.core.packs.paths import asset_dir, hf_cache_dir, sentinel_path
from app.core.packs.service import PackBusy, PackService
from app.main import app
from tests.test_packs_service import drain, types_of

BASE_URL = "http://127.0.0.1:8000"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
GLOVE_BYTES = b"pretend this is a gzip of GloVe"

HF_PACK = Pack(
    pack_id="test-hf", title="Test HF", description="d", pip=(),
    probe_modules=(), depends_on=(), install_mode="live",
    items=(ModelItem(item_id="tiny", kind="hf", repo_id="org/tiny",
                     approx_bytes=10, license="MIT"),
           ModelItem(item_id="other", kind="hf", repo_id="org/other",
                     approx_bytes=10, license="MIT")),
)
ASSET_PACK = Pack(
    pack_id="test-asset", title="Test asset", description="d", pip=(),
    probe_modules=(), depends_on=(), install_mode="live",
    items=(ModelItem(item_id="table", kind="asset",
                     url="https://example.invalid/table.gz",
                     filename="table.gz",
                     sha256=hashlib.sha256(GLOVE_BYTES).hexdigest(),
                     approx_bytes=10, license="PDDL-1.0"),),
)
#: Packages that are importable (``json``), so an export may run.
PIP_PACK = Pack(
    pack_id="test-pip", title="Test pip", description="d",
    pip=("tinypkg>=1",), probe_modules=("json",), depends_on=(),
    install_mode="live", items=(),
)


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path / "data"))
    state.invalidate()
    yield tmp_path
    state.invalidate()


def _hf_snapshot(item: ModelItem) -> Path:
    return (hf_cache_dir() / ("models--" + item.repo_id.replace("/", "--"))
            / "snapshots" / COMMIT)


def install_hf(pack: Pack, item: ModelItem, files: dict[str, bytes]) -> Path:
    """Leave what ``download.download_hf_item`` leaves."""
    snapshot = _hf_snapshot(item)
    for rel, data in files.items():
        (snapshot / rel).parent.mkdir(parents=True, exist_ok=True)
        (snapshot / rel).write_bytes(data)
    state.write_sentinel(pack.pack_id, item.item_id, {
        "schema": 1, "pack_id": pack.pack_id, "item_id": item.item_id,
        "kind": "hf", "repo_id": item.repo_id, "revision": item.revision,
        "snapshot_dir": str(snapshot), "bytes": 1, "at": "x",
    })
    state.invalidate()
    return snapshot


def install_asset(pack: Pack, item: ModelItem, data: bytes) -> Path:
    path = asset_dir() / item.filename
    path.write_bytes(data)
    state.write_sentinel(pack.pack_id, item.item_id, {
        "schema": 1, "pack_id": pack.pack_id, "item_id": item.item_id,
        "kind": "asset", "url": item.url, "path": str(path),
        "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
        "at": "x",
    })
    state.invalidate()
    return path


def forget(pack: Pack, item: ModelItem) -> None:
    """Remove an item from this machine, as ``cdui packs remove`` would."""
    flows.remove_item(pack, item.item_id)
    assert not state.item_state(pack, item).present


def export(pack, dest, item_ids=None, **kwargs):
    events: list[dict] = []
    outcome = bundle.export_pack(pack, item_ids, dest, emit=events.append,
                                 cancel_check=lambda: False, **kwargs)
    return outcome, events


def install(pack, source, item_ids=None, *, cancel=False):
    events: list[dict] = []
    outcome = flows.install_pack_from_bundle(
        pack, item_ids, source, emit=events.append,
        cancel_check=lambda: cancel)
    return outcome, events


def read_manifest(root: Path) -> dict:
    return json.loads((root / bundle.MANIFEST_NAME).read_text(encoding="utf-8"))


def write_manifest(root: Path, data: dict) -> None:
    (root / bundle.MANIFEST_NAME).write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def hf_bundle(tmp_path) -> Path:
    """A bundle of ``test-hf/tiny``, with the item then removed locally."""
    item = HF_PACK.items[0]
    install_hf(HF_PACK, item, {"config.json": b'{"a": 1}',
                               "1_Pooling/config.json": b"{}",
                               "model.safetensors": b"w" * 3000})
    dest = tmp_path / "usb" / "tiny"
    export(HF_PACK, dest)
    forget(HF_PACK, item)
    return dest


# ── export ───────────────────────────────────────────────────────────────


def test_export_writes_the_items_and_a_manifest_last(tmp_path):
    item = HF_PACK.items[0]
    install_hf(HF_PACK, item, {"config.json": b"{}",
                               "sub/model.safetensors": b"weights"})
    dest = tmp_path / "bundle"

    outcome, events = export(HF_PACK, dest)

    assert outcome.items == ("tiny",)
    assert outcome.wheels == () and outcome.skipped == ()
    manifest = read_manifest(dest)
    assert manifest["format"] == bundle.BUNDLE_FORMAT
    assert manifest["schema"] == bundle.BUNDLE_SCHEMA
    assert manifest["pack_id"] == "test-hf"
    assert manifest["catalog_version"] == bundle.catalog_version()
    assert manifest["platform"] == bundle.current_platform()
    [entry] = manifest["items"]
    assert entry["repo_id"] == "org/tiny" and entry["revision"] == "main"
    assert entry["commit"] == COMMIT
    assert {f["path"]: f["sha256"] for f in entry["files"]} == {
        "config.json": hashlib.sha256(b"{}").hexdigest(),
        "sub/model.safetensors": hashlib.sha256(b"weights").hexdigest(),
    }
    assert (dest / "items/tiny/sub/model.safetensors").read_bytes() == b"weights"
    assert "export:tiny" in [e.get("step") for e in events]
    assert events[-2]["type"] == "progress" and events[-2]["percent"] == 100.0
    assert not list(dest.glob(".manifest.json.tmp-*"))


def test_export_refuses_what_it_cannot_bundle(tmp_path):
    with pytest.raises(ValueError, match="restart mode"):
        export(get_pack("gpu-torch"), tmp_path / "a")
    with pytest.raises(ValueError, match="not installed here"):
        export(HF_PACK, tmp_path / "b", ["tiny"])
    with pytest.raises(ValueError, match="nothing of"):
        export(HF_PACK, tmp_path / "c")
    with pytest.raises(ValueError, match="has no item"):
        export(HF_PACK, tmp_path / "d", ["nope"])

    install_hf(HF_PACK, HF_PACK.items[0], {"config.json": b"{}"})
    busy = tmp_path / "e"
    busy.mkdir()
    (busy / "keep.txt").write_text("x")
    with pytest.raises(ValueError, match="not an empty directory"):
        export(HF_PACK, busy)

    unready = Pack(**{**PIP_PACK.__dict__, "probe_modules": ("no_such_mod_x",)})
    with pytest.raises(ValueError, match="packages are not installed"):
        export(unready, tmp_path / "f")


def test_export_refuses_a_disk_that_cannot_hold_it(tmp_path, monkeypatch):
    install_hf(HF_PACK, HF_PACK.items[0], {"config.json": b"{}"})
    monkeypatch.setattr(bundle.shutil, "disk_usage",
                        lambda path: type("U", (), {"free": 0})())
    with pytest.raises(PackInsufficientDisk):
        export(HF_PACK, tmp_path / "bundle")


def test_export_refuses_an_asset_that_no_longer_matches_the_catalog(
        tmp_path, monkeypatch):
    item = ASSET_PACK.items[0]
    install_asset(ASSET_PACK, item, GLOVE_BYTES)
    (asset_dir() / item.filename).write_bytes(b"changed on disk")
    with pytest.raises(PackInstallError, match="does not match"):
        export(ASSET_PACK, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_export_downloads_the_wheels_for_this_platform(tmp_path, monkeypatch):
    calls: list[list[str]] = []

    def fake_run_argv(argv, *, emit, cancel_check, cwd, tail=None,
                      conflict=None):
        calls.append(list(argv))
        dest = Path(argv[argv.index("--dest") + 1])
        (dest / "tinypkg-1.0-py3-none-any.whl").write_bytes(b"wheel")
        return 0

    monkeypatch.setattr(bundle, "wheel_requirements",
                        lambda pack: (["tinypkg==1.0"], ["torch==2.9.1+xpu"]))
    monkeypatch.setattr(runner, "run_argv", fake_run_argv)

    outcome, events = export(PIP_PACK, tmp_path / "bundle")

    [argv] = calls
    assert argv[-1] == "tinypkg==1.0"
    assert "--no-deps" in argv and "--only-binary=:all:" in argv
    assert outcome.wheels == ("tinypkg-1.0-py3-none-any.whl",)
    assert outcome.skipped == ("torch==2.9.1+xpu",)
    manifest = read_manifest(tmp_path / "bundle")
    assert manifest["wheels"] == [{
        "path": "tinypkg-1.0-py3-none-any.whl", "size": 5,
        "sha256": hashlib.sha256(b"wheel").hexdigest()}]
    assert manifest["skipped_packages"] == ["torch==2.9.1+xpu"]
    assert any("torch==2.9.1+xpu is not bundled" in e.get("line", "")
               for e in events)


def test_export_without_wheels_skips_the_download(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "run_argv",
                        lambda *a, **k: pytest.fail("pip download ran"))
    outcome, _ = export(PIP_PACK, tmp_path / "bundle", download_wheels=False)
    assert outcome.wheels == ()
    assert read_manifest(tmp_path / "bundle")["wheels"] == []


def test_a_failed_wheel_download_leaves_no_manifest(tmp_path, monkeypatch):
    def failing(argv, *, emit, cancel_check, cwd, tail=None, conflict=None):
        tail.append("ERROR: No matching distribution found for tinypkg==1.0")
        return 1

    monkeypatch.setattr(bundle, "wheel_requirements",
                        lambda pack: (["tinypkg==1.0"], []))
    monkeypatch.setattr(runner, "run_argv", failing)
    with pytest.raises(PackInstallError) as excinfo:
        export(PIP_PACK, tmp_path / "bundle")
    assert "No matching distribution" in excinfo.value.hint
    # What the failed export wrote is gone, so a retry can use the same path.
    assert not (tmp_path / "bundle").exists()

    # A directory that was there before (a USB stick's root) stays, empty.
    stick = tmp_path / "stick"
    stick.mkdir()
    with pytest.raises(PackInstallError):
        export(PIP_PACK, stick)
    assert stick.is_dir() and not any(stick.iterdir())


def test_pip_download_uses_this_interpreter(monkeypatch, tmp_path):
    monkeypatch.setattr(bundle.importlib.util, "find_spec",
                        lambda name: object())
    argv = bundle.pip_download_argv(["a==1"], tmp_path)
    assert argv[:4] == [sys.executable, "-m", "pip", "download"]

    # A uv venv has no pip of its own: uv runs one on this interpreter.
    monkeypatch.setattr(bundle.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(runner, "find_uv", lambda: "/bin/uv")
    argv = bundle.pip_download_argv(["a==1"], tmp_path)
    assert argv[:7] == ["/bin/uv", "tool", "run", "--python", sys.executable,
                        "pip", "download"]
    assert argv[-3:] == ["--dest", str(tmp_path), "a==1"]


class _Dist:
    def __init__(self, name, version, requires=(), direct_url=None):
        self.metadata = {"Name": name}
        self.version = version
        self.requires = list(requires)
        self._direct_url = direct_url

    def read_text(self, filename):
        return self._direct_url if filename == "direct_url.json" else None


def test_wheel_requirements_bundles_what_the_pack_adds_to_the_base(monkeypatch):
    editable = '{"url": "file:///x", "dir_info": {"editable": true}}'
    dists = [
        _Dist("codefyui-backend", "3.2.0", direct_url=editable,
              requires=["torch>=2.5", "numpy>=1",
                        'sentence-transformers>=3; extra == "llm-sentence"']),
        _Dist("torch", "2.9.1+cu130", requires=["filelock"]),
        _Dist("filelock", "3.0"),
        _Dist("numpy", "2.1.0"),
        _Dist("sentence-transformers", "5.1.0",
              requires=["transformers>=4", "torch>=2", "numpy",
                        'scikit-learn; extra == "train"',
                        "special-kernels", "from-url"]),
        _Dist("transformers", "4.57.0", requires=["tokenizers", "numpy"]),
        _Dist("tokenizers", "0.22.0"),
        _Dist("scikit-learn", "1.5.0"),
        _Dist("special-kernels", "1.0+cu130"),
        _Dist("from-url", "0.1", direct_url='{"url": "https://x/y.whl"}'),
    ]
    monkeypatch.setattr(bundle.importlib.metadata, "distributions",
                        lambda: iter(dists))
    pack = Pack(**{**PIP_PACK.__dict__,
                   "pip": ("sentence-transformers>=3.0,<6",)})

    pins, skipped = bundle.wheel_requirements(pack)

    assert pins == ["sentence-transformers==5.1.0", "tokenizers==0.22.0",
                    "transformers==4.57.0"]
    assert skipped == ["from-url==0.1", "special-kernels==1.0+cu130"]


# ── install: the round trip ──────────────────────────────────────────────


def test_an_hf_bundle_installs_where_a_download_would_have(hf_bundle):
    item = HF_PACK.items[0]

    outcome, events = install(HF_PACK, hf_bundle)

    assert outcome.items_done == ("tiny",) and not outcome.pip_installed
    probe = state.item_state(HF_PACK, item)
    assert probe.present
    assert probe.snapshot_dir == _hf_snapshot(item)
    assert (probe.snapshot_dir / "1_Pooling/config.json").read_bytes() == b"{}"
    assert (probe.snapshot_dir / "model.safetensors").stat().st_size == 3000
    repo = probe.snapshot_dir.parent.parent
    assert (repo / "refs" / "main").read_text(encoding="utf-8") == COMMIT
    sentinel = state.read_sentinel(sentinel_path("test-hf", "tiny"))
    assert sentinel["source"] == "bundle" and sentinel["bytes"] == 3010
    assert [e["step"] for e in events if e["type"] == "step_started"] == [
        "copy:tiny"]
    assert not list(probe.snapshot_dir.parent.glob(".*partial*"))


def test_an_asset_bundle_is_verified_against_the_catalog(tmp_path):
    item = ASSET_PACK.items[0]
    install_asset(ASSET_PACK, item, GLOVE_BYTES)
    dest = tmp_path / "bundle"
    export(ASSET_PACK, dest)
    forget(ASSET_PACK, item)

    install(ASSET_PACK, dest)

    probe = state.item_state(ASSET_PACK, item)
    assert probe.present
    assert probe.snapshot_dir.read_bytes() == GLOVE_BYTES
    assert not (asset_dir() / "table.gz.part").exists()


def test_the_glove_bundle_still_converts(tmp_path, monkeypatch):
    pack = get_pack("word-vectors")
    item = pack.items[0]
    converted: list[Path] = []
    monkeypatch.setattr(flows, "_convert_glove_step",
                        lambda path, *, emit: converted.append(path))
    dest = tmp_path / "bundle"
    (dest / "items" / item.item_id).mkdir(parents=True)
    (dest / "items" / item.item_id / item.filename).write_bytes(b"x")
    write_manifest(dest, {
        "format": bundle.BUNDLE_FORMAT, "schema": 1, "pack_id": pack.pack_id,
        "pip": [], "items": [{
            "item_id": item.item_id, "kind": "asset", "url": item.url,
            "filename": item.filename,
            "files": [{"path": item.filename, "size": 1,
                       "sha256": item.sha256}]}],
    })
    # The catalog's digest is for the real 69 MB file; these bytes fail it.
    with pytest.raises(PackInstallError, match="damaged"):
        install(pack, dest)
    assert converted == []
    assert not state.item_state(pack, item).present


def test_none_installs_what_is_in_the_bundle_and_missing_here(hf_bundle):
    tiny, other = HF_PACK.items
    source = bundle.load_bundle(hf_bundle, HF_PACK)
    assert bundle.select_items(source, None) == [tiny]
    install_hf(HF_PACK, tiny, {"config.json": b"{}"})
    assert bundle.select_items(source, None) == []
    with pytest.raises(BundleInvalid, match="does not contain 'other'"):
        bundle.select_items(source, ["other"])
    with pytest.raises(ValueError, match="has no item"):
        bundle.select_items(source, ["nope"])


def test_an_installed_snapshot_is_replaced_by_the_bundled_one(hf_bundle):
    item = HF_PACK.items[0]
    stale = install_hf(HF_PACK, item, {"stale.bin": b"old"})
    install(HF_PACK, hf_bundle, ["tiny"])
    assert not (stale / "stale.bin").exists()
    assert (stale / "config.json").exists()


# ── install: refusals before anything is touched ─────────────────────────


def _mutate(root: Path, change) -> None:
    data = read_manifest(root)
    change(data)
    write_manifest(root, data)


@pytest.mark.parametrize("change, message", [
    (lambda m: m.update(format="zip"), "not a CodefyUI pack bundle"),
    (lambda m: m.update(schema=2), "schema 2"),
    (lambda m: m.update(pack_id="test-asset"), "for pack 'test-asset'"),
    (lambda m: m.update(pip=["evil==1"]), "made for the packages"),
    (lambda m: m.update(items={}), "no item list"),
    (lambda m: m.update(wheels={}), "wheel list is not a list"),
    (lambda m: m["items"][0].update(item_id="evil"), "does not have"),
    (lambda m: m["items"][0].update(kind="asset"), "is kind 'hf'"),
    (lambda m: m["items"][0].update(revision="v2"), "catalog wants"),
    (lambda m: m["items"][0].update(repo_id="evil/model"), "catalog wants"),
    (lambda m: m["items"][0].update(commit="../x"), "no usable snapshot"),
    (lambda m: m["items"][0].update(files=[]), "lists no files"),
    (lambda m: m["items"][0]["files"].append(dict(m["items"][0]["files"][0])),
     "lists a file twice"),
    (lambda m: m["items"][0]["files"][0].update(path="../../etc/passwd"),
     "unsafe file path"),
    (lambda m: m["items"][0]["files"][0].update(path="/etc/passwd"),
     "unsafe file path"),
    (lambda m: m["items"][0]["files"][0].update(path="a\\b"),
     "unsafe file path"),
    (lambda m: m["items"][0]["files"][0].update(sha256="zz"), "bad sha256"),
    (lambda m: m["items"][0]["files"][0].update(size=-1), "bad size"),
    (lambda m: m["items"].append(dict(m["items"][0])), "in the manifest twice"),
    (lambda m: m.update(wheels=[{"path": "x.tar.gz", "size": 1,
                                 "sha256": "0" * 64}]), "not a wheel"),
    (lambda m: m["items"].__setitem__(0, "tiny"), "not an object"),
    (lambda m: m["items"][0]["files"].__setitem__(0, "f"), "not an object"),
])
def test_a_manifest_the_catalog_does_not_vouch_for_is_refused(
        hf_bundle, change, message):
    _mutate(hf_bundle, change)
    with pytest.raises(BundleInvalid, match=message):
        install(HF_PACK, hf_bundle)
    assert not state.item_state(HF_PACK, HF_PACK.items[0]).present


@pytest.mark.parametrize("change, message", [
    (lambda m: m["items"][0].update(url="https://evil.invalid/t.gz"),
     "different file"),
    (lambda m: m["items"][0]["files"][0].update(path="other.gz"),
     "must be exactly table.gz"),
    (lambda m: m["items"][0]["files"][0].update(sha256="1" * 64),
     "not the one the catalog records"),
])
def test_an_asset_must_be_the_catalog_file(tmp_path, change, message):
    install_asset(ASSET_PACK, ASSET_PACK.items[0], GLOVE_BYTES)
    dest = tmp_path / "bundle"
    export(ASSET_PACK, dest)
    _mutate(dest, change)
    with pytest.raises(BundleInvalid, match=message):
        bundle.load_bundle(dest, ASSET_PACK)


def test_a_bundle_cannot_add_a_pack_or_an_item_to_the_catalog(tmp_path):
    """The real catalog: a bundle for a known pack naming an unknown model."""
    pack = get_pack("sentence-embeddings")
    dest = tmp_path / "bundle"
    dest.mkdir()
    write_manifest(dest, {
        "format": bundle.BUNDLE_FORMAT, "schema": 1, "pack_id": pack.pack_id,
        "pip": list(pack.pip), "items": [{
            "item_id": "evil-model", "kind": "hf", "repo_id": "evil/model",
            "revision": "main", "commit": COMMIT,
            "files": [{"path": "x.bin", "size": 1, "sha256": "0" * 64}]}],
    })
    with pytest.raises(BundleInvalid, match="'evil-model'"):
        bundle.load_bundle(dest, pack)
    with pytest.raises(BundleInvalid, match="restart mode"):
        bundle.load_bundle(dest, Pack(**{**get_pack("gpu-torch").__dict__,
                                         "pack_id": pack.pack_id}))


def test_what_is_not_a_bundle_says_so(tmp_path):
    with pytest.raises(BundleInvalid, match="not a directory"):
        bundle.load_bundle(tmp_path / "nowhere", HF_PACK)
    tmp_path.joinpath("empty").mkdir()
    with pytest.raises(BundleInvalid, match="export did not finish"):
        bundle.load_bundle(tmp_path / "empty", HF_PACK)
    (tmp_path / "empty" / bundle.MANIFEST_NAME).write_text("not json")
    with pytest.raises(BundleInvalid, match="cannot read"):
        bundle.load_bundle(tmp_path / "empty", HF_PACK)
    (tmp_path / "empty" / bundle.MANIFEST_NAME).write_text("[]")
    with pytest.raises(BundleInvalid, match="not a JSON object"):
        bundle.load_bundle(tmp_path / "empty", HF_PACK)


def test_an_oversized_manifest_is_not_read(tmp_path, monkeypatch):
    tmp_path.joinpath("big").mkdir()
    (tmp_path / "big" / bundle.MANIFEST_NAME).write_text("{}")
    monkeypatch.setattr(bundle, "_MAX_MANIFEST_BYTES", 1)
    with pytest.raises(BundleInvalid, match="too large"):
        bundle.load_bundle(tmp_path / "big", HF_PACK)


def test_install_refuses_a_restart_mode_pack(hf_bundle):
    with pytest.raises(ValueError, match="restart mode"):
        install(get_pack("gpu-torch"), hf_bundle)


def test_a_symlink_out_of_the_bundle_is_refused(hf_bundle, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("not yours")
    target = hf_bundle / "items" / "tiny" / "config.json"
    target.unlink()
    os.symlink(secret, target)
    with pytest.raises(BundleInvalid, match="outside the bundle"):
        install(HF_PACK, hf_bundle)
    assert not state.item_state(HF_PACK, HF_PACK.items[0]).present


def test_a_damaged_file_installs_nothing(hf_bundle):
    (hf_bundle / "items/tiny/model.safetensors").write_bytes(b"x" * 3000)

    with pytest.raises(PackInstallError, match="damaged"):
        install(HF_PACK, hf_bundle)

    snapshot = _hf_snapshot(HF_PACK.items[0])
    assert not snapshot.exists()
    assert not list(snapshot.parent.glob(".*")) if snapshot.parent.exists() else True
    assert state.read_sentinel(sentinel_path("test-hf", "tiny")) is None


def test_the_disk_is_checked_before_anything_is_copied(hf_bundle, monkeypatch):
    def full(items):
        raise PackInsufficientDisk("full", needed=10, free=0)

    monkeypatch.setattr(download, "check_disk", full)
    with pytest.raises(PackInsufficientDisk):
        install(HF_PACK, hf_bundle)
    assert not _hf_snapshot(HF_PACK.items[0]).exists()


def test_cancel_stops_the_copy(hf_bundle):
    with pytest.raises(PackCancelled):
        install(HF_PACK, hf_bundle, cancel=True)
    assert not state.item_state(HF_PACK, HF_PACK.items[0]).present


def test_a_cancel_mid_file_unwinds_the_copy(tmp_path):
    source = tmp_path / "src.bin"
    source.write_bytes(b"x" * 10)
    with pytest.raises(PackCancelled):
        bundle.copy_hashed(source, tmp_path / "dst.bin",
                           cancel_check=lambda: True)


# ── install: the packages ────────────────────────────────────────────────


#: Packages that are NOT importable, so the install has to run pip.
UNREADY_PACK = Pack(**{**PIP_PACK.__dict__,
                       "probe_modules": ("no_such_module_for_bundles",)})


@pytest.fixture
def wheel_bundle(tmp_path, monkeypatch) -> Path:
    def fake_run_argv(argv, *, emit, cancel_check, cwd, tail=None,
                      conflict=None):
        dest = Path(argv[argv.index("--dest") + 1])
        (dest / "tinypkg-1.0-py3-none-any.whl").write_bytes(b"wheel")
        return 0

    monkeypatch.setattr(bundle, "wheel_requirements",
                        lambda pack: (["tinypkg==1.0"], []))
    monkeypatch.setattr(runner, "run_argv", fake_run_argv)
    dest = tmp_path / "bundle"
    export(PIP_PACK, dest)
    return dest


@pytest.fixture
def pip_calls(monkeypatch) -> list[dict]:
    calls: list[dict] = []

    def fake_run_pip(specs, *, constraints_path, emit, cancel_check, cwd,
                     tail=None, conflict=None, extra_args=()):
        calls.append({"specs": list(specs), "extra_args": list(extra_args),
                      "constraints": Path(constraints_path).read_text()})
        return calls[-1].get("returncode", 0)

    monkeypatch.setattr(runner, "run_pip", fake_run_pip)
    monkeypatch.setattr(flows, "verify_imports", lambda pack, *, emit: None)
    return calls


def test_packages_install_from_the_wheels_and_never_the_index(
        wheel_bundle, pip_calls):
    outcome, events = install(UNREADY_PACK, wheel_bundle)

    assert outcome.pip_installed
    [call] = pip_calls
    assert call["specs"] == ["tinypkg>=1"], "the catalog's specs, not the bundle's"
    assert call["extra_args"] == ["--no-index", "--find-links",
                                  str(wheel_bundle / "wheels")]
    # The same add-only constraints file a live install runs under.
    assert "\ntorch==" in "\n" + call["constraints"]
    assert "pip" in [e.get("step") for e in events]


def test_ready_packages_need_no_wheels(tmp_path, pip_calls):
    export(PIP_PACK, tmp_path / "bundle", download_wheels=False)
    outcome, _ = install(PIP_PACK, tmp_path / "bundle")
    assert not outcome.pip_installed and pip_calls == []


def test_missing_wheels_are_refused(tmp_path, pip_calls):
    export(PIP_PACK, tmp_path / "bundle", download_wheels=False)
    with pytest.raises(BundleInvalid, match="no wheels"):
        install(UNREADY_PACK, tmp_path / "bundle")
    assert pip_calls == []


def test_wheels_for_another_platform_are_refused(wheel_bundle, pip_calls):
    _mutate(wheel_bundle, lambda m: m.update(platform={
        "sys_platform": "win32", "machine": "amd64", "python": "3.11"}))
    with pytest.raises(BundleInvalid, match="win32/amd64 Python 3.11"):
        install(UNREADY_PACK, wheel_bundle)
    assert pip_calls == []


def test_a_damaged_or_missing_wheel_is_refused(wheel_bundle, pip_calls):
    wheel = wheel_bundle / "wheels" / "tinypkg-1.0-py3-none-any.whl"
    wheel.write_bytes(b"tampered")
    with pytest.raises(PackInstallError, match="does not match"):
        install(UNREADY_PACK, wheel_bundle)
    wheel.unlink()
    with pytest.raises(PackInstallError, match="missing"):
        install(UNREADY_PACK, wheel_bundle)
    assert pip_calls == []


def test_a_resolver_conflict_names_the_offline_command(
        wheel_bundle, monkeypatch):
    def conflicting(specs, *, constraints_path, emit, cancel_check, cwd,
                    tail=None, conflict=None, extra_args=()):
        conflict.set()
        return 1

    monkeypatch.setattr(runner, "run_pip", conflicting)
    with pytest.raises(PackNeedsRestart) as excinfo:
        install(UNREADY_PACK, wheel_bundle)
    assert "--no-index --find-links" in excinfo.value.command


def test_pip_install_argv_puts_the_extra_args_before_the_specs():
    argv = runner.pip_install_argv(
        ["tinypkg>=1"], constraints_path=Path("c.txt"),
        extra_args=("--no-index", "--find-links", "/w"))
    assert argv[-5:] == ["c.txt", "--no-index", "--find-links", "/w",
                         "tinypkg>=1"]


def test_run_argv_names_the_missing_tool(monkeypatch, tmp_path):
    def missing(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(runner.subprocess, "Popen", missing)
    events: list[dict] = []
    assert runner.run_argv(["/no/where/pip"], emit=events.append,
                           cancel_check=lambda: False, cwd=tmp_path) == 127
    assert events[-1]["line"] == ("pip was not found on PATH; install pip "
                                  "and try again")


# ── the service and the route ────────────────────────────────────────────


def _word_vectors_bundle(root: Path) -> Path:
    """A manifest for the real ``word-vectors`` pack. Its bytes are never
    read: the flow is faked wherever this is used."""
    pack = get_pack("word-vectors")
    item = pack.items[0]
    root.mkdir(parents=True, exist_ok=True)
    write_manifest(root, {
        "format": bundle.BUNDLE_FORMAT, "schema": 1, "pack_id": pack.pack_id,
        "pip": [], "items": [{
            "item_id": item.item_id, "kind": "asset", "url": item.url,
            "filename": item.filename,
            "files": [{"path": item.filename, "size": 1,
                       "sha256": item.sha256}]}],
    })
    return root


class _BundleFlow:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, pack, item_ids, from_dir, *, emit, cancel_check):
        self.calls.append((pack.pack_id, list(item_ids), from_dir))
        emit({"type": "log", "line": "copied"})


def _live_flow(*args, **kwargs):
    raise AssertionError("the download flow ran for a bundle install")


async def test_the_service_runs_a_bundle_install_as_a_job(tmp_path):
    source = _word_vectors_bundle(tmp_path / "bundle")
    flow = _BundleFlow()
    service = PackService(run_flow=_live_flow, run_bundle_flow=flow)

    job = await service.submit_install(get_pack("word-vectors"), None,
                                       from_dir=source)
    events, status = await drain(service, job.job_id)

    assert status == "done"
    assert job.from_dir == str(source)
    started = events[0]
    assert started["type"] == "job_started"
    assert started["items"] == ["glove-50d"]
    assert started["from_dir"] == str(source)
    assert "job_done" in types_of(events)
    assert flow.calls == [("word-vectors", ["glove-50d"], source)]


async def test_the_service_refuses_a_bad_bundle_before_any_job(tmp_path):
    service = PackService(run_flow=_live_flow, run_bundle_flow=_BundleFlow())
    with pytest.raises(BundleInvalid):
        await service.submit_install(get_pack("word-vectors"), None,
                                     from_dir=tmp_path)
    with pytest.raises(ValueError, match="restart mode"):
        await service.submit_install(get_pack("gpu-torch"), None,
                                     from_dir=tmp_path)
    with pytest.raises(ValueError, match="restart mode"):
        await service.submit_install(get_pack("word-vectors"), None,
                                     mode="restart", from_dir=tmp_path)
    assert service.current_job() is None


async def test_a_bundle_install_waits_for_a_running_one(tmp_path):
    source = _word_vectors_bundle(tmp_path / "bundle")
    release = asyncio.Event()
    loop = asyncio.get_running_loop()

    def slow(pack, item_ids, from_dir, *, emit, cancel_check):
        asyncio.run_coroutine_threadsafe(release.wait(), loop).result(10)

    service = PackService(run_flow=_live_flow, run_bundle_flow=slow)
    first = await service.submit_install(get_pack("word-vectors"), None,
                                         from_dir=source)
    with pytest.raises(PackBusy):
        await service.submit_install(get_pack("word-vectors"), None,
                                     from_dir=source)
    release.set()
    await drain(service, first.job_id)


@pytest.fixture
async def api(tmp_path):
    flow = _BundleFlow()
    service = PackService(run_flow=_live_flow, run_bundle_flow=flow)
    previous = getattr(app.state, "pack_service", None)
    app.state.pack_service = service
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url=BASE_URL,
            headers={TOKEN_HEADER: session_token()},
        ) as http:
            yield http, service, flow
    finally:
        await service.shutdown()
        if previous is None:
            if hasattr(app.state, "pack_service"):
                delattr(app.state, "pack_service")
        else:
            app.state.pack_service = previous


async def test_the_route_installs_from_a_bundle_directory(api, tmp_path):
    http, service, flow = api
    source = _word_vectors_bundle(tmp_path / "bundle")

    response = await http.post("/api/packs/word-vectors/install",
                               json={"from_dir": str(source)})

    assert response.status_code == 202, response.text
    await drain(service, response.json()["job_id"])
    assert flow.calls == [("word-vectors", ["glove-50d"], source)]


@pytest.mark.parametrize("body, status", [
    ({"from_dir": "relative/bundle"}, 422),
    ({"from_dir": "   "}, 422),
    ({"from_dir": "/x\x00y"}, 422),
    ({"from_dir": "<missing>"}, 400),
    ({"from_dir": "<missing>", "mode": "restart"}, 400),
])
async def test_the_route_refuses_a_bundle_it_cannot_use(api, tmp_path, body, status):
    http, service, flow = api
    # An absolute path that exists nowhere, spelled for this platform: a
    # literal ``/no/such/bundle`` is not absolute on Windows and is refused
    # as 422 by the validator before the 400 this case is about.
    if body.get("from_dir") == "<missing>":
        body = {**body, "from_dir": str(tmp_path / "no-such-bundle")}
    response = await http.post("/api/packs/word-vectors/install", json=body)
    assert response.status_code == status, response.text
    assert service.current_job() is None and flow.calls == []


async def test_a_bundle_install_is_loopback_only_like_every_install(
        api, tmp_path, monkeypatch):
    http, service, flow = api
    source = _word_vectors_bundle(tmp_path / "bundle")
    monkeypatch.setattr(settings, "HOST", "0.0.0.0")
    monkeypatch.setattr(settings, "ALLOW_REMOTE_PACK_INSTALL", False)

    response = await http.post("/api/packs/word-vectors/install",
                               json={"from_dir": str(source)})

    assert response.status_code == 403
    assert flow.calls == []
