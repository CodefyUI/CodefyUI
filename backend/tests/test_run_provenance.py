"""Node provenance, honest pack identity and worker run environments (#344).

The remainder of #344 after PR #749 (hooks 1-3):

1. ``core.content_hash``: a stat identity and a real content hash are two
   different answers; content hashing is incremental.
2. Hook 5, ``env_fingerprint``: a linked local pack is identified by its
   content, a builtin pack by the checkout commit (immutable only while the
   checkout is clean); an edit to a pack changes the digest.
3. A worker's run subprocess reports its own fingerprint with the agent's
   device facts, so an unchanged machine reports the agent's digest.
4. Hook 4, ``core.node_provenance``: each provider's record, unknown facts
   explicit, paths never absolute.
5. The run option, the store (MIGRATION_017), the run service and the run
   detail API; the worker path (child -> runner -> WorkerExecutor).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.core import content_hash
from app.core import env_fingerprint as fpmod
from app.core import node_provenance as prov
from app.core import worker_protocol as wp
from app.core.auth import TOKEN_HEADER, session_token
from app.core.db import Database
from app.core.run_service import (
    RunService,
    RunSubmitError,
    normalize_options,
)
from app.core.run_store import STATUS_SUCCEEDED, RunProvenance, RunStore
from app.main import app

# ── 1. content identity ───────────────────────────────────────────────────


def _tree(root: Path) -> Path:
    (root / "a").mkdir(parents=True)
    (root / "a" / "x.txt").write_bytes(b"hello")
    (root / "b.txt").write_bytes(b"world")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "m.cpython-311.pyc").write_bytes(b"junk")
    return root


def test_a_single_file_content_digest_is_its_sha256(tmp_path):
    path = tmp_path / "data.csv"
    path.write_bytes(b"a,b\n1,2\n")
    result = content_hash.content_identity([path], cache_dir=None)
    assert result["method"] == "sha256"
    assert result["digest"] == hashlib.sha256(b"a,b\n1,2\n").hexdigest()
    assert result["files"] == 1 and result["bytes"] == 8


def test_stat_and_content_identities_are_different_answers(tmp_path):
    root = _tree(tmp_path / "t")
    stat = content_hash.stat_identity([root])
    content = content_hash.content_identity([root], cache_dir=None)
    assert stat["method"] == "stat" and content["method"] == "sha256"
    assert stat["digest"] != content["digest"]
    # __pycache__ is not content.
    assert stat["files"] == content["files"] == 2
    # Touching a file changes the stat identity, not the content one.
    import os

    os.utime(root / "b.txt", ns=(1_000_000_000, 1_000_000_000))
    assert content_hash.stat_identity([root])["digest"] != stat["digest"]
    assert content_hash.content_identity(
        [root], cache_dir=None)["digest"] == content["digest"]


def test_content_hashing_is_incremental(tmp_path):
    root = _tree(tmp_path / "t")
    cache = tmp_path / "cache"
    first = content_hash.content_identity([root], cache_dir=cache)
    assert (first["hashed"], first["reused"]) == (2, 0)
    again = content_hash.content_identity([root], cache_dir=cache)
    assert (again["hashed"], again["reused"]) == (0, 2)
    assert again["digest"] == first["digest"]
    (root / "b.txt").write_bytes(b"WORLD!")
    changed = content_hash.content_identity([root], cache_dir=cache)
    assert (changed["hashed"], changed["reused"]) == (1, 1)
    assert changed["digest"] != first["digest"]
    refreshed = content_hash.content_identity([root], cache_dir=cache,
                                              refresh=True)
    assert refreshed["hashed"] == 2 and refreshed["digest"] == changed["digest"]
    # The cache never sits inside the tree it describes.
    assert not any(p.suffix == ".json" for p in root.rglob("*"))


def test_a_missing_or_truncated_tree_has_no_digest(tmp_path, monkeypatch):
    missing = content_hash.content_identity([tmp_path / "nope"],
                                            cache_dir=None)
    assert missing["exists"] is False and missing["digest"] is None
    assert content_hash.stat_identity([tmp_path / "nope"])["digest"] is None
    root = _tree(tmp_path / "t")
    monkeypatch.setattr(content_hash, "WALK_CAP", 1)
    truncated = content_hash.content_identity([root], cache_dir=None)
    assert truncated["truncated"] is True and truncated["digest"] is None
    assert content_hash.stat_identity([root])["digest"] is None


def test_an_unwritable_cache_only_costs_a_reread(tmp_path):
    root = _tree(tmp_path / "t")
    blocker = tmp_path / "cache-is-a-file"
    blocker.write_text("x")
    result = content_hash.content_identity([root], cache_dir=blocker)
    assert result["digest"] is not None and result["hashed"] == 2


def _symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except (OSError, NotImplementedError):
        pytest.skip("this platform does not allow symlinks here")


def test_a_symlinked_class_folder_is_part_of_the_identity(tmp_path):
    # ImageFolderDataset roots often link their class folders from another
    # disk (train/cats -> /mnt/x/cats); those files are the dataset.
    first, second = tmp_path / "x" / "cats", tmp_path / "y" / "cats"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "1.png").write_bytes(b"AAAA")
    (second / "1.png").write_bytes(b"BBBB")
    ds1, ds2 = tmp_path / "ds1", tmp_path / "ds2"
    ds1.mkdir()
    ds2.mkdir()
    _symlink(ds1 / "cats", first)
    _symlink(ds2 / "cats", second)
    one = content_hash.content_identity([ds1], cache_dir=None)
    two = content_hash.content_identity([ds2], cache_dir=None)
    assert one["files"] == two["files"] == 1
    assert one["digest"] is not None and one["digest"] != two["digest"]
    assert content_hash.stat_identity([ds1])["files"] == 1


def test_a_symlink_loop_is_not_followed_and_says_so(tmp_path):
    root = _tree(tmp_path / "t")
    _symlink(root / "a" / "back", root)
    result = content_hash.content_identity([root], cache_dir=None)
    assert result["files"] == 2 and result["digest"] is not None
    assert result["notes"] == ["not followed (a symlink loop): a/back"]
    assert content_hash.content_identity(
        [root], cache_dir=None)["digest"] == result["digest"]


@pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid")
                                        and os.geteuid() == 0),
                    reason="needs POSIX permissions and a non-root user")
def test_an_unreadable_directory_leaves_no_digest(tmp_path):
    root = _tree(tmp_path / "t")
    locked = root / "locked"
    locked.mkdir()
    (locked / "f.txt").write_text("x")
    locked.chmod(0)
    try:
        result = content_hash.content_identity([root], cache_dir=None)
        stat = content_hash.stat_identity([root])
    finally:
        locked.chmod(0o755)
    for answer in (result, stat):
        assert answer["unreadable"] == 1 and answer["digest"] is None
        assert answer["notes"][0].startswith("unreadable directory: locked")


def test_generated_content_is_not_hashed(tmp_path):
    root = _pack(tmp_path / "pack")
    for name in (".venv", "venv", "env", ".tox", ".nox", "build", "dist",
                 "my_pack.egg-info", "node_modules"):
        (root / name / "lib").mkdir(parents=True)
        (root / name / "lib" / "big.so").write_bytes(b"\0" * 10)
    (root / ".DS_Store").write_bytes(b"finder")
    (root / "Thumbs.db").write_bytes(b"explorer")
    packs = content_hash.PACK_SKIPS
    result = content_hash.content_identity([root], cache_dir=None,
                                           skips=packs)
    assert result["files"] == 2  # the manifest and nodes.py
    assert content_hash.stat_identity([root], skips=packs)["files"] == 2
    # The same tree as a dataset root keeps everything but node_modules.
    assert content_hash.content_identity([root], cache_dir=None)["files"] \
        == 2 + 8 + 2


def test_a_dataset_class_folder_named_like_build_output_is_hashed(tmp_path):
    # The generated-content list is for pack code only: ImageFolder reads a
    # class folder named build or env like any other.
    root = tmp_path / "train"
    for label in ("build", "env"):
        (root / label).mkdir(parents=True)
        (root / label / "1.png").write_bytes(label.encode())
    (root / "Thumbs.db").write_bytes(b"explorer")
    before = content_hash.content_identity([root], cache_dir=None)
    assert before["files"] == 3 and before["digest"] is not None
    assert content_hash.stat_identity([root])["files"] == 3
    (root / "env" / "1.png").write_bytes(b"changed")
    after = content_hash.content_identity([root], cache_dir=None)
    assert after["digest"] != before["digest"]


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_a_pack_in_a_git_checkout_is_listed_by_git(tmp_path):
    root = _pack(tmp_path / "pack")
    (root / ".gitignore").write_text("weights/\n", encoding="utf-8")
    (root / "weights").mkdir()
    (root / "weights" / "model.bin").write_bytes(b"\0" * 64)
    (root / "untracked.py").write_text("Y = 2\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "nodes.py",
                    "codefyui-plugin.json"], check=True)
    walked = content_hash.content_identity([root], cache_dir=None)
    listed = content_hash.content_identity([root], cache_dir=None,
                                           use_git=True,
                                           skips=content_hash.PACK_SKIPS)
    # Tracked and untracked files count, what .gitignore excludes does not.
    assert walked["files"] == 5
    assert listed["files"] == 4 and listed["digest"] is not None
    assert listed["digest"] == content_hash.content_identity(
        [root], cache_dir=None, use_git=True)["digest"]
    # Outside a checkout, use_git walks.
    plain = _pack(tmp_path / "plain")
    assert content_hash.content_identity([plain], cache_dir=None,
                                         use_git=True)["files"] == 2


def test_hashing_one_root_does_not_wait_for_another(tmp_path, monkeypatch):
    # The cache lock is per cache file and never held while files are read:
    # a first content hash of a large dataset must not stall the pack hash
    # of every other run start.
    big = _tree(tmp_path / "big")
    small = _pack(tmp_path / "pack")
    cache = tmp_path / "cache"
    reading = threading.Event()
    release = threading.Event()
    real = content_hash._sha256_file

    def slow(path: Path) -> str | None:
        if big in path.parents:
            reading.set()
            assert release.wait(30)
        return real(path)

    monkeypatch.setattr(content_hash, "_sha256_file", slow)
    worker = threading.Thread(target=content_hash.content_identity,
                              args=([big],), kwargs={"cache_dir": cache})
    results: list[dict[str, Any]] = []
    other = threading.Thread(target=lambda: results.append(
        content_hash.content_identity([small], cache_dir=cache)))
    worker.start()
    try:
        assert reading.wait(30)
        other.start()
        other.join(15)
        # The pack finished while the big tree is still being read.
        assert not other.is_alive() and worker.is_alive()
        assert results[0]["digest"] is not None
    finally:
        release.set()
        worker.join(30)
        other.join(30)


def test_concurrent_saves_of_one_cache_keep_both_entries(tmp_path):
    root = _tree(tmp_path / "t")
    cache = tmp_path / "cache"
    first = content_hash._HashCache(cache, root)
    second = content_hash._HashCache(cache, root)
    a, b = root / "a" / "x.txt", root / "b.txt"
    first.store(a, a.stat(), "1" * 64)
    second.store(b, b.stat(), "2" * 64)
    first.save()
    second.save()
    merged = content_hash._HashCache(cache, root)
    assert merged.lookup(a, a.stat()) == "1" * 64
    assert merged.lookup(b, b.stat()) == "2" * 64


def test_any_missing_or_unreadable_input_leaves_no_digest(tmp_path):
    present = _tree(tmp_path / "t")
    both = content_hash.content_identity([present, tmp_path / "gone"],
                                         cache_dir=None)
    assert both["exists"] is True and both["files"] == 2
    assert both["digest"] is None and both["notes"] == ["missing: gone"]
    assert content_hash.stat_identity(
        [present, tmp_path / "gone"])["digest"] is None
    _symlink(present / "dangling.txt", tmp_path / "nowhere")
    for answer in (content_hash.stat_identity([present]),
                   content_hash.content_identity([present], cache_dir=None)):
        assert answer["unreadable"] == 1 and answer["digest"] is None
        assert answer["notes"] == ["unreadable: t/dangling.txt"]


# ── 2. hook 5: pack identity ──────────────────────────────────────────────


def _pack(root: Path, body: str = "X = 1\n") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "codefyui-plugin.json").write_text('{"id": "p"}', encoding="utf-8")
    (root / "nodes.py").write_text(body, encoding="utf-8")
    return root


def test_a_linked_local_pack_is_identified_by_its_content(tmp_path,
                                                          monkeypatch):
    from app.core import plugin_loader

    pack = _pack(tmp_path / "src" / "linked")
    lock = {"plugins": {"linked": {
        "source_kind": "local", "source": str(pack), "path": str(pack),
        "enabled": True, "manifest": {"version": "0.0.1"}}}}
    monkeypatch.setattr(plugin_loader, "load_lockfile", lambda: lock)
    before = fpmod.collect_fingerprint()
    linked = before["plugins"]["linked"]
    assert linked["revision_kind"] == "content-sha256"
    assert linked["revision"] == linked["content_sha256"]
    assert len(linked["revision"]) == 64
    assert linked["immutable"] is False
    assert linked["source"] is None
    assert str(tmp_path) not in json.dumps(before)
    # Same files: same digest. An edit in place: a different one.
    assert fpmod.fingerprint_digest(fpmod.collect_fingerprint()) \
        == fpmod.fingerprint_digest(before)
    (pack / "nodes.py").write_text("X = 2\n", encoding="utf-8")
    after = fpmod.collect_fingerprint()
    assert after["plugins"]["linked"]["revision"] != linked["revision"]
    assert fpmod.fingerprint_digest(after) != fpmod.fingerprint_digest(before)


def test_a_builtin_pack_is_immutable_only_on_a_clean_checkout(tmp_path,
                                                              monkeypatch):
    from app.core import plugin_loader

    builtin_root = tmp_path / "plugins"
    _pack(builtin_root / "deep")
    monkeypatch.setattr(plugin_loader, "plugins_builtin_root",
                        lambda: builtin_root)
    entry = {"source_kind": "builtin", "source": "deep", "enabled": True}
    for dirty, immutable in ((False, True), (True, False)):
        identity = fpmod._plugin_identity(
            "deep", entry, {"commit": "c" * 40, "dirty": dirty},
            builtin_root / "deep")
        assert identity["revision"] == "c" * 40
        assert identity["revision_kind"] == "codefyui-commit"
        assert identity["immutable"] is immutable
        assert identity["content_sha256"] is not None
    no_commit = fpmod._plugin_identity("deep", entry,
                                       {"commit": None, "dirty": None},
                                       builtin_root / "deep")
    assert no_commit["revision"] is None and no_commit["immutable"] is False


# ── 3. a worker run subprocess's fingerprint ──────────────────────────────


def test_a_run_subprocess_takes_the_agents_device_facts(monkeypatch):
    host = fpmod.collect_fingerprint()
    # The run sees one card only; its CUDA facts would describe that card.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(fpmod, "_accelerators",
                        lambda: pytest.fail("read the narrowed devices"))
    child = fpmod.collect_fingerprint(host=host)
    assert child["accelerators"] == host["accelerators"]
    assert child["devices"] == host["devices"]
    assert child["env"]["CUDA_VISIBLE_DEVICES"] == \
        host["env"]["CUDA_VISIBLE_DEVICES"]
    # An unchanged machine: the agent's digest exactly.
    assert fpmod.fingerprint_digest(child) == fpmod.fingerprint_digest(host)
    # A host document without device facts is not used.
    monkeypatch.setattr(fpmod, "_accelerators", lambda: {"x": 1})
    assert fpmod.collect_fingerprint(host={"env": {}})["accelerators"] \
        == {"x": 1}


def test_the_child_reports_its_environment_before_the_graph_runs():
    from app.agent import child

    from .agent import _helpers as h

    host = fpmod.collect_fingerprint()
    spec = {"run_id": "envrun", "graph": h.tensor_graph(),
            "options": h.TENSOR_OPTIONS, "device": "cpu", "run_facts": True,
            "host_fingerprint": host}
    sent: list[tuple] = []
    thread = threading.Thread(target=child.run_assignment,
                              args=(spec, sent.append),
                              kwargs={"discover": False})
    thread.start()
    thread.join(60)
    kinds = [m[0] for m in sent]
    assert kinds[0] == child.MSG_ENVIRONMENT
    report = sent[0][1]
    assert report["digest"] == fpmod.fingerprint_digest(report["fingerprint"])
    assert report["digest"] == fpmod.fingerprint_digest(host)
    assert "num_threads" in report["numerics"]
    assert kinds[-1] == child.MSG_FINISHED
    # Without the capability nothing extra is sent.
    sent.clear()
    spec["run_facts"] = False
    thread = threading.Thread(target=child.run_assignment,
                              args=(spec, sent.append),
                              kwargs={"discover": False})
    thread.start()
    thread.join(60)
    assert child.MSG_ENVIRONMENT not in [m[0] for m in sent]


def test_the_runner_turns_child_reports_into_run_facts_frames(tmp_path):
    from app.agent import child
    from app.agent.runner import RunHandle, Runner

    runner = Runner(runs_dir=tmp_path / "runs", offered_devices={"cpu"})
    handle = RunHandle(run_id="r1", process=None, events_in=None,
                       control_out=None, log_path=tmp_path / "log")
    frame = runner._frame(handle, (child.MSG_ENVIRONMENT, {
        "fingerprint": {"schema_version": 2}, "digest": "d" * 64,
        "numerics": {"num_threads": 2}}))
    assert isinstance(frame, wp.RunFacts)
    assert frame.fingerprint.digest == "d" * 64
    assert frame.numerics == {"num_threads": 2}
    assert frame.seq == 1
    failed = runner._frame(handle, (child.MSG_ENVIRONMENT, {
        "fingerprint": None, "digest": None, "numerics": None}))
    assert failed.fingerprint is None and failed.seq == 2
    record = runner._frame(handle, (child.MSG_PROVENANCE, {"node_id": "n"}))
    assert record.provenance == {"node_id": "n"} and record.seq == 3


# ── 4. hook 4: the providers ──────────────────────────────────────────────


@pytest.fixture
def data_dirs(tmp_path, monkeypatch):
    models = tmp_path / "data" / "models"
    files = tmp_path / "data" / "files"
    models.mkdir(parents=True)
    files.mkdir(parents=True)
    monkeypatch.setattr(settings, "MODELS_DIR", models)
    monkeypatch.setattr(settings, "DATA_FILES_DIR", files)
    monkeypatch.setattr(settings, "PROJECT_DIR", None)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    return tmp_path / "data"


def _keys(record: dict[str, Any]) -> None:
    assert set(record) == {
        "schema_version", "node_id", "node_type", "kind", "mode", "source",
        "split", "revision", "content", "rows", "notes"}
    assert set(record["source"]) == {"type", "name", "location", "path",
                                     "subset"}
    assert set(record["revision"]) == {"requested", "resolved", "kind"}
    assert record["schema_version"] == prov.SCHEMA_VERSION


def test_csv_reader_records_the_file_and_its_rows(data_dirs):
    import torch

    (data_dirs / "files" / "iris.csv").write_bytes(b"a,b\n1,2\n3,4\n")
    outputs = {"tensor": torch.zeros(2, 2)}
    identity = prov.describe("n1", "CSVReader", {"path": "iris.csv"},
                             outputs, "identity")
    _keys(identity)
    assert identity["kind"] == "dataset"
    assert identity["source"] == {"type": "file", "name": None,
                                  "location": "files", "path": "iris.csv",
                                  "subset": None}
    assert identity["content"]["method"] == "stat"
    assert identity["rows"] == 2
    content = prov.describe("n1", "CSVReader", {"path": "iris.csv"},
                            outputs, "content")
    assert content["content"]["method"] == "sha256"
    assert content["content"]["digest"] == hashlib.sha256(
        b"a,b\n1,2\n3,4\n").hexdigest()
    # Unknown facts are explicit.
    assert content["revision"] == {"requested": None, "resolved": None,
                                   "kind": None}


def test_model_and_checkpoint_loaders_record_the_weights(data_dirs):
    (data_dirs / "models" / "net.pt").write_bytes(b"weights")
    for node_type in ("ModelLoader", "CheckpointLoader"):
        record = prov.describe("m", node_type, {"path": "net.pt"}, None,
                               "content")
        _keys(record)
        assert record["kind"] == "model"
        assert (record["source"]["location"], record["source"]["path"]) \
            == ("models", "net.pt")
        assert record["content"]["digest"] == hashlib.sha256(
            b"weights").hexdigest()
    empty = prov.describe("m", "ModelLoader", {"path": ""}, None, "identity")
    assert empty["content"] is None and "no file selected" in empty["notes"]


def test_a_torchvision_dataset_records_its_owned_files(data_dirs, tmp_path):
    root = tmp_path / "tv"
    (root / "MNIST" / "raw").mkdir(parents=True)
    (root / "MNIST" / "raw" / "train-images").write_bytes(b"img")
    (root / "unrelated").mkdir()
    (root / "unrelated" / "big.bin").write_bytes(b"x" * 100)
    record = prov.describe("d", "Dataset", {
        "name": "MNIST", "split": "train", "data_dir": str(root)},
        {"dataset": list(range(60))}, "identity")
    _keys(record)
    assert record["source"]["type"] == "torchvision"
    assert record["source"]["name"] == "MNIST"
    assert record["split"] == "train"
    assert record["rows"] == 60
    # Only the dataset's own directory, not the whole data_dir.
    assert record["content"]["files"] == 1
    assert record["revision"]["resolved"] is None


def test_an_image_folder_and_an_iterable_dataset(data_dirs):
    folder = data_dirs / "pets" / "train" / "cat"
    folder.mkdir(parents=True)
    (folder / "1.png").write_bytes(b"png")

    class Stream:  # an iterable dataset: no len
        def __iter__(self):
            return iter(())

    record = prov.describe("f", "ImageFolderDataset",
                           {"path": "pets", "split": "train"},
                           {"dataset": Stream()}, "content")
    assert record["source"]["location"] == "data"
    assert record["source"]["path"] == "pets/train"
    assert record["content"]["files"] == 1
    assert record["rows"] is None   # honest unknown



def test_an_image_folder_read_through_a_wired_path(data_dirs):
    """The path param is empty when the port is wired; the dataset's own
    root names the folder that was read."""
    folder = data_dirs / "pets" / "train" / "cat"
    folder.mkdir(parents=True)
    (folder / "1.png").write_bytes(b"png")
    dataset = SimpleNamespace(root=str((data_dirs / "pets" / "train").resolve()))

    record = prov.describe("f", "ImageFolderDataset",
                           {"path": "", "split": "train"},
                           {"dataset": dataset}, "content")
    assert record["source"]["path"] == "pets/train"
    assert record["content"]["files"] == 1
    assert "no folder selected" not in record["notes"]

def test_a_huggingface_dataset_records_the_resolved_commit(tmp_path,
                                                           data_dirs):
    sha = "77f3279092a1c1579b2250db8eafed0ad422088c"
    arrow = (tmp_path / "hfds" / "ylecun___mnist" / "mnist" / "0.0.0" / sha
             / "mnist-train.arrow")
    arrow.parent.mkdir(parents=True)
    arrow.write_bytes(b"arrow")

    class Wrapped:
        _ds = SimpleNamespace(cache_files=[{"filename": str(arrow)}])

        def __len__(self):
            return 5

    record = prov.describe("h", "HuggingFaceDataset",
                           {"dataset_name": "ylecun/mnist", "split": "train"},
                           {"dataset": Wrapped()}, "content")
    _keys(record)
    assert record["revision"] == {"requested": None, "resolved": sha,
                                  "kind": "hf-commit"}
    assert record["rows"] == 5
    assert record["content"]["digest"] == hashlib.sha256(b"arrow").hexdigest()
    assert record["source"]["location"] == "external"
    assert str(tmp_path) not in json.dumps(record)
    # Nothing local to read: the hub cache ref, else an explicit unknown.
    unknown = prov.describe("h", "HuggingFaceDataset",
                            {"dataset_name": "o/none"}, {"dataset": None},
                            "identity")
    assert unknown["revision"]["resolved"] is None
    assert unknown["content"] is None
    ref = tmp_path / "hub" / "datasets--o--ref" / "refs" / "main"
    ref.parent.mkdir(parents=True)
    ref.write_text("a" * 40)
    via_ref = prov.describe("h", "HuggingFaceDataset",
                            {"dataset_name": "o/ref"}, {"dataset": None},
                            "identity")
    assert via_ref["revision"]["resolved"] == "a" * 40


def test_kaggle_records_the_version_it_resolved(tmp_path, data_dirs):
    root = tmp_path / "kaggle" / "datasets" / "o" / "pets" / "versions" / "3"
    (root / "cat").mkdir(parents=True)
    (root / "cat" / "1.png").write_bytes(b"p")
    dataset = SimpleNamespace(root=str(root))
    record = prov.describe("k", "KaggleDataset", {"dataset_slug": "o/pets"},
                           {"dataset": dataset}, "identity")
    assert record["revision"] == {"requested": None, "resolved": "3",
                                  "kind": "kaggle-version"}
    pinned = prov.describe("k", "KaggleDataset",
                           {"dataset_slug": "o/pets/versions/2"},
                           {"dataset": dataset}, "identity")
    assert pinned["revision"]["requested"] == "2"


def test_a_pack_model_records_its_snapshot_commit(tmp_path, monkeypatch,
                                                  data_dirs):
    from app.nodes.llm import _packs_bridge

    sha = "b" * 40
    snapshot = tmp_path / "packs" / "hf" / "models--Qwen--x" / "snapshots" / sha
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    monkeypatch.setattr(_packs_bridge, "model_dir", lambda repo: snapshot)
    record = prov.describe("g", "HFTextGenerate",
                           {"model": "Qwen/Qwen2.5-0.5B-Instruct"}, None,
                           "identity")
    _keys(record)
    assert record["kind"] == "model"
    assert record["revision"]["resolved"] == sha
    assert record["revision"]["kind"] == "hf-commit"
    monkeypatch.setattr(_packs_bridge, "model_dir", lambda repo: None)
    missing = prov.describe("g", "HFTextGenerate", {}, None, "identity")
    assert missing["revision"]["resolved"] is None
    assert "the model snapshot is not downloaded" in missing["notes"]


def test_a_provider_that_fails_still_gives_a_record(monkeypatch):
    def broken(params, outputs, mode):
        raise OSError("disk gone")

    monkeypatch.setitem(prov.PROVIDERS, "ModelLoader", broken)
    record = prov.describe("m", "ModelLoader", {}, None, "content")
    _keys(record)
    assert record["kind"] == "model"
    assert record["notes"] == ["not collected: OSError: disk gone"]
    assert prov.describe("x", "NotProvided", {}, None, "content") is None


def test_a_partial_provider_record_gets_every_key(monkeypatch):
    # A plugin's provider may return only what it knows; the stored record
    # still has every key, so the Runs panel can render it.
    monkeypatch.setitem(prov.PROVIDERS, "PluginLoader",
                        lambda params, outputs, mode: {
                            "rows": 3, "source": {"name": "mine",
                                                  "bogus": 1},
                            "content": "not a mapping", "notes": "nope"})
    record = prov.describe("p", "PluginLoader", {}, None, "identity")
    _keys(record)
    assert record["rows"] == 3 and record["content"] is None
    assert record["source"]["name"] == "mine" and record["notes"] == []
    monkeypatch.setitem(prov.PROVIDERS, "PluginLoader",
                        lambda params, outputs, mode: None)
    empty = prov.describe("p", "PluginLoader", {}, None, "identity")
    _keys(empty)
    assert empty["notes"] == ["the provider returned no record"]


def test_a_filesystem_root_working_dir_never_makes_a_path_relative(
        data_dirs, tmp_path, monkeypatch):
    # A server started from "/" (a systemd unit's default WorkingDirectory,
    # a container's WORKDIR /) must not record the absolute path minus "/".
    target = tmp_path / "someone" / "data.csv"
    target.parent.mkdir()
    target.write_text("a\n1\n")
    monkeypatch.chdir(Path(target.resolve().anchor))
    assert prov.locate(target) == ("external", "data.csv")
    monkeypatch.setattr(settings, "MODELS_DIR",
                        Path(target.resolve().anchor) / "models")
    assert prov.locate(target) == ("external", "data.csv")


def test_the_home_directory_never_makes_a_path_relative(data_dirs, tmp_path,
                                                        monkeypatch):
    home = tmp_path / "home"
    target = home / "secret" / "data.csv"
    target.parent.mkdir(parents=True)
    target.write_text("a\n1\n")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.chdir(home)
    assert prov.locate(target) == ("external", "data.csv")
    # A directory below the home directory is still a root.
    monkeypatch.chdir(home / "secret")
    assert prov.locate(target) == ("working-dir", "data.csv")


# ── 5. option, store, service, API ────────────────────────────────────────


def test_the_option_is_stored_only_when_on():
    assert "provenance" not in normalize_options({})
    assert "provenance" not in normalize_options({"provenance": "off"})
    assert "provenance" not in normalize_options({"provenance": None})
    assert normalize_options({"provenance": " Content "})["provenance"] \
        == "content"
    with pytest.raises(RunSubmitError, match="provenance must be one of"):
        normalize_options({"provenance": "everything"})
    with pytest.raises(RunSubmitError):
        normalize_options({"provenance": True})


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "codefyui.db")
    database.connect()
    try:
        yield database
    finally:
        database.close()


@pytest.fixture
def store(db):
    return RunStore(db)


async def test_the_store_keeps_one_record_per_node_and_drops_them_with_the_run(
        store, db):
    run = await store.create_run(graph_snapshot={"nodes": [], "edges": []},
                                 options={}, provenance=RunProvenance())
    assert await store.record_provenance(run.id, {
        "node_id": "n", "node_type": "CSVReader", "kind": "dataset",
        "rows": 1})
    assert await store.record_provenance(run.id, {
        "node_id": "n", "node_type": "CSVReader", "kind": "dataset",
        "rows": 2})
    assert await store.list_provenance(run.id) == [{
        "node_id": "n", "node_type": "CSVReader", "kind": "dataset",
        "rows": 2}]
    assert not await store.record_provenance("missing", {"node_id": "n"})
    assert not await store.record_provenance(run.id, {"rows": 1})
    await store.mark_running(run.id)
    await store.mark_finished(run.id, STATUS_SUCCEEDED)
    assert await store.delete_run(run.id)
    assert db._conn.execute(
        "SELECT COUNT(*) FROM exec_run_provenance").fetchone()[0] == 0


async def test_a_database_without_the_provenance_table_still_serves_runs(
        store, db):
    # A development database migrated by a wave-d build while migration 017
    # was still empty is past 017 without the table.
    run = await store.create_run(graph_snapshot={"nodes": [], "edges": []},
                                 options={}, provenance=RunProvenance())
    db._conn.execute("DROP TABLE exec_run_provenance")
    assert await store.list_provenance(run.id) == []
    record = {"node_id": "n", "node_type": "CSVReader", "kind": "dataset"}
    assert await store.record_provenance(run.id, record)
    assert await store.list_provenance(run.id) == [record]


async def test_a_worker_runs_own_environment_replaces_the_hello_one(store):
    run = await store.create_run(graph_snapshot={"nodes": [], "edges": []},
                                 options={}, provenance=RunProvenance())
    hello = {"schema_version": 2, "marker": "agent"}
    facts = {"schema_version": 3, "fingerprint_source": "worker-hello",
             "resolved": {"device": "cuda:0"}}
    await store.mark_running(run.id, fingerprint=hello,
                             execution_facts=facts)
    own = {"schema_version": 2, "marker": "subprocess"}
    assert await store.record_run_environment(
        run.id, fingerprint=own, digest=fpmod.fingerprint_digest(own),
        numerics={"num_threads": 8}, source="worker-run")
    record = await store.get_run(run.id)
    assert record.fingerprint_digest == fpmod.fingerprint_digest(own)
    assert await store.get_fingerprint(record.fingerprint_digest) == own
    assert record.execution_facts["fingerprint_source"] == "worker-run"
    assert record.execution_facts["resolved"] == {
        "device": "cuda:0", "numerics": {"num_threads": 8}}
    # The hello document still resolves.
    assert await store.get_fingerprint(fpmod.fingerprint_digest(hello)) \
        == hello
    # Numerics alone (a subprocess that could not collect its document)
    # keep the digest it had.
    assert await store.record_run_environment(
        run.id, fingerprint=None, digest=None, numerics={"num_threads": 1},
        source="worker-run")
    again = await store.get_run(run.id)
    assert again.fingerprint_digest == record.fingerprint_digest
    never_started = await store.create_run(
        graph_snapshot={"nodes": [], "edges": []}, options={},
        provenance=RunProvenance())
    assert not await store.record_run_environment(
        never_started.id, fingerprint=own, digest=None, numerics=None,
        source="worker-run")


def _csv_graph(path: str) -> dict[str, Any]:
    return {
        "nodes": [{"id": "start", "type": "Start", "data": {"params": {}}},
                  {"id": "csv", "type": "CSVReader",
                   "data": {"params": {"path": path}}}],
        "edges": [{"id": "t", "source": "start", "target": "csv",
                   "sourceHandle": "trigger", "type": "trigger"}],
    }


async def _await_terminal(store: RunStore, run_id: str, timeout: float = 30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = await store.get_run(run_id)
        if record is not None and record.finished_at is not None:
            return record
        await asyncio.sleep(0.02)
    raise AssertionError(f"run {run_id} did not finish")


async def test_a_run_with_the_option_records_its_csv(store, tmp_path):
    path = tmp_path / "rows.csv"
    path.write_bytes(b"a,b\n1,2\n3,4\n5,6\n")
    service = RunService(store, shutdown_grace_s=2.0)
    try:
        on = await service.submit(_csv_graph(str(path)),
                                  options={"provenance": "content"},
                                  provenance=RunProvenance())
        off = await service.submit(_csv_graph(str(path)), options={},
                                   provenance=RunProvenance())
        on_record = await _await_terminal(store, on.run_id)
        off_record = await _await_terminal(store, off.run_id)
    finally:
        await service.shutdown()
    assert on_record.status == off_record.status == STATUS_SUCCEEDED
    [record] = await store.list_provenance(on.run_id)
    assert record["node_id"] == "csv" and record["mode"] == "content"
    assert record["rows"] == 3
    assert record["content"]["digest"] == hashlib.sha256(
        path.read_bytes()).hexdigest()
    assert await store.list_provenance(off.run_id) == []
    assert "provenance" not in off_record.options
    assert on_record.execution_facts["fingerprint_source"] == "process"
    # The option changes what is recorded, never what the run emits.
    on_events = [(e.type, e.payload) for e in
                 await store.get_events(on.run_id)
                 if e.type not in ("metric", "execution_start")]
    off_events = [(e.type, e.payload) for e in
                  await store.get_events(off.run_id)
                  if e.type not in ("metric", "execution_start")]
    assert len(on_events) >= 4
    assert on_events == off_events


async def test_the_run_detail_lists_its_provenance(tmp_path):
    path = tmp_path / "rows.csv"
    path.write_bytes(b"a\n1\n")
    database = Database(tmp_path / "api.db")
    database.connect()
    service = RunService(RunStore(database), shutdown_grace_s=5.0)
    app.state.db = database
    app.state.run_service = service
    try:
        async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url=f"http://127.0.0.1:{settings.PORT}",
                headers={TOKEN_HEADER: session_token()}) as client:
            submitted = await client.post("/api/runs", json={
                "graph": _csv_graph(str(path)),
                "options": {"provenance": "identity"}})
            assert submitted.status_code in (200, 201, 202), submitted.text
            run_id = submitted.json()["run_id"]
            await _await_terminal(service.store, run_id)
            detail = (await client.get(f"/api/runs/{run_id}")).json()
            listing = (await client.get("/api/runs")).json()
    finally:
        await service.shutdown()
        database.close()
        for attribute in ("db", "run_service"):
            if hasattr(app.state, attribute):
                delattr(app.state, attribute)
    [record] = detail["provenance"]
    assert record["content"]["method"] == "stat"
    assert record["source"]["location"] == "external"
    assert detail["options"]["provenance"] == "identity"
    # Only the detail carries the records.
    assert all("provenance" not in row for row in listing["runs"])


# ── the worker path on the control plane ──────────────────────────────────


async def test_worker_executor_routes_run_facts_to_the_sinks():
    from app.core.executors import RunSinks, WorkerExecutor

    environments: list[tuple] = []
    records: list[dict] = []

    async def noop(*args, **kwargs):
        return None

    async def environment(document, numerics):
        environments.append((document, numerics))

    async def provenance(record):
        records.append(record)

    sinks = RunSinks(emit=noop, record_metrics=noop, record_artifact=noop,
                     warn=noop, record_environment=environment,
                     record_provenance=provenance)
    request = SimpleNamespace(run_id="r1", sinks=sinks, placement=None)
    document = {"schema_version": 2, "marker": 1}
    await WorkerExecutor._run_facts(request, wp.RunFacts(
        run_id="r1", seq=1, numerics={"num_threads": 3},
        # A digest the document does not hash to: the document is kept.
        fingerprint=wp.FingerprintReport(document=document, digest="0" * 64)))
    await WorkerExecutor._run_facts(request, wp.RunFacts(
        run_id="r1", seq=2, provenance={"node_id": "n"}))
    assert environments == [(document, {"num_threads": 3})]
    assert records == [{"node_id": "n"}]

    async def failing(record):
        raise RuntimeError("db gone")

    sinks.record_provenance = failing
    # Logged, never raised: observability does not end a run.
    await WorkerExecutor._run_facts(request, wp.RunFacts(
        run_id="r1", seq=3, provenance={"node_id": "n"}))


async def test_the_service_verifies_a_reported_fingerprint(store):
    service = RunService(store, shutdown_grace_s=2.0)
    try:
        run = await store.create_run(
            graph_snapshot={"nodes": [], "edges": []}, options={},
            provenance=RunProvenance())
        await store.mark_running(run.id, execution_facts={
            "schema_version": 3, "resolved": {}})
        document = {"schema_version": 2, "worker": True}
        await service._record_environment(run.id, document,
                                          {"num_threads": 1})
        record = await store.get_run(run.id)
        assert record.fingerprint_digest == fpmod.fingerprint_digest(document)
        assert record.execution_facts["fingerprint_source"] == "worker-run"
        # A document that cannot be canonicalised is dropped, numerics kept.
        await service._record_environment(run.id, {"bad": float("nan")},
                                          {"num_threads": 2})
        again = await store.get_run(run.id)
        assert again.fingerprint_digest == record.fingerprint_digest
        assert again.execution_facts["resolved"]["numerics"] == {
            "num_threads": 2}
    finally:
        await service.shutdown()
