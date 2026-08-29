"""One install, start to finish -- the function the server job and the CLI
both call.

``install_pack_live`` is the only place the order of an install is written
down, and the order is the whole design:

* packages BEFORE downloads, because a failed pip run should cost seconds,
  not a 470 MB download that is then unusable;
* the disk check before either, because finding out at 90% that the disk was
  always too small wastes the download;
* ``state.invalidate()`` at the end WHATEVER happened, because a failed or
  cancelled install still changed what is on disk, and a probe cache that
  outlives it makes the UI lie in both directions.

The failure modes are the other half. uv cannot replace a package this
process has already imported, so a resolver conflict is not "the install
failed" -- it is "this one needs a restart", and the message has to carry
the command to type.

Nothing here starts a process or opens a socket: ``runner.run_pip``, the two
download functions and ``subprocess.run`` are all replaced.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest

from app.core.packs import download, flows, runner, state
from app.core.packs.catalog import ModelItem, Pack, get_item, get_pack
from app.core.packs.errors import (
    PackCancelled,
    PackInstallError,
    PackNeedsRestart,
)
from app.core.packs.paths import (
    asset_dir,
    hf_cache_dir,
    sentinel_dir,
    sentinel_path,
)


@pytest.fixture(autouse=True)
def user_data_dir(tmp_path, monkeypatch):
    """A throwaway cache root and a cold probe cache for every test."""
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path))
    state.invalidate()
    yield tmp_path
    state.invalidate()


class _Installer:
    """Records what an install did, with nothing actually installed."""

    def __init__(self):
        self.events: list[dict] = []
        self.pip_calls: list[dict] = []
        self.downloaded: list[str] = []
        self.disk_checked: list[list[str]] = []
        self.probes: list[list[str]] = []
        self.probe_kwargs: list[dict] = []
        self.invalidated = 0
        self.pip_returncode = 0
        self.pip_output: list[str] = []
        self.probe_returncode = 0

    @property
    def steps(self) -> list[str]:
        return [f"{event['type']} {event['step']}" for event in self.events
                if event["type"] in {"step_started", "step_done"}]


@pytest.fixture
def installer(monkeypatch, tmp_path):
    """Fake out everything ``install_pack_live`` delegates to."""
    fake = _Installer()

    def _run_pip(specs, *, constraints_path, emit, cancel_check, cwd,
                 tail=None):
        fake.pip_calls.append({
            "specs": list(specs), "cwd": cwd,
            "constraints": Path(constraints_path).read_text(encoding="utf-8"),
            "constraints_path": Path(constraints_path),
        })
        for line in fake.pip_output:
            emit({"type": "log", "line": line})
            if tail is not None:
                tail.append(line)
        return fake.pip_returncode

    def _download_hf(pack, item, *, emit, cancel_check):
        if cancel_check():
            raise PackCancelled("cancelled")
        fake.downloaded.append(item.item_id)
        target = hf_cache_dir() / item.item_id
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _download_asset(pack, item, *, emit, cancel_check):
        if cancel_check():
            raise PackCancelled("cancelled")
        fake.downloaded.append(item.item_id)
        target = asset_dir() / item.filename
        target.write_bytes(b"data")
        return target

    def _check_disk(items):
        fake.disk_checked.append([item.item_id for item in items])

    def _subprocess_run(argv, **kwargs):
        fake.probes.append(list(argv))
        fake.probe_kwargs.append(kwargs)
        return subprocess.CompletedProcess(argv, fake.probe_returncode,
                                           stdout="", stderr="boom")

    def _invalidate():
        fake.invalidated += 1

    monkeypatch.setattr(runner, "run_pip", _run_pip)
    monkeypatch.setattr(download, "download_hf_item", _download_hf)
    monkeypatch.setattr(download, "download_asset_item", _download_asset)
    monkeypatch.setattr(download, "check_disk", _check_disk)
    monkeypatch.setattr(state, "invalidate", _invalidate)
    monkeypatch.setattr(subprocess, "run", _subprocess_run)
    return fake


def _install(pack_id, item_ids, installer, *, cancel=False):
    return flows.install_pack_live(
        get_pack(pack_id), item_ids,
        emit=installer.events.append,
        cancel_check=lambda: cancel)


def test_error_types_are_reachable_from_the_package(installer):
    """Node code and the routes catch these; neither should have to know
    which module inside the package raised them."""
    import app.core.packs as packs

    assert packs.PackInstallError is PackInstallError
    assert packs.PackNeedsRestart is PackNeedsRestart
    assert packs.PackCancelled is PackCancelled
    assert issubclass(packs.PackInsufficientDisk, PackInstallError)
    # Cancelling is the system doing as it was told, so `except
    # PackInstallError` must not report it as a failure.
    assert not issubclass(PackCancelled, PackInstallError)


def test_a_restart_mode_pack_is_not_installable_live(installer):
    """gpu-torch has no pip specs, no probe modules and no items, so every
    step below would be a no-op and the install would report SUCCESS having
    changed nothing at all. A caller that gets here has a bug, which is why
    it is a ValueError rather than one of the install errors."""
    with pytest.raises(ValueError, match="restart mode"):
        _install("gpu-torch", None, installer)

    assert installer.events == []
    assert installer.disk_checked == []
    assert installer.downloaded == []
    # And it refused BEFORE touching anything -- no probe cache was dropped,
    # because nothing on this machine changed.
    assert installer.invalidated == 0


# -- the happy path --------------------------------------------------------


def test_step_order_and_outcome(installer, monkeypatch):
    """Disk, then packages, then downloads, then the import probe."""
    monkeypatch.setattr(state, "pip_ready", lambda pack: False)

    outcome = _install("sentence-embeddings", ["all-MiniLM-L6-v2"], installer)

    assert installer.steps == [
        "step_started pip",
        "step_done pip",
        "step_started download:all-MiniLM-L6-v2",
        "step_done download:all-MiniLM-L6-v2",
        "step_started verify",
        "step_done verify",
    ]
    assert installer.disk_checked == [["all-MiniLM-L6-v2"]]
    assert installer.downloaded == ["all-MiniLM-L6-v2"]
    assert outcome == flows.InstallOutcome(
        pack_id="sentence-embeddings", pip_installed=True,
        items_done=("all-MiniLM-L6-v2",))
    assert installer.invalidated == 1


def test_every_step_started_carries_a_label(installer, monkeypatch):
    """The step id is for the code; the label is what a learner reads."""
    monkeypatch.setattr(state, "pip_ready", lambda pack: False)

    _install("sentence-embeddings", ["all-MiniLM-L6-v2"], installer)

    started = [event for event in installer.events
               if event["type"] == "step_started"]
    assert started
    assert all(event.get("label") for event in started), started


def test_pip_runs_against_this_interpreter_under_a_constraints_file(
        installer, monkeypatch):
    """The constraints file pins what is installed HERE, so the install has
    to happen here too -- and the file has to still exist while uv reads it."""
    monkeypatch.setattr(state, "pip_ready", lambda pack: False)

    _install("sentence-embeddings", [], installer)

    call = installer.pip_calls[0]
    assert call["specs"] == list(get_pack("sentence-embeddings").pip)
    assert call["cwd"] == flows.BACKEND_DIR
    assert (flows.BACKEND_DIR / "pyproject.toml").exists()
    assert "pytest==" in call["constraints"] or "fastapi==" in call["constraints"]
    # The temporary directory is the job's, and it is gone afterwards.
    assert not call["constraints_path"].exists()


def test_pip_step_skipped_when_modules_importable(installer, monkeypatch):
    """A pack whose packages are already here goes straight to its models --
    re-running pip would be a minute of nothing for every extra model."""
    monkeypatch.setattr(state, "pip_ready", lambda pack: True)

    outcome = _install("sentence-embeddings", ["bge-small-zh-v1.5"], installer)

    assert installer.pip_calls == []
    assert "step_started pip" not in installer.steps
    assert outcome.pip_installed is False
    assert outcome.items_done == ("bge-small-zh-v1.5",)


def test_item_ids_none_installs_everything_not_already_present(
        installer, monkeypatch):
    """``None`` means "the whole pack", minus what is already downloaded."""
    monkeypatch.setattr(state, "pip_ready", lambda pack: True)
    pack = get_pack("sentence-embeddings")
    present = get_item(pack, "all-MiniLM-L6-v2")
    snapshot = hf_cache_dir() / "already-here"
    snapshot.mkdir(parents=True)
    state.write_sentinel(pack.pack_id, present.item_id, {
        "schema": 1, "pack_id": pack.pack_id, "item_id": present.item_id,
        "kind": "hf", "repo_id": present.repo_id, "revision": present.revision,
        "snapshot_dir": str(snapshot), "bytes": 1, "at": "2026-08-28T00:00:00Z",
    })

    outcome = _install("sentence-embeddings", None, installer)

    assert "all-MiniLM-L6-v2" not in outcome.items_done
    assert set(outcome.items_done) == {
        "paraphrase-multilingual-MiniLM-L12-v2", "bge-small-zh-v1.5",
        "multilingual-e5-small"}


def test_asset_pack_downloads_then_converts(installer):
    """GloVe arrives as a gzip and is converted once, after it lands."""
    outcome = _install("word-vectors", None, installer)

    assert installer.steps == [
        "step_started download:glove-50d",
        "step_done download:glove-50d",
        "step_started convert:glove-50d",
        "step_done convert:glove-50d",
    ]
    assert outcome.items_done == ("glove-50d",)
    # No packages to probe, so no verify step to run.
    assert installer.probes == []


def test_glove_convert_step_hands_the_converter_the_downloaded_file(
        installer, monkeypatch):
    """``ensure_npz`` is given the gz that just landed -- not a path guessed
    from the catalog -- and the bar it reports through is the learner's:
    unpacking 400k word vectors is its own wait, and forwarding the frames as
    ``glove-50d`` progress means the UI needs to know nothing new to draw it.
    """
    seen: dict = {}

    def _ensure_npz(gz_path, progress=None):
        seen["gz_path"] = gz_path
        seen["progress"] = progress
        progress({"bytes_done": 40, "bytes_total": 100, "percent": 40.0})
        return Path(gz_path).with_name("glove-50d.npz")

    module = types.ModuleType("app.nodes.llm._glove")
    module.ensure_npz = _ensure_npz
    monkeypatch.setitem(sys.modules, "app.nodes.llm._glove", module)

    _install("word-vectors", None, installer)

    item = get_item(get_pack("word-vectors"), "glove-50d")
    assert seen["gz_path"] == asset_dir() / item.filename
    assert seen["gz_path"].is_file(), "the converter was handed a path with no file"
    assert callable(seen["progress"])

    forwarded = [event for event in installer.events
                 if event["type"] == "progress"]
    assert forwarded == [{"type": "progress", "item": "glove-50d",
                          "bytes_done": 40, "bytes_total": 100,
                          "percent": 40.0}]
    logs = [event["line"] for event in installer.events
            if event["type"] == "log"]
    assert any("glove-50d.npz" in line for line in logs), logs
    assert "step_done convert:glove-50d" in installer.steps


def test_glove_progress_cannot_be_reported_against_another_item(
        installer, monkeypatch):
    """The converter's frames are stamped by the flow, not trusted from it:
    a payload naming a different item would move the wrong bar.

    What the converter leaves out is filled in to the types the event
    contract promises, not to a blanket ``None``: ``bytes_done`` is an int
    there, so "nothing counted yet" is 0, where an unknown total and a
    percent that cannot be computed genuinely are nothing.
    """
    def _ensure_npz(gz_path, progress=None):
        progress({"type": "log", "item": "all-MiniLM-L6-v2"})
        return Path(gz_path).with_name("glove-50d.npz")

    module = types.ModuleType("app.nodes.llm._glove")
    module.ensure_npz = _ensure_npz
    monkeypatch.setitem(sys.modules, "app.nodes.llm._glove", module)

    _install("word-vectors", None, installer)

    forwarded = [event for event in installer.events
                 if event["type"] == "progress"]
    assert forwarded == [{"type": "progress", "item": "glove-50d",
                          "bytes_done": 0, "bytes_total": None,
                          "percent": None}]


def test_converter_whose_own_dependency_is_missing_fails_the_install(
        installer, monkeypatch):
    """"The converter is not in this build" and "the converter is broken"
    are both ImportErrors and mean opposite things. Only the first may be
    shrugged off; the second has to be reported, or a GloVe pack that can
    never convert reports itself installed."""
    module = types.ModuleType("app.nodes.llm._glove")

    def _module_getattr(name):
        raise ImportError("No module named 'numpy'", name="numpy")

    module.__getattr__ = _module_getattr
    monkeypatch.setitem(sys.modules, "app.nodes.llm._glove", module)

    with pytest.raises(ImportError, match="numpy"):
        _install("word-vectors", None, installer)

    logs = [event["line"] for event in installer.events
            if event["type"] == "log"]
    assert not any("not available in this build" in line for line in logs), logs


def test_converter_module_without_ensure_npz_fails_the_install(
        installer, monkeypatch):
    """The trap the exception sets for whoever reads it.

    A converter module that is HERE but does not export ``ensure_npz``
    raises ``ImportError("cannot import name 'ensure_npz' ...")`` whose
    ``name`` is the converter's own module -- indistinguishable from the
    module being missing if you believe ``exc.name``. Believing it reports a
    GloVe pack that can never convert as installed, so presence is what
    decides, and a module that is here has to explain itself.
    """
    module = types.ModuleType("app.nodes.llm._glove")  # no ensure_npz
    monkeypatch.setitem(sys.modules, "app.nodes.llm._glove", module)

    with pytest.raises(ImportError) as raised:
        _install("word-vectors", None, installer)

    # The shape the guard used to be fooled by, asserted so a future guard
    # that leans on ``exc.name`` again fails here rather than in the field.
    assert raised.value.name == "app.nodes.llm._glove"
    assert "ensure_npz" in str(raised.value)

    logs = [event["line"] for event in installer.events
            if event["type"] == "log"]
    assert not any("not available in this build" in line for line in logs), logs
    assert "step_done convert:glove-50d" not in installer.steps


def test_missing_converter_is_a_log_line_not_a_failed_install(installer):
    """The GloVe converter lands in a later PR. Until it does, the download
    is still worth having, and the install must say so rather than fail."""
    _install("word-vectors", None, installer)

    logs = [event["line"] for event in installer.events
            if event["type"] == "log"]
    assert any("glove" in line.lower() for line in logs), logs
    assert "step_done convert:glove-50d" in installer.steps


# -- failures --------------------------------------------------------------


def test_resolver_conflict_becomes_needs_restart_with_command(
        installer, monkeypatch):
    """uv refusing to satisfy the pins means "not while the server is
    running" -- and the message has to say what to type instead."""
    monkeypatch.setattr(state, "pip_ready", lambda pack: False)
    installer.pip_returncode = 1
    installer.pip_output = [
        "  x No solution found when resolving dependencies:",
        "  Because torch==2.11.0+cu128 is installed and sentence-transformers",
        "  depends on torch>=2.12, we can conclude that ...",
    ]

    with pytest.raises(PackNeedsRestart) as failure:
        _install("sentence-embeddings", ["all-MiniLM-L6-v2"], installer)

    assert failure.value.command == (
        "cdui packs install sentence-embeddings --restart")
    assert "No solution found" in failure.value.hint
    assert installer.downloaded == [], "downloaded despite a failed pip run"
    assert installer.invalidated == 1


def test_ordinary_pip_failure_is_a_plain_install_error(installer, monkeypatch):
    """A failed download is not a restart-worthy conflict; telling a user to
    restart the server for a flaky network wastes their time."""
    monkeypatch.setattr(state, "pip_ready", lambda pack: False)
    installer.pip_returncode = 1
    installer.pip_output = ["error: Failed to fetch: https://pypi.org/simple/x/"]

    with pytest.raises(PackInstallError) as failure:
        _install("sentence-embeddings", ["all-MiniLM-L6-v2"], installer)

    assert not isinstance(failure.value, PackNeedsRestart)
    assert "Failed to fetch" in failure.value.hint


def test_verify_step_runs_import_probe(installer, monkeypatch):
    """The probe runs in a CHILD interpreter: a package installed a second
    ago is not importable in this one until it restarts."""
    monkeypatch.setattr(state, "pip_ready", lambda pack: True)

    _install("sentence-embeddings", [], installer)

    assert installer.probes == [
        [sys.executable, "-c", "import sentence_transformers, transformers"]]

    kwargs = installer.probe_kwargs[0]
    assert "PYTHONPATH" not in kwargs["env"], (
        "a dev shell's PYTHONPATH would put this repo inside the probe")
    assert kwargs["env"]["PYTHONUTF8"] == "1"
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["creationflags"] == runner.creation_flags()
    assert "shell" not in kwargs


def test_failed_import_probe_fails_the_install(installer, monkeypatch):
    """pip reporting success while the package cannot be imported is exactly
    the half-installed state the constraints file exists to prevent, and it
    must not be reported as a finished install."""
    monkeypatch.setattr(state, "pip_ready", lambda pack: True)
    installer.probe_returncode = 1

    with pytest.raises(PackInstallError) as failure:
        _install("sentence-embeddings", [], installer)

    assert "boom" in (failure.value.hint or "")
    assert installer.invalidated == 1


def test_unknown_item_id_is_value_error(installer):
    """An id that is not in the catalog is a caller mistake, not an install
    failure -- and nothing is touched before it is caught."""
    with pytest.raises(ValueError, match="no-such-model"):
        _install("sentence-embeddings", ["no-such-model"], installer)

    assert installer.disk_checked == []
    assert installer.invalidated == 0


def test_cancel_stops_before_the_next_item(installer, monkeypatch):
    """Stop between items, not only inside a download."""
    monkeypatch.setattr(state, "pip_ready", lambda pack: True)

    with pytest.raises(PackCancelled):
        _install("sentence-embeddings", None, installer, cancel=True)

    assert installer.downloaded == []
    assert installer.invalidated == 1


def test_invalidate_called_on_success_and_failure(installer, monkeypatch):
    """A failed install still changed the disk; a stale probe cache would
    then report the half-finished state as either done or absent."""
    monkeypatch.setattr(state, "pip_ready", lambda pack: True)

    _install("sentence-embeddings", ["bge-small-zh-v1.5"], installer)
    assert installer.invalidated == 1

    def _boom(pack, item, *, emit, cancel_check):
        raise PackInstallError("disk on fire")

    monkeypatch.setattr(download, "download_hf_item", _boom)
    with pytest.raises(PackInstallError):
        _install("sentence-embeddings", ["bge-small-zh-v1.5"], installer)

    assert installer.invalidated == 2


def test_disk_check_runs_before_anything_is_installed(installer, monkeypatch):
    """The precheck is the first thing that can refuse."""
    monkeypatch.setattr(state, "pip_ready", lambda pack: False)

    def _no_room(items):
        raise download.PackInsufficientDisk("full", needed=10, free=1)

    monkeypatch.setattr(download, "check_disk", _no_room)

    with pytest.raises(download.PackInsufficientDisk):
        _install("sentence-embeddings", ["all-MiniLM-L6-v2"], installer)

    assert installer.pip_calls == []
    assert installer.downloaded == []


# -- removal ---------------------------------------------------------------


def _hf_repo_layout(repo_id: str):
    """The directory huggingface_hub really builds for one repo.

    The BLOBS hold the bytes; the snapshot holds links (or, on a filesystem
    without them, copies) into that folder. Deleting only the snapshot frees
    little or nothing, which is the whole reason removal targets the repo
    folder.
    """
    folder = hf_cache_dir() / ("models--" + repo_id.replace("/", "--"))
    blob = folder / "blobs" / "deadbeef"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"x" * 512)
    snapshot = folder / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    return folder, snapshot


def _hf_sentinel_for(pack, item, snapshot_dir) -> None:
    state.write_sentinel(pack.pack_id, item.item_id, {
        "schema": 1, "pack_id": pack.pack_id, "item_id": item.item_id,
        "kind": "hf", "repo_id": item.repo_id, "revision": item.revision,
        "snapshot_dir": str(snapshot_dir), "bytes": 1,
        "at": "2026-08-28T00:00:00Z",
    })


def _asset_sentinel_for(pack, item, path) -> None:
    state.write_sentinel(pack.pack_id, item.item_id, {
        "schema": 1, "pack_id": pack.pack_id, "item_id": item.item_id,
        "kind": "asset", "url": item.url, "path": str(path), "bytes": 4,
        "sha256": "0" * 64, "at": "2026-08-28T00:00:00Z",
    })


def test_remove_item_deletes_the_whole_repo_folder(installer):
    """Uninstalling one model has to free its BYTES.

    They are in ``models--org--name/blobs``; the snapshot directory holds
    links into it. Deleting the snapshot alone would report 90 MB freed and
    free none of it, and the model would be re-listed as removed while the
    disk stayed exactly as full.
    """
    pack = get_pack("sentence-embeddings")
    item = get_item(pack, "all-MiniLM-L6-v2")
    folder, snapshot = _hf_repo_layout(item.repo_id)
    _hf_sentinel_for(pack, item, snapshot)

    assert flows.remove_item(pack, item.item_id) is True

    assert not folder.exists(), "the blobs were left behind"
    assert not sentinel_path(pack.pack_id, item.item_id).exists()
    assert hf_cache_dir().exists(), "the cache root went with it"


def test_remove_item_ignores_the_snapshot_path_a_sentinel_names(
        installer, tmp_path):
    """A sentinel is a file on disk, and a corrupt or hand-edited one must
    not steer a delete. The repo folder is derived from the CATALOG, so a
    sentinel naming somewhere else is simply not consulted."""
    pack = get_pack("sentence-embeddings")
    item = get_item(pack, "all-MiniLM-L6-v2")
    folder, _ = _hf_repo_layout(item.repo_id)
    outside = tmp_path / "not-the-cache"
    outside.mkdir()
    (outside / "important.txt").write_text("keep me", encoding="utf-8")
    _hf_sentinel_for(pack, item, outside)

    assert flows.remove_item(pack, item.item_id) is True

    assert (outside / "important.txt").read_text(encoding="utf-8") == "keep me"
    assert not folder.exists()


def test_remove_item_refuses_a_repo_folder_outside_the_hf_cache(
        installer, monkeypatch, tmp_path):
    """The derived folder is still checked: it must be a direct child of the
    Hugging Face cache and be named like one."""
    pack = get_pack("sentence-embeddings")
    item = get_item(pack, "all-MiniLM-L6-v2")
    monkeypatch.setattr(flows, "_hf_repo_folder_name",
                        lambda repo_id: "../../escaped")
    escaped = (hf_cache_dir() / ".." / ".." / "escaped").resolve()
    escaped.mkdir(parents=True, exist_ok=True)
    (escaped / "important.txt").write_text("keep me", encoding="utf-8")

    assert flows.remove_item(pack, item.item_id) is False

    assert (escaped / "important.txt").read_text(encoding="utf-8") == "keep me"


def test_remove_item_deletes_an_asset_file(installer):
    """The asset kind records a FILE, not a directory."""
    pack = get_pack("word-vectors")
    _install("word-vectors", None, installer)
    item = get_item(pack, "glove-50d")
    path = asset_dir() / item.filename
    _asset_sentinel_for(pack, item, path)
    assert path.is_file()

    assert flows.remove_item(pack, item.item_id) is True

    assert not path.exists()
    assert asset_dir().exists(), "the cache root went with it"


@pytest.mark.parametrize("where", ["cache-root", "sentinel-dir", "outside",
                                   "another-file"])
def test_remove_item_refuses_an_asset_path_that_is_not_its_own(
        installer, tmp_path, where):
    """An asset may only ever delete ONE path: its own file, in the asset
    directory. Every other answer -- the cache root itself, the directory the
    sentinels live in, somewhere off the cache entirely, or another pack's
    download -- is a sentinel telling us to delete something that is not the
    thing being removed."""
    pack = get_pack("word-vectors")
    item = get_item(pack, "glove-50d")
    (asset_dir() / item.filename).write_bytes(b"data")
    neighbour = asset_dir() / "someone-elses.bin"
    neighbour.write_bytes(b"data")
    outside = tmp_path / "elsewhere"
    outside.mkdir()

    target = {"cache-root": asset_dir(), "sentinel-dir": sentinel_dir(),
              "outside": outside, "another-file": neighbour}[where]
    if where == "sentinel-dir":
        target.mkdir(parents=True, exist_ok=True)
    _asset_sentinel_for(pack, item, target)

    flows.remove_item(pack, item.item_id)

    assert target.exists(), f"{where} was deleted"
    assert neighbour.exists()
    assert asset_dir().exists()
    assert hf_cache_dir().parent.exists()


@pytest.mark.parametrize("filename", ["", "..", "../escaped.bin",
                                     "sub/nested.bin"])
def test_remove_item_refuses_an_asset_filename_that_is_not_one_component(
        installer, filename):
    """A filename is ONE name, and every other spelling deletes the wrong
    thing: ``""`` joins to the cache root, ``".."`` to its parent, ``"../x"``
    to a file outside the cache, and a nested path to somebody else's
    subdirectory. None of them may be reached by removing a model."""
    item = ModelItem(item_id="nameless", kind="asset", filename=filename,
                     url="https://example.invalid/x", approx_bytes=1,
                     license="MIT")
    pack = Pack(pack_id="broken", title="t", description="d", pip=(),
                probe_modules=(), items=(item,), depends_on=(),
                install_mode="live")
    keep = asset_dir() / "keep.bin"
    keep.write_bytes(b"data")

    planted = None
    candidate = (asset_dir() / filename) if filename else None
    if candidate is not None and not candidate.is_dir():
        planted = candidate
        planted.parent.mkdir(parents=True, exist_ok=True)
        planted.write_bytes(b"data")
    _asset_sentinel_for(pack, item, asset_dir() / filename)

    assert flows.remove_item(pack, "nameless") is False

    assert keep.exists()
    assert asset_dir().exists()
    assert asset_dir().parent.exists()
    if planted is not None:
        assert planted.exists(), f"{filename} was deleted"


def test_remove_item_is_false_when_the_bytes_did_not_actually_go(
        installer, monkeypatch, caplog):
    """On Windows a file another process has open cannot be deleted, and
    ``rmtree(ignore_errors=True)`` says nothing about it. Reporting success
    would tell the user they had freed 90 MB they still have."""
    pack = get_pack("sentence-embeddings")
    item = get_item(pack, "all-MiniLM-L6-v2")
    folder, snapshot = _hf_repo_layout(item.repo_id)
    _hf_sentinel_for(pack, item, snapshot)
    monkeypatch.setattr(flows.shutil, "rmtree",
                        lambda path, **kwargs: None)

    with caplog.at_level("WARNING"):
        assert flows.remove_item(pack, item.item_id) is False

    assert folder.exists()
    assert any(str(folder) in record.message for record in caplog.records)


def test_remove_item_with_nothing_to_remove_is_false(installer):
    pack = get_pack("sentence-embeddings")

    assert flows.remove_item(pack, "all-MiniLM-L6-v2") is False


def test_remove_unknown_item_is_a_key_error(installer):
    with pytest.raises(KeyError):
        flows.remove_item(get_pack("sentence-embeddings"), "no-such-model")
