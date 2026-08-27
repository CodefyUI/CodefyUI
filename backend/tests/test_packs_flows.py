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
from pathlib import Path

import pytest

from app.core.packs import download, flows, runner, state
from app.core.packs.catalog import get_item, get_pack
from app.core.packs.errors import (
    PackCancelled,
    PackInstallError,
    PackNeedsRestart,
)
from app.core.packs.paths import asset_dir, hf_cache_dir, sentinel_path


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


def test_remove_item_deletes_the_snapshot_and_the_sentinel(installer):
    """Uninstalling one model frees its bytes and stops claiming it is there."""
    pack = get_pack("sentence-embeddings")
    _install("sentence-embeddings", ["all-MiniLM-L6-v2"], installer)
    item = get_item(pack, "all-MiniLM-L6-v2")
    snapshot = hf_cache_dir() / item.item_id
    state.write_sentinel(pack.pack_id, item.item_id, {
        "schema": 1, "pack_id": pack.pack_id, "item_id": item.item_id,
        "kind": "hf", "repo_id": item.repo_id, "revision": item.revision,
        "snapshot_dir": str(snapshot), "bytes": 1, "at": "2026-08-28T00:00:00Z",
    })
    assert snapshot.is_dir()

    assert flows.remove_item(pack, item.item_id) is True

    assert not snapshot.exists()
    assert not sentinel_path(pack.pack_id, item.item_id).exists()


def test_remove_item_refuses_paths_outside_cache(installer, tmp_path):
    """A sentinel is a file on disk. One naming ``C:/Windows`` -- corrupted,
    hand-edited, or restored from another machine -- must not turn "remove
    this model" into "delete that directory"."""
    pack = get_pack("sentence-embeddings")
    item = get_item(pack, "all-MiniLM-L6-v2")
    outside = tmp_path / "not-the-cache"
    outside.mkdir()
    (outside / "important.txt").write_text("keep me", encoding="utf-8")
    state.write_sentinel(pack.pack_id, item.item_id, {
        "schema": 1, "pack_id": pack.pack_id, "item_id": item.item_id,
        "kind": "hf", "repo_id": item.repo_id, "revision": item.revision,
        "snapshot_dir": str(outside), "bytes": 1, "at": "2026-08-28T00:00:00Z",
    })

    flows.remove_item(pack, item.item_id)

    assert (outside / "important.txt").read_text(encoding="utf-8") == "keep me"
    assert not sentinel_path(pack.pack_id, item.item_id).exists()


def test_remove_item_deletes_an_asset_file(installer):
    """The asset kind records a FILE, not a directory."""
    pack = get_pack("word-vectors")
    _install("word-vectors", None, installer)
    item = get_item(pack, "glove-50d")
    path = asset_dir() / item.filename
    state.write_sentinel(pack.pack_id, item.item_id, {
        "schema": 1, "pack_id": pack.pack_id, "item_id": item.item_id,
        "kind": "asset", "url": item.url, "path": str(path), "bytes": 4,
        "sha256": "0" * 64, "at": "2026-08-28T00:00:00Z",
    })
    assert path.is_file()

    assert flows.remove_item(pack, item.item_id) is True

    assert not path.exists()


def test_remove_item_with_nothing_to_remove_is_false(installer):
    pack = get_pack("sentence-embeddings")

    assert flows.remove_item(pack, "all-MiniLM-L6-v2") is False


def test_remove_unknown_item_is_a_key_error(installer):
    with pytest.raises(KeyError):
        flows.remove_item(get_pack("sentence-embeddings"), "no-such-model")
