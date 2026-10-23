"""Installing a plugin from a folder or an archive on this machine's disk.

The classroom case: a plugin on a USB stick, no GitHub. These tests follow
it through every layer it touches -- the parser, the shared archive
extraction and its caps, the private copy and its digest, the inspection,
the install flow and its lockfile entry, uninstall, the REST routes and the
CLI -- because the claim being tested is that a local source takes the SAME
road as a repository (manifest, AST gate, consent, staging, lockfile) and
not a shortcut like ``cdui plugin link``.

Nothing here touches the network, a real installer, or the user's own plugin
directory: the guards below fail the test if any of them is reached.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import stat
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path
from textwrap import dedent

import pytest
from httpx import ASGITransport, AsyncClient

import plugins as plugin_cli
from app.config import settings
from app.core import plugin_loader
from app.core.auth import TOKEN_HEADER, session_token
from app.core.packs import runner as packs_runner
from app.core.plugins import catalog as catalog_module
from app.core.plugins import flows, github, lifecycle, local
from app.core.plugins import inspect as plugin_inspect
from app.core.plugins.catalog import CatalogEntry
from app.core.plugins.errors import (
    AlreadyInstalled,
    ManifestError,
    NotUpdatable,
    PluginInstallError,
    ReservedPluginId,
    SourceRefused,
)
from app.core.plugins.service import PluginService
from app.core.plugins.sources import parse_source
from app.main import app

BASE_URL = f"http://127.0.0.1:{settings.PORT}"
_REAL_POPEN = packs_runner.subprocess.Popen

MANIFEST = dedent("""\
    [plugin]
    id = "classroom-demo"
    name = "Classroom Demo"
    version = "0.3.0"
    description = "Handed out on a USB stick."
    schema_version = 1
    """)

NODE = dedent("""\
    def double(x):
        return 2 * x
    """)


# ── guards ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def _explode(*_a, **_k):  # pragma: no cover - only runs on a bug
        raise AssertionError("a local-source test tried to reach the network")

    monkeypatch.setattr(github.urllib.request, "urlopen", _explode)


@pytest.fixture(autouse=True)
def no_installer(monkeypatch):
    def _explode(*_a, **_k):  # pragma: no cover - only runs on a bug
        raise AssertionError("a local-source test tried to run an installer")

    monkeypatch.setattr(packs_runner.subprocess, "Popen", _explode)


@pytest.fixture(autouse=True)
def user_root(tmp_path, monkeypatch) -> Path:
    target = tmp_path / "user" / "plugins"
    target.mkdir(parents=True)
    monkeypatch.setattr(plugin_loader, "plugins_user_root", lambda: target)
    monkeypatch.setattr(plugin_cli, "plugins_user_root", lambda: target)
    monkeypatch.setattr(plugin_cli, "_backend_reload", lambda: False)
    return target


@pytest.fixture(autouse=True)
def builtin_root(tmp_path, monkeypatch) -> Path:
    """An empty release: no pack ships, the catalog is empty."""
    target = tmp_path / "builtin"
    target.mkdir()
    monkeypatch.setattr(plugin_loader, "plugins_builtin_root", lambda: target)
    monkeypatch.setattr(plugin_cli, "plugins_builtin_root", lambda: target)
    monkeypatch.setattr(catalog_module, "catalog_entries", lambda: {})
    return target


# ── sources on disk ────────────────────────────────────────────────────────

def _plugin_folder(root: Path, *, manifest: str = MANIFEST,
                   extra: dict[str, str] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "cdui.plugin.toml").write_text(manifest, encoding="utf-8")
    (root / "nodes").mkdir(exist_ok=True)
    (root / "nodes" / "double.py").write_text(NODE, encoding="utf-8")
    for name, text in (extra or {}).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _zip_of(dest: Path, members: dict[str, str], *, prefix: str = "") -> Path:
    with zipfile.ZipFile(dest, "w") as zf:
        for name, text in members.items():
            zf.writestr(prefix + name, text)
    return dest


def _tgz_of(dest: Path, members: dict[str, str], *, prefix: str = "") -> Path:
    with tarfile.open(dest, "w:gz") as tf:
        for name, text in members.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(prefix + name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return dest


PLUGIN_FILES = {"cdui.plugin.toml": MANIFEST, "nodes/double.py": NODE}


def _entry(plugin_id: str = "classroom-demo") -> dict | None:
    return plugin_loader.load_lockfile().get("plugins", {}).get(plugin_id)


def _steps(events: list[dict]) -> list[str]:
    return [e["step"] for e in events if e["type"] == "step_started"]


def _install(spec: str, **decisions) -> tuple[flows.InstallOutcome, list[dict]]:
    found = plugin_inspect.inspect_source(spec, lockfile=plugin_loader.load_lockfile())
    plan = flows.plan_from_inspection(found, **decisions)
    events: list[dict] = []
    outcome = flows.install_plugin_live(plan, emit=events.append,
                                        cancel_check=lambda: False)
    return outcome, events


# ── parse_source ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("spec", ["/srv/usb/demo", "/srv/usb/demo.zip"])
def test_an_absolute_path_is_a_local_source(spec):
    # ``os.path.abspath`` rather than *spec* itself: on Windows a path that
    # starts with ``/`` is relative to the current drive, and the parser
    # records where it would really read from (``C:\\srv\\usb\\demo``).
    assert parse_source(spec, catalog={"plugins": {}}) == (
        "local", os.path.abspath(spec), "", "")


def test_relative_home_and_file_url_paths_are_made_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # ``expanduser`` reads HOME on POSIX and USERPROFILE on Windows.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    empty = {"plugins": {}}
    assert parse_source("./demo", catalog=empty).name_or_owner == str(tmp_path / "demo")
    assert parse_source("../demo", catalog=empty).name_or_owner == str(tmp_path.parent / "demo")
    assert parse_source(".", catalog=empty).name_or_owner == str(tmp_path)
    assert parse_source("~/demo.zip", catalog=empty).name_or_owner == str(
        tmp_path / "home" / "demo.zip")
    # ``url2pathname`` then ``abspath``, so the expectation is spelled the
    # way the platform spells it (a drive letter in front on Windows).
    assert parse_source("file:///srv/usb/my%20plugin", catalog=empty) == (
        "local", os.path.abspath(urllib.request.url2pathname("/srv/usb/my%20plugin")), "", "")
    assert parse_source("FILE://localhost/srv/x", catalog=empty).name_or_owner == os.path.abspath(
        urllib.request.url2pathname("/srv/x"))


def test_owner_slash_repo_is_still_a_repository_not_a_relative_path():
    """``plugins/x`` has always meant a GitHub repository; a relative path
    has to say ``./`` to be one."""
    assert parse_source("alice/extras", catalog={"plugins": {}}).kind == "github"


def test_a_catalog_name_still_wins():
    catalog = {"plugins": {"stats": {"kind": "builtin"}}}
    assert parse_source("stats", catalog=catalog).kind == "catalog"


def test_the_cli_parser_passes_the_local_kind_through(tmp_path):
    assert plugin_cli.parse_source(str(tmp_path)) == ("local", str(tmp_path), "", "")


def test_the_cli_refusal_lists_the_local_shapes(monkeypatch):
    monkeypatch.setenv("CODEFYUI_LANG", "en")
    with pytest.raises(ValueError) as excinfo:
        plugin_cli.parse_source("not a valid source spec")
    assert "local folder/archive path" in str(excinfo.value)


# ── archive extraction (shared with the GitHub tarball) ────────────────────

def test_a_zip_with_the_manifest_at_its_root(tmp_path):
    archive = _zip_of(tmp_path / "demo.zip", PLUGIN_FILES)
    dest = tmp_path / "out"
    dest.mkdir()
    assert github.extract_archive(archive, dest) == dest
    assert (dest / "nodes" / "double.py").read_text() == NODE


def test_a_zip_of_one_folder_made_on_a_mac(tmp_path):
    """macOS "Compress" adds ``__MACOSX/`` beside the folder; it is not a
    second plugin root."""
    members = {f"demo/{k}": v for k, v in PLUGIN_FILES.items()}
    members["__MACOSX/demo/._cdui.plugin.toml"] = "junk"
    archive = _zip_of(tmp_path / "demo.zip", members)
    dest = tmp_path / "out"
    dest.mkdir()
    assert github.extract_archive(archive, dest) == dest / "demo"


def test_a_tgz_with_one_top_level_directory(tmp_path):
    archive = _tgz_of(tmp_path / "demo.tgz", PLUGIN_FILES, prefix="demo-1/")
    dest = tmp_path / "out"
    dest.mkdir()
    assert github.extract_archive(archive, dest) == dest / "demo-1"


def test_two_top_level_directories_and_no_root_manifest_is_a_layout_refusal(tmp_path):
    archive = _zip_of(tmp_path / "two.zip", {"a/x.txt": "1", "b/y.txt": "2"})
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(SourceRefused) as excinfo:
        github.extract_archive(archive, dest)
    assert excinfo.value.reason == "layout"


@pytest.mark.parametrize("name", ["../evil.py", "/etc/evil.py", "C:/evil.py",
                                  "ok/../../evil.py", "..\\evil.py"])
def test_a_zip_member_that_escapes_is_refused(tmp_path, name):
    archive = _zip_of(tmp_path / "bad.zip", {**PLUGIN_FILES, name: "x"})
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(SourceRefused) as excinfo:
        github.extract_archive(archive, dest)
    assert excinfo.value.reason == "unsafe_member"
    assert not (tmp_path / "evil.py").exists()


def test_a_zip_symlink_member_is_refused(tmp_path):
    archive = tmp_path / "link.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for name, text in PLUGIN_FILES.items():
            zf.writestr(name, text)
        info = zipfile.ZipInfo("nodes/secret")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, "/etc/passwd")
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(SourceRefused) as excinfo:
        github.extract_archive(archive, dest)
    assert excinfo.value.reason == "unsafe_member"


def test_a_tarball_member_that_escapes_is_refused(tmp_path):
    archive = _tgz_of(tmp_path / "bad.tar.gz", {**PLUGIN_FILES, "../evil.py": "x"})
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(SourceRefused) as excinfo:
        github.extract_archive(archive, dest)
    assert excinfo.value.reason == "unsafe_member"


def test_the_member_count_cap_applies_to_every_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(github, "MAX_ARCHIVE_MEMBERS", 2)
    for builder, name in ((_zip_of, "many.zip"), (_tgz_of, "many.tgz")):
        archive = builder(tmp_path / name, {**PLUGIN_FILES, "c.txt": "3"})
        dest = tmp_path / f"out-{name}"
        dest.mkdir()
        with pytest.raises(SourceRefused) as excinfo:
            github.extract_archive(archive, dest)
        assert excinfo.value.reason == "too_many_files"
        assert list(dest.iterdir()) == []


def test_the_unpacked_size_cap_applies_to_a_zip(tmp_path, monkeypatch):
    monkeypatch.setattr(github, "MAX_EXTRACTED_BYTES", 1024)
    archive = _zip_of(tmp_path / "big.zip", {**PLUGIN_FILES, "big.bin": "0" * 4096})
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(SourceRefused) as excinfo:
        github.extract_archive(archive, dest)
    assert excinfo.value.reason == "too_large"


def test_a_zip_whose_stream_outgrows_its_header_is_stopped(tmp_path, monkeypatch):
    """The declared sizes pass; the bytes actually written are counted too."""
    archive = _zip_of(tmp_path / "liar.zip", {**PLUGIN_FILES, "big.bin": "0" * 4096})
    real = github.check_caps
    calls = []

    def header_is_fooled(name, members, unpacked):
        calls.append(unpacked)
        if len(calls) == 1:
            return None
        return real(name, members, unpacked)

    monkeypatch.setattr(github, "MAX_EXTRACTED_BYTES", 1024)
    monkeypatch.setattr(github, "check_caps", header_is_fooled)
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(SourceRefused) as excinfo:
        github.extract_archive(archive, dest)
    assert excinfo.value.reason == "too_large"
    assert len(calls) == 2


@pytest.mark.parametrize("name", ["broken.zip", "broken.tar.gz"])
def test_a_corrupt_archive_is_unreadable(tmp_path, name):
    archive = tmp_path / name
    archive.write_bytes(b"this is not an archive at all")
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(SourceRefused) as excinfo:
        github.extract_archive(archive, dest)
    assert excinfo.value.reason == "unreadable"


def test_an_unsupported_file_type_is_refused(tmp_path):
    archive = tmp_path / "demo.rar"
    archive.write_bytes(b"x")
    with pytest.raises(SourceRefused) as excinfo:
        github.extract_archive(archive, tmp_path)
    assert excinfo.value.reason == "not_supported"


def test_the_github_tarball_rule_is_unchanged(tmp_path):
    """A codeload tarball still needs exactly one top-level directory, and
    its refusal is still a PluginInstallError."""
    archive = _tgz_of(tmp_path / "two.tgz", {"a/x.txt": "1", "b/y.txt": "2"})
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(PluginInstallError) as excinfo:
        github.extract_tarball(archive, dest)
    assert isinstance(excinfo.value, SourceRefused)
    assert excinfo.value.reason == "layout"


# ── the private copy ───────────────────────────────────────────────────────

def test_resolve_tells_a_folder_from_an_archive_and_refuses_the_rest(tmp_path):
    folder = _plugin_folder(tmp_path / "demo")
    archive = _zip_of(tmp_path / "demo.zip", PLUGIN_FILES)
    assert local.resolve(str(folder)) == (folder, "folder")
    assert local.resolve(str(archive)) == (archive, "archive")
    other = tmp_path / "notes.txt"
    other.write_text("x")
    with pytest.raises(SourceRefused) as excinfo:
        local.resolve(str(other))
    assert excinfo.value.reason == "not_supported"
    with pytest.raises(SourceRefused) as excinfo:
        local.resolve(str(tmp_path / "missing"))
    assert excinfo.value.reason == "not_found"
    assert excinfo.value.path == str(tmp_path / "missing")


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs a FIFO")
def test_a_path_that_is_neither_file_nor_folder_is_not_supported(tmp_path):
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    with pytest.raises(SourceRefused) as excinfo:
        local.resolve(str(fifo))
    assert excinfo.value.reason == "not_supported"


def test_a_folder_copy_leaves_out_vcs_and_caches(tmp_path):
    folder = _plugin_folder(tmp_path / "demo", extra={
        ".git/config": "[core]", "__pycache__/x.pyc": "x",
        "node_modules/a/index.js": "x", ".DS_Store": "x",
    })
    tree = local.materialize(folder, "folder", tmp_path / "work0")
    names = sorted(p.relative_to(tree.root).as_posix()
                   for p in tree.root.rglob("*") if p.is_file())
    assert names == ["cdui.plugin.toml", "nodes/double.py"]


def test_the_folder_digest_follows_content_not_timestamps(tmp_path):
    folder = _plugin_folder(tmp_path / "demo")
    first = local.materialize(folder, "folder", tmp_path / "w1").digest
    os.utime(folder / "nodes" / "double.py", (time.time() - 1000,) * 2)
    assert local.materialize(folder, "folder", tmp_path / "w2").digest == first
    (folder / "nodes" / "double.py").write_text(NODE + "# edited\n")
    assert local.materialize(folder, "folder", tmp_path / "w3").digest != first


def test_an_archive_digest_is_the_files_own_sha256(tmp_path):
    import hashlib

    archive = _zip_of(tmp_path / "demo.zip", PLUGIN_FILES)
    tree = local.materialize(archive, "archive", tmp_path / "work")
    assert tree.digest == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert tree.form == "archive"


def test_a_symlink_in_a_folder_is_refused(tmp_path):
    folder = _plugin_folder(tmp_path / "demo")
    (folder / "nodes" / "secret").symlink_to("/etc/hosts")
    with pytest.raises(SourceRefused) as excinfo:
        local.materialize(folder, "folder", tmp_path / "work")
    assert excinfo.value.reason == "unsafe_member"


def test_a_symlinked_directory_in_a_folder_is_refused(tmp_path):
    folder = _plugin_folder(tmp_path / "demo")
    (folder / "elsewhere").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(SourceRefused) as excinfo:
        local.materialize(folder, "folder", tmp_path / "work")
    assert excinfo.value.reason == "unsafe_member"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs a FIFO")
def test_a_special_file_in_a_folder_is_refused(tmp_path):
    folder = _plugin_folder(tmp_path / "demo")
    os.mkfifo(folder / "pipe")
    with pytest.raises(SourceRefused) as excinfo:
        local.materialize(folder, "folder", tmp_path / "work")
    assert excinfo.value.reason == "unsafe_member"


def test_a_folder_past_the_caps_is_refused(tmp_path, monkeypatch):
    folder = _plugin_folder(tmp_path / "demo", extra={"big.bin": "0" * 4096})
    monkeypatch.setattr(github, "MAX_EXTRACTED_BYTES", 1024)
    with pytest.raises(SourceRefused) as excinfo:
        local.materialize(folder, "folder", tmp_path / "w1")
    assert excinfo.value.reason == "too_large"
    monkeypatch.setattr(github, "MAX_EXTRACTED_BYTES", 10**9)
    monkeypatch.setattr(github, "MAX_ARCHIVE_MEMBERS", 1)
    with pytest.raises(SourceRefused) as excinfo:
        local.materialize(folder, "folder", tmp_path / "w2")
    assert excinfo.value.reason == "too_many_files"


def test_an_archive_past_the_compressed_cap_is_refused_unopened(tmp_path, monkeypatch):
    archive = _zip_of(tmp_path / "demo.zip", PLUGIN_FILES)
    monkeypatch.setattr(github, "MAX_TARBALL_BYTES", 10)
    with pytest.raises(SourceRefused) as excinfo:
        local.materialize(archive, "archive", tmp_path / "work")
    assert excinfo.value.reason == "too_large"


def test_an_unreadable_folder_is_reported_as_such(tmp_path, monkeypatch):
    folder = _plugin_folder(tmp_path / "demo")

    def boom(*_a, **_k):
        raise PermissionError("denied")

    monkeypatch.setattr(local.shutil, "copyfile", boom)
    with pytest.raises(SourceRefused) as excinfo:
        local.materialize(folder, "folder", tmp_path / "work")
    assert excinfo.value.reason == "unreadable"


def test_an_unreadable_archive_is_reported_as_such(tmp_path, monkeypatch):
    archive = _zip_of(tmp_path / "demo.zip", PLUGIN_FILES)

    def boom(_path):
        raise PermissionError("denied")

    monkeypatch.setattr(local, "_file_digest", boom)
    with pytest.raises(SourceRefused) as excinfo:
        local.materialize(archive, "archive", tmp_path / "work")
    assert excinfo.value.reason == "unreadable"


def test_a_folder_copy_can_be_cancelled(tmp_path):
    from app.core.plugins.errors import PluginCancelled

    folder = _plugin_folder(tmp_path / "demo")
    with pytest.raises(PluginCancelled):
        local.materialize(folder, "folder", tmp_path / "work",
                          cancel_check=lambda: True)


# ── the inspection ─────────────────────────────────────────────────────────

def test_inspecting_a_folder_describes_it_and_asks_for_consent(tmp_path):
    folder = _plugin_folder(tmp_path / "demo")
    found = plugin_inspect.inspect_source(str(folder), lockfile={})
    assert found.kind == "local"
    assert found.plugin_id == "classroom-demo"
    assert found.source == found.path == str(folder)
    assert found.source_form == "folder"
    assert found.digest and len(found.digest) == 64
    assert found.consent_required is True
    assert found.sha is None and found.url is None and found.owner is None


def test_inspecting_an_archive_says_it_is_one(tmp_path):
    archive = _zip_of(tmp_path / "demo.zip", PLUGIN_FILES, prefix="demo/")
    found = plugin_inspect.inspect_source(str(archive), lockfile={})
    assert found.source_form == "archive"
    assert found.version == "0.3.0"


def test_a_folder_without_a_manifest_is_a_manifest_error(tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    with pytest.raises(ManifestError):
        plugin_inspect.inspect_source(str(folder), lockfile={})


def test_a_local_source_may_not_take_a_reserved_id(tmp_path, monkeypatch):
    folder = _plugin_folder(tmp_path / "demo")
    monkeypatch.setattr(catalog_module, "catalog_entries", lambda: {
        "classroom-demo": CatalogEntry(
            id="classroom-demo", kind="github", name="Demo", description="",
            repo="codefy/classroom-demo", ref="", official=True, chapters=(),
            tags=(), homepage="", path=None),
    })
    with pytest.raises(ReservedPluginId) as excinfo:
        plugin_inspect.inspect_source(str(folder), lockfile={})
    assert "codefy/classroom-demo" in excinfo.value.taken_by


def test_a_reinspection_of_the_installed_copy_reads_as_up_to_date(tmp_path):
    folder = _plugin_folder(tmp_path / "demo")
    _install(str(folder))
    again = plugin_inspect.inspect_source(str(folder),
                                          lockfile=plugin_loader.load_lockfile())
    assert again.mode == "update"
    assert again.up_to_date is True


# ── the install ────────────────────────────────────────────────────────────

def test_a_folder_installs_through_every_step_and_is_recorded(tmp_path, user_root):
    folder = _plugin_folder(tmp_path / "usb" / "demo")
    outcome, events = _install(str(folder))

    assert _steps(events) == ["resolve", "extract", "verify", "stage", "lock"]
    assert outcome.plugin_dir == user_root / "classroom-demo"
    assert (user_root / "classroom-demo" / "nodes" / "double.py").read_text() == NODE
    assert not (user_root / ".staging" / "classroom-demo").exists()
    entry = _entry()
    assert entry["source_kind"] == "local_copy"
    assert entry["source"] == entry["path"] == str(folder)
    assert entry["source_form"] == "folder"
    assert len(entry["sha256"]) == 64
    assert "url" not in entry and "sha" not in entry
    assert entry["enabled"] is True
    # The original is untouched: this is a copy, not a link.
    assert (folder / "cdui.plugin.toml").exists()


def test_an_archive_installs_and_records_its_own_digest(tmp_path, user_root):
    import hashlib

    archive = _tgz_of(tmp_path / "demo.tar.gz", PLUGIN_FILES, prefix="demo/")
    _install(str(archive))
    entry = _entry()
    assert entry["source_form"] == "archive"
    assert entry["sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert (user_root / "classroom-demo" / "cdui.plugin.toml").exists()


def test_the_installed_copy_is_discovered_from_the_user_root(tmp_path, user_root):
    folder = _plugin_folder(tmp_path / "demo")
    _install(str(folder))
    dirs = dict(plugin_loader.iter_plugin_dirs(
        plugin_loader.plugins_builtin_root(), user_root,
        plugin_loader.load_lockfile()))
    assert dirs["classroom-demo"] == user_root / "classroom-demo"


def test_a_folder_edited_after_the_review_is_refused(tmp_path, user_root):
    folder = _plugin_folder(tmp_path / "demo")
    found = plugin_inspect.inspect_source(str(folder), lockfile={})
    plan = flows.plan_from_inspection(found)
    (folder / "nodes" / "double.py").write_text("import subprocess\n")

    with pytest.raises(SourceRefused) as excinfo:
        flows.install_plugin_live(plan, emit=lambda e: None,
                                  cancel_check=lambda: False)
    assert excinfo.value.reason == "changed"
    assert not (user_root / "classroom-demo").exists()
    assert _entry() is None


def test_the_ast_gate_runs_on_a_local_source(tmp_path, user_root):
    folder = _plugin_folder(tmp_path / "demo", extra={
        "nodes/evil.py": "import subprocess\nsubprocess.run(['true'])\n"})
    with pytest.raises(PluginInstallError) as excinfo:
        _install(str(folder))
    assert "security scan" in str(excinfo.value)
    assert "evil.py" in (excinfo.value.hint or "")
    assert not (user_root / "classroom-demo").exists()


def test_a_capability_is_asked_for_like_any_other_source(tmp_path):
    from app.core.plugins.errors import ConsentRequired

    folder = _plugin_folder(tmp_path / "demo", manifest=MANIFEST + dedent("""\

        [security]
        capabilities = ["network"]
        """))
    found = plugin_inspect.inspect_source(str(folder), lockfile={})
    with pytest.raises(ConsentRequired):
        flows.plan_from_inspection(found)
    plan = flows.plan_from_inspection(found, accept_capabilities=["network"])
    assert plan.granted_capabilities == ("network",)


def test_installing_twice_needs_force(tmp_path, user_root):
    folder = _plugin_folder(tmp_path / "demo")
    _install(str(folder))
    found = plugin_inspect.inspect_source(str(folder),
                                          lockfile=plugin_loader.load_lockfile())
    # A plan built by hand as a fresh install is refused at the door.
    plan = flows.InstallPlan(**{**flows.plan_from_inspection(found).__dict__,
                                "mode": "install"})
    with pytest.raises(AlreadyInstalled):
        flows.install_plugin_live(plan, emit=lambda e: None,
                                  cancel_check=lambda: False)
    outcome, _ = _install(str(folder), force=True)
    assert outcome.replaced is True


def test_a_plan_with_no_reviewed_copy_is_refused(tmp_path):
    folder = _plugin_folder(tmp_path / "demo")
    found = plugin_inspect.inspect_source(str(folder), lockfile={})
    plan = flows.InstallPlan(**{**flows.plan_from_inspection(found).__dict__,
                                "digest": None})
    with pytest.raises(PluginInstallError, match="no reviewed copy"):
        flows.install_plugin_live(plan, emit=lambda e: None,
                                  cancel_check=lambda: False)


def test_a_local_copy_has_no_update_to_look_for(tmp_path):
    folder = _plugin_folder(tmp_path / "demo")
    _install(str(folder))
    with pytest.raises(NotUpdatable) as excinfo:
        plugin_inspect.updatable_entry("classroom-demo",
                                       lockfile=plugin_loader.load_lockfile())
    assert excinfo.value.source_kind == "local_copy"
    assert "cdui plugin install" in excinfo.value.hint


def test_uninstall_deletes_the_copy_and_offers_the_path_again(tmp_path, user_root):
    folder = _plugin_folder(tmp_path / "demo")
    _install(str(folder))
    outcome = lifecycle.uninstall_plugin("classroom-demo", builtin_ids=(),
                                         catalog={"plugins": {}})
    assert outcome.removed is True
    assert outcome.files_removed is True
    assert not (user_root / "classroom-demo").exists()
    assert folder.exists()
    # Quoted the way the hint quotes every path: a backslash path on Windows
    # comes back in double quotes, a plain POSIX one bare.
    assert outcome.reinstall_hint == f"cdui plugin install {lifecycle._shell_quote(str(folder))}"


def test_the_reinstall_hint_needs_the_source_to_still_be_there(tmp_path):
    entry = {"source_kind": "local_copy", "path": str(tmp_path / "gone")}
    assert lifecycle.reinstall_command(
        "x", entry, catalog={"plugins": {}}, builtin_ids=()) is None


def test_a_local_copy_is_identified_by_its_files_and_names_no_path(
        tmp_path, user_root, monkeypatch):
    """Run provenance: a copy has no commit, so its revision is the digest of
    the installed files, and the path it came from (this machine's) is
    withheld like a link's."""
    from app.core import env_fingerprint

    folder = _plugin_folder(tmp_path / "demo")
    _install(str(folder))
    # The content walk asks git for a file list; the installer guard would
    # turn that into "unreadable".
    monkeypatch.setattr(packs_runner.subprocess, "Popen", _REAL_POPEN)
    identity = env_fingerprint._plugin_identity(
        "classroom-demo", _entry(), {}, user_root / "classroom-demo")
    assert identity["source_kind"] == "local_copy"
    assert identity["source"] is None
    assert identity["revision_kind"] == "content-sha256"
    assert identity["revision"] == identity["content_sha256"]
    assert identity["immutable"] is False


def test_an_old_lockfile_entry_reads_unchanged(user_root):
    """A ``github_url`` entry written before local sources existed keeps its
    meaning: copied files, an update to look for."""
    assert "github_url" in plugin_loader.COPIED_SOURCE_KINDS
    assert "local" not in plugin_loader.COPIED_SOURCE_KINDS
    entry = {"source_kind": "github_url", "url": "https://github.com/a/b",
             "ref": "", "sha": "0" * 40}
    lockfile = {"plugins": {"b": entry}}
    assert plugin_inspect.updatable_entry("b", lockfile=lockfile) is entry


# ── the REST routes ────────────────────────────────────────────────────────

@pytest.fixture
async def client(monkeypatch):
    """A real service over the real flow, for the whole two-turn install."""
    monkeypatch.setattr(settings, "HOST", "127.0.0.1")
    service = PluginService(reload=lambda: {})
    previous = getattr(app.state, "plugin_service", None)
    app.state.plugin_service = service
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url=BASE_URL,
            headers={TOKEN_HEADER: session_token()},
        ) as http:
            yield http
    finally:
        await service.shutdown()
        if previous is None:
            if hasattr(app.state, "plugin_service"):
                delattr(app.state, "plugin_service")
        else:
            app.state.plugin_service = previous


async def _drain(client, job_id: str) -> tuple[list[dict], str]:
    events: list[dict] = []
    cursor, status = 0, "running"
    deadline = time.monotonic() + 20
    while status == "running":
        assert time.monotonic() < deadline, "job never finished"
        response = await client.get(f"/api/plugins/jobs/{job_id}/events",
                                    params={"cursor": cursor, "wait": 1.0})
        body = response.json()
        events.extend(body["events"])
        cursor, status = body["cursor"], body["status"]
    return events, status


async def test_the_panel_inspects_and_installs_a_folder(client, tmp_path, user_root):
    folder = _plugin_folder(tmp_path / "usb" / "demo")
    response = await client.post("/api/plugins/inspect",
                                 json={"source": str(folder)})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["kind"] == "local"
    assert body["path"] == str(folder)
    assert body["source_form"] == "folder"
    assert len(body["digest"]) == 64
    assert body["consent_required"] is True

    response = await client.post("/api/plugins/install",
                                 json={"inspection_id": body["inspection_id"]})
    assert response.status_code == 202, response.text
    events, status = await _drain(client, response.json()["job_id"])
    assert status == "done", events
    assert _entry()["source_kind"] == "local_copy"

    catalog = await client.get("/api/plugins/catalog")
    row = {e["id"]: e for e in catalog.json()["entries"]}["classroom-demo"]
    assert row["source_kind"] == "local_copy"
    assert row["source"] == str(folder)
    assert row["consent_required"] is True
    assert row["url"] is None

    response = await client.post("/api/plugins/classroom-demo/update")
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "not_updatable"


async def test_nothing_at_the_path_is_a_404_with_the_path(client, tmp_path):
    missing = str(tmp_path / "nowhere")
    response = await client.post("/api/plugins/inspect", json={"source": missing})
    assert response.status_code == 404
    assert response.json()["detail"] == {"code": "local_not_found", "path": missing}


async def test_an_unsafe_archive_is_a_coded_400(client, tmp_path):
    archive = _zip_of(tmp_path / "bad.zip", {**PLUGIN_FILES, "../evil.py": "x"})
    response = await client.post("/api/plugins/inspect",
                                 json={"source": str(archive)})
    assert response.status_code == 400
    assert response.json()["detail"] == {
        "code": "source_refused", "reason": "unsafe_member", "path": ""}


async def test_a_folder_with_no_manifest_is_invalid_manifest(client, tmp_path):
    response = await client.post("/api/plugins/inspect",
                                 json={"source": str(tmp_path)})
    assert response.status_code == 400
    assert response.json()["detail"] == {"code": "invalid_manifest"}


async def test_the_loopback_gate_still_applies(client, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "HOST", "0.0.0.0")
    monkeypatch.setattr(settings, "ALLOW_REMOTE_PLUGIN_INSTALL", False)
    response = await client.post("/api/plugins/inspect",
                                 json={"source": str(tmp_path)})
    assert response.status_code == 403


# ── the CLI ────────────────────────────────────────────────────────────────

def _cli_args(*sources: str, **overrides) -> argparse.Namespace:
    base = dict(source=list(sources), force=False, no_confirm=True,
                trust_author=False, accept_capabilities=False)
    base.update(overrides)
    return argparse.Namespace(**base)


def test_cdui_plugin_install_takes_a_folder(tmp_path, user_root, capsys, monkeypatch):
    monkeypatch.setenv("CODEFYUI_LANG", "en")
    folder = _plugin_folder(tmp_path / "demo")
    assert plugin_cli.cmd_install(_cli_args(str(folder))) == 0
    out = capsys.readouterr().out
    assert "Local folder (sha256 " in out
    assert "Installed: classroom-demo" in out
    assert _entry()["source_kind"] == "local_copy"


def test_cdui_plugin_install_takes_a_zip(tmp_path, user_root, capsys, monkeypatch):
    monkeypatch.setenv("CODEFYUI_LANG", "en")
    archive = _zip_of(tmp_path / "demo.zip", PLUGIN_FILES, prefix="demo/")
    assert plugin_cli.cmd_install(_cli_args(str(archive))) == 0
    assert "Local archive (sha256 " in capsys.readouterr().out
    assert (user_root / "classroom-demo" / "nodes" / "double.py").exists()


def test_cdui_plugin_install_refuses_a_missing_path(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("CODEFYUI_LANG", "en")
    assert plugin_cli.cmd_install(_cli_args(str(tmp_path / "nowhere"))) == 1
    captured = capsys.readouterr()
    assert "Nothing at" in captured.out + captured.err
    assert _entry() is None


def test_cdui_plugin_install_refuses_an_installed_copy_without_force(
        tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("CODEFYUI_LANG", "en")
    folder = _plugin_folder(tmp_path / "demo")
    assert plugin_cli.cmd_install(_cli_args(str(folder))) == 0
    assert plugin_cli.cmd_install(_cli_args(str(folder))) == 1
    assert "already installed" in capsys.readouterr().err
    assert plugin_cli.cmd_install(_cli_args(str(folder), force=True)) == 0


def test_cdui_plugin_update_skips_a_local_copy(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("CODEFYUI_LANG", "en")
    folder = _plugin_folder(tmp_path / "demo")
    plugin_cli.cmd_install(_cli_args(str(folder)))
    capsys.readouterr()
    assert plugin_cli.cmd_update(argparse.Namespace(plugin_id="classroom-demo")) == 0
    assert "installed from a local folder or archive" in capsys.readouterr().out


def test_cdui_plugin_info_reads_a_folder_without_installing(
        tmp_path, user_root, capsys, monkeypatch):
    monkeypatch.setenv("CODEFYUI_LANG", "en")
    folder = _plugin_folder(tmp_path / "demo")
    assert plugin_cli.cmd_info(argparse.Namespace(source_or_id=str(folder))) == 0
    out = capsys.readouterr().out
    assert "Classroom Demo" in out
    assert _entry() is None
    assert plugin_cli.cmd_info(
        argparse.Namespace(source_or_id=str(tmp_path / "nowhere"))) == 1


def test_project_freeze_skips_a_local_copy(tmp_path, monkeypatch, capsys):
    import project as project_cli

    monkeypatch.setenv("CODEFYUI_LANG", "en")
    folder = _plugin_folder(tmp_path / "demo")
    plugin_cli.cmd_install(_cli_args(str(folder)))
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / project_cli.MANIFEST_FILENAME).write_text(
        '[project]\nname = "p"\n', encoding="utf-8")
    monkeypatch.setattr(project_cli, "load_lockfile", plugin_loader.load_lockfile)
    capsys.readouterr()
    assert project_cli.cmd_freeze(argparse.Namespace(dir=str(proj))) == 0
    assert "Skipping linked/local plugin: classroom-demo" in capsys.readouterr().out
    assert json.dumps("classroom-demo") not in (
        proj / project_cli.MANIFEST_FILENAME).read_text()
