"""freeze writes github pins (skipping linked/local); restore installs each pin
BY ITS SHA, no-confirm, skipping already-satisfied pins (spec ID11)."""

import argparse

import project
from app.core.plugin_loader import tomllib


def _init(tmp_path):
    target = tmp_path / "svc"
    project.cmd_init(argparse.Namespace(dir=str(target), adopt=None, force=False))
    return target


def test_freeze_writes_github_pins_skips_local_and_builtin(tmp_path, monkeypatch):
    proj = _init(tmp_path)
    fake_lock = {"schema": 1, "plugins": {
        "pack-a": {"source_kind": "github_url",
                   "url": "https://github.com/o/pack-a", "ref": "v1.2",
                   "sha": "a" * 40, "enabled": True},
        "linked": {"source_kind": "local", "path": "/abs/linked",
                   "enabled": True},
        "builtin-pack": {"source_kind": "builtin", "source": "builtin-pack"},
    }}
    monkeypatch.setattr(project, "load_lockfile", lambda: fake_lock)
    assert project.cmd_freeze(argparse.Namespace(dir=str(proj))) == 0
    manifest = tomllib.loads(
        (proj / "codefyui.project.toml").read_text(encoding="utf-8"))
    assert manifest["plugins"]["pack-a"]["sha"] == "a" * 40
    assert manifest["plugins"]["pack-a"]["ref"] == "v1.2"
    assert "linked" not in manifest["plugins"]       # local skipped (warned)
    assert "builtin-pack" not in manifest["plugins"]  # builtins not pinned


# A manifest with user-added content freeze does not own: an unknown top-level
# scalar, an unknown key inside [project], an unknown top-level table (with a
# nested sub-table), and unknown keys inside [publish]. [plugins] holds a
# stale hand-written pin that freeze MUST replace (machine-owned section).
_CUSTOMIZED_MANIFEST = (
    'codename = "zeta"\n'
    '\n'
    '[project]\n'
    'name = "svc"\n'
    'format_version = 1\n'
    'requires_codefyui = ">=1.4"\n'
    'description = "my service"\n'
    '\n'
    '[plugins]\n'
    'stale-pack = { url = "https://github.com/o/stale", ref = "v0", '
    'sha = "' + "d" * 40 + '" }\n'
    '\n'
    '[my_notes]\n'
    'owner = "alice"\n'
    'priority = 7\n'
    'active = true\n'
    'tags = ["a", "b"]\n'
    '\n'
    '[my_notes.inner]\n'
    'deep = "kept"\n'
    '\n'
    '[publish]\n'
    'graph = "main"\n'
    'slug = "svc"\n'
    'record_io = true\n'
    'timeout_s = 30\n'
)


def _freeze_with_one_pin(proj, monkeypatch):
    fake_lock = {"schema": 1, "plugins": {
        "pack-a": {"source_kind": "github_url",
                   "url": "https://github.com/o/pack-a", "ref": "v1.2",
                   "sha": "a" * 40, "enabled": True}}}
    monkeypatch.setattr(project, "load_lockfile", lambda: fake_lock)
    assert project.cmd_freeze(argparse.Namespace(dir=str(proj))) == 0


def test_freeze_preserves_unknown_manifest_keys(tmp_path, monkeypatch):
    """Issue #87: freeze must round-trip keys it does not know about."""
    proj = _init(tmp_path)
    (proj / "codefyui.project.toml").write_text(
        _CUSTOMIZED_MANIFEST, encoding="utf-8")
    _freeze_with_one_pin(proj, monkeypatch)
    manifest = tomllib.loads(
        (proj / "codefyui.project.toml").read_text(encoding="utf-8"))

    # (a) unknown top-level scalar + unknown top-level table survive.
    assert manifest["codename"] == "zeta"
    assert manifest["my_notes"] == {
        "owner": "alice", "priority": 7, "active": True,
        "tags": ["a", "b"], "inner": {"deep": "kept"}}
    # (b) unknown key inside [project] survives; known keys intact.
    assert manifest["project"]["description"] == "my service"
    assert manifest["project"]["name"] == "svc"
    assert manifest["project"]["format_version"] == 1
    assert manifest["project"]["requires_codefyui"] == ">=1.4"
    # (c) unknown keys inside [publish] survive alongside the known ones.
    assert manifest["publish"] == {
        "graph": "main", "slug": "svc", "record_io": True, "timeout_s": 30}
    # [plugins] is machine-owned: regenerated from the lockfile, stale pin gone.
    assert manifest["plugins"] == {
        "pack-a": {"url": "https://github.com/o/pack-a", "ref": "v1.2",
                   "sha": "a" * 40}}


def test_freeze_twice_is_idempotent(tmp_path, monkeypatch):
    """A second freeze with the same lockfile rewrites the same bytes."""
    proj = _init(tmp_path)
    mpath = proj / "codefyui.project.toml"
    mpath.write_text(_CUSTOMIZED_MANIFEST, encoding="utf-8")
    _freeze_with_one_pin(proj, monkeypatch)
    first = mpath.read_text(encoding="utf-8")
    _freeze_with_one_pin(proj, monkeypatch)
    second = mpath.read_text(encoding="utf-8")
    assert second == first
    manifest = tomllib.loads(second)
    assert manifest["codename"] == "zeta"          # still there after 2 passes
    assert manifest["my_notes"]["inner"]["deep"] == "kept"
    assert manifest["publish"]["timeout_s"] == 30


def test_restore_installs_pin_by_sha(tmp_path, monkeypatch):
    proj = _init(tmp_path)
    (proj / "codefyui.project.toml").write_text(
        '[project]\nname = "svc"\nformat_version = 1\n\n'
        '[plugins]\npack-a = { url = "https://github.com/o/pack-a", '
        'ref = "v1.2", sha = "' + "b" * 40 + '" }\n', encoding="utf-8")
    monkeypatch.setattr(project, "load_lockfile",
                        lambda: {"schema": 1, "plugins": {}})
    calls = []

    def fake_install(owner, repo, ref, args, lockfile):
        calls.append((owner, repo, ref, args.pinned_sha, args.no_confirm))
        return 0

    monkeypatch.setattr(project, "_install_github", fake_install)
    assert project.cmd_restore(argparse.Namespace(dir=str(proj))) == 0
    assert calls == [("o", "pack-a", "v1.2", "b" * 40, True)]


def test_restore_skips_already_satisfied(tmp_path, monkeypatch):
    proj = _init(tmp_path)
    (proj / "codefyui.project.toml").write_text(
        '[project]\nname = "svc"\nformat_version = 1\n\n'
        '[plugins]\npack-a = { url = "https://github.com/o/pack-a", '
        'ref = "v1.2", sha = "' + "c" * 40 + '" }\n', encoding="utf-8")
    monkeypatch.setattr(project, "load_lockfile", lambda: {"schema": 1, "plugins": {
        "pack-a": {"source_kind": "github_url", "sha": "c" * 40}}})
    called = []
    monkeypatch.setattr(project, "_install_github",
                        lambda *a, **k: called.append(1) or 0)
    assert project.cmd_restore(argparse.Namespace(dir=str(proj))) == 0
    assert called == []  # already at the pinned sha -> no reinstall


def test_restore_pin_missing_url_sha_fails_without_install(tmp_path,
                                                           monkeypatch):
    """A pin table lacking url/sha cannot be installed: restore reports it
    (rc 1) and never attempts an install for it (issue #88)."""
    proj = _init(tmp_path)
    (proj / "codefyui.project.toml").write_text(
        '[project]\nname = "svc"\nformat_version = 1\n\n'
        '[plugins]\nbroken = { ref = "v1" }\n', encoding="utf-8")
    monkeypatch.setattr(project, "load_lockfile",
                        lambda: {"schema": 1, "plugins": {}})
    called = []
    monkeypatch.setattr(project, "_install_github",
                        lambda *a, **k: called.append(1) or 0)
    assert project.cmd_restore(argparse.Namespace(dir=str(proj))) == 1
    assert called == []  # nothing installable from an unenforceable pin


def test_restore_continues_batch_after_install_failure(tmp_path, monkeypatch):
    """One failing install (rc != 0) must not abort the batch: the next pin
    is still attempted, and the overall rc reports the failure (issue #88)."""
    proj = _init(tmp_path)
    (proj / "codefyui.project.toml").write_text(
        '[project]\nname = "svc"\nformat_version = 1\n\n'
        '[plugins]\n'
        'pack-a = { url = "https://github.com/o/pack-a", ref = "v1", '
        'sha = "' + "a" * 40 + '" }\n'
        'pack-b = { url = "https://github.com/o/pack-b", ref = "v2", '
        'sha = "' + "b" * 40 + '" }\n', encoding="utf-8")
    monkeypatch.setattr(project, "load_lockfile",
                        lambda: {"schema": 1, "plugins": {}})
    attempts = []

    def fake_install(owner, repo, ref, args, lockfile):
        attempts.append(repo)
        return 1 if repo == "pack-a" else 0

    monkeypatch.setattr(project, "_install_github", fake_install)
    assert project.cmd_restore(argparse.Namespace(dir=str(proj))) == 1
    assert attempts == ["pack-a", "pack-b"]  # pack-b still ran after the fail
