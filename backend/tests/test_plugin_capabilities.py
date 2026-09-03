"""core#133 -- the install-time conversation about declared capabilities.

Tier 1's whole point is that the user is TOLD what a plugin is asking for and
gets to say no, so these tests are about the conversation rather than about
the AST (``test_tiered_policy.py`` owns that half).

Every test here runs with stdin non-interactive, because that is what pytest
and every CI job give you -- which makes "refuse and name the flag" the
default path and means nothing in this file can ever block on ``input()``.
The interactive branch is reached by monkeypatching the one predicate that
asks, which is why that predicate is a named function rather than an inline
``sys.stdin.isatty()``.
"""

from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path
from textwrap import dedent

import pytest

import plugins as plugin_cli
from app.core import plugin_loader
from app.core.plugins.errors import GitHubError, PluginInstallError


# ── fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def isolated_lockfile(tmp_path, monkeypatch):
    """Redirect the lockfile AND the install directory to a temp dir.

    Both names have to be patched. ``scripts/plugins.py`` does
    ``from app.core.plugin_loader import plugins_user_root``, so it holds its
    own reference: patching only the one in ``plugin_loader`` moves the
    lockfile (whose writer resolves the name inside that module) while
    leaving ``_install_github`` copying real files into the real
    ``<USER_DATA>/plugins/``. That is not a hypothetical -- it happened while
    these tests were being written, and three plugin directories had to be
    swept out of a developer's actual user data.
    """
    target = tmp_path / "plugins"
    target.mkdir()
    monkeypatch.setattr(plugin_loader, "plugins_user_root", lambda: target)
    monkeypatch.setattr(plugin_cli, "plugins_user_root", lambda: target)
    monkeypatch.setattr(plugin_cli, "_backend_reload", lambda: False)
    return target


def _manifest_dict(capabilities=None, allowed=None) -> dict:
    manifest: dict = {
        "plugin": {
            "id": "cap-pack",
            "name": "Cap Pack",
            "version": "0.1.0",
            "schema_version": 1,
        }
    }
    security: dict = {}
    if capabilities is not None:
        security["capabilities"] = capabilities
    if allowed is not None:
        security["allowed_modules"] = allowed
    if security:
        manifest["security"] = security
    return manifest


def _install_args(**overrides) -> argparse.Namespace:
    base = dict(
        force=False,
        no_confirm=True,
        trust_author=False,
        accept_capabilities=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _out(capsys) -> str:
    captured = capsys.readouterr()
    return captured.out + captured.err


# ── the prompt ─────────────────────────────────────────────────────────────

def test_a_plugin_that_declares_nothing_is_never_asked_about(capsys):
    assert plugin_cli.capability_gate(_manifest_dict(), _install_args()) == ()
    assert "capabilit" not in _out(capsys).lower()


def test_a_capability_request_is_printed_before_it_is_decided(capsys):
    plugin_cli.capability_gate(_manifest_dict(["network"]), _install_args())
    printed = _out(capsys)
    assert "network" in printed
    # The summary, not just the word -- somebody is about to answer y/N.
    assert "reach the network" in printed or "連線網路" in printed


def test_the_prompt_says_a_capability_is_not_a_sandbox(capsys):
    plugin_cli.capability_gate(_manifest_dict(["filesystem"]), _install_args())
    printed = _out(capsys)
    assert "sandbox" in printed or "沙箱" in printed


def test_non_interactive_refuses_and_names_the_flag(capsys):
    granted = plugin_cli.capability_gate(_manifest_dict(["network"]), _install_args())
    assert granted is plugin_cli.CAPABILITIES_REFUSED
    assert "--accept-capabilities" in _out(capsys)


def test_accept_capabilities_grants_without_a_prompt(monkeypatch):
    def _explode(*_a, **_k):  # pragma: no cover - only runs on a bug
        raise AssertionError("--accept-capabilities must not prompt")

    monkeypatch.setattr("builtins.input", _explode)
    granted = plugin_cli.capability_gate(
        _manifest_dict(["network", "filesystem"]),
        _install_args(accept_capabilities=True),
    )
    assert granted == ("filesystem", "network")


@pytest.mark.parametrize("answer", ["y", "Y", "yes", " YES "])
def test_an_interactive_yes_grants(monkeypatch, answer):
    monkeypatch.setattr(plugin_cli, "_stdin_is_interactive", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _p: answer)
    granted = plugin_cli.capability_gate(_manifest_dict(["network"]), _install_args())
    assert granted == ("network",)


@pytest.mark.parametrize("answer", ["", "n", "no", "maybe", "yeah", "sure"])
def test_an_interactive_anything_else_refuses(monkeypatch, answer):
    monkeypatch.setattr(plugin_cli, "_stdin_is_interactive", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _p: answer)
    granted = plugin_cli.capability_gate(_manifest_dict(["network"]), _install_args())
    assert granted is plugin_cli.CAPABILITIES_REFUSED


def test_a_closed_stdin_reads_as_no_and_still_names_the_flag(monkeypatch, capsys):
    """``EOFError`` is what an install driven from a here-doc or a pipe hits --
    ``isatty()`` says yes and the read fails anyway. It must not traceback, and
    "Cancelled" on its own would leave the user with nothing to do
    differently."""
    monkeypatch.setattr(plugin_cli, "_stdin_is_interactive", lambda: True)

    def _eof(_p):
        raise EOFError

    monkeypatch.setattr("builtins.input", _eof)
    granted = plugin_cli.capability_gate(_manifest_dict(["network"]), _install_args())
    assert granted is plugin_cli.CAPABILITIES_REFUSED
    assert "--accept-capabilities" in _out(capsys)


def test_no_confirm_does_not_imply_accepting_capabilities():
    """``-y`` skips the "install from this URL?" question. Consenting to code
    that reaches the network is a different question and needs its own yes."""
    granted = plugin_cli.capability_gate(
        _manifest_dict(["network"]), _install_args(no_confirm=True)
    )
    assert granted is plugin_cli.CAPABILITIES_REFUSED


def test_a_previously_granted_set_is_not_re_asked(monkeypatch, capsys):
    """The ``update`` path: the same ask as last time gets no second prompt."""
    def _explode(*_a, **_k):  # pragma: no cover - only runs on a bug
        raise AssertionError("an update must not prompt for an unchanged ask")

    monkeypatch.setattr("builtins.input", _explode)
    granted = plugin_cli.capability_gate(
        _manifest_dict(["network"]),
        _install_args(prior_capabilities=["network", "filesystem"]),
    )
    assert granted == ("network",)
    assert "Re-using" in _out(capsys)


def test_an_update_that_grows_a_capability_stops(capsys):
    """Capability creep across an update is the supply-chain shape an update
    CAN catch: v1 asked for the network, v2 also wants your filesystem."""
    granted = plugin_cli.capability_gate(
        _manifest_dict(["network", "filesystem"]),
        _install_args(prior_capabilities=["network"]),
    )
    assert granted is plugin_cli.CAPABILITIES_REFUSED
    printed = _out(capsys)
    assert "filesystem" in printed
    assert "--accept-capabilities" in printed


def test_an_unknown_capability_is_refused_even_with_the_flag(capsys):
    granted = plugin_cli.capability_gate(
        _manifest_dict(["telepathy"]), _install_args(accept_capabilities=True)
    )
    assert granted is plugin_cli.CAPABILITIES_REFUSED
    assert "telepathy" in _out(capsys)


# ── manifest validation ────────────────────────────────────────────────────

def test_manifest_validation_rejects_an_unknown_capability():
    with pytest.raises(ValueError, match="Unknown capability"):
        plugin_cli.validate_manifest(_manifest_dict(["telepathy"]))


def test_manifest_validation_rejects_a_string_where_a_list_belongs():
    """``allowed_modules = "os"`` used to reach ``frozenset("os")`` -- a set of
    two single letters that unlocks nothing and prints nonsense."""
    with pytest.raises(ValueError, match="list of strings"):
        plugin_cli.validate_manifest(_manifest_dict(allowed="os"))
    with pytest.raises(ValueError, match="list of strings"):
        plugin_cli.validate_manifest(_manifest_dict(capabilities="network"))


def test_manifest_validation_accepts_a_well_formed_security_table():
    plugin_cli.validate_manifest(_manifest_dict(["network"], allowed=[]))
    plugin_cli.validate_manifest(_manifest_dict())          # no [security] at all


def test_install_parser_wires_accept_capabilities():
    parser = plugin_cli.build_parser()
    args = parser.parse_args(["install", "foo/bar"])
    assert args.accept_capabilities is False
    args = parser.parse_args(["install", "foo/bar", "--accept-capabilities"])
    assert args.accept_capabilities is True


# ── end to end: an install that records what it was granted ────────────────

def _tarball_of(root_name: str, files: dict[str, str], dest: Path) -> None:
    """Pack ``{relative path: text}`` under one top-level directory -- the
    shape ``_install_github`` expects from a GitHub codeload tarball."""
    with tarfile.open(dest, "w:gz") as tf:
        for rel, text in files.items():
            data = text.encode("utf-8")
            member = tarfile.TarInfo(f"{root_name}/{rel}")
            member.size = len(data)
            tf.addfile(member, io.BytesIO(data))


@pytest.fixture
def fake_github(monkeypatch):
    """Serve a synthetic tarball instead of reaching GitHub.

    Three names, in two modules, because an install reads GitHub twice. The
    ref and the manifest are read by ``inspect_github`` -- the half that
    happens BEFORE the download, so the user can be shown what they are
    agreeing to -- and it lives in ``app.core.plugins.github``. The tarball
    is still fetched through ``plugin_cli.download_tarball``: the flow takes a
    GitHub client and the CLI hands over its own module, which is what keeps
    this patch load-bearing. ``**_kw`` absorbs the ``cancel_check`` and
    ``progress`` keywords the flow passes.
    """
    def _make(files: dict[str, str]) -> None:
        monkeypatch.setattr(
            "app.core.plugins.github.resolve_sha", lambda o, r, ref: "0" * 40
        )
        monkeypatch.setattr(
            "app.core.plugins.github.fetch_manifest_text",
            lambda o, r, sha: files["cdui.plugin.toml"],
        )
        monkeypatch.setattr(
            plugin_cli,
            "download_tarball",
            lambda owner, repo, sha, dest, **_kw: _tarball_of(
                f"{repo}-main", files, dest
            ),
        )

    return _make


_LOGGER_MANIFEST = dedent("""\
    [plugin]
    id = "metric-logger"
    name = "Metric Logger"
    version = "0.1.0"
    schema_version = 1

    [security]
    capabilities = ["network"]
    """)

_LOGGER_NODE = dedent("""\
    import json

    import requests


    def post_metric(url, loss):
        return requests.post(url, data=json.dumps({"loss": loss}), timeout=5)
    """)


def test_a_wandb_shaped_plugin_installs_with_network_and_no_trust_author(
    isolated_lockfile, fake_github
):
    """Acceptance 1, through the real install path.

    ``requests`` is refused at Tier 0; the manifest asks for ``network``; the
    user accepts; the AST gate then passes with that grant and the lockfile
    records it. Nothing here passes ``--trust-author``, and ``trusted_modules``
    stays empty -- which is the whole point of the tier existing.
    """
    fake_github({
        "cdui.plugin.toml": _LOGGER_MANIFEST,
        "nodes/logger.py": _LOGGER_NODE,
    })
    rc = plugin_cli._install_github(
        "alice", "metric-logger", "",
        _install_args(accept_capabilities=True),
        plugin_loader.load_lockfile(),
    )
    assert rc == 0

    entry = plugin_loader.load_lockfile()["plugins"]["metric-logger"]
    assert entry["capabilities"] == ["network"]
    assert entry["trusted_modules"] == []
    assert entry["source_kind"] == "github_url"
    assert (isolated_lockfile / "metric-logger" / "nodes" / "logger.py").exists()


def test_the_same_plugin_is_refused_when_the_capability_is_not_granted(
    isolated_lockfile, fake_github, capsys
):
    """Acceptance 3 through the CLI: nothing is written, and the message names
    the flag that would have granted it."""
    fake_github({
        "cdui.plugin.toml": _LOGGER_MANIFEST,
        "nodes/logger.py": _LOGGER_NODE,
    })
    rc = plugin_cli._install_github(
        "alice", "metric-logger", "", _install_args(), plugin_loader.load_lockfile()
    )
    assert rc == 1
    assert plugin_loader.load_lockfile()["plugins"] == {}
    assert not (isolated_lockfile / "metric-logger").exists()
    assert "--accept-capabilities" in _out(capsys)


def test_a_plugin_whose_code_outruns_its_declaration_is_refused(
    isolated_lockfile, fake_github, capsys
):
    """Declaring ``network`` does not quietly buy ``subprocess``: the AST gate
    still runs, with the granted set and nothing more."""
    fake_github({
        "cdui.plugin.toml": _LOGGER_MANIFEST,
        "nodes/logger.py": "import subprocess\n\n\ndef go():\n    return subprocess\n",
    })
    rc = plugin_cli._install_github(
        "alice", "metric-logger", "",
        _install_args(accept_capabilities=True),
        plugin_loader.load_lockfile(),
    )
    assert rc == 1
    assert plugin_loader.load_lockfile()["plugins"] == {}
    assert "--trust-author" in _out(capsys)


def test_a_pure_compute_plugin_installs_with_zero_declarations(
    isolated_lockfile, fake_github
):
    """Acceptance 2, in the shape the shipped ``plugins/stats`` pack has: no
    ``[security]`` table at all, numpy and statistics only, no prompt."""
    fake_github({
        "cdui.plugin.toml": dedent("""\
            [plugin]
            id = "puremath"
            name = "Pure"
            version = "0.1.0"
            schema_version = 1
            """),
        "nodes/calc.py": dedent("""\
            import statistics

            import numpy as np


            def summarise(xs):
                return {"mean": float(np.mean(xs)), "sd": statistics.pstdev(xs)}
            """),
    })
    rc = plugin_cli._install_github(
        "alice", "puremath", "", _install_args(), plugin_loader.load_lockfile()
    )
    assert rc == 0
    assert plugin_loader.load_lockfile()["plugins"]["puremath"]["capabilities"] == []


def test_the_path_helpers_need_no_declaration_either(isolated_lockfile, fake_github):
    """The other half of Tier 0's widening: ``os.path`` is string work, and a
    plugin that joins two path components should not have to ask for the
    environment."""
    fake_github({
        "cdui.plugin.toml": dedent("""\
            [plugin]
            id = "pathy"
            name = "Pathy"
            version = "0.1.0"
            schema_version = 1
            """),
        "nodes/p.py": dedent("""\
            from os.path import basename, join


            def under(root, name):
                return join(root, basename(name))
            """),
    })
    rc = plugin_cli._install_github(
        "alice", "pathy", "", _install_args(), plugin_loader.load_lockfile()
    )
    assert rc == 0


def test_an_absent_capabilities_key_reads_as_empty_everywhere(
    isolated_lockfile, tmp_path
):
    """Acceptance 4, the migration.

    A lockfile entry written before core#133 has no ``capabilities`` key, and
    every reader has to treat that as ``[]`` rather than as ``None`` or a
    ``KeyError``.
    """
    work = tmp_path / "legacy"
    (work / "nodes").mkdir(parents=True)
    (work / "cdui.plugin.toml").write_text(
        dedent("""\
            [plugin]
            id = "legacy"
            name = "Legacy"
            version = "0.1.0"
            schema_version = 1
            """),
        encoding="utf-8",
    )
    (work / "nodes" / "__init__.py").write_text("", encoding="utf-8")

    assert plugin_cli.cmd_link(argparse.Namespace(path=str(work), force=False)) == 0
    lock = plugin_loader.load_lockfile()
    assert lock["plugins"]["legacy"]["capabilities"] == []

    lock["plugins"]["legacy"].pop("capabilities")      # the pre-#133 shape
    plugin_loader.save_lockfile(lock)

    assert plugin_cli.cmd_list(argparse.Namespace()) == 0
    assert plugin_cli.cmd_info(argparse.Namespace(source_or_id="legacy")) == 0


def test_list_shows_which_plugins_hold_which_capabilities(
    isolated_lockfile, fake_github, capsys
):
    """"Which of my plugins can reach the network" should be answerable
    without opening a JSON file in a text editor."""
    fake_github({
        "cdui.plugin.toml": _LOGGER_MANIFEST,
        "nodes/logger.py": _LOGGER_NODE,
    })
    plugin_cli._install_github(
        "alice", "metric-logger", "",
        _install_args(accept_capabilities=True),
        plugin_loader.load_lockfile(),
    )
    capsys.readouterr()

    assert plugin_cli.cmd_list(argparse.Namespace()) == 0
    assert "network" in _out(capsys)


def test_no_shipped_pack_declares_a_capability():
    """The first-party packs are the worked example for "Tier 0 is enough".

    If one of them ever needs a capability that is fine -- but it should be a
    deliberate change to this assertion, not something that appears in a
    manifest during an unrelated PR.
    """
    from app.core.plugin_loader import plugins_builtin_root

    for manifest_path in sorted(plugins_builtin_root().glob("*/cdui.plugin.toml")):
        manifest = plugin_loader.read_manifest_safe(manifest_path.parent)
        assert plugin_cli.manifest_capabilities(manifest) == (), manifest_path
        plugin_cli.validate_manifest(manifest)


# ── the install path around the prompt ─────────────────────────────────────

@pytest.mark.parametrize(
    "failure",
    [
        GitHubError("Not Found", status=404),
        PluginInstallError("Tarball exceeds 100 MB limit.", hint="alice/extras"),
    ],
    ids=["github-said-no", "over-the-size-cap"],
)
def test_a_download_that_fails_is_reported_rather_than_raised(
    failure, isolated_lockfile, monkeypatch, capsys
):
    """The download handler has to name what ``download_tarball`` raises.

    Both of these reach it as ``RuntimeError`` subclasses, which is an
    inheritance choice made three modules away in ``plugins/errors.py`` that
    nothing at the call site states. A cancel is deliberately NOT a
    ``RuntimeError``, so the day one is wired into this path a catch that
    leaned on the base class turns a failed install into a traceback.

    No tarball is served: the point is the failure. The install has to end at
    the same "Download failed" line and the same exit code either way, with
    nothing written to the lockfile.

    The ref and the manifest are served through ``app.core.plugins.github``
    because that is where the inspection reads them, and the inspection is
    what has to SUCCEED here for the download to be reached at all.
    """
    monkeypatch.setenv("CODEFYUI_LANG", "en")
    monkeypatch.setattr(
        "app.core.plugins.github.resolve_sha", lambda o, r, ref: "0" * 40
    )
    monkeypatch.setattr(
        "app.core.plugins.github.fetch_manifest_text",
        lambda o, r, sha: dedent("""\
            [plugin]
            id = "extras"
            name = "Extras"
            version = "0.1.0"
            schema_version = 1
            """),
    )

    def _fail(owner, repo, sha, dest, **kwargs):
        raise failure

    monkeypatch.setattr(plugin_cli, "download_tarball", _fail)

    rc = plugin_cli._install_github(
        "alice", "extras", "", _install_args(), plugin_loader.load_lockfile()
    )
    assert rc == 1
    printed = _out(capsys)
    assert "Download failed" in printed
    assert str(failure) in printed
    assert plugin_loader.load_lockfile()["plugins"] == {}
