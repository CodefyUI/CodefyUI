"""The reinstall command an uninstall hands back (#506).

``cdui plugin install <id>`` resolves only an id the catalog lists, so the
command has to come from where the plugin was installed from: its catalog
name, its repository and ref, or the folder it was linked from -- and no
command at all when none of those can be vouched for.
"""

from __future__ import annotations

import os

import pytest

from app.core.plugins.deps import _shell_quote
from app.core.plugins.lifecycle import reinstall_command
from app.core.plugins.sources import parse_source

CATALOG = {
    "schema": 1,
    "plugins": {
        "rl": {"kind": "builtin", "name": "RL"},
        "graph-copilot": {"kind": "github", "name": "Graph Copilot",
                          "repo": "CodefyUI/CodefyUI-Graph-Copilot", "ref": "v1"},
    },
}
BUILTINS = {"rl"}


def _hint(plugin_id, entry):
    return reinstall_command(plugin_id, entry, catalog=CATALOG,
                             builtin_ids=BUILTINS)


def test_a_builtin_pack_gets_its_catalog_name():
    assert _hint("rl", {"source_kind": "builtin", "source": "rl"}) == (
        "cdui plugin install rl")


def test_a_builtin_pack_the_catalog_no_longer_lists_gets_no_command():
    assert _hint("gone", {"source_kind": "builtin", "source": "gone"}) is None


def test_a_catalogue_github_plugin_gets_its_catalog_name():
    entry = {"source_kind": "github_url",
             "source": "CodefyUI/CodefyUI-Graph-Copilot@v1",
             "url": "https://github.com/CodefyUI/CodefyUI-Graph-Copilot",
             "ref": "v1", "catalog_id": "graph-copilot"}
    assert _hint("graph-copilot", entry) == "cdui plugin install graph-copilot"


@pytest.mark.parametrize(
    ("ref", "spec"),
    [("v1.2.3", "alice/extras@v1.2.3"), ("", "alice/extras"),
     ("feature/x", "alice/extras@feature/x")],
)
def test_a_github_plugin_outside_the_catalogue_gets_its_source_and_ref(ref, spec):
    """Its manifest id names nothing the installer can find, so the command
    names the repository and the ref it was installed from -- and that
    command parses back to that same repository."""
    entry = {"source_kind": "github_url", "source": spec,
             "url": "https://github.com/alice/extras", "ref": ref}
    hint = _hint("extras-pack", entry)
    assert hint == f"cdui plugin install {spec}"
    parsed = parse_source(hint.split()[-1], catalog=CATALOG)
    assert tuple(parsed) == ("github", "alice", "extras", ref)


def test_a_github_plugin_whose_id_matches_a_catalog_row_gets_its_own_source():
    """Installed by hand from another repository: the catalog's command
    would install a different plugin, so the recorded source wins."""
    entry = {"source_kind": "github_url", "source": "mallory/graph-copilot",
             "url": "https://github.com/mallory/graph-copilot", "ref": ""}
    assert _hint("graph-copilot", entry) == (
        "cdui plugin install mallory/graph-copilot")


def test_a_stale_catalog_id_falls_back_to_the_recorded_source():
    entry = {"source_kind": "github_url", "source": "alice/extras",
             "url": "https://github.com/alice/extras", "ref": "",
             "catalog_id": "no-longer-listed"}
    assert _hint("extras", entry) == "cdui plugin install alice/extras"


@pytest.mark.parametrize(
    "entry",
    [
        {"source_kind": "github_url", "url": "https://evil.example/alice/extras",
         "ref": ""},
        {"source_kind": "github_url", "ref": "v1"},
        {"source_kind": "github_url", "url": "https://github.com/alice/extras",
         "ref": "../../x"},
        {"source_kind": "github_url", "url": "https://github.com/alice/extras",
         "ref": "v1; rm -rf ~"},
        {"source_kind": "github_url", "url": "https://github.com/alice/extras",
         "ref": 3},
    ],
    ids=["other-host", "no-url", "walks-up", "shell", "not-text"],
)
def test_a_recorded_source_that_does_not_validate_gets_no_command(entry):
    assert _hint("extras", entry) is None


def test_a_linked_folder_that_is_still_there_gets_the_link_command(tmp_path):
    work = tmp_path / "my pack"
    work.mkdir()
    (work / "cdui.plugin.toml").write_text("[plugin]\n", encoding="utf-8")
    entry = {"source_kind": "local", "source": str(work), "path": str(work)}
    assert _hint("my-pack", entry) == (
        f"cdui plugin link {_shell_quote(str(work))}")


@pytest.mark.parametrize("make", ["missing", "no-manifest", "relative", "quote"])
def test_a_linked_folder_that_cannot_be_linked_again_gets_no_command(
    tmp_path, make
):
    """No catalog command is made up for it either."""
    # Windows refuses `"` in a folder name; `$` is refused by the same rule.
    odd = "we$ird" if os.name == "nt" else 'we"ird'
    work = tmp_path / (odd if make == "quote" else "pack")
    if make != "missing":
        work.mkdir()
    if make in ("relative", "quote"):
        (work / "cdui.plugin.toml").write_text("[plugin]\n", encoding="utf-8")
    path = "pack" if make == "relative" else str(work)
    entry = {"source_kind": "local", "source": path, "path": path}
    assert _hint("rl-local", entry) is None


def test_an_unknown_source_kind_gets_no_command():
    assert _hint("x", {"source_kind": "zip"}) is None
    assert _hint("x", {}) is None
