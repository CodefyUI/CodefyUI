"""The plugin catalog may only advertise nodes that actually install.

`cdui plugin search` and `cdui plugin info` print `plugins/registry.json`'s
and each manifest's prose verbatim (`scripts/plugins.py:1509,1657,1911`), and
that prose is where a student learns what to type into the node palette. A
name in there that no `.py` file registers is not a documentation nit: the
student types it, the palette returns nothing, and the lab stops.

Fourteen such names shipped before this file existed -- `Edu-Conv2d`,
`Edu-MaxPool2d`, `Edu-RNNCell`, `Edu-LSTMCell`, `Edu-LayerNorm`,
`Edu-Resample`, `Edu-DenoiseStep`, `Edu-SVM`, `Edu-DecisionTree`,
`Edu-SlidingWindow`, `Edu-VectorSimilarity`, `Edu-PPOClip`, `Edu-GRPO`,
`Edu-PreferenceLoss`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import dev  # scripts/dev.py -- conftest puts scripts/ on sys.path
from app.core.plugins.catalog import RESERVED_PLUGIN_IDS, validate_catalog
from app.core.plugins.manifest import REPO_RE

PLUGINS_DIR = dev.ROOT / "plugins"
REGISTRY = PLUGINS_DIR / "registry.json"

# Names shaped like a pack node. Bare CamelCase is deliberately excluded --
# the prose legitimately says things like "policy gradient" and "max-pool" as
# English, and only the hyphenated form is what a student types.
NODE_NAME_RE = re.compile(r"\b((?:Edu|Stats)-[A-Za-z0-9]+)\b")


def _registered_node_names() -> set[str]:
    """Every NODE_NAME a pack actually registers, read off disk."""
    names: set[str] = set()
    for path in PLUGINS_DIR.rglob("*_node.py"):
        text = path.read_text(encoding="utf-8")
        names.update(re.findall(r"""NODE_NAME\s*=\s*["']([^"']+)["']""", text))
    return names


def _pack_manifests() -> list[Path]:
    return sorted(PLUGINS_DIR.glob("*/cdui.plugin.toml"))


@pytest.fixture(scope="module")
def registered() -> set[str]:
    names = _registered_node_names()
    assert names, "found no NODE_NAME anywhere under plugins/ -- scan is broken"
    return names


def _registry_entries() -> dict[str, dict]:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))["plugins"]


def test_registry_advertises_only_nodes_that_exist(registered: set[str]):
    # Only the ``builtin`` rows are checked against this disk, because only
    # their nodes are ON this disk: a ``github`` row's pack lives in another
    # repository, so a name in its prose cannot be confirmed or denied here,
    # and scanning it would fail the honest entry rather than the dishonest
    # one. Every builtin row is still scanned whole (name and description
    # alike) -- the rule below is unchanged for them.
    builtin = {
        pack_id: entry
        for pack_id, entry in _registry_entries().items()
        if entry.get("kind") == "builtin"
    }
    text = json.dumps(builtin, ensure_ascii=False)
    phantom = sorted({n for n in NODE_NAME_RE.findall(text) if n not in registered})
    assert not phantom, (
        f"plugins/registry.json names nodes that no pack registers: {phantom}. "
        f"`cdui plugin search` prints these, and the palette has no such node."
    )


@pytest.mark.parametrize("manifest", _pack_manifests(), ids=lambda p: p.parent.name)
def test_manifest_advertises_only_nodes_that_exist(manifest: Path, registered: set[str]):
    text = manifest.read_text(encoding="utf-8")
    phantom = sorted({n for n in NODE_NAME_RE.findall(text) if n not in registered})
    assert not phantom, (
        f"{manifest.parent.name}/cdui.plugin.toml names nodes that do not exist: "
        f"{phantom}"
    )


def test_every_pack_in_the_registry_has_a_manifest():
    """A ``builtin`` row names a directory that ships here; a ``github`` row
    names a repository instead, and the two are installed by completely
    different code -- so each kind is checked against the thing it claims."""
    for pack_id, entry in _registry_entries().items():
        if entry.get("kind") == "github":
            repo = entry.get("repo", "")
            assert REPO_RE.match(repo), f"{pack_id}: repo {repo!r} is not owner/repo"
            assert "path" not in entry, (
                f"{pack_id}: a github entry claiming a path is dropped by "
                f"validate_catalog -- the two kinds install by different code"
            )
            if repo.startswith("CodefyUI/"):
                assert entry.get("official") is True, (
                    f"{pack_id}: a pack published by the CodefyUI org is what "
                    f"'official' means; the Plugin Center prints that badge"
                )
            assert entry.get("homepage", "").startswith(
                f"https://github.com/{repo}"
            ), f"{pack_id}: homepage must point at the repository it names"
            assert not (PLUGINS_DIR / pack_id).exists(), (
                f"{pack_id}: a github pack must not also sit in plugins/ -- "
                f"whichever copy loaded first would decide what the user got"
            )
            continue
        path = dev.ROOT / entry["path"]
        assert path.is_dir(), f"{pack_id}: registry path {entry['path']} does not exist"
        assert (path / "cdui.plugin.toml").is_file(), f"{pack_id}: no cdui.plugin.toml"


def test_catalog_validates():
    """Every row this repository ships survives ``validate_catalog``.

    It drops a malformed row with a log line rather than raising, so a typo
    in one entry costs the Plugin Center a card and costs CI nothing -- which
    is why the file has to be checked here instead.
    """
    raw = _registry_entries()
    entries = validate_catalog(json.loads(REGISTRY.read_text(encoding="utf-8")))
    assert set(entries) == set(raw), (
        f"validate_catalog dropped: {sorted(set(raw) - set(entries))}"
    )


def test_github_catalog_entries_are_not_reserved_builtin_ids():
    """A github row may not claim an id the routing table or a shipped pack
    already owns: ``/api/plugins/<id>`` and ``plugins/<id>/`` would both
    resolve to something other than the pack the catalog advertises."""
    entries = _registry_entries()
    github_ids = {
        pack_id for pack_id, entry in entries.items() if entry.get("kind") == "github"
    }
    builtin_ids = {
        pack_id for pack_id, entry in entries.items() if entry.get("kind") == "builtin"
    }
    assert github_ids, "the catalog is expected to advertise the org's own plugins"
    assert not (github_ids & RESERVED_PLUGIN_IDS)
    assert not (github_ids & builtin_ids)


def _preset_names() -> set[str]:
    names: set[str] = set()
    for path in (dev.ROOT / "backend" / "app" / "presets").glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        name = data.get("preset_name") or data.get("name")
        if name:
            names.add(str(name))
    return names


def test_example_graphs_only_reference_nodes_that_exist(registered: set[str]):
    """A shipped example that names a missing node fails on Run, not on load."""
    presets = _preset_names()
    broken: list[str] = []
    for graph_path in PLUGINS_DIR.rglob("examples/**/graph.json"):
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        for node in graph.get("nodes", []):
            node_type = str(node.get("type", ""))
            # Core nodes are bare and covered by test_builtin_examples.py.
            if ":" not in node_type:
                continue
            namespace, _, name = node_type.partition(":")
            # `preset:` is the built-in preset namespace, not a pack.
            known = presets if namespace == "preset" else registered
            if name not in known:
                rel = graph_path.relative_to(dev.ROOT).as_posix()
                broken.append(f"{rel} -> {node_type}")
    assert not broken, f"example graphs reference nodes that do not exist: {broken}"
