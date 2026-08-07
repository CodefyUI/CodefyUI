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


def test_registry_advertises_only_nodes_that_exist(registered: set[str]):
    text = REGISTRY.read_text(encoding="utf-8")
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
    entries = json.loads(REGISTRY.read_text(encoding="utf-8"))["plugins"]
    for pack_id, entry in entries.items():
        path = dev.ROOT / entry["path"]
        assert path.is_dir(), f"{pack_id}: registry path {entry['path']} does not exist"
        assert (path / "cdui.plugin.toml").is_file(), f"{pack_id}: no cdui.plugin.toml"


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
