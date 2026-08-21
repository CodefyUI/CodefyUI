"""Tests for plugin node namespacing (``c2:EduKNN`` registry keys).

Locks in the conflict-resolution contract:

    * Plugin nodes register under ``<plugin_id>:<NODE_NAME>`` so two plugins
      can both define ``EduKNN`` without overwriting each other.
    * Builtin nodes (loaded from ``app.nodes`` / ``app.custom_nodes``) keep
      bare ``NODE_NAME`` keys for backward compatibility with existing
      example graphs that ship in the main repo.
    * ``registry.get()`` accepts both forms — exact match wins, then a
      suffix fallback finds the unique match for bare lookups. Ambiguous
      bare lookups across multiple plugins log a warning and pick the
      alphabetically-first plugin so behavior stays deterministic.

These tests don't touch the live ``registry`` singleton — each test makes
its own ``NodeRegistry`` and seeds it directly so the cases are isolated
from whatever the test session's plugin install state happens to be.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from textwrap import dedent


from app.core import plugin_loader
from app.core.node_base import BaseNode, DataType, PortDefinition
from app.core.node_registry import NodeRegistry, _plugin_id_from_package, qualify
from app.core.preset_registry import PresetRegistry


# ── helpers ───────────────────────────────────────────────────────────────


def _make_node_class(name: str):
    """Build a throwaway BaseNode subclass with the given NODE_NAME."""

    class _Synthetic(BaseNode):
        NODE_NAME = name
        CATEGORY = "Test"
        DESCRIPTION = ""

        @classmethod
        def define_inputs(cls):
            return [PortDefinition(name="x", data_type=DataType.ANY)]

        @classmethod
        def define_outputs(cls):
            return [PortDefinition(name="y", data_type=DataType.ANY)]

        def execute(self, inputs, params, **_):
            return {}

    _Synthetic.__name__ = f"_{name}_FromPlugin_{id(name)}"
    return _Synthetic


# ── qualify() ─────────────────────────────────────────────────────────────


def test_qualify_uses_colon_separator():
    assert qualify("c2", "EduKNN") == "c2:EduKNN"


def test_qualify_bare_when_plugin_id_is_none():
    """Builtin nodes (no plugin id) keep their bare name."""
    assert qualify(None, "Linear") == "Linear"


def test_qualify_empty_plugin_id_treated_as_builtin():
    """Empty string is falsy → bare name, matches None case."""
    assert qualify("", "Linear") == "Linear"


# ── register / get round-trip ─────────────────────────────────────────────


def test_register_with_plugin_id_creates_qualified_key():
    reg = NodeRegistry()
    cls = _make_node_class("EduKNN")
    qualified = reg.register(cls, plugin_id="c2")
    assert qualified == "c2:EduKNN"
    assert "c2:EduKNN" in reg.nodes
    assert "EduKNN" not in reg.nodes  # bare key not created


def test_register_without_plugin_id_keeps_bare_key():
    reg = NodeRegistry()
    cls = _make_node_class("Linear")
    qualified = reg.register(cls)
    assert qualified == "Linear"
    assert "Linear" in reg.nodes


def test_two_plugins_same_node_name_coexist():
    """The canonical conflict scenario — both plugins keep their EduKNN."""
    reg = NodeRegistry()
    knn_c2 = _make_node_class("EduKNN")
    knn_other = _make_node_class("EduKNN")
    reg.register(knn_c2, plugin_id="c2")
    reg.register(knn_other, plugin_id="thirdparty")

    assert reg.get("c2:EduKNN") is knn_c2
    assert reg.get("thirdparty:EduKNN") is knn_other
    # Two distinct classes still in the registry
    assert reg.nodes["c2:EduKNN"] is not reg.nodes["thirdparty:EduKNN"]


# ── get() fallback ────────────────────────────────────────────────────────


def test_bare_lookup_falls_back_to_unique_qualified():
    """Old graphs with ``"type": "EduKNN"`` still load when only one plugin owns it."""
    reg = NodeRegistry()
    cls = _make_node_class("EduKNN")
    reg.register(cls, plugin_id="c2")

    assert reg.get("EduKNN") is cls


def test_bare_lookup_returns_none_when_no_match():
    reg = NodeRegistry()
    assert reg.get("DoesNotExist") is None


def test_qualified_lookup_returns_none_when_no_match():
    reg = NodeRegistry()
    reg.register(_make_node_class("EduKNN"), plugin_id="c2")
    assert reg.get("c99:EduKNN") is None


def test_ambiguous_bare_lookup_picks_alphabetically_first_and_warns(caplog):
    """Two plugins both export EduKNN; bare lookup deterministically picks alphabetically first."""
    reg = NodeRegistry()
    knn_c2 = _make_node_class("EduKNN")
    knn_zulu = _make_node_class("EduKNN")
    reg.register(knn_c2, plugin_id="c2")
    reg.register(knn_zulu, plugin_id="zulu")

    with caplog.at_level(logging.WARNING):
        resolved = reg.get("EduKNN")

    # "c2" < "zulu" alphabetically → "c2:EduKNN" wins
    assert resolved is knn_c2
    # And the warning message points the user at the qualified form.
    assert any("Ambiguous" in r.message for r in caplog.records)


def test_exact_qualified_match_bypasses_fallback_path():
    """When the registry key matches exactly there's no warning even with ambiguity."""
    reg = NodeRegistry()
    a = _make_node_class("EduKNN")
    b = _make_node_class("EduKNN")
    reg.register(a, plugin_id="c2")
    reg.register(b, plugin_id="zulu")

    # Explicit qualified lookup — no fallback, no ambiguity.
    assert reg.get("zulu:EduKNN") is b
    assert reg.get("c2:EduKNN") is a


# ── discover(): where the plugin id comes from ────────────────────────────
#
# The id a node is registered under has to be the id the plugin's MANIFEST
# declares, because that is the id every other subsystem already uses: the
# examples route's ``plugin:<id>`` path prefix, the install directory,
# ``cdui plugin list``, and the ``"type": "<id>:<NodeName>"`` that
# :func:`qualify` documents as the saved-graph form.
#
# That id cannot be recovered from the Python package name. A kebab-case id
# has to become snake_case to be importable at all (``official-template`` ->
# ``cdui_plugins.official_template``), and nothing in the package name says
# whether the underscore was a hyphen first. So the loader threads the real
# id through to ``discover``, which only falls back to reading it off the
# package name when no caller supplies one.


def _write_kebab_pack(root: Path, plugin_id: str, node_name: str) -> Path:
    """Install a minimal pack under *root* whose manifest id has a hyphen."""
    plugin_dir = root / plugin_id
    nodes = plugin_dir / "nodes"
    nodes.mkdir(parents=True)
    (plugin_dir / "cdui.plugin.toml").write_text(
        dedent(f"""            [plugin]
            id = "{plugin_id}"
            name = "Kebab pack {plugin_id}"
            version = "0.0.1"
            description = ""
            schema_version = 1
            """),
        encoding="utf-8",
    )
    (nodes / "__init__.py").write_text("", encoding="utf-8")
    (nodes / "demo_node.py").write_text(
        dedent(f"""            from app.core.node_base import BaseNode, DataType, PortDefinition


            class DemoNode(BaseNode):
                NODE_NAME = "{node_name}"
                CATEGORY = "Test"
                DESCRIPTION = ""

                @classmethod
                def define_inputs(cls):
                    return [PortDefinition(name="x", data_type=DataType.ANY)]

                @classmethod
                def define_outputs(cls):
                    return [PortDefinition(name="y", data_type=DataType.ANY)]

                def execute(self, inputs, params, **_):
                    return {{}}
            """),
        encoding="utf-8",
    )
    return plugin_dir


def test_kebab_case_pack_registers_under_its_hyphenated_manifest_id(tmp_path):
    """``official-template`` must not reach the registry as ``official_template``.

    Driven through the production pair of calls — ``install_plugin_finder``
    then ``discover`` — because the id is what gets lost between exactly
    those two when the loader hands over only the package name.
    """
    plugin_id = "kebab-case-pack"
    user_root = tmp_path / "userdata" / "plugins"
    user_root.mkdir(parents=True)
    _write_kebab_pack(user_root, plugin_id, "KebabDemo")
    lockfile = {
        "schema": 1,
        "plugins": {plugin_id: {"source_kind": "user", "enabled": True}},
    }

    reg = NodeRegistry()
    try:
        namespaces = plugin_loader.install_plugin_finder(
            tmp_path / "builtin", user_root, lockfile
        )
        assert [ns.plugin_id for ns in namespaces] == [plugin_id]
        for ns in namespaces:
            # The importable spelling and the real id travel together, which
            # is the whole point: the package name still has to be snake_case.
            assert ns.package_name == "cdui_plugins.kebab_case_pack.nodes"
            reg.discover(ns.nodes_dir, ns.package_name, plugin_id=ns.plugin_id)
    finally:
        plugin_loader.purge_plugin_modules(plugin_id)

    assert "kebab-case-pack:KebabDemo" in reg.nodes
    # The snake_case spelling belongs to the Python import path and nowhere
    # else — no graph, palette label, example path or CLI listing shows it.
    assert "kebab_case_pack:KebabDemo" not in reg.nodes


def test_plugin_discovery_wiring_keeps_the_hyphen_the_manifest_declares(
    tmp_path, monkeypatch
):
    """The wiring, not just the API.

    ``discover_plugin_nodes`` is the one loop the server lifespan, the
    project CLI and ``POST /api/plugins/reload`` all run, so a call site that
    stopped passing the id along would fail here rather than only on a
    student's canvas.
    """
    plugin_id = "kebab-case-reload"
    user_root = tmp_path / "userdata" / "plugins"
    user_root.mkdir(parents=True)
    _write_kebab_pack(user_root, plugin_id, "KebabReload")
    (user_root / "installed.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "plugins": {plugin_id: {"source_kind": "user", "enabled": True}},
            }
        ),
        encoding="utf-8",
    )
    # rediscover_all reads the lockfile itself rather than taking one, so the
    # user root has to be redirected as well as passed.
    monkeypatch.setattr(plugin_loader, "plugins_user_root", lambda: user_root)

    empty = tmp_path / "empty"  # no builtin nodes, custom nodes or presets
    reg = NodeRegistry()
    try:
        counts = plugin_loader.rediscover_all(
            reg,
            PresetRegistry(),
            nodes_dir=empty,
            custom_nodes_dir=empty,
            presets_dir=empty,
            builtin_root=empty,
            user_root=user_root,
        )
    finally:
        plugin_loader.purge_plugin_modules(plugin_id)

    assert counts["plugins"] == 1
    assert list(reg.nodes) == ["kebab-case-reload:KebabReload"]


# ── _plugin_id_from_package() ────────────────────────────────────────────


def test_plugin_id_detection_from_cdui_plugins_namespace():
    """Discoveries from ``cdui_plugins.<id>.nodes`` should detect ``<id>``.

    This is the FALLBACK only — the path taken when a caller passes no
    explicit ``plugin_id`` (builtin discovery, ad-hoc discovery in tests).
    It reads the Python package name, so the best it can ever return is the
    snake_case spelling: ``third_party_pack`` is the right answer here
    whether the manifest said ``third_party_pack`` or ``third-party-pack``,
    because the package name genuinely does not record which. The
    hyphen-preserving path is the one above, where the loader supplies the
    manifest id instead of leaving it to be guessed.
    """
    assert _plugin_id_from_package("cdui_plugins.c2.nodes") == "c2"
    assert _plugin_id_from_package("cdui_plugins.c2") == "c2"
    assert _plugin_id_from_package("cdui_plugins.third_party_pack.nodes") == "third_party_pack"


def test_plugin_id_detection_returns_none_for_builtin_namespace():
    """Builtin (``app.nodes``) discoveries must not gain a plugin prefix."""
    assert _plugin_id_from_package("app.nodes") is None
    assert _plugin_id_from_package("app.custom_nodes") is None
    assert _plugin_id_from_package("anything.else") is None
    # Edge case: empty-ish package names → None (defensive default).
    assert _plugin_id_from_package("") is None
    # ``cdui_plugins`` with nothing after it → None (no plugin id to extract).
    assert _plugin_id_from_package("cdui_plugins") is None
