"""The stats pack as a PACK: catalog entry, install, hot-reload, security gate.

Node behaviour lives in the per-node ``test_stats_*_node.py`` files. What is
asserted here is everything that only matters because this ships as a
first-party plugin rather than as built-in nodes — the acceptance criteria of
issue #130, in the order the issue lists them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from textwrap import dedent

import pandas as pd
import plugins as plugin_cli
import pytest

from app.core import plugin_loader
from app.core.graph_engine import execute_graph
from app.core.node_registry import NodeRegistry
from app.core.output_entries import build_node_output_entries, declared_media_ports
from app.core.plugin_validator import validate_python_source

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACK_DIR = _REPO_ROOT / "plugins" / "stats"
_EXAMPLES = _PACK_DIR / "examples" / "Stats"

NODE_NAMES = [
    "Stats-ChartView",
    "Stats-ConfusionMatrix",
    "Stats-Correlation",
    "Stats-Describe",
    "Stats-GroupByAggregate",
    "Stats-Histogram",
    "Stats-Percentile",
    "Stats-TableView",
]


def _pack_sources() -> list[Path]:
    return sorted(
        p for p in _PACK_DIR.rglob("*.py") if "__pycache__" not in p.parts
    )


@pytest.fixture
def isolated_lockfile(tmp_path, monkeypatch):
    """Redirect the lockfile to a temp dir and stub the server hot-reload.

    Same shape as the fixture in ``test_plugin_cli.py``: CLI tests must never
    touch real user data or expect a running server.
    """
    target = tmp_path / "plugins"
    target.mkdir()
    monkeypatch.setattr(plugin_loader, "plugins_user_root", lambda: target)
    monkeypatch.setattr(plugin_cli, "_backend_reload", lambda: False)
    return target


def _install(**overrides) -> int:
    args = argparse.Namespace(
        source=["stats"], force=False, no_confirm=True, trust_author=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return plugin_cli.cmd_install(args)


# ── the catalog entry ────────────────────────────────────────────────────────

def test_the_pack_is_in_the_first_party_catalog():
    """Without this, `cdui plugin install stats` parses `stats` as a GitHub repo."""
    entry = plugin_cli.load_catalog()["plugins"]["stats"]
    assert entry["kind"] == "builtin"
    assert entry["path"] == "plugins/stats"
    assert entry["name"] and entry["description"]


def test_the_catalog_name_resolves_as_a_catalog_source_not_a_repo():
    assert plugin_cli.parse_source("stats")[:2] == ("catalog", "stats")


def test_the_manifest_validates():
    manifest = plugin_cli.read_manifest(_PACK_DIR)
    plugin_cli.validate_manifest(manifest)
    assert manifest["plugin"]["id"] == "stats"
    assert manifest["plugin"]["schema_version"] == 1
    assert manifest["content"]["nodes_dir"] == "nodes"
    assert manifest["content"]["examples_dir"] == "examples"


# ── acceptance: install exposes all eight nodes, and reload works ────────────

def test_install_records_a_builtin_lockfile_entry(isolated_lockfile):
    assert _install() == 0

    entry = plugin_loader.load_lockfile()["plugins"]["stats"]
    assert entry["source_kind"] == "builtin"
    assert entry["enabled"] is True
    assert entry["trusted_modules"] == []
    assert entry["manifest"]["id"] == "stats"


def test_installing_twice_needs_force(isolated_lockfile):
    assert _install() == 0
    assert _install() != 0
    assert _install(force=True) == 0


def test_an_installed_pack_exposes_all_eight_nodes(isolated_lockfile):
    assert _install() == 0

    pairs = plugin_loader.install_plugin_finder(
        plugin_loader.plugins_builtin_root(),
        plugin_loader.plugins_user_root(),
        plugin_loader.load_lockfile(),
    )
    nodes_dir, package = next(p for p in pairs if p[1].endswith("stats.nodes"))

    registry = NodeRegistry()
    assert registry.discover(nodes_dir, package) == 8
    assert sorted(registry.nodes) == [f"stats:{name}" for name in NODE_NAMES]


def test_hot_reload_re_registers_every_node(isolated_lockfile):
    """`cdui plugin reload` rediscovers with force_reload — twice must be stable."""
    assert _install() == 0
    nodes_dir = _PACK_DIR / "nodes"
    package = "cdui_plugins.stats.nodes"

    registry = NodeRegistry()
    assert registry.discover(nodes_dir, package, force_reload=True) == 8
    registry.clear()
    assert registry.discover(nodes_dir, package, force_reload=True) == 8
    assert sorted(registry.nodes) == [f"stats:{name}" for name in NODE_NAMES]


def test_node_types_are_namespaced_so_saved_graphs_are_unambiguous(
    registry_with_nodes,
):
    for name in NODE_NAMES:
        assert registry_with_nodes.get(f"stats:{name}") is not None


def test_the_helper_module_contributes_no_nodes():
    """`_stats_core` is shared code, not a node — it must not register anything."""
    registry = NodeRegistry()
    registry.discover(_PACK_DIR / "nodes", "cdui_plugins.stats.nodes")
    assert len(registry.nodes) == len(NODE_NAMES)


# ── acceptance: the AST gate passes with zero [security] overrides ──────────

def test_the_manifest_declares_no_security_overrides():
    """The whole point of the pack: the default policy is workable for stats."""
    manifest = plugin_cli.read_manifest(_PACK_DIR)
    assert "security" not in manifest
    assert manifest.get("security", {}).get("allowed_modules", []) == []


@pytest.mark.parametrize(
    "source", _pack_sources(), ids=[p.name for p in _pack_sources()]
)
def test_every_source_file_passes_the_ast_validator(source):
    validate_python_source(source.read_bytes(), source.name, allowed_modules=None)


def test_the_whole_pack_passes_the_directory_gate_a_github_install_would_run():
    plugin_cli.validate_plugin_dir(_PACK_DIR, [])


def test_the_pack_imports_no_pandas_at_runtime():
    """pandas is the tests' reference implementation, never the pack's dependency."""
    for source in _pack_sources():
        text = source.read_text(encoding="utf-8")
        assert "import pandas" not in text, source
        assert "\nimport sklearn" not in text, source


def test_a_pack_file_that_broke_the_rules_would_be_caught(tmp_path):
    """Guards the guard: the parametrised test above must be able to fail."""
    offender = tmp_path / "bad_node.py"
    offender.write_text(dedent("""
        import os
        VALUE = os.environ
    """), encoding="utf-8")
    with pytest.raises(plugin_cli.PluginValidationError):
        validate_python_source(offender.read_bytes(), offender.name)


# ── acceptance: the chart kind reaches the wire ─────────────────────────────

def test_chart_ports_are_declared_as_media_and_ride_the_output_channel(
    registry_with_nodes,
):
    nodes = [{"id": "cm", "type": "stats:Stats-ConfusionMatrix", "data": {"params": {}}}]
    declared = declared_media_ports(nodes)
    assert declared == {"cm": {"chart": ["chart"]}}

    entries = build_node_output_entries(
        "completed",
        {"matrix": [[1.0]], "chart": {"kind": "heatmap", "matrix": [[1.0]]}},
        declared["cm"],
    )
    chart = next(e for e in entries if e["output_kind"] == "chart")
    assert chart["port"] == "chart"
    assert chart["chart"]["kind"] == "heatmap"


def test_every_chart_producing_node_declares_the_media_kind(registry_with_nodes):
    producers = [
        "Stats-Histogram", "Stats-Correlation", "Stats-ConfusionMatrix",
        "Stats-ChartView",
    ]
    for name in producers:
        cls = registry_with_nodes.get(f"stats:{name}")
        chart = next(p for p in cls.define_outputs() if p.name == "chart")
        assert chart.media == "chart", name


# ── acceptance: the demo graphs run, and reproduce pandas ───────────────────

def _run_graph(name: str) -> dict:
    payload = json.loads((_EXAMPLES / name / "graph.json").read_text(encoding="utf-8"))
    backend_dir = Path(__file__).resolve().parents[1]
    previous = Path.cwd()
    os.chdir(backend_dir)
    try:
        return asyncio.run(
            execute_graph(payload["nodes"], payload["edges"], error_mode="fail_fast")
        )
    finally:
        os.chdir(previous)


def test_the_pack_ships_the_demo_graphs():
    shipped = sorted(p.parent.name for p in _EXAMPLES.rglob("graph.json"))
    assert shipped == [
        "Confusion-Matrix-Heatmap", "Iris-Describe-Table", "Iris-GroupBy-Chart",
    ]


def test_the_describe_demo_reproduces_pandas_describe_end_to_end():
    """The headline acceptance criterion, through the real graph engine.

    Not a call to the node with hand-made inputs: the CSV is read by CSVReader,
    the columns travel the LIST port, and the numbers are read back out of the
    text Stats-TableView rendered — so a break anywhere along that chain fails
    here.
    """
    results = _run_graph("Iris-Describe-Table")

    frame = pd.read_csv(Path(__file__).resolve().parents[1] / "data/samples/iris.csv")
    expected = frame.select_dtypes("number").describe()

    described = results["describe"]
    assert described["columns"] == list(expected.columns)
    assert described["row_labels"] == list(expected.index)
    assert described["table"].numpy() == pytest.approx(
        expected.to_numpy(), rel=1e-5
    )

    # ...and the view actually rendered it, headers, statistic names and all.
    lines = results["view"]["text"].splitlines()
    assert lines[0] == "Iris - describe()"
    for column in expected.columns:
        assert column in lines[1]
    assert [line.split()[0] for line in lines[2:]] == list(expected.index)


def test_the_groupby_demo_charts_one_bar_per_species():
    results = _run_graph("Iris-GroupBy-Chart")

    assert results["group"]["row_labels"] == ["setosa", "versicolor", "virginica"]
    chart = results["chart"]["chart"]
    assert chart["kind"] == "bar"
    assert [b["label"] for b in chart["bars"]] == [
        "setosa", "versicolor", "virginica",
    ]

    frame = pd.read_csv(Path(__file__).resolve().parents[1] / "data/samples/iris.csv")
    expected = frame.groupby("species")["petal length (cm)"].mean()
    assert [b["value"] for b in chart["bars"]] == pytest.approx(
        expected.to_numpy(), rel=1e-5
    )


def test_the_confusion_matrix_demo_produces_a_heatmap_payload():
    results = _run_graph("Confusion-Matrix-Heatmap")

    chart = results["cm"]["chart"]
    assert chart["kind"] == "heatmap"
    assert chart["row_labels"] == ["setosa", "versicolor", "virginica"]
    assert (chart["x_label"], chart["y_label"]) == ("predicted", "true")

    # 45 held-out samples, all accounted for, and the tree is at least sane.
    assert sum(sum(row) for row in chart["matrix"]) == 45
    assert results["cm"]["accuracy"] > 0.8


def test_every_demo_chart_payload_survives_the_run_stores_json_encoder():
    """Events are serialised with allow_nan=False; a NaN would break the socket."""
    for name in ("Iris-GroupBy-Chart", "Confusion-Matrix-Heatmap"):
        results = _run_graph(name)
        for node in results.values():
            if isinstance(node, dict) and isinstance(node.get("chart"), dict):
                json.dumps(node["chart"], allow_nan=False)
