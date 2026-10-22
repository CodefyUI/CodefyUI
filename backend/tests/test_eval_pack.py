"""The eval pack as a PACK: catalog entry, manifest, discovery, gate, example.

Node behaviour lives in ``test_eval_nodes.py`` (offline, no transformers)
and ``test_eval_hf_local.py`` (real transformers models, skipped without the
library). What is asserted here only matters because this ships as a
first-party plugin pack (issue #139).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import plugins as plugin_cli
import pytest

from app.core import plugin_loader
from app.core.graph_engine import validate_graph
from app.core.node_registry import NodeRegistry
from app.core.plugin_validator import validate_python_source
from app.core.plugins.inspect import inspect_builtin
from app.core.script_policy import TIER0_DENIED_ATTRS
from cdui_plugins.eval.nodes._eval_core import is_commit_sha

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACK_DIR = _REPO_ROOT / "plugins" / "eval"
_EXAMPLE = _PACK_DIR / "examples" / "Eval" / "AB-Text-Classification" / "graph.json"
_ZH_NODES = _REPO_ROOT / "frontend" / "src" / "i18n" / "nodeLocales" / "zh-TW.ts"

NODE_NAMES = [
    "Eval-BLEU",
    "Eval-BatchInference",
    "Eval-CompareRuns",
    "Eval-ExactMatch",
    "Eval-F1Score",
    "Eval-HFModelLoader",
    "Eval-HFProcessorLoader",
    "Eval-JSONLDataset",
    "Eval-PromptTemplate",
    "Eval-ROUGE-L",
    "Eval-TopKAccuracy",
]

#: The palette summary caps ``test_api_nodes.py`` holds the built-ins to.
MAX_DESCRIPTION_CHARS = 56
MAX_ZH_DESCRIPTION_CHARS = 28


def _pack_sources() -> list[Path]:
    found = sorted(p for p in _PACK_DIR.rglob("*.py") if "__pycache__" not in p.parts)
    assert found, f"no Python sources under {_PACK_DIR} -- the AST gate would be a no-op"
    return found


def _discover() -> NodeRegistry:
    namespaces = plugin_loader.install_plugin_finder(
        plugin_loader.plugins_builtin_root(),
        _REPO_ROOT / "_phantom_user_root_for_tests",
        {"schema": 1, "plugins": {"eval": {"source_kind": "builtin", "source": "eval",
                                           "enabled": True}}},
    )
    ns = next(n for n in namespaces if n.plugin_id == "eval")
    registry = NodeRegistry()
    registry.discover(ns.nodes_dir, ns.package_name, plugin_id=ns.plugin_id)
    return registry


# ── catalog and manifest ─────────────────────────────────────────────────────

def test_the_pack_is_in_the_first_party_catalog():
    entry = plugin_cli.load_catalog()["plugins"]["eval"]
    assert entry["kind"] == "builtin"
    assert entry["path"] == "plugins/eval"
    assert entry["name"] and entry["description"]
    # Bilingual like its siblings: English sentence, Traditional Chinese glosses.
    assert re.search(r"[一-鿿]", entry["description"])
    for name in NODE_NAMES:
        assert name in entry["description"], name


def test_the_catalog_name_resolves_as_a_catalog_source():
    assert plugin_cli.parse_source("eval")[:2] == ("catalog", "eval")


def test_the_manifest_validates_and_declares_transformers_as_a_pack_dependency():
    manifest = plugin_cli.read_manifest(_PACK_DIR)
    plugin_cli.validate_manifest(manifest)
    assert manifest["plugin"]["id"] == "eval"
    assert manifest["content"] == {"nodes_dir": "nodes", "examples_dir": "examples"}
    assert manifest["python_deps"] == {"transformers": ">=5.0,<6"}
    assert "security" not in manifest


def test_an_install_would_bring_transformers():
    found = inspect_builtin("eval", lockfile={"schema": 1, "plugins": {}})
    assert found.python_deps == {"transformers": ">=5.0,<6"}
    assert found.capabilities == ()


# ── discovery ────────────────────────────────────────────────────────────────

def test_discovery_registers_exactly_the_eleven_nodes():
    registry = _discover()
    assert sorted(registry.nodes) == [f"eval:{name}" for name in NODE_NAMES]


def test_the_suite_registry_knows_the_pack(registry_with_nodes):
    for name in NODE_NAMES:
        assert registry_with_nodes.get(f"eval:{name}") is not None, name


def test_the_node_modules_import_without_transformers():
    """The palette scan imports every node module at startup, so none may
    import transformers at module level -- an install without it would lose
    the whole palette, not just this pack."""
    for source in _pack_sources():
        for line in source.read_text(encoding="utf-8").splitlines():
            if line.startswith(("import transformers", "from transformers")):
                pytest.fail(f"{source.name} imports transformers at module level")


@pytest.mark.parametrize("name", NODE_NAMES)
def test_every_node_has_an_english_summary_within_the_palette_cap(name):
    cls = _discover().get(f"eval:{name}")
    assert cls.DESCRIPTION and len(cls.DESCRIPTION) <= MAX_DESCRIPTION_CHARS
    assert cls.DETAILS
    texts = [cls.DESCRIPTION, cls.DETAILS]
    texts += [p.description for p in cls.define_inputs() + cls.define_outputs()]
    texts += [p.description for p in cls.define_params()]
    assert not [t for t in texts if re.search(r"[一-鿿]", t or "")], (
        "English in the node; the Chinese goes in nodeLocales/zh-TW.ts")


@pytest.mark.parametrize("name", NODE_NAMES)
def test_every_node_has_a_complete_zh_tw_entry(name):
    catalog = _ZH_NODES.read_text(encoding="utf-8")
    block = re.search(rf"\n  'eval:{re.escape(name)}': \{{(.*?)\n  \}},", catalog, re.DOTALL)
    assert block, f"eval:{name} has no zh-TW entry"
    body = block.group(1)
    description = re.search(r"\n    description: '([^']*)'", body)
    assert description and len(description.group(1)) <= MAX_ZH_DESCRIPTION_CHARS
    assert "\n    details:" in body
    cls = _discover().get(f"eval:{name}")
    for kind, items in (("params", cls.define_params()), ("inputs", cls.define_inputs()),
                        ("outputs", cls.define_outputs())):
        if not items:
            continue
        section = re.search(rf"\n    {kind}: \{{(.*?)\n    \}},", body, re.DOTALL)
        assert section, f"eval:{name} has no zh-TW {kind}"
        for item in items:
            assert re.search(rf"\n      {re.escape(item.name)}:\s", section.group(1)), (
                f"eval:{name}.{kind}.{item.name} renders in English under zh-TW")


def test_nodes_that_hand_out_models_or_log_metrics_are_not_cacheable():
    registry = _discover()
    for name in ("Eval-HFModelLoader", "Eval-BatchInference", "Eval-ExactMatch",
                 "Eval-F1Score", "Eval-TopKAccuracy", "Eval-BLEU", "Eval-ROUGE-L",
                 "Eval-JSONLDataset"):
        assert registry.get(f"eval:{name}").cacheable is False, name


# ── the AST gate ─────────────────────────────────────────────────────────────

#: The one module the pack imports beyond Tier 0, and the one file that does.
#: ``_eval_hub`` needs a ``threading.Lock``: transformers' lazy module is not
#: safe to resolve from two loaders at once, and the engine runs an A/B
#: graph's loaders concurrently. Built-in packs skip the gate at install, so
#: this costs nothing there; a GitHub fork would declare it in
#: ``[security].allowed_modules``.
_BEYOND_TIER0 = {"_eval_hub.py": ["threading"]}


@pytest.mark.parametrize("source", _pack_sources(), ids=[p.name for p in _pack_sources()])
def test_every_source_file_passes_the_ast_validator(source):
    """Every file passes at Tier 0 except for the one allowance above."""
    validate_python_source(
        source.read_bytes(), source.name,
        allowed_modules=_BEYOND_TIER0.get(source.name), denied_attributes=TIER0_DENIED_ATTRS,
    )


def test_the_allowance_is_needed_and_is_only_threading():
    """Guards the guard: without the allowance the gate refuses the file."""
    source = _PACK_DIR / "nodes" / "_eval_hub.py"
    with pytest.raises(plugin_cli.PluginValidationError, match="threading"):
        validate_python_source(source.read_bytes(), source.name,
                               denied_attributes=TIER0_DENIED_ATTRS)


def test_the_whole_pack_passes_the_directory_gate():
    plugin_cli.validate_plugin_dir(_PACK_DIR, ["threading"])


# ── the example graph ────────────────────────────────────────────────────────

def _example() -> dict:
    return json.loads(_EXAMPLE.read_text(encoding="utf-8"))


def test_the_example_validates():
    graph = _example()
    assert validate_graph(graph["nodes"], graph["edges"]) == []


def test_the_example_pins_both_checkpoints_to_a_commit():
    """A note quotes the numbers, so the weights behind them cannot move."""
    loaders = [n for n in _example()["nodes"] if n.get("type") == "eval:Eval-HFModelLoader"]
    assert len(loaders) == 2
    for loader in loaders:
        assert is_commit_sha(loader["data"]["params"]["revision"]), loader["id"]


def test_the_example_reads_the_bundled_sample():
    data = next(n for n in _example()["nodes"] if n.get("type") == "eval:Eval-JSONLDataset")
    path = data["data"]["params"]["path"]
    assert path == "eval:samples/sst2-mini-20.jsonl"
    assert (_PACK_DIR / "samples" / "sst2-mini-20.jsonl").is_file()


def test_the_example_compares_the_two_runs_and_charts_the_outcomes():
    graph = _example()
    into = {(e["target"], e.get("targetHandle")): (e["source"], e.get("sourceHandle"))
            for e in graph["edges"] if e.get("type") != "trigger"}
    assert into[("compare", "table_a")] == ("infer_a", "predictions")
    assert into[("compare", "table_b")] == ("infer_b", "predictions")
    assert into[("proc_a", "model")] == ("model_a", "model")
    assert into[("chart", "table")] == ("compare", "summary_table")
    infer_b = next(n for n in graph["nodes"] if n["id"] == "infer_b")
    assert infer_b["data"]["params"]["label_map"] == "LABEL_0=negative, LABEL_1=positive"


_VQA_EXAMPLE = _PACK_DIR / "examples" / "Eval" / "VQA-Shapes" / "graph.json"


@pytest.mark.parametrize("path", [_EXAMPLE, _VQA_EXAMPLE], ids=["ab", "vqa"])
def test_every_example_note_is_bilingual_and_sits_where_its_binding_says(path):
    graph = json.loads(path.read_text(encoding="utf-8"))
    by_id = {n["id"]: n for n in graph["nodes"]}
    notes = [n for n in graph["nodes"] if n.get("type") == "note"]
    assert notes
    for note in notes:
        data = note["data"]
        english, _, chinese = data["noteContent"].partition("\n\n")
        assert english and re.search(r"[一-鿿]", chinese), note["id"]
        assert not re.search(r"[一-鿿]", english), note["id"]
        bound = by_id[data["boundToNodeId"]]["position"]
        offset = data["boundOffset"]
        assert note["position"] == {"x": bound["x"] + offset["x"],
                                    "y": bound["y"] + offset["y"]}, note["id"]
