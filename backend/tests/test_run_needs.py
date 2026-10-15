"""``core.run_needs``: which runs need the data mount (#782).

The coverage table is the point: every built-in node whose source touches
the data directories must be listed in ``run_needs.COVERAGE``, and each
rule is pinned by a case below.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from app.core import run_needs
from app.core.node_base import BaseNode, ParamDefinition, ParamType
from app.core.node_registry import registry

NODES_DIR = Path(__file__).resolve().parents[1] / "app" / "nodes"

#: What a node file must contain to count as touching the data directories:
#: the four settings, the data root, the write rule, and the artifact and
#: checkpoint hooks.
_TOUCHES = re.compile(
    r"DATA_FILES_DIR|IMAGES_DIR|MODELS_DIR|MEDIA_DIR|data_root\(|"
    r"resolve_data_path|log_artifact|save_interrupt_checkpoint|"
    r"open_run_writer|ParamType\.(DATA_FILE|IMAGE_FILE|MODEL_FILE)")
_NODE_NAME = re.compile(r'NODE_NAME\s*=\s*"([^"]+)"')


def _touching_nodes() -> dict[str, str]:
    found: dict[str, str] = {}
    for path in sorted(NODES_DIR.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if _TOUCHES.search(source):
            for name in _NODE_NAME.findall(source):
                found[name] = str(path.relative_to(NODES_DIR))
    return found


def test_every_node_that_touches_the_data_directories_is_in_the_table():
    touching = _touching_nodes()
    # The issue counted 17 node files; the scan must not silently go blind.
    assert len(set(touching.values())) >= 17, touching
    missing = sorted(set(touching) - set(run_needs.COVERAGE))
    assert missing == [], (
        f"{missing} read or write under the data directories but "
        "run_needs.COVERAGE does not say how a run of them is placed; add a "
        "rule (or a 'none: ...' entry) and a case to test_run_needs.py")


def test_the_table_names_only_registered_nodes():
    unknown = [name for name in run_needs.COVERAGE
               if registry.get(name) is None]
    assert unknown == []


# ── one node at a time ───────────────────────────────────────────────────

CASES: list[tuple[str, dict[str, Any], list[str] | None]] = [
    # rule 1: typed params
    ("CSVReader", {"path": "train.csv"}, ["train.csv"]),
    ("CSVReader", {"path": "data/samples/iris.csv"}, None),
    ("CSVReader", {"path": ""}, None),
    ("ImageReader", {"path": "cat.png"}, ["cat.png"]),
    ("ModelLoader", {"path": "net.safetensors"}, ["net.safetensors"]),
    ("CheckpointLoader", {"path": "run.pt"}, ["run.pt"]),
    ("TextCorpusDataset", {"source": "local_file", "local_path": "c.txt"},
     ["c.txt"]),
    ("TextCorpusDataset", {"source": "huggingface", "local_path": "c.txt"},
     None),
    ("DocumentLoader", {"source": "uploaded_file", "file": "a.md"}, ["a.md"]),
    # rule 2: writers, whatever their params
    ("CheckpointSaver", {"path": "run.pt"}, []),
    ("ModelSaver", {}, []),
    ("ImageWriter", {}, []),
    ("VideoWrite", {}, []),
    # rule 3: node rules
    ("TrainingLoop", {}, None),
    ("TrainingLoop", {"checkpoint_every": 2}, []),
    ("TrainingLoop", {"checkpoint_every_steps": 50}, []),
    ("TrainingLoop", {"tensorboard": True}, []),
    ("TrainingLoop", {"checkpoint_every": "x"}, None),
    ("FileReader", {"path": "notes.txt"}, ["notes.txt"]),
    ("FileReader", {"path": ""}, None),
    ("ImageFolderDataset", {"path": "pets"}, ["pets"]),
    ("ImageFolderDataset", {"path": ""}, None),
    ("ImageBatchReader", {"directory": "frames"}, ["frames"]),
    ("VideoLoad", {}, ["clip.mp4"]),  # its default names a media file
    ("DocumentLoader", {"source": "directory"}, None),  # bundled sample
    ("DocumentLoader", {"source": "directory", "directory": "docs"},
     ["docs"]),
    ("GraphInput", {"type": "image", "default": "in.png"}, ["in.png"]),
    ("GraphInput", {"type": "image", "default": "in.png", "value": "b64"},
     None),
    ("GraphInput", {"type": "string", "default": "hello"}, None),
    # not needs
    ("Dataset", {}, None),
    ("HuggingFaceDataset", {}, None),
    ("LMTokenizedDataset", {"cache": True}, None),
    ("DiffusionTrainingLoop", {}, None),
    ("TensorCreate", {}, None),
]


@pytest.mark.parametrize(("node_type", "params", "expected"), CASES,
                         ids=[f"{c[0]}-{i}" for i, c in enumerate(CASES)])
def test_node_needs(node_type, params, expected):
    assert run_needs.node_needs(node_type, params) == expected


def test_every_table_entry_has_a_case():
    cased = {case[0] for case in CASES}
    assert set(run_needs.COVERAGE) <= cased


def test_a_plugin_nodes_typed_param_counts(monkeypatch):
    class PluginReader(BaseNode):
        NODE_NAME = "acme:Reader"

        @classmethod
        def define_inputs(cls):
            return []

        @classmethod
        def define_outputs(cls):
            return []

        @classmethod
        def define_params(cls):
            return [ParamDefinition(name="mode", param_type=ParamType.SELECT,
                                    default="file", options=["file", "none"]),
                    ParamDefinition(name="src", param_type=ParamType.DATA_FILE,
                                    default="",
                                    visible_when={"mode": "file"})]

        def execute(self, inputs, params, context=None):  # pragma: no cover
            return {}

    real_get = registry.get
    monkeypatch.setattr(registry, "get", lambda name: PluginReader
                        if name == "acme:Reader" else real_get(name))
    assert run_needs.node_needs("acme:Reader", {"src": "x.jsonl"}) == [
        "x.jsonl"]
    # A hidden param's leftover value is "not set".
    assert run_needs.node_needs("acme:Reader", {"mode": "none",
                                                "src": "x.jsonl"}) is None


# ── whole graphs ─────────────────────────────────────────────────────────


def _node(node_id: str, node_type: str, **params: Any) -> dict[str, Any]:
    return {"id": node_id, "type": node_type, "data": {"params": params}}


def _pure_tensor_graph() -> dict[str, Any]:
    return {"nodes": [_node("s", "Start"),
                      _node("x", "TensorCreate", shape="2,2", fill="ones",
                            value=0, requires_grad=False)],
            "edges": [{"id": "t", "source": "s", "target": "x",
                       "sourceHandle": "trigger", "type": "trigger"}]}


def test_a_pure_tensor_graph_needs_nothing():
    assert run_needs.needs(_pure_tensor_graph(), label="data") == {
        "mount": None, "files": []}


def test_a_csv_graph_needs_the_mount_and_lists_its_upload():
    graph = _pure_tensor_graph()
    _triggered(graph, _node("csv", "CSVReader", path="b.csv"))
    _triggered(graph, _node("csv2", "CSVReader", path="a.csv"))
    _triggered(graph, _node("csv3", "CSVReader", path="a.csv"))
    assert run_needs.needs(graph, label="nas") == {
        "mount": "nas", "files": ["a.csv", "b.csv"]}


def _triggered(graph: dict[str, Any], node: dict[str, Any]) -> None:
    graph["nodes"].append(node)
    graph["edges"].append({"id": f"t-{node['id']}", "source": "s",
                           "target": node["id"], "sourceHandle": "trigger",
                           "type": "trigger"})


def test_a_bypassed_or_unreachable_reader_is_not_a_need():
    graph = _pure_tensor_graph()
    reader = _node("csv", "CSVReader", path="z.csv")
    reader["data"]["bypassed"] = True
    _triggered(graph, reader)
    graph["nodes"].append(_node("draft", "CSVReader", path="d.csv"))
    assert run_needs.needs(graph, label="data") == {
        "mount": None, "files": []}


def test_a_node_inside_a_subgraph_counts():
    graph = _pure_tensor_graph()
    _triggered(graph, {"id": "blk", "type": "subgraph:sg1",
                       "data": {"params": {}}})
    graph["subgraphs"] = [{
        "id": "sg1", "name": "inner",
        "nodes": [_node("c", "CSVReader", path="in.csv")], "edges": [],
        "interface": {"inputs": [], "outputs": [], "triggerTargets": ["c"]}}]
    assert run_needs.needs(graph, label="data") == {
        "mount": "data", "files": ["in.csv"]}


def test_a_graph_that_does_not_prepare_is_read_node_by_node():
    graph = {"nodes": [_node("csv", "CSVReader", path="x.csv")],
             "edges": [{"id": "bad", "source": "csv", "target": "ghost",
                        "sourceHandle": "tensor", "targetHandle": "x"}]}
    assert run_needs.needs(graph, label="data") == {
        "mount": "data", "files": ["x.csv"]}


def test_the_label_defaults_to_the_control_planes(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "MOUNT_LABEL", "lab-nas")
    graph = _pure_tensor_graph()
    _triggered(graph, _node("c", "CSVReader", path="u.csv"))
    assert run_needs.needs(graph)["mount"] == "lab-nas"
