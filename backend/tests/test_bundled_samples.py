"""``data/samples/...`` names a sample that ships with the install, in every
mode and for every reader node (round 6, H0).

3.4.0 shipped with seven digits examples that failed in the default project
mode: ``CSVReader`` exempted the one path ``data/samples/iris.csv`` from the
project rule, so ``data/samples/digits.csv`` resolved to
``<project>/data/samples/digits.csv``, which does not exist. The examples
were tested in flat mode and the project-mode acceptance ran only examples
that read no file.

The first test is the one that stops that happening again: it walks every
example graph that ships (the core ones and the bundled plugins'), takes
every parameter value that starts with ``data/samples/``, and resolves it
through its node's own resolver with and without a project open, from a
working directory that is not ``backend/``. It lists no file names: a sample
added next time is covered the moment an example uses it.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.config import settings
from app.core import projects
from app.core.data_paths import (
    bundled_sample_candidates,
    bundled_sample_names,
    bundled_sample_relative,
    not_found_message,
    resolve_bundled_sample,
)
from app.core.graph_engine import execute_graph
from app.core.project import scaffold_project
from app.nodes.data.csv_reader_node import CSVReaderNode
from app.nodes.llm.document_loader_node import _resolve_document_path
from app.nodes.llm.text_corpus_dataset_node import _resolve_corpus_path
from tests._example_graphs import graph_nodes

_BACKEND_DIR = Path(__file__).resolve().parents[1]
_REPO = _BACKEND_DIR.parent

#: Each reader node's own resolver, keyed by node type. A node type that
#: appears in an example with a ``data/samples/`` value and is missing here
#: fails ``test_every_node_reading_a_sample_has_a_resolver_here`` -- add it.
RESOLVERS = {
    "CSVReader": lambda param, value: CSVReaderNode._resolve_path(value),
    "DocumentLoader": lambda param, value: _resolve_document_path(value, kind=param),
    "TextCorpusDataset": lambda param, value: _resolve_corpus_path(value),
}


def _example_graphs() -> list[Path]:
    return sorted([*_REPO.glob("examples/**/graph.json"),
                   *_REPO.glob("plugins/*/examples/**/graph.json")])


def _sample_params() -> list[tuple[str, str, str, str]]:
    """(graph, node type, param, value) for every ``data/samples/`` value."""
    found = []
    for path in _example_graphs():
        graph = json.loads(path.read_text(encoding="utf-8"))
        for node in graph_nodes(graph):
            node_type = node.get("type") or node.get("data", {}).get("type")
            for param, value in (node.get("data", {}).get("params") or {}).items():
                if isinstance(value, str) and value.replace("\\", "/").startswith("data/samples/"):
                    found.append((path.relative_to(_REPO).as_posix(), node_type, param, value))
    return found


SAMPLE_PARAMS = _sample_params()


def test_the_examples_do_read_bundled_samples():
    """Guards the parametrisation below against silently collecting nothing."""
    assert len(SAMPLE_PARAMS) >= 10
    assert {node_type for _, node_type, _, _ in SAMPLE_PARAMS} >= {"CSVReader"}


def test_every_node_reading_a_sample_has_a_resolver_here():
    missing = {node_type for _, node_type, _, _ in SAMPLE_PARAMS} - RESOLVERS.keys()
    assert not missing, f"add these node types to RESOLVERS: {sorted(missing)}"


@pytest.fixture
def elsewhere(tmp_path, monkeypatch):
    """A working directory that is not ``backend/``: the server's own cwd in
    project mode must not be what makes a sample resolve."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return cwd


@pytest.fixture
def open_project(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    scaffold_project(root, "Samples test")
    projects.activate(root)
    assert settings.PROJECT_DIR is not None
    return settings.PROJECT_DIR


@pytest.mark.parametrize("mode", ["flat", "project"])
@pytest.mark.parametrize(
    "graph,node_type,param,value", SAMPLE_PARAMS,
    ids=[f"{g}:{t}.{p}" for g, t, p, _ in SAMPLE_PARAMS])
def test_every_sample_an_example_names_resolves(mode, graph, node_type, param, value,
                                                 elsewhere, tmp_path, monkeypatch):
    if mode == "project":
        root = tmp_path / "proj"
        root.mkdir()
        scaffold_project(root, "Samples test")
        projects.activate(root)
    else:
        projects.activate(None)
        monkeypatch.setattr(settings, "PROJECT_DIR", None)
    resolved = RESOLVERS[node_type](param, value)
    assert resolved.exists(), f"{graph}: {node_type}.{param}={value!r} -> {resolved}"
    assert resolved == _BACKEND_DIR / value.replace("\\", "/")


# ── the shared rule ─────────────────────────────────────────────────────

@pytest.mark.parametrize("value,expected", [
    ("data/samples/digits.csv", "data/samples/digits.csv"),
    ("data\\samples\\rag", "data/samples/rag"),
    ("  data/samples/iris.csv ", "data/samples/iris.csv"),
    ("data/samples/../../secret.csv", None),
    ("data/samplesX/a.csv", None),
    ("digits.csv", None),
    ("/data/samples/iris.csv", None),
    ("samples/iris.csv", None),
])
def test_what_counts_as_a_bundled_sample(value, expected):
    assert bundled_sample_relative(value) == expected


def test_a_file_in_the_project_wins_over_the_bundled_one(open_project, elsewhere):
    own = open_project / "data" / "samples" / "digits.csv"
    own.parent.mkdir(parents=True)
    own.write_bytes(b"a,b\n1,2\n")
    assert resolve_bundled_sample("data/samples/digits.csv") == own
    out = CSVReaderNode().execute({}, {"path": "data/samples/digits.csv"})
    assert out["tensor"].shape == (1, 2)
    # The others still come from the install.
    assert resolve_bundled_sample("data/samples/iris.csv") == _BACKEND_DIR / "data/samples/iris.csv"


def test_the_project_is_looked_in_first(open_project, elsewhere):
    candidates = bundled_sample_candidates("data/samples/digits.csv")
    assert candidates[0] == open_project / "data/samples/digits.csv"
    assert candidates[-1] == _BACKEND_DIR / "data/samples/digits.csv"


def test_a_sample_still_cannot_climb_out_of_the_project(open_project, elsewhere):
    with pytest.raises(ValueError, match="escapes the project"):
        CSVReaderNode._resolve_path("data/samples/../../../outside.csv")


# ── the "not found" error ───────────────────────────────────────────────

def test_a_missing_sample_names_where_it_looked_and_what_exists(open_project, elsewhere):
    with pytest.raises(FileNotFoundError) as caught:
        CSVReaderNode().execute({}, {"path": "data/samples/digit.csv"})
    message = str(caught.value)
    assert "file not found" in message
    for candidate in bundled_sample_candidates("data/samples/digit.csv"):
        assert str(candidate) in message
    for name in bundled_sample_names():
        assert name in message
    assert "data/samples/digits.csv" in message


def test_a_guessed_bare_sample_name_lists_the_samples(open_project, elsewhere):
    message = not_found_message("CSVReader", "digits.csv", open_project / "digits.csv")
    assert str(open_project / "digits.csv") in message
    assert "Bundled samples: " in message
    assert "data/samples/digits.csv" in message


def test_an_unrelated_missing_file_does_not_list_the_samples(open_project, elsewhere):
    message = not_found_message("CSVReader", "mine/run.csv", open_project / "mine/run.csv")
    assert str(open_project / "mine/run.csv") in message
    assert "Bundled samples" not in message


def test_the_sample_list_is_read_from_the_install():
    names = bundled_sample_names()
    root = _BACKEND_DIR / "data" / "samples"
    for entry in root.iterdir():
        if not entry.name.startswith("."):
            assert f"data/samples/{entry.name}" in names
    assert all((_BACKEND_DIR / name).exists() for name in names)


def test_the_document_and_corpus_errors_carry_the_same_hint(open_project, elsewhere):
    from app.nodes.llm.document_loader_node import DocumentLoaderNode
    from app.nodes.llm.text_corpus_dataset_node import TextCorpusDatasetNode

    with pytest.raises(FileNotFoundError, match="Bundled samples: "):
        DocumentLoaderNode().execute({}, {"source": "directory", "directory": "data/samples/ragg"})
    with pytest.raises(FileNotFoundError, match="Bundled samples: "):
        TextCorpusDatasetNode().execute(
            {}, {"source": "local_file", "local_path": "data/samples/mix/fact.txt"})


# ── a digits example, run in project mode ───────────────────────────────

def _smallest_digits_example() -> Path:
    """The digits example with the fewest nodes (data-driven, not named)."""
    users = {graph for graph, node_type, _, value in SAMPLE_PARAMS
             if node_type == "CSVReader" and value.endswith("/digits.csv")
             and graph.startswith("examples/")}
    assert users, "no example reads the bundled digits any more"
    return min((_REPO / g for g in users),
               key=lambda p: len(json.loads(p.read_text(encoding="utf-8"))["nodes"]))


def test_a_digits_example_runs_in_project_mode(open_project, elsewhere):
    from app.core.execution_context import ExecutionContext

    graph = json.loads(_smallest_digits_example().read_text(encoding="utf-8"))
    results = asyncio.run(execute_graph(
        graph["nodes"], graph["edges"], error_mode="fail_fast",
        subgraphs=graph.get("subgraphs"),
        context=ExecutionContext(device="cpu", seed=graph.get("settings", {}).get("seed", 0))))
    csv_nodes = [n["id"] for n in graph_nodes(graph) if n.get("type") == "CSVReader"]
    assert csv_nodes
    for node_id in csv_nodes:
        assert results[node_id]["tensor"].shape[0] == 1797
