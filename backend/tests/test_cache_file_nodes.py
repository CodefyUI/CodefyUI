"""Regression tests for #116 - nodes reading external state are never cached.

``ExecutionCache.compute_key()`` hashes ``{node_type, params, upstream keys,
device}``; a file's *contents* never enter that key. A cacheable ``CSVReader``
therefore keeps handing back the tensor it read on the very first run, even
after the user edits the CSV on disk. Nodes whose output depends on state
outside the graph (disk, network, process environment) opt out with
``cacheable = False``, and ``graph_engine`` propagates that opt-out to every
node downstream of them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.cache import ExecutionCache
from app.core.graph_engine import execute_graph
from app.core.node_base import BaseNode
from app.core.node_registry import registry
from app.nodes.data.csv_reader_node import CSVReaderNode
from app.nodes.data.dataset_node import DatasetNode
from app.nodes.data.huggingface_dataset_node import HuggingFaceDatasetNode
from app.nodes.data.kaggle_dataset_node import KaggleDatasetNode
from app.nodes.data.synthetic_dataset_node import SyntheticDatasetNode
from app.nodes.io.checkpoint_node import CheckpointLoaderNode
from app.nodes.io.file_reader_node import FileReaderNode
from app.nodes.io.image_batch_reader_node import ImageBatchReaderNode
from app.nodes.io.image_reader_node import ImageReaderNode
from app.nodes.io.model_loader_node import ModelLoaderNode
from app.nodes.tensor_ops.add_node import AddNode
from app.nodes.tensor_ops.mean_node import MeanNode
from app.nodes.utility.flatten_node import FlattenNode

# Every built-in node whose execute() reads state the cache key cannot see.
# Audited across backend/app/nodes/ for #116; the trailing comment records
# what each one reaches for.
EXTERNAL_STATE_NODES = [
    CSVReaderNode,           # pandas.read_csv on a user-supplied path
    FileReaderNode,          # open() on a user-supplied path
    ImageReaderNode,         # PIL.Image.open on a user-supplied path
    ImageBatchReaderNode,    # globs a directory, opens every match
    DatasetNode,             # torchvision dataset root on disk (may download)
    HuggingFaceDatasetNode,  # HF Hub download + on-disk HF cache
    KaggleDatasetNode,       # kagglehub download + KAGGLE_* env credentials
    ModelLoaderNode,         # torch.load of a weights file
    CheckpointLoaderNode,    # torch.load of a checkpoint file
]

# Control group. These stay cacheable so the fix remains surgical (#116:
# "no other node's caching behavior changes"). SyntheticDataset is here on
# purpose: it sits in app/nodes/data/ beside the file readers but builds its
# samples from a seeded RNG, so caching it is correct.
PURE_NODES = [MeanNode, AddNode, FlattenNode, SyntheticDatasetNode]


def _node_name(node_cls: type[BaseNode]) -> str:
    return node_cls.NODE_NAME


@pytest.mark.parametrize("node_cls", EXTERNAL_STATE_NODES, ids=_node_name)
def test_external_state_node_is_not_cacheable(node_cls: type[BaseNode]) -> None:
    """Reading disk / network / env means the cache key cannot describe the
    output, so the node has to opt out.

    Checked through the registry as well: graph_engine reads ``cacheable`` off
    whatever ``registry.get(node_type)`` hands it, so the opt-out has to
    survive node discovery, not just the import at the top of this file.
    """
    assert node_cls.cacheable is False, (
        f"{node_cls.NODE_NAME} reads state outside the graph, so its cache key "
        "cannot describe its output. Set `cacheable = False` on the class."
    )
    assert registry.get(node_cls.NODE_NAME) is node_cls, (
        f"the registry serves a different {node_cls.NODE_NAME} class than the "
        "one asserted above, so the engine may not see the opt-out"
    )


@pytest.mark.parametrize("node_cls", PURE_NODES, ids=_node_name)
def test_pure_nodes_stay_cacheable(node_cls: type[BaseNode]) -> None:
    """Pure compute nodes keep the BaseNode default - the #116 fix must not
    widen into a blanket cache disable."""
    assert node_cls.cacheable is True


def _write_csv(path: Path, rows: list[tuple[float, float]]) -> None:
    lines = ["a,b"] + [f"{a},{b}" for a, b in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _graph(csv_path: Path) -> tuple[list[dict], list[dict]]:
    """Start -> CSVReader -> Mean(dim=0), plus an independent TensorCreate.

    ``witness`` is a pure, cacheable source on its own branch. It has to
    report "cached" on the second run, which proves the ExecutionCache is
    actually live in this harness - without it, the assertions below would
    also pass if caching silently never engaged.
    """
    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {
            "id": "csv",
            "type": "CSVReader",
            "data": {
                "params": {
                    "path": str(csv_path),
                    "target_column": "",
                    "include_columns": "",
                    "skip_header": True,
                }
            },
        },
        {"id": "mean", "type": "Mean", "data": {"params": {"dim": "0", "keepdim": False}}},
        {
            "id": "witness",
            "type": "TensorCreate",
            "data": {"params": {"shape": "2", "fill": "zeros"}},
        },
    ]
    edges = [
        {"id": "et1", "source": "start", "target": "csv", "sourceHandle": "trigger", "type": "trigger"},
        {"id": "et2", "source": "start", "target": "witness", "sourceHandle": "trigger", "type": "trigger"},
        {
            "id": "e1",
            "source": "csv",
            "target": "mean",
            "sourceHandle": "tensor",
            "targetHandle": "tensor",
        },
    ]
    return nodes, edges


async def _run(
    nodes: list[dict], edges: list[dict], cache: ExecutionCache
) -> dict[str, tuple[str, dict | None]]:
    """Execute once; return ``{node_id: (status, outputs)}``."""
    seen: dict[str, tuple[str, dict | None]] = {}

    async def track(node_id: str, status: str, data: dict | None) -> None:
        if status in ("completed", "cached"):
            seen[node_id] = (status, data)

    await execute_graph(nodes, edges, on_progress=track, cache=cache)
    return seen


@pytest.mark.asyncio
async def test_editing_csv_between_runs_yields_new_data(tmp_path: Path) -> None:
    """The #116 repro: run a graph reading a CSV, edit the CSV on disk, re-run
    against the same ExecutionCache. The second run must read the new file
    instead of replaying the tensor cached during the first run.
    """
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path, [(1.0, 2.0)])
    cache = ExecutionCache()
    nodes, edges = _graph(csv_path)

    first = await _run(nodes, edges, cache)
    assert first["csv"][0] == "completed"
    assert first["csv"][1]["tensor"].tolist() == [[1.0, 2.0]]

    _write_csv(csv_path, [(1.0, 2.0), (3.0, 4.0)])
    second = await _run(nodes, edges, cache)

    assert second["witness"][0] == "cached", (
        "the pure witness node must hit the cache, otherwise this test proves "
        "nothing about CSVReader escaping it"
    )
    assert second["csv"][0] == "completed", (
        "CSVReader must re-execute after the file on disk changed; it was "
        f"reported as {second['csv'][0]!r}"
    )
    assert second["csv"][1]["tensor"].tolist() == [[1.0, 2.0], [3.0, 4.0]]


@pytest.mark.asyncio
async def test_downstream_of_csv_reader_reruns_after_edit(tmp_path: Path) -> None:
    """Non-cacheability propagates: Mean sits downstream of CSVReader and must
    also re-execute, otherwise it serves a stale reduction of the old file.
    """
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path, [(1.0, 3.0)])
    cache = ExecutionCache()
    nodes, edges = _graph(csv_path)

    first = await _run(nodes, edges, cache)
    assert first["mean"][0] == "completed"
    assert first["mean"][1]["tensor"].tolist() == pytest.approx([1.0, 3.0])

    _write_csv(csv_path, [(1.0, 3.0), (3.0, 5.0)])
    second = await _run(nodes, edges, cache)

    assert second["mean"][0] == "completed", (
        "Mean must re-execute when its non-cacheable upstream re-executes; it "
        f"was reported as {second['mean'][0]!r}"
    )
    assert second["mean"][1]["tensor"].tolist() == pytest.approx([2.0, 4.0])


@pytest.mark.asyncio
async def test_csv_reader_is_not_cached_even_when_file_is_unchanged(
    tmp_path: Path,
) -> None:
    """The engine cannot tell an edited file from an untouched one, so the
    opt-out is unconditional: CSVReader re-reads on every run.
    """
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path, [(1.0, 2.0)])
    cache = ExecutionCache()
    nodes, edges = _graph(csv_path)

    await _run(nodes, edges, cache)
    second = await _run(nodes, edges, cache)

    assert second["csv"][0] == "completed"
    assert second["mean"][0] == "completed"
    assert second["witness"][0] == "cached"
