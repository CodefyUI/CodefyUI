"""Regression tests for #144 - content-aware cache keys.

#116 (PR #142) made every file-reading node ``cacheable = False`` outright:
correct, but it means a Dataset/CSV/model-weights root re-reads from disk on
EVERY run, even when nothing changed underneath the path. The cache key
built from ``params`` alone describes WHERE to read, never WHAT is there.

The fix: an optional ``BaseNode.cache_fingerprint(params)`` hook that a node
overrides to fold cheap external-state metadata (a file's size + mtime, or
a directory's aggregate) into ``ExecutionCache.compute_key``, so a change on
disk changes the key even though the ``path`` param did not.

This file covers, in order:
  1. the ``core.cache_fingerprint`` helpers (pure functions over the
     filesystem);
  2. ``ExecutionCache.compute_key``'s new ``fingerprint`` parameter;
  3. the engine wiring end to end, via a synthetic node -- proven BEFORE
     touching any production node, so a mistake in a real node's path
     resolution cannot masquerade as a mechanism failure;
  4. the eight production reader nodes #144 restores to ``cacheable =
     True`` (CSVReader, FileReader, ImageReader, ImageBatchReader,
     ModelLoader, CheckpointLoader, Dataset, ImageFolderDataset), each
     both unit-tested at the fingerprint level and, for a representative
     few, proven end to end: run, change the input on disk, run again,
     assert the new value (unchanged-file case proves caching came back;
     changed-file case proves it never went stale).

#145 (``GraphInput(type=image)`` serving stale pixels on a canvas run) rides
the exact same mechanism proven here -- section 3's synthetic-node test is
the RED-then-GREEN proof that the engine wiring itself is correct, before
any production node (including ``GraphInputNode``) was touched. The
node-specific half -- ``GraphInputNode.cache_fingerprint``'s own RED-then-
GREEN tests, plus an end-to-end canvas-run repro and the API-path
no-regression check -- live in ``test_graph_input_node.py`` alongside that
node's other behavioural tests, kept there rather than here so the #144 and
#145 changes stay separable commits despite sharing this mechanism.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import settings
from app.core.cache import ExecutionCache
from app.core.cache_fingerprint import (
    directory_fingerprint,
    path_fingerprint,
    paths_fingerprint,
)
from app.core.graph_engine import execute_graph
from app.core.node_base import BaseNode, DataType, PortDefinition
from app.core.node_registry import registry
from app.nodes.data.csv_reader_node import CSVReaderNode
from app.nodes.data.dataset_node import DatasetNode
from app.nodes.data.huggingface_dataset_node import HuggingFaceDatasetNode
from app.nodes.data.image_folder_dataset_node import ImageFolderDatasetNode
from app.nodes.data.kaggle_dataset_node import KaggleDatasetNode
from app.nodes.io.checkpoint_node import CheckpointLoaderNode
from app.nodes.io.file_reader_node import FileReaderNode
from app.nodes.io.image_batch_reader_node import ImageBatchReaderNode
from app.nodes.io.image_reader_node import ImageReaderNode
from app.nodes.io.model_loader_node import ModelLoaderNode

# ─────────────────────────────────────────────────────────────────────────
# 1. core.cache_fingerprint helpers
# ─────────────────────────────────────────────────────────────────────────


def test_path_fingerprint_missing_file():
    assert path_fingerprint(Path("no_such_file_xyz.bin")) == {"exists": False}


def test_path_fingerprint_reflects_size_and_mtime(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello")
    fp1 = path_fingerprint(p)
    assert fp1["exists"] is True
    assert fp1["size"] == 5

    p.write_text("hello world")  # size AND mtime change
    fp2 = path_fingerprint(p)
    assert fp2["size"] == 11
    assert fp2 != fp1


def test_path_fingerprint_same_content_same_fingerprint(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello")
    assert path_fingerprint(p) == path_fingerprint(p)


def test_path_fingerprint_detects_same_size_same_mtime_content_change(tmp_path):
    """Size + mtime alone is not always fine enough: two writes that land
    inside the same filesystem timestamp tick (plausible on Windows, whose
    clock-interrupt granularity is coarser than a fast back-to-back
    rewrite) and happen to produce the same byte count are, to (size,
    mtime), indistinguishable -- yet the content differs. Forcing an
    identical mtime here makes that collision deterministic instead of
    relying on incidental timing, and proves small files also get hashed.
    """
    import os

    p = tmp_path / "a.txt"
    p.write_text("1")
    fixed_time = p.stat().st_mtime
    fp1 = path_fingerprint(p)

    p.write_text("2")  # same size (1 byte), different content
    os.utime(p, (fixed_time, fixed_time))  # force identical mtime too
    fp2 = path_fingerprint(p)

    assert fp1 != fp2, (
        "same size + same mtime still must not fingerprint identically "
        "once content differs"
    )


def test_paths_fingerprint_aggregates_a_list(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("x")
    b.write_text("yy")
    fp1 = paths_fingerprint([a, b])
    assert fp1["count"] == 2
    assert fp1["total_size"] == 3

    b.write_text("yyyy")
    fp2 = paths_fingerprint([a, b])
    assert fp2["total_size"] == 5
    assert fp2 != fp1


def test_paths_fingerprint_skips_missing_entries(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("x")
    fp = paths_fingerprint([a, tmp_path / "ghost.txt"])
    assert fp["count"] == 1


def test_directory_fingerprint_missing_directory():
    assert directory_fingerprint(Path("no_such_dir_xyz")) == {"exists": False}


def test_directory_fingerprint_reflects_nested_file_change(tmp_path):
    (tmp_path / "sub").mkdir()
    f = tmp_path / "sub" / "data.bin"
    f.write_bytes(b"1234")

    fp1 = directory_fingerprint(tmp_path)
    assert fp1["exists"] is True
    assert fp1["count"] == 1
    assert fp1["total_size"] == 4

    f.write_bytes(b"12345678")  # replaced with different content/size
    fp2 = directory_fingerprint(tmp_path)
    assert fp2["total_size"] == 8
    assert fp2 != fp1


def test_directory_fingerprint_unchanged_tree_is_stable(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x")
    assert directory_fingerprint(tmp_path) == directory_fingerprint(tmp_path)


# ─────────────────────────────────────────────────────────────────────────
# 2. ExecutionCache.compute_key's fingerprint parameter
# ─────────────────────────────────────────────────────────────────────────


def test_compute_key_fingerprint_changes_the_key():
    k1 = ExecutionCache.compute_key("FileReader", {"path": "a.txt"}, [], fingerprint={"size": 1})
    k2 = ExecutionCache.compute_key("FileReader", {"path": "a.txt"}, [], fingerprint={"size": 2})
    assert k1 != k2


def test_compute_key_fingerprint_defaults_to_none():
    """Omitting fingerprint must equal passing None explicitly -- nodes with
    no external state (the overwhelming majority) must not see their key
    schema change shape depending on caller diligence."""
    k1 = ExecutionCache.compute_key("Conv2d", {"in_channels": 3}, ["abc"])
    k2 = ExecutionCache.compute_key("Conv2d", {"in_channels": 3}, ["abc"], fingerprint=None)
    assert k1 == k2


# ─────────────────────────────────────────────────────────────────────────
# 3. Engine wiring, via a synthetic node (mechanism proven before any
#    production node is touched)
# ─────────────────────────────────────────────────────────────────────────


class _FingerprintedReaderNode(BaseNode):
    """Reads one integer from a file. Cacheable, with a fingerprint hook --
    the minimal shape of every real reader node converted by #144.
    """

    NODE_NAME = "_FingerprintedReader144"
    CATEGORY = "Test"
    DESCRIPTION = "Reads an int from a file path param"
    cacheable = True

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def cache_fingerprint(cls, params):
        from app.core.cache_fingerprint import path_fingerprint

        return path_fingerprint(params.get("path", ""))

    def execute(self, inputs, params):
        text = Path(params["path"]).read_text()
        return {"value": int(text)}


@pytest.fixture(autouse=True)
def _register_fingerprinted_reader():
    registry._nodes["_FingerprintedReader144"] = _FingerprintedReaderNode
    yield
    registry._nodes.pop("_FingerprintedReader144", None)


def _fp_graph(path: Path) -> tuple[list[dict], list[dict]]:
    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {"id": "reader", "type": "_FingerprintedReader144", "data": {"params": {"path": str(path)}}},
    ]
    edges = [
        {"id": "et", "source": "start", "target": "reader", "sourceHandle": "trigger", "type": "trigger"},
    ]
    return nodes, edges


async def _run(nodes, edges, cache) -> dict[str, tuple[str, dict | None]]:
    seen: dict[str, tuple[str, dict | None]] = {}

    async def track(node_id, status, data):
        if status in ("completed", "cached"):
            seen[node_id] = (status, data)

    await execute_graph(nodes, edges, on_progress=track, cache=cache)
    return seen


@pytest.mark.asyncio
async def test_engine_reuses_cache_when_fingerprint_is_unchanged(tmp_path):
    """The whole point of restoring cacheable=True: an untouched file must
    hit the cache on the second run, not re-execute."""
    p = tmp_path / "n.txt"
    p.write_text("1")
    cache = ExecutionCache()
    nodes, edges = _fp_graph(p)

    first = await _run(nodes, edges, cache)
    assert first["reader"][0] == "completed"

    second = await _run(nodes, edges, cache)
    assert second["reader"][0] == "cached", (
        "an unchanged file must be served from cache -- otherwise restoring "
        "cacheable=True bought nothing"
    )


@pytest.mark.asyncio
async def test_engine_busts_cache_when_fingerprint_changes(tmp_path):
    """The #144/#145 repro at the mechanism level: run, change the file on
    disk (same path param), run again -- must see the new value, not the
    first run's cached one.
    """
    p = tmp_path / "n.txt"
    p.write_text("1")
    cache = ExecutionCache()
    nodes, edges = _fp_graph(p)

    first = await _run(nodes, edges, cache)
    assert first["reader"][1]["value"] == 1

    p.write_text("2")  # path param is identical; content is not
    second = await _run(nodes, edges, cache)

    assert second["reader"][0] == "completed", (
        "a changed file must bust the cache -- it was served from cache "
        f"instead (status={second['reader'][0]!r})"
    )
    assert second["reader"][1]["value"] == 2


# ─────────────────────────────────────────────────────────────────────────
# 4. The eight production reader nodes #144 restores to cacheable=True
# ─────────────────────────────────────────────────────────────────────────

RESTORED_READER_NODES = [
    CSVReaderNode,
    FileReaderNode,
    ImageReaderNode,
    ImageBatchReaderNode,
    ModelLoaderNode,
    CheckpointLoaderNode,
    DatasetNode,
    ImageFolderDatasetNode,
]

# Hit the network (and, for Kaggle, environment credentials) in addition to
# reading local files. A content fingerprint of the local cache cannot
# describe "the remote revision changed" or "the credentials changed", so
# these stay outside the #144 restoration -- see the node-level comments.
STILL_NETWORK_BOUND_NODES = [KaggleDatasetNode, HuggingFaceDatasetNode]


def _reader_node_name(node_cls: type[BaseNode]) -> str:
    return node_cls.NODE_NAME


@pytest.mark.parametrize("node_cls", RESTORED_READER_NODES, ids=_reader_node_name)
def test_restored_reader_node_is_cacheable_again(node_cls: type[BaseNode]) -> None:
    assert node_cls.cacheable is True, (
        f"{node_cls.NODE_NAME} should be cacheable again now that its cache "
        "key folds in a content fingerprint (#144)"
    )
    assert registry.get(node_cls.NODE_NAME) is node_cls, (
        f"the registry serves a different {node_cls.NODE_NAME} class than "
        "the one asserted above, so the engine may not see the fix"
    )


@pytest.mark.parametrize("node_cls", STILL_NETWORK_BOUND_NODES, ids=_reader_node_name)
def test_network_bound_nodes_remain_non_cacheable(node_cls: type[BaseNode]) -> None:
    assert node_cls.cacheable is False, (
        f"{node_cls.NODE_NAME} hits the network and credentials, which a "
        "file-content fingerprint cannot describe, so it must stay outside "
        "the #144 restoration"
    )


# ── per-node fingerprint unit tests ─────────────────────────────────────


def test_file_reader_fingerprint_changes_on_edit(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("one")
    fp1 = FileReaderNode.cache_fingerprint({"path": str(p)})
    p.write_text("a much longer replacement body")
    fp2 = FileReaderNode.cache_fingerprint({"path": str(p)})
    assert fp1 is not None
    assert fp1 != fp2


def test_model_loader_fingerprint_changes_on_resave(tmp_path):
    import torch

    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    p = settings.MODELS_DIR / "_fp_test_144.pt"
    try:
        torch.save({"a": 1}, p)
        fp1 = ModelLoaderNode.cache_fingerprint({"path": "_fp_test_144.pt"})
        torch.save({"a": 1, "b": 2}, p)
        fp2 = ModelLoaderNode.cache_fingerprint({"path": "_fp_test_144.pt"})
        assert fp1 is not None
        assert fp1 != fp2
    finally:
        p.unlink(missing_ok=True)


def test_checkpoint_loader_fingerprint_changes_on_resave(tmp_path):
    import torch

    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    p = settings.MODELS_DIR / "_fp_ckpt_144.pt"
    try:
        torch.save({"epoch": 1}, p)
        fp1 = CheckpointLoaderNode.cache_fingerprint({"path": "_fp_ckpt_144.pt"})
        torch.save({"epoch": 2}, p)
        fp2 = CheckpointLoaderNode.cache_fingerprint({"path": "_fp_ckpt_144.pt"})
        assert fp1 is not None
        assert fp1 != fp2
    finally:
        p.unlink(missing_ok=True)


def test_image_batch_reader_fingerprint_changes_when_a_matched_file_changes(tmp_path):
    (tmp_path / "a.png").write_bytes(b"aaa")
    (tmp_path / "b.png").write_bytes(b"bb")
    params = {"directory": str(tmp_path), "pattern": "*.png", "max_images": 0}

    fp1 = ImageBatchReaderNode.cache_fingerprint(params)
    (tmp_path / "b.png").write_bytes(b"bbbbbb")
    fp2 = ImageBatchReaderNode.cache_fingerprint(params)

    assert fp1 is not None
    assert fp1 != fp2


def test_dataset_fingerprint_changes_when_a_file_under_data_dir_changes(tmp_path):
    (tmp_path / "raw").mkdir()
    f = tmp_path / "raw" / "batch1"
    f.write_bytes(b"x" * 10)
    params = {"name": "MNIST", "split": "train", "data_dir": str(tmp_path)}

    fp1 = DatasetNode.cache_fingerprint(params)
    f.write_bytes(b"x" * 20)
    fp2 = DatasetNode.cache_fingerprint(params)

    assert fp1 is not None
    assert fp1 != fp2


def test_image_folder_dataset_fingerprint_changes_when_a_class_image_changes(tmp_path):
    (tmp_path / "train" / "cat").mkdir(parents=True)
    f = tmp_path / "train" / "cat" / "1.png"
    f.write_bytes(b"x" * 10)
    params = {"path": str(tmp_path), "split": "train"}

    fp1 = ImageFolderDatasetNode.cache_fingerprint(params)
    f.write_bytes(b"x" * 30)
    fp2 = ImageFolderDatasetNode.cache_fingerprint(params)

    assert fp1 is not None
    assert fp1 != fp2


# ── full end-to-end: run, change the input on disk, run again ──────────
#
# Two representative nodes (both take zero graph inputs, so the harness
# stays minimal) prove the mechanism holds all the way through the real
# engine, not just at the fingerprint-hook level.


def _write_csv(path: Path, rows: list[tuple[float, float]]) -> None:
    lines = ["a,b"] + [f"{a},{b}" for a, b in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.mark.asyncio
async def test_csv_reader_is_cached_when_unchanged_and_fresh_when_edited(tmp_path):
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path, [(1.0, 2.0)])
    cache = ExecutionCache()
    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {
            "id": "csv", "type": "CSVReader",
            "data": {"params": {
                "path": str(csv_path), "target_column": "",
                "include_columns": "", "skip_header": True,
            }},
        },
    ]
    edges = [
        {"id": "et", "source": "start", "target": "csv", "sourceHandle": "trigger", "type": "trigger"},
    ]

    first = await _run(nodes, edges, cache)
    assert first["csv"][0] == "completed"
    assert first["csv"][1]["tensor"].tolist() == [[1.0, 2.0]]

    # Unchanged: restoring cacheable=True must actually restore caching.
    second = await _run(nodes, edges, cache)
    assert second["csv"][0] == "cached", (
        "an untouched CSV must hit the cache -- otherwise #144 bought "
        "nothing over #116's blanket cacheable=False"
    )

    # Changed: the #144 repro -- edit on disk, rerun, must see new data.
    _write_csv(csv_path, [(1.0, 2.0), (3.0, 4.0)])
    third = await _run(nodes, edges, cache)
    assert third["csv"][0] == "completed", (
        "an edited CSV must bust the cache -- it was served from cache "
        f"instead (status={third['csv'][0]!r})"
    )
    assert third["csv"][1]["tensor"].tolist() == [[1.0, 2.0], [3.0, 4.0]]


@pytest.mark.asyncio
async def test_image_reader_is_cached_when_unchanged_and_fresh_when_replaced(tmp_path):
    from PIL import Image

    img_path = tmp_path / "pic.png"
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(img_path)  # red
    cache = ExecutionCache()
    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {
            "id": "reader", "type": "ImageReader",
            "data": {"params": {"path": str(img_path), "mode": "RGB", "resize": 0}},
        },
    ]
    edges = [
        {"id": "et", "source": "start", "target": "reader", "sourceHandle": "trigger", "type": "trigger"},
    ]

    first = await _run(nodes, edges, cache)
    assert first["reader"][0] == "completed"
    assert first["reader"][1]["image"][0, 0, 0].item() == pytest.approx(1.0)  # red channel

    second = await _run(nodes, edges, cache)
    assert second["reader"][0] == "cached", (
        "an untouched image must hit the cache -- otherwise #144 bought "
        "nothing over #116's blanket cacheable=False"
    )

    Image.new("RGB", (4, 4), color=(0, 255, 0)).save(img_path)  # replaced: green
    third = await _run(nodes, edges, cache)
    assert third["reader"][0] == "completed", (
        "a replaced image must bust the cache -- it was served from cache "
        f"instead (status={third['reader'][0]!r})"
    )
    assert third["reader"][1]["image"][1, 0, 0].item() == pytest.approx(1.0)  # green channel
