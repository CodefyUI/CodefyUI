"""Regression tests for #143 - writer nodes must not skip their write on a
cache hit.

A cache hit returns the RECORDED outputs dict without calling execute()
again. That is correct for a pure node (same inputs -> same outputs, so
skipping the recompute is invisible) and wrong for a node whose entire
purpose is a side effect: ImageWriter / ModelSaver / CheckpointSaver return
a path string (and a pass-through value), but the thing that actually
matters -- the file on disk -- is never touched on a hit. Delete the file
after the first run and a cache hit on the second run hands back the same
path string while the file stays gone.

Found during the #116 review: #116 scoped to nodes that READ external
state; these writer nodes were left cacheable on the (probabilistic, not
structural) assumption their upstream is typically a stateful/non-cacheable
model. A writer fed only by pure nodes stays fully cacheable and skips its
write on a cache hit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import settings
from app.core.cache import ExecutionCache
from app.core.graph_engine import execute_graph
from app.core.node_base import BaseNode, DataType, PortDefinition
from app.core.node_registry import registry
from app.nodes.io.checkpoint_node import CheckpointSaverNode
from app.nodes.io.image_writer_node import ImageWriterNode
from app.nodes.io.model_saver_node import ModelSaverNode

# The three side-effecting writers named in #143.
WRITER_NODES = [ImageWriterNode, ModelSaverNode, CheckpointSaverNode]


def _node_name(node_cls: type[BaseNode]) -> str:
    return node_cls.NODE_NAME


class _PureImageSourceNode(BaseNode):
    """Deterministic image-shaped tensor, standing in for any pure upstream
    that could feed ImageWriter. What matters for #143 is that the upstream
    is cacheable (so the whole graph is), not which node produces it --
    using a real production node here would tie the test to that node's own
    port-type rules rather than to the cache bug.
    """

    NODE_NAME = "_PureImageSource143"
    CATEGORY = "Test"
    DESCRIPTION = "Deterministic zero image tensor"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="image", data_type=DataType.IMAGE)]

    def execute(self, inputs, params):
        import torch

        return {"image": torch.zeros(3, 4, 4)}


@pytest.fixture(autouse=True)
def _register_pure_image_source():
    registry._nodes["_PureImageSource143"] = _PureImageSourceNode
    yield
    registry._nodes.pop("_PureImageSource143", None)


@pytest.mark.parametrize("node_cls", WRITER_NODES, ids=_node_name)
def test_writer_node_is_not_cacheable(node_cls: type[BaseNode]) -> None:
    """The write IS the node's output; a cache hit must not skip it."""
    assert node_cls.cacheable is False, (
        f"{node_cls.NODE_NAME}'s purpose is its side-effect write. A cache "
        "hit that skips execute() also skips that write, so it must opt out "
        "with `cacheable = False`."
    )


@pytest.mark.asyncio
async def test_deleting_the_output_between_runs_recreates_it(tmp_path: Path) -> None:
    """The #143 repro: a pure upstream feeds ImageWriter so the whole graph
    is cacheable end to end. Run, delete the output file, run again against
    the same cache. The file must come back -- a cache hit that skips the
    write is exactly the bug.
    """
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_name = "test_143_output.png"
    target = settings.MODELS_DIR.parent / "output" / out_name

    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {"id": "src", "type": "_PureImageSource143", "data": {"params": {}}},
        {
            "id": "writer",
            "type": "ImageWriter",
            "data": {"params": {"path": out_name, "format": "PNG"}},
        },
    ]
    edges = [
        {"id": "et", "source": "start", "target": "src", "sourceHandle": "trigger", "type": "trigger"},
        {
            "id": "e1", "source": "src", "target": "writer",
            "sourceHandle": "image", "targetHandle": "image",
        },
    ]

    cache = ExecutionCache()
    try:
        statuses: dict[str, str] = {}

        async def track(node_id, status, data):
            statuses[node_id] = status

        await execute_graph(nodes, edges, on_progress=track, cache=cache)
        assert statuses["writer"] == "completed"
        assert target.exists(), "first run must create the output file"

        target.unlink()
        assert not target.exists()

        await execute_graph(nodes, edges, on_progress=track, cache=cache)
        assert statuses["writer"] == "completed", (
            "ImageWriter was served from cache and its write was skipped "
            f"(status was {statuses['writer']!r})"
        )
        assert target.exists(), (
            "the output file the user deleted was not recreated on rerun"
        )
    finally:
        target.unlink(missing_ok=True)
