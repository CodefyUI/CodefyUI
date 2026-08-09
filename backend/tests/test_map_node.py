"""Tests for MapNode (preset-driven mapping over a list)."""

from __future__ import annotations

from typing import Any

import pytest

from app.core.execution_context import ExecutionContext, MetricSignal
from app.core.node_base import BaseNode, DataType, PortDefinition
from app.core.node_registry import registry as node_registry
from app.core.preset_registry import preset_registry
from app.nodes.data.transforms._base import SeededAugmentation, seeded_for_node
from app.nodes.data.transforms.random_horizontal_flip_node import (
    RandomHorizontalFlipNode,
)
from app.nodes.data.transforms.to_tensor_transform_node import (
    ToTensorTransformNode,
)
from app.nodes.dataflow.map_node import MapNode
from app.schemas.models import (
    ExposedPortSchema,
    InternalEdgeSchema,
    InternalNodeSchema,
    PresetDefinition,
)

PROBE_TYPE = "_MapContextProbe"
PRESET_NAME = "_MapContextProbePreset"


def test_node_metadata():
    assert MapNode.NODE_NAME == "Map"
    assert MapNode.CATEGORY == "Data Flow"


def test_non_list_input_raises():
    with pytest.raises(ValueError, match="list"):
        MapNode().execute({"items": "not a list"}, {"subgraph": "x"})


def test_empty_subgraph_name_raises():
    with pytest.raises(ValueError, match="subgraph parameter"):
        MapNode().execute({"items": [1, 2]}, {"subgraph": ""})


def test_unknown_subgraph_raises():
    with pytest.raises(ValueError, match="not found"):
        MapNode().execute({"items": [1, 2]}, {"subgraph": "definitely_not_a_real_preset"})


# ── the body runs WITH the run's context (#196) ──────────────────────────
#
# Before #196 the loop called ``instance.execute(inputs, params)`` directly
# -- the last node-execute call site in the repo that did not go through
# ``invoke_node`` -- so every node in a Map body saw ``context=None``. The
# visible cost was silent: ``seed_pipeline`` returns the raw pipeline when
# there is no context, so a seeded run's augmentation inside a Map was
# neither reproducible nor isolated, with no error and no warning.


def _augmenting_chain():
    chain = RandomHorizontalFlipNode().execute({}, {"p": 0.5})["transform"]
    return ToTensorTransformNode().execute(
        {"transform": chain}, {})["transform"]


class _ProbeNode(BaseNode):
    """Records what the Map body handed it, then passes its input through.

    ``seeded`` goes through the real ``seeded_for_node`` rather than
    inspecting the context, so the assertion is about the reported symptom
    (augmentation is unseeded) and not about the mechanism.
    """

    NODE_NAME = PROBE_TYPE
    CATEGORY = "Testing"

    #: One entry per execute, in call order. Reset by the fixture.
    seen: list[dict[str, Any]] = []

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY,
                               description="passthrough")]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY,
                               description="passthrough")]

    @classmethod
    def define_params(cls) -> list:
        return []

    def execute(self, inputs, params, progress_callback=None, *,
                context: Any = None) -> dict[str, Any]:
        seeded = seeded_for_node(_augmenting_chain(), context)
        if context is not None:
            context.log_metric("probe", 1.0, step=0)
        type(self).seen.append({
            "node_id": getattr(context, "current_node_id", None),
            "context": context,
            "seeded": isinstance(seeded, SeededAugmentation),
            "progress_callback": progress_callback,
        })
        return {"value": inputs.get("value")}


def _preset(node_ids: list[str]) -> PresetDefinition:
    """A chain of probes: ``node_ids[0] -> node_ids[1] -> ...``."""
    return PresetDefinition(
        preset_name=PRESET_NAME,
        category="Testing",
        description="",
        nodes=[InternalNodeSchema(id=n, type=PROBE_TYPE) for n in node_ids],
        edges=[
            InternalEdgeSchema(source=a, sourceHandle="value",
                               target=b, targetHandle="value")
            for a, b in zip(node_ids, node_ids[1:])
        ],
        exposed_inputs=[ExposedPortSchema(
            name="in", internal_node=node_ids[0], internal_port="value")],
        exposed_outputs=[ExposedPortSchema(
            name="out", internal_node=node_ids[-1], internal_port="value")],
        exposed_params=[],
    )


@pytest.fixture()
def probe_body():
    """Register the probe node type and a two-probe preset, then restore."""
    _ProbeNode.seen = []
    saved_nodes = dict(node_registry._nodes)
    saved_presets = dict(preset_registry._presets)
    node_registry._nodes[PROBE_TYPE] = _ProbeNode
    preset_registry._presets[PRESET_NAME] = _preset(["node_0", "node_1"])
    try:
        yield _ProbeNode
    finally:
        node_registry._nodes.clear()
        node_registry._nodes.update(saved_nodes)
        preset_registry._presets.clear()
        preset_registry._presets.update(saved_presets)
        _ProbeNode.seen = []


def _map(context, items=("a",)):
    return MapNode().execute({"items": list(items)},
                             {"subgraph": PRESET_NAME}, context=context)


def _context(node_id: str = "map1", seed: int | None = 7) -> ExecutionContext:
    context = ExecutionContext(seed=seed)
    context.current_node_id = node_id
    return context


def test_a_seeded_map_body_seeds_its_augmentation(probe_body):
    """The bug this item was filed for: silently unseeded augmentation."""
    result = _map(_context())
    assert result["results"] == ["a"]
    assert [call["seeded"] for call in probe_body.seen] == [True, True]


def test_an_unseeded_map_body_still_keeps_torchs_own_entropy(probe_body):
    """Passing the context must not start seeding a run that asked for none."""
    _map(_context(seed=None))
    assert [call["seeded"] for call in probe_body.seen] == [False, False]


def test_body_node_ids_are_qualified_with_the_maps_own(probe_body):
    """Preset internals are named ``node_0``, ``node_1``, ... so a bare inner
    id is unique only within one body. Three things key off this field --
    the transform seed label, ``StatefulModuleMixin``'s weight key and every
    signal the body emits -- and all three break on a collision."""
    _map(_context(node_id="map1"))
    assert [call["node_id"] for call in probe_body.seen] == [
        "map1__node_0", "map1__node_1"]


def test_two_nodes_in_one_body_draw_different_seed_streams(probe_body):
    """Two same-shaped transform nodes must not derive the same seed: the
    label is ``transform:<current_node_id>``, so a shared id would give both
    the same stream and undo the isolation seeding exists for."""
    context = _context()
    _map(context)
    ids = [call["node_id"] for call in probe_body.seen]
    assert len(set(ids)) == 2
    assert len({context.derive_seed(f"transform:{i}") for i in ids}) == 2


def test_two_map_nodes_sharing_one_preset_do_not_collide(probe_body):
    """The same preset under two Map nodes: the ids must still differ, or the
    two bodies would share ``(graph_id, current_node_id, structure_hash)``
    and therefore share weights."""
    _map(_context(node_id="map1"))
    _map(_context(node_id="map2"))
    assert len({call["node_id"] for call in probe_body.seen}) == 4


def test_the_maps_own_current_node_id_survives_the_loop(probe_body):
    """A per-node COPY, not mutate-and-restore (the rule graph_engine follows
    since #253), so the Map's own signals stay attributed to the Map."""
    context = _context(node_id="map1")
    _map(context, items=["a", "b"])
    assert context.current_node_id == "map1"


def test_the_body_shares_the_runs_collaborators_not_copies(probe_body):
    """Only the scalar id differs per node. The stop event, the outbox and
    the node-state store must be the SAME objects, or cancellation and
    logging would be silently dead inside a Map."""
    context = _context()
    _map(context)
    inner = probe_body.seen[0]["context"]
    assert inner is not context
    assert inner.outbox is context.outbox
    assert inner._stop_event is context._stop_event
    assert inner.seed == context.seed
    assert inner.device == context.device


def test_body_metrics_are_attributed_to_the_inner_node(probe_body):
    context = _context(node_id="map1")
    _map(context)
    signals, _dropped = context.outbox.drain()
    metrics = [s for s in signals if isinstance(s, MetricSignal)]
    assert [m.node_id for m in metrics] == ["map1__node_0", "map1__node_1"]


def test_the_body_does_not_get_the_maps_progress_channel(probe_body):
    """The engine binds that callback to the MAP node's id, so an inner
    training loop's per-batch frames would arrive on the canvas as the Map's
    own progress and race the per-item frames Map emits itself."""
    frames: list[dict] = []
    MapNode().execute({"items": ["a"]}, {"subgraph": PRESET_NAME},
                      frames.append, context=_context())
    assert [call["progress_callback"] for call in probe_body.seen] == [None, None]
    assert frames, "the Map's own per-item frames must still be emitted"


def test_a_body_run_without_a_context_still_works(probe_body):
    """``execute_graph`` in a test, an exported script: no context at all."""
    result = _map(None)
    assert result["results"] == ["a"]
    assert [call["node_id"] for call in probe_body.seen] == [None, None]
