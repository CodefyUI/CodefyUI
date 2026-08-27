"""Tests for the /api/presets surface, focused on the SECRET-param
guarantees added in the secret-params work (C1 / I3):

- a SECRET param (an LLM API key) is never EXPOSED as a preset param, and
- its raw VALUE is scrubbed out of the stored preset definition file.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from app.core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)
from app.core.preset_registry import preset_registry


@pytest.fixture
def _isolated_presets(tmp_path, monkeypatch):
    """Write created presets into a throwaway dir and restore the global
    preset registry afterward (create_preset clears + rediscovers it)."""
    monkeypatch.setattr("app.config.settings.PRESETS_DIR", tmp_path)
    saved = dict(preset_registry._presets)
    try:
        yield tmp_path
    finally:
        preset_registry._presets.clear()
        preset_registry._presets.update(saved)


def _llm_nodes():
    """A single LLMChat node with both secret keys filled in. Its ports are
    all optional (inputs) / present (output), so the endpoint auto-detects
    exposed ports and does not 400 on 'no unconnected ports'."""
    return [
        {"id": "n1", "type": "LLMChat", "position": {"x": 0, "y": 0},
         "data": {"params": {
             "provider": "ChatGPT API",
             "model": "gpt-5.2",
             "openai_api_key": "sk-should-not-persist",
             "anthropic_api_key": "sk-ant-should-not-persist",
         }}},
    ]


@pytest.mark.asyncio
async def test_create_preset_does_not_expose_secret_params(
    test_client, _isolated_presets,
):
    """C1: creating a preset from an LLMChat subgraph exposes NO secret
    param, while still exposing the ordinary ones."""
    resp = await test_client.post("/api/presets/create", json={
        "name": "LLM Preset",
        "nodes": _llm_nodes(),
        "edges": [],
    })
    assert resp.status_code == 200, resp.text
    preset = resp.json()

    exposed = {p["param_name"] for p in preset["exposed_params"]}
    assert "openai_api_key" not in exposed
    assert "anthropic_api_key" not in exposed
    # Non-secret params are still exposed for configuration.
    assert "model" in exposed
    assert "provider" in exposed
    # No exposed param carries a SECRET param_def either.
    assert all(
        (p["param_def"] or {}).get("param_type") != "secret"
        for p in preset["exposed_params"]
    )


@pytest.mark.asyncio
async def test_create_preset_scrubs_secret_values_from_disk(
    test_client, _isolated_presets,
):
    """I3: the raw secret VALUE never reaches the stored preset file, and the
    inner node's non-secret params survive."""
    resp = await test_client.post("/api/presets/create", json={
        "name": "LLM Preset",
        "nodes": _llm_nodes(),
        "edges": [],
    })
    assert resp.status_code == 200, resp.text
    preset = resp.json()

    inner = preset["nodes"][0]["params"]
    assert inner.get("openai_api_key", "") == ""
    assert inner.get("anthropic_api_key", "") == ""
    assert inner["model"] == "gpt-5.2"

    # And nothing leaked into the on-disk JSON.
    written = (_isolated_presets / "llm_preset.json").read_text()
    assert "sk-should-not-persist" not in written
    assert "sk-ant-should-not-persist" not in written
    # Sanity: the file really is the preset we created.
    assert json.loads(written)["preset_name"] == "LLM Preset"


@pytest.mark.asyncio
async def test_exposed_param_keeps_visibility_and_tier(
    test_client, _isolated_presets,
):
    """core#134: an exposed param must behave like the inner one it stands for.

    ``_resolve_param_def`` copies field by field, and before #134 it dropped
    ``visible_when`` entirely -- so a conditional param became unconditional
    the moment it was exposed through a preset, and the editor offered
    Adam's betas on an SGD node. ``advanced`` would have been lost the same
    way.
    """
    resp = await test_client.post("/api/presets/create", json={
        "name": "Optimizer Preset",
        "nodes": [
            {"id": "opt", "type": "Optimizer", "data": {"params": {
                "type": "Adam", "lr": 0.01, "betas": "0.9, 0.999",
            }}},
        ],
        "edges": [],
    })
    assert resp.status_code == 200, resp.text

    by_name = {p["param_name"]: p for p in resp.json()["exposed_params"]}
    betas = by_name["betas"]["param_def"]
    assert betas["advanced"] is True
    assert betas["visible_when"] == {"type": ["Adam", "AdamW", "NAdam", "RAdam"]}

    lr = by_name["lr"]["param_def"]
    assert lr["advanced"] is False
    assert lr["visible_when"] is None


class _PackedNode(BaseNode):
    """A node with a SELECT whose options need different optional packs."""

    NODE_NAME = "_PackedPresetTest"
    CATEGORY = "Test"
    DESCRIPTION = "Has a pack-gated option"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="table",
                param_type=ParamType.SELECT,
                default="demo-16d",
                options=["demo-16d", "glove-50d"],
                option_packs={"glove-50d": "word-vectors"},
            ),
        ]

    def execute(self, inputs: dict[str, Any],
                params: dict[str, Any]) -> dict[str, Any]:
        return {"value": None}


@pytest.fixture
def _packed_node():
    from app.core.node_registry import registry

    registry._nodes[_PackedNode.NODE_NAME] = _PackedNode
    yield _PackedNode
    registry._nodes.pop(_PackedNode.NODE_NAME, None)


@pytest.mark.asyncio
async def test_exposed_param_keeps_its_option_packs(
    test_client, _isolated_presets, _packed_node,
):
    """Same failure mode as core#134, one field later.

    ``_resolve_param_def`` copies field by field, so a pack-gated SELECT
    exposed through a preset would offer options nothing on this machine can
    load -- and the editor, seeing no gating, would let the learner pick one.
    """
    resp = await test_client.post("/api/presets/create", json={
        "name": "Packed Preset",
        "nodes": [
            {"id": "p", "type": _PackedNode.NODE_NAME,
             "data": {"params": {"table": "demo-16d"}}},
        ],
        "edges": [],
    })
    assert resp.status_code == 200, resp.text

    by_name = {p["param_name"]: p for p in resp.json()["exposed_params"]}
    assert by_name["table"]["param_def"]["option_packs"] == {
        "glove-50d": "word-vectors"}


@pytest.mark.asyncio
async def test_create_refuses_a_canvas_containing_a_subgraph_instance(
    test_client, _isolated_presets,
):
    """core#137: a preset cannot carry a graph-local subgraph reference.

    The stored preset is nodes + edges and nothing else -- there is no slot
    for a definition -- so a preset built from a canvas holding an instance
    would ship a `subgraph:<id>` node whose definition can never accompany
    it. It does not even work in the source graph: expansion runs subgraphs
    before presets and never revisits, so the instance reaches the executor
    unexpanded.

    Refused, not stripped: stripping would silently hand back a preset
    missing an arbitrary piece of what the user asked to package.
    """
    resp = await test_client.post("/api/presets/create", json={
        "name": "Blocky",
        "nodes": [
            {"id": "opt", "type": "Optimizer",
             "data": {"params": {"type": "Adam", "lr": 0.01}}},
            {"id": "blk", "type": "subgraph:inner", "data": {"params": {}}},
        ],
        "edges": [],
    })
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert "blk" in detail
    assert "expand" in detail.lower()
    # Nothing was written.
    assert list(_isolated_presets.glob("*.json")) == []


@pytest.mark.asyncio
async def test_create_still_accepts_a_canvas_with_no_instances(
    test_client, _isolated_presets,
):
    """The guard must not fire on an ordinary graph -- including one whose
    node type merely CONTAINS the word."""
    resp = await test_client.post("/api/presets/create", json={
        "name": "Plain",
        "nodes": [
            {"id": "opt", "type": "Optimizer",
             "data": {"params": {"type": "Adam", "lr": 0.01}}},
        ],
        "edges": [],
    })
    assert resp.status_code == 200, resp.text


# -- dynamic port counts survive preset extraction (#196) -----------------
#
# Both loops used to read the STATIC define_inputs() / define_outputs(),
# which answer for the DEFAULT params. A ComposeTransform(steps=5) therefore
# reported two ports however many it had, and step_3..step_5 never reached
# `exposed_inputs` -- so edges into them had nowhere to reattach when the
# preset was dropped back onto a canvas.


@pytest.mark.asyncio
async def test_preset_exposes_every_port_a_dynamic_node_actually_has(
    test_client, _isolated_presets,
):
    resp = await test_client.post("/api/presets/create", json={
        "name": "Five Steps",
        "nodes": [
            {"id": "cmp", "type": "ComposeTransform",
             "data": {"params": {"steps": 5}}},
        ],
        "edges": [],
    })
    assert resp.status_code == 200, resp.text

    exposed = {p["internal_port"] for p in resp.json()["exposed_inputs"]}
    assert exposed == {f"step_{i}" for i in range(1, 6)}


@pytest.mark.asyncio
async def test_preset_exposes_dynamic_ports_in_both_directions(
    test_client, _isolated_presets,
):
    """PythonScript varies inputs AND outputs, so it catches a fix applied to
    only one of the two loops."""
    resp = await test_client.post("/api/presets/create", json={
        "name": "Wide Script",
        "nodes": [
            {"id": "py", "type": "PythonScript",
             "data": {"params": {"input_ports": 4, "output_ports": 3,
                                 "code": "return {}"}}},
        ],
        "edges": [],
    })
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert {p["internal_port"] for p in body["exposed_inputs"]} == {
        f"in{i}" for i in range(1, 5)}
    assert {p["internal_port"] for p in body["exposed_outputs"]} == {
        f"out{i}" for i in range(1, 4)}


@pytest.mark.asyncio
async def test_preset_still_exposes_the_default_ports_of_a_static_node(
    test_client, _isolated_presets,
):
    """The dynamic form delegates to the static one for everything else, so
    a node holding no port param must be completely unaffected."""
    resp = await test_client.post("/api/presets/create", json={
        "name": "Static",
        "nodes": [
            {"id": "cmp", "type": "ComposeTransform", "data": {"params": {}}},
        ],
        "edges": [],
    })
    assert resp.status_code == 200, resp.text
    assert {p["internal_port"] for p in resp.json()["exposed_inputs"]} == {
        "step_1", "step_2"}


def test_registry_types_a_port_that_only_exists_at_this_port_count():
    """``_resolve_port_type`` read the same static definition, so a stored
    preset naming step_5 resolved to "ANY" and the canvas drew a grey handle
    where the wire is a TRANSFORM."""
    from app.core.node_registry import registry as node_registry

    preset = preset_registry._load_and_resolve({
        "preset_name": "Five Steps",
        "nodes": [{"id": "cmp", "type": "ComposeTransform",
                   "params": {"steps": 5}}],
        "edges": [],
        # data_type omitted on purpose: that is what makes the registry
        # resolve it from the node class.
        "exposed_inputs": [{"name": "s5", "internal_node": "cmp",
                            "internal_port": "step_5"}],
        "exposed_outputs": [],
    }, node_registry)

    assert preset.exposed_inputs[0].data_type == "TRANSFORM"
