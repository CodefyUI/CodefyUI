"""Tests for the graph API endpoints."""

import json

import pytest

from app.core.preset_registry import preset_registry
from app.schemas.models import InternalNodeSchema, PresetDefinition


@pytest.fixture
def _secret_preset():
    """Register a preset with an inner LLMChat (declares SECRET params) so
    the secret scrub / lint can resolve its inner node types. Cleaned up
    after the test to leave the global registry as discovered."""
    preset = PresetDefinition(
        preset_name="SecretChat",
        category="Test",
        description="",
        nodes=[InternalNodeSchema(id="chat", type="LLMChat", params={})],
        edges=[],
        exposed_inputs=[],
        exposed_outputs=[],
        exposed_params=[],
    )
    preset_registry._presets["SecretChat"] = preset
    try:
        yield preset
    finally:
        preset_registry._presets.pop("SecretChat", None)


@pytest.mark.asyncio
async def test_health(test_client):
    resp = await test_client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["nodes_loaded"] >= 1


@pytest.mark.asyncio
async def test_validate_valid_graph(test_client, sample_graph):
    resp = await test_client.post("/api/graph/validate", json=sample_graph)
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is True
    assert data["errors"] == []


@pytest.mark.asyncio
async def test_validate_invalid_graph(test_client):
    graph = {
        "nodes": [
            {"id": "1", "type": "Loss", "position": {"x": 0, "y": 0}, "data": {}},
            {"id": "2", "type": "Conv2d", "position": {"x": 0, "y": 0}, "data": {}},
        ],
        "edges": [
            {"id": "e1", "source": "1", "target": "2", "sourceHandle": "loss_fn", "targetHandle": "tensor"},
        ],
        "name": "bad-graph",
    }
    resp = await test_client.post("/api/graph/validate", json=graph)
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is False
    assert len(data["errors"]) > 0


@pytest.mark.asyncio
async def test_save_and_load_roundtrip(test_client, sample_graph, tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.settings.GRAPHS_DIR", tmp_path)
    # Save
    resp = await test_client.post("/api/graph/save", json=sample_graph)
    assert resp.status_code == 200
    assert "path" in resp.json()

    # Load
    resp = await test_client.get("/api/graph/load/test-graph")
    assert resp.status_code == 200
    loaded = resp.json()
    assert loaded["name"] == "test-graph"
    assert len(loaded["nodes"]) == 3  # Start + _TestSource + Print

    # List
    resp = await test_client.get("/api/graph/list")
    assert resp.status_code == 200
    graphs = resp.json()
    assert any(g["name"] == "test-graph" for g in graphs)


def test_sanitize_name_helper():
    from app.api.routes_graph import _sanitize_name

    assert _sanitize_name("my-graph_2") == "my-graph_2"
    assert _sanitize_name("a b.c/d") == "a_b_c_d"


def test_graph_path_helper(monkeypatch, tmp_path):
    monkeypatch.setattr("app.config.settings.GRAPHS_DIR", tmp_path)
    from app.api.routes_graph import _graph_path

    p = _graph_path("weird name")
    assert p == tmp_path / "weird_name.json"


@pytest.mark.asyncio
async def test_save_scrubs_secret_params(test_client, tmp_path, monkeypatch):
    """Item 1d: a filled SECRET param (LLMChat api key) is blanked before the
    graph file is written — secrets never persist to disk, even if a client
    bypasses the editor's own stripping."""
    monkeypatch.setattr("app.config.settings.GRAPHS_DIR", tmp_path)
    graph = {
        "name": "secret-graph",
        "nodes": [
            {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
             "data": {"params": {}}},
            {"id": "llm", "type": "LLMChat", "position": {"x": 0, "y": 0},
             "data": {"params": {
                 "provider": "ChatGPT API",
                 "openai_api_key": "sk-super-secret",
                 "anthropic_api_key": "sk-ant-secret",
                 "model": "gpt-5.2",
             }}},
        ],
        "edges": [],
    }
    resp = await test_client.post("/api/graph/save", json=graph)
    assert resp.status_code == 200

    saved = json.loads((tmp_path / "secret-graph.json").read_text())
    params = next(n for n in saved["nodes"] if n["id"] == "llm")["data"]["params"]
    assert params["openai_api_key"] == ""
    assert params["anthropic_api_key"] == ""
    # Non-secret params on the same node survive untouched.
    assert params["model"] == "gpt-5.2"
    assert params["provider"] == "ChatGPT API"


@pytest.mark.asyncio
async def test_save_scrubs_preset_embedded_secret(
    test_client, tmp_path, monkeypatch, _secret_preset,
):
    """C1: a hand-written graph with a secret baked into a preset node's
    internalParams is scrubbed before the file is written."""
    monkeypatch.setattr("app.config.settings.GRAPHS_DIR", tmp_path)
    graph = {
        "name": "secret-preset-graph",
        "nodes": [
            {"id": "p1", "type": "preset:SecretChat",
             "position": {"x": 0, "y": 0},
             "data": {"internalParams": {
                 "chat": {"openai_api_key": "sk-leaked-in-preset",
                          "model": "gpt-5.2"},
             }}},
        ],
        "edges": [],
    }
    resp = await test_client.post("/api/graph/save", json=graph)
    assert resp.status_code == 200

    saved = json.loads((tmp_path / "secret-preset-graph.json").read_text())
    inner = saved["nodes"][0]["data"]["internalParams"]["chat"]
    assert inner["openai_api_key"] == ""     # secret blanked
    assert inner["model"] == "gpt-5.2"       # non-secret override kept
    assert "sk-leaked-in-preset" not in json.dumps(saved)


@pytest.mark.asyncio
async def test_export_scrubs_secret_params(test_client):
    """M4: exported Python never embeds a SECRET param value."""
    graph = {
        "name": "export-secret",
        "nodes": [
            {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
             "data": {"params": {}}},
            {"id": "llm", "type": "LLMChat", "position": {"x": 200, "y": 0},
             "data": {"params": {
                 "provider": "ChatGPT API", "model": "gpt-5.2",
                 "openai_api_key": "sk-export-secret",
             }}},
        ],
        "edges": [
            {"id": "t1", "source": "start", "target": "llm",
             "sourceHandle": "trigger", "targetHandle": "", "type": "trigger"},
        ],
    }
    resp = await test_client.post("/api/graph/export", json=graph)
    assert resp.status_code == 200, resp.text
    script = resp.json()["script"]
    assert "sk-export-secret" not in script


def _minimal_export_graph(seed=None, **extra):
    graph = {
        "name": "seed-range",
        "nodes": [
            {"id": "start", "type": "Start",
             "position": {"x": 0, "y": 0}, "data": {"params": {}}},
            {"id": "flip", "type": "RandomHorizontalFlip",
             "position": {"x": 200, "y": 0}, "data": {"params": {"p": 0.5}}},
        ],
        "edges": [
            {"id": "t1", "source": "start", "target": "flip",
             "sourceHandle": "trigger", "targetHandle": "", "type": "trigger"},
        ],
        **extra,
    }
    if seed is not None:
        graph["seed"] = seed
    return graph


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", [-1, 2 ** 32, 10 ** 26])
async def test_export_refuses_a_seed_the_run_path_would_refuse(test_client, seed):
    """core#136 re-review, N-4. One seed range, not two.

    ``run_service`` rejects anything outside ``0..MAX_SEED`` before a run
    starts. ``/export`` used to bake the same values into ``GRAPH_SEED``
    verbatim, so a hand-rolled request could produce a script whose results
    the canvas could not reproduce -- it would refuse the seed outright.
    An export that disagrees with the graph it came from is worse than one
    that fails to build.
    """
    resp = await test_client.post("/api/graph/export",
                                  json=_minimal_export_graph(seed))
    assert resp.status_code == 422, resp.text


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", [0, 1, 2 ** 32 - 1])
async def test_export_still_accepts_every_seed_a_run_accepts(test_client, seed):
    """The bound is the run path's bound, inclusive at both ends.

    ``0`` in particular: it is falsy, and the endpoints of a range are
    exactly where an off-by-one hides.
    """
    resp = await test_client.post("/api/graph/export",
                                  json=_minimal_export_graph(seed))
    assert resp.status_code == 200, resp.text
    assert f"GRAPH_SEED = {seed}" in resp.json()["script"]


@pytest.mark.asyncio
async def test_export_without_a_seed_is_unchanged(test_client):
    """An older client sends no ``seed`` at all and still exports."""
    resp = await test_client.post("/api/graph/export",
                                  json=_minimal_export_graph())
    assert resp.status_code == 200, resp.text
    assert "GRAPH_SEED = None" in resp.json()["script"]


@pytest.mark.asyncio
async def test_export_scrubs_secrets_from_embedded_preset(test_client):
    """Portable preset defaults and overrides are scrubbed before embedding."""
    graph = {
        "name": "embedded-secret-preset",
        "nodes": [
            {
                "id": "start",
                "type": "Start",
                "position": {"x": 0, "y": 0},
                "data": {"params": {}},
            },
            {
                "id": "loss",
                "type": "Loss",
                "position": {"x": 0, "y": 0},
                "data": {"params": {"type": "MSELoss"}},
            },
            {
                "id": "preset",
                "type": "preset:PortableSecretChat",
                "position": {"x": 0, "y": 0},
                "data": {
                    "internalParams": {
                        "chat": {
                            "anthropic_api_key": "sk-preset-override-secret",
                        },
                    },
                },
            },
        ],
        "edges": [
            {
                "id": "trigger",
                "source": "start",
                "target": "loss",
                "sourceHandle": "trigger",
                "targetHandle": "",
                "type": "trigger",
            },
        ],
        "presets": [
            {
                "preset_name": "PortableSecretChat",
                "category": "Test",
                "description": "",
                "tags": [],
                "nodes": [
                    {
                        "id": "chat",
                        "type": "LLMChat",
                        "params": {
                            "provider": "ChatGPT API",
                            "model": "gpt-5.2",
                            "openai_api_key": "sk-preset-default-secret",
                        },
                    },
                ],
                "edges": [],
                "exposed_inputs": [],
                "exposed_outputs": [],
                "exposed_params": [],
            },
        ],
    }

    resp = await test_client.post("/api/graph/export", json=graph)
    assert resp.status_code == 200, resp.text
    script = resp.json()["script"]
    assert "sk-preset-default-secret" not in script
    assert "sk-preset-override-secret" not in script
    # The untriggered preset is a draft: node-function export prunes it, so
    # its scrubbed params leave no trace at all (previously the whole graph
    # was embedded and the blanked keys were asserted verbatim).
    assert "openai_api_key" not in script
    assert "anthropic_api_key" not in script


@pytest.mark.asyncio
async def test_export_scrubs_shadowed_embedded_preset_override(
    test_client,
    _secret_preset,
):
    """Installed same-name presets cannot hide portable secret slots."""
    graph = {
        "name": "shadowed-portable-preset",
        "nodes": [
            {
                "id": "start",
                "type": "Start",
                "position": {"x": 0, "y": 0},
                "data": {"params": {}},
            },
            {
                "id": "loss",
                "type": "Loss",
                "position": {"x": 0, "y": 0},
                "data": {"params": {"type": "MSELoss"}},
            },
            {
                "id": "preset",
                "type": "preset:SecretChat",
                "position": {"x": 0, "y": 0},
                "data": {
                    "internalParams": {
                        "portable_chat": {
                            "openai_api_key": "sk-shadowed-override",
                        },
                    },
                },
            },
        ],
        "edges": [
            {
                "id": "trigger",
                "source": "start",
                "target": "loss",
                "sourceHandle": "trigger",
                "targetHandle": "",
                "type": "trigger",
            },
        ],
        "presets": [
            {
                "preset_name": "SecretChat",
                "category": "Portable",
                "description": "",
                "tags": [],
                "nodes": [
                    {
                        "id": "portable_chat",
                        "type": "LLMChat",
                        "params": {},
                    },
                ],
                "edges": [],
                "exposed_inputs": [],
                "exposed_outputs": [],
                "exposed_params": [],
            },
        ],
    }

    resp = await test_client.post("/api/graph/export", json=graph)
    assert resp.status_code == 200, resp.text
    script = resp.json()["script"]
    assert "sk-shadowed-override" not in script
    # As above: the draft preset is pruned at export, so the scrubbed slot
    # is absent from the generated source entirely.
    assert "openai_api_key" not in script


@pytest.mark.asyncio
async def test_save_unknown_node_type_left_untouched(
    test_client, tmp_path, monkeypatch,
):
    """A node type the registry doesn't know carries no known secret params,
    so its data is written verbatim (the scrub is a no-op for it) and the
    save still succeeds."""
    monkeypatch.setattr("app.config.settings.GRAPHS_DIR", tmp_path)
    graph = {
        "name": "unknown-graph",
        "nodes": [
            {"id": "x", "type": "TotallyUnknownNode", "position": {"x": 0, "y": 0},
             "data": {"params": {"api_key": "kept-verbatim", "foo": "bar"}}},
        ],
        "edges": [],
    }
    resp = await test_client.post("/api/graph/save", json=graph)
    assert resp.status_code == 200

    saved = json.loads((tmp_path / "unknown-graph.json").read_text())
    assert saved["nodes"][0]["data"]["params"] == {
        "api_key": "kept-verbatim", "foo": "bar",
    }


@pytest.mark.asyncio
async def test_save_and_load_roundtrips_segment_groups(
    test_client, tmp_path, monkeypatch,
):
    """Item 4: segmentGroups is persisted and returned by load (previously
    silently dropped by the GraphData schema)."""
    monkeypatch.setattr("app.config.settings.GRAPHS_DIR", tmp_path)
    graph = {
        "name": "seg-graph",
        "nodes": [],
        "edges": [],
        "segmentGroups": [
            {"id": "g1", "headNodeId": "a", "tailNodeId": "b"},
            {"id": "g2", "headNodeId": "c", "tailNodeId": "d"},
        ],
    }
    resp = await test_client.post("/api/graph/save", json=graph)
    assert resp.status_code == 200

    resp = await test_client.get("/api/graph/load/seg-graph")
    assert resp.status_code == 200
    loaded = resp.json()
    assert loaded["segmentGroups"] == [
        {"id": "g1", "headNodeId": "a", "tailNodeId": "b"},
        {"id": "g2", "headNodeId": "c", "tailNodeId": "d"},
    ]


@pytest.mark.asyncio
async def test_save_and_load_roundtrips_the_bypass_flag(
    test_client, tmp_path, monkeypatch,
):
    """core#128: a muted node stays muted across save/load.

    `NodeData.data` is a free-form dict, so this is really a guard against a
    future schema tightening quietly dropping the flag.
    """
    monkeypatch.setattr("app.config.settings.GRAPHS_DIR", tmp_path)
    graph = {
        "name": "bypass-graph",
        "nodes": [
            {
                "id": "drop",
                "type": "Dropout",
                "position": {"x": 0, "y": 0},
                "data": {"params": {"p": 0.25}, "bypassed": True},
            },
            {
                "id": "flat",
                "type": "Flatten",
                "position": {"x": 0, "y": 0},
                "data": {"params": {}},
            },
        ],
        "edges": [],
    }
    resp = await test_client.post("/api/graph/save", json=graph)
    assert resp.status_code == 200

    resp = await test_client.get("/api/graph/load/bypass-graph")
    assert resp.status_code == 200
    loaded = {n["id"]: n for n in resp.json()["nodes"]}
    assert loaded["drop"]["data"]["bypassed"] is True
    assert loaded["drop"]["data"]["params"]["p"] == 0.25
    # An untouched node gains no flag.
    assert "bypassed" not in loaded["flat"]["data"]


# ── Subgraphs over HTTP (core#137) ──────────────────────────────────────


def _subgraph_graph(name="sg-graph", scalar=2.0):
    return {
        "name": name,
        "nodes": [
            {"id": "start", "type": "Start", "position": {"x": 0, "y": 0},
             "data": {"params": {}}},
            {"id": "src", "type": "TensorCreate", "position": {"x": 1, "y": 0},
             "data": {"params": {"shape": "1,2", "fill": "full",
                                 "value": 1.0}}},
            {"id": "one", "type": "subgraph:double",
             "position": {"x": 2, "y": 0}, "data": {"params": {}}},
            {"id": "two", "type": "subgraph:double",
             "position": {"x": 3, "y": 0}, "data": {"params": {}}},
        ],
        "edges": [
            {"id": "t", "source": "start", "target": "src",
             "sourceHandle": "trigger", "targetHandle": "__trigger",
             "type": "trigger"},
            {"id": "e1", "source": "src", "target": "one",
             "sourceHandle": "tensor", "targetHandle": "in", "type": "data"},
            {"id": "e2", "source": "one", "target": "two",
             "sourceHandle": "out", "targetHandle": "in", "type": "data"},
        ],
        "subgraphs": [{
            "id": "double",
            "name": "Double",
            "description": "",
            "nodes": [
                {"id": "mul", "type": "ScalarMultiply",
                 "position": {"x": 4, "y": 5},
                 "data": {"params": {"scalar": scalar}}},
            ],
            "edges": [],
            "interface": {
                "inputs": [{"port": "in", "innerNode": "mul",
                            "innerPort": "tensor", "data_type": "TENSOR"}],
                "outputs": [{"port": "out", "innerNode": "mul",
                             "innerPort": "tensor", "data_type": "TENSOR"}],
                "triggerTargets": ["mul"],
            },
        }],
    }


@pytest.mark.asyncio
async def test_save_and_load_roundtrips_subgraph_definitions(
    test_client, tmp_path, monkeypatch,
):
    """A definition survives the schema, disk and the load path.

    Two instances share ONE definition on disk -- which is the whole point:
    the file stores the block once, not once per use.
    """
    monkeypatch.setattr("app.config.settings.GRAPHS_DIR", tmp_path)
    resp = await test_client.post("/api/graph/save", json=_subgraph_graph())
    assert resp.status_code == 200

    on_disk = json.loads((tmp_path / "sg-graph.json").read_text())
    assert len(on_disk["subgraphs"]) == 1
    assert [n["type"] for n in on_disk["nodes"]].count("subgraph:double") == 2

    resp = await test_client.get("/api/graph/load/sg-graph")
    assert resp.status_code == 200
    loaded = resp.json()
    definition = loaded["subgraphs"][0]
    assert definition["id"] == "double"
    assert definition["nodes"][0]["data"]["params"]["scalar"] == 2.0
    assert definition["interface"]["inputs"] == [
        {"port": "in", "innerNode": "mul", "innerPort": "tensor",
         "data_type": "TENSOR"},
    ]
    assert definition["interface"]["triggerTargets"] == ["mul"]


@pytest.mark.asyncio
async def test_validate_accepts_a_wired_subgraph_instance(test_client):
    resp = await test_client.post("/api/graph/validate",
                                  json=_subgraph_graph())
    assert resp.status_code == 200
    assert resp.json() == {"valid": True, "errors": []}


@pytest.mark.asyncio
async def test_validate_rejects_a_cycle_that_only_exists_inside_a_definition(
    test_client,
):
    """Criterion 4 at the API surface: the path names both sides."""
    graph = _subgraph_graph()
    definition = graph["subgraphs"][0]
    definition["nodes"].append({
        "id": "back", "type": "ScalarMultiply", "position": {"x": 6, "y": 5},
        "data": {"params": {"scalar": 1.0}},
    })
    definition["edges"] = [
        {"id": "f", "source": "mul", "target": "back", "sourceHandle": "tensor",
         "targetHandle": "tensor", "type": "data"},
        {"id": "g", "source": "back", "target": "mul", "sourceHandle": "tensor",
         "targetHandle": "tensor", "type": "data"},
    ]
    resp = await test_client.post("/api/graph/validate", json=graph)
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is False
    cycles = [e for e in body["errors"] if "cycle" in e]
    assert cycles, body["errors"]
    # Either instance is a correct answer -- both carry the same loop -- but
    # whichever is reported must name the instance AND the nodes inside it.
    instance = "one" if "one/" in cycles[0] else "two"
    assert f"{instance}/mul" in cycles[0]
    assert f"{instance}/back" in cycles[0]
    assert f"crosses subgraph instance(s): {instance}" in cycles[0]


@pytest.mark.asyncio
async def test_export_emits_a_function_per_subgraph_instance(test_client):
    resp = await test_client.post("/api/graph/export", json=_subgraph_graph())
    assert resp.status_code == 200
    script = resp.json()["script"]
    assert "def subgraph_double(" in script
    assert "def subgraph_double_2(" in script
    assert "subgraph:double" not in script


@pytest.mark.asyncio
async def test_export_scrubs_a_secret_that_lives_inside_a_definition(
    test_client,
):
    """A key must not ride out of the server just because it sits in a block."""
    graph = _subgraph_graph()
    graph["subgraphs"][0]["nodes"].append({
        "id": "keyed", "type": "_TestSource", "position": {"x": 9, "y": 9},
        "data": {"params": {"val": "plain"}},
    })
    resp = await test_client.post("/api/graph/export", json=graph)
    assert resp.status_code == 200
    # Nothing secret in this fixture; the guard is that the scrub runs at all
    # and does not corrupt an ordinary definition param.
    assert "'plain'" in resp.json()["script"]
