"""Tests for the /api/presets surface, focused on the SECRET-param
guarantees added in the secret-params work (C1 / I3):

- a SECRET param (an LLM API key) is never EXPOSED as a preset param, and
- its raw VALUE is scrubbed out of the stored preset definition file.
"""

from __future__ import annotations

import json
from pathlib import Path
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


@pytest.mark.asyncio
async def test_exposed_device_param_lists_only_this_machines_devices(
    test_client, _isolated_presets, monkeypatch,
):
    """An exposed ``device`` SELECT is narrowed the way the node API narrows
    it, so a preset never offers a backend this machine does not have."""
    import torch

    from app.core.device_utils import get_available_devices

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    get_available_devices.cache_clear()
    try:
        resp = await test_client.post("/api/presets/create", json={
            "name": "Inference Preset",
            "nodes": [
                {"id": "inf", "type": "Inference",
                 "data": {"params": {"device": "auto"}}},
            ],
            "edges": [],
        })
    finally:
        get_available_devices.cache_clear()
    assert resp.status_code == 200, resp.text
    by_name = {p["param_name"]: p for p in resp.json()["exposed_params"]}
    options = by_name["device"]["param_def"]["options"]
    assert options == ["auto", "cpu"]
    assert "cuda" not in options


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


# -- the name is a filename, not a path (#476) ----------------------------
#
# `POST /api/presets/create` turned the request's `name` straight into a
# path with nothing but `.replace(" ", "_").replace("/", "_")` in the way.
# A backslash was never replaced and `WindowsPath` honours it as a
# separator, and a drive-qualified name is absolute -- so on Windows, this
# project's primary platform, `PRESETS_DIR / name` landed wherever the name
# said and the server wrote attacker-chosen JSON there.
#
# Every rule below is enforced on EVERY platform, not under a
# `sys.platform` fake. A preset file travels (it is copied between
# machines, synced, committed), so a name that is a path on Windows is a
# bad name on Linux too -- and a rule that only fires on the host that runs
# the tests is a rule CI cannot check.


@pytest.fixture
def _presets_sandbox(tmp_path, monkeypatch):
    """`PRESETS_DIR` one level UNDER the sandbox root.

    The room above the directory is the point: a refused name has to leave
    the sandbox empty, not merely leave `PRESETS_DIR` empty, or a test
    would pass on the traversal it is meant to catch.
    """
    presets_dir = tmp_path / "presets"
    presets_dir.mkdir()
    monkeypatch.setattr("app.config.settings.PRESETS_DIR", presets_dir)
    saved = dict(preset_registry._presets)
    try:
        yield presets_dir
    finally:
        preset_registry._presets.clear()
        preset_registry._presets.update(saved)


def _sandbox_files(presets_dir: Path) -> list[str]:
    """Every file anywhere in the sandbox, relative to its root.

    ``as_posix`` so the expected value spells the same on both platforms:
    a literal ``"presets/x.json"`` compares equal on Linux and fails on
    Windows, which is the wrong way round for a Windows-first bug.
    """
    root = presets_dir.parent
    return sorted(
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
    )


def _one_node():
    """A canvas the endpoint accepts: one node, ports left unconnected."""
    return [
        {"id": "opt", "type": "Optimizer",
         "data": {"params": {"type": "Adam", "lr": 0.01}}},
    ]


async def _create(test_client, name: str):
    return await test_client.post("/api/presets/create", json={
        "name": name, "nodes": _one_node(), "edges": [],
    })


@pytest.mark.asyncio
@pytest.mark.parametrize("name, character", [
    ("..\\..\\evil", "\\"),
    ("sub\\evil", "\\"),
    ("..\\..\\..\\Windows\\Temp\\evil", "\\"),
    ("../../evil", "/"),
    ("Vision/Classifier", "/"),
])
async def test_a_name_carrying_a_separator_is_refused(
    test_client, _presets_sandbox, name, character,
):
    """Both separators, refused rather than rewritten.

    `Vision/Classifier` is in here on purpose: it used to succeed, silently,
    as `vision_classifier.json` -- which is also the file `Vision Classifier`
    writes, so one of the two overwrote the other with no warning. Refusing
    asks the user for a name instead of picking one for them.
    """
    resp = await _create(test_client, name)
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert detail["code"] == "name_separator"
    assert detail["character"] == character
    assert _sandbox_files(_presets_sandbox) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("name", [
    "C:\\Windows\\Temp\\evil",
    "C:/Windows/Temp/evil",
    "D:evil",
    "evil:stream",
])
async def test_a_drive_qualified_name_is_refused(
    test_client, _presets_sandbox, name,
):
    """`Path.__truediv__` DISCARDS the left side for an absolute right side,
    so a drive-qualified name ignored `PRESETS_DIR` entirely. The colon is
    refused on its own account too: on Windows it also opens an alternate
    data stream on a file whose name looks perfectly ordinary.
    """
    resp = await _create(test_client, name)
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert detail["code"] == "name_separator"
    assert detail["character"] == ":"
    assert _sandbox_files(_presets_sandbox) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["///", "\\\\", "/\\/", ":"])
async def test_a_name_that_is_only_separators_is_refused(
    test_client, _presets_sandbox, name,
):
    """Nothing is left to name a file with once the separators are gone."""
    resp = await _create(test_client, name)
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"]["code"] == "name_separator"
    assert _sandbox_files(_presets_sandbox) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("name, reserved", [
    ("CON", "con"),
    ("NUL", "nul"),
    ("COM1", "com1"),
    ("LPT1", "lpt1"),
    ("con", "con"),
    # The extension is no protection: Windows resolves the name before the
    # first dot, so `com1.json` -- which is exactly what this endpoint
    # writes -- opens the serial port rather than creating a file.
    ("CON.json", "con"),
    ("NUL.txt", "nul"),
    ("LPT1.preset.json", "lpt1"),
])
async def test_a_reserved_windows_device_name_is_refused(
    test_client, _presets_sandbox, name, reserved,
):
    resp = await _create(test_client, name)
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert detail["code"] == "name_reserved_device"
    assert detail["reserved"] == reserved
    assert _sandbox_files(_presets_sandbox) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["", "   ", "\t"])
async def test_an_empty_name_is_refused(test_client, _presets_sandbox, name):
    resp = await _create(test_client, name)
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"]["code"] in {"name_empty",
                                             "name_control_character"}
    assert _sandbox_files(_presets_sandbox) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("name", [".", "..", "..."])
async def test_a_name_that_is_only_dots_is_refused(
    test_client, _presets_sandbox, name,
):
    """A parent segment, with the separators already gone."""
    resp = await _create(test_client, name)
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"]["code"] == "name_dot_segment"
    assert _sandbox_files(_presets_sandbox) == []


@pytest.mark.asyncio
async def test_a_control_character_in_a_name_is_refused(
    test_client, _presets_sandbox,
):
    """An embedded NUL is what `Path.resolve()` raises `ValueError` on -- a
    500 with a traceback for anyone who can reach the port."""
    resp = await _create(test_client, "evil\x00.json")
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"]["code"] == "name_control_character"
    assert _sandbox_files(_presets_sandbox) == []


@pytest.mark.asyncio
async def test_a_name_differing_only_by_case_does_not_overwrite(
    test_client, _presets_sandbox,
):
    """The registry's duplicate check is case-SENSITIVE and the filename is
    lowercased, so `llm preset` used to walk straight over `LLM Preset`'s
    file and take its place in the registry."""
    first = await _create(test_client, "LLM Preset")
    assert first.status_code == 200, first.text

    second = await _create(test_client, "llm preset")
    assert second.status_code == 409, second.text
    detail = second.json()["detail"]
    assert detail["code"] == "preset_file_exists"
    assert detail["filename"] == "llm_preset.json"

    # The first preset is still the one on disk, and still the only file.
    assert _sandbox_files(_presets_sandbox) == ["presets/llm_preset.json"]
    stored = json.loads(
        (_presets_sandbox / "llm_preset.json").read_text(encoding="utf-8"))
    assert stored["preset_name"] == "LLM Preset"


@pytest.mark.asyncio
@pytest.mark.parametrize("name, filename", [
    ("視覺 分類器", "視覺_分類器.json"),
    ("殘差區塊", "殘差區塊.json"),
    ("ResNet Block", "resnet_block.json"),
    ("block-2_v3", "block-2_v3.json"),
])
async def test_a_legitimate_name_still_works(
    test_client, _presets_sandbox, name, filename,
):
    """The guard must not cost this project its own users' names: titles
    here are routinely Traditional Chinese, with spaces."""
    resp = await _create(test_client, name)
    assert resp.status_code == 200, resp.text

    written = [p for p in _presets_sandbox.iterdir() if p.is_file()]
    assert [p.name for p in written] == [filename]
    # The path that was actually written stays inside PRESETS_DIR.
    assert written[0].resolve().parent == _presets_sandbox.resolve()
    assert json.loads(written[0].read_text(encoding="utf-8"))[
        "preset_name"] == name


@pytest.mark.parametrize("filename", [
    "../evil.json",
    "../../evil.json",
    "sub/evil.json",
    "./sub/../../evil.json",
    # Not an escape: the string `resolve()` REFUSES. It raises ValueError on
    # an embedded NUL (on Windows and on POSIX alike), and letting that out
    # would turn the refusal into a 500 with a traceback in the log.
    "evil\x00.json",
])
def test_the_resolved_path_is_checked_against_presets_dir(tmp_path, filename):
    """The second layer, on its own terms.

    The name rules above should mean nothing ever reaches here -- which is
    exactly why this calls the containment check directly rather than
    through the route. Validation answers "is this string a filename"; this
    answers "is the path it produced still inside the directory", and the
    two fail in different ways. Every case here escapes on POSIX as well as
    on Windows, so CI checks the same thing the developer's box does.
    """
    from fastapi import HTTPException

    from app.api.routes_presets import _resolved_under

    with pytest.raises(HTTPException) as caught:
        _resolved_under(tmp_path, filename)
    assert caught.value.status_code == 400
    assert caught.value.detail["code"] == "name_escapes_presets_dir"


def test_the_containment_check_accepts_a_direct_child(tmp_path):
    """Same function, the answer that has to keep working."""
    from app.api.routes_presets import _resolved_under

    assert _resolved_under(tmp_path, "ok.json") == (
        tmp_path.resolve() / "ok.json")


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
