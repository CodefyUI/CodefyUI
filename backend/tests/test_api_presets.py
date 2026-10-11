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

from app.core.execution_context import ExecutionContext
from app.core.graph_engine import (
    execute_graph,
    prepare_executable_graph,
    validate_graph,
)
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
    preset registry afterward (create_preset adds what it writes to it).

    Yields the dir a preset is written to, ``USER_PRESETS_DIR`` (#600). The
    built-in ``PRESETS_DIR`` points at a sibling, so a preset written into
    the wrong one is not where these tests look for it.
    """
    user_dir = tmp_path / "user"
    user_dir.mkdir()
    monkeypatch.setattr("app.config.settings.USER_PRESETS_DIR", user_dir)
    monkeypatch.setattr("app.config.settings.PRESETS_DIR", tmp_path / "builtin")
    saved = dict(preset_registry._presets)
    try:
        yield user_dir
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
    """`USER_PRESETS_DIR`, where a preset is written, one level UNDER the
    sandbox root.

    The room above the directory is the point: a refused name has to leave
    the sandbox empty, not merely leave the presets dir empty, or a test
    would pass on the traversal it is meant to catch. The built-in
    `PRESETS_DIR` is a sibling inside the same sandbox, so a write that
    lands there shows up as well.
    """
    presets_dir = tmp_path / "presets"
    presets_dir.mkdir()
    monkeypatch.setattr("app.config.settings.USER_PRESETS_DIR", presets_dir)
    monkeypatch.setattr("app.config.settings.PRESETS_DIR", tmp_path / "builtin")
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
@pytest.mark.parametrize("name, character", [
    ("What is this?", "?"),
    ("loss|acc", "|"),
    ("a*b", "*"),
    ("a<b", "<"),
    ("a>b", ">"),
    ('say "hi"', '"'),
])
async def test_a_character_windows_refuses_is_refused(
    test_client, _presets_sandbox, name, character,
):
    """#520: the name rule is shared with the upload routes now, and it
    refuses the rest of what Windows cannot store in a file name. Before,
    these were written as-is on Linux and answered 500 on Windows, where the
    write itself fails."""
    resp = await _create(test_client, name)
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == {"code": "name_reserved_character",
                                     "character": character}
    assert _sandbox_files(_presets_sandbox) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("length", [251, 300])
async def test_a_name_whose_file_would_be_too_long_is_refused(
    test_client, _presets_sandbox, length,
):
    """#520: measured on the file written, ``.json`` included. A 251-character
    name is a 256-character file name, which answered 500."""
    resp = await _create(test_client, "a" * length)
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == {"code": "name_too_long", "limit": 255}
    assert _sandbox_files(_presets_sandbox) == []


def test_a_name_whose_file_is_exactly_at_the_limit_is_kept():
    """The counterweight, without a write: a 255-character path component
    under a deep temp directory is a Windows MAX_PATH question, not this
    rule's."""
    from app.api.routes_presets import _preset_filename

    assert _preset_filename("a" * 250) == "a" * 250 + ".json"


@pytest.mark.asyncio
@pytest.mark.parametrize("name, reserved", [
    ("CONIN$", "conin$"),
    ("conout$.v2", "conout$"),
    ("COM\N{SUPERSCRIPT ONE}", "com\N{SUPERSCRIPT ONE}"),
])
async def test_the_other_device_spellings_are_refused(
    test_client, _presets_sandbox, name, reserved,
):
    resp = await _create(test_client, name)
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == {"code": "name_reserved_device",
                                     "reserved": reserved}
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
    # A device name only before the space, which the file name does not
    # keep: con_.json is an ordinary file on every Windows.
    ("CON ", "con_.json"),
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
    # The path that was actually written stays inside the presets dir.
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


# -- Start, its trigger wires and notes are left out (#600) -----------------
#
# Export takes the whole canvas, and a canvas needs a Start to be run and
# checked, so an exported preset carried one. The trigger wire out of it was
# stored without its `type`, and expansion brought it back as a DATA edge into
# a port called `__trigger`: a graph using the card validated clean and its
# run refused with "Invalid input port '__trigger' on TextInput". Start does
# no work, and a placed card starts its inner roots on its own trigger
# (#561), so both are left out rather than refused.


def _canvas_node(node_id: str, node_type: str,
                 params: dict | None = None) -> dict:
    return {"id": node_id, "type": node_type, "position": {"x": 0, "y": 0},
            "data": {"params": dict(params or {})}}


def _note(node_id: str) -> dict:
    """A canvas note the way the editor serializes one: no params at all."""
    return {"id": node_id, "type": "note", "position": {"x": 0, "y": 0},
            "data": {"noteKind": "text", "noteContent": "what the card does"}}


#: Start's trigger wire as the canvas sends it.
_CANVAS_TRIGGER = {"sourceHandle": "trigger", "targetHandle": "__trigger",
                   "type": "trigger"}

#: The same wire in every shape a request may send it: with both of its marks
#: (the type and the `__trigger` end), with one of them, or with neither, when
#: only its end on Start gives it away.
_TRIGGER_SHAPES = [
    pytest.param(_CANVAS_TRIGGER, id="as the canvas sends it"),
    pytest.param({"sourceHandle": "trigger", "targetHandle": "__trigger"},
                 id="without its type"),
    pytest.param({"sourceHandle": "trigger", "type": "trigger"},
                 id="without a target handle"),
    pytest.param({"sourceHandle": "trigger"}, id="with only its source handle"),
]


def _say_hi_canvas(trigger: dict) -> dict:
    """Start -> TextInput("hi") -> Print, with *trigger* as Start's wire."""
    return {
        "nodes": [
            _canvas_node("start", "Start"),
            _canvas_node("t", "TextInput", {"value": "hi"}),
            _canvas_node("p", "Print", {"label": "inner"}),
        ],
        "edges": [
            {"id": "et", "source": "start", "target": "t", **trigger},
            {"id": "e1", "source": "t", "target": "p",
             "sourceHandle": "text", "targetHandle": "value"},
        ],
    }


def _stored(presets_dir: Path, filename: str) -> dict:
    return json.loads((presets_dir / filename).read_text(encoding="utf-8"))


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", _TRIGGER_SHAPES)
async def test_export_leaves_out_start_and_its_trigger_wire(
    test_client, _isolated_presets, trigger,
):
    resp = await test_client.post("/api/presets/create", json={
        "name": "Say Hi", **_say_hi_canvas(trigger),
    })
    assert resp.status_code == 200, resp.text

    stored = _stored(_isolated_presets, "say_hi.json")
    # Numbered over the nodes that were kept.
    assert [(n["id"], n["type"]) for n in stored["nodes"]] == [
        ("node_0", "TextInput"), ("node_1", "Print")]
    assert stored["edges"] == [{
        "source": "node_0", "target": "node_1",
        "sourceHandle": "text", "targetHandle": "value",
    }]


@pytest.mark.asyncio
@pytest.mark.parametrize("marks", [
    pytest.param({"type": "trigger"}, id="by its type"),
    pytest.param({"targetHandle": "__trigger"}, id="by its trigger end"),
])
async def test_a_trigger_wire_from_another_node_is_left_out(
    test_client, _isolated_presets, marks,
):
    """A hand-rolled request can mark a wire from any node as a trigger. It
    carries no data either way, and kept it would come back as a DATA edge."""
    canvas = _say_hi_canvas(_CANVAS_TRIGGER)
    canvas["nodes"][0] = _canvas_node("a", "TextInput", {"value": "x"})
    canvas["edges"][0] = {"id": "w", "source": "a", "target": "t",
                          "sourceHandle": "text", **marks}
    resp = await test_client.post("/api/presets/create", json={
        "name": "Say Hi", **canvas,
    })
    assert resp.status_code == 200, resp.text

    assert _stored(_isolated_presets, "say_hi.json")["edges"] == [{
        "source": "node_1", "target": "node_2",
        "sourceHandle": "text", "targetHandle": "value",
    }]


@pytest.mark.asyncio
async def test_an_unwired_start_is_left_out_too(test_client, _isolated_presets):
    """Left out by its type, not by its wire: an unwired Start used to give
    the card a TRIGGER output named after it."""
    canvas = _say_hi_canvas(_CANVAS_TRIGGER)
    canvas["edges"] = canvas["edges"][1:]
    resp = await test_client.post("/api/presets/create", json={
        "name": "Say Hi", **canvas,
    })
    assert resp.status_code == 200, resp.text

    stored = _stored(_isolated_presets, "say_hi.json")
    assert [n["type"] for n in stored["nodes"]] == ["TextInput", "Print"]
    assert "TRIGGER" not in {p["data_type"] for p in stored["exposed_outputs"]}


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", _TRIGGER_SHAPES)
async def test_start_runs_a_card_exported_from_a_canvas_with_start(
    test_client, _isolated_presets, trigger,
):
    """The browser e2e failure, end to end: Start wired into the card."""
    resp = await test_client.post("/api/presets/create", json={
        "name": "Say Hi", **_say_hi_canvas(trigger),
    })
    assert resp.status_code == 200, resp.text

    nodes = [_canvas_node("start", "Start"),
             _canvas_node("c", "preset:Say Hi")]
    edges = [{"id": "t", "source": "start", "target": "c", **_CANVAS_TRIGGER}]
    assert validate_graph(nodes, edges) == []
    executable, _edges, _mapping = prepare_executable_graph(nodes, edges)
    assert sorted(n["id"] for n in executable) == [
        "c__node_0", "c__node_1", "start"]

    results = await execute_graph(nodes, edges, context=ExecutionContext(
        device="cpu", weights_persistent=False, graph_id="b2-export-start"))
    assert results["c__node_1"]["value"] == "hi"

    exported = await test_client.post("/api/graph/export", json={
        "name": "uses-say-hi", "nodes": nodes, "edges": edges})
    assert exported.status_code == 200, exported.text


@pytest.mark.asyncio
@pytest.mark.parametrize("nodes", [
    pytest.param([], id="empty"),
    pytest.param([_canvas_node("start", "Start")], id="only Start"),
    pytest.param([_canvas_node("start", "Start"), _note("n")],
                 id="Start and a note"),
])
async def test_a_canvas_with_nothing_but_start_is_refused(
    test_client, _isolated_presets, nodes,
):
    resp = await test_client.post("/api/presets/create", json={
        "name": "Nothing", "nodes": nodes, "edges": [],
    })
    assert resp.status_code == 400, resp.text
    # Coded, so the editor says it in the user's language (#618).
    assert resp.json()["detail"] == {"code": "preset_empty"}
    # Nothing was written, in either presets dir.
    assert list(_isolated_presets.parent.rglob("*.json")) == []


@pytest.mark.asyncio
async def test_a_note_on_the_canvas_is_left_out(test_client, _isolated_presets):
    """No preset can hold a note: the registry refused to load the file it
    was written into. That answered 500 and left the file behind, so every
    later export under the name was refused with 409."""
    canvas = _say_hi_canvas(_CANVAS_TRIGGER)
    canvas["nodes"].append(_note("n"))
    resp = await test_client.post("/api/presets/create", json={
        "name": "Say Hi", **canvas,
    })
    assert resp.status_code == 200, resp.text

    stored = _stored(_isolated_presets, "say_hi.json")
    assert [n["type"] for n in stored["nodes"]] == ["TextInput", "Print"]


# -- an exported preset is user content (#600) -------------------------------
#
# Exported presets were written into `backend/app/presets/`, the tracked
# directory of the built-in ones, where `cdui update` checks code out over
# untracked files. They now go to `USER_PRESETS_DIR` (`backend/data/presets/`
# by default), and every place that discovers presets reads it.


def _user_card_file() -> dict:
    """A preset file the way Export as Subgraph writes one."""
    return {
        "preset_name": "User Card",
        "category": "Custom",
        "description": "",
        "tags": [],
        "nodes": [{"id": "node_0", "type": "TextInput",
                   "params": {"value": "x"}}],
        "edges": [],
        "exposed_inputs": [],
        "exposed_outputs": [{"name": "node_0_text", "internal_node": "node_0",
                             "internal_port": "text"}],
        "exposed_params": [],
    }


def test_the_user_presets_dir_is_install_wide_user_content(
    tmp_path, monkeypatch,
):
    """What the docs promise about it: beside the saved graphs and outside
    the code tree, moved by its own variable, and one dir for every project
    (a graph that uses a preset carries its definition in `presets[]`)."""
    from app.config import Settings

    for name in ("CODEFYUI_USER_PRESETS_DIR", "CODEFYUI_GRAPHS_DIR",
                 "CODEFYUI_PROJECT_DIR"):
        monkeypatch.delenv(name, raising=False)
    install = Settings()
    assert install.USER_PRESETS_DIR == install.GRAPHS_DIR.parent / "presets"
    assert not install.USER_PRESETS_DIR.is_relative_to(install.NODES_DIR.parent)

    assert Settings(PROJECT_DIR=tmp_path).USER_PRESETS_DIR == (
        install.USER_PRESETS_DIR)

    monkeypatch.setenv("CODEFYUI_USER_PRESETS_DIR", str(tmp_path / "mine"))
    assert Settings().USER_PRESETS_DIR == tmp_path / "mine"


def _user_presets_dir(root: Path) -> Path:
    """A user presets dir holding the one preset of :func:`_user_card_file`."""
    user_dir = root / "user-presets"
    user_dir.mkdir()
    (user_dir / "user_card.json").write_text(
        json.dumps(_user_card_file()), encoding="utf-8")
    return user_dir


@pytest.mark.asyncio
async def test_an_exported_preset_is_written_to_the_user_presets_dir(
    test_client, _isolated_presets, tmp_path, monkeypatch,
):
    """Written where the saved graphs live, never into the built-in dir."""
    from app.config import settings

    builtin = tmp_path / "app" / "presets"
    builtin.mkdir(parents=True)
    # Not there yet, as on a fresh install: the first export creates it.
    user_dir = tmp_path / "data" / "presets"
    monkeypatch.setattr(settings, "PRESETS_DIR", builtin)
    monkeypatch.setattr(settings, "USER_PRESETS_DIR", user_dir)

    resp = await _create(test_client, "My Optimizer")
    assert resp.status_code == 200, resp.text

    assert [p.name for p in user_dir.iterdir()] == ["my_optimizer.json"]
    assert list(builtin.iterdir()) == []


@pytest.mark.asyncio
async def test_creating_a_preset_keeps_every_other_preset_listed(
    test_client, _isolated_presets,
):
    """create_preset cleared the registry and rediscovered one directory, so
    a preset a plugin pack supplied left the palette until Reload Nodes."""
    from app.core.node_registry import registry as node_registry

    preset_registry._presets["Pack Card"] = preset_registry._load_and_resolve(
        {**_user_card_file(), "preset_name": "Pack Card"}, node_registry)
    before = {p["preset_name"]
              for p in (await test_client.get("/api/presets")).json()}
    assert "Pack Card" in before

    resp = await _create(test_client, "Mine")
    assert resp.status_code == 200, resp.text

    after = {p["preset_name"]
             for p in (await test_client.get("/api/presets")).json()}
    assert after == before | {"Mine"}


@pytest.mark.asyncio
async def test_a_create_loads_only_the_new_preset(test_client, _isolated_presets):
    """At startup a plugin pack's preset wins a name clash with a user
    preset, because packs load last. Rereading the whole user dir after a
    create handed the name to the user's file until the next reload."""
    from app.core.node_registry import registry as node_registry

    (_isolated_presets / "clash.json").write_text(json.dumps(
        {**_user_card_file(), "preset_name": "Clash", "description": "mine"}),
        encoding="utf-8")
    preset_registry._presets["Clash"] = preset_registry._load_and_resolve(
        {**_user_card_file(), "preset_name": "Clash",
         "description": "from the pack"}, node_registry)

    resp = await _create(test_client, "Mine")
    assert resp.status_code == 200, resp.text

    assert preset_registry.get("Clash").description == "from the pack"
    assert preset_registry.get("Mine") is not None


@pytest.mark.asyncio
async def test_a_name_a_built_in_preset_file_has_is_refused(
    test_client, _isolated_presets,
):
    """The registry's duplicate check is case-sensitive, and the built-ins
    (with presets exported by older versions) sit in another dir than new
    exports, so `vision block` beside `Vision Block` would be two presets
    whose names differ only in case."""
    from app.config import settings
    from app.core.node_registry import registry as node_registry

    raw = {**_user_card_file(), "preset_name": "Vision Block"}
    settings.PRESETS_DIR.mkdir()
    (settings.PRESETS_DIR / "vision_block.json").write_text(
        json.dumps(raw), encoding="utf-8")
    preset_registry._presets["Vision Block"] = (
        preset_registry._load_and_resolve(raw, node_registry))

    resp = await _create(test_client, "vision block")
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"] == {"code": "preset_file_exists",
                                     "filename": "vision_block.json"}
    assert list(_isolated_presets.iterdir()) == []


@pytest.mark.asyncio
async def test_a_preset_that_fails_to_load_leaves_no_file_behind(
    test_client, _isolated_presets, monkeypatch,
):
    """The backstop after the write. The file written for a preset the
    registry refused stayed, so a retry under the same name was refused with
    409 instead of getting the same answer again. Nothing the editor sends
    reaches it any more -- a card is copied in and a node type this server
    does not have is refused before the write (#618) -- so the registry is
    made to refuse the file here."""
    def refuse(path, _node_registry):
        raise ValueError(f"{path.name} cannot be resolved")

    monkeypatch.setattr(preset_registry, "load_file", refuse)
    request = {"name": "Never Loads", **_say_hi_canvas(_CANVAS_TRIGGER)}

    first = await test_client.post("/api/presets/create", json=request)
    retry = await test_client.post("/api/presets/create", json=request)

    assert (first.status_code, retry.status_code) == (500, 500), retry.text
    assert retry.json() == first.json() == {
        "detail": "Failed to load created preset"}
    assert list(_isolated_presets.parent.rglob("*.json")) == []


def test_rediscover_all_reads_the_user_presets_dir_it_is_given(
    tmp_path, monkeypatch,
):
    from app.config import settings
    from app.core import plugin_loader
    from app.core.node_registry import NodeRegistry
    from app.core.preset_registry import PresetRegistry

    monkeypatch.setattr(plugin_loader, "load_lockfile",
                        plugin_loader.empty_lockfile)
    empty = tmp_path / "empty"  # no custom nodes, built-in presets or packs
    user_dir = _user_presets_dir(tmp_path)

    def rediscover(**user_presets: Path) -> tuple[list[str], int]:
        presets = PresetRegistry()
        counts = plugin_loader.rediscover_all(
            NodeRegistry(),
            presets,
            nodes_dir=settings.NODES_DIR,
            custom_nodes_dir=empty,
            presets_dir=empty,
            builtin_root=empty,
            user_root=empty,
            **user_presets,
        )
        return list(presets.presets), counts["presets"]

    assert rediscover(user_presets_dir=user_dir) == (["User Card"], 1)
    assert rediscover() == ([], 0)


def test_a_reload_keeps_the_exported_presets(tmp_path, monkeypatch):
    """Reload Nodes, the plugin routes and an exported script's runtime all
    rebuild the registry through ``rediscover_now``, which clears it first."""
    from app.config import settings
    from app.core import plugin_loader
    from app.core.plugins.reload import rediscover_now

    monkeypatch.setattr(settings, "USER_PRESETS_DIR", _user_presets_dir(tmp_path))
    monkeypatch.setattr(plugin_loader, "load_lockfile",
                        plugin_loader.empty_lockfile)
    saved = dict(preset_registry._presets)
    assert "User Card" not in saved
    try:
        rediscover_now()
        assert "User Card" in preset_registry.presets
    finally:
        preset_registry._presets.clear()
        preset_registry._presets.update(saved)


@pytest.mark.asyncio
async def test_a_restart_keeps_the_exported_presets(tmp_path, monkeypatch):
    """The server's startup reads the dir too. Driven the way
    test_main_lifespan.py drives it: user data and the database in tmp."""
    import app.main as main_mod
    from app.config import settings
    from app.core.auth import init_allowed_hosts

    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path / "userdata"))
    monkeypatch.setattr(settings, "DB_PATH", tmp_path / "db" / "lifespan.db")
    monkeypatch.setattr(settings, "PROJECT_DIR", None)
    monkeypatch.setattr(settings, "USER_PRESETS_DIR", _user_presets_dir(tmp_path))
    monkeypatch.setattr(main_mod, "setup_logging", lambda **kwargs: None)
    saved = dict(preset_registry._presets)
    assert "User Card" not in saved
    try:
        async with main_mod.lifespan(main_mod.app):
            assert "User Card" in preset_registry.presets
    finally:
        preset_registry._presets.clear()
        preset_registry._presets.update(saved)
        # The lifespan re-ran init_allowed_hosts with CORS origins added.
        init_allowed_hosts(settings.HOST, settings.PORT)


def test_project_validate_knows_the_exported_presets(tmp_path, monkeypatch):
    """`cdui project validate` builds its registry "exactly like the server";
    a graph using an exported preset was an unknown preset there."""
    import project
    from app.config import settings

    monkeypatch.setattr(settings, "USER_PRESETS_DIR", _user_presets_dir(tmp_path))
    monkeypatch.setattr(project, "load_lockfile",
                        lambda: {"schema": 1, "plugins": {}})
    saved = dict(preset_registry._presets)
    assert "User Card" not in saved
    try:
        project._init_registries_like_server()
        assert "User Card" in preset_registry.presets
    finally:
        preset_registry._presets.clear()
        preset_registry._presets.update(saved)


# -- a preset card and a muted node on the canvas (#618) ---------------------
#
# A card was copied into the preset as an inner node of type `preset:<name>`,
# which the registry cannot load (it resolves inner types through the node
# registry only), so the export answered 500 on every try. And only each
# node's `params` were copied, so a node muted on the canvas ran inside the
# preset. Both are now what a run of the canvas executes: the card opened up
# into its own nodes with its settings applied, the muted node left out.


def _labeler(name: str = "Labeler", label: str = "inner") -> dict:
    """A one-node preset: a Print, its input and output exposed."""
    return {
        "preset_name": name,
        "nodes": [{"id": "p", "type": "Print", "params": {"label": label}}],
        "edges": [],
        "exposed_inputs": [{"name": "value", "internal_node": "p",
                            "internal_port": "value"}],
        "exposed_outputs": [{"name": "value", "internal_node": "p",
                             "internal_port": "value"}],
        "exposed_params": [{"internal_node": "p", "param_name": "label",
                            "display_name": "Label"}],
    }


@pytest.fixture
def _installed_labeler(_isolated_presets):
    """``Labeler`` installed, as a built-in or an exported preset is."""
    from app.core.node_registry import registry as node_registry

    preset_registry._presets["Labeler"] = preset_registry._load_and_resolve(
        _labeler(), node_registry)
    return _isolated_presets


def _card(node_id: str, preset: str,
          internal_params: dict | None = None) -> dict:
    """A placed preset card the way the editor serializes one."""
    return {"id": node_id, "type": f"preset:{preset}",
            "position": {"x": 0, "y": 0},
            "data": {"params": {},
                     "internalParams": dict(internal_params or {})}}


def _muted(node: dict) -> dict:
    """*node* bypassed on the canvas, which writes ``data.bypassed``."""
    node["data"]["bypassed"] = True
    return node


def _wire(source: str, source_handle: str, target: str,
          target_handle: str) -> dict:
    return {"id": f"{source}-{target}", "source": source, "target": target,
            "sourceHandle": source_handle, "targetHandle": target_handle}


async def _export(test_client, name: str, nodes: list, edges: list = (),
                  presets: list | None = None):
    body = {"name": name, "nodes": nodes, "edges": list(edges)}
    if presets is not None:
        body["presets"] = presets
    return await test_client.post("/api/presets/create", json=body)


@pytest.mark.asyncio
async def test_a_preset_card_is_copied_in_as_its_own_nodes(
    test_client, _installed_labeler,
):
    """With the settings made on the card, and its unconnected port the new
    preset's -- and the file loads, so the export no longer answers 500."""
    resp = await _export(test_client, "Holds A Card", [
        _canvas_node("t", "TextInput", {"value": "hi"}),
        _card("card", "Labeler", {"p": {"label": "set on the card"}}),
    ], [_wire("t", "text", "card", "value")])
    assert resp.status_code == 200, resp.text

    stored = _stored(_installed_labeler, "holds_a_card.json")
    assert [(n["id"], n["type"], n["params"]) for n in stored["nodes"]] == [
        ("node_0", "TextInput", {"value": "hi"}),
        ("node_1", "Print", {"label": "set on the card"}),
    ]
    assert stored["edges"] == [{
        "source": "node_0", "target": "node_1",
        "sourceHandle": "text", "targetHandle": "value",
    }]
    assert stored["exposed_inputs"] == []
    assert [(p["internal_node"], p["internal_port"])
            for p in stored["exposed_outputs"]] == [("node_1", "value")]
    assert preset_registry.get("Holds A Card") is not None


@pytest.mark.asyncio
async def test_a_card_is_copied_from_the_definition_the_graph_carries(
    test_client, _installed_labeler,
):
    """The graph's own ``presets[]`` entry wins over an installed preset of
    the same name in a run, so it does here; and a preset only the graph
    carries is copied in too."""
    resp = await _export(test_client, "Owned", [
        _card("a", "Labeler"),
        _card("b", "Graph Only"),
    ], presets=[
        _labeler(label="the graph's"),
        _labeler("Graph Only", label="only in the graph"),
    ])
    assert resp.status_code == 200, resp.text

    assert [(n["type"], n["params"])
            for n in _stored(_installed_labeler, "owned.json")["nodes"]] == [
        ("Print", {"label": "the graph's"}),
        ("Print", {"label": "only in the graph"}),
    ]


@pytest.mark.asyncio
async def test_a_card_whose_preset_is_nowhere_is_refused_by_name(
    test_client, _isolated_presets,
):
    """Neither installed nor carried by the graph: refused before anything
    is written, so a retry gets the same answer rather than a 409."""
    request = {"name": "Lost Card", "nodes": [
        _canvas_node("t", "TextInput", {"value": "x"}),
        _card("card", "Nowhere"),
    ], "edges": []}

    first = await test_client.post("/api/presets/create", json=request)
    retry = await test_client.post("/api/presets/create", json=request)

    assert (first.status_code, retry.status_code) == (400, 400), retry.text
    assert first.json()["detail"] == retry.json()["detail"] == {
        "code": "preset_card_unknown", "preset": "Nowhere"}
    assert list(_isolated_presets.parent.rglob("*.json")) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("stranger, presets", [
    pytest.param(_canvas_node("x", "NotInstalledHere"), None,
                 id="on the canvas"),
    pytest.param(_muted(_canvas_node("x", "NotInstalledHere")), None,
                 id="bypassed on the canvas"),
    pytest.param(_card("x", "Strange Card"),
                 [{"preset_name": "Strange Card",
                   "nodes": [{"id": "s", "type": "NotInstalledHere"}],
                   "edges": []}],
                 id="inside a card's definition"),
])
async def test_a_node_type_this_server_does_not_have_is_refused_by_type(
    test_client, _isolated_presets, stranger, presets,
):
    """A plugin's node whose pack is missing or disabled -- a workspace
    import opens it as a placeholder -- cannot be stored: the registry
    refuses a file that names it. That answered 500 in English after the
    write; it is refused before the write, by type."""
    request = {"name": "Holds A Stranger", **_say_hi_canvas(_CANVAS_TRIGGER)}
    request["nodes"].append(stranger)
    if presets is not None:
        request["presets"] = presets

    first = await test_client.post("/api/presets/create", json=request)
    retry = await test_client.post("/api/presets/create", json=request)

    assert (first.status_code, retry.status_code) == (400, 400), retry.text
    assert first.json()["detail"] == retry.json()["detail"] == {
        "code": "preset_node_unknown", "type": "NotInstalledHere"}
    assert list(_isolated_presets.parent.rglob("*.json")) == []


@pytest.mark.asyncio
async def test_two_cards_of_one_preset_become_separate_nodes(
    test_client, _installed_labeler,
):
    resp = await _export(test_client, "Two Cards", [
        _card("a", "Labeler", {"p": {"label": "first"}}),
        _card("b", "Labeler", {"p": {"label": "second"}}),
    ])
    assert resp.status_code == 200, resp.text

    stored = _stored(_installed_labeler, "two_cards.json")
    assert [(n["id"], n["params"]) for n in stored["nodes"]] == [
        ("node_0", {"label": "first"}), ("node_1", {"label": "second"})]
    assert [p["name"] for p in stored["exposed_inputs"]] == [
        "node_0_value", "node_1_value"]
    assert [p["name"] for p in stored["exposed_outputs"]] == [
        "node_0_value", "node_1_value"]


@pytest.mark.asyncio
async def test_nodes_are_stored_in_the_order_data_flows(
    test_client, _installed_labeler,
):
    """The preset's dialog lists the stored nodes in order, and a card is
    opened up where it stood: one placed before the node feeding it was
    listed first, "Print -> TextInput" for TextInput -> Print. Each node now
    follows what feeds it; a node with no feeder keeps its place."""
    resp = await _export(test_client, "In Flow Order", [
        _card("card", "Labeler"),
        _canvas_node("loose", "Print", {"label": "loose"}),
        _canvas_node("t", "TextInput", {"value": "hi"}),
    ], [_wire("t", "text", "card", "value")])
    assert resp.status_code == 200, resp.text

    stored = _stored(_installed_labeler, "in_flow_order.json")
    assert [(n["id"], n["type"], n["params"]) for n in stored["nodes"]] == [
        ("node_0", "Print", {"label": "loose"}),
        ("node_1", "TextInput", {"value": "hi"}),
        ("node_2", "Print", {"label": "inner"}),
    ]
    assert stored["edges"] == [{
        "source": "node_1", "target": "node_2",
        "sourceHandle": "text", "targetHandle": "value",
    }]


@pytest.mark.asyncio
async def test_two_nodes_of_one_type_are_numbered_in_flow_order(
    test_client, _installed_labeler,
):
    """Two "Print - label" fields under one "Print" heading: the dialog could
    not say which was which. A repeated type is numbered in flow order, as
    the built-in presets number theirs ("Activation 1"), one heading each;
    a type that occurs once keeps its name."""
    resp = await _export(test_client, "Two Prints", [
        _card("card", "Labeler", {"p": {"label": "second"}}),
        _canvas_node("first", "Print", {"label": "first"}),
        _canvas_node("t", "TextInput", {"value": "hi"}),
    ], [_wire("t", "text", "first", "value"),
        _wire("first", "value", "card", "value")])
    assert resp.status_code == 200, resp.text

    stored = _stored(_installed_labeler, "two_prints.json")
    assert [n["params"] for n in stored["nodes"]] == [
        {"value": "hi"}, {"label": "first"}, {"label": "second"}]
    assert [(p["group"], p["display_name"])
            for p in stored["exposed_params"]] == [
        ("TextInput", "TextInput - value"),
        ("Print 1", "Print 1 - label"),
        ("Print 2", "Print 2 - label"),
    ]
    assert [p["description"] for p in stored["exposed_outputs"]] == [
        "Print 2: Pass-through"]


@pytest.mark.asyncio
async def test_exposed_ports_keep_canvas_order(test_client, _isolated_presets):
    """Map runs a preset through its FIRST exposed input and output, so the
    port lists keep the order they had before the nodes were stored in flow
    order: canvas order, a card's ports where the card stood. Here the flow
    order is solo, src, add, and the ports still list add's first."""
    resp = await _export(test_client, "Ports In Canvas Order", [
        _canvas_node("add", "Add"),
        _canvas_node("solo", "Print", {"label": "solo"}),
        _canvas_node("src", "TextInput", {"value": "x"}),
    ], [_wire("src", "text", "add", "tensor_a")])
    assert resp.status_code == 200, resp.text

    stored = _stored(_isolated_presets, "ports_in_canvas_order.json")
    assert [n["type"] for n in stored["nodes"]] == ["Print", "TextInput", "Add"]
    assert [(p["internal_node"], p["internal_port"])
            for p in stored["exposed_inputs"]] == [
        ("node_2", "tensor_b"), ("node_0", "value")]
    assert [(p["internal_node"], p["internal_port"])
            for p in stored["exposed_outputs"]] == [
        ("node_2", "tensor"), ("node_0", "value")]


@pytest.mark.asyncio
async def test_map_uses_the_ports_the_canvas_lists_first(
    test_client, _isolated_presets,
):
    """The same rule, run through Map: its result comes from the output the
    canvas lists first (the end of the Print chain), not from the loose
    TextInput that flow order puts ahead of it."""
    from app.nodes.dataflow.map_node import MapNode

    resp = await _export(test_client, "Map Body", [
        _canvas_node("last", "Print", {"label": "last"}),
        _canvas_node("loose", "TextInput", {"value": "loose"}),
        _canvas_node("first", "Print", {"label": "first"}),
    ], [_wire("first", "value", "last", "value")])
    assert resp.status_code == 200, resp.text

    out = MapNode().execute({"items": [1, 2]}, {"subgraph": "Map Body"})
    assert out["results"] == [1, 2]


@pytest.mark.asyncio
async def test_a_cycle_keeps_its_place_when_nodes_are_ordered(
    test_client, _isolated_presets,
):
    """A cycle has no flow order. A run refuses it; the export takes it as
    it always has, with the cycle's nodes last in canvas order, instead of
    failing on the ordering."""
    resp = await _export(test_client, "Loop", [
        _canvas_node("a", "Print", {"label": "a"}),
        _canvas_node("b", "Print", {"label": "b"}),
        _canvas_node("t", "TextInput", {"value": "hi"}),
    ], [_wire("a", "value", "b", "value"), _wire("b", "value", "a", "value")])
    assert resp.status_code == 200, resp.text

    assert [n["params"] for n in _stored(_isolated_presets, "loop.json")["nodes"]] == [
        {"value": "hi"}, {"label": "a"}, {"label": "b"}]


@pytest.mark.asyncio
async def test_a_card_nested_in_a_cards_definition_is_copied_in_too(
    test_client, _installed_labeler,
):
    """Expansion repeats until no card is left, as a run's does."""
    wrapper = {
        "preset_name": "Wrapper",
        "nodes": [{"id": "inner", "type": "preset:Labeler"}],
        "edges": [],
    }
    resp = await _export(test_client, "Nested", [_card("w", "Wrapper")],
                         presets=[wrapper])
    assert resp.status_code == 200, resp.text

    assert [(n["type"], n["params"])
            for n in _stored(_installed_labeler, "nested.json")["nodes"]] == [
        ("Print", {"label": "inner"})]


@pytest.mark.asyncio
async def test_a_muted_node_is_left_out_and_wired_past(
    test_client, _isolated_presets,
):
    """A run hands what fed the muted node to the node after it."""
    resp = await _export(test_client, "Skips One", [
        _canvas_node("t", "TextInput", {"value": "hi"}),
        _muted(_canvas_node("m", "Print", {"label": "muted"})),
        _canvas_node("p", "Print", {"label": "after"}),
    ], [_wire("t", "text", "m", "value"), _wire("m", "value", "p", "value")])
    assert resp.status_code == 200, resp.text

    stored = _stored(_isolated_presets, "skips_one.json")
    assert [(n["id"], n["type"], n["params"]) for n in stored["nodes"]] == [
        ("node_0", "TextInput", {"value": "hi"}),
        ("node_1", "Print", {"label": "after"}),
    ]
    assert stored["edges"] == [{
        "source": "node_0", "target": "node_1",
        "sourceHandle": "text", "targetHandle": "value",
    }]


@pytest.mark.asyncio
async def test_an_input_a_muted_node_leaves_empty_becomes_a_port(
    test_client, _isolated_presets,
):
    """Nothing fed the muted node, so nothing reaches the node after it:
    that input is unconnected, and an unconnected port is the preset's."""
    resp = await _export(test_client, "Fed Later", [
        _muted(_canvas_node("m", "Print")),
        _canvas_node("p", "Print", {"label": "after"}),
    ], [_wire("m", "value", "p", "value")])
    assert resp.status_code == 200, resp.text

    stored = _stored(_isolated_presets, "fed_later.json")
    assert [n["params"] for n in stored["nodes"]] == [{"label": "after"}]
    assert stored["edges"] == []
    assert [(p["internal_node"], p["internal_port"])
            for p in stored["exposed_inputs"]] == [("node_0", "value")]


@pytest.mark.asyncio
async def test_a_canvas_whose_every_node_is_muted_is_refused(
    test_client, _isolated_presets,
):
    resp = await _export(test_client, "All Muted",
                         [_muted(_canvas_node("m", "Print"))])
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == {"code": "preset_empty"}
    assert list(_isolated_presets.parent.rglob("*.json")) == []


#: Canvases a run of which is refused, each with the sentence the run says.
_REFUSED_AS_A_RUN_IS = [
    pytest.param(
        [_muted(_canvas_node("m", "TextInput", {"value": "x"})),
         _canvas_node("p", "Print")],
        [_wire("m", "text", "p", "value")], None,
        "Bypassed node m (TextInput): output 'text' (STRING) has no "
        "type-compatible input",
        id="a muted node with nothing to forward"),
    pytest.param(
        [_muted(_card("card", "Labeler")), _canvas_node("p", "Print")],
        [], [_labeler()],
        "Bypass is not supported on preset node(s): card",
        id="a muted card"),
    pytest.param(
        [_canvas_node("t", "TextInput"), _card("card", "Labeler")],
        [_wire("t", "text", "card", "nope")], [_labeler()],
        "Edge targets input port 'nope' which preset 'Labeler' does not "
        "expose (node card)",
        id="an edge on a port the card does not expose"),
    pytest.param(
        [_card("card", "Broken")], [],
        [{"preset_name": "Broken", "edges": []}],
        "Preset 'Broken' is in this graph but could not be read",
        id="a definition the graph carries but cannot read"),
    pytest.param(
        [_card("card", "Blocky")], [],
        [{"preset_name": "Blocky",
          "nodes": [{"id": "blk", "type": "subgraph:inner"}], "edges": []}],
        "Preset 'Blocky' contains subgraph instance(s) blk (node card)",
        id="a definition holding a block"),
    pytest.param(
        [_card("card", "Loop")], [],
        [{"preset_name": "Loop",
          "nodes": [{"id": "again", "type": "preset:Loop"}], "edges": []}],
        "Preset nesting exceeds the maximum depth of 10",
        id="a definition holding itself"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("nodes, edges, presets, sentence",
                         _REFUSED_AS_A_RUN_IS)
async def test_a_canvas_a_run_refuses_is_refused_in_the_runs_words(
    test_client, _isolated_presets, nodes, edges, presets, sentence,
):
    resp = await _export(test_client, "Refused", nodes, edges, presets)
    assert resp.status_code == 400, resp.text
    assert sentence in resp.json()["detail"]
    assert list(_isolated_presets.parent.rglob("*.json")) == []


@pytest.mark.asyncio
async def test_a_block_two_definitions_down_is_refused(
    test_client, _isolated_presets,
):
    """Checked on every expansion pass, not only the first: a block inside a
    nested definition was left in the stored preset as a ``subgraph:`` node,
    and the registry cannot load a file that holds one."""
    blocky = {"preset_name": "Blocky",
              "nodes": [{"id": "blk", "type": "subgraph:inner"}], "edges": []}
    outer = {"preset_name": "Holds Blocky",
             "nodes": [{"id": "mid", "type": "preset:Blocky"}], "edges": []}
    resp = await _export(test_client, "Refused", [_card("card", "Holds Blocky")],
                         presets=[outer, blocky])
    assert resp.status_code == 400, resp.text
    assert ("Preset 'Blocky' contains subgraph instance(s) blk "
            "(node card__mid)") in resp.json()["detail"]
    assert list(_isolated_presets.parent.rglob("*.json")) == []


@pytest.mark.asyncio
async def test_a_key_inside_a_card_is_not_written(
    test_client, _isolated_presets,
):
    """Blanked AFTER the card is opened up: neither the key typed into the
    card nor the one in the definition the graph carries was a param of a
    node on the canvas until then."""
    chat = {
        "preset_name": "Chat Card",
        "nodes": [{"id": "chat", "type": "LLMChat", "params": {
            "provider": "ChatGPT API", "model": "gpt-5.2",
            "openai_api_key": "sk-in-the-definition"}}],
        "edges": [],
    }
    resp = await _export(test_client, "Chatty", [
        _card("card", "Chat Card",
              {"chat": {"anthropic_api_key": "sk-ant-typed-into-the-card"}}),
    ], presets=[chat])
    assert resp.status_code == 200, resp.text

    written = (_isolated_presets / "chatty.json").read_text(encoding="utf-8")
    assert "sk-in-the-definition" not in written
    assert "sk-ant-typed-into-the-card" not in written
    assert json.loads(written)["nodes"][0]["params"]["model"] == "gpt-5.2"


@pytest.mark.asyncio
async def test_a_name_an_installed_preset_has_is_refused_with_a_code(
    test_client, _installed_labeler,
):
    """Coded like the other name refusals, so the editor can say it in the
    user's language and ask for another name."""
    resp = await _create(test_client, "Labeler")
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"] == {"code": "preset_exists", "name": "Labeler"}
    assert list(_installed_labeler.iterdir()) == []


@pytest.mark.asyncio
async def test_a_canvas_with_every_port_wired_is_refused_with_a_code(
    test_client, _isolated_presets,
):
    resp = await _export(test_client, "Closed", [
        _canvas_node("t", "TextInput", {"value": "x"}),
        _canvas_node("o", "GraphOutput", {"name": "out"}),
    ], [_wire("t", "text", "o", "value")])
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == {"code": "preset_no_ports"}
    assert list(_isolated_presets.parent.rglob("*.json")) == []


@pytest.mark.asyncio
async def test_a_part_of_the_canvas_start_does_not_reach_is_still_exported(
    test_client, _isolated_presets,
):
    """Export takes the whole canvas, as the docs say, unlike a run, which
    skips what nothing from Start reaches: here the `draft` chain, and the
    Print whose open input is meant to be the preset's port. Pruning like a
    run would drop that Print too, the part a preset is usually made for,
    and the run's pruning (forward reach plus its rescue passes) is not a
    helper this route can call. Pinned so that changing it is a decision."""
    resp = await _export(test_client, "Has Draft", [
        _canvas_node("start", "Start"),
        _canvas_node("ta", "TextInput", {"value": "hi"}),
        _canvas_node("pa", "Print", {"label": "a"}),
        _canvas_node("td", "TextInput", {"value": "draft"}),
        _canvas_node("pd", "Print", {"label": "draft"}),
        _canvas_node("port", "Print", {"label": "fed by the preset"}),
    ], [{"id": "t", "source": "start", "target": "ta", **_CANVAS_TRIGGER},
        _wire("ta", "text", "pa", "value"), _wire("td", "text", "pd", "value")])
    assert resp.status_code == 200, resp.text

    stored = _stored(_isolated_presets, "has_draft.json")
    assert [n["params"] for n in stored["nodes"]] == [
        {"value": "hi"}, {"label": "a"}, {"value": "draft"}, {"label": "draft"},
        {"label": "fed by the preset"}]
    assert [p["name"] for p in stored["exposed_inputs"]] == ["node_4_value"]


@pytest.mark.asyncio
async def test_a_preset_made_from_a_card_runs_with_its_input_wired(
    test_client, _installed_labeler,
):
    resp = await _export(test_client, "Outer", [
        _canvas_node("a", "Print", {"label": "outer"}),
        _card("card", "Labeler", {"p": {"label": "card"}}),
    ], [_wire("a", "value", "card", "value")])
    assert resp.status_code == 200, resp.text
    assert [p["name"] for p in resp.json()["exposed_inputs"]] == [
        "node_0_value"]

    nodes = [_canvas_node("start", "Start"),
             _canvas_node("t", "TextInput", {"value": "hi"}),
             _canvas_node("c", "preset:Outer")]
    edges = [{"id": "s", "source": "start", "target": "t", **_CANVAS_TRIGGER},
             _wire("t", "text", "c", "node_0_value")]
    assert validate_graph(nodes, edges) == []
    executable, _edges, _mapping = prepare_executable_graph(nodes, edges)
    assert sorted(n["id"] for n in executable) == [
        "c__node_0", "c__node_1", "start", "t"]

    results = await execute_graph(nodes, edges, context=ExecutionContext(
        device="cpu", weights_persistent=False, graph_id="f1-card-in-preset"))
    assert results["c__node_1"]["value"] == "hi"


# -- #699: the Training Pipeline preset trains on the graph's device ---------


def _pipeline_card(internal_params: dict) -> dict:
    return {
        "id": "train-pipeline",
        "type": "preset:Training Pipeline",
        "position": {"x": 0, "y": 0},
        "data": {"params": {}, "internalParams": internal_params},
    }


def _train_loop_device(internal_params: dict, graph_device: str) -> str:
    from types import SimpleNamespace

    from app.core.device_utils import resolve_node_device
    from app.core.graph_engine import expand_presets

    nodes, _, _ = expand_presets([_pipeline_card(internal_params)], [])
    (loop,) = [n for n in nodes if n["type"] == "TrainingLoop"]
    return resolve_node_device(
        loop["data"]["params"].get("device"), SimpleNamespace(device=graph_device))


def test_a_freshly_dropped_training_pipeline_follows_the_graph_device():
    assert _train_loop_device({}, "cuda") == "cuda"


def test_a_training_pipeline_saved_with_cpu_keeps_cpu():
    assert _train_loop_device({"train_loop": {"device": "cpu"}}, "cuda") == "cpu"
