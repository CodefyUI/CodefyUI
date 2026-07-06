"""Unit tests for the SECRET-param scrub / lint helpers (Item 1)."""

from __future__ import annotations

from typing import Any

import pytest

from app.core.node_base import (
    BaseNode,
    ParamDefinition,
    ParamType,
    PortDefinition,
)
from app.core.node_registry import registry
from app.core.preset_registry import preset_registry
from app.core.secret_params import (
    find_secret_violations,
    scrub_graph_secrets,
    secret_param_names,
)
from app.schemas.models import InternalNodeSchema, PresetDefinition


def test_secret_param_names_llmchat():
    assert secret_param_names("LLMChat") == {
        "openai_api_key", "anthropic_api_key",
    }


def test_secret_param_names_unknown_and_empty():
    assert secret_param_names("TotallyUnknownNode") == set()
    assert secret_param_names("") == set()
    # A node with no secret params (Print) returns an empty set.
    assert secret_param_names("Print") == set()


def test_scrub_blanks_only_nonempty_secrets():
    nodes = [
        {"id": "a", "type": "LLMChat", "data": {"params": {
            "openai_api_key": "sk-1", "anthropic_api_key": "", "model": "m"}}},
        {"id": "b", "type": "Print", "data": {"params": {"label": "x"}}},
    ]
    # Only the one non-empty secret is rewritten.
    assert scrub_graph_secrets(nodes) == 1
    assert nodes[0]["data"]["params"]["openai_api_key"] == ""
    assert nodes[0]["data"]["params"]["anthropic_api_key"] == ""
    assert nodes[0]["data"]["params"]["model"] == "m"   # non-secret untouched
    assert nodes[1]["data"]["params"] == {"label": "x"}  # no secret params


def test_scrub_tolerates_missing_or_malformed_params():
    nodes = [
        {"id": "a", "type": "LLMChat", "data": {}},            # no params key
        {"id": "b", "type": "LLMChat"},                        # no data key
        {"id": "c", "type": "LLMChat", "data": {"params": None}},  # wrong type
        {"id": "d", "type": "LLMChat", "data": {"params": {}}},    # empty
    ]
    assert scrub_graph_secrets(nodes) == 0


def test_find_violations_reports_node_and_param_sorted():
    nodes = [
        {"id": "a", "type": "LLMChat", "data": {"params": {
            "anthropic_api_key": "sk-ant", "openai_api_key": "sk-oai"}}},
        {"id": "b", "type": "LLMChat", "data": {"params": {
            "openai_api_key": ""}}},  # empty -> not a violation
        {"id": "c", "type": "Print", "data": {"params": {"label": "x"}}},
    ]
    # Param order within a node is deterministic (sorted).
    assert find_secret_violations(nodes) == [
        {"node_id": "a", "param": "anthropic_api_key"},
        {"node_id": "a", "param": "openai_api_key"},
    ]


def test_find_violations_flags_non_string_secret():
    # A hand-edited file could put a non-string truthy value in a secret slot.
    nodes = [{"id": "a", "type": "LLMChat",
              "data": {"params": {"openai_api_key": 12345}}}]
    assert find_secret_violations(nodes) == [
        {"node_id": "a", "param": "openai_api_key"},
    ]


# ── Preset-embedded secrets: internalParams walk (C1) ────────────────


@pytest.fixture
def _secret_preset():
    """Register a preset whose inner nodes include a real LLMChat (which
    declares two SECRET params); remove it after the test so the global
    preset registry is left as discovered."""
    preset = PresetDefinition(
        preset_name="SecretChat",
        category="Test",
        description="",
        nodes=[
            InternalNodeSchema(id="chat", type="LLMChat", params={}),
            InternalNodeSchema(id="printer", type="Print", params={}),
        ],
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


def test_scrub_blanks_preset_internal_secret(_secret_preset):
    nodes = [
        {"id": "p1", "type": "preset:SecretChat", "data": {"internalParams": {
            "chat": {"openai_api_key": "sk-leak", "model": "gpt-5.2"},
            "printer": {"label": "out"},
        }}},
    ]
    assert scrub_graph_secrets(nodes) == 1
    inner = nodes[0]["data"]["internalParams"]["chat"]
    assert inner["openai_api_key"] == ""       # secret blanked
    assert inner["model"] == "gpt-5.2"         # non-secret override kept
    assert nodes[0]["data"]["internalParams"]["printer"] == {"label": "out"}


def test_find_violations_reports_preset_internal_secret(_secret_preset):
    nodes = [
        {"id": "p1", "type": "preset:SecretChat", "data": {"internalParams": {
            "chat": {"anthropic_api_key": "sk-ant", "openai_api_key": ""},
        }}},
    ]
    # Only the non-empty secret; reported as <inner_id>.<param>.
    assert find_secret_violations(nodes) == [
        {"node_id": "p1", "param": "chat.anthropic_api_key"},
    ]


def test_preset_scrub_tolerates_missing_malformed_and_unknown(_secret_preset):
    nodes = [
        {"id": "a", "type": "preset:SecretChat", "data": {}},
        {"id": "b", "type": "preset:SecretChat",
         "data": {"internalParams": None}},
        {"id": "c", "type": "preset:SecretChat",
         "data": {"internalParams": {"chat": None}}},
        # Unknown preset: inner node types are unresolvable, so it is left
        # untouched (same philosophy as an unknown node type).
        {"id": "d", "type": "preset:NoSuchPreset",
         "data": {"internalParams": {"chat": {"openai_api_key": "x"}}}},
    ]
    assert scrub_graph_secrets(nodes) == 0
    assert find_secret_violations(nodes) == []
    # The unknown preset's baked secret is genuinely left as-is.
    assert nodes[3]["data"]["internalParams"]["chat"]["openai_api_key"] == "x"


# ── Plugin-namespaced node types (M5: suffix-fallback path) ──────────


class _FakeSecretPluginNode(BaseNode):
    NODE_NAME = "SecretPluginNode"
    CATEGORY = "Test"
    DESCRIPTION = ""

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [ParamDefinition(name="token", param_type=ParamType.SECRET,
                                default="")]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any],
                progress_callback: Any | None = None, *,
                context: Any = None) -> dict[str, Any]:
        return {}


@pytest.fixture
def _plugin_secret_node():
    registry._nodes["pk:SecretPluginNode"] = _FakeSecretPluginNode
    try:
        yield
    finally:
        registry._nodes.pop("pk:SecretPluginNode", None)


def test_secret_param_names_plugin_namespaced(_plugin_secret_node):
    # Exact qualified lookup.
    assert secret_param_names("pk:SecretPluginNode") == {"token"}
    # Bare lookup resolves through the registry's suffix-fallback scan.
    assert secret_param_names("SecretPluginNode") == {"token"}


def test_scrub_plugin_namespaced_secret(_plugin_secret_node):
    nodes = [
        {"id": "q", "type": "pk:SecretPluginNode",
         "data": {"params": {"token": "sk-plugin"}}},
        {"id": "r", "type": "SecretPluginNode",   # suffix-fallback resolves
         "data": {"params": {"token": "sk-bare"}}},
    ]
    assert scrub_graph_secrets(nodes) == 2
    assert nodes[0]["data"]["params"]["token"] == ""
    assert nodes[1]["data"]["params"]["token"] == ""
    assert find_secret_violations(nodes) == []
