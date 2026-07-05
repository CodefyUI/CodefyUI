"""Unit tests for the SECRET-param scrub / lint helpers (Item 1)."""

from __future__ import annotations

from app.core.secret_params import (
    find_secret_violations,
    scrub_graph_secrets,
    secret_param_names,
)


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
