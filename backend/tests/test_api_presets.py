"""Tests for the /api/presets surface, focused on the SECRET-param
guarantees added in the secret-params work (C1 / I3):

- a SECRET param (an LLM API key) is never EXPOSED as a preset param, and
- its raw VALUE is scrubbed out of the stored preset definition file.
"""

from __future__ import annotations

import json

import pytest

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
