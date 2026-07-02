"""Tests for the LLMChat workflow node."""

from __future__ import annotations

import pytest
import torch

from app.core.node_base import DataType, ParamType
from app.nodes.llm import llm_chat_node
from app.nodes.llm.llm_chat_node import LLMChatNode, _build_request, _normalize_provider


def params(**overrides):
    base = {
        "provider": "ChatGPT API",
        "model": "gpt-5.2",
        "prompt": "Prompt",
        "system_prompt": "System",
        "openai_api_key": "sk-test",
        "anthropic_api_key": "sk-ant-test",
        "ollama_base_url": "http://127.0.0.1:11434/v1",
        "max_tokens": 128,
        "temperature": 0.2,
    }
    base.update(overrides)
    return base


def test_node_metadata():
    assert LLMChatNode.NODE_NAME == "LLMChat"
    assert LLMChatNode.CATEGORY == "LLM"
    assert LLMChatNode.cacheable is False

    inputs = {p.name: p for p in LLMChatNode.define_inputs()}
    assert inputs["text"].data_type == DataType.STRING
    assert inputs["image"].data_type == DataType.IMAGE
    assert inputs["array"].data_type == DataType.LIST
    assert all(p.optional for p in inputs.values())

    outputs = LLMChatNode.define_outputs()
    assert len(outputs) == 1
    assert outputs[0].name == "text"
    assert outputs[0].data_type == DataType.STRING

    provider_param = next(p for p in LLMChatNode.define_params() if p.name == "provider")
    assert provider_param.param_type == ParamType.SELECT
    assert provider_param.options == ["ChatGPT API", "Codex", "Claude API", "Ollama"]


def test_provider_aliases():
    assert _normalize_provider("ChatGPT API") == "openai"
    assert _normalize_provider("Codex") == "openai-codex"
    assert _normalize_provider("Claude API") == "anthropic"
    assert _normalize_provider("Ollama") == "custom"
    with pytest.raises(ValueError, match="Unsupported"):
        _normalize_provider("Bedrock")


def test_execute_builds_openai_request(monkeypatch):
    seen = {}

    async def fake_collect(req, adapter, progress_callback=None):
        seen["req"] = req
        return "assistant text", {"input_tokens": 3, "output_tokens": 4}

    monkeypatch.setattr(llm_chat_node, "_collect_chat", fake_collect)
    res = LLMChatNode().execute(
        {"text": "hello", "array": [{"x": 1}]},
        params(),
    )

    assert res == {
        "text": "assistant text",
        "__usage__": {"input_tokens": 3, "output_tokens": 4},
    }
    req = seen["req"]
    assert req.provider == "openai"
    assert req.api_key == "sk-test"
    assert req.model == "gpt-5.2"
    assert req.max_tokens == 128
    assert req.temperature == 0.2
    assert req.messages[0].content == "System"
    user_content = req.messages[1].content
    assert isinstance(user_content, str)
    assert "Prompt" in user_content
    assert "hello" in user_content
    assert '"x": 1' in user_content


def test_ollama_maps_to_custom_provider(monkeypatch):
    seen = {}

    async def fake_collect(req, adapter, progress_callback=None):
        seen["req"] = req
        return "local", {}

    monkeypatch.setattr(llm_chat_node, "_collect_chat", fake_collect)
    res = LLMChatNode().execute(
        {},
        params(
            provider="Ollama",
            model="llama3.2",
            prompt="hi",
            system_prompt="",
            ollama_base_url="http://localhost:11434/v1/",
        ),
    )

    assert res == {"text": "local"}
    req = seen["req"]
    assert req.provider == "custom"
    assert req.base_url == "http://localhost:11434/v1/"
    assert req.api_key is None


def test_image_tensor_becomes_multimodal_content():
    req = _build_request(
        "openai",
        {"image": torch.zeros(3, 2, 2)},
        params(prompt="Describe it"),
    )

    content = req.messages[-1].content
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "Describe it"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_codex_image_input_falls_back_to_text():
    req = _build_request(
        "openai-codex",
        {"image": torch.zeros(3, 2, 2)},
        params(provider="Codex", prompt="Look"),
    )

    content = req.messages[-1].content
    assert isinstance(content, str)
    assert "Look" in content
    assert "text only" in content


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("CODEFYUI_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="requires openai_api_key"):
        _build_request("openai", {}, params(openai_api_key="", prompt="hi"))