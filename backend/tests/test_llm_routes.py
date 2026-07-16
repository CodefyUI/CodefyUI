"""HTTP-surface tests for /api/llm/* (chat SSE relay, models, codex auth)."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api import routes_llm
from app.core.llm_proxy import codex_auth, codex_models


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path))
    from app.config import settings
    from app.core.auth import TOKEN_HEADER, session_token
    from app.main import app
    with TestClient(app, base_url=f"http://127.0.0.1:{settings.PORT}") as c:
        c.headers[TOKEN_HEADER] = session_token()
        yield c


def mock_upstream(monkeypatch, handler):
    def factory():
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(routes_llm, "_client_factory", factory)


def chat_body(**overrides):
    base = {
        "provider": "openai",
        "model": "gpt-5.2",
        "messages": [{"role": "user", "content": "hi"}],
        "api_key": "sk-x",
    }
    base.update(overrides)
    return base


def sse_events_from(resp_text: str) -> list[dict]:
    return [json.loads(line[len("data: "):])
            for line in resp_text.splitlines() if line.startswith("data: ")]


def test_chat_streams_sse(client, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            'data: {"choices": [{"delta": {"content": "yo"}, "finish_reason": "stop"}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, content=body.encode(),
                              headers={"content-type": "text/event-stream"})
    mock_upstream(monkeypatch, handler)

    with client.stream("POST", "/api/llm/chat", json=chat_body()) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = sse_events_from(r.read().decode())
    assert events[0] == {"type": "text_delta", "text": "yo"}
    assert events[-1]["type"] == "done"


def test_chat_requires_key_for_key_providers(client):
    r = client.post("/api/llm/chat", json=chat_body(api_key=None))
    assert r.status_code == 400
    assert "api_key" in r.json()["detail"]


def test_chat_requires_base_url_for_custom(client):
    r = client.post("/api/llm/chat", json=chat_body(provider="custom", api_key=None))
    assert r.status_code == 400


def test_chat_codex_requires_login(client):
    r = client.post("/api/llm/chat", json=chat_body(provider="openai-codex", api_key=None))
    assert r.status_code == 400
    assert "sign" in r.json()["detail"].lower()


def test_chat_rejects_missing_session_token(client):
    from app.config import settings
    from app.main import app
    with TestClient(app, base_url=f"http://127.0.0.1:{settings.PORT}") as bare:
        r = bare.post("/api/llm/chat", json=chat_body())
    assert r.status_code in (401, 403)


def test_models_openai(client, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.openai.com/v1/models"
        assert request.headers["authorization"] == "Bearer sk-x"
        return httpx.Response(200, json={"data": [{"id": "gpt-5.2"}, {"id": "gpt-4.1"}]})
    mock_upstream(monkeypatch, handler)

    r = client.post("/api/llm/models", json={"provider": "openai", "api_key": "sk-x"})
    assert r.status_code == 200
    assert {"id": "gpt-4.1"} in r.json()["models"]
    assert r.json()["capabilities"] == {
        "reasoning_effort": True,
        "rich_model_catalog": False,
    }


def test_models_codex_fallback_is_rich_and_backward_compatible(client):
    codex_models.clear_cache()
    r = client.post("/api/llm/models", json={"provider": "openai-codex"})
    assert r.status_code == 200
    body = r.json()
    assert any(model["id"] == "gpt-5.5" for model in body["models"])
    sol = next(model for model in body["models"] if model["id"] == "gpt-5.6-sol")
    assert sol["default_reasoning_effort"] == "low"
    assert {item["effort"] for item in sol["supported_reasoning_efforts"]} == {
        "low", "medium", "high", "xhigh", "max",
    }
    assert body["capabilities"] == {
        "reasoning_effort": True,
        "rich_model_catalog": True,
    }


@pytest.mark.parametrize("provider", ["openrouter", "anthropic", "custom"])
def test_models_non_forwarding_provider_capabilities_are_false(
    client, monkeypatch, provider
):
    mock_upstream(
        monkeypatch,
        lambda request: httpx.Response(200, json={"data": [{"id": "model-1"}]}),
    )
    body = {"provider": provider, "api_key": "key"}
    if provider == "custom":
        body.update({"api_key": None, "base_url": "http://127.0.0.1:11434/v1"})
    r = client.post("/api/llm/models", json=body)
    assert r.status_code == 200
    assert r.json()["capabilities"] == {
        "reasoning_effort": False,
        "rich_model_catalog": False,
    }


def test_chat_rejects_codex_ultra_even_when_logged_in(client, monkeypatch):
    monkeypatch.setattr(codex_auth, "status", lambda: {"status": "logged_in"})
    r = client.post(
        "/api/llm/chat",
        json=chat_body(
            provider="openai-codex",
            model="gpt-5.6-sol",
            api_key=None,
            reasoning_effort="ultra",
        ),
    )
    assert r.status_code == 400
    assert "multi-agent orchestration" in r.json()["detail"]


def test_models_upstream_error_becomes_502(client, monkeypatch):
    mock_upstream(monkeypatch, lambda request: httpx.Response(401, json={}))
    r = client.post("/api/llm/models", json={"provider": "openai", "api_key": "bad"})
    assert r.status_code == 502


def test_codex_status_and_logout(client):
    r = client.get("/api/llm/codex/status")
    assert r.json() == {"status": "logged_out"}
    r = client.post("/api/llm/codex/logout")
    assert r.json() == {"status": "logged_out"}


def test_chat_adapter_crash_becomes_terminal_error_event(client, monkeypatch):
    async def exploding_adapter(req, http_client):
        yield {"type": "text_delta", "text": "partial"}
        raise RuntimeError("adapter bug")

    monkeypatch.setitem(routes_llm._ADAPTERS, "openai", exploding_adapter)

    with client.stream("POST", "/api/llm/chat", json=chat_body()) as r:
        assert r.status_code == 200
        events = sse_events_from(r.read().decode())
    assert events[0] == {"type": "text_delta", "text": "partial"}
    assert events[-1]["type"] == "error"
    assert "proxy error" in events[-1]["message"]
    assert "adapter bug" in events[-1]["message"]
