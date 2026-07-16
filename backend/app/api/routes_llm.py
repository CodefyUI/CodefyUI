"""Generic LLM provider proxy: unified streaming chat, model listing, and
the ChatGPT-account (codex) sign-in flow.

API keys arrive per-request from the browser, are forwarded upstream, and
are never logged or persisted server-side. Upstream hosts are fixed per
provider; only "custom" honors a user-supplied base URL.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..core.llm_proxy import anthropic, codex, codex_auth, codex_models, openai_like
from ..core.llm_proxy.events import error_event, sse_format
from ..core.llm_proxy.schema import ChatRequest, Provider

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/llm", tags=["llm"])

_KEY_PROVIDERS = {"openai", "openrouter", "anthropic"}

_ADAPTERS = {
    "openai": openai_like.stream_chat,
    "openrouter": openai_like.stream_chat,
    "custom": openai_like.stream_chat,
    "anthropic": anthropic.stream_chat,
    "openai-codex": codex.stream_chat,
}


def _client_factory() -> httpx.AsyncClient:
    """Indirection point so tests can swap in a MockTransport client."""
    return httpx.AsyncClient()


def _validate(req: ChatRequest) -> None:
    if req.provider in _KEY_PROVIDERS and not req.api_key:
        raise HTTPException(400, f"provider '{req.provider}' requires api_key")
    if req.provider == "custom" and not (req.base_url or "").startswith(("http://", "https://")):
        raise HTTPException(400, "custom provider requires an http(s) base_url")
    if req.provider == "openai-codex" and codex_auth.status()["status"] != "logged_in":
        raise HTTPException(400, "openai-codex requires ChatGPT sign-in - open Settings to sign in")
    if req.provider == "openai-codex" and req.reasoning_effort == "ultra":
        raise HTTPException(
            400,
            "reasoning_effort 'ultra' requires Codex multi-agent orchestration "
            "and is not supported by this proxy",
        )


@router.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    _validate(req)
    adapter = _ADAPTERS[req.provider]
    logger.info("llm chat: provider=%s model=%s messages=%d",
                req.provider, req.model, len(req.messages))

    async def gen() -> AsyncIterator[str]:
        client = _client_factory()
        try:
            async for event in adapter(req, client):
                yield sse_format(event)
        except Exception as exc:  # adapter bug must surface, not hang the stream
            logger.exception("llm chat stream crashed")
            yield sse_format(error_event(f"proxy error: {exc}"))
        finally:
            await client.aclose()

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "cache-control": "no-store",
        "x-accel-buffering": "no",
    })


class ModelsRequest(BaseModel):
    provider: Provider
    api_key: str | None = None
    base_url: str | None = None


_MODEL_ENDPOINTS = {
    "openai": ("https://api.openai.com/v1/models", "bearer"),
    "openrouter": ("https://openrouter.ai/api/v1/models", "bearer"),
    "anthropic": ("https://api.anthropic.com/v1/models", "x-api-key"),
}


def _model_capabilities(provider: Provider) -> dict[str, bool]:
    """Advertise only fields this proxy intentionally routes for a provider."""
    return {
        "reasoning_effort": provider in {"openai", "openai-codex"},
        # The Codex catalog is the only provider catalog currently enriched
        # with display names and effort metadata; other providers retain the
        # legacy id-only model shape.
        "rich_model_catalog": provider == "openai-codex",
    }


@router.post("/models")
async def list_models(req: ModelsRequest) -> dict:
    if req.provider == "openai-codex":
        client = _client_factory()
        try:
            result = await codex_models.list_models(client)
            result["capabilities"] = _model_capabilities(req.provider)
            return result
        finally:
            await client.aclose()

    if req.provider == "custom":
        base = (req.base_url or "").strip().rstrip("/")
        if not base.startswith(("http://", "https://")):
            raise HTTPException(400, "custom provider requires an http(s) base_url")
        url, auth_kind = f"{base}/models", "bearer"
    else:
        url, auth_kind = _MODEL_ENDPOINTS[req.provider]

    headers = {}
    if req.api_key:
        if auth_kind == "bearer":
            headers["authorization"] = f"Bearer {req.api_key}"
        else:
            headers["x-api-key"] = req.api_key
            headers["anthropic-version"] = "2023-06-01"

    client = _client_factory()
    try:
        resp = await client.get(url, headers=headers, timeout=20.0)
        if resp.status_code != 200:
            raise HTTPException(502, f"upstream {resp.status_code} from {req.provider}")
        data = resp.json().get("data", [])
        models = sorted({m.get("id", "") for m in data if isinstance(m, dict)} - {""})
        return {
            "models": [{"id": m} for m in models],
            "capabilities": _model_capabilities(req.provider),
        }
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"could not reach {req.provider}: {exc}") from exc
    finally:
        await client.aclose()


@router.post("/codex/login")
async def codex_login() -> dict:
    auth_url = await codex_auth.start_login()
    return {"auth_url": auth_url}


@router.get("/codex/status")
async def codex_status() -> dict:
    return codex_auth.status()


@router.post("/codex/logout")
async def codex_logout() -> dict:
    codex_auth.logout()
    codex_models.clear_cache()
    return {"status": "logged_out"}
