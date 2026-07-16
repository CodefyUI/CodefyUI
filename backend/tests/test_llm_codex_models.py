"""Dynamic, account-isolated Codex model catalog tests."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from app.core.llm_proxy import codex_auth, codex_models


@pytest.fixture(autouse=True)
def reset_catalog_cache():
    codex_models.clear_cache()
    yield
    codex_models.clear_cache()


def _model(model_id: str, **overrides):
    model = {
        "slug": model_id,
        "display_name": model_id.upper(),
        "description": f"Description for {model_id}",
        "visibility": "list",
        "supported_in_api": True,
        "default_reasoning_level": "medium",
        "supported_reasoning_levels": [
            {"effort": "low", "description": "Low"},
            {"effort": "medium", "description": "Medium"},
        ],
    }
    model.update(overrides)
    return model


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_live_catalog_is_sanitized_and_uses_expected_request(monkeypatch):
    async def fake_access(client, *, force_refresh=False):
        return "access-secret", "account-1"

    monkeypatch.setattr(codex_auth, "get_valid_access", fake_access)
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        seen["timeout"] = request.extensions["timeout"]
        return httpx.Response(200, headers={"etag": '"catalog-v1"'}, json={"models": [
            _model(
                "gpt-5.6-sol",
                priority=1,
                default_reasoning_level="ultra",
                supported_reasoning_levels=[
                    {"effort": "low", "description": "Low"},
                    {"effort": "ultra", "description": "Orchestration"},
                    {"effort": "High", "description": "Uppercase"},
                    {"effort": "very high", "description": "Space"},
                    {"reasoning_effort": "future_effort", "description": "Future"},
                ],
                model_provider="secret-provider-field",
                system_instructions="do not expose this",
            ),
            _model("hidden-model", priority=2, visibility="hide"),
            _model("unsupported-model", priority=3, supported_in_api=False),
            _model("bad model id", priority=4),
            _model("gpt-5.6-sol", priority=5, description="duplicate"),
        ]})

    async with _client(handler) as client:
        result = await codex_models.list_models(client)

    assert seen["url"] == (
        "https://chatgpt.com/backend-api/codex/models?client_version=0.144.0"
    )
    assert seen["headers"]["authorization"] == "Bearer access-secret"
    assert seen["headers"]["chatgpt-account-id"] == "account-1"
    assert seen["headers"]["originator"] == codex_auth.ORIGINATOR
    assert set(seen["timeout"].values()) == {5.0}
    assert result["source"] == "live"
    assert result["stale"] is False
    assert len(result["models"]) == 1

    model = result["models"][0]
    assert set(model) == {
        "id",
        "display_name",
        "description",
        "default_reasoning_effort",
        "supported_reasoning_efforts",
    }
    assert model["id"] == "gpt-5.6-sol"
    assert model["default_reasoning_effort"] == "low"
    assert model["supported_reasoning_efforts"] == [
        {"effort": "low", "description": "Low"},
        {"effort": "future_effort", "description": "Future"},
    ]
    assert "system_instructions" not in model
    assert "model_provider" not in model
    assert "priority" not in model


@pytest.mark.asyncio
async def test_fresh_cache_is_reused_for_the_same_account(monkeypatch):
    async def fake_access(client, *, force_refresh=False):
        return "token-1", "account-1"

    monkeypatch.setattr(codex_auth, "get_valid_access", fake_access)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"models": [_model("model-one")]})

    async with _client(handler) as client:
        first = await codex_models.list_models(client)
        second = await codex_models.list_models(client)

    assert first["source"] == "live"
    assert second["source"] == "cache"
    assert calls == 1


@pytest.mark.asyncio
async def test_cache_is_isolated_by_account(monkeypatch):
    current = {"account": "account-a"}

    async def fake_access(client, *, force_refresh=False):
        account = current["account"]
        return f"token-{account}", account

    monkeypatch.setattr(codex_auth, "get_valid_access", fake_access)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        account = request.headers["chatgpt-account-id"]
        return httpx.Response(200, json={"models": [_model(f"model-{account}")]})

    async with _client(handler) as client:
        account_a = await codex_models.list_models(client)
        current["account"] = "account-b"
        account_b = await codex_models.list_models(client)

    assert account_a["models"][0]["id"] == "model-account-a"
    assert account_b["models"][0]["id"] == "model-account-b"
    assert calls == 2


@pytest.mark.asyncio
async def test_expired_cache_revalidates_with_etag(monkeypatch):
    clock = {"now": 0.0}
    monkeypatch.setattr(codex_models, "monotonic", lambda: clock["now"])

    async def fake_access(client, *, force_refresh=False):
        return "token-1", "account-1"

    monkeypatch.setattr(codex_auth, "get_valid_access", fake_access)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                headers={"etag": '"v1"'},
                json={"models": [_model("model-one")]},
            )
        assert request.headers["if-none-match"] == '"v1"'
        return httpx.Response(304)

    async with _client(handler) as client:
        first = await codex_models.list_models(client)
        clock["now"] = codex_models.CACHE_TTL_S + 1
        second = await codex_models.list_models(client)

    assert first["source"] == "live"
    assert second["source"] == "cache"
    assert second["models"] == first["models"]
    assert calls == 2


@pytest.mark.asyncio
async def test_expired_cache_is_returned_stale_on_timeout(monkeypatch):
    clock = {"now": 0.0}
    monkeypatch.setattr(codex_models, "monotonic", lambda: clock["now"])

    async def fake_access(client, *, force_refresh=False):
        return "token-1", "account-1"

    monkeypatch.setattr(codex_auth, "get_valid_access", fake_access)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(200, json={"models": [_model("model-one")]})
        raise httpx.ReadTimeout("catalog timeout", request=request)

    async with _client(handler) as client:
        first = await codex_models.list_models(client)
        clock["now"] = codex_models.CACHE_TTL_S + 1
        second = await codex_models.list_models(client)

    assert first["source"] == "live"
    assert second["source"] == "stale"
    assert second["stale"] is True
    assert second["models"] == first["models"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rotated_account_id", ["", "account-1"], ids=["token-key", "account-key"]
)
async def test_legacy_token_key_rotation_preserves_stale_cache_on_timeout(
    monkeypatch, rotated_account_id
):
    clock = {"now": 0.0}
    session = {
        "stored_token": "legacy-token",
        "stored_account_id": "",
        "access_calls": 0,
    }
    monkeypatch.setattr(codex_models, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        codex_auth,
        "load_tokens",
        lambda: {
            "access_token": session["stored_token"],
            "account_id": session["stored_account_id"],
        },
    )

    async def rotating_access(client, *, force_refresh=False):
        session["access_calls"] += 1
        if session["access_calls"] == 1:
            return "legacy-token", ""
        # Mirror an auth refresh: the persisted access token changes before
        # get_valid_access returns, and the refresh may also recover a stable
        # account id for the legacy session.
        session["stored_token"] = "rotated-token"
        session["stored_account_id"] = rotated_account_id
        return "rotated-token", rotated_account_id

    monkeypatch.setattr(codex_auth, "get_valid_access", rotating_access)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            assert request.headers["authorization"] == "Bearer legacy-token"
            return httpx.Response(200, json={"models": [_model("model-one")]})
        assert request.headers["authorization"] == "Bearer rotated-token"
        assert request.headers["chatgpt-account-id"] == rotated_account_id
        raise httpx.ReadTimeout("catalog timeout", request=request)

    async with _client(handler) as client:
        first = await codex_models.list_models(client)
        clock["now"] = codex_models.CACHE_TTL_S + 1
        second = await codex_models.list_models(client)
        # The persisted token is now the rotated value.  A third failed fetch
        # proves the stale entry was moved to that key instead of being used
        # only once through an old-key fallback.
        third = await codex_models.list_models(client)

    assert first["source"] == "live"
    assert second["source"] == "stale"
    assert second["stale"] is True
    assert second["models"] == first["models"]
    assert third["source"] == "stale"
    assert third["models"] == first["models"]
    assert calls == 3


@pytest.mark.asyncio
async def test_401_session_switch_does_not_move_legacy_cache_to_new_account(
    monkeypatch,
):
    clock = {"now": 0.0}
    session = {"generation": 1, "token": "a-token", "account_id": ""}
    monkeypatch.setattr(codex_models, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        codex_auth, "session_generation", lambda: session["generation"]
    )
    monkeypatch.setattr(
        codex_auth,
        "load_tokens",
        lambda: {
            "access_token": session["token"],
            "account_id": session["account_id"],
        },
    )

    async def switching_access(client, *, force_refresh=False):
        if force_refresh:
            session.update({
                "generation": 2,
                "token": "b-token",
                "account_id": "account-b",
            })
        return session["token"], session["account_id"]

    monkeypatch.setattr(codex_auth, "get_valid_access", switching_access)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(200, json={"models": [_model("model-account-a")]})
        if calls == 2:
            assert request.headers["authorization"] == "Bearer a-token"
            return httpx.Response(401, json={"detail": "expired"})
        assert request.headers["authorization"] == "Bearer b-token"
        assert request.headers["chatgpt-account-id"] == "account-b"
        raise httpx.ReadTimeout("account-b catalog timeout", request=request)

    async with _client(handler) as client:
        first = await codex_models.list_models(client)
        clock["now"] = codex_models.CACHE_TTL_S + 1
        switched = await codex_models.list_models(client)

    a_key = codex_models._cache_key("", "a-token")
    b_key = codex_models._cache_key("account-b", "b-token")
    assert first["source"] == "live"
    assert switched["source"] == "fallback"
    assert a_key in codex_models._CACHE
    assert b_key not in codex_models._CACHE
    assert calls == 3


@pytest.mark.asyncio
async def test_clear_cache_prevents_inflight_success_from_repopulating(monkeypatch):
    async def fake_access(client, *, force_refresh=False):
        return "token-1", "account-1"

    monkeypatch.setattr(codex_auth, "get_valid_access", fake_access)
    request_started = asyncio.Event()
    release_response = asyncio.Event()
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            request_started.set()
            await release_response.wait()
            return httpx.Response(200, json={"models": [_model("model-one")]})
        raise httpx.ReadTimeout("catalog timeout", request=request)

    async with _client(handler) as client:
        pending = asyncio.create_task(codex_models.list_models(client))
        await request_started.wait()
        codex_models.clear_cache()
        release_response.set()
        invalidated = await pending
        after_clear = await codex_models.list_models(client)

    assert invalidated["source"] == "fallback"
    assert after_clear["source"] == "fallback"
    assert codex_models._CACHE == {}
    assert calls == 2


@pytest.mark.asyncio
async def test_concurrent_logout_during_401_refresh_returns_fallback(monkeypatch):
    calls = 0

    async def fake_access(client, *, force_refresh=False):
        if force_refresh:
            raise codex_auth.CodexNotLoggedIn("logged out during refresh")
        return "expired-token", "account-1"

    monkeypatch.setattr(codex_auth, "get_valid_access", fake_access)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"detail": "expired"})

    async with _client(handler) as client:
        result = await codex_models.list_models(client)

    assert calls == 1
    assert result["source"] == "fallback"
    assert any(model["id"] == "gpt-5.6-sol" for model in result["models"])


@pytest.mark.asyncio
async def test_logged_out_fallback_includes_gpt_5_6_and_excludes_ultra(monkeypatch):
    async def not_logged_in(client, *, force_refresh=False):
        raise codex_auth.CodexNotLoggedIn("logged out")

    monkeypatch.setattr(codex_auth, "get_valid_access", not_logged_in)
    async with _client(lambda request: httpx.Response(500)) as client:
        result = await codex_models.list_models(client)

    by_id = {model["id"]: model for model in result["models"]}
    assert result["source"] == "fallback"
    assert {"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5"} <= set(by_id)
    for model_id in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"):
        efforts = {
            item["effort"] for item in by_id[model_id]["supported_reasoning_efforts"]
        }
        assert "ultra" not in efforts


def test_sanitized_catalog_is_bounded_to_500_models():
    payload = {"models": [_model(f"model-{index}") for index in range(505)]}
    models = codex_models._sanitize_catalog(payload)
    assert len(models) == 500
