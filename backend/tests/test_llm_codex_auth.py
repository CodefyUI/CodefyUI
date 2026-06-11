"""Codex ChatGPT-OAuth tests: PKCE, JWT claims, storage, refresh, login flow."""

from __future__ import annotations

import base64
import hashlib
import json
import time
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from app.core.llm_proxy import codex_auth


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path))
    codex_auth.cancel_login()
    yield
    codex_auth.cancel_login()


def make_jwt(claims: dict) -> str:
    def b64(part: dict) -> str:
        raw = json.dumps(part).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
    return f"{b64({'alg': 'none'})}.{b64(claims)}.sig"


def seeded_tokens(*, exp_offset: int = 3600, account_id: str = "acc-1") -> dict:
    access = make_jwt({"exp": int(time.time()) + exp_offset})
    id_token = make_jwt({
        "email": "u@example.com",
        "https://api.openai.com/auth": {"chatgpt_account_id": account_id},
    })
    return {
        "id_token": id_token,
        "access_token": access,
        "refresh_token": "rt-1",
        "account_id": account_id,
        "last_refresh": codex_auth.now_iso(),
    }


# -- PKCE / JWT helpers ------------------------------------------------------

def test_pkce_pair_shapes():
    verifier, challenge = codex_auth.generate_pkce()
    assert len(verifier) == 86  # 64 random bytes, base64url no-pad
    expected = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    assert challenge == expected


def test_jwt_claim_extraction():
    tok = make_jwt({"email": "x@y.z",
                    "https://api.openai.com/auth": {"chatgpt_account_id": "acc-9"}})
    assert codex_auth.account_id_from_id_token(tok) == "acc-9"
    assert codex_auth.email_from_id_token(tok) == "x@y.z"
    assert codex_auth.account_id_from_id_token("garbage") is None


def test_authorize_url_contains_required_params():
    url = codex_auth.build_authorize_url(port=1455, state="st", challenge="ch")
    q = parse_qs(urlparse(url).query)
    assert urlparse(url).netloc == "auth.openai.com"
    assert q["client_id"] == [codex_auth.CLIENT_ID]
    assert q["response_type"] == ["code"]
    assert q["redirect_uri"] == ["http://localhost:1455/auth/callback"]
    assert q["code_challenge"] == ["ch"]
    assert q["code_challenge_method"] == ["S256"]
    assert q["state"] == ["st"]
    assert q["originator"] == ["codex_cli_rs"]
    assert q["codex_cli_simplified_flow"] == ["true"]
    assert "openid" in q["scope"][0]


# -- storage -----------------------------------------------------------------

def test_store_roundtrip_and_logout():
    assert codex_auth.load_tokens() is None
    codex_auth.save_tokens(seeded_tokens())
    loaded = codex_auth.load_tokens()
    assert loaded["account_id"] == "acc-1"
    assert codex_auth.status()["status"] == "logged_in"
    assert codex_auth.status()["email"] == "u@example.com"
    codex_auth.logout()
    assert codex_auth.load_tokens() is None
    assert codex_auth.status()["status"] == "logged_out"


# -- refresh logic ------------------------------------------------------------

def test_needs_refresh_on_expiry_and_age():
    fresh = seeded_tokens(exp_offset=3600)
    assert codex_auth.needs_refresh(fresh) is False
    expiring = seeded_tokens(exp_offset=60)  # < 5-minute window
    assert codex_auth.needs_refresh(expiring) is True
    old = seeded_tokens(exp_offset=3600)
    old["last_refresh"] = "2020-01-01T00:00:00+00:00"  # > 8 days ago
    assert codex_auth.needs_refresh(old) is True


@pytest.mark.asyncio
async def test_refresh_preserves_account_id_and_shape():
    codex_auth.save_tokens(seeded_tokens(exp_offset=10))
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        # Post-refresh access tokens omit the account-id claim (real-world
        # behavior) -- account_id must survive from the stored value.
        return httpx.Response(200, json={
            "access_token": make_jwt({"exp": int(time.time()) + 7200}),
            "refresh_token": "rt-2",
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        access, account = await codex_auth.get_valid_access(client)

    assert seen["url"] == "https://auth.openai.com/oauth/token"
    assert seen["body"]["grant_type"] == "refresh_token"
    assert seen["body"]["client_id"] == codex_auth.CLIENT_ID
    assert seen["body"]["refresh_token"] == "rt-1"
    assert account == "acc-1"
    stored = codex_auth.load_tokens()
    assert stored["refresh_token"] == "rt-2"
    assert stored["account_id"] == "acc-1"


@pytest.mark.asyncio
async def test_get_valid_access_raises_when_logged_out():
    async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(500))) as client:
        with pytest.raises(codex_auth.CodexNotLoggedIn):
            await codex_auth.get_valid_access(client)


# -- full login flow over a real localhost socket ------------------------------

@pytest.mark.asyncio
async def test_login_flow_end_to_end():
    def token_handler(request: httpx.Request) -> httpx.Response:
        body = dict(pair.split("=", 1) for pair in request.content.decode().split("&"))
        assert body["grant_type"] == "authorization_code"
        assert body["code"] == "the-code"
        return httpx.Response(200, json={
            "id_token": make_jwt({
                "email": "u@example.com",
                "https://api.openai.com/auth": {"chatgpt_account_id": "acc-7"},
            }),
            "access_token": make_jwt({"exp": int(time.time()) + 3600}),
            "refresh_token": "rt-7",
        })

    exchange_client = httpx.AsyncClient(transport=httpx.MockTransport(token_handler))
    auth_url = await codex_auth.start_login(ports=(0,), exchange_client=exchange_client)
    q = parse_qs(urlparse(auth_url).query)
    state = q["state"][0]
    port = codex_auth.active_login_port()
    assert port and q["redirect_uri"] == [f"http://localhost:{port}/auth/callback"]
    assert codex_auth.status()["status"] == "pending"

    async with httpx.AsyncClient() as browser:
        r = await browser.get(
            f"http://127.0.0.1:{port}/auth/callback",
            params={"code": "the-code", "state": state})
    assert r.status_code == 200
    assert "close this tab" in r.text

    await codex_auth.wait_login_settled()
    st = codex_auth.status()
    assert st["status"] == "logged_in"
    assert st["email"] == "u@example.com"
    assert codex_auth.load_tokens()["account_id"] == "acc-7"


@pytest.mark.asyncio
async def test_login_flow_rejects_bad_state():
    exchange_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    await codex_auth.start_login(ports=(0,), exchange_client=exchange_client)
    port = codex_auth.active_login_port()

    async with httpx.AsyncClient() as browser:
        r = await browser.get(
            f"http://127.0.0.1:{port}/auth/callback",
            params={"code": "x", "state": "WRONG"})
    assert r.status_code == 400
    assert codex_auth.status()["status"] == "pending"  # flow still waiting
