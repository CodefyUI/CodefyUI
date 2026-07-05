"""API-key helpers + the three Stage-2 auth dependencies (spec Section 7).

Unit half (dependencies exercised via a minimal starlette Request whose
.app is the real app); Task 4 appends the /api/keys endpoint tests.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any

import pytest
from fastapi import HTTPException, Request
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.core.api_keys import (
    KEY_PREFIX,
    PREFIX_DISPLAY_CHARS,
    hash_token,
    mint_token,
    require_api_key,
    require_api_key_or_session,
    require_session_token,
)
from app.core.auth import TOKEN_HEADER, session_token
from app.core.db import utc_now_iso
from app.main import app


def _fake_request(headers: dict[str, str] | None = None) -> Request:
    """A minimal Request whose .app is the real app, so dependencies can
    reach the app.state.db set by the app_db fixture."""
    encoded = [
        (k.lower().encode("latin-1"), v.encode("latin-1"))
        for k, v in (headers or {}).items()
    ]
    return Request({
        "type": "http",
        "method": "POST",
        "path": "/api/apps/x/invoke",
        "headers": encoded,
        "query_string": b"",
        "app": app,
    })


async def _insert_key(db, token: str, *, name: str = "unit",
                      revoked_at: str | None = None) -> int:
    token_hash = hash_token(token)
    prefix = token[:PREFIX_DISPLAY_CHARS]
    now = utc_now_iso()

    def _ins(conn: sqlite3.Connection) -> int:
        cur = conn.execute(
            "INSERT INTO api_keys (name, prefix, token_hash, created_at, "
            "revoked_at) VALUES (?, ?, ?, ?, ?)",
            (name, prefix, token_hash, now, revoked_at),
        )
        return cur.lastrowid

    return await db.run(_ins)


def test_mint_token_format():
    token = mint_token()
    assert token.startswith(KEY_PREFIX)
    assert len(token) >= len(KEY_PREFIX) + 40  # 32 CSPRNG bytes -> 43 chars
    assert mint_token() != token


def test_hash_token_is_sha256_hex_of_full_token():
    token = "cdui_example"
    assert hash_token(token) == hashlib.sha256(
        token.encode("utf-8")).hexdigest()
    assert len(hash_token(token)) == 64


@pytest.mark.asyncio
async def test_require_api_key_missing_or_malformed(app_db):
    for headers in (
        None,
        {"Authorization": "Basic abc"},
        {"Authorization": "Bearer "},
        {"Authorization": "Bearer"},
    ):
        result = await require_api_key(_fake_request(headers))
        assert not result.ok
        assert result.failure == (
            "missing or malformed Authorization: Bearer header"
        )


@pytest.mark.asyncio
async def test_require_api_key_non_cdui_bearer_self_diagnoses(app_db):
    # The expected failure mode: someone pastes the editor session token.
    result = await require_api_key(
        _fake_request({"Authorization": f"Bearer {session_token()}"})
    )
    assert not result.ok
    assert result.failure == (
        "this endpoint takes an API key (cdui_...), "
        "not the editor session token"
    )


@pytest.mark.asyncio
async def test_require_api_key_unknown_and_revoked(app_db):
    result = await require_api_key(
        _fake_request({"Authorization": "Bearer cdui_neverminted"})
    )
    assert not result.ok
    assert result.failure == "unknown or revoked API key"

    token = mint_token()
    await _insert_key(app_db, token, revoked_at=utc_now_iso())
    result = await require_api_key(
        _fake_request({"Authorization": f"Bearer {token}"})
    )
    assert not result.ok
    assert result.failure == "unknown or revoked API key"


@pytest.mark.asyncio
async def test_require_api_key_valid_returns_row_and_touches_last_used(app_db):
    token = mint_token()
    key_id = await _insert_key(app_db, token)
    result = await require_api_key(
        _fake_request({"Authorization": f"Bearer {token}"})
    )
    assert result.ok
    assert result.failure is None
    assert result.key_row["id"] == key_id
    assert result.key_row["name"] == "unit"

    def _last_used(conn: sqlite3.Connection) -> str | None:
        return conn.execute(
            "SELECT last_used_at FROM api_keys WHERE id = ?", (key_id,)
        ).fetchone()[0]

    assert await app_db.run(_last_used) is not None


@pytest.mark.asyncio
async def test_require_api_key_503_when_db_absent():
    if hasattr(app.state, "db"):
        delattr(app.state, "db")
    with pytest.raises(HTTPException) as exc_info:
        await require_api_key(
            _fake_request({"Authorization": "Bearer cdui_x"})
        )
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "database unavailable"


@pytest.mark.asyncio
async def test_require_session_token_shapes(app_db):
    # Valid: returns silently.
    await require_session_token(_fake_request({TOKEN_HEADER: session_token()}))
    # Invalid / missing: the exact auth_guard 403 shape.
    for headers in ({TOKEN_HEADER: "wrong"}, None):
        with pytest.raises(HTTPException) as exc_info:
            await require_session_token(_fake_request(headers))
        assert exc_info.value.status_code == 403
        assert exc_info.value.detail == (
            f"Missing or invalid {TOKEN_HEADER} header"
        )


@pytest.mark.asyncio
async def test_require_api_key_or_session_accepts_either_rejects_neither(app_db):
    token = mint_token()
    await _insert_key(app_db, token)
    # Session token alone works (Stage 3's editor UI reads runs with it).
    await require_api_key_or_session(
        _fake_request({TOKEN_HEADER: session_token()})
    )
    # API key alone works.
    await require_api_key_or_session(
        _fake_request({"Authorization": f"Bearer {token}"})
    )
    # Neither -> plain 401.
    with pytest.raises(HTTPException) as exc_info:
        await require_api_key_or_session(_fake_request())
    assert exc_info.value.status_code == 401


# ── /api/keys endpoints (Task 4) ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_key_returns_token_once_and_stores_only_hash(
    test_client, app_db,
):
    resp = await test_client.post("/api/keys", json={"name": "ci-bot"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body.keys()) == {"id", "name", "prefix", "token"}
    assert body["name"] == "ci-bot"
    assert body["token"].startswith(KEY_PREFIX)
    assert body["prefix"] == body["token"][:PREFIX_DISPLAY_CHARS]

    # The row stores the hash only.
    def _row(conn: sqlite3.Connection) -> dict[str, Any]:
        return dict(conn.execute(
            "SELECT * FROM api_keys WHERE id = ?", (body["id"],)
        ).fetchone())

    row = await app_db.run(_row)
    assert row["token_hash"] == hash_token(body["token"])
    assert body["token"] not in json.dumps(row)

    # ...and the plaintext appears nowhere in the DB file (+ WAL sidecar).
    await app_db.run(
        lambda conn: conn.execute("PRAGMA wal_checkpoint(FULL)").fetchone())
    blob = app_db.path.read_bytes()
    wal = app_db.path.with_name(app_db.path.name + "-wal")
    if wal.exists():
        blob += wal.read_bytes()
    assert body["token"].encode("ascii") not in blob


@pytest.mark.asyncio
async def test_list_keys_no_secrets_newest_first(test_client, app_db):
    first = (await test_client.post("/api/keys", json={"name": "a"})).json()
    second = (await test_client.post("/api/keys", json={"name": "b"})).json()
    resp = await test_client.get("/api/keys")
    assert resp.status_code == 200
    rows = resp.json()
    assert [r["id"] for r in rows] == [second["id"], first["id"]]
    for row in rows:
        assert set(row.keys()) == {
            "id", "name", "prefix", "created_at", "last_used_at", "revoked_at",
        }


@pytest.mark.asyncio
async def test_revoke_is_soft_and_row_stays_listed(test_client, app_db):
    created = (await test_client.post("/api/keys", json={"name": "r"})).json()
    resp = await test_client.post(f"/api/keys/{created['id']}/revoke")
    assert resp.status_code == 200
    assert resp.json()["revoked_at"] is not None

    rows = (await test_client.get("/api/keys")).json()
    assert len(rows) == 1                     # deliberately not DELETE
    assert rows[0]["revoked_at"] is not None

    # A revoked key immediately fails auth.
    result = await require_api_key(
        _fake_request({"Authorization": f"Bearer {created['token']}"})
    )
    assert not result.ok


@pytest.mark.asyncio
async def test_revoke_unknown_id_404(test_client, app_db):
    resp = await test_client.post("/api/keys/9999/revoke")
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "key_not_found"


@pytest.mark.asyncio
async def test_create_key_empty_name_422(test_client, app_db):
    resp = await test_client.post("/api/keys", json={"name": ""})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_keys_routes_require_session_token(app_db):
    # THE auth_guard GET gap this design closes: without the route-level
    # dependency, GET /api/keys would be world-readable (the middleware
    # never covered GETs, and now exempts this prefix entirely).
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url=f"http://127.0.0.1:{settings.PORT}",
    ) as anon:
        assert (await anon.get("/api/keys")).status_code == 403
        assert (await anon.post(
            "/api/keys", json={"name": "x"})).status_code == 403
        assert (await anon.post("/api/keys/1/revoke")).status_code == 403


@pytest.mark.asyncio
async def test_keys_routes_503_without_db(test_client):
    if hasattr(app.state, "db"):
        delattr(app.state, "db")
    resp = await test_client.get("/api/keys")
    assert resp.status_code == 503
    assert resp.json() == {"detail": "database unavailable"}
