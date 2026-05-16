"""Tests for WebSocket execution endpoint."""

import json

import pytest
from httpx import AsyncClient
from httpx_ws import aconnect_ws
from httpx_ws.transport import ASGIWebSocketTransport

from app.config import settings
from app.core.auth import TOKEN_QUERY_PARAM, session_token
from app.main import app

# Host that's in the production whitelist (see ``init_allowed_hosts`` in
# ``app.core.auth``). Tests use this everywhere so the host_guard middleware
# accepts the request — production rejects ``Host: test`` to close the
# DNS-rebinding hole, but the test transport doesn't go through DNS.
_BASE_URL = f"http://127.0.0.1:{settings.PORT}"
_WS_PATH_WITH_TOKEN = f"/ws/execution?{TOKEN_QUERY_PARAM}={session_token()}"


@pytest.mark.asyncio
async def test_ws_connect_and_execute():
    """Test that we can connect via WS and execute a simple graph."""
    async with AsyncClient(
        transport=ASGIWebSocketTransport(app=app),
        base_url=_BASE_URL,
    ) as client:
        async with aconnect_ws(_WS_PATH_WITH_TOKEN, client) as ws:
            # Send execute with Start -> _TestSource -> Print
            await ws.send_text(json.dumps({
                "action": "execute",
                "nodes": [
                    {"id": "start", "type": "Start", "data": {"params": {}}},
                    {"id": "1", "type": "_TestSource", "data": {"params": {}}},
                    {"id": "2", "type": "Print", "data": {"params": {"label": "b"}}},
                ],
                "edges": [
                    {"id": "et", "source": "start", "target": "1", "sourceHandle": "trigger", "type": "trigger"},
                    {"source": "1", "target": "2", "sourceHandle": "value", "targetHandle": "value"},
                ],
            }))

            # Collect messages until execution_complete or error
            messages = []
            for _ in range(20):
                msg = json.loads(await ws.receive_text())
                messages.append(msg)
                if msg["type"] in ("execution_complete", "execution_error"):
                    break

            types = [m["type"] for m in messages]
            assert "execution_start" in types
            assert "execution_complete" in types


@pytest.mark.asyncio
async def test_ws_unknown_action():
    """Unknown actions should return an error message."""
    async with AsyncClient(
        transport=ASGIWebSocketTransport(app=app),
        base_url=_BASE_URL,
    ) as client:
        async with aconnect_ws(_WS_PATH_WITH_TOKEN, client) as ws:
            await ws.send_text(json.dumps({"action": "foobar"}))
            msg = json.loads(await ws.receive_text())
            assert msg["type"] == "error"
            assert "foobar" in msg["error"]


# ── Origin policy ──────────────────────────────────────────────────────
# Regression guards for the WS handshake's same-origin / CORS_ORIGINS gate
# (see ws_execution.websocket_execution comment). The bug these protect
# against: end users on `cdui start` (single uvicorn at :8000) hit
# 403 Forbidden because the default CORS_ORIGINS only listed Vite dev
# ports (5173 / 5174 / 3000) and rejected the same-origin connection.


@pytest.mark.asyncio
async def test_ws_same_origin_allowed():
    """Same-origin connections must be accepted regardless of CORS_ORIGINS.

    This is the `cdui start` case — SPA and WS share host:port so Origin
    matches Host, and the handshake should succeed without any explicit
    allowlist entry for the production port.
    """
    async with AsyncClient(
        transport=ASGIWebSocketTransport(app=app),
        base_url=_BASE_URL,
    ) as client:
        async with aconnect_ws(
            _WS_PATH_WITH_TOKEN,
            client,
            headers={"origin": _BASE_URL},
        ) as ws:
            await ws.send_text(json.dumps({"action": "foobar"}))
            msg = json.loads(await ws.receive_text())
            assert msg["type"] == "error"  # handshake succeeded


@pytest.mark.asyncio
async def test_ws_cross_origin_in_allowlist_allowed():
    """Cross-origin from a CORS_ORIGINS entry must be accepted.

    Covers `cdui dev` — Vite at :5173 hitting the backend at :8000.
    """
    async with AsyncClient(
        transport=ASGIWebSocketTransport(app=app),
        base_url=_BASE_URL,
    ) as client:
        async with aconnect_ws(
            _WS_PATH_WITH_TOKEN,
            client,
            headers={"origin": "http://localhost:5173"},
        ) as ws:
            await ws.send_text(json.dumps({"action": "foobar"}))
            msg = json.loads(await ws.receive_text())
            assert msg["type"] == "error"


@pytest.mark.asyncio
async def test_ws_cross_origin_not_in_allowlist_rejected():
    """Cross-origin not in CORS_ORIGINS must be rejected at handshake.

    The ASGI test transport surfaces a close-before-accept as
    ``WebSocketDisconnect`` with our custom 4003 code (real browsers
    see uvicorn translate it into an HTTP 403 response).
    """
    from httpx_ws import WebSocketDisconnect

    async with AsyncClient(
        transport=ASGIWebSocketTransport(app=app),
        base_url=_BASE_URL,
    ) as client:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            async with aconnect_ws(
                _WS_PATH_WITH_TOKEN,
                client,
                headers={"origin": "http://attacker.example"},
            ) as _ws:
                pass  # connection should fail before we get here
        # 4003 is what ws_execution.py uses for "Origin not allowed".
        assert exc_info.value.code == 4003


# ── Token enforcement (new in security audit fixes) ─────────────────────


@pytest.mark.asyncio
async def test_ws_rejects_missing_token():
    """A connect without ?token=... must be closed with 4401."""
    from httpx_ws import WebSocketDisconnect

    async with AsyncClient(
        transport=ASGIWebSocketTransport(app=app),
        base_url=_BASE_URL,
    ) as client:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            async with aconnect_ws("/ws/execution", client) as _ws:
                pass
        assert exc_info.value.code == 4401


@pytest.mark.asyncio
async def test_ws_rejects_wrong_token():
    """A wrong token must be rejected with the same 4401."""
    from httpx_ws import WebSocketDisconnect

    async with AsyncClient(
        transport=ASGIWebSocketTransport(app=app),
        base_url=_BASE_URL,
    ) as client:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            async with aconnect_ws(
                f"/ws/execution?{TOKEN_QUERY_PARAM}=not-the-real-token",
                client,
            ) as _ws:
                pass
        assert exc_info.value.code == 4401


@pytest.mark.asyncio
async def test_ws_rejects_bad_host_header():
    """``Host: attacker.example`` must be rejected before token check.

    The 4003 code matches the Origin-rejected case; we still want it gone
    before any handshake state is built up.
    """
    from httpx_ws import WebSocketDisconnect

    async with AsyncClient(
        transport=ASGIWebSocketTransport(app=app),
        base_url="http://attacker.example",
    ) as client:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            async with aconnect_ws(_WS_PATH_WITH_TOKEN, client) as _ws:
                pass
        assert exc_info.value.code == 4003
