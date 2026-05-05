"""Tests for WebSocket execution endpoint."""

import json

import pytest
from httpx import ASGITransport, AsyncClient
from httpx_ws import aconnect_ws
from httpx_ws.transport import ASGIWebSocketTransport

from app.main import app


@pytest.mark.asyncio
async def test_ws_connect_and_execute():
    """Test that we can connect via WS and execute a simple graph."""
    async with AsyncClient(
        transport=ASGIWebSocketTransport(app=app),
        base_url="http://test",
    ) as client:
        async with aconnect_ws("/ws/execution", client) as ws:
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
        base_url="http://test",
    ) as client:
        async with aconnect_ws("/ws/execution", client) as ws:
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
        base_url="http://test",
    ) as client:
        # Origin matches the synthetic Host ("test") that ASGI assigns.
        async with aconnect_ws(
            "/ws/execution",
            client,
            headers={"origin": "http://test"},
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
        base_url="http://test",
    ) as client:
        async with aconnect_ws(
            "/ws/execution",
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
        base_url="http://test",
    ) as client:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            async with aconnect_ws(
                "/ws/execution",
                client,
                headers={"origin": "http://attacker.example"},
            ) as _ws:
                pass  # connection should fail before we get here
        # 4003 is what ws_execution.py uses for "Origin not allowed".
        assert exc_info.value.code == 4003
