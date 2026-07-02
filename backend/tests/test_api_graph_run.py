"""Tests for POST /api/graph/run/{name} and GET /api/graph/contract/{name}."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.core.node_base import BaseNode, DataType, PortDefinition
from app.core.run_output_store import RunOutputStore
from app.main import app

ENVELOPE_KEYS = {"status", "run_id", "graph", "device", "outputs", "error", "timing"}


# ── config: body cap setting ─────────────────────────────────────────────


def test_max_run_body_bytes_default():
    assert settings.MAX_RUN_BODY_BYTES == 64 * 1024 * 1024


def test_max_run_body_bytes_env_override(monkeypatch):
    monkeypatch.setenv("CODEFYUI_MAX_RUN_BODY_BYTES", "1024")
    from app.config import Settings

    assert Settings().MAX_RUN_BODY_BYTES == 1024


# ── envelope builders ────────────────────────────────────────────────

from app.api.routes_graph_run import build_envelope, error_response  # noqa: E402


def test_build_envelope_has_all_keys_with_nulls():
    env = build_envelope(status="ok", run_id="r1", graph="g", outputs={"y": 1})
    assert set(env.keys()) == ENVELOPE_KEYS
    assert env["status"] == "ok"
    assert env["run_id"] == "r1"
    assert env["graph"] == "g"
    assert env["device"] is None
    assert env["outputs"] == {"y": 1}
    assert env["error"] is None
    assert env["timing"] is None


def test_error_response_mirrors_status_and_keeps_all_keys():
    resp = error_response(
        409, run_id="r2", graph="g", code="invalid_contract",
        message="broken", details=["p1", "p2"],
    )
    assert resp.status_code == 409
    body = json.loads(resp.body)
    assert set(body.keys()) == ENVELOPE_KEYS
    assert body["status"] == "error"
    assert body["run_id"] == "r2"
    assert body["outputs"] is None
    assert body["error"] == {
        "code": "invalid_contract", "message": "broken",
        "node_id": None, "details": ["p1", "p2"],
    }
    assert body["timing"] is None


def test_error_response_carries_device_node_id_timing_when_known():
    resp = error_response(
        500, run_id="r3", graph="g", code="execution_error",
        message="node blew up", device="cpu", node_id="n7",
        timing={"total_s": 1.02},
    )
    body = json.loads(resp.body)
    assert body["device"] == "cpu"
    assert body["error"]["node_id"] == "n7"
    assert body["timing"] == {"total_s": 1.02}
