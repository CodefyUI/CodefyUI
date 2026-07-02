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
