"""Shared pytest fixtures for CodefyUI backend tests."""

import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

# Make scripts/ importable so CLI tests can `import plugins`.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from app.config import settings
from app.core.auth import TOKEN_HEADER, init_allowed_hosts, session_token
from app.core.node_base import BaseNode, DataType, PortDefinition
from app.core.node_registry import NodeRegistry, registry
from app.core.plugin_loader import install_plugin_finder, purge_all_plugin_modules
from app.core.preset_registry import preset_registry
from app.main import app

# Captured before the redirect below -- test_config_stage2.py asserts
# against this exact production value (see the fixture near the bottom of
# this file that hands it back for the duration of those tests only).
_DEFAULT_DB_PATH = settings.DB_PATH

# DB isolation: every test run's SQLite DB lives in a temp dir, never in
# backend/data/ (lifespan-driving TestClient tests would otherwise create a
# real codefyui.db there). Module-level on purpose: conftest import runs
# before any hook or fixture, so there is no ordering race.
settings.DB_PATH = Path(tempfile.mkdtemp(prefix="codefyui-test-db-")) / "codefyui-test.db"

# Tests use ``base_url="http://127.0.0.1:8000"`` which the production Host
# whitelist already accepts, but seed it explicitly here so tests don't rely
# on lifespan-time initialisation (lifespan runs once per app instance and
# ASGITransport doesn't always go through it).
init_allowed_hosts(settings.HOST, settings.PORT)

# Register the in-repo chapter plugin packs in the synthetic `cdui_plugins`
# namespace AT CONFTEST IMPORT TIME, before any test_*.py module is collected.
# Tests for Edu nodes import them from `cdui_plugins.{foundations,deep,rl}.nodes.*`
# — those imports happen during pytest's collection pass, which runs after conftest
# is imported, so the namespace must exist by then.
_REPO_ROOT = Path(__file__).resolve().parents[2]
purge_all_plugin_modules()
install_plugin_finder(
    builtin_root=_REPO_ROOT / "plugins",
    user_root=_REPO_ROOT / "_phantom_user_root_for_tests",  # never read
    lockfile={
        "schema": 1,
        "plugins": {
            "foundations": {"source_kind": "builtin", "source": "foundations"},
            "deep": {"source_kind": "builtin", "source": "deep"},
            "rl": {"source_kind": "builtin", "source": "rl"},
        },
    },
)


@pytest.fixture(autouse=True)
def _config_tests_see_default_db_path(request, monkeypatch):
    """test_config_stage2.py asserts the untouched production DB_PATH; hand
    it back for the duration of those tests only.

    A plain fixture, not a ``pytest_runtest_setup``/``teardown`` hook -- it
    only ever runs as part of normal per-item fixture resolution, which
    pytest guarantees happens before that item's test body regardless of
    collection order. ``monkeypatch`` restores the isolated path afterward,
    so no hand-rolled restore bookkeeping is needed.
    """
    if "test_config_stage2" in str(request.node.fspath):
        monkeypatch.setattr(settings, "DB_PATH", _DEFAULT_DB_PATH)
    yield


class _TestSourceNode(BaseNode):
    """Lightweight source node for tests -- no required inputs, no torch."""
    NODE_NAME = "_TestSource"
    CATEGORY = "Test"
    DESCRIPTION = "Emits a constant value"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        return {"value": params.get("val", "test")}


@pytest.fixture(scope="session", autouse=True)
def registry_with_nodes() -> NodeRegistry:
    """Discover all nodes once per test session, including chapter plugins."""
    if len(registry.nodes) == 0:
        registry.discover(settings.NODES_DIR, "app.nodes")
        registry.discover(settings.CUSTOM_NODES_DIR, "app.custom_nodes")
        # Plugin nodes — the three direction packs (foundations / deep / rl).
        for plugin_id in ("foundations", "deep", "rl"):
            plugin_nodes = _REPO_ROOT / "plugins" / plugin_id / "nodes"
            if plugin_nodes.exists():
                registry.discover(plugin_nodes, f"cdui_plugins.{plugin_id}.nodes")
        preset_registry.discover(settings.PRESETS_DIR, registry)
    registry._nodes["_TestSource"] = _TestSourceNode
    return registry


@pytest.fixture(autouse=True)
def _ensure_registry_intact(registry_with_nodes):
    """Run before every test: repopulate the registry if a prior test cleared it.

    ``POST /api/plugins/reload`` (and any test that calls ``rediscover_all``)
    clears every registry entry, including the manually-injected
    ``_TestSource`` synthetic node and the built-ins. Without this safety net,
    ws-execution tests that follow such a test see "Unknown node type".
    """
    if "_TestSource" not in registry._nodes:
        registry._nodes["_TestSource"] = _TestSourceNode
    if "Start" not in registry._nodes:
        # Wholesale rebuild — registry was nuked by an earlier reload.
        registry.discover(settings.NODES_DIR, "app.nodes")
        registry.discover(settings.CUSTOM_NODES_DIR, "app.custom_nodes")
        for plugin_id in ("foundations", "deep", "rl"):
            plugin_nodes = _REPO_ROOT / "plugins" / plugin_id / "nodes"
            if plugin_nodes.exists():
                registry.discover(plugin_nodes, f"cdui_plugins.{plugin_id}.nodes")
        registry._nodes["_TestSource"] = _TestSourceNode
    yield


@pytest.fixture
async def test_client():
    """Async HTTP client connected to the FastAPI app via ASGI transport.

    The base URL is chosen so the ``Host`` header (set automatically by
    httpx) matches the production whitelist seeded above. The session
    token is also pre-attached so tests don't need to know about it.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url=f"http://127.0.0.1:{settings.PORT}",
        headers={TOKEN_HEADER: session_token()},
    ) as client:
        yield client


@pytest.fixture
def sample_graph():
    """A minimal valid graph: Start -> _TestSource -> Print."""
    return {
        "nodes": [
            {"id": "start", "type": "Start", "position": {"x": -150, "y": 0}, "data": {"params": {}}},
            {"id": "1", "type": "_TestSource", "position": {"x": 0, "y": 0}, "data": {"params": {}}},
            {"id": "2", "type": "Print", "position": {"x": 200, "y": 0}, "data": {"params": {"label": "second"}}},
        ],
        "edges": [
            {"id": "et", "source": "start", "target": "1", "sourceHandle": "trigger", "type": "trigger"},
            {"id": "e1", "source": "1", "target": "2", "sourceHandle": "value", "targetHandle": "value"},
        ],
        "name": "test-graph",
        "description": "A test graph",
    }


@pytest.fixture
async def app_db(tmp_path):
    """Per-test Stage-2 Database on app.state (+ empty app_locks).

    PER TEST, never module/session-scoped: asyncio locks bind to the
    running event loop on first use. The lifespan does not run under
    httpx ASGITransport, so tests set app.state directly (the
    run_output_store precedent in test_api_graph_run.py).
    """
    from app.core.db import Database

    db = Database(tmp_path / "codefyui.db")
    db.connect()
    app.state.db = db
    app.state.app_locks = {}
    try:
        yield db
    finally:
        db.close()
        if hasattr(app.state, "db"):
            delattr(app.state, "db")
        if hasattr(app.state, "app_locks"):
            delattr(app.state, "app_locks")
