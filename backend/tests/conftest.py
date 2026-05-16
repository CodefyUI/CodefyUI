"""Shared pytest fixtures for CodefyUI backend tests."""

import sys
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

# Tests use ``base_url="http://127.0.0.1:8000"`` which the production Host
# whitelist already accepts, but seed it explicitly here so tests don't rely
# on lifespan-time initialisation (lifespan runs once per app instance and
# ASGITransport doesn't always go through it).
init_allowed_hosts(settings.HOST, settings.PORT)

# Register the in-repo chapter plugin packs in the synthetic `cdui_plugins`
# namespace AT CONFTEST IMPORT TIME, before any test_*.py module is collected.
# Tests for Edu nodes import them from `cdui_plugins.c{2,3,4}.nodes.*` — those
# imports happen during pytest's collection pass, which runs after conftest is
# imported, so the namespace must exist by then.
_REPO_ROOT = Path(__file__).resolve().parents[2]
purge_all_plugin_modules()
install_plugin_finder(
    builtin_root=_REPO_ROOT / "plugins",
    user_root=_REPO_ROOT / "_phantom_user_root_for_tests",  # never read
    lockfile={
        "schema": 1,
        "plugins": {
            "c1": {"source_kind": "builtin", "source": "c1"},
            "c2": {"source_kind": "builtin", "source": "c2"},
            "c3": {"source_kind": "builtin", "source": "c3"},
            "c4": {"source_kind": "builtin", "source": "c4"},
            "c5": {"source_kind": "builtin", "source": "c5"},
            "c6": {"source_kind": "builtin", "source": "c6"},
        },
    },
)


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
        # Plugin nodes — c{1..6} are the six textbook chapter packs.
        for chapter in ("c1", "c2", "c3", "c4", "c5", "c6"):
            plugin_nodes = _REPO_ROOT / "plugins" / chapter / "nodes"
            if plugin_nodes.exists():
                registry.discover(plugin_nodes, f"cdui_plugins.{chapter}.nodes")
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
        for chapter in ("c1", "c2", "c3", "c4", "c5", "c6"):
            plugin_nodes = _REPO_ROOT / "plugins" / chapter / "nodes"
            if plugin_nodes.exists():
                registry.discover(plugin_nodes, f"cdui_plugins.{chapter}.nodes")
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
