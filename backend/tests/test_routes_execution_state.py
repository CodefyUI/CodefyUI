"""Tests for /api/execution/state REST endpoints (A2)."""

import pytest
import torch.nn as nn

from app.core.node_state_store import NodeStateStore
from app.main import app


@pytest.fixture(autouse=True)
def _ensure_state_store():
    app.state.node_state_store = NodeStateStore(max_modules=20)
    yield


@pytest.mark.asyncio
async def test_reset_graph_clears_all_nodes_for_graph(test_client):
    store = app.state.node_state_store
    store.get_or_create("g1", "n1", "h", lambda: nn.Linear(4, 4))
    store.get_or_create("g1", "n2", "h", lambda: nn.Linear(4, 4))
    store.get_or_create("g2", "n1", "h", lambda: nn.Linear(4, 4))

    resp = await test_client.post(
        "/api/execution/state/reset",
        json={"graph_id": "g1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["scope"] == "graph"
    assert body["evicted"] == 2
    assert len(store) == 1


@pytest.mark.asyncio
async def test_reset_specific_nodes(test_client):
    store = app.state.node_state_store
    store.get_or_create("g1", "n1", "h", lambda: nn.Linear(4, 4))
    store.get_or_create("g1", "n2", "h", lambda: nn.Linear(4, 4))

    resp = await test_client.post(
        "/api/execution/state/reset",
        json={"graph_id": "g1", "node_ids": ["n1"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["scope"] == "nodes"
    assert body["evicted"] == 1
    assert len(store) == 1


@pytest.mark.asyncio
async def test_list_state_for_graph(test_client):
    store = app.state.node_state_store
    store.get_or_create("g1", "n1", "h", lambda: nn.Linear(4, 4))
    store.get_or_create("g1", "n2", "h", lambda: nn.Linear(4, 4))

    resp = await test_client.get("/api/execution/state/list?graph_id=g1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert sorted(body["node_ids"]) == ["n1", "n2"]


@pytest.mark.asyncio
async def test_list_state_total(test_client):
    store = app.state.node_state_store
    store.get_or_create("g1", "n1", "h", lambda: nn.Linear(4, 4))
    store.get_or_create("g2", "n1", "h", lambda: nn.Linear(4, 4))

    resp = await test_client.get("/api/execution/state/list")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
