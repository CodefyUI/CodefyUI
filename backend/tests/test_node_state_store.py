"""Tests for NodeStateStore (A2)."""

from __future__ import annotations

import torch
import torch.nn as nn

from app.core.node_state_store import NodeStateStore


def test_get_or_create_returns_same_instance_on_repeat_calls():
    store = NodeStateStore()
    builder_calls = {"n": 0}

    def build():
        builder_calls["n"] += 1
        return nn.Linear(4, 8)

    a = store.get_or_create("g1", "node1", "h1", build)
    b = store.get_or_create("g1", "node1", "h1", build)
    assert a is b
    assert builder_calls["n"] == 1


def test_get_or_create_evicts_siblings_on_structure_change():
    store = NodeStateStore()
    a = store.get_or_create("g1", "node1", "hash-old", lambda: nn.Linear(4, 8))
    b = store.get_or_create("g1", "node1", "hash-new", lambda: nn.Linear(4, 16))
    assert a is not b
    # Old hash should no longer resolve to the same instance.
    c = store.get_or_create("g1", "node1", "hash-old", lambda: nn.Linear(4, 8))
    assert c is not a, "structure-change eviction should drop the old key"


def test_reset_node_clears_all_hashes_for_a_node():
    store = NodeStateStore()
    store.get_or_create("g1", "n1", "h1", lambda: nn.Linear(2, 2))
    store.get_or_create("g1", "n2", "h1", lambda: nn.Linear(2, 2))
    evicted = store.reset_node("g1", "n1")
    assert evicted == 1
    assert len(store) == 1


def test_reset_graph_clears_all_nodes():
    store = NodeStateStore()
    store.get_or_create("g1", "n1", "h1", lambda: nn.Linear(2, 2))
    store.get_or_create("g1", "n2", "h1", lambda: nn.Linear(2, 2))
    store.get_or_create("g2", "n1", "h1", lambda: nn.Linear(2, 2))
    evicted = store.reset_graph("g1")
    assert evicted == 2
    assert len(store) == 1  # only g2/n1 remains


def test_lru_evicts_oldest_when_max_exceeded():
    store = NodeStateStore(max_modules=2)
    store.get_or_create("g1", "a", "h", lambda: nn.Linear(2, 2))
    store.get_or_create("g1", "b", "h", lambda: nn.Linear(2, 2))
    store.get_or_create("g1", "c", "h", lambda: nn.Linear(2, 2))
    assert len(store) == 2
    keys_for_a = store.keys_for_node("g1", "a")
    assert keys_for_a == [], "oldest entry (a) should have been evicted"


def test_iter_for_graph_returns_only_matching_graph():
    store = NodeStateStore()
    store.get_or_create("g1", "n1", "h", lambda: nn.Linear(2, 2))
    store.get_or_create("g2", "n1", "h", lambda: nn.Linear(2, 2))
    items = list(store.iter_for_graph("g1"))
    assert len(items) == 1
    assert items[0][0][0] == "g1"
