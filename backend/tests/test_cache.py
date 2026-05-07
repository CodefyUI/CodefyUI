"""Tests for node-level execution caching (Phase 5)."""

import pytest

from app.core.cache import ExecutionCache
from app.core.graph_engine import execute_graph
from app.core.node_base import BaseNode, DataType, PortDefinition
from app.core.node_registry import registry


class _CacheTestNode(BaseNode):
    """Lightweight node for cache tests (no torch dependency)."""
    NODE_NAME = "_CacheTest"
    CATEGORY = "Test"
    DESCRIPTION = "Returns a constant"

    @classmethod
    def define_inputs(cls):
        return []

    @classmethod
    def define_outputs(cls):
        return [PortDefinition(name="out", data_type=DataType.ANY)]

    def execute(self, inputs, params):
        return {"out": params.get("val", "default")}


@pytest.fixture(autouse=True)
def _register_cache_test_node():
    registry._nodes["_CacheTest"] = _CacheTestNode
    yield
    registry._nodes.pop("_CacheTest", None)


def _start_node(nid="start"):
    return {"id": nid, "type": "Start", "data": {"params": {}}}


def _trigger(eid, src, tgt):
    return {"id": eid, "source": src, "target": tgt, "sourceHandle": "trigger", "type": "trigger"}


def test_cache_compute_key_deterministic():
    k1 = ExecutionCache.compute_key("Conv2d", {"in_channels": 3}, ["abc"])
    k2 = ExecutionCache.compute_key("Conv2d", {"in_channels": 3}, ["abc"])
    assert k1 == k2


def test_cache_different_params_different_key():
    k1 = ExecutionCache.compute_key("Conv2d", {"in_channels": 3}, [])
    k2 = ExecutionCache.compute_key("Conv2d", {"in_channels": 64}, [])
    assert k1 != k2


def test_cache_put_and_get():
    cache = ExecutionCache()
    cache.put("key1", {"output": 42})
    assert cache.get("key1") == {"output": 42}
    assert cache.get("missing") is None


def test_cache_lru_eviction():
    cache = ExecutionCache(max_entries=2)
    cache.put("a", {"v": 1})
    cache.put("b", {"v": 2})
    cache.put("c", {"v": 3})  # evicts "a"
    assert cache.get("a") is None
    assert cache.get("b") == {"v": 2}
    assert cache.get("c") == {"v": 3}


def test_cache_lru_access_refreshes():
    cache = ExecutionCache(max_entries=2)
    cache.put("a", {"v": 1})
    cache.put("b", {"v": 2})
    cache.get("a")  # refresh "a"
    cache.put("c", {"v": 3})  # should evict "b" (least recently used)
    assert cache.get("a") == {"v": 1}
    assert cache.get("b") is None


@pytest.mark.asyncio
async def test_cache_hit_skips_execution():
    """Second run with same params should hit cache."""
    cache = ExecutionCache()
    run_count = 0

    async def count_runs(node_id, status, data):
        nonlocal run_count
        if status == "completed" and node_id != "start":
            run_count += 1

    nodes = [_start_node(), {"id": "1", "type": "_CacheTest", "data": {"params": {"val": "test"}}}]
    edges = [_trigger("et", "start", "1")]

    await execute_graph(nodes, edges, on_progress=count_runs, cache=cache)
    assert run_count == 1

    # Reset counter
    cached_count = 0

    async def count_cached(node_id, status, data):
        nonlocal cached_count
        if status == "cached" and node_id != "start":
            cached_count += 1

    await execute_graph(nodes, edges, on_progress=count_cached, cache=cache)
    assert cached_count == 1


@pytest.mark.asyncio
async def test_cache_invalidation_on_param_change():
    """Changing params should cause a cache miss."""
    cache = ExecutionCache()

    nodes_v1 = [_start_node(), {"id": "1", "type": "_CacheTest", "data": {"params": {"val": "v1"}}}]
    nodes_v2 = [_start_node(), {"id": "1", "type": "_CacheTest", "data": {"params": {"val": "v2"}}}]
    edges = [_trigger("et", "start", "1")]

    await execute_graph(nodes_v1, edges, cache=cache)

    statuses = {}

    async def track(node_id, status, data):
        statuses[node_id] = status

    await execute_graph(nodes_v2, edges, on_progress=track, cache=cache)
    # Should NOT be cached since param changed
    assert statuses.get("1") == "completed"


@pytest.mark.asyncio
async def test_cache_hit_still_populates_output_store():
    """Regression: if the first run primes the cache without record_outputs,
    a second run with record_outputs=True should still capture cached
    outputs into the store. Otherwise the Teaching Inspector sees nothing
    for nodes that were served from cache."""
    from app.core.run_output_store import RunOutputStore

    cache = ExecutionCache()
    store = RunOutputStore()

    nodes = [_start_node(), {"id": "1", "type": "_CacheTest", "data": {"params": {"val": "hello"}}}]
    edges = [_trigger("et", "start", "1")]

    # Run 1: populate cache, record_outputs=False (store stays empty)
    await execute_graph(
        nodes, edges,
        cache=cache, output_store=store,
        record_outputs=False, run_id="run-1",
    )
    assert await store.has_run("run-1") is False

    # Run 2: cache hits, record_outputs=True — store should now have the
    # cached value even though _CacheTestNode.execute never ran this time.
    statuses: dict[str, str] = {}

    async def track(node_id, status, data):
        statuses[node_id] = status

    await execute_graph(
        nodes, edges,
        on_progress=track, cache=cache, output_store=store,
        record_outputs=True, run_id="run-2",
    )
    assert statuses.get("1") == "cached"
    assert await store.get("run-2", "1", "out") == "hello"


class _NonCacheableNode(BaseNode):
    """Stateful-style node that returns a fresh value on every execution."""

    NODE_NAME = "_NonCacheable"
    CATEGORY = "Test"
    DESCRIPTION = "Returns a different output every call"
    cacheable = False

    @classmethod
    def define_inputs(cls):
        return []

    @classmethod
    def define_outputs(cls):
        return [PortDefinition(name="out", data_type=DataType.ANY)]

    def execute(self, inputs, params):
        # Use the call counter passed via params so each test invocation
        # produces a known sequence.
        _NonCacheableNode._counter = getattr(_NonCacheableNode, "_counter", 0) + 1
        return {"out": f"call-{_NonCacheableNode._counter}"}


class _PassthroughNode(BaseNode):
    """Cacheable node that passes its input through unchanged."""

    NODE_NAME = "_Passthrough"
    CATEGORY = "Test"
    DESCRIPTION = "Pass-through"
    cacheable = True

    @classmethod
    def define_inputs(cls):
        return [PortDefinition(name="in_value", data_type=DataType.ANY)]

    @classmethod
    def define_outputs(cls):
        return [PortDefinition(name="out", data_type=DataType.ANY)]

    def execute(self, inputs, params):
        return {"out": inputs.get("in_value")}


@pytest.mark.asyncio
async def test_non_cacheable_upstream_invalidates_downstream_cache():
    """Regression: when upstream is non-cacheable, the downstream node's cache
    key must NOT be reused across runs — otherwise a downstream cache hit
    returns a stale tensor whose actual upstream content has changed.

    Repro: Start → _NonCacheable → _Passthrough → output. Each run, the
    non-cacheable node returns a different value; the downstream Passthrough
    must observe that change rather than returning a cached "call-1" output
    indefinitely.
    """
    registry._nodes["_NonCacheable"] = _NonCacheableNode
    registry._nodes["_Passthrough"] = _PassthroughNode
    _NonCacheableNode._counter = 0
    try:
        cache = ExecutionCache()
        nodes = [
            _start_node(),
            {"id": "src", "type": "_NonCacheable", "data": {"params": {}}},
            {"id": "pass", "type": "_Passthrough", "data": {"params": {}}},
        ]
        edges = [
            _trigger("et", "start", "src"),
            {"id": "e1", "source": "src", "target": "pass", "sourceHandle": "out", "targetHandle": "in_value"},
        ]

        outputs1: dict[str, dict] = {}
        outputs2: dict[str, dict] = {}

        async def capture1(node_id, status, data):
            if status == "completed":
                outputs1[node_id] = data
            elif status == "cached":
                outputs1[node_id] = data

        async def capture2(node_id, status, data):
            if status == "completed":
                outputs2[node_id] = data
            elif status == "cached":
                outputs2[node_id] = data

        await execute_graph(nodes, edges, cache=cache, on_progress=capture1)
        await execute_graph(nodes, edges, cache=cache, on_progress=capture2)

        # First run: src = "call-1", pass = "call-1" (forwarded)
        # Second run: src = "call-2"; pass MUST also see "call-2", not the
        # stale "call-1" cached from run 1.
        assert outputs1["src"]["out"] == "call-1"
        assert outputs1["pass"]["out"] == "call-1"
        assert outputs2["src"]["out"] == "call-2"
        assert outputs2["pass"]["out"] == "call-2", (
            "Downstream cache must invalidate when upstream is non-cacheable; "
            f"got {outputs2['pass']['out']}"
        )
    finally:
        registry._nodes.pop("_NonCacheable", None)
        registry._nodes.pop("_Passthrough", None)
