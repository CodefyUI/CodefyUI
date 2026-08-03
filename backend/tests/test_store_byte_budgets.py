"""The three in-memory stores evict by BYTES, not only by count (#135).

Every test here sizes its payload deliberately, because the point of the
change is that a count cannot distinguish 256 scalars from 256 feature
maps. The budgets are passed explicitly rather than read from settings, so
these assertions do not move when the defaults do.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from app.config import Settings
from app.core.cache import ExecutionCache, execution_cache_stats
from app.core.node_state_store import NodeStateStore
from app.core.run_output_store import RunOutputStore

MB = 1024 * 1024


def _tensor(megabytes: float) -> torch.Tensor:
    """A float32 tensor of exactly *megabytes* MB."""
    return torch.zeros(int(megabytes * MB // 4), dtype=torch.float32)


#: Room for three 1 MB entries with change to spare. The half-megabyte of
#: slack is not decoration: an entry is its tensor PLUS the dict and port
#: name holding it, so three of them measure a little over 3 MB and a
#: budget of exactly 3 MB would evict one before the fourth ever arrived --
#: making the test pass for the wrong reason.
THREE_ENTRY_BUDGET = 3 * MB + MB // 2


# ── settings ──────────────────────────────────────────────────────────────


def test_the_three_budgets_have_the_specified_defaults():
    settings = Settings()
    assert settings.EXECUTION_CACHE_MAX_MB == 1024
    assert settings.RUN_OUTPUT_STORE_MAX_MB == 2048
    assert settings.NODE_STATE_STORE_MAX_MB == 1024


def test_the_budgets_are_env_overridable(monkeypatch):
    monkeypatch.setenv("CODEFYUI_EXECUTION_CACHE_MAX_MB", "7")
    monkeypatch.setenv("CODEFYUI_RUN_OUTPUT_STORE_MAX_MB", "8")
    monkeypatch.setenv("CODEFYUI_NODE_STATE_STORE_MAX_MB", "9")
    settings = Settings()
    assert settings.EXECUTION_CACHE_MAX_MB == 7
    assert settings.RUN_OUTPUT_STORE_MAX_MB == 8
    assert settings.NODE_STATE_STORE_MAX_MB == 9


def test_a_store_built_with_no_budget_reads_one_from_settings():
    cache = ExecutionCache()
    assert cache.max_bytes == Settings().EXECUTION_CACHE_MAX_MB * MB


# ── ExecutionCache ────────────────────────────────────────────────────────


def test_filling_the_cache_past_max_bytes_evicts_by_bytes():
    """The acceptance criterion, with sized dummy tensors.

    Four 1 MB entries into a 3 MB budget: the entry count never approaches
    its own limit, so anything evicted here was evicted for its size.
    """
    cache = ExecutionCache(max_entries=100, max_bytes=THREE_ENTRY_BUDGET)
    for name in ("a", "b", "c"):
        cache.put(name, {"out": _tensor(1)})
    assert len(cache) == 3
    assert cache.total_bytes == pytest.approx(3 * MB, rel=0.01)

    cache.put("d", {"out": _tensor(1)})
    assert cache.get("a") is None, "the least recently used entry should go"
    assert cache.get("b") is not None
    assert cache.get("d") is not None
    assert len(cache) == 3
    assert cache.total_bytes <= THREE_ENTRY_BUDGET


def test_byte_eviction_respects_recency():
    cache = ExecutionCache(max_entries=100, max_bytes=THREE_ENTRY_BUDGET)
    for name in ("a", "b", "c"):
        cache.put(name, {"out": _tensor(1)})
    cache.get("a")  # a is now the most recently used
    cache.put("d", {"out": _tensor(1)})
    assert cache.get("a") is not None
    assert cache.get("b") is None


def test_one_entry_larger_than_the_whole_budget_is_not_stored():
    """Storing it would evict everything and still not fit."""
    cache = ExecutionCache(max_entries=100, max_bytes=2 * MB)
    cache.put("small", {"out": _tensor(0.5)})
    cache.put("huge", {"out": _tensor(8)})
    assert cache.get("huge") is None
    assert cache.get("small") is not None
    assert cache.total_bytes < 2 * MB


def test_the_entry_count_limit_still_applies():
    """Both limits, and whichever binds first evicts."""
    cache = ExecutionCache(max_entries=2, max_bytes=1024 * MB)
    for name in ("a", "b", "c"):
        cache.put(name, {"out": _tensor(0.001)})
    assert len(cache) == 2
    assert cache.get("a") is None


def test_replacing_a_key_gives_its_bytes_back():
    cache = ExecutionCache(max_entries=100, max_bytes=8 * MB)
    cache.put("k", {"out": _tensor(2)})
    first = cache.total_bytes
    cache.put("k", {"out": _tensor(2)})
    assert cache.total_bytes == first
    assert len(cache) == 1


def test_clearing_one_node_drops_only_its_entries():
    cache = ExecutionCache(max_entries=100, max_bytes=64 * MB)
    cache.put("k1", {"out": _tensor(1)}, "nodeA")
    cache.put("k2", {"out": _tensor(1)}, "nodeA")
    cache.put("k3", {"out": _tensor(1)}, "nodeB")

    assert cache.clear_node("nodeA") == 2
    assert cache.get("k1") is None and cache.get("k2") is None
    assert cache.get("k3") is not None
    assert cache.total_bytes == pytest.approx(1 * MB, rel=0.01)
    # Idempotent, which matters because an OOM can fire twice in one run.
    assert cache.clear_node("nodeA") == 0


def test_clear_resets_the_byte_total():
    cache = ExecutionCache(max_entries=100, max_bytes=64 * MB)
    cache.put("k", {"out": _tensor(4)}, "node")
    cache.clear()
    assert cache.total_bytes == 0
    assert cache.clear_node("node") == 0


def test_a_zero_budget_disables_the_byte_limit():
    cache = ExecutionCache(max_entries=100, max_bytes=0)
    for index in range(4):
        cache.put(str(index), {"out": _tensor(4)})
    assert len(cache) == 4


def test_live_caches_are_visible_to_health():
    cache = ExecutionCache(max_entries=10, max_bytes=64 * MB)
    cache.put("k", {"out": _tensor(2)})
    stats = execution_cache_stats()
    assert stats["instances"] >= 1
    assert stats["bytes"] >= 2 * MB
    assert cache.stats()["entries"] == 1


# ── RunOutputStore ────────────────────────────────────────────────────────


async def test_the_output_store_evicts_whole_runs_by_bytes():
    store = RunOutputStore(max_runs=100, max_bytes=3 * MB)
    for run in ("r1", "r2", "r3"):
        await store.put(run, "n", "out", _tensor(1))
    assert await store.list_runs() == ["r1", "r2", "r3"]

    await store.put("r4", "n", "out", _tensor(1))
    assert await store.has_run("r1") is False
    assert await store.list_runs() == ["r2", "r3", "r4"]


async def test_overwriting_a_port_does_not_inflate_the_total():
    """A node inside a loop rewrites one port on every iteration."""
    store = RunOutputStore(max_runs=100, max_bytes=64 * MB)
    for _ in range(10):
        await store.put("r1", "n", "out", _tensor(1))
    stats = await store.stats()
    assert stats["bytes"] == pytest.approx(1 * MB, rel=0.01)
    assert stats["runs"] == 1


async def test_the_run_being_written_is_never_the_one_evicted():
    """Otherwise a single oversized run evicts itself forever."""
    store = RunOutputStore(max_runs=100, max_bytes=1 * MB)
    await store.put("r1", "n", "a", _tensor(4))
    assert await store.get("r1", "n", "a") is not None
    stats = await store.stats()
    assert stats["bytes"] > stats["max_bytes"]


async def test_deleting_a_run_gives_its_bytes_back():
    store = RunOutputStore(max_runs=100, max_bytes=64 * MB)
    await store.put("r1", "n", "out", _tensor(2))
    await store.put("r2", "n", "out", _tensor(2))
    assert await store.delete_run("r1") is True
    assert (await store.stats())["bytes"] == pytest.approx(2 * MB, rel=0.01)
    await store.clear()
    assert (await store.stats())["bytes"] == 0


async def test_the_run_count_limit_still_applies():
    store = RunOutputStore(max_runs=2, max_bytes=1024 * MB)
    for run in ("r1", "r2", "r3"):
        await store.put(run, "n", "out", torch.zeros(4))
    assert await store.list_runs() == ["r2", "r3"]


# ── NodeStateStore ────────────────────────────────────────────────────────


def _linear_of(megabytes: float) -> nn.Module:
    """A Linear whose weights are about *megabytes* MB."""
    width = int((megabytes * MB / 4) ** 0.5)
    return nn.Linear(width, width, bias=False)


def test_the_state_store_evicts_modules_by_bytes():
    store = NodeStateStore(max_modules=100, max_bytes=3 * MB)
    for node in ("a", "b", "c"):
        store.get_or_create("g", node, "h", lambda: _linear_of(1))
    assert len(store) == 3

    store.get_or_create("g", "d", "h", lambda: _linear_of(1))
    assert store.keys_for_node("g", "a") == []
    assert store.keys_for_node("g", "d") != []
    assert store.stats()["bytes"] <= 3 * MB


def test_the_module_just_built_is_never_the_one_evicted():
    """Handing back a module the store has already forgotten would silently
    lose that node's weights between runs -- the one thing it exists to
    prevent."""
    store = NodeStateStore(max_modules=100, max_bytes=1 * MB)
    module = store.get_or_create("g", "big", "h", lambda: _linear_of(4))
    assert store.keys_for_node("g", "big") != []
    assert module is store.get_or_create("g", "big", "h", lambda: _linear_of(4))


def test_resetting_gives_the_bytes_back():
    store = NodeStateStore(max_modules=100, max_bytes=64 * MB)
    store.get_or_create("g", "a", "h", lambda: _linear_of(1))
    store.get_or_create("g", "b", "h", lambda: _linear_of(1))
    assert store.stats()["bytes"] > 0

    assert store.reset_node("g", "a") == 1
    after_node = store.stats()["bytes"]
    assert after_node == pytest.approx(1 * MB, rel=0.05)

    store.reset_all()
    assert store.stats()["bytes"] == 0


def test_a_structural_change_gives_the_old_modules_bytes_back():
    store = NodeStateStore(max_modules=100, max_bytes=64 * MB)
    store.get_or_create("g", "a", "shape1", lambda: _linear_of(2))
    store.get_or_create("g", "a", "shape2", lambda: _linear_of(1))
    assert len(store) == 1
    assert store.stats()["bytes"] == pytest.approx(1 * MB, rel=0.05)


def test_two_threads_building_the_same_module_do_not_drift_the_total():
    """``get_or_create`` runs the builder OUTSIDE the lock, on purpose.

    Module construction can be slow and can take CUDA's own locks, so two
    executor threads can both miss on one key and both arrive at the
    insert. Before the guard, the second one's bytes were added on top of
    the first's (a drift that never recovers, so the budget eventually
    evicts live modules) and the key landed in the LRU deque twice (a
    phantom that makes the count limit evict early).
    """
    import threading

    store = NodeStateStore(max_modules=100, max_bytes=64 * MB)
    start = threading.Barrier(2)

    def build():
        start.wait(timeout=5)
        return _linear_of(1)

    threads = [threading.Thread(
        target=lambda: store.get_or_create("g", "n", "h", build))
        for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert len(store) == 1
    assert store.stats()["bytes"] == pytest.approx(1 * MB, rel=0.05)
    assert store.reset_node("g", "n") == 1
    assert store.stats()["bytes"] == 0


def test_discarding_trained_weights_is_reported_to_the_caller():
    """An eviction here is not a cache miss -- it throws away TRAINING.

    The byte budget binds long before the count does (a 1 GB budget holds
    about ten ResNet-50s, and the gradients this store deliberately does
    not re-count make the real figure smaller), so this is a path users
    reach, and the next run silently restarts that node from a fresh
    initialisation. The store therefore hands its caller the keys it
    discarded rather than only writing a log line.
    """
    store = NodeStateStore(max_modules=100, max_bytes=THREE_ENTRY_BUDGET)
    seen: list[tuple[list, str]] = []
    for node in ("a", "b", "c"):
        store.get_or_create("g", node, "h", lambda: _linear_of(1))
    assert seen == [], "nothing should have been evicted yet"

    store.get_or_create("g", "d", "h", lambda: _linear_of(1),
                        on_evict=lambda keys, reason: seen.append((keys, reason)))

    assert len(seen) == 1
    keys, reason = seen[0]
    assert reason == "bytes"
    assert [k[1] for k in keys] == ["a"]


def test_a_count_eviction_is_reported_with_its_own_reason():
    store = NodeStateStore(max_modules=2, max_bytes=1024 * MB)
    seen: list[tuple[list, str]] = []
    for node in ("a", "b"):
        store.get_or_create("g", node, "h", lambda: nn.Linear(2, 2))
    store.get_or_create("g", "c", "h", lambda: nn.Linear(2, 2),
                        on_evict=lambda keys, reason: seen.append((keys, reason)))
    assert [(len(k), r) for k, r in seen] == [(1, "count")]


def test_a_structural_reset_is_not_reported_as_a_budget_eviction():
    """Changing a node's shape drops its own old module on purpose.

    That is a deliberate reset, not the store running out of room, and
    warning about it would train users to ignore the warning that matters.
    """
    store = NodeStateStore(max_modules=100, max_bytes=64 * MB)
    seen: list = []
    store.get_or_create("g", "a", "shape1", lambda: _linear_of(1))
    store.get_or_create("g", "a", "shape2", lambda: _linear_of(1),
                        on_evict=lambda keys, reason: seen.append(reason))
    assert seen == []


def test_the_eviction_warning_reaches_the_run_event_stream():
    """The half a user watching the canvas can actually notice.

    ``run_warning`` is the same event the dropped-signal notice uses, which
    the canvas renders as a toast and the Runs panel as a log line.
    """
    from app.core.execution_context import ExecutionContext, WarningSignal
    from app.core.stateful_module import StatefulModuleMixin

    class _Layer(StatefulModuleMixin):
        def build_module(self, params):
            return _linear_of(1)

    context = ExecutionContext()
    context.node_state_store = NodeStateStore(
        max_modules=100, max_bytes=THREE_ENTRY_BUDGET)
    context.graph_id = "g"

    layer = _Layer()
    for node_id in ("a", "b", "c", "d"):
        context.current_node_id = node_id
        layer.get_or_build_module(context, {"width": 1})

    signals, _ = context.outbox.drain()
    warnings = [s for s in signals if isinstance(s, WarningSignal)]
    assert len(warnings) == 1
    assert warnings[0].kind == "node_state_evicted"
    assert "fresh initialisation" in warnings[0].detail
    assert "CODEFYUI_NODE_STATE_STORE_MAX_MB" in warnings[0].detail


def test_the_eviction_notice_cannot_fail_the_node_that_triggered_it():
    from app.core.stateful_module import _report_eviction

    class _Hostile:
        def log_warning(self, *a, **k):
            raise RuntimeError("reporting exploded")

    _report_eviction(_Hostile(), [("g", "n", "h")], "bytes")  # must not raise
    _report_eviction(_Hostile(), [], "bytes")


def test_the_module_count_limit_still_applies():
    store = NodeStateStore(max_modules=2, max_bytes=1024 * MB)
    for node in ("a", "b", "c"):
        store.get_or_create("g", node, "h", lambda: nn.Linear(2, 2))
    assert len(store) == 2
    assert store.keys_for_node("g", "a") == []


# ── /api/health ───────────────────────────────────────────────────────────


async def test_health_reports_what_the_stores_hold(test_client):
    response = await test_client.get("/api/health")
    caches = response.json()["caches"]
    # The execution cache is always reported: it is a process-wide
    # registry rather than something hung off app.state, so it does not
    # depend on the lifespan having run.
    assert set(caches["execution_cache"]) >= {
        "instances", "entries", "bytes", "max_bytes_each"}
    assert caches["execution_cache"]["max_bytes_each"] > 0
    # The other two are omitted rather than reported as zero when the
    # lifespan has not run (it does not, under httpx's ASGITransport) --
    # "empty" and "not running" must not read the same. The next test
    # covers what they say when they DO exist.
    for name in ("run_output_store", "node_state_store"):
        assert name not in caches or caches[name]["max_bytes"] > 0


async def test_health_reports_the_stores_when_they_exist(test_client):
    from app.main import app

    app.state.run_output_store = RunOutputStore(max_runs=2, max_bytes=8 * MB)
    app.state.node_state_store = NodeStateStore(max_modules=2,
                                                max_bytes=4 * MB)
    try:
        await app.state.run_output_store.put("r", "n", "out", _tensor(1))
        caches = (await test_client.get("/api/health")).json()["caches"]
        assert caches["run_output_store"]["runs"] == 1
        assert caches["run_output_store"]["bytes"] == pytest.approx(
            1 * MB, rel=0.01)
        assert caches["run_output_store"]["max_bytes"] == 8 * MB
        assert caches["node_state_store"]["modules"] == 0
        assert caches["node_state_store"]["max_bytes"] == 4 * MB
    finally:
        delattr(app.state, "run_output_store")
        delattr(app.state, "node_state_store")
