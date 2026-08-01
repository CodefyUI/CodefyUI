"""Tests for RunOutputStore."""

import asyncio

import pytest

from app.core.run_output_store import RunOutputStore


@pytest.mark.asyncio
async def test_put_and_get_roundtrip():
    store = RunOutputStore(max_runs=5)
    await store.put("r1", "n1", "output", [1, 2, 3])
    assert await store.get("r1", "n1", "output") == [1, 2, 3]


@pytest.mark.asyncio
async def test_missing_returns_none():
    store = RunOutputStore()
    assert await store.get("r1", "n1", "output") is None


@pytest.mark.asyncio
async def test_has_run():
    store = RunOutputStore()
    assert await store.has_run("r1") is False
    await store.put("r1", "n1", "p", 1)
    assert await store.has_run("r1") is True


@pytest.mark.asyncio
async def test_multiple_nodes_and_ports():
    store = RunOutputStore()
    await store.put("r1", "n1", "out1", 1)
    await store.put("r1", "n1", "out2", 2)
    await store.put("r1", "n2", "out1", 3)
    assert await store.get("r1", "n1", "out1") == 1
    assert await store.get("r1", "n1", "out2") == 2
    assert await store.get("r1", "n2", "out1") == 3


@pytest.mark.asyncio
async def test_list_ports():
    store = RunOutputStore()
    await store.put("r1", "n1", "a", 1)
    await store.put("r1", "n1", "b", 2)
    await store.put("r1", "n2", "c", 3)
    ports = await store.list_ports("r1")
    assert ports is not None
    assert set(ports) == {("n1", "a"), ("n1", "b"), ("n2", "c")}


@pytest.mark.asyncio
async def test_list_ports_unknown_run():
    store = RunOutputStore()
    assert await store.list_ports("missing") is None


@pytest.mark.asyncio
async def test_delete_run():
    store = RunOutputStore()
    await store.put("r1", "n1", "p", 1)
    assert await store.delete_run("r1") is True
    assert await store.has_run("r1") is False
    assert await store.delete_run("r1") is False


@pytest.mark.asyncio
async def test_lru_eviction_keeps_newest():
    store = RunOutputStore(max_runs=3)
    for i in range(5):
        await store.put(f"r{i}", "n", "p", i)
    runs = await store.list_runs()
    assert runs == ["r2", "r3", "r4"]
    assert await store.has_run("r0") is False
    assert await store.has_run("r1") is False
    assert await store.has_run("r2") is True
    assert await store.has_run("r4") is True


@pytest.mark.asyncio
async def test_same_run_id_does_not_evict_others():
    store = RunOutputStore(max_runs=2)
    await store.put("r1", "n", "p", 1)
    await store.put("r2", "n", "p", 2)
    # Re-putting into r1 should NOT move it in LRU order or evict r2
    await store.put("r1", "n", "q", 11)
    runs = await store.list_runs()
    assert set(runs) == {"r1", "r2"}
    assert await store.get("r1", "n", "q") == 11


@pytest.mark.asyncio
async def test_concurrent_puts():
    store = RunOutputStore(max_runs=100)

    async def worker(run_id: str, count: int):
        for i in range(count):
            await store.put(run_id, f"node{i}", "port", i)

    await asyncio.gather(
        *[worker(f"r{i}", 10) for i in range(10)]
    )
    for i in range(10):
        assert await store.has_run(f"r{i}")
        ports = await store.list_ports(f"r{i}")
        assert ports is not None
        assert len(ports) == 10


@pytest.mark.asyncio
async def test_clear():
    store = RunOutputStore()
    await store.put("r1", "n", "p", 1)
    await store.put("r2", "n", "p", 2)
    await store.clear()
    assert await store.list_runs() == []
    assert await store.has_run("r1") is False


# ── write serials (#129) ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_with_version_returns_value_and_serial():
    store = RunOutputStore()
    await store.put("r1", "n1", "p", "first")
    slot = await store.get_with_version("r1", "n1", "p")
    assert slot is not None
    value, version = slot
    assert value == "first"
    assert version > 0


@pytest.mark.asyncio
async def test_get_with_version_is_none_for_a_missing_port():
    store = RunOutputStore()
    await store.put("r1", "n1", "p", 1)
    assert await store.get_with_version("r1", "n1", "other") is None
    assert await store.get_with_version("r1", "other", "p") is None
    assert await store.get_with_version("other", "n1", "p") is None


@pytest.mark.asyncio
async def test_overwriting_a_port_bumps_its_serial():
    store = RunOutputStore()
    await store.put("r1", "n1", "p", "first")
    first = (await store.get_with_version("r1", "n1", "p"))[1]
    await store.put("r1", "n1", "p", "second")
    second = (await store.get_with_version("r1", "n1", "p"))[1]
    assert second > first
    assert await store.get("r1", "n1", "p") == "second"


@pytest.mark.asyncio
async def test_serials_are_unique_across_ports_and_runs():
    store = RunOutputStore()
    await store.put("r1", "n1", "a", 1)
    await store.put("r1", "n1", "b", 2)
    await store.put("r2", "n1", "a", 3)
    versions = [
        (await store.get_with_version(*key))[1]
        for key in (("r1", "n1", "a"), ("r1", "n1", "b"), ("r2", "n1", "a"))
    ]
    assert len(set(versions)) == 3


@pytest.mark.asyncio
async def test_serial_distinguishes_tensors_that_reuse_one_allocation():
    """The reason the serial exists rather than a fingerprint of the value.

    torch's caching allocator hands a freed block straight to the next tensor
    of the same shape, so a replacement can carry the previous tensor's
    ``data_ptr``, ``shape`` and ``dtype`` and be indistinguishable from it.
    """
    import torch

    store = RunOutputStore()
    seen_versions = set()
    seen_pointers = set()
    for i in range(6):
        await store.put("r1", "n1", "out", torch.full((512, 512), float(i)))
        value, version = await store.get_with_version("r1", "n1", "out")
        seen_versions.add(version)
        seen_pointers.add(value.data_ptr())
        assert float(value[0][0]) == float(i)

    assert len(seen_versions) == 6
    # If this ever stops holding, the allocator changed — the serial is still
    # correct, this assertion just no longer demonstrates why it is needed.
    assert len(seen_pointers) < 6
