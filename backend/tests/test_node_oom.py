"""An out-of-memory failure is a typed, actionable error (#135).

The allocator's own message ("Tried to allocate 2.00 GiB") says what did not
fit and nothing about what to change, so the engine rewrites it. These tests
raise the failure rather than causing it: provoking a real CUDA OOM in CI
would need a card, would take minutes, and would leave the process in a
state the next test inherits.
"""

from __future__ import annotations

import pytest
import torch

from app.core.cache import ExecutionCache
from app.core.error_handling import (
    OOM_HINTS,
    NodeOOMError,
    is_out_of_memory,
    memory_digest,
    release_cached_memory,
)
from app.core.execution_context import ExecutionContext
from app.core.graph_engine import execute_graph
from app.core.node_base import BaseNode, DataType, PortDefinition
from app.core.node_registry import registry


OOM_MESSAGE = (
    "CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total "
    "capacity of 15.99 GiB of which 0 bytes is free.")

#: `torch.OutOfMemoryError` arrived in torch 2.5, which is the floor
#: `backend/pyproject.toml` declares since core#192 -- so it is a plain
#: attribute access here, not a `getattr` with a fallback. If this line ever
#: raises `AttributeError`, the installed torch is below the declared floor
#: and that is the bug.
_TORCH_OOM = __import__("torch").OutOfMemoryError


def _cuda_oom() -> BaseException:
    """The exception this torch raises when a CUDA allocation does not fit."""
    return _TORCH_OOM(OOM_MESSAGE)


class _HungryNode(BaseNode):
    """Raises whatever ``boom`` is set to, once per instance."""

    NODE_NAME = "_Hungry"
    CATEGORY = "Test"
    DESCRIPTION = "Fails on demand"

    #: Class attribute so a test can swap it between runs without
    #: re-registering the node.
    boom: BaseException | None = None

    @classmethod
    def define_inputs(cls):
        return []

    @classmethod
    def define_outputs(cls):
        return [PortDefinition(name="out", data_type=DataType.ANY)]

    def execute(self, inputs, params):
        if _HungryNode.boom is not None:
            raise _HungryNode.boom
        return {"out": "ok"}


@pytest.fixture(autouse=True)
def _register():
    registry._nodes["_Hungry"] = _HungryNode
    _HungryNode.boom = None
    yield
    registry._nodes.pop("_Hungry", None)
    _HungryNode.boom = None


def _graph():
    return (
        [{"id": "start", "type": "Start", "data": {"params": {}}},
         {"id": "n1", "type": "_Hungry", "data": {"params": {}}}],
        [{"id": "e", "source": "start", "target": "n1",
          "sourceHandle": "trigger", "type": "trigger"}],
    )


# ── classification ────────────────────────────────────────────────────────


def test_the_dedicated_torch_oom_type_is_recognised():
    """By TYPE, not by message -- the message match is the fallback."""
    assert is_out_of_memory(_TORCH_OOM("no message about memory here")) is True


def test_both_spellings_of_the_oom_class_are_recognised():
    """``torch.cuda.OutOfMemoryError`` is an ALIAS from 2.5 on (core#192).

    ``is_out_of_memory`` used to look both names up through ``getattr``
    because 2.0-2.4 -- which the old ``torch>=2.0.0`` floor admitted -- only
    had the ``torch.cuda`` one. With the floor at 2.5 the two names are one
    object, so the single check covers both. This test is what would catch
    a future release splitting them apart again, which is the only thing
    that could make the removed branch matter.
    """
    import torch

    assert torch.cuda.OutOfMemoryError is torch.OutOfMemoryError
    assert is_out_of_memory(torch.OutOfMemoryError("silent")) is True
    assert is_out_of_memory(torch.cuda.OutOfMemoryError("silent")) is True


def test_the_declared_torch_floor_is_the_one_the_code_assumes():
    """The floor and the code that leans on it must not drift apart.

    ``is_out_of_memory`` reaches ``torch.OutOfMemoryError`` as a plain
    attribute now. Lowering the floor back below 2.5 without restoring the
    older spelling's branch would leave the classifier falling through to
    the MESSAGE match on those builds -- silently, since a message match
    still returns True for a real CUDA OOM and only fails for the ones
    torch raises without one.
    """
    import re
    from pathlib import Path

    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml")
    spec = re.search(r'"torch>=([\d.]+)"', pyproject.read_text(encoding="utf-8"))
    assert spec is not None, "no torch floor declared in backend/pyproject.toml"
    major, minor = (int(part) for part in spec.group(1).split(".")[:2])
    assert (major, minor) >= (2, 5), (
        f"torch floor is {spec.group(1)}; torch.OutOfMemoryError needs >= 2.5")


def test_a_torch_oom_is_recognised():
    assert is_out_of_memory(_cuda_oom()) is True


def test_an_mps_style_runtime_error_is_recognised():
    assert is_out_of_memory(RuntimeError("MPS backend out of memory")) is True


def test_a_plain_memory_error_is_recognised():
    assert is_out_of_memory(MemoryError()) is True


def test_an_unrelated_failure_is_not_dressed_up_as_an_oom():
    assert is_out_of_memory(RuntimeError("shape mismatch")) is False
    assert is_out_of_memory(ValueError("out of memory")) is False


def test_the_digest_is_empty_off_cuda():
    assert memory_digest("cpu") == ""
    assert memory_digest("") == ""


def test_releasing_memory_off_cuda_does_nothing_and_says_nothing():
    release_cached_memory("cpu")  # must not raise


# ── the message ───────────────────────────────────────────────────────────


def test_the_error_names_the_node_the_device_and_every_hint():
    error = NodeOOMError.from_exception(
        _cuda_oom(), node_id="n1", node_type="TrainingLoop", device="cuda:0")
    text = str(error)
    assert "TrainingLoop" in text and "n1" in text and "cuda:0" in text
    for hint in OOM_HINTS:
        assert hint in text
    # The allocator's own words are kept: they say what did not fit.
    assert "Tried to allocate 2.00 GiB" in text
    assert error.node_id == "n1"
    assert error.hints == OOM_HINTS


def test_the_hints_name_the_levers_this_release_actually_provides():
    joined = " ".join(OOM_HINTS)
    assert "batch_size" in joined
    assert "bf16" in joined
    assert "accumulate_steps" in joined


# ── through the engine ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_an_oom_in_a_node_surfaces_as_a_node_oom_error():
    _HungryNode.boom = _cuda_oom()
    nodes, edges = _graph()
    with pytest.raises(NodeOOMError) as caught:
        await execute_graph(nodes, edges)
    assert caught.value.node_id == "n1"
    assert caught.value.node_type == "_Hungry"
    assert "accumulate_steps" in str(caught.value)


@pytest.mark.asyncio
async def test_an_oom_is_not_retried():
    """Retry mode re-runs a failing node; an OOM is the one it must not.

    The same allocation on the same device gets the same answer, so a retry
    only delays the message telling the user what to change. Note the run
    still finishes the way ``retry`` mode says it should -- the error is
    RECORDED rather than raised -- because the no-retry rule is about how
    many times the node runs, not about overriding the error policy the
    submitter chose.
    """
    attempts = 0
    errors: list[str] = []

    class _CountingNode(_HungryNode):
        def execute(self, inputs, params):
            nonlocal attempts
            attempts += 1
            raise _cuda_oom()

    async def observe(node_id, status, data):
        if status == "error":
            errors.append(data["error"])

    registry._nodes["_Hungry"] = _CountingNode
    nodes, edges = _graph()
    await execute_graph(nodes, edges, on_progress=observe,
                        error_mode="retry", max_retries=3)
    assert attempts == 1
    assert errors and "accumulate_steps" in errors[0]


@pytest.mark.asyncio
async def test_an_ordinary_failure_is_still_retried():
    """The no-retry rule is scoped to OOM and nothing else."""
    attempts = 0

    class _FlakyNode(_HungryNode):
        def execute(self, inputs, params):
            nonlocal attempts
            attempts += 1
            raise RuntimeError("shape mismatch")

    registry._nodes["_Hungry"] = _FlakyNode
    nodes, edges = _graph()
    await execute_graph(nodes, edges, error_mode="retry", max_retries=2)
    assert attempts == 3


@pytest.mark.asyncio
async def test_an_oom_drops_that_nodes_cache_entries_and_leaves_the_rest():
    """Recovery is targeted: the failing node's cached tensors, not the cache.

    A first run caches both nodes' outputs. The second run OOMs in one of
    them, and the entry holding tensors on the device that just ran out has
    to go -- while the sibling's entry, which had nothing to do with it,
    stays. That difference is what distinguishes a targeted drop from a
    ``cache.clear()``.
    """
    cache = ExecutionCache()
    nodes, edges = _graph()
    nodes.append({"id": "n2", "type": "_Hungry",
                  "data": {"params": {"val": 2}}})
    edges.append({"id": "e2", "source": "start", "target": "n2",
                  "sourceHandle": "trigger", "type": "trigger"})

    await execute_graph(nodes, edges, cache=cache)
    assert len(cache) == 3  # Start plus both _Hungry nodes

    class _OnlyN1Fails(_HungryNode):
        def execute(self, inputs, params):
            raise _cuda_oom()

    registry._nodes["_Hungry"] = _OnlyN1Fails
    with pytest.raises(NodeOOMError):
        await execute_graph(nodes, edges, cache=cache, changed_nodes=["n1"])

    assert cache.clear_node("n1") == 0, "n1's entry should already be gone"
    assert cache.clear_node("n2") == 1, "n2's entry must have survived"


@pytest.mark.asyncio
async def test_the_engine_still_runs_the_next_graph_after_an_oom():
    """The acceptance criterion: the server stays healthy for the next run."""
    _HungryNode.boom = _cuda_oom()
    nodes, edges = _graph()
    with pytest.raises(NodeOOMError):
        await execute_graph(nodes, edges)

    _HungryNode.boom = None
    outputs = await execute_graph(nodes, edges)
    assert outputs["n1"]["out"] == "ok"


@pytest.mark.asyncio
async def test_continue_mode_records_the_oom_and_carries_on():
    """``error_mode="continue"`` is still honoured -- OOM only skips RETRIES."""
    _HungryNode.boom = _cuda_oom()
    statuses: list[tuple[str, str]] = []

    async def observe(node_id, status, data):
        statuses.append((node_id, status))
        if status == "error" and node_id == "n1":
            assert "accumulate_steps" in data["error"]

    nodes, edges = _graph()
    await execute_graph(nodes, edges, on_progress=observe,
                        error_mode="continue")
    assert ("n1", "error") in statuses


# ── crossed with determinism (#193) ──────────────────────────────────────
#
# Both paths reach for PROCESS-GLOBAL state: OOM recovery calls
# ``torch.cuda.empty_cache()``, and a deterministic run holds a refcounted
# scope over ``torch.use_deterministic_algorithms``. Neither had a test
# saying what happens when one unwinds through the other.


@pytest.fixture
def _determinism_restored():
    """Give torch its flags back whatever the test does to them."""
    previously = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previously, warn_only=warn_only)


@pytest.mark.asyncio
async def test_oom_recovery_runs_INSIDE_the_deterministic_scope(
    _determinism_restored, monkeypatch,
):
    """Ordering, asserted from inside the recovery rather than after it.

    The question a crossover test has to answer is not "did both happen"
    but "which one was still in force while the other ran". Recovery that
    fired AFTER the scope unwound would be touching the allocator with the
    process's determinism setting already handed back — a different, and
    much harder to reason about, ordering than the one the code intends.

    ``release_cached_memory`` is the probe point because it is the last
    thing the recovery does and the one that touches the allocator. It is
    stubbed rather than allowed to run: on a CPU-only box it returns before
    reaching ``empty_cache()`` at all (see its ``kind != "cuda"`` guard), so
    letting it run would assert nothing while LOOKING like it asserted the
    interaction. This test pins the ordering; it does not pretend to
    exercise a real CUDA free.
    """
    from app.core import graph_engine as engine
    from app.core.seeding import determinism_depth

    seen: list[tuple[bool, int]] = []

    def _probe(device: str = "") -> None:
        seen.append((torch.are_deterministic_algorithms_enabled(),
                     determinism_depth()))

    monkeypatch.setattr(engine, "release_cached_memory", _probe)

    _HungryNode.boom = _cuda_oom()
    nodes, edges = _graph()
    with pytest.raises(NodeOOMError):
        await execute_graph(
            nodes, edges,
            context=ExecutionContext(device="cpu", deterministic=True))

    assert seen, "OOM recovery never ran"
    enabled, depth = seen[0]
    assert enabled, "recovery ran after determinism was handed back"
    assert depth == 1, f"expected to be inside exactly one scope, saw {depth}"


@pytest.mark.asyncio
async def test_an_oom_in_a_deterministic_run_still_unwinds_the_scope(
    _determinism_restored,
):
    """And the NEXT run is unaffected — which is the whole point of #134.

    An OOM leaves ``execute_graph`` by the same ``finally`` that owns the
    scope's exit, so this is the ordinary path rather than a corner. It is
    pinned because the two mechanisms are independent and nothing else says
    they compose: a future recovery step that returned early, or unwound by
    some other route, would strand the depth and weld determinism on for
    the life of the process.
    """
    from app.core.seeding import determinism_depth

    _HungryNode.boom = _cuda_oom()
    nodes, edges = _graph()
    with pytest.raises(NodeOOMError):
        await execute_graph(
            nodes, edges,
            context=ExecutionContext(device="cpu", deterministic=True))

    assert determinism_depth() == 0, "the OOM stranded the scope's depth"
    assert not torch.are_deterministic_algorithms_enabled()

    # The next run asked for nothing and must get nothing.
    _HungryNode.boom = None
    await execute_graph(nodes, edges, context=ExecutionContext(device="cpu"))
    assert not torch.are_deterministic_algorithms_enabled()
