"""An out-of-memory failure is a typed, actionable error (#135).

The allocator's own message ("Tried to allocate 2.00 GiB") says what did not
fit and nothing about what to change, so the engine rewrites it. These tests
raise the failure rather than causing it: provoking a real CUDA OOM in CI
would need a card, would take minutes, and would leave the process in a
state the next test inherits.
"""

from __future__ import annotations

import pytest

from app.core.cache import ExecutionCache
from app.core.error_handling import (
    OOM_HINTS,
    NodeOOMError,
    is_out_of_memory,
    memory_digest,
    release_cached_memory,
)
from app.core.graph_engine import execute_graph
from app.core.node_base import BaseNode, DataType, PortDefinition
from app.core.node_registry import registry


OOM_MESSAGE = (
    "CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total "
    "capacity of 15.99 GiB of which 0 bytes is free.")

#: `torch.OutOfMemoryError` arrived in torch 2.5 and the declared floor is
#: lower, so the tests raise whichever shape the INSTALLED torch actually
#: produces. That is not a workaround -- it is the same reason
#: `is_out_of_memory` checks the type and the message: on an older build a
#: CUDA allocation failure really is a bare `RuntimeError`, and the handler
#: has to work there too.
_TORCH_OOM = getattr(__import__("torch"), "OutOfMemoryError", None)
has_oom_type = _TORCH_OOM is not None
requires_oom_type = pytest.mark.skipif(
    not has_oom_type, reason="torch < 2.5 has no OutOfMemoryError type")


def _cuda_oom() -> BaseException:
    """The exception this torch raises when a CUDA allocation does not fit."""
    return (_TORCH_OOM or RuntimeError)(OOM_MESSAGE)


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


@requires_oom_type
def test_the_dedicated_torch_oom_type_is_recognised():
    """By TYPE, not by message -- the message match is the fallback."""
    assert is_out_of_memory(_TORCH_OOM("no message about memory here")) is True


def test_both_spellings_of_the_oom_class_are_recognised():
    """``torch.cuda.OutOfMemoryError`` is the one that exists on 2.0-2.4.

    That range is exactly what this project's declared floor still admits,
    so checking only the 2.5+ ``torch.OutOfMemoryError`` spelling would
    leave those builds falling through to the MESSAGE match -- which is the
    fallback, not the answer. On 2.5+ the two names are one object and this
    asserts the same thing twice, which costs nothing and keeps the
    guarantee pinned if they ever diverge again.
    """
    import torch

    for spelling in ("OutOfMemoryError",):
        cls = getattr(torch, spelling, None)
        if isinstance(cls, type):
            assert is_out_of_memory(cls("silent")) is True
        cuda_cls = getattr(torch.cuda, spelling, None)
        if isinstance(cuda_cls, type):
            assert is_out_of_memory(cuda_cls("silent")) is True
    # At least one of the two must exist on any supported build.
    assert (isinstance(getattr(torch, "OutOfMemoryError", None), type)
            or isinstance(getattr(torch.cuda, "OutOfMemoryError", None), type))


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
