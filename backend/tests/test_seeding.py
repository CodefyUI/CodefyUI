"""Reproducibility: one seed in, the same loss curve out (core#134).

The acceptance criterion this file exists for is deliberately empirical --
"two runs with the same seed produce bitwise-identical loss curves on CPU,
and different seeds differ". Nothing here asserts that a seeding CALL was
made; every test runs a real graph through the real engine and compares the
numbers that came out, because a seeded run that still drifts is exactly the
failure a call-count assertion cannot see.

The fixture dataset is built from ``torch.arange`` -- no randomness at all --
so the ONLY sources of entropy in the graph are the ones seeding has to
control: ``SequentialModel``'s weight init and ``DataLoader``'s shuffle
order. If either escaped the seed, these tests fail.
"""

from __future__ import annotations

import asyncio
import os
import threading
from typing import Any

import pytest
import torch

from app.core.execution_context import ExecutionContext
from app.core.graph_engine import execute_graph
from app.core.node_base import BaseNode, DataType, PortDefinition
from app.core.node_registry import registry
from app.core.seeding import (
    CUBLAS_WORKSPACE_CONFIG,
    MAX_SEED,
    apply_determinism,
    determinism_depth,
    deterministic_scope,
    derive_seed,
    make_generator,
    make_worker_init_fn,
    seed_rngs,
)

FIXTURE_NODE = "_SeedFixtureDataset"

#: Small enough that three epochs cost milliseconds, wide enough that two
#: differently-initialised models cannot accidentally agree to 7 decimals.
_SAMPLES = 64
_FEATURES = 8
_CLASSES = 3


class _SeedFixtureDatasetNode(BaseNode):
    """A completely deterministic supervised dataset. No RNG whatsoever.

    Test-local on purpose: every shipped dataset node either downloads
    something or generates from its own seed, and both would muddy the
    question this file asks.
    """

    NODE_NAME = FIXTURE_NODE
    CATEGORY = "Data"
    DESCRIPTION = "Fixed tensor dataset for reproducibility tests."

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="dataset", data_type=DataType.DATASET)]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        features = (
            torch.arange(_SAMPLES * _FEATURES, dtype=torch.float32)
            .reshape(_SAMPLES, _FEATURES)
            / (_SAMPLES * _FEATURES)
        )
        targets = torch.arange(_SAMPLES) % _CLASSES
        return {"dataset": torch.utils.data.TensorDataset(features, targets)}


@pytest.fixture(autouse=True)
def _fixture_node():
    """Register the fixture node for this module only, then take it back."""
    registry._nodes[FIXTURE_NODE] = _SeedFixtureDatasetNode
    yield
    registry._nodes.pop(FIXTURE_NODE, None)


_MODEL_SPEC = (
    '{"version":2,'
    '"nodes":['
    '{"id":"in","type":"Input","ports":[{"id":"p_x","name":"x"}]},'
    f'{{"id":"l1","type":"Linear","params":{{"in_features":{_FEATURES},"out_features":16}}}},'
    '{"id":"r1","type":"ReLU"},'
    f'{{"id":"l2","type":"Linear","params":{{"in_features":16,"out_features":{_CLASSES}}}}},'
    '{"id":"out","type":"Output","ports":[{"id":"p_y","name":"y"}]}'
    '],'
    '"edges":['
    '{"id":"e1","source":"in","sourceHandle":"p_x","target":"l1"},'
    '{"id":"e2","source":"l1","target":"r1"},'
    '{"id":"e3","source":"r1","target":"l2"},'
    '{"id":"e4","source":"l2","target":"out","targetHandle":"p_y"}'
    ']}'
)


def _edge(source: str, source_handle: str, target: str, target_handle: str) -> dict:
    return {
        "id": f"{source}.{source_handle}->{target}.{target_handle}",
        "source": source, "sourceHandle": source_handle,
        "target": target, "targetHandle": target_handle,
    }


def _training_graph(*, epochs: int = 3, loop_params: dict | None = None,
                    loss_params: dict | None = None) -> tuple[list, list]:
    """A real MLP training graph: fixture data -> loader -> train."""
    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {"id": "data", "type": FIXTURE_NODE, "data": {"params": {}}},
        {"id": "loader", "type": "DataLoader",
         "data": {"params": {"batch_size": 8, "shuffle": True}}},
        {"id": "model", "type": "SequentialModel",
         "data": {"params": {"layers": _MODEL_SPEC}}},
        {"id": "opt", "type": "Optimizer",
         "data": {"params": {"type": "SGD", "lr": 0.1}}},
        {"id": "loss", "type": "Loss",
         "data": {"params": {"type": "CrossEntropyLoss", **(loss_params or {})}}},
        {"id": "train", "type": "TrainingLoop",
         "data": {"params": {"epochs": epochs, "device": "cpu",
                             **(loop_params or {})}}},
    ]
    edges = [
        {"id": "t1", "source": "start", "target": "data",
         "sourceHandle": "trigger", "type": "trigger"},
        {"id": "t2", "source": "start", "target": "model",
         "sourceHandle": "trigger", "type": "trigger"},
        {"id": "t3", "source": "start", "target": "loss",
         "sourceHandle": "trigger", "type": "trigger"},
        _edge("data", "dataset", "loader", "dataset"),
        _edge("loader", "dataloader", "train", "dataloader"),
        _edge("model", "model", "opt", "model"),
        _edge("model", "model", "train", "model"),
        _edge("opt", "optimizer", "train", "optimizer"),
        _edge("loss", "loss_fn", "train", "loss_fn"),
    ]
    return nodes, edges


async def _run(seed: int | None, **graph_kwargs) -> list[float]:
    """Execute the graph once and return the per-epoch training losses."""
    nodes, edges = _training_graph(**graph_kwargs)
    context = ExecutionContext(device="cpu", seed=seed)
    outputs = await execute_graph(nodes, edges, context=context)
    return outputs["train"]["losses"].tolist()


# ── the acceptance criterion ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_same_seed_gives_bitwise_identical_loss_curves():
    """Two runs, one seed, identical floats -- not "close", EQUAL.

    ``==`` on floats is the point. An almost-equal comparison would pass on
    a run that drifts in the last few bits, which is precisely the state
    reproducibility is supposed to rule out.
    """
    first = await _run(1234)
    second = await _run(1234)

    assert len(first) == 3
    assert first == second, (
        f"same seed diverged:\n  run 1: {first}\n  run 2: {second}")


@pytest.mark.asyncio
async def test_different_seeds_give_different_loss_curves():
    """The seed has to MATTER, or the test above proves nothing.

    A graph that is accidentally deterministic (a fixed init, a loader that
    never really shuffles) would satisfy the equality test above while being
    completely unseeded. This is the control.
    """
    first = await _run(1234)
    other = await _run(4321)

    assert first != other, f"different seeds agreed exactly: {first}"


@pytest.mark.asyncio
async def test_unseeded_runs_are_not_forced_to_agree():
    """No seed means no promise -- the old behaviour is untouched.

    Two unseeded runs re-initialise the model from whatever torch's global
    state happens to be, so they differ. Asserting this keeps the seeding
    path from quietly becoming "always seeded with 0".
    """
    first = await _run(None)
    second = await _run(None)

    assert first != second


@pytest.mark.asyncio
@pytest.mark.parametrize("seed,expected_workers", [(7, 1), (None, 4)])
async def test_a_seeded_run_serialises_node_execution(
    monkeypatch, seed, expected_workers,
):
    """A seeded run drops to one worker; an unseeded one keeps its four.

    Per-node seeding writes the PROCESS-GLOBAL RNGs, so concurrency would
    let one node reseed another mid-execute. This is the mechanism the
    equality test above depends on, asserted directly so a future change to
    the scheduler cannot silently remove it.

    ``monkeypatch`` rather than a manual save/restore: the target is an
    attribute of the real ``asyncio`` module, and a test that failed before
    its ``finally`` would leave the whole session's event loop wrapped.
    """
    import asyncio as real_asyncio

    import app.core.graph_engine as engine

    observed: list[int] = []
    real_semaphore = real_asyncio.Semaphore

    def _spy(value):
        observed.append(value)
        return real_semaphore(value)

    monkeypatch.setattr(engine.asyncio, "Semaphore", _spy)

    nodes, edges = _training_graph(epochs=1)
    context = ExecutionContext(device="cpu", seed=seed, max_workers=4)
    await execute_graph(nodes, edges, context=context)

    assert observed and observed[0] == expected_workers


@pytest.mark.asyncio
async def test_seed_survives_a_reordered_graph():
    """Node ids, not execution order, decide what each node draws.

    Shuffling the node list changes the order the engine walks ties in. With
    a per-node derived seed the numbers must not move; with a single
    up-front ``torch.manual_seed`` they would.
    """
    nodes, edges = _training_graph()
    reordered = [nodes[0]] + list(reversed(nodes[1:]))

    baseline = ExecutionContext(device="cpu", seed=99)
    shuffled = ExecutionContext(device="cpu", seed=99)

    first = (await execute_graph(nodes, edges, context=baseline))["train"]["losses"]
    second = (await execute_graph(reordered, edges, context=shuffled))["train"]["losses"]

    assert first.tolist() == second.tolist()


# ── the primitives ────────────────────────────────────────────────────────


def test_derive_seed_is_stable_and_label_separated():
    assert derive_seed(42, "a") == derive_seed(42, "a")
    assert derive_seed(42, "a") != derive_seed(42, "b")
    assert derive_seed(42, "a") != derive_seed(43, "a")
    assert 0 <= derive_seed(42, "a") <= MAX_SEED


def test_derive_seed_does_not_depend_on_python_hash_randomisation():
    """The value is pinned, so a PYTHONHASHSEED-sensitive rewrite is caught.

    ``hash()`` differs per process; a run seeded today and repeated tomorrow
    would then differ for no visible reason. Freezing one known output makes
    that regression impossible to introduce silently.
    """
    assert derive_seed(0, "node-1") == 3878883895


def test_seed_rngs_makes_all_three_generators_reproducible():
    import random

    import numpy as np

    seed_rngs(2026)
    drawn = (random.random(), float(np.random.rand()), torch.rand(1).item())
    seed_rngs(2026)
    again = (random.random(), float(np.random.rand()), torch.rand(1).item())

    assert drawn == again


def test_seed_rngs_ignores_none():
    """``None`` must be a no-op, not "seed with 0"."""
    seed_rngs(12345)
    before = torch.rand(1).item()
    seed_rngs(12345)
    seed_rngs(None)
    assert torch.rand(1).item() == before


def test_make_generator_is_reproducible_and_independent():
    first = make_generator(derive_seed(5, "loader"))
    second = make_generator(derive_seed(5, "loader"))
    assert torch.randperm(16, generator=first).tolist() == \
        torch.randperm(16, generator=second).tolist()
    assert make_generator(None) is None


def test_worker_init_fn_gives_each_worker_its_own_stream():
    init = make_worker_init_fn(11)
    assert init is not None

    def _draw(worker_id: int) -> float:
        init(worker_id)
        return torch.rand(1).item()

    assert _draw(0) == _draw(0)
    assert _draw(0) != _draw(1)
    assert make_worker_init_fn(None) is None


def test_apply_determinism_is_warn_only():
    """Strict mode would kill a run on the first non-deterministic op.

    Asserted through torch's own accessor rather than a mock: the flag has
    to be really set, and it has to be the forgiving variant.
    """
    previously = torch.are_deterministic_algorithms_enabled()
    try:
        apply_determinism(True)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        # False never turns it back off -- it is a process-wide setting this
        # function does not own.
        apply_determinism(False)
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(previously)


# ── the wiring ────────────────────────────────────────────────────────────


def test_execution_context_derives_and_generates():
    seeded = ExecutionContext(seed=3)
    assert seeded.derive_seed("x") == derive_seed(3, "x")
    assert seeded.make_generator("x") is not None

    unseeded = ExecutionContext()
    assert unseeded.derive_seed("x") is None
    assert unseeded.make_generator("x") is None


# ── determinism is scoped to the run, not latched (#188 review, I4) ──────


class _DeterminismProbeNode(BaseNode):
    """Records what torch's determinism flag was WHILE the graph ran."""

    NODE_NAME = "_DeterminismProbe"
    CATEGORY = "Utility"
    DESCRIPTION = "Test node: samples the deterministic flag mid-run."

    #: ``(enabled, warn_only)`` sampled INSIDE the run. Both are restored on
    #: the way out, so neither can be read afterwards.
    seen: list[tuple[bool, bool]] = []

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        _DeterminismProbeNode.seen.append((
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
        ))
        return {"value": None}


@pytest.fixture
def _determinism_probe():
    registry._nodes["_DeterminismProbe"] = _DeterminismProbeNode
    _DeterminismProbeNode.seen = []
    previously = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        yield _DeterminismProbeNode
    finally:
        torch.use_deterministic_algorithms(previously, warn_only=warn_only)
        registry._nodes.pop("_DeterminismProbe", None)


def _probe_graph(loop_params: dict | None = None) -> tuple[list, list]:
    return (
        [
            {"id": "start", "type": "Start", "data": {"params": {}}},
            {"id": "probe", "type": "_DeterminismProbe", "data": {"params": {}}},
        ],
        [{"id": "t1", "source": "start", "target": "probe",
          "sourceHandle": "trigger", "type": "trigger"}],
    )


@pytest.mark.asyncio
async def test_context_deterministic_flag_reaches_torch(_determinism_probe):
    """It must be ON while the graph runs — sampled by a node, not after."""
    nodes, edges = _probe_graph()
    await execute_graph(
        nodes, edges,
        context=ExecutionContext(device="cpu", deterministic=True))

    # warn_only, deliberately: strict mode would kill the run on the first
    # op with no deterministic kernel.
    assert _determinism_probe.seen == [(True, True)]


@pytest.mark.asyncio
async def test_determinism_is_handed_back_when_the_run_ends(_determinism_probe):
    """A deterministic run must not latch the whole server process.

    Before the #188 review ``apply_determinism(False)`` was a deliberate
    no-op, so one run turned determinism on for every LATER run until
    restart. The setting is a property of a run; it now dies with it.
    """
    nodes, edges = _probe_graph()
    await execute_graph(
        nodes, edges,
        context=ExecutionContext(device="cpu", deterministic=True))
    assert not torch.are_deterministic_algorithms_enabled()

    # The next run, which asked for nothing, must compute the old way.
    _determinism_probe.seen = []
    await execute_graph(nodes, edges, context=ExecutionContext(device="cpu"))
    assert _determinism_probe.seen == [(False, False)]


@pytest.mark.asyncio
async def test_a_node_cannot_latch_determinism_past_its_own_run(
    _determinism_probe,
):
    """``TrainingLoop.deterministic`` is a NODE param on a saved graph.

    Opening someone else's graph must not change how every later run in the
    server computes — which is why the scope is entered unconditionally
    rather than only when the run asked for determinism.
    """
    nodes, edges = _training_graph(
        epochs=1, loop_params={"deterministic": True})
    await execute_graph(nodes, edges, context=ExecutionContext(device="cpu"))

    assert not torch.are_deterministic_algorithms_enabled()


@pytest.mark.asyncio
async def test_determinism_restores_an_already_on_setting(_determinism_probe):
    """Restore means "back to what it was", not "off"."""
    torch.use_deterministic_algorithms(True, warn_only=True)
    nodes, edges = _probe_graph()
    await execute_graph(nodes, edges, context=ExecutionContext(device="cpu"))

    assert torch.are_deterministic_algorithms_enabled()


# ── overlapping runs must not latch it either (#188 re-review, D1) ───────


def test_overlapping_scopes_restore_what_the_first_one_found():
    """The exact interleaving the re-review measured, made harmless.

    ``enter(True)`` reads "off" and sets on; ``enter(False)`` then reads
    "on"; ``exit(True)`` restores off; ``exit(False)`` puts its poisoned
    reading back — and determinism was on for every later run in that
    server, forever. Both runs are unseeded in the report, so neither takes
    the exclusion gate, and the cpu queue admits two.

    Refcounting is what fixes it: the baseline belongs to the first scope
    open, and only the last one to close puts it back.
    """
    previously = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    outer = deterministic_scope(True)
    inner = deterministic_scope(False)
    try:
        outer.__enter__()
        inner.__enter__()
        assert torch.are_deterministic_algorithms_enabled()
        assert determinism_depth() == 2

        outer.__exit__(None, None, None)
        # The run that asked for determinism has finished, but the run
        # sharing the process has not: taking it away here would change how
        # THAT run computes half way through.
        assert torch.are_deterministic_algorithms_enabled()
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == CUBLAS_WORKSPACE_CONFIG

        inner.__exit__(None, None, None)
        assert not torch.are_deterministic_algorithms_enabled()
        assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
        assert determinism_depth() == 0
    finally:
        torch.use_deterministic_algorithms(previously, warn_only=warn_only)
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


def test_a_raising_scope_still_gives_its_count_back():
    """A leaked depth would mean determinism is never restored again."""
    with pytest.raises(RuntimeError):
        with deterministic_scope(True):
            raise RuntimeError("boom")

    assert determinism_depth() == 0
    assert not torch.are_deterministic_algorithms_enabled()


@pytest.mark.asyncio
async def test_two_overlapping_runs_do_not_latch_determinism(
    _determinism_probe,
):
    """The same interleaving, driven through two real overlapping runs.

    One run asks for determinism, the other does not, and they are made to
    overlap the way the cpu queue's depth of 2 lets them: the deterministic
    run finishes FIRST, which is the order that used to leave the flag on
    for the life of the process.

    The plain run is dragged into its neighbour's determinism while they
    overlap — unavoidable for a process-global torch setting, and bounded:
    it cannot outlive the last run that wanted it. That is what this
    asserts.
    """
    deterministic_entered = threading.Event()
    plain_entered = threading.Event()
    deterministic_finished = threading.Event()
    seen: dict[str, bool] = {}

    class _DeterministicHalf(BaseNode):
        NODE_NAME = "_OverlapDeterministic"
        CATEGORY = "Test"
        DESCRIPTION = "Waits until the plain run is inside its scope too."

        @classmethod
        def define_inputs(cls):
            return []

        @classmethod
        def define_outputs(cls):
            return [PortDefinition(name="value", data_type=DataType.ANY)]

        def execute(self, inputs, params):
            deterministic_entered.set()
            assert plain_entered.wait(timeout=10), "the plain run never started"
            seen["deterministic"] = torch.are_deterministic_algorithms_enabled()
            return {"value": None}

    class _PlainHalf(_DeterministicHalf):
        NODE_NAME = "_OverlapPlain"
        DESCRIPTION = "Outlives the deterministic run, then finishes."

        def execute(self, inputs, params):
            plain_entered.set()
            assert deterministic_entered.wait(timeout=10)
            assert deterministic_finished.wait(timeout=10), (
                "the deterministic run never finished")
            seen["plain"] = torch.are_deterministic_algorithms_enabled()
            return {"value": None}

    registry._nodes[_DeterministicHalf.NODE_NAME] = _DeterministicHalf
    registry._nodes[_PlainHalf.NODE_NAME] = _PlainHalf

    def _graph(node_type: str):
        return (
            [{"id": "start", "type": "Start", "data": {"params": {}}},
             {"id": "n", "type": node_type, "data": {"params": {}}}],
            [{"id": "t1", "source": "start", "target": "n",
              "sourceHandle": "trigger", "type": "trigger"}],
        )

    async def _deterministic_run():
        nodes, edges = _graph(_DeterministicHalf.NODE_NAME)
        try:
            await execute_graph(
                nodes, edges,
                context=ExecutionContext(device="cpu", deterministic=True))
        finally:
            deterministic_finished.set()

    try:
        plain_nodes, plain_edges = _graph(_PlainHalf.NODE_NAME)
        await asyncio.gather(
            _deterministic_run(),
            execute_graph(plain_nodes, plain_edges,
                          context=ExecutionContext(device="cpu")),
        )
    finally:
        registry._nodes.pop(_DeterministicHalf.NODE_NAME, None)
        registry._nodes.pop(_PlainHalf.NODE_NAME, None)

    assert seen["deterministic"] is True, "the run that asked for it lost it"
    assert seen["plain"] is True, "borrowed while they overlapped"
    assert not torch.are_deterministic_algorithms_enabled(), (
        "determinism outlived both runs")
    assert determinism_depth() == 0


# ── the engine's hand-entered scope must be unskippable (#190) ──────────
#
# ``execute_graph`` enters the scope with a bare ``__enter__`` and pairs it
# in a ``finally``, which means the pairing is not the language's job and
# every statement in between is a place it can be lost. Under refcounting a
# lost exit does not cost one run its restore -- it strands the depth above
# zero for the life of the PROCESS, which suppresses every later scope's
# baseline capture as well.


@pytest.fixture
def _determinism_depth_isolated():
    """Contain a leak so a FAILING test here does not poison the session.

    The failure under test is exactly "the depth never comes back", and a
    stranded depth would turn a dozen unrelated tests red for a reason none
    of them names. The assertions still run first -- this only cleans up
    after them.
    """
    import app.core.seeding as seeding

    try:
        yield
    finally:
        seeding._DETERMINISM_DEPTH = 0
        seeding._DETERMINISM_BASELINE = None


def _assert_scope_closed_before_the_exception_left(depth_at_raise: int) -> None:
    """Both tests below assert INSIDE their ``except``, and here is why.

    Measured while writing them: with the guard removed the depth really is
    1 at the moment the exception is caught -- and 0 again as soon as the
    ``except`` block ends. ``deterministic_scope`` is a generator-based
    context manager, so once the traceback (which holds ``execute_graph``'s
    frame, which holds the generator) is released, CPython's refcounting
    finalises the suspended generator, which throws GeneratorExit at the
    ``yield`` and runs the very ``finally`` that was skipped.

    That is worth writing down, because it makes #190's severity claim --
    "determinism is never restored again in that process" -- too strong for
    CPython as it stands. What is true is narrower and still worth fixing:
    the scope stays open for as long as anything holds the traceback (a
    stored exception on a run row, a task that nobody retrieved, a
    ``sys.last_value``), it depends on an interpreter detail no code here
    states, and it evaporates on any implementation without prompt
    refcounting.

    So the property to pin is the one that does not depend on the garbage
    collector: the scope was closed BEFORE the exception left the engine.
    Asserting after ``pytest.raises`` would pass either way and prove
    nothing -- it did, until this was measured.
    """
    assert depth_at_raise == 0, (
        "the scope was still open when the exception left execute_graph; "
        "it was closed later only because the traceback was released and "
        "the generator got finalised")


@pytest.mark.asyncio
async def test_a_pump_that_cannot_be_started_still_gives_the_depth_back(
    _determinism_probe, _determinism_depth_isolated, monkeypatch,
):
    """The pump task is created BETWEEN ``__enter__`` and the guarded body.

    ``asyncio.create_task`` raises ``RuntimeError`` on a loop that is
    closing, which is precisely when a server is being torn down and least
    able to notice that determinism has been welded on.

    Only the pump's own creation is failed -- matched on the task name --
    so nothing else in the run is disturbed by the patch.
    """
    from app.core import graph_engine as engine

    real_create_task = asyncio.create_task

    def _fail_the_pump(coro, *args, name=None, **kwargs):
        if name and name.startswith("outbox-pump"):
            coro.close()  # or Python warns it was never awaited
            raise RuntimeError("event loop is closed")
        return real_create_task(coro, *args, name=name, **kwargs)

    monkeypatch.setattr(engine.asyncio, "create_task", _fail_the_pump)

    nodes, edges = _probe_graph()
    try:
        await execute_graph(
            nodes, edges,
            context=ExecutionContext(device="cpu", deterministic=True))
    except RuntimeError as exc:
        assert "event loop is closed" in str(exc)
        _assert_scope_closed_before_the_exception_left(determinism_depth())
    else:
        pytest.fail("the pump was supposed to fail to start")

    assert not torch.are_deterministic_algorithms_enabled()


@pytest.mark.asyncio
async def test_a_pump_that_dies_mid_run_still_gives_the_depth_back(
    _determinism_probe, _determinism_depth_isolated,
):
    """``await pump_task`` re-raises whatever killed the pump.

    ``_pump`` catches ``Exception`` around ``_deliver`` but NOT around its
    own ``outbox.wait()``, and catches nothing at all above ``Exception``,
    so this is narrow rather than closed. The exit used to be that await's
    PEER statement, so anything coming back out of it went straight past.

    Raising from ``wait`` is the honest injection point: it is the one line
    of the pump with no handler over it.
    """
    from app.core.execution_context import EventOutbox

    class _ExplodingOutbox(EventOutbox):
        async def wait(self) -> None:
            raise RuntimeError("pump died")

    nodes, edges = _probe_graph()
    context = ExecutionContext(device="cpu", deterministic=True,
                               outbox=_ExplodingOutbox())
    try:
        await execute_graph(nodes, edges, context=context)
    except RuntimeError as exc:
        assert "pump died" in str(exc)
        _assert_scope_closed_before_the_exception_left(determinism_depth())
    else:
        pytest.fail("the pump was supposed to die")

    assert not torch.are_deterministic_algorithms_enabled()


def test_deterministic_scope_restores_the_cublas_env_var():
    """The env var leaks into every subprocess spawned afterwards.

    It does nothing in THIS process once the CUDA context exists, but a
    DataLoader worker started later would inherit it.
    """
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    with deterministic_scope(True):
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == CUBLAS_WORKSPACE_CONFIG
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    try:
        with deterministic_scope(True):
            pass
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    finally:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
