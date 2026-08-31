"""Sweep compilation and storage (#140).

The REST surface has its own module (test_api_sweeps.py); the probe node is
duplicated there on purpose, the way test_api_runs.py duplicates
test_run_queue.py's, so test modules stay independent.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from typing import Any

import pytest

from app.core.db import Database
from app.core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)
from app.core.node_registry import registry
from app.core.run_store import MetricPoint, RunProvenance, RunStore
from app.core.sweep_compiler import (
    SweepCompileError,
    compile_sweep,
    planner_prng,
    sample_unique_ranks,
)
from app.core.sweep_store import (
    SWEEP_STATE_CANCELLING,
    SWEEP_STATE_FAILED,
    SWEEP_STATE_FINISHED,
    SWEEP_STATE_RUNNING,
    HarvestEntry,
    SweepStore,
    SweepVariant,
    rank_variants,
    variant_is_terminal,
)


class _SweepProbeNode(BaseNode):
    """Every param type a sweep can meet, plus the ones it must refuse."""

    NODE_NAME = "_SweepProbe"
    CATEGORY = "Test"
    DESCRIPTION = "Logs a val_loss derived from its swept params"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="lr", param_type=ParamType.FLOAT,
                            default=0.001, min_value=0.0, max_value=1.0),
            ParamDefinition(name="weight_decay", param_type=ParamType.FLOAT,
                            default=0.0),
            ParamDefinition(name="epochs", param_type=ParamType.INT,
                            default=1),
            ParamDefinition(name="shuffle", param_type=ParamType.BOOL,
                            default=False),
            ParamDefinition(name="optimizer", param_type=ParamType.SELECT,
                            default="adam", options=["adam", "sgd"]),
            ParamDefinition(name="note", param_type=ParamType.STRING,
                            default=""),
            ParamDefinition(name="api_key", param_type=ParamType.SECRET,
                            default=""),
            ParamDefinition(name="weights", param_type=ParamType.MODEL_FILE,
                            default=""),
        ]

    def execute(self, inputs: dict[str, Any],
                params: dict[str, Any]) -> dict[str, Any]:
        return {"value": inputs.get("value")}


@pytest.fixture(autouse=True)
def _register_probe():
    registry._nodes["_SweepProbe"] = _SweepProbeNode
    yield
    registry._nodes.pop("_SweepProbe", None)


# ── helpers ───────────────────────────────────────────────────────────────


def _base_graph(*extra_nodes: dict[str, Any]) -> dict[str, Any]:
    return {
        "nodes": [
            {"id": "start", "type": "Start", "data": {"params": {}}},
            {"id": "probe", "type": "_SweepProbe", "data": {"params": {}}},
            *extra_nodes,
        ],
        "edges": [],
    }


def _compile(spec, graph=None, *, max_runs=32, max_params=4, max_domain=32):
    return compile_sweep(graph if graph is not None else _base_graph(), spec,
                         max_runs=max_runs, max_params=max_params,
                         max_domain=max_domain)


def _values(param: str, values: list[Any], node_id: str = "probe"):
    return {"node_id": node_id, "param": param, "values": values}


def _range(param: str, node_id: str = "probe", **fields: Any):
    return {"node_id": node_id, "param": param, "range": fields}


def _grid(*params: dict[str, Any]) -> dict[str, Any]:
    return {"method": "grid", "params": list(params)}


def _domain(param_spec: dict[str, Any]) -> tuple[Any, ...]:
    return _compile(_grid(param_spec)).params[0].domain


def _refusal(spec, graph=None, **kw) -> str:
    with pytest.raises(SweepCompileError) as excinfo:
        _compile(spec, graph, **kw)
    return str(excinfo.value)


# ── determinism (spec 2.10, 9.5) ──────────────────────────────────────────


def test_mulberry32_matches_the_javascript_stream():
    """Raw uint32 streams, computed from optimizer.ts:307-317's algorithm.
    The tightest fixture there is: independent of the sampler, and the one
    that catches a float-returning port immediately."""
    def draws(seed: int, n: int) -> list[int]:
        nxt = planner_prng(seed)
        return [nxt() for _ in range(n)]

    assert draws(0, 4) == [1144304738, 1416247, 958946056, 627933444]
    assert draws(42, 4) == [2581720956, 1925393290, 3661312704, 2876485805]
    assert draws(123456789, 6) == [1107202814, 4169434471, 3372958138,
                                   885470128, 1301683845, 3208624240]


def test_mulberry32_returns_a_raw_uint32_not_a_float():
    """The JS original has no `/ 4294967296`; its only consumer is
    `next() % remaining`. A textbook port that divides diverges on some
    draws and silently produces a different variant list (spec 2.9)."""
    nxt = planner_prng(2026)
    for _ in range(64):
        value = nxt()
        assert isinstance(value, int) and not isinstance(value, bool)
        assert 0 <= value < 2 ** 32


def test_sample_unique_ranks_matches_the_copilot_fixture():
    # optimizer.test.ts:192-220 uses domains [5, 2] with a baseline at
    # index 3, so it draws 7 of the 9 non-baseline slots and shifts them:
    # [9, 8, 5, 7, 0, 6, 2] after `rank >= 3 ? rank + 1 : rank`. v1 has no
    # baseline (spec 3.1), so it uses the unshifted form below.
    assert sample_unique_ranks(9, 7, 123456789) == [8, 7, 4, 6, 0, 5, 2]


def test_sample_unique_ranks_is_a_permutation_when_exhaustive():
    """count == total proves the sparse Fisher-Yates bookkeeping rather
    than just the PRNG."""
    drawn = sample_unique_ranks(6, 6, 7)
    assert drawn == [4, 0, 2, 3, 1, 5]
    assert sorted(drawn) == list(range(6))


def test_the_same_seed_reproduces_the_identical_variant_list():
    """#140 acceptance criterion 2."""
    spec = {"method": "random", "seed": 123456789, "samples": 6,
            "params": [_values("lr", [0.1, 0.01, 0.001, 0.0001, 0.5]),
                       _values("epochs", [1, 2])]}
    first = _compile(spec)
    second = _compile(spec)
    assert [v.assignment for v in first.variants] == \
           [v.assignment for v in second.variants]
    assert [v.domain_index for v in first.variants] == \
           [v.domain_index for v in second.variants]


def test_a_different_seed_produces_a_different_variant_list():
    # Inequality only, mirroring optimizer.test.ts:222-227 -- pinning a
    # second exact sequence buys nothing the first one has not already.
    def order(seed: int) -> list[int]:
        return [v.domain_index for v in _compile(
            {"method": "random", "seed": seed, "samples": 6,
             "params": [_values("lr", [0.1, 0.01, 0.001, 0.0001, 0.5]),
                        _values("epochs", [1, 2])]}).variants]

    assert order(123456789) != order(987654321)


def test_random_never_repeats_a_combination():
    compiled = _compile({"method": "random", "seed": 20260831, "samples": 5,
                         "params": [_values("lr", [0.1, 0.01, 0.001, 0.5]),
                                    _values("epochs", [1, 2, 3])]})
    indices = [v.domain_index for v in compiled.variants]
    assert len(set(indices)) == 5
    assert compiled.total_combinations == 12


def test_samples_above_the_space_is_refused():
    message = _refusal({"method": "random", "seed": 1, "samples": 9,
                        "params": [_values("epochs", [1, 2, 3])]})
    assert "9" in message and "3" in message


# ── the recorded planner seed (RULING 1, spec 2.2) ────────────────────────


def test_the_compiled_sweep_records_the_planner_seed():
    """The route fills sweeps.seed from the compiled object rather than
    re-reading and re-validating spec['seed'], so the seed the sampler used
    and the seed the row records cannot drift apart."""
    compiled = _compile({"method": "random", "seed": 20260831, "samples": 3,
                         "params": [_values("epochs", [1, 2, 3])]})
    assert compiled.seed == 20260831


def test_a_grid_without_a_seed_records_none():
    """A grid enumerates everything, so it consumes no seed. None is the
    honest record of 'the caller asked for nothing' -- not 0, which is a
    real seed a caller may have meant."""
    assert _compile(_grid(_values("epochs", [1, 2]))).seed is None


def test_a_grid_with_a_seed_records_it():
    """RULING 1: the seed is still recorded even though the planner never
    draws with it, because it is the base for per-variant execution seeds
    and because a stored spec should say what was asked for."""
    spec = _grid(_values("epochs", [1, 2]))
    spec["seed"] = 7
    assert _compile(spec).seed == 7


def test_the_recorded_seed_agrees_with_the_variant_list_it_produced():
    """The one with teeth: re-draw the ranks from the RECORDED seed with the
    public sampler and demand the compiler's own variant order back. A
    compiler that recorded one seed and sampled with another passes every
    other test in this section and fails this one."""
    spec = {"method": "random", "seed": 123456789, "samples": 6,
            "params": [_values("lr", [0.1, 0.01, 0.001, 0.0001, 0.5]),
                       _values("epochs", [1, 2])]}
    first = _compile(spec)
    second = _compile(spec)
    assert first.seed == second.seed == 123456789
    assert [v.assignment for v in first.variants] == \
           [v.assignment for v in second.variants]
    assert [v.domain_index for v in first.variants] == \
        sample_unique_ranks(first.total_combinations, 6, first.seed)


# ── grid shape (spec 3.1, 9.3) ────────────────────────────────────────────


def test_grid_cardinality_is_the_product_of_the_domains():
    """3x2 compiles exactly 6 variants -- no baseline subtraction.
    #140 acceptance criterion 1."""
    compiled = _compile(_grid(_values("lr", [0.0001, 0.001, 0.01]),
                              _values("weight_decay", [0.0, 0.0001])))
    assert compiled.total_combinations == 6
    assert len(compiled.variants) == 6
    assert [v.index for v in compiled.variants] == [0, 1, 2, 3, 4, 5]
    assert [v.domain_index for v in compiled.variants] == [0, 1, 2, 3, 4, 5]


def test_grid_varies_the_last_param_fastest():
    """Row-major / C order, so a comparison table is reproducible from the
    spec alone. The exact table is spec 3.1's, over this module's params."""
    compiled = _compile(_grid(_values("epochs", [3, 4]),
                              _values("shuffle", [False, True]),
                              _values("optimizer", ["adam", "sgd"])))
    assert [v.assignment for v in compiled.variants] == [
        (3, False, "adam"), (3, False, "sgd"),
        (3, True, "adam"), (3, True, "sgd"),
        (4, False, "adam"), (4, False, "sgd"),
        (4, True, "adam"), (4, True, "sgd"),
    ]


def test_the_cap_errors_and_never_truncates():
    message = _refusal(_grid(_values("lr", [0.1, 0.2, 0.3, 0.4, 0.5]),
                             _values("weight_decay", [0.0, 0.1, 0.2, 0.3, 0.4]),
                             _values("epochs", [1, 2, 3, 4, 5])))
    assert "125" in message and "32" in message
    assert "narrow the domains" in message


# ── range expansion (spec 2.5, 2.6, 9.3) ──────────────────────────────────


def test_linear_float_range_includes_both_endpoints_exactly():
    # `==` against the literals, never `is`: identity is meaningless on
    # floats after a round-trip and vacuously true for `float(x) is x`.
    assert _domain(_range("lr", min=0.0, max=1.0, count=5,
                          scale="linear", type="float")) == \
        (0.0, 0.25, 0.5, 0.75, 1.0)


def test_log_float_range_hits_the_decades_exactly():
    assert _domain(_range("lr", min=1e-4, max=1e-1, count=4,
                          scale="log", type="float")) == \
        (0.0001, 0.001, 0.01, 0.1)


def test_log_int_range_doubles():
    assert _domain(_range("epochs", min=16, max=128, count=4,
                          scale="log", type="int")) == (16, 32, 64, 128)


def test_int_range_rounds_half_away_from_zero():
    """The midpoint is chosen so the two rules DIFFER: 1..4 in 3 points
    interpolates [1.0, 2.5, 4.0], and Python's banker's `round(2.5)` is 2
    while half-away-from-zero is 3. A 1.5-style midpoint is identical under
    both rules and would satisfy a wrong implementation."""
    assert _domain(_range("epochs", min=1, max=4, count=3,
                          scale="linear", type="int")) == (1, 3, 4)


def test_a_collapsed_int_range_dedupes_and_shrinks_the_grid():
    compiled = _compile(_grid(_range("epochs", min=1, max=3, count=10,
                                     scale="linear", type="int")))
    assert compiled.params[0].domain == (1, 2, 3)
    assert compiled.total_combinations == 3
    assert len(compiled.variants) == 3


def test_count_one_returns_the_min():
    for scale in ("linear", "log"):
        assert _domain(_range("lr", min=0.01, max=0.5, count=1,
                              scale=scale, type="float")) == (0.01,)


def test_log_range_with_a_non_positive_min_is_refused():
    message = _refusal(_grid(_range("lr", min=0.0, max=0.1, count=3,
                                    scale="log", type="float")))
    assert "params[0]" in message and "positive min" in message


def test_duplicate_values_are_refused():
    assert "repeats" in _refusal(_grid(_values("epochs", [1, 1, 2])))


def test_non_finite_values_are_refused():
    """json.loads("NaN") is nan and json.loads("1e999") is inf, and both
    pass every other check in the spec before corrupting a stored row."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        message = _refusal(_grid(_values("lr", [0.1, bad])))
        assert "values[1]" in message and "finite" in message


def test_non_finite_range_bounds_are_refused():
    message = _refusal(_grid(_range("lr", min=0.1, max=float("inf"), count=3,
                                    scale="linear", type="float")))
    assert "bounds must be finite" in message


def test_a_huge_integer_in_values_is_refused_not_an_overflow():
    """Python ints are UNBOUNDED, so json.loads happily parses a 400-digit
    integer literal and `math.isfinite` RAISES OverflowError on it instead
    of returning False. Uncaught, that is a 500 from one crafted body on a
    surface whose whole contract is that a bad spec is a 400 -- the integer
    sibling of the json.loads("1e999") case, and the same defect class as
    test_a_non_dict_node_entry_is_a_400_not_a_500.

    Both an INT and a FLOAT param, because the value never reaches the
    param's type at all: the envelope refuses it before any address is
    resolved."""
    for param in ("epochs", "weight_decay"):
        message = _refusal(_grid(_values(param, [1, 10 ** 400])))
        assert "values[1]" in message and "finite" in message


def test_a_huge_integer_range_bound_is_refused_not_an_overflow():
    """Same hazard through the other door: range bounds are checked for
    finiteness before anything else in the range envelope."""
    message = _refusal(_grid(_range("weight_decay", min=1, max=10 ** 400,
                                    count=3, scale="linear", type="float")))
    assert "bounds must be finite" in message


def test_a_large_but_representable_number_still_compiles():
    """The guard refuses what cannot become a float AT ALL, not what is
    merely large -- otherwise it would be a blanket size limit nobody asked
    for. 10**300 and 1e308 are both ordinary finite floats and a sweep may
    use them."""
    compiled = _compile(_grid(_values("epochs", [1, 10 ** 300]),
                              _values("weight_decay", [1e308])))
    assert compiled.params[0].domain == (1, 10 ** 300)
    assert compiled.params[1].domain == (1e308,)


# ── the type-acceptance matrix (spec 3.4.1, 9.3) ──────────────────────────


def test_a_float_param_accepts_an_integer_and_stores_a_float():
    compiled = _compile(_grid(_values("lr", [1, 0.1])))
    assert compiled.params[0].domain == (1.0, 0.1)
    stored = compiled.variants[0].graph["nodes"][1]["data"]["params"]["lr"]
    assert stored == 1.0 and isinstance(stored, float)


def test_an_int_param_accepts_a_whole_float_and_stores_an_int():
    compiled = _compile(_grid(_values("epochs", [2.0])))
    stored = compiled.variants[0].graph["nodes"][1]["data"]["params"]["epochs"]
    assert stored == 2 and isinstance(stored, int)


def test_true_is_not_a_number_for_an_int_or_float_param():
    """bool subclasses int and True == 1, so a naive isinstance check would
    silently accept `true` as 1 -- not a 400, and not a domain of size 2."""
    assert "contains True" in _refusal(_grid(_values("lr", [1, True])))
    assert "contains True" in _refusal(_grid(_values("epochs", [1, True])))


def test_a_bool_param_refuses_1_and_0():
    assert "bool param" in _refusal(_grid(_values("shuffle", [0, 1])))


def test_coercion_makes_the_domain_homogeneous_before_dedup():
    """Both coerce to 1.0, so this is a DUPLICATE error and not a domain of
    size 2 -- which proves the coerce/dedup/bounds ordering (rule 4)."""
    assert "repeats" in _refusal(_grid(_values("lr", [1, 1.0])))


# ── graph handling (spec 3.5, 9.3) ────────────────────────────────────────


def test_a_non_dict_node_entry_is_a_400_not_a_500():
    """normalize_graph validates that `nodes` is a non-empty list and
    nothing about its contents, so {"nodes": ["x"]} reaches the compiler
    intact and `node["id"]` on a str would be an uncaught 500."""
    graph = {"nodes": ["x", {"id": "probe", "type": "_SweepProbe",
                             "data": {"params": {}}}], "edges": []}
    message = _refusal(_grid(_values("lr", [0.1], node_id="ghost")), graph)
    assert "no node with id 'ghost'" in message


def test_compile_never_mutates_the_base_graph():
    graph = _base_graph()
    before = copy.deepcopy(graph)
    _compile(_grid(_values("lr", [0.1, 0.2])), graph)
    assert graph == before


def test_each_variant_graph_is_independent():
    compiled = _compile(_grid(_values("lr", [0.1, 0.2])))
    compiled.variants[0].graph["nodes"][1]["data"]["params"]["lr"] = 99.0
    assert compiled.variants[1].graph["nodes"][1]["data"]["params"]["lr"] == 0.2


def test_overrides_land_in_data_params():
    compiled = _compile(_grid(_values("lr", [0.0001, 0.001, 0.01]),
                              _values("weight_decay", [0.0, 0.0001])))
    params = compiled.variants[3].graph["nodes"][1]["data"]["params"]
    assert params == {"lr": 0.001, "weight_decay": 0.0001}


def test_a_param_absent_from_data_params_is_still_settable():
    """Nodes read params.get(name, default), so writing a key that was not
    there takes effect. Only unreachable CONTAINERS are refused."""
    graph = {"nodes": [{"id": "probe", "type": "_SweepProbe", "data": {}}],
             "edges": []}
    compiled = _compile(_grid(_values("lr", [0.25])), graph)
    assert compiled.variants[0].graph["nodes"][0]["data"]["params"] == \
        {"lr": 0.25}


# ── RULING 3: one test per reachability failure (spec 3.4, 9.4) ───────────


def test_unknown_node_id_is_refused():
    assert "no node with id 'nope'" in \
        _refusal(_grid(_values("lr", [0.1], node_id="nope")))


def test_duplicate_node_id_is_refused():
    graph = _base_graph({"id": "probe", "type": "_SweepProbe",
                         "data": {"params": {}}})
    message = _refusal(_grid(_values("lr", [0.1])), graph)
    assert "2 nodes with id 'probe'" in message
    assert "ids must be unique" in message


def test_preset_node_param_is_refused():
    graph = {"nodes": [{"id": "pipe", "type": "preset:Training Pipeline",
                        "data": {"params": {}}}], "edges": []}
    message = _refusal(_grid(_values("lr", [0.1], node_id="pipe")), graph)
    assert "preset instance" in message
    assert "data.internalParams" in message


def test_subgraph_node_param_is_refused():
    graph = {"nodes": [{"id": "blk", "type": "subgraph:blk",
                        "data": {"params": {}}}], "edges": []}
    message = _refusal(_grid(_values("lr", [0.1], node_id="blk")), graph)
    assert "subgraph instance" in message
    assert "cannot be overridden per instance" in message


def test_unknown_param_name_is_refused():
    message = _refusal(_grid(_values("momentum", [0.9])))
    assert "'probe'" in message and "'momentum'" in message


def test_secret_param_is_refused():
    """RULING 4 stores each variant's chosen params on the durable sweeps
    row, so a swept SECRET would be written there in the clear."""
    message = _refusal(_grid(_values("api_key", ["sk-1", "sk-2"])))
    assert "'probe.api_key' is a secret parameter" in message


def test_unsupported_param_type_is_refused():
    message = _refusal(_grid(_values("weights", ["a.pt", "b.pt"])))
    assert "param type 'model_file'" in message


def test_value_outside_min_value_is_refused():
    message = _refusal(_grid(_values("lr", [0.1, -1.0])))
    assert "'probe.lr' must be >= 0.0, got -1.0" in message


def test_select_value_not_in_options_is_refused():
    message = _refusal(_grid(_values("optimizer", ["adam", "rmsprop"])))
    assert "does not accept 'rmsprop'" in message
    assert "['adam', 'sgd']" in message


def test_wrong_scalar_type_for_an_int_param_is_refused():
    assert "int param but the domain contains 0.5" in \
        _refusal(_grid(_values("epochs", [1, 0.5])))


# ── SweepStore (spec 4.2, 4.3) ────────────────────────────────────────────


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "codefyui.db")
    database.connect()
    try:
        yield database
    finally:
        database.close()


@pytest.fixture
def sweeps(db):
    return SweepStore(db)


def _variant(index: int, **overrides) -> SweepVariant:
    fields = {"index": index, "domain_index": index, "run_id": None,
              "params": [{"node_id": "probe", "param": "lr",
                          "value": 0.1 * (index + 1)}],
              "seed": None, "objective": None, "status": None,
              "harvested_at": None}
    fields.update(overrides)
    return SweepVariant(**fields)


async def _new_sweep(store: SweepStore, count: int = 2, **overrides):
    fields = {"method": "grid", "seed": 123456789, "seed_variants": False,
              "spec": {"method": "grid", "params": []},
              "objective": {"metric": "val_loss", "direction": "minimize"},
              "variants": [_variant(i) for i in range(count)],
              "name": "lr sweep"}
    fields.update(overrides)
    return await store.create_sweep(**fields)


async def test_create_and_get_a_sweep_round_trip(sweeps):
    created = await _new_sweep(sweeps)
    assert created.state == SWEEP_STATE_RUNNING
    assert created.finished_at is None and created.error is None
    fetched = await sweeps.get_sweep(created.id)
    assert fetched == created
    assert fetched.seed_variants is False        # 0/1 comes back as a bool
    assert [v.index for v in fetched.variants] == [0, 1]
    assert fetched.variants[0].run_id is None
    assert await sweeps.get_sweep("nope") is None


async def test_create_sweep_assigns_each_variant_index_exactly_once(sweeps):
    """The store is the ONLY thing that can guarantee this.

    ``(sweep_id, sweep_variant)`` is an INDEX, not a constraint -- SQLite
    cannot add a UNIQUE column via ``ADD COLUMN`` (MIGRATION_004) -- so the
    database accepts two variant 3s without complaint, and every later
    reader of the sweep would then see one of them twice. ``index`` IS the
    entry's position in the list (spec 4.2), so the writer derives it and
    never trusts what it was handed; ``domain_index``, which is the
    caller's own datum, is left exactly as given.
    """
    created = await _new_sweep(sweeps, variants=[
        _variant(3, domain_index=7), _variant(3, domain_index=2),
        _variant(0, domain_index=5)])
    assert [v.index for v in created.variants] == [0, 1, 2]
    fetched = await sweeps.get_sweep(created.id)
    assert fetched.variants == created.variants   # returned == stored
    assert [v.domain_index for v in fetched.variants] == [7, 2, 5]
    # With a duplicate index in the blob this patch would have hit two
    # entries, and two children would claim the same sweep_variant.
    assert await sweeps.set_variant_run(created.id, 1, run_id="r1", seed=None)
    patched = await sweeps.get_sweep(created.id)
    assert [v.run_id for v in patched.variants] == [None, "r1", None]


async def test_a_non_finite_objective_never_reaches_the_blob(sweeps):
    """``_dumps``' ``allow_nan=False``, and why it must never be relaxed.

    Python's json emits the bare tokens ``NaN`` / ``Infinity``, which are a
    CPython extension and not JSON. Starlette renders every response with
    ``allow_nan=False``, so one of those inside the durable blob is a 500 on
    EVERY later read of the sweep -- permanently, because retention deletes
    finished runs but never a sweeps row. Refusing at the door costs one
    loud failure on the write instead, and the write is a single autocommit
    INSERT, so nothing is stored.
    """
    for bad, unwritten in ((float("nan"), "nan-sweep"),
                           (float("inf"), "inf-sweep")):
        with pytest.raises(ValueError, match="not JSON compliant"):
            await _new_sweep(sweeps, sweep_id=unwritten,
                             variants=[_variant(0, objective=bad)])
        assert await sweeps.get_sweep(unwritten) is None


async def test_set_variant_run_patches_one_entry_and_leaves_the_rest(sweeps):
    created = await _new_sweep(sweeps, count=3)
    assert await sweeps.set_variant_run(created.id, 1, run_id="r1", seed=42)
    fetched = await sweeps.get_sweep(created.id)
    assert fetched.variants[1].run_id == "r1"
    assert fetched.variants[1].seed == 42
    assert fetched.variants[0].run_id is None
    assert fetched.variants[2].run_id is None
    assert not await sweeps.set_variant_run(created.id, 9, run_id="x",
                                            seed=None)


async def test_concurrent_variant_patches_do_not_lose_each_other(sweeps):
    """The single-closure rule, asserted rather than merely documented.

    ``set_variant_run`` is the only read-modify-write in this module, and it
    is safe ONLY because its SELECT, its patch and its UPDATE all live in
    one ``fn(conn)``. Split into a ``Database.run`` read plus a separate
    ``Database.run`` write -- the tidy-looking refactor a maintainer reaches
    for when they notice ``get_sweep`` already does the read -- every call
    still returns True and every patch but the last is silently lost, because
    all of them read the same blob before any of them writes.

    Spec 5.2's submit loop patches one variant per submit, so a real sweep can
    have several of these in flight at once; a lost update there is a
    sweep-to-child link gone for good, reported as ``status: "missing"``
    forever while the child run sits in ``exec_runs`` with a correct
    ``sweep_id``.
    """
    created = await _new_sweep(sweeps, count=4)
    patched = await asyncio.gather(*(
        sweeps.set_variant_run(created.id, i, run_id=f"r{i}", seed=i)
        for i in range(4)))
    assert all(patched)
    fetched = await sweeps.get_sweep(created.id)
    assert [v.run_id for v in fetched.variants] == ["r0", "r1", "r2", "r3"]
    assert [v.seed for v in fetched.variants] == [0, 1, 2, 3]


async def test_mark_failed_records_the_error_and_stamps_finished_at(sweeps):
    created = await _new_sweep(sweeps)
    assert await sweeps.mark_failed(created.id, "run service is shutting down")
    fetched = await sweeps.get_sweep(created.id)
    assert fetched.state == SWEEP_STATE_FAILED
    assert fetched.error == "run service is shutting down"
    assert fetched.finished_at is not None


async def test_set_state_moves_a_running_sweep(sweeps):
    """The positive half, which the two negative tests never observe.

    Spec 7.3's cancel is built entirely on this call, so a ``set_state`` that
    quietly moves nothing would surface a task later as a cancelled sweep
    that keeps reporting ``running`` to the panel.
    """
    created = await _new_sweep(sweeps)
    assert await sweeps.set_state(created.id, SWEEP_STATE_CANCELLING)
    fetched = await sweeps.get_sweep(created.id)
    assert fetched.state == SWEEP_STATE_CANCELLING
    assert not await sweeps.set_state("nope", SWEEP_STATE_CANCELLING)


async def test_set_state_never_overwrites_failed(sweeps):
    """`failed` on a sweep means something went wrong with the SWEEP, not
    with the training, and a later cancel must not paper over it."""
    created = await _new_sweep(sweeps)
    await sweeps.mark_failed(created.id, "boom")
    assert not await sweeps.set_state(created.id, SWEEP_STATE_CANCELLING)
    assert (await sweeps.get_sweep(created.id)).state == SWEEP_STATE_FAILED


async def test_an_unknown_sweep_state_is_refused(sweeps):
    created = await _new_sweep(sweeps)
    with pytest.raises(ValueError, match="unknown sweep state"):
        await sweeps.set_state(created.id, "cancelled")


# ── harvest + ranking (spec 6.3, 6.4, 9.7) ────────────────────────────────


async def _child(store: RunStore, sweep_id: str, index: int, *,
                 status: str = "succeeded",
                 metrics: dict[str, float] | None = None,
                 series: dict[str, list[tuple[int, float]]] | None = None,
                 ) -> str:
    """One child run. *metrics* logs a single point per series at step 0;
    *series* logs a whole ``[(step, value), ...]`` in the order given, so a
    test can pin which point of a series the harvest actually reads."""
    record = await store.create_run(
        graph_snapshot={"nodes": [], "edges": []}, options={},
        provenance=RunProvenance(), sweep_id=sweep_id, sweep_variant=index)
    for name, value in (metrics or {}).items():
        await store.log_metrics(record.id, [MetricPoint(name, value, 0)])
    for name, points in (series or {}).items():
        for step, value in points:
            await store.log_metrics(record.id, [MetricPoint(name, value, step)])
    if status != "queued":
        await store.mark_finished(record.id, status)
    return record.id


def test_rank_variants_sorts_by_objective_and_keeps_the_unrankable():
    """Fed OUT of index order on purpose, both halves of the list.

    ``sorted`` is stable, so an input that already arrives index-ordered
    lets a key with NO ``index`` component agree with a correct one by
    accident -- and then nothing pins the tiebreak that makes the ordering
    total. Variants 0 and 3 tie at 0.5 and arrive 3-before-0; the unranked
    4 and 1 arrive 4-before-1.
    """
    variants = [_variant(3, objective=0.5), _variant(4, objective=None),
                _variant(0, objective=0.5), _variant(1, objective=None),
                _variant(2, objective=0.2)]
    ranked = rank_variants(variants, direction="minimize")
    assert [(v.index, rank) for v, rank in ranked] == [
        (2, 1), (0, 2), (3, 3),      # ties break on index ASCENDING
        (1, None), (4, None),        # unrankable, appended in index order
    ]


def test_rank_variants_maximize_still_breaks_ties_on_index_ascending():
    """The tie arrives 2-before-0, so this pins the tiebreak and not just
    the direction. ``sorted(..., reverse=True)`` is documented STABLE and
    does not reverse equal elements, so a ``reverse=True`` implementation
    with no index key answers [(1,1),(2,2),(0,3)] here and fails.
    """
    variants = [_variant(2, objective=0.5), _variant(1, objective=0.9),
                _variant(0, objective=0.5)]
    ranked = rank_variants(variants, direction="maximize")
    assert [(v.index, rank) for v, rank in ranked] == [(1, 1), (0, 2), (2, 3)]


def test_an_unknown_rank_direction_is_refused():
    """``set_state`` raises on ITS unknown vocabulary; this matches it.

    Silently minimizing on ``"Maximize"``, ``"max"`` or ``""`` would present
    the WORST variant as `best` -- a confidently inverted answer, which is
    worse than an error. Spec 6.2 makes `direction` required precisely
    because inferring it would be a heuristic on a user-chosen string; the
    route validates it too, and this is the half that does not depend on a
    caller getting it right.
    """
    for bad in ("Maximize", "max", ""):
        with pytest.raises(ValueError, match="unknown sweep direction"):
            rank_variants([_variant(0, objective=0.5)], direction=bad)


def test_a_failed_variant_that_logged_the_objective_is_still_ranked():
    """#140 acceptance criterion 3: hiding a real number because the run
    ended badly is exactly the silent disappearance it forbids."""
    variants = [_variant(0, objective=0.3, status="failed"),
                _variant(1, objective=0.9, status="succeeded")]
    ranked = rank_variants(variants, direction="minimize")
    assert [(v.index, rank) for v, rank in ranked] == [(0, 1), (1, 2)]


def test_variant_is_terminal_answers_each_clause_with_a_LIVE_child_row():
    """Every clause, exercised where ``child_exists=True``.

    Asserting only ``child_exists=False`` cases lets the row-is-gone clause
    answer all of them, and then the function can be collapsed to
    ``not child_exists or child_status in TERMINAL_STATUSES`` with nothing
    noticing. Seam B never reaches the earlier clauses -- it removes the
    doomed ids from ``live`` first -- so Task 7's seam A, where the child
    row is still there, is the only caller that will, which is exactly why
    they need pinning now rather than then.
    """
    # A harvested status is the answer even while the child row is alive
    # and says something else. This is what a `missing` variant relies on.
    assert variant_is_terminal(_variant(0, run_id="r", status="succeeded"),
                               "running", child_exists=True)
    assert variant_is_terminal(_variant(0, run_id="r", status="succeeded"),
                               None, child_exists=False)
    # No run at all is NOT terminal -- see the sweep-level test below.
    assert not variant_is_terminal(_variant(0, run_id=None), "running",
                                   child_exists=True)
    assert not variant_is_terminal(_variant(0, run_id=None), None,
                                   child_exists=False)
    # The row is gone and nothing harvested it: spec 5.3's "missing", and
    # the clause that stops a hand-deleted child wedging a sweep forever.
    assert variant_is_terminal(_variant(0, run_id="r"), None,
                               child_exists=False)
    # A live child answers on its own status.
    assert variant_is_terminal(_variant(0, run_id="r"), "succeeded",
                               child_exists=True)
    assert not variant_is_terminal(_variant(0, run_id="r"), "running",
                                   child_exists=True)


async def test_harvest_is_idempotent_and_records_an_absent_objective(sweeps):
    created = await _new_sweep(sweeps, count=2)
    await sweeps.set_variant_run(created.id, 0, run_id="r0", seed=None)
    await sweeps.set_variant_run(created.id, 1, run_id="r1", seed=None)

    first = await sweeps.harvest(
        created.id,
        entries={0: HarvestEntry(objective=0.25, status="succeeded"),
                 1: HarvestEntry(objective=None, status="failed")},
        finished=True)
    assert first.state == SWEEP_STATE_FINISHED
    assert first.finished_at is not None
    assert first.variants[0].objective == 0.25
    assert first.variants[1].objective is None
    assert first.variants[1].status == "failed"
    assert first.variants[1].harvested_at is not None

    # "harvested, no value" is a recorded fact, not a retry.
    second = await sweeps.harvest(
        created.id, entries={1: HarvestEntry(objective=9.9, status="failed")},
        finished=True)
    assert second.variants[1].objective is None
    assert second.finished_at == first.finished_at    # stamped once

    # A sweep that is gone answers None rather than inventing a row.
    assert await sweeps.harvest("nope", entries={}, finished=True) is None


async def test_concurrent_harvests_do_not_lose_each_other(sweeps):
    """The single-closure rule again, for the SECOND writer of the blob.

    ``set_variant_run``'s own atomicity test cannot cover ``harvest``: they
    are different methods, and the tidy-looking refactor -- read with
    ``get_sweep``, patch in Python, write with a second ``Database.run`` --
    is available to each of them separately. Split that way, all four calls
    below read the same blob before any of them writes, every one still
    returns a full record, and only the last patch survives.

    Spec 6.3 runs a harvest on every GET and every cancel, so several are
    genuinely in flight at once whenever a sweep is being polled; a lost
    update there silently discards a harvested objective and, because
    ``harvested_at`` was written for it, nothing ever retries.
    """
    created = await _new_sweep(sweeps, count=4)
    for index in range(4):
        await sweeps.set_variant_run(created.id, index, run_id=f"r{index}",
                                     seed=None)
    patched = await asyncio.gather(*(
        sweeps.harvest(created.id,
                       entries={index: HarvestEntry(objective=float(index),
                                                    status="succeeded")},
                       finished=False)
        for index in range(4)))
    assert all(record is not None for record in patched)
    fetched = await sweeps.get_sweep(created.id)
    assert [v.index for v in fetched.variants] == [0, 1, 2, 3]   # order kept
    assert [v.objective for v in fetched.variants] == [0.0, 1.0, 2.0, 3.0]


async def test_seam_b_harvests_the_LAST_point_of_the_objective_series(
        db, sweeps):
    """Spec 6.1: the objective is the series' LAST point by ``step``.

    Not the first, not the best epoch, and not the last one WRITTEN. Every
    other sweep test logs exactly one point per series, so the ordering in
    ``_last_metric_value`` changes none of their answers and its
    ``ORDER BY step DESC, id DESC`` could be flipped with nothing noticing.
    Seam B is a SECOND implementation of the rule ``latest_metrics``
    already implements -- that one is pinned in test_run_store.py, this one
    was not -- and spec 6.1 requires the two to agree.

    The blast radius is why this matters more here than in a chart: a sweep
    pruned before its first GET is harvested only by seam B, and RULING 4
    makes whatever it stored the PERMANENT answer. The children are gone
    and nothing recomputes it.

    Three points, arriving out of step order, give four implementations
    four different answers: last-by-step 0.5 (correct), first-by-step 0.9,
    best 0.1, last-WRITTEN 0.9. A run that overfits after its best epoch
    therefore ranks on where it ended up, which is the honest reading of
    #140's "final metric".
    """
    store = RunStore(db)
    created = await _new_sweep(sweeps, count=1)
    run_id = await _child(store, created.id, 0, series={
        "val_loss": [(1, 0.1), (2, 0.5), (0, 0.9)]})
    await sweeps.set_variant_run(created.id, 0, run_id=run_id, seed=None)

    # No GET first, so seam A never runs and seam B is the only path.
    assert await store.prune(keep_last=0) == 1
    fetched = await sweeps.get_sweep(created.id)
    assert fetched.variants[0].objective == 0.5
    assert fetched.variants[0].harvested_at is not None


async def test_the_objective_is_harvested_before_the_delete(db, sweeps):
    """RULING 4, seam B: prune with NO prior read, and the value survives."""
    store = RunStore(db)
    created = await _new_sweep(sweeps, count=2)
    for index in range(2):
        run_id = await _child(store, created.id, index,
                              metrics={"val_loss": 0.4 - 0.1 * index})
        await sweeps.set_variant_run(created.id, index, run_id=run_id,
                                     seed=None)

    assert await store.prune(keep_last=0) == 2
    fetched = await sweeps.get_sweep(created.id)
    assert [v.objective for v in fetched.variants] == [0.4, 0.30000000000000004]
    assert [v.status for v in fetched.variants] == ["succeeded", "succeeded"]
    assert all(v.harvested_at is not None for v in fetched.variants)


async def test_a_sweep_mid_submit_loop_is_never_stamped_finished(db, sweeps):
    """A variant with NO run yet is not a variant that is done.

    ``prune_retention()`` runs after every ``_finalize``, on the same event
    loop that ``POST /api/sweeps`` is still submitting on, so the first
    child of a sweep can finish and be pruned while the later variants
    legitimately still carry ``run_id: null``. Counting those as terminal
    stamps the sweep ``finished`` on its FIRST child -- at
    ``RUN_RETENTION_KEEP_LAST = 0``, a documented setting, every time and
    not as a race. It never recovers, because ``_write_variants`` refuses
    to re-stamp a sweep that already reads ``finished``, so a polling table
    would stop polling and a half-empty comparison table would be read as
    the sweep's answer.

    The submit loop BREAKING is a different case and does not need this:
    spec 5.2 has the route call ``mark_failed`` itself, and neither seam
    ever overwrites ``failed``.
    """
    store = RunStore(db)
    created = await _new_sweep(sweeps, count=3)
    run_id = await _child(store, created.id, 0, metrics={"val_loss": 0.4})
    await sweeps.set_variant_run(created.id, 0, run_id=run_id, seed=None)

    assert await store.prune(keep_last=0) == 1
    fetched = await sweeps.get_sweep(created.id)
    assert fetched.variants[0].objective == 0.4      # harvested all the same
    assert fetched.state == SWEEP_STATE_RUNNING      # ... but NOT finished
    assert fetched.finished_at is None


async def test_a_failing_harvest_does_not_stop_retention(db, caplog,
                                                         monkeypatch):
    """Retention is unattended and irreplaceable; the harvest is not.

    ``prune`` is the only thing bounding ``exec_runs``, its cascaded
    children, checkpoint files and TensorBoard logdirs, and it runs behind
    the run task's blanket ``except Exception`` -- so an exception raised
    inside its transaction aborts retention for EVERY run, on every later
    pass, permanently, and surfaces as one log line rather than a failed
    request. One unreadable ``sweeps.variants`` cell must not fill the
    user's disk.
    """
    store = RunStore(db)
    record = await store.create_run(
        graph_snapshot={"nodes": [], "edges": []}, options={},
        provenance=RunProvenance())
    await store.mark_finished(record.id, "succeeded")

    def _boom(conn, where_clause, params):
        raise ValueError("sweeps.variants for sweep 'x' is dict, expected "
                         "a JSON array")

    monkeypatch.setattr("app.core.sweep_store.harvest_doomed", _boom)
    with caplog.at_level(logging.WARNING, logger="app.core.run_store"):
        assert await store.prune(keep_last=0) == 1
    assert await store.get_run(record.id) is None
    assert "harvest" in caplog.text
    assert "sweeps.variants" in caplog.text      # the cause, not just "oops"


async def _corrupt_variants(db: Database, sweep_id: str) -> None:
    """Make one ``sweeps`` row unreadable — a JSON object where the blob
    must be an array, which ``SweepRecord.from_row`` deliberately refuses
    to paper over rather than turning into a plausible empty sweep."""
    await db.run(lambda conn: conn.execute(
        "UPDATE sweeps SET variants = ? WHERE id = ?",
        ('{"not": "a list"}', sweep_id)))


async def test_one_unharvestable_sweep_does_not_cost_the_others_their_results(
        db, sweeps):
    """Per-sweep isolation: one bad row costs only its OWN sweep.

    Catching around the whole harvest keeps retention alive but unwinds the
    loop, so every other sweep in the same pass loses its harvested numbers
    too. ``prune`` sweeps the entire table in one call, so those sweeps are
    unrelated to the broken one — a single corrupt row would quietly cost a
    healthy sweep the results RULING 4 exists to preserve, and the children
    holding them are deleted in the same transaction.
    """
    store = RunStore(db)
    broken = await _new_sweep(sweeps, count=1, name="broken")
    healthy = await _new_sweep(sweeps, count=1, name="healthy")
    # Children created in this order so the doomed SELECT — a table scan,
    # therefore rowid order — hands the broken sweep to the loop FIRST, and
    # the healthy one is harvested strictly after the failure.
    for sweep, value in ((broken, 0.7), (healthy, 0.25)):
        run_id = await _child(store, sweep.id, 0, metrics={"val_loss": value})
        await sweeps.set_variant_run(sweep.id, 0, run_id=run_id, seed=None)
    await _corrupt_variants(db, broken.id)

    assert await store.prune(keep_last=0) == 2
    fetched = await sweeps.get_sweep(healthy.id)
    assert fetched.variants[0].objective == 0.25
    assert fetched.variants[0].status == "succeeded"
    assert fetched.state == SWEEP_STATE_FINISHED


async def test_an_unharvestable_sweep_is_logged_with_its_id(db, sweeps,
                                                            caplog):
    """A swallowed failure nobody can trace is barely better than a silent
    one: the id is what makes the row findable afterwards."""
    store = RunStore(db)
    broken = await _new_sweep(sweeps, count=1)
    run_id = await _child(store, broken.id, 0, metrics={"val_loss": 0.7})
    await sweeps.set_variant_run(broken.id, 0, run_id=run_id, seed=None)
    await _corrupt_variants(db, broken.id)

    with caplog.at_level(logging.WARNING):
        assert await store.prune(keep_last=0) == 1

    # Asserted on the RECORD, not on caplog.text: the outer backstop in
    # prune logs `exc_info=True`, and the ValueError's own message already
    # names the sweep, so a substring check over the rendered traceback
    # would pass with no per-sweep isolation at all.
    isolated = [record for record in caplog.records
                if record.name == "app.core.sweep_store"
                and record.levelno >= logging.WARNING]
    assert len(isolated) == 1
    assert broken.id in isolated[0].getMessage()     # findable, not "a sweep"
    assert "expected a JSON array" in str(isolated[0].exc_info[1])
    # The blanket catch one level up never had to fire.
    assert not [record for record in caplog.records
                if record.name == "app.core.run_store"
                and "could not harvest" in record.getMessage()]
    assert await store.get_run(run_id) is None       # the delete proceeded


async def test_the_sweep_finishes_through_seam_b_without_a_read(db, sweeps):
    """Catches a seam-B harvest that copies values but forgets the state."""
    store = RunStore(db)
    created = await _new_sweep(sweeps, count=2)
    for index in range(2):
        run_id = await _child(store, created.id, index,
                              metrics={"val_loss": 0.5})
        await sweeps.set_variant_run(created.id, index, run_id=run_id,
                                     seed=None)
    await store.prune(keep_last=0)
    fetched = await sweeps.get_sweep(created.id)
    assert fetched.state == SWEEP_STATE_FINISHED
    assert fetched.finished_at is not None


async def test_a_diverged_variant_is_unranked(db, sweeps):
    """A non-finite last point is stored as SQL NULL and omitted from the
    series map, so a diverged loss must not render as a suspiciously good
    one -- it is unranked, never ranked at 0.0."""
    store = RunStore(db)
    created = await _new_sweep(sweeps, count=1)
    run_id = await _child(store, created.id, 0,
                          metrics={"val_loss": float("nan")})
    await sweeps.set_variant_run(created.id, 0, run_id=run_id, seed=None)
    await store.prune(keep_last=0)
    fetched = await sweeps.get_sweep(created.id)
    assert fetched.variants[0].objective is None
    assert fetched.variants[0].status == "succeeded"
    assert rank_variants(fetched.variants, direction="minimize") == \
        [(fetched.variants[0], None)]
