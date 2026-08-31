"""Sweep compilation and storage (#140).

The REST surface has its own module (test_api_sweeps.py); the probe node is
duplicated there on purpose, the way test_api_runs.py duplicates
test_run_queue.py's, so test modules stay independent.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from app.core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)
from app.core.node_registry import registry
from app.core.sweep_compiler import (
    SweepCompileError,
    compile_sweep,
    planner_prng,
    sample_unique_ranks,
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
