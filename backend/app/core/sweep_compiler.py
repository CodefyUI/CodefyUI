"""Grid / random sweep compilation (#140) — pure, LLM-free, deterministic.

Turns a ``sweep_spec`` plus a base graph into an ordered list of complete
variant graphs. No database, no HTTP, no node execution, and the input graph
is NEVER modified: every variant is an independent ``deepcopy`` with its
overrides written into ``data.params``.

Ported bit-exactly from Graph Copilot's browser-side planner
(``CodefyUI-Plugin-Graph-Copilot/ui/src/agent/optimizer.ts``): ``mulberry32``
(:307-317), ``sampleUniqueRanks`` (:319-336) and the mixed-radix decode
``assignmentAt`` (:285-294). Range expansion and per-variant execution seeds
have no reference implementation there and are new design (spec 2.5, 2.9).

RULING 3: an address that cannot be set is REFUSED, never guessed. A preset
instance's params live in ``data.internalParams`` and a subgraph instance's
come from its definition, so neither is reachable by a ``{node_id, param}``
address; both are 400s that say why.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Callable

from .node_base import ParamDefinition, ParamType
from .node_registry import registry

_GRID = "grid"
_RANDOM = "random"
_METHODS = (_GRID, _RANDOM)
_SCALES = ("linear", "log")
_RANGE_TYPES = ("int", "float")

#: The planner seed space, matching optimizer.ts:462-466's own bound.
MAX_PLANNER_SEED = 0xFFFFFFFF

#: The param types a sweep can enumerate (spec 3.4, check 7).
SWEEPABLE_PARAM_TYPES = (ParamType.INT, ParamType.FLOAT, ParamType.BOOL,
                         ParamType.STRING, ParamType.SELECT)


class SweepCompileError(ValueError):
    """A sweep_spec that cannot be compiled. REST maps this to 400.

    Mirrors ``RunSubmitError`` (``run_service.py:435-436``), which
    ``routes_runs.py:264-265`` maps the same way.
    """


def planner_prng(seed: int) -> Callable[[], int]:
    """``createPlannerPrng`` (optimizer.ts:307-317), ported.

    Returns RAW uint32 draws, not floats in ``[0, 1)``. The JS original has
    no ``/ 4294967296`` and its only consumer is ``next() % remaining``; a
    textbook mulberry32 that divides and multiplies back would diverge on
    some draws, giving the same seed a different variant list.

    Translation: ``x >>> 0`` is ``x & 0xFFFFFFFF``; ``Math.imul(a, b)`` is
    ``(a * b) & 0xFFFFFFFF`` once both operands are masked non-negative.
    """
    state = seed & 0xFFFFFFFF

    def next_uint32() -> int:
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        value = state
        value = ((value ^ (value >> 15)) * (value | 1)) & 0xFFFFFFFF
        value = (value ^ (value + (((value ^ (value >> 7))
                                    * (value | 61)) & 0xFFFFFFFF))) & 0xFFFFFFFF
        return (value ^ (value >> 14)) & 0xFFFFFFFF

    return next_uint32


def sample_unique_ranks(total: int, count: int, seed: int) -> list[int]:
    """``count`` distinct ranks out of ``0..total-1``, in DRAW order.

    Sparse Fisher-Yates (``sampleUniqueRanks``, optimizer.ts:319-336): one
    draw per VARIANT rather than per param, so uniqueness is structural and
    there is no draw-and-reject loop. Draw order is preserved deliberately —
    sorting would throw away the reproducibility the pinned vectors prove.
    """
    nxt = planner_prng(seed)
    swaps: dict[int, int] = {}
    selected: list[int] = []
    for draw_index in range(count):
        remaining = total - draw_index
        draw = nxt() % remaining
        chosen = swaps.get(draw, draw)
        last_position = remaining - 1
        last_value = swaps.get(last_position, last_position)
        if draw != last_position:
            swaps[draw] = last_value
        else:
            swaps.pop(draw, None)
        swaps.pop(last_position, None)
        selected.append(chosen)
    return selected


@dataclass(frozen=True)
class CompiledParam:
    """One swept address plus its final, deduplicated, coerced domain."""

    node_id: str
    param: str
    #: The ``ParamType`` value: "int"|"float"|"bool"|"string"|"select".
    param_type: str
    domain: tuple[Any, ...]


@dataclass(frozen=True)
class CompiledVariant:
    """One variant: a whole, independent graph with its overrides applied."""

    #: 0-based position in ``CompiledSweep.variants``.
    index: int
    #: Which cell of the cartesian product this is. Equal to ``index`` for a
    #: grid; for a random sweep it is the drawn rank, recorded so a reader
    #: can verify the draw.
    domain_index: int
    #: One value per ``CompiledSweep.params``, in the same order.
    assignment: tuple[Any, ...]
    graph: dict[str, Any]


@dataclass(frozen=True)
class CompiledSweep:
    params: tuple[CompiledParam, ...]
    variants: tuple[CompiledVariant, ...]
    #: prod(len(p.domain)). May exceed len(variants) for a random sweep.
    total_combinations: int
    #: The PLANNER seed, exactly as ``_validate_seed`` accepted it: the seed
    #: that drove the sampling for a ``random`` sweep, whatever the caller
    #: sent for a ``grid``, and None when a grid sent nothing. RULING 1
    #: records it either way, and the route fills ``sweeps.seed`` from HERE
    #: rather than re-reading ``spec["seed"]`` -- so the seed the sampler
    #: used and the seed the row stores cannot drift apart.
    seed: int | None


def _round_half_away_from_zero(value: float) -> int:
    """NOT Python's ``round``, which is banker's rounding: ``round(0.5)`` is
    0 and ``round(2.5)`` is 2, so a batch-size domain would silently skip
    values by a rule nobody reading the spec would predict."""
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


def _expand_range(lo: float, hi: float, count: int, scale: str,
                  range_type: str) -> list[Any]:
    """Spec 2.5, to the last float.

    Both endpoints are pinned by EXPLICIT ASSIGNMENT after the
    interpolation, never by trusting the arithmetic: ``10 ** log10(1e-4)``
    is not bit-identical to ``1e-4``, and a learning-rate column reading
    ``0.00010000000000000009`` in a classroom table is a bug report. This is
    ``numpy.linspace``'s own endpoint discipline.

    Base **10** for the log scale, not ``e``. The interpolation is
    mathematically base-independent and NOT so in IEEE-754: the ``ln`` form
    gives ``[0.00010000000000000009, ...]`` where ``log10`` gives
    ``[0.0001, 0.001, 0.01, 0.1]``. The base is observable, so it is pinned.
    """
    if count == 1:
        values: list[float] = [lo]
    elif scale == "log":
        a, b = math.log10(lo), math.log10(hi)
        values = [10 ** (a + (b - a) * i / (count - 1)) for i in range(count)]
        values[0], values[-1] = lo, hi
    else:
        values = [lo + (hi - lo) * i / (count - 1) for i in range(count)]
        values[0], values[-1] = lo, hi

    typed: list[Any] = ([_round_half_away_from_zero(v) for v in values]
                        if range_type == "int" else list(values))

    # Dedup by exact equality, first-seen order. The DEDUPLICATED length is
    # the domain size everywhere afterwards -- a cap computed on the
    # pre-dedup count would lie. A range that collapses is not an error:
    # {1, 3, count 10, linear, int} genuinely has three integer points.
    out: list[Any] = []
    for value in typed:
        if value not in out:
            out.append(value)
    return out


def _is_finite_number(value: Any) -> bool:
    """Is this number usable at all -- a finite float, or an int a float can
    hold?

    ``math.isfinite`` alone is not enough, and the gap is reachable from an
    ordinary JSON body. Python ints are UNBOUNDED, so ``json.loads`` parses a
    400-digit integer literal into an ``int`` no float can represent, and
    ``math.isfinite`` then RAISES ``OverflowError: int too large to convert
    to float`` rather than returning False. On an in-request compiler that is
    an uncaught 500 on a surface whose whole contract is that a bad spec is a
    400 -- the integer sibling of the ``json.loads("1e999")`` case spec 2.7
    already refuses by design, and the same class as a non-dict node entry.

    Callers refuse with the SPEC'S EXISTING non-finite message rather than a
    dedicated "too large" one: a caller does not need to know whether their
    number was too big or not a number, only that it cannot be used. This is
    the single finiteness predicate for the module, so every entry point --
    both envelope checks and the type matrix -- is covered by construction.
    """
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _coerce(value: Any, param_type: ParamType) -> Any | None:
    """Spec 3.4.1's matrix. The coerced value, or None when unacceptable.

    None is unambiguous as a rejection because ``null`` is not a sweepable
    scalar in the first place (spec 2.1).
    """
    if param_type is ParamType.BOOL:
        return value if isinstance(value, bool) else None
    if param_type in (ParamType.STRING, ParamType.SELECT):
        return value if isinstance(value, str) else None
    if isinstance(value, bool):
        # bool subclasses int and True == 1: without this guard `true`
        # silently becomes the integer 1. Same rule normalize_options
        # already applies to `seed` (run_service.py:544).
        return None
    if param_type is ParamType.INT:
        if isinstance(value, int):
            return int(value)
        if (isinstance(value, float) and _is_finite_number(value)
                and value.is_integer()):
            return int(value)
        return None
    if param_type is ParamType.FLOAT:
        # _is_finite_number, not math.isfinite: this is also what stops the
        # float(value) below raising OverflowError on an unbounded int.
        if isinstance(value, (int, float)) and _is_finite_number(value):
            return float(value)
        return None
    return None


def _validate_seed(spec: dict[str, Any], method: str) -> int | None:
    """The sweep's own PLANNER seed: it selects which combinations exist.

    Required for ``random``, optional for ``grid`` (a grid enumerates
    everything, so the seed selects nothing) — but still recorded, because
    RULING 1 makes it the base for per-variant execution seeds. The
    ``seed_variants`` interaction is the ROUTE's rule (spec 5.2 rule 3), not
    this function's: the compiler never sees that flag.
    """
    seed = spec.get("seed")
    if seed is None:
        if method == _RANDOM:
            raise SweepCompileError(
                "sweep_spec.seed is required for method 'random' and must "
                "be an integer from 0 to 4294967295")
        return None
    if isinstance(seed, bool) or not isinstance(seed, int) \
            or not 0 <= seed <= MAX_PLANNER_SEED:
        # Its OWN wording, not the "required for method 'random'" one: a
        # grid sweep carrying seed=2**32 has an out-of-range seed, not a
        # missing one.
        raise SweepCompileError(
            "sweep_spec.seed must be an integer from 0 to 4294967295")
    return seed


def _validate_samples(spec: dict[str, Any], method: str,
                      max_runs: int) -> int | None:
    """FORBIDDEN for a grid: on a grid `samples` could only mean truncate,
    and a truncated grid is a table whose missing rows are invisible."""
    samples = spec.get("samples")
    if method == _GRID:
        if samples is not None:
            raise SweepCompileError(
                "sweep_spec.samples is only allowed for method 'random'")
        return None
    if samples is None:
        raise SweepCompileError(
            "sweep_spec.samples is required for method 'random'")
    if isinstance(samples, bool) or not isinstance(samples, int) \
            or not 1 <= samples <= max_runs:
        raise SweepCompileError(
            f"sweep_spec.samples must be an integer from 1 to {max_runs}")
    return samples


def _validated_range(i: int, spec: Any, *, max_domain: int) -> list[Any]:
    if not isinstance(spec, dict):
        raise SweepCompileError(
            f"sweep_spec.params[{i}] needs either 'values' or 'range'")
    lo, hi = spec.get("min"), spec.get("max")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
               and _is_finite_number(v) for v in (lo, hi)):
        raise SweepCompileError(
            f"sweep_spec.params[{i}].range bounds must be finite numbers")
    count = spec.get("count")
    if isinstance(count, bool) or not isinstance(count, int) \
            or not 1 <= count <= max_domain:
        raise SweepCompileError(
            f"sweep_spec.params[{i}].range.count must be an integer from 1 "
            f"to {max_domain}")
    if hi < lo:
        raise SweepCompileError(
            f"sweep_spec.params[{i}].range.max must be >= range.min")
    scale = spec.get("scale")
    if scale not in _SCALES:
        raise SweepCompileError(
            f"unknown range scale {scale!r}; expected 'linear' or 'log'")
    range_type = spec.get("type")
    if range_type not in _RANGE_TYPES:
        raise SweepCompileError(
            f"unknown range type {range_type!r}; expected 'int' or 'float'")
    if scale == "log" and lo <= 0:
        raise SweepCompileError(
            f"sweep_spec.params[{i}].range needs a positive min for a log "
            "scale (log of 0 or a negative number is undefined)")
    if range_type == "int" and not (float(lo).is_integer()
                                    and float(hi).is_integer()):
        raise SweepCompileError(
            f"sweep_spec.params[{i}].range needs whole-number bounds for "
            "type 'int'")
    return _expand_range(float(lo), float(hi), count, scale, range_type)


def _raw_domain(i: int, entry: dict[str, Any], *,
                max_domain: int) -> list[Any]:
    """The pre-coercion domain. Runs for EVERY param before any address is
    resolved, so a NaN is refused with the envelope message naming
    ``values[j]`` rather than with a type message (spec 3.4.1)."""
    values = entry.get("values")
    range_spec = entry.get("range")
    if values is not None and range_spec is not None:
        raise SweepCompileError(
            f"sweep_spec.params[{i}] may not carry both 'values' and 'range'")
    if values is None and range_spec is None:
        raise SweepCompileError(
            f"sweep_spec.params[{i}] needs either 'values' or 'range'")
    if values is None:
        return _validated_range(i, range_spec, max_domain=max_domain)

    if not isinstance(values, list) or not 1 <= len(values) <= max_domain:
        raise SweepCompileError(
            f"sweep_spec.params[{i}].values must contain 1 to {max_domain} "
            "values")
    for j, value in enumerate(values):
        if value is None or isinstance(value, (list, dict)):
            raise SweepCompileError(
                f"sweep_spec.params[{i}].values must contain only numbers, "
                "booleans or strings")
        if isinstance(value, (int, float)) and not _is_finite_number(value):
            # A non-finite passes every other check and then corrupts things
            # quietly: RunStore._dumps rewrites it to null in the stored
            # snapshot, and Starlette's JSONResponse renders with
            # allow_nan=False, so a survivor wedges every later read of the
            # sweep with a 500 inside the renderer.
            raise SweepCompileError(
                f"sweep_spec.params[{i}].values[{j}] must be a finite "
                "number; NaN and Infinity are not sweepable")
    return list(values)


def _node_index(base_graph: dict[str, Any]) -> dict[str, list[dict]]:
    """Every addressable node keyed by id, ids mapping to a LIST.

    NON-DICT entries are SKIPPED rather than indexed. ``normalize_graph``
    (``run_service.py:619-655``) validates that ``nodes`` is a non-empty
    list and nothing about its contents, so ``{"nodes": ["x"]}`` reaches
    here intact; ``node["id"]`` on a ``str`` raises ``TypeError``, which on
    an in-request compiler is an uncaught 500 on a surface that promises
    400s. A non-dict entry simply matches no ``node_id``, which is the
    honest answer -- the graph really does not contain an addressable node
    with that id.

    Values are lists so check 1 can tell one node from two: a ``node_id``
    over a duplicated id would set one node and leave the other alone, with
    no way to say which (the hazard ``secret_params.py:37-41`` names).
    """
    out: dict[str, list[dict]] = {}
    for node in base_graph.get("nodes") or []:
        if isinstance(node, dict) and isinstance(node.get("id"), str):
            out.setdefault(node["id"], []).append(node)
    return out


def _coerced_domain(i: int, node_id: str, param_name: str,
                    definition: ParamDefinition, domain: list[Any], *,
                    from_values: bool) -> list[Any]:
    """Checks 8-10, in the order spec 3.4.1 rule 4 fixes: coerce every
    value, THEN duplicates, THEN bounds.

    Duplicates first would reintroduce the ``1 == True`` collision; bounds
    first would compare a bound against a value the sweep will not use.
    Coercion is what makes a domain type-homogeneous, which is what makes
    the exact-equality dedup correct.
    """
    address = f"{node_id}.{param_name}"
    coerced: list[Any] = []
    for value in domain:
        out = _coerce(value, definition.param_type)
        if out is None:
            raise SweepCompileError(
                f"sweep_spec.params[{i}]: '{address}' is a "
                f"{definition.param_type.value} param but the domain "
                f"contains {value!r}")
        if definition.param_type is ParamType.SELECT \
                and out not in definition.options:
            raise SweepCompileError(
                f"sweep_spec.params[{i}]: '{address}' does not accept "
                f"{out!r}; its options are {definition.options}")
        coerced.append(out)

    if from_values:
        # A range that collapses is arithmetic and is deduped (spec 2.5
        # step 3); a hand-written list that repeats a value is a typo the
        # caller can fix.
        seen: list[Any] = []
        for value in coerced:
            if value in seen:
                raise SweepCompileError(
                    f"sweep_spec.params[{i}].values repeats {value!r}; "
                    "list each value once")
            seen.append(value)

    if definition.param_type in (ParamType.INT, ParamType.FLOAT):
        for value in coerced:
            if definition.min_value is not None \
                    and value < definition.min_value:
                raise SweepCompileError(
                    f"sweep_spec.params[{i}]: '{address}' must be >= "
                    f"{definition.min_value}, got {value}")
            if definition.max_value is not None \
                    and value > definition.max_value:
                raise SweepCompileError(
                    f"sweep_spec.params[{i}]: '{address}' must be <= "
                    f"{definition.max_value}, got {value}")
    return coerced


def _compile_param(i: int, entry: dict[str, Any], domain: list[Any],
                   index: dict[str, list[dict]]) -> CompiledParam:
    """Checks 1-10 of spec 3.4 for one address. RULING 3 lives here."""
    node_id = entry.get("node_id")
    param_name = entry.get("param")
    matches = index.get(node_id, []) if isinstance(node_id, str) else []
    if not matches:
        raise SweepCompileError(
            f"sweep_spec.params[{i}]: no node with id {node_id!r} in "
            "base_graph")
    if len(matches) > 1:
        raise SweepCompileError(
            f"sweep_spec.params[{i}]: base_graph has {len(matches)} nodes "
            f"with id {node_id!r}; ids must be unique for a sweep to "
            "address them")
    node = matches[0]

    data = node.get("data")
    if data is not None and not isinstance(data, dict):
        raise SweepCompileError(
            f"sweep_spec.params[{i}]: node {node_id!r} has no 'data' object "
            "to set params on")
    stored = (data or {}).get("params")
    if stored is not None and not isinstance(stored, dict):
        raise SweepCompileError(
            f"sweep_spec.params[{i}]: node {node_id!r} has a 'data.params' "
            "that is not an object")

    node_type = node.get("type")
    if isinstance(node_type, str) and node_type.startswith("preset:"):
        # A preset node's inner params live in
        # data.internalParams[<inner id>][<param>] (graph_engine.py:102);
        # writing data.params[name] on the preset node reaches nothing.
        raise SweepCompileError(
            f"sweep_spec.params[{i}]: node {node_id!r} is a preset "
            "instance; its inner params live in data.internalParams, which "
            "a {node_id, param} address cannot reach")
    if isinstance(node_type, str) and node_type.startswith("subgraph:"):
        # graph_engine.py:466-467: "Params ride along verbatim from the
        # definition. v1 has no per-instance overrides".
        raise SweepCompileError(
            f"sweep_spec.params[{i}]: node {node_id!r} is a subgraph "
            "instance; a subgraph's params come from its definition and "
            "cannot be overridden per instance")

    node_cls = registry.get(node_type) if isinstance(node_type, str) else None
    if node_cls is None:
        raise SweepCompileError(
            f"sweep_spec.params[{i}]: node {node_id!r} has type "
            f"{node_type!r}, which is not a known node type; a sweep can "
            "only set params on registered nodes")

    definition = next((p for p in node_cls.define_params()
                       if p.name == param_name), None)
    if definition is None:
        raise SweepCompileError(
            f"sweep_spec.params[{i}]: node {node_id!r} (type {node_type!r}) "
            f"has no param named {param_name!r}")
    if definition.param_type is ParamType.SECRET:
        # RULING 4 stores each variant's chosen params on the DURABLE sweeps
        # row, so a swept secret would be written there in the clear --
        # while every other persisting path scrubs it.
        raise SweepCompileError(
            f"sweep_spec.params[{i}]: '{node_id}.{param_name}' is a secret "
            "parameter and cannot be swept")
    if definition.param_type not in SWEEPABLE_PARAM_TYPES:
        raise SweepCompileError(
            f"sweep_spec.params[{i}]: '{node_id}.{param_name}' has param "
            f"type '{definition.param_type.value}', which a sweep cannot "
            "enumerate")

    return CompiledParam(
        node_id=node_id, param=param_name,
        param_type=definition.param_type.value,
        domain=tuple(_coerced_domain(
            i, node_id, param_name, definition, domain,
            from_values=entry.get("values") is not None)))


def _assignment_at(params: tuple[CompiledParam, ...],
                   domain_index: int) -> tuple[Any, ...]:
    """Mixed-radix decode (``assignmentAt``, optimizer.ts:285-294).

    The LAST-declared param varies fastest and the first varies slowest --
    row-major / C order -- so a comparison table is reproducible from the
    spec alone. The cartesian product is never materialised as tuples.
    """
    values: list[Any] = [None] * len(params)
    remainder = domain_index
    for i in range(len(params) - 1, -1, -1):
        domain = params[i].domain
        values[i] = domain[remainder % len(domain)]
        remainder //= len(domain)
    return tuple(values)


def _build_variant(base_graph: dict[str, Any],
                   params: tuple[CompiledParam, ...],
                   index: int, domain_index: int) -> CompiledVariant:
    """One independent deep copy with its overrides written in.

    The base graph is NEVER modified: deepcopy first, then write — the same
    contract ``split_graph_secrets`` states (``secret_params.py:388-392``).
    The compiler must NOT pre-scrub secrets: ``RunService.submit`` runs
    ``normalize_graph`` and ``split_graph_secrets`` on each copy itself.
    """
    assignment = _assignment_at(params, domain_index)
    graph = copy.deepcopy(base_graph)
    nodes = {node["id"]: node for node in graph["nodes"]
             if isinstance(node, dict) and isinstance(node.get("id"), str)}
    for param, value in zip(params, assignment):
        node = nodes[param.node_id]      # present: check 1 used this filter
        data = node.get("data")
        if not isinstance(data, dict):
            # An explicit "data": null passes check 2 as "absent", so
            # setdefault alone would call a method on None.
            data = {}
            node["data"] = data
        stored = data.get("params")
        if not isinstance(stored, dict):
            stored = {}
            data["params"] = stored
        stored[param.param] = value
    return CompiledVariant(index=index, domain_index=domain_index,
                           assignment=assignment, graph=graph)


def compile_sweep(base_graph: dict[str, Any], spec: dict[str, Any], *,
                  max_runs: int, max_params: int,
                  max_domain: int) -> CompiledSweep:
    """Validate, expand and enumerate. Raises ``SweepCompileError`` (-> 400).

    Pure: nothing is written anywhere and ``base_graph`` is not touched, so
    a 400 never leaves a partial sweep behind.

    The caps are PASSED IN rather than read from ``settings`` here, so a
    caller states the policy it is enforcing and a test can override it --
    the rule ``QueueLimits.from_settings`` follows for the same reason
    (``run_service.py:383-397``).
    """
    method = spec.get("method")
    if method not in _METHODS:
        raise SweepCompileError(
            f"unknown sweep method {method!r}; expected 'grid' or 'random'")

    raw_params = spec.get("params")
    if not isinstance(raw_params, list) \
            or not 1 <= len(raw_params) <= max_params:
        raise SweepCompileError(
            f"sweep_spec.params must contain 1 to {max_params} entries")

    seed = _validate_seed(spec, method)
    samples = _validate_samples(spec, method, max_runs)

    # Phase 1: the envelope, for every param, before any address exists.
    seen: set[tuple[Any, Any]] = set()
    domains: list[list[Any]] = []
    for i, entry in enumerate(raw_params):
        if not isinstance(entry, dict):
            raise SweepCompileError(
                f"sweep_spec.params[{i}] needs either 'values' or 'range'")
        address = (entry.get("node_id"), entry.get("param"))
        if address in seen:
            raise SweepCompileError(
                f"sweep_spec.params[{i}] repeats the address "
                f"'{address[0]}.{address[1]}'; each address may appear once")
        seen.add(address)
        domains.append(_raw_domain(i, entry, max_domain=max_domain))

    # Phase 2: resolve each address, then coerce against its ParamDefinition.
    index = _node_index(base_graph)
    params = tuple(_compile_param(i, raw_params[i], domains[i], index)
                   for i in range(len(raw_params)))

    total = 1
    for param in params:
        total *= len(param.domain)

    if method == _RANDOM:
        assert samples is not None      # _validate_samples guarantees it
        if samples > total:
            raise SweepCompileError(
                f"sweep_spec.samples asks for {samples} distinct "
                f"combinations but the search space has only {total}")
        variant_count = samples
    else:
        variant_count = total

    # Checked BEFORE any variant is built, so a 10x10x10 grid costs one
    # multiplication rather than a thousand deep copies. Never truncate: a
    # truncated grid is a table whose missing rows are invisible.
    if variant_count > max_runs:
        raise SweepCompileError(
            f"sweep would compile {variant_count} variants but "
            f"MAX_SWEEP_RUNS is {max_runs}; narrow the domains or lower "
            "'samples' instead of truncating the sweep")

    if method == _RANDOM:
        # Draw order, NOT sorted order -- that is what the pinned vectors
        # prove, and sorting would throw the proof away. No baseline shift:
        # optimizer.ts:474-475's remap exists only to skip a baseline slot,
        # and v1 has no baseline (spec 3.1).
        domain_indices = sample_unique_ranks(total, samples, seed or 0)
    else:
        domain_indices = list(range(total))

    variants = tuple(_build_variant(base_graph, params, i, domain_index)
                     for i, domain_index in enumerate(domain_indices))
    return CompiledSweep(params=params, variants=variants,
                         total_combinations=total, seed=seed)
