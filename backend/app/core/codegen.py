"""Generate a runnable, single-file Python program for a CodefyUI graph.

The exporter plans the *structure* of the program at export time — one
function per executable node, one function per connected flow, and a
``run_graph()`` orchestrator that mirrors the graph engine's execution
order — while each node's *behavior* stays delegated to the canonical node
implementation through :func:`app.core.graph_engine.invoke_node`, the same
helper the engine itself uses.  Presets are expanded here, at export time,
with the engine's own :func:`prepare_executable_graph`, so the script runs
exactly the executable graph the canvas ran on this machine.  The generated
file does not need a running CodefyUI server, but it does need to be
launched with a compatible CodefyUI backend environment.

Injection safety: every graph-controlled string (node IDs, types, names,
labels, paths, parameters, port names) is embedded only inside
``ascii()``/``repr()``-generated literals, and every Python identifier the
script declares (function, argument, and variable names) comes from a strict
sanitizer — graph data can therefore never become source fragments.
"""

from __future__ import annotations

import builtins
import copy
import keyword
import math
import re
from collections import deque
from typing import Any

from .api_contract import (
    GRAPH_INPUT_TYPE,
    GRAPH_OUTPUT_TYPE,
    check_wiring,
    derive_contract,
)
from .graph_engine import (
    SUBGRAPH_TYPE_PREFIX,
    BypassLink,
    build_preset_fallback,
    outermost_container,
    prepare_executable_graph,
    resolve_bypass,
    topological_sort,
)
from .node_base import ParamType
from .node_registry import registry
from .secret_params import scrub_graph_secrets


# ── Identifier sanitizing ────────────────────────────────────────────────

_WORDS = (
    frozenset(keyword.kwlist)
    | frozenset(keyword.softkwlist)
    | frozenset(dir(builtins))
)

# Names a flow body relies on; node-derived locals must never shadow them.
_RESERVED_LOCALS = frozenset(
    {"ctx", "results", "provided", "run_graph", "main"}
)


def _slug(text: Any, fallback: str) -> str:
    """Lowercase *text* and reduce it to the ``[a-z0-9_]`` identifier charset."""
    slug = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return slug or fallback


def _identifier(
    raw: Any,
    *,
    used: set[str],
    avoid: frozenset[str] | set[str],
    fallback: str,
) -> str:
    """Derive a safe, unique Python identifier from graph-controlled text."""
    base = _slug(raw, fallback)
    if base[0].isdigit():
        base = f"v_{base}"
    if base in _WORDS or base in avoid:
        base = f"{base}_"
    candidate = base
    counter = 1
    while candidate in used or candidate in avoid or candidate in _WORDS:
        counter += 1
        candidate = f"{base}_{counter}"
    used.add(candidate)
    return candidate


# ── Literal embedding ────────────────────────────────────────────────────


def _literal(value: Any) -> str:
    """ASCII-only Python literal for JSON-derived data.

    ``ascii()`` escapes every non-ASCII character and all quoting, so the
    emitted source stays ASCII regardless of graph content.  Non-finite
    floats need spelling out because ``repr(float("inf"))`` is ``inf``,
    which is not a Python literal.
    """
    if value is None or isinstance(value, bool):
        return repr(value)
    if isinstance(value, str):
        return ascii(value)
    if isinstance(value, int):
        return repr(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return repr(value)
        if math.isnan(value):
            return "float('nan')"
        return "float('inf')" if value > 0 else "float('-inf')"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_literal(item) for item in value) + "]"
    if isinstance(value, dict):
        items = ", ".join(
            f"{_literal(key)}: {_literal(val)}" for key, val in value.items()
        )
        return "{" + items + "}"
    raise TypeError(
        f"graph value of type {type(value).__name__} cannot be embedded "
        "in generated Python"
    )


def _comment_text(text: str) -> str:
    """Make arbitrary text safe for a ``#`` comment (single printable line)."""
    return re.sub(r"[^\x20-\x7e]", "?", text)


# ── Inline source params (core#131) ──────────────────────────────────────


def _code_param_names(node_type: str) -> set[str]:
    """Params of *node_type* declared as ``ParamType.CODE``.

    Those hold Python the *user* wrote -- the PythonScript node's body today.
    Collapsed onto one line by ``_literal`` they are an unreadable wall of
    ``\\n`` escapes, which is a poor thing to hand someone who exported their
    graph precisely to read and edit it.
    """
    node_cls = registry.get(node_type)
    if node_cls is None:
        return set()
    try:
        return {
            param.name
            for param in node_cls.define_params()
            if param.param_type == ParamType.CODE
        }
    except Exception:  # noqa: BLE001 - a node class that cannot describe
        return set()   # itself must not break the whole export


def _emit_params(params: dict, node_type: str, node_id: str) -> list[str]:
    """Emit the ``params = {...}`` assignment for one node function.

    Ordinary params stay a one-line literal. A ``CODE`` param is spelled out
    one SOURCE line per string literal, under a provenance comment, so the
    script reads as the script -- and stays the live value the node runs, not
    a stale copy in a comment: edit a line here and the exported graph runs
    the edit.

    Injection safety is unchanged from every other value in this file. The
    lines are ``ascii()`` literals, never raw source, so a ``'''`` or a
    newline inside the user's code cannot terminate the literal and become
    program text.
    """
    code_names = _code_param_names(node_type) & set(params)
    inline = {
        name for name in code_names
        if isinstance(params[name], str) and params[name]
    }
    if not inline:
        return [f"    params = {_literal(params)}"]

    lines = ["    params = {"]
    for key, value in params.items():
        if key not in inline:
            lines.append(f"        {_literal(key)}: {_literal(value)},")
            continue
        lines.append(
            f"        # ---- {_literal(key)} of node {_literal(node_id)}: "
            "the script this node runs, verbatim ----"
        )
        lines.append(f"        {_literal(key)}: (")
        lines.extend(
            f"            {_literal(source_line)}"
            for source_line in value.splitlines(keepends=True)
        )
        lines.append("        ),")
    lines.append("    }")
    return lines


# ── Static script scaffolding ────────────────────────────────────────────

_SCRIPT_HEADER = '''#!/usr/bin/env python3
"""Runnable CodefyUI graph export.

This file was generated from a CodefyUI graph.  Its structure mirrors the
canvas: one function per node, one function per connected flow, and
``run_graph()`` executing the flows in the graph engine's order.  Node
parameters are plain literals -- edit them freely.  Every node function
delegates to the canonical CodefyUI node implementation, so behavior stays
identical to running the graph on the canvas.

It requires a compatible CodefyUI backend Python environment (including
every custom/plugin node and third-party dependency used by the graph).

The canvas's seed is baked in below as ``GRAPH_SEED`` and is the default for
``--seed``, so a seeded graph reproduces here exactly as it did on the
canvas -- same weight init, same shuffling, same crops and flips.  Pass
``--no-seed`` for fresh entropy.  Note that TensorBoard event files are NOT
written by an exported script: it records no artifacts, so there would be no
row referencing the directory.

Development checkout examples:
  Windows: backend/.venv/Scripts/python.exe exported_graph.py
  macOS/Linux: backend/.venv/bin/python exported_graph.py

Use ``--help`` for device, input, timeout, and project-directory options.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any
'''

_RUNTIME_PRELUDE = '''_ABSENT = object()  # "no value flowed here" -- deliberately distinct from None
_RT: Any = None  # CodefyUI runtime namespace, populated by _load_runtime()
_VERBOSE = False
_DEADLINE: float | None = None  # monotonic instant set by --timeout


def _port(result, name):
    """Fetch one output port; a key the node did not produce stays absent."""
    return result.get(name, _ABSENT)


def _pick(*candidates):
    """Multi-edge fan-in: the first present candidate wins.

    Candidates are listed in reverse edge order, matching the graph engine,
    where the last edge whose source port produced a value overwrites
    earlier ones.
    """
    for value in candidates:
        if value is not _ABSENT:
            return value
    return _ABSENT


def _call(node_type, node_id, params, ctx, inputs=None):
    """Instantiate *node_type* and run it through the engine's invoke_node.

    ``inputs`` keys are the real port-name strings (they are not always
    valid Python identifiers); absent values are filtered out so the node
    sees exactly the keys the engine would deliver.
    """
    if _DEADLINE is not None and time.monotonic() > _DEADLINE:
        raise TimeoutError(
            f"--timeout reached before node {node_id!r}; remaining nodes were not run"
        )
    node_cls = _RT.registry.get(node_type)
    if node_cls is None:
        raise RuntimeError(f"Unknown node type: {node_type}")
    if ctx is not None:
        ctx.current_node_id = node_id
        if ctx.seed is not None:
            # What the graph engine does before every node: a seed derived
            # from (run seed, node id), so a node's weight init, dropout
            # mask, shuffle order and augmentation depend on the run seed
            # and its own identity and on NOTHING else -- not on how many
            # numbers the nodes before it drew.
            _RT.seed_rngs(ctx.derive_seed(node_id))
    wired = {
        key: value for key, value in (inputs or {}).items() if value is not _ABSENT
    }
    if _VERBOSE:
        print(f"[{node_id}] running", file=sys.stderr)

    def progress(data, _node_id=node_id):
        # The engine always hands nodes a live callback; only the printing
        # is gated on verbosity.
        if _VERBOSE:
            print(f"[{_node_id}] progress", file=sys.stderr)

    try:
        result = _RT.invoke_node(
            node_cls(), wired, params, progress_callback=progress, context=ctx
        )
    except Exception as exc:
        print(f"[{node_id}] error: {exc}", file=sys.stderr)
        raise
    if _VERBOSE:
        print(f"[{node_id}] completed", file=sys.stderr)
    return result
'''

_CLI_TAIL = '''def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the CodefyUI graph embedded in this Python file.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "Execution device: auto, cpu, cuda, cuda:N (a specific GPU), "
            "or mps (default: auto). An unavailable backend falls back to "
            "the CPU and an unusable GPU index falls back to the default "
            "GPU; either way the substitution is logged."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=(
            "Optional soft timeout in seconds, checked between nodes: the "
            "node already running finishes, and no further nodes start."
        ),
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=None,
        help=(
            "CodefyUI project directory used to resolve project assets. "
            "Set this when the graph refers to project-local models/images."
        ),
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--inputs-json",
        default=None,
        help="JSON object to inject into GraphInput nodes.",
    )
    input_group.add_argument(
        "--inputs-file",
        type=Path,
        default=None,
        help="UTF-8 JSON file containing GraphInput values.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=GRAPH_SEED,
        help=(
            "Run seed. Every node is seeded from (seed, node id), so weight "
            "init, shuffling and augmentation reproduce exactly. Defaults to "
            "the seed set on the canvas when this file was exported; pass "
            "--no-seed for fresh entropy on every run."
        ),
    )
    parser.add_argument(
        "--no-seed",
        dest="seed",
        action="store_const",
        const=None,
        help="Ignore the exported seed and use torch's own entropy.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=GRAPH_DETERMINISTIC,
        help=(
            "Ask torch for reproducible kernels as well as a fixed seed "
            "(warn_only, so an op with no deterministic implementation warns "
            "rather than failing). Slower."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-node progress and a traceback on failure.",
    )
    return parser


def _load_inputs(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.inputs_json is None and args.inputs_file is None:
        return None

    try:
        raw = (
            args.inputs_file.read_text(encoding="utf-8")
            if args.inputs_file is not None
            else args.inputs_json
        )
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read inputs JSON: {exc}") from exc

    if not isinstance(value, dict):
        raise ValueError("inputs JSON must be an object")
    return value


def _load_runtime(project_dir: Path | None) -> SimpleNamespace:
    # Settings are created when app.config is imported, so apply the project
    # override before importing any CodefyUI module.
    if project_dir is not None:
        os.environ["CODEFYUI_PROJECT_DIR"] = str(project_dir.resolve())

    try:
        from app.core import api_contract
        from app.core.device_utils import describe_accelerator, resolve_device
        from app.core.execution_context import ExecutionContext
        from app.core.graph_engine import invoke_node
        from app.core.node_registry import registry
        from app.core.runtime import initialize_runtime
        from app.core.seeding import deterministic_scope, seed_rngs
    except (ImportError, ModuleNotFoundError) as exc:
        missing = getattr(exc, "name", None) or str(exc)
        raise RuntimeError(
            "CodefyUI backend runtime is unavailable "
            f"(failed import: {missing}). Run this file with the Python "
            "environment from a compatible CodefyUI installation."
        ) from exc

    return SimpleNamespace(
        api_contract=api_contract,
        describe_accelerator=describe_accelerator,
        resolve_device=resolve_device,
        ExecutionContext=ExecutionContext,
        invoke_node=invoke_node,
        registry=registry,
        initialize_runtime=initialize_runtime,
        deterministic_scope=deterministic_scope,
        seed_rngs=seed_rngs,
    )


def _print_problems(title: str, problems: list[Any]) -> None:
    print(title, file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)


def _run(args: argparse.Namespace) -> int:
    global _RT, _DEADLINE
    try:
        request_inputs = _load_inputs(args)
        _RT = _load_runtime(args.project_dir)
        discovery_stream = (
            redirect_stdout(sys.stderr) if _HAS_GRAPH_OUTPUT else nullcontext()
        )
        with discovery_stream:
            _RT.initialize_runtime()
    except (RuntimeError, ValueError) as exc:
        print(f"Exported graph setup failed: {exc}", file=sys.stderr)
        if args.verbose:
            raise
        return 2

    unresolved = sorted(
        node_type
        for node_type in _REQUIRED_NODE_TYPES
        if _RT.registry.get(node_type) is None
    )
    if unresolved:
        details = "; ".join(f"Unknown node type: {t}" for t in unresolved)
        print(f"Exported graph validation failed: {details}", file=sys.stderr)
        return 2

    for title, problems in _STATIC_PROBLEMS:
        _print_problems(title, problems)
    if _STATIC_PROBLEMS:
        return 2

    contract = _RT.api_contract.derive_contract(_IO_NODES) if _IO_NODES else None

    provided: dict[str, Any] = {}
    if request_inputs is not None:
        if contract is None or not contract.inputs:
            print(
                "Inputs were supplied, but this graph has no GraphInput nodes.",
                file=sys.stderr,
            )
            return 2
        patched, input_errors = _RT.api_contract.inject_inputs(
            _IO_NODES, contract, request_inputs
        )
        if input_errors:
            _print_problems("Graph inputs are invalid:", input_errors)
            return 2
        provided = {
            node["id"]: node["data"]["params"]["value"]
            for node in patched
            if node.get("type") == "GraphInput"
            and "value" in node.get("data", {}).get("params", {})
        }

    try:
        requested_device = (args.device or "auto").strip().lower() or "auto"
        resolved_device = (
            _RT.describe_accelerator()["default"]
            if requested_device == "auto"
            else _RT.resolve_device(requested_device)
        )
        context = _RT.ExecutionContext(
            device=resolved_device,
            weights_persistent=False,
            graph_id=f"export:{GRAPH_NAME}",
            # The canvas seed is baked in as the default above, so an
            # exported graph reproduces what the canvas produced without
            # anyone remembering to pass a flag.
            seed=None if args.seed is None else int(args.seed),
            deterministic=bool(args.deterministic),
        )

        if args.timeout is not None:
            _DEADLINE = time.monotonic() + args.timeout

        # Contract runners reserve stdout for their final JSON object.  Nodes
        # such as Print still remain visible on stderr.  Flows run on the main
        # thread with no event loop, like the engine's worker threads, so
        # nodes that call asyncio.run() internally keep working.
        output_stream = (
            redirect_stdout(sys.stderr) if _HAS_GRAPH_OUTPUT else nullcontext()
        )
        # The engine wraps every run the same way and hands the process
        # setting back afterwards, so a TrainingLoop whose own deterministic
        # param is on cannot leak that state past this run.
        with _RT.deterministic_scope(context.deterministic), output_stream:
            results = run_graph(context, provided)
    except Exception as exc:
        print(f"Graph execution failed: {exc}", file=sys.stderr)
        if args.verbose:
            raise
        return 1

    if _HAS_GRAPH_OUTPUT and contract is not None:
        collected, missing = _RT.api_contract.collect_outputs(contract, results)
        if missing:
            _print_problems("Graph outputs are missing:", missing)
            return 1
        try:
            serialized = {
                name: _RT.api_contract.serialize_output(value)
                for name, value in collected.items()
            }
            output_json = json.dumps(
                serialized,
                ensure_ascii=False,
                sort_keys=True,
            )
        except Exception as exc:
            print(f"Graph output serialization failed: {exc}", file=sys.stderr)
            if args.verbose:
                raise
            return 1
        print(output_json)

    print(
        f"CodefyUI graph {GRAPH_NAME!r} completed on {resolved_device} "
        f"({len(results)} node results).",
        file=sys.stderr,
    )
    return 0


def main() -> None:
    global _VERBOSE
    args = _parser().parse_args()
    _VERBOSE = args.verbose
    raise SystemExit(_run(args))


if __name__ == "__main__":
    main()
'''


# ── Generation planning ──────────────────────────────────────────────────


def _split_flows(
    exec_nodes: list[dict],
    exec_edges: list[dict],
    order: list[str],
) -> list[list[str]]:
    """Group nodes into weakly-connected components over ALL executable edges.

    Components are ordered by their first appearance in *order*; the nodes
    inside each component keep the global topological order.  Components
    share no data by construction, so running them one after another is
    observably equivalent to the engine's level schedule.
    """
    parent = {node["id"]: node["id"] for node in exec_nodes}

    def find(node_id: str) -> str:
        root = node_id
        while parent[root] != root:
            root = parent[root]
        while parent[node_id] != root:  # path compression
            parent[node_id], node_id = root, parent[node_id]
        return root

    for edge in exec_edges:
        parent[find(edge["source"])] = find(edge["target"])

    groups: dict[str, list[str]] = {}
    for node_id in order:
        groups.setdefault(find(node_id), []).append(node_id)
    return list(groups.values())


def _static_problems(
    nodes: list[dict],
    edges: list[dict],
    has_graph_output: bool,
) -> list[tuple[str, list[str]]]:
    """Compute the contract/wiring problems today's runner reports at startup.

    At most one titled group is returned — the runner exits on the first
    failing check, so later groups would never have been shown anyway.
    """
    has_contract_nodes = any(
        node.get("type") in (GRAPH_INPUT_TYPE, GRAPH_OUTPUT_TYPE)
        for node in nodes
    )
    if not has_contract_nodes:
        return []

    contract = derive_contract(nodes)
    # A canvas-style graph may use GraphInput defaults without declaring a
    # GraphOutput.  That is still runnable; ignore only that one API-contract
    # complaint and retain every other name/type/default validation error.
    contract_problems = [
        problem
        for problem in contract.problems
        if not (
            not has_graph_output
            and problem.startswith("graph has no GraphOutput node")
        )
    ]
    if contract_problems:
        return [("Graph I/O contract is invalid:", contract_problems)]

    wiring = check_wiring(nodes, edges, contract)
    wiring_problems = [
        *(f"untriggered GraphInput: {name}" for name in wiring.untriggered),
        *(f"unreachable GraphOutput: {name}" for name in wiring.unreachable),
    ]
    if wiring_problems:
        return [("Graph I/O wiring is invalid:", wiring_problems)]
    return []


def generate_python(
    nodes: list[dict],
    edges: list[dict],
    name: str = "Untitled",
    presets: list[dict] | None = None,
    *,
    seed: int | None = None,
    deterministic: bool = False,
    subgraphs: list[dict] | None = None,
) -> str:
    """Return a runnable Python program for the graph *nodes* / *edges*.

    Presets are expanded and drafts pruned with the engine's own
    :func:`prepare_executable_graph`, so the emitted functions cover exactly
    the executable graph the canvas would run.  Raises
    :class:`app.core.graph_engine.GraphValidationError` for graphs the
    engine would refuse to run.

    *subgraphs* are the graph's own subgraph definitions (core#137). Each
    INSTANCE becomes one ``subgraph_<name>`` function holding its inner node
    calls, so the exported file keeps the block structure the canvas shows
    instead of dissolving it into a flat run of node calls.

    *seed* and *deterministic* are the canvas's reproducibility settings,
    baked in as the defaults for the generated ``--seed`` /
    ``--deterministic`` flags.  Without them the exported script built an
    unseeded ``ExecutionContext``, so ``derive_seed`` returned ``None``, no
    ``SeededAugmentation`` was ever installed, and an exported augmenting
    graph produced different crops on every invocation -- while the
    augmentation docs promised the opposite.
    """
    preset_fallback = build_preset_fallback(presets or [])
    exec_nodes, exec_edges, internal_to_preset = prepare_executable_graph(
        nodes,
        edges,
        preset_fallback=preset_fallback,
        subgraphs=subgraphs,
    )
    # Preset expansion can pull params straight from server-side registry
    # definitions, which the export route's payload scrub never saw.  Scrub
    # the expanded graph (on a copy, so callers' node dicts stay untouched):
    # a SECRET value must never be embedded in an exported file.
    exec_nodes = copy.deepcopy(exec_nodes)
    scrub_graph_secrets(exec_nodes)
    order = topological_sort(exec_nodes, exec_edges)

    node_by_id = {node["id"]: node for node in exec_nodes}
    seq_by_id = {node_id: seq for seq, node_id in enumerate(order, start=1)}

    # Wiring map: data edges only, in edges[] array order per target handle.
    # Trigger edges are ordering markers; Start produces {}, so the engine's
    # conditional injection never fires for them.
    wired_by_target: dict[str, dict[str, list[tuple[str, str]]]] = {}
    for edge in exec_edges:
        if edge.get("type", "data") == "trigger":
            continue
        handles = wired_by_target.setdefault(edge["target"], {})
        handles.setdefault(edge.get("targetHandle", ""), []).append(
            (edge["source"], edge.get("sourceHandle", ""))
        )

    flows = _split_flows(exec_nodes, exec_edges, order)
    raw_by_id = {node.get("id"): node for node in nodes}

    func_names = {
        node_id: (
            f"n{seq_by_id[node_id]:02d}_"
            f"{_slug(node_by_id[node_id].get('type', ''), 'node')}"
        )
        for node_id in order
    }
    flow_names = {f"flow_{index}" for index in range(1, len(flows) + 1)}

    # ── Subgraph grouping (core#137) ─────────────────────────────────────
    #
    # `internal_to_preset` maps every expanded inner node to its container.
    # Walking it to the OUTERMOST container (with `outermost_container`, the
    # engine's own cycle-guarded walk) and asking the RAW graph what that
    # container was separates subgraph instances from presets: a preset stays
    # flat (its origin is a comment), an instance becomes a function, because
    # the canvas shows it as one box the user can open and edit.
    def _outermost_container(node_id: str) -> str | None:
        return outermost_container(node_id, internal_to_preset)

    subgraph_of: dict[str, str] = {}
    subgraph_members: dict[str, list[str]] = {}
    for node_id in order:
        root = _outermost_container(node_id)
        if root is None:
            continue
        raw_type = str((raw_by_id.get(root) or {}).get("type", ""))
        if not raw_type.startswith(SUBGRAPH_TYPE_PREFIX):
            continue
        subgraph_of[node_id] = root
        subgraph_members.setdefault(root, []).append(node_id)

    # ── Is each instance schedulable as ONE call? ────────────────────────
    #
    # An instance becomes one function CALLED ONCE, at the position of its
    # FIRST member in topological order -- so every member runs there, not
    # where it sits in the order.  Two conditions have to hold for that to
    # mean the same thing as running the members in place, and BOTH are
    # checked here rather than assumed:
    #
    #   1. Every value the block reads from outside is already assigned at
    #      that position.  This is the real form of "no outside node sits in
    #      the middle of the block": an outside node fed BY a member and
    #      feeding one is necessarily ordered after the first member, so it
    #      is caught -- but so is an unrelated feeder that Kahn's tie-break
    #      happened to place late, which a pure convexity test would miss.
    #   2. The members all live in ONE flow.  `_split_flows` cuts the graph
    #      into weakly-connected components emitted as separate functions;
    #      an instance straddling two of them would be CALLED from each, and
    #      every one of its members would run twice.
    #
    # The editor refuses to COLLAPSE a selection an outside node sits in the
    # middle of, but that refusal is not an invariant codegen may lean on:
    # two edge drags after a perfectly legal collapse recreate the shape, and
    # an imported, plugin-produced or hand-edited graph never went through
    # the editor at all.  The engine runs all of these correctly, so the
    # exported script must too.
    #
    # An instance that fails either condition is dropped from `subgraph_of`
    # and `subgraph_members`, which is all it takes for every consumer below
    # (`_source_expr`, `subgraph_func_names`, the "Subgraph functions"
    # section, `_emit_flow`) to fall back to the ordinary flat path.  Its
    # members are then emitted inline, in their true topological positions;
    # the block loses its function, never its identity, because
    # `_preset_origin` still stamps each member with the subgraph it is from.
    data_sources: dict[str, set[str]] = {}
    for edge in exec_edges:
        # Trigger edges carry no value and are ignored by `topological_sort`,
        # so they constrain neither the reads nor the order emitted here.
        if edge.get("type", "data") == "trigger":
            continue
        data_sources.setdefault(edge["target"], set()).add(edge["source"])

    flow_of = {
        node_id: index
        for index, flow_members in enumerate(flows)
        for node_id in flow_members
    }

    def _not_one_call(members: list[str]) -> str | None:
        """Why this instance cannot be one call -- ``None`` when it can."""
        spread = {flow_of[member] for member in members}
        if len(spread) > 1:
            return (
                f"its members are split across {len(spread)} flows that run "
                "one after another"
            )
        call_at = min(seq_by_id[member] for member in members)
        # Walk ANCESTORS, not just direct sources: a value the block reads
        # only exists once the chain that produces it has run.
        seen = set(members)  # seeded, so anything reached below is outside
        queue = deque(members)
        late: list[str] = []
        while queue:
            for source in data_sources.get(queue.popleft(), ()):
                if source in seen:
                    continue
                seen.add(source)
                queue.append(source)
                position = seq_by_id.get(source)
                if position is not None and position > call_at:
                    late.append(source)
        if late:
            blocker = min(late, key=lambda node_id: seq_by_id[node_id])
            return f"node {blocker!r} runs in the middle of it"
        return None

    # member id -> instance id, for instances that failed the test above.
    inline_subgraph_of: dict[str, str] = {}
    inline_notes: dict[str, str] = {}
    for instance_id in list(subgraph_members):
        reason = _not_one_call(subgraph_members[instance_id])
        if reason is None:
            continue
        raw_type = str((raw_by_id.get(instance_id) or {}).get("type", ""))
        sid = raw_type[len(SUBGRAPH_TYPE_PREFIX):] or instance_id
        inline_notes[instance_id] = (
            f"subgraph {ascii(sid)} (node {ascii(instance_id)}) is inlined "
            f"here instead of called as one function: {reason}."
        )
        for member in subgraph_members.pop(instance_id):
            del subgraph_of[member]
            inline_subgraph_of[member] = instance_id

    used_subgraph_names: set[str] = set()
    subgraph_func_names: dict[str, str] = {}
    for instance_id in subgraph_members:
        raw_type = str((raw_by_id.get(instance_id) or {}).get("type", ""))
        label = raw_type[len(SUBGRAPH_TYPE_PREFIX):] or instance_id
        subgraph_func_names[instance_id] = _identifier(
            f"subgraph_{label}",
            used=used_subgraph_names,
            avoid=frozenset(
                _RESERVED_LOCALS | set(func_names.values()) | flow_names
            ),
            fallback="subgraph",
        )

    # Result locals: derived from node IDs, never shadowing anything a flow
    # body references (helpers, node functions, flow functions).
    local_avoid = frozenset(
        _RESERVED_LOCALS
        | set(func_names.values())
        | flow_names
        | set(subgraph_func_names.values())
    )
    used_locals: set[str] = set()
    local_names = {
        node_id: _identifier(
            node_id, used=used_locals, avoid=local_avoid, fallback="node"
        )
        for node_id in order
    }

    # Per-function parameter names for wired input ports.  Dict keys at the
    # call site stay the REAL port-name strings; only the identifiers are
    # sanitized.
    input_params: dict[str, dict[str, str]] = {}
    for node_id in order:
        is_graph_input = node_by_id[node_id].get("type") == GRAPH_INPUT_TYPE
        avoid = frozenset(
            {"ctx", "params", "value"} if is_graph_input else {"ctx", "params"}
        )
        used_params: set[str] = set()
        input_params[node_id] = {
            handle: _identifier(
                handle, used=used_params, avoid=avoid, fallback="input"
            )
            for handle in wired_by_target.get(node_id, {})
        }

    # ── Bypassed nodes (core#128) ────────────────────────────────────────
    #
    # `prepare_executable_graph` already removed them, so the script would
    # simply never mention them.  Resolving the RAW graph a second time here
    # is purely so the export SHOWS what was muted and what each of its
    # outputs stood in for -- a commented-out assignment, sitting immediately
    # above the first node that consumes the pass-through.
    #
    # It re-resolves rather than reusing the executable pass because the
    # resolution is internal to that call; the only cost is a second port
    # lookup per bypassed node, and the only divergence is a bypassed node fed
    # by a PRESET (whose expanded internals the raw graph does not have), which
    # simply has no local variable to name -- handled below.
    bypass = resolve_bypass(nodes, edges)
    kept_ids = {kept.get("id") for kept in bypass.nodes}
    # Graph order, so the emitted comments are deterministic.
    bypassed_ids = [
        node.get("id")
        for node in nodes
        if node.get("id") is not None and node.get("id") not in kept_ids
    ]
    bypassed_id_set = set(bypassed_ids)
    links_by_node: dict[str, list[BypassLink]] = {}
    for link in bypass.links:
        links_by_node.setdefault(link.node_id, []).append(link)

    raw_downstream: dict[str, list[str]] = {}
    for edge in edges:
        if edge.get("type", "data") == "data":
            raw_downstream.setdefault(edge["source"], []).append(edge["target"])

    def _bypass_anchor(node_id: str) -> str | None:
        """Earliest emitted node that consumes this bypass, through chains."""
        best: str | None = None
        seen = {node_id}
        queue = deque([node_id])
        while queue:
            for target in raw_downstream.get(queue.popleft(), []):
                if target in seen:
                    continue
                seen.add(target)
                if target in bypassed_id_set:
                    queue.append(target)
                elif target in seq_by_id and (
                    best is None or seq_by_id[target] < seq_by_id[best]
                ):
                    best = target
        return best

    def _bypass_lines(node_id: str) -> list[str]:
        node_type = str((raw_by_id.get(node_id) or {}).get("type", ""))
        lines = [
            f"# BYPASSED node {_literal(node_id)} ({_literal(node_type)}) -- "
            "not executed; its inputs pass straight through:"
        ]
        node_links = links_by_node.get(node_id, [])
        if not node_links:
            # A pass-through is only ever resolved for an output something
            # downstream reads, so an empty link list means nothing consumed
            # this node -- it says nothing about what was wired INTO it.
            lines.append("#     (nothing downstream consumed it)")
            return lines
        stand_in = _slug(node_id, "node")
        for link in node_links:
            source_local = local_names.get(link.source)
            rhs = (
                f"_port({source_local}, {_literal(link.source_handle)})"
                if source_local is not None
                else (
                    f"output {_literal(link.source_handle)} of node "
                    f"{_literal(link.source)}"
                )
            )
            lines.append(
                f"#     {stand_in}[{_literal(link.output)}] = {rhs}"
                f"  # via input {_literal(link.input)}"
            )
        return lines

    # node_id -> comment block emitted just before it inside its flow body.
    bypass_comments: dict[str, list[str]] = {}
    orphan_bypass_lines: list[str] = []
    for node_id in bypassed_ids:
        anchor = _bypass_anchor(node_id)
        if anchor is None:
            orphan_bypass_lines.extend(_bypass_lines(node_id))
        else:
            bypass_comments.setdefault(anchor, []).extend(_bypass_lines(node_id))

    def _preset_origin(node_id: str) -> str | None:
        root = _outermost_container(node_id)
        if root is None:
            return None
        raw_type = (raw_by_id.get(root) or {}).get("type", "")
        if raw_type.startswith("preset:"):
            preset_name = raw_type[len("preset:"):]
            return f"# from preset {ascii(preset_name)} (node {ascii(root)})"
        if raw_type.startswith(SUBGRAPH_TYPE_PREFIX):
            sid = raw_type[len(SUBGRAPH_TYPE_PREFIX):]
            return f"# from subgraph {ascii(sid)} (node {ascii(root)})"
        return f"# from preset node {ascii(root)}"

    def _emit_node_function(node_id: str) -> str:
        node = node_by_id[node_id]
        node_type = node.get("type", "")
        params = node.get("data", {}).get("params", {})
        handles = wired_by_target.get(node_id, {})
        pnames = input_params[node_id]
        is_graph_input = node_type == GRAPH_INPUT_TYPE

        signature = ["ctx"]
        if is_graph_input:
            signature.append("value=_ABSENT")
        if handles:
            signature.append("*")
            signature.extend(f"{pnames[handle]}=_ABSENT" for handle in handles)

        lines = [f"def {func_names[node_id]}({', '.join(signature)}):"]
        lines.append(f"    {ascii(f'{node_type} - node {node_id!r}.')}")
        origin = _preset_origin(node_id)
        if origin is not None:
            lines.append(f"    {origin}")

        call_args = [_literal(node_type), _literal(node_id)]
        if params or is_graph_input:
            lines.extend(_emit_params(params, node_type, node_id))
            call_args.append("params")
        else:
            call_args.append("{}")
        if is_graph_input:
            # api_contract.inject_inputs writes caller-supplied values into
            # params["value"]; merging here reproduces that exactly.
            lines.append("    if value is not _ABSENT:")
            lines.append("        params = {**params, 'value': value}")
        call_args.append("ctx")
        if handles:
            items = ", ".join(
                f"{_literal(handle)}: {pnames[handle]}" for handle in handles
            )
            call_args.append(f"inputs={{{items}}}")
        lines.append(f"    return _call({', '.join(call_args)})")
        return "\n".join(lines) + "\n"

    def _flow_summary(member_ids: list[str]) -> str:
        parts = [node_by_id[member].get("type", "") for member in member_ids]
        summary = " -> ".join(parts)
        while len(summary) > 72 and len(parts) > 1:  # drop whole steps first
            parts.pop()
            summary = " -> ".join(parts) + " -> ..."
        if len(summary) > 72:
            summary = summary[:69] + "..."
        return summary

    def _source_expr(source: str, *, inside_subgraph: bool) -> str:
        """How a consumer names another node's outputs.

        A flow body keeps the readable local (``dataset``); a subgraph body,
        and anything reading ACROSS a boundary, goes through ``results`` --
        members of a subgraph live in a different function scope, so their
        locals are simply not in scope where the flow needs them.
        """
        if inside_subgraph or source in subgraph_of:
            return f"results[{_literal(source)}]"
        return local_names[source]

    def _member_statement(member: str, *, inside_subgraph: bool) -> list[str]:
        """The call that runs one node, indented four spaces."""
        lines: list[str] = []
        for comment in bypass_comments.get(member, ()):
            lines.append(f"    {comment}")
        node = node_by_id[member]
        kwargs: list[str] = []
        if node.get("type") == GRAPH_INPUT_TYPE:
            kwargs.append(f"value=provided.get({_literal(member)}, _ABSENT)")
        pnames = input_params[member]
        for handle, sources in wired_by_target.get(member, {}).items():
            ports = [
                f"_port("
                f"{_source_expr(source, inside_subgraph=inside_subgraph)}, "
                f"{_literal(source_handle)})"
                for source, source_handle in sources
            ]
            if len(ports) == 1:
                value_expr = ports[0]
            else:
                # Reverse edge order: in the engine the last edge whose
                # source port produced a value wins.
                value_expr = f"_pick({', '.join(reversed(ports))})"
            kwargs.append(f"{pnames[handle]}={value_expr}")
        target = f"results[{_literal(member)}]"
        if not inside_subgraph:
            target = f"{local_names[member]} = {target}"
        assign = f"    {target} = "
        if not kwargs:
            lines.append(f"{assign}{func_names[member]}(ctx)")
        else:
            lines.append(f"{assign}{func_names[member]}(")
            lines.append("        ctx,")
            lines.extend(f"        {kwarg}," for kwarg in kwargs)
            lines.append("    )")
        return lines

    def _emit_subgraph(instance_id: str) -> str:
        raw_type = str((raw_by_id.get(instance_id) or {}).get("type", ""))
        sid = raw_type[len(SUBGRAPH_TYPE_PREFIX):] or instance_id
        members = subgraph_members[instance_id]
        lines = [
            f"def {subgraph_func_names[instance_id]}"
            "(ctx, results, provided):"
        ]
        lines.append(
            f"    {ascii(f'subgraph {sid!r} - instance {instance_id!r}.')}"
        )
        lines.append(f"    # {_comment_text(_flow_summary(members))}")
        for member in members:
            lines.extend(_member_statement(member, inside_subgraph=True))
        return "\n".join(lines) + "\n"

    def _emit_flow(index: int, member_ids: list[str]) -> str:
        lines = [f"def flow_{index}(ctx, results, provided):"]
        lines.append(f"    {ascii(_flow_summary(member_ids))}")
        # A subgraph instance is emitted ONCE, at the position of its first
        # member in topological order, which runs every member there.  Only
        # instances CHECKED to be schedulable that way -- all outside values
        # they read already assigned, all members in this one flow -- reach
        # `subgraph_of`; the rest arrive with their members ungrouped and
        # take the ordinary per-node path below, each in its true position.
        emitted_subgraphs: set[str] = set()
        noted_inline: set[str] = set()
        for member in member_ids:
            instance_id = subgraph_of.get(member)
            if instance_id is not None:
                if instance_id in emitted_subgraphs:
                    continue
                emitted_subgraphs.add(instance_id)
                lines.append(
                    f"    {subgraph_func_names[instance_id]}"
                    "(ctx, results, provided)"
                )
                continue
            inline_id = inline_subgraph_of.get(member)
            if inline_id is not None and inline_id not in noted_inline:
                # Say WHY the block is not one function; a silent flattening
                # reads like the exporter simply forgot about subgraphs.
                noted_inline.add(inline_id)
                lines.append(f"    # {_comment_text(inline_notes[inline_id])}")
            lines.extend(_member_statement(member, inside_subgraph=False))
        return "\n".join(lines) + "\n"

    # ── Baked data ───────────────────────────────────────────────────────
    io_nodes = [
        {
            "id": node.get("id", ""),
            "type": node.get("type", ""),
            "data": {"params": node.get("data", {}).get("params", {})},
        }
        for node in nodes
        if node.get("type") in (GRAPH_INPUT_TYPE, GRAPH_OUTPUT_TYPE)
    ]
    has_graph_output = any(
        node.get("type") == GRAPH_OUTPUT_TYPE for node in nodes
    )
    required_types = sorted({node.get("type", "") for node in exec_nodes})

    literals = (
        f"GRAPH_NAME = {_literal(name)}\n"
        f"_HAS_GRAPH_OUTPUT = {has_graph_output!r}\n"
        "\n"
        "# Reproducibility, baked from the canvas at export time (core#136).\n"
        "# These are the DEFAULTS for --seed / --deterministic; override them\n"
        "# on the command line, or edit them here.\n"
        f"GRAPH_SEED = {_literal(seed)}\n"
        f"GRAPH_DETERMINISTIC = {bool(deterministic)!r}\n"
        "\n"
        "# Raw GraphInput/GraphOutput declarations (data, not code), consumed\n"
        "# by the canonical app.core.api_contract helpers at run time.\n"
        f"_IO_NODES = {_literal(io_nodes)}\n"
        "\n"
        "# Contract/wiring problems detected when this file was generated.\n"
        f"_STATIC_PROBLEMS = "
        f"{_literal(_static_problems(nodes, edges, has_graph_output))}\n"
        "\n"
        "# Node types this graph needs, verified against the registry after\n"
        "# runtime discovery.\n"
        f"_REQUIRED_NODE_TYPES = {_literal(required_types)}\n"
    )

    # ── Assembly ─────────────────────────────────────────────────────────
    sections: list[str] = [_SCRIPT_HEADER, literals, _RUNTIME_PRELUDE]

    banner = "# " + "=" * 28 + " Node functions " + "=" * 28
    node_sections: list[str] = [banner + "\n"]
    # Bypassed nodes nothing downstream consumes have no flow member to sit
    # above, so they are recorded here rather than dropped without trace.
    if orphan_bypass_lines:
        node_sections.append("\n".join(orphan_bypass_lines) + "\n")
    for index, member_ids in enumerate(flows, start=1):
        header = _comment_text(_flow_summary(member_ids))
        node_sections.append(f"# ---- Flow {index}: {header} ----\n")
        node_sections.extend(
            _emit_node_function(member) for member in member_ids
        )
    sections.append("\n\n".join(node_sections))

    if subgraph_members:
        banner = "# " + "=" * 26 + " Subgraph functions " + "=" * 26
        subgraph_sections: list[str] = [banner + "\n"]
        subgraph_sections.extend(
            _emit_subgraph(instance_id) for instance_id in subgraph_members
        )
        sections.append("\n\n".join(subgraph_sections))

    banner = "# " + "=" * 28 + " Flow functions " + "=" * 28
    flow_sections: list[str] = [banner + "\n"]
    flow_sections.extend(
        _emit_flow(index, member_ids)
        for index, member_ids in enumerate(flows, start=1)
    )
    flow_calls = "".join(
        f"    flow_{index}(ctx, results, provided)\n"
        for index in range(1, len(flows) + 1)
    )
    flow_sections.append(
        "def run_graph(ctx, provided):\n"
        '    """Execute every flow in engine order; return {node_id: outputs}."""\n'
        "    results = {}\n"
        f"{flow_calls}"
        "    return results\n"
    )
    sections.append("\n\n".join(flow_sections))

    sections.append(_CLI_TAIL)
    return "\n\n".join(sections)
