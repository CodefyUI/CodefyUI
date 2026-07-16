"""Generate a runnable, single-file Python runner for a CodefyUI graph.

The exporter deliberately reuses CodefyUI's node registry and graph engine
instead of translating a small subset of nodes into handwritten source.  That
keeps exported execution aligned with the canvas for built-in, custom, and
plugin nodes, including future node types that the exporter does not know
about.  The generated file does not need a running CodefyUI server, but it
does need to be launched with a compatible CodefyUI backend environment.

All graph-controlled data is embedded as one JSON string literal.  Node IDs,
graph names, labels, paths, and parameters are therefore data rather than
Python identifiers or source fragments.
"""

from __future__ import annotations

import json


_GRAPH_JSON_SENTINEL = "__CODEFYUI_GRAPH_JSON_LITERAL__"


_SCRIPT_TEMPLATE = '''#!/usr/bin/env python3
"""Runnable CodefyUI graph export.

This file contains the graph and runs it without starting the CodefyUI server.
It requires a compatible CodefyUI backend Python environment (including every
custom/plugin node and third-party dependency used by the graph).

Development checkout examples:
  Windows: backend/.venv/Scripts/python.exe exported_graph.py
  macOS/Linux: backend/.venv/bin/python exported_graph.py

Use ``--help`` for device, input, timeout, and project-directory options.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from typing import Any


GRAPH_JSON = __CODEFYUI_GRAPH_JSON_LITERAL__


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the CodefyUI graph embedded in this Python file.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device: auto, cpu, cuda, or mps (default: auto).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=(
            "Optional soft timeout in seconds. A synchronous node already "
            "running in a worker thread may finish before the process exits."
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


def _load_runtime(project_dir: Path | None):
    # Settings are created when app.config is imported, so apply the project
    # override before importing any CodefyUI module.
    if project_dir is not None:
        os.environ["CODEFYUI_PROJECT_DIR"] = str(project_dir.resolve())

    try:
        from app.core import api_contract
        from app.core.device_utils import describe_accelerator, resolve_device
        from app.core.execution_context import ExecutionContext
        from app.core.graph_engine import (
            GraphValidationError,
            build_preset_fallback,
            execute_graph,
            prepare_executable_graph,
        )
        from app.core.runtime import initialize_runtime
    except (ImportError, ModuleNotFoundError) as exc:
        missing = getattr(exc, "name", None) or str(exc)
        raise RuntimeError(
            "CodefyUI backend runtime is unavailable "
            f"(failed import: {missing}). Run this file with the Python "
            "environment from a compatible CodefyUI installation."
        ) from exc

    return (
        api_contract,
        describe_accelerator,
        resolve_device,
        ExecutionContext,
        GraphValidationError,
        build_preset_fallback,
        execute_graph,
        prepare_executable_graph,
        initialize_runtime,
    )


def _print_problems(title: str, problems: list[Any]) -> None:
    print(title, file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)


async def _run(args: argparse.Namespace) -> int:
    try:
        request_inputs = _load_inputs(args)
        graph = json.loads(GRAPH_JSON)
        has_graph_output = any(
            node.get("type") == "GraphOutput" for node in graph["nodes"]
        )
        (
            api_contract,
            describe_accelerator,
            resolve_device,
            ExecutionContext,
            GraphValidationError,
            build_preset_fallback,
            execute_graph,
            prepare_executable_graph,
            initialize_runtime,
        ) = _load_runtime(args.project_dir)
        discovery_stream = (
            redirect_stdout(sys.stderr) if has_graph_output else nullcontext()
        )
        with discovery_stream:
            initialize_runtime()
    except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Exported graph setup failed: {exc}", file=sys.stderr)
        if args.verbose:
            raise
        return 2

    nodes = graph["nodes"]
    edges = graph["edges"]
    graph_name = graph.get("name", "Untitled")
    preset_fallback = build_preset_fallback(graph.get("presets", []))

    try:
        preflight_stream = (
            redirect_stdout(sys.stderr) if has_graph_output else nullcontext()
        )
        with preflight_stream:
            prepare_executable_graph(
                nodes,
                edges,
                preset_fallback=preset_fallback,
            )
    except GraphValidationError as exc:
        print(f"Exported graph validation failed: {exc}", file=sys.stderr)
        return 2

    has_contract_nodes = any(
        node.get("type") in ("GraphInput", "GraphOutput") for node in nodes
    )
    contract = api_contract.derive_contract(nodes) if has_contract_nodes else None

    if contract is not None:
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
            _print_problems("Graph I/O contract is invalid:", contract_problems)
            return 2

        wiring = api_contract.check_wiring(nodes, edges, contract)
        wiring_problems = [
            *(f"untriggered GraphInput: {name}" for name in wiring.untriggered),
            *(f"unreachable GraphOutput: {name}" for name in wiring.unreachable),
        ]
        if wiring_problems:
            _print_problems("Graph I/O wiring is invalid:", wiring_problems)
            return 2

    if request_inputs is not None:
        if contract is None or not contract.inputs:
            print(
                "Inputs were supplied, but this graph has no GraphInput nodes.",
                file=sys.stderr,
            )
            return 2
        nodes, input_errors = api_contract.inject_inputs(
            nodes, contract, request_inputs
        )
        if input_errors:
            _print_problems("Graph inputs are invalid:", input_errors)
            return 2

    try:
        requested_device = (args.device or "auto").strip().lower() or "auto"
        resolved_device = (
            describe_accelerator()["default"]
            if requested_device == "auto"
            else resolve_device(requested_device)
        )
        context = ExecutionContext(
            device=resolved_device,
            weights_persistent=False,
            graph_id=f"export:{graph_name}",
        )

        def on_progress(
            node_id: str, status: str, data: dict[str, Any] | None
        ) -> None:
            if args.verbose or status == "error":
                suffix = ""
                if status == "error" and data:
                    suffix = f": {data.get('error', data)}"
                print(f"[{node_id}] {status}{suffix}", file=sys.stderr)

        # Contract runners reserve stdout for their final JSON object.  Nodes
        # such as Print still remain visible on stderr, including when they run
        # inside the engine's worker threads.
        output_stream = redirect_stdout(sys.stderr) if has_graph_output else nullcontext()
        with output_stream:
            execution = execute_graph(
                nodes,
                edges,
                on_progress=on_progress,
                context=context,
                error_mode="fail_fast",
                preset_fallback=preset_fallback,
            )
            result = (
                await execution
                if args.timeout is None
                else await asyncio.wait_for(execution, timeout=args.timeout)
            )
    except Exception as exc:
        print(f"Graph execution failed: {exc}", file=sys.stderr)
        if args.verbose:
            raise
        return 1

    if has_graph_output and contract is not None:
        collected, missing = api_contract.collect_outputs(contract, result)
        if missing:
            _print_problems("Graph outputs are missing:", missing)
            return 1
        try:
            serialized = {
                name: api_contract.serialize_output(value)
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
        f"CodefyUI graph {graph_name!r} completed on {resolved_device} "
        f"({len(result)} node results).",
        file=sys.stderr,
    )
    return 0


def main() -> None:
    args = _parser().parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
'''


def generate_python(
    nodes: list[dict],
    edges: list[dict],
    order: list[str] | None = None,
    name: str = "Untitled",
    presets: list[dict] | None = None,
) -> str:
    """Return a runnable Python script containing *nodes* and *edges*.

    ``order`` remains accepted for compatibility with the original direct
    code generator.  The production graph engine computes execution levels at
    runtime, so source generation no longer turns node IDs into Python
    variables or depends on this precomputed ordering.
    """

    del order
    graph = {"name": name, "nodes": nodes, "edges": edges}
    if presets:
        graph["presets"] = presets
    graph_json = json.dumps(
        graph,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return _SCRIPT_TEMPLATE.replace(_GRAPH_JSON_SENTINEL, repr(graph_json), 1)
