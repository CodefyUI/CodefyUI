#!/usr/bin/env python3
"""
CodefyUI Graph CLI Runner
==========================
Execute a graph.json directly from the command line without starting the server.

Usage:
    python run_graph.py <path_to_graph.json>
    python run_graph.py ../examples/Usage_Example/CNN-MNIST/TrainCNN-MNIST/graph.json
    python run_graph.py ../examples/Usage_Example/CNN-MNIST/TrainCNN-MNIST/graph.json --validate-only
    python run_graph.py ../examples/Usage_Example/CNN-MNIST/TrainCNN-MNIST/graph.json --verbose
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Ensure the backend package is importable
sys.path.insert(0, str(Path(__file__).parent))

from app.config import settings
from app.core.graph_engine import GraphValidationError, execute_graph, expand_presets, validate_graph
from app.core.logging_config import setup_logging
from app.core.node_registry import registry
from app.core.preset_registry import preset_registry

logger = logging.getLogger("codefyui.cli")


def _init_registries() -> None:
    """Discover all nodes and presets."""
    n = registry.discover(settings.NODES_DIR, "app.nodes")
    c = registry.discover(settings.CUSTOM_NODES_DIR, "app.custom_nodes")
    p = preset_registry.discover(settings.PRESETS_DIR, registry)
    logger.info("%d built-in nodes, %d custom nodes, %d presets", n, c, p)


def _on_progress(node_id: str, status: str, data: dict[str, Any] | None) -> None:
    """CLI progress callback — logs node execution status."""
    if status == "running":
        logger.info("  [%s] running...", node_id)
    elif status == "completed":
        # Summarize outputs
        parts = []
        if data:
            for key, val in data.items():
                if hasattr(val, "shape"):
                    parts.append(f"{key}: Tensor{list(val.shape)}")
                elif hasattr(val, "parameters"):
                    n_params = sum(p.numel() for p in val.parameters())
                    parts.append(f"{key}: Model({n_params:,} params)")
                elif isinstance(val, (int, float)):
                    parts.append(f"{key}: {val}")
                elif isinstance(val, str) and len(val) > 80:
                    parts.append(f"{key}: str({len(val)} chars)")
                else:
                    parts.append(f"{key}: {type(val).__name__}")
        summary = ", ".join(parts) if parts else "ok"
        logger.info("  [%s] completed  ->  %s", node_id, summary)
    elif status == "error":
        err = data.get("error", "unknown") if data else "unknown"
        logger.error("  [%s] ERROR: %s", node_id, err)


async def run(
    graph_path: str,
    *,
    validate_only: bool = False,
    verbose: bool = False,
    device: str | None = None,
) -> None:
    t0 = time.time()

    # Load graph
    path = Path(graph_path)
    if not path.exists():
        logger.error("File not found: %s", path)
        sys.exit(1)

    graph = json.loads(path.read_text(encoding="utf-8"))
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    name = graph.get("name", path.stem)

    logger.info("=" * 60)
    logger.info("  Graph: %s", name)
    logger.info("  File:  %s", path)
    logger.info("  Nodes: %d, Edges: %d", len(nodes), len(edges))
    logger.info("=" * 60)

    # Expand presets
    expanded_nodes, expanded_edges, preset_map = expand_presets(nodes, edges)
    if len(expanded_nodes) != len(nodes):
        logger.info("Expanded %d nodes -> %d (presets resolved)", len(nodes), len(expanded_nodes))

    # Validate
    logger.info("Checking graph...")
    errors = validate_graph(expanded_nodes, expanded_edges)
    if errors:
        logger.error("Validation FAILED:")
        for e in errors:
            logger.error("  - %s", e)
        sys.exit(1)
    logger.info("Validation OK")

    if validate_only:
        logger.info("(--validate-only: skipping execution)")
        return

    # Execute
    logger.info("Starting graph execution...")
    logger.info("-" * 60)
    context = None
    if device:
        from app.core.device_utils import resolve_device
        from app.core.execution_context import ExecutionContext

        resolved = resolve_device(device)
        logger.info("Global device: %s", resolved)
        context = ExecutionContext(device=resolved)
    try:
        outputs = await execute_graph(
            nodes,
            edges,
            on_progress=_on_progress if verbose else _on_progress,
            context=context,
        )
    except GraphValidationError as e:
        logger.error("Validation error: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Runtime error: %s", e)
        if verbose:
            logger.exception("Full traceback:")
        sys.exit(1)

    elapsed = time.time() - t0
    logger.info("-" * 60)
    logger.info("Completed in %.1fs", elapsed)

    # Log final summary
    logger.info("=" * 60)
    logger.info("  Final Outputs")
    logger.info("=" * 60)
    for node_id, result in outputs.items():
        display_id = node_id
        # Shorten internal preset node IDs
        if "__" in node_id:
            preset_id, internal_id = node_id.split("__", 1)
            display_id = f"{preset_id}/{internal_id}"
        for key, val in result.items():
            if hasattr(val, "shape"):
                logger.info("  %s.%s = Tensor%s", display_id, key, list(val.shape))
            elif hasattr(val, "parameters"):
                n_params = sum(p.numel() for p in val.parameters())
                logger.info("  %s.%s = Model(%s params)", display_id, key, f"{n_params:,}")
            elif isinstance(val, (int, float)):
                logger.info("  %s.%s = %s", display_id, key, val)
            elif isinstance(val, str) and len(val) > 120:
                logger.info("  %s.%s = str(%d chars)", display_id, key, len(val))
            else:
                r = repr(val)
                if len(r) > 120:
                    r = r[:117] + "..."
                logger.info("  %s.%s = %s", display_id, key, r)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute a CodefyUI graph.json from the command line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("graph", help="Path to graph.json file")
    parser.add_argument("--validate-only", action="store_true", help="Only validate, do not execute")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output and tracebacks")
    parser.add_argument("--device", default=None, help="Global compute device: cpu / cuda / mps (default cpu)")
    args = parser.parse_args()

    level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=level)

    _init_registries()
    asyncio.run(run(args.graph, validate_only=args.validate_only, verbose=args.verbose, device=args.device))


if __name__ == "__main__":
    main()
