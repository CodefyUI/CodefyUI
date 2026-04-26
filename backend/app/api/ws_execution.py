import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from urllib.parse import urlparse

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..config import settings
from ..core.cache import ExecutionCache
from ..core.execution_context import CancellationError, ExecutionContext
from ..core.graph_engine import GraphValidationError, execute_graph

logger = logging.getLogger(__name__)

router = APIRouter()


def _summarize_single(value: Any) -> dict[str, Any]:
    """Generate a human-readable summary for a single output value."""
    try:
        import torch

        if isinstance(value, torch.Tensor):
            summary: dict[str, Any] = {
                "type": "tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            if value.numel() > 0 and value.is_floating_point():
                summary["min"] = round(float(value.min()), 4)
                summary["max"] = round(float(value.max()), 4)
                summary["mean"] = round(float(value.mean()), 4)
            elif value.numel() > 0:
                summary["min"] = int(value.min())
                summary["max"] = int(value.max())
            return summary
        if isinstance(value, torch.nn.Module):
            param_count = sum(p.numel() for p in value.parameters())
            return {
                "type": "model",
                "class": value.__class__.__name__,
                "params": param_count,
                "trainable": sum(p.numel() for p in value.parameters() if p.requires_grad),
            }
    except ImportError:
        pass
    if isinstance(value, (int, float, bool)):
        return {"type": "scalar", "value": value}
    if isinstance(value, str):
        summary: dict[str, Any] = {"type": "string", "value": value[:200]}
        rel = _models_dir_relative(value)
        if rel is not None:
            summary["download_path"] = rel
        return summary
    return {"type": type(value).__name__, "repr": repr(value)[:200]}


def _models_dir_relative(value: str) -> str | None:
    """If *value* points to an existing file under ``MODELS_DIR``, return
    the relative path (POSIX-style) so the frontend can build a download URL.
    Returns ``None`` otherwise — keeps the check silent on any unexpected input.
    """
    try:
        p = Path(value).resolve()
        if not p.is_file():
            return None
        models_dir = settings.MODELS_DIR.resolve()
        if not p.is_relative_to(models_dir):
            return None
        return p.relative_to(models_dir).as_posix()
    except (OSError, ValueError):
        return None


def _summarize_outputs(result: dict[str, Any]) -> dict[str, Any]:
    """Summarize all output ports of a node result."""
    summary = {}
    for key, val in result.items():
        if key.startswith("__"):
            continue
        summary[key] = _summarize_single(val)
    return summary


@router.websocket("/ws/execution")
async def websocket_execution(ws: WebSocket):
    # Validate Origin header against allowed CORS origins
    origin = ws.headers.get("origin")
    if origin:
        allowed = {urlparse(o).netloc for o in settings.CORS_ORIGINS}
        if urlparse(origin).netloc not in allowed:
            await ws.close(code=4003, reason="Origin not allowed")
            return

    await ws.accept()

    current_task: asyncio.Task | None = None
    current_context: ExecutionContext | None = None
    cache = ExecutionCache()

    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)

            action = data.get("action")
            if action == "execute":
                # Cancel any existing execution first
                if current_task and not current_task.done():
                    if current_context:
                        current_context.cancel()
                    current_task.cancel()

                nodes = data.get("nodes", [])
                edges = data.get("edges", [])
                error_mode = data.get("error_mode", "fail_fast")
                max_retries = data.get("max_retries", 0)
                changed_nodes = data.get("changed_nodes")  # partial re-execution hint
                run_id = data.get("run_id")
                record_outputs = bool(data.get("record_outputs", False))
                output_store = getattr(ws.app.state, "run_output_store", None)

                # Educational feature flags. Default values keep behaviour
                # unchanged for clients that don't send them.
                verbose_mode = bool(data.get("verbose_mode", False))
                graph_id = str(data.get("graph_id", "") or "")
                weights_persistent = bool(data.get("weights_persistent", True))
                backward_mode = bool(data.get("backward_mode", False))
                auto_backward = bool(data.get("auto_backward", False))
                node_state_store = getattr(ws.app.state, "node_state_store", None)

                current_context = ExecutionContext(
                    verbose=verbose_mode,
                    graph_id=graph_id,
                    weights_persistent=weights_persistent,
                    node_state_store=node_state_store,
                    backward_mode=backward_mode,
                    auto_backward=auto_backward,
                )

                async def on_progress(node_id: str, status: str, result: dict[str, Any] | None) -> None:
                    msg: dict[str, Any] = {
                        "type": "node_status",
                        "node_id": node_id,
                        "status": status,
                    }
                    if result and status == "error":
                        msg["error"] = result.get("error", "")
                    if result and status == "progress":
                        msg["progress"] = result
                    if result and status in ("completed", "cached"):
                        # Forward log output (from Print node etc.)
                        if "__log__" in result:
                            msg["log"] = str(result["__log__"])
                        # Forward base64 image data so the frontend can display it
                        for key, val in result.items():
                            if key.startswith("__"):
                                continue
                            if isinstance(val, str) and len(val) > 200 and val[:20].isalnum():
                                msg["image"] = val
                                break
                        # Generate output summaries for edge inspection.
                        # Forwarded on both "completed" and "cached" so the
                        # frontend edge tooltip + Inspector still populate
                        # when nodes are served from ExecutionCache.
                        msg["output_summary"] = _summarize_outputs(result)
                    await ws.send_text(json.dumps(msg))

                async def _run() -> None:
                    try:
                        start_msg: dict[str, Any] = {"type": "execution_start"}
                        if run_id:
                            start_msg["run_id"] = run_id
                        await ws.send_text(json.dumps(start_msg))
                        await execute_graph(
                            nodes,
                            edges,
                            on_progress=on_progress,
                            context=current_context,
                            error_mode=error_mode,
                            max_retries=max_retries,
                            cache=cache,
                            changed_nodes=changed_nodes,
                            run_id=run_id,
                            output_store=output_store,
                            record_outputs=record_outputs,
                        )
                        await ws.send_text(json.dumps({"type": "execution_complete"}))
                    except CancellationError:
                        await ws.send_text(json.dumps({"type": "execution_stopped"}))
                    except GraphValidationError as e:
                        await ws.send_text(json.dumps({"type": "execution_error", "error": str(e)}))
                    except Exception as e:
                        await ws.send_text(json.dumps({"type": "execution_error", "error": str(e)}))

                current_task = asyncio.create_task(_run())

            elif action == "stop":
                if current_context:
                    current_context.cancel()
                if current_task and not current_task.done():
                    current_task.cancel()
                else:
                    await ws.send_text(json.dumps({"type": "execution_stopped"}))

            elif action == "clear_cache":
                cache.clear()
                await ws.send_text(json.dumps({"type": "cache_cleared"}))

            else:
                await ws.send_text(json.dumps({"type": "error", "error": f"Unknown action: {action}"}))
    except WebSocketDisconnect:
        if current_context:
            current_context.cancel()
        if current_task and not current_task.done():
            current_task.cancel()
