"""Backward-pass orchestration for the gradient inspector (A3).

When ``ExecutionContext.backward_mode`` is on, ``graph_engine``:

  1. After each node completes, calls :func:`attach_retain_grad` on its
     result tensors so non-leaf ``.grad`` is preserved through the
     ``.backward()`` call.
  2. Tracks every captured (node_id, port) → tensor in ``ctx.grad_targets``.
  3. After the entire graph finishes, calls :func:`select_backward_target`
     to pick (or skip) a tensor to backward through.
  4. If a target is found, runs :func:`run_backward` then
     :func:`capture_grads` to write ``{port}__grad`` entries (and
     ``__weight_grad__{name}`` for persisted modules) into ``RunOutputStore``.

Storage convention:
  * Forward port gradients use port name ``{forward_port}__grad`` and an
    accompanying ``{forward_port}__grad__meta`` containing the health dict.
  * Persisted module weight gradients use ``__weight_grad__{param_name}``
    and ``__weight_grad__{param_name}__meta``.
  * Both forms have leading ``__`` so they don't pollute edge tooltips.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    import torch
    from .node_state_store import NodeStateStore
    from .run_output_store import RunOutputStore


# ── Health heuristic ────────────────────────────────────────────────


def grad_health(grad: "torch.Tensor") -> dict[str, Any]:
    """Classify a gradient as vanishing / exploding / healthy and return stats."""
    import torch

    if not isinstance(grad, torch.Tensor) or grad.numel() == 0:
        return {"status": "unknown", "norm": 0.0, "mean": 0.0, "max": 0.0}
    g = grad.detach().float().abs()
    norm = float(g.norm())
    mean = float(g.mean())
    max_ = float(g.max())
    if norm < 1e-7 or mean < 1e-8:
        status = "vanishing"
    elif norm > 1e3 or max_ > 1e2:
        status = "exploding"
    else:
        status = "healthy"
    return {"status": status, "norm": norm, "mean": mean, "max": max_}


# ── Retain-grad walker ──────────────────────────────────────────────


def attach_retain_grad(
    value: Any,
    sink: dict[tuple[str, str], "torch.Tensor"],
    node_id: str,
    port: str,
) -> None:
    """Walk a result value (tensor / dict / list / tuple) and:
       * call ``retain_grad()`` on each non-leaf floating tensor with grad,
       * record (node_id, port) → tensor in ``sink`` so we can read ``.grad``
         after backward.

    Sub-keys in dicts/lists get derived port names (``port[0]``, ``port.x``).
    """
    import torch

    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            try:
                if value.requires_grad and not value.is_leaf:
                    value.retain_grad()
                # Even leaf tensors are worth tracking — their .grad will
                # still be populated if they participate in the graph.
                if value.requires_grad:
                    sink[(node_id, port)] = value
            except RuntimeError:
                # Some tensors (already detached, view of int, …) reject
                # retain_grad — silently skip rather than crash the run.
                pass
        return

    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(k, str) and k.startswith("__"):
                continue
            attach_retain_grad(v, sink, node_id, f"{port}.{k}" if port else str(k))
        return

    if isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            attach_retain_grad(v, sink, node_id, f"{port}[{i}]")
        return


# ── Backward-target selection ───────────────────────────────────────


def select_backward_target(
    nodes: list[dict],
    outputs: dict[str, dict[str, Any]],
    *,
    auto_backward: bool,
) -> "tuple[torch.Tensor, str] | None":
    """Pick the tensor to call ``.backward()`` on.

    Order of preference:
      1. Output of any ``BackwardOnce`` node — the user explicitly marked it.
      2. If a ``TrainingLoop`` node was executed, return ``None`` so we
         don't double-backward over its already-completed pass.
      3. If ``auto_backward`` is True, pick the highest-rank floating tensor
         from a leaf node and return ``loss = tensor.sum()`` (synthetic loss).
      4. Otherwise ``None`` — caller should skip backward.
    """
    import torch

    # 1. Explicit BackwardOnce node.
    for n in nodes:
        if n.get("type") == "BackwardOnce" and n["id"] in outputs:
            t = outputs[n["id"]].get("tensor")
            if isinstance(t, torch.Tensor) and t.requires_grad and t.is_floating_point():
                return (t.sum(), f"BackwardOnce({n['id']}).sum()")
            return None

    # 2. TrainingLoop owns its own backward — defer.
    if any(n.get("type") == "TrainingLoop" for n in nodes):
        return None

    # 3. Auto-synthetic loss on a leaf.
    if auto_backward:
        # A "leaf" node has no outgoing data edges. Find candidate tensors.
        leaf_ids: set[str] = set(outputs.keys())
        leaf_node_ids = list(leaf_ids)
        # Pick tensors with the largest numel as the "interesting" output.
        best: tuple[float, "torch.Tensor", str] | None = None
        for nid in leaf_node_ids:
            for port, val in outputs[nid].items():
                if port.startswith("__"):
                    continue
                if isinstance(val, torch.Tensor) and val.is_floating_point() and val.requires_grad:
                    score = float(val.numel())
                    if best is None or score > best[0]:
                        best = (score, val, f"{nid}.{port}")
        if best is not None:
            return (best[1].sum(), f"auto({best[2]}).sum()")

    return None


# ── Backward + capture ──────────────────────────────────────────────


def run_backward(loss: "torch.Tensor") -> None:
    if loss.requires_grad:
        loss.backward()


async def capture_grads(
    grad_targets: dict[tuple[str, str], "torch.Tensor"],
    node_state_store: "NodeStateStore | None",
    graph_id: str,
    run_id: str,
    output_store: "RunOutputStore",
) -> None:
    """Write captured gradients to the run output store.

    Per-port: ``{port}__grad`` + ``{port}__grad__meta``.
    Per-module weight: ``__weight_grad__{param_name}`` + ``__meta``.
    """
    import torch

    # Per-port gradients.
    for (node_id, port), tensor in grad_targets.items():
        grad = getattr(tensor, "grad", None)
        if grad is None:
            continue
        await output_store.put(run_id, node_id, f"{port}__grad", grad.detach())
        await output_store.put(
            run_id, node_id, f"{port}__grad__meta", grad_health(grad),
        )

    # Per-module weight gradients (only when A2 NodeStateStore is active).
    if node_state_store is None or not graph_id:
        return
    for (gid, node_id, _h), module in node_state_store.iter_for_graph(graph_id):
        if not isinstance(module, torch.nn.Module):
            continue
        for name, param in module.named_parameters():
            if param.grad is None:
                continue
            await output_store.put(
                run_id, node_id, f"__weight_grad__{name}", param.grad.detach()
            )
            await output_store.put(
                run_id, node_id, f"__weight_grad__{name}__meta", grad_health(param.grad)
            )


def zero_module_grads(node_state_store: "NodeStateStore | None", graph_id: str) -> None:
    """Zero accumulated ``.grad`` on every persisted module for this graph.

    Called BEFORE the forward pass when ``backward_mode`` is on so
    repeated runs don't accumulate gradients across runs.
    """
    if node_state_store is None or not graph_id:
        return
    for _key, module in node_state_store.iter_for_graph(graph_id):
        try:
            for p in module.parameters():
                if p.grad is not None:
                    p.grad = None
        except Exception:
            pass


def iter_input_leaves(inputs: Any) -> Iterable["torch.Tensor"]:
    """Yield floating-point tensors from inputs that should require_grad
    (so the backward pass can flow back to them)."""
    import torch

    stack: list[Any] = [inputs]
    while stack:
        v = stack.pop()
        if isinstance(v, torch.Tensor):
            if v.is_floating_point():
                yield v
        elif isinstance(v, dict):
            stack.extend(v.values())
        elif isinstance(v, (list, tuple)):
            stack.extend(v)
