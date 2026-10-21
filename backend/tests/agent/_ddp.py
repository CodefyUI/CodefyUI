"""Graphs and entry points for the ``strategy: ddp`` tests (#792).

Module level imports nothing heavy: a *spawn* child imports this module to
find :func:`cpu_ddp_child`, and the rank stand-ins below must start fast.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from app.agent import child, ddp

from ._helpers import node, trigger, wire

#: A plan for two gloo ranks on the CPU: what CI can run.
CPU_PLAN = {"rank_devices": ["cpu", "cpu"], "backend": "gloo"}

#: ``x`` (n x 3) and ``y = x @ w + noise``, the same on every call.
GEN_CODE = (
    "import torch\n"
    "\n"
    "def run(inputs, params):\n"
    "    g = torch.Generator().manual_seed(0)\n"
    "    x = torch.randn(params['n'], 3, generator=g)\n"
    "    w = torch.tensor([[1.0], [-2.0], [0.5]])\n"
    "    y = x @ w + 0.1 * torch.randn(params['n'], 1, generator=g)\n"
    "    return {'out1': x, 'out2': y}\n"
)

#: Class labels 0/1 except at index ``bad``, which is 7: out of range for
#: a two-class model. With two ranks and no shuffle, rank ``bad % 2`` gets
#: it in its first batch.
BAD_LABEL_CODE = (
    "import torch\n"
    "\n"
    "def run(inputs, params):\n"
    "    g = torch.Generator().manual_seed(0)\n"
    "    x = torch.randn(params['n'], 3, generator=g)\n"
    "    y = (x[:, 0] > 0).long()\n"
    "    if params.get('bad', -1) >= 0:\n"
    "        y[params['bad']] = 7\n"
    "    return {'out1': x, 'out2': y}\n"
)


def linear_layers(in_features: int = 3, out_features: int = 1) -> str:
    return json.dumps({
        "version": 2,
        "nodes": [
            {"id": "in", "type": "Input", "ports": [{"id": "p_x", "name": "x"}]},
            {"id": "l0", "type": "Linear",
             "params": {"in_features": in_features,
                        "out_features": out_features}},
            {"id": "out", "type": "Output",
             "ports": [{"id": "p_y", "name": "y"}]},
        ],
        "edges": [
            {"id": "e0", "source": "in", "target": "l0", "sourceHandle": "p_x"},
            {"id": "eout", "source": "l0", "target": "out",
             "targetHandle": "p_y"},
        ],
    }, separators=(",", ":"))


def training_graph(*, batch_size: int, epochs: int = 3,
                   strategy: str | None = "ddp", n: int = 32,
                   train_params: dict[str, Any] | None = None,
                   save_as: str | None = "trained.pt", classify: bool = False,
                   bad: int | None = None, shuffle: bool = False,
                   lr: float = 0.1) -> dict[str, Any]:
    """Start -> data -> TensorDataset -> DataLoader, SequentialModel ->
    Optimizer, Loss -> TrainingLoop -> ModelSaver."""
    gen_params: dict[str, Any] = {
        "code": BAD_LABEL_CODE if classify else GEN_CODE,
        "input_ports": 1, "output_ports": 2,
        "output_types": "TENSOR,TENSOR", "n": n}
    if bad is not None:
        gen_params["bad"] = bad
    params = {"epochs": epochs, "device": "auto", **(train_params or {})}
    if strategy is not None:
        params["strategy"] = strategy
    nodes = [
        node("start", "Start"),
        node("gen", "PythonScript", **gen_params),
        node("ds", "TensorDataset",
             target_type="class" if classify else "number"),
        node("dl", "DataLoader", batch_size=batch_size, shuffle=shuffle,
             num_workers=0),
        node("model", "SequentialModel",
             layers=linear_layers(3, 2 if classify else 1)),
        node("opt", "Optimizer", type="SGD", lr=lr, weight_decay=0.0),
        node("loss", "Loss",
             type="CrossEntropyLoss" if classify else "MSELoss"),
        node("train", "TrainingLoop", **params),
    ]
    edges = [
        trigger("start", "gen"),
        trigger("start", "model"),
        trigger("start", "loss"),
        wire("gen", "out1", "ds", "features"),
        wire("gen", "out2", "ds", "targets"),
        wire("ds", "dataset", "dl", "dataset"),
        wire("model", "model", "opt", "model"),
        wire("model", "model", "train", "model"),
        wire("dl", "dataloader", "train", "dataloader"),
        wire("opt", "optimizer", "train", "optimizer"),
        wire("loss", "loss_fn", "train", "loss_fn"),
    ]
    if save_as is not None:
        nodes.append(node("save", "ModelSaver", path=save_as))
        edges.append(wire("train", "model", "save", "model"))
    return {"nodes": nodes, "edges": edges}


def with_validation(graph: dict[str, Any], *, batch_size: int) -> dict[str, Any]:
    """*graph* with a second DataLoader over the same dataset wired to the
    loop's ``val_dataloader``."""
    graph["nodes"].append(node("vdl", "DataLoader", batch_size=batch_size,
                               shuffle=False, num_workers=0))
    graph["edges"] += [wire("ds", "dataset", "vdl", "dataset"),
                       wire("vdl", "dataloader", "train", "val_dataloader")]
    return graph


def with_step_scheduler(graph: dict[str, Any], *, step_size: int
                        ) -> dict[str, Any]:
    """*graph* with a StepLR scheduler on the loop. A ``step_size`` past
    the loop's epochs makes the loop warn (``lr_schedule_mismatch``)."""
    graph["nodes"].append(node("sched", "LRScheduler", type="StepLR",
                               step_size=step_size, gamma=0.5))
    graph["edges"] += [wire("opt", "optimizer", "sched", "optimizer"),
                       wire("sched", "scheduler", "train", "lr_scheduler")]
    return graph


def cpu_ddp_child(spec: dict[str, Any], events_out: Any,
                  control_in: Any) -> None:
    """``child.main`` for a ``ddp`` run on two gloo CPU ranks: what the
    agent runs for a two-card set, on the CPU."""
    spec = dict(spec)
    spec["ddp"] = dict(CPU_PLAN)
    child.main(spec, events_out, control_in)


# ── rank stand-ins for the launcher tests (no torch) ─────────────────────


def _behaviour(spec: dict[str, Any], rank: int) -> dict[str, Any]:
    return dict((spec.get("fake_ranks") or {}).get(str(rank)) or {})


def fake_rank(spec: dict[str, Any], plan: dict[str, Any], rank: int,
              rendezvous: str, events_out: Any, control_in: Any) -> None:
    """A rank that follows ``spec["fake_ranks"][str(rank)]``:

    - ``limit``: a ``memory_limit`` payload to send first;
    - ``messages``: messages to send (lists become tuples);
    - ``ready``: rank 0 sends ``ddp_ready``;
    - ``wait_go``: wait for ``go`` (or a cancel) before going on;
    - ``finish``: the ``finished`` payload, sent after the rest; without
      it the rank waits for a cancel, then reports ``cancelled``
      (unless ``ignore_cancel``, then it sleeps until killed);
    - ``delay_s``: sleep this long before finishing or crashing;
    - ``crash``: exit with this code instead of finishing;
    - ``pid_file``: write the pid there first.
    """
    child.ignore_interrupts()
    do = _behaviour(spec, rank)
    if do.get("pid_file"):
        with open(do["pid_file"], "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
    received: list[Any] = []

    def recv(timeout: float = 30.0) -> Any:
        if control_in.poll(timeout):
            try:
                message = control_in.recv()
            except EOFError:
                return ("eof",)
            received.append(message)
            return message
        return None

    if do.get("limit") is not None:
        events_out.send((child.MSG_MEMORY_LIMIT, do["limit"]))
    if do.get("ready"):
        events_out.send((ddp.MSG_DDP_READY,))
    if do.get("wait_go"):
        while True:
            message = recv()
            if message is None or message[0] in (ddp.MSG_GO, child.MSG_CANCEL,
                                                 "eof"):
                break
        events_out.send((child.MSG_WARNING, {
            "kind": "fake", "detail": f"rank {rank} got {message}",
            "node_id": None}))
    for message in do.get("messages") or []:
        events_out.send(tuple(message))
    if do.get("delay_s"):
        time.sleep(float(do["delay_s"]))
    if do.get("crash") is not None:
        os._exit(int(do["crash"]))
    if do.get("finish") is not None:
        events_out.send((child.MSG_FINISHED, do["finish"]))
        if do.get("linger_s"):
            time.sleep(float(do["linger_s"]))
        return
    while True:
        message = recv(60.0)
        if message is not None and message[0] in (child.MSG_CANCEL, "eof"):
            break
    if do.get("ignore_cancel"):
        time.sleep(120)
    events_out.send((child.MSG_FINISHED, {"status": "cancelled",
                                          "error": None,
                                          "peak_memory": None}))
