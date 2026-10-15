"""The event sequence a default submission produces, pinned (#765).

Epic #764 splits CodefyUI into a control plane and workers. Its first
invariant is that a server with no worker enrolled, given a submission with
no ``target`` option, behaves exactly as 3.0.2 did: same queue, same
in-process executor, same events. This module is what makes "exactly" a
check rather than a promise.

Three graphs go through ``POST /api/runs`` with the real lifespan around
the app (instance lock, node discovery, the RunService it builds) and the
default settings, and their events are read back over
``GET /api/runs/{id}/events`` until the run is terminal:

* ``tensor``: a pure tensor chain;
* ``training``: a tiny regression trained for one epoch;
* ``preset``: a preset card (``Simple CNN Classifier``), whose inner nodes
  report under their expanded ids.

Each graph carries ``settings.seed`` (no submit options at all). A seeded
run is a serial run, so the order nodes in one level complete in is the
engine's level order and can be pinned; an unseeded run lets independent
nodes (Start and whatever it triggers, at the least) finish in thread
order, which no snapshot can hold.

Every event is reduced to ``(type, node_id, status, payload keys)`` and the
run row to its terminal status, and the result is compared with
``tests/snapshots/single_node_parity.json``. Nothing in it depends on the
machine: no timestamps, durations, values, paths, run ids or cursors. Two
things are timed by the wall clock and are normalised for that: the training
loop's per-batch progress frames (a run of identical consecutive rows is
collapsed to one) and metric batches (left out of the sequence; the snapshot
records only that one arrived).

A changed snapshot is a reviewed, deliberate commit. To regenerate it after
an intended change, run this module with ``UPDATE_SINGLE_NODE_PARITY_SNAPSHOT=1``
and commit the diff together with the change that caused it.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

import app.main as main_mod
from app.config import settings
from app.core.auth import TOKEN_HEADER, init_allowed_hosts, session_token

SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "single_node_parity.json"
UPDATE_ENV = "UPDATE_SINGLE_NODE_PARITY_SNAPSHOT"

#: A cold CI runner imports torch and trains one epoch inside this.
RUN_TIMEOUT_S = 120.0

TERMINAL_STATUSES = {"succeeded", "failed", "cancelled", "interrupted"}

#: Flushed on a size-or-time cadence; see `_submit_and_follow`.
METRIC_EVENT = "metric"


# ── The three graphs ────────────────────────────────────────────────────


def _node(node_id: str, node_type: str, **params: Any) -> dict[str, Any]:
    return {"id": node_id, "type": node_type, "data": {"params": params}}


def _trigger(source: str, target: str) -> dict[str, Any]:
    return {"id": f"t-{source}-{target}", "source": source, "target": target,
            "sourceHandle": "trigger", "type": "trigger"}


def _wire(source: str, source_handle: str, target: str,
          target_handle: str) -> dict[str, Any]:
    return {"id": f"e-{source}-{source_handle}-{target}-{target_handle}",
            "source": source, "sourceHandle": source_handle,
            "target": target, "targetHandle": target_handle}


def _tensor_graph() -> dict[str, Any]:
    """Start -> TensorCreate -> ScalarMultiply -> Activation -> Print."""
    return {
        "nodes": [
            _node("start", "Start"),
            _node("x", "TensorCreate", shape="2,3", fill="ones", value=0,
                  requires_grad=False),
            _node("scale", "ScalarMultiply", scalar=2.5),
            _node("act", "Activation", function="relu"),
            _node("show", "Print", label="tensor"),
        ],
        "edges": [
            _trigger("start", "x"),
            _wire("x", "tensor", "scale", "tensor"),
            _wire("scale", "tensor", "act", "tensor"),
            _wire("act", "tensor", "show", "value"),
        ],
        "settings": {"seed": 0},
    }


_REGRESSION_DATA = (
    "import torch\n"
    "\n"
    "def run(inputs, params):\n"
    "    g = torch.Generator().manual_seed(0)\n"
    "    x = torch.randn(32, 2, generator=g)\n"
    "    y = x @ torch.tensor([1.5, -2.0]) + 0.5\n"
    "    return {'out1': x, 'out2': y}\n"
)

_LINEAR_MODEL = json.dumps({
    "version": 2,
    "nodes": [
        {"id": "in", "type": "Input", "ports": [{"id": "p_x", "name": "x"}]},
        {"id": "l0", "type": "Linear",
         "params": {"in_features": 2, "out_features": 1}},
        {"id": "out", "type": "Output", "ports": [{"id": "p_y", "name": "y"}]},
    ],
    "edges": [
        {"id": "e0", "source": "in", "target": "l0", "sourceHandle": "p_x"},
        {"id": "eout", "source": "l0", "target": "out", "targetHandle": "p_y"},
    ],
}, separators=(",", ":"))


def _training_graph() -> dict[str, Any]:
    """One epoch of linear regression on 32 generated rows, seeded."""
    return {
        "nodes": [
            _node("start", "Start"),
            _node("gen", "PythonScript", code=_REGRESSION_DATA, input_ports=1,
                  output_ports=2, input_types="TENSOR",
                  output_types="TENSOR,TENSOR"),
            _node("model", "SequentialModel", layers=_LINEAR_MODEL),
            _node("loss", "Loss", type="MSELoss"),
            _node("ds", "TensorDataset", target_type="number"),
            _node("dl", "DataLoader", batch_size=8, shuffle=False),
            _node("opt", "Optimizer", type="SGD", lr=0.05),
            _node("train", "TrainingLoop", epochs=1, device="cpu"),
            _node("show", "Print", label="trained"),
        ],
        "edges": [
            _trigger("start", "gen"),
            _trigger("start", "model"),
            _trigger("start", "loss"),
            _wire("gen", "out1", "ds", "features"),
            _wire("gen", "out2", "ds", "targets"),
            _wire("ds", "dataset", "dl", "dataset"),
            _wire("model", "model", "opt", "model"),
            _wire("model", "model", "train", "model"),
            _wire("dl", "dataloader", "train", "dataloader"),
            _wire("opt", "optimizer", "train", "optimizer"),
            _wire("loss", "loss_fn", "train", "loss_fn"),
            _wire("train", "model", "show", "value"),
        ],
        "settings": {"seed": 0},
    }


def _preset_graph() -> dict[str, Any]:
    """A shipped preset card between a tensor and a Print."""
    return {
        "nodes": [
            _node("start", "Start"),
            _node("img", "TensorCreate", shape="1,1,8,8", fill="ones", value=0,
                  requires_grad=False),
            {"id": "cnn", "type": "preset:Simple CNN Classifier",
             "data": {"params": {}}},
            _node("show", "Print", label="features"),
        ],
        "edges": [
            _trigger("start", "img"),
            _wire("img", "tensor", "cnn", "tensor"),
            _wire("cnn", "tensor", "show", "value"),
        ],
        "settings": {"seed": 0},
    }


GRAPHS = {
    "tensor": _tensor_graph,
    "training": _training_graph,
    "preset": _preset_graph,
}


# ── Reducing a run to what is pinned ────────────────────────────────────


def _event_row(event: dict[str, Any]) -> list[Any]:
    payload = event.get("payload") or {}
    if not isinstance(payload, dict):
        payload = {}
    return [
        event["type"],
        payload.get("node_id"),
        payload.get("status"),
        sorted(payload),
    ]


def _collapse_repeats(rows: list[list[Any]]) -> list[list[Any]]:
    """Drop a row identical to the one before it: the training loop's
    per-batch progress frames are throttled by wall clock, so how many of
    them a run produces depends on the machine, not on the code."""
    collapsed: list[list[Any]] = []
    for row in rows:
        if not collapsed or collapsed[-1] != row:
            collapsed.append(row)
    return collapsed


# ── The real lifespan ───────────────────────────────────────────────────


@pytest.fixture
async def served_app(tmp_path, monkeypatch):
    """The app with its real lifespan running and default settings.

    The database moves to *tmp_path* so the run rows are this test's own;
    the user data dir is already the session's temporary one (conftest).
    ``setup_logging`` is a no-op because the real one clears the root
    logger's handlers, which would detach pytest's capture mid-test.
    """
    monkeypatch.setattr(settings, "DB_PATH", tmp_path / "db" / "parity.db")
    monkeypatch.setattr(main_mod, "setup_logging", lambda **kwargs: None)
    async with main_mod.lifespan(main_mod.app):
        async with AsyncClient(
            transport=ASGITransport(app=main_mod.app),
            base_url=f"http://127.0.0.1:{settings.PORT}",
            headers={TOKEN_HEADER: session_token()},
        ) as client:
            yield client
    # The lifespan re-ran init_allowed_hosts with CORS origins added; hand
    # back the exact whitelist conftest seeded.
    init_allowed_hosts(settings.HOST, settings.PORT)


async def _submit_and_follow(client: AsyncClient,
                             graph: dict[str, Any]) -> dict[str, Any]:
    response = await client.post("/api/runs", json={"graph": graph})
    assert response.status_code == 200, response.text
    run_id = response.json()["run_id"]

    events: list[dict[str, Any]] = []
    cursor = 0
    deadline = time.monotonic() + RUN_TIMEOUT_S
    while True:
        page = await client.get(f"/api/runs/{run_id}/events",
                                params={"cursor": cursor, "wait": 1.0})
        assert page.status_code == 200, page.text
        body = page.json()
        events.extend(body["events"])
        cursor = body["cursor"]
        if body["status"] in TERMINAL_STATUSES and not body["active"]:
            # One more read: the last events may land after the status flips.
            tail = (await client.get(f"/api/runs/{run_id}/events",
                                     params={"cursor": cursor})).json()
            events.extend(tail["events"])
            break
        assert time.monotonic() < deadline, (
            f"run {run_id} still {body['status']} after {RUN_TIMEOUT_S}s")
        await asyncio.sleep(0)

    record = (await client.get(f"/api/runs/{run_id}")).json()
    # A metric batch is flushed by size or by wall clock, so where in the
    # sequence it lands depends on the machine; that one exists does not.
    metric_events = [e for e in events if e["type"] == METRIC_EVENT]
    rows = _collapse_repeats([
        _event_row(event) for event in events if event["type"] != METRIC_EVENT
    ])
    return {
        "submit_status": response.json()["status"],
        "queue_key": record["queue_key"],
        "status": record["status"],
        "events": rows,
        "metric_events": bool(metric_events),
    }


async def test_default_submissions_match_the_snapshot(served_app):
    actual = {
        name: await _submit_and_follow(served_app, build())
        for name, build in GRAPHS.items()
    }

    if os.environ.get(UPDATE_ENV) == "1":
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_PATH.write_text(
            json.dumps(actual, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        pytest.skip(f"{UPDATE_ENV}=1: wrote {SNAPSHOT_PATH.name}")

    expected = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    assert set(actual) == set(expected), "the snapshot names other graphs"
    for name in GRAPHS:
        assert actual[name]["status"] == "succeeded", (
            f"{name}: run ended {actual[name]['status']}")
        assert actual[name] == expected[name], (
            f"{name}: the default submit path changed. If that is intended, "
            f"rerun with {UPDATE_ENV}=1 and commit the snapshot diff.\n"
            f"actual:\n{json.dumps(actual[name], indent=1)}")
