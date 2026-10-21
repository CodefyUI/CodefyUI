"""``TrainingLoop`` ``strategy: ddp`` inside the run subprocess (#792).

Three layers:

- the plan and the graphs (pure functions, no processes);
- the launcher's relay with stand-in ranks (``_ddp.fake_rank``: spawned,
  no torch), for what it forwards, merges, drops and stops;
- real runs: ``child.main`` in a spawned process, which starts two
  ``gloo`` ranks on the CPU and trains a small regression. CI has no GPU;
  the NCCL path is checked on CUDA hardware (see the PR).
"""

from __future__ import annotations

import multiprocessing
import threading
import time
from pathlib import Path
from typing import Any

import psutil
import pytest

from app.agent import child, ddp
from app.agent.runner import Runner
from app.core import distributed
from app.core import worker_protocol as wp
from app.core.mounts import data_root_env

from . import _ddp
from . import _helpers as h

SPAWN = multiprocessing.get_context("spawn")


# ── the plan ──────────────────────────────────────────────────────────────


def _spec(graph: dict[str, Any], devices: list[str] | None = None,
          **kw: Any) -> dict[str, Any]:
    return {"run_id": "d1", "graph": graph, "options": {}, "device": "cpu",
            "env": {"devices": devices or []}, **kw}


def test_a_two_card_set_with_a_ddp_loop_gets_one_rank_per_card():
    plan = ddp.plan_for(_spec(_ddp.training_graph(batch_size=4),
                              ["cuda:2", "cuda:5"]))
    assert plan == {"world_size": 2, "rank_devices": ["cuda:0", "cuda:1"],
                    "backend": "nccl", "loop_ids": ["train"]}


@pytest.mark.parametrize("devices", [[], ["cuda:0"], ["cpu"], ["mps"]])
def test_fewer_than_two_cards_runs_in_one_process(devices):
    assert ddp.plan_for(_spec(_ddp.training_graph(batch_size=4),
                              devices)) is None


def test_a_graph_without_a_ddp_loop_runs_in_one_process():
    for strategy in (None, "single"):
        graph = _ddp.training_graph(batch_size=4, strategy=strategy)
        assert ddp.plan_for(_spec(graph, ["cuda:0", "cuda:1"])) is None


def test_an_explicit_plan_sets_the_ranks_and_the_backend():
    spec = _spec(_ddp.training_graph(batch_size=4), ddp=dict(_ddp.CPU_PLAN))
    assert ddp.plan_for(spec) == {
        "world_size": 2, "rank_devices": ["cpu", "cpu"], "backend": "gloo",
        "loop_ids": ["train"]}
    one_card = _spec(_ddp.training_graph(batch_size=4),
                     ddp={"rank_devices": ["cuda:0"]})
    assert ddp.plan_for(one_card)["backend"] == "nccl"
    mixed = _spec(_ddp.training_graph(batch_size=4),
                  ddp={"rank_devices": ["cuda:0", "cpu"]})
    assert ddp.plan_for(mixed)["backend"] == "gloo"
    assert ddp.plan_for(_spec(_ddp.training_graph(batch_size=4),
                              ddp={"rank_devices": []})) is None


def _two_loops(*, chained: bool) -> dict[str, Any]:
    graph = _ddp.training_graph(batch_size=4, save_as=None)
    second = h.node("train2", "TrainingLoop", epochs=1, strategy="ddp")
    graph["nodes"].append(second)
    for port in ("dataloader", "optimizer", "loss_fn"):
        source = {"dataloader": "dl", "optimizer": "opt", "loss_fn": "loss"}
        graph["edges"].append(h.wire(source[port], port, "train2", port))
    graph["edges"].append(h.wire("train" if chained else "model", "model",
                                 "train2", "model"))
    return graph


def test_two_ddp_loops_must_be_in_sequence():
    assert ddp.plan_for(_spec(_two_loops(chained=True),
                              ["cuda:0", "cuda:1"]))["loop_ids"] == [
        "train", "train2"]
    with pytest.raises(distributed.StrategyError, match="neither one feeds"):
        ddp.plan_for(_spec(_two_loops(chained=False), ["cuda:0", "cuda:1"]))


def test_bypassed_and_nested_loops_are_not_ddp_loops():
    graph = _ddp.training_graph(batch_size=4)
    graph["nodes"][-2]["data"]["bypassed"] = True
    assert distributed.ddp_loop_ids(graph) == []
    nested = {"nodes": [], "edges": [], "subgraphs": [
        {"id": "blk", "nodes": [h.node("t", "TrainingLoop", strategy="ddp")]}]}
    assert distributed.ddp_loop_ids(nested) == []
    assert distributed.nested_ddp_loops(nested) == ["blk/t"]


def test_the_other_ranks_run_only_the_loops_and_what_feeds_them():
    graph = _ddp.training_graph(batch_size=4)
    pruned = distributed.rank_graph(graph, ["train"])
    ids = {node["id"] for node in pruned["nodes"]}
    assert ids == {"start", "gen", "ds", "dl", "model", "opt", "loss",
                   "train"}
    assert all(edge["target"] != "save" for edge in pruned["edges"])


def test_each_rank_caps_only_its_own_card():
    spec = _spec(_ddp.training_graph(batch_size=4), ["cuda:0", "cuda:1"],
                 memory_budgets={"cuda:0": 5, "cuda:1": 7})
    plan = ddp.plan_for(spec)
    spec["ddp"] = {"rank_devices": ["cuda:0", "cuda:1"]}
    assert ddp.rank_spec(spec, plan, 1)["memory_budgets"] == {"cuda:1": 7}
    assert ddp.rank_spec(spec, plan, 1)["device"] == "cuda:1"
    assert "ddp" not in ddp.rank_spec(spec, plan, 0)


def test_a_graph_the_ranks_cannot_share_fails_before_anything_starts():
    sent: list[tuple] = []
    child.run_spec(_spec(_two_loops(chained=False), ["cuda:0", "cuda:1"]),
                   sent.append, None)
    assert sent[-1][0] == child.MSG_FINISHED
    assert sent[-1][1]["status"] == "failed"
    assert "neither one feeds" in sent[-1][1]["error"]


# ── the strategy outside a rank ───────────────────────────────────────────


def test_strategy_single_and_a_missing_param_change_nothing():
    for params in ({}, {"strategy": "single"}, {"strategy": " Single "}):
        assert distributed.data_parallel_for(params, None) == (None, None)
    with pytest.raises(ValueError, match="not one of"):
        distributed.data_parallel_for({"strategy": "fsdp"}, None)


def _in_process(graph: dict[str, Any]) -> list[tuple]:
    sent: list[tuple] = []
    thread = threading.Thread(
        target=child.run_assignment,
        args=({"run_id": "one", "graph": graph, "options": {"seed": 0},
               "device": "cpu"}, sent.append),
        kwargs={"discover": False})
    thread.start()
    thread.join(120)
    assert not thread.is_alive()
    return sent


def _train_output(sent: list[tuple]) -> dict[str, Any]:
    for message in sent:
        if message[0] == child.MSG_EVENT \
                and message[2].get("node_id") == "train" \
                and message[2].get("status") == "progress":
            for output in message[2].get("outputs") or []:
                progress = output.get("progress") or {}
                if progress.get("event") == "config":
                    return progress["config"]
    raise AssertionError("no config frame")


def test_ddp_outside_a_worker_run_trains_in_one_process_and_says_so():
    sent = _in_process(_ddp.training_graph(batch_size=8, save_as=None))
    assert sent[-1][1]["status"] == "succeeded"
    warnings = [m[1] for m in sent if m[0] == child.MSG_WARNING]
    assert [w["kind"] for w in warnings] == [
        distributed.FALLBACK_WARNING_KIND]
    assert warnings[0]["node_id"] == "train"
    assert "trains in one process" in warnings[0]["detail"]
    config = _train_output(sent)
    assert config["strategy"] == "single"
    assert config["strategy_requested"] == "ddp"


def test_a_single_loop_reports_no_strategy_at_all():
    sent = _in_process(_ddp.training_graph(batch_size=8, save_as=None,
                                           strategy=None))
    assert sent[-1][1]["status"] == "succeeded"
    assert not [m for m in sent if m[0] == child.MSG_WARNING]
    config = _train_output(sent)
    assert "strategy" not in config and "strategy_requested" not in config


# ── the launcher, with stand-in ranks ─────────────────────────────────────


def _launch(fake_ranks: dict[str, Any], *, world: int = 2,
            agent: list[Any] | None = None, budgets: dict[str, int] | None = None,
            timeout: float = 60.0) -> tuple[list[tuple], float]:
    """Run :func:`ddp.launch` with :func:`_ddp.fake_rank` ranks. *agent*:
    ``(delay_s, message)`` pairs sent down the agent pipe (``"close"``
    closes it). Returns what reached the agent and how long it took."""
    spec = {"run_id": "f1", "graph": {}, "fake_ranks": fake_ranks}
    if budgets:
        spec["memory_budgets"] = budgets
    plan = {"world_size": world,
            "rank_devices": [f"cuda:{i}" for i in range(world)],
            "backend": "gloo", "loop_ids": ["train"]}
    control_in, control_out = SPAWN.Pipe(duplex=False)
    sent: list[tuple] = []
    lock = threading.Lock()

    def send(message: tuple) -> None:
        with lock:
            sent.append(message)

    started = time.monotonic()
    thread = threading.Thread(target=ddp.launch,
                              args=(spec, plan, send, control_in),
                              kwargs={"entry": _ddp.fake_rank, "context": SPAWN})
    thread.start()
    for delay, message in agent or []:
        time.sleep(delay)
        if message == "close":
            control_out.close()
        else:
            control_out.send(message)
    thread.join(timeout)
    assert not thread.is_alive(), f"the launcher did not return: {sent}"
    return sent, time.monotonic() - started


def _finished(sent: list[tuple]) -> dict[str, Any]:
    finished = [m[1] for m in sent if m[0] == child.MSG_FINISHED]
    assert len(finished) == 1, sent
    return finished[0]


def test_rank_zeros_messages_are_the_runs_and_the_others_are_dropped():
    event = ["event", "node_status", {"node_id": "a", "status": "running"}]
    metric = ["metrics", [{"name": "loss", "value": 1.0, "step": 1,
                           "node_id": "train"}]]
    warning = ["warning", {"kind": "k", "detail": "same", "node_id": None}]
    sent, _ = _launch({
        "0": {"messages": [event, metric, warning],
              "finish": {"status": "succeeded", "error": None,
                         "peak_memory": {"cuda:0": {"allocated_bytes": 1}}}},
        "1": {"messages": [event, metric, warning, warning],
              "finish": {"status": "succeeded", "error": None,
                         "peak_memory": {"cuda:1": {"allocated_bytes": 2}}}},
    })
    assert [m for m in sent if m[0] == "event"] == [tuple(event)]
    assert [m for m in sent if m[0] == "metrics"] == [tuple(metric)]
    # Rank 1's copy of a warning rank 0 already gave is dropped too.
    assert len([m for m in sent if m[0] == "warning"]) == 1
    finished = _finished(sent)
    assert finished["status"] == "succeeded"
    assert finished["peak_memory"] == {"cuda:0": {"allocated_bytes": 1},
                                       "cuda:1": {"allocated_bytes": 2}}
    assert sent[-1][0] == child.MSG_FINISHED


def test_rank_zero_entering_the_loop_starts_the_others():
    sent, _ = _launch({
        "0": {"ready": True, "wait_go": False,
              "finish": {"status": "succeeded", "error": None,
                         "peak_memory": None}, "linger_s": 1.0},
        "1": {"wait_go": True, "finish": {"status": "succeeded",
                                          "error": None, "peak_memory": None}},
    })
    details = [m[1]["detail"] for m in sent if m[0] == "warning"]
    assert details == ["rank 1 got ('go',)"]


def test_a_failing_rank_ends_the_run_and_stops_the_others():
    sent, took = _launch({
        "0": {},  # trains until it is cancelled
        "1": {"finish": {"status": "failed", "error": "ValueError: boom",
                         "peak_memory": None}},
    })
    finished = _finished(sent)
    assert finished["status"] == "failed"
    assert finished["error"] == "DDP rank 1 of 2 failed: ValueError: boom"
    assert took < 30


def test_a_rank_that_dies_without_reporting_fails_the_run(tmp_path):
    sent, _ = _launch({"0": {}, "1": {"crash": 3}})
    finished = _finished(sent)
    assert finished["status"] == "failed"
    assert "DDP rank 1 of 2 exited (exit code 3) without reporting" \
        in finished["error"]


def test_a_peer_that_died_first_is_named_over_rank_zeros_error():
    """A rank killed mid-collective breaks rank 0's collective, and rank 0
    may report that before the launcher sees the rank exit. The run names
    the rank that died."""
    sent, _ = _launch({
        "0": {"finish": {"status": "failed", "peak_memory": None,
                         "error": "gloo: Connection reset by peer"}},
        "1": {"delay_s": 0.3, "crash": 9},
    })
    finished = _finished(sent)
    assert finished["status"] == "failed"
    assert finished["error"].startswith(
        "DDP rank 1 of 2 exited (exit code 9) without reporting")


def test_a_rank_that_dies_after_rank_zero_succeeded_changes_nothing():
    sent, _ = _launch({
        "0": {"finish": {"status": "succeeded", "error": None,
                         "peak_memory": None}},
        "1": {"delay_s": 0.3, "crash": 9},
    })
    assert _finished(sent)["status"] == "succeeded"


def test_a_cancel_reaches_every_rank():
    sent, _ = _launch({"0": {}, "1": {}},
                      agent=[(1.0, (child.MSG_CANCEL,))])
    assert _finished(sent)["status"] == "cancelled"


def test_a_failure_after_a_cancel_is_a_cancel():
    sent, _ = _launch({
        "0": {}, "1": {"wait_go": True,
                       "finish": {"status": "failed", "error": "broken pipe",
                                  "peak_memory": None}}},
        agent=[(1.0, (child.MSG_CANCEL,))])
    assert _finished(sent) == {"status": "cancelled", "error": None,
                               "peak_memory": None}


def test_a_rank_that_ignores_its_cancel_is_terminated(tmp_path):
    pid_file = tmp_path / "rank1.pid"
    sent, took = _launch({
        "0": {"finish": {"status": "succeeded", "error": None,
                         "peak_memory": None}},
        "1": {"ignore_cancel": True, "pid_file": str(pid_file)}})
    assert _finished(sent)["status"] == "succeeded"
    assert took < ddp.MERGE_WAIT_S + ddp.STOP_GRACE_S + ddp.KILL_AFTER_S + 20
    assert not psutil.pid_exists(int(pid_file.read_text())) or \
        psutil.Process(int(pid_file.read_text())).status() == \
        psutil.STATUS_ZOMBIE


def test_the_caps_go_first_as_one_message():
    event = ["event", "node_status", {"node_id": "a", "status": "running"}]
    sent, _ = _launch({
        "0": {"limit": {"cuda:0": {"budget_bytes": 5}}, "messages": [event],
              "finish": {"status": "succeeded", "error": None,
                         "peak_memory": None}, "linger_s": 1.0},
        "1": {"wait_go": False, "messages": [["warning", {
            "kind": "slow", "detail": "x", "node_id": None}]],
              "limit": {"cuda:1": {"budget_bytes": 7}},
              "finish": {"status": "succeeded", "error": None,
                         "peak_memory": None}},
    }, budgets={"cuda:0": 5, "cuda:1": 7})
    assert sent[0] == (child.MSG_MEMORY_LIMIT,
                       {"cuda:0": {"budget_bytes": 5},
                        "cuda:1": {"budget_bytes": 7}})
    assert tuple(event) in sent
    assert sent[-1][0] == child.MSG_FINISHED


def test_a_cap_that_fails_on_one_rank_fails_the_run_before_it_starts():
    sent, _ = _launch({
        "0": {"limit": {"cuda:0": {"budget_bytes": 5}},
              "messages": [["event", "node_status", {"node_id": "a"}]]},
        "1": {"finish": {"status": "failed", "error": "RuntimeError: cap",
                         "peak_memory": None}},
    }, budgets={"cuda:0": 5, "cuda:1": 7})
    assert [m[0] for m in sent] == [child.MSG_FINISHED]
    assert "cap" in sent[0][1]["error"]


def test_a_closed_agent_pipe_cancels_every_rank():
    sent, _ = _launch({"0": {}, "1": {}}, agent=[(1.0, "close")])
    assert _finished(sent)["status"] == "cancelled"


def test_a_rank_whose_launcher_is_gone_exits(monkeypatch):
    exits: list[int] = []
    monkeypatch.setattr(ddp, "ORPHAN_EXIT_S", 0.05)
    monkeypatch.setattr(ddp.os, "_exit", exits.append)
    reader, writer = SPAWN.Pipe(duplex=False)

    class Context:
        cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

    context, go = Context(), threading.Event()
    thread = threading.Thread(target=ddp.rank_listener,
                              args=(reader, context, None, go))
    thread.start()
    writer.send(("unknown",))
    writer.close()
    thread.join(5)
    deadline = time.monotonic() + 5
    while not exits and time.monotonic() < deadline:
        time.sleep(0.02)
    assert context.cancelled and go.is_set() and exits == [1]


def test_the_listener_hands_reads_to_rank_zeros_outputs(monkeypatch):
    # The pipe closing at the end must not exit this process.
    monkeypatch.setattr(ddp, "_orphaned", lambda: None)
    reader, writer = SPAWN.Pipe(duplex=False)
    calls: list[tuple] = []

    class Outputs:
        def answer(self, *args: Any) -> None:
            calls.append(("answer", *args))

        def drop(self, run_id: str) -> None:
            calls.append(("drop", run_id))

    class Context:
        def cancel(self) -> None:
            calls.append(("cancel",))

    thread = threading.Thread(target=ddp.rank_listener,
                              args=(reader, Context(), Outputs()))
    thread.start()
    writer.send((child.MSG_FETCH, "c1", "ports", "r", {}))
    writer.send((child.MSG_DROP_OUTPUTS, "r"))
    writer.send((child.MSG_CANCEL,))
    writer.send(())
    deadline = time.monotonic() + 5
    while len(calls) < 3 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert calls == [("answer", "c1", "ports", "r", {}), ("drop", "r"),
                     ("cancel",)]
    writer.close()
    thread.join(10)


# ── real runs: two gloo ranks on the CPU ──────────────────────────────────


class ChildRun:
    """``child.main`` (via *entry*) in a spawned process, as the agent
    starts it, with its messages collected on a thread."""

    def __init__(self, tmp_path: Path, graph: dict[str, Any], *,
                 entry: Any = _ddp.cpu_ddp_child,
                 options: dict[str, Any] | None = None) -> None:
        data = tmp_path / "data"
        data.mkdir(parents=True, exist_ok=True)
        env_set, unset = data_root_env(data)
        self.data = data
        self.log = tmp_path / "output.log"
        spec = {"run_id": "g1", "graph": graph,
                "options": options or {"seed": 0}, "device": "cpu",
                "env": {"set": env_set, "unset": unset},
                "log_path": str(self.log)}
        events_in, events_out = SPAWN.Pipe(duplex=False)
        control_in, self.control = SPAWN.Pipe(duplex=False)
        self.process = SPAWN.Process(target=entry,
                                     args=(spec, events_out, control_in))
        self.process.start()
        events_out.close()
        control_in.close()
        self.messages: list[tuple] = []
        self.changed = threading.Event()
        self.descendants: set[int] = set()
        self.reader = threading.Thread(target=self._read, args=(events_in,),
                                       daemon=True)
        self.reader.start()

    def _read(self, events_in: Any) -> None:
        while True:
            try:
                message = events_in.recv()
            except (EOFError, OSError):
                break
            self.messages.append(message)
            self.changed.set()
        self.changed.set()

    def wait_for(self, predicate: Any, timeout: float = 120.0) -> Any:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for message in list(self.messages):
                if predicate(message):
                    return message
            self.track()
            self.changed.clear()
            self.changed.wait(0.2)
        raise AssertionError(f"no such message in {timeout}s: "
                             f"{self.messages[-5:]}\n{self.tail()}")

    def track(self) -> None:
        """Remember every process under the run subprocess."""
        try:
            for proc in psutil.Process(self.process.pid).children(
                    recursive=True):
                self.descendants.add(proc.pid)
        except psutil.Error:
            pass

    def finish(self, timeout: float = 180.0) -> dict[str, Any]:
        message = self.wait_for(lambda m: m[0] == child.MSG_FINISHED, timeout)
        self.process.join(60)
        assert self.process.exitcode == 0, self.tail()
        self.reader.join(10)
        return message[1]

    def tail(self) -> str:
        try:
            return self.log.read_text(encoding="utf-8", errors="replace")[-4000:]
        except OSError:
            return ""

    def assert_no_process_left(self) -> None:
        assert self.descendants, "the ranks were never seen"
        deadline = time.monotonic() + 15
        alive = self.descendants
        while time.monotonic() < deadline:
            alive = set()
            for pid in self.descendants:
                try:
                    if psutil.Process(pid).status() != psutil.STATUS_ZOMBIE:
                        alive.add(pid)
                except psutil.Error:
                    pass
            if not alive:
                return
            time.sleep(0.2)
        raise AssertionError(f"processes left behind: {alive}")

    def of(self, kind: str) -> list[Any]:
        return [m for m in self.messages if m[0] == kind]

    def metric(self, name: str) -> list[float]:
        return [point["value"] for message in self.of(child.MSG_METRICS)
                for point in message[1] if point["name"] == name]


def _reference(tmp_path: Path, graph: dict[str, Any], monkeypatch: Any
               ) -> tuple[list[tuple], Path]:
    """*graph* trained the ordinary way, in this process."""
    from app.config import settings

    models = tmp_path / "reference" / "models"
    models.mkdir(parents=True)
    monkeypatch.setattr(settings, "MODELS_DIR", models)
    sent = _in_process(graph)
    assert sent[-1][1]["status"] == "succeeded", sent[-1]
    return sent, models


def test_two_ranks_train_like_one_process_on_the_doubled_batch(
        tmp_path, monkeypatch):
    """Gradients are synchronised: two ranks at batch 4, each on its half
    of every batch of 8, end on the weights one process reaches at batch
    8, and report the same losses. A rank that trained on its half alone
    would not."""
    import torch

    epochs = 3
    run = ChildRun(tmp_path, _ddp.training_graph(
        batch_size=4, epochs=epochs, train_params={"checkpoint_every": 1}))
    finished = run.finish()
    assert finished["status"] == "succeeded", run.tail()
    sent, reference_models = _reference(
        tmp_path, _ddp.training_graph(batch_size=8, epochs=epochs,
                                      strategy="single"), monkeypatch)

    trained = torch.load(run.data / "models" / "trained.pt")
    expected = torch.load(reference_models / "trained.pt")
    assert trained.keys() == expected.keys()
    for key in expected:
        assert torch.allclose(trained[key], expected[key], atol=1e-5), key

    reference_losses = [point["value"] for message in sent
                        if message[0] == child.MSG_METRICS
                        for point in message[1]
                        if point["name"] == "train_loss"]
    assert run.metric("train_loss") == pytest.approx(reference_losses,
                                                     rel=1e-5)
    # One copy of everything: rank 0's.
    assert len(run.metric("train_loss")) == epochs
    checkpoints = [m[1] for m in run.of(child.MSG_ARTIFACT)
                   if m[1]["kind"] == "checkpoint"]
    assert len(checkpoints) == epochs
    periodic = sorted(p.name for p in
                      (run.data / "models" / "periodic").iterdir())
    assert periodic == [f"g1-train-e{e}.pt" for e in range(1, epochs + 1)]
    finished_nodes = [m[2]["node_id"] for m in run.of(child.MSG_EVENT)
                      if m[2].get("status") == "completed"]
    assert finished_nodes.count("save") == 1
    config = next(o["progress"]["config"] for m in run.of(child.MSG_EVENT)
                  for o in m[2].get("outputs") or []
                  if (o.get("progress") or {}).get("event") == "config")
    assert config["strategy"] == "ddp" and config["world_size"] == 2
    assert config["backend"] == "gloo"
    assert config["batch_size"] == 4 and config["effective_batch_size"] == 8
    # Each rank ran the loop; only rank 0 ran what follows it.
    log = run.tail()
    assert "rank 0" in log and "rank 1" in log
    assert not [m for m in run.of(child.MSG_WARNING)
                if m[1]["kind"] == distributed.FALLBACK_WARNING_KIND]


def test_validation_and_accuracy_are_summed_over_the_ranks(
        tmp_path, monkeypatch):
    graph = _ddp.training_graph(batch_size=4, epochs=2, classify=True,
                                bad=None, save_as=None)
    graph = _with_validation(graph)
    run = ChildRun(tmp_path, graph)
    assert run.finish()["status"] == "succeeded", run.tail()
    reference = _ddp.training_graph(batch_size=8, epochs=2, classify=True,
                                    save_as=None, strategy="single")
    sent, _ = _reference(tmp_path, _with_validation(reference), monkeypatch)

    def series(messages: list[tuple], name: str) -> list[float]:
        return [p["value"] for m in messages if m[0] == child.MSG_METRICS
                for p in m[1] if p["name"] == name]

    assert run.metric("val_loss") == pytest.approx(series(sent, "val_loss"),
                                                   rel=1e-5)
    assert run.metric("val_accuracy") == pytest.approx(
        series(sent, "val_accuracy"), rel=1e-9)


def _with_validation(graph: dict[str, Any]) -> dict[str, Any]:
    graph["nodes"].append(h.node("vdl", "DataLoader", batch_size=next(
        n for n in graph["nodes"] if n["id"] == "dl")["data"]["params"][
            "batch_size"], shuffle=False, num_workers=0))
    graph["edges"].append(h.wire("ds", "dataset", "vdl", "dataset"))
    graph["edges"].append(h.wire("vdl", "dataloader", "train",
                                 "val_dataloader"))
    return graph


def _chained(batch_size: int, strategy: str) -> dict[str, Any]:
    """Two loops in sequence: the second trains the first one's model with
    the first one's optimizer, and saves it."""
    graph = _ddp.training_graph(batch_size=batch_size, epochs=2,
                                strategy=strategy, save_as=None)
    graph["nodes"].append(h.node("train2", "TrainingLoop", epochs=2,
                                 device="auto", strategy=strategy))
    graph["nodes"].append(h.node("save", "ModelSaver", path="trained.pt"))
    graph["edges"] += [h.wire("train", "model", "train2", "model"),
                       h.wire("train", "optimizer", "train2", "optimizer"),
                       h.wire("dl", "dataloader", "train2", "dataloader"),
                       h.wire("loss", "loss_fn", "train2", "loss_fn"),
                       h.wire("train2", "model", "save", "model")]
    return graph


def test_two_ddp_loops_in_sequence_train_like_one_process(
        tmp_path, monkeypatch):
    import torch

    run = ChildRun(tmp_path, _chained(4, "ddp"))
    assert run.finish()["status"] == "succeeded", run.tail()
    _, reference = _reference(tmp_path, _chained(8, "single"), monkeypatch)
    trained = torch.load(run.data / "models" / "trained.pt")
    expected = torch.load(reference / "trained.pt")
    for key in expected:
        assert torch.allclose(trained[key], expected[key], atol=1e-5), key
    assert len(run.metric("train_loss")) == 4


def test_a_cancel_stops_every_rank_and_rank_zero_writes_one_checkpoint(
        tmp_path):
    run = ChildRun(tmp_path, _ddp.training_graph(
        batch_size=2, epochs=100000, n=64, lr=0.001))
    run.wait_for(lambda m: m[0] == child.MSG_METRICS)
    run.track()
    run.control.send((child.MSG_CANCEL,))
    finished = run.finish()
    assert finished["status"] == "cancelled", run.tail()
    checkpoints = [m[1] for m in run.of(child.MSG_ARTIFACT)
                   if m[1]["kind"] == "checkpoint"]
    assert len(checkpoints) == 1
    assert checkpoints[0]["meta"]["reason"] == "interrupted"
    assert len(list((run.data / "models" / "interrupted").iterdir())) == 1
    run.assert_no_process_left()


def test_a_failing_rank_fails_the_run_with_its_error(tmp_path):
    # Index 1 has label 7 for a two-class model: rank 1's first batch.
    run = ChildRun(tmp_path, _ddp.training_graph(
        batch_size=4, epochs=50, classify=True, bad=1, save_as=None))
    run.wait_for(lambda m: m[0] == child.MSG_EVENT
                 and m[2].get("node_id") == "train", timeout=120)
    run.track()
    finished = run.finish()
    assert finished["status"] == "failed", run.tail()
    assert finished["error"].startswith("DDP rank 1 of 2 failed: ")
    assert "out of bounds" in finished["error"]
    run.assert_no_process_left()


def test_a_rank_killed_mid_training_fails_the_run_and_leaves_nothing(
        tmp_path):
    run = ChildRun(tmp_path, _ddp.training_graph(
        batch_size=2, epochs=100000, n=64, lr=0.001))
    run.wait_for(lambda m: m[0] == child.MSG_METRICS)
    run.track()
    ranks = sorted(psutil.Process(run.process.pid).children(),
                   key=lambda p: p.create_time())
    assert len(ranks) == 2
    ranks[1].kill()
    finished = run.finish()
    assert finished["status"] == "failed", run.tail()
    assert finished["error"].startswith("DDP rank 1 of 2 exited (exit code ")
    run.assert_no_process_left()


def test_a_killed_launcher_leaves_no_rank_behind(tmp_path):
    run = ChildRun(tmp_path, _ddp.training_graph(
        batch_size=2, epochs=100000, n=64, lr=0.001))
    run.wait_for(lambda m: m[0] == child.MSG_METRICS)
    run.track()
    run.process.kill()
    run.process.join(30)
    run.assert_no_process_left()


# ── through the agent's Runner ────────────────────────────────────────────


async def test_the_agent_runs_a_ddp_run_like_any_other(tmp_path):
    sink = h.FrameSink()
    runner = Runner(runs_dir=tmp_path / "agent-runs", send=sink,
                    offered_devices={"cpu"}, child_entry=_ddp.cpu_ddp_child)
    graph = _ddp.training_graph(batch_size=4, epochs=2, save_as=None)
    await runner.assign(wp.Assign(run_id="a1", graph=graph,
                                  options={"seed": 0}))
    finished = await sink.wait_for(
        lambda f: isinstance(f, wp.RunFinished), timeout=180)
    frames = sink.of("a1")
    assert finished.status == "succeeded", finished
    assert isinstance(frames[0], wp.RunStarted)
    assert [f.seq for f in frames] == list(range(1, len(frames) + 1))
    assert len([f for f in frames if isinstance(f, wp.Metrics)
                for p in f.points if p.name == "train_loss"]) == 2
    handle = runner.runs.get("a1")
    assert handle is None or handle.finished


async def test_the_agent_cancels_a_ddp_run(tmp_path):
    sink = h.FrameSink()
    runner = Runner(runs_dir=tmp_path / "agent-runs", send=sink,
                    offered_devices={"cpu"}, child_entry=_ddp.cpu_ddp_child)
    graph = _ddp.training_graph(batch_size=2, epochs=100000, n=64, lr=0.001,
                                save_as=None)
    await runner.assign(wp.Assign(run_id="a2", graph=graph,
                                  options={"seed": 0}))
    await sink.wait_for(lambda f: isinstance(f, wp.Metrics), timeout=180)
    launcher = psutil.Process(runner.runs["a2"].process.pid)
    started = {launcher.pid} | {
        p.pid for p in launcher.children(recursive=True)}
    assert len(started) >= 3  # the launcher and its two ranks
    assert await runner.cancel("a2", grace_s=60)
    finished = await sink.wait_for(
        lambda f: isinstance(f, wp.RunFinished), timeout=90)
    assert finished.status == "cancelled", finished
    # Every process the run started is gone with it.
    left = set()
    for pid in started:
        try:
            if psutil.Process(pid).status() != psutil.STATUS_ZOMBIE:
                left.add(pid)
        except psutil.Error:
            pass
    assert not left
