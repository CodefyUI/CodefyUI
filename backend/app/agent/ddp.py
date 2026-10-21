"""A ``ddp`` run: one process per card inside the run subprocess (#792).

When a run's graph has a top-level ``TrainingLoop`` with ``strategy: ddp``
and its device set has two or more CUDA cards, the run subprocess
(``child.main``) does not execute the graph itself. It becomes the
*launcher*: :func:`launch` starts one *rank* process per card, in the
``spawn`` context, gives them a rendezvous on a free loopback port, and
relays what they report to the agent over its one pipe. No rank talks to
the agent.

- Every rank sees all of the run's cards (the launcher's
  ``CUDA_VISIBLE_DEVICES``) and trains on its own, ``cuda:<rank>``
  (``torch.cuda.set_device``), so NCCL can use peer-to-peer copies.
- Rank 0 executes the whole graph through ``child.run_assignment`` exactly
  as a single-process run does, so its events, metrics, artifacts and
  recorded outputs are the run's. The launcher forwards them unchanged.
- A rank other than 0 executes only what the ``ddp`` loops need
  (``core.distributed.rank_graph``), with no signal consumer: it records no
  artifact and so writes no checkpoint and no TensorBoard file. It starts
  its graph once rank 0 enters its first ``ddp`` loop (rank 0 sends
  ``ddp_ready``, the launcher answers the others with ``go``), so what runs
  upstream of the loop, a dataset download say, runs on rank 0 first.
  Its events and metrics are dropped; its warnings are forwarded once
  each; its ``finished`` decides only whether the run failed.
- A cancel from the agent goes to every rank. The ``ddp`` loops agree on
  the stop flag every batch (``DataParallel.stop_checker``), so every rank
  stops at the same batch and rank 0 writes the interrupt checkpoint.
- The run ends with rank 0's ``finished``, its ``peak_memory`` merged with
  the other ranks' (each reads the card it trained on). A rank that fails
  or exits without reporting before that ends the run at once, ``failed``
  with that rank's error (``cancelled`` after a cancel): the launcher stops
  the other ranks (a cancel, then terminate, then kill) and reports.
- The launcher exits only once every rank has exited, so the agent frees
  the run's cards only when no rank holds them. A rank whose launcher is
  gone (its control pipe closes) cancels its run and exits within
  :data:`ORPHAN_EXIT_S`, so a killed launcher leaves no rank behind.

With memory budgets (#788) each rank caps its own card and reports it; the
launcher sends one ``memory_limit`` with all of them, before anything
else, and holds rank 0's other messages until then.

This module imports nothing heavy at module level: the launcher never
imports torch.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable

from . import child
from ..core import distributed

logger = logging.getLogger(__name__)

#: Rank 0 -> launcher: the other ranks may start their graph.
MSG_DDP_READY = "ddp_ready"
#: Launcher -> a rank other than 0: start the graph.
MSG_GO = "go"

#: Collective timeout. A rank other than 0 must reach the loop within this
#: many seconds of rank 0 (it runs the loop's ancestors first).
COLLECTIVE_TIMEOUT_S = 1800.0
#: After rank 0 reports, how long the launcher waits for the other ranks'
#: reports (their peak memory) before it forwards rank 0's.
MERGE_WAIT_S = 2.0
#: After the run's outcome is known, how long a rank still running has to
#: exit after its cancel before it is terminated, then killed.
STOP_GRACE_S = 3.0
KILL_AFTER_S = 2.0
#: After the agent's pipe closes (every rank was cancelled), how long the
#: ranks have before the launcher stops them.
AGENT_GONE_GRACE_S = 30.0
#: How long a rank whose launcher is gone takes before it exits anyway.
ORPHAN_EXIT_S = 5.0
#: Bytes of a crashed rank's output log that go with its error.
TAIL_BYTES = 2048

#: Seconds between two checks of the launcher's loop.
POLL_S = 0.2


# ── the plan ──────────────────────────────────────────────────────────────


def _cuda_index(device: str) -> int | None:
    kind, _, index = str(device).strip().lower().partition(":")
    if kind != "cuda" or not index.isdigit():
        return None
    return int(index)


def plan_for(spec: dict[str, Any]) -> dict[str, Any] | None:
    """How *spec* runs across ranks, or None to run it in one process.

    ``{"world_size", "rank_devices", "backend", "loop_ids"}``. A plan exists
    when the graph has a top-level ``ddp`` loop and the run's device set
    (``spec["env"]["devices"]``, what ``runner.child_environment`` gave the
    subprocess) is two or more CUDA cards; rank *r* trains on the run's
    ``cuda:r``, with ``nccl``. ``spec["ddp"]`` sets ``rank_devices`` and
    ``backend`` explicitly instead (the tests, and a check on hardware
    with fewer cards: two ``gloo`` ranks on the CPU or on one card).
    Raises ``StrategyError`` for ``ddp`` loops that cannot share one run.
    """
    graph = spec.get("graph") or {}
    loop_ids = distributed.ddp_loop_ids(graph)
    if not loop_ids:
        return None
    explicit = spec.get("ddp")
    if isinstance(explicit, dict):
        rank_devices = [str(d) for d in explicit.get("rank_devices") or []]
        backend = explicit.get("backend")
        if not rank_devices:
            return None
    else:
        devices = list((spec.get("env") or {}).get("devices") or [])
        indices = [_cuda_index(device) for device in devices]
        if len(devices) < distributed.MIN_DEVICES or None in indices:
            return None
        rank_devices = [f"cuda:{rank}" for rank in range(len(devices))]
        backend = None
    if backend is None:
        backend = "nccl" if all(_cuda_index(d) is not None
                                for d in rank_devices) else "gloo"
    distributed.check_loops_ordered(graph, loop_ids)
    return {"world_size": len(rank_devices), "rank_devices": rank_devices,
            "backend": str(backend), "loop_ids": loop_ids}


def free_port() -> int:
    """A free TCP port on the loopback interface, for the rendezvous."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def rank_spec(spec: dict[str, Any], plan: dict[str, Any],
              rank: int) -> dict[str, Any]:
    """*spec* as rank *rank* executes it: its own device, and the memory
    budget of that device only (each rank caps the card it trains on)."""
    device = plan["rank_devices"][rank]
    out = dict(spec)
    out["device"] = device
    budgets = spec.get("memory_budgets") or {}
    out["memory_budgets"] = {name: value for name, value in budgets.items()
                             if name == device}
    out.pop("ddp", None)
    return out


# ── a rank ────────────────────────────────────────────────────────────────


def init_process_group(plan: dict[str, Any], rank: int, port: int,
                       device: str) -> tuple[str, Any]:
    """Join the run's process group; ``(backend, gloo group or None)``."""
    import torch
    import torch.distributed as dist

    backend = plan["backend"]
    if backend == "nccl" and not dist.is_nccl_available():
        logger.warning("NCCL is not available in this torch build; the "
                       "ranks communicate with gloo instead")
        backend = "gloo"
    if _cuda_index(device) is not None:
        torch.cuda.set_device(torch.device(device))
    dist.init_process_group(
        backend, init_method=f"tcp://127.0.0.1:{port}", rank=rank,
        world_size=int(plan["world_size"]),
        timeout=timedelta(seconds=COLLECTIVE_TIMEOUT_S))
    # The stop flag and the metric sums are a few CPU numbers: a gloo
    # group carries them without a device round trip.
    group = dist.new_group(backend="gloo") if backend != "gloo" else None
    return backend, group


def _leave_process_group() -> None:
    import sys

    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception:  # noqa: BLE001 -- the process is about to exit
        pass


def _orphaned() -> None:
    """The launcher is gone: exit after :data:`ORPHAN_EXIT_S`, whatever the
    run is doing (a rank waiting in a collective never sees its cancel)."""
    timer = threading.Timer(ORPHAN_EXIT_S, os._exit, args=(1,))
    timer.daemon = True
    timer.start()


def rank_listener(control_in: Any, context: Any, outputs: Any = None,
                  go: threading.Event | None = None) -> None:
    """Thread body: what the launcher sends a rank. ``cancel`` cancels the
    run (and releases a rank waiting for ``go``); reads of rank 0's
    recorded outputs go to *outputs*; a closed pipe is a cancel and, after
    :data:`ORPHAN_EXIT_S`, an exit."""
    while True:
        try:
            message = control_in.recv()
        except (EOFError, OSError):
            context.cancel()
            if go is not None:
                go.set()
            _orphaned()
            return
        if not isinstance(message, tuple) or not message:
            continue
        kind = message[0]
        if kind == child.MSG_CANCEL:
            context.cancel()
            if go is not None:
                go.set()
        elif kind == MSG_GO and go is not None:
            go.set()
        elif outputs is not None and kind == child.MSG_FETCH \
                and len(message) == 5:
            outputs.answer(*message[1:])
        elif outputs is not None and kind == child.MSG_DROP_OUTPUTS \
                and len(message) == 2:
            outputs.drop(message[1])


def rank_main(spec: dict[str, Any], plan: dict[str, Any], rank: int,
              port: int, events_out: Any, control_in: Any) -> None:
    """``Process`` target of one rank. The launcher's environment (the
    run's ``CUDA_VISIBLE_DEVICES`` and data settings) is inherited."""
    child.ignore_interrupts()
    child._redirect_output(spec.get("log_path"))
    logging.basicConfig(
        level=logging.INFO, force=True,
        format=f"%(asctime)s rank {rank} %(levelname)s %(name)s: %(message)s")
    send = child._safe_sender(events_out)
    try:
        if rank == 0:
            run_rank_zero(spec, plan, port, send, control_in)
        else:
            run_rank(spec, plan, rank, port, send, control_in)
    finally:
        _leave_process_group()
        try:
            events_out.close()
        except OSError:
            pass


def run_rank_zero(spec: dict[str, Any], plan: dict[str, Any], port: int,
                  send: child.Send, control_in: Any, *,
                  discover: bool = True) -> None:
    """Rank 0: the run as ``child.run_assignment`` executes it, inside the
    process group. Never raises."""
    device = plan["rank_devices"][0]

    def prepare(context: Any) -> None:
        backend, group = init_process_group(plan, 0, port, device)
        distributed.install(distributed.RankState(
            rank=0, world_size=int(plan["world_size"]), device=device,
            backend=backend, loop_ids=frozenset(plan["loop_ids"]),
            group=group, release=lambda: send((MSG_DDP_READY,))))

    child.run_assignment(rank_spec(spec, plan, 0), send, control_in,
                         discover=discover, listener=rank_listener,
                         prepare=prepare)


def run_rank(spec: dict[str, Any], plan: dict[str, Any], rank: int,
             port: int, send: child.Send, control_in: Any, *,
             discover: bool = True) -> None:
    """A rank other than 0: join the group, wait for ``go``, execute the
    ``ddp`` loops and their ancestors, send ``finished``. Never raises."""
    status, error, peak_memory = "failed", None, None
    try:
        import asyncio

        from ..core.executors import PeakMemoryScope
        from ..core.runtime import initialize_runtime

        own = rank_spec(spec, plan, rank)
        device = own["device"]
        if own["memory_budgets"]:
            send((child.MSG_MEMORY_LIMIT,
                  child.apply_memory_budgets(own["memory_budgets"])))
        if discover:
            initialize_runtime()
        options = dict(spec.get("options") or {})
        context = child.build_context(spec["run_id"], options, device)
        go = threading.Event()
        if control_in is not None:
            threading.Thread(target=rank_listener,
                             args=(control_in, context, None, go),
                             name="rank-listener", daemon=True).start()
        backend, group = init_process_group(plan, rank, port, device)
        distributed.install(distributed.RankState(
            rank=rank, world_size=int(plan["world_size"]), device=device,
            backend=backend, loop_ids=frozenset(plan["loop_ids"]),
            group=group))
        go.wait()
        if context.should_stop():
            status = "cancelled"
        else:
            graph = distributed.rank_graph(
                child._normalised_graph(spec["graph"]), plan["loop_ids"])

            async def execute() -> tuple[str, str | None, Any]:
                from ..core.execution_context import CancellationError
                from ..core.graph_engine import (
                    build_preset_fallback,
                    execute_graph,
                )
                from ..core.seeding import seed_rngs

                scope = PeakMemoryScope(device, exclusive=True).open()
                try:
                    seed_rngs(options.get("seed"))
                    # No on_signal: nothing this rank logs is recorded, so
                    # nothing it would write as an artifact is written.
                    # fail_fast whatever the run's error mode: a node that
                    # failed here must end this rank, or rank 0 would wait
                    # in the loop for a rank that never arrives.
                    await execute_graph(
                        graph["nodes"], graph["edges"], context=context,
                        error_mode="fail_fast", run_id=spec["run_id"],
                        preset_fallback=build_preset_fallback(
                            graph["presets"]),
                        subgraphs=graph["subgraphs"])
                    outcome: tuple[str, str | None] = ("succeeded", None)
                except CancellationError:
                    outcome = ("cancelled", None)
                except Exception as exc:  # noqa: BLE001 -- reported
                    logger.warning("rank %d failed: %s", rank, exc,
                                   exc_info=True)
                    outcome = ("failed", f"{type(exc).__name__}: {exc}")
                peak = scope.read(exact=True)
                scope.close()
                return outcome[0], outcome[1], peak

            status, error, peak_memory = asyncio.run(execute())
    except BaseException as exc:  # noqa: BLE001 -- reported, then we exit
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        import traceback

        traceback.print_exc()
    send((child.MSG_FINISHED, {"status": status, "error": error,
                               "peak_memory": peak_memory}))


# ── the launcher ──────────────────────────────────────────────────────────


@dataclass
class _Rank:
    index: int
    process: Any
    events_in: Any
    control_out: Any
    #: Its ``finished`` payload, once received.
    report: dict[str, Any] | None = None
    #: Its pipe reached EOF.
    closed: bool = False
    #: Its process exited.
    exited: bool = False
    #: When it was terminated (monotonic), None if never.
    terminated_at: float | None = None
    killed: bool = False
    #: Why it failed (it reported ``failed``, or exited without reporting),
    #: when that came after rank 0's report: the likelier cause of a
    #: failure rank 0 reports (a peer gone breaks its collective).
    failure: str | None = None


@dataclass
class _Launch:
    ranks: list[_Rank]
    send: child.Send
    world_size: int
    log_path: str | None = None
    cancel_requested: bool = False
    agent_gone: bool = False
    #: The outcome went to the agent: only ``output_answer`` follows.
    reported: bool = False
    #: When the outcome was decided; the stragglers are stopped from then.
    decided_at: float | None = None
    #: Rank 0's report and when it came, held for :data:`MERGE_WAIT_S`.
    rank0_report: dict[str, Any] | None = None
    rank0_at: float | None = None
    #: Ranks that cap a card (#788) and what they applied, by rank.
    limits_expected: set[int] = field(default_factory=set)
    limits: dict[int, dict[str, Any]] = field(default_factory=dict)
    limit_sent: bool = False
    #: Rank 0's messages that arrived before every cap did.
    held: list[tuple] = field(default_factory=list)
    #: Warnings rank 0 gave, and those another rank gave first: each is
    #: forwarded once, whichever rank says it first.
    rank0_warnings: set[tuple] = field(default_factory=set)
    other_warnings: set[tuple] = field(default_factory=set)

    # ── to the agent ──

    def _out(self, message: tuple) -> None:
        if self.limit_sent or not self.limits_expected:
            self.send(message)
        else:
            self.held.append(message)

    def _maybe_send_limits(self) -> None:
        if self.limit_sent or not self.limits_expected \
                or set(self.limits) < self.limits_expected:
            return
        merged: dict[str, Any] = {}
        for rank in sorted(self.limits):
            merged.update(self.limits[rank])
        self.limit_sent = True
        self.send((child.MSG_MEMORY_LIMIT, merged))
        held, self.held = self.held, []
        for message in held:
            self.send(message)

    def _merged_peak(self) -> dict[str, Any] | None:
        merged: dict[str, Any] = {}
        reports = [self.rank0_report] + [r.report for r in self.ranks[1:]]
        for report in reports:
            peak = (report or {}).get("peak_memory")
            if isinstance(peak, dict):
                for device, value in peak.items():
                    merged.setdefault(device, value)
        return merged or None

    def report(self, payload: dict[str, Any]) -> None:
        """Send the run's ``finished`` (once) and start stopping the ranks
        still running."""
        if self.reported:
            return
        self.reported = True
        if self.decided_at is None:
            self.decided_at = time.monotonic()
        if self.limits_expected and not self.limit_sent:
            # A cap failed somewhere: the run never started (#788).
            self.held = []
        self.send((child.MSG_FINISHED, payload))
        for rank in self.ranks:
            if not rank.exited and rank.report is None:
                self.tell(rank, (child.MSG_CANCEL,))

    def fail(self, error: str) -> None:
        """End the run because a rank failed, before rank 0 reported."""
        if self.reported:
            return
        if self.cancel_requested:
            payload = {"status": "cancelled", "error": None}
        else:
            payload = {"status": "failed", "error": error}
        payload["peak_memory"] = self._merged_peak()
        self.report(payload)

    # ── to the ranks ──

    @staticmethod
    def tell(rank: _Rank, message: tuple) -> None:
        try:
            rank.control_out.send(message)
        except (BrokenPipeError, EOFError, OSError, ValueError):
            pass

    def cancel_all(self) -> None:
        self.cancel_requested = True
        for rank in self.ranks:
            if not rank.exited:
                self.tell(rank, (child.MSG_CANCEL,))

    # ── from the agent ──

    def from_agent(self, message: Any) -> None:
        if not isinstance(message, tuple) or not message:
            return
        kind = message[0]
        if kind == child.MSG_CANCEL:
            self.cancel_all()
        elif kind in (child.MSG_FETCH, child.MSG_DROP_OUTPUTS):
            self.tell(self.ranks[0], message)

    # ── from a rank ──

    def from_rank(self, rank: _Rank, message: Any) -> None:
        if not isinstance(message, tuple) or not message:
            return
        kind = message[0]
        if kind == child.MSG_MEMORY_LIMIT and len(message) == 2:
            self.limits[rank.index] = dict(message[1] or {})
            self._maybe_send_limits()
            return
        if kind == MSG_DDP_READY:
            if rank.index == 0:
                for other in self.ranks[1:]:
                    self.tell(other, (MSG_GO,))
            return
        if kind == child.MSG_FINISHED and len(message) == 2:
            payload = dict(message[1] or {})
            rank.report = payload
            if rank.index == 0:
                if self.rank0_report is None:
                    self.rank0_report = payload
                    self.rank0_at = time.monotonic()
                    if self.decided_at is None:
                        self.decided_at = self.rank0_at
            elif payload.get("status") == "failed":
                self._rank_failed(
                    rank, f"DDP rank {rank.index} of {self.world_size} "
                          f"failed: {payload.get('error') or 'no error'}")
            return
        key = None
        if kind == child.MSG_WARNING and len(message) == 2 \
                and isinstance(message[1], dict):
            key = (message[1].get("kind"), message[1].get("detail"),
                   message[1].get("node_id"))
        if rank.index == 0:
            if self.reported and kind != child.MSG_OUTPUT_ANSWER:
                return
            if key is not None:
                self.rank0_warnings.add(key)
                if key in self.other_warnings:
                    self.other_warnings.discard(key)
                    return  # already forwarded, from another rank
            self._out(message)
            return
        if key is not None and not self.reported \
                and key not in self.rank0_warnings \
                and key not in self.other_warnings:
            self.other_warnings.add(key)
            self._out(message)
        # Events, metrics and artifacts of the other ranks repeat rank 0's.

    def _rank_failed(self, rank: _Rank, error: str) -> None:
        """A rank other than 0 failed: that ends the run, unless rank 0
        reported already (then it may explain rank 0's failure)."""
        if self.rank0_report is None:
            self.fail(error)
        else:
            rank.failure = error

    def crashed(self, rank: _Rank) -> None:
        """*rank* exited without reporting."""
        code = getattr(rank.process, "exitcode", None)
        error = (f"DDP rank {rank.index} of {self.world_size} exited "
                 f"(exit code {code}) without reporting")
        tail = _tail(self.log_path)
        if tail:
            error = f"{error}\n--- output tail ---\n{tail}"
        if rank.index == 0:
            self.fail(error)
        else:
            self._rank_failed(rank, error)

    # ── housekeeping ──

    def settle(self, now: float) -> None:
        """Forward rank 0's report once the others reported or
        :data:`MERGE_WAIT_S` passed, and stop the ranks that outlive the
        outcome."""
        if self.rank0_report is not None and not self.reported:
            others_done = all(r.report is not None or r.exited
                              for r in self.ranks[1:])
            if others_done or now - (self.rank0_at or now) >= MERGE_WAIT_S:
                payload = dict(self.rank0_report)
                if self.cancel_requested and payload.get("status") == "failed":
                    # A rank that stopped first broke rank 0's collective.
                    payload.update(status="cancelled", error=None)
                elif payload.get("status") == "failed":
                    # A rank that failed or died first is why rank 0's
                    # collective failed: name that one.
                    cause = next((r.failure for r in self.ranks[1:]
                                  if r.failure), None)
                    if cause is not None:
                        payload["error"] = cause
                payload["peak_memory"] = self._merged_peak()
                self.report(payload)
        if self.agent_gone and self.decided_at is None:
            # Their cancel gives rank 0 the time to write its interrupt
            # checkpoint; a rank that hangs is stopped after that.
            self.decided_at = now + AGENT_GONE_GRACE_S
        if self.decided_at is None:
            return
        for rank in self.ranks:
            if rank.exited:
                continue
            if rank.index == 0 and rank.report is not None:
                continue  # writing its recorded outputs, on its own clock
            if rank.terminated_at is None \
                    and now - self.decided_at >= STOP_GRACE_S:
                rank.terminated_at = now
                _stop(rank.process, kill=False)
            elif rank.terminated_at is not None and not rank.killed \
                    and now - rank.terminated_at >= KILL_AFTER_S:
                rank.killed = True
                _stop(rank.process, kill=True)

    def terminate_all(self) -> None:
        for rank in self.ranks:
            if not rank.exited:
                _stop(rank.process, kill=False)


def _stop(process: Any, *, kill: bool) -> None:
    try:
        if kill:
            process.kill()
        else:
            process.terminate()
    except Exception:  # noqa: BLE001 -- already gone
        pass


def _tail(path: str | None) -> str:
    if not path:
        return ""
    try:
        with open(path, "rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - TAIL_BYTES))
            return handle.read().decode("utf-8", "replace")
    except OSError:
        return ""


def launch(spec: dict[str, Any], plan: dict[str, Any], send: child.Send,
           control_in: Any, *, entry: Callable[..., None] = rank_main,
           context: Any = None) -> None:
    """Start the ranks, relay them, report the run; return once every rank
    has exited. Never raises. *entry* is the rank processes' target
    (:func:`rank_main`; the tests pass light stand-ins)."""
    import multiprocessing
    from multiprocessing.connection import wait

    ctx = context or multiprocessing.get_context("spawn")
    ranks: list[_Rank] = []
    try:
        port = free_port()
        for index in range(int(plan["world_size"])):
            events_in, events_out = ctx.Pipe(duplex=False)
            control_in_r, control_out = ctx.Pipe(duplex=False)
            # Not daemonic: a daemonic process may not start children,
            # and a DataLoader with num_workers does.
            process = ctx.Process(
                target=entry,
                args=(spec, plan, index, port, events_out, control_in_r),
                name=f"codefyui-run-{spec.get('run_id')}-rank{index}",
                daemon=False)
            process.start()
            events_out.close()
            control_in_r.close()
            ranks.append(_Rank(index=index, process=process,
                               events_in=events_in, control_out=control_out))
    except BaseException as exc:  # noqa: BLE001 -- reported, then we exit
        for rank in ranks:
            _stop(rank.process, kill=True)
        send((child.MSG_FINISHED, {
            "status": "failed", "peak_memory": None,
            "error": f"could not start the DDP ranks: "
                     f"{type(exc).__name__}: {exc}"}))
        for rank in ranks:
            rank.process.join(KILL_AFTER_S)
        return
    logger.info("run %s: %d DDP ranks (%s) on %s, rendezvous on port %d",
                spec.get("run_id"), len(ranks), plan["backend"],
                ", ".join(plan["rank_devices"]), port)
    state = _Launch(ranks=ranks, send=send, world_size=len(ranks),
                    log_path=spec.get("log_path"))
    budgets = spec.get("memory_budgets") or {}
    state.limits_expected = {
        index for index, device in enumerate(plan["rank_devices"])
        if device in budgets}
    _on_sigterm(state)
    try:
        _relay(state, control_in, wait)
    except BaseException as exc:  # noqa: BLE001 -- the ranks must not stay
        logger.exception("the DDP launcher failed")
        state.terminate_all()
        state.fail(f"the DDP launcher failed: {type(exc).__name__}: {exc}")
    for rank in ranks:
        rank.process.join(KILL_AFTER_S)
        if rank.process.is_alive():
            _stop(rank.process, kill=True)
            rank.process.join(KILL_AFTER_S)
        for end in (rank.events_in, rank.control_out):
            try:
                end.close()
            except OSError:
                pass
    if not state.reported:
        # Every rank exited and none said how the run went.
        state.fail("the DDP ranks exited without reporting")


def _on_sigterm(state: _Launch) -> None:
    """The agent stops a run that ignores its cancel with SIGTERM: take the
    ranks down too. (Windows terminates outright; the ranks then see their
    pipe close and exit on their own, see :func:`rank_listener`.)"""
    import signal

    def handler(signum: int, frame: Any) -> None:
        state.terminate_all()
        for rank in state.ranks:
            try:
                rank.process.join(KILL_AFTER_S)
                if rank.process.is_alive():
                    rank.process.kill()
            except Exception:  # noqa: BLE001
                pass
        os._exit(1)

    try:
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, handler)
    except (ValueError, OSError, AttributeError):
        pass


def _relay(state: _Launch, control_in: Any, wait: Callable[..., list]) -> None:
    """The launcher's loop: until every rank has exited."""
    by_conn = {id(rank.events_in): rank for rank in state.ranks}
    by_sentinel = {rank.process.sentinel: rank for rank in state.ranks}

    def drain(rank: _Rank) -> None:
        while not rank.closed:
            try:
                if not rank.events_in.poll():
                    return
                message = rank.events_in.recv()
            except (EOFError, OSError):
                rank.closed = True
                return
            state.from_rank(rank, message)

    while not all(rank.exited for rank in state.ranks):
        waitables: list[Any] = []
        if control_in is not None and not state.agent_gone:
            waitables.append(control_in)
        waitables += [rank.events_in for rank in state.ranks if not rank.closed]
        waitables += [rank.process.sentinel for rank in state.ranks
                      if not rank.exited]
        for ready in wait(waitables, POLL_S):
            if ready is control_in:
                try:
                    state.from_agent(control_in.recv())
                except (EOFError, OSError):
                    # Nobody is left to report to: stop the ranks.
                    state.agent_gone = True
                    state.cancel_all()
                continue
            rank = by_conn.get(id(ready))
            if rank is not None:
                drain(rank)
                continue
            rank = by_sentinel.get(ready)
            if rank is not None:
                drain(rank)
                rank.exited = True
                # The sentinel can fire a moment before the process can be
                # reaped (its exit code read).
                rank.process.join(1.0)
                rank.closed = True
                if rank.report is None and not state.reported \
                        and rank.terminated_at is None:
                    state.crashed(rank)
        state.settle(time.monotonic())
