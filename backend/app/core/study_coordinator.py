"""The durable study coordinator (#343).

A study is an experiment the SERVER owns: N variants of a graph, each run R
times with a distinct seed per repetition, ranked on one exact objective.
Graph Copilot's ``run_graph_experiments`` used to keep all of that in the
browser tab that started it, so closing the tab abandoned the study. Here
the plan lives in the database (``core.study_store``) and this module drives
it through the ordinary run queue, so a study finishes after its browser has
gone, survives a restart, and resumes without re-running a repetition that
already completed.

How it is driven
----------------
Deterministically, by one function: :meth:`StudyCoordinator.tick`. A tick
reads the study and its attempts, reconciles them with the child runs
(``_reconcile``), and then does whatever the study's status calls for:
submit the next pending attempts up to ``budget.max_parallel_runs``, settle
the study when every repetition is done, or settle a cancel. Ticks for one
study are serialised by a per-study ``asyncio.Lock``. A tick is triggered by
the submit, by every child reaching a terminal status (the Run Service's
terminal listener), by ``cancel`` / ``resume``, and by every ``GET`` of the
study (read repair, as sweeps do). Nothing polls on a timer.

Submission order is repetition-major: every variant's repetition 0, then
every variant's repetition 1, and so on, so a partially-run study already
compares every variant on the same seeds.

Linkage and recovery
--------------------
An attempt names its child BEFORE the child exists: the coordinator mints the
run id, writes it onto the attempt as ``submitting``, then submits the run
with ``exec_runs.study_attempt_id`` stamped on it, then confirms the attempt
as ``queued``. A failure between the submit and the confirmation leaves an
attempt that is ``submitting``; the next tick (or the next boot) looks the
child up by its stamp and relinks it (``attempt_relinked``), or, when no
child carries the stamp, records that the submission never created a run
(``attempt_lost``) and gives the repetition a new attempt. Either way no
repetition is ever run twice by the coordinator.

Restart policy
--------------
The Run Service retires every active run as ``interrupted`` at startup.
:meth:`recover` then reconciles every active study: its children's evidence
is harvested, a ``cancel_requested`` study settles as ``cancelled``, and any
other unfinished study becomes ``interrupted`` and WAITS for an explicit
``resume``. Only ``resume_policy: "auto"`` resumes it on its own. Resuming
never re-runs a completed repetition: a repetition whose current attempt was
cancelled, interrupted or lost gets a new attempt record (attempt + 1, same
seed), and the old attempt stays as history.

SECRET parameters are not persisted (#251): the variant graphs are stored
scrubbed and the values are held in this process's memory. A study resumed
after a restart therefore runs with blank SECRET values, exactly like any
other run whose submitter's process is gone.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Sequence
from uuid import uuid4

from .run_service import (
    MAX_NAME_LENGTH,
    MAX_SEED,
    RunService,
    RunServiceUnavailable,
    RunSubmitError,
    _Broadcast,
    _swallow_task_result,
)
from .run_store import TERMINAL_STATUSES
from .secret_params import (
    SecretVault,
    restore_graph_secrets,
    split_graph_secrets,
)
from .study_store import (
    ATTEMPT_ACTIVE,
    ATTEMPT_CANCELLED,
    ATTEMPT_FINAL,
    ATTEMPT_INTERRUPTED,
    ATTEMPT_LOST,
    ATTEMPT_PENDING,
    ATTEMPT_QUEUED,
    ATTEMPT_RUNNING,
    ATTEMPT_SUBMITTING,
    RESUME_AUTO,
    STUDY_ACTIVE,
    STUDY_CANCEL_REQUESTED,
    STUDY_CANCELLED,
    STUDY_FAILED,
    STUDY_INTERRUPTED,
    STUDY_QUEUED,
    STUDY_RESUMABLE,
    STUDY_RUNNING,
    STUDY_SUCCEEDED,
    AttemptRecord,
    IdempotencyConflict,
    NewVariant,
    StudyEventRecord,
    StudyRecord,
    StudyStateError,
    StudyStore,
    current_attempts,
    rank_study,
)

logger = logging.getLogger(__name__)

#: How long ``shutdown`` waits for in-flight ticks before cancelling them.
_SHUTDOWN_TICK_GRACE_S = 5.0


def repetition_seed(study_seed: int, repetition: int) -> int:
    """The execution seed of one repetition: ``(study seed + r) mod 2^32``.

    Every variant of a repetition runs with the SAME seed (common random
    numbers, so a comparison within a repetition is paired), every
    repetition gets a different one, and a retry of a repetition reuses its
    seed, so a resumed study runs what the original would have.
    """
    return (study_seed + repetition) % (MAX_SEED + 1)


@dataclass(frozen=True)
class StudySpec:
    """A validated study submission, as the route hands it over."""

    name: str | None
    objective: dict[str, Any]
    repetitions: int
    seed: int
    options: dict[str, Any]
    budget: dict[str, Any]
    resume_policy: str
    idempotency_key: str | None
    #: ``(label, ops or None, resolved graph WITH secrets)`` per variant.
    variants: Sequence[tuple[str, list[dict[str, Any]] | None,
                             dict[str, Any]]]
    #: The request fields that identify it for idempotency, with the seed
    #: AS SUBMITTED (None when the server chose one) and graphs omitted;
    #: the scrubbed graphs are folded in by ``request_digest``.
    identity: dict[str, Any]


def request_digest(identity: dict[str, Any],
                   scrubbed_graphs: Sequence[Any]) -> str:
    """sha256 of the canonical request, SECRET values excluded.

    Secrets are scrubbed before hashing so the stored digest cannot be used
    to test guesses of a key. The consequence is that a retry that differs
    ONLY in a SECRET value counts as the same request.
    """
    canonical = json.dumps({"request": identity, "graphs": list(scrubbed_graphs)},
                           sort_keys=True, separators=(",", ":"),
                           allow_nan=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _child_name(study: StudyRecord, label: str, repetition: int) -> str:
    """``<study> / <label> / r<n>``, cut to fit ``MAX_NAME_LENGTH``."""
    suffix = f" / {label} / r{repetition + 1}"
    prefix = study.name or "study"
    room = MAX_NAME_LENGTH - len(suffix)
    if room <= 0:
        return f"r{repetition + 1} {label}"[:MAX_NAME_LENGTH].strip() or "study"
    return (prefix[:room].rstrip() + suffix)[:MAX_NAME_LENGTH]


class StudyCoordinator:
    """Owns every study's scheduling in this process. One per app."""

    def __init__(self, service: RunService, store: StudyStore) -> None:
        self.service = service
        self.store = store
        self._locks: dict[str, asyncio.Lock] = {}
        self._signals: dict[str, _Broadcast] = {}
        self._secrets: dict[tuple[str, int], SecretVault] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        self._closed = False
        service.add_terminal_listener(self._on_terminal)

    # ── plumbing ─────────────────────────────────────────────────────────

    def _lock(self, study_id: str) -> asyncio.Lock:
        lock = self._locks.get(study_id)
        if lock is None:
            lock = self._locks[study_id] = asyncio.Lock()
        return lock

    def _notify(self, study_id: str) -> None:
        signal = self._signals.get(study_id)
        if signal is not None:
            signal.notify()

    def _schedule_tick(self, study_id: str) -> None:
        if self._closed:
            return
        try:
            task = asyncio.ensure_future(self._tick_quietly(study_id))
        except RuntimeError:  # pragma: no cover - no running loop
            return
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        task.add_done_callback(_swallow_task_result)

    async def _tick_quietly(self, study_id: str) -> None:
        try:
            await self.tick(study_id)
        except Exception:
            logger.warning("study %s: a scheduled tick failed", study_id,
                           exc_info=True)

    async def _on_terminal(self, run_id: str, status: str) -> None:
        """The Run Service's terminal listener: copy the child's evidence
        onto its attempt NOW (retention runs right after this returns),
        then tick its study in the background."""
        touched = await self.store.harvest_runs([run_id])
        for study_id in touched:
            self._notify(study_id)
            self._schedule_tick(study_id)

    async def shutdown(self) -> None:
        """Stop scheduling; wait briefly for in-flight ticks. Call BEFORE
        ``RunService.shutdown`` so a child retired by the shutdown does not
        make a tick submit a replacement into a closing queue."""
        self._closed = True
        pending = [task for task in self._tasks if not task.done()]
        if pending:
            _done, still = await asyncio.wait(
                pending, timeout=_SHUTDOWN_TICK_GRACE_S)
            for task in still:
                task.cancel()
            if still:
                await asyncio.gather(*still, return_exceptions=True)
        self._secrets.clear()

    # ── submission ───────────────────────────────────────────────────────

    async def create_study(self, spec: StudySpec) -> tuple[StudyRecord, bool]:
        """Persist the study (or find it by idempotency key), then tick.

        Raises ``study_store.IdempotencyConflict`` for a key reused with a
        different request, and ``RunSubmitError`` /
        ``RunServiceUnavailable`` for a NEW study whose children the run
        service would refuse (see ``RunService.check_placeable_batch``).
        """
        scrubbed: list[NewVariant] = []
        vaults: list[SecretVault] = []
        for label, ops, graph in spec.variants:
            stored, vault = split_graph_secrets(graph,
                                                unknown_types_as_secret=True)
            scrubbed.append(NewVariant(label=label, ops=ops,
                                       graph=dict(stored)))
            vaults.append(vault)
        digest = request_digest(spec.identity, [v.graph for v in scrubbed])
        # The idempotency lookup comes FIRST, so whether a retry gets its
        # study back (or a 409) never depends on what the workers are doing
        # now; only a NEW study is checked for placement (#777). The store
        # repeats the lookup inside its insert transaction, which is what
        # settles two concurrent submits with one key.
        if spec.idempotency_key is not None:
            existing = await self.store.find_by_idempotency_key(
                spec.idempotency_key)
            if existing is not None:
                if existing.request_digest != digest:
                    raise IdempotencyConflict(spec.idempotency_key,
                                              existing.id)
                return existing, False
        # Every child inherits options.target; refuse what the run service
        # would refuse for any of them (a worker it has never seen, a device
        # nobody offers, record_outputs on a worker) before any row exists.
        self.service.check_placeable_batch(
            (graph for _label, _ops, graph in spec.variants), spec.options)
        study, created = await self.store.create_study(
            name=spec.name, objective=spec.objective,
            repetitions=spec.repetitions, seed=spec.seed,
            options=spec.options, budget=spec.budget,
            resume_policy=spec.resume_policy,
            idempotency_key=spec.idempotency_key, request_digest=digest,
            variants=scrubbed,
            repetition_seeds=[repetition_seed(spec.seed, r)
                              for r in range(spec.repetitions)])
        if created:
            for index, vault in enumerate(vaults):
                if vault:
                    self._secrets[(study.id, index)] = vault
            self._notify(study.id)
            study = await self.tick(study.id) or study
        return study, created

    # ── the tick ─────────────────────────────────────────────────────────

    async def tick(self, study_id: str) -> StudyRecord | None:
        """Reconcile one study and advance it. Returns the row after."""
        async with self._lock(study_id):
            return await self._tick_locked(study_id)

    async def _tick_locked(self, study_id: str) -> StudyRecord | None:
        study = await self.store.get_study(study_id)
        if study is None:
            return None
        if await self._reconcile(study_id):
            self._notify(study_id)
        study = await self._advance(study)
        self._notify(study_id)
        return study

    async def _advance(self, study: StudyRecord) -> StudyRecord:
        if study.status in (STUDY_QUEUED, STUDY_RUNNING):
            attempts = await self.store.list_attempts(study.id)
            current = current_attempts(attempts)
            stopped = [a for a in current.values()
                       if a.status in (ATTEMPT_CANCELLED, ATTEMPT_INTERRUPTED)]
            if stopped:
                # A child stopped outside the study (someone cancelled the
                # run itself, or the server is going away). Do not paper over
                # it with a silent retry: stop scheduling and wait for a
                # resume, which is the documented way to re-run it.
                moved = await self.store.transition(
                    study.id, to=STUDY_INTERRUPTED,
                    expected=(STUDY_QUEUED, STUDY_RUNNING),
                    event_type="study_interrupted", payload={
                        "reason": "a child run was stopped outside the study",
                        "attempts": [{"attempt_id": a.id, "run_id": a.run_id,
                                      "status": a.status} for a in stopped]})
                return moved or study
            if study.status == STUDY_QUEUED:
                study = await self.store.transition(
                    study.id, to=STUDY_RUNNING, expected=(STUDY_QUEUED,),
                    event_type="study_started") or study
                if study.status != STUDY_RUNNING:
                    return study
            for attempt in current.values():
                if attempt.status == ATTEMPT_LOST:
                    await self.store.add_attempt(
                        attempt, "the previous submission never created a "
                                 "run")
            study = await self._schedule(study)
            if study.status == STUDY_RUNNING:
                study = await self._settle_if_done(study)
            return study
        if study.status == STUDY_CANCEL_REQUESTED:
            current = current_attempts(await self.store.list_attempts(study.id))
            if not any(a.status in ATTEMPT_ACTIVE for a in current.values()):
                moved = await self.store.transition(
                    study.id, to=STUDY_CANCELLED,
                    expected=(STUDY_CANCEL_REQUESTED,),
                    event_type="study_cancelled", payload={
                        "completed_repetitions_kept": sum(
                            1 for a in current.values()
                            if a.status in ATTEMPT_FINAL)})
                return moved or study
        return study

    async def _settle_if_done(self, study: StudyRecord) -> StudyRecord:
        attempts = await self.store.list_attempts(study.id)
        current = current_attempts(attempts)
        if len(current) == 0 or not all(a.status in ATTEMPT_FINAL
                                        for a in current.values()):
            return study
        variants = await self.store.list_variants(study.id)
        ranking = rank_study(len(variants), study.repetitions, attempts,
                             direction=study.objective["direction"])
        summary = [{key: row[key] for key in (
            "variant_index", "rank", "tie", "tied_with", "mean", "std",
            "count", "eligible")} for row in ranking]
        moved = await self.store.transition(
            study.id, to=STUDY_SUCCEEDED, expected=(STUDY_RUNNING,),
            event_type="study_finished", payload={"ranking": summary})
        return moved or study

    async def _schedule(self, study: StudyRecord) -> StudyRecord:
        """Submit pending attempts up to the parallel budget, in
        repetition-major order. Returns the study row (it may have failed).
        """
        attempts = await self.store.list_attempts(study.id)
        current = current_attempts(attempts)
        active = sum(1 for a in current.values() if a.status in ATTEMPT_ACTIVE)
        parallel = int(study.budget.get("max_parallel_runs") or 1)
        max_runs = study.budget.get("max_runs")
        used = sum(1 for a in attempts if a.status != ATTEMPT_PENDING)
        pending = sorted((a for a in current.values()
                          if a.status == ATTEMPT_PENDING),
                         key=lambda a: (a.repetition, a.variant_index))
        labels: dict[int, str] | None = None
        for attempt in pending:
            if active >= parallel:
                break
            if max_runs is not None and used >= max_runs:
                if active == 0:
                    unfinished = sum(1 for a in current.values()
                                     if a.status not in ATTEMPT_FINAL)
                    return await self.store.transition(
                        study.id, to=STUDY_FAILED, expected=(STUDY_RUNNING,),
                        event_type="study_failed",
                        error=(f"budget.max_runs ({max_runs}) is used up "
                               f"with {unfinished} repetition(s) "
                               "unfinished")) or study
                break
            if labels is None:
                labels = {v.index: v.label
                          for v in await self.store.list_variants(study.id)}
            outcome = await self._submit_attempt(
                study, attempt, labels.get(attempt.variant_index, "?"))
            if outcome is False:
                return await self.store.get_study(study.id) or study
            active += 1
            used += 1
        return study

    async def _submit_attempt(self, study: StudyRecord,
                              attempt: AttemptRecord, label: str) -> bool:
        """One child: linkage first, then the submit, then the confirmation.

        False means "stop scheduling in this tick": the study failed, the
        service is shutting down, or a write failed and the next tick
        reconciles what is left.
        """
        graph = await self.store.get_variant_graph(study.id,
                                                   attempt.variant_index)
        if graph is None:  # pragma: no cover - variants are never deleted
            return False
        restore_graph_secrets(graph, self._secrets.get(
            (study.id, attempt.variant_index)))
        run_id = uuid4().hex
        began = await self.store.begin_submission(attempt.id, run_id)
        if began is None:
            return True
        self._notify(study.id)
        options = {**study.options, "seed": attempt.seed}
        try:
            result = await self.service.submit(
                graph, options=options,
                name=_child_name(study, label, attempt.repetition),
                run_id=run_id, study_id=study.id,
                study_attempt_id=attempt.id)
        except RunSubmitError as exc:
            await self.store.mark_lost(
                attempt.id, f"the run service refused the child: {exc}")
            await self.store.transition(
                study.id, to=STUDY_FAILED, expected=(STUDY_RUNNING,),
                event_type="study_failed",
                error=f"could not submit variant {attempt.variant_index} "
                      f"repetition {attempt.repetition}: {exc}")
            return False
        except RunServiceUnavailable:
            # The child row may already exist (the service refuses after
            # persisting when it starts shutting down mid-submit); the
            # attempt stays `submitting` and the next boot relinks it.
            return False
        except Exception:
            logger.warning("study %s: submitting attempt %s failed",
                           study.id, attempt.id, exc_info=True)
            return False
        try:
            await self.store.confirm_submission(attempt.id, result.status)
        except Exception:
            # The acceptance case: the child exists, its linkage did not
            # land. Left `submitting`; the next tick finds the child by its
            # stamp. Nothing is submitted again for this repetition.
            logger.warning(
                "study %s: attempt %s was submitted as run %s but recording "
                "it failed; the next tick relinks it by its stamp",
                study.id, attempt.id, result.run_id, exc_info=True)
            return False
        self._notify(study.id)
        return True

    async def _reconcile(self, study_id: str) -> bool:
        """Bring the attempts in line with their children. True if anything
        was written.

        1. ``submitting`` attempts: relink the child found by its stamp, or
           mark the attempt ``lost`` when no child carries it.
        2. active attempts: ``queued -> running`` when the child started, a
           terminal child's evidence harvested, and an attempt whose child
           row is gone before its evidence was copied marked ``lost``.
        """
        changed = False
        attempts = await self.store.list_attempts(study_id)
        submitting = [a for a in attempts if a.status == ATTEMPT_SUBMITTING]
        if submitting:
            found = await self.service.store.find_runs_by_study_attempt(
                [a.id for a in submitting])
            for attempt in submitting:
                child = found.get(attempt.id)
                if child is None:
                    await self.store.mark_lost(
                        attempt.id, "no run carries this attempt's stamp; "
                                    "the submission never created one")
                else:
                    await self.store.relink(attempt.id, child)
                changed = True
            attempts = await self.store.list_attempts(study_id)
        terminal: list[str] = []
        for attempt in attempts:
            if attempt.status not in (ATTEMPT_QUEUED, ATTEMPT_RUNNING) \
                    or attempt.run_id is None:
                continue
            child = await self.service.store.get_run(attempt.run_id)
            if child is None:
                await self.store.mark_lost(
                    attempt.id, f"run {attempt.run_id} was deleted before "
                                "its result was recorded",
                    expected=(ATTEMPT_QUEUED, ATTEMPT_RUNNING))
                changed = True
            elif child.status in TERMINAL_STATUSES:
                terminal.append(child.id)
            elif child.status == ATTEMPT_RUNNING \
                    and attempt.status == ATTEMPT_QUEUED:
                await self.store.mark_started(attempt.id, child.started_at)
                changed = True
        if terminal and await self.store.harvest_runs(terminal):
            changed = True
        return changed

    # ── cancel / resume ──────────────────────────────────────────────────

    async def cancel(self, study_id: str) -> tuple[StudyRecord, list[str]]:
        """Stop scheduling and ask every active child to stop.

        Completed attempts are retained; never-submitted ones stay
        ``pending`` so a resume picks them up. Returns the study and the
        run ids a cancel was delivered to. Raises ``KeyError`` for an
        unknown study and ``StudyStateError`` for a succeeded one.
        """
        async with self._lock(study_id):
            study = await self.store.get_study(study_id)
            if study is None:
                raise KeyError(study_id)
            if study.status == STUDY_SUCCEEDED:
                raise StudyStateError(
                    f"study {study_id} already succeeded; there is nothing "
                    "to cancel")
            if study.status == STUDY_CANCELLED:
                return study, []
            await self._reconcile(study_id)
            if study.status != STUDY_CANCEL_REQUESTED:
                study = await self.store.transition(
                    study_id, to=STUDY_CANCEL_REQUESTED,
                    expected=STUDY_ACTIVE | STUDY_RESUMABLE,
                    event_type="study_cancel_requested") or study
            delivered: list[str] = []
            current = current_attempts(await self.store.list_attempts(study_id))
            # One reschedule after the loop, so cancelling one child does
            # not start a queued sibling that the next iteration cancels.
            hold = getattr(self.service, "deferred_reschedule",
                           contextlib.nullcontext)
            with hold():
                for attempt in sorted(current.values(),
                                      key=lambda a: (a.repetition,
                                                     a.variant_index)):
                    if attempt.status in ATTEMPT_ACTIVE and attempt.run_id:
                        outcome = await self.service.cancel(attempt.run_id)
                        if outcome is not None and outcome.cancelled:
                            delivered.append(attempt.run_id)
            await self._reconcile(study_id)
            study = await self._advance(
                await self.store.get_study(study_id) or study)
            self._notify(study_id)
            return study, delivered

    async def resume(self, study_id: str, *,
                     reason: str = "explicit resume") -> StudyRecord:
        """Reopen an interrupted, cancelled or failed study and schedule it.

        Raises ``KeyError`` / ``StudyStateError`` like :meth:`cancel`.
        """
        async with self._lock(study_id):
            await self._reconcile(study_id)
            await self.store.resume(study_id, reason=reason)
            return await self._tick_locked(study_id) or (
                await self.store.get_study(study_id))  # type: ignore[return-value]

    # ── startup ──────────────────────────────────────────────────────────

    async def recover(self) -> int:
        """STARTUP ONLY, after ``RunService.recover_interrupted``: settle
        every study the previous process left active. Returns how many."""
        settled = 0
        for study_id in await self.store.list_study_ids(STUDY_ACTIVE):
            async with self._lock(study_id):
                study = await self.store.get_study(study_id)
                if study is None or study.status not in STUDY_ACTIVE:
                    continue
                await self._reconcile(study_id)
                current = current_attempts(
                    await self.store.list_attempts(study_id))
                if study.status == STUDY_CANCEL_REQUESTED:
                    await self.store.transition(
                        study_id, to=STUDY_CANCELLED,
                        expected=(STUDY_CANCEL_REQUESTED,),
                        event_type="study_cancelled",
                        payload={"reason": "the server restarted while the "
                                           "cancel was in progress"})
                elif current and all(a.status in ATTEMPT_FINAL
                                     for a in current.values()):
                    if study.status == STUDY_QUEUED:
                        study = await self.store.transition(
                            study_id, to=STUDY_RUNNING,
                            expected=(STUDY_QUEUED,),
                            event_type="study_started") or study
                    await self._settle_if_done(study)
                else:
                    await self.store.transition(
                        study_id, to=STUDY_INTERRUPTED,
                        expected=STUDY_ACTIVE,
                        event_type="study_interrupted", payload={
                            "reason": "the server restarted; resume the "
                                      "study to continue it",
                            "attempts": [
                                {"attempt_id": a.id, "run_id": a.run_id,
                                 "status": a.status}
                                for a in current.values()
                                if a.status == ATTEMPT_INTERRUPTED]})
                settled += 1
            study = await self.store.get_study(study_id)
            if study is not None and study.status == STUDY_INTERRUPTED \
                    and study.resume_policy == RESUME_AUTO:
                await self.resume(
                    study_id, reason="resume_policy is auto; resumed after "
                                     "a server restart")
        if settled:
            logger.warning("settled %d study(ies) left active by the previous "
                           "process", settled)
        return settled

    # ── events ───────────────────────────────────────────────────────────

    async def wait_for_events(
        self, study_id: str, *, after_cursor: int = 0, limit: int = 500,
        wait: float = 0.0,
    ) -> list[StudyEventRecord]:
        """Events after *after_cursor*, long-polling up to *wait* seconds
        while the study is still active. Edge-triggered like runs'."""
        deadline = time.monotonic() + max(wait, 0.0)
        signal = self._signals.get(study_id)
        if signal is None:
            signal = self._signals[study_id] = _Broadcast()
        while True:
            waiter = signal.waiter()
            events = await self.store.get_events(
                study_id, after_cursor=after_cursor, limit=limit)
            if events:
                return events
            study = await self.store.get_study(study_id)
            if study is None or study.status not in STUDY_ACTIVE:
                return []
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return []
            try:
                await asyncio.wait_for(waiter.wait(), remaining)
            except asyncio.TimeoutError:
                return []


__all__ = [
    "STUDY_RESUMABLE",
    "StudyCoordinator",
    "StudySpec",
    "repetition_seed",
    "request_digest",
]
