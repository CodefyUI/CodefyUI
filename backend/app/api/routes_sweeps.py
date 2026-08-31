"""REST surface for parameter sweeps (#140).

    POST   /api/sweeps                compile + queue N variants  -> 201

``GET /api/sweeps/{id}`` (the ranked comparison table, JSON or CSV) and
``POST /api/sweeps/{id}/cancel`` land in their own commits. The conventions
below are the module's rather than one route's, so they are stated once here.

Conventions are ``routes_runs.py``'s, deliberately: plain ``dict`` returns
and no ``response_model``, request models declared locally with
``extra="forbid"``, every error a ``{"detail": ...}`` (the 9-key run envelope
belongs to ``routes_graph_run`` / ``routes_apps``), and a 503 when the
service is not on ``app.state`` -- the lifespan does not run under httpx's
``ASGITransport``.

Auth follows the house rule in ``main.py`` exactly: ``auth_guard`` requires
the session token for the mutating routes and lets the GET through like every
other GET in the app. This router is deliberately NOT added to
``_AUTH_EXEMPT_PREFIXES`` -- that list exists for ``/api/apps`` and
``/api/keys``, which carry per-route auth dependencies instead, and joining
it without one would silently drop authentication entirely
(``test_api_sweeps.test_sweeps_routes_are_not_under_an_auth_exempt_prefix``
guards it).

**One deliberate deviation from ``routes_runs.py``: ``POST /api/sweeps``
answers 201, not 200.** A sweep submit creates a new addressable resource
with its own URL that the caller goes on to poll, which is what 201 means.
``POST /api/runs`` answers 200 and predates any such resource-shaped
surface; it is not worth changing now, and this note exists so a later
reviewer does not "fix" one to match the other.

No long poll and no pagination: a sweep has at most ``MAX_SWEEP_RUNS``
variants, and a caller who wants live progress long-polls each child's
``GET /api/runs/{id}/events``, which already works.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    field_validator,
)

from ..config import settings
from ..core.run_service import (
    LANE_INTERACTIVE,
    MAX_SEED,
    RunService,
    RunServiceUnavailable,
    RunSubmitError,
    normalize_graph,
    normalize_name,
    normalize_options,
    # Private, and imported on purpose: retrieving a detached task's outcome
    # so asyncio does not log it is ONE two-line implementation, and
    # `_mark_failed_on_the_way_out` below is modelled on the RunGate release
    # that already uses it.
    _swallow_task_result,
)
from ..core.run_store import RunProvenance
from ..core.sweep_compiler import (
    CompiledSweep,
    CompiledVariant,
    SweepCompileError,
    compile_sweep,
)
from ..core.sweep_store import SweepRecord, SweepStore, SweepVariant

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sweeps", tags=["sweeps"])

#: A variant whose run cannot be reached: it never got one (the failed
#: submit loop), or its row was pruned or deleted before any read harvested
#: it. Not an ``exec_runs.status`` -- it is the absence of one.
_STATUS_MISSING = "missing"


def _get_service(request: Request) -> RunService:
    service = getattr(request.app.state, "run_service", None)
    if service is None:
        raise HTTPException(status_code=503,
                            detail="run service not initialised")
    return service


def _get_sweep_store(request: Request) -> SweepStore:
    store = getattr(request.app.state, "sweep_store", None)
    if store is None:
        raise HTTPException(status_code=503,
                            detail="sweep store not initialised")
    return store


# ── request models ────────────────────────────────────────────────────────
#
# extra="forbid" on every one of them is load-bearing, the same way it is in
# routes_packs.py:80-86: it is what makes "the client cannot smuggle in an
# unknown key" a guarantee of the schema rather than of every handler
# remembering. OPTION_KEYS enforces the same closed-key discipline for run
# options.


class SweepRangeModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # StrictInt | StrictFloat rather than a plain `float`. Measured on
    # pydantic 2.12: a lax `float` field accepts JSON `true` and coerces it
    # to 1.0, which would smuggle a bool past the bool-is-not-a-number rule
    # this design applies everywhere else. The union still accepts a JSON
    # integer, which {min: 16, max: 128} needs.
    min: StrictInt | StrictFloat
    max: StrictInt | StrictFloat
    count: int = Field(strict=True)
    scale: Literal["linear", "log"]
    type: Literal["int", "float"]


class SweepParamModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    param: str
    # Exact JSON types survive this union (measured): 1 stays int, true
    # stays bool, 2.0 stays float. That is what lets the compiler's type
    # matrix see what the caller actually wrote rather than a coercion.
    values: list[int | float | bool | str] | None = None
    range: SweepRangeModel | None = None


class SweepSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["grid", "random"]
    # strict, for the same reason `count` is: on a lax int field
    # `seed: true` is silently the integer 1 and `seed: "7"` is 7.
    seed: int | None = Field(default=None, strict=True)
    samples: int | None = Field(default=None, strict=True)
    params: list[SweepParamModel]


class SweepObjectiveModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str = Field(min_length=1, max_length=128)
    direction: Literal["minimize", "maximize"]

    @field_validator("metric")
    @classmethod
    def _strip_metric(cls, value: str) -> str:
        """Validated as a non-empty bounded string and NOTHING else.

        Deliberately not checked against a known-series list: the name is
        whatever a node computes, plugins and custom nodes can log anything,
        and at submit time no variant has run so there is nothing to check
        against.
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError("objective.metric must not be blank")
        return stripped


class CreateSweepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_graph: dict[str, Any]
    sweep_spec: SweepSpecModel
    #: REQUIRED, with no default. A default of "val_loss" would produce an
    #: all-empty ranking for the majority of graphs a user is likely to
    #: sweep first (spec 6.5), with nothing on screen explaining why, and
    #: `direction` cannot be inferred from a user-chosen string either.
    objective: SweepObjectiveModel
    #: Run options for EVERY variant, passed verbatim to normalize_options.
    #: `objective` and `seed_variants` are top-level rather than options
    #: keys: OPTION_KEYS is a closed set, so smuggling one in there would be
    #: a 400 from the wrong validator with the wrong message.
    options: Any = None
    name: str | None = None
    #: RULING 1. A run carrying options["seed"] takes a PROCESS-WIDE
    #: exclusive lock, so seeding all 32 variants would serialise the sweep
    #: regardless of the queue's concurrency AND freeze every interactive
    #: canvas run for the sweep's whole duration.
    seed_variants: bool = False


# ── payload helpers ───────────────────────────────────────────────────────


def _variant_params(compiled: CompiledSweep,
                    variant: CompiledVariant) -> list[dict[str, Any]]:
    """``[{node_id, param, value}]`` in DECLARED order -- which is the order
    the CSV columns and the comparison table use."""
    return [{"node_id": param.node_id, "param": param.param, "value": value}
            for param, value in zip(compiled.params, variant.assignment)]


def _placeholder_variant(compiled: CompiledSweep,
                         variant: CompiledVariant) -> SweepVariant:
    """The stored entry before its run exists; ``run_id`` and ``seed`` are
    patched in as each submit returns (spec 5.2 step 7)."""
    return SweepVariant(
        index=variant.index, domain_index=variant.domain_index, run_id=None,
        params=_variant_params(compiled, variant), seed=None, objective=None,
        status=None, harvested_at=None)


def _compiled_spec(spec: dict[str, Any],
                   compiled: CompiledSweep) -> dict[str, Any]:
    """The normalised spec with each param's EXPANDED domain folded in.

    ``sweeps.spec`` is documented as "the compiled, normalised sweep_spec",
    and this is what makes it so. The read side needs the expanded,
    deduplicated domain -- it is what shows a caller that an int range
    collapsed -- and it cannot recompute it: recompiling needs the base
    graph, which lives only on each child's snapshot and may have been
    pruned, and it would re-resolve every address against a registry that
    may have changed since submit.
    """
    return {
        **spec,
        "params": [{**raw, "domain": list(param.domain),
                    "param_type": param.param_type}
                   for raw, param in zip(spec["params"], compiled.params)],
    }


def _spec_param_payload(sweep: SweepRecord) -> list[dict[str, Any]]:
    """The swept addresses with their expanded domains, for both routes."""
    return [{"node_id": entry.get("node_id"), "param": entry.get("param"),
             "domain": list(entry.get("domain") or [])}
            for entry in sweep.spec.get("params", [])]


#: Strong references to the detached writes fired by
#: ``_mark_failed_on_the_way_out``. asyncio holds only a WEAK reference to a
#: running task, so without this a settle fired while the request is being
#: cancelled could be garbage-collected mid-flight. Mirrors ``RunGate``'s own
#: ``self._wakes`` (``run_service.py:1038-1040``).
_PENDING_MARKS: set[asyncio.Task[Any]] = set()


async def _mark_failed_on_the_way_out(store: SweepStore, sweep_id: str,
                                      error: str) -> None:
    """Settle the sweep row while this request may itself be cancelled.

    Shaped exactly like ``RunGate._release``'s notification
    (``run_service.py:1029-1049``) and for the same reason: a bare
    ``await store.mark_failed(...)`` inside a ``finally`` IS a cancellation
    point, and Starlette runs a handler inside an anyio cancel scope where
    every checkpoint after a cancel raises again -- so the write would never
    be reached. Firing it as its OWN task and shielding the wait lets the
    cancel take this await while the write runs on to completion.

    Two properties, both required:

    * **the CancelledError still propagates.** ``shield`` re-raises the
      cancel that landed on this await, and ``except Exception`` does not
      catch a ``BaseException``, so the request still ends cancelled. This
      write is a side effect on the way out, never a recovery.
    * **a failure here never replaces the exception on its way out.** This
      runs in a ``finally``, so a raise would substitute itself for whatever
      the handler was already failing with and the original would be gone.
      Swallowing costs nothing that was not already lost -- the row is
      unsettled either way -- while losing the caller's own error is new
      damage.

    Not durable against the loop itself going away: a process-wide shutdown
    cancels every task and can take the detached write with it. Recovering a
    sweep across a restart is RULING 2's explicit hand-off to #343.
    """
    try:
        task = asyncio.ensure_future(store.mark_failed(sweep_id, error))
    except RuntimeError:  # pragma: no cover - the loop is already gone
        return
    _PENDING_MARKS.add(task)
    task.add_done_callback(_PENDING_MARKS.discard)
    task.add_done_callback(_swallow_task_result)
    try:
        await asyncio.shield(task)
    except Exception:  # noqa: BLE001 - see above; CancelledError is a
        # BaseException and so still propagates, which is required.
        logger.warning("sweep %s: settling it as failed on the way out of "
                       "the submit loop failed", sweep_id, exc_info=True)


def _refuse_unrecordable_outputs(request: Request, options: dict[str, Any],
                                 variant_count: int) -> None:
    """RULE 5 of spec 5.2 -- the only one that needs the compiled count.

    ``RunOutputStore`` keeps only the newest ``max_runs`` runs (20 by
    default), so a 32-variant sweep with ``record_outputs`` on evicts its
    own earliest variants before it finishes -- silently, with nothing
    erroring and ``/api/execution/outputs`` simply answering nothing.

    Best-effort by design: the store is shared with the canvas, so even a
    sweep under the cap can be evicted by concurrent captures. The job here
    is to refuse the case that CANNOT work, not to promise the rest will.
    The bound is read from the live store rather than a literal 20, so
    raising it raises this limit with it.

    The store is absent under httpx's ASGITransport and in any embedded host
    without one; SKIP the rule then rather than refusing -- with no output
    store there is nothing to evict and nothing to warn about. The same
    tolerance ``routes_runs.delete_run`` shows for the same object.
    """
    if not options.get("record_outputs"):
        return
    store = getattr(request.app.state, "run_output_store", None)
    if store is None:
        return
    if variant_count > store.max_runs:
        raise HTTPException(
            status_code=400,
            detail=(f"options.record_outputs cannot be used for a sweep of "
                    f"{variant_count} variants: captured outputs are kept "
                    f"for only the newest {store.max_runs} runs, so the "
                    "earliest variants' outputs would be evicted before the "
                    "sweep finished. Submit the variants you want to "
                    "inspect as individual runs instead."))


@router.post("", status_code=201)
async def create_sweep(body: CreateSweepRequest, request: Request):
    """Compile a ``sweep_spec`` into N variant graphs and queue every one.

    Nothing is written until compilation has finished, and compilation is
    pure, so every 400 below leaves no rows at all. The order of the checks
    is the whole transaction story (spec 5.2).
    """
    service = _get_service(request)
    store = _get_sweep_store(request)
    spec = body.sweep_spec.model_dump()

    # Rules 1-3: cross-field refusals the option validator cannot see.
    # `options` may legitimately be None or a non-dict; normalize_options
    # below is what refuses the non-dict, with its own message.
    raw_options = body.options if isinstance(body.options, dict) else {}
    if "seed" in raw_options:
        raise HTTPException(
            status_code=400,
            detail="options.seed is owned by the sweep; set sweep_spec.seed "
                   "and seed_variants instead")
    # The STRIPPED value, not the raw one: normalize_options strips
    # `lane` (run_service.py:596), so a padded " interactive " would pass
    # a raw comparison, normalise INTO the interactive lane, and be caught
    # only by submit's own guard inside the loop -- a 500 plus a permanent
    # `failed` sweeps row, where this rule promises a 400 and no rows at
    # all. NOT lowercased, because normalize_options does not lowercase
    # either: lanes are case-sensitive labels and an unrecognised one
    # QUEUES rather than bypassing (run_service.py:242-244), so
    # "Interactive" is an ordinary queued lane and refusing it here would
    # refuse a legal sweep.
    raw_lane = raw_options.get("lane")
    if isinstance(raw_lane, str) and raw_lane.strip() == LANE_INTERACTIVE:
        raise HTTPException(
            status_code=400,
            detail="a sweep cannot use the interactive lane; interactive "
                   "submits are capped at RUN_INTERACTIVE_MAX_CONCURRENT "
                   "and refused past it, never queued")
    if body.seed_variants and spec.get("seed") is None:
        raise HTTPException(
            status_code=400,
            detail="sweep_spec.seed is required when seed_variants is true; "
                   "the per-variant execution seeds are derived from it")

    try:
        # Rule 4, then the envelope, then the compiler. The caps are read
        # from settings PER REQUEST, never captured into a module constant,
        # so CODEFYUI_MAX_SWEEP_RUNS actually reaches them.
        options = normalize_options(body.options)
        name = normalize_name(body.name)
        graph = normalize_graph(body.base_graph)
        compiled = compile_sweep(
            graph, spec,
            max_runs=settings.MAX_SWEEP_RUNS,
            max_params=settings.MAX_SWEEP_PARAMS,
            max_domain=settings.MAX_SWEEP_DOMAIN)
    except (RunSubmitError, SweepCompileError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _refuse_unrecordable_outputs(request, options, len(compiled.variants))

    # ONE capture for all N. create_run's default runs RunProvenance.capture
    # in a thread -- two git subprocesses plus a lockfile read in project
    # mode -- and thirty-two of those is thirty-two times the cost for one
    # answer. tests/test_run_store.py:123-127 skips it the same way.
    provenance = await asyncio.to_thread(RunProvenance.capture)

    # Minted HERE rather than left to create_sweep, so the settle-on-the-way-
    # out `finally` below knows the id even when this frame never sees the
    # record: Database.run finishes its worker before re-raising a
    # cancellation (db.py:225-243), so a cancel delivered at create_sweep's
    # own await leaves the row written and this coroutine unwinding past it.
    sweep_id = uuid4().hex
    submitted: list[dict[str, Any]] = []
    marked_failed = False
    try:
        # The sweep row goes FIRST: exec_runs.sweep_id has an enforced
        # foreign key, so inserting a child before its parent exists fails.
        #
        # The seed comes off the COMPILED sweep, not off `spec`:
        # CompiledSweep.seed is the planner seed the sampler actually used,
        # and reading it from there is what stops the stored seed and the
        # sampled variant list from ever drifting apart
        # (sweep_compiler.CompiledSweep.seed says so).
        sweep = await store.create_sweep(
            sweep_id=sweep_id,
            method=spec["method"], seed=compiled.seed,
            seed_variants=body.seed_variants,
            spec=_compiled_spec(spec, compiled),
            objective=body.objective.model_dump(),
            variants=[_placeholder_variant(compiled, variant)
                      for variant in compiled.variants],
            name=name)

        for variant in compiled.variants:
            variant_options = dict(options)
            seed: int | None = None
            if body.seed_variants:
                # The modulo is mandatory: normalize_options rejects a seed
                # above MAX_SEED, so a base seed near the top of the range
                # would make late variants fail submission with an error the
                # caller never wrote. `compiled.seed` is not None here --
                # rule 3 above refused that case, and the compiler records
                # the seed verbatim. When seed_variants is off the seed
                # stays None -- the unseeded case, which takes the SHARED
                # exclusion and lets variants overlap up to the device's
                # concurrency limit.
                seed = (compiled.seed + variant.index) % (MAX_SEED + 1)
                variant_options["seed"] = seed
            try:
                result = await service.submit(
                    variant.graph, options=variant_options, name=name,
                    provenance=provenance, sweep_id=sweep.id,
                    sweep_variant=variant.index)
                # Patched once per variant, not once at the end: a single
                # patch at the end would lose every id if the loop broke
                # half-way, which is precisely the case RULING 2 says must
                # stay visible.
                #
                # INSIDE the guard, not after it. The duty is that every
                # path out of this loop either fills in every run id or
                # marks the sweep failed, and a store fault here would
                # otherwise take a third path: a transient fault can fail
                # one write and allow the next, so it would exit with
                # neither -- leaving a `running` row that no harvest seam
                # can ever settle, because a run_id-None variant is not
                # terminal (spec 4.3).
                await store.set_variant_run(sweep.id, variant.index,
                                            run_id=result.run_id, seed=seed)
            except Exception as exc:
                # The children already created are LEFT ALONE. They are real
                # queued runs and shutdown/startup recovery retires them as
                # `interrupted` -- visible, not vanished, per RULING 2. The
                # sweep id goes in the body so nothing is unfindable, and
                # the row leaves `running`: a variant with run_id None is
                # not terminal, so without this the sweep would never
                # settle.
                await store.mark_failed(sweep.id, str(exc))
                marked_failed = True
                status = (503 if isinstance(exc, RunServiceUnavailable)
                          else 500)
                raise HTTPException(
                    status_code=status,
                    detail=(f"{exc}; sweep {sweep.id} was created with "
                            f"{len(submitted)} of {len(compiled.variants)} "
                            "variants and is marked failed")) from exc
            # Appended only once the patch landed, so `len(submitted)` in
            # the message above counts variants FULLY recorded on the row. A
            # child whose patch failed is still findable by
            # exec_runs.sweep_id.
            submitted.append({
                "index": variant.index, "domain_index": variant.domain_index,
                "run_id": result.run_id, "status": result.status,
                "seed": seed, "params": _variant_params(compiled, variant)})
    finally:
        # EVERY other way out of the block above. There is no `return`,
        # `break` or `continue` in it, so "every other way" means an
        # exception this handler did not raise itself -- and the `except
        # Exception` above cannot see the BaseException ones. The important
        # one is not exotic: a browser tab closing during a 32-variant
        # submit cancels the request task, which lands a CancelledError
        # between two submits.
        #
        # `failed` is the honest end state then, and it is the spec's own
        # words for it (4.3: "the submit loop broke part-way and some
        # variants have no run at all"). Without this the row keeps
        # `running` forever -- a variant holding a null run id is not
        # terminal, so no harvest seam can ever settle it -- while the
        # children already submitted keep running and stay inspectable,
        # which is RULING 2's promise.
        if not marked_failed and len(submitted) < len(compiled.variants):
            cause = sys.exc_info()[1]
            cause_name = "no error" if cause is None else type(cause).__name__
            await _mark_failed_on_the_way_out(
                store, sweep_id,
                f"the submit loop did not finish ({cause_name}); "
                f"{len(submitted)} of {len(compiled.variants)} variants "
                "were created")

    return {
        "sweep_id": sweep.id, "state": sweep.state, "method": sweep.method,
        "seed": sweep.seed, "seed_variants": sweep.seed_variants,
        "objective": sweep.objective,
        "total_combinations": compiled.total_combinations,
        "params": _spec_param_payload(sweep),
        "variants": submitted,
    }
