"""REST surface for durable studies (#343).

    POST   /api/studies                 submit (idempotent with a key) -> 201 / 200
    GET    /api/studies                 list, newest first
    GET    /api/studies/{id}            status, attempts, aggregates, rankings
    GET    /api/studies/{id}/events     audit log after a cursor, optional long poll
    POST   /api/studies/{id}/cancel     stop scheduling, cancel active children
    POST   /api/studies/{id}/resume     re-run what did not complete

Conventions are ``routes_sweeps.py``'s: plain ``dict`` returns, request
models with ``extra="forbid"``, every error a ``{"detail": ...}``, a 503 when
the coordinator is not on ``app.state``. Auth is ``auth_guard``'s house rule:
the POSTs need the session token, the GETs are open. Not under an
auth-exempt prefix.

The study itself lives in ``core.study_store`` and is driven by
``core.study_coordinator``; nothing here keeps state. Any client (or plugin)
can list the studies and re-attach to one by id: there is no affinity to
the tab that submitted it.
"""

from __future__ import annotations

import secrets
from typing import Any, Literal

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from ..config import settings
from ..core.run_service import (
    LANE_INTERACTIVE,
    MAX_SEED,
    RunServiceUnavailable,
    RunSubmitError,
    normalize_graph,
    normalize_name,
)
from ..core.run_store import RunRecord
from ..core.study_coordinator import StudyCoordinator, StudySpec
from ..core.study_store import (
    ATTEMPT_STATUSES,
    RESUME_EXPLICIT,
    STUDY_STATUSES,
    AttemptRecord,
    IdempotencyConflict,
    StudyRecord,
    StudyStateError,
    current_attempts,
    rank_study,
)
from ..core.sweep_compiler import SweepCompileError, compile_sweep

router = APIRouter(prefix="/api/studies", tags=["studies"])

#: Upper bound on ``repetitions``. A study is N x R child runs; the variant
#: count is bounded by ``MAX_SWEEP_RUNS`` like a sweep's.
MAX_STUDY_REPETITIONS = 32
#: Upper bound on ``budget.max_parallel_runs``.
MAX_STUDY_PARALLEL_RUNS = 32
MAX_LABEL_LENGTH = 64
MAX_IDEMPOTENCY_KEY_LENGTH = 200
IDEMPOTENCY_HEADER = "Idempotency-Key"


def _get_coordinator(request: Request) -> StudyCoordinator:
    coordinator = getattr(request.app.state, "study_coordinator", None)
    if coordinator is None:
        raise HTTPException(status_code=503,
                            detail="study coordinator not initialised")
    return coordinator


# ── request models ────────────────────────────────────────────────────────


class StudyObjectiveModel(BaseModel):
    """The EXACT objective: one producer node, one series, one direction.

    ``node_id`` is required, unlike a sweep's: a study's ranking feeds
    promotion decisions, and a name-only objective that several nodes log
    has no single answer. The value of a repetition is the last point that
    node logged under ``metric``.
    """

    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1, max_length=512)
    metric: str = Field(min_length=1, max_length=128)
    direction: Literal["minimize", "maximize"]

    @field_validator("node_id", "metric")
    @classmethod
    def _strip(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


class StudyOpModel(BaseModel):
    """One param override applied to ``base_graph``."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    param: str
    value: StrictBool | StrictInt | StrictFloat | StrictStr


class StudyVariantModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=MAX_LABEL_LENGTH)
    #: A whole graph, or ...
    graph: dict[str, Any] | None = None
    #: ... param overrides of the request's ``base_graph`` (``[]`` is the
    #: base graph itself). Exactly one of the two.
    ops: list[StudyOpModel] | None = None

    @field_validator("label")
    @classmethod
    def _strip_label(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("label must not be blank")
        return stripped

    @model_validator(mode="after")
    def _one_source(self) -> "StudyVariantModel":
        if (self.graph is None) == (self.ops is None):
            raise ValueError("a variant needs exactly one of 'graph' or "
                             "'ops'")
        return self


class StudyBudgetModel(BaseModel):
    """Server-enforced limits. ``max_runs`` counts every SUBMITTED attempt,
    retries included; ``max_parallel_runs`` caps the children in flight."""

    model_config = ConfigDict(extra="forbid")

    max_runs: int | None = Field(default=None, strict=True, ge=1)
    max_parallel_runs: int = Field(default=1, strict=True, ge=1,
                                   le=MAX_STUDY_PARALLEL_RUNS)


class CreateStudyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    objective: StudyObjectiveModel
    base_graph: dict[str, Any] | None = None
    variants: list[StudyVariantModel] = Field(min_length=1)
    repetitions: int = Field(strict=True, ge=1, le=MAX_STUDY_REPETITIONS)
    #: The study seed; repetition r runs with ``(seed + r) mod 2^32``.
    #: Omitted -> the server picks one and records it.
    seed: int | None = Field(default=None, strict=True, ge=0, le=MAX_SEED)
    #: Run options copied into every child. ``seed`` is owned by the study
    #: and the interactive lane is refused, as for sweeps.
    options: dict[str, Any] | None = None
    budget: StudyBudgetModel = Field(default_factory=StudyBudgetModel)
    resume_policy: Literal["explicit", "auto"] = RESUME_EXPLICIT
    idempotency_key: str | None = Field(
        default=None, min_length=1, max_length=MAX_IDEMPOTENCY_KEY_LENGTH)


# ── submission helpers ────────────────────────────────────────────────────


def _resolve_variant(index: int, variant: StudyVariantModel,
                     base: dict[str, Any] | None) -> dict[str, Any]:
    """The variant's complete graph. ``ops`` go through the sweep compiler,
    so an address is refused for exactly the reasons a sweep refuses it
    (unknown node, preset or subgraph instance, SECRET param, type or bound
    violation)."""
    if variant.graph is not None:
        return normalize_graph(variant.graph)
    if base is None:
        raise HTTPException(
            status_code=400,
            detail=f"variants[{index}] uses 'ops' but the request has no "
                   "base_graph")
    ops = variant.ops or []
    if not ops:
        return base
    spec = {"method": "grid", "params": [
        {"node_id": op.node_id, "param": op.param, "values": [op.value]}
        for op in ops]}
    try:
        compiled = compile_sweep(base, spec, max_runs=1, max_params=len(ops),
                                 max_domain=1)
    except SweepCompileError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc).replace("sweep_spec.params[",
                                    f"variants[{index}].ops[")) from exc
    return compiled.variants[0].graph


def _build_spec(body: CreateStudyRequest,
                idempotency_key: str | None) -> StudySpec:
    if len(body.variants) > settings.MAX_SWEEP_RUNS:
        raise HTTPException(
            status_code=400,
            detail=f"a study has at most {settings.MAX_SWEEP_RUNS} variants "
                   "(MAX_SWEEP_RUNS)")
    labels = [variant.label for variant in body.variants]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise HTTPException(
            status_code=400,
            detail=f"variant labels must be unique; repeated: {duplicates}")
    total = len(body.variants) * body.repetitions
    if body.budget.max_runs is not None and body.budget.max_runs < total:
        raise HTTPException(
            status_code=400,
            detail=f"budget.max_runs ({body.budget.max_runs}) is below the "
                   f"{total} runs the study schedules "
                   f"({len(body.variants)} variants x {body.repetitions} "
                   "repetitions)")
    raw_options = dict(body.options or {})
    if "seed" in raw_options:
        raise HTTPException(
            status_code=400,
            detail="options.seed is owned by the study; set 'seed' and each "
                   "repetition gets (seed + repetition)")
    lane = raw_options.get("lane")
    if isinstance(lane, str) and lane.strip() == LANE_INTERACTIVE:
        raise HTTPException(
            status_code=400,
            detail="a study cannot use the interactive lane; its children "
                   "are queued")
    try:
        # `options` itself is validated by the placement check in
        # create_study (once per distinct settings.device), after the
        # idempotency lookup.
        name = normalize_name(body.name)
        base = (None if body.base_graph is None
                else normalize_graph(body.base_graph))
        graphs = [_resolve_variant(index, variant, base)
                  for index, variant in enumerate(body.variants)]
    except RunSubmitError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    seed = body.seed if body.seed is not None \
        else secrets.randbelow(MAX_SEED + 1)
    identity = body.model_dump(exclude={"idempotency_key", "base_graph"})
    for entry in identity["variants"]:
        entry.pop("graph", None)
    return StudySpec(
        name=name,
        objective=body.objective.model_dump(),
        repetitions=body.repetitions,
        seed=seed,
        options=raw_options,
        budget=body.budget.model_dump(),
        resume_policy=body.resume_policy,
        idempotency_key=idempotency_key,
        variants=[(variant.label,
                   None if variant.ops is None
                   else [op.model_dump() for op in variant.ops],
                   graph)
                  for variant, graph in zip(body.variants, graphs)],
        identity=identity,
    )


# ── payloads ──────────────────────────────────────────────────────────────


def _study_summary(study: StudyRecord) -> dict[str, Any]:
    return {
        "study_id": study.id, "name": study.name, "status": study.status,
        "revision": study.revision, "objective": study.objective,
        "repetitions": study.repetitions, "seed": study.seed,
        "budget": study.budget, "resume_policy": study.resume_policy,
        "idempotency_key": study.idempotency_key, "error": study.error,
        "created_at": study.created_at, "updated_at": study.updated_at,
        "finished_at": study.finished_at,
    }


def _counts(current: dict[tuple[int, int], AttemptRecord]) -> dict[str, int]:
    counts = {status: 0 for status in sorted(ATTEMPT_STATUSES)}
    for attempt in current.values():
        counts[attempt.status] = counts.get(attempt.status, 0) + 1
    return counts


def _attempt_payload(attempt: AttemptRecord,
                     runs: dict[str, RunRecord]) -> dict[str, Any]:
    """An attempt, plus where its child run is placed (#777) while that
    run exists: ``worker_id``, ``devices`` and ``queue_key`` off the run
    row, as a sweep variant reports them (``any`` / ``any/<device>`` until
    a ``target: any`` child starts). Omitted when the run is gone."""
    payload = attempt.as_json()
    run = runs.get(attempt.id)
    if run is not None and run.id == attempt.run_id:
        payload["worker_id"] = run.worker_id
        payload["devices"] = run.devices
        payload["queue_key"] = run.queue_key
    return payload


async def _study_detail(coordinator: StudyCoordinator,
                        study: StudyRecord) -> dict[str, Any]:
    store = coordinator.store
    variants = await store.list_variants(study.id)
    attempts = await store.list_attempts(study.id)
    current = current_attempts(attempts)
    # ONE indexed read for every child, keyed by the attempt stamp.
    runs = await coordinator.service.store.find_runs_by_study_attempt(
        [a.id for a in attempts if a.run_id is not None])
    ranking = rank_study(len(variants), study.repetitions, attempts,
                         direction=study.objective["direction"])
    by_variant: dict[int, list[AttemptRecord]] = {}
    for attempt in attempts:
        by_variant.setdefault(attempt.variant_index, []).append(attempt)
    variant_rows = []
    for variant, row in zip(variants, ranking):
        variant_rows.append({
            "index": variant.index, "label": variant.label,
            "ops": variant.ops,
            "repetitions": [
                {"repetition": repetition,
                 "seed": (current[(variant.index, repetition)].seed
                          if (variant.index, repetition) in current
                          else None),
                 "current": (_attempt_payload(
                     current[(variant.index, repetition)], runs)
                     if (variant.index, repetition) in current else None)}
                for repetition in range(study.repetitions)],
            "attempts": [_attempt_payload(a, runs)
                         for a in by_variant.get(variant.index, [])],
            "aggregate": {"count": row["count"], "mean": row["mean"],
                          "std": row["std"], "values": row["values"]},
            "eligible": row["eligible"],
            "ineligible_reasons": row["ineligible_reasons"],
            "rank": row["rank"], "tie": row["tie"],
            "tied_with": row["tied_with"],
        })
    ranked = sorted((row for row in variant_rows if row["rank"] is not None),
                    key=lambda row: (row["rank"], row["index"]))
    return {
        **_study_summary(study),
        "options": study.options,
        "counts": _counts(current),
        "attempts_submitted": sum(1 for a in attempts
                                  if a.status != "pending"),
        "variants": variant_rows,
        "rankings": [{"rank": row["rank"], "variant_index": row["index"],
                      "label": row["label"], "mean": row["aggregate"]["mean"],
                      "std": row["aggregate"]["std"],
                      "count": row["aggregate"]["count"], "tie": row["tie"],
                      "tied_with": row["tied_with"]} for row in ranked],
        "last_cursor": await store.latest_cursor(study.id),
    }


def _not_found(study_id: str) -> HTTPException:
    return HTTPException(status_code=404,
                         detail=f"study '{study_id}' not found")


# ── routes ────────────────────────────────────────────────────────────────


@router.post("", status_code=201)
async def create_study(
    body: CreateStudyRequest, request: Request, response: Response,
    idempotency_header: str | None = Header(default=None,
                                            alias=IDEMPOTENCY_HEADER),
):
    """Submit a study: N variants x R repetitions of queued child runs.

    With an idempotency key (the ``Idempotency-Key`` header, or the body's
    ``idempotency_key``), a retry of the same request returns the SAME study
    with 200 and creates nothing; the key reused for a different request is
    a 409. Every 400 leaves no rows: the study, its variants and all N x R
    attempts are written in one transaction, after validation.
    """
    coordinator = _get_coordinator(request)
    header_key = None if idempotency_header is None \
        else idempotency_header.strip() or None
    if header_key is not None \
            and len(header_key) > MAX_IDEMPOTENCY_KEY_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"{IDEMPOTENCY_HEADER} is at most "
                   f"{MAX_IDEMPOTENCY_KEY_LENGTH} characters")
    if header_key is not None and body.idempotency_key is not None \
            and header_key != body.idempotency_key:
        raise HTTPException(
            status_code=400,
            detail=f"the {IDEMPOTENCY_HEADER} header and the body's "
                   "idempotency_key differ; send one, or the same value")
    key = header_key or body.idempotency_key
    spec = _build_spec(body, key)
    try:
        study, created = await coordinator.create_study(spec)
    except IdempotencyConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RunSubmitError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RunServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    payload = await _study_detail(coordinator, study)
    payload["created"] = created
    if not created:
        response.status_code = 200
    return payload


@router.get("")
async def list_studies(
    request: Request,
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """Every study, newest first. ``status`` takes one status or several
    separated by ``|`` (``queued|running|cancel_requested`` finds the ones
    still in flight)."""
    coordinator = _get_coordinator(request)
    wanted = None if status is None else [
        part.strip() for part in status.split("|") if part.strip()]
    if wanted is not None:
        unknown = sorted(set(wanted) - STUDY_STATUSES)
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"unknown study status {unknown}; expected one of "
                       f"{sorted(STUDY_STATUSES)}")
    studies, total = await coordinator.store.list_studies(
        status=wanted, limit=limit, offset=offset)
    counts = await coordinator.store.attempt_counts([s.id for s in studies])
    return {
        "studies": [{**_study_summary(study),
                     "counts": counts.get(study.id, {})}
                    for study in studies],
        "total": total, "limit": limit, "offset": offset,
    }


@router.get("/{study_id}")
async def get_study(study_id: str, request: Request):
    """Status, every attempt with its run id and evidence, per-variant
    aggregates and the rankings. Reconciles the study first (read repair),
    so a child that finished while nobody was watching is reflected."""
    coordinator = _get_coordinator(request)
    study = await coordinator.tick(study_id)
    if study is None:
        raise _not_found(study_id)
    return await _study_detail(coordinator, study)


@router.get("/{study_id}/events")
async def get_study_events(
    study_id: str, request: Request,
    cursor: int = Query(default=0, ge=0),
    wait: float = Query(default=0.0, ge=0.0, le=60.0),
    limit: int = Query(default=500, ge=1, le=2000),
):
    """The audit log strictly after *cursor*, oldest first. ``wait`` long
    polls while the study is active and the tail is empty; the returned
    ``cursor`` is where to resume and never moves backwards."""
    coordinator = _get_coordinator(request)
    study = await coordinator.store.get_study(study_id)
    if study is None:
        raise _not_found(study_id)
    events = await coordinator.wait_for_events(
        study_id, after_cursor=cursor, limit=limit, wait=wait)
    if wait > 0:
        study = await coordinator.store.get_study(study_id) or study
    page = [{"cursor": e.cursor, "type": e.type, "payload": e.payload,
             "ts": e.ts} for e in events]
    return {"study_id": study_id, "status": study.status,
            "revision": study.revision, "events": page,
            "cursor": page[-1]["cursor"] if page else cursor}


@router.post("/{study_id}/cancel")
async def cancel_study(study_id: str, request: Request):
    """Stop scheduling and cancel the active children. Completed attempts
    are kept; a study reads ``cancel_requested`` until every child it asked
    to stop has stopped, then ``cancelled``. Cancelling a cancelled study is
    a 200 no-op; a succeeded one is a 409."""
    coordinator = _get_coordinator(request)
    try:
        study, delivered = await coordinator.cancel(study_id)
    except KeyError as exc:
        raise _not_found(study_id) from exc
    except StudyStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {**await _study_detail(coordinator, study),
            "cancelled_runs": delivered}


@router.post("/{study_id}/resume")
async def resume_study(study_id: str, request: Request):
    """Continue an interrupted, cancelled or failed study. Completed
    repetitions are skipped; a cancelled, interrupted or lost one gets a new
    attempt. Any other status is a 409."""
    coordinator = _get_coordinator(request)
    try:
        study = await coordinator.resume(study_id)
    except KeyError as exc:
        raise _not_found(study_id) from exc
    except StudyStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return await _study_detail(coordinator, study)
