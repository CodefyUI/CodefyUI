"""Plugin-scoped durable jobs (#343 increment 6): ``/api/plugins/{id}/jobs``.

    GET    /api/plugins/{plugin_id}/jobs/kinds              what may be submitted
    POST   /api/plugins/{plugin_id}/jobs                    submit (key required) -> 202 / 200
    GET    /api/plugins/{plugin_id}/jobs                    this plugin's jobs, newest first
    GET    /api/plugins/{plugin_id}/jobs/{job_id}           one job
    GET    /api/plugins/{plugin_id}/jobs/{job_id}/events    audit log after ``after``
    POST   /api/plugins/{plugin_id}/jobs/{job_id}/cancel    stop it
    POST   /api/plugins/{plugin_id}/jobs/{job_id}/resume    continue it
    GET    /api/plugins/{plugin_id}/jobs/{job_id}/variants/{index}/graph

A job of kind ``study`` IS a study (``core.plugin_jobs``): the routes map
one-to-one onto ``/api/studies`` and share its helpers, with three
differences the contract asks for. The plugin is the study's ``owner``, so a
plugin sees and controls only its own jobs (another plugin's job id is a
404); an idempotency key is REQUIRED and scoped to the plugin; a submit
answers ``202 Accepted`` (``200`` for a retry). Every job also appears in
``GET /api/studies`` and in the web app's Studies panel, with its owner.

Auth is ``auth_guard``'s house rule, as for studies: the POSTs need the
session token (a plugin's ``api.http.fetch`` adds it), the GETs are open.
The plugin must be installed and enabled to submit or resume.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Header, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..core.plugin_jobs import (
    JOBS_API_VERSION,
    KIND_STUDY,
    PluginDisabled,
    PluginNotFound,
    get_job_kind,
    job_kinds,
    require_enabled_plugin,
)
from ..core.run_service import RunServiceUnavailable, RunSubmitError
from ..core.study_store import (
    STUDY_STATUSES,
    IdempotencyConflict,
    StudyRecord,
    StudyStateError,
)
from .routes_studies import (
    IDEMPOTENCY_HEADER,
    MAX_IDEMPOTENCY_KEY_LENGTH,
    CancelStudyRequest,
    CreateStudyRequest,
    ResumeStudyRequest,
    _build_spec,
    _get_coordinator,
    _study_detail,
    _study_summary,
)

router = APIRouter(prefix="/api/plugins", tags=["plugin-jobs"])


class CreatePluginJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str = Field(min_length=1, max_length=64)
    spec_version: int = Field(strict=True, ge=1)
    #: The kind's spec. For ``study``: a ``POST /api/studies`` body.
    spec: dict[str, Any]
    idempotency_key: str | None = Field(
        default=None, min_length=1, max_length=MAX_IDEMPOTENCY_KEY_LENGTH)


def _job_payload(plugin_id: str, detail: dict[str, Any]) -> dict[str, Any]:
    return {"job_id": detail["study_id"], "plugin_id": plugin_id,
            "kind": KIND_STUDY, "spec_version": 1,
            "api_version": JOBS_API_VERSION, **detail}


def _plugin_error(exc: Exception) -> HTTPException:
    if isinstance(exc, PluginNotFound):
        return HTTPException(status_code=404,
                             detail=f"plugin '{exc}' is not installed")
    return HTTPException(status_code=409,
                         detail=f"plugin '{exc}' is disabled; enable it to "
                                "start or resume its jobs")


async def _owned(request: Request, plugin_id: str,
                 job_id: str) -> StudyRecord:
    """The job, when it exists AND belongs to *plugin_id*; else 404 (the
    same answer for both, so a plugin cannot probe another's job ids)."""
    coordinator = _get_coordinator(request)
    study = await coordinator.store.get_study(job_id)
    if study is None or study.owner != plugin_id:
        raise HTTPException(
            status_code=404,
            detail=f"job '{job_id}' not found for plugin '{plugin_id}'")
    return study


@router.get("/{plugin_id}/jobs/kinds")
async def list_job_kinds(plugin_id: str):
    """The job kinds this server runs, for a plugin to feature-check
    (an older server answers 404 here, and the plugin keeps its own
    in-page engine)."""
    return {"plugin_id": plugin_id, "api_version": JOBS_API_VERSION,
            "kinds": [kind.as_json() for kind in job_kinds()]}


@router.post("/{plugin_id}/jobs", status_code=202)
async def create_job(
    plugin_id: str, body: CreatePluginJobRequest, request: Request,
    response: Response,
    idempotency_header: str | None = Header(default=None,
                                            alias=IDEMPOTENCY_HEADER),
):
    """Submit a job. The ``Idempotency-Key`` header (or ``idempotency_key``
    in the body or in the spec) is required: a retry with it returns the
    same job with 200 and creates nothing, the key reused for another
    request is a 409. ``kind`` must be registered and ``spec_version`` the
    kind's (400 otherwise)."""
    coordinator = _get_coordinator(request)
    try:
        require_enabled_plugin(plugin_id)
    except (PluginNotFound, PluginDisabled) as exc:
        raise _plugin_error(exc) from exc
    kind = get_job_kind(body.kind)
    if kind is None:
        raise HTTPException(
            status_code=400,
            detail=f"unknown job kind {body.kind!r}; this server runs "
                   f"{[k.kind for k in job_kinds()]}")
    if body.spec_version != kind.spec_version:
        raise HTTPException(
            status_code=400,
            detail=f"job kind {kind.kind!r} takes spec_version "
                   f"{kind.spec_version}, not {body.spec_version}")
    try:
        spec = CreateStudyRequest.model_validate(body.spec)
    except ValidationError as exc:
        raise RequestValidationError([
            {**error, "loc": ("body", "spec", *error["loc"])}
            for error in exc.errors(include_url=False, include_context=False)
        ]) from exc
    header_key = None if idempotency_header is None \
        else idempotency_header.strip() or None
    keys = {key for key in (header_key, body.idempotency_key,
                            spec.idempotency_key) if key is not None}
    if len(keys) > 1:
        raise HTTPException(
            status_code=400,
            detail="the idempotency keys of the header, the body and the "
                   "spec differ; send one, or the same value")
    if not keys:
        raise HTTPException(
            status_code=400,
            detail=f"a plugin job needs an {IDEMPOTENCY_HEADER} header (or "
                   "idempotency_key), so a retry never starts it twice")
    key = keys.pop()
    if len(key) > MAX_IDEMPOTENCY_KEY_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"{IDEMPOTENCY_HEADER} is at most "
                   f"{MAX_IDEMPOTENCY_KEY_LENGTH} characters")
    study_spec = _build_spec(spec, key, owner=plugin_id)
    try:
        study, created = await coordinator.create_study(study_spec)
    except IdempotencyConflict as exc:
        raise HTTPException(
            status_code=409,
            detail=f"idempotency key {key!r} was already used for job "
                   f"{exc.study_id} with a different request; use a new "
                   "key for a new job") from exc
    except RunSubmitError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RunServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    payload = _job_payload(plugin_id, await _study_detail(coordinator, study))
    payload["created"] = created
    if not created:
        response.status_code = 200
    return payload


@router.get("/{plugin_id}/jobs")
async def list_jobs(
    plugin_id: str, request: Request,
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """This plugin's jobs, newest first; ``status`` as for studies."""
    coordinator = _get_coordinator(request)
    wanted = None if status is None else [
        part.strip() for part in status.split("|") if part.strip()]
    if wanted is not None:
        unknown = sorted(set(wanted) - STUDY_STATUSES)
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"unknown job status {unknown}; expected one of "
                       f"{sorted(STUDY_STATUSES)}")
    studies, total = await coordinator.store.list_studies(
        status=wanted, limit=limit, offset=offset, owner=plugin_id)
    counts = await coordinator.store.attempt_counts([s.id for s in studies])
    return {
        "plugin_id": plugin_id,
        "jobs": [{"job_id": study.id, "kind": KIND_STUDY,
                  **_study_summary(study),
                  "counts": counts.get(study.id, {})} for study in studies],
        "total": total, "limit": limit, "offset": offset,
    }


@router.get("/{plugin_id}/jobs/{job_id}")
async def get_job(plugin_id: str, job_id: str, request: Request):
    coordinator = _get_coordinator(request)
    await _owned(request, plugin_id, job_id)
    study = await coordinator.tick(job_id)
    assert study is not None
    return _job_payload(plugin_id, await _study_detail(coordinator, study))


@router.get("/{plugin_id}/jobs/{job_id}/events")
async def get_job_events(
    plugin_id: str, job_id: str, request: Request,
    after: int | None = Query(default=None, ge=0),
    cursor: int = Query(default=0, ge=0),
    wait: float = Query(default=0.0, ge=0.0, le=60.0),
    limit: int = Query(default=500, ge=1, le=2000),
):
    """The job's events strictly after ``after`` (the contract's name;
    ``cursor`` is accepted too, as on ``/api/studies``)."""
    coordinator = _get_coordinator(request)
    study = await _owned(request, plugin_id, job_id)
    position = cursor if after is None else after
    events = await coordinator.wait_for_events(
        job_id, after_cursor=position, limit=limit, wait=wait)
    if wait > 0:
        study = await coordinator.store.get_study(job_id) or study
    page = [{"cursor": e.cursor, "sequence": e.cursor, "type": e.type,
             "payload": e.payload, "ts": e.ts} for e in events]
    return {"job_id": job_id, "plugin_id": plugin_id, "status": study.status,
            "revision": study.revision, "events": page,
            "cursor": page[-1]["cursor"] if page else position}


@router.post("/{plugin_id}/jobs/{job_id}/cancel")
async def cancel_job(plugin_id: str, job_id: str, request: Request,
                     body: CancelStudyRequest | None = Body(default=None)):
    coordinator = _get_coordinator(request)
    await _owned(request, plugin_id, job_id)
    try:
        study, delivered = await coordinator.cancel(
            job_id, expected_revision=(
                None if body is None else body.expected_revision))
    except StudyStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {**_job_payload(plugin_id, await _study_detail(coordinator, study)),
            "cancelled_runs": delivered}


@router.post("/{plugin_id}/jobs/{job_id}/resume")
async def resume_job(plugin_id: str, job_id: str, request: Request,
                     body: ResumeStudyRequest | None = Body(default=None)):
    coordinator = _get_coordinator(request)
    await _owned(request, plugin_id, job_id)
    try:
        require_enabled_plugin(plugin_id)
    except (PluginNotFound, PluginDisabled) as exc:
        raise _plugin_error(exc) from exc
    try:
        study = await coordinator.resume(
            job_id,
            budget=(None if body is None or body.budget is None
                    else body.budget.stored()),
            expected_revision=(None if body is None
                               else body.expected_revision))
    except StudyStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _job_payload(plugin_id, await _study_detail(coordinator, study))


@router.get("/{plugin_id}/jobs/{job_id}/variants/{index}/graph")
async def get_job_variant_graph(plugin_id: str, job_id: str, index: int,
                                request: Request):
    """The graph one variant ran (``SECRET`` values blank), for the plugin
    to open as a tab after the user picks a winner."""
    coordinator = _get_coordinator(request)
    await _owned(request, plugin_id, job_id)
    graph = await coordinator.store.get_variant_graph(job_id, index)
    if graph is None:
        raise HTTPException(status_code=404,
                            detail=f"job '{job_id}' has no variant {index}")
    variants = await coordinator.store.list_variants(job_id)
    label = next((v.label for v in variants if v.index == index), None)
    return {"job_id": job_id, "variant_index": index, "label": label,
            "graph": graph}
