"""REST surface for the Package Center (optional packs).

    GET    /api/packs                          the catalog + what is installed
    POST   /api/packs/{id}/install             start a job -> 202 {job_id}
    POST   /api/packs/jobs/{job_id}/cancel     cooperative stop
    GET    /api/packs/jobs/{job_id}/events     replay after a cursor, long poll
    DELETE /api/packs/{id}/items/{item_id}     delete one downloaded model

Auth follows the house rule in ``main.py`` exactly: ``auth_guard`` requires the
session token for the mutating routes, and the GET is open like every other
read in the app -- it is what the editor polls to draw the panel. This router
is deliberately NOT in ``_AUTH_EXEMPT_PREFIXES``; the exemption exists for
``/api/apps`` and ``/api/keys``, which carry per-route dependencies instead,
and adding one here would silently drop authentication from every install.

On top of that, every MUTATING route is refused unless the server is bound to
loopback. Starting a package install is the most privileged thing this app can
do -- it runs a package manager against the interpreter serving the request --
and "whoever can reach the port" is the wrong audience for it. A classroom or
office install that deliberately serves the LAN opts back in with
``CODEFYUI_ALLOW_REMOTE_PACK_INSTALL=1``.

What the client may ask for is bounded by the CATALOG: a pack id, and the ids
of items inside it. No pip spec, no repo id and no URL from a request body
ever reaches a subprocess -- see ``core/packs/catalog.py``, which is the whole
attack surface of the feature.

There is no route for removing a pack's PACKAGES, and no CLI command either:
uninstalling pip packages from inside the process that imported them is how
you get a half-loaded interpreter serving requests. ``cdui packs remove
<pack_id> <item_id>`` deletes one downloaded MODEL and, for the packages,
PRINTS the ``uv pip uninstall --python <venv python> ...`` line to run by
hand, from a terminal, with the server stopped.
"""

from __future__ import annotations

import asyncio
import os
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, field_validator

from ..config import settings
from ..core.packs import catalog, flows, restart, state
from ..core.packs.catalog import ModelItem, Pack
from ..core.packs.errors import (
    PackInstallError,
    PackInsufficientDisk,
    RestartRefused,
)
from ..core.packs.service import (
    PackBusy,
    PackJob,
    PackService,
    RestartUnavailable,
    UnknownJob,
)
from ..core.packs.state import ItemState, PackState

router = APIRouter(prefix="/api/packs", tags=["packs"])

#: Bind addresses that mean "this machine only". ``0.0.0.0`` and ``::`` are
#: deliberately absent: they are wildcards, reachable from the LAN, and a
#: server on one of them is exactly the case this gate exists for.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

_REMOTE_REFUSAL = (
    "Installing packs is only allowed from the computer that runs the "
    "server. Set CODEFYUI_ALLOW_REMOTE_PACK_INSTALL=1 to override.")

#: What ``cdui start`` / ``cdui dev`` put in the environment, so the panel can
#: tell a supervised server (which a later release can restart for the user)
#: from a bare ``uvicorn app.main:app``.
_LAUNCH_MODES = frozenset({"start", "dev"})


class InstallRequest(BaseModel):
    """The whole install request body. ``extra="forbid"`` is load-bearing:
    it is what makes "the client cannot smuggle in a package spec" a
    guarantee of the schema rather than of every handler remembering."""

    model_config = ConfigDict(extra="forbid")

    #: Which items to fetch. Omitted means "the whole pack, minus what is
    #: already downloaded".
    items: list[str] | None = None
    mode: Literal["live", "restart"] | None = None
    #: Only meaningful for the GPU pack: which torch wheel to install.
    variant: str | None = None

    @field_validator("variant")
    @classmethod
    def _known_variant(cls, value: str | None) -> str | None:
        """Refuse a wheel name that is not one of ours.

        This value comes back to the user as part of a command line to paste
        into a shell (``cdui install --gpu <variant>``), and a later release
        hands it to the installer. Neither is a place for a free-form string
        from a request body, so the allowlist is enforced by the SCHEMA --
        anything else is a 422 before a line of route code runs.
        """
        if value is not None and value not in restart.VARIANTS:
            raise ValueError(
                f"unknown torch variant {value!r}; expected one of "
                f"{', '.join(restart.VARIANTS)}")
        return value


def remote_install_allowed() -> bool:
    """May a request start an install at all, given how the server is bound?"""
    host = settings.HOST.strip().strip("[]").lower()
    return host in _LOOPBACK_HOSTS or bool(settings.ALLOW_REMOTE_PACK_INSTALL)


def _require_local_install() -> None:
    """Dependency for every mutating route. See the module docstring."""
    if not remote_install_allowed():
        raise HTTPException(status_code=403, detail=_REMOTE_REFUSAL)


def _service(request: Request) -> PackService:
    """The service the lifespan built, or 503.

    Optional on ``app.state`` for the same reason every other store is: the
    lifespan does not run under httpx's ASGITransport, so a test reaches
    these routes with nothing there unless it installs one.
    """
    service = getattr(request.app.state, "pack_service", None)
    if service is None:
        raise HTTPException(status_code=503,
                            detail="Package Center is not available")
    return service


def _require_pack(pack_id: str) -> Pack:
    pack = catalog.find_pack(pack_id)
    if pack is None:
        raise HTTPException(status_code=404, detail=f"unknown pack '{pack_id}'")
    return pack


def _active_job(service: PackService) -> PackJob | None:
    """The job that is running right now, if any. A FINISHED job is kept for
    its events, but it is not what the panel calls active."""
    job = service.current_job()
    return job if job is not None and not job.terminal else None


# ── GET /api/packs ────────────────────────────────────────────────────────


def _item_status(item: ItemState, active: PackJob | None) -> str:
    if active is not None and active.current_step == f"download:{item.item_id}":
        return "downloading"
    return "present" if item.present else "missing"


def _pack_status(pack: Pack, probed: PackState, active: PackJob | None) -> str:
    """What the panel's pill says for one pack.

    ``partial`` means "some progress, but not all of it" -- and progress is
    NOT ``pip_ready`` on its own. A pack with nothing to probe (``word-vectors``
    ships data and no packages) is vacuously pip-ready on every machine, so
    reading that as progress would show a permanent "partial" for a pack
    nobody has downloaded anything for.
    """
    if active is not None:
        return "installing"
    if probed.installed:
        return "installed"
    progressed = ((bool(pack.probe_modules) and probed.pip_ready)
                  or any(item.present for item in probed.items))
    return "partial" if progressed else "not_installed"


def _install_command(pack: Pack, gpu: dict) -> str:
    """The terminal command shown next to one pack.

    A restart-mode pack is a wheel swap the installer owns rather than
    something ``cdui packs install`` can do, so its command names a torch
    variant -- and that variant is already in the ``gpu`` block this request
    computed. Reading it from there keeps ``list_packs`` to exactly ONE GPU
    probe, which is what lets that probe be moved off the event loop:
    calling ``restart.install_command_for`` for the GPU pack would run a
    second one right here, on the loop thread.
    """
    if pack.install_mode == "restart":
        return gpu["install_command"]
    return restart.install_command_for(pack)


def _item_payload(item: ModelItem, probed: ItemState,
                  active: PackJob | None) -> dict:
    # Both `repo_id` and `url` are always present, one of them null: the
    # frontend types one item shape, not one per kind.
    return {
        "id": item.item_id,
        "kind": item.kind,
        "repo_id": item.repo_id,
        "url": item.url,
        "size_bytes": item.approx_bytes,
        # What the install writes BESIDE the download (the GloVe npz), as
        # its own field and deliberately NOT folded into `size_bytes` or
        # `size_bytes_total`: those two answer "what comes down the wire",
        # which is what a progress bar and a download prompt are about. A UI
        # that wants to say "and N MB more on disk" has the number here.
        "derived_bytes": item.derived_bytes,
        "license": item.license,
        "status": _item_status(probed, active),
    }


def _pack_payload(pack: Pack, probed: PackState, active: PackJob | None,
                  gpu: dict) -> dict:
    mine = active if active is not None and active.pack_id == pack.pack_id else None
    items = [
        _item_payload(item, item_probe, mine)
        for item, item_probe in zip(pack.items, probed.items)
    ]
    return {
        "id": pack.pack_id,
        "title": pack.title,
        "description": pack.description,
        "install_mode": pack.install_mode,
        "status": _pack_status(pack, probed, mine),
        "pip_ready": probed.pip_ready,
        "usable": probed.usable,
        "depends_on": list(pack.depends_on),
        "blocked_by": list(probed.blocked_by),
        "pip": [{"spec": spec} for spec in pack.pip],
        "items": items,
        # What is still to fetch, so the UI can say "470 MB" before the
        # click. Already-downloaded items contribute nothing, which makes
        # this 0 for an installed pack without a special case.
        "size_bytes_total": sum(
            item.approx_bytes
            for item, item_probe in zip(pack.items, probed.items)
            if not item_probe.present),
        "install_command": _install_command(pack, gpu),
    }


@router.get("")
async def list_packs(request: Request) -> dict:
    """Every pack, what is installed, and what this machine can install.

    The one route the panel polls. Deliberately cheap: ``state.probe_all``
    caches until something changes it, and the GPU facts are detected once
    per process (see ``restart.gpu_info``), so polling this while a download
    runs costs a dict walk.

    That first detection, though, is a ``shutil.which`` and an ``nvidia-smi``
    subprocess with a five second timeout, so it goes to a thread. On the
    loop it would stall every other request -- including the install long
    poll -- for as long as a wedged driver takes to answer.
    """
    service = _service(request)
    active = _active_job(service)
    probed = state.probe_all()
    launch_mode = os.environ.get("CODEFYUI_MANAGED")
    gpu = await asyncio.to_thread(restart.gpu_info)

    return {
        "packs": [_pack_payload(pack, probed[pack.pack_id], active, gpu)
                  for pack in catalog.iter_packs()],
        "active_job": (None if active is None
                       else {"job_id": active.job_id, "pack_id": active.pack_id}),
        "last_restart_job": restart.read_last_restart(),
        "remote_install_allowed": remote_install_allowed(),
        "launch_mode": launch_mode if launch_mode in _LAUNCH_MODES else "unknown",
        # Whether the panel may offer "install and restart" at all, rather
        # than only the command to paste. Narrower than ``launch_mode``: it
        # also wants the launcher still on disk and the kill switch off, and
        # it stays on the loop deliberately -- it is one ``stat`` of a local
        # path, not the subprocess ``gpu_info`` above needs a thread for.
        "restart_available": restart.restart_available(),
        "gpu": gpu,
    }


# ── POST /api/packs/{pack_id}/install ─────────────────────────────────────


@router.post("/{pack_id}/install", status_code=202,
             dependencies=[Depends(_require_local_install)])
async def install_pack(pack_id: str, request: Request,
                       body: InstallRequest | None = None):
    """Start installing a pack. 202 and a ``job_id`` to follow.

    Everything that can be known before the job starts is refused here rather
    than reported as a failed job thirty seconds later: an unknown item, a
    dependency that is not ready, an install already running, a disk that
    cannot hold the download, and a restart-mode request this server cannot
    honour or cannot honour yet. A 202 means the job exists.

    For a RESTART-mode install the 202 is the last thing this process says:
    the job it names is already ``needs_restart``, the helper that will
    finish the install is running, and the server stops itself half a second
    later. The client follows the job's events exactly as it would a live
    one -- it just reaches the terminal event immediately, and then waits for
    the server to come back.
    """
    pack = _require_pack(pack_id)
    service = _service(request)
    payload = body or InstallRequest()

    if payload.items is not None:
        known = {item.item_id for item in pack.items}
        unknown = [item_id for item_id in payload.items if item_id not in known]
        if unknown:
            return JSONResponse(
                status_code=400,
                content={"detail": f"pack '{pack_id}' has no item "
                                   f"'{unknown[0]}'"})

    blocked = list(state.probe_all()[pack_id].blocked_by)
    if blocked:
        # Installing would finish and still leave the pack unusable, so the
        # honest answer is "install that one first", with the ids to do it.
        return JSONResponse(
            status_code=400,
            content={"detail": f"{pack.title} needs these packs first: "
                               f"{', '.join(blocked)}",
                     "blocked_by": blocked})

    try:
        job = await service.submit_install(
            pack, payload.items, mode=payload.mode or "live",
            variant=payload.variant)
    except PackBusy as exc:
        return JSONResponse(status_code=409,
                            content={"detail": str(exc), "job_id": exc.job_id})
    except RestartUnavailable as exc:
        return JSONResponse(status_code=409,
                            content={"detail": str(exc),
                                     "command": exc.command})
    except RestartRefused as exc:
        # Also a 409, and deliberately the same shape plus ``reason``: to the
        # client both mean "not from in here, not now", and the command is
        # the way round either. Caught BEFORE PackInsufficientDisk and the
        # generic branch below -- all three are PackInstallError subclasses,
        # and nothing failed here.
        return JSONResponse(status_code=409,
                            content={"detail": str(exc), "reason": exc.reason,
                                     "command": exc.command})
    except PackInsufficientDisk as exc:
        # 507 Insufficient Storage: the request was fine, the disk is not.
        return JSONResponse(status_code=507,
                            content={"detail": str(exc), "needed": exc.needed,
                                     "free": exc.free})
    except PackInstallError as exc:
        # The restart helper could not be started. Nothing was installed, no
        # job exists to follow, and the message is the only thing anyone has
        # -- so it travels rather than becoming an anonymous 500 page.
        return JSONResponse(status_code=500, content={"detail": str(exc)})
    except ValueError as exc:
        # A request the catalog cannot honour: an item id this pack does not
        # have, or a restart-mode install of a pack with no packages in it.
        # The client's fault, and nothing was started.
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    return {"job_id": job.job_id}


# ── the job routes ────────────────────────────────────────────────────────


def _job_not_found(job_id: str) -> HTTPException:
    """Only the most recent job is kept, so "gone" and "never existed" are
    the same answer -- and the client's next move is the same either way."""
    return HTTPException(status_code=404, detail=f"job '{job_id}' not found")


@router.post("/jobs/{job_id}/cancel",
             dependencies=[Depends(_require_local_install)])
async def cancel_job(job_id: str, request: Request) -> dict:
    """Ask the running install to stop.

    ``cancelled`` reports whether the request did anything: False for a job
    that had already finished. Both are 200 -- asking twice is not an error,
    and a cancel that raced the last step is normal.
    """
    service = _service(request)
    try:
        cancelled = await service.cancel(job_id)
    except UnknownJob:
        raise _job_not_found(job_id) from None
    return {"job_id": job_id, "cancelled": cancelled}


@router.get("/jobs/{job_id}/events")
async def get_job_events(
    job_id: str,
    request: Request,
    cursor: int = Query(default=0, ge=0),
    wait: float = Query(default=0.0, ge=0.0, le=60.0),
    limit: int = Query(default=500, ge=1, le=2000),
) -> dict:
    """Events strictly after *cursor*, oldest first.

    ``wait`` seconds of long polling when the tail is empty: the request
    parks on an in-process wake-up (never a retry loop) and returns the
    moment an event lands, the job ends, or the deadline passes. A finished
    job answers immediately regardless of ``wait``, and its events stay
    readable until the next install replaces them -- which is what makes the
    last page of a failed install fetchable at all.

    The returned ``cursor`` is where to resume, and never moves backwards.
    """
    service = _service(request)
    try:
        events, next_cursor, status = await service.wait_for_events(
            job_id, after_cursor=cursor, limit=limit, wait=wait)
    except UnknownJob:
        # Also reachable AFTER the park: a job that ends while a poll is
        # waiting on it, followed by a new install, takes its events with it.
        raise _job_not_found(job_id) from None
    return {"job_id": job_id, "status": status, "events": events,
            "cursor": next_cursor}


# ── DELETE /api/packs/{pack_id}/items/{item_id} ───────────────────────────


@router.delete("/{pack_id}/items/{item_id}",
               dependencies=[Depends(_require_local_install)])
async def delete_pack_item(pack_id: str, item_id: str, request: Request):
    """Delete one downloaded model and forget it.

    ``removed`` is ``flows.remove_item``'s answer verbatim, and False is a
    real result rather than "nothing to do": the sentinel is always cleared,
    but on Windows a file another process holds open survives the delete, and
    the caller is entitled to know its disk did not come back.

    Only ITEMS. A pack's pip packages are not items, and nothing removes
    them for you -- doing it from inside the process that imported them
    leaves a half-loaded interpreter serving requests. ``cdui packs remove
    <pack_id> <item_id>`` deletes one model and PRINTS the ``uv pip
    uninstall --python <venv python> ...`` line for the packages, to run
    from a terminal with the server stopped.
    """
    pack = _require_pack(pack_id)
    if not any(item.item_id == item_id for item in pack.items):
        raise HTTPException(
            status_code=404,
            detail=f"pack '{pack_id}' has no item '{item_id}'")

    service = _service(request)
    active = _active_job(service)
    if active is not None and active.pack_id == pack_id:
        # Deleting the repo folder a download is writing into leaves a
        # half-populated cache the sentinel then vouches for.
        return JSONResponse(
            status_code=409,
            content={"detail": f"an install of '{pack_id}' is running; "
                               f"cancel it before removing its files",
                     "job_id": active.job_id})

    # Off the loop: this is an rmtree over hundreds of megabytes.
    removed = await asyncio.to_thread(flows.remove_item, pack, item_id)
    return {"pack_id": pack_id, "item_id": item_id, "removed": removed}
