"""Projects: which one is open, switching, creating, cloning (phase 4A).

``GET  /api/projects``          the open project, recents, the .env status
``POST /api/projects/open``     switch to a project folder
``POST /api/projects/create``   scaffold + git init + open
``POST /api/projects/clone``    clone an https repository + open
``GET  /api/projects/import``   what importing the flat graphs would copy
``POST /api/projects/import``   copy them (and their files) into a project
``POST /api/projects/trust``    load the open project's .env after consent
``DELETE /api/projects/recent`` drop an entry from the recent list
``GET  /api/projects/browse``   list sub-folders (loopback only)

The writes (POST, DELETE) need the session token, like every write under
``/api/`` (``auth_guard``); the reads (GET) follow the same policy as the
other read routes and need none -- ``browse`` is limited to loopback
instead. None of these routes is checked against ``X-CodefyUI-Project``:
they are how a window changes or learns the open project. A switch answers
with the new project; the editor then reloads the page. What makes a
switch refuse is in ``core.project_switch``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from ..config import settings
from ..core import projects
from ..core.auth import bound_to_loopback, is_loopback_peer
from ..core.fs_paths import canonical_path
from ..core.project import GraphAmbiguityError, init_new_project_repo, scaffold_project
from ..core.project_switch import (
    PathRefused,
    SwitchRefused,
    busy_reasons,
    check_project_path,
    refused_location,
    switch_project,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/projects", tags=["projects"])

#: A project folder name: portable on every OS this runs on.
_NAME = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_WINDOWS_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)})
#: How many sub-folders ``browse`` lists at most.
BROWSE_LIMIT = 500


def name_problem(name: str) -> str | None:
    """Why *name* cannot be a project folder name, or None."""
    if not _NAME.fullmatch(name):
        return ("use 1-64 letters, digits, '.', '_' or '-' (no spaces or "
                "other characters)")
    if name.startswith("."):
        return "a project name cannot start with '.'"
    if name.endswith("."):
        return "a project name cannot end with '.'"
    if name.split(".")[0].upper() in _WINDOWS_RESERVED:
        return f"{name} is a reserved name on Windows"
    return None


def _error(status: int, code: str, message: str,
           details: Any = None) -> HTTPException:
    return HTTPException(status_code=status, detail={
        "code": code, "message": message, "details": details})


def _project_view(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return {"path": str(path), "name": projects.project_name(path),
            "project_id": projects.project_id(path)}


def _env_view() -> dict[str, Any]:
    env = projects.current_env.as_dict()
    project = settings.PROJECT_DIR
    env["sha256"] = (projects.env_digest(project)
                     if project is not None and env["names"] else None)
    env["needs_consent"] = env["state"] in ("untrusted", "changed")
    return env


def _state(request: Request) -> Any:
    return request.app.state


def _opened(result_project: Path | None, *, extra: dict[str, Any] | None = None
            ) -> dict[str, Any]:
    env = _env_view()
    answer = {
        "project": _project_view(result_project),
        "project_id": projects.project_id(result_project),
        "env": env,
        # Variables some libraries read only at import or start-up: changing
        # them at run time has no effect until the server restarts.
        "env_restart_recommended": env["restart_recommended"],
        "mods": _mods_view(result_project),
    }
    if extra:
        answer.update(extra)
    return answer


async def _switch(request: Request, target: Path) -> Path | None:
    try:
        result = await switch_project(_state(request), target)
    except SwitchRefused as exc:
        raise HTTPException(status_code=409, detail=exc.detail()) from None
    await _broadcast(request, result.project)
    return result.project


async def _broadcast(request: Request, project: Path | None) -> None:
    from .ws_execution import broadcast_project_changed

    try:
        await broadcast_project_changed({
            "type": "project_changed",
            "project": None if project is None else str(project),
            "project_id": projects.project_id(project),
        })
    except Exception:  # noqa: BLE001 - other windows are a courtesy
        logger.exception("could not tell the other windows about the switch")


# ── Reading ───────────────────────────────────────────────────────────────

@router.get("")
async def list_projects(request: Request) -> dict[str, Any]:
    registry = projects.load_registry()
    recent = []
    for entry in registry["recent"]:
        path = Path(entry["path"])
        recent.append({**entry, "missing": not projects.is_project(path)})
    decision = projects.startup_decision
    current = settings.PROJECT_DIR
    return {
        "current": _project_view(current),
        "project_id": projects.project_id(current),
        "projects_root": str(Path(settings.PROJECTS_ROOT).expanduser()),
        "recent": recent,
        "env": _env_view(),
        "mods": _mods_view(current),
        "startup": None if decision is None else {
            "reason": decision.reason,
            "created": decision.created,
            "warnings": list(decision.warnings),
        },
        "flat_graphs": (projects.count_flat_graphs() if current is None
                        else 0),
        "browse_allowed": _browse_allowed(request),
    }


# ── Switching ─────────────────────────────────────────────────────────────

class OpenRequest(BaseModel):
    path: str


@router.post("/open")
async def open_project(body: OpenRequest, request: Request) -> dict[str, Any]:
    try:
        target = await asyncio.to_thread(check_project_path, body.path)
    except PathRefused as exc:
        raise HTTPException(status_code=exc.status, detail=exc.detail()) from None
    if settings.PROJECT_DIR is not None and target == settings.PROJECT_DIR:
        return _opened(target, extra={"unchanged": True})
    project = await _switch(request, target)
    return _opened(project, extra={"unchanged": False})


class CreateRequest(BaseModel):
    name: str
    parent: str | None = None


def _parent_dir(raw: str | None) -> Path:
    if raw is None or not raw.strip():
        parent = Path(settings.PROJECTS_ROOT).expanduser()
        parent.mkdir(parents=True, exist_ok=True)
    else:
        parent = Path(raw.strip()).expanduser()
        if not parent.is_absolute():
            raise _error(400, "invalid_path", f"{raw} is not an absolute path")
        if not parent.is_dir():
            raise _error(400, "not_found", f"{parent} is not an existing folder")
    return canonical_path(parent)


async def _refuse_if_busy(request: Request) -> None:
    """Refuse before making anything on disk when the switch that follows
    would be refused anyway."""
    reasons = await busy_reasons(_state(request))
    if reasons:
        raise HTTPException(status_code=409, detail=SwitchRefused(
            reasons[0]["code"], "Cannot switch projects: "
            + "; ".join(r["message"] for r in reasons), reasons).detail())


def _new_project_target(name: str, parent_raw: str | None) -> Path:
    """Where ``create`` (and an import into a new project) puts *name*."""
    problem = name_problem(name)
    if problem is not None:
        raise _error(400, "invalid_name", problem)
    target = _parent_dir(parent_raw) / name
    reason = refused_location(target)
    if reason is not None:
        raise _error(403, "path_refused", reason)
    if target.is_symlink() or (target.exists() and (
            not target.is_dir() or any(target.iterdir()))):
        raise _error(409, "exists",
                     f"{target} already exists and is not empty; pick another "
                     "name or open it")
    return target


def _make_project(target: Path, name: str) -> bool:
    """Scaffold, mark owned (this install made it), git init and commit."""
    scaffold_project(target, name)
    projects.mark_owned(target)
    return init_new_project_repo(target)


@router.post("/create")
async def create_project(body: CreateRequest, request: Request) -> dict[str, Any]:
    name = body.name.strip()
    target = _new_project_target(name, body.parent)
    await _refuse_if_busy(request)
    git_ready = await asyncio.to_thread(_make_project, target, name)
    project = await _switch(request, canonical_path(target))
    return _opened(project, extra={"created": str(canonical_path(target)),
                                   "git": git_ready})


# ── Importing the flat-mode graphs (R9) ───────────────────────────────────

@router.get("/import")
async def import_preview() -> dict[str, Any]:
    """The flat graphs and the files they reference, with sizes. Reads only."""
    from ..core.project_import import plan_import

    try:
        plan = await asyncio.to_thread(plan_import)
    except GraphAmbiguityError as exc:
        raise _error(409, "ambiguous_graph", str(exc)) from None
    return plan.as_dict()


class ImportRequest(BaseModel):
    #: A new project of this name under ``parent`` (default: the projects
    #: root). Or ``path``: an existing project. Neither: the open project.
    name: str | None = None
    parent: str | None = None
    path: str | None = None
    #: Copy the referenced models, images and media too (default on).
    include_assets: bool = True


@router.post("/import")
async def import_flat_graphs(body: ImportRequest, request: Request
                             ) -> dict[str, Any]:
    """Copy the flat graphs (and their files) into a project, then open it.

    Copies only: the flat folder is left as it was, and a graph or file
    the project already has is kept, not replaced (reported in
    ``skipped`` / ``kept``).
    """
    from ..core.project_import import run_import

    created = False
    if body.name:
        name = body.name.strip()
        target = _new_project_target(name, body.parent)
        created = True
    elif body.path:
        try:
            target = await asyncio.to_thread(check_project_path, body.path)
        except PathRefused as exc:
            raise HTTPException(status_code=exc.status,
                                detail=exc.detail()) from None
    elif settings.PROJECT_DIR is not None:
        target = settings.PROJECT_DIR
    else:
        raise _error(400, "no_target", "name a new project or an existing one")
    switching = settings.PROJECT_DIR is None or target != settings.PROJECT_DIR
    if switching:
        await _refuse_if_busy(request)

    def _do() -> Any:
        if created:
            _make_project(target, name)
        return run_import(target, include_assets=body.include_assets,
                          keep_existing=not created)

    try:
        result = await asyncio.to_thread(_do)
    except GraphAmbiguityError as exc:
        raise _error(409, "ambiguous_graph", str(exc)) from None
    project = (await _switch(request, canonical_path(target)) if switching
               else settings.PROJECT_DIR)
    return _opened(project, extra={
        "created": str(canonical_path(target)) if created else None,
        "imported": result.graphs.adopted,
        "skipped": [{"file": f, "reason": r} for f, r in result.graphs.skipped],
        "copied": result.copied,
        "kept": result.kept,
    })


class CloneRequest(BaseModel):
    url: str
    parent: str | None = None


@router.post("/clone")
async def clone_project_route(body: CloneRequest, request: Request
                              ) -> dict[str, Any]:
    from ..core.plugins.errors import GitError, GitUnavailable, SourceRefused
    from ..core.project_clone import CloneRefused, clone_project, repo_name

    parent = _parent_dir(body.parent)
    reason = refused_location(parent / (repo_name(body.url.strip()) or "x"))
    if reason is not None:
        raise _error(403, "path_refused", reason)
    await _refuse_if_busy(request)
    try:
        result = await asyncio.to_thread(clone_project, body.url, parent,
                                         name_check=name_problem)
    except CloneRefused as exc:
        raise _error(exc.status, exc.code, exc.message, exc.hint) from None
    except GitUnavailable as exc:
        raise _error(503, "git_unavailable", str(exc),
                     getattr(exc, "hint", None)) from None
    except SourceRefused as exc:
        raise _error(400, f"clone_{exc.reason}", str(exc),
                     getattr(exc, "hint", None)) from None
    except GitError as exc:
        status = 504 if exc.reason == "timeout" else 400
        raise _error(status, f"clone_{exc.reason}", str(exc),
                     getattr(exc, "hint", None)) from None
    project = await _switch(request, result.path)
    return _opened(project, extra={"cloned": str(result.path),
                                   "head": result.head,
                                   "uses_lfs": result.uses_lfs})


# ── Trust ─────────────────────────────────────────────────────────────────

class TrustRequest(BaseModel):
    #: The sha256 of the .env the user was shown (``env.sha256``). Consent
    #: is for those contents: a file that changed since is refused.
    sha256: str


@router.post("/trust")
async def trust_project_env(body: TrustRequest) -> dict[str, Any]:
    project = settings.PROJECT_DIR
    if project is None:
        raise _error(400, "no_project", "no project is open")
    digest = projects.env_digest(project)
    if digest is None:
        raise _error(400, "no_env", f"{project} has no .env")
    if digest != body.sha256:
        raise _error(409, "env_changed",
                     ".env changed since it was shown; review it again")
    projects.trust_env(project, digest)
    projects.load_project_env(project)
    return _opened(project)


# ── Project mods (phase 4C) ───────────────────────────────────────────────

def _mods_view(project: Path | None) -> dict[str, Any]:
    """The open project's mods: the trust state and whether they are
    loaded. The full scan is ``GET /api/projects/mods``."""
    from ..core import mods

    if project is None:
        return {"state": "none", "loaded": False, "needs_consent": False}
    status = mods.current_status(project)
    status["needs_consent"] = status["state"] in ("untrusted", "changed")
    if status["needs_consent"]:
        # What a "not now" is remembered against: the same tree is not
        # asked about again this session, a changed one is.
        status["digest"] = mods.mods_digest(mods.walk_mods(project))
    return status


@router.get("/mods")
async def read_project_mods() -> dict[str, Any]:
    """What the trust prompt lists: every file under ``mods/``, each node
    file's scan (gate verdict, capabilities, node names), refused entries,
    the digest, and the trust state. Nothing is imported to answer."""
    from ..core import mods

    project = settings.PROJECT_DIR
    if project is None:
        raise _error(400, "no_project", "no project is open")
    tree = mods.walk_mods(project)
    answer = mods.scan_mods(project, tree)
    answer.update(_mods_view(project))
    answer["trust"] = mods.mods_trust_state(project, tree)
    return answer


class TrustModsRequest(BaseModel):
    #: The digest of the tree the user was shown (``GET /mods`` ``digest``).
    sha256: str
    #: Also run the files the scan had findings for ("trust these files
    #: completely"). Without it those files are not loaded.
    full: bool = False


@router.post("/trust-mods")
async def trust_project_mods(body: TrustModsRequest) -> dict[str, Any]:
    from ..core import mods

    project = settings.PROJECT_DIR
    if project is None:
        raise _error(400, "no_project", "no project is open")
    tree = mods.walk_mods(project)
    if tree.refused:
        raise _error(409, "mods_blocked", "mods/ holds entries that are never loaded; "
                     "see GET /api/projects/mods", details=[
                         {"path": p, "reason": r} for p, r in tree.refused])
    if mods.mods_digest(tree) != body.sha256:
        raise _error(409, "mods_changed",
                     "mods/ changed since it was shown; review it again")
    mods.record_trust(project, tree, full=body.full)
    mods.reload_project_mods(project)
    return _opened(project)


# ── Recents and browsing ──────────────────────────────────────────────────

class RecentRequest(BaseModel):
    path: str


@router.delete("/recent")
async def remove_recent(body: RecentRequest) -> dict[str, Any]:
    if not projects.remove_recent(body.path):
        raise _error(404, "not_in_recent", f"{body.path} is not in the list")
    return {"removed": body.path}


def _browse_allowed(request: Request) -> bool:
    peer = request.client.host if request.client is not None else None
    return bound_to_loopback() and is_loopback_peer(peer)


@router.get("/browse")
async def browse(request: Request, dir: str | None = Query(default=None),
                 hidden: bool = Query(default=False)) -> dict[str, Any]:
    """Sub-folders of *dir* (default: the projects root), never file names.

    Only for a server that listens on loopback, asked from this machine: a
    LAN instance would otherwise list its disk to anyone on the network.
    """
    if not _browse_allowed(request):
        raise _error(403, "browse_refused",
                     "folder browsing is only available on this machine "
                     "(the server listens beyond loopback, or the request "
                     "is remote); type the full path instead")
    if dir is None or not dir.strip():
        base = Path(settings.PROJECTS_ROOT).expanduser()
        if not base.is_dir():
            base = Path.home()
    else:
        base = Path(dir.strip()).expanduser()
        if not base.is_absolute():
            raise _error(400, "invalid_path", f"{dir} is not an absolute path")

    def _list() -> dict[str, Any]:
        root = canonical_path(base)
        if not root.is_dir():
            raise _error(400, "not_found", f"{root} is not an existing folder")
        entries = []
        truncated = False
        try:
            children = sorted(os.scandir(root), key=lambda e: e.name.lower())
        except OSError as exc:
            raise _error(403, "unreadable",
                         f"{root} cannot be read: {exc.strerror}") from None
        for entry in children:
            if not hidden and entry.name.startswith("."):
                continue
            try:
                if not entry.is_dir():
                    continue
            except OSError:
                continue
            if len(entries) >= BROWSE_LIMIT:
                truncated = True
                break
            path = Path(entry.path)
            entries.append({"name": entry.name, "path": str(path),
                            "is_project": projects.is_project(path)})
        parent = root.parent if root.parent != root else None
        return {"dir": str(root), "parent": None if parent is None else str(parent),
                "is_project": projects.is_project(root),
                "entries": entries, "truncated": truncated}

    return await asyncio.to_thread(_list)
