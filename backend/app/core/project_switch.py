"""Switching the open project while the server runs (phase 4A).

``POST /api/projects/open`` ends here. A switch moves every path the server
resolves (graphs, assets, a run's checkpoints, the ``.env`` in the process)
from one project to another, so it is refused while anything is still
working against the old one, and it is all-or-nothing: a failure part way
puts the old project back.

Refused (409, :class:`SwitchRefused`; the message names what is busy):

``runs_active``
    A run is running or queued: canvas runs, ``/api/graph/run``, app
    invokes, sweep variants and study children all go through RunService.
    A submit already in progress counts too.
``study_active``
    A study is queued, running or being cancelled: its coordinator would
    submit the next child into the other project.
``install_active``
    A plugin or pack install job is running.
``switch_in_progress``
    Another switch holds the lock.
``git_busy``
    A git operation holds ``GitService``'s lock (tried, never waited for).
``worker_mount``
    A connected worker declares this server's mount label with a directory
    of its own (``--mount data=...``): its paths point into the old project
    and only restarting it changes that.

Steps, under the switch lock and the git lock, with RunService closed to
new submits: check the guards again (a run submitted meanwhile is caught),
:func:`projects.activate` (roots and ``.env``), clear the in-memory stores
that hold the old project's state, then record the project as current in
``projects.json``. Anything failing after ``activate`` re-activates the old
project before the error propagates; the registry is written last, so a
failed switch leaves it as it was.

Paths are checked by :func:`check_project_path` (400 / 403,
:class:`PathRefused`); ``create`` and ``clone`` apply
:func:`refused_location` to their targets as well.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import _user_data_root, settings
from . import projects
from .project import PROJECT_MANIFEST

logger = logging.getLogger(__name__)

#: One switch at a time.
_switch_lock = asyncio.Lock()


class SwitchRefused(Exception):
    """The server is busy with the open project (409)."""

    def __init__(self, code: str, message: str,
                 reasons: list[dict[str, str]] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.reasons = reasons or [{"code": code, "message": message}]

    def detail(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message,
                "details": self.reasons}


class PathRefused(Exception):
    """The path cannot be a project (400) or may not be one (403)."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message

    def detail(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": None}


# ── Paths ─────────────────────────────────────────────────────────────────

#: Directories that are never a project nor contain one (the directory or
#: anything under it).
_SYSTEM_TREES_POSIX = ("/etc", "/usr", "/bin", "/sbin", "/boot", "/dev",
                       "/proc", "/sys", "/System", "/Library/Apple")
#: Directories that may contain projects but are not one themselves.
_SYSTEM_EXACT_POSIX = ("/", "/var", "/private", "/opt", "/Library", "/Users",
                       "/home", "/tmp", "/Volumes", "/mnt", "/media")


def _windows_trees() -> list[Path]:
    import os

    roots = [os.environ.get("SystemRoot", r"C:\Windows"),
             os.environ.get("ProgramFiles", r"C:\Program Files"),
             os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
             os.environ.get("ProgramData", r"C:\ProgramData")]
    return [Path(r) for r in roots if r]


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _install_dir() -> Path:
    """The CodefyUI checkout or install (``backend/``'s parent)."""
    return Path(__file__).resolve().parents[3]


def refused_location(path: Path) -> str | None:
    """Why *path* (resolved) may not hold a project, or None.

    System directories, the install directory (except a projects root that
    a developer put inside it), the per-user data directory (it holds the
    session token, the lockfile and ``trust.json``) and the plugin
    directories.
    """
    home = Path.home().resolve()
    if path == home:
        return "the home directory itself cannot be a project"
    if sys.platform == "win32":
        anchor = Path(path.anchor)
        if path == anchor:
            return f"{path} is a drive root"
        for tree in _windows_trees():
            if _under(path, tree.resolve()):
                return f"{path} is inside the system directory {tree}"
    else:
        # macOS resolves /etc, /var and /tmp into /private: test both forms.
        forms = {str(path)}
        if str(path).startswith("/private/"):
            forms.add(str(path)[len("/private"):])
        for exact in _SYSTEM_EXACT_POSIX:
            if exact in forms:
                return f"{path} is a system directory"
        for tree in _SYSTEM_TREES_POSIX:
            if any(form == tree or form.startswith(tree + "/") for form in forms):
                return f"{path} is inside the system directory {tree}"
    install = _install_dir()
    projects_root = Path(settings.PROJECTS_ROOT).expanduser().resolve()
    if _under(path, install) and not _under(path, projects_root):
        return f"{path} is inside the CodefyUI install ({install})"
    user_data = _user_data_root().resolve()
    if _under(path, user_data):
        return (f"{path} is inside CodefyUI's own data directory "
                f"({user_data})")
    for name in ("PLUGINS_USER_DIR", "PLUGINS_BUILTIN_DIR", "CUSTOM_NODES_DIR"):
        plugins = Path(getattr(settings, name)).expanduser().resolve()
        if _under(path, plugins):
            return f"{path} is inside the plugin directory {plugins}"
    return None


def check_project_path(raw: str | Path) -> Path:
    """*raw* resolved, if it is a project this server may open."""
    text = str(raw).strip()
    if not text:
        raise PathRefused(400, "invalid_path", "no path given")
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise PathRefused(400, "invalid_path",
                          f"{text} is not an absolute path")
    try:
        path = path.resolve(strict=True)
    except (OSError, RuntimeError):
        raise PathRefused(400, "not_found", f"{text} does not exist") from None
    if not path.is_dir():
        raise PathRefused(400, "not_a_directory", f"{path} is not a folder")
    reason = refused_location(path)
    if reason is not None:
        raise PathRefused(403, "path_refused", reason)
    if not (path / PROJECT_MANIFEST).is_file():
        raise PathRefused(400, "not_a_project",
                          f"{path} has no {PROJECT_MANIFEST}; it is not a "
                          "CodefyUI project")
    try:
        tomllib.loads((path / PROJECT_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise PathRefused(400, "invalid_manifest",
                          f"{path / PROJECT_MANIFEST} could not be read: "
                          f"{exc}") from None
    return path


# ── Guards ────────────────────────────────────────────────────────────────

def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


async def busy_reasons(state: Any) -> list[dict[str, str]]:
    """Everything that keeps the open project from being switched now."""
    reasons: list[dict[str, str]] = []
    service = getattr(state, "run_service", None)
    if service is not None:
        active = service.active_run_ids()
        queued = service.queued_run_ids()
        submitting = service.submits_in_flight()
        if active or queued or submitting:
            parts = []
            if active:
                parts.append(_plural(len(active), "run is running",
                                     "runs are running"))
            if queued:
                parts.append(_plural(len(queued), "run is queued",
                                     "runs are queued"))
            if submitting:
                parts.append(_plural(submitting, "run is being submitted",
                                     "runs are being submitted"))
            reasons.append({
                "code": "runs_active",
                "message": (" and ".join(parts) + " in this project; wait for "
                            "them or cancel them in the Runs tab first"),
            })
    coordinator = getattr(state, "study_coordinator", None)
    if coordinator is not None:
        from .study_store import STUDY_ACTIVE

        studies, total = await coordinator.store.list_studies(
            status=sorted(STUDY_ACTIVE), limit=3)
        if total:
            names = ", ".join(f"'{s.name or s.id}'" for s in studies)
            more = "" if total <= len(studies) else f" and {total - len(studies)} more"
            reasons.append({
                "code": "study_active",
                "message": (f"{_plural(total, 'study is', 'studies are')} "
                            f"still running ({names}{more}); wait for "
                            f"{'it' if total == 1 else 'them'} or cancel "
                            "in the Studies view first"),
            })
    for attr, what in (("plugin_service", "plugin"), ("pack_service", "pack")):
        installer = getattr(state, attr, None)
        job = installer.current_job_id() if installer is not None else None
        if job is not None:
            reasons.append({
                "code": "install_active",
                "message": (f"a {what} install is running (job {job}); "
                            "wait for it to finish"),
            })
    git = getattr(state, "git_service", None)
    if git is not None and (git.lock.locked() or git.network_lock.locked()):
        reasons.append({
            "code": "git_busy",
            "message": ("a git operation is running in Source Control; "
                        "wait for it to finish"),
        })
    registry = getattr(state, "worker_registry", None)
    if registry is not None:
        from .mounts import control_plane_label
        from .workers import STATUS_ONLINE

        label = control_plane_label()
        for worker in registry.workers():
            if worker.status != STATUS_ONLINE:
                continue
            path = (worker.mounts or {}).get(label)
            if path is None:
                continue
            reasons.append({
                "code": "worker_mount",
                "message": (
                    f"worker '{worker.name}' declares the mount '{label}' "
                    f"at {path}, which is this project's folder on that "
                    "machine; stop it, switch, then start it again with "
                    f"--mount {label}=<the new project's folder there>"),
            })
    return reasons


def _first(reasons: list[dict[str, str]]) -> SwitchRefused:
    message = "; ".join(r["message"] for r in reasons)
    return SwitchRefused(reasons[0]["code"], "Cannot switch projects: "
                         + message, reasons)


# ── The switch ────────────────────────────────────────────────────────────

@dataclass
class SwitchResult:
    previous: Path | None
    project: Path | None
    env: projects.EnvStatus


async def _clear_stores(state: Any) -> None:
    """Drop the in-memory state that belongs to the old project's tabs and
    runs (persisted modules, captured outputs, port statistics)."""
    node_states = getattr(state, "node_state_store", None)
    if node_states is not None:
        node_states.reset_all()
    outputs = getattr(state, "run_output_store", None)
    if outputs is not None:
        await outputs.clear()
    port_stats = getattr(state, "port_stats_cache", None)
    if port_stats is not None:
        port_stats.clear()


async def switch_project(state: Any, target: Path | None) -> SwitchResult:
    """Open *target* (already checked; None: flat mode, for tests) in place
    of the open project, or raise :class:`SwitchRefused`."""
    if _switch_lock.locked():
        raise SwitchRefused("switch_in_progress",
                            "Cannot switch projects: another switch is in "
                            "progress")
    async with _switch_lock:
        reasons = await busy_reasons(state)
        if reasons:
            raise _first(reasons)
        git = getattr(state, "git_service", None)
        if git is not None:
            if git.lock.locked():
                raise _first([{"code": "git_busy", "message": (
                    "a git operation is running in Source Control; wait "
                    "for it to finish")}])
            await git.lock.acquire()
        service = getattr(state, "run_service", None)
        if service is not None:
            service.close_for_switch(
                "the project is being switched; submit again in a moment")
        try:
            # Again, with submits closed: a run that arrived between the
            # first check and closing is caught here.
            reasons = await busy_reasons(state)
            reasons = [r for r in reasons if r["code"] != "git_busy"]
            if reasons:
                raise _first(reasons)
            previous = settings.PROJECT_DIR
            try:
                env = projects.activate(target)
                await _clear_stores(state)
                if target is not None:
                    projects.record_opened(target)
            except BaseException:
                logger.exception("switching to %s failed; reopening %s",
                                 target, previous or "flat mode")
                projects.activate(previous)
                raise
        finally:
            if service is not None:
                service.reopen_after_switch()
            if git is not None:
                git.lock.release()
    logger.info("project switched: %s -> %s", previous or "flat mode",
                settings.PROJECT_DIR or "flat mode")
    return SwitchResult(previous=previous, project=settings.PROJECT_DIR,
                        env=env)
