"""Projects at run time (phase 4A): the registry, trust, and which project is open.

Three files under the per-user data root, never inside a project (a repo
must not vouch for itself):

``projects.json``
    The recently opened projects and the current one. Start-up opens the
    current one again when nothing more explicit says otherwise.
``trust.json``
    Per resolved project path: ``owned`` (this install created or imported
    it) and the sha256 of the ``.env`` the user agreed to load. Phase 4C adds
    the hash of ``mods/``.

:func:`activate` is the one way a project becomes the open one, at start-up
and on a switch: it re-derives the data roots (``Settings.derive_project_roots``),
takes the previous project's ``.env`` back out of the process and loads the
new one's -- only when it is trusted, and never a blocked name.
:func:`resolve_startup_project` decides what start-up opens.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Settings, _user_data_root, settings
from .dotenv import apply_project_env, is_blocked, read_dotenv, unload_project_env
from .project import PROJECT_MANIFEST, _atomic_write, collect_graph_files, read_project_manifest

logger = logging.getLogger(__name__)

REGISTRY_FILE = "projects.json"
TRUST_FILE = "trust.json"
#: How many recent projects the registry keeps.
MAX_RECENT = 20
#: The default project's directory name under ``PROJECTS_ROOT``.
DEFAULT_PROJECT_NAME = "workspace"

_lock = threading.RLock()


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _key(path: Path | str) -> str:
    return str(Path(path).expanduser().resolve())


def _read_json(path: Path) -> dict[str, Any]:
    """The file's object, or {} when it is absent. A corrupt file is moved
    aside (``<name>.corrupt-<time>``) and reported, never silently reused."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return {}
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("not a JSON object")
        return data
    except ValueError as exc:
        aside = path.with_name(f"{path.name}.corrupt-{datetime.datetime.now():%Y%m%d%H%M%S}")
        try:
            path.replace(aside)
        except OSError:
            aside = path
        logger.warning("%s was unreadable (%s); moved aside to %s and started empty",
                       path, exc, aside)
        return {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def registry_path() -> Path:
    return _user_data_root() / REGISTRY_FILE


def trust_path() -> Path:
    return _user_data_root() / TRUST_FILE


# ── What a project is ─────────────────────────────────────────────────────

def is_project(path: Path) -> bool:
    return (path / PROJECT_MANIFEST).is_file()


def project_name(path: Path) -> str:
    name = read_project_manifest(path).get("project", {})
    name = name.get("name") if isinstance(name, dict) else None
    return name if isinstance(name, str) and name.strip() else path.name


# ── Registry ──────────────────────────────────────────────────────────────

def load_registry() -> dict[str, Any]:
    """``{"current": str | None, "recent": [{"path", "name", "last_opened"}]}``."""
    with _lock:
        data = _read_json(registry_path())
    recent = [entry for entry in data.get("recent", [])
              if isinstance(entry, dict) and isinstance(entry.get("path"), str)]
    current = data.get("current")
    return {"current": current if isinstance(current, str) else None, "recent": recent}


def _save_registry(data: dict[str, Any]) -> None:
    _write_json(registry_path(), {"version": 1, **data})


def record_opened(path: Path) -> None:
    """Make *path* the current project and the most recent one."""
    key = _key(path)
    with _lock:
        data = load_registry()
        recent = [entry for entry in data["recent"] if entry["path"] != key]
        recent.insert(0, {"path": key, "name": project_name(Path(key)), "last_opened": _now()})
        _save_registry({"current": key, "recent": recent[:MAX_RECENT]})


def remove_recent(path: str) -> bool:
    """Drop *path* from the recent list (never from disk). The current
    project stays current."""
    key = _key(path)
    with _lock:
        data = load_registry()
        recent = [entry for entry in data["recent"] if entry["path"] != key]
        if len(recent) == len(data["recent"]):
            return False
        _save_registry({"current": data["current"], "recent": recent})
        return True


# ── Trust ─────────────────────────────────────────────────────────────────

def env_digest(project: Path) -> str | None:
    try:
        return hashlib.sha256((project / ".env").read_bytes()).hexdigest()
    except OSError:
        return None


def _trust_entries() -> dict[str, Any]:
    entries = _read_json(trust_path()).get("projects", {})
    return entries if isinstance(entries, dict) else {}


def _update_trust(project: Path, **changes: Any) -> None:
    key = _key(project)
    with _lock:
        entries = _trust_entries()
        entry = entries.get(key) if isinstance(entries.get(key), dict) else {}
        entry.update(changes, updated=_now())
        entries[key] = entry
        _write_json(trust_path(), {"version": 1, "projects": entries})


def mark_owned(project: Path) -> None:
    """This install created or imported *project*: its ``.env`` is trusted
    whatever it later holds, as a file the user wrote."""
    _update_trust(project, owned=True)


def trust_env(project: Path, digest: str | None) -> None:
    """The user agreed to load the ``.env`` whose sha256 is *digest*."""
    _update_trust(project, env_sha256=digest)


def env_trust_state(project: Path) -> str:
    """``none`` (no .env), ``trusted``, ``untrusted`` (never agreed) or
    ``changed`` (agreed to different contents)."""
    digest = env_digest(project)
    if digest is None:
        return "none"
    entry = _trust_entries().get(_key(project))
    if not isinstance(entry, dict):
        return "untrusted"
    if entry.get("owned") is True or entry.get("env_sha256") == digest:
        return "trusted"
    return "changed" if entry.get("env_sha256") else "untrusted"


# ── The open project ──────────────────────────────────────────────────────

@dataclass
class EnvStatus:
    """What happened to the open project's ``.env``. Names only, never values."""
    state: str = "none"
    names: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)
    applied: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)
    restart: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"state": self.state, "names": self.names, "blocked": self.blocked,
                "applied": self.applied, "kept": self.kept,
                "restart_recommended": self.restart}


#: The open project's .env status, for ``GET /api/projects``.
current_env = EnvStatus()

#: The project :func:`activate` last opened. A PROJECT_DIR that differs was
#: set by someone else -- the operator's --project -- and is explicit.
_activated: Path | None = None


def load_project_env(project: Path, *, explicit: bool = False) -> EnvStatus:
    """Load *project*'s ``.env`` if it may be loaded, and record how it went.

    ``explicit`` is the project the operator named at start-up: loaded
    whatever it holds, without the blocklist, as before phase 4A, and its
    current contents recorded as trusted.
    """
    global current_env
    values = read_dotenv(project / ".env")
    if values is None:
        current_env = EnvStatus()
        return current_env
    names = sorted(values)
    blocked = sorted(name for name in names if is_blocked(name))
    state = "trusted" if explicit else env_trust_state(project)
    status = EnvStatus(state=state, names=names, blocked=[] if explicit else blocked)
    if state == "trusted":
        report = apply_project_env(values, enforce_blocklist=not explicit)
        status.applied, status.kept, status.restart = report["applied"], report["kept"], report["restart"]
        if explicit:
            trust_env(project, env_digest(project))
    current_env = status
    return status


def activate(project: Path | None, *, explicit: bool = False) -> EnvStatus:
    """Make *project* (None: flat mode) the open one in this process.

    Re-derives the data roots, keeps ``CODEFYUI_PROJECT_DIR`` in step for
    child processes, takes the previous project's ``.env`` out and loads
    this one's. Nothing here checks whether a switch is safe; that is
    ``core.project_switch``'s job.
    """
    global current_env, _activated
    unload_project_env()
    resolved = Path(project).expanduser().resolve() if project is not None else None
    settings.derive_project_roots(resolved)
    _activated = resolved
    if resolved is None:
        os.environ.pop("CODEFYUI_PROJECT_DIR", None)
        current_env = EnvStatus()
        return current_env
    os.environ["CODEFYUI_PROJECT_DIR"] = str(resolved)
    return load_project_env(resolved, explicit=explicit)


# ── Start-up ──────────────────────────────────────────────────────────────

@dataclass
class StartupDecision:
    #: ``explicit`` (--project), ``flat-pinned`` (CODEFYUI_GRAPHS_DIR),
    #: ``registry``, ``flat-graphs`` (old graphs found, import offered),
    #: ``default`` (the default project), ``flat`` (nothing else applied).
    reason: str
    project: Path | None
    #: Graphs in the old flat directory, offered for import (``flat-graphs``).
    flat_graphs: int = 0
    created: bool = False
    warnings: list[str] = field(default_factory=list)


#: What start-up decided, for ``GET /api/projects``.
startup_decision: StartupDecision | None = None


def flat_graphs_dir() -> Path:
    """The pre-project graphs directory (``backend/data/graphs``)."""
    return Path(Settings.model_fields["GRAPHS_DIR"].default)


def count_flat_graphs(directory: Path | None = None) -> int:
    directory = directory or flat_graphs_dir()
    if not directory.is_dir():
        return 0
    try:
        return len(collect_graph_files(directory))
    except Exception:  # noqa: BLE001 -- an ambiguous pair still means "has graphs"
        return sum(1 for _ in directory.glob("*.json"))


def _decide() -> StartupDecision:
    if settings.PROJECT_DIR is not None and settings.PROJECT_DIR != _activated:
        # Set before this process opened anything itself: --project (or a
        # test standing in for it).
        return StartupDecision("explicit", settings.PROJECT_DIR)
    if "GRAPHS_DIR" in settings._explicit_roots:
        return StartupDecision("flat-pinned", None)
    warnings: list[str] = []
    current = load_registry()["current"]
    if current:
        if is_project(Path(current)):
            return StartupDecision("registry", Path(current))
        warnings.append(f"The last open project {current} is gone or has no "
                        f"{PROJECT_MANIFEST}; not reopening it")
    flat = count_flat_graphs()
    if flat:
        return StartupDecision("flat-graphs", None, flat_graphs=flat, warnings=warnings)
    if not settings.DEFAULT_PROJECT:
        return StartupDecision("flat", None, warnings=warnings)
    target = Path(settings.PROJECTS_ROOT).expanduser() / DEFAULT_PROJECT_NAME
    if is_project(target):
        return StartupDecision("default", target, warnings=warnings)
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        warnings.append(f"{target} exists but is not a CodefyUI project; leaving it "
                        "alone and starting without a project")
        return StartupDecision("flat", None, warnings=warnings)
    from .project import init_project_repo, scaffold_project

    scaffold_project(target)
    if not init_project_repo(target):
        warnings.append(f"git is not installed; {target} was created without a repository")
    mark_owned(target)
    return StartupDecision("default", target, created=True, warnings=warnings)


def resolve_startup_project() -> StartupDecision:
    """Decide which project start-up opens, open it, and record it.

    Precedence (reviews/phase-4a-design.md): --project, an explicit
    CODEFYUI_GRAPHS_DIR (flat), the registry's current project, old graphs
    in the flat directory (flat, import offered), then the default project
    (``CODEFYUI_DEFAULT_PROJECT=0`` stops at flat mode).
    """
    global startup_decision
    decision = _decide()
    for warning in decision.warnings:
        logger.warning("%s", warning)
    status = activate(decision.project, explicit=decision.reason == "explicit")
    if decision.project is not None:
        record_opened(decision.project)
    if status.state in ("untrusted", "changed"):
        logger.warning("The project's .env was not loaded: it has not been "
                       "trusted in this install (open the project menu to review it)")
    startup_decision = decision
    return decision
