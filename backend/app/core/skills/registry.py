"""Where skills are found, which one of a name wins, and reading their files.

Sources, highest priority first. A skill's ``id`` is ``<source>/<folder>``.

==============  ==================================================
``project``     ``<PROJECT_DIR>/skills/<name>/`` (project mode only)
``user``        ``<USER_DATA>/skills/<name>/`` (installed or placed by hand)
``plugin:<id>`` ``<plugin dir>/skills/<name>/`` (enabled plugins only)
``builtin``     ``<repo>/skills/<name>/``
``extra``       each directory in ``CODEFYUI_SKILLS_EXTRA_DIRS``
==============  ==================================================

Two skills of one name: the first valid, enabled one in that order is the
skill of that name, and every other entry of the name carries
``shadowed_by`` with its id. Invalid and disabled entries never shadow
anything, so disabling a project's copy of a skill brings the built-in one
back rather than leaving the agent with neither. Every entry is listed,
shadowed or not, so the list says why a skill is not in use.

Nothing is cached: a scan is a directory listing and a few small reads,
and a user who drops a folder into ``skills/`` sees it on the next request.
A broken skill costs that skill, never the list.

Links
-----
A skill that comes from someone else's repository (the ``project`` and
``plugin:*`` sources) must stay inside that source's ``skills/`` folder: a
skill folder that is a symbolic link leading out of it is listed as invalid
(``unsafe_link``) and nothing in it is read. Without this, a cloned project
holding ``skills/x -> ~/.ssh`` would hand the files under it to an agent.
The ``user``, ``builtin`` and ``extra`` sources are folders on the user's own
machine, where linking ``~/.claude/skills/foo`` in is the point, so a linked
skill folder is followed there. In every source ``SKILL.md`` itself must
resolve inside its skill folder, and a file read must too.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from ...config import _user_data_root, settings
from .. import plugin_loader
from ..data_paths import resolve_under
from .spec import MAX_SKILL_FILE_BYTES, SKILL_FILE, Issue, ParsedSkill, parse_skill

#: The largest file ``read_file`` returns whole; past it the text is cut.
MAX_READ_BYTES = 256 * 1024

#: How many files a skill's listing walks before it stops counting. A skill
#: is a handful of files; an ``extra`` directory pointed at a home folder by
#: mistake must not make every list request walk it.
MAX_LISTED_FILES = 1000

#: Directories a listing does not descend into.
_SKIP_DIRS = frozenset({".git", ".hg", ".svn", "__pycache__", "node_modules",
                        ".venv", "venv", ".mypy_cache", ".pytest_cache"})

_STATE_FILE = "state.json"

_KIND_BY_DIR = {"references": "reference", "scripts": "script", "assets": "asset"}



class SkillNotFound(LookupError):
    pass


class SkillPathRefused(ValueError):
    pass


class SkillFileNotText(ValueError):
    pass


@dataclass
class Skill:
    id: str
    source: str
    folder: Path
    parsed: ParsedSkill
    enabled: bool = True
    shadowed_by: str | None = None
    files: list[dict[str, Any]] = field(default_factory=list)
    #: False when a link made the skill unsafe to read: no file is served.
    readable: bool = True

    @property
    def name(self) -> str:
        return self.parsed.name

    @property
    def valid(self) -> bool:
        return self.parsed.valid

    def summary(self) -> dict[str, Any]:
        """The list route's entry."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.parsed.description,
            "source": self.source,
            "license": self.parsed.license,
            "compatibility": self.parsed.compatibility,
            "metadata": self.parsed.metadata,
            "allowed_tools": self.parsed.allowed_tools,
            "enabled": self.enabled,
            "valid": self.valid,
            "issues": [issue.as_dict() for issue in self.parsed.issues],
            "shadowed_by": self.shadowed_by,
            "file_count": len(self.files),
            "has_scripts": any(f["kind"] == "script" for f in self.files),
        }

    def detail(self) -> dict[str, Any]:
        """The single-skill route's answer: the summary, the body, the files."""
        return {**self.summary(), "body": self.parsed.body, "files": self.files}


# ── roots ──────────────────────────────────────────────────────────────────

def user_skills_root() -> Path:
    """``<USER_DATA>/skills``. Read per call: tests and dev tooling repoint
    ``CODEFYUI_USER_DATA_DIR``."""
    return _user_data_root() / "skills"


def project_skills_root() -> Path | None:
    return settings.PROJECT_DIR / "skills" if settings.PROJECT_DIR else None


def extra_roots() -> list[Path]:
    raw = settings.SKILLS_EXTRA_DIRS or ""
    return [Path(os.path.expanduser(part)) for part in raw.split(os.pathsep)
            if part.strip()]


def _is_contained_source(source: str) -> bool:
    return source == "project" or source.startswith("plugin:")


def source_roots() -> list[tuple[str, Path]]:
    """``(source, directory of skill folders)`` in priority order."""
    roots: list[tuple[str, Path]] = []
    project = project_skills_root()
    if project is not None:
        roots.append(("project", project))
    roots.append(("user", user_skills_root()))
    for plugin_id, plugin_dir in plugin_loader.iter_plugin_dirs(
        plugin_loader.plugins_builtin_root(), plugin_loader.plugins_user_root(),
        plugin_loader.load_lockfile(),
    ):
        roots.append((f"plugin:{plugin_id}", plugin_dir / "skills"))
    roots.append(("builtin", settings.SKILLS_BUILTIN_DIR))
    for extra in extra_roots():
        roots.append(("extra", extra))
    return roots


# ── enabled state ──────────────────────────────────────────────────────────

def _state_path() -> Path:
    return user_skills_root() / _STATE_FILE


def disabled_ids() -> set[str]:
    try:
        data = json.loads(_state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    items = data.get("disabled") if isinstance(data, dict) else None
    return {item for item in items if isinstance(item, str)} if isinstance(items, list) else set()


def set_enabled(skill_id: str, enabled: bool) -> None:
    disabled = disabled_ids()
    if enabled:
        disabled.discard(skill_id)
    else:
        disabled.add(skill_id)
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"disabled": sorted(disabled)}, indent=2) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


# ── discovery ──────────────────────────────────────────────────────────────

def _kind(relative: PurePosixPath) -> str:
    return _KIND_BY_DIR.get(relative.parts[0], "other") if len(relative.parts) > 1 else "other"


def list_files(folder: Path) -> list[dict[str, Any]]:
    """Every file in a skill but ``SKILL.md``, as ``{path, size, kind}``.

    Symbolic links are not followed into directories, and a linked file is
    listed only when it resolves inside the skill: what cannot be read is
    not offered.
    """
    out: list[dict[str, Any]] = []
    try:
        base = folder.resolve()
    except OSError:
        return out
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in _SKIP_DIRS and not d.startswith("."))
        current = Path(dirpath)
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            full = current / name
            relative = PurePosixPath(full.relative_to(base).as_posix())
            if str(relative) == SKILL_FILE:
                continue
            resolved = resolve_under(base, str(relative))
            if resolved is None:
                continue
            try:
                if not resolved.is_file():
                    continue
                size = resolved.stat().st_size
            except OSError:
                continue
            out.append({"path": str(relative), "size": size, "kind": _kind(relative)})
            if len(out) >= MAX_LISTED_FILES:
                return out
    return out


def _read_skill_md(path: Path) -> tuple[str | None, Issue | None]:
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_SKILL_FILE_BYTES + 1)
    except OSError as exc:
        return None, Issue("error", "unreadable", f"SKILL.md could not be read: {exc.strerror}.")
    if len(raw) > MAX_SKILL_FILE_BYTES:
        return None, Issue("error", "too_large",
                           f"SKILL.md is larger than {MAX_SKILL_FILE_BYTES // 1024} KB.")
    try:
        return raw.decode("utf-8"), None
    except UnicodeDecodeError:
        return None, Issue("error", "not_utf8", "SKILL.md is not UTF-8 text.")


def _unsafe(source: str, folder: Path, message: str) -> Skill:
    parsed = ParsedSkill(name=folder.name,
                         issues=[Issue("error", "unsafe_link", message)])
    return Skill(id=f"{source}/{folder.name}", source=source, folder=folder,
                 parsed=parsed, readable=False)


def load_skill(source: str, folder: Path, *, root: Path | None = None) -> Skill:
    """One skill folder, never raising for what is inside it.

    *root* is the source's ``skills/`` folder. For a ``project`` or
    ``plugin:*`` source the skill folder must resolve inside it.
    """
    try:
        real = folder.resolve()
        # ``root.parent`` is the project or plugin folder: a ``skills/`` that
        # is itself a link out of it leads out just the same.
        if root is not None and _is_contained_source(source) and (
                not real.is_relative_to(root.resolve())
                or not root.resolve().is_relative_to(root.parent.resolve())):
            return _unsafe(source, folder,
                           "The skill folder is a link that leads outside the "
                           "project's or plugin's skills folder; it is not read.")
        if not (folder / SKILL_FILE).resolve().is_relative_to(real):
            return _unsafe(source, folder,
                           "SKILL.md is a link that leads outside the skill "
                           "folder; it is not read.")
    except (OSError, RuntimeError):
        return _unsafe(source, folder, "The skill folder could not be resolved.")
    text, problem = _read_skill_md(folder / SKILL_FILE)
    if text is None:
        parsed = ParsedSkill(name=folder.name, issues=[problem] if problem else [])
    else:
        parsed = parse_skill(text, folder.name)
    return Skill(id=f"{source}/{folder.name}", source=source, folder=folder,
                 parsed=parsed, files=list_files(folder))


def _skill_folders(root: Path) -> list[Path]:
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return []
    folders = []
    for entry in entries:
        if entry.name.startswith("."):
            continue
        try:
            if entry.is_dir() and (entry / SKILL_FILE).is_file():
                folders.append(entry)
        except OSError:
            continue
    return folders


def discover() -> list[Skill]:
    """Every skill in every source, in priority order, with ``enabled`` and
    ``shadowed_by`` filled in."""
    disabled = disabled_ids()
    skills: list[Skill] = []
    seen_ids: set[str] = set()
    for source, root in source_roots():
        for folder in _skill_folders(root):
            skill = load_skill(source, folder, root=root)
            if skill.id in seen_ids:
                # Two ``extra`` directories holding one name: the first is
                # the one its id names.
                continue
            seen_ids.add(skill.id)
            skill.enabled = skill.id not in disabled
            skills.append(skill)
    winners: dict[str, str] = {}
    for skill in skills:
        if skill.valid and skill.enabled and skill.name not in winners:
            winners[skill.name] = skill.id
    for skill in skills:
        winner = winners.get(skill.name)
        if winner is not None and winner != skill.id:
            skill.shadowed_by = winner
    return skills


def get_skill(skill_id: str) -> Skill:
    for skill in discover():
        if skill.id == skill_id:
            return skill
    raise SkillNotFound(skill_id)


def active_skills() -> list[Skill]:
    """The skills an agent is told about: enabled, valid, not shadowed."""
    return [s for s in discover() if s.enabled and s.valid and s.shadowed_by is None]


# ── reading one file ───────────────────────────────────────────────────────

def _looks_absolute(path: str) -> bool:
    return (path.startswith(("/", "\\"))
            or (len(path) >= 2 and path[1] == ":")
            or PurePosixPath(path).is_absolute())


def resolve_skill_file(skill: Skill, path: str) -> Path:
    """*path* inside *skill*, or :class:`SkillPathRefused`.

    Refused: an absolute path, a ``..`` segment, and anything that resolves
    outside the skill's folder -- a symbolic link included, since the
    comparison is made after ``resolve``.
    """
    if not isinstance(path, str) or not path.strip() or "\x00" in path:
        raise SkillPathRefused("The file path is empty or not a path.")
    if _looks_absolute(path):
        raise SkillPathRefused("The file path must be relative to the skill.")
    if not skill.readable:
        raise SkillPathRefused("This skill is not read: a link makes it unsafe.")
    segments = [seg for seg in path.replace("\\", "/").split("/") if seg]
    if ".." in segments:
        raise SkillPathRefused("The file path may not contain '..'.")
    if any(seg.startswith(".") or seg in _SKIP_DIRS for seg in segments[:-1]) \
            or (segments and segments[-1].startswith(".")):
        raise SkillPathRefused("Hidden files and folders such as .git are not served.")
    resolved = resolve_under(skill.folder, path)
    if resolved is None:
        raise SkillPathRefused("The file path leaves the skill's folder.")
    return resolved


def _decode_text(raw: bytes, truncated: bool) -> str | None:
    """*raw* as UTF-8, or None for a binary file. A cut can split the last
    character, so a truncated read may drop up to three trailing bytes."""
    if b"\x00" in raw[:8192]:
        return None
    for cut in range(0, 4 if truncated else 1):
        try:
            return (raw[:-cut] if cut else raw).decode("utf-8")
        except UnicodeDecodeError:
            continue
    return None


def read_file(skill: Skill, path: str) -> dict[str, Any]:
    """``{path, content, truncated}`` for one text file of *skill*.

    :raises SkillPathRefused: the path is not inside the skill.
    :raises FileNotFoundError: there is no file there.
    :raises SkillFileNotText: the file is binary (or not UTF-8).
    """
    resolved = resolve_skill_file(skill, path)
    if not resolved.is_file():
        raise FileNotFoundError(path)
    with resolved.open("rb") as handle:
        raw = handle.read(MAX_READ_BYTES + 1)
    truncated = len(raw) > MAX_READ_BYTES
    raw = raw[:MAX_READ_BYTES]
    text = _decode_text(raw, truncated)
    if text is None:
        raise SkillFileNotText(path)
    relative = resolved.relative_to(skill.folder.resolve()).as_posix()
    return {"path": relative, "content": text, "truncated": truncated}
