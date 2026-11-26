"""Installing skills into the ``user`` source.

Three kinds of source, told apart by shape:

- an HTTPS git URL (any host, github.com included), optionally ``@ref`` and
  ``#<sub/path>``: cloned with :func:`app.core.plugins.git_source.clone`,
  which runs git with prompts off, only https, a deadline and a size watch;
- a ``.zip``, ``.tar.gz`` or ``.tgz`` on the server's disk;
- a folder on the server's disk.

A source may hold one skill or many. Where they are is looked up in this
order (:func:`find_skill_roots`), without following links:

1. ``SKILL.md`` at the source's root: one skill;
2. one top-level folder with ``SKILL.md`` in it: one skill (an archive of
   the folder rather than of its contents);
3. ``skills/*/SKILL.md``, ``.claude/skills/*/SKILL.md`` and
   ``.agents/skills/*/SKILL.md``: every one is a skill (a skill repository
   or a Claude Code plugin). Also looked for inside a single top-level
   folder, which is how a downloaded repository archive is shaped.

A folder source is scanned in place and only the skill folders found are
copied. An archive is unpacked, and a git source cloned, into a staging
directory (``<USER_DATA>/skills/.staging-*``, which the listing skips because
of its dot) under a whole-source cap of :data:`MAX_SOURCE_ENTRIES` entries and
:data:`MAX_SOURCE_BYTES` bytes. Each skill is then copied on its own under
:data:`MAX_FILES` entries (files and folders) and :data:`MAX_TOTAL_BYTES`
bytes, with no symbolic link and nothing outside the copy, validated, and
moved into place under the ``name`` in its frontmatter. A name already
installed is skipped (``exists``), never overwritten; one invalid skill does
not stop the others. Whatever happens, the staging directory is removed.

Nothing in a skill is run: no script, no dependency install, no git hook
(a clone runs none, and git-lfs filters are switched off). The skill is text
an agent may read.
"""

from __future__ import annotations

import os
import shutil
import stat
import tarfile
import tempfile
import threading
import urllib.parse
import zipfile
from pathlib import Path, PurePosixPath

from ..data_paths import resolve_under
from ..git.errors import redact
from ..plugins import git_source
from ..plugins.errors import GitError, GitUnavailable, PluginCancelled, SourceRefused
from ..plugins.local import path_from_spec
from ..plugins.sources import parse_git_url
from .registry import load_skill, user_skills_root
from .spec import SKILL_FILE, declared_name, name_problem

#: Caps on ONE installed skill: entries (files and folders) and bytes.
MAX_FILES = 500
MAX_TOTAL_BYTES = 20 * 1024 * 1024
#: Caps on a whole unpacked archive or clone, which may hold many skills.
MAX_SOURCE_ENTRIES = 5000
MAX_SOURCE_BYTES = 100 * 1024 * 1024
_CHUNK = 64 * 1024

_SKIP_DIRS = frozenset({".git", ".hg", ".svn", "__pycache__", "node_modules",
                        ".venv", "venv"})
_SKIP_FILES = frozenset({".DS_Store", "Thumbs.db"})
_IGNORED_TOP_LEVEL = frozenset({"__MACOSX", ".DS_Store"})
#: Where a repository keeps several skills, relative to its root.
_COLLECTIONS = ("skills", ".claude/skills", ".agents/skills")

#: One install at a time: a clone can hold a worker thread for minutes.
_install_lock = threading.Lock()


class SkillInstallError(Exception):
    """A refused install. ``status`` is the HTTP answer, ``code`` what a
    client branches on, ``hint`` the detail behind the one-line message."""

    def __init__(self, message: str, *, code: str, status: int = 400,
                 hint: str | None = None, skipped: list[dict] | None = None):
        super().__init__(message)
        self.code = code
        self.status = status
        self.hint = hint
        self.skipped = skipped

    def as_detail(self) -> dict:
        detail: dict = {"code": self.code, "message": redact(str(self)),
                        "hint": redact(self.hint) if self.hint else None}
        if self.skipped is not None:
            detail["skipped"] = self.skipped
        return detail


class _Budget:
    """The running entry count and byte total of one copy or unpack."""

    def __init__(self, label: str, *, max_entries: int, max_bytes: int):
        self.label = label
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.entries = 0
        self.bytes = 0

    def add_entry(self, size: int = 0) -> None:
        """One file or folder, plus its size."""
        self.entries += 1
        if self.entries > self.max_entries:
            raise SkillInstallError(
                f"{self.label} holds more than {self.max_entries} files and folders.",
                code="too_many_files")
        self.add_bytes(size)

    def add_bytes(self, size: int) -> None:
        self.bytes += size
        if self.bytes > self.max_bytes:
            raise SkillInstallError(
                f"{self.label} is larger than {self.max_bytes // (1024 * 1024)} MB.",
                code="too_large")


def _skill_budget(label: str) -> _Budget:
    return _Budget(label, max_entries=MAX_FILES, max_bytes=MAX_TOTAL_BYTES)


def _source_budget(label: str) -> _Budget:
    return _Budget(label, max_entries=MAX_SOURCE_ENTRIES, max_bytes=MAX_SOURCE_BYTES)


# ── copying, unpacking, counting ──────────────────────────────────────────

def _unsafe(what: str, hint: str = "A skill may hold files and folders only, no links.") -> SkillInstallError:
    return SkillInstallError(f"{what} is not a regular file or folder.",
                             code="unsafe_member", hint=hint)


def _copy_folder(source: Path, dest: Path, budget: _Budget) -> None:
    """Copy one skill folder, refusing links and counting every entry."""
    dest.mkdir(parents=True)
    for dirpath, dirnames, filenames in os.walk(source, followlinks=False):
        current = Path(dirpath)
        target_dir = dest / current.relative_to(source)
        keep = []
        for name in sorted(dirnames):
            if name in _SKIP_DIRS:
                continue
            if (current / name).is_symlink():
                raise _unsafe(str((current / name).relative_to(source)))
            budget.add_entry()
            keep.append(name)
        dirnames[:] = keep
        for name in keep:
            (target_dir / name).mkdir()
        for name in sorted(filenames):
            if name in _SKIP_FILES:
                continue
            path = current / name
            mode = path.lstat().st_mode
            if not stat.S_ISREG(mode):
                raise _unsafe(str(path.relative_to(source)))
            budget.add_entry(path.lstat().st_size)
            shutil.copyfile(path, target_dir / name, follow_symlinks=False)


def _count_tree(root: Path, budget: _Budget) -> None:
    """Hold an already-written tree (a clone) to *budget*."""
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for _ in dirnames:
            budget.add_entry()
        for name in filenames:
            try:
                budget.add_entry(os.lstat(os.path.join(dirpath, name)).st_size)
            except OSError:
                budget.add_entry()


def _archive_member(name: str) -> PurePosixPath | None:
    """The relative path an archive member names, or None when it would
    escape: absolute, a drive, or a ``..`` segment. Refused, never cleaned."""
    normalized = name.replace("\\", "/")
    if normalized.startswith("/") or (len(normalized) > 1 and normalized[1] == ":"):
        return None
    parts = [p for p in normalized.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        return None
    return PurePosixPath(*parts) if parts else PurePosixPath()


def _member_refused(archive: Path, name: str) -> SkillInstallError:
    return SkillInstallError(
        f"{archive.name} holds a member this build will not unpack.",
        code="unsafe_member",
        hint=f"{name!r}: a skill archive may not write outside its folder or "
             f"carry links or devices.")


def _extract_zip(archive: Path, dest: Path) -> None:
    budget = _source_budget(archive.name)
    try:
        with zipfile.ZipFile(archive) as zf:
            infos = zf.infolist()
            if len(infos) > MAX_SOURCE_ENTRIES:
                raise SkillInstallError(
                    f"{archive.name} holds more than {MAX_SOURCE_ENTRIES} files and folders.",
                    code="too_many_files")
            if sum(i.file_size for i in infos) > MAX_SOURCE_BYTES:
                raise SkillInstallError(
                    f"{archive.name} unpacks to more than "
                    f"{MAX_SOURCE_BYTES // (1024 * 1024)} MB.", code="too_large")
            root = dest.resolve()
            for info in infos:
                relative = _archive_member(info.filename)
                mode = (info.external_attr >> 16) & 0o170000
                if relative is None or mode not in (0, stat.S_IFREG, stat.S_IFDIR):
                    raise _member_refused(archive, info.filename)
                if not relative.parts:
                    continue
                target = root.joinpath(*relative.parts)
                if not target.resolve().is_relative_to(root):
                    raise _member_refused(archive, info.filename)
                budget.add_entry()
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, target.open("wb") as out:
                    while chunk := src.read(_CHUNK):
                        budget.add_bytes(len(chunk))
                        out.write(chunk)
    except (zipfile.BadZipFile, zipfile.LargeZipFile, EOFError, OSError,
            NotImplementedError, RuntimeError) as exc:
        raise SkillInstallError(f"{archive.name} could not be unpacked.",
                                code="unreadable", hint=str(exc)) from exc


def _extract_tar(archive: Path, dest: Path) -> None:
    """Walk the tar one member at a time, stopping at the first refusal or
    the first member past the caps, before anything is written."""
    budget = _source_budget(archive.name)
    members: list[tarfile.TarInfo] = []
    try:
        with tarfile.open(archive, "r:gz") as tf:
            for member in tf:
                if not (member.isfile() or member.isdir()) \
                        or _archive_member(member.name) is None:
                    raise _member_refused(archive, member.name)
                budget.add_entry(member.size if member.isfile() else 0)
                members.append(member)
            tf.extractall(dest, members=members, filter="data")
    except tarfile.FilterError as exc:
        raise SkillInstallError(
            f"{archive.name} holds a member this build will not unpack.",
            code="unsafe_member", hint=str(exc)) from exc
    except (tarfile.TarError, EOFError, OSError) as exc:
        raise SkillInstallError(f"{archive.name} could not be unpacked.",
                                code="unreadable", hint=str(exc)) from exc


def _archive_kind(path: Path) -> str | None:
    name = path.name.lower()
    if name.endswith(".zip"):
        return "zip"
    if name.endswith((".tar.gz", ".tgz")):
        return "tar"
    return None


# ── finding the skills in a source ────────────────────────────────────────

def _is_dir_not_link(path: Path) -> bool:
    try:
        return stat.S_ISDIR(path.lstat().st_mode)
    except OSError:
        return False


def _has_skill_file(folder: Path) -> bool:
    try:
        return stat.S_ISREG((folder / SKILL_FILE).lstat().st_mode)
    except OSError:
        return False


def _subdirs(folder: Path) -> list[Path]:
    try:
        entries = sorted(folder.iterdir())
    except OSError:
        return []
    return [e for e in entries if e.name not in _IGNORED_TOP_LEVEL
            and not e.name.startswith(".") and _is_dir_not_link(e)]


def _collections(base: Path) -> list[Path]:
    found: list[Path] = []
    for relative in _COLLECTIONS:
        folder = base
        for part in relative.split("/"):
            folder = folder / part
            if not _is_dir_not_link(folder):
                break
        else:
            found.extend(d for d in _subdirs(folder) if _has_skill_file(d))
    return found


def find_skill_roots(base: Path) -> list[Path]:
    """Every skill folder in *base*, in the order the module docstring gives.
    Never follows a link below *base*."""
    if _has_skill_file(base):
        return [base]
    tops = _subdirs(base)
    if len(tops) == 1:
        if _has_skill_file(tops[0]):
            return [tops[0]]
        inside = _collections(tops[0])
        if inside:
            return inside
    return _collections(base)


# ── locating a source ─────────────────────────────────────────────────────

def _remote_path(source: str) -> bool:
    """A UNC path or a ``file://`` URL with a host: looking one up makes the
    server open an SMB connection (and, on Windows, offer its NTLM hash)."""
    if source.startswith(("\\\\", "//")):
        return True
    if source.lower().startswith("file:"):
        host = urllib.parse.urlsplit(source).netloc
        return host not in ("", "localhost")
    return False


def _check_subpath(sub: str) -> None:
    segments = [seg for seg in sub.replace("\\", "/").split("/") if seg not in ("", ".")]
    if (not segments or sub.startswith(("/", "\\")) or ":" in sub or "\x00" in sub
            or any(seg == ".." or seg in _SKIP_DIRS for seg in segments)):
        raise SkillInstallError(
            f"'{sub}' is not a folder path inside the repository.",
            code="bad_subpath",
            hint="Use a relative path such as skills/my-skill, with no '..'.")


def _clone(url: str, ref: str, sub: str, staging: Path) -> Path:
    clone_dir = staging / "clone"
    try:
        git_source.clone(url, ref, clone_dir, allow_github=True)
    except GitUnavailable as exc:
        raise SkillInstallError(str(exc), code="git_unavailable", hint=exc.hint) from exc
    except GitError as exc:
        raise SkillInstallError(str(exc), code=f"git_{exc.reason}", hint=exc.hint) from exc
    except SourceRefused as exc:
        raise SkillInstallError(str(exc), code=exc.reason, hint=exc.hint) from exc
    except PluginCancelled as exc:  # pragma: no cover - no cancel is wired
        raise SkillInstallError("The clone was cancelled.", code="cancelled") from exc
    _count_tree(clone_dir, _source_budget(url))
    if not sub:
        return clone_dir
    target = resolve_under(clone_dir, sub)
    if target is None or not _is_dir_not_link(clone_dir / sub) or not target.is_dir():
        raise SkillInstallError(f"The repository has no folder '{sub}'.",
                                code="bad_subpath")
    return target


def locate(source: str, staging: Path) -> Path:
    """The folder to look for skills in: the source folder itself, or the
    unpacked archive or clone under *staging*."""
    raw = source.strip()
    url_part, hash_mark, sub = raw.partition("#")
    git = parse_git_url(url_part, allow_github=True)
    if git is not None:
        if hash_mark:
            _check_subpath(sub)
        return _clone(git[0], git[1], sub if hash_mark else "", staging)
    if raw.lower().startswith(("http://", "https://", "git@", "ssh://")):
        raise SkillInstallError(
            "Only an https URL of a git repository (optionally ending in @ref "
            "and #sub/folder) can be installed from the network.",
            code="not_supported",
            hint="Download an archive and install the file, or clone the "
                 "repository and install its folder.")
    if _remote_path(raw):
        raise SkillInstallError(
            "A network share is not a source this server reads.",
            code="not_supported",
            hint="Copy the skill to a local folder first.")
    path = Path(path_from_spec(raw))
    if path.is_dir():
        return path
    if path.is_file():
        kind = _archive_kind(path)
        if kind is None:
            raise SkillInstallError(
                f"{path.name} is not an archive this build unpacks.",
                code="not_supported", hint="Use a .zip, .tar.gz or .tgz file, or a folder.")
        if path.stat().st_size > MAX_SOURCE_BYTES:
            raise SkillInstallError(
                f"{path.name} is larger than {MAX_SOURCE_BYTES // (1024 * 1024)} MB.",
                code="too_large")
        tree = staging / "extracted"
        tree.mkdir()
        (_extract_zip if kind == "zip" else _extract_tar)(path, tree)
        return tree
    raise SkillInstallError(
        f"Nothing at {raw}.", code="not_found",
        hint="The path is read on the machine the CodefyUI server runs on.")


# ── the install ───────────────────────────────────────────────────────────

def _declared(skill_root: Path):
    try:
        return declared_name((skill_root / SKILL_FILE).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return None


def _install_one(source: Path, work: Path, root: Path, taken: set[str]) -> dict:
    """Copy, check and move one skill folder; returns its summary."""
    _copy_folder(source, work, _skill_budget(source.name))
    candidate = load_skill("user", work)
    if not candidate.valid:
        errors = "; ".join(i.message for i in candidate.parsed.issues if i.level == "error")
        raise SkillInstallError("SKILL.md is not a valid skill.",
                                code="invalid_skill", hint=errors)
    name = _declared(work)
    problem = name_problem(name) if isinstance(name, str) else "it is missing"
    if problem is not None:
        raise SkillInstallError(
            "SKILL.md has no valid 'name'.", code="invalid_name",
            hint=f"The name names the installed folder, and {problem}.")
    if name in taken:
        raise SkillInstallError(f"The source holds two skills named '{name}'.",
                                code="duplicate")
    target = root / name
    if target.exists() or target.is_symlink():
        raise SkillInstallError(
            f"A skill named '{name}' is already installed.",
            code="exists", status=409,
            hint="Remove it first to install this one in its place.")
    os.replace(work, target)
    taken.add(name)
    return load_skill("user", target).summary()


def install(source: str) -> dict:
    """Install every skill in *source*: ``{"installed": [...], "skipped": [...]}``.

    :raises SkillInstallError: 400 for a refused source; 409 ``exists`` when
        every skill found is already installed; 409 ``busy`` while another
        install runs.
    """
    if not isinstance(source, str) or not source.strip():
        raise SkillInstallError("No source was given.", code="no_source")
    if not _install_lock.acquire(blocking=False):
        raise SkillInstallError("Another skill install is running.", code="busy",
                                status=409, hint="Try again when it has finished.")
    try:
        return _install(source)
    finally:
        _install_lock.release()


def _install(source: str) -> dict:
    root = user_skills_root()
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=root))
    try:
        base = locate(source, staging)
        found = find_skill_roots(base)
        if not found:
            raise SkillInstallError(
                "The source has no SKILL.md where a skill is looked for.",
                code="no_skill",
                hint="A skill is a folder with SKILL.md in it: at the source's "
                     "root, in its one top-level folder, or under skills/, "
                     ".claude/skills/ or .agents/skills/.")
        installed: list[dict] = []
        skipped: list[dict] = []
        taken: set[str] = set()
        for index, folder in enumerate(found):
            try:
                installed.append(_install_one(folder, staging / f"skill-{index}", root, taken))
            except SkillInstallError as exc:
                if len(found) == 1:
                    raise
                skipped.append({"name": folder.name, "code": exc.code,
                                "message": redact(str(exc))})
        if not installed:
            every_exists = all(s["code"] == "exists" for s in skipped)
            raise SkillInstallError(
                "No skill in the source could be installed.",
                code="exists" if every_exists else "nothing_installed",
                status=409 if every_exists else 400, skipped=skipped)
        return {"installed": installed, "skipped": skipped}
    finally:
        git_source._remove_tree(staging)


def remove(skill_folder: Path) -> None:
    """Delete an installed skill.

    A skill linked into the user folder by hand (``ln -s ~/.claude/skills/x``)
    loses only its link: the folder it points at is the user's own and is
    left exactly as it was. A real folder is deleted only when it sits
    directly in the user skills folder, and a deletion that leaves anything
    behind is an error, never a success.
    """
    root = user_skills_root()
    if skill_folder.is_symlink():
        skill_folder.unlink()
        return
    if skill_folder.resolve().parent != root.resolve():
        raise SkillInstallError("That folder is not an installed skill.",
                                code="not_installed")
    git_source._remove_tree(skill_folder)
    if skill_folder.exists() or skill_folder.is_symlink():
        raise SkillInstallError(f"{skill_folder.name} could not be fully deleted.",
                                code="delete_failed", status=500)
