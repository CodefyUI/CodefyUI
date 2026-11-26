"""Installing a skill into the ``user`` source.

Three kinds of source, told apart by shape:

- an HTTPS git URL (any host, github.com included), optionally ``@ref``:
  cloned with :func:`app.core.plugins.git_source.clone`, which runs git
  with prompts off, only https, a deadline and a size watch;
- a ``.zip``, ``.tar.gz`` or ``.tgz`` on the server's disk;
- a folder on the server's disk.

Every kind is first made into a private copy under a staging directory
(``<USER_DATA>/skills/.staging-*``, which the listing skips because of its
dot), under this module's caps: :data:`MAX_FILES` files and
:data:`MAX_TOTAL_BYTES` bytes, no symbolic links, nothing that would land
outside the copy. Only then is ``SKILL.md`` looked for and read, and only a
skill whose frontmatter is valid -- a description, and a ``name`` that
follows the specification -- is moved into place, under that name. A name
already installed is refused (409), never overwritten. Whatever happens,
the staging directory is removed, so a refused source leaves nothing.

Nothing in the skill is run: no script, no dependency install, no git
hook (a clone runs none). The skill is text an agent may read.
"""

from __future__ import annotations

import os
import shutil
import stat
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from ..git.errors import redact
from ..plugins import git_source
from ..plugins.errors import GitError, GitUnavailable, PluginCancelled, SourceRefused
from ..plugins.local import path_from_spec
from ..plugins.sources import parse_git_url
from .registry import load_skill, user_skills_root
from .spec import SKILL_FILE, declared_name, name_problem

MAX_FILES = 500
MAX_TOTAL_BYTES = 20 * 1024 * 1024
_CHUNK = 64 * 1024

_SKIP_DIRS = frozenset({".git", ".hg", ".svn", "__pycache__", "node_modules",
                        ".venv", "venv"})
_SKIP_FILES = frozenset({".DS_Store", "Thumbs.db"})
_IGNORED_TOP_LEVEL = frozenset({"__MACOSX", ".DS_Store"})


class SkillInstallError(Exception):
    """A refused install. ``status`` is the HTTP answer, ``code`` what a
    client branches on, ``hint`` the detail behind the one-line message."""

    def __init__(self, message: str, *, code: str, status: int = 400,
                 hint: str | None = None):
        super().__init__(message)
        self.code = code
        self.status = status
        self.hint = hint

    def as_detail(self) -> dict[str, str | None]:
        return {"code": self.code, "message": redact(str(self)),
                "hint": redact(self.hint) if self.hint else None}


class _Budget:
    """The running file count and byte total of one copy."""

    def __init__(self, label: str):
        self.label = label
        self.files = 0
        self.bytes = 0

    def add_file(self, size: int = 0) -> None:
        self.files += 1
        if self.files > MAX_FILES:
            raise SkillInstallError(
                f"{self.label} holds more than {MAX_FILES} files.",
                code="too_many_files")
        self.add_bytes(size)

    def add_bytes(self, size: int) -> None:
        self.bytes += size
        if self.bytes > MAX_TOTAL_BYTES:
            raise SkillInstallError(
                f"{self.label} is larger than {MAX_TOTAL_BYTES // (1024 * 1024)} MB.",
                code="too_large")


# ── materialising a source ────────────────────────────────────────────────

def _copy_folder(source: Path, dest: Path, label: str) -> None:
    budget = _Budget(label)
    dest.mkdir(parents=True)
    for dirpath, dirnames, filenames in os.walk(source, followlinks=False):
        current = Path(dirpath)
        target_dir = dest / current.relative_to(source)
        keep = []
        for name in sorted(dirnames):
            if name in _SKIP_DIRS:
                continue
            if (current / name).is_symlink():
                raise SkillInstallError(
                    f"{(current / name).relative_to(source)} is a symbolic link.",
                    code="unsafe_member",
                    hint="A skill may not contain symbolic links.")
            keep.append(name)
        dirnames[:] = keep
        for name in keep:
            (target_dir / name).mkdir()
        for name in sorted(filenames):
            if name in _SKIP_FILES:
                continue
            path = current / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise SkillInstallError(
                    f"{path.relative_to(source)} is not a regular file.",
                    code="unsafe_member",
                    hint="A skill may hold files and folders only, no links.")
            budget.add_file(path.stat().st_size)
            shutil.copyfile(path, target_dir / name)


def _zip_member(name: str) -> PurePosixPath | None:
    """The relative path a zip member names, or None when it would escape."""
    normalized = name.replace("\\", "/")
    if normalized.startswith("/") or (len(normalized) > 1 and normalized[1] == ":"):
        return None
    parts = [p for p in normalized.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        return None
    return PurePosixPath(*parts) if parts else PurePosixPath()


def _extract_zip(archive: Path, dest: Path) -> None:
    budget = _Budget(archive.name)
    try:
        with zipfile.ZipFile(archive) as zf:
            infos = zf.infolist()
            files = [i for i in infos if not i.is_dir()]
            if len(files) > MAX_FILES:
                raise SkillInstallError(
                    f"{archive.name} holds more than {MAX_FILES} files.",
                    code="too_many_files")
            if sum(i.file_size for i in files) > MAX_TOTAL_BYTES:
                raise SkillInstallError(
                    f"{archive.name} unpacks to more than "
                    f"{MAX_TOTAL_BYTES // (1024 * 1024)} MB.", code="too_large")
            root = dest.resolve()
            for info in infos:
                relative = _zip_member(info.filename)
                mode = (info.external_attr >> 16) & 0o170000
                if relative is None or mode not in (0, stat.S_IFREG, stat.S_IFDIR):
                    raise SkillInstallError(
                        f"{archive.name} holds a member this build will not unpack.",
                        code="unsafe_member",
                        hint=f"{info.filename!r}: a skill archive may not write "
                             f"outside its folder or carry links.")
                if not relative.parts:
                    continue
                target = root.joinpath(*relative.parts)
                if not target.resolve().is_relative_to(root):
                    raise SkillInstallError(
                        f"{archive.name} holds a member this build will not unpack.",
                        code="unsafe_member", hint=repr(info.filename))
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                budget.add_file()
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
    try:
        with tarfile.open(archive, "r:gz") as tf:
            members = tf.getmembers()
            for member in members:
                if not (member.isfile() or member.isdir()):
                    raise SkillInstallError(
                        f"{archive.name} holds a member this build will not unpack.",
                        code="unsafe_member",
                        hint=f"{member.name!r}: a skill archive may hold files "
                             f"and folders only, no links or devices.")
            files = [m for m in members if m.isfile()]
            if len(files) > MAX_FILES:
                raise SkillInstallError(
                    f"{archive.name} holds more than {MAX_FILES} files.",
                    code="too_many_files")
            if sum(m.size for m in files) > MAX_TOTAL_BYTES:
                raise SkillInstallError(
                    f"{archive.name} unpacks to more than "
                    f"{MAX_TOTAL_BYTES // (1024 * 1024)} MB.", code="too_large")
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


def _clone(url: str, ref: str, staging: Path) -> Path:
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
    tree = staging / "tree"
    _copy_folder(clone_dir, tree, url)
    return tree


def materialize(source: str, staging: Path) -> Path:
    """A private copy of *source* under *staging*; returns its root."""
    git = parse_git_url(source.strip(), allow_github=True)
    if git is not None:
        return _clone(*git, staging)
    if source.strip().lower().startswith(("http://", "https://", "git@", "ssh://")):
        raise SkillInstallError(
            "Only an https URL of a git repository (optionally ending in @ref) "
            "can be installed from the network.",
            code="not_supported",
            hint="Download an archive and install the file, or clone the "
                 "repository and install its folder.")
    path = Path(path_from_spec(source.strip()))
    if path.is_dir():
        tree = staging / "tree"
        _copy_folder(path, tree, path.name)
        return tree
    if path.is_file():
        kind = _archive_kind(path)
        if kind is None:
            raise SkillInstallError(
                f"{path.name} is not an archive this build unpacks.",
                code="not_supported", hint="Use a .zip, .tar.gz or .tgz file, or a folder.")
        if path.stat().st_size > MAX_TOTAL_BYTES:
            raise SkillInstallError(
                f"{path.name} is larger than {MAX_TOTAL_BYTES // (1024 * 1024)} MB.",
                code="too_large")
        tree = staging / "tree"
        tree.mkdir()
        (_extract_zip if kind == "zip" else _extract_tar)(path, tree)
        return tree
    raise SkillInstallError(
        f"Nothing at {source}.", code="not_found",
        hint="The path is read on the machine the CodefyUI server runs on.")


def find_skill_root(tree: Path) -> Path:
    """The folder holding ``SKILL.md``: the copy's root, or its one
    top-level directory (an archive of the folder rather than its contents)."""
    if (tree / SKILL_FILE).is_file():
        return tree
    roots = [p for p in sorted(tree.iterdir())
             if p.is_dir() and p.name not in _IGNORED_TOP_LEVEL]
    if len(roots) == 1 and (roots[0] / SKILL_FILE).is_file():
        return roots[0]
    raise SkillInstallError(
        "The source has no SKILL.md at its root.",
        code="no_skill",
        hint="A skill is a folder with SKILL.md in it. Install that folder, "
             "or an archive of it.")


# ── the install ───────────────────────────────────────────────────────────

def install(source: str) -> dict:
    """Install *source* into the user source; returns the new skill's summary.

    :raises SkillInstallError: with ``status`` 400 for a refused source and
        409 when a skill of that name is already installed.
    """
    if not isinstance(source, str) or not source.strip():
        raise SkillInstallError("No source was given.", code="no_source")
    root = user_skills_root()
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=root))
    try:
        tree = materialize(source, staging)
        skill_root = find_skill_root(tree)
        candidate = load_skill("user", skill_root)
        if not candidate.valid:
            errors = "; ".join(i.message for i in candidate.parsed.issues
                               if i.level == "error")
            raise SkillInstallError("SKILL.md is not a valid skill.",
                                    code="invalid_skill", hint=errors)
        name = _declared_name(skill_root)
        problem = name_problem(name) if isinstance(name, str) else "it is missing"
        if problem is not None:
            raise SkillInstallError(
                "SKILL.md has no valid 'name'.", code="invalid_name",
                hint=f"The name names the installed folder, and {problem}.")
        target = root / name
        if target.exists():
            raise SkillInstallError(
                f"A skill named '{name}' is already installed.",
                code="exists", status=409,
                hint="Remove it first to install this one in its place.")
        os.replace(skill_root, target)
        return load_skill("user", target).summary()
    finally:
        git_source._remove_tree(staging)


def _declared_name(skill_root: Path):
    try:
        text = (skill_root / SKILL_FILE).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return declared_name(text)


def remove(skill_folder: Path) -> None:
    """Delete an installed skill's folder."""
    git_source._remove_tree(skill_folder)
