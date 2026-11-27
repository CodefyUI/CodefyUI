"""Clone a project from an https git URL (phase 4A).

The plugin installer's hardening, reused: https only (``protocol.allow``),
no credential helper, no prompt, no symlinks checked out, no submodules,
the git-lfs smudge filter off, and the same time and unpacked-size caps
(``plugins.git_source``). Unlike a plugin clone this one keeps the whole
history, since the result is a repository the user goes on working in.

The destination is ``<parent>/<repository name>`` and must not exist. A
clone that fails, or that turns out not to be a CodefyUI project (no
``codefyui.project.toml``), is removed -- only the folder this clone
created -- so nothing half-made is left behind.

A cloned project is not trusted: its ``.env`` waits for the user's consent
like any folder this install did not create.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

from .plugins import git_source
from .plugins.errors import GitError
from .plugins.sources import parse_git_url
from .project import PROJECT_MANIFEST


class CloneRefused(Exception):
    """The URL or the destination cannot be used (400 / 409)."""

    def __init__(self, status: int, code: str, message: str,
                 hint: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.hint = hint


@dataclass(frozen=True)
class CloneResult:
    path: Path
    head: str
    #: ``.gitattributes`` routes files through git-lfs; they were not
    #: fetched (the smudge filter is off) and ``git lfs pull`` gets them.
    uses_lfs: bool


_NAME_TAIL = re.compile(r"[^A-Za-z0-9._-]")


def repo_name(url: str) -> str:
    """The folder name a clone of *url* gets: its last path segment without
    ``.git``, as ``git clone`` itself would choose."""
    last = url.rstrip("/").rsplit("/", 1)[-1]
    if last.endswith(".git"):
        last = last[:-4]
    return last


def uses_lfs(root: Path) -> bool:
    try:
        text = (root / ".gitattributes").read_text(encoding="utf-8",
                                                    errors="replace")
    except OSError:
        return False
    return any("filter=lfs" in line and not line.lstrip().startswith("#")
               for line in text.splitlines())


def clone_project(url: str, parent: Path, *, name_check=None) -> CloneResult:
    """Clone *url* into ``parent/<repo name>``. Blocking; run it in a thread.

    *name_check* validates the folder name (the same rule ``create`` uses)
    and returns why it is refused, or None.
    """
    url = url.strip()
    parsed = parse_git_url(url, allow_github=True)
    if parsed is None:
        raise CloneRefused(400, "invalid_url",
                           f"{url!r} is not an https git repository URL")
    url, ref = parsed
    name = repo_name(url)
    reason = name_check(name) if name_check is not None else None
    if reason is not None or not name or _NAME_TAIL.search(name):
        raise CloneRefused(400, "invalid_name",
                           f"the repository name {name!r} cannot be a folder "
                           f"name here{': ' + reason if reason else ''}")
    dest = parent / name
    if dest.exists() or dest.is_symlink():
        raise CloneRefused(409, "exists",
                           f"{dest} already exists; clone into another "
                           "folder or open that one")
    git = git_source.require_git()
    run = {"label": f"Cloning {url}",
           "deadline": time.monotonic() + git_source.CLONE_TIMEOUT_S,
           "cancel_check": None, "watch": dest}
    argv = [git, *git_source._config_args(), "clone", "--quiet",
            "--no-recurse-submodules"]
    if ref:
        argv += ["--branch", ref]
    try:
        code, stderr = git_source._run(
            [*argv, "--", git_source._remote_for(url), str(dest)], **run)
        if code != 0:
            raise git_source._clone_error(url, ref, stderr)
        if not (dest / PROJECT_MANIFEST).is_file():
            raise CloneRefused(
                400, "not_a_project",
                f"{url} has no {PROJECT_MANIFEST} at its root, so it is not a "
                "CodefyUI project; the clone was removed",
                hint=("Add the manifest to the repository (cdui project init "
                      "does), or clone it yourself and adopt it."))
        head = git_source._head(git, dest)
    except BaseException:
        if dest.exists() and not dest.is_symlink():
            git_source._remove_tree(dest)
        raise
    return CloneResult(path=dest.resolve(), head=head, uses_lfs=uses_lfs(dest))


__all__ = ["CloneRefused", "CloneResult", "GitError", "clone_project",
           "repo_name", "uses_lfs"]
