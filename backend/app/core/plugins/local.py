"""A plugin that arrives as a folder or an archive on this machine's disk.

The classroom case: a plugin handed out on a USB stick or a shared drive,
where GitHub is out of reach. Unlike ``cdui plugin link``, which loads the
author's own tree in place and skips the AST gate, this is an INSTALL: the
files are copied into ``<USER_DATA>/plugins/<id>/`` through the whole flow
(manifest, gate, consent, dependencies, staging, lockfile), because a folder
somebody else handed you is third-party code like any repository.

Everything here works on a private COPY. :func:`materialize` copies a folder
(or unpacks an archive) into a working directory first, under the same
member-count and size caps a GitHub tarball gets, and every later step reads
that copy. Reading the USB stick once is what stops a file that changes
between the scan and the copy from being installed unscanned.

The digest is how an install proves it is installing what was reviewed: the
sha256 of the archive file, or of the copied tree (every file's relative
path and bytes, in sorted order). :func:`materialize` computes it on the
copy; the install recomputes it and refuses a mismatch.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from . import github
from .errors import PluginCancelled, SourceRefused

#: Directories a folder copy leaves out: version control, caches, virtual
#: environments and JavaScript dependencies. None of them is part of a
#: plugin, and a ``.venv`` alone can be past the size cap.
SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "node_modules",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
})

#: Files a folder copy leaves out, for the same reason.
SKIP_FILES = frozenset({".DS_Store", "Thumbs.db"})

SourceForm = Literal["folder", "archive"]


@dataclass(frozen=True)
class LocalTree:
    """A private copy of a local source, ready to be read and scanned."""

    root: Path
    form: SourceForm
    digest: str


def path_from_spec(spec: str) -> str:
    """The absolute path a local source spec names, as a string.

    ``file://`` URLs are decoded (``file:///C:/x`` included), ``~`` is
    expanded, and a relative path is made absolute against the current
    directory: the shell's for the CLI, the server's for the Plugin Center.
    Symlinks are not resolved, so the path recorded is the one the user
    typed.
    """
    if spec.lower().startswith("file://"):
        parsed = urllib.parse.urlsplit(spec)
        netloc = parsed.netloc if parsed.netloc not in ("", "localhost") else ""
        raw = urllib.request.url2pathname(parsed.path)
        spec = f"//{netloc}{raw}" if netloc else raw
    return os.path.abspath(os.path.expanduser(spec))


def resolve(path_str: str) -> tuple[Path, SourceForm]:
    """What is at *path_str*: a folder or a supported archive. Reads nothing.

    :raises SourceRefused: ``not_found`` or ``not_supported``.
    """
    path = Path(path_str)
    if path.is_dir():
        return path, "folder"
    if path.is_file():
        if github.archive_format(path) is None:
            raise SourceRefused(
                f"{path.name} is not an archive this build unpacks.",
                reason="not_supported",
                path=path_str,
                hint="Use a .zip, .tar.gz or .tgz file, or the plugin's folder.",
            )
        return path, "archive"
    if path.exists():
        raise SourceRefused(
            f"{path_str} is neither a folder nor a file.",
            reason="not_supported",
            path=path_str,
        )
    raise SourceRefused(
        f"Nothing at {path_str}.",
        reason="not_found",
        path=path_str,
        hint=(
            "The path is read on the machine the CodefyUI server runs on; "
            "check that the drive is mounted and the spelling is right."
        ),
    )


def _file_digest(path: Path) -> str:
    """sha256 of one file's bytes, as hex."""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(github.CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_digest(root: Path) -> str:
    """sha256 over every file under *root*: sorted relative path and bytes.

    The same tree copied twice gives the same digest whatever its
    timestamps say. Only regular files count; a copy made by
    :func:`copy_folder` has nothing else in it.
    """
    entries: list[tuple[str, str]] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            full = Path(dirpath) / name
            entries.append((full.relative_to(root).as_posix(), _file_digest(full)))
    h = hashlib.sha256()
    for relative, digest in sorted(entries):
        h.update(relative.encode("utf-8"))
        h.update(b"\0")
        h.update(digest.encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def copy_folder(
    source: Path,
    dest: Path,
    *,
    cancel_check: Callable[[], bool] | None = None,
) -> None:
    """Copy a plugin folder to *dest* under the archive caps.

    The caps are :data:`~.github.MAX_ARCHIVE_MEMBERS` files and
    :data:`~.github.MAX_EXTRACTED_BYTES` bytes, the ones a GitHub tarball is
    held to, counted as the walk goes so a folder past them is refused
    before it is all read. A symbolic link anywhere in the tree is refused:
    the one in a folder from somebody else can point at any file on this
    machine. :data:`SKIP_DIRS` and :data:`SKIP_FILES` are left out.

    :raises SourceRefused: ``too_many_files``, ``too_large``,
        ``unsafe_member`` or ``unreadable``.
    :raises PluginCancelled: when *cancel_check* goes true between files.
    """
    files = 0
    total = 0
    dest.mkdir(parents=True)
    try:
        for dirpath, dirnames, filenames in os.walk(source, followlinks=False):
            current = Path(dirpath)
            target_dir = dest / current.relative_to(source)
            keep: list[str] = []
            for name in sorted(dirnames):
                if name in SKIP_DIRS:
                    continue
                if (current / name).is_symlink():
                    _refuse_link(source, current / name)
                keep.append(name)
            dirnames[:] = keep
            for name in keep:
                (target_dir / name).mkdir()
            for name in sorted(filenames):
                if name in SKIP_FILES:
                    continue
                path = current / name
                mode = path.lstat().st_mode
                if stat.S_ISLNK(mode):
                    _refuse_link(source, path)
                if not stat.S_ISREG(mode):
                    raise SourceRefused(
                        f"{path.relative_to(source)} is not a regular file.",
                        reason="unsafe_member",
                        path=str(source),
                        hint="A plugin folder may hold files and folders only.",
                    )
                files += 1
                total += path.stat().st_size
                github.check_caps(source.name, files, total)
                shutil.copyfile(path, target_dir / name)
                if cancel_check is not None and cancel_check():
                    raise PluginCancelled(f"copy of {source} cancelled")
    except OSError as exc:
        raise SourceRefused(
            f"{source} could not be read.",
            reason="unreadable",
            path=str(source),
            hint=str(exc),
        ) from exc


def _refuse_link(source: Path, link: Path) -> None:
    raise SourceRefused(
        f"{link.relative_to(source)} is a symbolic link.",
        reason="unsafe_member",
        path=str(source),
        hint=(
            "A plugin folder may not contain symbolic links; replace the link "
            "with the file it points to."
        ),
    )


def materialize(
    path: Path,
    form: SourceForm,
    workdir: Path,
    *,
    cancel_check: Callable[[], bool] | None = None,
) -> LocalTree:
    """Copy or unpack *path* into *workdir* and say where the plugin is.

    An archive is capped at :data:`~.github.MAX_TARBALL_BYTES` on disk
    before it is opened (the compressed cap a download is held to), then
    unpacked by :func:`~.github.extract_archive`. Its digest is the archive
    file's own, so the number the lockfile records is one a person can
    check with ``sha256sum``.
    """
    if form == "folder":
        copy = workdir / "tree"
        copy_folder(path, copy, cancel_check=cancel_check)
        return LocalTree(root=copy, form=form, digest=tree_digest(copy))
    size = path.stat().st_size
    if size > github.MAX_TARBALL_BYTES:
        cap_mb = github.MAX_TARBALL_BYTES // (1024 * 1024)
        raise SourceRefused(
            f"{path.name} is larger than {cap_mb} MB.",
            reason="too_large",
            path=str(path),
            hint=f"It is {size // (1024 * 1024)} MB; nothing was unpacked.",
        )
    try:
        digest = _file_digest(path)
    except OSError as exc:
        raise SourceRefused(
            f"{path} could not be read.", reason="unreadable", path=str(path),
            hint=str(exc),
        ) from exc
    unpacked = workdir / "extracted"
    unpacked.mkdir(parents=True)
    root = github.extract_archive(path, unpacked)
    return LocalTree(root=root, form=form, digest=digest)
