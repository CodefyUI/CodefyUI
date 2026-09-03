"""What changed in one file, and what that file says at one ref.

Both of these are open GETs -- the tab reads a diff the way it reads a
status -- so both start at the same place: the ``.env`` guard. A dotenv file
is refused at ANY ref, including the ones deep in history, because
``.gitignore`` does not un-commit a secret that was committed once and a
read endpoint that will hand it back is the whole exposure.

After that the shape of both functions is the same three questions:

* **which two things are being compared.** ``worktree`` is index against
  disk, ``index`` is HEAD against index, ``commit`` is a commit against its
  first parent -- and the ``old_ref`` / ``new_ref`` strings say so in the
  tab's own words, so the header above a diff is not a guess.
* **is there anything to show.** git answers "Binary files ... differ"
  rather than a patch for a PNG, and an untracked file has no index side to
  diff against at all -- that one becomes ``--no-index`` against
  ``/dev/null``, which git for Windows understands as well as any POSIX
  git.
* **how much of it to hand over.** A patch is capped at
  :data:`MAX_PATCH_BYTES` and a file at :data:`MAX_FILE_BYTES`, both cut in
  BYTES and then decoded, so a multi-byte character split by the cut costs
  one replacement character instead of an exception. ``truncated`` says it
  happened; ``size`` is what git had before the cut.

``--no-ext-diff`` is on every one of these commands, including the
``--no-index`` one. Without it a repository whose config sets
``diff.external`` -- or an inherited ``GIT_EXTERNAL_DIFF`` -- makes a diff
request run that program on the server, which is a command line the user
never typed and this process would then wait for.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .errors import GitError, classify_failure
from .models import DiffResponse, FileAtRef
from .paths import is_env_secret_path, validate_rel_path, validate_sha
from .repo import first_parent, read_status, resolve_commit
from .runner import T_READ, run_git

#: The most patch text one response carries. A diff past a megabyte is not
#: something anybody reads in a side panel; it is a generated file or a
#: vendored tree, and the tab says so rather than sending it.
MAX_PATCH_BYTES = 1024 * 1024

#: The most of one file's content a response carries. Twice the patch cap,
#: because a side-by-side view asks for BOTH sides in full.
MAX_FILE_BYTES = 2 * 1024 * 1024

#: How much of a file is read to decide whether it is text. A NUL in the
#: first few kilobytes is what every diff tool means by "binary", and it is
#: also what git itself uses.
BINARY_SNIFF_BYTES = 8 * 1024

#: The three things a diff can be about.
DIFF_SCOPES = ("worktree", "index", "commit")

#: The refs a file may be read at, besides a commit id.
NAMED_REFS = ("HEAD", "index", "worktree")

#: The empty side of an added file. A literal, even on Windows: git for
#: Windows understands ``/dev/null`` in a pathspec and prints it back in the
#: patch header, and ``NUL`` would be a file named NUL.
DEV_NULL = "/dev/null"

#: git's way of saying there is no patch to show.
_BINARY_MARKER = "Binary files "

#: What ``cat-file`` says when the blob is not there. Anything else that
#: exits 128 is a repository problem, not a missing file, and is classified
#: like any other failure.
_MISSING_BLOB_PHRASES = (
    # cat-file blob HEAD:<path that is not in HEAD>
    "does not exist",
    # cat-file blob <ref that is not an object>:<path>, and an unborn HEAD
    "invalid object",
    "not a valid object name",
    # cat-file blob <valid sha>:<path added after that commit>
    "exists on disk, but not in",
)


def _readable_path(root: Path, path: str) -> str:
    """Validate *path*, and refuse a dotenv file whatever ref it was asked at.

    The guard is not about the working copy: a ``.env`` that was committed
    once is in every later tree, and both of these functions can read one.
    """
    clean = validate_rel_path(root, path)
    if is_env_secret_path(clean):
        raise GitError("ignored", 403, f"{clean} is not served",
                       hint="dotenv files hold secrets and are never read "
                            "through this API")
    return clean


def diff(root: Path, path: str, scope: str, *, sha: str | None = None,
         blobs: bool = False) -> DiffResponse:
    """The patch for one file, in one of the three scopes.

    *scope* is ``worktree`` (index against the file on disk), ``index``
    (HEAD against the index) or ``commit`` (that commit against its first
    parent, which needs *sha*).

    *blobs* additionally fills ``old_text`` / ``new_text`` with both whole
    sides, for a side-by-side view. It costs two more git reads, so it is
    off by default; a side that does not exist at that ref sets
    ``old_missing`` / ``new_missing`` instead of failing the request, since
    "this file was added here" is an answer and not an error.
    """
    clean = _readable_path(root, path)
    if scope not in DIFF_SCOPES:
        raise GitError("invalid_value", 400,
                       f"scope must be one of {', '.join(DIFF_SCOPES)}")
    plan = _plan(root, clean, scope, sha)

    # ``diff`` exits 1 to mean "there are differences" under ``--no-index``,
    # and 0 otherwise; neither is a failure.
    result = run_git(plan.args, cwd=root, timeout=T_READ, ok_codes=(0, 1),
                     read_only=True)
    raw = result.stdout
    patch = raw[:MAX_PATCH_BYTES].decode("utf-8", errors="replace")

    response = DiffResponse(
        patch=patch,
        binary=_is_binary_patch(patch),
        truncated=len(raw) > MAX_PATCH_BYTES,
        old_ref=plan.old_ref,
        new_ref=plan.new_ref,
        # Known before anything is read: an untracked file has no index
        # side, and a root commit has no parent.
        old_missing=plan.old_source is None)
    if blobs:
        _fill_blobs(root, clean, response, plan)
    return response


@dataclass
class _Plan:
    """What one diff request compares, decided once.

    ``old_ref`` / ``new_ref`` are what the TAB shows above the diff (git's
    own notation, ``<sha>^``); ``old_source`` / ``new_source`` are what
    :func:`file_at_ref` has to be asked to get each whole side, which is not
    always the same string -- ``<sha>^`` is not a ref a closed grammar can
    accept, so the parent is resolved to its own id. A ``None`` source means
    that side does not exist.
    """

    args: list[str]
    old_ref: str | None
    new_ref: str
    old_source: str | None
    new_source: str


def _plan(root: Path, path: str, scope: str, sha: str | None) -> _Plan:
    """The command and the two sides for *scope*."""
    if scope == "worktree":
        # An UNTRACKED file has no index side, and ``git diff`` says nothing
        # at all about one -- an empty patch for a file the user can see is
        # the most confusing possible answer. ``--no-index`` against
        # ``/dev/null`` gives the patch that shows it as added, which it is.
        if path in {entry.path for entry in read_status(root).untracked}:
            return _Plan(["diff", "--no-index", "--no-color", "--no-ext-diff",
                          "--", DEV_NULL, path],
                         "index", "worktree", None, "worktree")
        return _Plan(["diff", "--no-color", "--no-ext-diff", "-M", "--", path],
                     "index", "worktree", "index", "worktree")

    if scope == "index":
        return _Plan(["diff", "--cached", "--no-color", "--no-ext-diff", "-M",
                      "--", path],
                     "HEAD", "index", "HEAD", "index")

    if not sha:
        raise GitError("invalid_value", 400,
                       "a commit diff needs the commit it is about")
    # The first parent is resolved rather than asked for with
    # ``--first-parent``: on a merge that option does not restrict
    # ``diff-tree`` to one parent (git 2.53 prints one patch per parent), so
    # the tab would show the same file twice with two different diffs. A
    # root commit has no parent and is diffed against the empty tree.
    commit = resolve_commit(root, sha)
    parent = first_parent(root, commit)
    trees = [parent, commit] if parent is not None else ["--root", commit]
    return _Plan(["diff-tree", "-M", "-p", "--no-color", "--no-ext-diff",
                  "--no-commit-id", *trees, "--", path],
                 f"{commit}^" if parent is not None else None,
                 commit, parent, commit)


def _is_binary_patch(patch: str) -> bool:
    """Did git say "Binary files ... differ" instead of giving a patch?

    Anchored to the START of a line: the same sentence inside a hunk arrives
    with a ``+``, ``-`` or space in front of it, and a text file that
    happens to contain it must not be reported as binary.
    """
    return any(line.startswith(_BINARY_MARKER) and line.rstrip().endswith("differ")
               for line in patch.splitlines())


def _fill_blobs(root: Path, path: str, response: DiffResponse,
                plan: _Plan) -> None:
    """Read both whole sides into *response*, for a side-by-side view.

    A side that does not exist at its ref -- an added file has no old side,
    a deleted one no new side -- sets the ``missing`` flag and leaves the
    text None. That is not an error: it is the ordinary shape of half the
    diffs the tab shows.
    """
    if plan.old_source is not None:
        response.old_text, response.old_missing = _side(root, path,
                                                        plan.old_source)
    response.new_text, response.new_missing = _side(root, path,
                                                    plan.new_source)


def _side(root: Path, path: str, ref: str) -> tuple[str | None, bool]:
    """``(text, missing)`` for one side of a diff."""
    try:
        return file_at_ref(root, path, ref).text, False
    except GitError as exc:
        if exc.code == "not_found":
            return None, True
        raise


def file_at_ref(root: Path, path: str, ref: str) -> FileAtRef:
    """One file's content at one ref: ``HEAD``, ``index``, ``worktree``, a sha.

    The ``.env`` guard runs first, before anything is read and whatever the
    ref is.

    A WORKTREE read is the one that needs a second question asked, because
    the file on disk is not necessarily one git knows about: it must be
    tracked or present in the status as untracked (otherwise there is
    nothing to serve and it is a 404), and it must not be IGNORED -- an
    ignored file is a 403, because ``.gitignore`` is where a project says
    which files are not part of it, and a read endpoint that serves them
    anyway makes that promise worthless.
    """
    clean = _readable_path(root, path)
    if ref == "worktree":
        return _from_worktree(root, clean)
    if ref == "index":
        return _from_object(root, f":0:{clean}")
    if ref == "HEAD":
        return _from_object(root, f"HEAD:{clean}")
    return _from_object(root, f"{validate_sha(ref)}:{clean}")


def _from_object(root: Path, spec: str) -> FileAtRef:
    """Read a blob out of the object database (``<ref>:<path>``).

    ``cat-file`` exits 128 for every kind of "there is nothing there": the
    path is not in that tree, the ref is not an object, HEAD is unborn. All
    of those are a 404 for a file view -- the phrases are in
    :data:`_MISSING_BLOB_PHRASES`, read off git 2.53 -- and anything else
    that exits 128 is classified like any other git failure, so a corrupt
    object database is not reported as a missing file.
    """
    result = run_git(["cat-file", "blob", spec], cwd=root, timeout=T_READ,
                     ok_codes=(0, 128), read_only=True)
    if result.returncode != 0:
        message = result.err.lower()
        if any(phrase in message for phrase in _MISSING_BLOB_PHRASES):
            raise GitError("not_found", 404, f"no {spec} in this repository",
                           stderr=result.err.strip())
        raise classify_failure(result.argv, result.returncode, result.err)
    return _content(result.stdout, len(result.stdout))


def _from_worktree(root: Path, path: str) -> FileAtRef:
    """Read the file on disk, once git agrees it is part of the project."""
    ignored = run_git(["check-ignore", "-q", "--", path], cwd=root,
                      timeout=T_READ, ok_codes=(0, 1), read_only=True)
    if ignored.returncode == 0:
        raise GitError("ignored", 403, f"{path} is ignored by this repository",
                       hint="ignored files are not part of the project and "
                            "are not served")

    tracked = run_git(["ls-files", "--error-unmatch", "--", path], cwd=root,
                      timeout=T_READ, ok_codes=(0, 1), read_only=True)
    if tracked.returncode != 0:
        if path not in {entry.path for entry in read_status(root).untracked}:
            raise GitError("not_found", 404, f"git does not know {path}")

    target = root / path
    try:
        size = target.stat().st_size
        with target.open("rb") as handle:
            # One byte past the cap, so that "was there more" is answered
            # without reading a file that does not fit in memory.
            raw = handle.read(MAX_FILE_BYTES + 1)
    except OSError as exc:
        raise GitError("not_found", 404, f"{path} could not be read",
                       stderr=str(exc)) from None
    return _content(raw, size)


def _content(raw: bytes, size: int) -> FileAtRef:
    """Turn bytes into the response, with the cap and the binary sniff.

    A binary file gets an EMPTY text and the flag, rather than a decoded
    approximation of itself: the tab draws a placeholder for it, and
    replacement characters would be a screenful of nothing that looks like
    content.

    *size* is what the file really is, before any cut -- so a truncated
    response can still say how big the thing it truncated was.
    """
    if b"\x00" in raw[:BINARY_SNIFF_BYTES]:
        return FileAtRef(text="", binary=True, size=size,
                         truncated=size > MAX_FILE_BYTES)
    return FileAtRef(text=raw[:MAX_FILE_BYTES].decode("utf-8", errors="replace"),
                     binary=False, size=size, truncated=size > MAX_FILE_BYTES)
