"""History: one page of commits, and the files one commit touched.

Two reads, both shaped by the same problem -- a commit message is arbitrary
text, and so is a filename. So neither of these parses lines:

* the log asks for one record per commit, fields separated by ``\\x1f``
  (ASCII unit separator) and records by NUL (``-z``). The message goes LAST,
  so anything it contains -- newlines, a stray ``\\x1f``, a subject that
  looks like a header -- lands in the body field where it belongs, because
  the split has a maximum.
* the file list is ``--name-status -z``, where the status and each path are
  their own NUL-terminated token. A rename is three tokens, not a line with
  tabs in it.

Paging is "ask for one more than the page and drop it": ``has_more`` is then
a fact about the page just read rather than a count of the whole history,
which on a big repository is a walk nobody asked for.

A MERGE is the awkward case, and the reason :func:`commit_files` resolves
the first parent itself: ``diff-tree -m --first-parent`` does not restrict
the diff to that parent (measured on git 2.53 -- it prints one diff per
parent, concatenated, so a merge appeared to touch every file changed on
either side), and ``--diff-merges=first-parent``, which would, is newer than
the git this tab supports. Two trees and one extra ``rev-parse`` say exactly
what was meant, on every version.
"""

from __future__ import annotations

from pathlib import Path

from .errors import GitError
from .models import CommitInfo, FileKind, GitFile, LogResponse
from .repo import commit_trees
from .runner import T_READ, run_git
from .status import kind_from_letter

#: How many commits one page holds unless the caller says otherwise, and
#: the most it may hold. The cap is not about the server -- ``git log`` is
#: cheap -- but about the tab: a page is a scroll position, and a client
#: asking for ten thousand commits has a paging bug, not a big repository.
#:
#: Both live HERE and the route imports them, so there is one owner. They
#: were two numbers once (a route that defaulted to 30 in front of a service
#: that defaulted to 20), which is the kind of disagreement nothing fails
#: over and everybody eventually trips on.
DEFAULT_LOG_LIMIT = 30
MAX_LOG_LIMIT = 100

#: ASCII unit separator: the one character a commit message is least likely
#: to contain, and (unlike a tab or a pipe) not something git will ever
#: insert itself. It is not impossible in a message, which is why the split
#: below has a maximum and the body is last.
FIELD_SEP = "\x1f"

#: The fields, in order: full sha, short sha, parents, author name, author
#: email, author date, ref names, subject, body.
#:
#: ``%x1f`` -- NOT ``%1f``, which git prints literally (measured on 2.53) --
#: is a literal byte in git's pretty format.
LOG_FORMAT = "%H%x1f%h%x1f%P%x1f%an%x1f%ae%x1f%at%x1f%D%x1f%s%x1f%b"

#: How many fields the format above produces.
LOG_FIELDS = 9

#: What ``%D`` puts between two ref names.
REF_SEP = ", "

#: git's rename/copy status letters, the only two that carry a second path.
_TWO_PATH_LETTERS = ("R", "C")


def log(root: Path, *, skip: int = 0,
        limit: int = DEFAULT_LOG_LIMIT) -> LogResponse:
    """One page of *root*'s history, newest first.

    *skip* and *limit* are the page; the routes clamp them, and they are
    checked again here because this is also called directly (by a test, by
    a future CLI) and ``--max-count`` is not a place to put an unchecked
    number.

    An UNBORN branch is not a failure: a repository that has just been
    initialised has no HEAD to walk, and ``git log`` says so with a fatal
    error that would otherwise be reported as one. The answer is an empty
    page with ``unborn`` set, which is the screen the tab draws for it.
    """
    if skip < 0:
        raise GitError("invalid_value", 400, "skip may not be negative")
    if not 1 <= limit <= MAX_LOG_LIMIT:
        raise GitError("invalid_value", 400,
                       f"limit must be between 1 and {MAX_LOG_LIMIT}")

    head = run_git(["rev-parse", "--verify", "--quiet", "HEAD"], cwd=root,
                   timeout=T_READ, ok_codes=(0, 1), read_only=True)
    if head.returncode != 0:
        return LogResponse(commits=[], has_more=False, unborn=True)

    # One more than the page, so that "is there another page" is answered by
    # the read itself rather than by counting the history.
    result = run_git(["log", "-z", f"--format={LOG_FORMAT}",
                      f"--max-count={limit + 1}", f"--skip={skip}", "HEAD",
                      "--"],
                     cwd=root, timeout=T_READ, read_only=True)

    rows = [row for row in result.out.split("\x00") if row]
    has_more = len(rows) > limit
    return LogResponse(
        commits=[commit for commit in (_commit(row) for row in rows[:limit])
                 if commit is not None],
        has_more=has_more,
        unborn=False)


def _commit(row: str) -> CommitInfo | None:
    """One record of the log format, or None if it is not one.

    The split has a MAXIMUM: the body is the last field and may contain
    anything, including the separator itself, and everything after the
    eighth one is body.

    A row with too few fields is dropped rather than raised on. This runs on
    a history the user did not write, one row can be malformed for reasons
    that are nobody's fault (a truncated pipe, a future git), and a page
    missing one commit is a smaller failure than a history panel that shows
    nothing.
    """
    parts = row.split(FIELD_SEP, LOG_FIELDS - 1)
    if len(parts) < LOG_FIELDS:
        return None
    sha, short, parents, name, email, timestamp, refs, subject, body = parts
    if not sha:
        return None
    return CommitInfo(
        sha=sha,
        short=short,
        parents=parents.split(),
        author_name=name,
        author_email=email,
        authored_at=_timestamp(timestamp),
        refs=[ref.strip() for ref in refs.split(REF_SEP) if ref.strip()],
        subject=subject,
        # git ends %b with a newline and starts it after the blank line that
        # follows the subject; the text between is the author's, indentation
        # included.
        body=body.strip("\n"))


def _timestamp(text: str) -> int:
    """``%at`` as an integer; 0 when it is not one.

    A commit with an unreadable date is still a commit worth showing, and
    the browser formats this field -- 0 is a date it can render.
    """
    try:
        return int(text)
    except ValueError:
        return 0


def commit_files(root: Path, sha: str) -> list[GitFile]:
    """The files one commit changed, against its first parent.

    A ROOT commit has no parent, so ``--root`` makes git diff it against the
    empty tree -- every file in it is an addition, which is what it is. A
    MERGE is diffed against its first parent explicitly; see the module
    docstring for why git's own ``--first-parent`` cannot do that here.

    ``xy`` on each file is the commit's letter followed by ``.`` -- the
    porcelain v2 spelling for "changed in the tree, nothing in the working
    copy" -- so the frontend can read a commit's file the same way it reads
    a status entry.
    """
    _, _, trees = commit_trees(root, sha)
    result = run_git(["diff-tree", "-r", "-M", "-z", "--name-status",
                      "--no-commit-id", *trees, "--"],
                     cwd=root, timeout=T_READ, read_only=True)
    return _name_status(result.stdout)


def _name_status(data: bytes) -> list[GitFile]:
    """Parse ``--name-status -z``: a status token, then one or two paths.

    Each token is decoded on its own, like the status parser does and for
    the same reason: under ``-z`` git prints a filename's bytes exactly as
    the filesystem holds them, and one unreadable name must cost that name
    a few replacement characters rather than cost the request.

    A truncated record at the end is dropped: it is the tail of an output
    that stopped, not evidence that what came before it is wrong.
    """
    tokens = [chunk.decode("utf-8", errors="replace")
              for chunk in data.split(b"\x00")]
    files: list[GitFile] = []
    index = 0
    while index < len(tokens):
        status = tokens[index]
        index += 1
        if not status:
            continue
        letter = status[0]
        renamed = letter in _TWO_PATH_LETTERS
        needed = 2 if renamed else 1
        if index + needed > len(tokens):
            break
        if renamed:
            orig_path, path = tokens[index], tokens[index + 1]
        else:
            orig_path, path = None, tokens[index]
        index += needed
        if not path:
            continue
        kind: FileKind = kind_from_letter(letter)
        files.append(GitFile(path=path, orig_path=orig_path, kind=kind,
                             xy=f"{letter}.", score=_score(status)))
    return files


def _score(status: str) -> int | None:
    """The similarity number in ``R100`` / ``C75``; None when there is none."""
    digits = status[1:]
    if not digits.isdigit():
        return None
    return int(digits)
