"""The stash: work set aside, and put back.

A stash is a stack of commits nobody can see from a branch, and the four
things that decide the shape of this module are all consequences of that:

* **The index is git's, never the caller's.** Every operation here that
  names a stash addresses it as ``stash@{N}`` -- built from an integer this
  module has just seen git itself print, and never from a string the client
  sent. ``drop`` and ``pop`` destroy work, and "the fourth stash" is the
  kind of argument a stale panel gets wrong; the list is re-read on every
  one of them and an index that is not in it is a 404 rather than a
  command. The index also comes out of ``%gd`` rather than out of the row's
  POSITION in the list: a row this parser could not read would otherwise
  shift every index below it by one, and the operation that shift lands on
  is the one that deletes.
* **The format is the PRETTY printer's, not ``for-each-ref``'s.** ``git
  stash list`` is ``git log`` wearing another name, so the separator is
  ``%x1f`` -- the opposite spelling to ``refs.BRANCH_FORMAT``, which is
  ``for-each-ref``'s. Both are measured on git 2.53 and both modules say
  so, because they look like the same mistake in opposite directions.
* **The subject is in the MIDDLE, so the record is read from both ends.**
  ``%gs`` is git's reflog subject, and for an unnamed stash it ends with
  the base commit's SUBJECT -- arbitrary text, which really can contain a
  0x1f byte (measured). The two fields around it cannot: ``%gd`` is
  ``stash@{N}`` and ``%at`` is digits. So the parse takes the first field
  from the left, the last from the right, and everything between is the
  subject -- the same property ``log.py`` and ``refs.py`` buy by putting
  their free-text field last, without moving a field R3 fixed.
* **Two answers git gives on STDOUT.** A stash with nothing to save is
  exit 0 and a sentence, and a pop that conflicts is exit 1 with an EMPTY
  stderr -- so a classification made from stderr alone calls the most
  ordinary failure this module has a 500. Both streams are joined before
  they are classified, exactly as ``service.commit_changes`` does for the
  empty commit.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from .errors import GitError, classify_failure
from .models import StashInfo
from .paths import validate_stash_message
from .runner import T_LOCAL, T_READ, run_git

#: ASCII unit separator, as ``log.py`` and ``refs.py`` use it.
FIELD_SEP = "\x1f"

#: ``%gd`` the reflog selector (``stash@{0}``), ``%gs`` the reflog subject,
#: ``%at`` the author time. ``%x1f`` -- NOT ``%1f`` -- is the literal byte
#: here: ``stash list`` takes ``log``'s pretty format, which is the
#: OPPOSITE of ``for-each-ref`` (see :data:`~app.core.git.refs.BRANCH_FORMAT`).
STASH_FORMAT = "%gd%x1f%gs%x1f%at"

#: ``stash@{N}``, as ``%gd`` prints it. The index is parsed back out of it
#: because it is what every write here is addressed by -- see the module
#: docstring.
_SELECTOR_RE = re.compile(r"^stash@\{(\d+)\}$")

#: ``git stash push`` with a named message: ``On <branch>: <message>``.
_NAMED = "On "

#: ``git stash push`` without one: ``WIP on <branch>: <sha> <subject>``.
_WIP = "WIP on "

#: What git writes where the branch would be when HEAD is detached. Not a
#: branch name -- ``StashInfo.branch`` is None for it, the same reading
#: ``BranchesResponse.current`` gives a detached HEAD.
_NO_BRANCH = "(no branch)"

#: What separates the branch from the rest of a reflog subject. A colon
#: cannot appear in a ref name (git's own rule), so the FIRST one ends the
#: branch even when the message that follows holds more of them.
_BRANCH_END = ": "

#: git's answer, on STDOUT and with exit 0, to a stash that would hold
#: nothing. Its exact wording is a promise ``LC_ALL=C`` in the runner's
#: environment makes -- the same promise every row of ``errors._RULES``
#: rests on.
NOTHING_TO_STASH = "No local changes to save"

#: git's answer to a stash in a repository whose branch has no commits: a
#: stash is a commit, and there is nothing for it to sit on yet.
NO_INITIAL_COMMIT = "You do not have the initial commit yet"


# --- reading the stack -------------------------------------------------------


def list_stashes(root: Path) -> list[StashInfo]:
    """Every stash of *root*, newest first, as git orders them.

    ``-z`` terminates each record with a NUL and prints no trailing
    newline, so unlike ``for-each-ref`` there is nothing to strip from the
    front of a record; an empty stack is an empty string.
    """
    result = run_git(["stash", "list", "-z", f"--format={STASH_FORMAT}"],
                     cwd=root, timeout=T_READ, read_only=True)
    stashes: list[StashInfo] = []
    for record in result.out.split("\x00"):
        entry = _entry(record)
        if entry is not None:
            stashes.append(entry)
    return stashes


def _entry(record: str) -> StashInfo | None:
    """One record of :data:`STASH_FORMAT`, or None if it is not one.

    Read from both ends because the free-text field is in the middle (see
    the module docstring). A record whose selector is not ``stash@{N}`` is
    dropped rather than raised on -- the same choice ``refs._row`` and
    ``log._commit`` make, and here it also means an unreadable row can
    never be ADDRESSED, which is the safer half of dropping it.
    """
    if not record:
        return None
    selector, sep, rest = record.partition(FIELD_SEP)
    if not sep:
        return None
    subject, sep, when = rest.rpartition(FIELD_SEP)
    if not sep:
        return None
    match = _SELECTOR_RE.match(selector)
    if match is None:
        return None
    branch, message = _subject(subject)
    # ``message or selector``: the model promises a non-empty label, and a
    # blank row in a list is a row nobody can tell from the one below it.
    # git always writes a reflog subject, so this is the malformed case
    # keeping its promise rather than a shape anybody has seen.
    return StashInfo(index=int(match.group(1)), message=message or selector,
                     branch=branch, created_at=_timestamp(when))


def _subject(subject: str) -> tuple[str | None, str]:
    """``%gs`` as ``(the branch it was made on, what to show for it)``.

    Two shapes, and the difference is whether the user gave a message:
    ``On <branch>: <message>`` when they did, ``WIP on <branch>: <sha>
    <subject>`` when they did not. The message half is what comes back for
    the first and the WHOLE subject for the second -- because "WIP on main:
    9f2c1ab Add the loader" is a description of that stash and its base
    commit is the only description there is, while "9f2c1ab Add the loader"
    on its own would read as a commit the user made.

    The ``WIP on `` test runs FIRST and both are ``startswith``: a message
    of "WIP on main: deadbee spoof" arrives as ``On main: WIP on main:
    deadbee spoof`` (measured), so a substring test would read the user's
    own words as git's and lose the branch.

    Anything that is neither -- a stash written by another tool, a reflog
    somebody rewrote -- keeps its whole subject and reports no branch. It
    is still a row worth showing and still a stash worth dropping. So does
    a named shape whose message half is EMPTY: ``validate_stash_message``
    means this API cannot make one, and a row with nothing written on it is
    a row nobody can tell from the row below.
    """
    if subject.startswith(_WIP):
        branch, sep, _rest = subject[len(_WIP):].partition(_BRANCH_END)
        return (_branch(branch), subject) if sep else (None, subject)
    if subject.startswith(_NAMED):
        branch, sep, message = subject[len(_NAMED):].partition(_BRANCH_END)
        return (_branch(branch), message or subject) if sep else (None, subject)
    return None, subject


def _branch(name: str) -> str | None:
    """The branch a stash was made on; None when it was made off one."""
    return None if name == _NO_BRANCH else name


def _timestamp(text: str) -> int:
    """``%at`` as an integer; 0 when it is not one."""
    try:
        return int(text)
    except ValueError:
        return 0


# --- changing the stack ------------------------------------------------------


def stash_push(root: Path, *, message: str | None = None,
               include_untracked: bool = True) -> dict[str, Any]:
    """Set the working tree aside, and answer with what was saved.

    *include_untracked* defaults to True because the button says "Stash
    Changes" and a new file is a change: leaving it behind would set aside
    half the work and say nothing about the other half.

    The message rides in ``--message=<m>``, attached rather than as a
    second argument, so a message that begins with ``-`` is a message and
    not an option (measured: ``--message=-x`` makes a stash called ``-x``).

    **This is the one command in the package that runs WITHOUT
    ``--literal-pathspecs``, and the option does not refuse it -- it
    corrupts it.** Measured on git 2.53: with the option,
    ``--include-untracked`` puts the untracked file in the stash and LEAVES
    IT in the working tree, exit 0 and no warning, so the tab reports a
    stash that did half of what it says -- and the next pop of that entry
    fails with "new.txt already exists, no checkout". (The removal is an
    internal ``git clean`` whose pathspec the option changes the meaning
    of.) Turning it off is safe by construction, and by a stronger argument
    than ``check-ignore``'s: this argv holds NO PATHSPEC AT ALL. The
    subcommand takes none, the message travels inside ``--message=`` as an
    option value, and nothing the caller sent is positional -- so there is
    nothing for pathspec magic to be read out of, whatever the option says.

    Two answers that are not failures to git and are refusals here. A stash
    with NOTHING to save is exit 0 and a sentence on stdout -- not a 409
    ``nothing_to_commit``, which is a different sentence about a different
    button, but a 400 saying there is nothing to stash; the tab disables the
    menu item when the tree is clean, so what this really guards is the race
    between a fifteen-second-old panel and a click. And a stash in a
    repository with no commits at all is git's "You do not have the initial
    commit yet" -- a 500 without this, and the state every new repository
    starts in. It is answered the way ``commit_changes`` answers an amend
    with nothing to amend: a 404 whose hint is the fact.
    """
    text = validate_stash_message(message) if message is not None else None
    args = ["stash", "push"]
    if include_untracked:
        args.append("--include-untracked")
    if text is not None:
        args.append(f"--message={text}")

    # ``literal_pathspecs=False``: see the docstring. It is not a tidy-up.
    result = run_git(args, cwd=root, timeout=T_LOCAL, ok_codes=(0, 1, 128),
                     literal_pathspecs=False)
    if result.returncode != 0:
        if NO_INITIAL_COMMIT in result.err:
            raise GitError("not_found", 404, "there is nothing to stash yet",
                           hint="this branch has no commits yet")
        raise classify_failure(result.argv, result.returncode,
                               f"{result.out}\n{result.err}")
    if NOTHING_TO_STASH in result.out:
        raise GitError("invalid_value", 400, "there is nothing to stash",
                       hint="nothing to stash")
    # The stack's newest entry, which is where a push always lands.
    return {"stash": 0, "message": text,
            "include_untracked": include_untracked}


def stash_pop(root: Path, index: int) -> dict[str, Any]:
    """Put a stash back and remove it -- unless putting it back conflicts.

    git keeps the entry when the merge does not apply cleanly, and that is
    the behaviour rather than a fallback: the changes are in the working
    tree with markers around them, and the copy in the stack is the only
    thing standing between the user and losing them if the resolution goes
    wrong. So the 409 this raises says ``conflict``, the tab draws the merge
    group, and the stash is still there afterwards.
    """
    return _restore(root, "pop", index)


def stash_apply(root: Path, index: int) -> dict[str, Any]:
    """Put a stash back and KEEP it in the stack."""
    return _restore(root, "apply", index)


def _restore(root: Path, command: Literal["pop", "apply"],
             index: int) -> dict[str, Any]:
    """``pop`` or ``apply``: the same command with a different last word.

    Both classify against BOTH streams, and that is the whole reason this
    is not two lines of ``run_git``. A pop that conflicts prints
    ``CONFLICT (content): ...`` on STDOUT and leaves stderr EMPTY (measured
    on git 2.53), so stderr alone classifies the commonest failure here as
    ``git_failed`` -- a 500 for a conflict the tab knows exactly what to do
    with. A pop REFUSED for an uncommitted change to the same file is the
    other way round, on stderr, and classifies as ``dirty_tree`` from
    either; joining does not disturb it.
    """
    ref = _addressed(root, index)
    result = run_git(["stash", command, ref], cwd=root, timeout=T_LOCAL,
                     ok_codes=(0, 1, 128))
    if result.returncode != 0:
        raise classify_failure(result.argv, result.returncode,
                               f"{result.out}\n{result.err}")
    return {"stash": index}


def stash_drop(root: Path, index: int) -> dict[str, Any]:
    """Throw a stash away. Nothing in the working tree moves.

    The one operation here that destroys and cannot be undone from the tab,
    which is why the index is re-read first: dropping the wrong entry of a
    stack the panel last saw fifteen seconds ago is exactly the mistake
    :func:`_addressed` exists to make impossible.
    """
    ref = _addressed(root, index)
    run_git(["stash", "drop", "-q", ref], cwd=root, timeout=T_LOCAL)
    return {"stash": index}


def _addressed(root: Path, index: int) -> str:
    """``stash@{N}`` for an index git is listing RIGHT NOW, or a 404.

    Membership in a fresh list, not ``0 <= index < len``: the two agree
    whenever every row parsed, and when one did not, this is the reading
    that cannot address a stash the caller did not mean. git's own refusal
    for a bad index ("error: stash@{7} is not a valid reference") is a 404
    too, so this is not the only thing standing between a stale panel and a
    wrong answer -- but it is the one with a hint saying what to do.
    """
    if index not in {entry.index for entry in list_stashes(root)}:
        raise GitError("not_found", 404, f"there is no stash at {index}",
                       hint="the stash list has changed since it was read; "
                            "reload it")
    return f"stash@{{{index}}}"
