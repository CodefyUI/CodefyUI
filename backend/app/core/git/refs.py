"""Branches and remotes: what the repository can point at, and where.

Everything in here is a REF or a name for one, and refs are where the
Source Control tab stops being about files. Four things are decided once,
here, rather than per operation:

* **The format string is not the log's format string.** ``git log``'s
  pretty printer spells a literal byte ``%x1f`` and prints ``%1f`` verbatim;
  ``for-each-ref`` spells the same byte ``%1f`` and prints ``%x1f``
  verbatim. Both measured on git 2.53, both in this repository -- see
  ``log.LOG_FORMAT``'s comment for the other half. They look like the same
  mistake in opposite directions, and a "fix" that made them agree would
  break whichever one it touched, so :data:`BRANCH_FORMAT` says so out loud.
* **The subject goes LAST**, for the reason ``log.py`` puts the body last: a
  commit subject is arbitrary text and can contain the separator itself, so
  the split has a maximum and everything past the seventh field is subject.
* **A ``<remote>/HEAD`` is a pointer, not a branch.** ``git clone`` leaves a
  symbolic ref at ``refs/remotes/origin/HEAD`` saying which branch the
  remote calls its default, and ``for-each-ref`` lists it beside the real
  ones -- where ``%(refname:short)`` renders it as the bare word ``origin``,
  which would draw a remote branch with no name. The format asks for
  ``%(symref)`` and skips every row that has one: a symbolic ref is the only
  thing that can be one here, and the test does not depend on what it is
  called.
* **The full ``%(refname)``, not the short one.** ``refs/heads`` and
  ``refs/remotes`` arrive in ONE stream, and the short forms cannot be told
  apart -- a local branch really can be called ``origin/main``. The prefix
  is what separates them, and stripping it is more predictable than git's
  own shortening rules.

**Nothing here puts a user's string on a command line without a validator
in front of it, and the VALIDATOR is the guarantee -- not ``--``.** A
branch name goes through :func:`check_ref_format`, a remote name and a URL
through ``paths.validate_remote_name`` / ``validate_remote_url``, and a
start point through ``validate_sha`` when it looks like a commit id. Every
one of them refuses a leading ``-``, which is the whole attack:
``--upload-pack=`` is a branch name to a shell that is not there and a
command to git.

``--`` is not what stops it, and the argv below (R9's, verbatim) does not
lean on it. Measured on git 2.53: ``switch``, ``switch -c``, ``switch
--track``, ``remote add``, ``remote set-url`` and ``remote remove`` all
ACCEPT a ``--``, and it buys less than it looks like. ``git remote add --
-y <url>`` exits 0 and creates a remote genuinely called ``-y``: the
separator made the option-shaped name legal rather than harmless. ``git
branch -- -x`` refuses with "'-x' is not a valid branch name" -- the NAME
grammar, again, not the separator. So a future operation added to this
module may put a ``--`` in its argv for tidiness, but it owes the same
validator either way.

Two failures are decided HERE rather than by the classifier, for the reason
``commit_changes`` decides "nothing to amend" itself: git reports them with
an ``error: `` opening that the classifier deliberately refuses to read as
git's voice (a failing hook opens the same way), so "there is no branch
called that" would otherwise be a 500 on a panel that is up to fifteen
seconds out of date.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from .errors import GitError, redact
from .models import BranchesResponse, BranchInfo, RemoteBranchInfo, RemoteInfo
from .paths import (
    SHA_RE,
    validate_branch_name,
    validate_remote_name,
    validate_remote_url,
    validate_sha,
)
from .runner import T_LOCAL, T_READ, T_STATUS, run_git

#: ASCII unit separator, the same one ``log.py`` uses and for the same
#: reason: the one character a commit subject is least likely to hold.
FIELD_SEP = "\x1f"

#: The fields, in order: full ref name, the ref it points at when it is
#: symbolic, short sha, upstream, upstream tracking, the HEAD marker, commit
#: date, subject.
#:
#: ``%1f`` -- NOT ``%x1f``, which ``for-each-ref`` prints literally -- is a
#: literal byte here. This is the OPPOSITE of ``log.LOG_FORMAT``, where the
#: pretty printer wants ``%x1f`` and prints ``%1f`` literally. Both measured
#: on git 2.53; the two formats are different parsers wearing the same
#: syntax, and making them look alike breaks one of them.
BRANCH_FORMAT = (
    "%(refname)%1f%(symref)%1f%(objectname:short)%1f%(upstream:short)"
    "%1f%(upstream:track)%1f%(HEAD)%1f%(committerdate:unix)%1f%(subject)%00")

#: How many fields the format above produces.
BRANCH_FIELDS = 8

#: What ``%(HEAD)`` prints for the branch HEAD is on. Every other row gets a
#: SPACE, not an empty field.
HEAD_MARKER = "*"

_HEADS = "refs/heads/"
_REMOTES = "refs/remotes/"


def check_ref_format(root: Path, name: str) -> str:
    """Would git accept *name* as a branch name? Returns it, or raises 400.

    Two checks, and the ORDER is the point. ``git check-ref-format`` is the
    authority on the rules that are git's (a trailing ``.lock``, ``..``, a
    control character) and knows nothing about ours: it accepts
    ``refs/heads/-x`` quite happily -- measured, exit 0 -- and ``-x`` on a
    command line is an option, not a branch. So the regex
    (:data:`~app.core.git.paths.BRANCH_NAME_RE`, via
    ``validate_branch_name``) runs FIRST and is what stops that one; git
    then rejects what it alone knows about, with exit 1.

    Every non-zero exit is the same answer. ``check-ref-format`` documents 1
    for "not a valid ref name" and uses 128 for a call it could not make
    sense of at all (a missing argument, an option it does not know), and
    the second is still "git will not take this name" as far as a browser is
    concerned -- reporting it as a 500 would be a server error for a
    question the user asked wrongly.

    This is the gate in front of every branch name that reaches a command
    line, including the ones that only NAME an existing branch (a rename's
    old name, a checkout's target). It is the grammar and not a ``--``
    separator that makes those safe -- see the module docstring: ``switch``
    does take one, and taking one is not the same as being protected by it.
    """
    validate_branch_name(name)
    result = run_git(["check-ref-format", f"refs/heads/{name}"], cwd=root,
                     timeout=T_STATUS, ok_codes=(0, 1, 128), read_only=True)
    if result.returncode != 0:
        raise GitError("invalid_ref", 400, f"git will not accept {name!r} "
                                           f"as a branch name")
    return name


# --- reading the refs --------------------------------------------------------


def list_branches(root: Path) -> BranchesResponse:
    """Every branch of *root*, local and remote-tracking, in git's own order.

    One ``for-each-ref`` for the branches and one ``symbolic-ref`` for what
    HEAD is: the second is not a nicety, because ``%(HEAD)`` marks nothing at
    all when HEAD is detached, and it is also the only way to name the
    branch of an UNBORN repository -- which has a current branch and no ref
    to list it from.
    """
    current, detached = _head(root)
    result = run_git(["for-each-ref", f"--format={BRANCH_FORMAT}",
                      "refs/heads", "refs/remotes"],
                     cwd=root, timeout=T_READ, read_only=True)

    local: list[BranchInfo] = []
    remote: list[RemoteBranchInfo] = []
    for record in result.out.split("\x00"):
        row = _row(record)
        if row is None:
            continue
        refname, symref, sha, upstream, track, head, when, subject = row
        if symref:
            # ``refs/remotes/<r>/HEAD`` -- which branch the remote calls its
            # default. A pointer at one of the rows below it, not a row.
            continue
        if refname.startswith(_HEADS):
            ahead, behind, gone = _track(track, bool(upstream))
            local.append(BranchInfo(
                name=refname[len(_HEADS):], sha=sha,
                current=head == HEAD_MARKER,
                upstream=upstream or None, ahead=ahead, behind=behind,
                gone=gone, subject=subject, committed_at=_timestamp(when)))
        elif refname.startswith(_REMOTES):
            rest = refname[len(_REMOTES):]
            remote_name, slash, name = rest.partition("/")
            if not slash or not name:
                continue
            remote.append(RemoteBranchInfo(
                name=name, remote=remote_name, sha=sha, subject=subject,
                committed_at=_timestamp(when)))
    return BranchesResponse(current=current, detached=detached, local=local,
                            remote=remote)


def _head(root: Path) -> tuple[str | None, bool]:
    """``(the branch HEAD is on, is HEAD detached)``.

    ``symbolic-ref`` answers both in one call and answers them for an unborn
    branch too, where ``rev-parse`` has nothing to resolve: a repository that
    has just been initialised is ON ``main`` with no commit and no ref, and
    the tab has a name to show for it.
    """
    result = run_git(["symbolic-ref", "--quiet", "--short", "HEAD"], cwd=root,
                     timeout=T_STATUS, ok_codes=(0, 1), read_only=True)
    if result.returncode != 0:
        return None, True
    return result.out.strip() or None, False


def _row(record: str) -> tuple[str, ...] | None:
    """One record of :data:`BRANCH_FORMAT`, or None if it is not one.

    The split has a MAXIMUM, so a separator inside a commit subject stays in
    the subject; a record with too few fields is dropped rather than raised
    on, for the reason ``log._commit`` drops one -- a branch list missing a
    row is a smaller failure than a panel that shows nothing.

    ``for-each-ref`` ends every record with a newline AFTER the NUL, so each
    chunk but the first opens with one.
    """
    text = record.lstrip("\n")
    if not text:
        return None
    parts = text.split(FIELD_SEP, BRANCH_FIELDS - 1)
    if len(parts) < BRANCH_FIELDS or not parts[0]:
        return None
    return tuple(parts)


def _track(text: str, has_upstream: bool) -> tuple[int | None, int | None, bool]:
    """``%(upstream:track)`` as ``(ahead, behind, gone)``.

    git prints ``[ahead 1, behind 2]``, either half alone, ``[gone]``, or
    nothing at all -- and "nothing at all" means two different things: no
    upstream is configured, or there is one and the branch is level with it.
    The caller knows which, so it says, and the difference is null versus
    zero: the tab draws nothing for a branch with no upstream and "up to
    date" for one that is level.
    """
    if not has_upstream:
        return None, None, False
    body = text.strip().strip("[]")
    if body == "gone":
        return None, None, True
    ahead = behind = 0
    for part in body.split(","):
        words = part.split()
        if len(words) != 2 or not words[1].isdigit():
            continue
        if words[0] == "ahead":
            ahead = int(words[1])
        elif words[0] == "behind":
            behind = int(words[1])
    return ahead, behind, False


def _timestamp(text: str) -> int:
    """``%(committerdate:unix)`` as an integer; 0 when it is not one."""
    try:
        return int(text)
    except ValueError:
        return 0


def list_remotes(root: Path) -> list[RemoteInfo]:
    """Every configured remote of *root*, with its fetch and push URLs.

    ``remote -v`` prints two tab-separated lines per remote -- the fetch URL
    and the push URL, which are the same string unless somebody set
    ``remote.<name>.pushurl`` -- and the split takes the LAST space rather
    than the first, so a URL that somehow contains one keeps it and only the
    ``(fetch)`` / ``(push)`` marker is taken off the end.

    **Both URLs are redacted before they leave this function**, and that is
    not defence in depth -- it is the only thing standing between a token
    and the network. ``GET /api/git/remotes`` is an open, unauthenticated
    read like every other GET in this app, this server is deliberately
    servable to a LAN (issue #247), and
    ``https://alice:ghp_xxx@github.com/owner/repo.git`` is a remote URL git
    accepts and people really do paste into ``git remote add`` --
    ``validate_remote_url`` accepts it too, on purpose, because refusing it
    would only mean the user configures it from a terminal instead.
    :func:`~app.core.git.errors.redact` is the function this package
    already has for this exact string, and its own docstring says a route
    "still owes the same care to anything else it echoes back".

    What survives is the scheme, the host and the path -- the part that
    makes the row worth showing -- so what the tab draws is a URL the user
    recognises and cannot copy a credential out of. The cost is that a
    harmless ``ssh://git@host/...`` loses its username to the same mask;
    that is the right trade for a string nobody can tell apart from a
    password on sight, and it is why :class:`RemoteInfo` says these values
    are for DISPLAY and must never be written back.
    """
    result = run_git(["remote", "-v"], cwd=root, timeout=T_READ,
                     read_only=True)
    found: dict[str, dict[str, str]] = {}
    for line in result.out.splitlines():
        name, tab, rest = line.partition("\t")
        if not tab or not name:
            continue
        url, space, marker = rest.rpartition(" ")
        if not space:
            url, marker = rest, ""
        urls = found.setdefault(name, {})
        if marker == "(push)":
            urls["push"] = url
        elif marker == "(fetch)":
            urls["fetch"] = url
    return [RemoteInfo(name=name,
                       fetch_url=redact(urls.get("fetch", "")),
                       push_url=redact(urls.get("push", "")))
            for name, urls in found.items()]


# --- writing the refs --------------------------------------------------------


def create_branch(root: Path, name: str, *, checkout: bool = True,
                  start_point: str | None = None) -> dict[str, Any]:
    """Make a branch, and go to it unless the caller said not to.

    Two commands, not one with a flag: ``switch -c`` makes the branch AND
    moves the working tree, and ``branch`` makes it and leaves the user where
    they are. Which one runs is also what decides whether this write can
    change a file on disk, which is why the service reads the same flag to
    pick the kind of mutation it is.
    """
    check_ref_format(root, name)
    args = ["switch", "-c", name] if checkout else ["branch", "--", name]
    if start_point is not None:
        args.append(_start_point(root, start_point))
    run_git(args, cwd=root, timeout=T_LOCAL)
    return {"branch": name, "checkout": checkout, "start_point": start_point}


def _start_point(root: Path, start_point: str) -> str:
    """Where a new branch starts: a ref name, or a commit id.

    A commit id is checked by shape alone and never reaches
    ``check-ref-format``, which would accept it anyway (a sha is a perfectly
    legal ref name) at the cost of a process. Whether the object EXISTS is
    git's answer a moment later, as a 404.
    """
    if SHA_RE.match(start_point.strip().lower()):
        return validate_sha(start_point)
    return check_ref_format(root, start_point)


def checkout(root: Path, target: str, *,
             kind: Literal["local", "remote"]) -> dict[str, Any]:
    """Go to a branch: one that exists, or a new one tracking a remote's.

    ``--track`` is the whole difference. Switching to ``origin/main`` cannot
    mean "put HEAD on the remote-tracking ref" -- that is a detached HEAD on
    a ref the next fetch moves -- so it means what the button says: create
    the local ``main``, set its upstream, and go there.

    A dirty working tree is git's refusal and not a check here: git compares
    what would be overwritten against what has changed, which is a finer
    answer than "there are modifications" ("would be overwritten by
    checkout" classifies as ``dirty_tree``), and re-deciding it would refuse
    switches that are perfectly safe.
    """
    check_ref_format(root, target)
    if kind == "remote":
        remote_name, slash, name = target.partition("/")
        if not slash or not name:
            raise GitError("invalid_ref", 400,
                           f"{target!r} does not name a remote branch",
                           hint="a remote branch is <remote>/<branch>")
        validate_remote_name(remote_name)
        run_git(["switch", "--track", target], cwd=root, timeout=T_LOCAL)
        return {"branch": name, "target": target, "kind": kind}

    run_git(["switch", target], cwd=root, timeout=T_LOCAL)
    return {"branch": target, "target": target, "kind": kind}


def rename_branch(root: Path, name: str, new_name: str) -> dict[str, Any]:
    """Give a branch another name; the branch may be the current one."""
    check_ref_format(root, name)
    check_ref_format(root, new_name)
    _require_branch(root, name)
    run_git(["branch", "-m", "--", name, new_name], cwd=root, timeout=T_LOCAL)
    return {"branch": new_name, "previous": name}


def delete_branch(root: Path, name: str, *,
                  force: bool = False) -> dict[str, Any]:
    """Delete a branch. ``force`` is ``-D``: delete it unmerged.

    The CURRENT branch is refused here rather than by git, and it is a 400
    with a code and not a 500 with a sentence: git answers with "cannot
    delete branch 'main' used by worktree at ..." (measured, exit 1), which
    no classification rule should have to know and which names a path the
    user did not ask about.

    Unmerged is git's own answer and stays git's: ``-d`` refuses with "the
    branch 'x' is not fully merged", the tab asks again, and the second
    request carries ``force`` -- which is a decision the user makes, so it
    travels as one rather than being taken here.
    """
    check_ref_format(root, name)
    current, _ = _head(root)
    if current is not None and current == name:
        raise GitError("invalid_value", 400,
                       f"{name} is the branch you are on",
                       hint="switch to another branch first")
    _require_branch(root, name)
    run_git(["branch", "-D" if force else "-d", "--", name], cwd=root,
            timeout=T_LOCAL)
    return {"branch": name, "forced": force}


def _require_branch(root: Path, name: str) -> None:
    """Refuse with a 404 when there is no local branch called *name*.

    git says "error: branch 'x' not found" for a delete and "fatal: no
    branch named 'x'" for a rename -- the first under an opening the
    classifier will not read as git's, the second in words no rule knows --
    so both are a 500 without this. The branch list a click comes from is up
    to fifteen seconds old, which makes this the ordinary race and not the
    exotic one.
    """
    result = run_git(["show-ref", "--verify", "--quiet",
                      f"{_HEADS}{name}"],
                     cwd=root, timeout=T_STATUS, ok_codes=(0, 1),
                     read_only=True)
    if result.returncode != 0:
        raise GitError("not_found", 404, f"there is no branch named {name!r}",
                       hint="the branch list has changed since it was read; "
                            "reload it")


def add_remote(root: Path, name: str, url: str) -> dict[str, Any]:
    """Point a name at a repository somewhere else.

    A duplicate name is git's answer and a code of its own
    (``remote_exists``): the alternative -- checking the list first and
    refusing here -- would be a second answer to the same question that can
    disagree with git's between the two calls.
    """
    validate_remote_name(name)
    validate_remote_url(url)
    run_git(["remote", "add", name, url], cwd=root, timeout=T_LOCAL)
    return {"remote": name, "url": url}


def set_remote_url(root: Path, name: str, url: str) -> dict[str, Any]:
    """Point an existing remote somewhere else."""
    validate_remote_name(name)
    validate_remote_url(url)
    run_git(["remote", "set-url", name, url], cwd=root, timeout=T_LOCAL)
    return {"remote": name, "url": url}


def remove_remote(root: Path, name: str) -> dict[str, Any]:
    """Forget a remote, and every ``branch.<x>.remote`` that named it."""
    validate_remote_name(name)
    run_git(["remote", "remove", name], cwd=root, timeout=T_LOCAL)
    return {"remote": name}
