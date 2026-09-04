"""Talking to a remote: fetch, the two halves of a pull, push and publish.

Everything else in this package finishes in milliseconds against a lock
git already holds. These four do not: they open a connection, and the
difference is not only that they are slower.

* **A pull is never ``git pull``.** It is a ``fetch`` and then a ``merge``,
  written out, because the two halves need different things. The fetch
  talks to the network and touches no file in the working tree; the merge
  touches every file and talks to nobody. Spelling them separately is what
  lets the service hold the SLOW half outside the mutation lock -- so a
  commit is not refused for the twenty seconds somebody's fetch is
  transferring -- and take that lock only for the fast local half, which is
  the one that can collide with a commit. ``git pull`` would hold both for
  the length of both, and its failure modes are the union of the two with
  no way to say which happened.
* **``@{u}`` is a literal and never a name the caller sent.** The merge
  half's whole argv is fixed text: the branch it merges is whatever
  ``branch.<current>.merge`` says, which is git's own config, so there is
  nothing here for a request to reach. That is deliberate -- it is the one
  place in this package where the safe answer was to make the value
  unreachable rather than to validate it.
* **Which remote is a decision, and it is made once** (:func:`resolve_remote`,
  R6). The upstream's, when the branch has one; the only one, when there is
  exactly one; otherwise a refusal that says which of the two problems it
  is -- no remote at all, or several and no way to choose. The tab never
  sends a remote except when publishing, so this is the answer nearly every
  request gets. A PLAIN push is the exception: its argv names no remote,
  git takes the destination from the upstream, and this module only reads
  that decision back (:func:`_tracked_remote`) rather than making one git
  would not.
* **Two of these commands are ``run_git``'s ordinary shape and one is
  not.** ``fetch`` and ``push`` report everything on stderr, so the runner's
  own classification is right for them. ``merge`` reports a CONFLICT on
  STDOUT with an empty stderr (measured on git 2.53), which is
  ``stash pop``'s trap exactly -- so the merge joins both streams before it
  classifies, or the commonest outcome a pull has would be a 500.

None of these commands takes a pathspec, so ``--literal-pathspecs`` (which
every call here keeps) has nothing to act on, and none of them runs ``git
clean`` under the covers the way ``stash push`` does -- checked against
the exemption list in :func:`~app.core.git.runner.run_git`, because a
command that silently did half its job with the option on is a bug this
package has already paid for once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from . import refs, repo
from .errors import GitError, classify_failure
from .models import GitStatus
from .paths import validate_remote_name
from .runner import T_LOCAL, T_NETWORK, T_STATUS, run_git

#: How a pull brings the remote's work in. ``ff-only`` moves the branch or
#: refuses; ``merge`` writes a merge commit and can leave conflicts.
PullStrategy = Literal["ff-only", "merge"]

#: The upstream branch, in git's own syntax. A FIXED LITERAL: it is the one
#: ref this package merges, it comes out of the repository's own config
#: rather than out of a request, and ``paths.validate_branch_name`` refuses
#: ``@{`` in everything a caller CAN send -- so there is no spelling of a
#: branch name that could arrive here and be mistaken for it.
UPSTREAM = "@{u}"

#: The merge each strategy is. ``--ff-only`` refuses anything that is not a
#: fast-forward ("Not possible to fast-forward", which classifies as
#: ``diverged``); ``--no-edit`` takes git's own merge message rather than
#: opening an editor -- which the runner's ``GIT_EDITOR=:`` would already
#: have made a no-op, but a server should not depend on an editor being
#: broken to avoid opening one.
MERGE_ARGS: dict[str, tuple[str, ...]] = {
    "ff-only": ("merge", "--ff-only", UPSTREAM),
    "merge": ("merge", "--no-edit", UPSTREAM),
}


# --- which remote ------------------------------------------------------------


def resolve_remote(root: Path, status: GitStatus,
                   requested: str | None = None) -> str:
    """Which remote this operation talks to (R6), or the refusal it is.

    Three answers in order, and the first that applies wins.

    A remote the CALLER named is used -- once it is confirmed to be one of
    this repository's. That check is not politeness: ``git fetch -- nope``
    answers a name that is not configured with "'nope' does not appear to
    be a git repository" followed by "Could not read from remote
    repository" (measured), which classifies as ``network`` -- so a Publish
    click on a panel drawn before somebody removed the remote would tell
    the user their connection is down. A 404 says what actually happened,
    the way ``refs._require_branch`` does for a branch that has gone.

    Otherwise the UPSTREAM's remote, which is the answer for every branch
    anybody has already published: ``origin/main`` means ``origin``, and
    everything before the first slash is the remote even for a branch
    called ``feat/x``.

    Otherwise the only remote there is, and a refusal when that is not a
    question with one answer: none at all is ``no_remote`` (409 -- there is
    nothing to configure a choice between), several is a 400 saying to
    publish the branch first, which is the click that MAKES the choice and
    records it.
    """
    if requested is not None:
        return _known_remote(root, requested)
    upstream = _upstream_remote(status)
    if upstream is not None:
        return upstream

    remotes = refs.list_remotes(root)
    if not remotes:
        raise _no_remote()
    if len(remotes) > 1:
        raise GitError("invalid_value", 400,
                       "this branch does not say which remote it belongs to",
                       hint="several remotes -- publish the branch first")
    return validate_remote_name(remotes[0].name)


def _upstream_remote(status: GitStatus) -> str | None:
    """The remote the upstream names, or None when there is no upstream.

    Validated even though it came out of git's own config: a remote name
    that reached a command line unchecked is the one shape this module is
    not allowed to have.
    """
    if not status.upstream:
        return None
    return validate_remote_name(status.upstream.partition("/")[0])


def _no_remote() -> GitError:
    """The refusal for a repository with nothing configured to talk to."""
    return GitError("no_remote", 409, "this repository has no remote",
                    hint="add a remote before pushing or fetching")


def _tracked_remote(root: Path, status: GitStatus) -> str | None:
    """Where a PLAIN push goes: git's decision, read here rather than made.

    A plain push is the one operation here that does not go through
    :func:`resolve_remote`, because its argv names no remote: git sends
    the branch to the remote its upstream records, and its answer for a
    branch that has none -- "The current branch main has no upstream
    branch", ``no_upstream`` (measured) -- is the code the Publish button
    hangs off. Resolving the remote the way a fetch does answered that
    same state ``invalid_value`` whenever several remotes existed, before
    git could speak, and R10 has no button for that code.

    So the upstream's remote when there is one, and None when there is
    not. git may still push then: ``push.default=current`` sends the
    branch to the only remote, or to ``origin`` among several (measured),
    from config this package does not read -- and a remote it did not
    choose would be a guess dressed as an answer.

    One state IS refused first. With no remote at all git says "No
    configured push destination", the same sentence it prints when
    several remotes exist and none is called ``origin`` -- so no
    classifier row can tell "add a remote" from "publish the branch", and
    ``no_remote`` is what every other operation here answers for the
    state (R14 hides the button for it). It costs one ``remote -v``, only
    on a branch with no upstream, where git was about to refuse anyway.
    """
    upstream = _upstream_remote(status)
    if upstream is not None:
        return upstream
    if not refs.list_remotes(root):
        raise _no_remote()
    return None


def _known_remote(root: Path, name: str) -> str:
    """*name*, once it is one of this repository's remotes; else a 404."""
    validate_remote_name(name)
    if name not in {remote.name for remote in refs.list_remotes(root)}:
        raise GitError("not_found", 404, f"there is no remote named {name!r}",
                       hint="the remote list has changed since it was read; "
                            "reload it")
    return name


# --- the operations ----------------------------------------------------------


def fetch(root: Path, remote: str | None = None) -> dict[str, Any]:
    """Bring the remote's refs up to date. Nothing on disk moves.

    ``--prune`` is not optional here, and it is why the branch list can be
    trusted: without it a remote-tracking ref outlives the branch it
    mirrors, so a branch somebody deleted last week goes on being offered
    as something to switch to, and ``%(upstream:track)`` goes on reporting
    a count against a ref that is not there. With it, the same fetch that
    finds new work also removes what is gone, and ``BranchInfo.gone`` says
    so on the next read.
    """
    name = resolve_remote(root, repo.read_status(root), remote)
    run_git(["fetch", "--prune", "--", name], cwd=root, timeout=T_NETWORK)
    return {"remote": name}


def merge_upstream(root: Path,
                   strategy: PullStrategy = "ff-only") -> dict[str, Any]:
    """The second half of a pull: merge what the fetch brought in.

    Local, and quick, which is the point of it being its own step -- see
    the module docstring.

    **Both streams are classified**, for the reason ``stash._restore``
    does it: a merge that conflicts exits 1 with ``CONFLICT (content):
    ...`` on STDOUT and an EMPTY stderr (measured on git 2.53), so stderr
    alone answers the most ordinary failure a pull has with ``git_failed``
    -- a 500 for the state the tab draws its whole merge group for. Joined,
    it is ``conflict`` (409), the merge is left IN PROGRESS with the
    markers on disk, and the panel picks it up from
    ``status.merge_in_progress``.

    The other refusals classify from stderr and are unchanged by joining:
    ``--ff-only`` against a diverged branch is "fatal: Not possible to
    fast-forward, aborting." (``diverged``, exit 128), and a branch with no
    upstream -- or one whose upstream was deleted and pruned -- is
    ``no_upstream``, which is the code the Publish button hangs off.

    ``head_moved`` is read rather than parsed out of git's prose: HEAD
    before and after, two ``rev-parse`` calls that cost nothing next to the
    fetch that preceded them. It is what tells the tab whether to say
    "pulled" or "already up to date", and "Already up to date." is a
    sentence in a language the person reading may not have.
    """
    before = _head_sha(root)
    # ``T_LOCAL``, not ``T_NETWORK``: this half talks to nobody. What it can
    # take time over is the user's own hooks, which is what that budget is
    # for -- and a merge still running after thirty seconds is holding the
    # index lock, not a socket.
    result = run_git(list(MERGE_ARGS[strategy]), cwd=root, timeout=T_LOCAL,
                     ok_codes=(0, 1, 128))
    if result.returncode != 0:
        raise classify_failure(result.argv, result.returncode,
                               f"{result.out}\n{result.err}")
    return {"step": "merge", "strategy": strategy,
            "head_moved": _head_sha(root) != before}


def _head_sha(root: Path) -> str | None:
    """The commit HEAD is on, or None on an unborn branch.

    ``rev-parse`` exits 128 with nothing to resolve before the first
    commit, which is not a failure to a caller that is asking whether HEAD
    moved: it did not, and both sides of the comparison say so the same
    way.
    """
    result = run_git(["rev-parse", "--verify", "--quiet", "HEAD"], cwd=root,
                     timeout=T_STATUS, ok_codes=(0, 1, 128), read_only=True)
    return result.out.strip() or None


def push(root: Path, *, remote: str | None = None,
         set_upstream: bool = False) -> dict[str, Any]:
    """Send this branch to its remote, adopting it first when publishing.

    Two argv, and the difference is a branch that has been sent somewhere
    before.

    A PLAIN push has no positional arguments at all: git sends the current
    branch to the ref its upstream names, which is the only destination
    that keeps the ahead/behind in the panel meaning anything. A branch
    with no upstream is git's own refusal ("The current branch main has no
    upstream branch", ``no_upstream``), and the tab answers it with the
    Publish button rather than by guessing -- so no remote is resolved for
    it here, only read back for the answer (:func:`_tracked_remote`).

    A PUBLISH is ``push -u -- <remote> <branch>``: a remote that was named
    or resolved, the branch HEAD is on, and ``-u`` to record the pairing so
    every later push is the plain one. The branch goes through
    ``check_ref_format`` even though it came from git a moment ago --
    that is the module's rule for every name that reaches a command line,
    and the one place it could be skipped is the one place a reader would
    have to check.

    Two states are refused before either runs, because git's own answers
    for them are worse than a code:

    * a DETACHED HEAD. git's plain push says ``detached_head`` and its
      publish would need a ``HEAD:<name>`` refspec nobody asked for; more
      to the point, a user's ``push.default=matching`` turns a detached
      plain push into "send every branch that exists on both sides", which
      is a write nobody clicked for.
    * an UNBORN branch. ``push -u`` answers it with "error: src refspec
      main does not match any" -- a 404 whose message is about a refspec
      the user never typed. ``nothing_to_commit`` is the code that says
      what is actually true: there is nothing here to send yet. It is a
      409 like the rest of the "the repository is not in a state for this"
      family, and the hint carries the fact.
    """
    # First, and before any process: a request that names a remote without
    # asking to publish is a client bug, and it costs nothing to say so.
    if remote is not None and not set_upstream:
        raise GitError("invalid_value", 400,
                       "a plain push goes where the upstream says",
                       hint="send set_upstream=true to publish this branch "
                            "to a remote you name")

    status = repo.read_status(root)
    branch = _pushable_branch(status)
    if not set_upstream:
        name = _tracked_remote(root, status)
        run_git(["push"], cwd=root, timeout=T_NETWORK)
        return {"remote": name, "branch": branch, "published": False}

    name = resolve_remote(root, status, remote)
    refs.check_ref_format(root, branch)
    run_git(["push", "-u", "--", name, branch], cwd=root, timeout=T_NETWORK)
    return {"remote": name, "branch": branch, "published": True}


def _pushable_branch(status: GitStatus) -> str:
    """The branch a push is about, or the refusal this repository's state is.

    Decided from the same three flags the header is drawn from, so the
    answer can never disagree with what the person clicking can see -- the
    argument ``service.abort_merge`` and ``stash._refuse_a_half_finished_merge``
    both make.
    """
    if status.detached or status.branch is None:
        raise GitError("detached_head", 409, "HEAD is not on a branch",
                       hint="switch to a branch before pushing")
    if status.unborn:
        raise GitError("nothing_to_commit", 409,
                       "there is nothing to push yet",
                       hint="this branch has no commits yet")
    return status.branch
