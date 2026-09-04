"""The Source Control tab's one entry point: what may be asked, and when.

Every route in ``routes_git`` calls a method here and nothing else. What
this layer adds to the plain functions underneath it is the three decisions
that cannot be made per operation:

* **WHICH repository.** The project directory and nothing else --
  ``settings.PROJECT_DIR``, injected so a test never has to monkeypatch it.
  :func:`resolve_repo` resolves the state in one order (no project, no git,
  a git too old, not the top level of a repository, ready) and the first
  answer wins. The ``not_repo`` case carries ``nested_toplevel`` when the
  project sits INSIDE some other repository -- the CodefyUI checkout, a
  home directory somebody once ran ``git init`` in -- because the one thing
  the tab must never do is operate on that repository. It would look like
  it worked.
* **ONE mutation at a time.** git serialises writes with the index lock and
  answers a loser with a message about ``.git/index.lock`` that means
  nothing to anybody. So a write takes an ``asyncio.Lock`` first, and a
  second write while one is running is refused immediately with
  :class:`~app.core.git.errors.GitBusy` naming the operation that holds it.
  The check and the acquisition have no ``await`` between them, which is
  what makes "one at a time" true rather than likely. READS never take the
  lock: a status poll must not queue behind a commit that is running the
  user's hooks.
* **TWO locks, because a fetch is not a commit.** A network operation can
  run for a minute and touches no file in the working tree, so putting it
  on the mutation lock would refuse every commit for as long as somebody's
  push was transferring -- a panel that goes dead while the thing it is
  doing works fine. :attr:`GitService.network_lock` is its own queue: one
  network operation at a time (two pushes to the same branch is a race
  nobody wins), local writes untouched. A ``pull`` is the operation that
  needs both, one after the other, and it is written out as two steps for
  exactly that reason -- the fetch under the network lock, the merge under
  the mutation one. Nothing here ever takes the mutation lock and THEN the
  network lock, which is what keeps two locks from being a deadlock.
* **EVERY write answers with the status it left behind.** ``mutate`` reads a
  fresh status after the operation and returns it with the result, so the
  tab never draws a panel one operation out of date, and computes
  ``changed_paths`` -- which files this actually moved -- for the editor to
  reload. If that read fails, the REQUEST fails: a write whose result cannot
  be read back is not a success with a hole in it.

Nothing here blocks the event loop. Every git call happens inside
``asyncio.to_thread``, including the ones the checks themselves make, which
is why the sync helpers below take a ``root`` and know nothing about the
service: the same functions are what a test calls directly.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, TypeVar

from ...config import settings
from . import diff as diff_ops
from . import log as log_ops
from . import network as network_ops
from . import refs as refs_ops
from . import repo
from . import stash as stash_ops
from .errors import GitBusy, GitError, classify_failure
from .models import (
    BranchesResponse,
    DiffResponse,
    FileAtRef,
    GitFile,
    GitStatus,
    Identity,
    LogResponse,
    MutationResult,
    RemoteInfo,
    RepoInfo,
    StashInfo,
    StatusResponse,
)
from .paths import (
    validate_commit_message,
    validate_rel_paths,
)
# Re-exported: the branch-name gate moved to ``refs`` with the operations
# that need it, and this is still the name every caller learned it under.
from .refs import check_ref_format  # noqa: F401
from .runner import (
    T_LOCAL,
    # Re-exported: the network timeout is passed by ``network.py`` now, and
    # this is still the name a caller outside the package would look for it
    # under. The four live in ``runner`` because every module that runs git
    # needs them and this one imports all of those.
    T_NETWORK,  # noqa: F401
    T_READ,
    T_STATUS,
    git_executable,
    git_version,
    run_git,
)

_T = TypeVar("_T")

#: The oldest git the tab works with. ``restore`` and ``switch`` -- which
#: are how unstage and discard are spelled here, because ``checkout`` means
#: four different things -- arrived in 2.23. Below it the operations exist
#: under other names with other footguns, and refusing with a version number
#: is a better answer than half a tab.
MIN_GIT_VERSION = (2, 23, 0)

#: The states in which no git command can be run at all -- as opposed to
#: ``not_repo``, which ``init`` exists to fix.
_UNUSABLE_STATES = ("no_project", "git_missing", "git_too_old")

#: How many paths one git process is given by the whole-tree writes, which
#: name every file they touch (see :func:`_run_over`). Windows stops a
#: command line at about 32,000 characters; 200 paths is a few kilobytes
#: with room for the longest name anybody has.
PATHS_PER_PROCESS = 200


# --- the repository --------------------------------------------------------


def open_project_dir() -> Path | None:
    """The project directory the server was started with, or None.

    A function rather than a value read at import time: ``settings`` is
    mutable in tests and the CLI, and a service built once at startup must
    still see the directory the server was actually given.
    """
    return settings.PROJECT_DIR


def resolve_repo(project_dir: Path | None) -> RepoInfo:
    """How far the tab can get with *project_dir*, and why not further.

    Cheap checks first, so the common failure (no project open) costs no
    process at all, and the one that needs git (``--show-toplevel``) runs
    only once the git it needs is known to be there and new enough.

    A version that cannot be READ is reported as ``git_too_old``: something
    on PATH is called git and will not say what it is, so it cannot be
    confirmed to have the commands this tab is built on. ``git_version`` is
    then None, which is the frontend's clue that the number is not the
    problem.
    """
    if project_dir is None:
        return RepoInfo(state="no_project")

    root = Path(project_dir)
    where = str(root)
    if git_executable() is None:
        return RepoInfo(state="git_missing", project_dir=where)

    version = git_version()
    text = _version_text(version)
    if version is None or version < MIN_GIT_VERSION:
        return RepoInfo(state="git_too_old", project_dir=where,
                        git_version=text)

    if not root.is_dir():
        return RepoInfo(state="not_repo", project_dir=where, git_version=text)
    toplevel = _toplevel(root)
    if toplevel is None:
        return RepoInfo(state="not_repo", project_dir=where, git_version=text)
    if not _same_dir(toplevel, root):
        # The project is inside SOMEBODY ELSE'S repository. Reported, and
        # still ``not_repo``: every operation the tab offers would be
        # applied to that repository, which nobody asked about.
        return RepoInfo(state="not_repo", project_dir=where, git_version=text,
                        nested_toplevel=str(toplevel))
    return RepoInfo(state="ready", project_dir=where, git_version=text)


def _toplevel(root: Path) -> Path | None:
    """The top level of the repository *root* is in, or None if there is none.

    ``rev-parse`` exits 128 both for "not a repository" and for a directory
    git cannot enter; both mean the same thing here.
    """
    result = run_git(["rev-parse", "--show-toplevel"], cwd=root,
                     timeout=T_STATUS, ok_codes=(0, 128), read_only=True)
    text = result.out.strip()
    if result.returncode != 0 or not text:
        return None
    return Path(text)


def _same_dir(left: Path, right: Path) -> bool:
    """Do two paths name the same directory?

    Resolved and case-folded, because git prints ``D:/Project`` where the
    settings hold ``D:\\project`` and a substring comparison of those two is
    how a nested repository gets mistaken for the right one.
    """
    return (os.path.normcase(str(left.resolve()))
            == os.path.normcase(str(right.resolve())))


def _version_text(version: tuple[int, int, int] | None) -> str | None:
    """``(2, 53, 0)`` as ``"2.53.0"``; None stays None."""
    return None if version is None else ".".join(str(part) for part in version)


def _state_error(info: RepoInfo) -> GitError:
    """The failure that a non-``ready`` state is, for an operation that needs one.

    ``GET /api/git/status`` reports these as a 200 with a state, because
    "no project is open" is a screen; every other route has to fail, and
    this is the one place that decides how.
    """
    if info.state == "no_project":
        return GitError("no_project", 409,
                        hint="start the server with --project <dir>")
    if info.state == "git_missing":
        return GitError("git_missing", 503,
                        hint="install git and make sure it is on PATH")
    if info.state == "git_too_old":
        return GitError(
            "git_too_old", 409,
            hint=f"this git is {info.git_version or 'unreadable'}; "
                 f"{'.'.join(str(part) for part in MIN_GIT_VERSION)} or newer "
                 f"is needed")
    if info.nested_toplevel:
        return GitError(
            "not_repo", 409,
            "the project directory is inside another repository",
            hint=f"{info.nested_toplevel} is a repository, but the project "
                 f"directory is not; initialise one here to use this tab")
    return GitError("not_repo", 409,
                    hint="initialise a repository in the project directory")


# --- the operations, as functions of a root --------------------------------


def _selection(paths: Sequence[str] | None, all_paths: bool
               ) -> list[str] | None:
    """One of the two forms: these paths, or the whole tree (``None``).

    The same rule ``PathsRequest`` enforces on the wire, enforced again here
    because this is not only reached from a route. ``git add -A --`` with an
    empty pathspec stages EVERYTHING, so a caller that means "the selection"
    and passes an empty one must be refused rather than promoted.
    """
    if all_paths:
        if paths:
            raise GitError("invalid_value", 400,
                           "send either paths or all, not both")
        return None
    if not paths:
        raise GitError("invalid_path", 400, "no paths were given",
                       hint="use all=true to act on the whole tree")
    return list(paths)


def _writable_paths(root: Path, paths: Sequence[str]) -> list[str]:
    """The paths of one write, validated and checked for a link in the way.

    Every NAMED path a mutation acts on goes through here, and nowhere else
    does: ``validate_rel_path`` allows a link (it must -- see
    ``repo.refuse_link_parents``), and a path whose PARENT is a link or a
    Windows junction resolves inside the project and passes every other
    check while naming a file somewhere else entirely. That is measured, not
    theoretical: it is how ``discard`` once deleted the file a gitignored
    folder held.

    The whole-tree forms ask the same question through
    :func:`_tree_selection`, and they have to: git DOES descend a junction.
    Measured on Windows -- with ``proj/notes`` a junction to a folder
    outside the project, ``git status`` lists the files under it as
    untracked and ``git add -A -- .`` stages them into this repository,
    which is the same write the per-path guard refuses. The destructive
    half of ``discard`` is the exception and stays a whole-tree ``clean -fd
    -- .``: git removes the LINK, never what it points at (measured, in
    both the junction and the symlink shape).
    """
    clean = validate_rel_paths(root, paths)
    for path in clean:
        repo.refuse_link_parents(root, path)
    return clean


def _tree_selection(root: Path, paths: Iterable[str]
                    ) -> tuple[list[str], list[str]]:
    """Split a whole-tree write into ``(what to do, what to leave alone)``.

    The paths come from a fresh status -- git's own spelling of every file
    it would touch -- and each of them meets the SAME guard a named path
    does, because "all" is not a different kind of consent. Without it,
    ``git add -A -- .`` staged files from outside the project through a
    junction while ``stage(["notes/keys.txt"])`` was refused: the same
    write, allowed or not depending on which button the user pressed.

    A refused path is SKIPPED and reported, not fatal. One link in a
    project must not make "stage everything" impossible, and the caller
    puts the skipped list in ``detail`` so the tab can say what it left
    alone rather than quietly doing less than it claimed.

    Sorted and deduplicated: a file can be in two groups of one status
    (``MM`` is staged and unstaged), and a stable order makes the argv --
    and the skipped list the tab shows -- reproducible.
    """
    kept: list[str] = []
    skipped: list[str] = []
    for path in sorted(set(paths)):
        if repo.link_parent_refusal(root, path) is None:
            kept.append(path)
        else:
            skipped.append(path)
    return kept, skipped


def _run_over(root: Path, args: Sequence[str], paths: Sequence[str], *,
              timeout: float) -> None:
    """Run ``git <args> -- <paths>``, in chunks a command line can hold.

    A working tree can hold more paths than Windows allows in one command
    line (about 32,000 characters), and a whole-tree write names every one
    of them. :data:`PATHS_PER_PROCESS` at a time is a few kilobytes per
    process, which git spends longer working on than it costs to start.

    An empty list runs NOTHING, which is the point rather than an edge
    case: every command these arguments spell treats an absent pathspec as
    "the whole tree", so "nothing to do" must never reach git as "do it
    all".
    """
    for start in range(0, len(paths), PATHS_PER_PROCESS):
        chunk = paths[start:start + PATHS_PER_PROCESS]
        run_git([*args, "--", *chunk], cwd=root, timeout=timeout)


def stage_paths(root: Path, paths: Sequence[str] | None = None
                ) -> dict[str, Any]:
    """Stage *paths*, or everything a fresh status names when *paths* is None.

    ``add -A`` for both forms, because "stage this file" has to include
    staging its DELETION -- a file the user deleted is a change like any
    other, and plain ``add`` would silently skip it.

    The whole-tree form is the status's own three groups -- unstaged,
    untracked and conflicted -- named one by one instead of ``.``, so that
    a path reaching out of the project through a link is skipped and said
    so (:func:`_tree_selection`). A conflicted path is in the list because
    ``add`` is how a resolution is marked, which is what "Stage All" means
    in the middle of a merge.
    """
    if paths is None:
        status = repo.read_status(root)
        kept, skipped = _tree_selection(
            root, (entry.path
                   for group in (status.unstaged, status.untracked,
                                 status.conflicted)
                   for entry in group))
        _run_over(root, ["add", "-A"], kept, timeout=T_LOCAL)
        return {"all": True, "skipped": skipped}
    clean = _writable_paths(root, paths)
    run_git(["add", "-A", "--", *clean], cwd=root, timeout=T_LOCAL)
    return {"paths": clean}


def unstage_paths(root: Path, paths: Sequence[str] | None = None
                  ) -> dict[str, Any]:
    """Take *paths* (or everything) back out of the index.

    An UNBORN branch needs the other spelling: ``restore --staged`` and
    ``reset`` both resolve HEAD to know what to restore TO, and there is no
    HEAD yet, so the answer for a repository whose first commit has not
    happened is ``rm --cached`` -- remove it from the index, leave it on
    disk. Which state we are in is read, not guessed.

    The whole-tree form names the staged files rather than running a bare
    ``reset``, for the reason in :func:`_tree_selection`; the same list goes
    to whichever of the two commands this repository's HEAD allows. A
    conflicted path is NOT in it: it is not staged, and taking it out of
    the index would throw away the three versions a merge tool needs.

    A staged RENAME is two paths and one entry. porcelain v2 reports it as
    a single ``2`` record carrying the new name and the old one, so a list
    built from ``entry.path`` alone names only half of it -- and restoring
    half a rename leaves the other half staged: ``restore --staged b.txt``
    after ``git mv a.txt b.txt`` leaves ``a.txt`` staged as a DELETION
    (measured on git 2.53), so "unstage everything" left the index dirty
    and the panel showed a deletion nobody asked for. Both names go in.
    """
    status = repo.read_status(root)
    unborn = status.unborn
    if paths is None:
        kept, skipped = _tree_selection(
            root, (name for entry in status.staged
                   for name in (entry.path, entry.orig_path) if name))
        args = (["rm", "--cached", "-r", "-q"] if unborn
                else ["restore", "--staged"])
        _run_over(root, args, kept, timeout=T_LOCAL)
        return {"all": True, "skipped": skipped}

    clean = _writable_paths(root, paths)
    args = (["rm", "--cached", "-r", "-q", "--", *clean] if unborn
            else ["restore", "--staged", "--", *clean])
    run_git(args, cwd=root, timeout=T_LOCAL)
    return {"paths": clean}


def discard_paths(root: Path, paths: Sequence[str] | None = None
                  ) -> dict[str, Any]:
    """Throw away the working-tree changes to *paths* (or to everything).

    The only operation here that DESTROYS something the user cannot get
    back, so every part of it is decided from a fresh status rather than
    from what the request claims:

    * a tracked file with unstaged changes is restored from the index --
      ``restore --worktree`` only, so a file that is staged AND modified
      (``MM``) keeps what was staged. That is what "Discard Changes" does in
      every editor that has the button, and the other reading (throwing the
      staged version away too) is not recoverable.
    * an untracked file is DELETED (``clean -f``), which is the only thing
      "discard" can mean for a file that has no other copy.
    * a path in neither list is a 400. It is either already clean or gone,
      and the request was built from a status that has since changed.
    * a path THROUGH a link or a junction is refused before anything runs
      (:func:`_writable_paths`): ``clean -f`` down one deletes the file at
      the other end of it, which is how this operation once emptied a
      gitignored folder of secrets.
    * a SUBMODULE is refused. ``restore --worktree`` on a gitlink succeeds
      and does nothing at all (measured on git 2.53), so the tab would
      report a discard that did not happen; the changes are in the other
      repository, where this tab does not go.

    The whole-tree form restores the unstaged files by NAME and then runs
    ``clean -fd``, never ``-x``: ignored files are the user's ``.env``,
    their virtual environment and their model weights, and "discard my
    changes" has never meant "delete those".
    """
    if paths is None:
        # The whole-tree restore NAMES its files rather than passing ``.``,
        # and there are three reasons, all measured on git 2.53.
        #
        # A path that reaches its file through a link or a junction is
        # skipped and reported, like the other two whole-tree writes -- see
        # ``_tree_selection``.
        #
        # ``restore --worktree -- .`` exits 1 with "error: path 'a.txt' is
        # unmerged" as soon as one conflicted file sits beside an ordinary
        # modification -- and it restores NOTHING, so "discard everything"
        # during a conflict answered 500 and left every change in place.
        # The pathspec cannot express "all but the unmerged ones"; the
        # unstaged group already is that list, because an unmerged path is
        # a ``u`` record and is not in it.
        #
        # It also exits 1 for "pathspec '.' did not match any file(s) known
        # to git" when the INDEX is empty -- a repository somebody has just
        # initialised -- and the clean would then never run. An empty list
        # skips the process instead of tolerating a failure it cannot tell
        # apart from a real one.
        restore, skipped = _tree_selection(
            root, (entry.path for entry in repo.read_status(root).unstaged))
        _run_over(root, ["restore", "--worktree"], restore, timeout=T_LOCAL)
        # The one whole-tree ``.`` that stays, because it is the safe half:
        # ``clean -fd`` removes the LINK itself and never what it points at
        # (measured, both shapes -- the junction went, the folder outside
        # the project it pointed at was untouched). Naming the untracked
        # files instead would leave every dangling link in place, which is
        # not what "discard everything" means.
        run_git(["clean", "-fd", "--", "."], cwd=root, timeout=T_LOCAL)
        return {"all": True, "skipped": skipped}

    clean = _writable_paths(root, paths)
    submodules = repo.submodule_paths(root, clean)
    if submodules:
        raise GitError("invalid_path", 400,
                       f"{submodules[0]} is a submodule",
                       hint=repo.SUBMODULE_HINT)

    status = repo.read_status(root)
    tracked = {entry.path for entry in status.unstaged}
    untracked = {entry.path for entry in status.untracked}
    unknown = [path for path in clean
               if path not in tracked and path not in untracked]
    if unknown:
        raise GitError("path_not_in_status", 400,
                       f"{unknown[0]} has no changes to discard",
                       hint="the status has changed since it was read; "
                            "reload it")

    # Both lists are computed before either command runs, so a request that
    # names one of each is one decision and not two.
    restore = [path for path in clean if path in tracked]
    remove = [path for path in clean if path in untracked and path not in tracked]
    if restore:
        run_git(["restore", "--worktree", "--", *restore], cwd=root,
                timeout=T_LOCAL)
    if remove:
        run_git(["clean", "-f", "--", *remove], cwd=root, timeout=T_LOCAL)
    return {"paths": clean, "restored": len(restore), "removed": len(remove)}


def commit_changes(root: Path, message: str, *, all_paths: bool = False,
                   amend: bool = False) -> dict[str, Any]:
    """Make a commit. Returns the sha it made.

    *all_paths* is VS Code's "Commit All": stage the whole tree first, which
    is one ``add -A`` and not a different commit command.

    The message travels on STDIN (``--file=-``) and is never an argument: it
    is the one value here that is meant to contain newlines, and an argument
    starting with ``-`` is an option.

    Two failures need help from this function rather than from the
    classifier. AMENDING with no commit to amend is a 404 decided from the
    status, because git answers it with "You have nothing to amend", which
    is not a phrase any rule should have to know. And a commit that finds
    nothing to commit says so on STDOUT -- "nothing added to commit but
    untracked files present" -- so both streams are joined before they are
    classified; stderr alone would make the most ordinary empty commit a
    500.
    """
    text = validate_commit_message(message)
    if amend and repo.read_status(root).unborn:
        raise GitError("not_found", 404, "there is no commit to amend",
                       hint="this branch has no commits yet")

    if all_paths:
        run_git(["add", "-A", "--", "."], cwd=root, timeout=T_LOCAL)

    args = ["commit", "-q"]
    if amend:
        args.append("--amend")
    args.append("--file=-")
    result = run_git(args, cwd=root, timeout=T_LOCAL,
                     input_bytes=text.encode("utf-8"),
                     # 1 is an empty commit or a hook that said no, 128 an
                     # identity or a lock problem: all of them are handled
                     # below, against BOTH streams.
                     ok_codes=(0, 1, 128))
    if result.returncode != 0:
        raise classify_failure(result.argv, result.returncode,
                               f"{result.out}\n{result.err}")

    sha = run_git(["rev-parse", "HEAD"], cwd=root, timeout=T_STATUS,
                  read_only=True).out.strip()
    return {"sha": sha, "short": sha[:7]}


def init_repo(root: Path) -> dict[str, Any]:
    """``git init`` plus the scaffold that keeps ``.env`` out of history."""
    repo.init(root)
    return {"scaffold": repo.ensure_scaffold(root)}


# --- finishing a merge -------------------------------------------------------


def abort_merge(root: Path) -> dict[str, Any]:
    """Undo a merge that is half-done, and put the tree back as it was.

    The pre-check is not defence in depth: ``git merge --abort`` with no
    ``MERGE_HEAD`` says "fatal: There is no merge to abort (MERGE_HEAD
    missing)." and exits 128, which matches no classification rule and
    reached the browser as a 500. Deciding it here makes it a 404 whose
    hint is the fact, and it is decided from ``merge_in_progress`` -- the
    SAME flag the button is drawn from, so the answer can never disagree
    with what the person clicking it can see.
    """
    if not repo.read_status(root).merge_in_progress:
        raise GitError("not_found", 404, "there is no merge to abort",
                       hint="no merge in progress")
    run_git(["merge", "--abort"], cwd=root, timeout=T_LOCAL)
    return {}


def resolve(root: Path, path: str,
            side: Literal["ours", "theirs", "mark"]) -> dict[str, Any]:
    """Settle one conflicted file: keep ours, take theirs, or mark it done.

    ``ours`` and ``theirs`` are ``checkout --ours|--theirs`` followed by an
    ``add``; ``mark`` is the ``add`` alone, which is what a person who has
    edited the file by hand means by "resolved" -- git does not read the
    file, it reads the index, and staging an unmerged path is exactly how a
    resolution is recorded.

    **The path must be one the status calls conflicted, and that check is
    the safety.** ``git checkout --ours -- b.txt`` on a file that is merely
    MODIFIED exits 0 and silently restores it from HEAD (measured on git
    2.53) -- the user's edit is gone, with no error and nothing to undo
    from. A stale panel that offered "Keep mine" on the wrong row would
    destroy work; a 400 saying the file is not in the status tells the tab
    to reload instead.

    It is a THIRD status read -- the service already reads one before and
    one after every worktree operation -- and that is the price knowingly
    paid. A resolve is one deliberate click per conflicted file, not a
    poll, and the read is what keeps a command that destroys silently away
    from a row the panel drew fifteen seconds ago.
    """
    clean = _writable_paths(root, [path])[0]
    conflicted = {entry.path for entry in repo.read_status(root).conflicted}
    if clean not in conflicted:
        raise GitError("path_not_in_status", 400,
                       f"{clean} is not one of the conflicted files",
                       hint="the status has changed since it was read; "
                            "reload it")
    if side != "mark":
        run_git(["checkout", f"--{side}", "--", clean], cwd=root,
                timeout=T_LOCAL)
    run_git(["add", "--", clean], cwd=root, timeout=T_LOCAL)
    return {"path": clean, "side": side}


# --- what changed ----------------------------------------------------------


def _entries(status: GitStatus) -> set[tuple[str, str]]:
    """``(path, xy)`` for every file in every group of *status*.

    The PAIR, not the path: a file that goes from ``MM`` to ``M.`` has
    changed even though it is in both statuses, and a file that appears in
    two groups is two entries in both.
    """
    return {(entry.path, entry.xy)
            for group in (status.staged, status.unstaged, status.untracked,
                          status.conflicted)
            for entry in group}


def changed_paths(root: Path, before: GitStatus | None, after: GitStatus,
                  paths: Sequence[str] = ()) -> list[str]:
    """Which files one operation moved, for the editor to reload.

    Three sources, unioned: what the status says is different, what the
    request itself named (a discard that restored a file to exactly what the
    index had leaves no trace in the status, and the editor still has the
    other version open), and -- when HEAD moved -- what the two commits
    differ in, which is how a commit or an amend reports the files it
    swallowed.

    The HEAD diff is best-effort: it is a nicety for the UI, and a commit
    that succeeded must not be reported as a failure because the diff after
    it did not run.
    """
    changed = set(paths)
    if before is not None:
        changed.update(path for path, _ in _entries(before) ^ _entries(after))
        if before.head and after.head and before.head != after.head:
            changed.update(_names_between(root, before.head, after.head))
    return sorted(changed)


def _names_between(root: Path, before_head: str, after_head: str) -> list[str]:
    """The files two commits differ in; empty when that cannot be answered."""
    try:
        result = run_git(["diff", "--name-only", "-z", before_head, after_head,
                          "--"],
                         cwd=root, timeout=T_READ, read_only=True)
    except GitError:
        return []
    return [name for name in result.out.split("\x00") if name]


# --- the service ------------------------------------------------------------


class GitService:
    """One instance on ``app.state``; every git route goes through it."""

    def __init__(self,
                 *,
                 project_dir: Callable[[], Path | None] | None = None) -> None:
        # Injected so a test can point the service at a temporary repository
        # without monkeypatching global settings -- the same reason
        # ``PackService`` takes its flow as an argument.
        self._project_dir = project_dir if project_dir is not None else open_project_dir
        #: Held for the length of one mutation. Public because the routes
        #: report "busy" from it and a test parks on it.
        self.lock = asyncio.Lock()
        #: What the lock is being held FOR, so a refusal can name it.
        self.current_op: str | None = None
        #: Held for the length of one NETWORK operation, which is a
        #: different queue: a fetch may run for a minute and moves no file,
        #: so a commit must not wait behind it (see the module docstring).
        self.network_lock = asyncio.Lock()
        #: What the network lock is being held FOR.
        self.current_network_op: str | None = None

    # --- reads (never take the lock) ------------------------------------

    async def repo_info(self) -> RepoInfo:
        """Whether the tab can talk to a repository at all, and why not."""
        return await asyncio.to_thread(resolve_repo, self._project_dir())

    async def status(self) -> StatusResponse:
        """The repository, and its status when there is one to read.

        Never an error for a repository that is not there: the state is the
        answer, and ``status`` is None beside it.
        """
        return await asyncio.to_thread(self._status)

    def _status(self) -> StatusResponse:
        root = self._project_dir()
        info = resolve_repo(root)
        if info.state != "ready" or root is None:
            return StatusResponse(repo=info)
        return StatusResponse(repo=info, status=repo.read_status(root))

    async def identity(self) -> Identity:
        """``user.name`` / ``user.email``, and where each comes from."""
        return await self._read(repo.read_identity)

    async def log(self, *, skip: int = 0,
                  limit: int = log_ops.DEFAULT_LOG_LIMIT) -> LogResponse:
        """One page of history, newest first."""
        return await self._read(
            lambda root: log_ops.log(root, skip=skip, limit=limit))

    async def commit_files(self, sha: str) -> list[GitFile]:
        """The files one commit changed, against its first parent."""
        return await self._read(lambda root: log_ops.commit_files(root, sha))

    async def diff(self, path: str, scope: str, *, sha: str | None = None,
                   blobs: bool = False) -> DiffResponse:
        """The patch for one file, in one of the three scopes."""
        return await self._read(
            lambda root: diff_ops.diff(root, path, scope, sha=sha,
                                       blobs=blobs))

    async def file_at_ref(self, path: str, ref: str) -> FileAtRef:
        """One file's content at ``HEAD``, ``index``, ``worktree`` or a sha."""
        return await self._read(
            lambda root: diff_ops.file_at_ref(root, path, ref))

    async def branches(self) -> BranchesResponse:
        """Every branch, local and remote-tracking, and what HEAD is on."""
        return await self._read(refs_ops.list_branches)

    async def remotes(self) -> list[RemoteInfo]:
        """Every configured remote, with its fetch and push URLs."""
        return await self._read(refs_ops.list_remotes)

    async def stashes(self) -> list[StashInfo]:
        """The stash stack, newest first."""
        return await self._read(stash_ops.list_stashes)

    async def _read(self, fn: Callable[[Path], _T]) -> _T:
        """Run *fn* against the ready repository, off the loop, without the lock.

        Reads outnumber writes by a wide margin here -- the tab polls the
        status and opens diffs while a commit is running -- and none of them
        take the index lock, so making them wait for the mutation lock would
        only make the UI stall for no gain.
        """
        return await asyncio.to_thread(lambda: fn(self._require_ready()))

    # --- writes (one at a time) ------------------------------------------

    async def init(self) -> MutationResult:
        """Make the project directory a repository, with the shared scaffold."""
        return await self.mutate("init", init_repo, worktree=False,
                                 require_repo=False)

    async def stage(self, paths: Sequence[str] | None = None, *,
                    all_paths: bool = False) -> MutationResult:
        """Stage *paths*, or everything when *all_paths*."""
        selection = _selection(paths, all_paths)
        return await self.mutate(
            "stage", lambda root: stage_paths(root, selection), worktree=True)

    async def unstage(self, paths: Sequence[str] | None = None, *,
                      all_paths: bool = False) -> MutationResult:
        """Take *paths*, or everything, back out of the index."""
        selection = _selection(paths, all_paths)
        return await self.mutate(
            "unstage", lambda root: unstage_paths(root, selection),
            worktree=True)

    async def discard(self, paths: Sequence[str] | None = None, *,
                      all_paths: bool = False) -> MutationResult:
        """Throw away working-tree changes. The one write that destroys."""
        selection = _selection(paths, all_paths)
        return await self.mutate(
            "discard", lambda root: discard_paths(root, selection),
            worktree=True)

    async def commit(self, message: str, *, all_paths: bool = False,
                     amend: bool = False) -> MutationResult:
        """Commit the index (or the whole tree), optionally amending."""
        return await self.mutate(
            "commit",
            lambda root: commit_changes(root, message, all_paths=all_paths,
                                        amend=amend),
            worktree=True)

    async def set_identity(self, name: str | None = None,
                           email: str | None = None) -> MutationResult:
        """Write ``user.name`` / ``user.email`` into THIS repository."""
        return await self.mutate(
            "set_identity", lambda root: repo.write_identity(root, name, email),
            worktree=False)

    # --- the refs (branches and remotes) ----------------------------------
    #
    # ``worktree`` is the one thing these have to get right, and it is not
    # decorative: it is what makes ``changed_paths`` be read, and
    # ``changed_paths`` is what tells an open editor that the file under it
    # has been replaced. A checkout replaces files; creating a branch you
    # do not go to replaces nothing; renaming one moves no file at all. A
    # write that claimed the wrong one would either cost a status read per
    # click or leave the canvas showing a graph that is no longer on disk.

    async def create_branch(self, name: str, *, checkout: bool = True,
                            start_point: str | None = None) -> MutationResult:
        """Make a branch; go to it unless ``checkout`` says otherwise."""
        return await self.mutate(
            "create_branch",
            lambda root: refs_ops.create_branch(root, name, checkout=checkout,
                                                start_point=start_point),
            worktree=checkout)

    async def checkout(self, target: str, *,
                       kind: Literal["local", "remote"] = "local"
                       ) -> MutationResult:
        """Go to a local branch, or to a new one tracking a remote's."""
        return await self.mutate(
            "checkout",
            lambda root: refs_ops.checkout(root, target, kind=kind),
            worktree=True)

    async def rename_branch(self, name: str, new_name: str) -> MutationResult:
        """Give a branch another name."""
        return await self.mutate(
            "rename_branch",
            lambda root: refs_ops.rename_branch(root, name, new_name),
            worktree=False)

    async def delete_branch(self, name: str, *,
                            force: bool = False) -> MutationResult:
        """Delete a branch; ``force`` deletes one that is not merged."""
        return await self.mutate(
            "delete_branch",
            lambda root: refs_ops.delete_branch(root, name, force=force),
            worktree=False)

    async def add_remote(self, name: str, url: str) -> MutationResult:
        """Point a name at a repository somewhere else."""
        return await self.mutate(
            "add_remote", lambda root: refs_ops.add_remote(root, name, url),
            worktree=False)

    async def set_remote_url(self, name: str, url: str) -> MutationResult:
        """Point an existing remote somewhere else."""
        return await self.mutate(
            "set_remote_url",
            lambda root: refs_ops.set_remote_url(root, name, url),
            worktree=False)

    async def remove_remote(self, name: str) -> MutationResult:
        """Forget a remote."""
        return await self.mutate(
            "remove_remote", lambda root: refs_ops.remove_remote(root, name),
            worktree=False)

    # --- the stash, and finishing a merge ---------------------------------
    #
    # Five of these six move files and say ``worktree=True``. A PUSH does,
    # as much as a pop does -- it takes the working tree back to what HEAD
    # has, which is the same replacement a checkout makes and the same
    # reason an open editor has to be told. Only ``stash_drop`` is False:
    # it deletes a ref and touches nothing on disk.
    #
    # The last two methods have the same names as the module functions they
    # call, unlike ``stage`` / ``stage_paths``: a method name is not in
    # scope inside its own body, so ``abort_merge`` and ``resolve`` below
    # resolve to the module-level functions. The plan names both, and
    # spelling them differently here would put two names on one operation.

    async def stash_push(self, message: str | None = None, *,
                         include_untracked: bool = True) -> MutationResult:
        """Set the working tree aside as a new stash."""
        return await self.mutate(
            "stash_push",
            lambda root: stash_ops.stash_push(
                root, message=message, include_untracked=include_untracked),
            worktree=True)

    async def stash_pop(self, index: int) -> MutationResult:
        """Put a stash back and remove it from the stack."""
        return await self.mutate(
            "stash_pop", lambda root: stash_ops.stash_pop(root, index),
            worktree=True)

    async def stash_apply(self, index: int) -> MutationResult:
        """Put a stash back and keep it in the stack."""
        return await self.mutate(
            "stash_apply", lambda root: stash_ops.stash_apply(root, index),
            worktree=True)

    async def stash_drop(self, index: int) -> MutationResult:
        """Throw a stash away. Nothing in the working tree moves."""
        return await self.mutate(
            "stash_drop", lambda root: stash_ops.stash_drop(root, index),
            worktree=False)

    async def abort_merge(self) -> MutationResult:
        """Undo a half-finished merge and put the tree back as it was."""
        return await self.mutate("abort_merge", abort_merge, worktree=True)

    async def resolve(self, path: str,
                      side: Literal["ours", "theirs", "mark"]
                      ) -> MutationResult:
        """Settle one conflicted file: keep ours, take theirs, or mark it."""
        return await self.mutate(
            "resolve", lambda root: resolve(root, path, side), worktree=True)

    # --- talking to a remote (the other lock) -----------------------------
    #
    # Four operations, and only two shapes underneath them. ``fetch`` and
    # ``push`` are one command under the NETWORK lock. ``pull`` and
    # ``sync`` are those steps run in order, each taking and releasing its
    # own lock, because an ``asyncio.Lock`` is not reentrant and because
    # holding the network lock across the local merge would refuse other
    # people's fetches for no reason.
    #
    # Every step of a pull says ``pull`` and every step of a sync says
    # ``sync``, on whichever lock it takes: a 409 names the operation the
    # USER started, not the command it happens to be on. That is also what
    # the tab's busy bar shows.

    async def fetch(self, remote: str | None = None) -> MutationResult:
        """Bring the remote's refs up to date, and prune what is gone."""
        return await self._fetch("fetch", remote)

    async def pull(self, strategy: network_ops.PullStrategy = "ff-only"
                   ) -> MutationResult:
        """Fetch, then merge the upstream into the branch HEAD is on.

        Two steps and two locks (R1), never ``git pull``: the slow half is
        the one that must not hold the mutation lock, and the fast half is
        the one that must. The answer is the MERGE's result -- the status
        it left, the files it replaced -- with the fetch's remote folded
        into the detail, because "which remote did this come from" is a
        fact only the first step has.
        """
        fetched = await self._step("fetch", self._fetch("pull", None))
        merged = await self._step("merge",
                                  self._merge_upstream("pull", strategy))
        return merged.model_copy(update={
            "detail": {**merged.detail,
                       "remote": fetched.detail.get("remote")}})

    async def push(self, remote: str | None = None, *,
                   set_upstream: bool = False) -> MutationResult:
        """Send this branch; ``set_upstream`` is Publish rather than Push."""
        return await self._push("publish" if set_upstream else "push",
                                remote, set_upstream=set_upstream)

    async def sync(self) -> MutationResult:
        """Pull, then push -- or publish, for a branch that has neither.

        One button for "make the two sides the same", and which steps that
        takes is decided from the status rather than asked for: a branch
        with a live upstream is fetch, merge, push; a branch with no
        upstream -- or one whose upstream has been deleted and pruned -- has
        nothing to pull FROM, so it is a publish, which is the same thing
        the header offers for both of those states (R14).

        It stops at the first failure, and every step that ran has already
        happened: a sync whose push is refused has still merged, and the
        result the caller never sees is the reason the store refreshes the
        status after a failure as well as after a success.

        ``changed_paths`` is the union over the steps, not the last one's.
        The merge is the step that replaces files and the push is the one
        that answers, so returning the push's list alone would tell an open
        editor that nothing had changed under it.
        """
        status = await self._read(repo.read_status)
        if status.upstream and not status.upstream_gone:
            fetched = await self._step("fetch", self._fetch("sync", None))
            merged = await self._step("merge",
                                      self._merge_upstream("sync", "ff-only"))
            last = await self._step(
                "push", self._push("sync", None, set_upstream=False))
            steps = ["fetch", "merge", "push"]
            results = [fetched, merged, last]
            head_moved = bool(merged.detail.get("head_moved"))
        else:
            last = await self._step(
                "publish", self._push("sync", None, set_upstream=True))
            steps = ["publish"]
            results = [last]
            head_moved = False

        return last.model_copy(update={
            "changed_paths": sorted({path for result in results
                                     for path in result.changed_paths}),
            "detail": {"steps": steps, "head_moved": head_moved,
                       "published": bool(last.detail.get("published")),
                       "remote": last.detail.get("remote"),
                       "branch": last.detail.get("branch")}})

    async def _fetch(self, op: str, remote: str | None) -> MutationResult:
        """One fetch, under the network lock, named by whoever asked for it."""
        return await self.network(
            op, lambda root: network_ops.fetch(root, remote))

    async def _merge_upstream(self, op: str,
                              strategy: network_ops.PullStrategy
                              ) -> MutationResult:
        """The local half of a pull: the MUTATION lock, and a worktree op."""
        return await self.mutate(
            op, lambda root: network_ops.merge_upstream(root, strategy),
            worktree=True)

    async def _push(self, op: str, remote: str | None, *,
                    set_upstream: bool) -> MutationResult:
        """One push or publish, under the network lock."""
        return await self.network(
            op, lambda root: network_ops.push(root, remote=remote,
                                              set_upstream=set_upstream))

    async def _step(self, step: str, call: Awaitable[_T]) -> _T:
        """Await one step of a pull or a sync, naming it if it fails.

        A failure raises, so the ``detail`` that would have said which step
        this was never reaches the caller -- and "the push was refused" and
        "the fetch never got there" are different things to do next. The
        step goes in the ``hint``, and only when git left that slot empty:
        ``classify_failure`` never sets one, so this fills a gap rather than
        overwriting the one useful sentence a refusal came with.
        """
        try:
            return await call
        except GitError as exc:
            if exc.hint is None:
                exc.hint = f"the {step} step failed"
            raise

    async def mutate(self, op: str,
                     fn: Callable[[Path], dict[str, Any] | None], *,
                     worktree: bool, require_repo: bool = True
                     ) -> MutationResult:
        """Run one write under the lock and answer with the status it left.

        *fn* does the work and returns its ``detail`` -- the operation's own
        extras, and by convention a ``paths`` key naming what it acted on,
        which is one of the three things ``changed_paths`` is built from.

        *worktree* says whether this operation can move files. A status is
        then read BEFORE it as well, so the difference between the two is
        knowable; ``init`` and an identity write skip that read because
        there is nothing to compare.

        *require_repo* is False for ``init`` alone: it is the one write that
        runs when the answer to "is this a repository" is no.

        :raises GitBusy: another mutation holds the lock. Nothing was
            attempted, and the same request usually works a moment later.
        """
        # The check and the acquisition, with no await between them: an
        # asyncio.Lock that is free is taken without suspending, so two
        # requests cannot both pass this check.
        if self.lock.locked():
            raise GitBusy(self.current_op or op)
        async with self.lock:
            self.current_op = op
            try:
                return await asyncio.to_thread(
                    self._mutate, fn, worktree=worktree,
                    require_repo=require_repo)
            finally:
                self.current_op = None

    async def network(self, op: str,
                      fn: Callable[[Path], dict[str, Any] | None]
                      ) -> MutationResult:
        """Run one NETWORK operation and answer with the status it left.

        :meth:`mutate`'s twin, on the other lock and with one flag fewer.
        Everything after the lock is the same transaction on the same
        worker thread, so a fetch answers with a fresh status exactly as a
        commit does -- which is the whole point of it being a
        ``MutationResult``: what a fetch changes is the ahead/behind the
        header draws.

        ``worktree`` is not a parameter because there is nothing to decide:
        fetch, push and publish move refs and never a file, so there is no
        BEFORE status worth reading. The one network operation that does
        move files is a pull's merge half, and that runs through
        :meth:`mutate` like every other local write.

        :raises GitBusy: another network operation holds the lock -- with
            ``op`` naming it, so the tab can say "wait for the push" rather
            than "wait". A LOCAL write is not blocked by this lock and must
            not be: the point of the second lock is that a commit made
            during somebody's fetch still works.
        """
        # Check then acquire, with no await between them -- see ``mutate``.
        if self.network_lock.locked():
            raise GitBusy(self.current_network_op or op)
        async with self.network_lock:
            self.current_network_op = op
            try:
                return await asyncio.to_thread(self._mutate, fn,
                                               worktree=False,
                                               require_repo=True)
            finally:
                self.current_network_op = None

    def _mutate(self, fn: Callable[[Path], dict[str, Any] | None], *,
                worktree: bool, require_repo: bool) -> MutationResult:
        """The whole write, on one worker thread.

        One thread for the state check, the operation and both status reads,
        because they are one transaction as far as the caller is concerned
        and splitting them over several ``to_thread`` hops would let another
        request's write land in the middle of them.

        The status read afterwards is NOT guarded: if it fails, the request
        fails with its error. ``MutationResult.status`` is required, and a
        write that cannot be read back is a failed request rather than a
        result with a hole in it.
        """
        root = self._require_ready() if require_repo else self._require_project()
        before = repo.read_status(root) if worktree else None
        detail = fn(root) or {}
        after = repo.read_status(root)
        named = detail.get("paths")
        return MutationResult(
            status=after,
            changed_paths=changed_paths(
                root, before, after,
                named if isinstance(named, list) else ()),
            head=after.head,
            detail=detail)

    # --- the state checks --------------------------------------------------

    def _require_ready(self) -> Path:
        """The project directory, or the failure its state is."""
        root = self._project_dir()
        info = resolve_repo(root)
        if info.state != "ready" or root is None:
            raise _state_error(info)
        return Path(root)

    def _require_project(self) -> Path:
        """The project directory, without requiring a repository in it.

        For ``init``, which is the operation whose whole purpose is that
        there is not one yet -- including the ``nested_toplevel`` case,
        where making the project its own repository is exactly the fix.
        """
        root = self._project_dir()
        info = resolve_repo(root)
        if info.state in _UNUSABLE_STATES or root is None:
            raise _state_error(info)
        return Path(root)
