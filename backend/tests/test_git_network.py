"""Fetch, pull, push, publish and sync -- against a real ``file://`` remote.

Nothing here fakes git and nothing here reaches a network. A bare
repository in ``tmp_path`` is a perfectly good server: git runs the same
protocol over a ``file://`` URL that it runs over ssh, so a push really is
refused for being behind, a ``--prune`` really does remove a branch
somebody deleted, and a diverged pull really does answer with the sentence
the classifier is written against. What is left out is only the part that
would make these tests flaky -- credentials and a host.

Five of these tests exist because git 2.53 does not do what a plan would
assume, all measured before they were written:

* ``git merge`` that CONFLICTS exits 1 with an empty stderr -- the
  ``CONFLICT (add/add)`` line is on stdout -- so a pull classified from
  stderr alone calls its commonest failure ``git_failed``;
* ``merge --ff-only @{u}`` on a branch with no upstream says "fatal: no
  upstream configured for branch 'main'", which shares no phrase with the
  two sentences the ``no_upstream`` row was written from;
* the same command on a branch whose upstream was DELETED and pruned says
  "merge: @{u} - not something we can merge", under an opening
  (``merge: ``) that is neither ``fatal: `` nor ``error: ``;
* ``git fetch -- <a name that is not a remote>`` reports "Could not read
  from remote repository", which classifies as ``network`` -- so a stale
  Publish would tell the user their connection is down;
* ``push -u`` on an unborn branch says "error: src refspec main does not
  match any", which is a 404 about a refspec the user never typed.

The fixtures come from ``test_git_service`` so the files cannot drift
apart, including the two this task added: ``bare_remote`` and ``clone_of``.
"""

from __future__ import annotations

import asyncio
import shutil
import threading
from pathlib import Path

import pytest

from app.core.git import network, refs, repo as repo_ops
from app.core.git.errors import GitBusy, GitError, _subcommand
from app.core.git.models import GitStatus, MutationResult
from app.core.git.runner import LITERAL_PATHSPECS, GitResult

# ``isolated_git`` and ``make_repo`` are fixtures, used by NAME rather than
# by reference -- what pytest wants and what ruff cannot see. The other four
# are plain helpers, and the three fixtures below wrap them here for the
# reason ``test_git_service`` gives beside them.
from tests.test_git_service import (  # noqa: F401
    Repo,
    isolated_git,
    make_bare_remote,
    make_clone,
    make_repo,
    remote_url,
)

pytestmark = pytest.mark.skipif(shutil.which("git") is None,
                                reason="the host has no git")


@pytest.fixture
def repo(make_repo) -> Repo:  # noqa: F811
    """A repository with the project scaffold and one commit."""
    return make_repo()


@pytest.fixture
def bare_remote(tmp_path) -> Path:
    """An empty bare repository, to stand in for a server."""
    return make_bare_remote(tmp_path)


@pytest.fixture
def clone_of(tmp_path):
    """Factory: a second working copy of a bare remote."""

    def _clone(bare: Path, name: str = "clone") -> Repo:
        return make_clone(bare, tmp_path, name)

    return _clone


def _publish(repo: Repo, bare, branch: str = "main") -> Repo:
    """Add ``origin`` and push *branch* to it -- with RAW git.

    The arrange half of a test never runs the code under test: a publish
    built with ``service.push`` would pass whenever the service and the
    test were wrong the same way.
    """
    if "origin" not in repo.git("remote"):
        repo.git("remote", "add", "origin", remote_url(bare))
    repo.git("push", "-q", "-u", "origin", branch)
    return repo


def _blocking_run_git(started: threading.Event, release: threading.Event):
    """A ``run_git`` that hangs until it is let go -- a slow network, made local.

    Only ``network.run_git`` is replaced with it, so the status reads and
    the remote list (which live in other modules) still run for real and
    the operation blocks exactly where a fetch blocks: on the wire.
    """

    def _run(args, **kwargs):
        started.set()
        assert release.wait(timeout=10), "the blocked git was never released"
        return GitResult(argv=["git", *args], returncode=0, stdout=b"",
                         stderr=b"")

    return _run


# --- what each operation tells the locks -------------------------------------


async def test_every_network_op_names_itself_and_takes_the_right_lock(
        repo, monkeypatch):
    """Which lock, and under which name -- two facts nothing else pins.

    The name is what a 409 ``busy`` carries in ``detail.op`` and what the
    busy bar prints, and every step of a pull says ``pull`` while every
    step of a sync says ``sync``: a refusal names the operation the USER
    started, not the command it is on at that moment. The lock is the whole
    point of the second one existing -- a fetch on the mutation lock would
    refuse commits for as long as it ran.
    """
    network_ops: list[str] = []
    mutations: list[tuple[str, bool]] = []

    async def _network(op, fn):
        network_ops.append(op)
        return MutationResult(status=GitStatus())

    async def _mutate(op, fn, *, worktree, require_repo=True):
        mutations.append((op, worktree))
        return MutationResult(status=GitStatus())

    service = repo.service
    monkeypatch.setattr(service, "network", _network)
    monkeypatch.setattr(service, "mutate", _mutate)

    await service.fetch()
    await service.pull()
    await service.push()
    await service.push(set_upstream=True)
    await service.sync()

    assert network_ops == ["fetch", "pull", "push", "publish", "sync"]
    # Only the merge half of a pull is a mutation, and it moves files.
    assert mutations == [("pull", True)]


async def test_a_sync_with_an_upstream_runs_all_three_steps(repo, bare_remote,
                                                            monkeypatch):
    """...and every one of them under the name the user pressed."""
    _publish(repo, bare_remote)
    network_ops: list[str] = []
    mutations: list[str] = []

    async def _network(op, fn):
        network_ops.append(op)
        return MutationResult(status=GitStatus())

    async def _mutate(op, fn, *, worktree, require_repo=True):
        mutations.append(op)
        return MutationResult(status=GitStatus())

    service = repo.service
    monkeypatch.setattr(service, "network", _network)
    monkeypatch.setattr(service, "mutate", _mutate)

    await service.sync()

    assert network_ops == ["sync", "sync"]
    assert mutations == ["sync"]


async def test_a_second_network_op_while_one_runs_is_refused(
        repo, bare_remote, monkeypatch):
    """One connection at a time: two pushes to one branch is a race nobody
    wins, and the refusal names the operation that is running."""
    _publish(repo, bare_remote)
    started, release = threading.Event(), threading.Event()
    monkeypatch.setattr(network, "run_git", _blocking_run_git(started, release))
    service = repo.service
    fetching = asyncio.create_task(service.fetch())

    try:
        assert await asyncio.to_thread(started.wait, 5)
        with pytest.raises(GitBusy) as excinfo:
            await service.push()
        assert excinfo.value.code == "busy"
        assert excinfo.value.op == "fetch"
    finally:
        release.set()
        await fetching


async def test_a_commit_during_a_fetch_still_works(repo, bare_remote,
                                                   monkeypatch):
    """The reason there are two locks at all.

    A fetch can run for a minute against a slow remote. If it held the
    mutation lock, every commit, stage and discard in the panel would be
    refused for that minute -- a tab that goes dead while the thing it is
    doing is working perfectly.
    """
    _publish(repo, bare_remote)
    started, release = threading.Event(), threading.Event()
    monkeypatch.setattr(network, "run_git", _blocking_run_git(started, release))
    service = repo.service
    fetching = asyncio.create_task(service.fetch())

    try:
        assert await asyncio.to_thread(started.wait, 5)
        repo.write("during.txt", "written while a fetch was running\n")
        repo.git("add", "-A", "--", ".")
        result = await asyncio.wait_for(service.commit("during a fetch"),
                                        timeout=15)
        assert result.detail["sha"]
    finally:
        release.set()
        await fetching


async def test_a_fetch_during_a_local_write_still_works(repo, bare_remote):
    """And the other way round: the two queues do not see each other."""
    _publish(repo, bare_remote)
    service = repo.service

    async with service.lock:
        service.current_op = "commit"
        result = await asyncio.wait_for(service.fetch(), timeout=15)
    service.current_op = None

    assert result.detail["remote"] == "origin"


async def test_a_pull_whose_merge_meets_a_local_write_is_refused(
        repo, bare_remote):
    """The merge half is an ordinary mutation and queues like one.

    The fetch has already happened by then, which is not a problem: a fetch
    changes no file and the next pull will merge what it brought.
    """
    _publish(repo, bare_remote)
    service = repo.service

    async with service.lock:
        service.current_op = "commit"
        with pytest.raises(GitBusy) as excinfo:
            await service.pull()
    service.current_op = None

    assert excinfo.value.op == "commit"


# --- which remote (R6) -------------------------------------------------------


async def test_the_upstream_decides_which_remote_a_fetch_uses(
        repo, bare_remote, tmp_path):
    """A branch that is tracking something is not a question."""
    _publish(repo, bare_remote)
    second = tmp_path / "second.git"
    second.mkdir()
    Repo(second).git("init", "--bare", "-q")
    repo.git("remote", "add", "backup", remote_url(second))

    result = await repo.service.fetch()

    assert result.detail == {"remote": "origin"}


async def test_the_only_remote_is_used_when_there_is_no_upstream(repo,
                                                                 bare_remote):
    """One remote is not a choice either."""
    repo.git("remote", "add", "elsewhere", remote_url(bare_remote))

    result = await repo.service.fetch()

    assert result.detail == {"remote": "elsewhere"}


async def test_a_repository_with_no_remote_is_refused(repo):
    """409, not a 500 and not a network error: there is nothing configured
    to talk to, which is a state the tab draws its own screen for."""
    with pytest.raises(GitError) as excinfo:
        await repo.service.fetch()

    assert excinfo.value.code == "no_remote"
    assert excinfo.value.status == 409
    assert excinfo.value.hint


async def test_several_remotes_and_no_upstream_needs_a_publish(repo,
                                                               bare_remote,
                                                               tmp_path):
    """The one case where the server cannot decide, and says which click can.

    Guessing would send somebody's work to a mirror they only ever read
    from. Publishing is the click that makes the choice AND records it, so
    every request after it has an answer.
    """
    second = tmp_path / "second.git"
    second.mkdir()
    Repo(second).git("init", "--bare", "-q")
    repo.git("remote", "add", "origin", remote_url(bare_remote))
    repo.git("remote", "add", "backup", remote_url(second))

    with pytest.raises(GitError) as excinfo:
        await repo.service.fetch()

    assert excinfo.value.code == "invalid_value"
    assert excinfo.value.status == 400
    assert "publish" in (excinfo.value.hint or "")


async def test_a_remote_that_is_not_configured_is_404(repo, bare_remote):
    """git would call this a network failure; it is a stale panel.

    ``git fetch -- nope`` says "'nope' does not appear to be a git
    repository" and then "Could not read from remote repository", which
    classifies as ``network`` -- so the user would be told to check their
    connection because somebody removed a remote fifteen seconds ago.
    """
    _publish(repo, bare_remote)

    with pytest.raises(GitError) as excinfo:
        await repo.service.fetch("nope")

    assert excinfo.value.code == "not_found"
    assert excinfo.value.status == 404
    assert "reload" in (excinfo.value.hint or "")


@pytest.mark.parametrize("name", ["-oProxyCommand=curl", "a b", "he/re"])
async def test_a_remote_name_that_is_not_one_never_reaches_git(repo, name):
    """The validator, not ``--``, is what keeps an option off the argv."""
    with pytest.raises(GitError) as excinfo:
        await repo.service.fetch(name)

    assert excinfo.value.code == "invalid_value"
    assert excinfo.value.status == 400


# --- fetch --------------------------------------------------------------------


async def test_fetch_updates_how_far_behind_the_branch_is(repo, bare_remote,
                                                          clone_of):
    """What the header's ahead/behind is drawn from, and why a fetch is a
    ``MutationResult``: nothing on disk moved and the panel still changed."""
    _publish(repo, bare_remote)
    clone = clone_of(bare_remote)
    clone.commit("from the clone", {"b.txt": "bee\n"})
    clone.git("push", "-q")

    result = await repo.service.fetch()

    assert result.status.behind == 1
    assert result.status.ahead == 0
    assert result.changed_paths == []


async def test_fetch_prunes_a_branch_the_remote_no_longer_has(
        repo, bare_remote, clone_of):
    """``--prune`` is why the branch list can be trusted.

    Without it the remote-tracking ref outlives the branch it mirrors, so a
    branch somebody deleted last week goes on being offered as somewhere to
    switch to and goes on being counted against.
    """
    _publish(repo, bare_remote)
    repo.git("switch", "-q", "-c", "side")
    repo.commit("side work", {"s.txt": "side\n"})
    _publish(repo, bare_remote, "side")
    clone = clone_of(bare_remote)
    clone.git("push", "-q", "origin", "--delete", "side")

    await repo.service.fetch()

    side = next(branch for branch in refs.list_branches(repo.root).local
                if branch.name == "side")
    assert side.gone is True
    assert [row.name for row in refs.list_branches(repo.root).remote] == ["main"]


async def test_fetch_says_which_remote_it_talked_to(repo, bare_remote):
    """``detail`` is exactly one key, and Task 4 is typed against it."""
    _publish(repo, bare_remote)

    result = await repo.service.fetch()

    assert set(result.detail) == {"remote"}


# --- push and publish ---------------------------------------------------------


async def test_publish_sets_the_upstream_and_sends_the_branch(repo,
                                                              bare_remote):
    """``-u`` is the whole difference: it records where this branch goes,
    so every push after it is the plain one."""
    repo.git("remote", "add", "origin", remote_url(bare_remote))

    result = await repo.service.push(set_upstream=True)

    assert result.detail == {"remote": "origin", "branch": "main",
                             "published": True}
    assert result.status.upstream == "origin/main"
    assert Repo(bare_remote).git("rev-parse", "refs/heads/main").strip() \
        == repo.head()


async def test_publish_can_be_told_which_remote(repo, bare_remote, tmp_path):
    """Several remotes is the case Publish exists to settle."""
    second = tmp_path / "second.git"
    second.mkdir()
    Repo(second).git("init", "--bare", "-q")
    repo.git("remote", "add", "origin", remote_url(bare_remote))
    repo.git("remote", "add", "backup", remote_url(second))

    result = await repo.service.push("backup", set_upstream=True)

    assert result.detail["remote"] == "backup"
    assert result.status.upstream == "backup/main"
    assert Repo(second).git("rev-parse", "refs/heads/main").strip() \
        == repo.head()


async def test_push_after_publishing_sends_the_new_commit(repo, bare_remote):
    _publish(repo, bare_remote)
    repo.commit("more work", {"a.txt": "one\ntwo\nthree\n"})

    result = await repo.service.push()

    assert result.detail == {"remote": "origin", "branch": "main",
                             "published": False}
    assert result.status.ahead == 0
    assert Repo(bare_remote).git("rev-parse", "refs/heads/main").strip() \
        == repo.head()


async def test_pushing_a_branch_with_no_upstream_says_so(repo, bare_remote):
    """git's own answer, and the one the tab turns into a Publish button."""
    repo.git("remote", "add", "origin", remote_url(bare_remote))

    with pytest.raises(GitError) as excinfo:
        await repo.service.push()

    assert excinfo.value.code == "no_upstream"
    assert excinfo.value.status == 409


async def test_a_push_the_remote_is_ahead_of_is_refused(repo, bare_remote,
                                                        clone_of):
    """Somebody else pushed first: ``non_fast_forward``, and nothing here
    force-pushes to fix it."""
    _publish(repo, bare_remote)
    clone = clone_of(bare_remote)
    clone.commit("from the clone", {"b.txt": "bee\n"})
    clone.git("push", "-q")
    repo.commit("mine", {"c.txt": "sea\n"})

    with pytest.raises(GitError) as excinfo:
        await repo.service.push()

    assert excinfo.value.code == "non_fast_forward"
    assert excinfo.value.status == 409


async def test_publishing_from_a_detached_head_is_refused(repo, bare_remote):
    """There is no branch to record an upstream for -- and a user whose
    ``push.default`` is ``matching`` would otherwise have every branch that
    exists on both sides sent, from one click that said nothing of the
    sort."""
    repo.git("remote", "add", "origin", remote_url(bare_remote))
    repo.git("checkout", "-q", "--detach", "HEAD")

    with pytest.raises(GitError) as excinfo:
        await repo.service.push(set_upstream=True)

    assert excinfo.value.code == "detached_head"
    assert excinfo.value.status == 409


async def test_publishing_before_the_first_commit_is_refused(make_repo,  # noqa: F811
                                                             bare_remote):
    """git answers this with "error: src refspec main does not match any" --
    a 404 about a refspec the user never typed. There is nothing to send
    yet, and that is a sentence the tab already has."""
    fresh = make_repo("unborn", first_commit=False)
    fresh.git("remote", "add", "origin", remote_url(bare_remote))

    with pytest.raises(GitError) as excinfo:
        await fresh.service.push(set_upstream=True)

    assert excinfo.value.code == "nothing_to_commit"
    assert excinfo.value.status == 409
    assert "no commits" in (excinfo.value.hint or "")


async def test_a_plain_push_that_names_a_remote_is_refused(repo, bare_remote):
    """Obeying it would send the commits somewhere the branch is not
    tracking, and the ahead/behind in the panel would go on counting
    against the old one -- the sort of "it worked" nobody can debug."""
    _publish(repo, bare_remote)

    with pytest.raises(GitError) as excinfo:
        await repo.service.push("origin")

    assert excinfo.value.code == "invalid_value"
    assert excinfo.value.status == 400
    assert "set_upstream" in (excinfo.value.hint or "")


# --- pull ---------------------------------------------------------------------


async def test_pull_fast_forwards_and_says_head_moved(repo, bare_remote,
                                                      clone_of):
    """The ordinary pull: two steps, and the files the merge replaced come
    back in ``changed_paths`` for an open editor to reload from."""
    _publish(repo, bare_remote)
    clone = clone_of(bare_remote)
    clone.commit("from the clone", {"b.txt": "bee\n"})
    clone.git("push", "-q")

    result = await repo.service.pull()

    assert result.detail == {"step": "merge", "strategy": "ff-only",
                             "head_moved": True, "remote": "origin"}
    assert result.changed_paths == ["b.txt"]
    assert repo.read("b.txt") == "bee\n"
    assert result.status.behind == 0


async def test_a_pull_with_nothing_to_do_says_head_did_not_move(repo,
                                                                bare_remote):
    """"Already up to date." is a sentence in a language the person
    reading may not have; ``head_moved`` is the same fact as a boolean."""
    _publish(repo, bare_remote)

    result = await repo.service.pull()

    assert result.detail["head_moved"] is False
    assert result.changed_paths == []


async def test_a_diverged_branch_will_not_fast_forward(repo, bare_remote,
                                                       clone_of):
    """``ff-only`` refuses rather than writing a merge commit nobody asked
    for. The tab turns this code into the button that asks."""
    _publish(repo, bare_remote)
    clone = clone_of(bare_remote)
    clone.commit("from the clone", {"b.txt": "bee\n"})
    clone.git("push", "-q")
    repo.commit("mine", {"c.txt": "sea\n"})

    with pytest.raises(GitError) as excinfo:
        await repo.service.pull()

    assert excinfo.value.code == "diverged"
    assert excinfo.value.status == 409
    assert repo_ops.read_status(repo.root).merge_in_progress is False


async def test_merging_the_remote_changes_after_that_refusal(repo, bare_remote,
                                                             clone_of):
    """The second click: the same route with ``strategy: "merge"``."""
    _publish(repo, bare_remote)
    clone = clone_of(bare_remote)
    clone.commit("from the clone", {"b.txt": "bee\n"})
    clone.git("push", "-q")
    repo.commit("mine", {"c.txt": "sea\n"})

    result = await repo.service.pull("merge")

    assert result.detail["strategy"] == "merge"
    assert result.detail["head_moved"] is True
    assert "b.txt" in result.changed_paths
    assert result.status.ahead == 2  # mine, and the merge


async def test_a_pull_whose_merge_conflicts_leaves_the_merge_in_progress(
        repo, bare_remote, clone_of):
    """The failure the whole merge group is drawn for.

    git reports it on STDOUT with an EMPTY stderr, so the classification
    has to join both streams -- from stderr alone this is ``git_failed``, a
    500 for the most ordinary thing that can happen to a pull.
    """
    _publish(repo, bare_remote)
    clone = clone_of(bare_remote)
    clone.commit("from the clone", {"shared.txt": "theirs\n"})
    clone.git("push", "-q")
    repo.commit("mine", {"shared.txt": "ours\n"})

    with pytest.raises(GitError) as excinfo:
        await repo.service.pull("merge")

    assert excinfo.value.code == "conflict"
    assert excinfo.value.status == 409
    assert (repo.root / ".git" / "MERGE_HEAD").exists()
    status = repo_ops.read_status(repo.root)
    assert status.merge_in_progress is True
    assert [entry.path for entry in status.conflicted] == ["shared.txt"]


async def test_pulling_a_branch_with_no_upstream_says_so(repo, bare_remote):
    """``fatal: no upstream configured for branch 'main'`` -- git's own
    voice, in words the ``no_upstream`` row did not know before G3."""
    repo.git("remote", "add", "origin", remote_url(bare_remote))

    with pytest.raises(GitError) as excinfo:
        await repo.service.pull()

    assert excinfo.value.code == "no_upstream"
    assert excinfo.value.status == 409


async def test_pulling_a_branch_whose_upstream_is_gone_says_so(
        repo, bare_remote, clone_of):
    """The upstream is configured and the ref is not there any more.

    git says "merge: @{u} - not something we can merge" (exit 1), under an
    opening that is neither ``fatal: `` nor ``error: ``. It is
    ``no_upstream`` and not ``not_found`` because the answer is the same
    one the header already offers for this state: publish it again.
    """
    _publish(repo, bare_remote)
    repo.git("switch", "-q", "-c", "side")
    repo.commit("side work", {"s.txt": "side\n"})
    _publish(repo, bare_remote, "side")
    clone = clone_of(bare_remote)
    clone.git("push", "-q", "origin", "--delete", "side")

    with pytest.raises(GitError) as excinfo:
        await repo.service.pull()

    assert excinfo.value.code == "no_upstream"
    assert excinfo.value.status == 409
    # The fetch half ran: that is what pruned the ref out from under it.
    assert repo_ops.read_status(repo.root).upstream_gone is True


# --- sync ---------------------------------------------------------------------


async def test_sync_pulls_and_then_pushes(repo, bare_remote):
    """One click for "make the two sides the same"."""
    _publish(repo, bare_remote)
    repo.commit("mine", {"c.txt": "sea\n"})

    result = await repo.service.sync()

    assert result.detail == {"steps": ["fetch", "merge", "push"],
                             "head_moved": False, "published": False,
                             "remote": "origin", "branch": "main"}
    assert result.status.ahead == 0
    assert Repo(bare_remote).git("rev-parse", "refs/heads/main").strip() \
        == repo.head()


async def test_sync_carries_the_files_the_merge_replaced(repo, bare_remote,
                                                         clone_of):
    """``changed_paths`` is the union over the steps.

    The merge is the step that replaces files and the push is the step that
    answers, so returning the push's list alone would tell an open editor
    that nothing had changed under it.
    """
    _publish(repo, bare_remote)
    clone = clone_of(bare_remote)
    clone.commit("from the clone", {"b.txt": "bee\n"})
    clone.git("push", "-q")

    result = await repo.service.sync()

    assert result.detail["steps"] == ["fetch", "merge", "push"]
    assert result.detail["head_moved"] is True
    assert result.changed_paths == ["b.txt"]


async def test_sync_publishes_a_branch_that_has_never_been_pushed(repo,
                                                                  bare_remote):
    """There is nothing to pull FROM, so the whole sync is the publish --
    which is also the only button the header offers in this state."""
    repo.git("remote", "add", "origin", remote_url(bare_remote))

    result = await repo.service.sync()

    assert result.detail == {"steps": ["publish"], "head_moved": False,
                             "published": True, "remote": "origin",
                             "branch": "main"}
    assert result.status.upstream == "origin/main"


async def test_sync_publishes_a_branch_whose_upstream_is_gone(
        repo, bare_remote, clone_of):
    """Same rule, same reason: a branch whose upstream was deleted has
    nothing to pull either, and re-publishing re-creates it."""
    _publish(repo, bare_remote)
    repo.git("switch", "-q", "-c", "side")
    repo.commit("side work", {"s.txt": "side\n"})
    _publish(repo, bare_remote, "side")
    clone = clone_of(bare_remote)
    clone.git("push", "-q", "origin", "--delete", "side")
    # Raw git, so the state under test is arrived at without the code under
    # test: the prune is what makes the upstream GONE rather than behind.
    repo.git("fetch", "--prune", "-q", "origin")

    result = await repo.service.sync()

    assert result.detail["steps"] == ["publish"]
    assert result.detail["published"] is True
    assert Repo(bare_remote).git("rev-parse", "refs/heads/side").strip() \
        == repo.head()


async def test_sync_stops_at_the_first_failure_and_says_which_step(
        repo, bare_remote, clone_of):
    """The push must not run once the merge has refused.

    A sync that pushed anyway would send a branch that does not have the
    remote's work in it, which is the ``non_fast_forward`` the merge just
    told us about -- or, on a remote that allows it, a history somebody
    else's commits fell out of.
    """
    _publish(repo, bare_remote)
    clone = clone_of(bare_remote)
    clone.commit("from the clone", {"b.txt": "bee\n"})
    clone.git("push", "-q")
    remote_head = Repo(bare_remote).git("rev-parse", "refs/heads/main").strip()
    repo.commit("mine", {"c.txt": "sea\n"})

    with pytest.raises(GitError) as excinfo:
        await repo.service.sync()

    assert excinfo.value.code == "diverged"
    assert excinfo.value.hint == "the merge step failed"
    assert Repo(bare_remote).git("rev-parse",
                                 "refs/heads/main").strip() == remote_head


async def test_a_step_that_failed_with_advice_keeps_it(repo):
    """The step name fills an EMPTY hint and never replaces git's own.

    ``classify_failure`` leaves the slot empty, which is what makes this
    safe; a pre-flight refusal like this one arrives with the sentence that
    says what to do, and losing it to "the fetch step failed" would be a
    worse answer, not a better-labelled one.
    """
    with pytest.raises(GitError) as excinfo:
        await repo.service.sync()

    assert excinfo.value.code == "no_remote"
    assert excinfo.value.hint == "add a remote before pushing or fetching"


# --- the argv ------------------------------------------------------------------


async def test_a_pull_that_deletes_a_file_removes_it(repo, bare_remote,
                                                     clone_of):
    """R20, as a behaviour rather than an argument.

    ``stash push`` with ``--literal-pathspecs`` stored an untracked file
    and LEFT IT in the working tree -- exit 0, no warning, half the job
    done. A merge is the other command here that removes files, so the
    shape worth pinning is the same one: what the remote deleted has to be
    gone from disk, the directory it emptied with it, and both have to
    reach ``changed_paths`` for an editor that has the file open.

    (Measured directly as well, with the option and without: fetch,
    ``--prune``, merge and push behave identically. Neither of them runs
    ``git clean`` under the covers, which is what made stash's case
    different.)
    """
    _publish(repo, bare_remote)
    repo.commit("a file and a folder", {"doomed.txt": "bye\n",
                                        "sub/nested.txt": "bye\n"})
    repo.git("push", "-q")
    clone = clone_of(bare_remote)
    (clone.root / "doomed.txt").unlink()
    (clone.root / "sub" / "nested.txt").unlink()
    clone.commit("delete them")
    clone.git("push", "-q")

    result = await repo.service.pull()

    assert not (repo.root / "doomed.txt").exists()
    assert not (repo.root / "sub").exists()
    assert result.changed_paths == ["doomed.txt", "sub/nested.txt"]


async def test_every_network_command_keeps_literal_pathspecs(
        repo, bare_remote, clone_of, monkeypatch):
    """R20: the exemption list is ``stash push`` and nothing here.

    None of these subcommands takes a pathspec and none of them runs ``git
    clean`` under the covers -- which is what made ``stash push`` silently
    do half its job with the option on. Pinned at the argv rather than
    argued in a docstring, because the next command added here will be
    added by somebody reading this file.
    """
    _publish(repo, bare_remote)
    clone = clone_of(bare_remote)
    clone.commit("from the clone", {"b.txt": "bee\n"})
    clone.git("push", "-q")
    real = network.run_git
    seen: list[list[str]] = []

    def _record(args, **kwargs):
        result = real(args, **kwargs)
        seen.append(result.argv)
        return result

    monkeypatch.setattr(network, "run_git", _record)

    await repo.service.pull()
    repo.commit("mine", {"c.txt": "sea\n"})
    await repo.service.push()

    assert [argv for argv in seen if LITERAL_PATHSPECS not in argv] == []
    # ...and that the list above really is every command this module runs.
    assert {_subcommand(argv) for argv in seen} == {"fetch", "rev-parse",
                                                    "merge", "push"}
