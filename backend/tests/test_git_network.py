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

from app.core.git import network, refs, repo as repo_ops, service as service_ops
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


@pytest.mark.parametrize("names", [("origin", "backup"), ("alpha", "beta")],
                         ids=["one-is-origin", "none-is-origin"])
async def test_a_plain_push_with_several_remotes_and_no_upstream_is_gits_own(
        repo, bare_remote, tmp_path, names):
    """``no_upstream`` -- git's answer, not a choice made for it.

    A plain push names no remote, so several of them is not a question the
    server has to settle before git runs. Settling it the way a fetch does
    answered this state ``invalid_value``, a code R10 has no button for,
    where git's own answer is the one the Publish button hangs off. git
    says it in two voices, both measured: "The current branch main has no
    upstream branch" when one of the remotes is ``origin``, and "No
    configured push destination" when none is -- it has no default to fall
    back on -- which the ``no_upstream`` row learned for this test.
    """
    second = tmp_path / "second.git"
    second.mkdir()
    Repo(second).git("init", "--bare", "-q")
    repo.git("remote", "add", names[0], remote_url(bare_remote))
    repo.git("remote", "add", names[1], remote_url(second))

    with pytest.raises(GitError) as excinfo:
        await repo.service.push()

    assert excinfo.value.code == "no_upstream"
    assert excinfo.value.status == 409


async def test_a_plain_push_with_no_remote_is_refused_before_push_runs(
        repo, monkeypatch):
    """The one state a plain push still refuses on its own.

    git's sentence for it, "No configured push destination", is the same
    one it prints for several remotes with no ``origin`` among them, so
    the classifier cannot tell "add a remote" from "publish". The answer is
    the one every other operation gives for no remote at all -- and the
    one the header hides its button for. Metadata reads may run, but the
    mutating push must not.
    """
    real_run_git = network.run_git

    def _no_push(args, **kwargs):
        if args and args[0] == "push":
            pytest.fail("push ran")
        return real_run_git(args, **kwargs)

    monkeypatch.setattr(network, "run_git", _no_push)

    with pytest.raises(GitError) as excinfo:
        await repo.service.push()

    assert excinfo.value.code == "no_remote"
    assert excinfo.value.status == 409
    assert excinfo.value.hint


async def test_plain_push_uses_a_direct_remote_push_default(repo, bare_remote):
    """A URL can be Git's push destination without being a named remote."""
    target = remote_url(bare_remote)
    repo.git("config", "remote.pushDefault", target)
    repo.git("config", "push.default", "current")
    assert repo.git("remote").strip() == ""

    result = await repo.service.push()

    assert result.detail == {"remote": target, "branch": "main",
                             "published": False}
    assert Repo(bare_remote).git("rev-parse", "refs/heads/main").strip() \
        == repo.head()


def _fake_push(monkeypatch, stdout: bytes, *, returncode: int = 0,
               stderr: bytes = b""):
    """Answer the plain push with *stdout* and leave every other git real.

    Only the mutating command is faked. The metadata reads around it --
    the status, ``for-each-ref``, ``remote -v`` -- still run against the
    repository on disk, which is what makes an assertion about
    ``detail.remote`` an assertion about the real resolver.
    """
    real_run_git = network.run_git

    def _push(args, **kwargs):
        if args == ["push", "--porcelain"]:
            return GitResult(argv=["git", "push", "--porcelain"],
                             returncode=returncode, stdout=stdout,
                             stderr=stderr)
        return real_run_git(args, **kwargs)

    monkeypatch.setattr(network, "run_git", _push)


@pytest.mark.parametrize("with_a_to_line", [True, False],
                         ids=["porcelain-names-the-destination",
                              "porcelain-names-nothing"])
async def test_plain_push_never_reports_credentials(repo, monkeypatch,
                                                    with_a_to_line):
    """A direct push destination is display-safe before it reaches detail.

    Both shapes, because there are two masks on this path and the stdout
    decides which one answers. With a ``To`` line the destination comes
    from the push's own porcelain output (``_pushed_remote``); without one
    -- an up-to-date push under some gits, or any future output change --
    it falls back to the configured target ``_tracked_remote`` resolved.
    A test that arranged a credential and then exercised only one of them
    would read as a guard on both.
    """
    token = "ghp_16C7e42F292c6912E7710c838347Ae178B4a"
    target = f"https://alice:{token}@example.invalid/repo.git"
    repo.git("config", "remote.pushDefault", target)
    repo.git("config", "push.default", "current")
    stdout = f"To {target}\nDone\n".encode() if with_a_to_line else b"Done\n"
    _fake_push(monkeypatch, stdout)

    result = await repo.service.push()

    assert token not in result.detail["remote"]
    assert result.detail["remote"] == (
        "https://alice:***@example.invalid/repo.git")


async def test_a_push_destination_that_is_not_a_url_is_not_reported(
        repo, monkeypatch):
    """``detail.remote`` is a closed contract, not an echo of git config.

    ``%(push:remotename)`` repeats ``remote.pushDefault`` verbatim, so
    whatever string is in the user's configuration -- any length, any
    shape -- would otherwise become a field of a JSON response. A value
    that is neither one of this repository's remotes nor a URL
    ``validate_remote_url`` accepts is reported as nothing at all. The
    push itself is git's business and still runs.
    """
    repo.git("config", "remote.pushDefault", "a target nobody configured")
    repo.git("config", "push.default", "current")
    pushed: list[list[str]] = []
    real_run_git = network.run_git

    def _push(args, **kwargs):
        if args and args[0] == "push":
            pushed.append(list(args))
            return GitResult(argv=["git", *args], returncode=0,
                             stdout=b"Done\n", stderr=b"")
        return real_run_git(args, **kwargs)

    monkeypatch.setattr(network, "run_git", _push)

    result = await repo.service.push()

    assert result.detail == {"remote": None, "branch": "main",
                             "published": False}
    assert pushed == [["push", "--porcelain"]]


async def test_branch_push_remote_overrides_the_upstream(repo, bare_remote,
                                                         tmp_path):
    """Git prefers ``branch.<name>.pushRemote`` to the upstream remote."""
    _publish(repo, bare_remote)
    backup = make_bare_remote(tmp_path, "backup.git")
    repo.git("remote", "add", "backup", remote_url(backup))
    origin_head = Repo(bare_remote).git("rev-parse", "refs/heads/main").strip()
    repo.commit("for backup", {"backup.txt": "backup\n"})
    repo.git("config", "branch.main.pushRemote", "backup")

    result = await repo.service.push()

    assert result.detail["remote"] == "backup"
    assert Repo(backup).git("rev-parse", "refs/heads/main").strip() == repo.head()
    assert Repo(bare_remote).git("rev-parse", "refs/heads/main").strip() \
        == origin_head


async def test_remote_push_default_overrides_the_upstream(repo, bare_remote,
                                                          tmp_path):
    """Git prefers ``remote.pushDefault`` to the upstream remote."""
    _publish(repo, bare_remote)
    backup = make_bare_remote(tmp_path, "backup.git")
    repo.git("remote", "add", "backup", remote_url(backup))
    origin_head = Repo(bare_remote).git("rev-parse", "refs/heads/main").strip()
    repo.commit("for backup", {"backup.txt": "backup\n"})
    repo.git("config", "remote.pushDefault", "backup")

    result = await repo.service.push()

    assert result.detail["remote"] == "backup"
    assert Repo(backup).git("rev-parse", "refs/heads/main").strip() == repo.head()
    assert Repo(bare_remote).git("rev-parse", "refs/heads/main").strip() \
        == origin_head


async def test_a_plain_push_reports_the_effective_remote(repo, bare_remote):
    """``detail.remote`` is Git's effective destination, not a guess.

    With ``push.default=current`` Git pushes a branch that has no upstream
    to the only remote. The push is real, and the answer names the remote
    Git selected even though no upstream exists yet.
    """
    repo.git("remote", "add", "elsewhere", remote_url(bare_remote))
    repo.git("config", "push.default", "current")

    result = await repo.service.push()

    assert result.detail == {"remote": "elsewhere", "branch": "main",
                             "published": False}
    assert Repo(bare_remote).git("rev-parse", "refs/heads/main").strip() \
        == repo.head()


async def test_a_sole_remote_answers_a_porcelain_that_named_nothing(
        repo, bare_remote, monkeypatch):
    """The last resort when the push output says where it went to nobody.

    Ref-filter reports no destination for one remote plus
    ``push.default=current``, so the name has to come from the push itself
    -- and git 2.53 always writes a ``To`` line, which the URL match then
    resolves. This is the same state with that line absent: a git that
    stops printing it, or an output shape a future version changes. With
    one remote there is still only one answer it can be, and returning
    None there would blank the remote out of a toast for no reason.
    """
    repo.git("remote", "add", "elsewhere", remote_url(bare_remote))
    repo.git("config", "push.default", "current")
    _fake_push(monkeypatch, b"Done\n")

    result = await repo.service.push()

    assert result.detail == {"remote": "elsewhere", "branch": "main",
                             "published": False}


async def test_a_plain_push_names_the_remote_git_fell_back_to(repo,
                                                              bare_remote,
                                                              tmp_path):
    """Several remotes and no upstream: the answer is still a NAME.

    Git's implicit fallback to ``origin`` is one ref-filter does not
    report, so ``%(push:remotename)`` is empty here and the effective
    destination has to be recovered from the push's own ``To`` line. With
    one remote the sole-remote branch would answer it; with two, only the
    URL match does -- and the difference between a name and a raw
    ``file://`` path in ``detail.remote`` is what the toast and the panel
    read.
    """
    backup = make_bare_remote(tmp_path, "backup.git")
    repo.git("remote", "add", "origin", remote_url(bare_remote))
    repo.git("remote", "add", "backup", remote_url(backup))
    repo.git("config", "push.default", "current")

    result = await repo.service.push()

    assert result.detail == {"remote": "origin", "branch": "main",
                             "published": False}
    assert Repo(bare_remote).git("rev-parse", "refs/heads/main").strip() \
        == repo.head()
    assert Repo(backup).git("for-each-ref", "refs/heads").strip() == "", \
        "the other remote was pushed to as well"


async def test_a_sole_remote_is_not_the_answer_when_the_push_went_elsewhere(
        repo, bare_remote, tmp_path):
    """The sole-remote guess is a last resort, under the destination match.

    One remote, and a ``remote.pushDefault`` that is a bare PATH: git takes
    it -- a push destination may be a URL or a path and not only a name --
    and the commit lands in that repository rather than in ``origin``.
    Ref-filter repeats the path, which is not a target ``detail.remote``
    will echo, so the configured target reaches ``_pushed_remote`` as None:
    the SAME None a ref-filter that reported nothing at all produces.
    Answering that None with "the only remote there is" would name a
    repository this push never opened, in a field whose whole job is to say
    where the work went. The ``To`` line settles it first -- it matches no
    configured remote's push URL, so the answer is the display form, and for
    a path that is nothing.
    """
    second = make_bare_remote(tmp_path, "second.git")
    repo.git("remote", "add", "origin", remote_url(bare_remote))
    repo.git("config", "remote.pushDefault", second.as_posix())
    repo.git("config", "push.default", "current")

    result = await repo.service.push()

    assert result.detail == {"remote": None, "branch": "main",
                             "published": False}
    assert Repo(second).git("rev-parse", "refs/heads/main").strip() \
        == repo.head()
    assert Repo(bare_remote).git("for-each-ref", "refs/heads").strip() == "", \
        "origin was pushed to as well"


async def test_push_default_nothing_is_a_user_error(repo, bare_remote):
    """A host config that disables plain push is actionable, not a 500.

    ``push_config`` and NOT ``invalid_value``: the tab rewrites an
    ``invalid_value`` from a push into ``no_upstream``, which draws "this
    branch is not published yet" over a branch that plainly is, and offers
    a Publish button whose ``push -u`` would repoint the user's upstream.
    """
    _publish(repo, bare_remote)
    repo.commit("not pushed", {"local.txt": "local\n"})
    repo.git("config", "push.default", "nothing")

    with pytest.raises(GitError) as excinfo:
        await repo.service.push()

    assert excinfo.value.code == "push_config"
    assert excinfo.value.status == 409
    assert excinfo.value.hint


async def test_simple_push_with_a_differently_named_upstream_is_a_user_error(
        repo, bare_remote):
    """Simple mode's name guard is host configuration, not a server fault.

    The other half of ``push_config``, and the one with the sharper edge:
    this branch HAS an upstream, it is just called something else, so a
    Publish button offered here would create a second remote branch and
    rewrite ``branch.main.merge`` to point at it.
    """
    _publish(repo, bare_remote)
    repo.git("push", "-q", "-u", "origin", "main:other")
    repo.commit("not pushed", {"local.txt": "local\n"})
    repo.git("config", "push.default", "simple")

    with pytest.raises(GitError) as excinfo:
        await repo.service.push()

    assert excinfo.value.code == "push_config"
    assert excinfo.value.status == 409
    assert excinfo.value.hint


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


async def test_a_refused_porcelain_push_keeps_its_verdict_in_the_evidence(
        repo, bare_remote, monkeypatch):
    """The line the code came from has to survive ``stderr_tail``.

    ``--porcelain`` moves the per-ref verdict onto STDOUT while git's own
    ``error:`` and ``hint:`` prose stays on stderr, and a forge is free to
    print a banner of ``remote:`` lines above both. The tail keeps the last
    twenty lines of whatever it is handed, so joining stdout first drops the
    one line the classification was derived from -- the tab's Details
    disclosure then shows twenty lines of somebody's banner and no reason.
    Classification itself scans the whole text and is order-independent, so
    stderr goes first and the verdict sits at the end.
    """
    _publish(repo, bare_remote)
    repo.commit("mine", {"c.txt": "sea\n"})
    banner = "".join(f"remote: house rule {n}\n" for n in range(25))
    _fake_push(
        monkeypatch,
        b"To file:///srv/mirrors/repo.git\n"
        b"!\trefs/heads/main:refs/heads/main\t[rejected] (fetch first)\n"
        b"Done\n",
        returncode=1,
        stderr=(banner + "error: failed to push some refs\n").encode())

    with pytest.raises(GitError) as excinfo:
        await repo.service.push()

    assert excinfo.value.code == "non_fast_forward"
    assert "[rejected] (fetch first)" in (excinfo.value.stderr or "")


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


async def test_sync_stops_at_the_first_failure_without_an_english_hint(
        repo, bare_remote, clone_of):
    """The push must not run once the merge has refused.

    A sync that pushed anyway would send a branch that does not have the
    remote's work in it, which is the ``non_fast_forward`` the merge just
    told us about -- or, on a remote that allows it, a history somebody
    else's commits fell out of.

    A CLASSIFIED refusal carries no step name. The tab has a translated
    sentence for ``diverged`` and draws the hint beside it, so "the merge
    step failed" arrived as raw English under a Chinese sentence -- and it
    said nothing the code did not already say. Which step it was is still
    in ``stderr``, which the Details disclosure shows.
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
    assert excinfo.value.hint is None
    assert Repo(bare_remote).git("rev-parse",
                                 "refs/heads/main").strip() == remote_head


async def test_an_unclassified_step_failure_still_says_which_step(
        repo, monkeypatch):
    """``git_failed`` is the one code the step name is still worth having.

    Nothing is known about that failure -- no row of the table matched --
    so the tab shows git's own words, and "the merge step failed" is then
    the only thing saying WHERE in a three-step sync it happened.
    """
    async def _boom(op, fn):
        raise GitError("git_failed", 500, "git push failed (exit 1)")

    # ``repo.service`` builds a NEW service on every read, so the one under
    # test has to be held before it is patched.
    service = repo.service
    monkeypatch.setattr(service, "network", _boom)

    with pytest.raises(GitError) as excinfo:
        await service.sync()

    assert excinfo.value.code == "git_failed"
    assert excinfo.value.hint == "the publish step failed"


async def test_a_step_that_failed_with_advice_keeps_it(repo):
    """The step name fills an EMPTY hint and never replaces git's own.

    ``classify_failure`` leaves the slot empty for every code but the few
    in ``errors.CODE_HINTS``, which is what makes this safe; a pre-flight
    refusal like this one arrives with the sentence that says what to do,
    and losing it to "the fetch step failed" would be a worse answer, not a
    better-labelled one. The same holds for a ``push_config`` inside a
    sync: "check push.default" beats "the push step failed".
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

    WHAT IS RECORDED is the ``run_git`` of the FOUR modules a pull or a
    push reaches, and nothing else:

    * ``network`` -- ``fetch``, ``merge``, ``push``, the ``for-each-ref``
      push resolver and the two ``rev-parse`` HEAD reads;
    * ``refs`` -- ``remote -v``;
    * ``repo`` -- the ``status`` the service reads around every mutation,
      and the ``rev-parse --git-path`` that goes with it;
    * ``service`` -- ``rev-parse --show-toplevel`` for the project root,
      and the ``diff --name-only -z`` that turns two HEADs into
      ``changed_paths``.

    Git run any other way is invisible here -- the raw ``subprocess`` the
    fixtures arrange with, above all. Recording one module and calling the
    result an inventory is how this test spent three rounds not seeing
    ``remote -v``, then the status read, then the two the service itself
    makes.
    """
    _publish(repo, bare_remote)
    clone = clone_of(bare_remote)
    clone.commit("from the clone", {"b.txt": "bee\n"})
    clone.git("push", "-q")
    seen: list[list[str]] = []

    def _recorder(real):
        def _record(args, **kwargs):
            result = real(args, **kwargs)
            seen.append(result.argv)
            return result
        return _record

    monkeypatch.setattr(network, "run_git", _recorder(network.run_git))
    monkeypatch.setattr(refs, "run_git", _recorder(refs.run_git))
    monkeypatch.setattr(repo_ops, "run_git", _recorder(repo_ops.run_git))
    monkeypatch.setattr(service_ops, "run_git", _recorder(service_ops.run_git))

    await repo.service.pull()
    repo.commit("mine", {"c.txt": "sea\n"})
    await repo.service.push()

    assert [argv for argv in seen if LITERAL_PATHSPECS not in argv] == []
    # ...and that the set below really is every subcommand those four run.
    assert {_subcommand(argv) for argv in seen} == {
        "diff", "fetch", "for-each-ref", "rev-parse", "merge", "push",
        "remote", "status"}
    # The pull contributes none of these: its remote comes from the
    # upstream. So this count is the plain push's, and it is one.
    assert len([argv for argv in seen if _subcommand(argv) == "remote"]) == 1
