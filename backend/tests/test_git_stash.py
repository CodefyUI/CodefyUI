"""The stash and the end of a merge, against real repositories in ``tmp_path``.

Nothing here fakes git, and four of these tests exist because git 2.53 does
not do what a plan would assume:

* ``git stash push`` with nothing to save exits **0** and says so on
  stdout, so "it worked" and "there was nothing to do" are the same return
  code;
* ``git stash pop`` that CONFLICTS exits 1 with an **empty stderr** -- the
  ``CONFLICT (content)`` line is on stdout -- so a classification made from
  stderr alone calls it ``git_failed``;
* ``git checkout --ours -- <file>`` on a file that is merely modified exits
  0 and silently restores it from HEAD, which is why a resolve of a
  non-conflicted path has to be refused before it runs;
* a stash's reflog subject ends with the base commit's SUBJECT, which is
  arbitrary text and really can contain the field separator.

A fake would have agreed with all four wrong ideas.

The fixtures come from ``test_git_service`` so that the two files cannot
drift apart -- including ``_conflicted``, which builds a real mid-merge
conflict and is what every resolve and abort test starts from.
"""

from __future__ import annotations

import shutil

import pytest

from app.core.git import stash
from app.core.git.errors import GitError
from app.core.git.models import GitStatus, MutationResult
from app.core.git.runner import LITERAL_PATHSPECS

# Fixtures, used by NAME rather than by reference -- what pytest wants and
# what ruff cannot see. ``_conflicted`` is a helper, not a fixture: it takes
# a repository and leaves it mid-merge. The one-commit ``repo`` fixture is
# rebuilt below rather than imported, for the reason ``test_git_refs`` gives.
from tests.test_git_service import (  # noqa: F401
    Repo,
    _conflicted,
    isolated_git,
    make_repo,
)

pytestmark = pytest.mark.skipif(shutil.which("git") is None,
                                reason="the host has no git")


@pytest.fixture
def repo(make_repo) -> Repo:  # noqa: F811
    """A repository with the project scaffold and one commit."""
    return make_repo()


# --- what each write tells the lock ------------------------------------------


async def test_every_write_here_names_itself_and_says_if_it_moves_files(
        repo, monkeypatch):
    """Two facts nothing else pins, and both are read by somebody else.

    The ``op`` string is what a 409 ``busy`` carries in ``detail.op``, so a
    typo in it reaches the tab as a busy bar with no sentence in it. The
    ``worktree`` flag is what makes ``changed_paths`` be computed, which is
    what an open editor reloads from -- and a PUSH moves files as much as a
    pop does, taking the working tree back to what HEAD has. Only ``drop``
    is False: it deletes a ref and touches nothing on disk.
    """
    seen: list[tuple[str, bool]] = []

    async def record(op, fn, *, worktree, require_repo=True):
        seen.append((op, worktree))
        return MutationResult(status=GitStatus())

    service = repo.service
    monkeypatch.setattr(service, "mutate", record)

    await service.stash_push("m")
    await service.stash_pop(0)
    await service.stash_apply(0)
    await service.stash_drop(0)
    await service.abort_merge()
    await service.resolve("a.txt", "ours")

    assert seen == [("stash_push", True), ("stash_pop", True),
                    ("stash_apply", True), ("stash_drop", False),
                    ("abort_merge", True), ("resolve", True)]


# --- reading the stack -------------------------------------------------------


async def test_a_named_stash_keeps_the_message_it_was_given(repo):
    repo.write("a.txt", "two\n")

    await repo.service.stash_push("saving this for later")

    entry = stash.list_stashes(repo.root)[0]
    assert entry.index == 0
    assert entry.message == "saving this for later"
    assert entry.branch == "main"
    assert entry.created_at > 0


async def test_an_unnamed_stash_shows_the_whole_wip_line(repo):
    """The base commit is the only description an unnamed stash has, and
    "9f2c1ab first" on its own would read as a commit the user made."""
    repo.write("a.txt", "two\n")

    await repo.service.stash_push()

    entry = stash.list_stashes(repo.root)[0]
    assert entry.message.startswith("WIP on main: ")
    assert entry.message.endswith(" first")
    assert entry.branch == "main"


async def test_the_newest_stash_is_first(repo):
    repo.write("a.txt", "two\n")
    await repo.service.stash_push("older")
    repo.write("a.txt", "three\n")
    await repo.service.stash_push("newer")

    stashes = stash.list_stashes(repo.root)

    assert [entry.index for entry in stashes] == [0, 1]
    assert [entry.message for entry in stashes] == ["newer", "older"]


async def test_a_repository_with_no_stashes_lists_none(repo):
    assert stash.list_stashes(repo.root) == []


async def test_a_stash_made_off_a_branch_has_no_branch(repo):
    """git writes ``(no branch)`` where the name would be, which is a
    placeholder and not a name -- the same reading ``BranchesResponse``
    gives a detached HEAD."""
    repo.git("checkout", "-q", "--detach", "HEAD")
    repo.write("a.txt", "two\n")

    await repo.service.stash_push("off a branch")

    entry = stash.list_stashes(repo.root)[0]
    assert entry.branch is None
    assert entry.message == "off a branch"


async def test_a_branch_with_slashes_in_it_survives_the_subject(repo):
    repo.git("switch", "-q", "-c", "feat/deep/name")
    repo.write("a.txt", "two\n")

    await repo.service.stash_push("on a deep branch")

    assert stash.list_stashes(repo.root)[0].branch == "feat/deep/name"


async def test_a_message_that_looks_like_gits_own_is_still_the_users(repo):
    """``--message="WIP on main: deadbee spoof"`` arrives as ``On main: WIP
    on main: deadbee spoof`` (measured), so the shape test has to be a
    prefix test and not a search."""
    repo.write("a.txt", "two\n")

    await repo.service.stash_push("WIP on main: deadbee spoof")

    entry = stash.list_stashes(repo.root)[0]
    assert entry.branch == "main"
    assert entry.message == "WIP on main: deadbee spoof"


async def test_a_message_holding_colons_keeps_all_of_them(repo):
    repo.write("a.txt", "two\n")

    await repo.service.stash_push("fix: the loader: part two")

    entry = stash.list_stashes(repo.root)[0]
    assert entry.branch == "main"
    assert entry.message == "fix: the loader: part two"


async def test_a_separator_in_the_base_subject_stays_in_the_message(make_repo):  # noqa: F811
    """``%gs`` is in the MIDDLE of the record and ends with the base
    commit's subject, which is arbitrary text -- so the record is read from
    both ends. A left-to-right split would put "ject here" in ``created_at``.
    """
    repo = make_repo(first_commit=False)
    repo.write("a.txt", "one\n")
    repo.git("add", "-A")
    repo.git("commit", "-q", "-m", "sub\x1fject here")
    repo.write("a.txt", "two\n")

    await repo.service.stash_push()

    entry = stash.list_stashes(repo.root)[0]
    assert entry.message.endswith("sub\x1fject here")
    assert entry.created_at > 0
    assert entry.index == 0


@pytest.mark.parametrize("record", [
    pytest.param("", id="empty"),
    pytest.param("stash@{0}", id="one-field"),
    pytest.param("stash@{0}\x1fOn main: x", id="two-fields"),
    pytest.param("refs/stash\x1fOn main: x\x1f1700000000", id="not-a-selector"),
    pytest.param("stash@{x}\x1fOn main: x\x1f1700000000", id="not-a-number"),
])
def test_a_row_that_cannot_be_addressed_is_dropped(record):
    """Dropped rather than raised on, and dropping it is the SAFE half: an
    index this parser guessed at is an index a drop would act on."""
    assert stash._entry(record) is None


@pytest.mark.parametrize("subject, branch, message", [
    pytest.param("something else entirely", None, "something else entirely",
                 id="another-tools-reflog"),
    pytest.param("On main: ", "main", "On main: ", id="an-empty-message"),
    pytest.param("On main", None, "On main", id="no-colon-at-all"),
    pytest.param("", None, "stash@{0}", id="no-subject-at-all"),
])
def test_a_subject_that_is_not_gits_two_shapes_keeps_all_of_itself(
        subject, branch, message):
    """A stash another tool wrote, or one with nothing written on it. Still
    a row worth showing, still a stash worth dropping -- and never an empty
    line the user cannot tell from the row below."""
    entry = stash._entry(f"stash@{{0}}\x1f{subject}\x1f1700000000")

    assert entry is not None
    assert entry.branch == branch
    assert entry.message == message


def test_a_timestamp_that_is_not_one_is_zero():
    entry = stash._entry("stash@{0}\x1fOn main: x\x1flater")

    assert entry is not None
    assert entry.created_at == 0


# --- pushing -----------------------------------------------------------------


async def test_push_takes_the_working_tree_back_to_head(repo):
    repo.write("a.txt", "two\n")

    result = await repo.service.stash_push("mine")

    assert repo.read("a.txt") == "one\ntwo\n"
    assert result.detail == {"stash": 0, "message": "mine",
                             "include_untracked": True}
    assert result.status.stash_count == 1
    assert result.changed_paths == ["a.txt"]


async def test_push_takes_untracked_files_with_it(repo):
    """The whole round trip, because the failure this pins is SILENT.

    Under ``--literal-pathspecs`` -- which every other command in this
    package runs with -- git stores the untracked file and leaves it in the
    working tree, exit 0 and no warning (measured on git 2.53). The stash
    that makes is broken in a way nothing notices until the pop, which then
    refuses with "new.txt already exists, no checkout".
    """
    repo.write("new.txt", "new\n")

    result = await repo.service.stash_push("with the new file")

    assert not (repo.root / "new.txt").exists()
    assert "new.txt" in result.changed_paths

    await repo.service.stash_pop(0)

    assert repo.read("new.txt") == "new\n"


async def test_push_can_be_asked_to_leave_untracked_files_alone(repo):
    repo.write("a.txt", "two\n")
    repo.write("new.txt", "new\n")

    result = await repo.service.stash_push("tracked only",
                                           include_untracked=False)

    assert (repo.root / "new.txt").exists()
    assert repo.read("a.txt") == "one\ntwo\n"
    assert result.detail["include_untracked"] is False


async def test_push_without_a_message_still_stashes(repo):
    repo.write("a.txt", "two\n")

    result = await repo.service.stash_push()

    assert result.detail["message"] is None
    assert result.status.stash_count == 1


async def test_a_staged_change_goes_into_the_stash_too(repo):
    repo.write("a.txt", "two\n")
    repo.git("add", "-A")

    result = await repo.service.stash_push("staged")

    assert repo.read("a.txt") == "one\ntwo\n"
    assert result.status.staged == []


async def test_a_message_that_is_too_long_is_refused_before_git_runs(repo):
    repo.write("a.txt", "two\n")

    with pytest.raises(GitError) as excinfo:
        await repo.service.stash_push("x" * 501)

    assert excinfo.value.code == "invalid_value"
    assert excinfo.value.status == 400
    assert stash.list_stashes(repo.root) == []


async def test_a_message_that_is_an_option_is_still_a_message(repo):
    """``--message=<m>`` is attached, so a leading ``-`` never becomes an
    option of its own."""
    repo.write("a.txt", "two\n")

    await repo.service.stash_push("-upload-pack=whoami")

    assert stash.list_stashes(repo.root)[0].message == "-upload-pack=whoami"


async def test_a_message_that_is_gits_nothing_to_stash_sentence_still_stashes(
        repo):
    """The stdout twin of the ``WIP on ...`` spoof.

    git echoes the message into stdout (``Saved working directory and index
    state On main: <message>``), so a SUBSTRING test for "No local changes
    to save" reports 400 "there is nothing to stash" for a push that set
    the work aside and reset the tree -- measured against this module
    before the test became ``startswith``. git prints that sentence as the
    whole of stdout when it means it; its warnings go to stderr.
    """
    repo.write("a.txt", "two\n")

    result = await repo.service.stash_push("No local changes to save")

    assert result.detail["stash"] == 0
    assert repo.read("a.txt") == "one\ntwo\n"
    entry = stash.list_stashes(repo.root)[0]
    assert entry.message == "No local changes to save"


async def test_stashing_a_clean_tree_is_refused_rather_than_silent(repo):
    """git exits 0 and says so on stdout, so without this the tab reports a
    stash that does not exist. The menu item is disabled when the tree is
    clean, which makes this the race guard it looks like."""
    with pytest.raises(GitError) as excinfo:
        await repo.service.stash_push("nothing here")

    assert excinfo.value.code == "invalid_value"
    assert excinfo.value.status == 400
    assert excinfo.value.hint == "nothing to stash"


async def test_an_untracked_file_alone_is_nothing_to_stash_without_u(repo):
    """git's own rule, not this server's: without ``--include-untracked``
    an untracked file is not a local change."""
    repo.write("new.txt", "new\n")

    with pytest.raises(GitError) as excinfo:
        await repo.service.stash_push("tracked only", include_untracked=False)

    assert excinfo.value.code == "invalid_value"


async def test_stashing_before_the_first_commit_is_404_not_500(make_repo):  # noqa: F811
    """The state every repository starts in. git says "You do not have the
    initial commit yet", which matches no classification rule."""
    repo = make_repo(first_commit=False)
    repo.write("a.txt", "one\n")

    with pytest.raises(GitError) as excinfo:
        await repo.service.stash_push("too early")

    assert excinfo.value.code == "not_found"
    assert excinfo.value.status == 404
    assert excinfo.value.hint == "this branch has no commits yet"


async def test_stashing_across_an_unfinished_merge_is_refused(repo):
    """git exits 1 with "could not write index" here, which matches no
    rule -- and the merge must survive being asked."""
    _conflicted(repo)

    with pytest.raises(GitError) as excinfo:
        await repo.service.stash_push("mid merge")

    assert excinfo.value.code == "merge_in_progress"
    assert excinfo.value.status == 409
    assert (repo.root / ".git" / "MERGE_HEAD").exists()
    assert stash.list_stashes(repo.root) == []


async def test_stashing_across_a_resolved_merge_is_refused_too(repo):
    """The dangerous half, and the one this guard exists for: with every
    conflict staged git exits **0** and DELETES ``MERGE_HEAD`` (measured on
    git 2.53). The merge is gone, popping the stash back does not bring it
    back, and the panel says "stashed"."""
    _conflicted(repo)
    await repo.service.resolve("a.txt", "theirs")

    with pytest.raises(GitError) as excinfo:
        await repo.service.stash_push("mid merge")

    assert excinfo.value.code == "merge_in_progress"
    assert excinfo.value.status == 409
    assert (repo.root / ".git" / "MERGE_HEAD").exists()
    assert repo.read("a.txt") == "side\n"
    assert stash.list_stashes(repo.root) == []


async def test_stashing_with_an_unmerged_index_and_no_merge_is_refused(repo):
    """The sibling state, which ``merge_in_progress`` does not reach: a
    conflicting ``stash pop`` leaves unmerged paths and no ``MERGE_HEAD``,
    and the tab draws a merge group with Stash Changes in the same menu."""
    repo.write("a.txt", "stashed\n")
    await repo.service.stash_push("mine")
    repo.commit("on top", {"a.txt": "committed\n"})
    with pytest.raises(GitError):
        await repo.service.stash_pop(0)
    assert not (repo.root / ".git" / "MERGE_HEAD").exists()

    with pytest.raises(GitError) as excinfo:
        await repo.service.stash_push("after the conflict")

    assert excinfo.value.code == "conflict"
    assert excinfo.value.status == 409
    assert len(stash.list_stashes(repo.root)) == 1


# --- popping, applying, dropping ---------------------------------------------


async def test_pop_puts_the_work_back_and_empties_the_stack(repo):
    repo.write("a.txt", "two\n")
    await repo.service.stash_push("mine")

    result = await repo.service.stash_pop(0)

    assert repo.read("a.txt") == "two\n"
    assert result.detail == {"stash": 0}
    assert result.status.stash_count == 0
    assert result.changed_paths == ["a.txt"]


async def test_apply_puts_the_work_back_and_keeps_the_stash(repo):
    repo.write("a.txt", "two\n")
    await repo.service.stash_push("mine")

    result = await repo.service.stash_apply(0)

    assert repo.read("a.txt") == "two\n"
    assert result.status.stash_count == 1
    assert stash.list_stashes(repo.root)[0].message == "mine"


async def test_drop_removes_the_one_that_was_named(repo):
    """The index is git's, so dropping ``1`` of two leaves ``0``'s work."""
    repo.write("a.txt", "two\n")
    await repo.service.stash_push("older")
    repo.write("a.txt", "three\n")
    await repo.service.stash_push("newer")

    result = await repo.service.stash_drop(1)

    assert result.detail == {"stash": 1}
    assert [entry.message for entry in stash.list_stashes(repo.root)] == ["newer"]
    assert repo.read("a.txt") == "one\ntwo\n"


async def test_drop_moves_no_file(repo):
    repo.write("a.txt", "two\n")
    await repo.service.stash_push("mine")

    result = await repo.service.stash_drop(0)

    assert result.changed_paths == []
    assert repo.read("a.txt") == "one\ntwo\n"


@pytest.mark.parametrize("call", ["stash_pop", "stash_apply", "stash_drop"])
async def test_an_index_that_is_not_in_the_list_is_404(repo, call):
    repo.write("a.txt", "two\n")
    await repo.service.stash_push("the only one")

    with pytest.raises(GitError) as excinfo:
        await getattr(repo.service, call)(1)

    assert excinfo.value.code == "not_found"
    assert excinfo.value.status == 404
    # Nothing was attempted: the stash is still there.
    assert len(stash.list_stashes(repo.root)) == 1


async def test_an_index_on_an_empty_stack_is_404(repo):
    with pytest.raises(GitError) as excinfo:
        await repo.service.stash_pop(0)

    assert excinfo.value.code == "not_found"


async def test_a_pop_that_conflicts_is_409_and_keeps_the_stash(repo):
    """git's stderr is EMPTY here -- the ``CONFLICT (content)`` line is on
    stdout -- so this test is what stops the commonest failure of this
    operation being reported as a 500."""
    repo.write("a.txt", "stashed\n")
    await repo.service.stash_push("mine")
    repo.commit("on top", {"a.txt": "committed\n"})

    with pytest.raises(GitError) as excinfo:
        await repo.service.stash_pop(0)

    assert excinfo.value.code == "conflict"
    assert excinfo.value.status == 409
    assert len(stash.list_stashes(repo.root)) == 1
    assert "<<<<<<<" in repo.read("a.txt")


async def test_an_apply_that_conflicts_is_409_too(repo):
    repo.write("a.txt", "stashed\n")
    await repo.service.stash_push("mine")
    repo.commit("on top", {"a.txt": "committed\n"})

    with pytest.raises(GitError) as excinfo:
        await repo.service.stash_apply(0)

    assert excinfo.value.code == "conflict"
    assert len(stash.list_stashes(repo.root)) == 1


async def test_a_pop_blocked_by_a_local_edit_says_dirty_tree(repo):
    """A different failure with a different answer: git refuses BEFORE
    merging, and this one really is on stderr."""
    repo.write("a.txt", "stashed\n")
    await repo.service.stash_push("mine")
    repo.write("a.txt", "local\n")

    with pytest.raises(GitError) as excinfo:
        await repo.service.stash_pop(0)

    assert excinfo.value.code == "dirty_tree"
    assert repo.read("a.txt") == "local\n"
    assert len(stash.list_stashes(repo.root)) == 1


async def test_a_pop_blocked_by_a_recreated_file_says_dirty_tree(repo):
    """A project directory that regenerates a file is an ordinary flow, and
    the answer used to be "there is nothing to commit".

    ``stash pop`` prints a whole ``git status`` on stdout whose last line
    is one of the ``nothing_to_commit`` phrases, so joining the streams --
    which is what makes a real conflict classifiable at all -- put this
    failure on the wrong row until git's own "already exists, no checkout"
    was added to ``dirty_tree``.
    """
    repo.write("n.txt", "generated\n")
    await repo.service.stash_push("with untracked")
    repo.write("n.txt", "generated again\n")

    with pytest.raises(GitError) as excinfo:
        await repo.service.stash_pop(0)

    assert excinfo.value.code == "dirty_tree"
    assert excinfo.value.status == 409
    assert len(stash.list_stashes(repo.root)) == 1
    assert repo.read("n.txt") == "generated again\n"


async def test_only_the_push_runs_without_literal_pathspecs(repo, monkeypatch):
    """The exemption pinned at the ARGV, not through its symptom.

    ``test_push_takes_untracked_files_with_it`` does fail today when the
    exemption is reverted, but it is a behavioural pin: on a git that fixed
    the bug it would pass either way and the exemption could be deleted
    without a test noticing. Every other stash command keeps the option --
    ``pop`` and ``apply`` take a ``stash@{N}``, not a pathspec, and are
    measurably fine with it.
    """
    seen: list[list[str]] = []
    real = stash.run_git

    def record(args, **kwargs):
        result = real(args, **kwargs)
        seen.append(list(result.argv))
        return result

    monkeypatch.setattr(stash, "run_git", record)
    repo.write("a.txt", "two\n")
    await repo.service.stash_push("mine")
    await repo.service.stash_pop(0)
    await repo.service.stash_push("again")
    await repo.service.stash_apply(0)
    await repo.service.stash_drop(0)

    def argvs_for(*words: str) -> list[list[str]]:
        matched = [argv for argv in seen
                   if all(word in argv for word in words)]
        assert matched, (words, seen)
        return matched

    for argv in argvs_for("stash", "push"):
        assert LITERAL_PATHSPECS not in argv, argv
    for command in ("list", "pop", "apply", "drop"):
        for argv in argvs_for("stash", command):
            assert LITERAL_PATHSPECS in argv, argv


# --- resolving a merge -------------------------------------------------------


async def test_keeping_ours_takes_the_version_we_were_on(repo):
    """``ours`` IS what HEAD has, so the file ends up matching HEAD and
    nothing is staged -- the merge is still in progress, waiting for its
    commit."""
    _conflicted(repo)

    result = await repo.service.resolve("a.txt", "ours")

    assert repo.read("a.txt") == "main\n"
    assert result.detail == {"path": "a.txt", "side": "ours"}
    assert result.status.conflicted == []
    assert result.status.merge_in_progress is True
    assert "a.txt" in result.changed_paths


async def test_taking_theirs_stages_the_version_that_came_in(repo):
    _conflicted(repo)

    result = await repo.service.resolve("a.txt", "theirs")

    assert repo.read("a.txt") == "side\n"
    assert result.status.conflicted == []
    assert [entry.path for entry in result.status.staged] == ["a.txt"]


async def test_marking_resolved_stages_whatever_is_on_disk(repo):
    """What a person who edited the file by hand means by "resolved": git
    reads the index, not the file."""
    _conflicted(repo)
    repo.write("a.txt", "half of each\n")

    result = await repo.service.resolve("a.txt", "mark")

    assert repo.read("a.txt") == "half of each\n"
    assert result.status.conflicted == []
    assert [entry.path for entry in result.status.staged] == ["a.txt"]


async def test_a_resolved_merge_commits_as_a_merge(repo):
    """The whole point of leaving MERGE_HEAD alone: the commit box makes a
    merge commit because git can see one is pending."""
    _conflicted(repo)
    await repo.service.resolve("a.txt", "theirs")

    result = await repo.service.commit("merged")

    assert result.status.merge_in_progress is False
    parents = repo.git("rev-list", "--parents", "-n", "1", "HEAD").split()
    assert len(parents) == 3, "a merge commit has two parents"


async def test_resolving_a_tracked_file_that_is_not_conflicted_is_refused(repo):
    """The shape the refusal actually exists for: a TRACKED file with an
    uncommitted edit.

    On an untracked path git refuses anyway ("did not match any file(s)
    known to git"), so a test written that way passes for the wrong
    reason. Here ``git checkout --ours -- b.txt`` exits 0 and silently
    restores b.txt from HEAD (measured on git 2.53) -- the edit is gone,
    with no error and nothing to undo from -- which makes this refusal the
    only thing between a stale panel and a lost edit.
    """
    repo.commit("add b", {"b.txt": "committed\n"})
    _conflicted(repo)
    repo.write("b.txt", "edited by the user\n")

    with pytest.raises(GitError) as excinfo:
        await repo.service.resolve("b.txt", "ours")

    assert excinfo.value.code == "path_not_in_status"
    assert excinfo.value.status == 400
    assert repo.read("b.txt") == "edited by the user\n"


async def test_resolving_a_file_git_has_never_heard_of_is_refused(repo):
    """The other half, which git would also refuse -- but with a 404 and a
    sentence about pathspecs rather than one about this panel."""
    _conflicted(repo)
    repo.write("b.txt", "an untracked file\n")

    with pytest.raises(GitError) as excinfo:
        await repo.service.resolve("b.txt", "theirs")

    assert excinfo.value.code == "path_not_in_status"
    assert repo.read("b.txt") == "an untracked file\n"


async def test_resolving_a_file_with_no_merge_at_all_is_refused(repo):
    repo.write("a.txt", "two\n")

    with pytest.raises(GitError) as excinfo:
        await repo.service.resolve("a.txt", "mark")

    assert excinfo.value.code == "path_not_in_status"
    assert repo.read("a.txt") == "two\n"


async def test_a_path_that_leaves_the_project_never_reaches_git(repo):
    _conflicted(repo)

    with pytest.raises(GitError) as excinfo:
        await repo.service.resolve("../outside.txt", "ours")

    assert excinfo.value.code == "invalid_path"
    assert excinfo.value.status == 400


# --- aborting a merge --------------------------------------------------------


async def test_abort_puts_the_tree_back_as_it_was(repo):
    _conflicted(repo)

    result = await repo.service.abort_merge()

    assert repo.read("a.txt") == "main\n"
    assert result.status.conflicted == []
    assert result.status.merge_in_progress is False
    assert result.detail == {}
    assert "a.txt" in result.changed_paths


async def test_abort_after_a_half_finished_resolution_still_works(repo):
    _conflicted(repo)
    await repo.service.resolve("a.txt", "theirs")

    result = await repo.service.abort_merge()

    assert repo.read("a.txt") == "main\n"
    assert result.status.merge_in_progress is False


async def test_aborting_with_no_merge_is_404(repo):
    """git says "There is no merge to abort (MERGE_HEAD missing)", which
    matches no classification rule and was a 500."""
    with pytest.raises(GitError) as excinfo:
        await repo.service.abort_merge()

    assert excinfo.value.code == "not_found"
    assert excinfo.value.status == 404
    assert excinfo.value.hint == "no merge in progress"


async def test_a_stash_pop_conflict_is_not_a_merge_to_abort(repo):
    """There is no MERGE_HEAD after one, so "Abort Merge" would be the
    wrong button -- and git would refuse it too."""
    repo.write("a.txt", "stashed\n")
    await repo.service.stash_push("mine")
    repo.commit("on top", {"a.txt": "committed\n"})
    with pytest.raises(GitError):
        await repo.service.stash_pop(0)

    with pytest.raises(GitError) as excinfo:
        await repo.service.abort_merge()

    assert excinfo.value.code == "not_found"
