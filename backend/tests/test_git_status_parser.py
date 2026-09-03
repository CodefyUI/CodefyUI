"""Reading git's own account of the working tree.

Everything the Source Control tab draws comes from one status read, so a
misread here is not a stack trace -- it is a file the user cannot see, a
rename shown as a delete plus an add, or an "up to date" badge on a branch
whose upstream was deleted last week. The parser is pure for exactly this
reason: every one of those cases is a byte string, and a byte string can be
written down.

The fixtures are REAL output, captured from ``git status --porcelain=v2
--branch --show-stash --untracked-files=all -z`` on git 2.53 against
repositories built for each case -- including the ones that are awkward to
produce on purpose (a copy record, delete-vs-modify, a submodule, an
upstream deleted from under a tracking branch). The helpers below rebuild
records of the same shape for the variations, and
:func:`test_a_real_status_reads_the_way_git_printed_it` holds one capture
verbatim so the helpers themselves are checked against git rather than
against my memory of it.

Non-ASCII paths are written as escapes so this file stays ASCII (a cp950
console is a machine we support); ``-z`` plus ``core.quotepath=false`` means
git hands over the bytes of a filename unaltered, so the CJK and the
leading-space cases here are literally what the parser will see.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.git.models import (
    CommitRequest,
    GitStatus,
    IdentityRequest,
    MutationResult,
    PathsRequest,
)
from app.core.git.status import kind_from_letter, parse_porcelain_v2

#: Object names, spelled out because a shortened one would not survive the
#: field-counting this parser does.
_OLD = "052d31effb03dcc923165ae3f6640bb6409b33cb"
_NEW = "eeacbdb6b832e024b53d4148bf37fbb054e2f44d"
_THIRD = "d709994d5213d86eb6a40c3124defc8349686627"
_NONE = "0" * 40


def payload(*tokens: str) -> bytes:
    """The tokens as git writes them: NUL-TERMINATED, not NUL-separated.

    The trailing NUL is git's, and it leaves an empty final token that the
    parser has to ignore -- so every fixture here carries it.
    """
    return "".join(f"{token}\0" for token in tokens).encode("utf-8")


def ordinary(xy: str, path: str, sub: str = "N...") -> str:
    """A ``1`` record: a tracked file that was changed but not moved."""
    return f"1 {xy} {sub} 100644 100644 100644 {_OLD} {_NEW} {path}"


def moved(xy: str, score: str, path: str) -> str:
    """A ``2`` record. The path it came FROM is the next token, not a field."""
    return f"2 {xy} N... 100644 100644 100644 {_OLD} {_NEW} {score} {path}"


def unmerged(xy: str, path: str) -> str:
    """A ``u`` record: three stages of one conflicted file."""
    return (f"u {xy} N... 100644 100644 100644 100644 "
            f"{_OLD} {_NEW} {_THIRD} {path}")


#: The headers of a repository on ``main`` with one commit and no remote.
_ON_MAIN = (f"# branch.oid {_OLD}", "# branch.head main")


# --- the branch headers ----------------------------------------------------


def test_an_unborn_branch_has_a_name_but_no_commit():
    """``git init`` then nothing: the tab still shows "main", and "unborn"
    is what tells the service to unstage with ``rm --cached``."""
    status = parse_porcelain_v2(payload("# branch.oid (initial)",
                                        "# branch.head main"))

    assert status.unborn is True
    assert status.head is None
    assert status.branch == "main"
    assert status.detached is False


def test_a_detached_head_has_a_commit_but_no_branch():
    status = parse_porcelain_v2(payload(f"# branch.oid {_OLD}",
                                        "# branch.head (detached)"))

    assert status.detached is True
    assert status.branch is None
    assert status.head == _OLD
    assert status.unborn is False


def test_a_branch_without_an_upstream_counts_nothing():
    status = parse_porcelain_v2(payload(*_ON_MAIN))

    assert status.upstream is None
    assert (status.ahead, status.behind) == (None, None)
    assert status.upstream_gone is False


def test_ahead_and_behind_come_from_the_ab_header():
    status = parse_porcelain_v2(payload(
        *_ON_MAIN, "# branch.upstream origin/main", "# branch.ab +3 -1"))

    assert status.upstream == "origin/main"
    assert (status.ahead, status.behind) == (3, 1)
    assert status.upstream_gone is False


def test_a_branch_in_sync_is_still_counted():
    """``+0 -0`` is an answer, and it must not read as "no upstream"."""
    status = parse_porcelain_v2(payload(
        *_ON_MAIN, "# branch.upstream origin/main", "# branch.ab +0 -0"))

    assert (status.ahead, status.behind) == (0, 0)
    assert status.upstream_gone is False


def test_an_upstream_with_no_counts_is_a_gone_upstream():
    """A tracked remote branch that was deleted and pruned.

    git keeps printing ``# branch.upstream`` and simply stops printing
    ``# branch.ab`` (checked against git 2.53), so this pair -- configured
    but uncounted -- is the whole signal porcelain v2 carries. Without it
    the tab would show a branch with no counts as if it were in sync.
    """
    status = parse_porcelain_v2(payload(
        *_ON_MAIN, "# branch.upstream origin/main"))

    assert status.upstream == "origin/main"
    assert (status.ahead, status.behind) == (None, None)
    assert status.upstream_gone is True


def test_half_an_ab_header_is_not_half_an_answer():
    """A wrong number on a badge is worse than no badge."""
    status = parse_porcelain_v2(payload(
        *_ON_MAIN, "# branch.upstream origin/main", "# branch.ab +3"))

    assert (status.ahead, status.behind) == (None, None)


def test_the_stash_count_comes_from_its_header():
    status = parse_porcelain_v2(payload(*_ON_MAIN, "# stash 2"))

    assert status.stash_count == 2


def test_no_stash_header_means_no_stashes():
    """``--show-stash`` prints the line only when the stack is not empty."""
    assert parse_porcelain_v2(payload(*_ON_MAIN)).stash_count == 0


# --- one file, one group ---------------------------------------------------


@pytest.mark.parametrize("xy,group,kind", [
    pytest.param("A.", "staged", "added", id="added-to-the-index"),
    pytest.param("M.", "staged", "modified", id="staged-edit"),
    pytest.param("D.", "staged", "deleted", id="staged-delete"),
    pytest.param("T.", "staged", "typechange", id="staged-typechange"),
    pytest.param(".M", "unstaged", "modified", id="worktree-edit"),
    pytest.param(".D", "unstaged", "deleted", id="worktree-delete"),
    pytest.param(".T", "unstaged", "typechange", id="worktree-typechange"),
])
def test_a_letter_on_one_side_lands_in_one_group(xy, group, kind):
    """``.`` is porcelain v2's "nothing happened on this side"."""
    status = parse_porcelain_v2(payload(*_ON_MAIN, ordinary(xy, "a.txt")))

    others = {"staged", "unstaged"} - {group}
    entries = getattr(status, group)
    assert [(f.path, f.kind, f.xy) for f in entries] == [("a.txt", kind, xy)]
    assert all(getattr(status, name) == [] for name in others)
    assert status.untracked == [] and status.conflicted == []


def test_a_file_changed_on_both_sides_is_in_both_groups():
    """The ``MM`` case, and the reason the two groups are built from X and Y
    separately: the tab offers "unstage" and "discard" on the same file, and
    both entries keep the full ``MM`` so the UI can say why."""
    status = parse_porcelain_v2(payload(*_ON_MAIN, ordinary("MM", "both.txt")))

    assert [(f.path, f.kind, f.xy) for f in status.staged] == [
        ("both.txt", "modified", "MM")]
    assert [(f.path, f.kind, f.xy) for f in status.unstaged] == [
        ("both.txt", "modified", "MM")]


def test_the_two_sides_can_disagree_about_what_happened():
    """Staged as an edit, then deleted on disk: one file, two kinds."""
    status = parse_porcelain_v2(payload(*_ON_MAIN, ordinary("MD", "a.txt")))

    assert [f.kind for f in status.staged] == ["modified"]
    assert [f.kind for f in status.unstaged] == ["deleted"]


def test_an_untracked_file_carries_the_letters_git_omits():
    """Porcelain v2 gives a ``?`` record no XY at all. Every other UI built
    on git spells that state ``??``, and giving it the same two characters
    here means nothing downstream needs a special case for one group."""
    status = parse_porcelain_v2(payload(*_ON_MAIN, "? new.txt"))

    assert [(f.path, f.kind, f.xy) for f in status.untracked] == [
        ("new.txt", "untracked", "??")]
    assert status.staged == [] and status.unstaged == []


def test_an_ignored_entry_is_dropped():
    """Only ``--ignored`` produces these, which we never pass -- and there
    is no group for them, so they must not become untracked files."""
    status = parse_porcelain_v2(payload(*_ON_MAIN, "! secret.env",
                                        "? seen.txt"))

    assert [f.path for f in status.untracked] == ["seen.txt"]


# --- renames and copies ----------------------------------------------------


def test_a_rename_remembers_where_it_came_from():
    status = parse_porcelain_v2(payload(
        *_ON_MAIN, moved("R.", "R100", "new.txt"), "old.txt"))

    assert len(status.staged) == 1
    entry = status.staged[0]
    assert (entry.path, entry.orig_path) == ("new.txt", "old.txt")
    assert (entry.kind, entry.xy, entry.score) == ("renamed", "R.", 100)
    assert status.unstaged == []


def test_a_copy_is_not_a_rename():
    """``2 C.`` with the original still there: git prints this once copy
    detection is on, and "copied" is a different sentence for the user."""
    status = parse_porcelain_v2(payload(
        *_ON_MAIN, moved("C.", "C75", "copied.txt"), "orig.txt"))

    entry = status.staged[0]
    assert (entry.kind, entry.orig_path, entry.score) == (
        "copied", "orig.txt", 75)


def test_only_the_moved_side_carries_the_old_path():
    """``RM``: renamed in the index, then edited on disk. The worktree half
    is an edit to the NEW path -- showing "old -> new" against it would say
    the file had moved twice."""
    status = parse_porcelain_v2(payload(
        *_ON_MAIN, moved("RM", "R100", "new.txt"), "old.txt"))

    staged, unstaged = status.staged[0], status.unstaged[0]
    assert (staged.kind, staged.orig_path, staged.score) == (
        "renamed", "old.txt", 100)
    assert (unstaged.kind, unstaged.orig_path, unstaged.score) == (
        "modified", None, None)
    assert unstaged.path == "new.txt"


def test_the_old_path_is_the_next_token_not_a_field():
    """Both halves have spaces in them. Under ``-z`` the separator between
    them is a NUL, which is the only reason this is unambiguous."""
    status = parse_porcelain_v2(payload(
        *_ON_MAIN,
        moved("R.", "R087", "docs/new name.txt"), "docs/old name.txt",
        "? untracked one.txt"))

    entry = status.staged[0]
    assert (entry.path, entry.orig_path) == (
        "docs/new name.txt", "docs/old name.txt")
    assert entry.score == 87
    assert [f.path for f in status.untracked] == ["untracked one.txt"]


# --- conflicts -------------------------------------------------------------


@pytest.mark.parametrize("xy", ["UU", "AA", "DU", "UD", "DD"])
def test_a_conflicted_file_keeps_its_letters(xy):
    """All of these are one kind and several situations: ``DU`` is "we
    deleted it, they changed it", where "keep ours" means "keep it deleted".
    The resolve buttons come from ``xy``, so it has to survive."""
    status = parse_porcelain_v2(payload(*_ON_MAIN, unmerged(xy, "both.txt")))

    assert [(f.path, f.kind, f.xy) for f in status.conflicted] == [
        ("both.txt", "conflict", xy)]
    assert status.staged == [] and status.unstaged == []


def test_a_conflict_does_not_hide_the_rest_of_the_status():
    status = parse_porcelain_v2(payload(
        *_ON_MAIN, ordinary("A.", "added.txt"), unmerged("AA", "both.txt"),
        "? new.txt"))

    assert [f.path for f in status.staged] == ["added.txt"]
    assert [f.path for f in status.conflicted] == ["both.txt"]
    assert [f.path for f in status.untracked] == ["new.txt"]


# --- paths -----------------------------------------------------------------


@pytest.mark.parametrize("path", [
    pytest.param("src/main.py", id="ordinary"),
    pytest.param("my notes/two words.txt", id="spaces"),
    pytest.param(" leading.txt", id="leading-space"),
    pytest.param("trailing .txt", id="inner-space"),
    # No quoting anywhere in this pipeline, so a CJK name is an ordinary
    # name -- it arrives as the bytes the filesystem holds.
    pytest.param("\u8cc7\u6599/\u8a13\u7df4.csv", id="cjk"),
    pytest.param("\u30c7\u30fc\u30bf/\u30e2\u30c7\u30eb.pt", id="kana"),
    # Legal on every filesystem git supports, and the whole reason this
    # parser splits on NUL rather than on newlines.
    pytest.param("two\nlines.txt", id="newline"),
])
def test_a_path_survives_whatever_it_is_called(path):
    status = parse_porcelain_v2(payload(
        *_ON_MAIN, ordinary(".M", path), f"? {path}"))

    assert [f.path for f in status.unstaged] == [path]
    assert [f.path for f in status.untracked] == [path]


def test_bytes_that_are_not_utf8_do_not_lose_the_status():
    """A repository made on a cp950 machine has filenames that are not
    UTF-8. One unreadable NAME must cost that name a replacement character,
    not cost the user the panel."""
    raw = payload(*_ON_MAIN, "? placeholder", "? seen.txt").replace(
        b"placeholder", b"caf\xe9.txt")

    status = parse_porcelain_v2(raw)

    assert [f.path for f in status.untracked] == ["caf\ufffd.txt", "seen.txt"]


# --- what git might say tomorrow -------------------------------------------


def test_an_unknown_record_type_is_skipped_not_raised():
    """A status read is the tab's heartbeat. If a future git grows a record
    type, the cost of skipping it is one missing row; the cost of raising is
    a panel that shows nothing and an error nobody can act on."""
    status = parse_porcelain_v2(payload(
        *_ON_MAIN, "9 something entirely new", ordinary(".M", "a.txt")))

    assert [f.path for f in status.unstaged] == ["a.txt"]


def test_a_truncated_record_is_skipped():
    status = parse_porcelain_v2(payload(
        *_ON_MAIN, "1 .M N... 100644", ordinary(".M", "a.txt")))

    assert [f.path for f in status.unstaged] == ["a.txt"]


def test_a_rename_missing_its_second_token_is_skipped():
    status = parse_porcelain_v2(payload(
        *_ON_MAIN, moved("R.", "R100", "new.txt")))

    assert status.staged == []


def test_output_that_stops_mid_record_is_not_an_index_error():
    """The last token with no NUL after it -- a rename whose second token
    never arrived. Not something git does, but this parser reads whatever a
    process handed back, and running off the end of the token list would be
    a traceback where a short status would do."""
    raw = (payload(*_ON_MAIN)
           + moved("R.", "R100", "new.txt").encode("utf-8"))

    status = parse_porcelain_v2(raw)

    assert status.staged == []
    assert status.branch == "main"


def test_a_submodule_is_an_ordinary_entry():
    """The ``sub`` field stops being ``N...`` and the modes become
    ``160000``; neither is a field this parser reads, and a submodule the
    user changed still has to show up."""
    status = parse_porcelain_v2(payload(
        *_ON_MAIN,
        f"1 .M SC.. 160000 160000 160000 {_OLD} {_NEW} vendor/sub"))

    assert [(f.path, f.kind) for f in status.unstaged] == [
        ("vendor/sub", "modified")]


def test_empty_output_is_an_empty_status():
    """git prints headers for every real status, so this is a "cannot
    happen" -- and it still must not be a traceback."""
    status = parse_porcelain_v2(b"")

    assert status == GitStatus()


def test_a_clean_repository_has_four_empty_lists():
    status = parse_porcelain_v2(payload(
        *_ON_MAIN, "# branch.upstream origin/main", "# branch.ab +0 -0"))

    assert status.staged == []
    assert status.unstaged == []
    assert status.untracked == []
    assert status.conflicted == []
    assert status.stash_count == 0


# --- the flags the parser cannot know --------------------------------------


def test_the_in_progress_flags_default_to_false():
    """Porcelain v2 says nothing about a merge or a rebase; the answer is
    whether MERGE_HEAD / rebase-merge / rebase-apply exist, which needs a
    git call the service makes right after this one."""
    status = parse_porcelain_v2(payload(*_ON_MAIN))

    assert status.merge_in_progress is False
    assert status.rebase_in_progress is False


def test_the_in_progress_flags_are_the_callers_to_set():
    status = parse_porcelain_v2(payload(*_ON_MAIN, unmerged("UU", "a.txt")),
                                merge_in_progress=True,
                                rebase_in_progress=True)

    assert status.merge_in_progress is True
    assert status.rebase_in_progress is True


# --- the whole thing, exactly as git printed it -----------------------------

#: Captured from git 2.53 in a repository holding one of everything: a file
#: added to the index, one edited in both places, one deleted on disk, one
#: edited and staged, one edited on disk, a rename that was then edited, a
#: staged delete, and an untracked file.
_REAL_CAPTURE = (
    "# branch.oid d8ae323989562f6475c4830b455e0027b0316eab",
    "# branch.head main",
    "1 A. N... 000000 100644 100644 "
    "0000000000000000000000000000000000000000 "
    "d2eb92c3f437753a118e0fb686bbc3d3bba96b63 added.txt",
    "1 MM N... 100644 100644 100644 "
    "052d31effb03dcc923165ae3f6640bb6409b33cb "
    "eeacbdb6b832e024b53d4148bf37fbb054e2f44d both.txt",
    "1 .D N... 100644 100644 000000 "
    "052d31effb03dcc923165ae3f6640bb6409b33cb "
    "052d31effb03dcc923165ae3f6640bb6409b33cb gone.txt",
    "1 M. N... 100644 100644 100644 "
    "052d31effb03dcc923165ae3f6640bb6409b33cb "
    "60b8d2378999073340b43595a43455b277449738 keep.txt",
    "1 .M N... 100644 100644 100644 "
    "052d31effb03dcc923165ae3f6640bb6409b33cb "
    "052d31effb03dcc923165ae3f6640bb6409b33cb mod.txt",
    "2 RM N... 100644 100644 100644 "
    "052d31effb03dcc923165ae3f6640bb6409b33cb "
    "052d31effb03dcc923165ae3f6640bb6409b33cb R100 new.txt",
    "old.txt",
    "1 D. N... 100644 000000 000000 "
    "052d31effb03dcc923165ae3f6640bb6409b33cb "
    "0000000000000000000000000000000000000000 stagedgone.txt",
    "? untracked one.txt",
)


def test_a_real_status_reads_the_way_git_printed_it():
    """One capture, asserted whole: order included.

    git sorts by path, and the tab lists files in the order they arrive, so
    the order is part of the output too -- and this is the test that would
    catch a field-counting mistake the helpers above and I made together.
    """
    status = parse_porcelain_v2(payload(*_REAL_CAPTURE))

    assert status.branch == "main"
    assert status.head == "d8ae323989562f6475c4830b455e0027b0316eab"
    assert [(f.path, f.kind, f.xy) for f in status.staged] == [
        ("added.txt", "added", "A."),
        ("both.txt", "modified", "MM"),
        ("keep.txt", "modified", "M."),
        ("new.txt", "renamed", "RM"),
        ("stagedgone.txt", "deleted", "D."),
    ]
    assert [(f.path, f.kind, f.xy) for f in status.unstaged] == [
        ("both.txt", "modified", "MM"),
        ("gone.txt", "deleted", ".D"),
        ("mod.txt", "modified", ".M"),
        ("new.txt", "modified", "RM"),
    ]
    assert [f.path for f in status.untracked] == ["untracked one.txt"]
    assert status.conflicted == []
    assert [f.orig_path for f in status.staged if f.kind == "renamed"] == [
        "old.txt"]


# --- kind_from_letter ------------------------------------------------------


@pytest.mark.parametrize("letter,kind", [
    ("M", "modified"), ("A", "added"), ("D", "deleted"),
    ("R", "renamed"), ("C", "copied"), ("T", "typechange"),
])
def test_every_status_letter_has_a_kind(letter, kind):
    assert kind_from_letter(letter) == kind


@pytest.mark.parametrize("letter", ["X", ".", "", "u", "m"])
def test_an_unexpected_letter_is_reported_as_modified(letter):
    """git has grown letters before. Raising would turn one odd file into a
    failed status read and dropping it would hide a change the user made;
    "something changed here" is the one thing every letter has in common,
    and the exact letters travel alongside in ``xy``."""
    assert kind_from_letter(letter) == "modified"


# --- the request models ----------------------------------------------------


def test_paths_request_takes_a_list_of_paths():
    request = PathsRequest(paths=["a.txt", "b/c.txt"])

    assert request.paths == ["a.txt", "b/c.txt"]
    assert request.all is False


def test_paths_request_takes_the_whole_tree():
    request = PathsRequest(all=True)

    assert request.paths is None
    assert request.all is True


@pytest.mark.parametrize("body", [
    pytest.param({}, id="neither"),
    pytest.param({"paths": ["a.txt"], "all": True}, id="both"),
    pytest.param({"paths": [], "all": True}, id="both-with-an-empty-list"),
    pytest.param({"paths": []}, id="empty-list"),
    pytest.param({"paths": None, "all": False}, id="spelled-out-neither"),
])
def test_paths_request_wants_exactly_one_form(body):
    """``git add -A --`` with an empty pathspec stages the WHOLE tree, so
    "nothing was selected" must never be able to arrive as "all" -- and a
    body carrying both would otherwise be decided by whichever branch a
    handler happened to test first."""
    with pytest.raises(ValidationError):
        PathsRequest(**body)


def test_paths_request_refuses_a_key_nobody_defined():
    """``extra="forbid"`` is what makes "the client cannot smuggle in an
    argument" a property of the schema instead of of every handler."""
    with pytest.raises(ValidationError):
        PathsRequest(paths=["a.txt"], force=True)


def test_commit_request_defaults_to_one_plain_commit():
    request = CommitRequest(message="fix the thing")

    assert (request.all, request.amend) == (False, False)


def test_commit_request_refuses_a_key_nobody_defined():
    with pytest.raises(ValidationError):
        CommitRequest(message="x", author="someone else")


def test_identity_request_may_set_one_field():
    """The other stays unset rather than being cleared."""
    request = IdentityRequest(email="me@example.com")

    assert request.name is None
    assert request.email == "me@example.com"


def test_identity_request_refuses_a_key_nobody_defined():
    with pytest.raises(ValidationError):
        IdentityRequest(name="me", scope="global")


def test_a_mutation_result_must_carry_the_fresh_status():
    """Present AND not null. The frontend is typed from these models, so a
    nullable status here would be a branch it has to handle and the service
    is required never to produce; a write that cannot be read back
    afterwards is a failed request, not a result with a hole in it."""
    with pytest.raises(ValidationError):
        MutationResult()

    with pytest.raises(ValidationError):
        MutationResult(status=None)

    assert MutationResult(status=GitStatus()).changed_paths == []


def test_an_empty_git_status_is_a_clean_repository():
    """The default of every field is "git did not mention it"."""
    status = GitStatus()

    assert (status.staged, status.unstaged) == ([], [])
    assert (status.untracked, status.conflicted) == ([], [])
    assert status.branch is None and status.head is None
    assert status.stash_count == 0
    assert status.upstream_gone is False
