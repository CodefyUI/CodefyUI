"""Turning git's prose into a code the Source Control tab can act on.

The tab is used by people who may not read English, so nothing git prints
is ever shown to them directly: every failure becomes a ``code`` the
frontend translates. That makes the classifier a piece of user-facing
behaviour, and it fails quietly -- a mis-read stderr does not raise, it
tells the user the wrong thing to do next ("log in" for a network outage,
"retry" for a wrong password).

So every row of the table gets a test with real git output, and the two
rows whose ORDER is the only thing keeping them apart -- auth before
network, both of which an ssh failure ends with the same line -- get one
each in both directions.
"""

from __future__ import annotations

import pytest

from app.core.git.errors import (
    CODES,
    LAST_STDERR_LINES,
    GitBusy,
    GitError,
    classify_failure,
    default_message,
    redact,
    stderr_tail,
)

#: The whole vocabulary, written out again here rather than derived from
#: ``CODES``: this is the list from the G1 plan, and a test that read the
#: table it is checking would pass no matter what the table said.
_PLANNED_CODES = {
    "no_project", "git_missing", "git_too_old", "not_repo", "busy", "timeout",
    "auth_required", "network", "non_fast_forward", "diverged", "conflict",
    "dirty_tree", "no_upstream", "no_remote", "nothing_to_commit",
    "identity_missing", "detached_head", "merge_in_progress", "branch_exists",
    "branch_not_merged", "signing_failed", "not_found", "invalid_path",
    "invalid_ref", "invalid_url", "invalid_value", "path_not_in_status",
    "ignored", "git_failed",
    # G3's two, from the plan's own ruling: a duplicate ``git remote add``,
    # and a push a server-side rule refused.
    "remote_exists", "remote_rejected",
}

#: One representative stderr per classified row, as git prints it under
#: ``LC_ALL=C``. The two ssh cases are the split described in the module
#: docstring: same last line, opposite meanings.
_SAMPLES = [
    pytest.param(
        "fatal: could not read Username for 'https://github.com': "
        "terminal prompts disabled\n",
        "auth_required", id="prompts-disabled"),
    pytest.param(
        "remote: Invalid username or token. Password authentication is not "
        "supported.\nfatal: Authentication failed for 'https://github.com/a/b/'\n",
        "auth_required", id="wrong-token"),
    pytest.param(
        "fatal: unable to access 'https://github.com/a/b/': "
        "The requested URL returned error: 403\n",
        "auth_required", id="http-403"),
    pytest.param(
        "git@github.com: Permission denied (publickey).\n"
        "fatal: Could not read from remote repository.\n\n"
        "Please make sure you have the correct access rights\n",
        "auth_required", id="ssh-key-refused"),
    pytest.param(
        "Host key verification failed.\n"
        "fatal: Could not read from remote repository.\n",
        "auth_required", id="ssh-host-key"),
    pytest.param(
        # A server that offers password auth first: ssh's method list is in
        # the SERVER's order, so the exact "(publickey" phrase is absent and
        # only the pairing with the remote-read line classifies this.
        "git@example.com: Permission denied (password,publickey).\n"
        "fatal: Could not read from remote repository.\n",
        "auth_required", id="ssh-password-first"),
    pytest.param(
        "ssh: connect to host github.com port 22: Connection timed out\n"
        "fatal: Could not read from remote repository.\n",
        "network", id="ssh-unreachable"),
    pytest.param(
        "fatal: unable to access 'https://github.com/a/b/': "
        "Could not resolve host: github.com\n",
        "network", id="dns"),
    pytest.param(
        " ! [rejected]        main -> main (fetch first)\n"
        "error: failed to push some refs to 'https://github.com/a/b'\n"
        "hint: Updates were rejected because the remote contains work that you\n",
        "non_fast_forward", id="push-rejected"),
    pytest.param(
        " ! [rejected]        main -> main (non-fast-forward)\n",
        "non_fast_forward", id="push-non-ff"),
    # A server-side rule said no: a pre-receive hook, or a protected branch.
    # Copied off git 2.53 pushing to a bare repository whose pre-receive
    # hook exits 1. Note that it does NOT contain the ``[rejected]`` of the
    # rows above -- the character before "rejected" is a space.
    pytest.param(
        "remote: this branch is protected\n"
        "To file:///srv/mirrors/repo.git\n"
        " ! [remote rejected] main -> main (pre-receive hook declined)\n"
        "error: failed to push some refs to 'file:///srv/mirrors/repo.git'\n",
        "remote_rejected", id="push-refused-by-the-server"),
    # ``git remote add`` on a name that is taken (measured, exit 3). git's
    # own voice under an ``error: `` opening, which is why its row anchors
    # to the whole ``error: remote `` rather than to the prefix set.
    pytest.param(
        "error: remote origin already exists.\n",
        "remote_exists", id="duplicate-remote"),
    pytest.param(
        "fatal: Need to specify how to reconcile divergent branches.\n",
        "diverged", id="pull-divergent"),
    pytest.param(
        "fatal: Not possible to fast-forward, aborting.\n",
        "diverged", id="ff-only"),
    pytest.param(
        "Auto-merging a.txt\nCONFLICT (content): Merge conflict in a.txt\n"
        "Automatic merge failed; fix conflicts and then commit the result.\n",
        "conflict", id="merge-conflict"),
    # The OTHER moment a conflict is reported: ``git commit`` refusing to
    # run while the index still holds an unmerged path. Copied off git 2.53
    # (exit 128); it shares no phrase with the merge-time lines above.
    pytest.param(
        "error: Committing is not possible because you have unmerged files.\n"
        "hint: Fix them up in the work tree, and then use 'git add/rm <file>'\n"
        "hint: as appropriate to mark resolution and make a commit.\n"
        "fatal: Exiting because of an unresolved conflict.\n",
        "conflict", id="commit-blocked-by-a-conflict"),
    pytest.param(
        # The fatal line on its own: a wrapper (or a future git) may print
        # only its own summary, so each phrase has to be enough by itself.
        "fatal: Exiting because of an unresolved conflict.\n",
        "conflict", id="unresolved-conflict-alone"),
    pytest.param(
        "error: Your local changes to the following files would be "
        "overwritten by merge:\n\ta.txt\n"
        "Please commit your changes or stash them before you merge.\n"
        "Aborting\n",
        "dirty_tree", id="dirty-tree"),
    pytest.param(
        # ``stash pop`` restoring an untracked file onto one that is back on
        # disk. The WHOLE of what the caller classifies is both streams
        # joined, and the stdout half is a ``git status`` whose last line is
        # a ``nothing_to_commit`` phrase -- so this sample carries both, and
        # it is the row ORDER that makes the answer "move it out of the
        # way" rather than "there is nothing to commit".
        "Already up to date.\nOn branch main\nUntracked files:\n\tn.txt\n\n"
        "nothing added to commit but untracked files present "
        '(use "git add" to track)\n'
        "The stash entry is kept in case you need it again.\n"
        "\nn.txt already exists, no checkout\n"
        "error: could not restore untracked files from stash\n",
        "dirty_tree", id="stash-pop-onto-a-recreated-file"),
    pytest.param(
        "fatal: The current branch feat has no upstream branch.\n",
        "no_upstream", id="no-upstream"),
    pytest.param(
        "There is no tracking information for the current branch.\n",
        "no_upstream", id="no-tracking"),
    pytest.param(
        'no changes added to commit (use "git add" and/or "git commit -a")\n',
        "nothing_to_commit", id="nothing-staged"),
    pytest.param(
        "Author identity unknown\n\n*** Please tell me who you are.\n"
        "fatal: unable to auto-detect email address\n",
        "identity_missing", id="no-identity"),
    pytest.param(
        "fatal: You are not currently on a branch.\n",
        "detached_head", id="detached"),
    pytest.param(
        "fatal: a branch named 'feat' already exists\n",
        "branch_exists", id="branch-exists"),
    pytest.param(
        "error: the branch 'feat' is not fully merged\n",
        "branch_not_merged", id="branch-not-merged"),
    pytest.param(
        "error: gpg failed to sign the data\n"
        "fatal: failed to write commit object\n",
        "signing_failed", id="signing"),
    pytest.param(
        "error: pathspec 'nope.txt' did not match any file(s) known to git\n",
        "not_found", id="bad-pathspec"),
    pytest.param(
        "fatal: ambiguous argument 'deadbee': unknown revision or path not in "
        "the working tree.\n",
        "not_found", id="unknown-revision"),
    pytest.param(
        "fatal: bad revision 'refs/heads/gone'\n",
        "not_found", id="bad-revision"),
    # The next four are git 2.53's own words for the G1 read operations,
    # copied off a real run: opening a file at a ref where it does not
    # exist is the most ordinary miss the tab can produce, and none of the
    # phrases above it says so.
    pytest.param(
        "fatal: path 'no/such/file' does not exist in 'HEAD'\n",
        "not_found", id="missing-path-at-a-ref"),
    pytest.param(
        "fatal: path 'no/such/file' does not exist (neither on disk nor in "
        "the index)\n",
        "not_found", id="missing-path-in-the-index"),
    pytest.param(
        "fatal: invalid object name 'nosuchref'.\n",
        "not_found", id="unknown-ref"),
    pytest.param(
        "fatal: Needed a single revision\n",
        "not_found", id="unverifiable-sha"),
    pytest.param(
        "error: No such remote 'nope'\n",
        "not_found", id="unknown-remote"),
    pytest.param(
        "error: stash@{9} is not a valid reference\n",
        "not_found", id="unknown-stash"),
    # The two G3 added to the ``not_found`` row: switching to a branch and
    # branching from a start point that were both there when the panel was
    # drawn and are not there now.
    pytest.param(
        "fatal: invalid reference: feat\n",
        "not_found", id="branch-gone"),
    pytest.param(
        "fatal: not a valid object name: 'deadbee'\n",
        "not_found", id="start-point-gone"),
    pytest.param(
        "error: could not lock config file .git/config: File exists\n",
        "git_failed", id="unrecognised"),
    pytest.param(
        # INDENTED, so it is not git's own sentence: git never indents a
        # ``fatal:``, and a hook that pretty-prints its output would
        # otherwise get to choose the code the user is shown.
        "    fatal: config not found\n",
        "git_failed", id="an-indented-fatal-is-hook-prose"),
]

_ARGV = ["git", "-C", "/p", "-c", "core.quotepath=false", "-c",
         "core.askPass=", "-c", "color.ui=never", "push", "origin", "main"]


@pytest.mark.parametrize("stderr,code", _SAMPLES)
def test_classification(stderr, code):
    """Real git output lands on the row the plan says it lands on."""
    error = classify_failure(_ARGV, 128, stderr)

    assert error.code == code
    assert error.status == CODES[code][0]


def test_the_ssh_split_is_the_row_order():
    """Both ssh failures end with the same line; only the line ABOVE it says
    which one it was, so the auth row has to be tried first."""
    tail = "fatal: Could not read from remote repository.\n"

    refused = classify_failure(_ARGV, 128, "Permission denied (publickey).\n" + tail)
    unreachable = classify_failure(_ARGV, 128,
                                   "ssh: connect to host h port 22: "
                                   "Connection refused\n" + tail)
    bare = classify_failure(_ARGV, 128, tail)

    assert refused.code == "auth_required"
    assert unreachable.code == "network"
    # Nothing explains it: the remote could not be read, and that is all we
    # can honestly say.
    assert bare.code == "network"


@pytest.mark.parametrize("line", [
    "fatal: could not read Username for 'https://github.com': "
    "terminal prompts disabled",
    "fatal: could not read Password for 'https://u@github.com': "
    "No such device or address",
    "fatal: Authentication failed for 'https://github.com/a/b/'",
    "remote: HTTP Basic: Access denied",
    "remote: Invalid username or password.",
    "remote: Invalid username or token. Password authentication is not "
    "supported for Git operations.",
    # No "Could not read from remote repository" line under it: the phrase
    # has to be enough on its own, because a wrapper (or a future git) may
    # not print git's own summary line.
    "git@github.com: Permission denied (publickey).",
    "Host key verification failed.",
    "fatal: unable to access 'https://github.com/a/b/': "
    "The requested URL returned error: 401",
])
def test_a_credential_phrase_is_enough_on_its_own(line):
    """Each phrase classifies by itself -- the row is an OR, not a recipe."""
    assert classify_failure(_ARGV, 128, line + "\n").code == "auth_required"


@pytest.mark.parametrize("line", [
    # git says this one on STDOUT, which is why classify_failure takes text
    # and not a GitResult: the commit caller joins both streams.
    "nothing to commit, working tree clean",
    'no changes added to commit (use "git add" and/or "git commit -a")',
])
def test_either_way_of_saying_nothing_to_commit_is_enough(line):
    assert classify_failure(_ARGV, 1, line + "\n").code == "nothing_to_commit"


def test_untracked_files_only_is_also_nothing_to_commit():
    """git's third phrasing, and the one it shares no words with the others.

    A repository whose only changes are new files answers a commit with
    "nothing ADDED to commit" -- which contains neither "nothing to commit"
    nor "no changes added to commit". It is the most ordinary empty commit
    there is (a first commit where nobody staged anything), and without its
    own phrase it was a 500. Measured against git 2.53, on stdout, exit 1.
    """
    stdout = (
        "On branch main\n"
        "Untracked files:\n"
        '  (use "git add <file>..." to include in what will be committed)\n'
        "\tnew.py\n"
        "\n"
        'nothing added to commit but untracked files present (use "git add" '
        "to track)\n")

    assert classify_failure(_ARGV, 1, stdout).code == "nothing_to_commit"


def test_a_local_permission_error_is_not_a_login_problem():
    """"Permission denied" on a lock file is not "your key was refused" --
    the auth row pairs it with the remote-read line for exactly this."""
    error = classify_failure(
        _ARGV, 128,
        "error: open('.git/index.lock'): Permission denied\n")

    assert error.code == "git_failed"


def test_an_unrecognised_failure_keeps_the_tail_of_stderr():
    """The only code that hands the raw text on: there is nothing else to
    go on, and the last lines are where git puts the reason."""
    stderr = "\n".join(f"line {n}" for n in range(50)) + "\n"

    error = classify_failure(_ARGV, 1, stderr)

    assert error.code == "git_failed"
    assert error.status == 500
    assert error.stderr is not None
    assert error.stderr.splitlines() == [
        f"line {n}" for n in range(50 - LAST_STDERR_LINES, 50)]


def test_a_classified_failure_keeps_the_evidence_too():
    """A code is a CLAIM about what went wrong, and a claim with nothing
    under it cannot be argued with.

    This test used to assert the opposite -- that a translated code needs no
    raw English attached -- and that premise was the defect: when the
    classification is wrong (a hook's output caught by an ordinary English
    phrase, say) the user got a confident wrong answer AND the sentence that
    would have explained it was gone. What a route puts in the response body
    is still the route's decision.
    """
    error = classify_failure(_ARGV, 128, "fatal: Authentication failed for 'x'\n")

    assert error.code == "auth_required"
    assert error.stderr == "fatal: Authentication failed for 'x'"


def test_stderr_that_says_nothing_is_reported_as_nothing():
    """``rev-parse --verify --quiet`` fails silently by design. An empty tail
    is the honest answer -- git failed and said nothing -- and it is still a
    string, so a route never has to tell None and "" apart."""
    error = classify_failure(_ARGV, 1, "")

    assert error.code == "git_failed"
    assert error.stderr == ""


# --- hook output is not git output -----------------------------------------


@pytest.mark.parametrize("stderr", [
    pytest.param("ruff: command not found\n", id="linter-not-installed"),
    pytest.param("node_modules/.bin/eslint: not found\n", id="tool-not-found"),
    pytest.param("error: pre-commit hook: .venv does not exist\n",
                 id="hook-missing-venv"),
    pytest.param("error: cannot lock ref: reference already exists\n",
                 id="not-a-branch-that-exists"),
    # TWO lines, and the anchored one comes first: this is what a real
    # failing hook looks like, and it pins the anchor to the LINE rather
    # than to the stream. A matcher that searched the whole text would find
    # "not found" on the second line, next to a "fatal: " on the first, and
    # answer "git found no such object" for a missing linter.
    pytest.param("fatal: cannot run .git/hooks/pre-commit: No such file or "
                 "directory\nruff: command not found\n",
                 id="hook-that-could-not-be-run"),
])
def test_a_failing_hook_is_not_a_missing_object(stderr):
    """``commit`` runs the user's hooks, and their output lands on this same
    stream. "not found" from a lint hook must not become "git found no such
    object or path" -- an answer that is wrong, confident, and (before the
    tail was kept) unaccompanied by the line that explains it.

    ``error: `` is deliberately not one of the prefixes that anchors a
    phrase: git uses it for what it can carry on past, and a hook uses it
    for everything.
    """
    error = classify_failure(_ARGV, 1, stderr)

    assert error.code == "git_failed"
    assert error.stderr == stderr.strip()


@pytest.mark.parametrize("stderr,code", [
    ("fatal: path 'x' does not exist in 'HEAD'\n", "not_found"),
    ("fatal: repository 'https://github.com/a/b.git' not found\n", "not_found"),
    ("remote: Repository not found.\n", "not_found"),
    ("fatal: a branch named 'feat' already exists\n", "branch_exists"),
])
def test_an_anchored_phrase_still_catches_git_saying_it(stderr, code):
    """The anchor narrows WHO is speaking, not what git means."""
    assert classify_failure(_ARGV, 128, stderr).code == code


def test_a_forge_saying_already_exists_is_still_branch_exists():
    """``remote_exists`` sits BELOW ``branch_exists`` so that row keeps
    everything it means today: a phrase arriving through ``remote: `` is
    the far side talking about a ref, and answering it with "a remote with
    that name already exists" would be a confident lie."""
    stderr = "remote: error: refusing to create the tag: it already exists\n"

    assert classify_failure(_ARGV, 1, stderr).code == "branch_exists"


@pytest.mark.parametrize("stderr", [
    pytest.param("    error: remote origin already exists.\n", id="indented"),
    pytest.param("error: the remote origin already exists.\n", id="mid-line"),
    pytest.param("error: remote origin is not there.\n", id="another-sentence"),
])
def test_only_gits_own_duplicate_remote_sentence_is_remote_exists(stderr):
    """The anchor is the whole opening ``error: remote ``, because
    ``error: `` alone belongs to every failing hook. Anything a hook could
    plausibly print stays ``git_failed`` -- with git's own tail attached."""
    assert classify_failure(_ARGV, 1, stderr).code == "git_failed"


def test_a_server_refusal_beats_a_non_fast_forward():
    """One push can report both, one ref each. "The server said no" is the
    half a pull cannot fix, so it is the answer the user gets."""
    stderr = (" ! [rejected]        old -> old (non-fast-forward)\n"
              " ! [remote rejected] main -> main (protected branch hook "
              "declined)\n")

    assert classify_failure(_ARGV, 1, stderr).code == "remote_rejected"


def test_the_failing_subcommand_is_named():
    """The message is for a log, and a log line without the subcommand is a
    line nobody can place. The fixed prefix must not be mistaken for it."""
    error = classify_failure(_ARGV, 1, "error: unrecognised\n")

    assert "push" in error.message
    assert "core.quotepath" not in error.message


def test_a_command_that_is_all_options_still_has_a_message():
    """No subcommand to name, and no "git  failed" with a hole in it."""
    error = classify_failure(["git", "--version"], 1, "boom\n")

    assert error.message == "git failed (exit 1)"


def test_stderr_tail_drops_the_trailing_blank_lines():
    assert stderr_tail("a\nb\n\n\n") == "a\nb"
    assert stderr_tail("a\nb\nc\n", limit=2) == "b\nc"


# --- the vocabulary --------------------------------------------------------


def test_every_planned_code_exists():
    """The table in the plan and the table in the code are the same table."""
    assert set(CODES) == _PLANNED_CODES


def test_every_code_has_a_plausible_status():
    assert all(400 <= status <= 599 for status, _ in CODES.values())


def test_every_message_is_plain_ascii():
    """No translated prose in core: the frontend translates the CODE, and a
    second translation here would be one nobody maintains. (Also the reason
    the whole backend stays ASCII -- a cp950 console cannot print the rest.)"""
    for code, (_, message) in CODES.items():
        assert message.isascii(), code
        assert message == message.strip()


def test_default_message_survives_an_unknown_code():
    """An unknown code must not be a KeyError in an error path."""
    assert default_message("something_new") == "something_new"


def test_git_error_carries_what_a_route_needs():
    error = GitError("not_found", 404, "no such commit", hint="try HEAD",
                     stderr="fatal: bad object")

    assert (error.code, error.status) == ("not_found", 404)
    assert error.message == "no such commit"
    assert error.hint == "try HEAD"
    assert str(error) == "no such commit"


def test_git_error_falls_back_to_the_tables_message():
    assert GitError("not_repo", 409).message == CODES["not_repo"][1]


def test_git_busy_names_the_operation_that_holds_the_lock():
    """A caller that catches this says "wait for X", not "busy"."""
    busy = GitBusy("commit")

    assert isinstance(busy, GitError)
    assert (busy.code, busy.status) == ("busy", 409)
    assert busy.op == "commit"
    assert "commit" in busy.message


# --- what leaves the server -------------------------------------------------


#: A token that is obviously one, so a failing assertion below says which
#: string escaped rather than pointing at a plausible-looking word.
TOKEN = "ghp_16C7e42F292c6912E7710c838347Ae178B4a"


@pytest.mark.parametrize("text", [
    pytest.param(
        f"fatal: unable to access 'https://octocat:{TOKEN}@github.com/o/r/': "
        "The requested URL returned error: 403\n",
        id="password-in-a-url"),
    pytest.param(
        f"fatal: could not read Username for 'https://x-access-token:{TOKEN}"
        "@github.com': terminal prompts disabled\n",
        id="installation-token-in-a-url"),
    pytest.param(
        f"remote: HTTP Basic: Access denied for token {TOKEN}\n",
        id="token-on-its-own"),
    pytest.param(
        f"x-access-token:{TOKEN}\n", id="header-form-without-a-url"),
])
def test_a_credential_never_survives_redaction(text):
    """Every shape the same token can arrive in, masked."""
    assert TOKEN not in redact(text)


def test_redaction_keeps_the_fact_that_makes_the_error_actionable():
    """The host and the path stay: "which remote failed" is the answer."""
    masked = redact(
        f"fatal: unable to access 'https://octocat:{TOKEN}@github.com/o/r/': "
        "The requested URL returned error: 403\n")

    assert "github.com/o/r/" in masked
    assert "https://***@github.com/o/r/" in masked
    assert "The requested URL returned error: 403" in masked


@pytest.mark.parametrize("prefix", ["ghp_", "github_pat_", "gho_", "ghs_",
                                    "ghu_", "glpat-"])
def test_every_published_token_prefix_is_masked(prefix):
    """One row per issuer prefix: a new one must be added deliberately."""
    assert redact(f"remote: rejected {prefix}AbC123_def-456 here") == (
        "remote: rejected *** here")


def test_redaction_leaves_ordinary_git_output_alone():
    """It runs on EVERY error, so it may not damage the other 28 codes.

    The two shapes that look like a credential and are not: an ssh remote
    (``git@host:path`` -- a username, and the only thing that says which
    host) and an email address, which the identity errors are about.
    """
    plain = ("fatal: repository 'git@github.com:owner/repo.git' not found\n"
             "Author identity unknown: author@example.com\n")

    assert redact(plain) == plain


def test_redaction_masks_more_than_one_credential_in_one_line():
    """A push failure prints the URL twice, and a mask that stops at the
    first one is a leak with a plausible-looking body."""
    masked = redact(f"https://a:{TOKEN}@h/x https://b:{TOKEN}@h/y")

    assert masked == "https://***@h/x https://***@h/y"


def test_an_unencoded_at_sign_in_a_password_does_not_end_the_mask():
    """A URL splits its authority at the LAST ``@``, so the mask must too.

    ``https://user:p@ss@github.com/o/r`` is a URL git accepts and prints
    back verbatim. A mask that stopped at the FIRST ``@`` left
    ``https://***@ss@github.com/o/r`` -- half the password, in a string
    that still looks redacted, which is worse than not masking at all.
    """
    assert redact("https://user:p@ss@github.com/o/r") == (
        "https://***@github.com/o/r")


def test_the_mask_never_runs_past_the_authority():
    """Greedy to the last ``@`` before a slash or a space, and no further.

    The pin for the class that made the test above pass: it may not swallow
    a path segment containing an ``@``, and it may not join two URLs on one
    line into a single match.
    """
    assert redact("https://a:pw@h/x@y") == "https://***@h/x@y"
    assert redact("https://a:p@w@h/x https://b:p@w@h/y") == (
        "https://***@h/x https://***@h/y")


def test_a_classified_failure_is_already_redacted():
    """The mask runs here too, not only on the way out of a route.

    A route is still what decides what reaches a browser -- but a credential
    that was never put INTO the exception cannot escape through a log line,
    a test fixture, or the next caller somebody writes. Redaction is
    idempotent, so the second pass at the route costs nothing.
    """
    error = classify_failure(
        _ARGV, 128,
        f"fatal: unable to access 'https://octocat:{TOKEN}@github.com/o/r/': "
        "The requested URL returned error: 403\n")

    assert error.code == "auth_required"
    assert error.stderr is not None
    assert TOKEN not in error.stderr
    # And the half that makes it actionable is still there.
    assert "github.com/o/r/" in error.stderr


def test_redaction_survives_the_empty_string():
    """It runs unconditionally on a message that a code may not have set."""
    assert redact("") == ""
