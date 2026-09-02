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
    pytest.param(
        "error: Your local changes to the following files would be "
        "overwritten by merge:\n\ta.txt\n"
        "Please commit your changes or stash them before you merge.\n"
        "Aborting\n",
        "dirty_tree", id="dirty-tree"),
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
    pytest.param(
        "error: could not lock config file .git/config: File exists\n",
        "git_failed", id="unrecognised"),
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


def test_a_classified_failure_does_not_leak_stderr():
    """A code the frontend can translate needs no raw English attached."""
    error = classify_failure(_ARGV, 128, "fatal: Authentication failed for 'x'\n")

    assert error.stderr is None


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
