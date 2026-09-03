"""What a git command failed at, in terms the Source Control tab can act on.

git says everything in prose, on stderr, in English -- and the person
reading the editor may not read English at all. So nothing here returns a
sentence to display: every failure carries a ``code`` from a closed
vocabulary (:data:`CODES`), and the frontend owns the wording. The
``message`` is for logs and for a developer reading a 500; ``hint`` is the
one extra fact a code cannot carry (which branch, which file); ``stderr``
is the last :data:`LAST_STDERR_LINES` lines of what git actually said, and
EVERY failure carries it.

That last part is deliberate and was once the other way round. A code the
frontend can translate looks like it needs no raw text attached -- until
the classification is wrong, and then the user is told something confident
and false with nothing left to explain it. The commit route runs the
user's own hooks, whose output lands on this same stream, so a
misclassification is not hypothetical. Keeping the tail on every failure
costs a few hundred bytes and makes every wrong answer diagnosable; what
reaches the browser is the route's decision, not this module's.

Four decisions here are not stylistic:

* **one exception type, not fifteen.** A route needs the HTTP status and a
  string the frontend can switch on; a caller further up needs to catch
  "any git failure" without importing a taxonomy. So the taxonomy lives in
  ``code``, which is data, and the only subclass is :class:`GitBusy` --
  because a caller that wants to say "wait for the other operation" has to
  be able to catch exactly that, and it carries an extra field.
* **classification is a table, read top to bottom.** The alternative is a
  chain of ``if "..." in stderr`` that grows a new branch per bug report,
  where the order that makes it correct is invisible. Here the order IS the
  table: :data:`_RULES` is ordered, the first row that matches wins, and
  the rows that must come first (auth before network -- see below) are
  adjacent to a comment saying why.
* **the auth-vs-network split.** An ssh failure ALWAYS ends with "Could not
  read from remote repository", whether the key was refused or the host was
  unreachable, and those are opposite instructions for the user. The real
  reason is on the line above it, so the auth row carries the credential
  phrases and sits ABOVE the network row: "Permission denied", "publickey"
  or "Host key" anywhere in the same stderr means the connection worked and
  the credentials did not.
* **three phrases are anchored to git's own voice.** "not found", "already
  exists" and "does not exist" are ordinary English, and this stream is not
  only git's: ``commit`` runs the user's hooks and prints whatever they
  print. ``ruff: command not found`` from a failing lint hook must not be
  answered with "git found no such object or path". :class:`Anchored` says
  the phrase only counts on a line that opens with one of
  :data:`GIT_MESSAGE_PREFIXES`.

Matching is case-insensitive substring matching, deliberately: git's own
wording is stable under ``LC_ALL=C``, but the ``remote:`` lines a forge
prints through it are not, and neither is the capitalisation of the
``fatal:``/``error:`` prefix across versions.

Stdlib only, and imported by ``git/__init__.py``: catching a git failure
never drags the subprocess machinery in with it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

#: How much of git's stderr an unclassified failure keeps. Enough to hold a
#: hook's whole complaint, short enough to put in a toast.
LAST_STDERR_LINES = 20

#: The whole error vocabulary: ``code -> (HTTP status, fallback message)``.
#:
#: The frontend translates the CODE, so these messages are for logs, for a
#: developer reading a 500, and for the rare client that shows the raw
#: detail. ASCII and English on purpose -- a translated string here would be
#: a second, worse translation layer competing with the real one.
#:
#: Not every code is produced by :func:`classify_failure`: the pre-flight
#: ones (``no_project``, ``not_repo``, ``no_remote``, ``merge_in_progress``,
#: ``busy``) are raised by the service before a process starts, and the 400s
#: by the validators in ``paths.py``. They live in one table anyway, because
#: "what can this API return" is a question with one answer.
CODES: dict[str, tuple[int, str]] = {
    # Pre-flight: the repository is not in a state where git can be asked.
    "no_project": (409, "no project directory is open"),
    "git_missing": (503, "git was not found on PATH"),
    "git_too_old": (409, "this git is too old for the source control tab"),
    "not_repo": (409, "the project directory is not a git repository"),
    "busy": (409, "another git operation is already running"),
    "merge_in_progress": (409, "a merge is already in progress"),
    "no_remote": (409, "this repository has no remote"),
    # The process itself.
    "timeout": (504, "git took too long and was stopped"),
    # Talking to a remote.
    "auth_required": (409, "git needs credentials this server cannot supply"),
    "network": (409, "git could not reach the remote"),
    "non_fast_forward": (409, "the remote has commits this branch does not"),
    "diverged": (409, "the local and the remote branch have diverged"),
    "no_upstream": (409, "this branch has no upstream branch"),
    # The working tree.
    "conflict": (409, "the merge left conflicts to resolve"),
    "dirty_tree": (409, "uncommitted changes would be overwritten"),
    "nothing_to_commit": (409, "there is nothing to commit"),
    "identity_missing": (409, "git has no user.name / user.email to commit with"),
    "detached_head": (409, "HEAD is not on a branch"),
    "branch_exists": (409, "that branch already exists"),
    "branch_not_merged": (409, "that branch is not fully merged"),
    "signing_failed": (409, "signing the commit failed"),
    "not_found": (404, "git found no such object or path"),
    # Validation (``paths.py``, and the service's freshness checks).
    "invalid_path": (400, "that path is not allowed"),
    "invalid_ref": (400, "that name is not a valid git ref"),
    "invalid_url": (400, "that remote URL is not allowed"),
    "invalid_value": (400, "that value is not allowed"),
    "path_not_in_status": (400, "that path is not in the current status"),
    "ignored": (403, "that file is ignored and is not served"),
    # Everything else.
    "git_failed": (500, "git failed"),
}


def default_message(code: str) -> str:
    """The fallback message for *code*; the code itself when it is unknown.

    An unknown code is not an error here. It would be one more way for a
    failure to become a traceback instead of a response, and the frontend
    already has to survive a code it does not know.
    """
    known = CODES.get(code)
    return known[1] if known is not None else code


class GitError(Exception):
    """A git operation failed, in a shape a route can answer with.

    ``code`` is the closed-vocabulary string the frontend switches on,
    ``status`` the HTTP status the route returns, ``message`` a plain-ASCII
    English sentence for logs, ``hint`` the one fact the code cannot carry,
    and ``stderr`` the tail of git's own output -- present on every failure
    :func:`classify_failure` produces, so that a wrong code is still a
    diagnosable one (see the module docstring). What a route puts in a
    response body is the route's decision.
    """

    def __init__(self, code: str, status: int, message: str | None = None,
                 hint: str | None = None, stderr: str | None = None):
        self.code = code
        self.status = status
        self.message = message if message is not None else default_message(code)
        self.hint = hint
        self.stderr = stderr
        super().__init__(self.message)


class GitBusy(GitError):
    """Another mutation holds the lock. Nothing was attempted.

    Its own class, and the only subclass, because the caller's response is
    different in kind: nothing failed and nothing changed, so the answer is
    "the other operation is still running" and the same request usually
    works a moment later. ``op`` names that other operation, so the UI can
    say which one rather than "busy".
    """

    def __init__(self, op: str, message: str | None = None,
                 hint: str | None = None):
        super().__init__(
            "busy", CODES["busy"][0],
            message if message is not None
            else f"another git operation is already running ({op})",
            hint=hint)
        self.op = op


#: The openings git puts on its own sentences. Matched at the very START of
#: a line -- an INDENTED ``fatal:`` is not git's, because git never indents
#: its own, and a hook that pretty-prints its output must not be able to
#: pick the code the user is shown -- and the set is short on purpose.
#:
#: ``error: `` is NOT here, and that is the whole point of anchoring: git
#: uses it for the problems it can carry on past, and it is also the most
#: common opening in the output of a failing hook (``error: pre-commit
#: hook: .venv does not exist``). Every sentence in which git ITSELF says
#: "not found", "already exists" or "does not exist" about an object,
#: path or ref opens with ``fatal: `` or comes back from the far side as
#: ``remote: ``.
#:
#: The cost is known and accepted: ``git remote add`` says "error: remote
#: origin already exists." for a duplicate, which now falls through to
#: ``git_failed`` -- with git's own sentence attached, which is what makes
#: that acceptable. G3 owns remotes and can add a precise row for it.
GIT_MESSAGE_PREFIXES: tuple[str, ...] = ("fatal: ", "remote: ")


@dataclass(frozen=True)
class Anchored:
    """A phrase that only counts when git is the one saying it.

    Wraps a phrase so ordinary that finding it anywhere in the stream means
    nothing -- "not found", "already exists" -- and requires it to sit on a
    line opening with one of :data:`GIT_MESSAGE_PREFIXES`. The hooks a
    commit runs write to the same stderr, and their prose must not be read
    as git's.
    """

    phrase: str


#: A pattern is a string ("this text appears in stderr"), an
#: :class:`Anchored` phrase ("git itself said this"), or a tuple of strings
#: ("all of these appear"). Rows are tried in order and the first row with a
#: matching pattern wins, so a row that must beat a later one goes above it.
_Pattern = str | Anchored | tuple[str, ...]

#: stderr phrase -> (code, status). ORDER IS PART OF THE MEANING.
#:
#: The phrases are git's own, as printed under ``LC_ALL=C``; matched
#: case-insensitively (see the module docstring). Every row here is a row of
#: the classification table in the G1 plan.
_RULES: tuple[tuple[str, int, tuple[_Pattern, ...]], ...] = (
    # FIRST, and above ``network`` on purpose: an ssh failure of either kind
    # ends with "Could not read from remote repository", so the credential
    # phrases have to be looked for before that line is allowed to mean
    # "the network was down".
    ("auth_required", 409, (
        "terminal prompts disabled",
        "could not read Username",
        "could not read Password",
        "Authentication failed",
        "Permission denied (publickey",
        "Host key verification failed",
        "Invalid username or password",
        "Invalid username or token",
        "HTTP Basic: Access denied",
        "The requested URL returned error: 401",
        "The requested URL returned error: 403",
        # The split itself: the remote read failed AND a line above it says
        # the credentials are why. A PAIR, so that an ordinary local
        # "Permission denied" on a lock file is not reported as "log in".
        #
        # One pair is enough for the three phrases the split is about.
        # "publickey" only ever appears inside ssh's own "Permission denied
        # (publickey,...)" -- which this pair matches whatever order the
        # server lists its methods in, including the "(password,publickey)"
        # that the exact phrase above does not -- and "Host key" only ever
        # appears as the sentence above. A pair for each of those would be a
        # row that no input can reach.
        ("Could not read from remote repository", "Permission denied"),
    )),
    ("network", 409, (
        "Could not resolve host",
        "unable to access",
        "Connection timed out",
        "Connection refused",
        # Reached only when no credential phrase matched above.
        "Could not read from remote repository",
    )),
    ("non_fast_forward", 409, (
        ("[rejected]", "non-fast-forward"),
        ("[rejected]", "fetch first"),
        ("[rejected]", "stale info"),
    )),
    ("diverged", 409, (
        "Not possible to fast-forward",
        "Need to specify how to reconcile divergent branches",
        "Diverging branches can't be fast-forwarded",
    )),
    # ``stash pop`` reports its conflicts with the same "CONFLICT (" line
    # ``merge`` does, so it needs no row of its own.
    #
    # The last two phrases are a different moment: ``git commit`` REFUSING
    # to run while the index still holds an unmerged path. It says "error:
    # Committing is not possible because you have unmerged files." and
    # "fatal: Exiting because of an unresolved conflict." (measured on git
    # 2.53, exit 128), which share no words with the merge-time lines above
    # -- so without them the most ordinary click there is during a conflict,
    # the commit button, was a 500.
    ("conflict", 409, (
        "CONFLICT (",
        "Automatic merge failed",
        "fix conflicts and then commit",
        "you have unmerged files",
        "unresolved conflict",
    )),
    ("dirty_tree", 409, (
        "would be overwritten by",
        "Please commit your changes or stash them",
    )),
    ("no_upstream", 409, (
        "has no upstream branch",
        "There is no tracking information",
    )),
    # git says all three of these on STDOUT, not stderr. A caller that runs
    # ``commit`` therefore has to hand the stdout text in as well -- see
    # :func:`classify_failure`.
    #
    # The third is the one a repository with nothing but new files answers
    # with, and it shares no phrase with the other two ("nothing ADDED to
    # commit"): without it, the most ordinary empty commit there is -- a
    # first commit where nobody staged anything -- would be a 500.
    ("nothing_to_commit", 409, (
        "nothing to commit",
        "no changes added to commit",
        "nothing added to commit but untracked files present",
    )),
    ("identity_missing", 409, (
        "Please tell me who you are",
        "Author identity unknown",
        "empty ident name",
    )),
    ("detached_head", 409, (
        "You are not currently on a branch",
    )),
    # Above ``not_found``: "a branch named 'x' already exists" would
    # otherwise be reported as a missing one by the "not found" phrase.
    ("branch_exists", 409, (
        Anchored("already exists"),
    )),
    ("branch_not_merged", 409, (
        "is not fully merged",
    )),
    ("signing_failed", 409, (
        "gpg failed to sign",
        "error: gpg",
    )),
    # The last row before the fallback, so these can only take a failure
    # away from ``git_failed``. Every phrase was read off git 2.53 running
    # the G1 read operations against a missing path, ref or stash -- a file
    # the user opens at a ref where it does not exist is the most ordinary
    # request the tab makes, and a 500 for it would be plainly wrong.
    ("not_found", 404, (
        # Anchored: two words of ordinary English that a failing lint hook
        # prints as readily as git does ("ruff: command not found").
        Anchored("not found"),
        "did not match any",
        "unknown revision",
        "bad revision",
        "No such remote",
        "is not a valid reference",
        # cat-file blob <ref>:<path> / :0:<path>. Anchored for the same
        # reason: "does not exist" is a sentence anything can write.
        Anchored("does not exist"),
        # cat-file blob <unknown ref>:<path>
        "invalid object name",
        # rev-parse --verify <unknown sha>, without --quiet
        "Needed a single revision",
    )),
)


def _matches(pattern: _Pattern, haystack: str, lines: Sequence[str]) -> bool:
    """Does *pattern* match this stderr?

    *haystack* is the whole stream and *lines* its lines, both already
    lower-cased by the caller; the patterns are written in git's own casing
    so the table stays readable next to real output, and are lowered here.

    The lines are NOT stripped, and that is the anchor doing its job. git
    puts its own ``fatal: `` at the very start of a line; a line that opens
    with whitespace and then ``fatal:`` is something a hook printed, and
    reading it as git's would let a script's prose pick the code the user
    is shown.
    """
    if isinstance(pattern, Anchored):
        phrase = pattern.phrase.lower()
        return any(line.startswith(GIT_MESSAGE_PREFIXES) and phrase in line
                   for line in lines)  # the line's FIRST characters
    if isinstance(pattern, str):
        return pattern.lower() in haystack
    return all(part.lower() in haystack for part in pattern)


def _subcommand(argv: Sequence[str]) -> str:
    """The git subcommand in *argv*, for a message a developer can place.

    *argv* is the full command line including the fixed prefix, so this
    steps over the executable, the ``-C <dir>`` and every ``-c <setting>``
    to find the first bare word. Empty for a command line that is nothing
    but options (``git --version``).
    """
    index = 1
    while index < len(argv):
        token = argv[index]
        if token in ("-C", "-c"):
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        return token
    return ""


def stderr_tail(stderr: str, limit: int = LAST_STDERR_LINES) -> str:
    """The last *limit* lines of *stderr*, with the trailing blank ones gone.

    git puts the sentence that matters last (the ``fatal:`` line, or the
    hook's own output), so a tail is the right end to keep.
    """
    lines = stderr.rstrip().splitlines()
    return "\n".join(lines[-limit:])


#: The userinfo of any URL: everything between ``://`` and the ``@`` that
#: ends it. Whatever it holds goes, because a password is not recognisable
#: on sight -- ``https://alice:hunter2@host/`` and
#: ``https://x-access-token:ghs_...@host/`` are the same shape.
#:
#: ``@`` is deliberately NOT excluded from the class, so the match runs
#: greedily to the LAST ``@`` before a slash or a space. A URL parser splits
#: the authority at the last one too, which means an unencoded ``@`` inside
#: the password is a URL git accepts -- and a class that stopped at the
#: first one turned ``https://user:p@ss@github.com/o/r`` into
#: ``https://***@ss@github.com/o/r``: half the password, in a string that
#: still looks redacted. The class cannot cross a ``/`` or whitespace, so it
#: can never run past the authority into a path or a second URL.
_USERINFO_RE = re.compile(r"://[^/\s]*@")

#: The GitHub App installation token, which also appears OUTSIDE a URL --
#: in an ``Authorization`` line a forge echoes back, or in a hook's own
#: output. The prefix is kept so the sentence still reads.
_ACCESS_TOKEN_RE = re.compile(r"(?i)(x-access-token:)[^\s@/]+")

#: The token shapes whose prefixes their issuers publish. A word starting
#: with one of these is a credential and nothing else.
_TOKEN_WORD_RE = re.compile(
    r"(?i)\b(?:github_pat_|ghp_|gho_|ghs_|ghu_|glpat-)[A-Za-z0-9_-]+")


def redact(text: str) -> str:
    """Mask the credentials a git failure can be carrying.

    ``auth_required`` is the failure most likely to reach a screen, and it
    is also the one whose stderr can hold a working password: git names the
    remote it could not reach, and a URL somebody once pasted into ``git
    remote add`` may carry ``https://user:ghp_xxx@github.com/owner/repo``
    inside it. Unredacted, that token is then drawn in the tab, pasted into
    a bug report and left in whatever the browser logs.

    Three shapes, all applied, because one line can carry more than one:
    the userinfo of a URL, the ``x-access-token:<value>`` a GitHub App
    installation token is spelled as, and the token WORDS whose prefixes
    are published (``ghp_``, ``github_pat_``, ``gho_``, ``ghs_``, ``ghu_``,
    ``glpat-``).

    What is deliberately KEPT is everything else -- the scheme, the host,
    the path and git's own sentence. Masking the whole URL would take the
    one fact that makes the error actionable ("which remote?") with it, and
    a user who cannot see which repository failed will paste the raw
    command into a terminal instead, which is worse in every way.

    This is a last line of defence and not the first: the app stores no
    credentials, passes none, and runs git with the prompts turned off. A
    token shape nobody has published is not recognisable here, so a route
    still owes the same care to anything else it echoes back.
    """
    masked = _USERINFO_RE.sub("://***@", text)
    masked = _ACCESS_TOKEN_RE.sub(r"\1***", masked)
    return _TOKEN_WORD_RE.sub("***", masked)


def classify_failure(argv: Sequence[str], returncode: int,
                     stderr: str) -> GitError:
    """Turn one failed git command into a :class:`GitError` with a code.

    *stderr* is git's decoded error output. A caller whose command explains
    itself on STDOUT instead -- ``git commit`` prints "nothing to commit"
    there -- passes both streams joined, which is why this takes text rather
    than reading a :class:`~app.core.git.runner.GitResult`.

    A return code the caller declared acceptable never gets here: that
    decision belongs to ``run_git``'s ``ok_codes``, because only the caller
    knows that ``diff`` exits 1 for "there are differences".

    Unrecognised output is ``git_failed`` (500). Every result -- recognised
    or not -- carries the last :data:`LAST_STDERR_LINES` lines in
    ``.stderr``, because a code is a claim about what went wrong and a
    wrong claim with no evidence under it cannot be argued with. Empty
    stderr gives an empty tail, which is itself the honest answer: git
    failed and said nothing (``--quiet`` does that).

    That tail is :func:`redact`ed HERE as well as at the route. The route is
    still the place that decides what reaches a browser, but a credential
    that is never put into the exception cannot escape through a log line, a
    test fixture or the next caller somebody writes -- and redaction is
    idempotent, so doing it twice costs nothing and forgetting it once
    costs a token.
    """
    haystack = stderr.lower()
    lines = haystack.splitlines()
    tail = redact(stderr_tail(stderr))
    for code, status, patterns in _RULES:
        if any(_matches(pattern, haystack, lines) for pattern in patterns):
            return GitError(code, status, stderr=tail)
    label = f"git {_subcommand(argv)}".strip()
    return GitError(
        "git_failed", CODES["git_failed"][0],
        f"{label} failed (exit {returncode})",
        stderr=tail)
