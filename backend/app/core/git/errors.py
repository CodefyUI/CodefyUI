"""What a git command failed at, in terms the Source Control tab can act on.

git says everything in prose, on stderr, in English -- and the person
reading the editor may not read English at all. So nothing here returns a
sentence to display: every failure carries a ``code`` from a closed
vocabulary (:data:`CODES`), and the frontend owns the wording. The
``message`` is for logs and for a developer reading a 500; ``hint`` is the
one extra fact a code cannot carry (which branch, which file); ``stderr``
is the raw tail, and only the "we do not know what this is" code sets it.

Three decisions here are not stylistic:

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

Matching is case-insensitive substring matching, deliberately: git's own
wording is stable under ``LC_ALL=C``, but the ``remote:`` lines a forge
prints through it are not, and neither is the capitalisation of the
``fatal:``/``error:`` prefix across versions.

Stdlib only, and imported by ``git/__init__.py``: catching a git failure
never drags the subprocess machinery in with it.
"""

from __future__ import annotations

from collections.abc import Sequence

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
    and ``stderr`` the raw tail of git's own output -- set only for
    ``git_failed``, where there is nothing else to go on.
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


#: A pattern is either a string ("this text appears in stderr") or a tuple of
#: strings ("all of these appear"). Rows are tried in order and the first row
#: with a matching pattern wins, so a row that must beat a later one goes
#: above it.
_Pattern = str | tuple[str, ...]

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
    ("conflict", 409, (
        "CONFLICT (",
        "Automatic merge failed",
        "fix conflicts and then commit",
    )),
    ("dirty_tree", 409, (
        "would be overwritten by",
        "Please commit your changes or stash them",
    )),
    ("no_upstream", 409, (
        "has no upstream branch",
        "There is no tracking information",
    )),
    # git says both of these on STDOUT, not stderr. A caller that runs
    # ``commit`` therefore has to hand the stdout text in as well -- see
    # :func:`classify_failure`.
    ("nothing_to_commit", 409, (
        "nothing to commit",
        "no changes added to commit",
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
        "already exists",
    )),
    ("branch_not_merged", 409, (
        "is not fully merged",
    )),
    ("signing_failed", 409, (
        "gpg failed to sign",
        "error: gpg",
    )),
    ("not_found", 404, (
        "not found",
        "did not match any",
        "unknown revision",
        "bad revision",
        "No such remote",
        "is not a valid reference",
    )),
)


def _matches(pattern: _Pattern, haystack: str) -> bool:
    """Is *pattern* (a phrase, or all of a tuple of phrases) in *haystack*?

    *haystack* is already lower-cased by the caller; the patterns are
    written in git's own casing so the table stays readable next to real
    output, and are lowered here.
    """
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

    Unrecognised output is ``git_failed`` (500) carrying the last
    :data:`LAST_STDERR_LINES` lines -- the one code that hands the raw text
    on, because there is nothing else to say about it.
    """
    haystack = stderr.lower()
    for code, status, patterns in _RULES:
        if any(_matches(pattern, haystack) for pattern in patterns):
            return GitError(code, status)
    label = f"git {_subcommand(argv)}".strip()
    return GitError(
        "git_failed", CODES["git_failed"][0],
        f"{label} failed (exit {returncode})",
        stderr=stderr_tail(stderr))
