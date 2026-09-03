"""Closed grammars for everything the browser is allowed to name.

The Source Control tab sends paths, branch names, remote names, URLs and
messages, and every one of them ends up on a git command line. git's command
line has no quoting problem to exploit -- there is no shell here -- but it
has something subtler: an argument that begins with ``-`` is an OPTION, and
several git options do more than they look like. ``--upload-pack=`` runs a
command on a fetch. So the rule these functions enforce is not "escape it",
it is "it has to look like the thing it claims to be, or it does not go".

Closed grammars, not blocklists: each validator says what IS allowed and
refuses the rest. A blocklist of dangerous prefixes has to be updated every
time git grows an option; ``^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`` does not.

Three of them are about more than syntax:

* **paths are confined to the repository.** ``..`` and an absolute path are
  refused by shape, and then the resolved path is checked against the
  resolved root -- which is the check that also catches a symlink inside the
  repository pointing out of it. Backslashes are refused outright rather
  than translated: the frontend sends POSIX paths because git speaks POSIX
  paths, and accepting ``a\b`` would mean guessing whether that is a
  Windows separator or a filename a Linux user is entitled to create. A
  colon is refused anywhere, for the reason a drive letter is: on Windows it
  opens an alternate data stream instead of the file it appears to name.
* **an empty path list is refused**, because ``git add -A --`` with no
  pathspec stages the WHOLE tree. "Nothing selected" must never be able to
  arrive as "everything".
* **``.env`` is not a path like the others.** :func:`is_env_secret_path`
  exists so the file and diff reads -- which are open GETs -- can refuse it
  at any ref: a secret that was committed once stays in history, and
  ``.gitignore`` does not un-commit it.

Deliberately subprocess-free. Branch names get a regex pre-check here and
the real ``git check-ref-format`` in the service, so this module stays
importable, instant and testable without a git on PATH.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import NoReturn

from .errors import GitError

#: How many paths one request may name. A stage-all of a big working tree
#: goes through the ``.`` form instead, so this is a cap on an explicit
#: SELECTION -- past a few hundred, the user meant "all".
MAX_PATHS = 500

#: A commit message long enough for anything (git itself has no limit), and
#: short enough that it cannot be used to fill a disk one commit at a time.
MAX_COMMIT_MESSAGE = 10_000

#: A stash message is one line in ``git stash list``.
MAX_STASH_MESSAGE = 500

#: Names on both sides of ``Author: <name> <email>``.
MAX_IDENTITY = 255

#: Refs and URLs. Both are far past anything real -- a filesystem gives up
#: on a ref name well before 255 -- and exist so a validator cannot be
#: handed a megabyte to match against.
MAX_BRANCH_NAME = 255
MAX_REMOTE_URL = 2048

#: A branch name, pre-check only: no leading whitespace / ``@`` / ``-``, and
#: no whitespace or control character anywhere. ``git check-ref-format``
#: (the service) is the authority on the rest -- trailing ``.lock``, ``..``,
#: ``~``, ``^``, ``:`` -- and this exists so an obviously wrong name is a
#: 400 with a reason rather than a subprocess. Exported because Task 3's
#: service and its tests need the same rule.
BRANCH_NAME_RE = re.compile(r"^[^\s@-][^\s\x00-\x1f]*$")

#: A remote name. git allows more than this; the tab does not need it.
REMOTE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

#: ``git@github.com:owner/repo.git`` -- the scp-like form, which has no
#: scheme and so has to be recognised by shape.
SCP_URL_RE = re.compile(r"^[A-Za-z0-9._-]+@[A-Za-z0-9.-]+:[^\s]+$")

#: The schemes ``GIT_ALLOW_PROTOCOL`` lets through, spelled out again here
#: so a bad URL is a 400 with an explanation rather than a git failure.
ALLOWED_URL_SCHEMES = ("https://", "ssh://", "file://")

#: Transport helpers whose "URL" is a command line git will run
#: (``ext::sh -c ...``). Refused by name as well as by the scheme
#: allowlist above, because this one is worth failing loudly.
REFUSED_URL_SCHEMES = ("ext::", "fd::")

#: A full or abbreviated commit id. Verified for real with
#: ``rev-parse --verify`` in the service; this is the shape check.
SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")

#: A drive letter, i.e. an absolute Windows path in disguise: ``C:/secrets``
#: is not relative to anything.
_DRIVE_RE = re.compile(r"^[A-Za-z]:")

#: C0 controls and DEL. NUL is the separator git's ``-z`` output uses, and
#: the rest have no business in a name that will be parsed back out of it.
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")

_WHITESPACE_RE = re.compile(r"\s")


def _refuse(code: str, message: str, hint: str | None = None) -> NoReturn:
    """Refuse a value: always a 400, always with a code the frontend knows."""
    raise GitError(code, 400, message, hint=hint)


def validate_rel_path(root: Path, path: str) -> str:
    """Check *path* is a plain relative path inside *root*; return it normalised.

    The returned string is POSIX, without ``./`` segments, doubled slashes
    or a trailing slash -- so two spellings of the same file cannot be two
    entries in a request, and what goes on the command line is what the
    status output would have called it.

    The containment check resolves both sides, which is what makes it hold
    for a symlink as well as for ``..``: a link inside the repository that
    points outside it resolves outside the root and is refused here rather
    than serving whatever it points at.
    """
    if not path:
        _refuse("invalid_path", "a path is required")
    if _CONTROL_RE.search(path):
        _refuse("invalid_path", "a path may not contain control characters")
    if "\\" in path:
        _refuse("invalid_path", "a path must use / as its separator")
    if path.startswith("/") or _DRIVE_RE.match(path):
        _refuse("invalid_path", "a path must be relative to the project")
    # ANY colon, not only a drive letter. On Windows ``ab:c.txt`` names the
    # ALTERNATE DATA STREAM ``c.txt`` of the file ``ab`` -- a second, hidden
    # fork of a file, which is not the file it appears to name and is not
    # what a diff of ``ab:c.txt`` would show. git cannot check such a name
    # out on Windows either, so no path git ever prints in a status contains
    # one, and a request naming it is asking for something else.
    if ":" in path:
        _refuse("invalid_path", "a path may not contain ':'")

    pure = PurePosixPath(path)
    if ".." in pure.parts:
        _refuse("invalid_path", "a path may not contain '..'")
    if not pure.parts:
        _refuse("invalid_path", "a path is required")

    normalised = pure.as_posix()
    # After normalisation, so that "./-rf" is refused for the same reason
    # "-rf" is: git would read it as an option, not a file.
    if normalised.startswith("-"):
        _refuse("invalid_path", "a path may not start with '-'")

    root_resolved = Path(os.path.normcase(str(root.resolve())))
    resolved = Path(os.path.normcase(str((root / normalised).resolve())))
    if resolved == root_resolved or not resolved.is_relative_to(root_resolved):
        _refuse("invalid_path", "a path must stay inside the project")
    return normalised


def validate_rel_paths(root: Path, paths: Sequence[str]) -> list[str]:
    """Validate a whole request's worth of paths; return them normalised.

    An EMPTY list is refused, and that is the point of the function existing
    beside :func:`validate_rel_path`: the commands these paths go to
    (``add -A --``, ``restore --worktree --``, ``clean -f --``) treat "no
    pathspec" as "the entire working tree", so a UI bug that sent an empty
    selection would discard every change in the project.

    The count is checked before the paths are, so an absurd request costs
    one comparison rather than 50,000 filesystem resolutions.
    """
    if not paths:
        _refuse("invalid_path", "no paths were given")
    if len(paths) > MAX_PATHS:
        _refuse("invalid_path", f"at most {MAX_PATHS} paths per request")
    return [validate_rel_path(root, path) for path in paths]


def validate_branch_name(name: str) -> str:
    """Check *name* looks like a branch name. Returns it unchanged.

    A pre-check only -- ``git check-ref-format refs/heads/<name>`` in the
    service is what actually decides -- so this refuses the cases that must
    never reach a command line at all: a leading ``-`` (an option), and
    ``@{`` (the reflog syntax, which would make ``main@{yesterday}`` a
    perfectly valid ref that is not the branch the user named).
    """
    if not name:
        _refuse("invalid_ref", "a branch name is required")
    if len(name) > MAX_BRANCH_NAME:
        _refuse("invalid_ref", f"a branch name may not exceed {MAX_BRANCH_NAME} characters")
    if "@{" in name:
        _refuse("invalid_ref", "a branch name may not contain '@{'")
    if not BRANCH_NAME_RE.match(name):
        _refuse("invalid_ref", "a branch name may not contain spaces or start with '-' or '@'")
    return name


def validate_remote_name(name: str) -> str:
    """Check *name* is a remote name (``origin``, ``upstream``, ``fork-2``)."""
    if not name:
        _refuse("invalid_value", "a remote name is required")
    if not REMOTE_NAME_RE.match(name):
        _refuse("invalid_value",
                "a remote name may only contain letters, digits, '.', '_' and '-'")
    return name


def validate_remote_url(url: str) -> str:
    """Check *url* is a remote this server is willing to hand to git.

    Whitespace is refused rather than trimmed: a URL with a space in it is
    either a paste accident the user should see, or two arguments trying to
    look like one. The scheme allowlist mirrors ``GIT_ALLOW_PROTOCOL``, so
    a URL that would fail inside git fails here instead -- with a reason.
    """
    if not url:
        _refuse("invalid_url", "a remote URL is required")
    if len(url) > MAX_REMOTE_URL:
        _refuse("invalid_url", f"a remote URL may not exceed {MAX_REMOTE_URL} characters")
    if _WHITESPACE_RE.search(url) or _CONTROL_RE.search(url):
        _refuse("invalid_url", "a remote URL may not contain whitespace")
    if url.startswith("-"):
        _refuse("invalid_url", "a remote URL may not start with '-'")

    lowered = url.lower()
    if lowered.startswith(REFUSED_URL_SCHEMES):
        _refuse("invalid_url", "that transport runs a command and is not allowed",
                hint="use an https://, ssh:// or file:// URL")
    if lowered.startswith(ALLOWED_URL_SCHEMES) or SCP_URL_RE.match(url):
        return url
    _refuse("invalid_url", "a remote URL must be https://, ssh://, file:// or user@host:path")


def validate_commit_message(message: str) -> str:
    """Check *message* is a usable commit message; return it stripped.

    Newlines and tabs stay -- a commit message is a subject, a blank line
    and a body, and this is the one value here that is meant to be several
    lines. Only NUL is refused, because it would truncate the message on its
    way through stdin.
    """
    if "\x00" in message:
        _refuse("invalid_value", "a commit message may not contain NUL")
    text = message.strip()
    if not text:
        _refuse("invalid_value", "a commit message is required")
    if len(text) > MAX_COMMIT_MESSAGE:
        _refuse("invalid_value",
                f"a commit message may not exceed {MAX_COMMIT_MESSAGE} characters")
    return text


def validate_stash_message(message: str) -> str:
    """Check *message* is a one-line stash description; return it stripped.

    Unlike a commit message this one IS a single line: it goes in
    ``--message=<m>`` and comes back as one row of ``git stash list``, so a
    newline in it would make the list unreadable.
    """
    text = message.strip()
    if not text:
        _refuse("invalid_value", "a stash message is required")
    if len(text) > MAX_STASH_MESSAGE:
        _refuse("invalid_value",
                f"a stash message may not exceed {MAX_STASH_MESSAGE} characters")
    if _CONTROL_RE.search(text):
        _refuse("invalid_value", "a stash message must be a single line")
    return text


def _identity_field(value: str | None, label: str, *,
                    require_at: bool = False) -> str | None:
    """One half of ``user.name`` / ``user.email``, stripped, or None.

    None passes through so a request that sets only one of the two does not
    have to invent the other.
    """
    if value is None:
        return None
    text = value.strip()
    if not text:
        _refuse("invalid_value", f"a {label} is required")
    if len(text) > MAX_IDENTITY:
        _refuse("invalid_value", f"a {label} may not exceed {MAX_IDENTITY} characters")
    if _CONTROL_RE.search(text):
        _refuse("invalid_value", f"a {label} may not contain control characters")
    if text.startswith("-"):
        _refuse("invalid_value", f"a {label} may not start with '-'")
    if require_at and "@" not in text:
        _refuse("invalid_value", "an email address must contain '@'")
    return text


def validate_identity(name: str | None = None,
                      email: str | None = None) -> tuple[str | None, str | None]:
    """Check the identity a commit would be made with; return it stripped.

    Both halves are optional -- the route sets whichever was sent -- but
    neither may be blank, contain a newline (it would forge a second header
    line in the commit object) or start with ``-``.
    """
    return (_identity_field(name, "name"),
            _identity_field(email, "email", require_at=True))


def validate_sha(sha: str) -> str:
    """Check *sha* is an abbreviated-or-full commit id; return it lower-cased.

    Shape only. Whether the object exists is
    ``rev-parse --verify --quiet <sha>^{commit}``'s answer, in the service,
    because that is the only place that knows the repository.
    """
    text = sha.strip().lower()
    if not SHA_RE.match(text):
        _refuse("invalid_ref", "a commit id must be 7 to 40 hexadecimal characters")
    return text


def is_env_secret_path(path: str | os.PathLike[str]) -> bool:
    """Is *path* a dotenv file -- i.e. a file that probably holds secrets?

    ``.env`` and ``.env.<anything>`` are, ``.env.example`` is not (it exists
    to be read), and neither is ``env.txt``. Only the final segment counts,
    so ``config/.env`` is caught as well as ``.env``.

    Compared case-insensitively because the filesystems this runs on mostly
    are: on Windows, ``.ENV`` and ``.env`` are one file, and a check that
    only knew the lowercase spelling would serve it.
    """
    name = re.split(r"[\\/]", str(path))[-1].lower()
    if name == ".env":
        return True
    return name.startswith(".env.") and name != ".env.example"
