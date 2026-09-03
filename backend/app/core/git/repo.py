"""The repository itself: make one, read its state, say who commits in it.

Everything here is a plain function of ``(root, ...)`` that runs git and
returns a model -- no service, no lock, no event loop -- so a test can call
it against a real temporary repository, and the service above can call the
same function inside ``asyncio.to_thread``.

What lives here is the questions that are about the REPOSITORY rather than
about one file or one commit's contents:

* :func:`init` and :func:`ensure_scaffold` -- turning a project directory
  into a repository, with the same ``.gitignore`` ``cdui project init``
  writes. The scaffold is not decoration: its first line is ``.env``, and a
  repository that does not ignore it will sooner or later have the user's
  API keys in its history, where no later ``.gitignore`` can reach them.
* :func:`read_status` -- one ``status --porcelain=v2`` read plus the two
  flags that format does not carry. Every other module needs it: a diff has
  to know whether a path is untracked, a discard has to know what is
  actually changed, and every mutation reads it again afterwards.
* :func:`read_identity` / :func:`write_identity` -- ``user.name`` and
  ``user.email``, and WHERE they come from. A commit made with the wrong
  name is a commit that has to be amended, so the tab shows the scope
  before it lets one be made, and only ever writes ``--local``.
* :func:`resolve_commit` / :func:`first_parent` -- turning the sha a browser
  sent into one git has verified, and finding the commit a merge should be
  compared against. Both live here rather than in ``log`` or ``diff``
  because both of those need them, and neither may import the other.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from pathlib import Path

from ..project import PROJECT_GITATTRIBUTES, PROJECT_GITIGNORE
from .errors import GitError
from .models import ConfigScope, GitStatus, Identity
from .paths import validate_identity, validate_sha
from .runner import T_LOCAL, T_STATUS, run_git
from .status import parse_porcelain_v2

#: The line ``.gitignore`` has to contain for the project's secrets to stay
#: out of history. Compared exactly: ``.env*`` also hides ``.env.example``,
#: which is meant to be committed.
ENV_IGNORE_LINE = ".env"

#: What to tell somebody whose stage / unstage / discard went through a
#: link. Worded differently from the read side's "symbolic links are not
#: served" on purpose: nothing was served, and what was refused was a WRITE
#: aimed through somebody else's directory.
LINK_PARENT_HINT = "paths through a link are not managed here"

#: The status read, in full. ``--untracked-files=all`` lists FILES rather
#: than the directories the default collapses them into, which is what lets
#: the tab offer (and ``discard`` remove) one new file out of a new folder;
#: ``--show-stash`` is the only place porcelain v2 reports the stash depth;
#: ``-z`` is what makes a filename with a newline in it safe to parse.
STATUS_ARGS: tuple[str, ...] = (
    "status", "--porcelain=v2", "--branch", "--show-stash",
    "--untracked-files=all", "-z",
)

#: The three files that say a merge or a rebase is half-finished. Asked for
#: in ONE ``rev-parse``, which answers with one line each, in this order.
IN_PROGRESS_PATHS: tuple[str, ...] = ("MERGE_HEAD", "rebase-merge",
                                      "rebase-apply")

#: ``\`` before any character in a C-quoted config origin. git quotes an
#: origin that contains a backslash -- which every absolute Windows path
#: does -- as ``file:"C:\\Users\\me\\.gitconfig"``.
_ESCAPE_RE = re.compile(r"\\(.)")


# --- becoming a repository -------------------------------------------------


def init(root: Path) -> None:
    """``git init`` in *root*. The caller has already decided *root* may be.

    Only ever the project directory (the service checks that): a stray
    ``git init`` one level up would make the tab operate on a repository
    containing the user's whole home directory.

    Re-running this on a directory that is ALREADY a repository is safe and
    deliberate -- git re-initialises it, which changes nothing about the
    history, the index or the working tree -- so a stale UI that offers the
    button twice cannot destroy anything.
    """
    if not root.is_dir():
        raise GitError("not_found", 404,
                       f"the project directory {root} does not exist",
                       hint="open a project directory that exists")
    run_git(["init", "-q"], cwd=root, timeout=T_LOCAL)


def ensure_scaffold(root: Path) -> list[str]:
    """Give *root* the ``.gitignore`` / ``.gitattributes`` a project needs.

    Returns the names of the files this changed, which is what the tab shows
    after an init ("created .gitignore").

    Never rewrites a file the user already has. A missing ``.gitignore`` is
    written from :data:`~app.core.project.PROJECT_GITIGNORE`; an existing one
    that does not ignore ``.env`` gets that one line appended, and nothing
    else. An existing ``.gitattributes`` is left exactly as it is: its rules
    change how git stores every file in the repository, and appending to
    somebody's own line-ending policy would be a worse bug than the one the
    default line prevents.

    Appending starts with a newline, so a file whose last line has no
    terminator does not become ``*.pklo.env``. A blank line in a
    ``.gitignore`` is ignored by git, which is why that is the safe way
    round. The bytes already in the file are never re-encoded -- the append
    is binary -- so a ``.gitignore`` saved as anything other than UTF-8
    survives it.

    An explicit ``!.env`` (an un-ignore rule) is not treated as the line
    being present, so the appended rule wins: this file holds the user's API
    keys, and the tab is not the place where "commit my secrets" is honoured
    by accident.
    """
    written: list[str] = []

    gitignore = root / ".gitignore"
    if not gitignore.exists():
        gitignore.write_bytes(PROJECT_GITIGNORE.encode("utf-8"))
        written.append(".gitignore")
    elif not _ignores_env(gitignore):
        with gitignore.open("ab") as handle:
            handle.write(b"\n" + ENV_IGNORE_LINE.encode("ascii") + b"\n")
        written.append(".gitignore")

    attributes = root / ".gitattributes"
    if not attributes.exists():
        attributes.write_bytes(PROJECT_GITATTRIBUTES.encode("utf-8"))
        written.append(".gitattributes")

    return written


def _ignores_env(path: Path) -> bool:
    """Would this ``.gitignore`` already keep a FILE called ``.env`` out?

    The question is git's, not the author's. Two spellings answer yes --
    ``.env`` and ``/.env``, either of them with trailing SPACES, which git
    drops before matching -- and reading only the first made
    :func:`ensure_scaffold` append a second ``.env`` under a line that
    plainly already did the job.

    Everything else answers NO, including the near-misses that look like
    they should count. ``.env/`` is a directory-only pattern and does not
    match a file; ``.env  # secrets`` is not a rule with a comment on it,
    because ``.gitignore`` has no trailing comments -- it is a pattern
    matching a file with that entire name; ``  .env`` keeps its leading
    spaces, so it matches a file whose name starts with two of them; and
    ``.env\\t`` keeps its trailing TAB, because spaces are the only trailing
    whitespace git drops (measured: ``git status`` lists ``.env`` as
    untracked against every one of them). Whoever wrote any of them
    believes their secrets are ignored and they are not, so the line goes
    in. A duplicate rule is harmless; a missing one is the leak this whole
    scaffold exists to prevent, and that asymmetry is what decides every
    uncertain case here.

    An un-ignore rule is the same call for a different reason: ``!.env``
    does not count, the appended rule goes in after it, and the last
    matching rule is the one git obeys. This file holds the user's API keys,
    and the tab is not the place where "commit my secrets" is honoured by
    accident.

    Decoded with ``errors="replace"`` because this only ever asks a question
    about ASCII: a ``.gitignore`` in some other encoding must not make this
    raise, and the bytes are never written back.
    """
    try:
        text = path.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return False
    return any(_ignore_rule(line) == ENV_IGNORE_LINE
               for line in text.splitlines())


def _ignore_rule(line: str) -> str:
    """The file one ``.gitignore`` line matches, as far as this check goes.

    Not a gitignore parser and not trying to be -- no globs, no negation
    semantics, no directory matching. It removes the only two things git
    itself removes from a line before matching it against a file at the top
    level: trailing SPACES, which git drops unless they are escaped, and
    the leading ``/`` that anchors a rule to the repository root, which for
    a root-level file changes nothing about what it matches.

    SPACES ONLY, and that is the whole reason this is a function rather
    than a ``strip()``. ``str.rstrip()`` with no argument also eats a TAB
    and a non-breaking space, and git keeps both: measured on git 2.53,
    ``.env\\t``, ``.env\\t ``, ``.env\\xa0`` and ``/.env\\t`` all leave the
    file listed as ``?? .env``. Reading one of those as the rule already
    being present would leave the working line unwritten -- the leak
    direction.

    LEADING whitespace is NOT removed either, for the same reason: a line
    reading ``  .env`` is a pattern for a file whose name begins with two
    spaces, and a ``.gitignore`` saying that does not ignore ``.env`` at
    all.

    Everything else is left exactly as written, so it simply does not
    compare equal: a glob (``*.env``), a directory-only rule (``.env/``), a
    line with what its author took for a comment on the end (``.env  #
    secrets`` -- git has no trailing comments, so that pattern matches a
    file with that whole name), and an un-ignore rule (``!.env``). Every one
    of those makes the scaffold append its own line, which is the safe
    direction to be wrong in: a duplicate rule does nothing, and a missing
    one is the user's API keys in a public repository.
    """
    text = line.rstrip(" ")
    if text.startswith("/"):
        text = text[1:]
    return text


# --- a path that does not stay where it says it is --------------------------


def path_redirects(root: Path, segments: Sequence[str]) -> bool:
    """Does *root* joined with *segments* land somewhere other than it says?

    The check that catches a Windows JUNCTION -- the redirection ``mklink
    /J`` makes, which ``Path.is_symlink`` answers False for and which git
    walks through like any other directory. Resolving both sides and
    comparing is what sees one: the lexical path is what the request NAMED,
    the resolved path is where the filesystem actually goes, and a
    difference between them means something in the chain redirected.

    The root's own links cancel out, because both sides start from
    ``root.resolve()``: a project directory that is itself reached through a
    link -- ``/tmp`` on macOS is one -- does not make every path in it a
    refusal. Compared with ``normcase``, because ``resolve`` hands back the
    case the filesystem stores and the request carries the case the user
    typed.

    Takes SEGMENTS rather than a path so that the two callers can ask about
    different parts of the same path: the read side asks about all of it
    (``diff._refuse_a_link``), the write side about its parents only
    (:func:`refuse_link_parents`). The comparison itself is subtle enough
    that it must exist once.
    """
    lexical = root.resolve().joinpath(*segments)
    actual = root.joinpath(*segments).resolve()
    return os.path.normcase(str(actual)) != os.path.normcase(str(lexical))


def refuse_link_parents(root: Path, path: str) -> None:
    """Refuse a WRITE whose path goes through a link or a junction.

    Measured, and the reason this exists: with ``notes`` a junction to a
    gitignored ``secrets`` folder, discarding ``notes/keys.txt`` deleted the
    real ``secrets/keys.txt``. Every check before this one passed and was
    right to -- the path is relative, holds no ``..``, and RESOLVES inside
    the project, which is exactly what a link to another folder of the same
    project does. git then cleaned the file at the other end of it.

    The FINAL component may be a link, and that is not an oversight. To git
    a symlink is an ordinary tracked file (a blob holding the path it points
    at), so staging one, or restoring one somebody deleted, is a legitimate
    operation, and refusing it would leave the tab unable to manage a
    repository that contains one. What is refused is a link in the MIDDLE of
    a path, where the redirection is not the thing being operated on but the
    way to something else.

    Both checks, for the same reason the read side runs both: ``is_symlink``
    sees a symbolic link at any depth and cannot see a junction;
    :func:`path_redirects` sees either.
    """
    parents = path.split("/")[:-1]
    current = root
    for segment in parents:
        current = current / segment
        if current.is_symlink():
            raise GitError("invalid_path", 400,
                           f"{path} goes through a symbolic link",
                           hint=LINK_PARENT_HINT)
    if path_redirects(root, parents):
        raise GitError("invalid_path", 400,
                       f"{path} does not stay where it says it is",
                       hint=LINK_PARENT_HINT)


# --- reading the state -----------------------------------------------------


def read_status(root: Path) -> GitStatus:
    """One full status read of *root*, parsed.

    Two commands, because porcelain v2 does not say whether a merge or a
    rebase is half-finished and the tab has to disable half its buttons when
    one is. ``rev-parse --git-path`` answers all three questions at once and
    costs one process; whether each path EXISTS is then a filesystem
    question, not git's.

    Read-only, so it runs without taking the index lock: this is polled
    while the tab is open, and a status that occasionally lags is better
    than a status that fights a real operation for the lock.
    """
    result = run_git(list(STATUS_ARGS), cwd=root, timeout=T_STATUS,
                     read_only=True)
    merge, rebase = _in_progress(root)
    return parse_porcelain_v2(result.stdout, merge_in_progress=merge,
                              rebase_in_progress=rebase)


def _in_progress(root: Path) -> tuple[bool, bool]:
    """``(merge, rebase)`` -- is one of them half-finished in *root*?

    ``--git-path`` prints paths RELATIVE to the working tree it was run in
    (``.git/MERGE_HEAD``), and absolute ones when the git directory is
    somewhere else, and ``root / line`` is correct for both: joining an
    absolute path replaces the root.

    Fewer than three lines means a git that answered something this does not
    understand. "No merge in progress" is the safe answer to that -- it
    leaves the buttons enabled and lets git itself refuse -- and a status
    read must not fail over a flag.
    """
    result = run_git(["rev-parse", *(arg for path in IN_PROGRESS_PATHS
                                     for arg in ("--git-path", path))],
                     cwd=root, timeout=T_STATUS, read_only=True)
    lines = result.out.splitlines()
    if len(lines) < len(IN_PROGRESS_PATHS):
        return False, False
    merge_head, rebase_merge, rebase_apply = (root / line for line in lines[:3])
    return merge_head.exists(), rebase_merge.exists() or rebase_apply.exists()


# --- the index -------------------------------------------------------------

#: The index mode of a gitlink -- a submodule's entry in its parent. Not a
#: file: what is at that path is another repository, and every operation
#: this tab offers would be about the wrong one.
GITLINK_MODE = "160000"


def index_entries(root: Path, paths: Sequence[str]) -> list[tuple[str, str]]:
    """``(mode, path)`` for every index entry matching *paths*.

    One reader for the two questions the index answers that no other command
    does: which paths are SUBMODULES (mode :data:`GITLINK_MODE`, which the
    status parser does not carry), and which names a pathspec actually
    matches -- ``sub`` matching ``sub/.env`` is how a request for one file
    becomes a diff of a whole directory.

    An UNMERGED path appears once per stage, all three with the same name;
    the caller that cares about "exactly this path and nothing else" should
    compare a SET of the names for that reason.
    """
    result = run_git(["ls-files", "--stage", "-z", "--", *paths], cwd=root,
                     timeout=T_STATUS, read_only=True)
    entries: list[tuple[str, str]] = []
    for line in result.out.split("\x00"):
        if not line:
            continue
        # ``<mode> <sha> <stage>\t<path>``
        head, _, path = line.partition("\t")
        mode = head.split(" ", 1)[0]
        if path:
            entries.append((mode, path))
    return entries


def submodule_paths(root: Path, paths: Sequence[str]) -> list[str]:
    """Which of *paths* the index holds as submodules.

    Asked of the INDEX rather than of the disk, because that is where the
    answer is unambiguous: mode 160000 is a gitlink and nothing else is.
    """
    return [path for mode, path in index_entries(root, paths)
            if mode == GITLINK_MODE]


# --- who commits -----------------------------------------------------------


def read_identity(root: Path) -> Identity:
    """``user.name`` / ``user.email`` as this repository sees them.

    Two reads rather than one ``--list``, because the answer wanted is the
    EFFECTIVE value of each -- what a commit made right now would carry --
    and that is what ``--get`` returns for each key on its own.
    """
    name, name_scope = _config_value(root, "user.name")
    email, email_scope = _config_value(root, "user.email")
    return Identity(name=name, email=email, name_scope=name_scope,
                    email_scope=email_scope)


def _config_value(root: Path, key: str) -> tuple[str | None, ConfigScope | None]:
    """One config value and where it came from; ``(None, None)`` when unset.

    ``git config --get`` exits 1 for "no such key", which is an answer and
    not a failure -- hence ``ok_codes``.
    """
    result = run_git(["config", "--show-origin", "--get", key], cwd=root,
                     timeout=T_STATUS, ok_codes=(0, 1), read_only=True)
    if result.returncode != 0:
        return None, None
    origin, _, value = result.out.rstrip("\n").partition("\t")
    text = value.strip()
    if not text:
        return None, None
    return text, _scope_from_origin(origin)


def _scope_from_origin(origin: str) -> ConfigScope:
    """Which config file git read a value from, as the three scopes.

    git says this as a path, not a scope (``--show-scope`` would say it
    directly, and needs a git newer than the one this tab supports), so the
    path is what has to be read:

    * ``file:.git/config`` -- relative, because git prints it relative to
      the working tree -- or any absolute path ending the same way is THIS
      repository: ``local``.
    * a ``.gitconfig`` anywhere, or an XDG ``.config/git/config``, is the
      user's own: ``global``.
    * anything else -- ``/etc/gitconfig``, a path some ``GIT_CONFIG_GLOBAL``
      invented, ``command line:``, a config blob -- is reported as
      ``system``, the scope that means "not yours to edit here". The tab
      only ever WRITES local, so a wrong guess here costs a label and
      nothing else.
    """
    path = _origin_path(origin)
    if path is None:
        return "system"
    if path.endswith(".git/config"):
        return "local"
    if ".gitconfig" in path or path.endswith("/.config/git/config"):
        return "global"
    return "system"


def _origin_path(origin: str) -> str | None:
    """The filename out of a ``file:<path>`` origin, POSIX and lower-cased.

    None for an origin that is not a file (``command line:``, ``blob:...``).
    An origin containing a backslash arrives C-quoted --
    ``file:"C:\\\\Users\\\\me\\\\.gitconfig"`` -- so the quotes come off and
    every ``\\x`` collapses to ``x``, which is right for the ``\\\\`` and
    ``\\"`` git actually emits.
    """
    if not origin.startswith("file:"):
        return None
    raw = origin[len("file:"):]
    if len(raw) >= 2 and raw.startswith('"') and raw.endswith('"'):
        raw = _ESCAPE_RE.sub(r"\1", raw[1:-1])
    return raw.replace("\\", "/").lower()


def write_identity(root: Path, name: str | None = None,
                   email: str | None = None) -> dict[str, str]:
    """Set ``user.name`` / ``user.email`` in THIS repository's config.

    Returns what was written, which is what the tab echoes back. Either half
    may be left out; a request with neither writes nothing, which is a no-op
    and not an error.

    ``--local`` always. Writing the user's global config from a web request
    would change every repository on the machine, including ones this app
    has never been pointed at.

    The value is a positional argument to ``git config <name> <value>``, so
    it cannot be read as an option whatever it contains -- but it is
    validated anyway (no leading ``-``, no control characters, an ``@`` in
    the email), because a name with a newline in it would forge a second
    header line in every commit object it signs.
    """
    clean_name, clean_email = validate_identity(name, email)
    written: dict[str, str] = {}
    if clean_name is not None:
        run_git(["config", "--local", "user.name", clean_name], cwd=root,
                timeout=T_LOCAL)
        written["name"] = clean_name
    if clean_email is not None:
        run_git(["config", "--local", "user.email", clean_email], cwd=root,
                timeout=T_LOCAL)
        written["email"] = clean_email
    return written


# --- commits ---------------------------------------------------------------


def resolve_commit(root: Path, sha: str) -> str:
    """The full sha of the commit *sha* names. 404 when there is no such one.

    Shape first (:func:`~app.core.git.paths.validate_sha`), so a ref like
    ``HEAD~3`` or ``--output=/etc/passwd`` never reaches a command line, and
    then ``rev-parse --verify --quiet <sha>^{commit}``, which answers with
    the full id and exits 1 -- silently, with no stderr to classify -- for
    an id no object has. ``^{commit}`` is what makes a tag or a tree name
    fail here rather than three commands later.
    """
    clean = validate_sha(sha)
    result = run_git(["rev-parse", "--verify", "--quiet", f"{clean}^{{commit}}"],
                     cwd=root, timeout=T_STATUS, ok_codes=(0, 1),
                     read_only=True)
    full = result.out.strip()
    if result.returncode != 0 or not full:
        raise GitError("not_found", 404, f"no commit {clean} in this repository",
                       hint="the history may have been rewritten; reload the log")
    return full


def first_parent(root: Path, commit: str) -> str | None:
    """The sha of *commit*'s first parent, or None for a root commit.

    What a merge commit should be compared against. git's own
    ``--first-parent`` does NOT restrict ``diff-tree`` to it (measured on
    2.53: with ``-m`` it prints one diff per parent, concatenated), and the
    option that would -- ``--diff-merges=first-parent`` -- is newer than the
    git this tab supports. Asking for the parent and diffing the two trees
    is one extra process and is exactly right on every version.
    """
    result = run_git(["rev-parse", "--verify", "--quiet", f"{commit}^"],
                     cwd=root, timeout=T_STATUS, ok_codes=(0, 1),
                     read_only=True)
    parent = result.out.strip()
    return parent if result.returncode == 0 and parent else None
