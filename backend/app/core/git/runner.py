"""The only place in the app that starts a git process.

Everything the Source Control tab does ends up here, and it is one function
on purpose: a second call site is a second chance to forget one of the five
things below, and every one of them fails in a way that is hard to trace
back to the command that caused it.

* **Never a shell, and never user input as an option.** Every argument is a
  list element, every path and ref is validated against a closed grammar in
  ``paths.py``, and the caller puts them after ``--`` wherever git accepts
  it. A branch called ``--upload-pack=rm -rf /`` is then a branch name that
  does not exist, not a command.
* **Never interactive.** A server has no terminal, so a git that decides to
  ask for a password does not get an answer -- it gets a request that never
  returns and a worker that never comes back. :data:`NON_INTERACTIVE_ENV`
  plus ``-c core.askPass=`` turn every prompt into an immediate failure,
  which the classifier then reports as ``auth_required``. ``GIT_EDITOR=:``
  does the same for the editor a merge or a commit would otherwise open.
* **Binary pipes.** Diff output is bytes: it carries the file's own CRLF
  line endings, and on a repository with mixed encodings it is not valid
  UTF-8 at all. ``text=True`` would rewrite the first and crash on the
  second, so the pipes stay binary and decoding is
  :attr:`GitResult.out` / :attr:`GitResult.err`'s job -- utf-8 with
  ``errors="replace"``, which is what ``core/project.py`` has always done
  for git output on a cp950 machine.
* **A killable process tree.** ``creation_flags()`` gives the child its own
  Windows process group so that a timeout can reach ``git`` AND the
  ``ssh``/``curl`` it spawned; ``stop_process()`` is the same kill the
  Package Center uses. A git that hangs on a network read holds an index
  lock, so leaving one behind breaks the NEXT request too.
* **An environment scrubbed of the caller's repository.** ``GIT_DIR`` and
  its five friends (:data:`REPOSITORY_ENV_VARS`) name a repository; if the
  server was started from a shell inside a git hook, every command here
  would silently operate on THAT repository instead of the project. The
  base is ``packs.runner.pip_env()``, which already drops the variables
  that relocate a Python interpreter.

``git_executable``, ``git_version`` and the ``core.sshCommand`` probe cache,
and the shape of the caching is deliberate: a POSITIVE answer lasts for the
life of the process (git does not move), a NEGATIVE one for
:data:`MISSING_RECHECK_S` seconds, so a machine without git -- or with a git
whose config cannot be read -- does not spawn a process on every poll of the
status endpoint, and still notices a repair a minute later without a
restart. ``_reset_for_tests`` exists because that cache would otherwise make
the second test in a file depend on the first.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Container, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..packs.runner import creation_flags, pip_env, stop_process
from .errors import GitError, classify_failure

#: The options every call carries, between the executable and the caller's
#: arguments (the ``-C <cwd>`` that precedes them is per call).
#:
#: ``core.quotepath=false`` stops git octal-escaping non-ASCII filenames, so
#: a CJK path arrives as UTF-8 bytes rather than ``\346\226\207``.
#: ``core.askPass=`` empties any askpass helper the user's config names --
#: without it, a machine with Git Credential Manager configured pops a
#: dialog on the SERVER and the request hangs until it is answered.
#: ``color.ui=never`` keeps ANSI escapes out of output that is parsed.
GIT_CONFIG_ARGS: tuple[str, ...] = (
    "-c", "core.quotepath=false",
    "-c", "core.askPass=",
    "-c", "color.ui=never",
)

#: The environment every call runs under.
#:
#: ``GIT_TERMINAL_PROMPT=0`` and ``GCM_INTERACTIVE=never`` make git and Git
#: Credential Manager fail instead of asking; ``SSH_ASKPASS_REQUIRE=never``
#: stops ssh reaching for a GUI passphrase prompt; ``GIT_EDITOR`` and
#: ``GIT_SEQUENCE_EDITOR`` are ``:`` (the no-op command) so a merge commit
#: message or a rebase todo is accepted as-is instead of opening an editor
#: nobody can see. ``LC_ALL=C`` pins the wording the classifier matches on.
#: ``GIT_ALLOW_PROTOCOL`` is an allowlist: it disables ``ext::`` -- the
#: transport whose "URL" is a command line to run -- along with every other
#: helper the validators do not accept.
NON_INTERACTIVE_ENV: dict[str, str] = {
    "GIT_TERMINAL_PROMPT": "0",
    "GCM_INTERACTIVE": "never",
    "SSH_ASKPASS_REQUIRE": "never",
    "GIT_EDITOR": ":",
    "GIT_SEQUENCE_EDITOR": ":",
    "LC_ALL": "C",
    "GIT_ALLOW_PROTOCOL": "https:ssh:file",
}

#: Variables that tell git which repository to work on. Every one of them
#: overrides the ``-C <cwd>`` this module passes, so an inherited value
#: would move every command to another repository without changing a single
#: argument. Dropped, never overwritten: an empty ``GIT_DIR`` is not the
#: same thing as no ``GIT_DIR``.
REPOSITORY_ENV_VARS: tuple[str, ...] = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_NAMESPACE",
)

#: What we set ``GIT_SSH_COMMAND`` to when the user's setup has no opinion:
#: batch mode turns ssh's passphrase and host-key questions into an
#: immediate failure. Never applied over a value the user chose -- theirs
#: may name a key, a jump host or a Windows OpenSSH binary that ours does
#: not know about.
DEFAULT_GIT_SSH_COMMAND = "ssh -oBatchMode=yes"

#: How long a NEGATIVE answer about git is trusted. Long enough that a
#: status poll does not re-scan PATH, short enough that installing git does
#: not need a server restart.
MISSING_RECHECK_S = 30.0

#: ``git --version`` and the one-off ``core.sshCommand`` probe: both are
#: local reads that touch no network, so anything past a couple of seconds
#: means something is wrong rather than slow.
VERSION_TIMEOUT_S = 5.0
SSH_PROBE_TIMEOUT_S = 5.0

#: How long a killed process gets to hand back its pipes.
DRAIN_TIMEOUT_S = 3.0

#: ``git version 2.53.0.windows.2`` -- the vendor suffix is deliberately not
#: captured. Git for Windows and Apple's git both append their own build
#: numbering, and a version check that tried to order those would be
#: comparing two vendors' counters.
_VERSION_RE = re.compile(r"git version (\d+)\.(\d+)(?:\.(\d+))?")

# --- caches (see the module docstring) -------------------------------------

_executable: str | None = None
_missing_until: float | None = None
_version: tuple[int, int, int] | None = None
_version_probed_at: float | None = None
_ssh_configured: bool | None = None
_ssh_probe_failed_at: float | None = None


def _reset_for_tests() -> None:
    """Forget everything cached about the host's git.

    Every cache in this module is process-wide, which is right for a server
    and wrong for a test file: without this, whether ``git_version`` spawns
    a process would depend on which test ran first.
    """
    global _executable, _missing_until, _version, _version_probed_at
    global _ssh_configured, _ssh_probe_failed_at
    _executable = None
    _missing_until = None
    _version = None
    _version_probed_at = None
    _ssh_configured = None
    _ssh_probe_failed_at = None


def git_executable() -> str | None:
    """The host's ``git``, or None when there is none on PATH."""
    global _executable, _missing_until

    if _executable is not None:
        return _executable
    now = time.monotonic()
    if _missing_until is not None and now < _missing_until:
        return None
    found = shutil.which("git")
    if found is None:
        _missing_until = now + MISSING_RECHECK_S
        return None
    _executable = found
    return found


def _probe_cwd() -> Path:
    """A directory to run the calls that are not about a repository from.

    ``git --version`` and the ``core.sshCommand`` probe still go through the
    fixed prefix, and ``-C`` needs somewhere real to point at.

    The temporary directory, and NEVER ``Path.cwd()``: the server's working
    directory is its own checkout, which is a git repository the Source
    Control tab must never consult. A ``core.sshCommand`` in THAT
    repository's config would otherwise decide how the user's PROJECT
    reaches its remotes -- a setting from a repository nobody asked about,
    silently applied to another one. Only the global and system config can
    answer this question, and the temp directory is the cheapest place to
    stand where neither a repository nor a deleted working directory can
    interfere.
    """
    return Path(tempfile.gettempdir())


def git_version() -> tuple[int, int, int] | None:
    """``(major, minor, patch)`` of the host's git, or None if it cannot be read.

    Asked once and remembered: the answer cannot change while the process
    runs, and the caller is a status endpoint the editor polls. A failure to
    read it is remembered for :data:`MISSING_RECHECK_S` only, so a broken or
    half-installed git does not become permanent.
    """
    global _version, _version_probed_at

    if _version is not None:
        return _version
    now = time.monotonic()
    if _version_probed_at is not None and now - _version_probed_at < MISSING_RECHECK_S:
        return None
    _version_probed_at = now

    try:
        # The base environment, not git_env(): a version check has no use
        # for an ssh command, and probing for one from here would run a git
        # process to find out how to run a git process.
        result = _run(["--version"], cwd=_probe_cwd(), timeout=VERSION_TIMEOUT_S,
                      env=_base_git_env())
    except GitError:
        return None
    match = _VERSION_RE.search(result.out)
    if match is None:
        return None
    major, minor, patch = match.group(1), match.group(2), match.group(3)
    _version = (int(major), int(minor), int(patch or 0))
    return _version


def _base_git_env() -> dict[str, str]:
    """The environment without the ``GIT_SSH_COMMAND`` decision.

    Split out because the probe that makes that decision is itself a git
    call: it runs with this, which is why asking "is core.sshCommand set?"
    cannot recurse into asking it again.
    """
    env = pip_env()
    for name in REPOSITORY_ENV_VARS:
        env.pop(name, None)
    env.update(NON_INTERACTIVE_ENV)
    return env


def _ssh_command_configured() -> bool:
    """Has the user's git config already got a ``core.sshCommand``?

    Asked at most once per process, and only when the environment does not
    already answer it. The directory does not matter much -- a setting that
    is meant to apply to the user's remotes lives in their global config --
    so this deliberately does not need a repository to be open.

    Any failure counts as "no": git missing, a timeout, a config file the
    user cannot read. The consequence of guessing wrong here is one ssh run
    in batch mode instead of theirs, which fails fast; the consequence of
    raising would be that a config problem broke every git command.

    A failure is not remembered as an ANSWER -- a guess must not outlive the
    condition that produced it and go on overriding a setting the user made
    -- but it is remembered as a failure, for :data:`MISSING_RECHECK_S`
    seconds. A host whose config genuinely cannot be read (an unreadable
    ``.gitconfig``, a git that exits 128 on every ``config --get``) would
    otherwise pay a whole process for this question on every single command,
    forever: ``git_env`` is called once per git call, and the status
    endpoint is polled while the tab is open. One probe per interval is the
    same bargain :func:`git_executable` strikes for a missing binary.

    A missing git is the exception, and is deliberately NOT remembered here:
    nothing was probed, and ``git_executable`` already has its own recheck
    window for exactly that case.
    """
    global _ssh_configured, _ssh_probe_failed_at

    if _ssh_configured is not None:
        return _ssh_configured
    now = time.monotonic()
    if (_ssh_probe_failed_at is not None
            and now - _ssh_probe_failed_at < MISSING_RECHECK_S):
        return False
    try:
        result = _run(["config", "--get", "core.sshCommand"], cwd=_probe_cwd(),
                      timeout=SSH_PROBE_TIMEOUT_S, env=_base_git_env(),
                      ok_codes=(0, 1))
    except GitError as exc:
        if exc.code != "git_missing":
            _ssh_probe_failed_at = now
        return False
    _ssh_configured = result.returncode == 0 and bool(result.out.strip())
    return _ssh_configured


def git_env(*, read_only: bool = False) -> dict[str, str]:
    """The environment every git process here runs under.

    *read_only* adds ``GIT_OPTIONAL_LOCKS=0``, which is what lets a status
    read run while the user has a merge tool open: git normally takes the
    index lock to refresh its cache, and a status poll every two seconds
    that fights a real operation for that lock is worse than a status that
    is occasionally a little stale.
    """
    env = _base_git_env()
    if not env.get("GIT_SSH_COMMAND") and not _ssh_command_configured():
        env["GIT_SSH_COMMAND"] = DEFAULT_GIT_SSH_COMMAND
    if read_only:
        env["GIT_OPTIONAL_LOCKS"] = "0"
    return env


@dataclass
class GitResult:
    """One finished git command: what was run, and what it said.

    ``stdout``/``stderr`` are BYTES (see the module docstring). ``out`` and
    ``err`` are the decoded views, and every caller that wants text should
    use them rather than decoding again with different options -- a
    ``UnicodeDecodeError`` from a diff is a 500 for a file the user can see
    on disk.
    """

    argv: list[str]
    returncode: int
    stdout: bytes
    stderr: bytes

    @property
    def out(self) -> str:
        """``stdout`` as text; undecodable bytes become U+FFFD."""
        return self.stdout.decode("utf-8", errors="replace")

    @property
    def err(self) -> str:
        """``stderr`` as text; undecodable bytes become U+FFFD."""
        return self.stderr.decode("utf-8", errors="replace")


def _argv(git: str, cwd: Path, args: Sequence[str]) -> list[str]:
    """The full command line: executable, ``-C <cwd>``, fixed options, *args*.

    *cwd* is resolved, so the ``-C`` git is handed is an absolute path with
    no ``..`` in it even when the caller kept a relative one.
    """
    return [git, "-C", str(cwd.resolve()), *GIT_CONFIG_ARGS, *args]


def _drain(proc: subprocess.Popen) -> None:
    """Collect the pipes of a process that has just been killed.

    Nothing here wants the output -- the point is to close the pipes and
    reap the child, so a timeout does not leave a ``ResourceWarning`` and a
    zombie behind. Every failure is a pipe that is already gone.
    """
    try:
        proc.communicate(timeout=DRAIN_TIMEOUT_S)
    except (subprocess.TimeoutExpired, OSError, ValueError):
        pass


def _run(args: Sequence[str], *, cwd: Path, timeout: float,
         env: dict[str, str], input_bytes: bytes | None = None,
         ok_codes: Container[int] = (0,)) -> GitResult:
    """Run one git command with a PREBUILT environment.

    Private because the environment is not the caller's business:
    :func:`run_git` is the entry point, and the only reason this seam exists
    is that building the environment can itself need a git call
    (:func:`_ssh_command_configured`).
    """
    git = git_executable()
    if git is None:
        raise GitError("git_missing", 503,
                       hint="install git and make sure it is on PATH")
    argv = _argv(git, cwd, args)

    try:
        proc = subprocess.Popen(
            argv,
            # No ``cwd=``: ``-C`` already puts git in the right directory,
            # and a cwd Popen cannot chdir into raises FileNotFoundError --
            # which would be reported below as "git is not installed".
            stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            creationflags=creation_flags(),
        )
    except FileNotFoundError:
        raise GitError("git_missing", 503,
                       hint="install git and make sure it is on PATH") from None
    except OSError as exc:
        # A git that cannot be executed at all -- a permission bit, a
        # broken symlink on PATH. Structured, because an OSError escaping
        # here is a traceback in the browser instead of an error message.
        raise GitError("git_failed", 500, "git could not be started",
                       stderr=str(exc)) from None

    try:
        stdout, stderr = proc.communicate(input_bytes, timeout=timeout)
    except subprocess.TimeoutExpired:
        # Kill the TREE: git's own children (ssh, curl, a credential
        # helper) are what tend to be blocked, and they hold the index lock
        # that the next request needs.
        stop_process(proc)
        _drain(proc)
        raise GitError("timeout", 504,
                       f"git took longer than {timeout:g}s and was stopped")

    result = GitResult(argv=argv, returncode=proc.returncode,
                       stdout=stdout or b"", stderr=stderr or b"")
    if result.returncode not in ok_codes:
        raise classify_failure(argv, result.returncode, result.err)
    return result


def run_git(args: Sequence[str], *, cwd: Path, timeout: float,
            input_bytes: bytes | None = None,
            ok_codes: Container[int] = (0,),
            read_only: bool = False) -> GitResult:
    """Run ``git <args>`` in *cwd* and return what it said.

    *args* is everything after the fixed prefix -- the subcommand and its
    options -- and the caller is responsible for putting validated paths and
    refs after ``--``. *timeout* is in seconds and is not optional: every
    command here can block on something (a lock, a network read, a hook),
    and a request that never returns costs a worker.

    *input_bytes* is written to stdin, which is how a commit message reaches
    ``commit --file=-`` without ever being an argument. Without it stdin is
    ``DEVNULL``, so a git that decides to read from it gets EOF rather than
    waiting.

    *ok_codes* are the return codes that are NOT failures for this command:
    ``diff`` exits 1 to mean "there are differences" and ``config --get``
    exits 1 to mean "unset". Anything outside it raises -- ``GitError`` with
    a code from :func:`~app.core.git.errors.classify_failure`, or
    ``git_missing`` / ``timeout`` for the two failures that have no stderr
    to classify.

    *read_only* marks a command that only reads, so it can run without
    taking the index lock (see :func:`git_env`).
    """
    return _run(args, cwd=cwd, timeout=timeout,
                env=git_env(read_only=read_only),
                input_bytes=input_bytes, ok_codes=ok_codes)
