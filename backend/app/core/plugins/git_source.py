"""A plugin in a repository on any HTTPS git host: GitLab, Gitea, Codeberg...

GitHub has an API this package uses to read one manifest before anything is
downloaded (:mod:`.github`). Other hosts do not share one, so a repository
there is read the way ``git`` reads it: a shallow clone into a temporary
directory, at the ref that was asked for, and the commit it landed on is the
pin. From the clone on, the source is a folder, and the install takes the
local-folder road (:mod:`.local`): the copy under the caps, the manifest,
the AST gate, consent, staging, the lockfile.

Nothing typed reaches ``git`` as an option. The URL has been through
:func:`~.sources.parse_git_url` (https, a host name with no user name in it,
no ``.``/``..`` segment) and is checked again here; it is passed after
``--``; a ref is the value of ``--branch``, or, for the commit fallback, a
hex string. ``protocol.allow=never`` with only ``https`` switched back on
means a redirect or a submodule cannot move the clone to ``file://`` or
``ssh://``, and ``GIT_TERMINAL_PROMPT=0`` means a private repository fails
instead of waiting for a password on a server nobody is watching.

The clone runs in a child process this module polls: a cancel check and a
deadline between polls, and the size of the clone directory against
:data:`~.github.MAX_EXTRACTED_BYTES` once a second, so a Stop button, a
stalled host and a repository that is mostly a dataset all end it.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from . import github
from .errors import GitError, GitUnavailable, PluginCancelled, SourceRefused
from .sources import parse_git_url

#: Seconds one clone may take, all of its git processes together.
CLONE_TIMEOUT_S = 300.0

#: How often the child is polled for exit, cancel and the deadline.
POLL_S = 0.1

#: How often the clone directory's size is added up.
SIZE_CHECK_S = 1.0

#: The protocols the clone may use. Tests add ``file`` to clone a bare
#: repository in a temporary directory; nothing else should.
ALLOWED_PROTOCOLS: tuple[str, ...] = ("https",)

_SHA = re.compile(r"^[0-9a-f]{40}$")
_HEX_REF = re.compile(r"^[0-9a-fA-F]{7,40}$")

#: stderr lines that mean "there is nothing at that address you may read".
#: A private repository answers with a credential prompt, which
#: ``GIT_TERMINAL_PROMPT=0`` turns into one of these.
_NOT_FOUND = re.compile(
    r"not found|does not exist|could not read username|authentication failed"
    r"|terminal prompts disabled|returned error: 40[134]"
    r"|does not appear to be a git repository",
    re.IGNORECASE,
)

#: Variables that would point a child ``git`` at another repository or
#: another configuration than the clone it is asked to make.
_SCRUBBED_ENV = (
    "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_COUNT", "GIT_NAMESPACE", "GIT_ASKPASS", "SSH_ASKPASS",
)

_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def git_executable() -> str | None:
    """The ``git`` on PATH, or ``None``."""
    return shutil.which("git")


def require_git() -> str:
    """The ``git`` on PATH, or :class:`~.errors.GitUnavailable`."""
    found = git_executable()
    if found is None:
        raise GitUnavailable(
            "Installing from a git host needs git, and none is on this "
            "server's PATH.",
            hint=(
                "Install git (https://git-scm.com/downloads) on the machine "
                "the CodefyUI server runs on, restart the server, and try "
                "again. GitHub repositories and local folders do not need it."
            ),
        )
    return found


def _remote_for(url: str) -> str:
    """The address ``git`` is given for *url*. The identity, except in tests,
    which point an https URL at a bare repository on disk."""
    return url


def _config_args() -> list[str]:
    args = ["-c", "protocol.allow=never"]
    for protocol in ALLOWED_PROTOCOLS:
        args += ["-c", f"protocol.{protocol}.allow=always"]
    return args + [
        "-c", "core.symlinks=false",
        "-c", "credential.helper=",
        "-c", "submodule.recurse=false",
        "-c", "advice.detachedHead=false",
    ]


def _env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _SCRUBBED_ENV}
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "never"
    env["LC_ALL"] = "C"
    return env


def _remove_tree(path: Path) -> None:
    """Delete a half-made clone. git writes its objects read-only, which
    Windows will not delete until the bit is cleared."""
    def _clear_and_retry(func, target, _exc):
        try:
            os.chmod(target, stat.S_IWRITE)
            func(target)
        except OSError:
            pass

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_clear_and_retry)
    else:
        shutil.rmtree(path, onerror=_clear_and_retry)


def _tree_bytes(root: Path) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_size
            except OSError:
                pass
    return total


def _tail(stderr: str, lines: int = 3) -> str:
    """The last few lines git wrote, cleaned for a terminal or a dialog."""
    kept = [_CONTROL.sub("", line).strip() for line in stderr.splitlines()]
    kept = [line for line in kept if line]
    return " / ".join(kept[-lines:])[:500]


def _run(
    argv: list[str],
    *,
    label: str,
    deadline: float,
    cancel_check: Callable[[], bool] | None,
    watch: Path | None = None,
) -> tuple[int, str]:
    """Run one git command to completion; returns ``(exit code, stderr)``.

    Killed, and the right exception raised, when *cancel_check* goes true,
    when *deadline* passes, or when *watch* grows past the unpacked-size cap.
    stderr goes to a temporary file rather than a pipe, so a chatty child
    can never block on a full pipe this loop is not reading.
    """
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=err, env=_env(),
        )
        last_size_check = time.monotonic()
        try:
            while True:
                try:
                    code = proc.wait(timeout=POLL_S)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if cancel_check is not None and cancel_check():
                    raise PluginCancelled(f"{label} cancelled")
                now = time.monotonic()
                if now > deadline:
                    raise GitError(
                        f"{label} took longer than {CLONE_TIMEOUT_S:.0f}s.",
                        reason="timeout",
                        hint="The host is slow or unreachable; try again later.",
                    )
                if watch is not None and now - last_size_check >= SIZE_CHECK_S:
                    last_size_check = now
                    size = _tree_bytes(watch)
                    if size > github.MAX_EXTRACTED_BYTES:
                        cap_mb = github.MAX_EXTRACTED_BYTES // (1024 * 1024)
                        raise SourceRefused(
                            f"{label} grew past {cap_mb} MB.",
                            reason="too_large",
                            hint="Keep datasets and model weights out of the "
                                 "plugin's repository.",
                        )
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        err.seek(0)
        return code, err.read().decode("utf-8", "replace")


def _clone_error(url: str, ref: str, stderr: str) -> GitError:
    target = f"{url}@{ref}" if ref else url
    reason = "not_found" if _NOT_FOUND.search(stderr) else "failed"
    message = (
        f"{target} has no repository or ref this server can read."
        if reason == "not_found"
        else f"git could not clone {target}."
    )
    return GitError(message, reason=reason, hint=_tail(stderr) or None)


def _head(git: str, dest: Path) -> str:
    result = subprocess.run(
        [git, "-C", str(dest), "rev-parse", "HEAD"],
        capture_output=True, text=True, env=_env(), timeout=30,
    )
    sha = result.stdout.strip()
    if result.returncode != 0 or not _SHA.fullmatch(sha):
        raise GitError(f"git could not read the commit of {dest.name}.",
                       reason="failed", hint=_tail(result.stderr) or None)
    return sha


def clone(
    url: str,
    ref: str,
    dest: Path,
    *,
    want_sha: str | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> str:
    """Clone *url* at *ref* into *dest* (which must not exist); returns the
    full commit sha checked out.

    A shallow clone of the branch or tag; when that fails and *ref* looks
    like a commit, a full clone and a checkout of that commit. *want_sha*
    is the commit an inspection pinned: when the ref has moved on since,
    that commit is fetched and checked out instead, and an install that
    cannot reach it is refused as ``changed`` rather than installing a
    commit nobody reviewed.

    :raises GitUnavailable: no git on PATH.
    :raises GitError: ``not_found``, ``timeout`` or ``failed``.
    :raises SourceRefused: ``too_large``, or ``changed`` for *want_sha*.
    :raises PluginCancelled: when *cancel_check* goes true.
    """
    git = require_git()
    if parse_git_url(url + (f"@{ref}" if ref else "")) != (url, ref):
        raise GitError(f"{url!r} is not an https repository URL this build "
                       f"clones.", reason="failed")
    if want_sha is not None and not _SHA.fullmatch(want_sha):
        raise GitError(f"{want_sha!r} is not a commit.", reason="failed")
    deadline = time.monotonic() + CLONE_TIMEOUT_S
    remote = _remote_for(url)
    label = f"Cloning {url}" + (f"@{ref}" if ref else "")
    run = {"label": label, "deadline": deadline, "cancel_check": cancel_check,
           "watch": dest}

    argv = [git, *_config_args(), "clone", "--quiet", "--depth", "1"]
    if ref:
        argv += ["--branch", ref]
    code, stderr = _run([*argv, "--", remote, str(dest)], **run)
    if code != 0:
        if not (ref and _HEX_REF.fullmatch(ref)):
            raise _clone_error(url, ref, stderr)
        # Not a branch or a tag: a commit, which --branch cannot name. The
        # whole history is needed to find it, under the same caps.
        _remove_tree(dest)
        code, stderr = _run(
            [git, *_config_args(), "clone", "--quiet", "--no-checkout",
             "--", remote, str(dest)], **run)
        if code != 0:
            raise _clone_error(url, ref, stderr)
        code, stderr = _run(
            [git, "-C", str(dest), "checkout", "--quiet", "--detach", ref, "--"],
            **run)
        if code != 0:
            raise GitError(f"{ref} is not a commit in {url}.", reason="not_found",
                           hint=_tail(stderr) or None)

    head = _head(git, dest)
    if want_sha is None or head == want_sha:
        return head
    code, stderr = _run(
        [git, "-C", str(dest), *_config_args(), "fetch", "--quiet", "--depth",
         "1", "origin", want_sha], **run)
    if code == 0:
        code, stderr = _run(
            [git, "-C", str(dest), "checkout", "--quiet", "--detach", want_sha,
             "--"], **run)
    if code != 0:
        raise SourceRefused(
            f"{url}" + (f"@{ref}" if ref else "") + " moved after it was "
            f"reviewed, and the reviewed commit {want_sha[:7]} cannot be "
            f"fetched.",
            reason="changed",
            hint="Review the source again and install what that shows.",
        )
    return _head(git, dest)
