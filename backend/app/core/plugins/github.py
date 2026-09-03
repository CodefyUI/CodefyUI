"""The only place this project talks to GitHub about a plugin.

Three requests, and each one has a different failure worth telling apart:
resolve a ref to a commit, read that commit's manifest, download that
commit's tarball. They were three ad-hoc ``urlopen`` calls in the CLI, each
translating ``HTTPError`` into its own sentence, and the sentences did not
say the same thing about the same failure. A GUI cannot work from sentences
at all, so the answer here is :class:`~.errors.GitHubError` with a
``status``: 404 is a typo in the repo name, 403 is usually the rate limit,
and ``None`` means the request never reached GitHub -- three different next
steps for the user, and none of them recoverable from message text.

The message is GitHub's own when GitHub sent one. Its JSON bodies say
"Not Found", "API rate limit exceeded for ...", "Bad credentials"; the HTTP
reason phrase says "Forbidden" for all three. Quoting the body is what makes
a rate-limited install say so.

``CODEFYUI_GITHUB_TOKEN`` is read from the environment at CALL time, not at
import: a server process that gains a token must not have to restart to use
it. It is attached as a bearer header and never logged, echoed, or put in an
exception -- this module logs nothing at all, which is the cheapest way to
be sure.

The download is streamed rather than read whole, and takes a *cancel_check*
and a *progress* callback, because the caller may be a job the user can stop
from a browser: a 100 MB read inside one ``resp.read()`` is a Stop button
that does nothing for a minute and a progress bar with two frames.
"""

from __future__ import annotations

import http.client
import json
import os
import tarfile
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from urllib.error import HTTPError, URLError

from app.core.plugin_loader import MANIFEST_FILENAME

from .errors import GitHubError, PluginCancelled, PluginInstallError

USER_AGENT = "cdui-plugin-installer/0.1"
MAX_TARBALL_BYTES = 100 * 1024 * 1024  # 100 MB

#: How much a tarball may unpack to. :data:`MAX_TARBALL_BYTES` caps the
#: COMPRESSED stream, and gzip of a file of zeroes runs to about 1000:1 --
#: so 100 MB off the wire is tens of gigabytes on the disk, written into the
#: user's plugin directory before anything downstream gets a chance to
#: object. Five times the download cap is generous for real source (a
#: repository of text compresses maybe 4:1) and nowhere near a bomb.
MAX_EXTRACTED_BYTES = 500 * 1024 * 1024  # 500 MB

#: Read size for the tarball stream. Small enough that a cancel is felt
#: immediately on a slow link, large enough that a 100 MB download is not
#: 1600 syscalls.
CHUNK_BYTES = 64 * 1024

#: Minimum seconds between two ``progress`` calls. Four frames a second is
#: smooth to a human and is orders of magnitude fewer events than one per
#: 64 KB chunk would send down a WebSocket. The first and last frames are
#: forced past it -- a download that finishes inside a throttle window would
#: otherwise leave the bar stopped at 97%.
PROGRESS_MIN_INTERVAL_S = 0.25


def _headers() -> dict[str, str]:
    """Request headers, with the token when the environment has one.

    Unauthenticated GitHub allows 60 requests an hour per IP, which a
    classroom behind one NAT exhausts in a morning -- ``CODEFYUI_GITHUB_TOKEN``
    is how a school raises that. Read here, per request, so a token exported
    after the server started still works; returned rather than cached so no
    copy of it outlives the call.
    """
    headers = {"User-Agent": USER_AGENT}
    token = os.environ.get("CODEFYUI_GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _from_http_error(exc: HTTPError) -> GitHubError:
    """An :class:`~.errors.GitHubError` carrying GitHub's own words.

    The body is JSON with a ``message`` on every GitHub API error, and that
    message is the one that distinguishes the three things a 403 can mean.
    Anything unreadable falls back to the HTTP reason phrase, which is at
    least always there.
    """
    message = ""
    try:
        body = exc.read()
    except Exception:  # a body that cannot be read is not the failure to report
        body = b""
    if body:
        try:
            payload = json.loads(body.decode("utf-8", "replace"))
        except ValueError:
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("message"), str):
            message = payload["message"]
    return GitHubError(message or str(exc.reason), status=exc.code)


def _from_url_error(exc: BaseException) -> GitHubError:
    """A :class:`~.errors.GitHubError` for a request that never got a status.

    DNS, TLS, a refused connection, a timeout: no HTTP status exists, and
    ``None`` is what tells the caller to talk about the network rather than
    about the repository name.
    """
    return GitHubError(str(getattr(exc, "reason", exc)), status=None)


def _discard(path: Path) -> None:
    """Delete a partial download, never raising over it.

    Cleanup runs while another exception is in flight, and a file the
    filesystem will not let go of must not replace the failure that is
    actually worth reporting.
    """
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _gh_get(url: str, timeout: float = 30.0) -> bytes:
    """GET *url* from GitHub, or raise :class:`~.errors.GitHubError`.

    Every request in this module goes through here, so the token, the user
    agent and the two shapes of failure are decided once.

    ``http.client.HTTPException`` is caught alongside the OSError family
    because it is NOT one: ``InvalidURL`` -- what a ref with a space or a
    control character in it becomes by the time ``putrequest`` sees the URL
    -- would otherwise travel out of here as itself, past every caller that
    catches ``GitHubError`` and past the CLI's ``except RuntimeError``, and
    turn a bad ref into a traceback. It never reached GitHub, so it has no
    status, which is exactly what ``None`` says.
    """
    req = urllib.request.Request(url, headers=_headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except HTTPError as exc:
        raise _from_http_error(exc) from exc
    except (URLError, TimeoutError, http.client.HTTPException) as exc:
        raise _from_url_error(exc) from exc


def resolve_sha(owner: str, repo: str, ref: str) -> str:
    """Convert tag / branch / short-sha to a full 40-char SHA.

    The failure is re-raised naming the repository and the ref, because
    GitHub's own "Not Found" answers a question the user never asked out
    loud: it is the repo, the ref, or the spelling of either, and only the
    caller's sentence can say which three things to check. The ``status``
    rides along unchanged.

    A 200 that is not JSON is a failure too, and its own one: a captive
    portal, a school proxy's block page or a transparent MITM all answer
    "200 OK" with HTML. That is a ``json.JSONDecodeError`` -- a ``ValueError``
    that no caller of this function catches -- so it becomes a
    :class:`~.errors.GitHubError` with no status, because whatever answered
    was not GitHub.
    """
    target = ref or "HEAD"
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{target}"
    try:
        body = _gh_get(url)
    except GitHubError as exc:
        if exc.status is None:
            raise GitHubError(
                f"GitHub API request failed: {exc}", status=None
            ) from exc
        raise GitHubError(
            f"GitHub API returned {exc.status} for {owner}/{repo}@{target}: {exc}",
            status=exc.status,
        ) from exc
    try:
        data = json.loads(body)
    except ValueError as exc:
        raise GitHubError(
            f"GitHub API request for {owner}/{repo}@{target} did not return "
            f"JSON; something on the way answered instead of GitHub",
            status=None,
        ) from exc
    sha = data.get("sha") if isinstance(data, dict) else None
    if not sha:
        raise GitHubError(
            f"GitHub API response for {owner}/{repo}@{target} is missing 'sha'"
        )
    return sha


def fetch_manifest_text(owner: str, repo: str, sha: str) -> str:
    """The ``cdui.plugin.toml`` of one commit, as text.

    Read from ``raw.githubusercontent.com`` at a full SHA rather than out of
    the tarball, so that inspecting a plugin -- which is what a person does
    BEFORE agreeing to install it -- costs one small file instead of the
    whole repository. Pinning the SHA is what makes the manifest that was
    shown the manifest that gets installed.

    Raises :class:`~.errors.GitHubError` like everything else here, and
    ``UnicodeDecodeError`` for a file that is not UTF-8 -- a manifest that
    is not text is a disk answer, not a network one, and its caller can say
    something better than "request failed".
    """
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{sha}/{MANIFEST_FILENAME}"
    return _gh_get(url).decode("utf-8")


def _content_length(resp: object) -> int | None:
    """The response's ``Content-Length``, or ``None`` when it has none.

    ``None`` rather than 0: a progress bar told the total is zero renders as
    finished, while one told nothing renders as counting up, which is the
    truth about a chunked response.
    """
    headers = getattr(resp, "headers", None)
    raw = headers.get("Content-Length") if headers is not None else None
    try:
        total = int(raw)
    except (TypeError, ValueError):
        return None
    return total or None


def download_tarball(
    owner: str,
    repo: str,
    sha: str,
    dest: Path,
    *,
    cancel_check: Callable[[], bool] | None = None,
    progress: Callable[[int, int | None], None] | None = None,
) -> None:
    """Stream one commit's tarball to *dest*.

    Streamed, and cancelled BETWEEN CHUNKS, because the caller may be a job
    a browser can stop: a check that only ran between whole downloads would
    not be reached until the download the user is trying to stop had
    finished.

    *progress* is called ``(bytes_done, bytes_total_or_None)`` at most every
    :data:`PROGRESS_MIN_INTERVAL_S`, always once before the first byte and
    always once after the last.

    Nothing partial is left behind. A cancelled or over-cap download removes
    *dest*, so no caller can mistake half a tarball for a file it may open.
    """
    url = f"https://codeload.github.com/{owner}/{repo}/tar.gz/{sha}"
    req = urllib.request.Request(url, headers=_headers())
    bytes_read = 0
    total: int | None = None
    last_report = 0.0
    reported = False

    def _report(*, force: bool) -> None:
        nonlocal last_report, reported
        if progress is None:
            return
        now = time.monotonic()
        if not force and reported and now - last_report < PROGRESS_MIN_INTERVAL_S:
            return
        last_report = now
        reported = True
        progress(bytes_read, total)

    try:
        with urllib.request.urlopen(req, timeout=60.0) as resp, dest.open("wb") as fout:
            total = _content_length(resp)
            _report(force=True)
            while True:
                chunk = resp.read(CHUNK_BYTES)
                if not chunk:
                    break
                bytes_read += len(chunk)
                if bytes_read > MAX_TARBALL_BYTES:
                    cap_mb = MAX_TARBALL_BYTES // (1024 * 1024)
                    raise PluginInstallError(
                        f"Tarball exceeds {cap_mb} MB limit.",
                        hint=(
                            f"{owner}/{repo}@{sha[:7]} is larger than the "
                            f"{cap_mb} MB this build will download; nothing "
                            f"was kept."
                        ),
                    )
                fout.write(chunk)
                if cancel_check is not None and cancel_check():
                    raise PluginCancelled(
                        f"download of {owner}/{repo}@{sha[:7]} cancelled"
                    )
                _report(force=False)
            _report(force=True)
    except HTTPError as exc:
        _discard(dest)
        raise _from_http_error(exc) from exc
    except (URLError, TimeoutError) as exc:
        _discard(dest)
        raise _from_url_error(exc) from exc
    except BaseException:
        # Everything else -- the cap, a cancel, a disk error, Ctrl-C -- leaves
        # the same rule behind it: no half a tarball where a caller looks.
        _discard(dest)
        raise


def extract_tarball(tar_path: Path, dest_dir: Path) -> Path:
    """Unpack *tar_path* into *dest_dir* and return its single root directory.

    ``filter="data"`` is the load-bearing argument: without it a tarball can
    write outside *dest_dir* through ``../`` members, absolute paths, links
    and device nodes -- and this one comes off the internet.

    The unpacked size is added up BEFORE a single member is written, because
    the alternative is finding out at the point where the disk is already
    full: ``filter="data"`` says where the bytes may go and nothing says how
    many there may be, and the ratio between the two caps is gzip's, not the
    author's. The index has to be read to do it, which for a gzip stream
    means decompressing it once more than a bare ``extractall`` would --
    paid on every install, in exchange for a cap that cannot be reached.

    Exactly one top-level directory, or nothing is installed. A GitHub
    codeload tarball is always ``<repo>-<ref>/...`` and that root is what the
    manifest is read from; a tarball with two of them is not a plugin whose
    root can be guessed, and guessing would mean installing whichever one the
    filesystem happened to list first.

    Every ``tarfile`` failure becomes a :class:`~.errors.PluginInstallError`.
    A truncated download and a file that is not a gzip at all both arrive
    here as ``TarError``, which is a class no caller of an installer has any
    reason to know about -- and one that travelled past every ``except`` on
    the way out to a traceback.
    """
    try:
        with tarfile.open(tar_path, "r:gz") as tf:
            members = tf.getmembers()
            unpacked = sum(member.size for member in members)
            if unpacked > MAX_EXTRACTED_BYTES:
                cap_mb = MAX_EXTRACTED_BYTES // (1024 * 1024)
                raise PluginInstallError(
                    f"{tar_path.name} unpacks to more than {cap_mb} MB.",
                    hint=(
                        f"Its members add up to {unpacked // (1024 * 1024)} MB, "
                        f"which is past the {cap_mb} MB this build will write; "
                        f"nothing was unpacked."
                    ),
                )
            tf.extractall(dest_dir, filter="data", members=members)
    except tarfile.TarError as exc:
        raise PluginInstallError(
            f"{tar_path.name} could not be unpacked.",
            hint=f"tarfile could not read it ({exc}); retry the install.",
        ) from exc
    roots = [p for p in sorted(dest_dir.iterdir()) if p.is_dir()]
    if len(roots) != 1:
        raise PluginInstallError(
            "A plugin tarball must hold exactly one top-level directory.",
            hint=(
                f"{tar_path.name} unpacked to {len(roots)}: "
                f"{', '.join(p.name for p in roots) or '(none)'}"
            ),
        )
    return roots[0]
