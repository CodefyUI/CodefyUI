"""The installers and `cdui update` stop instead of quietly installing `main`.

All three entry points look the latest release up before they touch git, so
that the backend checkout matches the prebuilt frontend they download. Until
2.8.9, "could not look it up" was not an error anywhere:

* ``install.ps1`` and ``cdui update`` printed one warning line and installed
  ``main``; ``cdui update`` then downloaded the latest RELEASE's frontend onto
  that ``main`` checkout -- the version drift the pinning exists to prevent.
* ``install.sh`` meant to do the same, but under ``set -euo pipefail`` the
  failed lookup ended the script on its assignment line with no message at
  all, so its "falling back to main" warning was dead code.

The lookup is GitHub's unauthenticated REST API: 60 requests an hour per IP.
A classroom or an office behind one NAT address uses that up, and a network
blip does the same. So each entry point now asks the release PAGE second --
it redirects to ``.../releases/tag/<tag>`` and is not under the API limit --
and only when both fail does it stop, naming the two ways out:
``CODEFYUI_RELEASE_TAG`` to pin a release, ``CODEFYUI_FORCE_BUILD=1`` to
install ``main`` on purpose. A tag the user names is used even with
``CODEFYUI_FORCE_BUILD=1`` (installers) or pnpm on PATH (``cdui update``).

Nothing here reaches the network: ``dev.urlopen`` is replaced in every test,
and the shell installers run with ``curl`` / ``Invoke-RestMethod`` /
``Invoke-WebRequest`` stubbed.
"""

from __future__ import annotations

import functools
import http.client
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace
from typing import NamedTuple
from urllib.error import HTTPError, URLError

import pytest

import dev  # scripts/dev.py -- conftest puts scripts/ on sys.path

API_URL = "https://api.github.com/repos/CodefyUI/CodefyUI/releases/latest"
PAGE_URL = "https://github.com/CodefyUI/CodefyUI/releases/latest"
TAG_URL = "https://github.com/CodefyUI/CodefyUI/releases/tag/{}"
#: Where the release page sends you when a repo has no release yet.
NO_RELEASE_URL = "https://github.com/CodefyUI/CodefyUI/releases"


def _read(rel: str) -> str:
    return (dev.ROOT / rel).read_text(encoding="utf-8")


# ── a scripted GitHub for dev.urlopen ────────────────────────────────────────


class _Resp:
    """Just enough of an HTTP response: a body for the API, a final URL for
    the release page (urlopen has already followed the redirect).
    ``read_error`` is raised by ``read()``: a connection that drops mid-answer."""

    def __init__(self, url: str, body: bytes = b"", read_error=None) -> None:
        self._url = url
        self._body = body
        self._read_error = read_error

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, *_args) -> bytes:
        if self._read_error is not None:
            raise self._read_error
        return self._body

    def geturl(self) -> str:
        return self._url


class _FakeGitHub:
    """Stands in for ``dev.urlopen``.

    ``api`` is the API's JSON answer (a dict) or the exception its request
    raises; ``page`` is the release page's final URL after redirects, or the
    exception. Either may also be a ready ``_Resp``, handed back as is.
    ``requests`` records ``(method, url)`` in order, because which lookups
    ran -- and that the page was asked with HEAD -- is the behaviour.
    """

    def __init__(self, *, api=None, page=None) -> None:
        self.api = URLError("offline") if api is None else api
        self.page = URLError("offline") if page is None else page
        self.requests: list[tuple[str, str]] = []

    def __call__(self, req, timeout=None):
        url, method = req.full_url, req.get_method()
        self.requests.append((method, url))
        if url == API_URL:
            answer = self.api
        elif url == PAGE_URL:
            answer = self.page
        else:
            raise AssertionError(f"unexpected request: {method} {url}")
        if isinstance(answer, BaseException):
            raise answer
        if isinstance(answer, _Resp):
            return answer
        if url == API_URL:
            return _Resp(url, json.dumps(answer).encode("utf-8"))
        return _Resp(answer)


def _no_network(req, timeout=None):
    raise AssertionError(f"no lookup expected, got {req.get_method()} {req.full_url}")


def _rate_limited() -> HTTPError:
    """What the API answers once an IP has used its 60 requests this hour."""
    body = io.BytesIO(b'{"message": "API rate limit exceeded"}')
    return HTTPError(API_URL, 403, "rate limit exceeded", {}, body)


@pytest.fixture(autouse=True)
def no_tag_env(monkeypatch):
    """Neither variable set, whatever the developer's shell exports, and no
    lookup allowed unless the test installs its own ``_FakeGitHub``.

    ``setenv`` before ``delenv`` on purpose: ``delenv`` of a variable that is
    not set records nothing to undo, so the ``CODEFYUI_RELEASE_TAG`` that
    ``update()`` writes into ``os.environ`` would outlive the test.
    """
    for name in ("CODEFYUI_RELEASE_TAG", "CODEFYUI_FORCE_BUILD"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.setattr(dev, "urlopen", _no_network)


# ── _resolve_release_tag: API first, then the release page ──────────────────


@pytest.mark.parametrize("api", [
    pytest.param(_rate_limited(), id="rate-limited"),
    pytest.param(URLError("Name or service not known"), id="unreachable"),
    pytest.param(TimeoutError("timed out"), id="timeout"),
    # Raised raw by http.client's getresponse(), not wrapped in a URLError.
    pytest.param(http.client.RemoteDisconnected("Remote end closed connection without response"),
                 id="remote-disconnected"),
    pytest.param({"message": "no tag here"}, id="answer-without-tag"),
])
def test_resolver_asks_the_release_page_when_the_api_fails(monkeypatch, api):
    github = _FakeGitHub(api=api, page=TAG_URL.format("2.8.8"))
    monkeypatch.setattr(dev, "urlopen", github)

    assert dev._resolve_release_tag() == "2.8.8"
    # HEAD: the tag is in the redirect target, the page body is never needed.
    assert github.requests == [("GET", API_URL), ("HEAD", PAGE_URL)]


@pytest.mark.parametrize("error", [
    pytest.param(http.client.RemoteDisconnected("Remote end closed connection without response"),
                 id="remote-disconnected"),
    pytest.param(http.client.IncompleteRead(b'{"tag_na'), id="incomplete-read"),
])
def test_resolver_asks_the_release_page_when_the_api_answer_breaks_off(monkeypatch, error):
    """A connection that drops while the answer is read raises from
    http.client, which urllib does not wrap: it must still reach the page
    lookup, not end `cdui update` with a traceback."""
    github = _FakeGitHub(api=_Resp(API_URL, read_error=error), page=TAG_URL.format("2.8.8"))
    monkeypatch.setattr(dev, "urlopen", github)

    assert dev._resolve_release_tag() == "2.8.8"
    assert github.requests == [("GET", API_URL), ("HEAD", PAGE_URL)]


def test_resolver_gives_up_when_the_page_connection_drops(monkeypatch):
    monkeypatch.setattr(dev, "urlopen", _FakeGitHub(
        api=_rate_limited(),
        page=http.client.RemoteDisconnected("Remote end closed connection without response")))
    assert dev._resolve_release_tag() is None


def test_resolver_takes_the_api_answer_without_asking_the_page(monkeypatch):
    github = _FakeGitHub(api={"tag_name": "2.8.8"})
    monkeypatch.setattr(dev, "urlopen", github)

    assert dev._resolve_release_tag() == "2.8.8"
    assert github.requests == [("GET", API_URL)]


def test_resolver_gives_up_when_both_lookups_fail(monkeypatch):
    monkeypatch.setattr(dev, "urlopen", _FakeGitHub(api=_rate_limited()))
    assert dev._resolve_release_tag() is None


def test_resolver_rejects_a_redirect_that_names_no_release(monkeypatch):
    """A repo with no release redirects to the release list, not a tag."""
    monkeypatch.setattr(dev, "urlopen",
                        _FakeGitHub(api=_rate_limited(), page=NO_RELEASE_URL))
    assert dev._resolve_release_tag() is None


def test_resolver_decodes_the_tag_in_the_redirect(monkeypatch):
    monkeypatch.setattr(dev, "urlopen", _FakeGitHub(
        api=_rate_limited(), page=TAG_URL.format("v3.0.0%2Bexam")))
    assert dev._resolve_release_tag() == "v3.0.0+exam"


def test_resolver_returns_an_explicit_tag_without_a_lookup(monkeypatch):
    monkeypatch.setenv("CODEFYUI_RELEASE_TAG", "2.8.7")
    assert dev._resolve_release_tag() == "2.8.7"


# ── update(): which source it checks out ─────────────────────────────────────


@pytest.fixture
def updatable(tmp_path, monkeypatch):
    """`update()` with every destructive edge redirected at tmp_path.

    The fixture of the same name in test_dev_shared_host.py, minus its
    ``CODEFYUI_FORCE_BUILD=1``: these tests are about the release path.
    ``shutil.rmtree(DIST_DIR)`` runs inline in this process, so DIST_DIR
    must point at a throwaway directory, not the real frontend/dist.
    """
    root = tmp_path / "codefyui"
    (root / ".git").mkdir(parents=True)
    dist = root / "frontend" / "dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text("<html></html>", encoding="utf-8")
    monkeypatch.setattr(dev, "ROOT", root)
    monkeypatch.setattr(dev, "VENV", root / "backend" / ".venv")
    monkeypatch.setattr(dev, "DIST_DIR", dist)
    monkeypatch.setattr(dev, "SERVER_PIDFILE", tmp_path / "server.pid")
    monkeypatch.setattr(dev, "SERVER_ADDRFILE", tmp_path / "server.addr")
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path / "userdata"))
    calls: list[list[str]] = []
    seen_tag: list = []

    def fake_install(**_kw):
        calls.append(["install"])
        # install() picks the dist to download from this variable.
        seen_tag.append(os.environ.get("CODEFYUI_RELEASE_TAG"))

    monkeypatch.setattr(dev, "run", lambda cmd, **kw: calls.append(list(cmd)))
    monkeypatch.setattr(dev, "install", fake_install)
    monkeypatch.setattr(dev, "_resolve_update_options", lambda argv: ("skip", False))
    monkeypatch.setattr(sys, "argv", ["cdui", "update"])
    return SimpleNamespace(calls=calls, dist=dist, seen_tag=seen_tag)


def _pnpm_on_path(monkeypatch, present: bool) -> None:
    monkeypatch.setattr(dev.shutil, "which",
                        lambda name: f"/usr/bin/{name}" if present and name == "pnpm" else None)


MAIN = [["git", "fetch", "origin", "main", "--depth", "1"],
        ["git", "checkout", "-B", "main", "FETCH_HEAD"]]


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_update_refuses_when_the_release_cannot_be_found(updatable, monkeypatch, capsys, lang):
    """No git, no install, the served dist left alone -- and a message an
    admin can act on, in either language."""
    monkeypatch.setattr(dev, "LANG", lang)
    _pnpm_on_path(monkeypatch, present=False)
    monkeypatch.setattr(dev, "urlopen", _FakeGitHub(api=_rate_limited()))

    with pytest.raises(SystemExit) as exc:
        dev.update()

    assert exc.value.code == 1
    assert updatable.calls == [], updatable.calls
    assert (updatable.dist / "index.html").exists()
    message = capsys.readouterr().err
    assert "CODEFYUI_RELEASE_TAG" in message
    assert "CODEFYUI_FORCE_BUILD" in message


def test_update_pins_the_tag_the_release_page_names(updatable, monkeypatch):
    _pnpm_on_path(monkeypatch, present=False)
    monkeypatch.setattr(dev, "urlopen", _FakeGitHub(
        api=_rate_limited(), page=TAG_URL.format("2.8.8")))

    dev.update()

    assert ["git", "checkout", "-f", "2.8.8"] in updatable.calls
    assert not any(cmd in updatable.calls for cmd in MAIN)
    assert updatable.seen_tag == ["2.8.8"]


@pytest.mark.parametrize("pnpm, force_build", [
    pytest.param(True, "", id="pnpm-on-path"),
    pytest.param(False, "1", id="force-build"),
    pytest.param(True, "1", id="both"),
])
def test_update_checks_out_an_explicit_tag_over_the_build_path(
        updatable, monkeypatch, pnpm, force_build):
    """The user named a version: checking out `main` instead is the same
    silent substitution as the lookup fallback was."""
    monkeypatch.setenv("CODEFYUI_RELEASE_TAG", "2.8.7")
    if force_build:
        monkeypatch.setenv("CODEFYUI_FORCE_BUILD", force_build)
    _pnpm_on_path(monkeypatch, present=pnpm)

    dev.update()

    assert ["git", "checkout", "-f", "2.8.7"] in updatable.calls
    assert not any(cmd in updatable.calls for cmd in MAIN)
    assert updatable.calls[-1] == ["install"]
    assert updatable.seen_tag == ["2.8.7"]


@pytest.mark.parametrize("pnpm, force_build, tag", [
    pytest.param(False, "1", None, id="force-build"),
    pytest.param(True, "", None, id="pnpm-on-path"),
    pytest.param(True, "", "latest", id="pnpm-and-tag-latest"),
])
def test_update_still_tracks_main_on_the_build_path(
        updatable, monkeypatch, pnpm, force_build, tag):
    """The explicit ways to `main` stay; the literal `latest` is not a tag."""
    if force_build:
        monkeypatch.setenv("CODEFYUI_FORCE_BUILD", force_build)
    if tag:
        monkeypatch.setenv("CODEFYUI_RELEASE_TAG", tag)
    _pnpm_on_path(monkeypatch, present=pnpm)

    dev.update()

    assert updatable.calls == [*MAIN, ["install"]]


def test_update_resolves_a_literal_latest_on_the_release_path(updatable, monkeypatch):
    monkeypatch.setenv("CODEFYUI_RELEASE_TAG", "latest")
    _pnpm_on_path(monkeypatch, present=False)
    github = _FakeGitHub(api={"tag_name": "2.8.8"})
    monkeypatch.setattr(dev, "urlopen", github)

    dev.update()

    assert github.requests == [("GET", API_URL)]
    assert ["git", "checkout", "-f", "2.8.8"] in updatable.calls


# ── the shell installers: what the text must (not) say ───────────────────────


def test_installers_no_longer_fall_back_to_main():
    for rel in ("install.sh", "install.ps1"):
        assert "改用 main" not in _read(rel), rel


def test_installers_ask_the_release_page_after_the_api():
    # The closing quote matters: the dist download URL starts the same way
    # (".../releases/latest/download/...").
    assert re.search(r'https://github\.com/\$\{RELEASE_REPO\}/releases/latest"',
                     _read("install.sh"))
    assert re.search(r'https://github\.com/\$ReleaseRepo/releases/latest"',
                     _read("install.ps1"))


def test_installer_refusals_name_both_ways_out():
    assert re.search(r'die "[^"]*CODEFYUI_RELEASE_TAG[^"]*CODEFYUI_FORCE_BUILD[^"]*"',
                     _read("install.sh"))
    assert re.search(r'Die "[^"]*CODEFYUI_RELEASE_TAG[^"]*CODEFYUI_FORCE_BUILD[^"]*"',
                     _read("install.ps1"))


# ── the shell installers, run for real with the network stubbed ─────────────
#
# Each test cuts the tag decision out of the installer -- its die helper, its
# resolver function and the block that calls them -- and runs that text in
# bash or PowerShell after stubs for the commands that would reach GitHub.
# The cut is by pattern; when a pattern stops matching, the installer was
# restructured and the pattern has to follow it.


class _Case(NamedTuple):
    api_tag: str | None    # the API's tag_name; None = the request fails
    final_url: str | None  # the release page's final URL; None = it fails
    release_tag: str       # CODEFYUI_RELEASE_TAG as the installer sees it
    force_build: bool      # CODEFYUI_FORCE_BUILD=1
    outcome: str           # "pinned:<tag>", "main" or "refused"
    offline: bool = False  # True: no request may be made at all
    shape: str = "ps5"     # PowerShell only: which .BaseResponse to fake


_CASES = {
    "api-answers": _Case("2.8.8", None, "latest", False, "pinned:2.8.8"),
    "api-down-page-answers": _Case(None, TAG_URL.format("2.8.8"), "latest", False,
                                   "pinned:2.8.8"),
    "both-down": _Case(None, None, "latest", False, "refused"),
    "no-release-yet": _Case(None, NO_RELEASE_URL, "latest", False, "refused"),
    "explicit-tag": _Case(None, None, "2.8.7", False, "pinned:2.8.7", offline=True),
    "explicit-tag-beats-force-build": _Case(None, None, "2.8.7", True, "pinned:2.8.7",
                                            offline=True),
    "force-build-tracks-main": _Case(None, None, "latest", True, "main", offline=True),
}

_PS_CASES = {
    **_CASES,
    # PowerShell 7 hands back an HttpResponseMessage, 5.1 an HttpWebResponse.
    "api-down-page-answers-pwsh7": _CASES["api-down-page-answers"]._replace(shape="ps7"),
}


def _cut(text: str, pattern: str, what: str) -> str:
    match = re.search(pattern, text, re.M | re.S)
    assert match, f"{what} not found in the installer; update the pattern"
    return match.group(0)


def _cut_helpers(text: str, names: tuple[str, ...], pattern: str) -> str:
    """The installer's one-line message helpers (step, ok, warn, die)."""
    return "".join(_cut(text, pattern.format(re.escape(name)), name) for name in names)


@functools.lru_cache(maxsize=None)
def _working(name: str, *probe: str) -> str | None:
    """The executable if it is on PATH and runs. On Windows `bash` can be the
    WSL launcher with no distribution behind it, which exits non-zero."""
    exe = shutil.which(name)
    if exe is None:
        return None
    try:
        result = subprocess.run([exe, *probe], capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    return exe if result.returncode == 0 else None


def _assert_outcome(case: _Case, proc: subprocess.CompletedProcess) -> None:
    out = proc.stdout.decode("utf-8", "replace")
    err = proc.stderr.decode("utf-8", "replace")
    both = f"--- stdout\n{out}\n--- stderr\n{err}"
    if case.outcome == "refused":
        assert proc.returncode == 1, both
        assert "CODEFYUI_RELEASE_TAG" in out + err, both
        assert "CODEFYUI_FORCE_BUILD" in out + err, both
        assert "PINNED=" not in out, both  # stopped before the clone
    elif case.outcome == "main":
        assert proc.returncode == 0, both
        assert "PINNED=[] RELEASE_TAG=[latest]" in out, both
    else:
        tag = case.outcome.split(":", 1)[1]
        assert proc.returncode == 0, both
        assert f"PINNED=[{tag}] RELEASE_TAG=[{tag}]" in out, both
    # Both ways round, so the stubs' marker is known to arrive when they run.
    assert ("STUB-REQUEST " in err) is not case.offline, both


# fd 3 because install.sh sends curl's stderr to /dev/null.
_SH_STUBS = r"""
exec 3>&2
RED='' GREEN='' YELLOW='' BLUE='' BOLD='' NC=''
curl() {
  local a url=""
  for a in "$@"; do
    case "$a" in https://*) url="$a" ;; esac
  done
  echo "STUB-REQUEST $url" >&3
  case "$url" in
    https://api.github.com/repos/CodefyUI/CodefyUI/releases/latest)
      [[ -n "$FAKE_API_TAG" ]] || return 22
      printf '{\n  "url": "x",\n  "tag_name": "%s",\n  "name": "%s"\n}\n' \
        "$FAKE_API_TAG" "$FAKE_API_TAG" ;;
    https://github.com/CodefyUI/CodefyUI/releases/latest)
      # Like real curl -f -w: the URL is printed even when the request fails.
      if [[ -n "$FAKE_FINAL_URL" ]]; then printf '%s' "$FAKE_FINAL_URL"; return 0; fi
      printf '%s' "$url"; return 22 ;;
    *) return 6 ;;
  esac
}
"""


def _sh_harness(text: str, case: _Case) -> str:
    text = text.replace("\r\n", "\n")
    assert "\nset -euo pipefail\n" in text  # the harness below assumes it
    return "\n".join([
        "set -euo pipefail",
        f"FAKE_API_TAG={shlex.quote(case.api_tag or '')}",
        f"FAKE_FINAL_URL={shlex.quote(case.final_url or '')}",
        'RELEASE_REPO="CodefyUI/CodefyUI"',
        f"RELEASE_TAG={shlex.quote(case.release_tag)}",
        f"FORCE_BUILD={'1' if case.force_build else '0'}",
        _SH_STUBS,
        _cut_helpers(text, ("step", "ok", "warn", "die"), r"^{}\(\) +\{{[^\n]*\n"),
        _cut(text, r"^resolve_release_tag\(\) \{\n.*?^\}\n", "resolve_release_tag()"),
        _cut(text, r'^PINNED_TAG=""\n.*?^fi\n', "the tag decision block"),
        'echo "PINNED=[$PINNED_TAG] RELEASE_TAG=[$RELEASE_TAG]"',
    ]) + "\n"


@pytest.mark.parametrize("name", list(_CASES))
def test_install_sh_tag_decision(name):
    bash = _working("bash", "-c", "exit 0")
    if bash is None:
        pytest.skip("no working bash here")
    case = _CASES[name]
    # Bytes in, so no CRLF is added on Windows; bash reads the script from stdin.
    proc = subprocess.run([bash, "-s"], input=_sh_harness(_read("install.sh"), case).encode(),
                          capture_output=True, timeout=120)
    _assert_outcome(case, proc)


_PS_STUBS = r"""
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
function Write-Net($what) { [Console]::Error.WriteLine("STUB-REQUEST $what") }
function Invoke-RestMethod {
    param([switch]$UseBasicParsing, $Uri, $TimeoutSec, $Headers)
    Write-Net $Uri
    if ($FakeApiTag) { return [pscustomobject]@{ tag_name = $FakeApiTag; name = $FakeApiTag } }
    throw 'The remote server returned an error: (403) Forbidden.'
}
function Invoke-WebRequest {
    param([switch]$UseBasicParsing, $Method, $Uri, $TimeoutSec, $Headers)
    Write-Net "$Method $Uri"
    if (-not $FakeFinalUrl) { throw 'Unable to connect to the remote server' }
    $final = [uri]$FakeFinalUrl
    if ($FakeShape -eq 'ps7') {
        return [pscustomobject]@{ BaseResponse = [pscustomobject]@{
            RequestMessage = [pscustomobject]@{ RequestUri = $final } } }
    }
    return [pscustomobject]@{ BaseResponse = [pscustomobject]@{ ResponseUri = $final } }
}
"""


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _ps_harness(text: str, case: _Case) -> str:
    text = text.replace("\r\n", "\n")
    assert "\n$ErrorActionPreference = 'Stop'\n" in text  # the stubs assume it
    return "\n".join([
        _PS_STUBS,
        f"$FakeApiTag = {_ps_quote(case.api_tag or '')}",
        f"$FakeFinalUrl = {_ps_quote(case.final_url or '')}",
        f"$FakeShape = {_ps_quote(case.shape)}",
        "$ReleaseRepo = 'CodefyUI/CodefyUI'",
        f"$ReleaseTag = {_ps_quote(case.release_tag)}",
        f"$ForceBuild = {'$true' if case.force_build else '$false'}",
        _cut_helpers(text, ("Step", "Ok", "Warn", "Die"), r"^function {}\(\$msg\)[^\n]*\n"),
        _cut(text, r"^function Resolve-ReleaseTag \{\n.*?^\}\n", "function Resolve-ReleaseTag"),
        _cut(text, r"^\$PinnedTag = \$null\n.*?^\}\n", "the tag decision block"),
        'Write-Host "PINNED=[$PinnedTag] RELEASE_TAG=[$ReleaseTag]"',
    ]) + "\n"


@pytest.mark.parametrize("name", list(_PS_CASES))
def test_install_ps1_tag_decision(name, tmp_path):
    # Windows PowerShell 5.1 first: it is what the README one-liner runs.
    ps = (_working("powershell", "-NoProfile", "-NonInteractive", "-Command", "exit 0")
          or _working("pwsh", "-NoProfile", "-NonInteractive", "-Command", "exit 0"))
    if ps is None:
        pytest.skip("no PowerShell here")
    case = _PS_CASES[name]
    script = tmp_path / "tag_decision.ps1"
    # BOM: Windows PowerShell 5.1 reads a BOM-less script in the ANSI code page.
    script.write_text(_ps_harness(_read("install.ps1"), case), encoding="utf-8-sig")
    env = {**os.environ, "POWERSHELL_TELEMETRY_OPTOUT": "1", "POWERSHELL_UPDATECHECK": "Off"}
    proc = subprocess.run([ps, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                           "-File", str(script)], capture_output=True, timeout=120, env=env)
    _assert_outcome(case, proc)
