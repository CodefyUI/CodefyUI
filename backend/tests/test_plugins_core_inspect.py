"""The half of a plugin install that happens before anybody says yes.

Reading a source, deciding what has already been consented to, and moving
bytes off GitHub -- all three used to live in ``scripts/plugins.py``, where
the only way to reach them was to run the CLI. These tests exercise them as
what they now are: functions the server can call, with no terminal, no
lockfile on disk and no network.

Nothing here touches GitHub. An autouse fixture replaces ``urlopen`` in the
GitHub client with one that fails the test if it is ever reached, and the
two suites that need a transport (``_gh_get`` and ``download_tarball``)
install their own fake over the top of it. A test in this file that starts
making real requests fails loudly rather than becoming slow and flaky in
somebody else's CI run.
"""

from __future__ import annotations

import http.client
import io
import tarfile
import urllib.request
from pathlib import Path
from textwrap import dedent
from urllib.error import HTTPError, URLError

import pytest

from app.core.plugins import catalog as catalog_module
from app.core.plugins import consent, github
from app.core.plugins import inspect as plugin_inspect
from app.core.plugins.errors import (
    ConsentRequired,
    GitHubError,
    ManifestError,
    NotInstalled,
    NotUpdatable,
    PluginCancelled,
    PluginInstallError,
    ReservedPluginId,
)


# ── fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail any test that reaches an actual socket.

    The functions under test are the ones that talk to GitHub, so "we
    remembered to fake it" cannot be left to each test to get right.
    """
    def _explode(*_a, **_k):  # pragma: no cover - only runs on a bug
        raise AssertionError("a test in this file tried to reach the network")

    monkeypatch.setattr(github.urllib.request, "urlopen", _explode)


def _tarball_of(files: dict[str, str], dest: Path) -> None:
    """Pack ``{path inside the archive: text}`` into a gzipped tar."""
    with tarfile.open(dest, "w:gz") as tf:
        for rel, text in files.items():
            data = text.encode("utf-8")
            member = tarfile.TarInfo(rel)
            member.size = len(data)
            tf.addfile(member, io.BytesIO(data))


@pytest.fixture
def fake_github(monkeypatch):
    """Serve one commit -- a sha, a manifest and a tarball -- with no network."""
    def _make(manifest_text: str, *, sha: str = "a" * 40, files=None):
        monkeypatch.setattr(github, "resolve_sha", lambda o, r, ref: sha)
        monkeypatch.setattr(
            github, "fetch_manifest_text", lambda o, r, s: manifest_text
        )
        payload = {"cdui.plugin.toml": manifest_text, **(files or {})}
        monkeypatch.setattr(
            github,
            "download_tarball",
            lambda owner, repo, s, dest, **kw: _tarball_of(
                {f"{repo}-main/{rel}": text for rel, text in payload.items()}, dest
            ),
        )
        return sha

    return _make


PLAIN_MANIFEST = dedent("""\
    [plugin]
    id = "extras"
    name = "Extras"
    version = "1.2.0"
    description = "A third-party pack."
    schema_version = 1
    """)

GREEDY_MANIFEST = dedent("""\
    [plugin]
    id = "extras"
    name = "Extras"
    version = "2.0.0"
    schema_version = 1

    [security]
    capabilities = ["network", "filesystem"]
    allowed_modules = ["subprocess"]

    [frontend]
    entry = "dist/index.js"

    [lessons]
    chapters = ["I9"]
    lessons = ["I9-1"]
    """)


def _installed(**overrides) -> dict:
    entry = {
        "source_kind": "github_url",
        "source": "alice/extras",
        "url": "https://github.com/alice/extras",
        "ref": "",
        "sha": "b" * 40,
        "manifest": {"id": "extras", "version": "1.0.0"},
        "capabilities": ["network"],
        "trusted_modules": [],
        "enabled": True,
    }
    entry.update(overrides)
    return {"plugins": {"extras": entry}}


# ── inspecting a pack that ships with the release ──────────────────────────

def test_a_builtin_pack_is_official_and_asks_nobody_for_consent():
    """A pack that arrived with the release came through a pull request in
    this repository -- there is no third party here to be asked about."""
    found = plugin_inspect.inspect_builtin("stats", lockfile={})
    assert found.kind == "builtin"
    assert found.mode == "install"
    assert found.plugin_id == "stats"
    assert found.catalog_id == "stats"
    assert found.official is True
    assert found.consent_required is False
    assert found.python_deps == {}
    assert found.warnings == ()
    assert found.url is None and found.sha is None
    assert found.name and found.version and found.description


def test_a_builtin_packs_dependency_is_reported_without_installing_it():
    """``edu`` needs model2vec, and the person deciding whether to install it
    is entitled to know that before a pip runs, not after."""
    found = plugin_inspect.inspect_builtin("edu", lockfile={})
    assert found.python_deps == {"model2vec": ">=0.8.0"}
    assert found.chapters == ("I1", "I2")
    assert found.lessons[:2] == ("I1-1", "I1-2")


def test_a_builtin_pack_in_the_lockfile_reads_as_an_up_to_date_update():
    """No sha on either side, and that is the honest answer: the copy on
    disk IS the copy that was installed."""
    lockfile = {"plugins": {"stats": {"source_kind": "builtin", "enabled": True}}}
    found = plugin_inspect.inspect_builtin("stats", lockfile=lockfile)
    assert found.mode == "update"
    assert found.up_to_date is True
    assert found.installed == {
        "sha": None,
        "version": "",
        "capabilities": (),
        "trusted_modules": (),
        "enabled": True,
        "source_kind": "builtin",
    }


# ── inspecting a repository ────────────────────────────────────────────────

def test_a_repository_is_read_at_the_resolved_sha(fake_github):
    sha = fake_github(PLAIN_MANIFEST)
    found = plugin_inspect.inspect_github("alice", "extras", "v1.2.0", lockfile={})
    assert found.kind == "github"
    assert found.mode == "install"
    assert found.plugin_id == "extras"
    assert found.sha == sha
    assert found.ref == "v1.2.0"
    assert found.url == "https://github.com/alice/extras"
    assert found.source == "alice/extras@v1.2.0"
    assert found.version == "1.2.0"
    assert found.consent_required is True
    assert found.catalog_id is None and found.official is False
    assert found.installed is None and found.up_to_date is False


def test_an_inspection_keeps_the_repository_apart_from_the_url(fake_github):
    """An install takes an owner and a repo; ``source`` and ``url`` are
    formatted strings. Recovering the pair by re-parsing one of those is how
    two spellings of the same repository start to drift apart."""
    fake_github(PLAIN_MANIFEST)
    found = plugin_inspect.inspect_github("alice", "extras", "v1.2.0", lockfile={})
    assert (found.owner, found.repo) == ("alice", "extras")


def test_a_pack_that_ships_here_names_no_repository():
    """``None`` because there is nothing to resolve or download, not because
    the field was forgotten."""
    found = plugin_inspect.inspect_builtin("stats", lockfile={})
    assert found.owner is None and found.repo is None


def test_a_pinned_sha_is_never_re_resolved(monkeypatch, fake_github):
    """The restore path: re-resolving a tag that has since moved would
    install something other than what was pinned."""
    fake_github(PLAIN_MANIFEST)

    def _explode(*_a):  # pragma: no cover - only runs on a bug
        raise AssertionError("a pinned install must not resolve the ref")

    monkeypatch.setattr(github, "resolve_sha", _explode)
    found = plugin_inspect.inspect_github(
        "alice", "extras", "v1.2.0", lockfile={}, pinned_sha="c" * 40
    )
    assert found.sha == "c" * 40


def test_everything_a_greedy_plugin_asks_for_is_reported_with_both_warnings(
    fake_github,
):
    fake_github(GREEDY_MANIFEST)
    found = plugin_inspect.inspect_github("alice", "extras", "", lockfile={})
    # Sorted, not manifest order: ``normalize_capabilities`` is what a
    # hand-written list goes through, and it de-duplicates and sorts.
    assert found.capabilities == ("filesystem", "network")
    assert found.allowed_modules == ("subprocess",)
    assert found.has_frontend is True
    assert found.chapters == ("I9",) and found.lessons == ("I9-1",)
    assert found.consent_required is True
    assert found.warnings == (
        plugin_inspect.FRONTEND_WARNING,
        plugin_inspect.ALLOWED_MODULES_WARNING.format(modules="subprocess"),
    )
    assert "subprocess" in found.warnings[1]


def test_a_manifest_with_no_plugin_table_is_refused(fake_github):
    fake_github('name = "not a manifest"\n')
    with pytest.raises(ManifestError):
        plugin_inspect.inspect_github("alice", "extras", "", lockfile={})


@pytest.mark.parametrize("plugin_id", ["edu", "catalog"])
def test_a_repository_may_not_claim_a_reserved_id(plugin_id, fake_github):
    """``edu`` is a pack this build ships and ``catalog`` is a route under
    /api/plugins/; either would be decided by the filesystem or the router
    rather than by the install."""
    fake_github(
        f'[plugin]\nid = "{plugin_id}"\nversion = "1"\nschema_version = 1\n'
    )
    with pytest.raises(PluginInstallError) as excinfo:
        plugin_inspect.inspect_github("alice", "extras", "", lockfile={})
    assert plugin_id in str(excinfo.value)


def test_the_reserved_id_refusal_carries_the_id_and_what_holds_it(fake_github):
    """Two audiences, one exception. The CLI prints the sentence and the hint,
    so both are asserted here word for word -- they are what a terminal user
    reads. A panel cannot show a sentence and draw a control from it, so the
    id and its holder travel as ATTRIBUTES: the route used to recover the id
    with a regular expression over the message, which made this wording part
    of the HTTP contract and would have turned every reserved id into
    ``invalid_manifest`` the day somebody rephrased it."""
    fake_github('[plugin]\nid = "catalog"\nversion = "1"\nschema_version = 1\n')

    with pytest.raises(ReservedPluginId) as excinfo:
        plugin_inspect.inspect_github("alice", "extras", "", lockfile={})
    refusal = excinfo.value

    assert refusal.plugin_id == "catalog"
    assert refusal.taken_by == catalog_module.RESERVED_BY_ROUTE
    assert str(refusal) == "Plugin id 'catalog' is reserved by this build."
    assert refusal.hint == (
        "alice/extras declares an id that names a route under /api/plugins/; "
        "it cannot be installed under that id."
    )
    # Still the base class every existing caller catches -- ``_install_github``
    # catches ``RuntimeError`` around this call and prints what it gets.
    assert isinstance(refusal, PluginInstallError)


def test_a_fork_may_not_claim_the_catalog_id_of_an_official_plugin(fake_github):
    """The third clause of the reserved-id rule, which the CLI enforced and
    this did not. ``self-learning`` is a ``github`` catalog row: its id is
    not reserved outright -- its own author installs the repository under it
    -- so an id-only check passes ``mallory/evil`` straight through, and the
    id is what the lockfile, the catalog card and ``/api/plugins/{id}`` all
    key on."""
    fake_github(
        '[plugin]\nid = "self-learning"\nversion = "1"\nschema_version = 1\n'
    )
    with pytest.raises(PluginInstallError) as excinfo:
        plugin_inspect.inspect_github("mallory", "evil", "", lockfile={})
    hint = excinfo.value.hint or ""
    assert "CodefyUI/CodefyUI-Plugin-Self-Learning" in hint
    assert "mallory/evil" in hint


def test_the_repository_the_catalog_names_may_claim_its_own_id(fake_github):
    """The other half, and the reason the clause is about the repository
    rather than the id: refusing every catalog id would make the official
    pack the one thing the catalog advertises and nobody can install."""
    fake_github(
        '[plugin]\nid = "self-learning"\nversion = "1"\nschema_version = 1\n'
    )
    found = plugin_inspect.inspect_github(
        "CodefyUI", "CodefyUI-Plugin-Self-Learning", "", lockfile={}
    )
    assert found.plugin_id == "self-learning"
    # Case-insensitively, because GitHub owners and repositories are.
    assert plugin_inspect.inspect_github(
        "codefyui", "codefyui-plugin-self-learning", "", lockfile={}
    ).plugin_id == "self-learning"


# ── an update compared against what was consented to ───────────────────────

def test_an_update_names_the_capability_this_version_added(fake_github):
    fake_github(GREEDY_MANIFEST, sha="d" * 40)
    found = plugin_inspect.inspect_github(
        "alice", "extras", "", lockfile=_installed()
    )
    assert found.mode == "update"
    assert found.up_to_date is False
    assert found.installed["sha"] == "b" * 40
    assert found.installed["version"] == "1.0.0"
    assert found.installed["capabilities"] == ("network",)
    # ``network`` was already granted; ``filesystem`` and the module are new.
    assert found.capabilities_added == ("filesystem",)
    assert found.allowed_modules_added == ("subprocess",)


def test_an_unchanged_sha_reads_as_up_to_date(fake_github):
    fake_github(PLAIN_MANIFEST, sha="b" * 40)
    found = plugin_inspect.inspect_github(
        "alice", "extras", "", lockfile=_installed()
    )
    assert found.up_to_date is True
    assert found.capabilities_added == ()


def test_the_lockfile_is_what_says_what_was_granted(fake_github):
    """A plugin that rewrote its manifest after install must not be shown as
    if the new ask is what the user agreed to."""
    fake_github(GREEDY_MANIFEST)
    found = plugin_inspect.inspect_github(
        "alice", "extras", "", lockfile=_installed(capabilities=[])
    )
    assert found.installed["capabilities"] == ()
    assert found.capabilities_added == ("filesystem", "network")


# ── dispatch: a typed source, and an installed one ─────────────────────────

def test_a_catalog_name_inspects_the_pack_that_ships_here():
    found = plugin_inspect.inspect_source("stats", lockfile={})
    assert found.kind == "builtin" and found.plugin_id == "stats"


def test_a_catalog_github_entry_carries_its_id_and_official_flag(
    monkeypatch, fake_github
):
    """The catalog is the only thing entitled to call a pack official, so
    that flag has to survive the trip through parse_source."""
    monkeypatch.setattr(
        catalog_module,
        "load_catalog",
        lambda *a, **k: {"schema": 1, "plugins": {"extras": {
            "kind": "github",
            "name": "Extras",
            "repo": "alice/extras",
            "ref": "v1.2.0",
            "official": True,
        }}},
    )
    fake_github(PLAIN_MANIFEST)
    found = plugin_inspect.inspect_source("extras", lockfile={})
    assert found.kind == "github"
    assert found.catalog_id == "extras"
    assert found.official is True
    assert found.ref == "v1.2.0"


def test_free_text_is_never_official(fake_github):
    fake_github(PLAIN_MANIFEST)
    found = plugin_inspect.inspect_source("alice/extras@v1.2.0", lockfile={})
    assert found.official is False
    assert found.catalog_id is None


def test_an_installed_plugin_is_re_inspected_from_its_recorded_source(
    fake_github,
):
    fake_github(PLAIN_MANIFEST, sha="e" * 40)
    found = plugin_inspect.inspect_installed("extras", lockfile=_installed())
    assert found.mode == "update"
    assert found.url == "https://github.com/alice/extras"
    assert found.sha == "e" * 40
    # A repository the catalog never heard of stays what it is.
    assert found.catalog_id is None and found.official is False


SELF_LEARNING_REPO = "CodefyUI/CodefyUI-Plugin-Self-Learning"
SELF_LEARNING_MANIFEST = dedent("""\
    [plugin]
    id = "self-learning"
    name = "Self-Learning"
    version = "1.1.0"
    schema_version = 1
    """)


def _installed_official(**overrides) -> dict:
    entry = {
        "source_kind": "github_url",
        "source": SELF_LEARNING_REPO,
        "url": f"https://github.com/{SELF_LEARNING_REPO}",
        "ref": "",
        "sha": "b" * 40,
        "manifest": {"id": "self-learning", "version": "1.0.0"},
        "capabilities": [],
        "trusted_modules": [],
        "enabled": True,
    }
    entry.update(overrides)
    return {"plugins": {"self-learning": entry}}


def test_an_update_of_a_catalog_pack_still_knows_it_is_the_catalog_pack(
    fake_github,
):
    """The recorded ``catalog_id`` is what ``cdui plugin install <name>``
    wrote down precisely so a later reader can tell the catalog's own pack
    from free text carrying the same id. Dropping it on re-inspection made
    every update of an official plugin read as third-party."""
    fake_github(SELF_LEARNING_MANIFEST, sha="f" * 40)
    found = plugin_inspect.inspect_installed(
        "self-learning", lockfile=_installed_official(catalog_id="self-learning")
    )
    assert found.mode == "update"
    assert found.catalog_id == "self-learning"
    assert found.official is True


def test_the_badge_is_re_derived_when_the_install_recorded_no_catalog_id(
    fake_github,
):
    """Installed by URL rather than by name -- the same code from the same
    repository by a longer road, and ``is_official``'s repository match is
    what says so. The panel's badge and this dialog must not disagree."""
    fake_github(SELF_LEARNING_MANIFEST, sha="f" * 40)
    found = plugin_inspect.inspect_installed(
        "self-learning", lockfile=_installed_official()
    )
    assert found.catalog_id is None, "nothing was recorded, so nothing is claimed"
    assert found.official is True


@pytest.mark.parametrize(
    "plugin_id, lockfile",
    [
        ("nothing", {"plugins": {}}),
        ("extras", {"plugins": {"extras": {"source_kind": "builtin"}}}),
        ("extras", {"plugins": {"extras": {"source_kind": "github_url"}}}),
    ],
)
def test_a_plugin_with_no_repository_to_update_from_says_so(plugin_id, lockfile):
    with pytest.raises(PluginInstallError):
        plugin_inspect.inspect_installed(plugin_id, lockfile=lockfile)


def test_a_plugin_nobody_has_is_a_different_refusal_from_one_that_cannot_update():
    """Two facts, two classes. A route answers them with different statuses
    -- 404 for a plugin this install does not have, 400 for one that has no
    repository behind it -- and both used to arrive as a bare
    ``PluginInstallError``, which left the caller choosing a status code by
    matching on an English sentence."""
    with pytest.raises(NotInstalled) as missing:
        plugin_inspect.inspect_installed("nothing", lockfile={"plugins": {}})
    assert missing.value.plugin_id == "nothing"

    with pytest.raises(NotUpdatable) as builtin:
        plugin_inspect.inspect_installed(
            "extras", lockfile={"plugins": {"extras": {"source_kind": "builtin"}}}
        )
    assert builtin.value.source_kind == "builtin"
    # The hint is the whole answer for a built-in pack: it updates, just not
    # from here.
    assert "cdui update" in builtin.value.hint

    with pytest.raises(NotUpdatable) as linked:
        plugin_inspect.inspect_installed(
            "extras", lockfile={"plugins": {"extras": {"source_kind": "local"}}}
        )
    assert linked.value.source_kind == "local"

    with pytest.raises(NotUpdatable) as nameless:
        plugin_inspect.inspect_installed(
            "extras", lockfile={"plugins": {"extras": {"source_kind": "github_url"}}}
        )
    # Installed from a repository, and the record no longer says which.
    assert nameless.value.source_kind == "github_url"


def test_updatable_entry_hands_back_the_record_it_approved():
    """What ``PluginService.update`` asks before it goes anywhere near the
    network -- the same rule ``inspect_installed`` applies on the thread."""
    entry = {"source_kind": "github_url", "url": "https://github.com/alice/extras"}
    approved = plugin_inspect.updatable_entry(
        "extras", lockfile={"plugins": {"extras": entry}}
    )
    assert approved is entry


# ── the GitHub client ──────────────────────────────────────────────────────

class _FakeResponse:
    """Just enough of an HTTP response for a streamed read."""

    def __init__(self, chunks: list[bytes], headers: dict | None = None):
        self._chunks = list(chunks)
        self.headers = headers if headers is not None else {}

    def read(self, size: int = -1) -> bytes:
        return self._chunks.pop(0) if self._chunks else b""

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _BrokenResponse:
    """A response that dies while its body is being read.

    The one shape ``urlopen``'s own translation cannot cover: the request
    succeeded, so nothing raised at ``urlopen`` time, and the connection
    breaks afterwards -- which is what a reset peer, a TLS error mid-stream
    and a read timeout all look like from here.
    """

    def __init__(self, exc: BaseException):
        self._exc = exc
        self.headers: dict = {}

    def read(self, size: int = -1) -> bytes:
        raise self._exc

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _http_error(code: int, body: bytes = b"", reason: str = "Some Reason"):
    return HTTPError(
        "https://api.github.com/x", code, reason, {}, io.BytesIO(body)
    )


@pytest.fixture
def fake_urlopen(monkeypatch):
    """Install a fake transport and hand back the requests it saw."""
    seen: list = []

    def _install(result):
        def _urlopen(req, timeout=None):
            seen.append(req)
            if isinstance(result, Exception):
                raise result
            return result() if callable(result) else result

        monkeypatch.setattr(github.urllib.request, "urlopen", _urlopen)
        return seen

    return _install


def test_the_token_is_sent_only_when_the_environment_has_one(
    monkeypatch, fake_urlopen
):
    """60 requests an hour is what a classroom behind one NAT gets without a
    token -- and a token in a header nobody asked for is a leak."""
    monkeypatch.delenv("CODEFYUI_GITHUB_TOKEN", raising=False)
    seen = fake_urlopen(lambda: _FakeResponse([b"{}"]))
    github._gh_get("https://api.github.com/x")
    assert seen[-1].get_header("Authorization") is None
    assert seen[-1].get_header("User-agent") == github.USER_AGENT

    monkeypatch.setenv("CODEFYUI_GITHUB_TOKEN", "ghp_secret")
    github._gh_get("https://api.github.com/x")
    assert seen[-1].get_header("Authorization") == "Bearer ghp_secret"


def test_the_token_does_not_follow_a_redirect_to_another_host(monkeypatch):
    """``codeload.github.com`` redirects a tarball to
    ``objects.githubusercontent.com`` as a matter of course, and urllib's
    redirect handler rebuilds the next request out of ``req.headers``
    verbatim -- Python 3.11 strips no credentials. Sent as an UNREDIRECTED
    header, the token goes on THIS request and on nothing built out of it.

    Driven through urllib's own ``HTTPRedirectHandler`` rather than a mock of
    it: the thing being pinned is what the library does with the request this
    module hands it.
    """
    monkeypatch.setenv("CODEFYUI_GITHUB_TOKEN", "ghp_secret")
    req = github._request("https://codeload.github.com/alice/extras/tar.gz/f")

    # The first request carries it: unredirected headers are still sent.
    assert req.get_header("Authorization") == "Bearer ghp_secret"
    assert req.unredirected_hdrs["Authorization"] == "Bearer ghp_secret"

    forwarded = urllib.request.HTTPRedirectHandler().redirect_request(
        req, None, 302, "Found", {},
        "https://objects.githubusercontent.com/somewhere-else",
    )
    assert forwarded.get_header("Authorization") is None
    assert "ghp_secret" not in str(forwarded.headers)
    assert "ghp_secret" not in str(forwarded.unredirected_hdrs)
    # The user agent is not a credential and still travels.
    assert forwarded.get_header("User-agent") == github.USER_AGENT


@pytest.mark.parametrize("broken", [
    ConnectionResetError(104, "Connection reset by peer"),
    OSError(5, "Input/output error"),
    TimeoutError("timed out"),
])
def test_a_connection_that_breaks_mid_body_is_a_transport_failure(
    fake_urlopen, broken
):
    """``urlopen`` returned, so its own translation is behind us: the read is
    where a reset peer, a TLS error and a stream timeout arrive, as a bare
    ``OSError``. Uncaught, one of those left this module as itself and
    reached callers that read an ``OSError`` as "the file is not there" --
    the Plugin Center's ``/inspect`` answers 400 `invalid_manifest` for one,
    which reports a dropped connection as a manifest that is not a manifest.
    """
    fake_urlopen(lambda: _BrokenResponse(broken))

    with pytest.raises(GitHubError) as excinfo:
        github._gh_get("https://api.github.com/x")
    assert excinfo.value.status is None, "it never got an HTTP status"

    # And through the capped read, which is the one an inspection makes.
    fake_urlopen(lambda: _BrokenResponse(broken))
    with pytest.raises(GitHubError):
        github.fetch_manifest_text("alice", "extras", "f" * 40)


def test_a_manifest_larger_than_the_cap_is_refused_rather_than_read(
    monkeypatch, fake_urlopen
):
    """The manifest is read before anybody has agreed to anything, and the
    server keeps a whole one per stored inspection -- so an unbounded read
    lets the file at the other end decide how much memory a repository
    nobody trusts occupies, once per inspection."""
    monkeypatch.setattr(github, "MAX_MANIFEST_BYTES", 8)
    fake_urlopen(lambda: _FakeResponse([b"[plugin]\nid = 'extras'\n"]))

    with pytest.raises(ManifestError) as excinfo:
        github.fetch_manifest_text("alice", "extras", "f" * 40)
    assert "larger than" in str(excinfo.value)

    monkeypatch.setattr(github, "MAX_MANIFEST_BYTES", 1024)
    fake_urlopen(lambda: _FakeResponse([PLAIN_MANIFEST.encode("utf-8")]))
    assert github.fetch_manifest_text("alice", "extras", "f" * 40) == PLAIN_MANIFEST


def test_a_missing_repository_and_a_broken_github_are_different_answers(
    fake_urlopen,
):
    """404 is a typo in the repo name and 502 is GitHub; the status is the
    only thing that tells the caller which sentence to show."""
    fake_urlopen(_http_error(404, b'{"message": "Not Found"}', "Not Found"))
    with pytest.raises(GitHubError) as excinfo:
        github.resolve_sha("alice", "nope", "")
    assert excinfo.value.status == 404
    assert "alice/nope@HEAD" in str(excinfo.value)
    assert "Not Found" in str(excinfo.value)

    fake_urlopen(_http_error(502, b"<html>bad gateway</html>", "Bad Gateway"))
    with pytest.raises(GitHubError) as excinfo:
        github.resolve_sha("alice", "extras", "")
    assert excinfo.value.status == 502
    assert "Bad Gateway" in str(excinfo.value), "falls back to the reason phrase"


def test_a_rate_limited_install_is_told_it_was_rate_limited(fake_urlopen):
    """GitHub's reason phrase for this is "Forbidden", which reads as a
    permissions problem. Its body says what actually happened."""
    fake_urlopen(
        _http_error(
            403,
            b'{"message": "API rate limit exceeded for 1.2.3.4."}',
            "Forbidden",
        )
    )
    with pytest.raises(GitHubError) as excinfo:
        github.resolve_sha("alice", "extras", "")
    assert excinfo.value.status == 403
    assert "rate limit" in str(excinfo.value)


def test_a_request_that_never_reached_github_has_no_status(fake_urlopen):
    fake_urlopen(URLError("getaddrinfo failed"))
    with pytest.raises(GitHubError) as excinfo:
        github.resolve_sha("alice", "extras", "")
    assert excinfo.value.status is None
    assert "getaddrinfo failed" in str(excinfo.value)


def test_a_200_that_is_not_json_is_a_github_failure_with_no_status(fake_urlopen):
    """A captive portal, a school proxy's block page and a transparent MITM
    all answer "200 OK" with HTML. ``json.loads`` raises a ``ValueError`` for
    that, which no caller of ``resolve_sha`` catches -- the CLI's
    ``except RuntimeError`` included -- so it would have been a traceback on
    the first school network that intercepts GitHub."""
    fake_urlopen(lambda: _FakeResponse([b"<html>Access denied</html>"]))
    with pytest.raises(GitHubError) as excinfo:
        github.resolve_sha("alice", "extras", "v1")
    assert excinfo.value.status is None, "whatever answered was not GitHub"
    assert "did not return JSON" in str(excinfo.value)
    assert "alice/extras@v1" in str(excinfo.value)


def test_a_url_http_client_will_not_send_is_a_github_failure(fake_urlopen):
    """``InvalidURL`` is an ``http.client.HTTPException``, which is not an
    ``OSError`` and so not a ``URLError`` -- it escaped the whole translation
    layer. It is what a ref with a space or a control character in it becomes
    by the time ``putrequest`` sees the URL."""
    fake_urlopen(http.client.InvalidURL("URL can't contain control characters"))
    with pytest.raises(GitHubError) as excinfo:
        github.resolve_sha("alice", "extras", "v 1")
    assert excinfo.value.status is None
    assert "control characters" in str(excinfo.value)


def test_a_commit_response_without_a_sha_is_refused(fake_urlopen):
    fake_urlopen(lambda: _FakeResponse([b'{"documentation_url": "..."}']))
    with pytest.raises(GitHubError) as excinfo:
        github.resolve_sha("alice", "extras", "v1")
    assert "missing 'sha'" in str(excinfo.value)


def test_the_manifest_is_fetched_from_the_commit_not_the_branch(fake_urlopen):
    seen = fake_urlopen(lambda: _FakeResponse([PLAIN_MANIFEST.encode("utf-8")]))
    text = github.fetch_manifest_text("alice", "extras", "f" * 40)
    assert text == PLAIN_MANIFEST
    assert seen[-1].full_url == (
        f"https://raw.githubusercontent.com/alice/extras/{'f' * 40}/"
        "cdui.plugin.toml"
    )


# ── downloading ────────────────────────────────────────────────────────────

def _chunks(count: int, size: int = 8) -> list[bytes]:
    return [bytes([65 + i % 26]) * size for i in range(count)]


def test_a_download_reports_progress_at_both_ends_and_throttles_between(
    monkeypatch, fake_urlopen, tmp_path
):
    """A bar that stops at 97% because the download finished inside a
    throttle window is worse than no bar."""
    monkeypatch.setattr(github, "PROGRESS_MIN_INTERVAL_S", 3600.0)
    body = _chunks(4)
    fake_urlopen(lambda: _FakeResponse(body, {"Content-Length": "32"}))
    seen: list[tuple[int, int | None]] = []
    dest = tmp_path / "src.tar.gz"

    github.download_tarball(
        "alice", "extras", "f" * 40, dest, progress=lambda d, t: seen.append((d, t))
    )
    assert seen == [(0, 32), (32, 32)], "first and last are forced past the throttle"
    assert dest.read_bytes() == b"".join(body)


def test_every_chunk_reports_when_nothing_is_throttled(
    monkeypatch, fake_urlopen, tmp_path
):
    monkeypatch.setattr(github, "PROGRESS_MIN_INTERVAL_S", 0.0)
    fake_urlopen(lambda: _FakeResponse(_chunks(3)))
    seen: list[tuple[int, int | None]] = []

    github.download_tarball(
        "alice", "extras", "f" * 40, tmp_path / "src.tar.gz",
        progress=lambda d, t: seen.append((d, t)),
    )
    assert seen == [(0, None), (8, None), (16, None), (24, None), (24, None)]
    assert all(total is None for _, total in seen), "no Content-Length, no total"


def test_a_cancelled_download_stops_between_chunks_and_leaves_nothing(
    fake_urlopen, tmp_path
):
    """Stop has to be felt inside the download. A check that only ran between
    whole downloads would not be reached until the one being stopped had
    finished."""
    fake_urlopen(lambda: _FakeResponse(_chunks(50)))
    dest = tmp_path / "src.tar.gz"
    calls = {"n": 0}

    def _cancel() -> bool:
        calls["n"] += 1
        return calls["n"] >= 2

    with pytest.raises(PluginCancelled):
        github.download_tarball(
            "alice", "extras", "f" * 40, dest, cancel_check=_cancel
        )
    assert not dest.exists(), "half a tarball must not be left where a caller looks"


def test_a_download_over_the_cap_is_refused_by_name(
    monkeypatch, fake_urlopen, tmp_path
):
    monkeypatch.setattr(github, "MAX_TARBALL_BYTES", 1024 * 1024)
    fake_urlopen(lambda: _FakeResponse(_chunks(2, size=700 * 1024)))
    dest = tmp_path / "src.tar.gz"

    with pytest.raises(PluginInstallError) as excinfo:
        github.download_tarball("alice", "extras", "f" * 40, dest)
    assert "1 MB" in str(excinfo.value)
    assert "alice/extras" in (excinfo.value.hint or "")
    assert not dest.exists()


def test_a_download_that_fails_is_reported_as_a_github_failure(
    fake_urlopen, tmp_path
):
    fake_urlopen(_http_error(404, b'{"message": "Not Found"}', "Not Found"))
    dest = tmp_path / "src.tar.gz"
    with pytest.raises(GitHubError) as excinfo:
        github.download_tarball("alice", "extras", "f" * 40, dest)
    assert excinfo.value.status == 404
    assert not dest.exists()


# ── extracting ─────────────────────────────────────────────────────────────

def test_a_tarball_with_one_root_returns_that_root(tmp_path):
    tar = tmp_path / "src.tar.gz"
    _tarball_of({"extras-main/cdui.plugin.toml": PLAIN_MANIFEST}, tar)
    dest = tmp_path / "out"
    dest.mkdir()
    root = github.extract_tarball(tar, dest)
    assert root == dest / "extras-main"
    assert (root / "cdui.plugin.toml").read_text(encoding="utf-8") == PLAIN_MANIFEST


@pytest.mark.parametrize(
    "files",
    [
        {"a/x.py": "x = 1", "b/y.py": "y = 2"},
        {"loose.py": "x = 1"},
    ],
)
def test_a_tarball_whose_root_cannot_be_guessed_is_refused(files, tmp_path):
    """Two roots is not a plugin whose root can be picked; guessing would
    install whichever one the filesystem happened to list first."""
    tar = tmp_path / "src.tar.gz"
    _tarball_of(files, tar)
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(PluginInstallError):
        github.extract_tarball(tar, dest)


def test_a_tarball_that_cannot_be_read_is_an_install_failure(tmp_path):
    """A truncated download and a file that is not a gzip both arrive as
    ``TarError`` -- a class no caller of an installer catches, which is how
    one used to travel all the way out to a traceback."""
    tar = tmp_path / "src.tar.gz"
    tar.write_bytes(b"this is not a gzip stream")
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(PluginInstallError) as excinfo:
        github.extract_tarball(tar, dest)
    assert "src.tar.gz" in str(excinfo.value)
    assert "retry" in (excinfo.value.hint or "")


def test_a_tarball_caught_escaping_is_not_reported_as_a_read_failure(tmp_path):
    """``tarfile.FilterError`` IS a ``TarError``, so a hostile tarball used to
    come out of here as "tarfile could not read it; retry the install" -- and
    retrying is the one thing nobody should do about an archive that was
    caught trying to write outside the directory it was unpacked into."""
    tar = tmp_path / "src.tar.gz"
    with tarfile.open(tar, "w:gz") as tf:
        data = PLAIN_MANIFEST.encode("utf-8")
        member = tarfile.TarInfo("extras-main/cdui.plugin.toml")
        member.size = len(data)
        tf.addfile(member, io.BytesIO(data))
        link = tarfile.TarInfo("extras-main/passwd")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        tf.addfile(link)
    dest = tmp_path / "out"
    dest.mkdir()

    with pytest.raises(PluginInstallError) as excinfo:
        github.extract_tarball(tar, dest)

    said = str(excinfo.value) + (excinfo.value.hint or "")
    assert "src.tar.gz" in said
    assert "outside the directory it is unpacked into" in said
    assert "retry" not in said.lower(), "nothing about this is transient"


def test_a_tarball_that_unpacks_past_the_cap_is_refused_before_it_is_written(
    monkeypatch, tmp_path
):
    """The download cap is on the COMPRESSED stream, and gzip of a file of
    zeroes runs about 1000:1 -- so it says nothing about what lands on the
    disk, which is where the space runs out."""
    monkeypatch.setattr(github, "MAX_EXTRACTED_BYTES", 1024 * 1024)
    tar = tmp_path / "src.tar.gz"
    _tarball_of({"extras-main/big.bin": "x" * (2 * 1024 * 1024)}, tar)
    dest = tmp_path / "out"
    dest.mkdir()

    with pytest.raises(PluginInstallError) as excinfo:
        github.extract_tarball(tar, dest)
    assert "1 MB" in str(excinfo.value)
    assert list(dest.iterdir()) == [], "the sum is checked before extractall"


# ── the consent arithmetic ─────────────────────────────────────────────────

def test_what_this_answer_covers_and_what_is_still_outstanding():
    decision = consent.decide_capabilities(
        ("network", "filesystem"), accepted=("network",)
    )
    assert decision.granted == ("network",)
    assert decision.missing == ("filesystem",)
    assert decision.reused_prior is False
    assert decision.grew == (), "no earlier install, so nothing has grown"


def test_a_prior_grant_covers_a_capability_and_says_that_it_did():
    decision = consent.decide_capabilities(
        ("network",), prior=("network", "filesystem")
    )
    assert decision.granted == ("network",)
    assert decision.missing == ()
    assert decision.reused_prior is True
    assert decision.grew == ()


def test_a_capability_this_version_added_is_growth_even_before_it_is_decided():
    decision = consent.decide_capabilities(
        ("network", "process-env"), prior=("network",)
    )
    assert decision.grew == ("process-env",)
    assert decision.missing == ("process-env",)
    assert decision.reused_prior is True


def test_no_earlier_install_and_an_empty_one_are_different_questions():
    """``None`` is "never installed"; ``()`` is "installed and granted
    nothing", and only the second makes a first request into growth."""
    assert consent.decide_capabilities(("network",), prior=None).grew == ()
    assert consent.decide_capabilities(("network",), prior=()).grew == ("network",)


def test_the_answer_keeps_the_order_the_manifest_asked_in():
    """A dialog lists these beside the manifest; a re-sorted list reads as a
    different request."""
    decision = consent.decide_capabilities(
        ("process-env", "network", "filesystem"),
        accepted=("filesystem", "process-env"),
    )
    assert decision.granted == ("process-env", "filesystem")
    assert decision.missing == ("network",)


def test_nothing_declared_needs_no_trust():
    assert consent.check_trust((), trust_author=False) is None


def test_trusting_the_author_is_what_unlocks_the_modules():
    assert consent.check_trust(("subprocess",), trust_author=True) is None


def test_an_untrusted_module_request_raises_with_the_whole_list():
    """The caller's next move is to show the list and ask, so the list has to
    be on the exception rather than re-derived from its sentence."""
    with pytest.raises(ConsentRequired) as excinfo:
        consent.check_trust(("subprocess", "ctypes"), trust_author=False)
    assert excinfo.value.allowed_modules == ("subprocess", "ctypes")
    assert excinfo.value.missing_capabilities == ()
    assert "subprocess" in str(excinfo.value)


def test_a_module_the_user_already_trusted_is_not_a_second_decision():
    assert consent.check_trust(
        ("subprocess",), trust_author=False, prior_trusted=["subprocess"]
    ) is None


def test_one_new_module_is_a_new_decision_however_short_the_list_was():
    with pytest.raises(ConsentRequired) as excinfo:
        consent.check_trust(
            ("subprocess", "ctypes"), trust_author=False, prior_trusted=["subprocess"]
        )
    assert excinfo.value.allowed_modules == ("subprocess", "ctypes")
