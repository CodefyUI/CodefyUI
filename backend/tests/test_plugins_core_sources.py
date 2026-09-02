"""The plugin rules, tested where they now live rather than through the CLI.

``test_plugin_cli.py`` already covers all of this through ``plugins.py`` --
and must keep doing so, because that is the proof the extraction changed no
behaviour. What it cannot cover is the half of the contract only a route will
use: a refusal that carries the catalog ids as a tuple instead of a sentence,
a ``CatalogEntry`` with fields a template can interpolate, and the accessors
that answer about a manifest nobody promised was valid.

So this file asks the questions a GUI asks. It leans on the real
``plugins/registry.json`` and the real ``edu`` / ``stats`` manifests wherever
the answer should be true of what this repository actually ships -- a
synthetic fixture would keep passing after someone shipped a catalog entry
the Plugin Center cannot render.
"""

from __future__ import annotations

import logging

import pytest

from app.core import plugin_loader
from app.core.plugins import (
    ConsentRequired,
    GitHubError,
    ManifestError,
    PluginCancelled,
    PluginInstallError,
    PluginNeedsRestart,
    SourceError,
)
from app.core.plugins.catalog import (
    RESERVED_PLUGIN_IDS,
    CatalogEntry,
    catalog_entries,
    catalog_entry,
    github_catalog_packs,
    load_catalog,
    validate_catalog,
)
from app.core.plugins.errors import UnknownCatalogName, UnparseableSource
from app.core.plugins.manifest import (
    manifest_allowed_modules,
    manifest_has_frontend,
    manifest_python_deps,
    read_manifest,
    validate_manifest,
)
from app.core.plugins.sources import ParsedSource, parse_github_url, parse_source

CATALOG_LOGGER = "app.core.plugins.catalog"


# ── the failure vocabulary ─────────────────────────────────────────────────

def test_the_install_failures_carry_what_a_dialog_needs():
    """Nothing raises these yet -- the install paths come in a later change --
    so this is what stops a typo in an attribute name from being found by a
    route instead of by a test."""
    assert PluginInstallError("x", hint="pip said no").hint == "pip said no"
    assert PluginInstallError("x").hint is None

    restart = PluginNeedsRestart("x", command="cdui plugin install foo")
    assert restart.command == "cdui plugin install foo"
    assert isinstance(restart, PluginInstallError)

    consent = ConsentRequired(
        "x", missing_capabilities=["network"], allowed_modules=("os",)
    )
    assert consent.missing_capabilities == ("network",)
    assert consent.allowed_modules == ("os",)
    assert isinstance(consent, PluginInstallError)
    assert ConsentRequired("x").missing_capabilities == ()

    assert GitHubError("x", status=404).status == 404
    assert GitHubError("x").status is None, "no status means it never got that far"


def test_a_cancel_is_not_an_install_failure():
    """The system doing as it was told. An ``except PluginInstallError`` that
    caught this would report the user's own Stop as something that broke."""
    assert not isinstance(PluginCancelled(), PluginInstallError)


def test_the_refusals_stay_the_exception_types_their_callers_already_catch():
    assert issubclass(ManifestError, ValueError)
    assert issubclass(SourceError, ValueError)
    assert issubclass(UnknownCatalogName, SourceError)
    assert issubclass(UnparseableSource, SourceError)
    assert issubclass(PluginInstallError, RuntimeError)


# ── parse_source ───────────────────────────────────────────────────────────

def test_a_catalog_id_resolves_to_the_pack_that_ships_here():
    parsed = parse_source("foundations")
    assert isinstance(parsed, ParsedSource)
    assert parsed == ParsedSource("catalog", "foundations", "", "")


@pytest.mark.parametrize("spec", ["rl", "RL", "Rl"])
def test_the_catalog_lookup_is_case_insensitive_and_answers_lower_case(spec):
    """The id is a directory name, so the answer has to be the spelling on
    disk however the user typed it."""
    assert parse_source(spec) == ParsedSource("catalog", "rl", "", "")


def test_the_catalog_wins_over_the_github_short_form():
    """``deep`` is a legal repository name too, and a bare word that IS a
    catalog id must resolve to the pack shipped here rather than to whatever
    happens to be on GitHub."""
    assert parse_source("deep").kind == "catalog"


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("alice/extras", ParsedSource("github", "alice", "extras", "")),
        ("alice/extras@v1.2.3", ParsedSource("github", "alice", "extras", "v1.2.3")),
        (
            "https://github.com/alice/extras",
            ParsedSource("github", "alice", "extras", ""),
        ),
        (
            "https://github.com/alice/extras.git",
            ParsedSource("github", "alice", "extras", ""),
        ),
        (
            "https://github.com/alice/extras@v2",
            ParsedSource("github", "alice", "extras", "v2"),
        ),
        (
            "https://github.com/alice/extras.git@v2",
            ParsedSource("github", "alice", "extras", "v2"),
        ),
        (
            "http://www.github.com/alice/extras",
            ParsedSource("github", "alice", "extras", ""),
        ),
    ],
)
def test_the_github_shapes_all_reach_the_same_owner_and_repo(spec, expected):
    """One field, four spellings: a person with a plugin in mind should not
    have to know which of them the installer prefers."""
    assert parse_source(spec) == expected


@pytest.mark.parametrize(
    "spec, expected_ref",
    [
        ("alice/extras@v1.2", "v1.2"),
        ("https://github.com/alice/extras@v1.2", "v1.2"),
        ("alice/extras@feat/x", "feat/x"),
        ("https://github.com/alice/extras@feat/x", "feat/x"),
        ("alice/extras@0.1.0-rc.1", "0.1.0-rc.1"),
    ],
)
def test_the_two_source_shapes_read_the_same_refs(spec, expected_ref):
    """One grammar for the ref, whichever door the source came in: the URL
    form used to end in ``(?:@(.+))?`` and the short form in ``[\\w./-]+``, so
    the same ref was legal typed one way and unparseable typed the other."""
    assert parse_source(spec).ref == expected_ref


@pytest.mark.parametrize(
    "spec",
    [
        "https://github.com/alice/extras@a b",     # a space: InvalidURL later
        "https://github.com/alice/extras@v1\tx",   # a control character
        "alice/extras@../../x",
        "https://github.com/alice/extras@../../x",
        "alice/extras@..",
        "alice/extras@feat/../../x",
        "alice/extras@feat/..",
    ],
)
def test_a_ref_that_walks_up_or_cannot_be_sent_is_unparseable(spec):
    """A ref is interpolated into ``/repos/<o>/<r>/commits/<ref>``, so a
    ``..`` segment asks a different endpoint than the one the user named --
    and a ref with a space in it never reaches GitHub at all. Both are
    refused at the string the user typed rather than three modules later."""
    with pytest.raises(UnparseableSource):
        parse_source(spec)


def test_a_ref_with_two_dots_inside_a_word_is_still_a_ref():
    """Only ``..`` standing alone between slashes is the traversal; ``v1..2``
    is a legal tag name and refusing it would be a rule about spelling."""
    assert parse_source("alice/extras@v1..2").ref == "v1..2"


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://github.com/alice/extras", ("alice", "extras")),
        ("https://github.com/alice/extras.git", ("alice", "extras")),
        ("https://github.com/alice/extras/", ("alice", "extras")),
        ("http://www.github.com/alice/extras", ("alice", "extras")),
        ("https://github.com/alice/extras@v2", ("alice", "extras")),
        # The host is half of the answer. The path of this one ends in the
        # owner and repository of a plugin the catalog vouches for.
        ("https://evil.example.com/CodefyUI/CodefyUI-Plugin-Self-Learning", None),
        ("https://gitlab.com/alice/extras", None),
        ("https://github.com.evil.example/alice/extras", None),
        ("alice/extras", None),   # a short form is not a URL
        ("", None),
    ],
)
def test_a_recorded_url_names_a_repository_only_when_it_is_on_github(url, expected):
    """What a lockfile reader needs and what a source parser needs are the
    same question with the catalog taken out: which repository is this, and
    is it one at all."""
    assert parse_github_url(url) == expected


def test_an_unknown_bare_word_names_the_catalog_it_was_not_in():
    """A bare word can only have been meant as a catalog name, so the answer
    is "yours does not have that one" plus the ids it does -- as DATA, so the
    CLI's bilingual sentence and a route's JSON both build from the same
    refusal."""
    with pytest.raises(UnknownCatalogName) as excinfo:
        parse_source("edo")
    error = excinfo.value
    assert error.spec == "edo"
    assert set(error.known) == set(load_catalog()["plugins"])
    assert error.known == tuple(sorted(error.known)), "sorted, for a stable message"
    assert error.catalog_path == plugin_loader.plugins_builtin_root() / "registry.json"


def test_an_unknown_bare_word_against_an_unreadable_catalog_says_so_with_no_ids():
    """An empty catalog fails every name including the valid ones, and the
    empty ``known`` is what lets a caller say that instead of blaming the
    name."""
    with pytest.raises(UnknownCatalogName) as excinfo:
        parse_source("foundations", catalog={"schema": 1, "plugins": {}})
    assert excinfo.value.known == ()


def test_an_injected_catalog_is_what_gets_searched():
    """The seam ``scripts/plugins.py`` needs: its tests fake the catalog by
    patching the CLI's own loader, and this must answer from what it is
    handed rather than from the registry on disk."""
    catalog = {"schema": 1, "plugins": {"only-this": {"kind": "builtin"}}}
    assert parse_source("only-this", catalog=catalog).kind == "catalog"
    with pytest.raises(UnknownCatalogName):
        parse_source("foundations", catalog=catalog)


@pytest.mark.parametrize(
    "spec",
    [
        "not a valid source spec",
        "",
        "https://gitlab.com/alice/extras",
        "alice/extras/deeper",
        "-leading-dash",
    ],
)
def test_garbage_is_refused_as_unparseable(spec):
    with pytest.raises(UnparseableSource) as excinfo:
        parse_source(spec)
    assert excinfo.value.spec == spec


def test_both_refusals_stay_catchable_as_ValueError():
    """``cmd_install``, ``cmd_info`` and ``scripts/project.py`` all catch
    ``ValueError`` around this call and none of them was changed."""
    for spec in ("edo", "not a valid source spec"):
        with pytest.raises(ValueError):
            parse_source(spec)


# ── the catalog this build ships ───────────────────────────────────────────

def test_reserved_ids_are_the_fixed_segments_under_api_plugins():
    assert RESERVED_PLUGIN_IDS == frozenset({
        "catalog", "inspect", "install", "jobs", "generation", "reload",
    })
    assert isinstance(RESERVED_PLUGIN_IDS, frozenset)


def test_no_pack_this_build_ships_claims_a_reserved_id():
    """The point of the set: a pack called ``install`` would sit where the
    install route already lives, and the router would decide which wins."""
    assert not (set(catalog_entries()) & RESERVED_PLUGIN_IDS)


def test_every_entry_the_repository_ships_survives_validation():
    """A catalog entry the Plugin Center cannot render is a pack a student
    cannot install, and it would ship green without this."""
    raw = load_catalog()["plugins"]
    entries = catalog_entries()
    assert set(entries) == set(raw)
    assert {"edu", "foundations", "deep", "stats", "rl"} <= set(entries)


def test_a_real_entry_is_carried_across_field_by_field():
    entries = catalog_entries()
    stats = entries["stats"]
    assert isinstance(stats, CatalogEntry)
    assert stats.id == "stats"
    assert stats.kind == "builtin"
    assert stats.path == "plugins/stats"
    assert stats.tags == ("statistics", "chart", "table", "eda", "confusion-matrix")
    assert "Descriptive statistics" in stats.description
    # Absent optional values are renderable rather than None.
    assert (stats.ref, stats.homepage, stats.chapters) == ("", "", ())
    assert stats.official is False
    assert stats.repo is None
    assert entries["edu"].chapters == ("I1", "I2")


def test_github_catalog_packs_lists_the_repository_entries_sorted_by_id(monkeypatch):
    """The shape is in the schema so a Plugin Center never has to invent it,
    and nothing ships through it today -- which is exactly why it is tested
    against a catalog that does."""
    monkeypatch.setattr("app.core.plugins.catalog.load_catalog", lambda: {
        "schema": 1,
        "plugins": {
            "zebra": {"kind": "github", "name": "Z", "repo": "alice/zebra"},
            "here": {"kind": "builtin", "name": "H", "path": "plugins/here"},
            "aardvark": {"kind": "github", "name": "A", "repo": "bob/aardvark"},
        },
    })
    assert [entry.id for entry in github_catalog_packs()] == ["aardvark", "zebra"]


def test_catalog_entry_is_case_insensitive():
    assert catalog_entry("Stats") == catalog_entry("stats")
    assert catalog_entry("STATS").id == "stats"
    assert catalog_entry("no-such-pack") is None


# ── validate_catalog drops, never raises ───────────────────────────────────

def _one(entry: dict, plugin_id: str = "pack") -> dict:
    return {"schema": 1, "plugins": {plugin_id: entry}}


GOOD_BUILTIN = {"kind": "builtin", "name": "Pack", "path": "plugins/pack"}
GOOD_GITHUB = {"kind": "github", "name": "Pack", "repo": "alice/extras"}


def test_a_github_entry_keeps_its_repo_ref_and_flags():
    entries = validate_catalog(_one({
        "kind": "github",
        "name": "Extras",
        "description": "third-party",
        "repo": "alice/extras",
        "ref": "v1.2.3",
        "homepage": "https://example.invalid",
        "tags": ["extra", 7],
        "official": True,
    }))
    entry = entries["pack"]
    assert entry.repo == "alice/extras"
    assert entry.ref == "v1.2.3"
    assert entry.homepage == "https://example.invalid"
    assert entry.official is True
    assert entry.tags == ("extra",), "a non-string tag is dropped, not coerced"
    assert entry.path is None


@pytest.mark.parametrize(
    "plugin_id, entry, because",
    [
        ("Caps", GOOD_BUILTIN, "not a valid plugin id"),
        ("trailing-", GOOD_BUILTIN, "not a valid plugin id"),
        ("pack", "not a table", "entry is not a table"),
        ("pack", {"kind": "svn", "path": "p"}, "unknown kind"),
        ("pack", {"kind": "builtin", "name": "P"}, "a builtin entry needs a path"),
        ("pack", {"kind": "builtin", "path": ""}, "a builtin entry needs a path"),
        ("pack", {"kind": "github", "name": "P"}, "a github entry needs"),
        ("pack", {"kind": "github", "repo": "no-slash"}, "a github entry needs"),
        (
            "pack",
            {"kind": "github", "repo": "alice/extras", "path": "plugins/pack"},
            "cannot also declare a path",
        ),
    ],
)
def test_a_malformed_entry_is_dropped_with_one_log_line(
    plugin_id, entry, because, caplog
):
    """Dropped rather than raised: one bad row must not empty the Plugin
    Center, hide the four good packs beside it, or take ``cdui plugin list``
    down with it. The log line is how the bad row gets fixed."""
    with caplog.at_level(logging.WARNING, logger=CATALOG_LOGGER):
        assert validate_catalog(_one(entry, plugin_id)) == {}
    dropped = [r for r in caplog.records if r.name == CATALOG_LOGGER]
    assert len(dropped) == 1, [r.getMessage() for r in dropped]
    message = dropped[0].getMessage()
    assert because in message, message
    assert repr(plugin_id) in message, message


def test_one_bad_entry_never_takes_the_good_ones_with_it(caplog):
    data = {"schema": 1, "plugins": {
        "good": GOOD_BUILTIN,
        "alsogood": GOOD_GITHUB,
        "Bad Id": GOOD_BUILTIN,
        "broken": {"kind": "builtin"},
    }}
    with caplog.at_level(logging.WARNING, logger=CATALOG_LOGGER):
        entries = validate_catalog(data)
    assert set(entries) == {"good", "alsogood"}
    dropped = [r for r in caplog.records if r.name == CATALOG_LOGGER]
    assert len(dropped) == 2, [r.getMessage() for r in dropped]


@pytest.mark.parametrize(
    "data", [None, [], "registry", 7, {}, {"plugins": None}, {"plugins": ["a"]}]
)
def test_a_catalog_that_is_not_a_catalog_is_empty_rather_than_an_exception(
    data, caplog
):
    """Reached with a hand-edited or half-written ``registry.json``. Every
    caller of this is on the way to showing a list, and a traceback there
    takes down commands that have nothing to do with the catalog."""
    with caplog.at_level(logging.WARNING, logger=CATALOG_LOGGER):
        assert validate_catalog(data) == {}
    assert caplog.records, "silence would leave nobody to fix the file"


# ── manifest accessors ─────────────────────────────────────────────────────

def _shipped_manifest(plugin_id: str) -> dict:
    return read_manifest(plugin_loader.plugins_builtin_root() / plugin_id)


def test_the_edu_pack_declares_the_dependency_its_install_has_to_fetch():
    """``edu`` is the reason ``python_deps`` exists: ``SentenceEmbedding``
    needs model2vec, and an install that skipped it would register a node
    that raises on first run."""
    assert manifest_python_deps(_shipped_manifest("edu")) == {"model2vec": ">=0.8.0"}


def test_a_pack_with_no_dependencies_answers_an_empty_table():
    assert manifest_python_deps(_shipped_manifest("stats")) == {}


def test_the_shipped_packs_ask_for_no_extra_modules_and_no_browser_code():
    """Both are trust decisions, and the built-in packs make neither: they
    run on the default allowlist and add nothing to the editor."""
    for plugin_id in ("edu", "stats"):
        manifest = _shipped_manifest(plugin_id)
        assert manifest_allowed_modules(manifest) == []
        assert manifest_has_frontend(manifest) is False


def test_a_manifest_that_asks_for_everything_is_read_back_in_full():
    manifest = {
        "plugin": {"id": "greedy", "schema_version": 1},
        "security": {"allowed_modules": ["subprocess", "socket"]},
        "python_deps": {"requests": ">=2", "rich": ""},
        "frontend": {"entry": "dist/index.js"},
    }
    assert manifest_allowed_modules(manifest) == ["subprocess", "socket"]
    assert manifest_python_deps(manifest) == {"requests": ">=2", "rich": ""}
    assert manifest_has_frontend(manifest) is True


@pytest.mark.parametrize(
    "manifest, because",
    [
        ({}, "[plugin]"),
        ({"plugin": "greedy"}, "[plugin]"),
        ({"plugin": {"id": "greedy"}}, "schema_version"),
        ({"plugin": {"id": "greedy", "schema_version": 2}}, "schema_version"),
        ({"plugin": {"id": "Greedy", "schema_version": 1}}, "Invalid plugin id"),
        (
            {"plugin": {"id": "greedy", "schema_version": 1}, "security": "os"},
            "[security] must be a table",
        ),
        (
            {
                "plugin": {"id": "greedy", "schema_version": 1},
                "security": {"allowed_modules": "os"},
            },
            "must be a list of strings",
        ),
        (
            {
                "plugin": {"id": "greedy", "schema_version": 1},
                "security": {"capabilities": ["telepathy"]},
            },
            "Unknown capability",
        ),
    ],
)
def test_a_manifest_this_build_will_not_install_is_refused_by_name(manifest, because):
    """The refusal names the offending key. A manifest is hand-written by
    somebody who cannot see the installer, so "invalid manifest" costs them a
    guessing game -- and ``allowed_modules = "os"`` is the shape that proved
    it, granting ``{"o", "s"}`` and failing much later somewhere else."""
    with pytest.raises(ManifestError) as excinfo:
        validate_manifest(manifest)
    assert because in str(excinfo.value)


@pytest.mark.parametrize(
    "manifest, expected",
    [
        ({}, []),
        ({"security": "os"}, []),
        ({"security": {}}, []),
        # The shape that named this rule: a bare string reaching
        # ``frozenset("os")`` is ``{"o", "s"}`` -- never a module list.
        ({"security": {"allowed_modules": "os"}}, []),
        ({"security": {"allowed_modules": ["os", 7]}}, ["os"]),
    ],
)
def test_allowed_modules_describes_a_manifest_nobody_promised_was_valid(
    manifest, expected
):
    """``validate_manifest`` refuses each of these loudly at install time.
    The accessor is what an inspect view uses on a manifest already on disk,
    and there it has to answer rather than raise."""
    assert manifest_allowed_modules(manifest) == expected


@pytest.mark.parametrize(
    "manifest",
    [
        {},
        {"frontend": "dist/index.js"},
        {"frontend": {}},
        {"frontend": {"entry": ""}},
        {"frontend": {"entry": 1}},
    ],
)
def test_a_frontend_that_is_not_an_entry_path_is_no_frontend(manifest):
    assert manifest_has_frontend(manifest) is False


@pytest.mark.parametrize("manifest", [{}, {"python_deps": None}, {"python_deps": []}])
def test_python_deps_answers_an_empty_table_for_anything_that_is_not_one(manifest):
    assert manifest_python_deps(manifest) == {}
