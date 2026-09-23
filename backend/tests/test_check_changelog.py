"""scripts/check_changelog.py holds CHANGELOG.md to RELEASING.md step 1 (#462).

Step 1 of "Before you tag" -- promote ``## [Unreleased]`` to the new version
-- had no check at all: a release PR could bump the version, leave the
changelog half promoted, and go green. These tests run the check against
the real CHANGELOG.md on every PR, and against half-done promotions built
from a copy of it.

Versions are read off the file's newest heading rather than written down,
so a release does not have to edit this file: the promotion under test is
always to one patch above whatever is newest.
"""

from __future__ import annotations

import datetime as dt
import re
import shutil
import subprocess
from pathlib import Path

import pytest

import check_changelog as ccl  # scripts/check_changelog.py -- conftest puts scripts/ on sys.path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RELEASE_BUILD = _REPO_ROOT / ".github" / "workflows" / "release-build.yml"
_COMPARE = "https://github.com/CodefyUI/CodefyUI/compare/"
EM = "\N{EM DASH}"

#: A version heading as CHANGELOG.md writes it. Read here with a regex of
#: its own, so the fixtures do not depend on the parser under test.
_VERSION_HEADING = re.compile(
    r"^## \[(\d+\.\d+\.\d+[^\]]*)\] " + EM + r" (\d{4}-\d{2}-\d{2})$", re.MULTILINE)


def _changelog() -> str:
    return (_REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")


def _pyproject_version() -> str:
    text = (_REPO_ROOT / "backend" / "pyproject.toml").read_text(encoding="utf-8")
    return re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE).group(1)


def _newest(text: str) -> tuple[str, dt.date, str]:
    """The newest version, its date, and the version below it."""
    found = _VERSION_HEADING.findall(text)
    assert len(found) >= 2, (
        "CHANGELOG.md has fewer than two version headings; "
        "test_main_is_promoted_to_the_version_it_claims says what is wrong")
    (version, date), (previous, _) = found[:2]
    return version, dt.date.fromisoformat(date), previous


def _newest_heading(text: str) -> str:
    return _VERSION_HEADING.search(text).group(0)


def _next(version: str) -> str:
    """One patch above *version*: the release every promotion here adds."""
    major, minor, patch = re.match(r"(\d+)\.(\d+)\.(\d+)", version).groups()
    return f"{major}.{minor}.{int(patch) + 1}"


def _with_an_entry(text: str) -> str:
    """*text* with a pending fix under ``## [Unreleased]``, as a PR leaves it."""
    return text.replace(
        "## [Unreleased]\n",
        "## [Unreleased]\n\n### Fixed\n\n- **Something was wrong.** It is not now. ([#999])\n",
        1)


def _promote(text: str, *, fresh: bool = True, below: bool = False,
             dash: str = EM, date: dt.date | None = None, link: bool = True,
             repoint: bool = True, base: str | None = None,
             stale: bool = False) -> str:
    """Step 1, done to a copy of *text* for the next patch release.

    With no keywords the promotion is complete. Each keyword skips or botches
    one part: *fresh* is the new empty ``## [Unreleased]``; *below* adds the
    new heading under whatever ``[Unreleased]`` holds instead of renaming it;
    *dash* and *date* make the new heading; *link* adds its compare link;
    *repoint* moves the ``[Unreleased]`` link to start at it; *base* is where
    its own link starts; and *stale* leaves the old ``[Unreleased]`` link
    above the new one.
    """
    current, current_date, _ = _newest(text)
    new = _next(current)
    # Found rather than spelled out: the check only cares where the link
    # starts, so a file that compares to HEAD instead of main is just as good.
    unreleased = re.search(r"^\[Unreleased\]: (\S+/compare/)\S+?\.\.\.(\S+)$",
                           text, re.MULTILINE)
    assert text.count("## [Unreleased]\n") == 1 and unreleased, (
        "CHANGELOG.md has no single ## [Unreleased] heading or no [Unreleased] "
        "compare link; test_main_is_promoted_to_the_version_it_claims says what is wrong")
    old_link, prefix, end = unreleased.group(0), unreleased.group(1), unreleased.group(2)
    when = date or current_date + dt.timedelta(days=1)
    heading = f"## [{new}] {dash} {when.isoformat()}\n"
    if below:
        text = text.replace(_newest_heading(text), heading + "\n" + _newest_heading(text), 1)
    else:
        text = text.replace(
            "## [Unreleased]\n", ("## [Unreleased]\n\n" if fresh else "") + heading, 1)
    links = [f"[Unreleased]: {prefix}{new}...{end}" if repoint else old_link]
    if stale:
        links.insert(0, old_link)
    if link:
        links.append(f"[{new}]: {_COMPARE}{base or current}...{new}")
    return text.replace(old_link, "\n".join(links), 1)


# -- the rules on every PR ------------------------------------------------------

def test_a_version_bump_without_a_promoted_changelog_is_refused():
    text = _changelog()
    current = _newest(text)[0]
    found = ccl.problems(text, _next(current))
    assert any(f"[{current}]" in p and _next(current) in p for p in found), found


def test_a_complete_promotion_passes():
    """The control for every refusal below: without it, a check that refused
    everything would pass all of them."""
    text = _changelog()
    assert ccl.problems(_promote(text), _next(_newest(text)[0])) == []


def test_main_is_promoted_to_the_version_it_claims(capsys):
    """The real files. A PR that bumps the version fails here until
    CHANGELOG.md is promoted to match."""
    assert ccl.problems(_changelog(), _pyproject_version()) == []
    assert ccl.main([]) == 0, capsys.readouterr().out


def test_an_ordinary_pr_passes_with_or_without_an_unreleased_entry():
    text = _changelog()
    version = _newest(text)[0]
    entry = f"{_with_an_entry(text).rstrip()}\n[#999]: https://github.com/CodefyUI/CodefyUI/pull/999\n"
    # Two PRs adding the same [#NNN] link is harmless, so it is not refused.
    repeated = re.search(r"^\[#\d+\]: \S+$", text, re.MULTILINE).group(0)
    assert ccl.problems(text, version) == []
    assert ccl.problems(entry, version) == []
    assert ccl.problems(f"{text.rstrip()}\n{repeated}\n", version) == []


def _previous(text: str) -> str:
    return _newest(text)[2]


def _a_day_early(text: str) -> dt.date:
    return _newest(text)[1] - dt.timedelta(days=1)


def _new_link_twice(text: str) -> str:
    promoted = _promote(text)
    new = re.escape(_next(_newest(text)[0]))
    line = re.search(rf"^\[{new}\]: \S+$", promoted, re.MULTILINE).group(0)
    return f"{promoted.rstrip()}\n{line}\n"


def _new_link_copied(text: str) -> str:
    """The new link made by copying the line below it and renaming only the label."""
    current, _, previous = _newest(text)
    new = _next(current)
    return _promote(text).replace(f"[{new}]: {_COMPARE}{current}...{new}",
                                  f"[{new}]: {_COMPARE}{previous}...{current}", 1)


@pytest.mark.parametrize(("promote", "bumped", "names"), [
    pytest.param(lambda t: t, True, "rename ## [Unreleased]",
                 id="version bumped, nothing promoted"),
    pytest.param(_promote, False, "bump backend/pyproject.toml",
                 id="promoted, version not bumped"),
    pytest.param(lambda t: _promote(t, fresh=False), True,
                 "there is no ## [Unreleased] heading",
                 id="no fresh [Unreleased] above"),
    pytest.param(lambda t: _promote(t, link=False), True, "has no link definition",
                 id="no link for the new version"),
    pytest.param(lambda t: _promote(t, repoint=False), True, "has to start at",
                 id="[Unreleased] still compares from the old version"),
    pytest.param(lambda t: _promote(t, base=_previous(t)), True, "compares from",
                 id="new link compares from two releases back"),
    pytest.param(lambda t: _promote(t, stale=True), True, "is defined 2 times",
                 id="stale [Unreleased] link left above the new one"),
    pytest.param(_new_link_twice, True, "is defined 2 times",
                 id="new link defined twice"),
    pytest.param(_new_link_copied, True, "has to end",
                 id="new link copied from the one below"),
    pytest.param(lambda t: _promote(t, dash="-"), True, "em dash",
                 id="heading written with a hyphen"),
    pytest.param(lambda t: _promote(t, date=_a_day_early(t)), True, "is dated",
                 id="dated before the release below it"),
])
def test_a_half_done_promotion_is_refused(promote, bumped, names):
    text = _changelog()
    current = _newest(text)[0]
    found = ccl.problems(promote(text), _next(current) if bumped else current)
    assert any(names in p for p in found), found


@pytest.mark.parametrize(("break_it", "names"), [
    pytest.param(lambda t: t.replace("## [Unreleased]\n", "## Notes\n\n## [Unreleased]\n", 1),
                 "has to be the first ## heading", id="a heading above [Unreleased]"),
    pytest.param(lambda t: t.replace("## [Unreleased]\n", "## [Unreleased]\n\n## [Unreleased]\n", 1),
                 "## [Unreleased] appears 2 times", id="[Unreleased] twice"),
    pytest.param(lambda t: t.replace("## [Unreleased]\n", "## [Unreleased]\n\n## Notes\n", 1),
                 "the heading below ## [Unreleased] is ## Notes",
                 id="a heading that is not a version below [Unreleased]"),
    pytest.param(lambda t: t.replace(_newest_heading(t), _newest_heading(t)[:-10] + "2026-02-30", 1),
                 "2026-02-30 is not a date", id="a date that does not exist"),
    pytest.param(lambda t: t.replace(_newest_heading(t), f"{_newest_heading(t)}\n\n{_newest_heading(t)}", 1),
                 "a release has one section", id="a version heading twice"),
    pytest.param(lambda t: _VERSION_HEADING.sub("", t), "there is no version heading",
                 id="no version heading at all"),
    pytest.param(lambda t: re.sub(r"^\[Unreleased\]: \S+\n", "", t, count=1, flags=re.MULTILINE),
                 "[Unreleased] has no link definition", id="no [Unreleased] link"),
])
def test_a_broken_structure_is_refused(break_it, names):
    """The same file, version unchanged, broken in a way no promotion step
    produces on its own -- a bad merge of CHANGELOG.md, for one."""
    text = _changelog()
    found = ccl.problems(break_it(text), _newest(text)[0])
    assert any(names in p for p in found), found


def _new_section_below_the_last_release(text: str) -> str:
    """A complete promotion whose new section went below the release before it."""
    current, _, previous = _newest(text)
    promoted = _promote(text)
    new = re.escape(_next(current))
    section = re.search(rf"^## \[{new}\] .*?(?=^## )", promoted,
                        re.MULTILINE | re.DOTALL).group(0)
    promoted = promoted.replace(section, "", 1)
    older = re.search(rf"^## \[{re.escape(previous)}\] ", promoted, re.MULTILINE).start()
    return promoted[:older] + section + promoted[older:]


def test_a_new_section_below_the_last_release_is_reported_once():
    """Moving the section up is the whole fix, so it is the one thing said.
    Read in file order, the version, link and date rules would each ask for a
    wrong fix, down to bumping backend/pyproject.toml back to the old version."""
    text = _changelog()
    moved = _new_section_below_the_last_release(text)
    new = _next(_newest(text)[0])
    found = ccl.problems(moved, new)
    assert len(found) == 1 and f"move the ## [{new}] section up" in found[0], found
    noon = dt.datetime.combine(_newest(_promote(text))[1], dt.time(12), dt.timezone.utc)
    assert ccl.tag_problems(moved, new, noon) == []


def test_versions_sort_by_number_not_text():
    """2.8.10 is newer than 2.8.9, and a release candidate sorts just below its
    release. A label that is not a version leaves the file order alone."""
    labels = ["2.8.9", "2.9.0rc1", "2.8.10", "2.9.0", "2.9.0rc2"]
    assert ccl._newest_first(labels) == ["2.9.0", "2.9.0rc2", "2.9.0rc1", "2.8.10", "2.8.9"]
    assert ccl._newest_first(["2.8.9", "Yanked", "2.8.10"]) == ["2.8.9", "Yanked", "2.8.10"]


def test_the_unreleased_link_may_end_anywhere():
    """The rules check where [Unreleased] starts, not where it ends: main,
    HEAD and a branch name compare the same way."""
    text = re.sub(r"^(\[Unreleased\]: \S+\.\.\.)\S+$", r"\g<1>HEAD", _changelog(),
                  count=1, flags=re.MULTILINE)
    assert re.search(r"^\[Unreleased\]: \S+\.\.\.HEAD$", text, re.MULTILINE)
    current = _newest(text)[0]
    assert ccl.problems(text, current) == []
    assert ccl.problems(_promote(text), _next(current)) == []


def test_crlf_is_read_like_lf():
    """A Windows checkout is CRLF (core.autocrlf), and a caller may pass the
    text on without universal newlines."""
    text = _changelog()
    current = _newest(text)[0]
    crlf = text.replace("\n", "\r\n")
    assert ccl.problems(crlf, current) == []
    assert ccl.problems(crlf, _next(current)) != []


def test_a_heading_is_read_the_way_markdown_renders_it():
    """Markdown drops the spaces around a heading's text, so ``##  [Unreleased]``
    is the same heading and must not be reported missing."""
    text = _changelog()
    spaced = text.replace("## [Unreleased]\n", "##  [Unreleased]  \n", 1)
    assert ccl.problems(spaced, _newest(text)[0]) == []


def test_the_version_is_read_from_the_project_table():
    other = '[tool.other]\nversion = "0.1.0"\n'
    assert ccl._project_version(other + '\n[project]\nname = "x"\nversion = "2.8.5"\n') == "2.8.5"
    assert ccl._project_version(other) is None


# -- the rules on a tag -----------------------------------------------------------

def test_the_tag_has_to_name_the_newest_heading_and_carry_its_utc_date():
    text = _changelog()
    promoted = _promote(text)
    new = _next(_newest(text)[0])
    noon = dt.datetime.combine(_newest(promoted)[1], dt.time(12), dt.timezone.utc)
    assert ccl.tag_problems(promoted, new, noon) == []
    assert ccl.tag_problems(promoted, f"v{new}", noon) == []
    assert ccl.tag_problems(promoted, new, noon + dt.timedelta(days=1)) != []
    # A tag on the wrong commit and a misnamed tag look the same from here, so
    # the message gives both fixes, the second with the name to use.
    wrong = ccl.tag_problems(promoted, _next(new), noon)
    assert len(wrong) == 1 and f"re-tag this commit as {new}" in wrong[0], wrong
    assert f"re-tag this commit as v{new}" in ccl.tag_problems(promoted, f"v{_next(new)}", noon)[0]
    assert ccl.tag_problems("## [Unreleased]\n", new, noon) != []


def test_a_tag_whose_entries_were_left_under_unreleased_is_refused():
    """Step 1 half done the one way the PR rules cannot see: the new heading
    added below the [Unreleased] entries instead of renaming [Unreleased].
    The structure is sound, but the release section is empty and its entries
    still read as unreleased. At the release commit [Unreleased] holds no
    entries, so the tag rules require that."""
    text = _with_an_entry(_changelog())
    new = _next(_newest(text)[0])
    below = _promote(text, below=True)
    noon = dt.datetime.combine(_newest(below)[1], dt.time(12), dt.timezone.utc)
    assert ccl.problems(below, new) == []
    found = ccl.tag_problems(below, new, noon)
    assert any("## [Unreleased] still holds entries" in p for p in found), found
    # The control: renamed instead, the entries move with the heading.
    assert ccl.tag_problems(_promote(text), new, noon) == []
    # A line of prose is not an entry. 2.1.0, 2.1.1 and 2.2.0 were tagged with
    # "Nothing yet." under [Unreleased], and they were promoted in full.
    placeholder = _promote(text).replace("## [Unreleased]\n", "## [Unreleased]\n\nNothing yet.\n", 1)
    assert ccl.tag_problems(placeholder, new, noon) == []


@pytest.mark.parametrize(("heading", "tag", "created", "ships"), [
    # 2.8.2 was promoted the day before its tag and shipped that way (#458).
    pytest.param(f"## [2.8.2] {EM} 2026-09-16", "2.8.2", "2026-09-17T05:22:48+00:00", False,
                 id="2.8.2 dated a day before its tag"),
    # 2.8.5 was tagged at 05:12 on 09-23 in UTC+8, which is still 09-22 in UTC.
    pytest.param(f"## [2.8.5] {EM} 2026-09-22", "2.8.5", "2026-09-23T05:12:16+08:00", True,
                 id="2.8.5 tagged at UTC+8, dated in UTC"),
    # The same tag dated with the tagger's local date: the trap itself.
    pytest.param(f"## [2.8.5] {EM} 2026-09-23", "2.8.5", "2026-09-23T05:12:16+08:00", False,
                 id="2.8.5 dated with the local date"),
])
def test_the_real_incidents(heading, tag, created, ships):
    text = f"## [Unreleased]\n\n{heading}\n"
    found = ccl.tag_problems(text, tag, dt.datetime.fromisoformat(created))
    assert (found == []) is ships, found


def test_a_time_without_an_offset_is_refused():
    """A naive datetime would be taken as local time, which is the mistake
    the UTC rule exists to prevent."""
    with pytest.raises(ValueError):
        ccl.tag_problems(f"## [Unreleased]\n\n## [2.8.5] {EM} 2026-09-22\n", "2.8.5",
                         dt.datetime(2026, 9, 22, 21, 12, 16))


def _tagged_repo(tmp_path: Path, monkeypatch, heading_date: str) -> Path:
    """A repository at 2.8.5 with an annotated tag 2.8.5, created when the
    real one was: 05:12:16 on 2026-09-23 in UTC+8, 21:12:16 on 09-22 in UTC.

    The machine's git config is left out, and the identity and the date come
    from the environment, so this runs where git was never configured.
    """
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    config = tmp_path / "gitconfig"
    config.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for role in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{role}_NAME", "Release Test")
        monkeypatch.setenv(f"GIT_{role}_EMAIL", "release-test@example.com")
    monkeypatch.setenv("GIT_COMMITTER_DATE", "2026-09-23T05:12:16+08:00")

    root = tmp_path / "repo"
    (root / "backend").mkdir(parents=True)
    changelog = "\n".join([
        "# Changelog", "",
        "## [Unreleased]", "",
        f"## [2.8.5] {EM} {heading_date}", "",
        f"## [2.8.4] {EM} 2026-09-20", "",
        f"[Unreleased]: {_COMPARE}2.8.5...main",
        f"[2.8.5]: {_COMPARE}2.8.4...2.8.5",
        f"[2.8.4]: {_COMPARE}2.8.3...2.8.4", "",
    ])
    (root / "CHANGELOG.md").write_bytes(changelog.encode("utf-8"))
    (root / "backend" / "pyproject.toml").write_bytes(
        b'[project]\nname = "codefyui-backend"\nversion = "2.8.5"\n')
    for args in (("init", "-q"), ("add", "-A"), ("commit", "-q", "-m", "2.8.5"),
                 ("tag", "-a", "2.8.5", "-m", "2.8.5")):
        subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True)
    return root


def test_the_tag_date_is_read_in_utc(tmp_path, monkeypatch):
    root = _tagged_repo(tmp_path, monkeypatch, "2026-09-22")
    created = ccl.tag_created("2.8.5", root)
    assert created == dt.datetime(2026, 9, 22, 21, 12, 16, tzinfo=dt.timezone.utc)
    # UTC itself, not the same instant at the tagger's offset: the date is
    # what gets compared, and at +08:00 it reads 2026-09-23.
    assert created.utcoffset() == dt.timedelta(0)
    assert created.date() == dt.date(2026, 9, 22)
    assert ccl.tag_created("9.9.9", root) is None


@pytest.mark.parametrize(("heading_date", "code"), [("2026-09-22", 0), ("2026-09-23", 1)])
def test_tag_mode_checks_the_files_at_the_tag(tmp_path, monkeypatch, capsys, heading_date, code):
    root = _tagged_repo(tmp_path, monkeypatch, heading_date)
    # What the working tree says no longer matters once the tag exists.
    (root / "CHANGELOG.md").write_text("# Changelog\n", encoding="utf-8")
    assert ccl.main(["--tag", "2.8.5", "--root", str(root)]) == code
    assert "2026-09-22T21:12:16Z" in capsys.readouterr().out
    assert ccl.main(["--root", str(root)]) == 1
    assert ccl.main(["--tag", "9.9.9", "--root", str(root)]) == 1


def _copy(root: Path, changelog: str, version: str) -> Path:
    """CHANGELOG.md and the real backend/pyproject.toml, set to *version*."""
    pyproject = (_REPO_ROOT / "backend" / "pyproject.toml").read_text(encoding="utf-8")
    pyproject = re.sub(r'^version = "[^"]+"', f'version = "{version}"', pyproject,
                       count=1, flags=re.MULTILINE)
    (root / "backend").mkdir(parents=True)
    (root / "CHANGELOG.md").write_text(changelog, encoding="utf-8")
    (root / "backend" / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    return root


def test_the_cli_exits_1_on_a_half_done_copy_and_0_on_an_untouched_one(tmp_path, capsys):
    text = _changelog()
    current = _newest(text)[0]
    assert ccl.main(["--root", str(_copy(tmp_path / "untouched", text, current))]) == 0
    assert ccl.main(["--root", str(_copy(tmp_path / "bumped", text, _next(current)))]) == 1
    assert (f"CHANGELOG.md: newest version heading is [{current}] but "
            f"backend/pyproject.toml says {_next(current)}") in capsys.readouterr().out
    # A file that cannot be read, or a pyproject.toml with no version, is a failure too.
    assert ccl.main(["--root", str(tmp_path / "nothing-here")]) == 1
    unversioned = _copy(tmp_path / "unversioned", text, current)
    (unversioned / "backend" / "pyproject.toml").write_text('[project]\nname = "x"\n', encoding="utf-8")
    assert ccl.main(["--root", str(unversioned)]) == 1


def test_release_build_runs_the_check_on_a_tag_push():
    """Release Build never runs on a PR, so a PR has to see that its step is
    still there, the house rule test_release_legal_files.py states. The step
    comes before the release is created, or a refused tag leaves a draft."""
    lines = _RELEASE_BUILD.read_text(encoding="utf-8").splitlines()
    runs = [i for i, line in enumerate(lines)
            if "python3 scripts/check_changelog.py --tag" in line]
    creates = [i for i, line in enumerate(lines)
               if line.strip() == "- name: Create or update release"]
    assert len(runs) == 1, (
        f"expected one `python3 scripts/check_changelog.py --tag` line in "
        f"{_RELEASE_BUILD.name}, found {len(runs)}")
    assert len(creates) == 1 and runs[0] < creates[0], (
        "the CHANGELOG check has to run before the release is created")
