#!/usr/bin/env python3
"""Release guard (#462): CHANGELOG.md has to be promoted to the version it ships as.

Step 1 of "Before you tag" in .github/RELEASING.md is done by hand: rename
``## [Unreleased]`` to the new version, open a fresh ``## [Unreleased]``
above it, and repoint the compare links at the bottom of the file. Nothing
read CHANGELOG.md before this script, so a release PR could bump the
version, leave the changelog half promoted, and go green.

Two sets of rules:

* ``problems`` checks the file against the version in
  backend/pyproject.toml. backend/tests/test_check_changelog.py runs it on
  every PR. An ordinary PR changes neither the version nor a released
  section, whether or not it adds to ``[Unreleased]``, so it passes; a
  release PR passes once step 1 is done in full.
* ``tag_problems`` checks the file against a tag: the newest heading has to
  name the tag and carry the date the tag was created, in UTC, and
  ``[Unreleased]`` has to hold no entries. Release Build runs it on a tag
  push, and the maintainer runs it between ``git tag`` and ``git push``.

Standard library only, and no tomllib: the Python 3.10 CI job imports this.

Usage (from anywhere; resolves the repo root from this file's own path):

    python scripts/check_changelog.py              # the working tree
    python scripts/check_changelog.py --tag X.Y.Z  # the files at a tag

Exit 0: nothing to fix. Exit 1: each problem is printed with its fix.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

# Force UTF-8 on Windows so the em dashes in the messages print whatever the
# console code page is. Mirrors the guard in scripts/check_control_bytes.py.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CHANGELOG = "CHANGELOG.md"
_PYPROJECT = "backend/pyproject.toml"
_COMPARE = "https://github.com/CodefyUI/CodefyUI/compare/"
_STEP = 'step 1 of "Before you tag" in .github/RELEASING.md'

EM_DASH = "\N{EM DASH}"
_FORMAT = f"## [X.Y.Z] {EM_DASH} YYYY-MM-DD"

_HEADING = re.compile(r"## (.*)")
_LINK_DEF = re.compile(r" {0,3}\[([^\]]+)\]:\s*(\S+)")
_BRACKETED = re.compile(r"\[([^\]]+)\](.*)")
_DATED = re.compile(" " + EM_DASH + r" (\d{4}-\d{2}-\d{2})")
_COMPARED = re.compile(r"/compare/([^/]+?)\.\.\.([^/]+)$")
_VERSION_KEY = re.compile(r"""\s*version\s*=\s*["']([^"']+)["']""")
_VERSION_NUMBER = re.compile(r"(\d+)\.(\d+)\.(\d+)(?:rc(\d+))?")
#: An entry under a heading: a subsection heading or a list item.
_ENTRY = re.compile(r"#{3,6} |[-*+] |\d+[.)] ")


def _parse(text: str) -> tuple[list[str], dict[str, list[str]]]:
    """The ``## `` headings, and each link label's URLs, in file order.

    Labels are keyed case-folded because Markdown matches them without case.
    A label defined twice keeps both URLs: Markdown renders the first, so a
    stale definition left above a new one wins.
    """
    headings: list[str] = []
    links: dict[str, list[str]] = {}
    for line in text.splitlines():
        heading = _HEADING.match(line)
        if heading:
            headings.append(heading.group(1).strip())
            continue
        link = _LINK_DEF.match(line)
        if link:
            links.setdefault(link.group(1).casefold(), []).append(link.group(2))
    return headings, links


def _versions(headings: list[str]) -> list[tuple[str, str, str]]:
    """(heading, label, what follows the label) for each ``## [label]``
    heading other than ``[Unreleased]``, newest first."""
    found = []
    for heading in headings:
        bracketed = _BRACKETED.fullmatch(heading)
        if bracketed and bracketed.group(1) != "Unreleased":
            found.append((heading, bracketed.group(1), bracketed.group(2)))
    return found


def _version_key(label: str) -> tuple[int, int, int, int, int] | None:
    """How *label* sorts as a version: X.Y.Z, with X.Y.ZrcN just below X.Y.Z.
    None for anything else, which the order rules then leave where it is."""
    number = _VERSION_NUMBER.fullmatch(label)
    if number is None:
        return None
    major, minor, patch, rc = number.groups()
    return int(major), int(minor), int(patch), 0 if rc else 1, int(rc or 0)


def _newest_first(labels: list[str]) -> list[str]:
    """*labels* newest first by version, or as given if one is not a version."""
    if any(_version_key(label) is None for label in labels):
        return labels
    return sorted(labels, key=_version_key, reverse=True)


def _unreleased_entries(text: str) -> list[str]:
    """The entries under the first ``## [Unreleased]``: its subsection headings
    and list items, up to the next ``## `` heading.

    A line of prose is not an entry. 2.1.0, 2.1.1 and 2.2.0 were promoted in
    full and tagged with "Nothing yet." under [Unreleased], while every state
    of main that held a real pending change had a subsection or a list item.
    """
    entries: list[str] | None = None
    for line in text.splitlines():
        heading = _HEADING.match(line)
        if heading:
            if entries is not None:
                break
            if heading.group(1).strip() == "[Unreleased]":
                entries = []
        elif entries is not None and _ENTRY.match(line.strip()):
            entries.append(line.strip())
    return entries or []


def _defined_twice(label: str, count: int) -> str:
    return (f"[{label}]: is defined {count} times, and Markdown uses the first: "
            f"delete the stale one")


def problems(text: str, version: str) -> list[str]:
    """Why CHANGELOG.md *text* is not promoted to *version*, the version in
    backend/pyproject.toml. Empty when it is.

    test_all_version_fields_agree holds the other two version fields to
    pyproject.toml, so comparing with that one covers all three.
    """
    headings, links = _parse(text)
    versions = _versions(headings)
    labels = [label for _, label, _ in versions]
    newest = f"## {versions[0][0]}" if versions else "the newest version heading"
    found: list[str] = []

    # [Unreleased] comes first, once.
    unreleased = headings.count("[Unreleased]")
    if unreleased == 0:
        found.append(f"there is no ## [Unreleased] heading: open an empty one above {newest}")
    elif headings[0] != "[Unreleased]":
        found.append(f"## [Unreleased] has to be the first ## heading, and ## {headings[0]} "
                     f"is above it")
    if unreleased > 1:
        found.append(f"## [Unreleased] appears {unreleased} times: keep one, above {newest}")

    # The newest release sits right below it, dated, and is the version shipped.
    if not versions:
        found.append(f"there is no version heading: the newest release goes below "
                     f"## [Unreleased] as {_FORMAT}")
    elif unreleased:
        below = headings.index("[Unreleased]") + 1
        if below < len(headings) and not _BRACKETED.fullmatch(headings[below]):
            found.append(f"the heading below ## [Unreleased] is ## {headings[below]}: the "
                         f"newest release comes first, as {_FORMAT}")

    # Each version once, in file order, with its date where the heading has one.
    dates: dict[str, dt.date | None] = {}
    for heading, label, rest in versions:
        if label in dates:
            continue
        dates[label] = None
        dated = _DATED.fullmatch(rest)
        if dated is None:
            found.append(f"## {heading}: write it as ## [{label}] {EM_DASH} YYYY-MM-DD, "
                         f"with an em dash (U+2014) and one space either side")
            continue
        try:
            dates[label] = dt.date.fromisoformat(dated.group(1))
        except ValueError:
            found.append(f"## {heading}: {dated.group(1)} is not a date")
    for label in dates:
        if labels.count(label) > 1:
            found.append(f"## [{label}] appears {labels.count(label)} times: a release has "
                         f"one section")

    # Newest first. A section put below an older release is reported where it
    # sits, and the rules after this read the versions in number order: moving
    # the section up is the whole fix, so they must not ask for more.
    order = list(dates)
    for above, below in zip(order, order[1:]):
        key_above, key_below = _version_key(above), _version_key(below)
        if key_above and key_below and key_below > key_above:
            found.append(f"## [{below}] is below ## [{above}], an older release: move the "
                         f"## [{below}] section up, since the newest release comes first")
    order = _newest_first(order)

    if order and order[0] != version:
        if version in dates:
            found.append(f"newest version heading is [{order[0]}] but {_PYPROJECT} says "
                         f"{version}: bump {_PYPROJECT}, backend/uv.lock (run uv lock) and "
                         f'frontend/package.json to {order[0]}, step 2 of "Before you tag"')
        else:
            found.append(f"newest version heading is [{order[0]}] but {_PYPROJECT} says "
                         f"{version}: rename ## [Unreleased] to ## [{version}] {EM_DASH} "
                         f"YYYY-MM-DD and open a fresh ## [Unreleased] above it")

    # [Unreleased] compares from the newest release. Only the first definition
    # is checked, because it is the one Markdown renders.
    targets = links.get("unreleased", [])
    if not targets:
        found.append(f"[Unreleased] has no link definition: add [Unreleased]: "
                     f"{_COMPARE}{order[0] if order else 'X.Y.Z'}...main")
    else:
        if len(targets) > 1:
            found.append(_defined_twice("Unreleased", len(targets)))
        compared = _COMPARED.search(targets[0])
        if order and (compared is None or compared.group(1) != order[0]):
            found.append(f"[Unreleased]: {targets[0]} has to start at {order[0]}: make it "
                         f"{_COMPARE}{order[0]}...main")

    # Each release compares from the one below it. The oldest has nothing
    # below it, so only its link's head is checked.
    for i, label in enumerate(order):
        below = order[i + 1] if i + 1 < len(order) else None
        want = f"[{label}]: {_COMPARE}{below or '<previous>'}...{label}"
        targets = links.get(label.casefold(), [])
        if not targets:
            found.append(f"## [{label}] has no link definition: add {want}")
            continue
        if len(targets) > 1:
            found.append(_defined_twice(label, len(targets)))
        compared = _COMPARED.search(targets[0])
        if compared is None or compared.group(2) != label:
            found.append(f"[{label}]: {targets[0]} has to end ...{label}: make it {want}")
        elif below is not None and compared.group(1) != below:
            found.append(f"[{label}]: {targets[0]} compares from {compared.group(1)}, but the "
                         f"heading below [{label}] is [{below}]: make it {want}")

    # A release is not dated before the one it follows.
    if len(order) > 1:
        newer, older = dates[order[0]], dates[order[1]]
        if newer and older and newer < older:
            found.append(f"## [{order[0]}] is dated {newer}, before ## [{order[1]}] below it "
                         f"({older}): date it with the day its tag is created, in UTC")
    return found


def tag_problems(text: str, tag: str, created: dt.datetime) -> list[str]:
    """Why CHANGELOG.md *text* cannot ship as *tag*, created at *created*.

    The newest heading has to name the tag, less one leading ``v``, and carry
    the date the tag was created in UTC. At UTC+8 the local date runs a day
    ahead for eight hours of every day: 2.8.2 shipped dated 2026-09-16 with
    a tag created at 2026-09-17T05:22:48Z. *created* has to carry its UTC
    offset, because a naive datetime would be read as local time.

    ``[Unreleased]`` has to hold no entries. At the release commit every entry
    has moved under the new heading; entries still there mean the new heading
    was added below them instead of renaming ``[Unreleased]``, which the PR
    rules cannot tell from an ordinary PR, or that the tag is on a later commit.
    """
    if created.utcoffset() is None:
        raise ValueError("created needs a UTC offset; a naive datetime is read as local time")
    utc = created.astimezone(dt.timezone.utc)
    versions = _versions(_parse(text)[0])
    if not versions:
        return [f"there is no version heading to compare tag {tag} with"]
    newest = _newest_first([label for _, label, _ in versions])[0]
    heading, label, rest = next(section for section in versions if section[1] == newest)
    name = tag.removeprefix("v")
    if label != name:
        # A tag on the wrong commit and a misnamed tag look the same from here.
        renamed = tag[:len(tag) - len(name)] + label
        return [f"tag {tag} is not the newest version, ## {heading} is: if the tag is on "
                f"the wrong commit, tag the commit that promotes [Unreleased] to [{name}]; "
                f"if it has the wrong name, re-tag this commit as {renamed}"]
    found = []
    dated = _DATED.fullmatch(rest)
    if dated is None or dated.group(1) != utc.date().isoformat():
        found.append(f"## {heading} has to carry {utc.date()}, the UTC date tag {tag} was "
                     f"created on ({utc:%Y-%m-%dT%H:%M:%SZ}): fix the heading in a PR, then "
                     f"delete the tag and tag the new merge commit")
    entries = _unreleased_entries(text)
    if entries:
        found.append(f"## [Unreleased] still holds entries at tag {tag}, the first being "
                     f"{entries[0]!r}: the tag ships them, but CHANGELOG.md lists them as "
                     f"unreleased. If ## [{name}] was added below them instead of renaming "
                     f"## [Unreleased], move them under ## [{name}] in a PR and tag the new "
                     f"merge commit; if the tag is on a commit after the release, tag the "
                     f"release commit instead")
    return found


def _git(root: Path, *args: str) -> str:
    # Decoded as UTF-8 whatever the platform: the files carry em dashes, and
    # the Windows default (cp950 on a zh-TW machine) garbles them.
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True, encoding="utf-8", check=True,
    ).stdout


def tag_created(tag: str, root: Path = _REPO_ROOT) -> dt.datetime | None:
    """When *tag* was created, in UTC, or None if there is no such tag.

    ``creatordate`` is the tagger date of an annotated tag. It is read as a
    Unix time and converted here, because ``%(taggerdate:short)`` prints the
    tagger's local date, which is the UTC+8 trap RELEASING.md describes.
    """
    out = _git(root, "for-each-ref", "--format=%(creatordate:unix)",
               f"refs/tags/{tag}").strip()
    if not out:
        return None
    return dt.datetime.fromtimestamp(int(out), dt.timezone.utc)


def _project_version(pyproject: str) -> str | None:
    """The ``version`` in pyproject.toml's ``[project]`` table.

    Found by the table's header line rather than parsed as TOML, because
    tomllib arrived in Python 3.11.
    """
    table = None
    for line in pyproject.splitlines():
        if line.startswith("["):
            table = line.split("#", 1)[0].strip()
        elif table == "[project]":
            key = _VERSION_KEY.match(line)
            if key:
                return key.group(1)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_changelog.py",
        description="Check that CHANGELOG.md is promoted to the version it ships as.")
    parser.add_argument(
        "--tag",
        help="read both files at this tag, and check that the newest heading names "
             "it and carries the UTC date it was created on")
    parser.add_argument(
        "--root", type=Path, default=_REPO_ROOT,
        help="the repository to check (default: the one this script is in)")
    args = parser.parse_args(argv)

    created = None
    try:
        if args.tag is None:
            changelog = (args.root / _CHANGELOG).read_text(encoding="utf-8")
            pyproject = (args.root / _PYPROJECT).read_text(encoding="utf-8")
        else:
            created = tag_created(args.tag, args.root)
            if created is None:
                print(f"error: there is no tag {args.tag} in {args.root}", file=sys.stderr)
                return 1
            changelog = _git(args.root, "show", f"refs/tags/{args.tag}:{_CHANGELOG}")
            pyproject = _git(args.root, "show", f"refs/tags/{args.tag}:{_PYPROJECT}")
    except subprocess.CalledProcessError as exc:
        print(f"error: {exc.stderr.strip() or exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    version = _project_version(pyproject)
    if version is None:
        print(f"{_PYPROJECT}: there is no version in its [project] table")
        return 1
    found = problems(changelog, version)
    if created is not None:
        found += tag_problems(changelog, args.tag, created)
    for problem in found:
        print(f"{_CHANGELOG}: {problem}")
    if found:
        print(f"[FAIL] {len(found)} problem{'s' if len(found) > 1 else ''}; see {_STEP}")
        return 1
    if created is None:
        print(f"[OK] {_CHANGELOG} is promoted to {version}, the version in {_PYPROJECT}")
    else:
        print(f"[OK] tag {args.tag} was created {created:%Y-%m-%dT%H:%M:%SZ}, and "
              f"{_CHANGELOG} at the tag names it with that UTC date, {created.date()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
