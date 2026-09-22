"""The release tarball carries the legal files and every file they point to.

``frontend-dist.tar.gz`` is packed from ``frontend/dist``, and one ``cp`` line
in ``.github/workflows/release-build.yml`` copies the legal files in beside
the bundle. NOTICE ends its commercial paragraph with "See
COMMERCIAL-LICENSE.md." and THIRD_PARTY_NOTICES.md links the same file, but
that line never listed it (#459): the reader the commercial offer is written
for -- someone reviewing the tarball off the Releases page -- followed both
references to a file the tarball did not have. NOTICE and
COMMERCIAL-LICENSE.md then send that reader on to CONTRIBUTING.md for the
inbound dual-licensing term the commercial path rests on, and that file was
missing too.

The rule has no exceptions: the legal files, and every repository-root file
any of them names, travel together. What they name is read off the files
rather than repeated here, so the next file one of them starts pointing to
fails this test instead of quietly missing from a release. The workflows'
checks of the tarball list the same files again, and are held to the line.
Neither release workflow runs on a pull request, so this is the only check
that sees any of those lists before a tag.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RELEASE_BUILD = _REPO_ROOT / ".github" / "workflows" / "release-build.yml"
_INSTALL_CHECK = _REPO_ROOT / ".github" / "workflows" / "install-check.yml"

#: The legal files, read for the files they name. A file they name ships
#: without being read in turn: CONTRIBUTING.md rides along for the inbound
#: terms NOTICE and COMMERCIAL-LICENSE.md point to, and the rest of its links
#: belong to the developer guide it also is. A new license document goes here.
_LEGAL = ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "COMMERCIAL-LICENSE.md")

#: ``cp LICENSE NOTICE ... frontend/dist/`` in the "Package dist" step.
_CP_INTO_DIST = re.compile(
    r"^[ \t]*cp[ \t]+(?P<files>[^\n]+?)[ \t]+frontend/dist/?[ \t]*$", re.MULTILINE)

#: ``for f in LICENSE NOTICE ...; do``: release-build.yml's check of the
#: archive, and install-check.yml's check of what install.sh unpacked.
_BASH_LOOP = re.compile(
    r"^[ \t]*for[ \t]+f[ \t]+in[ \t]+(?P<files>[^;\n]+);[ \t]*do[ \t]*$", re.MULTILINE)

#: ``foreach ($f in 'LICENSE', 'NOTICE', ...) {``: install-check.yml's check
#: of what install.ps1 unpacked.
_PWSH_LOOP = re.compile(
    r"^[ \t]*foreach[ \t]*\([ \t]*\$f[ \t]+in[ \t]+(?P<files>[^)\n]+)\)[ \t]*\{[ \t]*$",
    re.MULTILINE)

#: A ``./`` where a path starts. ``./CONTRIBUTING.md`` is the root file as much
#: as ``CONTRIBUTING.md`` is; in ``../CONTRIBUTING.md`` or ``docs/./X`` the
#: file is somewhere else, so those are left alone.
_LEADING_DOT_SLASH = re.compile(r"(?<![\w./-])\./")


def _names(text: str, name: str) -> bool:
    """Does *text* name the file *name* on its own, not inside another name?

    LICENSE is not named by "COMMERCIAL-LICENSE.md", NOTICE is not named by
    "THIRD_PARTY_NOTICES.md", and "backend/pyproject.toml" is a file in a
    subdirectory, not the root file of that name. A leading ``./`` is dropped
    first, since it is an ordinary way to spell a link to the root file.
    """
    text = _LEADING_DOT_SLASH.sub("", text)
    return re.search(rf"(?<![\w./-]){re.escape(name)}(?![\w-]|\.\w)", text) is not None


def _the_one(pattern: re.Pattern[str], workflow: Path, what: str) -> str:
    """The file list in the one line of *workflow* that *pattern* matches."""
    found = pattern.findall(workflow.read_text(encoding="utf-8"))
    assert len(found) == 1, (
        f"expected one {what} in {workflow.name}, found {len(found)}; the "
        f"legal files are read off it")
    return found[0]


def _copied_into_dist() -> list[str]:
    return _the_one(_CP_INTO_DIST, _RELEASE_BUILD, "`cp ... frontend/dist/` line").split()


@pytest.mark.parametrize(("text", "name", "named"), [
    ("See COMMERCIAL-LICENSE.md.", "COMMERCIAL-LICENSE.md", True),
    ("see [LICENSE](LICENSE).", "LICENSE", True),
    ("see [the guide](./CONTRIBUTING.md).", "CONTRIBUTING.md", True),
    ("see ../CONTRIBUTING.md", "CONTRIBUTING.md", False),
    ("See COMMERCIAL-LICENSE.md.", "LICENSE", False),
    ("THIRD_PARTY_NOTICES.md, which is", "NOTICE", False),
    ("`backend/pyproject.toml`", "pyproject.toml", False),
    ("the cdui.cmd launcher", "cdui", False),
])
def test_a_file_counts_as_named_only_on_its_own(text, name, named):
    assert _names(text, name) is named


def test_every_root_file_the_legal_files_name_is_copied_into_the_tarball():
    root_files = [p.name for p in _REPO_ROOT.iterdir() if p.is_file()]
    named_by: dict[str, list[str]] = {}
    for legal in _LEGAL:
        text = (_REPO_ROOT / legal).read_text(encoding="utf-8")
        for name in root_files:
            if name != legal and _names(text, name):
                named_by.setdefault(name, []).append(legal)

    missing = sorted((set(_LEGAL) | set(named_by)) - set(_copied_into_dist()))
    why = [f"{name} (named by {', '.join(named_by[name])})" if name in named_by
           else f"{name} (a legal file)" for name in missing]
    assert not missing, (
        f"{'; '.join(why)}: not copied into frontend/dist by "
        f"{_RELEASE_BUILD.name}, so the release tarball sends its reader to a "
        f"file it does not carry. Add it to the cp line and to the checks in "
        f"release-build.yml and install-check.yml.")


def test_every_check_of_the_tarball_names_the_files_the_cp_line_copies():
    """Three checks list the files again: release-build.yml's, on the archive
    before it is attached, and install-check.yml's two, on what install.sh
    and install.ps1 unpacked. They list the names rather than read them off
    the cp line -- a file dropped from that line has to fail there, not agree
    with it -- so this test is what holds the four lists together. Order does
    not matter; the names do."""
    copied = sorted(_copied_into_dist())
    loop = "`for f in ...; do` loop"
    checks = {
        f"the archive check in {_RELEASE_BUILD.name}":
            _the_one(_BASH_LOOP, _RELEASE_BUILD, loop).split(),
        f"the install.sh check in {_INSTALL_CHECK.name}":
            _the_one(_BASH_LOOP, _INSTALL_CHECK, loop).split(),
        f"the install.ps1 check in {_INSTALL_CHECK.name}": [
            name.strip().strip("'\"") for name in _the_one(
                _PWSH_LOOP, _INSTALL_CHECK, "`foreach ($f in ...)` loop").split(",")],
    }

    disagree = [f"{where} lists {' '.join(sorted(names))}"
                for where, names in checks.items() if sorted(names) != copied]
    assert not disagree, (
        f"the cp line in {_RELEASE_BUILD.name} copies {' '.join(copied)}, but "
        f"{'; '.join(disagree)}. Change the four lists together.")
