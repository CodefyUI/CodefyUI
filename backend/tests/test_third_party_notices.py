"""THIRD_PARTY_NOTICES.md lists the dependencies the manifests declare.

Section 1 of the notices lists the frontend's production dependencies (the
``dependencies`` of ``frontend/package.json``) and section 2 the backend's
direct runtime dependencies (``[project.dependencies]`` of
``backend/pyproject.toml``), one table each. Nothing kept either table in
step with its manifest: #497 declared ``packaging`` and added its row by hand
only because a reviewer noticed (#505).

Names only. The versions in the frontend table are the ones that were
verified, not the declared ranges, so they are allowed to differ.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # 3.10 backport -- same API.

_REPO_ROOT = Path(__file__).resolve().parents[2]
_NOTICES = _REPO_ROOT / "THIRD_PARTY_NOTICES.md"
_PACKAGE_JSON = _REPO_ROOT / "frontend" / "package.json"
_PYPROJECT = _REPO_ROOT / "backend" / "pyproject.toml"

#: "@xyflow/react (React Flow)", "tomli (Python < 3.11 only)": a note after
#: the name, not part of it.
_TRAILING_NOTE = re.compile(r"\s*\(.*\)$")


def _table_names(section: str) -> list[str]:
    """The first column of the first table under the ``## `` heading naming *section*."""
    lines = _NOTICES.read_text(encoding="utf-8").splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("## ") and section in line]
    assert len(starts) == 1, (
        f"expected one '## ' heading with {section!r} in {_NOTICES.name}, found {len(starts)}")
    rows: list[str] = []
    for line in lines[starts[0] + 1:]:
        if line.startswith("#"):
            break
        if line.startswith("|"):
            rows.append(line)
        elif rows:
            break
    # rows[0] is the header row and rows[1] the |---| row under it.
    assert len(rows) > 2, f"no table under the {section!r} heading of {_NOTICES.name}"
    return [_TRAILING_NOTE.sub("", row.split("|")[1].strip()) for row in rows[2:]]


def _mismatch(table: str, manifest: str, listed: list[str], declared: set[str]) -> str:
    """Empty when *listed* and *declared* agree, else what to change and where."""
    problems = []
    twice = sorted({name for name in listed if listed.count(name) > 1})
    if twice:
        problems.append(f"listed twice: {', '.join(twice)}")
    missing = sorted(declared - set(listed))
    if missing:
        problems.append(f"declared in {manifest} but not listed: {', '.join(missing)}")
    extra = sorted(set(listed) - declared)
    if extra:
        problems.append(f"listed but not declared in {manifest}: {', '.join(extra)}")
    if not problems:
        return ""
    return f"{_NOTICES.name}, {table}: {'; '.join(problems)}. Update the table to match."


def test_the_frontend_table_lists_every_production_dependency():
    declared = set(json.loads(_PACKAGE_JSON.read_text(encoding="utf-8"))["dependencies"])
    listed = _table_names("Frontend dependencies")

    problem = _mismatch("section 1", "frontend/package.json", listed, declared)
    assert not problem, problem


def test_the_backend_table_lists_every_direct_runtime_dependency():
    # Compared as PEP 503 names, the way an installer compares them: the table
    # writes Pillow and huggingface_hub, as their projects do.
    project = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["project"]
    declared = {canonicalize_name(Requirement(spec).name) for spec in project["dependencies"]}
    listed = [canonicalize_name(name) for name in _table_names("Backend Python dependencies")]

    problem = _mismatch("section 2", "backend/pyproject.toml [project.dependencies]", listed, declared)
    assert not problem, problem
