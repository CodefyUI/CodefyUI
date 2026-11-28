"""Phase 4C step 1: walking, hashing and scanning a project's mods/, and
the trust record. Nothing here imports a mod."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from app.core import mods, projects

from .test_projects_startup import isolated  # noqa: F401  (fixture)

NODE = (
    "from app.core.node_base import BaseNode\n\n\n"
    "class Focal(BaseNode):\n"
    "    NODE_NAME = 'FocalLoss'\n"
    "    CATEGORY = 'Loss'\n"
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))  # LF on every platform
    return path


def _project(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        _write(root / "mods" / rel, text)
    return root


def _symlink(link: Path, target: Path, directory: bool = False) -> None:
    try:
        os.symlink(target, link, target_is_directory=directory)
    except (OSError, NotImplementedError):
        pytest.skip("this platform does not allow creating symlinks here")


# ── walk and digest ──────────────────────────────────────────────────────

def test_roles_and_digest(tmp_path):
    p = _project(tmp_path / "p", {"nodes/loss.py": NODE, "nodes/helper.py": "X = 1\n",
                                  "presets/aug.json": "{}", "README.md": "hi\n"})
    tree = mods.walk_mods(p)
    assert tree.refused == []
    assert tree.node_files == ["nodes/helper.py", "nodes/loss.py"]
    assert {r: f.role for r, f in tree.files.items()}["presets/aug.json"] == "preset"
    assert tree.files["README.md"].role == "other"
    first = mods.mods_digest(tree)
    assert first == mods.mods_digest(mods.walk_mods(p))
    _write(p / "mods" / "README.md", "changed\n")  # data changes the digest too (R15)
    assert mods.mods_digest(mods.walk_mods(p)) != first


def test_no_mods_folder_is_no_digest(tmp_path):
    assert mods.mods_digest(mods.walk_mods(tmp_path)) is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_the_executable_bit_is_part_of_the_digest(tmp_path):
    p = _project(tmp_path / "p", {"nodes/loss.py": NODE})
    before = mods.mods_digest(mods.walk_mods(p))
    os.chmod(p / "mods" / "nodes" / "loss.py", 0o755)
    assert mods.mods_digest(mods.walk_mods(p)) != before


def test_refused_entries(tmp_path, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside")
    _write(outside / "secret.py", "print('x')\n")
    p = _project(tmp_path / "p", {"nodes/loss.py": NODE,
                                  "nodes/pkg/__init__.py": "open('/tmp/x','w')\n",
                                  "nodes/old.cpython-311.pyc": "junk",
                                  "nodes/__pycache__/loss.cpython-311.pyc": "junk"})
    _symlink(p / "mods" / "nodes" / "leak.py", outside / "secret.py")
    _symlink(p / "mods" / "shared", outside, directory=True)
    refused = dict(mods.walk_mods(p).refused)
    assert set(refused) == {"nodes/pkg/", "nodes/old.cpython-311.pyc", "nodes/__pycache__/",
                            "nodes/leak.py", "shared/"}
    assert "symbolic link" in refused["nodes/leak.py"]
    assert "package" in refused["nodes/pkg/"]


def test_mods_itself_a_symlink_is_refused(tmp_path, tmp_path_factory):
    outside = tmp_path_factory.mktemp("mods-elsewhere")
    _write(outside / "nodes" / "x.py", NODE)
    (tmp_path / "p").mkdir()
    _symlink(tmp_path / "p" / "mods", outside, directory=True)
    tree = mods.walk_mods(tmp_path / "p")
    assert tree.files == {} and tree.refused == [("mods", "mods/ is a symbolic link")]


# ── scan ─────────────────────────────────────────────────────────────────

def test_scan_lists_findings_capabilities_names_and_duplicates(tmp_path):
    p = _project(tmp_path / "p", {
        "nodes/loss.py": NODE,
        "nodes/net.py": "import socket\n" + NODE.replace("FocalLoss", "Fetch"),
        "nodes/bad.py": "import os\nos.system('rm -rf ~')\n",
        "nodes/dup.py": NODE,
    })
    scan = mods.scan_mods(p)
    files = {f["path"]: f for f in scan["files"]}
    assert files["nodes/loss.py"]["finding"] is None
    assert files["nodes/loss.py"]["nodes"] == ["FocalLoss"]
    assert files["nodes/bad.py"]["finding"]["line"] in (1, 2)
    assert [n["capability"] for n in files["nodes/net.py"]["needs"]] == ["network"]
    assert scan["duplicates"] == {"FocalLoss": ["nodes/dup.py", "nodes/loss.py"]}
    assert scan["notice"].startswith("These files are Python.")


def test_scanning_imports_nothing(tmp_path):
    marker = tmp_path / "ran"
    p = _project(tmp_path / "p", {"nodes/x.py": f"open({str(marker)!r}, 'w').write('1')\n"})
    mods.scan_mods(p)
    mods.mods_trust_state(p)
    assert not marker.exists()


# ── trust ────────────────────────────────────────────────────────────────

def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@e",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@e",
                        "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"})


def test_a_clone_is_untrusted_then_trusted_then_changed(isolated):  # noqa: F811
    p = _project(isolated / "p", {"nodes/loss.py": NODE})
    assert mods.mods_trust_state(p)["state"] == "untrusted"
    mods.record_trust(p, mods.walk_mods(p), full=False)
    assert mods.mods_trust_state(p)["state"] == "trusted"
    _write(p / "mods" / "nodes" / "loss.py", NODE + "# edit\n")
    _write(p / "mods" / "nodes" / "new.py", "X = 2\n")
    state = mods.mods_trust_state(p)
    assert state["state"] == "changed"
    assert state["changed"] == {"added": ["nodes/new.py"], "removed": [],
                                "changed": ["nodes/loss.py"]}
    # The record lives in the user's data, never in the project.
    assert not any(f.name == "trust.json" for f in p.rglob("*"))


def test_presets_alone_need_no_prompt_and_refusals_block(isolated, tmp_path_factory):  # noqa: F811
    p = _project(isolated / "p", {"presets/aug.json": "{}"})
    assert mods.mods_trust_state(p)["state"] == "none"
    _write(p / "mods" / "nodes" / "__pycache__" / "x.pyc", "junk")
    assert mods.mods_trust_state(p)["state"] == "blocked"


def test_an_owned_project_without_git_is_trusted(isolated):  # noqa: F811
    p = _project(isolated / "p", {"nodes/loss.py": NODE})
    projects.mark_owned(p)
    assert mods.mods_trust_state(p)["state"] == "trusted"


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_an_owned_project_asks_when_mods_changed_with_head(isolated):  # noqa: F811
    p = _project(isolated / "p", {"nodes/loss.py": NODE})
    _git(p, "init", "-q")
    _git(p, "add", "-A")
    _git(p, "commit", "-qm", "one")
    projects.mark_owned(p)
    assert mods.mods_trust_state(p)["state"] == "trusted"   # first open: the owner's
    mods.record_trust(p, mods.walk_mods(p), full=True)
    # The owner edits a file: HEAD did not move, no prompt.
    _write(p / "mods" / "nodes" / "loss.py", NODE + "# mine\n")
    assert mods.mods_trust_state(p)["state"] == "trusted"
    # A commit moves HEAD with changed mods (as a pull would): ask.
    _git(p, "add", "-A")
    _git(p, "commit", "-qm", "two")
    assert mods.mods_trust_state(p)["state"] == "changed"
