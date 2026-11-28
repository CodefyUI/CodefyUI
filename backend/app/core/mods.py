"""Project mods (phase 4C): ``<project>/mods/nodes/*.py`` and
``<project>/mods/presets/*.json``.

A project's Python is code that runs in this server with the user's
permissions, and a project can come from anywhere (a clone, a pull). So the
mods are never imported by discovery. This module is the only place anything
under ``mods/`` is read, and it reads in this order:

1. :func:`walk_mods` -- every entry under ``mods/``, without following a
   symbolic link and without importing anything. Links, non-regular files,
   compiled files, ``__pycache__`` and packages (a sub-folder of
   ``mods/nodes/`` with an ``__init__.py``) are *refused*: any refusal blocks
   the whole of ``mods/``.
2. :func:`mods_digest` -- the sha256 of the whole tree (paths, modes, bytes).
3. :func:`scan_mods` -- for each node file, the plugin gate's verdict (a
   hint, not a sandbox), the capabilities its imports would need, and the
   node names it defines, read from the AST.
4. :func:`mods_trust_state` -- whether ``<USER_DATA>/trust.json`` holds the
   user's agreement to exactly this tree (never the project itself: a repo
   must not vouch for itself).
5. :func:`load_project_mods` -- only then: each trusted file's bytes are
   checked against the per-file hash the user agreed to, compiled and
   executed into a module of the project's own package, through no path
   finder and with no bytecode.

Project nodes register as ``project:<NODE_NAME>`` (``provider = project``): a
project node never shares a name with a built-in, plugin or global custom
node, and the bare-name lookup never reaches one.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import os
import stat
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MODS_DIR = "mods"
NODES_DIR = "nodes"
PRESETS_DIR = "presets"
#: The plugin id project nodes are qualified with; no plugin may use it.
PROJECT_PREFIX = "project"
#: The largest node file the loader reads (the graph-folder file cap).
MAX_MOD_BYTES = 16 * 1024 * 1024

#: The sentence every prompt carries, in the order the review asked for.
SCAN_IS_A_HINT = (
    "These files are Python. Trusting them runs them in this CodefyUI server "
    "with your permissions. The scan result is a hint, not a sandbox."
)


@dataclass(frozen=True)
class ModFile:
    path: str            # relative to mods/, posix
    sha256: str
    size: int
    executable: bool
    role: str            # "node" (loaded), "preset" (loaded), "other" (data, unused)


@dataclass
class ModsTree:
    """What :func:`walk_mods` found. ``refused`` blocks the whole tree."""

    root: Path
    files: dict[str, ModFile] = field(default_factory=dict)
    refused: list[tuple[str, str]] = field(default_factory=list)

    @property
    def node_files(self) -> list[str]:
        return sorted(p for p, f in self.files.items() if f.role == "node")

    @property
    def exists(self) -> bool:
        return bool(self.files or self.refused)


def _role(rel: str) -> str:
    parts = rel.split("/")
    if len(parts) == 2 and parts[0] == NODES_DIR and parts[1].endswith(".py"):
        return "node"
    if len(parts) == 2 and parts[0] == PRESETS_DIR and parts[1].endswith(".json"):
        return "preset"
    return "other"


def _compiled(name: str) -> bool:
    from .plugins.gate import loader_suffix

    suffix = loader_suffix(Path(name))
    return suffix is not None and suffix not in (".py", ".pyw")


def _read_no_follow(path: Path) -> bytes:
    """*path*'s bytes, refusing to open it through a symbolic link."""
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise OSError(f"{path} is not a regular file")
        if info.st_size > MAX_MOD_BYTES:
            raise OSError(f"{path} is {info.st_size} bytes, over {MAX_MOD_BYTES}")
        chunks = []
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def walk_mods(project: Path) -> ModsTree:
    """Every entry under ``<project>/mods``, without following a link."""
    root = Path(project) / MODS_DIR
    tree = ModsTree(root=root)
    if os.path.islink(root):
        tree.refused.append((MODS_DIR, "mods/ is a symbolic link"))
        return tree
    if not root.is_dir():
        return tree
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(dirpath)
        rel_dir = base.relative_to(root).as_posix()
        for name in sorted(dirnames):
            path = base / name
            rel = (path.relative_to(root)).as_posix()
            reason = None
            if path.is_symlink():
                reason = "a symbolic link to a folder"
            elif name == "__pycache__":
                reason = "compiled bytecode (__pycache__); it could run instead of the source"
            elif rel_dir == NODES_DIR and (path / "__init__.py").exists():
                reason = "a package (a folder with __init__.py); packages are not loaded"
            if reason:
                tree.refused.append((rel + "/", reason))
                dirnames.remove(name)
        dirnames.sort()
        for name in sorted(filenames):
            path = base / name
            rel = path.relative_to(root).as_posix()
            info = os.lstat(path)
            if stat.S_ISLNK(info.st_mode):
                tree.refused.append((rel, "a symbolic link"))
                continue
            if not stat.S_ISREG(info.st_mode):
                tree.refused.append((rel, "not a regular file"))
                continue
            if _compiled(name):
                tree.refused.append((rel, "a compiled file; the scan cannot read it"))
                continue
            try:
                data = _read_no_follow(path)
            except OSError as exc:
                tree.refused.append((rel, f"cannot be read: {exc}"))
                continue
            tree.files[rel] = ModFile(
                path=rel, sha256=hashlib.sha256(data).hexdigest(), size=len(data),
                executable=bool(info.st_mode & 0o111), role=_role(rel))
    return tree


def mods_digest(tree: ModsTree) -> str | None:
    """sha256 over every file's path, mode and bytes; None for no mods."""
    if not tree.files and not tree.refused:
        return None
    h = hashlib.sha256()
    for rel in sorted(tree.files):
        f = tree.files[rel]
        h.update(f"{rel}\0{'x' if f.executable else '-'}\0{f.sha256}\n".encode("utf-8"))
    for rel, reason in sorted(tree.refused):
        h.update(f"!{rel}\0{reason}\n".encode("utf-8"))
    return h.hexdigest()


# ── The scan (never imports) ──────────────────────────────────────────────

def _node_names(source: str) -> list[str]:
    """``NODE_NAME = "..."`` string literals assigned in a class body."""
    names = []
    try:
        module = ast.parse(source)
    except SyntaxError:
        return names
    for node in ast.walk(module):
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            targets = item.targets if isinstance(item, ast.Assign) else (
                [item.target] if isinstance(item, ast.AnnAssign) else [])
            value = getattr(item, "value", None)
            if any(isinstance(t, ast.Name) and t.id == "NODE_NAME" for t in targets) \
                    and isinstance(value, ast.Constant) and isinstance(value.value, str) \
                    and value.value:
                names.append(value.value)
    return names


def _capabilities_needed(source: str) -> list[str]:
    from .security_tiers import capability_for_module

    found: set[str] = set()
    try:
        module = ast.parse(source)
    except SyntaxError:
        return []
    for node in ast.walk(module):
        roots: list[str] = []
        if isinstance(node, ast.Import):
            roots = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            roots = [node.module.split(".")[0]]
        for root in roots:
            capability = capability_for_module(root)
            if capability:
                found.add(capability)
    return sorted(found)


def scan_file(rel: str, data: bytes) -> dict[str, Any]:
    """One node file: the gate's verdict, the capabilities its imports
    would need, its node names. Nothing is imported."""
    from .plugin_validator import PluginValidationError, validate_python_source
    from .plugins.gate import _denied_attributes_for
    from .security_tiers import describe_capability

    text = data.decode("utf-8", "replace")
    finding = None
    try:
        validate_python_source(data, rel, allowed_modules=[], capabilities=[],
                               denied_attributes=_denied_attributes_for([]))
    except PluginValidationError as exc:
        finding = {"line": exc.lineno, "message": str(exc)}
    except Exception as exc:  # noqa: BLE001 -- the scan must not crash the prompt
        finding = {"line": None, "message": f"the scan could not read this file: {exc}"}
    needs = _capabilities_needed(text)
    return {
        "path": rel,
        "lines": text.count("\n") + (0 if text.endswith("\n") or not text else 1),
        "finding": finding,
        "needs": [{"capability": c, "summary": describe_capability(c)} for c in needs],
        "nodes": _node_names(text),
    }


def scan_mods(project: Path, tree: ModsTree | None = None) -> dict[str, Any]:
    """The prompt's content: every file, each node file's scan, refused
    entries, duplicate node names (both files refused), and the digest."""
    tree = tree or walk_mods(project)
    scans = {}
    for rel in tree.node_files:
        try:
            scans[rel] = scan_file(rel, _read_no_follow(tree.root / rel))
        except OSError as exc:
            tree.refused.append((rel, f"cannot be read: {exc}"))
    owners: dict[str, list[str]] = {}
    for rel, scan in scans.items():
        for name in scan["nodes"]:
            owners.setdefault(name, []).append(rel)
    duplicates = {name: files for name, files in owners.items() if len(files) > 1}
    for name, files in duplicates.items():
        for rel in files:
            scans[rel].setdefault("duplicates", []).append(name)
    files = []
    for rel in sorted(tree.files):
        f = tree.files[rel]
        entry: dict[str, Any] = {"path": rel, "size": f.size, "role": f.role,
                                 "sha256": f.sha256}
        if rel in scans:
            entry.update({k: v for k, v in scans[rel].items() if k != "path"})
        files.append(entry)
    return {
        "digest": mods_digest(tree),
        "files": files,
        "refused": [{"path": p, "reason": r} for p, r in sorted(tree.refused)],
        "duplicates": duplicates,
        "notice": SCAN_IS_A_HINT,
    }


# ── Trust ─────────────────────────────────────────────────────────────────

def _git_head(project: Path) -> str | None:
    """The project's HEAD commit, or None (no git, not a repository, unborn)."""
    try:
        proc = subprocess.run(["git", "-C", str(project), "rev-parse", "--verify", "-q", "HEAD"],
                              capture_output=True, text=True, timeout=10,
                              stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None
    head = proc.stdout.strip()
    return head if proc.returncode == 0 and head else None


def _git_repo(project: Path) -> bool:
    try:
        proc = subprocess.run(["git", "-C", str(project), "rev-parse", "--is-inside-work-tree"],
                              capture_output=True, text=True, timeout=10,
                              stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def _entry(project: Path) -> dict[str, Any]:
    from . import projects

    entry = projects._trust_entries().get(projects._key(project))
    return entry if isinstance(entry, dict) else {}


def record_trust(project: Path, tree: ModsTree, *, full: bool) -> None:
    """The user agreed to run exactly *tree* (and, with *full*, also the
    files the scan had findings for)."""
    from . import projects

    projects._update_trust(
        project,
        mods_sha256=mods_digest(tree),
        mods_files={rel: f.sha256 for rel, f in tree.files.items()},
        mods_head=_git_head(project),
        mods_full=bool(full),
        mods_trusted_at=projects._now(),
    )


def mods_trust_state(project: Path, tree: ModsTree | None = None) -> dict[str, Any]:
    """``{state, digest, changed: {added, changed, removed}}``.

    ``none``: no Python under ``mods/`` (presets alone are data: loaded).
    ``blocked``: something refused (a link, bytecode, a package); nothing loads.
    ``trusted``: the record matches this tree.
    ``untrusted``: never agreed. ``changed``: agreed to a different tree.

    An owned project (created or imported here) is trusted without asking
    while its HEAD has not moved since the record (the owner editing their
    own files), or when it is not a git repository at all; a change that
    came with a moved HEAD (a pull, a checkout) asks like any other.
    """
    tree = tree or walk_mods(project)
    digest = mods_digest(tree)
    answer: dict[str, Any] = {"digest": digest, "changed": None}
    if tree.refused:
        return {**answer, "state": "blocked"}
    if not tree.node_files:
        return {**answer, "state": "none"}
    entry = _entry(project)
    if entry.get("mods_sha256") == digest:
        return {**answer, "state": "trusted", "full": bool(entry.get("mods_full"))}
    if entry.get("owned") is True:
        head = _git_head(project)
        if not _git_repo(project) or "mods_sha256" not in entry \
                or entry.get("mods_head") == head:
            return {**answer, "state": "trusted", "owned": True,
                    "full": bool(entry.get("mods_full", True))}
    old = entry.get("mods_files") if isinstance(entry.get("mods_files"), dict) else None
    if old is None:
        return {**answer, "state": "untrusted"}
    now = {rel: f.sha256 for rel, f in tree.files.items()}
    answer["changed"] = {
        "added": sorted(set(now) - set(old)),
        "removed": sorted(set(old) - set(now)),
        "changed": sorted(r for r in set(now) & set(old) if now[r] != old[r]),
    }
    return {**answer, "state": "changed"}
