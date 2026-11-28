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
import json
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


#: Why a file under mods/ is never loaded (role ``other``): the prompt says
#: so, so nobody agrees thinking it will run.
NOT_LOADED_REASON = ("never loaded: only the .py files directly in mods/nodes/ "
                     "(not __init__.py) and mods/presets/*.json are read")


def _role(rel: str) -> str:
    parts = rel.split("/")
    if len(parts) == 2 and parts[0] == NODES_DIR and parts[1].endswith(".py") \
            and parts[1] != "__init__.py":
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


def _import_at(source: str, line: int | None) -> str | None:
    """The root module imported on *line*, if that line is an import."""
    if line is None:
        return None
    try:
        module = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(module):
        if getattr(node, "lineno", None) != line:
            continue
        if isinstance(node, ast.Import) and node.names:
            return node.names[0].name.split(".")[0]
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            return node.module.split(".")[0]
    return None


def _mods_finding(source: str, exc: Exception) -> dict[str, Any]:
    """``{line, kind, module?, capability?, construct?, message}``.

    ``kind``: ``import`` (a module that needs a capability; ``capability`` is
    its name, described by ``describe_capability``), ``blocked_import`` (a
    module nothing unlocks), ``refused`` (a call or attribute the scan does
    not allow; ``construct`` is the name it quoted, if any) or ``syntax``.
    ``message`` is the English sentence (CLI); the editor words it from the
    fields."""
    import re

    from .security_tiers import capability_for_module, describe_capability

    line = getattr(exc, "lineno", None)
    text = str(exc)
    where = f"line {line}: " if line else ""
    if text.startswith("Syntax error"):
        return {"line": line, "kind": "syntax",
                "message": f"{where}the file is not valid Python"}
    module = _import_at(source, line)
    if module is not None:
        capability = capability_for_module(module)
        if capability:
            return {"line": line, "kind": "import", "module": module,
                    "capability": capability,
                    "message": f"{where}imports {module}: needs {capability} "
                               f"({describe_capability(capability)})"}
        return {"line": line, "kind": "blocked_import", "module": module,
                "message": f"{where}imports {module}, which the scan never allows"}
    quoted = re.search(r"'([^']{1,60})'", text)
    construct = quoted.group(1) if quoted else None
    shown = f" ({construct})" if construct else ""
    return {"line": line, "kind": "refused", "construct": construct,
            "message": f"{where}uses a call or attribute the scan does not allow{shown}"}


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
        # The gate's verdict and line, in the words of a project's mods: its
        # own text tells a plugin author to edit a plugin manifest, which a
        # project does not have.
        finding = _mods_finding(text, exc)
    except Exception:  # noqa: BLE001 -- the scan must not crash the prompt
        finding = {"line": None, "kind": "unreadable",
                   "message": "the scan could not read this file"}
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
    # Files that load first, then the ones that never do.
    for rel in sorted(tree.files, key=lambda r: (tree.files[r].role == "other", r)):
        f = tree.files[rel]
        entry: dict[str, Any] = {"path": rel, "size": f.size, "role": f.role,
                                 "sha256": f.sha256, "loads": f.role != "other"}
        if f.role == "other":
            entry["reason"] = NOT_LOADED_REASON
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


def record_baseline(project: Path) -> None:
    """For an owned project: the HEAD and mods/ tree as they are now, even
    when there are no mods (the "empty" baseline). Called when a project is
    marked owned and at every server load of an owned project."""
    record_trust(project, walk_mods(project), full=True)


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
        # The owner's own edits load without asking only against a baseline
        # (the HEAD and tree recorded when it was marked owned and at every
        # load, mods or not) whose HEAD has not moved -- or outside git. No
        # baseline (a project owned before 4C) asks: code that arrived with
        # a pull must never run on the strength of "no record yet".
        if not _git_repo(project) or (
                "mods_head" in entry and entry.get("mods_head") == _git_head(project)):
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


# ── Loading (only after trust) ────────────────────────────────────────────

import importlib.abc  # noqa: E402
import importlib.machinery  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import types  # noqa: E402

_lock = threading.RLock()


@dataclass
class LoadedMods:
    project: str | None = None
    package: str | None = None
    finder: Any = None
    nodes: list[str] = field(default_factory=list)
    presets: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    state: str = "none"
    #: The digest of the tree the last load looked at (loaded or not).
    digest: str | None = None


_loaded = LoadedMods()
#: Every project whose mods this process has executed. Unloading is not
#: isolation, so once another project's code has run here, a restart is
#: recommended.
_ran: set[str] = set()

#: True only in the server process (set by the app's lifespan). Anywhere
#: else -- the CLI, an exported script, ``run_graph.py`` -- mods load only on
#: a trust record that matches the tree exactly: no owner shortcut, and
#: nothing is recorded.
_server_process = False


def set_server_process() -> None:
    global _server_process
    _server_process = True


def package_name(project: Path) -> str:
    from . import projects

    return f"codefyui_project_{projects.project_id(project)}"


class _ModsFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serves ``<package>.<stem>`` for the trusted top-level node files and
    nothing else: the bytes are read once (no link followed), checked
    against the hash the user agreed to, compiled and executed. No path
    finder and no bytecode is ever involved, so a ``.pyc`` in the project
    cannot be what runs."""

    def __init__(self, package: str, root: Path, allowed: dict[str, str]) -> None:
        self.package = package
        self.root = root          # mods/nodes
        self.allowed = allowed    # stem -> sha256

    def find_spec(self, fullname, path=None, target=None):  # noqa: D401
        if not fullname.startswith(self.package + "."):
            return None
        stem = fullname[len(self.package) + 1:]
        if "." in stem or stem not in self.allowed:
            return None
        return importlib.machinery.ModuleSpec(
            fullname, self, origin=str(self.root / f"{stem}.py"))

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        stem = module.__name__[len(self.package) + 1:]
        path = self.root / f"{stem}.py"
        data = _read_no_follow(path)
        if hashlib.sha256(data).hexdigest() != self.allowed[stem]:
            raise ImportError(f"mods/nodes/{stem}.py changed after it was trusted; "
                              "it was not loaded")
        module.__file__ = str(path)
        module.__package__ = self.package
        code = compile(data, str(path), "exec", dont_inherit=True)
        exec(code, module.__dict__)  # noqa: S102 -- trusted by the user, see module docstring


def _purge(package: str | None) -> None:
    if not package:
        return
    for name in [n for n in sys.modules if n == package or n.startswith(package + ".")]:
        del sys.modules[name]


def unload_project_mods() -> None:
    """Unregister the open project's mods: registry entries, presets, its
    modules and its finder. Not isolation: code that ran may have left
    effects that only a restart removes."""
    from .node_registry import registry
    from .preset_registry import preset_registry

    global _loaded
    with _lock:
        registry.unregister_prefix(f"{PROJECT_PREFIX}:")
        for name in _loaded.presets:
            preset_registry.remove(name)
        if _loaded.finder is not None and _loaded.finder in sys.meta_path:
            sys.meta_path.remove(_loaded.finder)
        _purge(_loaded.package)
        _loaded = LoadedMods()


def _load(project: Path, tree: ModsTree, entry_files: dict[str, str], full: bool) -> None:
    from .node_base import BaseNode
    from .node_registry import registry
    from .preset_registry import preset_registry

    package = package_name(project)
    scan = scan_mods(project, tree)
    by_path = {f["path"]: f for f in scan["files"]}
    loaded = LoadedMods(project=str(project), package=package, state="trusted")
    allowed: dict[str, str] = {}
    for rel in tree.node_files:
        info = by_path.get(rel, {})
        stem = rel.split("/", 1)[1][:-3]
        if not stem.isidentifier():
            loaded.skipped.append({"path": rel, "reason": "the file name is not a Python identifier"})
            continue
        if info.get("duplicates"):
            loaded.skipped.append({"path": rel, "reason": "defines a node name another file "
                                   "also defines: " + ", ".join(info["duplicates"])})
            continue
        if info.get("finding") and not full:
            loaded.skipped.append({"path": rel, "reason": "the scan had a finding; trust "
                                   "these files completely to load it"})
            continue
        if entry_files.get(rel) != tree.files[rel].sha256:
            loaded.skipped.append({"path": rel, "reason": "not the file that was trusted"})
            continue
        allowed[stem] = tree.files[rel].sha256
    pkg = types.ModuleType(package)
    pkg.__path__ = []  # nothing on disk: only the finder below serves it
    pkg.__package__ = package
    sys.modules[package] = pkg
    finder = _ModsFinder(package, tree.root / NODES_DIR, allowed)
    sys.meta_path.insert(0, finder)
    loaded.finder = finder
    for stem in sorted(allowed):
        name = f"{package}.{stem}"
        try:
            module = sys.modules.get(name) or importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 -- one bad mod must not stop the rest
            loaded.skipped.append({"path": f"nodes/{stem}.py", "reason": f"failed to load: {exc}"})
            logger.warning("project mod %s failed to load: %s", stem, exc)
            continue
        _ran.add(str(project))
        for obj in vars(module).values():
            if isinstance(obj, type) and obj is not BaseNode and getattr(obj, "__module__", "") == name:
                try:
                    is_node = issubclass(obj, BaseNode)
                except TypeError:
                    continue
                if is_node and getattr(obj, "NODE_NAME", None):
                    loaded.nodes.append(registry.register(obj, plugin_id=PROJECT_PREFIX))
    # Presets after nodes: one whose internal nodes did not load is skipped.
    for rel in sorted(r for r, f in tree.files.items() if f.role == "preset"):
        try:
            raw = json.loads(_read_no_follow(tree.root / rel).decode("utf-8"))
            # Parsed and resolved, not registered yet: a name already taken
            # (a built-in's) must not be replaced.
            preset = preset_registry._load_and_resolve(raw, registry)
        except Exception as exc:  # noqa: BLE001 -- e.g. it uses a node that did not load
            loaded.skipped.append({"path": rel, "reason": f"not loaded: {exc}"})
            continue
        if preset_registry.get(preset.preset_name) is not None:
            loaded.skipped.append({"path": rel, "reason": f"a preset named "
                                   f"{preset.preset_name!r} already exists"})
            continue
        preset_registry.add(preset)
        loaded.presets.append(preset.preset_name)
    global _loaded
    _loaded = loaded


def load_project_mods(project: Path | None, *, cli: bool = False) -> dict[str, Any]:
    """Load *project*'s mods if, and only if, the user trusted exactly this
    tree (or it is an owned project whose HEAD has not moved). *cli*: the
    command line loads only on a matching trust record. Returns the status."""
    cli = cli or not _server_process
    with _lock:
        unload_project_mods()
        if project is None:
            return current_status(None)
        tree = walk_mods(project)
        state = mods_trust_state(project, tree)
        status = state["state"]
        owned = _entry(project).get("owned") is True
        if owned and not cli and status in ("none", "trusted"):
            # The baseline, at every load, mods or not: what a later pull is
            # compared with.
            record_trust(project, tree, full=True)
        if status == "none" and tree.files:
            # Presets alone: data, loaded without a prompt.
            _load(project, tree, {}, full=False)
            _loaded.state = "none"
        elif status == "trusted":
            entry = _entry(project)
            if cli and entry.get("mods_sha256") != state["digest"]:
                _loaded.project, _loaded.state = str(project), "untrusted"
            else:
                _load(project, tree, entry.get("mods_files") or {},
                      full=bool(entry.get("mods_full")))
        else:
            _loaded.project, _loaded.state = str(project), status
        _loaded.project = str(project)
        _loaded.digest = state["digest"]
        return current_status(project)


def reload_project_mods(project: Path | None) -> dict[str, Any]:
    return load_project_mods(project)


def current_status(project: Path | None) -> dict[str, Any]:
    """``{state, loaded, nodes, presets, skipped, restart_recommended}``
    for *project* as this process last loaded it."""
    mine = project is not None and _loaded.project == str(project)
    # The tree on disk now, against the one the last load looked at: a pull
    # since then is reported, never loaded, until the next load checks it.
    disk_changed = bool(mine and mods_digest(walk_mods(project)) != _loaded.digest)
    return {
        "disk_changed": disk_changed,
        "state": _loaded.state if mine else "none",
        "loaded": bool(mine and (_loaded.nodes or _loaded.presets)),
        "nodes": list(_loaded.nodes) if mine else [],
        "presets": list(_loaded.presets) if mine else [],
        "skipped": list(_loaded.skipped) if mine else [],
        "restart_recommended": bool(_ran - {str(project)}),
    }


def unloaded_reason(node_type: str) -> str | None:
    """Why a ``project:`` type is not registered, in a sentence a run,
    validate or publish error can carry; None for any other type."""
    if not str(node_type).startswith(f"{PROJECT_PREFIX}:"):
        return None
    return ("it comes from this project's mods/ folder, which is not loaded: "
            "review and trust it in the editor (Review mods/), or with "
            "`cdui project mods --trust`")
