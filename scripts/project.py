"""`cdui project ...` CLI (spec Section 8). Runs under the backend venv via
scripts/dev.py's _dispatch_project_subcommand hop, so `import app.*` works.

This module mirrors scripts/plugins.py: a build_parser() with subcommands, a
main(argv) that dispatches to args._func. This file ships `init`, `validate`, `migrate`, `git-setup`,
`freeze`, `restore`, `mods`, and `publish`.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from urllib.error import HTTPError, URLError

# Reuse the tested bilingual print helpers + stdio reconfigure from plugins.py
# (both are scripts/ siblings loaded onto sys.path by the dispatcher).
from plugins import (
    USER_AGENT,
    _install_git,
    _install_github,
    _read_session_token,
    _reconfigure_stdio,
    _stdin_is_interactive,
    err,
    info,
    ok,
    parse_source,
    section,
    t,
    warn,
)

from app.core.plugin_loader import (
    iter_plugin_dirs,
    load_lockfile,
)
from app.core.project import (
    PROJECT_MANIFEST,
    GraphAmbiguityError,
    adopt_graphs,
    check_pin_issues,
    init_project_repo,
    scaffold_project,
)

MANIFEST_FILENAME = PROJECT_MANIFEST


def cmd_init(args: argparse.Namespace) -> int:
    target = Path(args.dir).expanduser().resolve()
    section(f"建立專案：{target}", f"Creating project: {target}")

    if target.exists() and any(target.iterdir()) and not args.adopt and not args.force:
        err(
            f"目錄非空：{target}（用 --adopt <src> 遷移既有 graphs，或 --force）",
            f"Directory not empty: {target} (use --adopt <src> to migrate an "
            "existing graphs dir, or --force)",
        )
        return 1

    # Shared with the server's own project creation (app.core.project).
    scaffold_project(target)

    # Adopt: copy every *.json from src into graphs/ and split immediately.
    if args.adopt:
        rc = _adopt(Path(args.adopt).expanduser().resolve(), target)
        if rc != 0:
            return rc

    # git init, NO initial commit (spec D3 / Section 5).
    if init_project_repo(target):
        info("已初始化 git 儲存庫（未建立 commit）",
             "Initialized a git repo (no commit made)")
        info("接下來：設定 git 身分並提交：",
             "Next: set a git identity and commit:")
        print('    git config user.name  "You"')
        print('    git config user.email "you@example.com"')
        print("    git add -A && git commit -m \"Initial project\"")
    else:
        warn("找不到 git；已建立專案但沒有版本庫",
             "git not found; created the project without a repository")

    ok(f"專案就緒：{target}", f"Project ready: {target}")
    print(t(f"    啟動：cdui start --project {target}",
            f"    Start it: cdui start --project {target}"))
    return 0


def _init_registries_like_server(proj: Path | None = None) -> dict | None:
    """Discover builtin + custom + PLUGIN nodes and presets exactly like the
    server lifespan (main.py) -- run_graph.py's init loads no plugins and is
    NOT sufficient (spec ID3). With *proj*, also that project's mods/, but
    only when the trust record matches the tree exactly (phase 4C); returns
    the mods status then."""
    import app.core.plugin_loader as plugin_loader
    from app.config import settings
    from app.core.node_registry import registry
    from app.core.plugin_loader import discover_plugin_nodes
    from app.core.preset_registry import preset_registry

    registry.discover(settings.NODES_DIR, "app.nodes")
    registry.discover(settings.CUSTOM_NODES_DIR, "app.custom_nodes")
    lockfile = load_lockfile()
    builtin_root = plugin_loader.plugins_builtin_root()
    user_root = plugin_loader.plugins_user_root()
    discover_plugin_nodes(registry, builtin_root, user_root, lockfile)
    preset_registry.discover(settings.PRESETS_DIR, registry)
    preset_registry.discover(settings.USER_PRESETS_DIR, registry)
    for _pid, pdir in iter_plugin_dirs(builtin_root, user_root, lockfile):
        preset_registry.discover(pdir / "presets", registry)
    if proj is None:
        return None
    from app.core import mods

    return mods.load_project_mods(proj, cli=True)


def _validate_one_graph(store, base: str) -> list[str]:
    """The publish six-gate pre-flight, IN PUBLISH ORDER (routes_apps): secret
    -> contract -> entry -> wiring -> validate. Returns problems (empty=pass)."""
    from app.core import api_contract
    from app.core.graph_engine import (
        build_preset_fallback,
        find_entry_points,
        validate_graph,
    )
    from app.core.device_utils import DEVICE_PATTERN
    from app.core.project import FORMAT_VERSION
    from app.core.secret_params import find_secret_violations

    from app.core.graph_storage import GraphStorageError

    try:
        if store.storage_of(base) == "dir":
            data, repairs = store.read_folder(base)
        else:
            data, repairs = store.read_logic(base), []
    except GraphStorageError as e:
        return [f"unreadable: {e}"]
    if repairs:
        # A folder that only reads after a repair is a problem to fix (open
        # it in the editor and save, or fix the files): never passed quietly.
        return [f"needs_repair: {r.kind} in {r.path}"
                + (f" ({r.detail})" if r.detail else "") for r in repairs]
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    fb = build_preset_fallback(data.get("presets", []))

    fmt = data.get("format_version", 1)
    if isinstance(fmt, int) and fmt > FORMAT_VERSION:
        # Read policy: warn-never-block (spec ID8).
        warn(f"{base}: format_version {fmt} is newer than this build "
             f"(known {FORMAT_VERSION})",
             f"{base}: format_version {fmt} is newer than this build "
             f"(known {FORMAT_VERSION})")

    # 0. settings.device, the same vocabulary the save and run paths accept.
    graph_settings = data.get("settings")
    if graph_settings is not None:
        if not isinstance(graph_settings, dict):
            return ["invalid_settings: settings must be an object"]
        value = graph_settings.get("device")
        if isinstance(value, str) and not value.strip():
            # A blank string is "no assignment"; the save and run paths
            # read it the same way.
            value = None
        if value is not None and (
                not isinstance(value, str)
                or DEVICE_PATTERN.match(value.strip().lower()) is None):
            return [f"invalid_settings: settings.device {value!r}; expected "
                    "cpu, auto, cuda, cuda:N, mps or mps:N"]

    # 1. secrets (publish checks these FIRST, routes_apps.py).
    violations = find_secret_violations(nodes)
    if violations:
        v = violations[0]
        return [f"secret_in_graph: node '{v['node_id']}' param '{v['param']}'"]
    # 2. contract.
    contract = api_contract.derive_contract(nodes)
    if contract.problems:
        msg = f"invalid_contract: {contract.problems[0]}"
        if not contract.outputs:
            # The mixed-project story (issue #86): a canvas-only training
            # graph has no API surface, and that is fine ON THE CANVAS --
            # but every publishable graph needs a declared output. Name the
            # escape hatch instead of leaving CI users to reverse it.
            msg += (" -- a publishable graph needs a declared GraphOutput; "
                    "for a canvas-only graph (e.g. training), either add a "
                    "real output or validate only your publish targets: "
                    "`cdui project validate . --graph <name>`")
        return [msg]
    # 3. entry points.
    if not find_entry_points(nodes, edges):
        return ["no_entry_points: wire a Start node into every GraphInput"]
    # 4. wiring.
    wiring = api_contract.check_wiring(nodes, edges, contract)
    if wiring.untriggered:
        return [f"untriggered_input: {wiring.untriggered[0]}"]
    if wiring.unreachable:
        return [f"unreachable_output: {wiring.unreachable[0]}"]
    # 5. validate_graph (unknown-type errors get the specific ID3 message).
    errors = validate_graph(nodes, edges, preset_fallback=fb)
    if errors:
        first = errors[0]
        if first.startswith("Unknown node type") and " -- " not in first:
            first = first + " -- run `cdui project restore` (or install the plugin that provides it)"
        return [f"invalid_graph: {first}"]
    return []


def _check_env_not_tracked(proj: Path) -> bool:
    """True on FAILURE: .env is tracked by git. No git -> notice + pass."""
    if not shutil.which("git"):
        info("git 不存在，略過 .env 追蹤檢查",
             "git not found; skipping the .env-tracked check")
        return False
    res = subprocess.run(["git", "-C", str(proj), "ls-files", ".env"],
                         capture_output=True, text=True)
    if res.stdout.strip():
        err(".env 已被 git 追蹤（絕不可提交機密）",
            ".env is tracked by git -- secrets must never be committed")
        return True
    return False


def _check_pins(manifest: dict, strict: bool) -> bool:
    """True on FAILURE (only when strict). Missing/mismatched pins warn, and
    with --strict become errors (spec ID3). Malformed (non-table) pins are
    warn-and-skip on EVERY surface -- never a failure, even under --strict
    (shared rule: app.core.project.check_pin_issues, issue #85)."""
    pins = manifest.get("plugins", {}) or {}
    if not pins:
        return False
    failed = False
    for issue in check_pin_issues(manifest, load_lockfile()):
        pid = issue.plugin_id
        if issue.kind == "malformed":
            msg = (f"pin '{pid}' in [plugins] is malformed (expected a table "
                   'like { url = "...", ref = "...", sha = "..." }) -- skipping')
            warn(msg, msg)
            continue
        if issue.kind == "missing":
            _warn_or_err(strict, f"pinned plugin '{pid}' is not installed -- "
                                 "run `cdui project restore`")
        else:  # sha_mismatch
            _warn_or_err(strict,
                         f"plugin '{pid}' sha {(issue.installed_sha or '')[:7]} "
                         f"!= pinned {issue.pinned_sha[:7]} -- run restore")
        failed = failed or strict
    return failed


def _warn_or_err(strict: bool, msg: str) -> None:
    (err if strict else warn)(msg, msg)


def cmd_validate(args: argparse.Namespace) -> int:
    proj = Path(args.dir).expanduser().resolve()
    section(f"驗證專案：{proj}", f"Validating project: {proj}")

    manifest_path = proj / MANIFEST_FILENAME
    if not manifest_path.exists():
        err(f"找不到 manifest：{manifest_path}",
            f"Manifest not found: {manifest_path}")
        return 1
    from app.core.plugin_loader import tomllib
    try:
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        err(f"manifest 解析失敗：{e}", f"Manifest parse error: {e}")
        return 1

    mods_status = _init_registries_like_server(proj) or {}
    state = mods_status.get("state", "none")
    if state in ("untrusted", "changed", "blocked"):
        warn(f"mods/ 未載入（{state}）；用到 project: 節點的 graph 會失敗。"
             "檢視並信任：cdui project mods --trust",
             f"mods/ is not loaded ({state}); graphs that use project: nodes "
             "will fail. Review and trust it with: cdui project mods --trust")

    from app.core.graph_storage import GraphStore

    store = GraphStore.for_project(proj)
    try:
        # Shared canonical-vs-legacy collection rule (app.core.graph_storage).
        graphs = store.stems()
    except GraphAmbiguityError as e:
        err(str(e), str(e))
        return 1

    # --graph filter (issue #86): mixed projects keep canvas-only graphs
    # (train...) next to publish targets (serve...); let CI gate just the
    # targets. An unknown name is an ERROR -- a typo must never turn the
    # gate into a vacuous pass.
    requested = getattr(args, "graph", None)
    if requested:
        missing = [n for n in requested if n not in graphs]
        if missing:
            avail = ", ".join(graphs) or "(none)"
            err(f"graphs/ 沒有這些 graph：{', '.join(missing)}（現有：{avail}）",
                f"No such graph in graphs/: {', '.join(missing)} "
                f"(available: {avail})")
            return 1
        wanted = set(requested)
        graphs = [n for n in graphs if n in wanted]

    all_ok = True
    for base in graphs:
        problems = _validate_one_graph(store, base)
        if problems:
            all_ok = False
            err(f"FAIL {base}: {problems[0]}", f"FAIL {base}: {problems[0]}")
            # Every further problem on its own line (a folder graph's
            # repairs are listed one by one).
            for more in problems[1:]:
                err(f"     {base}: {more}", f"     {base}: {more}")
        else:
            ok(f"PASS {base}", f"PASS {base}")

    pin_fail = _check_pins(manifest, args.strict)
    env_fail = _check_env_not_tracked(proj)
    _check_mods_block(proj, manifest)

    if not all_ok or pin_fail or env_fail:
        err("驗證失敗", "Validation FAILED")
        return 1
    # Always print the checked count: "Validation passed" over an empty
    # graphs/ would be a vacuous green in CI logs (issue #86).
    checked = len(graphs)
    noun = "graph" if checked == 1 else "graphs"
    ok(f"驗證通過（已檢查 {checked} 個 graph）",
       f"Validation passed ({checked} {noun} checked)")
    return 0


def _toml_key(key: str) -> str:
    """A bare TOML key when possible, else a basic-quoted key."""
    if key and all((ch.isascii() and ch.isalnum()) or ch in "-_" for ch in key):
        return key
    return json.dumps(key)


def _toml_value(value: object) -> str:
    """Serialize one tomllib-parsed value back to TOML. json.dumps produces a
    valid TOML basic string (its escapes are a subset of TOML's) except for
    DEL, which JSON leaves raw but TOML requires escaped."""
    if isinstance(value, bool):  # before int: bool is an int subclass
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value).replace("\x7f", "\\u007f")
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        items = ", ".join(f"{_toml_key(k)} = {_toml_value(v)}"
                          for k, v in value.items())
        return "{ " + items + " }" if items else "{}"
    raise TypeError(f"unsupported TOML value type: {type(value).__name__}")


_TABLE_HEADER = __import__("re").compile(r"^\s*\[\[?\s*([^\]]+?)\s*\]\]?\s*(#.*)?$")


def _set_toml_table(text: str, header: str, body: list[str] | None) -> str:
    """*text* with the table ``[header]`` holding exactly *body* (lines),
    edited in place: every other line -- comments included -- stays as it
    was, and a comment line right under the header is kept with it. A new
    table goes at the end; ``body=None`` removes the table."""
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        m = _TABLE_HEADER.match(line)
        if m and m.group(1) == header and not line.lstrip().startswith("[["):
            start = i
            break
    if start is None:
        if body is None:
            return text
        out = text.rstrip("\n")
        return (out + "\n\n" if out else "") + "\n".join([f"[{header}]", *body]) + "\n"
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if _TABLE_HEADER.match(lines[j]):
            end = j
            break
    # The blank lines and comments that come before the next table belong
    # to it, not to this one.
    tail = end
    while tail > start + 1 and (not lines[tail - 1].strip()
                                or lines[tail - 1].lstrip().startswith("#")):
        tail -= 1
    if body is None:
        new = lines[:start] + lines[tail:]
        while new and not new[-1].strip():
            new.pop()
        return "\n".join(new) + "\n" if new else ""
    keep = []
    k = start + 1
    while k < tail and lines[k].lstrip().startswith("#"):
        if lines[k] not in body:   # the writer's own note is in *body* already
            keep.append(lines[k])
        k += 1
    gap = [""] if tail < len(lines) else []
    new = lines[:start + 1] + keep + body + gap + lines[tail:]
    # Never two blank lines where one was.
    out: list[str] = []
    for line in new:
        if not line.strip() and out and not out[-1].strip():
            continue
        out.append(line)
    return "\n".join(out).rstrip("\n") + "\n"


def _plugins_table_body(pins: dict) -> list[str]:
    body = ["# written by `cdui project freeze`; installed by `cdui project restore`"]
    for pid in sorted(pins):
        p = pins[pid]
        body.append(f'{pid} = {{ url = "{p["url"]}", ref = "{p["ref"]}", '
                    f'sha = "{p["sha"]}" }}')
    return body


def _mods_table(proj: Path) -> dict:
    """What ``[mods]`` should say about *proj*'s mods/: per file its sha256
    and, for a node file, the node names it defines (read from the source);
    for a preset, its name. Informational: never used for trust."""
    from app.core import mods

    tree = mods.walk_mods(proj)
    table: dict = {}
    for rel in sorted(tree.files):
        f = tree.files[rel]
        entry: dict = {"sha256": f.sha256}
        if f.role == "node":
            entry["nodes"] = mods._node_names(
                mods._read_no_follow(tree.root / rel).decode("utf-8", "replace"))
        elif f.role == "preset":
            try:
                name = json.loads(mods._read_no_follow(tree.root / rel)).get("preset_name")
            except (ValueError, OSError, AttributeError):
                name = None
            if isinstance(name, str):
                entry["preset"] = name
        table[rel] = entry
    return table


def _mods_table_body(table: dict) -> list[str]:
    body = ["# written by `cdui project freeze` / `cdui project mods --write`; "
            "informational, never used for trust"]
    for rel, entry in table.items():
        body.append(f"{json.dumps(rel)} = {_toml_value(entry)}")
    return body


def _write_mods_block(manifest_path: Path, proj: Path) -> int:
    """Rewrite ``[mods]`` in place; returns how many files it lists."""
    table = _mods_table(proj)
    text = manifest_path.read_text(encoding="utf-8")
    text = _set_toml_table(text, "mods", _mods_table_body(table) if table else None)
    manifest_path.write_bytes(text.encode("utf-8"))
    return len(table)


def _check_mods_block(proj: Path, manifest: dict) -> bool:
    """Warn when the manifest's ``[mods]`` does not describe mods/ as it
    is (a stale block misleads a reviewer). A warning, never a failure, and
    never a trust decision. True when it was stale."""
    listed = manifest.get("mods")
    actual = _mods_table(proj)
    if listed is None and not actual:
        return False
    if listed == actual:
        return False
    warn("manifest 的 [mods] 與 mods/ 不符；更新：cdui project mods --write",
         "The manifest's [mods] does not match mods/; update it with: "
         "cdui project mods --write")
    return True


def _set_manifest_graph_format(manifest_path: Path, fmt: str) -> None:
    """Point ``[storage] graph_format`` at *fmt*, touching nothing else."""
    import re

    text = manifest_path.read_text(encoding="utf-8")
    line = f'graph_format = "{fmt}"'
    if re.search(r"(?m)^\s*graph_format\s*=.*$", text):
        text = re.sub(r"(?m)^\s*graph_format\s*=.*$", line, text, count=1)
    elif re.search(r"(?m)^\[storage\]\s*$", text):
        text = re.sub(r"(?m)^\[storage\]\s*$", f"[storage]\n{line}", text, count=1)
    else:
        text = text.rstrip("\n") + f"\n\n[storage]\n{line}\n"
    manifest_path.write_bytes(text.encode("utf-8"))


def cmd_migrate(args: argparse.Namespace) -> int:
    """Rewrite graphs in the other storage form, verifying each one reads
    back the same before its old form is removed."""
    proj = Path(args.dir).expanduser().resolve()
    to = args.to
    section(f"轉換專案的圖為 {to}：{proj}", f"Migrating graphs to {to}: {proj}")
    manifest_path = proj / MANIFEST_FILENAME
    if not manifest_path.exists():
        err(f"找不到 manifest：{manifest_path}", f"Manifest not found: {manifest_path}")
        return 1

    from app.core.graph_storage import GraphAmbiguityError, GraphStorageError, GraphStore

    store = GraphStore.for_project(proj)
    try:
        graphs = store.stems()
    except GraphAmbiguityError as e:
        err(str(e), str(e))
        return 1
    requested = getattr(args, "graph", None)
    if requested:
        missing = [n for n in requested if n not in graphs]
        if missing:
            err(f"graphs/ 沒有這些 graph：{', '.join(missing)}",
                f"No such graph in graphs/: {', '.join(missing)}")
            return 1
        graphs = [n for n in graphs if n in set(requested)]

    failed = converted = 0
    for stem in graphs:
        try:
            result = store.convert(stem, to)
        except GraphStorageError as e:
            failed += 1
            err(f"未轉換 {stem}：{e}", f"NOT MIGRATED {stem}: {e}")
            continue
        if result.status == "converted":
            converted += 1
            ok(f"{result.old} → {result.new}", f"{result.old} -> {result.new}")
        else:
            ok(f"{stem}：已經是 {to}", f"{stem}: already {to}")

    if not requested and not failed:
        # The whole project moved, so new graphs follow it.
        _set_manifest_graph_format(manifest_path, to)
        ok(f"manifest：[storage] graph_format = \"{to}\"",
           f"manifest: [storage] graph_format = \"{to}\"")
    if converted:
        warn("git 會把轉換看成刪檔加新檔：轉換之後 `git log --follow` 接不到轉換前的歷史。"
             "要看之前的歷史，用舊路徑查，例如 `git log -- graphs/<name>.graph.json`。"
             "請把這次轉換單獨 commit。",
             "git sees the migration as files deleted and files added, so "
             "`git log --follow` does not reach history from before it. For "
             "that history ask for the old path, e.g. `git log -- "
             "graphs/<name>.graph.json`. Commit the migration on its own.")
    return 1 if failed else 0


def cmd_git_setup(args: argparse.Namespace) -> int:
    """Define the graph merge driver in this repository's ``.git/config``,
    for merges run in a terminal (the editor defines it per command)."""
    proj = Path(args.dir).expanduser().resolve()
    section(f"設定 git 合併：{proj}", f"Setting up git merges: {proj}")
    if not (proj / MANIFEST_FILENAME).exists():
        err(f"找不到 manifest：{proj / MANIFEST_FILENAME}",
            f"Manifest not found: {proj / MANIFEST_FILENAME}")
        return 1
    if shutil.which("git") is None:
        err("找不到 git", "git was not found")
        return 1
    from app.core.git.runner import merge_driver_config
    from app.core.project import PROJECT_GITATTRIBUTES

    inside = subprocess.run(["git", "-C", str(proj), "rev-parse", "--git-dir"],
                            capture_output=True, text=True)
    if inside.returncode != 0:
        err("這個專案不是 git repository", "This project is not a git repository")
        return 1
    options = merge_driver_config()
    for _flag, setting in zip(options[0::2], options[1::2]):
        key, _, value = setting.partition("=")
        subprocess.run(["git", "-C", str(proj), "config", "--local", key, value],
                       check=True)
        ok(f"{key} = {value}", f"{key} = {value}")
    # A project made before phase 4B has no `merge=codefyui` lines, and the
    # driver applies only to the files they name.
    attributes = proj / ".gitattributes"
    have = attributes.read_text(encoding="utf-8") if attributes.exists() else ""
    wanted = [line for line in PROJECT_GITATTRIBUTES.splitlines()
              if "merge=codefyui" in line and line not in have.splitlines()]
    if wanted:
        text = have if have.endswith("\n") or not have else have + "\n"
        attributes.write_bytes((text + "\n".join(wanted) + "\n").encode("utf-8"))
        warn(f".gitattributes 加入了 {len(wanted)} 行（merge=codefyui）；請 commit。",
             f"Added {len(wanted)} line(s) to .gitattributes (merge=codefyui); "
             "commit them.")
    info("之後在終端機合併時，圖的線、layout 與索引會依鍵合併。這個設定只在這台機器的這個 clone。",
         "Merges run in a terminal now merge a graph's wires, layout and index "
         "by key. The setting lives in this clone on this machine only.")
    return 0


def _used_plugin_ids(proj: Path) -> tuple[set[str], list[str]]:
    """The plugin ids the project's graphs use (read off the qualified node
    types, so a plugin that is not installed still counts), and the graphs
    that could not be read (which fail the command: a pin set must not
    shrink silently)."""
    from app.core.graph_storage import GraphStore, _node_types

    store = GraphStore.for_project(proj)
    used: set[str] = set()
    unreadable: list[str] = []
    for stem in store.stems():
        try:
            if store.storage_of(stem) == "dir":
                logic, _repairs = store.read_folder(stem)
            else:
                logic = store.read_logic(stem)
        except Exception as e:  # noqa: BLE001 -- reported, then the command fails
            unreadable.append(f"{stem}: {e}")
            continue
        for node_type in _node_types(logic):
            prefix, sep, _name = str(node_type).partition(":")
            if sep and prefix not in ("project", "subgraph"):
                used.add(prefix)
    return used, unreadable


def cmd_freeze(args: argparse.Namespace) -> int:
    proj = Path(args.dir).expanduser().resolve()
    manifest_path = proj / MANIFEST_FILENAME
    if not manifest_path.exists():
        err(f"找不到 manifest：{manifest_path}", f"Manifest not found: {manifest_path}")
        return 1
    section("凍結外掛版本", "Freezing plugin pins")
    used = None
    if getattr(args, "used_only", False):
        try:
            used, unreadable = _used_plugin_ids(proj)
        except GraphAmbiguityError as e:
            err(str(e), str(e))
            return 1
        if unreadable:
            for line in unreadable:
                err(f"無法讀取 graph：{line}", f"Cannot read graph: {line}")
            return 1
    lock = load_lockfile().get("plugins", {})
    pins: dict = {}
    for pid, entry in lock.items():
        if used is not None and pid not in used:
            continue  # installed, but no graph here uses it
        kind = entry.get("source_kind")
        if kind in ("local", "local_copy"):
            warn(f"略過本地連結或本機來源：{pid}",
                 f"Skipping linked/local plugin: {pid} (machine-specific path)")
            continue
        if kind == "builtin":
            continue  # ships with cdui; nothing to pin
        url, sha, ref = entry.get("url"), entry.get("sha"), entry.get("ref", "")
        if not url or not sha:
            warn(f"略過 {pid}（缺 url/sha）", f"Skipping {pid} (missing url/sha)")
            continue
        pins[pid] = {"url": url, "ref": ref, "sha": sha}
        ok(f"釘選 {pid} @ {sha[:7]}", f"Pinned {pid} @ {sha[:7]}")
    for pid in sorted((used or set()) - set(lock)):
        warn(f"{pid} 有 graph 使用但沒有安裝，無法釘選",
             f"{pid} is used by a graph but not installed here; it cannot be pinned")
    # In place: the user's comments and every table freeze does not own stay.
    text = manifest_path.read_text(encoding="utf-8")
    text = _set_toml_table(text, "plugins", _plugins_table_body(pins))
    manifest_path.write_bytes(text.encode("utf-8"))
    count = _write_mods_block(manifest_path, proj)
    ok(f"已寫入 {len(pins)} 個釘選", f"Wrote {len(pins)} pin(s) to the manifest")
    if count:
        ok(f"[mods] 列出 {count} 個檔案", f"[mods] lists {count} file(s)")
    return 0


def cmd_restore(args: argparse.Namespace) -> int:
    proj = Path(args.dir).expanduser().resolve()
    manifest_path = proj / MANIFEST_FILENAME
    from app.core.plugin_loader import tomllib
    if not manifest_path.exists():
        err(f"找不到 manifest：{manifest_path}", f"Manifest not found: {manifest_path}")
        return 1
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    pins = manifest.get("plugins", {}) or {}
    if not pins:
        info("manifest 沒有外掛釘選", "No plugin pins in the manifest")
        return 0
    section("還原外掛版本", "Restoring plugin pins")
    lockfile = load_lockfile()
    installed = lockfile.get("plugins", {})
    rc_all = 0
    for pid, pin in pins.items():
        sha, url, ref = pin.get("sha"), pin.get("url"), pin.get("ref", "")
        if not sha or not url:
            err(f"{pid}：釘選缺 url/sha", f"{pid}: pin missing url/sha")
            rc_all = 1
            continue
        cur = installed.get(pid)
        if cur and cur.get("sha") == sha:
            ok(f"{pid} 已是 {sha[:7]}", f"{pid} already at {sha[:7]}")
            continue
        kind, owner, repo, _ref = parse_source(url)
        # Something the lockfile does not have yet (phase 4C): the manifest
        # came with the project, so what it installs is the project author's
        # choice, not the user's. The install shows each plugin's source,
        # trusted modules and capabilities and asks, in a terminal; without
        # one only --yes installs it. Capabilities this machine granted
        # before are reused by the install's own gate.
        yes = bool(getattr(args, "yes", False))
        if not yes and not _stdin_is_interactive():
            err(f"{pid}：需要確認才能安裝（在終端機執行，或加 --yes）",
                f"{pid}: installing it needs your confirmation -- run this in "
                "a terminal, or pass --yes")
            rc_all = 1
            continue
        inst_args = argparse.Namespace(
            force=True, no_confirm=yes, trust_author=True,
            accept_capabilities=yes, pinned_sha=sha,
            consent_flag="--yes")
        # Trust _install_github's return code (spec ID11: install BY the pinned
        # sha). It never re-resolves the ref when pinned_sha is set, so success
        # (rc == 0) already means the installed plugin is at this exact sha.
        if kind == "git":
            # A repository on another git host: ``owner`` is its URL. The
            # clone checks out the pinned commit whatever the ref says now.
            rc = _install_git(owner, ref, inst_args, lockfile)
        else:
            rc = _install_github(owner, repo, ref, inst_args, lockfile)
        if rc != 0:
            rc_all = 1
    return rc_all


def _print_mods_scan(scan: dict, state: dict) -> None:
    from app.core.mods import SCAN_IS_A_HINT

    for f in scan["files"]:
        size = f"{f['size']} B"
        lines = f", {f['lines']} lines" if "lines" in f else ""
        print(f"  mods/{f['path']}  ({size}{lines})")
        if not f.get("loads", True):
            print("    never loaded (only .py files directly in mods/nodes/, "
                  "not __init__.py, and mods/presets/*.json are read)")
            continue
        if f.get("role") != "node":
            continue
        finding = f.get("finding")
        if finding:
            warn(f"    掃描：{finding['message']}", f"    scan: {finding['message']}")
        else:
            print("    scan: no findings")
        for need in f.get("needs") or ():
            print(f"    uses: {need['summary']}")
        if f.get("nodes"):
            print("    nodes: " + ", ".join(f"project:{n}" for n in f["nodes"]))
        if f.get("duplicates"):
            warn("    另一個檔案也定義了同名節點，兩者都不載入",
                 "    another file defines the same node name; neither loads")
    for r in scan["refused"]:
        err(f"  mods/{r['path']}：{r['reason']}", f"  mods/{r['path']}: {r['reason']}")
    changed = state.get("changed") or {}
    for kind in ("added", "changed", "removed"):
        for rel in changed.get(kind) or ():
            print(f"  {kind}: mods/{rel}")
    print(f"  sha256: {scan['digest']}")
    print(f"  trust: {state['state']}")
    info(SCAN_IS_A_HINT, SCAN_IS_A_HINT)


def cmd_mods(args: argparse.Namespace) -> int:
    """Show the project's mods/ scan and hash; --trust records consent to
    run exactly this tree; --write rewrites the manifest's [mods]."""
    from app.core import mods

    proj = Path(args.dir).expanduser().resolve()
    manifest_path = proj / MANIFEST_FILENAME
    if not manifest_path.exists():
        err(f"找不到 manifest：{manifest_path}", f"Manifest not found: {manifest_path}")
        return 1
    section(f"專案 mods：{proj}", f"Project mods: {proj}")
    tree = mods.walk_mods(proj)
    if not tree.exists:
        info("沒有 mods/ 資料夾", "No mods/ folder")
    scan = mods.scan_mods(proj, tree)
    state = mods.mods_trust_state(proj, tree)
    _print_mods_scan(scan, state)
    rc = 0
    if args.trust:
        if tree.refused:
            err("mods/ 裡有永遠不會載入的項目，無法信任；先移除它們",
                "mods/ holds entries that are never loaded; remove them first. "
                "Nothing was trusted.")
            return 1
        if not tree.node_files:
            info("mods/ 沒有 Python 檔，不需要信任", "No Python files in mods/; nothing to trust")
        else:
            if not args.yes:
                if not _stdin_is_interactive():
                    err("需要確認：在終端機執行，或加 --yes（等於同意執行這些程式碼）",
                        "This needs your confirmation: run it in a terminal, or pass "
                        "--yes (which is consent to run this code)")
                    return 1
                try:
                    answer = input(f"  {t('信任並允許執行這些檔案？', 'Trust these files and let CodefyUI run them?')} "
                                   "[y/N]: ").strip().lower()
                except EOFError:
                    answer = ""
                if answer not in ("y", "yes"):
                    warn("已取消，沒有信任任何檔案", "Cancelled; nothing was trusted")
                    return 1
            # The tree read again: consent is for what was on the screen.
            again = mods.walk_mods(proj)
            if mods.mods_digest(again) != scan["digest"]:
                err("mods/ 在顯示之後變更了，沒有信任；請重新執行",
                    "mods/ changed after it was shown; nothing was trusted. Run "
                    "this again.")
                return 1
            mods.record_trust(proj, again, full=bool(args.full))
            ok(f"已信任 mods/（sha256 {scan['digest'][:12]}…）",
               f"Trusted mods/ (sha256 {scan['digest'][:12]}…)")
    if args.write:
        count = _write_mods_block(manifest_path, proj)
        ok(f"已更新 [mods]（{count} 個檔案）", f"Updated [mods] ({count} file(s))")
    return rc


def _server_base() -> tuple[str, str]:
    """(base_url, host_header) for the local server, mirroring the plugin CLI's
    port resolution (CODEFYUI_PORT env, then settings.PORT, then 8000). The
    client always connects on loopback, which is always whitelisted."""
    port = 8000
    env = os.environ.get("CODEFYUI_PORT", "").strip()
    if env.isdigit():
        port = int(env)
    else:
        try:
            from app.config import settings
            port = int(settings.PORT)
        except Exception:  # noqa: BLE001
            port = 8000
    netloc = f"127.0.0.1:{port}"
    return f"http://{netloc}", netloc


def _http_get_json(url: str, host: str) -> dict | None:
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Host": host})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (URLError, HTTPError, OSError, ValueError):
        return None


def _http_post_json(url: str, host: str, token: str, body: dict) -> dict | None:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, method="POST", data=data, headers={
        "User-Agent": USER_AGENT, "Host": host,
        "Content-Type": "application/json", "X-CodefyUI-Token": token,
        "Content-Length": str(len(data))})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            out = json.loads(resp.read().decode("utf-8"))
            out["_status"] = resp.status
            return out
    except HTTPError as e:
        try:
            out = json.loads(e.read().decode("utf-8"))
        except Exception:  # noqa: BLE001
            out = {}
        out["_status"] = e.code
        return out
    except (URLError, OSError, ValueError):
        return None


def cmd_publish(args: argparse.Namespace) -> int:
    proj = Path(args.dir).expanduser().resolve()
    manifest_path = proj / MANIFEST_FILENAME
    from app.core.plugin_loader import tomllib
    if not manifest_path.exists():
        err(f"找不到 manifest：{manifest_path}", f"Manifest not found: {manifest_path}")
        return 1
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    defaults = manifest.get("publish", {}) or {}
    graph = args.graph or defaults.get("graph")
    slug = args.slug or defaults.get("slug")
    if not graph:
        err("需要 --graph 或 manifest [publish].graph",
            "A graph is required: pass --graph or set [publish].graph")
        return 1
    if not slug:
        err("需要 --slug 或 manifest [publish].slug",
            "A slug is required: pass --slug or set [publish].slug")
        return 1

    token = _read_session_token()
    if token is None:
        err("找不到 session token -- 伺服器未執行？先 cdui start --project .",
            "No session token -- is the server running? Run `cdui start --project .`")
        return 1
    base, host = _server_base()

    # Local-only verification (ID4): the server must have THIS project open.
    health = _http_get_json(f"{base}/api/health", host)
    if health is None:
        err(f"無法連線到 {base}/api/health", f"Cannot reach {base}/api/health")
        return 1
    server_project = health.get("project")
    if server_project is None or Path(server_project).resolve() != proj:
        err(f"伺服器開啟的專案（{server_project}）與 {proj} 不符",
            f"Server has a different project open ({server_project}) -- run "
            f"`cdui start --project {proj}` first")
        return 1

    from app.core.project import git_provenance
    commit, dirty = git_provenance(proj)
    if commit is None:
        print("NOTE: not a git repo -- publishing with NULL provenance")
    elif dirty is None:
        # rev-parse resolved a commit but `git status` failed: the tree
        # state is UNKNOWN. Record null (the schema's "unknown"), never a
        # fabricated clean=false (issue #86).
        print("NOTE: `git status` failed -- working-tree state unknown; "
              "recording git_dirty as null")
    elif dirty:
        # Loud, locale-independent warning (ID12).
        print("=" * 70)
        print("WARNING: working tree is DIRTY -- the recorded commit does NOT "
              "match the published bytes.")
        print("=" * 70)

    # create only when the slug is the manifest's committed [publish].slug
    # (a deliberate target) or the user passed --create: an explicitly
    # typed --slug with a typo must hit the server's app_not_found 404
    # instead of silently minting a second app (issue #86). The slug itself
    # travels in the URL path only -- PublishRequest declares no such field.
    create = bool(args.create or args.slug is None)
    body: dict = {"graph": graph, "create": create, "note": args.note}
    if "record_io" in defaults:
        body["record_io"] = bool(defaults["record_io"])
    if commit is not None:
        body["git_commit"] = commit
        body["git_dirty"] = None if dirty is None else bool(dirty)

    resp = _http_post_json(f"{base}/api/apps/{slug}/publish", host, token, body)
    if resp is None or resp.get("_status", 500) != 200:
        detail = resp.get("detail") if isinstance(resp, dict) else resp
        err(f"發佈失敗：{detail}", f"Publish failed: {detail}")
        if (not create and isinstance(detail, dict)
                and detail.get("code") == "app_not_found"):
            info(f"slug '{slug}' 尚不存在 -- 第一次發佈新 slug 請加 --create",
                 f"slug '{slug}' does not exist yet -- pass --create to "
                 "create it on first publish")
        return 1
    prov = (f" (git {commit[:7]}{' dirty' if dirty else ''})"
            if commit else " (no provenance)")
    ok(f"已發佈 {slug} v{resp.get('version')}{prov}",
       f"Published {slug} v{resp.get('version')}{prov}")
    return 0


def _adopt(src: Path, target: Path) -> int:
    if not src.is_dir():
        err(f"--adopt 來源不是目錄：{src}", f"--adopt source is not a directory: {src}")
        return 1
    try:
        result = adopt_graphs(src, target)
    except GraphAmbiguityError as e:
        err(str(e), str(e))
        return 1
    for name, problem in result.skipped:
        warn(f"略過 {name}：{problem}", f"Skipping {name}: {problem}")
    count = len(result.adopted)
    ok(f"已採用並拆分 {count} 個 graph", f"Adopted and split {count} graph(s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cdui project", description="Manage a CodefyUI project directory")
    sub = p.add_subparsers(dest="project_cmd", required=True)

    p_init = sub.add_parser("init", help="Scaffold a new project directory")
    p_init.add_argument("dir", help="path to create the project in")
    p_init.add_argument("--adopt", default=None,
                        help="copy every *.json from this dir into graphs/ and split")
    p_init.add_argument("--force", action="store_true",
                        help="write into an existing non-empty directory")
    p_init.set_defaults(_func=cmd_init)

    p_val = sub.add_parser("validate", help="Validate every graph (publish pre-flight) + project checks")
    p_val.add_argument("dir", help="project directory to validate")
    p_val.add_argument("--graph", action="append", default=None, metavar="NAME",
                       help="validate only the named graph (repeatable); lets "
                            "CI gate publish targets while canvas-only graphs "
                            "(no GraphOutput) live in the same project")
    p_val.add_argument("--strict", action="store_true",
                       help="treat missing/mismatched plugin pins as errors")
    p_val.set_defaults(_func=cmd_validate)

    p_mig = sub.add_parser(
        "migrate", help="Store graphs as folders (dir) or single files (file)")
    p_mig.add_argument("dir", help="project directory")
    p_mig.add_argument("--to", choices=("dir", "file"), default="dir",
                       help="the storage form to convert to (default: dir)")
    p_mig.add_argument("--graph", action="append", default=None, metavar="NAME",
                       help="migrate only the named graph (repeatable); without "
                            "it every graph moves and the manifest's "
                            "[storage] graph_format is set to match")
    p_mig.set_defaults(_func=cmd_migrate)

    p_gs = sub.add_parser(
        "git-setup", help="Define the graph merge driver in this repo's .git/config")
    p_gs.add_argument("dir", help="project directory")
    p_gs.set_defaults(_func=cmd_git_setup)

    p_freeze = sub.add_parser("freeze", help="Write installed github plugin pins into the manifest")
    p_freeze.add_argument("dir", help="project directory")
    p_freeze.add_argument("--used-only", action="store_true",
                          help="pin only the plugins this project's graphs use")
    p_freeze.set_defaults(_func=cmd_freeze)

    p_restore = sub.add_parser("restore", help="Install the manifest's plugin pins by their exact SHA")
    p_restore.add_argument("dir", help="project directory")
    p_restore.add_argument("--yes", action="store_true",
                           help="install pins the lockfile does not have without "
                                "asking, granting what they request")
    p_restore.set_defaults(_func=cmd_restore)

    p_mods = sub.add_parser("mods", help="Show the scan and hash of the project's mods/, "
                                         "and trust it")
    p_mods.add_argument("dir", help="project directory")
    p_mods.add_argument("--trust", action="store_true",
                        help="record consent to run exactly these files (asks first)")
    p_mods.add_argument("--yes", action="store_true",
                        help="with --trust: do not ask. This is consent to run the code")
    p_mods.add_argument("--full", action="store_true",
                        help="with --trust: also run files the scan had findings for")
    p_mods.add_argument("--write", action="store_true",
                        help="rewrite the manifest's informational [mods] block")
    p_mods.set_defaults(_func=cmd_mods)

    p_pub = sub.add_parser("publish", help="Publish a graph to the local server, recording git provenance")
    p_pub.add_argument("dir", help="project directory")
    p_pub.add_argument("--graph", default=None, help="saved graph name (default: manifest [publish].graph)")
    p_pub.add_argument("--slug", default=None, help="published slug (default: manifest [publish].slug)")
    p_pub.add_argument("--note", default=None, help="optional immutable version note")
    p_pub.add_argument("--create", action="store_true",
                       help="allow creating a NEW app for an explicitly "
                            "passed --slug (a slug from the manifest "
                            "[publish].slug creates automatically; without "
                            "this flag a misspelled --slug fails with 404 "
                            "instead of minting a second app)")
    p_pub.set_defaults(_func=cmd_publish)

    return p


def main(argv: list[str] | None = None) -> int:
    _reconfigure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args._func(args)


if __name__ == "__main__":
    sys.exit(main())
