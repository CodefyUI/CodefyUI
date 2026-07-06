"""`cdui project ...` CLI (spec Section 8). Runs under the backend venv via
scripts/dev.py's _dispatch_project_subcommand hop, so `import app.*` works.

This module mirrors scripts/plugins.py: a build_parser() with subcommands, a
main(argv) that dispatches to args._func. This file ships `init` and
`validate`; restore / freeze / publish are added by later tasks.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Reuse the tested bilingual print helpers + stdio reconfigure from plugins.py
# (both are scripts/ siblings loaded onto sys.path by the dispatcher).
from plugins import (
    _install_github,
    _reconfigure_stdio,
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

MANIFEST_FILENAME = "codefyui.project.toml"

_PROJECT_TOML = """[project]
name = "{name}"
format_version = 1
requires_codefyui = ">=1.4"   # advisory metadata; NOT enforced

[plugins]
# written by `cdui project freeze`; installed by `cdui project restore`
# some-pack = {{ url = "https://github.com/owner/repo", ref = "v1.2", sha = "<40-hex>" }}

[publish]                      # optional single default target
# graph = "my-graph"
# slug = "my-service"
# record_io = true
"""

_GITIGNORE = """.env
*.pt
*.pth
*.safetensors
*.onnx
*.ckpt
*.pkl
__pycache__/
*.db
.DS_Store
# interrupted atomic writes
*.tmp-*
"""

_GITATTRIBUTES = "layout/*.layout.json linguist-generated=true\n"

_ENV_EXAMPLE = """# CodefyUI project secrets (runtime only). Copy to .env and fill in.
# .env is gitignored; this .env.example is committed as the required-keys template.
#
# Loaded at server start with os.environ.setdefault semantics (an already-set
# environment variable wins). Values are execution-time only and are never
# written to a saved graph. CODEFYUI_* SERVER CONFIG keys placed here are
# IGNORED (server settings materialize before .env loads); pass `cdui start`
# flags or the process environment for those.
#
# LLM keys, read at node execute time:
# OPENAI_API_KEY=
# CODEFYUI_OPENAI_API_KEY=
# ANTHROPIC_API_KEY=
# CODEFYUI_ANTHROPIC_API_KEY=
"""

_README = """# {name}

A CodefyUI **project directory**: a self-contained git repo where the graph
files ARE the storage. Open it with:

    cdui start --project .

## Layout

    codefyui.project.toml   project manifest (name, plugin pins, default publish target)
    graphs/                 <name>.graph.json  -- logic (nodes/edges/params/presets)
    layout/                 <name>.layout.json -- node positions (reviewable, generated)
    assets/images/          IMAGES_DIR
    assets/models/          MODELS_DIR (weights land here; gitignored)
    assets/data/            CSVs / datasets / FileReader inputs
    assets/output/          ImageWriter output (created on demand)
    .env.example            committed template of required secret keys
    .env                    your secrets (gitignored, never committed)

## Quickstart

    cdui start --project .          # edit graphs in the browser; Save writes the pair
    cdui project validate .         # CI gate: every graph must pass the publish pre-flight
    cdui project publish . --graph my-graph --slug my-service   # local publish + provenance

## Version control

Set a git identity if you have not:

    git config user.name  "You"
    git config user.email "you@example.com"

Then:

    git add -A && git commit -m "Initial project"

Commit `.env.example`, never `.env`. Commit a small **fetch script** that
downloads large data on demand -- do NOT commit datasets or weights (they are
gitignored). Model checkpoints written to `assets/models/` and images written
to `assets/output/` are local artifacts, not source.

## Continuous integration

Run `cdui project restore .` THEN `cdui project validate .` (restore installs
the plugin pins the graphs need; validate needs the full backend env incl.
torch -- cache the venv in CI). Validate `.` -- do NOT feed a raw `*.json` glob
to the validator, or it would pick up `layout/*.layout.json` files too.
"""


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

    # Scaffold directories.
    for rel in ("graphs", "layout", "assets/images", "assets/models",
                "assets/data"):
        (target / rel).mkdir(parents=True, exist_ok=True)
        # Keep otherwise-empty dirs trackable by git.
        keep = target / rel / ".gitkeep"
        if not any((target / rel).iterdir()):
            keep.write_text("", encoding="utf-8")

    # Scaffold files (never clobber an existing manifest/README under --force).
    name = target.name
    _write_if_absent(target / MANIFEST_FILENAME, _PROJECT_TOML.format(name=name))
    _write_if_absent(target / ".gitignore", _GITIGNORE)
    _write_if_absent(target / ".gitattributes", _GITATTRIBUTES)
    _write_if_absent(target / ".env.example", _ENV_EXAMPLE)
    _write_if_absent(target / "README.md", _README.format(name=name))

    # Adopt: copy every *.json from src into graphs/ and split immediately.
    if args.adopt:
        rc = _adopt(Path(args.adopt).expanduser().resolve(), target)
        if rc != 0:
            return rc

    # git init, NO initial commit (spec D3 / Section 5).
    if shutil.which("git"):
        if not (target / ".git").exists():
            subprocess.run(["git", "init"], cwd=target, capture_output=True)
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


def _collect_graphs(graphs_dir: Path) -> list[tuple[str, Path]]:
    """(base_name, file) for each graph: canonical `.graph.json` + legacy
    `.json`; raises RuntimeError naming both files on ambiguity (spec ID2)."""
    files: dict[str, Path] = {}
    if not graphs_dir.is_dir():
        return []
    for f in sorted(graphs_dir.glob("*.graph.json")):
        files[f.name[: -len(".graph.json")]] = f
    for f in sorted(graphs_dir.glob("*.json")):
        if f.name.endswith(".graph.json"):
            continue
        base = f.stem
        if base in files:
            raise RuntimeError(
                f"Graph '{base}' is ambiguous: both {files[base].name} and "
                f"{f.name} exist. Remove one.")
        files[base] = f
    return sorted(files.items())


def _init_registries_like_server() -> None:
    """Discover builtin + custom + PLUGIN nodes and presets exactly like the
    server lifespan (main.py) -- run_graph.py's init loads no plugins and is
    NOT sufficient (spec ID3)."""
    import app.core.plugin_loader as plugin_loader
    from app.config import settings
    from app.core.node_registry import registry
    from app.core.plugin_loader import install_plugin_finder
    from app.core.preset_registry import preset_registry

    registry.discover(settings.NODES_DIR, "app.nodes")
    registry.discover(settings.CUSTOM_NODES_DIR, "app.custom_nodes")
    lockfile = load_lockfile()
    builtin_root = plugin_loader.plugins_builtin_root()
    user_root = plugin_loader.plugins_user_root()
    for nodes_dir, pkg_name in install_plugin_finder(builtin_root, user_root, lockfile):
        registry.discover(nodes_dir, pkg_name)
    preset_registry.discover(settings.PRESETS_DIR, registry)
    for _pid, pdir in iter_plugin_dirs(builtin_root, user_root, lockfile):
        preset_registry.discover(pdir / "presets", registry)


def _validate_one_graph(path: Path, base: str) -> list[str]:
    """The publish six-gate pre-flight, IN PUBLISH ORDER (routes_apps): secret
    -> contract -> entry -> wiring -> validate. Returns problems (empty=pass)."""
    from app.core import api_contract
    from app.core.graph_engine import (
        build_preset_fallback,
        find_entry_points,
        validate_graph,
    )
    from app.core.project import FORMAT_VERSION
    from app.core.secret_params import find_secret_violations

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as e:
        return [f"unreadable: {e}"]
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

    # 1. secrets (publish checks these FIRST, routes_apps.py).
    violations = find_secret_violations(nodes)
    if violations:
        v = violations[0]
        return [f"secret_in_graph: node '{v['node_id']}' param '{v['param']}'"]
    # 2. contract.
    contract = api_contract.derive_contract(nodes)
    if contract.problems:
        return [f"invalid_contract: {contract.problems[0]}"]
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
        if first.startswith("Unknown node type"):
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
    with --strict become errors (spec ID3)."""
    pins = manifest.get("plugins", {}) or {}
    if not pins:
        return False
    installed = load_lockfile().get("plugins", {})
    failed = False
    for pid, pin in pins.items():
        entry = installed.get(pid)
        pinned_sha = pin.get("sha") if isinstance(pin, dict) else None
        if entry is None:
            _warn_or_err(strict, f"pinned plugin '{pid}' is not installed -- "
                                 "run `cdui project restore`")
            failed = failed or strict
        elif pinned_sha and entry.get("sha") != pinned_sha:
            _warn_or_err(strict, f"plugin '{pid}' sha {entry.get('sha', '')[:7]} "
                                 f"!= pinned {pinned_sha[:7]} -- run restore")
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

    _init_registries_like_server()

    try:
        graphs = _collect_graphs(proj / "graphs")
    except RuntimeError as e:
        err(str(e), str(e))
        return 1

    all_ok = True
    for base, path in graphs:
        problems = _validate_one_graph(path, base)
        if problems:
            all_ok = False
            err(f"FAIL {base}: {problems[0]}", f"FAIL {base}: {problems[0]}")
        else:
            ok(f"PASS {base}", f"PASS {base}")

    pin_fail = _check_pins(manifest, args.strict)
    env_fail = _check_env_not_tracked(proj)

    if not all_ok or pin_fail or env_fail:
        err("驗證失敗", "Validation FAILED")
        return 1
    ok("驗證通過", "Validation passed")
    return 0


def _render_manifest(project_tbl: dict, pins: dict, publish: dict) -> str:
    """Regenerate the manifest with a machine-written [plugins] table (spec
    5). Comments in [plugins] are not preserved -- it is a generated section."""
    lines = ["[project]"]
    lines.append(f'name = "{project_tbl.get("name", "")}"')
    lines.append(f'format_version = {int(project_tbl.get("format_version", 1))}')
    if project_tbl.get("requires_codefyui"):
        lines.append(f'requires_codefyui = "{project_tbl["requires_codefyui"]}"')
    lines.append("")
    lines.append("[plugins]")
    lines.append("# written by `cdui project freeze`; installed by `cdui project restore`")
    for pid in sorted(pins):
        p = pins[pid]
        lines.append(
            f'{pid} = {{ url = "{p["url"]}", ref = "{p["ref"]}", '
            f'sha = "{p["sha"]}" }}')
    lines.append("")
    if publish:
        lines.append("[publish]")
        for k in ("graph", "slug"):
            if publish.get(k):
                lines.append(f'{k} = "{publish[k]}"')
        if "record_io" in publish:
            lines.append(f'record_io = {"true" if publish["record_io"] else "false"}')
        lines.append("")
    return "\n".join(lines) + "\n"


def cmd_freeze(args: argparse.Namespace) -> int:
    proj = Path(args.dir).expanduser().resolve()
    manifest_path = proj / MANIFEST_FILENAME
    from app.core.plugin_loader import tomllib
    if not manifest_path.exists():
        err(f"找不到 manifest：{manifest_path}", f"Manifest not found: {manifest_path}")
        return 1
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    section("凍結外掛版本", "Freezing plugin pins")
    pins: dict = {}
    for pid, entry in load_lockfile().get("plugins", {}).items():
        kind = entry.get("source_kind")
        if kind == "local":
            warn(f"略過本地連結：{pid}",
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
    manifest_path.write_text(
        _render_manifest(manifest.get("project", {}), pins,
                         manifest.get("publish", {})),
        encoding="utf-8")
    ok(f"已寫入 {len(pins)} 個釘選", f"Wrote {len(pins)} pin(s) to the manifest")
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
        _kind, owner, repo, _ref = parse_source(url)
        inst_args = argparse.Namespace(
            force=True, no_confirm=True, trust_author=True, pinned_sha=sha)
        # Trust _install_github's return code (spec ID11: install BY the pinned
        # sha). It never re-resolves the ref when pinned_sha is set, so success
        # (rc == 0) already means the installed plugin is at this exact sha.
        rc = _install_github(owner, repo, ref, inst_args, lockfile)
        if rc != 0:
            rc_all = 1
    return rc_all


def _write_if_absent(path: Path, content: str) -> None:
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def _adopt(src: Path, target: Path) -> int:
    from app.core.project import write_graph_pair

    if not src.is_dir():
        err(f"--adopt 來源不是目錄：{src}", f"--adopt source is not a directory: {src}")
        return 1
    graphs_dir = target / "graphs"
    layout_dir = target / "layout"
    count = 0
    for jf in sorted(src.glob("*.json")):
        if jf.name.endswith(".layout.json"):
            continue
        base = jf.name[:-len(".graph.json")] if jf.name.endswith(".graph.json") else jf.stem
        try:
            payload = json.loads(jf.read_text(encoding="utf-8"))
        except (ValueError, OSError) as e:
            warn(f"略過 {jf.name}：{e}", f"Skipping {jf.name}: {e}")
            continue
        write_graph_pair(
            graphs_dir / f"{base}.graph.json",
            layout_dir / f"{base}.layout.json",
            payload,
        )
        count += 1
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
    p_val.add_argument("--strict", action="store_true",
                       help="treat missing/mismatched plugin pins as errors")
    p_val.set_defaults(_func=cmd_validate)

    p_freeze = sub.add_parser("freeze", help="Write installed github plugin pins into the manifest")
    p_freeze.add_argument("dir", help="project directory")
    p_freeze.set_defaults(_func=cmd_freeze)

    p_restore = sub.add_parser("restore", help="Install the manifest's plugin pins by their exact SHA")
    p_restore.add_argument("dir", help="project directory")
    p_restore.set_defaults(_func=cmd_restore)

    return p


def main(argv: list[str] | None = None) -> int:
    _reconfigure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args._func(args)


if __name__ == "__main__":
    sys.exit(main())
