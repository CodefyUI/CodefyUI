"""`cdui project ...` CLI (spec Section 8). Runs under the backend venv via
scripts/dev.py's _dispatch_project_subcommand hop, so `import app.*` works.

This module mirrors scripts/plugins.py: a build_parser() with subcommands, a
main(argv) that dispatches to args._func. This file ships `init`, `validate`,
`freeze`, `restore`, and `publish`.
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
    _install_github,
    _read_session_token,
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
from app.core.project import (
    GraphAmbiguityError,
    check_pin_issues,
    collect_graph_files,
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
    cdui project publish . --graph my-graph --slug my-service --create   # first publish + provenance

A slug declared in `codefyui.project.toml` `[publish]` creates its app on
first `cdui project publish .` automatically; an explicitly passed `--slug`
needs `--create` the first time (so a typo cannot mint a second app).

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

`cdui project freeze` rewrites `codefyui.project.toml` in place: keys you add
are preserved, but comments are not, and `[plugins]` is fully regenerated.

## Continuous integration

Run `cdui project restore .` THEN `cdui project validate .` (restore installs
the plugin pins the graphs need; validate needs the full backend env incl.
torch -- cache the venv in CI). Validate `.` -- do NOT feed a raw `*.json` glob
to the validator, or it would pick up `layout/*.layout.json` files too. A
canvas-only graph (no GraphOutput, e.g. a training graph) fails the publish
pre-flight; give it a real output or validate only your publish targets with
`cdui project validate . --graph <name>`.
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

    _init_registries_like_server()

    try:
        # Shared canonical-vs-legacy collection rule (app.core.project).
        graphs = collect_graph_files(proj / "graphs")
    except GraphAmbiguityError as e:
        err(str(e), str(e))
        return 1

    # --graph filter (issue #86): mixed projects keep canvas-only graphs
    # (train...) next to publish targets (serve...); let CI gate just the
    # targets. An unknown name is an ERROR -- a typo must never turn the
    # gate into a vacuous pass.
    requested = getattr(args, "graph", None)
    if requested:
        by_name = dict(graphs)
        missing = [n for n in requested if n not in by_name]
        if missing:
            avail = ", ".join(n for n, _ in graphs) or "(none)"
            err(f"graphs/ 沒有這些 graph：{', '.join(missing)}（現有：{avail}）",
                f"No such graph in graphs/: {', '.join(missing)} "
                f"(available: {avail})")
            return 1
        wanted = set(requested)
        graphs = [(n, p) for n, p in graphs if n in wanted]

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


def _render_extra_table(header: str, table: dict) -> list[str]:
    """Render a user-owned table freeze knows nothing about as a [header]
    section, nested dicts becoming [header.sub] sections (document order)."""
    lines = [f"[{header}]"]
    subtables = []
    for k, v in table.items():
        if isinstance(v, dict):
            subtables.append((k, v))
        else:
            lines.append(f"{_toml_key(k)} = {_toml_value(v)}")
    lines.append("")
    for k, v in subtables:
        lines.extend(_render_extra_table(f"{header}.{_toml_key(k)}", v))
    return lines


def _render_manifest(manifest: dict, pins: dict) -> str:
    """Regenerate the manifest with a machine-written [plugins] table (spec 5)
    while round-tripping every key freeze does not own (#87): unknown top-level
    keys/tables and unknown keys inside [project]/[publish] are preserved in
    document order. tomllib discards comments, so comments are NOT preserved.
    [plugins] is a generated section: its content is replaced wholesale."""
    project_tbl = manifest.get("project", {}) or {}
    publish = manifest.get("publish", {}) or {}
    extras = {k: v for k, v in manifest.items()
              if k not in ("project", "plugins", "publish")}

    lines: list[str] = []
    # Root-level (non-table) keys must precede the first [table] header.
    root_scalars = [(k, v) for k, v in extras.items() if not isinstance(v, dict)]
    for k, v in root_scalars:
        lines.append(f"{_toml_key(k)} = {_toml_value(v)}")
    if root_scalars:
        lines.append("")
    lines.append("[project]")
    lines.append(f'name = "{project_tbl.get("name", "")}"')
    lines.append(f'format_version = {int(project_tbl.get("format_version", 1))}')
    if project_tbl.get("requires_codefyui"):
        lines.append(f'requires_codefyui = "{project_tbl["requires_codefyui"]}"')
    for k, v in project_tbl.items():
        if k not in ("name", "format_version", "requires_codefyui"):
            lines.append(f"{_toml_key(k)} = {_toml_value(v)}")
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
        for k, v in publish.items():
            if k not in ("graph", "slug", "record_io"):
                lines.append(f"{_toml_key(k)} = {_toml_value(v)}")
        lines.append("")
    for k, v in extras.items():
        if isinstance(v, dict):
            lines.extend(_render_extra_table(_toml_key(k), v))
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
        _render_manifest(manifest, pins), encoding="utf-8")
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
    try:
        # Shared canonical-vs-legacy rule (app.core.project): skips
        # `*.layout.json`, and a source holding BOTH `x.json` and
        # `x.graph.json` aborts naming both files instead of silently
        # letting one overwrite the other (issue #85).
        pairs = collect_graph_files(src)
    except GraphAmbiguityError as e:
        err(str(e), str(e))
        return 1
    count = 0
    for base, jf in pairs:
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
    if count:
        # cmd_init scaffolds `.gitkeep` placeholders BEFORE adoption runs;
        # once real graph/layout files have landed, drop the stray ones.
        for d in (graphs_dir, layout_dir):
            (d / ".gitkeep").unlink(missing_ok=True)
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

    p_freeze = sub.add_parser("freeze", help="Write installed github plugin pins into the manifest")
    p_freeze.add_argument("dir", help="project directory")
    p_freeze.set_defaults(_func=cmd_freeze)

    p_restore = sub.add_parser("restore", help="Install the manifest's plugin pins by their exact SHA")
    p_restore.add_argument("dir", help="project directory")
    p_restore.set_defaults(_func=cmd_restore)

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
