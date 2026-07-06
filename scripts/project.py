"""`cdui project ...` CLI (spec Section 8). Runs under the backend venv via
scripts/dev.py's _dispatch_project_subcommand hop, so `import app.*` works.

This module mirrors scripts/plugins.py: a build_parser() with subcommands, a
main(argv) that dispatches to args._func. This file ships `init`; validate /
restore / freeze / publish are added by later tasks.
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
from plugins import _reconfigure_stdio, err, info, ok, section, t, warn

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

    return p


def main(argv: list[str] | None = None) -> int:
    _reconfigure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args._func(args)


if __name__ == "__main__":
    sys.exit(main())
