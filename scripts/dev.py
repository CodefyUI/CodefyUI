#!/usr/bin/env python3
"""CodefyUI 跨平台任務執行器。

用法（建議）：
    cdui <command>                 # 若已透過 install 腳本加到 PATH
    ./cdui <command>               # 從專案根目錄執行
    python scripts/dev.py <command>

指令：
    install     安裝所有依賴（backend；frontend dist 從 release 下載或本地 build）
    update      拉取最新版本並重新安裝依賴（git pull + install）
    build       建置 frontend dist（需 Node + pnpm，給開發者）
    dev         啟動開發伺服器（HMR；需 Node + pnpm）
    start       啟動 production（單一 uvicorn，用 frontend/dist；不需 Node）
    stop        停止所有服務
    test        執行 backend 測試
    clean       移除虛擬環境、node_modules 與 frontend/dist
    uninstall   解除安裝：clean + 移除全域 cdui launcher

環境變數：
    CODEFYUI_RELEASE_TAG    指定要下載的 release tag（預設：latest）
    CODEFYUI_FORCE_BUILD    設為 1 強制本地 build，不下載 release dist
"""

import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Force UTF-8 on Windows so we can print non-ASCII (Chinese headings etc.)
# without hitting cp1252 UnicodeEncodeError in CI / default consoles.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

ROOT = Path(__file__).resolve().parent.parent  # dev.py lives in <root>/scripts/
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"
DIST_DIR = FRONTEND_DIR / "dist"
DIST_INDEX = DIST_DIR / "index.html"
VENV = BACKEND_DIR / ".venv"
VENV_BIN = VENV / ("Scripts" if sys.platform == "win32" else "bin")
VENV_PY = VENV_BIN / ("python.exe" if sys.platform == "win32" else "python")

RELEASE_REPO = "treeleaves30760/CodefyUI"
RELEASE_ASSET = "frontend-dist.tar.gz"


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def _exec_into_venv_if_available() -> None:
    """Re-exec into backend/.venv's Python when it exists.

    Lets `python dev.py <cmd>` work transparently with any outer interpreter
    (uv-managed, system, or a temp env) — we hand off to the venv's Python so
    subprocess calls run against the installed deps.
    """
    if not VENV_PY.exists():
        return
    try:
        if Path(sys.executable).resolve() == VENV_PY.resolve():
            return
    except OSError:
        return
    import os
    os.execv(str(VENV_PY), [str(VENV_PY)] + sys.argv)


def _ensure_uv() -> None:
    if shutil.which("uv"):
        return
    print("=== uv 未安裝，正在自動安裝 ===")
    if sys.platform == "win32":
        subprocess.run(
            ["powershell", "-c", "irm https://astral.sh/uv/install.ps1 | iex"],
            check=True,
        )
    else:
        subprocess.run(
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            shell=True,
            check=True,
        )
    # 安裝後重新啟動自身，讓新 PATH 生效
    import os
    os.execv(sys.executable, [sys.executable] + sys.argv)


# ── Helpers ───────────────────────────────────────────────────────────────────

def run(cmd: list, cwd: Path = ROOT) -> None:
    # On Windows, subprocess doesn't search PATHEXT for relative commands,
    # so tools that ship as .cmd (e.g. pnpm.cmd) raise FileNotFoundError.
    # Delegating to cmd.exe via shell=True lets Windows resolve them, and
    # list2cmdline quotes our args safely.
    if sys.platform == "win32":
        subprocess.run(subprocess.list2cmdline(cmd), cwd=cwd, check=True, shell=True)
    else:
        subprocess.run(cmd, cwd=cwd, check=True)


def _stream(proc: subprocess.Popen, prefix: str) -> None:
    assert proc.stdout is not None
    for raw in iter(proc.stdout.readline, b""):
        print(f"{prefix} {raw.decode(errors='replace').rstrip()}", flush=True)


def _release_dist_url() -> str:
    """Build the GitHub release asset URL.

    `latest` redirects to the most recent non-prerelease — when CI publishes
    a pre-release (e.g. ``1.0.0rcN``), pin the tag explicitly via the env var.
    """
    tag = os.environ.get("CODEFYUI_RELEASE_TAG", "latest").strip() or "latest"
    if tag == "latest":
        return f"https://github.com/{RELEASE_REPO}/releases/latest/download/{RELEASE_ASSET}"
    return f"https://github.com/{RELEASE_REPO}/releases/download/{tag}/{RELEASE_ASSET}"


def fetch_release_dist() -> bool:
    """Download + extract prebuilt frontend dist from a GitHub release.

    Returns True on success. Used as a fallback when pnpm isn't available so
    end users can install without Node.
    """
    url = _release_dist_url()
    print(f"=== 下載 frontend dist：{url} ===")

    try:
        req = Request(url, headers={"User-Agent": "cdui-installer"})
        with urlopen(req, timeout=120) as resp:
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
                shutil.copyfileobj(resp, tmp)
                tarball = tmp.name
    except (URLError, HTTPError, TimeoutError) as e:
        print(f"  下載失敗：{e}")
        return False

    try:
        # Extract into a fresh dir so a half-extracted previous attempt can't
        # leave stray files behind.
        if DIST_DIR.exists():
            shutil.rmtree(DIST_DIR)
        DIST_DIR.mkdir(parents=True)

        # Python 3.12+ requires an explicit `filter=` to silence a
        # DeprecationWarning + future security default. Use the safer
        # `data` filter when available; older Python ignores the kwarg.
        with tarfile.open(tarball, "r:gz") as tf:
            extract_kwargs: dict = {}
            if hasattr(tarfile, "data_filter"):
                extract_kwargs["filter"] = "data"
            tf.extractall(DIST_DIR, **extract_kwargs)
    except (tarfile.TarError, OSError) as e:
        print(f"  解壓失敗：{e}")
        return False
    finally:
        try:
            os.unlink(tarball)
        except OSError:
            pass

    if not DIST_INDEX.exists():
        print("  解壓後找不到 index.html，可能 release asset 內容有誤")
        return False

    print(f"=== Frontend dist 解壓完成：{DIST_DIR} ===")
    return True


def _install_frontend_deps_if_needed() -> None:
    """For dev mode: ensure node_modules exists when pnpm is available."""
    if not (FRONTEND_DIR / "node_modules").exists():
        print("=== Frontend: 首次執行，安裝 node_modules ===")
        run(["pnpm", "install"], cwd=FRONTEND_DIR)


def _warn_if_dist_stale() -> None:
    """Print a warning when frontend/dist is older than the newest src file.

    Only meaningful in a developer checkout (frontend/src present) — release
    tarball installs ship without src so the comparison is skipped silently.
    """
    src = FRONTEND_DIR / "src"
    if not src.is_dir():
        return
    try:
        dist_mtime = DIST_INDEX.stat().st_mtime
    except OSError:
        return

    src_mtime = 0.0
    for p in src.rglob("*"):
        try:
            if p.is_file():
                src_mtime = max(src_mtime, p.stat().st_mtime)
        except OSError:
            continue

    if src_mtime <= dist_mtime:
        return

    from datetime import datetime
    delta_min = (src_mtime - dist_mtime) / 60
    src_when = datetime.fromtimestamp(src_mtime).strftime("%Y-%m-%d %H:%M")
    dist_when = datetime.fromtimestamp(dist_mtime).strftime("%Y-%m-%d %H:%M")
    print(
        f"\n⚠️  警告：frontend/dist 比 src 舊 {delta_min:.0f} 分鐘\n"
        f"    dist mtime: {dist_when}\n"
        f"    src  mtime: {src_when}\n"
        f"    若想看到最新前端，請先執行 'cdui build' 重新打包。\n",
        file=sys.stderr,
    )


# ── Commands ──────────────────────────────────────────────────────────────────

def install() -> None:
    if VENV.exists():
        print("=== Backend: 虛擬環境已存在，跳過建立 ===")
    else:
        print("=== Backend: 建立虛擬環境 ===")
        run(["uv", "venv", "--python", "3.11"], cwd=BACKEND_DIR)

    print("=== Backend: 安裝依賴 ===")
    run(["uv", "pip", "install", "-e", ".[dev]"], cwd=BACKEND_DIR)

    print("=== Backend: 安裝 PyTorch ===")
    run(["uv", "pip", "install", "torch", "torchvision", "gymnasium", "safetensors"],
        cwd=BACKEND_DIR)

    # Frontend: three branches in priority order.
    #   1. dist already present — nothing to do
    #   2. CODEFYUI_FORCE_BUILD=1 — local build path (developer)
    #   3. pnpm available — local build path (developer)
    #   4. fall back to downloading the release asset (end user, no Node)
    force_build = os.environ.get("CODEFYUI_FORCE_BUILD", "").strip() in ("1", "true", "yes")

    if DIST_INDEX.exists() and not force_build:
        print("=== Frontend: dist 已存在，略過 ===")
    elif force_build or shutil.which("pnpm"):
        if not shutil.which("pnpm"):
            print("錯誤：CODEFYUI_FORCE_BUILD=1 但找不到 pnpm", file=sys.stderr)
            sys.exit(1)
        print("=== Frontend: 安裝 node_modules ===")
        run(["pnpm", "install"], cwd=FRONTEND_DIR)
        print("=== Frontend: 建置 dist ===")
        run(["pnpm", "build"], cwd=FRONTEND_DIR)
    else:
        print("=== Frontend: 未偵測到 pnpm，改下載 release dist ===")
        if not fetch_release_dist():
            print(
                "\n錯誤：無法取得 frontend dist。可選擇其一：\n"
                "  1. 安裝 Node.js 24+ 與 pnpm 後重跑 cdui install\n"
                "  2. 設定 CODEFYUI_RELEASE_TAG=<tag> 指定特定 release\n"
                "  3. 檢查網路連線後重試",
                file=sys.stderr,
            )
            sys.exit(1)

    print("=== 安裝完成 ===")


def update() -> None:
    """拉取 main branch 的最新版本並重新同步依賴。"""
    if not (ROOT / ".git").exists():
        print("錯誤：此目錄不是 git clone，無法 update", file=sys.stderr)
        sys.exit(1)
    print("=== 拉取最新版本（main）===")
    # Explicit remote/branch so the command works even on a detached HEAD or
    # a branch that doesn't track upstream.
    run(["git", "fetch", "origin", "main"], cwd=ROOT)
    run(["git", "checkout", "main"], cwd=ROOT)
    run(["git", "merge", "--ff-only", "origin/main"], cwd=ROOT)

    # Old dist is for the previous source — wipe it so install() re-downloads
    # (or re-builds, when pnpm is on PATH) for the new code.
    if DIST_DIR.exists():
        print("=== 移除舊 frontend/dist ===")
        shutil.rmtree(DIST_DIR, ignore_errors=True)

    install()
    print("=== 更新完成 ===")


def build() -> None:
    """建置 frontend dist（需 Node + pnpm）。"""
    if not shutil.which("pnpm"):
        print("錯誤：build 需要 pnpm。請先安裝 Node.js 24+ 與 pnpm。", file=sys.stderr)
        sys.exit(1)
    if not (FRONTEND_DIR / "node_modules").exists():
        print("=== Frontend: 安裝 node_modules ===")
        run(["pnpm", "install"], cwd=FRONTEND_DIR)
    print("=== Frontend: 建置 dist ===")
    run(["pnpm", "build"], cwd=FRONTEND_DIR)
    print(f"=== 建置完成：{DIST_DIR} ===")


def start() -> None:
    """Production 模式：單一 uvicorn 由 FastAPI 直接 serve dist。"""
    if not DIST_INDEX.exists():
        print(
            "錯誤：找不到 frontend/dist/index.html\n"
            "  請執行 'cdui install'（下載 release dist）"
            " 或 'cdui build'（本地 build，需 pnpm）。",
            file=sys.stderr,
        )
        sys.exit(1)
    _warn_if_dist_stale()
    uvicorn = str(VENV_BIN / "uvicorn")
    print("=== CodefyUI 啟動（Ctrl+C 停止）===")
    print("    開啟 → http://localhost:8000")
    print("")
    run([uvicorn, "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=BACKEND_DIR)


def dev() -> None:
    if not shutil.which("pnpm"):
        print(
            "錯誤：dev 模式需要 pnpm（HMR）。請安裝 Node.js 24+ 與 pnpm，\n"
            "  或改用 'cdui start' 跑 production 模式（不需 Node）。",
            file=sys.stderr,
        )
        sys.exit(1)
    _install_frontend_deps_if_needed()

    uvicorn = str(VENV_BIN / "uvicorn")
    backend_cmd = [uvicorn, "app.main:app", "--reload"]
    frontend_cmd = ["pnpm", "dev"]

    shell = sys.platform == "win32"

    print("=== 啟動 CodefyUI（Ctrl+C 停止）===")
    print("    backend  → http://localhost:8000")
    print("    frontend → http://localhost:5173")
    print("")

    backend = subprocess.Popen(
        backend_cmd,
        cwd=BACKEND_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=shell,
    )
    frontend = subprocess.Popen(
        frontend_cmd,
        cwd=FRONTEND_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=shell,
    )

    threading.Thread(target=_stream, args=(backend, "[backend] "), daemon=True).start()
    threading.Thread(target=_stream, args=(frontend, "[frontend]"), daemon=True).start()

    try:
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        print("\n=== 停止服務 ===")
        backend.terminate()
        frontend.terminate()
        backend.wait()
        frontend.wait()


def stop() -> None:
    print("=== 停止所有服務 ===")
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/IM", "uvicorn.exe"], capture_output=True)
        subprocess.run(["taskkill", "/F", "/FI", "WINDOWTITLE eq vite*"], capture_output=True)
    else:
        subprocess.run(["pkill", "-f", "uvicorn app.main:app"], capture_output=True)
        subprocess.run(["pkill", "-f", "vite"], capture_output=True)
    print("=== 完成 ===")


def test() -> None:
    pytest = str(VENV_BIN / "pytest")
    run([pytest], cwd=BACKEND_DIR)


def clean() -> None:
    print("=== 清除虛擬環境、node_modules 與 frontend/dist ===")
    shutil.rmtree(VENV, ignore_errors=True)
    shutil.rmtree(FRONTEND_DIR / "node_modules", ignore_errors=True)
    shutil.rmtree(DIST_DIR, ignore_errors=True)
    print("=== 完成 ===")


def uninstall() -> None:
    """移除 venv、node_modules，以及全域 cdui launcher stub。"""
    clean()
    launcher = (
        Path.home() / ".local" / "bin" / ("cdui.cmd" if sys.platform == "win32" else "cdui")
    )
    if launcher.exists() or launcher.is_symlink():
        try:
            launcher.unlink()
            print(f"=== 已移除 launcher：{launcher} ===")
        except OSError as e:
            print(f"=== 無法移除 launcher {launcher}：{e} ===")
    else:
        print(f"=== 未發現 launcher（{launcher}），跳過 ===")
    print(f"=== 解除安裝完成。若要完全移除，請手動刪除：{ROOT} ===")


# ── Entry point ───────────────────────────────────────────────────────────────

COMMANDS = {
    "install": install,
    "update": update,
    "build": build,
    "dev": dev,
    "start": start,
    "stop": stop,
    "test": test,
    "clean": clean,
    "uninstall": uninstall,
}

# Commands that mutate or remove the venv must run from the outer interpreter,
# never from the venv's Python (Windows can't delete a running exe; update
# rewrites deps in-place).
_SKIP_VENV_EXEC = {"install", "update", "clean", "uninstall"}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    if sys.argv[1] not in _SKIP_VENV_EXEC:
        _exec_into_venv_if_available()
    _ensure_uv()
    COMMANDS[sys.argv[1]]()
