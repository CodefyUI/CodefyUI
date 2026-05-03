#!/usr/bin/env python3
"""CodefyUI 跨平台任務執行器。

用法（建議）：
    cdui <command>                 # 若已透過 install 腳本加到 PATH
    ./cdui <command>               # 從專案根目錄執行
    python scripts/dev.py <command>

指令：
    install     安裝所有節點需要的依賴（含 PyTorch wheel 選擇）
                旗標：--gpu {auto|cpu|cu118|cu121|cu124|cu126|cu128|rocm6.1|rocm6.2|mps|skip}
                      --dev / --no-dev   是否安裝測試工具（pytest 等）
                      --yes / -y         略過互動，自動偵測 + 非 dev
                從 TTY 不帶旗標執行會跳出互動選單；從非 TTY（curl|bash、CI）走 --yes。
    update      拉取最新版本並重新安裝依賴（接受同 install 的旗標）
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
    CODEFYUI_GPU            預設 --gpu 值（命令列旗標仍會覆蓋）
    CODEFYUI_DEV            預設 --dev 值；1/true/yes 開、0/false/no 關
"""

import argparse
import os
import platform
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


# ── Install: PyTorch wheel selection ──────────────────────────────────────────

# Mapping from `--gpu` choice → PyTorch wheel index URL.
#   None      → let PyPI resolve via `-e .` (auto-detected fallback / mps)
#   "__skip__" → don't touch torch at all (preserves user's manual override)
TORCH_INDEX_URLS: dict[str, str | None] = {
    "auto":    None,                                            # resolved at runtime
    "cpu":     "https://download.pytorch.org/whl/cpu",
    "cu118":   "https://download.pytorch.org/whl/cu118",
    "cu121":   "https://download.pytorch.org/whl/cu121",
    "cu124":   "https://download.pytorch.org/whl/cu124",
    "cu126":   "https://download.pytorch.org/whl/cu126",
    "cu128":   "https://download.pytorch.org/whl/cu128",
    "rocm6.1": "https://download.pytorch.org/whl/rocm6.1",
    "rocm6.2": "https://download.pytorch.org/whl/rocm6.2",
    "mps":     None,                                            # default PyPI on Apple Silicon
    "skip":    "__skip__",                                      # leave torch untouched
}


def _recommended_cu_for_driver(driver_version: str) -> str:
    """Map an NVIDIA driver version to the latest compatible PyTorch CUDA wheel.

    PyTorch's compat matrix shifts each release; these floors are deliberately
    conservative — better to suggest an older wheel than ship one the driver
    can't load. Users can override via the menu / --gpu flag.
    """
    try:
        major = int(driver_version.split(".")[0])
    except (ValueError, IndexError):
        return "cu121"
    if major >= 560:
        return "cu128"
    if major >= 555:
        return "cu126"
    if major >= 545:
        return "cu124"
    if major >= 530:
        return "cu121"
    if major >= 520:
        return "cu118"
    return "cpu"


def detect_gpu() -> tuple[str, str]:
    """Best-effort GPU detection. Returns ``(display_label, recommended_key)``.

    The recommended_key is one of TORCH_INDEX_URLS' keys (excluding "auto" /
    "skip"). Detection failures collapse to ("CPU only", "cpu") — never raises.
    """
    if platform.system() == "Darwin":
        if platform.machine() in ("arm64", "aarch64"):
            return ("Apple Silicon (MPS)", "mps")
        return ("macOS x86_64", "cpu")

    if shutil.which("nvidia-smi"):
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5, check=True,
            )
            first = (proc.stdout or "").strip().splitlines()[0] if proc.stdout else ""
            if first:
                name, _, driver = first.partition(",")
                name, driver = name.strip(), driver.strip()
                cu = _recommended_cu_for_driver(driver)
                return (f"{name} (driver {driver})", cu)
        except (subprocess.SubprocessError, OSError, ValueError, IndexError):
            pass

    if platform.system() == "Linux" and shutil.which("rocm-smi"):
        return ("AMD GPU (ROCm)", "rocm6.2")

    return ("CPU only", "cpu")


def _parse_install_args(argv_tail: list[str]) -> argparse.Namespace:
    """Parse the flags passed to `cdui install` / `cdui update`."""
    p = argparse.ArgumentParser(
        prog="cdui install",
        description=(
            "Install backend (with PyTorch wheel + dev tooling choice) and "
            "frontend. From a TTY without flags, runs an interactive menu."
        ),
    )
    p.add_argument(
        "--gpu",
        choices=list(TORCH_INDEX_URLS.keys()),
        default=None,
        help="PyTorch wheel variant; auto-detect if omitted.",
    )
    dev_grp = p.add_mutually_exclusive_group()
    dev_grp.add_argument(
        "--dev", dest="dev", action="store_true", default=None,
        help="Install dev tooling (pytest, httpx, ...).",
    )
    dev_grp.add_argument(
        "--no-dev", dest="dev", action="store_false",
        help="Skip dev tooling (default).",
    )
    p.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip prompts; equivalent to --gpu auto --no-dev when nothing else set.",
    )
    return p.parse_args(argv_tail)


def _prompt_install_options(detected_label: str, detected_gpu: str) -> tuple[str, bool]:
    """Interactive menu for GPU + dev choice. Stays inside the terminal — no curses."""
    options = ["auto", "cpu", "cu118", "cu121", "cu124", "cu126", "cu128",
               "rocm6.1", "rocm6.2", "mps", "skip"]

    print()
    print("=== CodefyUI install ===")
    print(f"偵測到：{detected_label}")
    print()
    print("選擇 PyTorch wheel：")
    for i, opt in enumerate(options, 1):
        marker = ""
        if opt == "auto":
            marker = f"  → {detected_gpu}（依偵測結果）"
        elif opt == detected_gpu:
            marker = "  ← 偵測到"
        print(f"  [{i:>2}] {opt}{marker}")
    print()

    while True:
        raw = input("選擇（直接 Enter = 1, auto）：").strip() or "1"
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                gpu = options[idx]
                break
        except ValueError:
            pass
        print(f"  輸入無效，請填 1 到 {len(options)}")

    raw = input("是否安裝 dev 測試工具（pytest, httpx 等）？[y/N]：").strip().lower()
    dev = raw in ("y", "yes")

    print(f"\n→ gpu={gpu}, dev={dev}\n")
    return gpu, dev


def _resolve_install_options(argv_tail: list[str]) -> tuple[str, bool]:
    """Combine CLI flags + env vars + interactive prompt into a final (gpu, dev)."""
    args = _parse_install_args(argv_tail)
    detected_label, detected_gpu = detect_gpu()

    gpu = args.gpu or os.environ.get("CODEFYUI_GPU", "").strip() or None
    if gpu is not None and gpu not in TORCH_INDEX_URLS:
        print(f"錯誤：未知的 --gpu 值 {gpu!r}（合法值：{', '.join(TORCH_INDEX_URLS)}）",
              file=sys.stderr)
        sys.exit(2)

    dev = args.dev
    if dev is None:
        env_dev = os.environ.get("CODEFYUI_DEV", "").strip().lower()
        if env_dev in ("1", "true", "yes"):
            dev = True
        elif env_dev in ("0", "false", "no"):
            dev = False

    interactive = (
        not args.yes
        and gpu is None
        and dev is None
        and sys.stdin.isatty()
    )
    if interactive:
        gpu, dev = _prompt_install_options(detected_label, detected_gpu)
    else:
        if gpu is None:
            gpu = "auto"
        if dev is None:
            dev = False
        print(f"=== CodefyUI install: gpu={gpu}, dev={dev}（偵測：{detected_label}）===")

    if gpu == "auto":
        gpu = detected_gpu

    return gpu, dev


# ── Commands ──────────────────────────────────────────────────────────────────

def install(gpu: str, dev: bool) -> None:
    """Backend + frontend install. Caller resolves `gpu` / `dev` choices."""
    if VENV.exists():
        print("=== Backend: 虛擬環境已存在，跳過建立 ===")
    else:
        print("=== Backend: 建立虛擬環境 ===")
        run(["uv", "venv", "--python", "3.11"], cwd=BACKEND_DIR)

    # Step 1: PyTorch wheel — installed BEFORE `-e .` so the variant satisfies
    # the `torch>=2.0.0` dependency without re-resolving from PyPI default.
    index_url = TORCH_INDEX_URLS.get(gpu)
    if index_url == "__skip__":
        print("=== Backend: 略過 PyTorch 安裝（保留現有版本）===")
    elif index_url is None:
        print(f"=== Backend: PyTorch 走 PyPI 預設（gpu={gpu}）===")
    else:
        print(f"=== Backend: 安裝 PyTorch（{gpu}）— {index_url} ===")
        run(["uv", "pip", "install", "torch", "torchvision",
             "--index-url", index_url], cwd=BACKEND_DIR)

    # Step 2: project + every node's runtime deps. `gymnasium` / `safetensors` /
    # `tiktoken` etc. are all in [project.dependencies] now — no separate
    # explicit install needed.
    spec = ".[dev]" if dev else "."
    print(f"=== Backend: 安裝依賴（{spec}）===")
    run(["uv", "pip", "install", "-e", spec], cwd=BACKEND_DIR)

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


def install_command() -> None:
    """Entry-point shim for `cdui install`: parse argv → resolve → install."""
    gpu, dev = _resolve_install_options(sys.argv[2:])
    install(gpu=gpu, dev=dev)


def update() -> None:
    """拉取 main branch 的最新版本並重新同步依賴。Accepts the same flags as install."""
    if not (ROOT / ".git").exists():
        print("錯誤：此目錄不是 git clone，無法 update", file=sys.stderr)
        sys.exit(1)
    print("=== 拉取最新版本（main）===")
    # Explicit remote/branch so the command works even on a detached HEAD or
    # a branch that doesn't track upstream.
    run(["git", "fetch", "origin", "main"], cwd=ROOT)
    run(["git", "checkout", "main"], cwd=ROOT)
    run(["git", "merge", "--ff-only", "origin/main"], cwd=ROOT)

    # Old dist is for the previous source — wipe it so install re-downloads
    # (or re-builds, when pnpm is on PATH) for the new code.
    if DIST_DIR.exists():
        print("=== 移除舊 frontend/dist ===")
        shutil.rmtree(DIST_DIR, ignore_errors=True)

    gpu, dev = _resolve_install_options(sys.argv[2:])
    install(gpu=gpu, dev=dev)
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
    "install": install_command,
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
