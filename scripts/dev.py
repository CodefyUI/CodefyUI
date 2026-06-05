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
                預設在背景執行（關掉 terminal 也會繼續跑），用 cdui status /
                cdui stop 管理。加 --foreground / -f 則在前景執行（Ctrl+C 停止）。
    status      顯示系統與伺服器狀態儀表板（像 btop / k9s：CPU、記憶體、
                磁碟、GPU、行程、伺服器 PID 與健康檢查）
                預設持續刷新（每 2 秒，Ctrl+C 離開）；輸出被導向管線或非互動
                環境時自動改為只輸出一次。
                旗標：[秒] 或 -w [秒]    自訂刷新間隔（如 cdui status 1）
                      --once / -1       只輸出一次
    stop        停止所有服務（含背景伺服器）
    test        執行 backend 測試
    clean       移除虛擬環境、node_modules 與 frontend/dist
    uninstall   解除安裝：clean + 移除全域 cdui launcher

    plugin <subcmd> ...
                與 cdui plugin 完全相同的介面，但 lockfile 寫到 repo 內的
                <repo>/.codefyui_dev/plugins/ 而不是 %LOCALAPPDATA%\\codefyui，
                讓多個 dev clone 互不干擾。官方 foundations/deep/rl direction
                pack 預設不會安裝，需要時逐一裝即可。範例：
                    python scripts/dev.py plugin install deep
                    python scripts/dev.py plugin install owner/repo@main
                    python scripts/dev.py plugin list
                    python scripts/dev.py plugin enable deep     # 啟用
                    python scripts/dev.py plugin disable deep    # 停用（檔案保留）
                    python scripts/dev.py plugin uninstall deep  # 從 lockfile 移除

環境變數：
    CODEFYUI_RELEASE_TAG    指定要下載的 release tag（預設：latest）
    CODEFYUI_FORCE_BUILD    設為 1 強制本地 build，不下載 release dist
    CODEFYUI_GPU            預設 --gpu 值（命令列旗標仍會覆蓋）
    CODEFYUI_DEV            預設 --dev 值；1/true/yes 開、0/false/no 關
    CODEFYUI_USER_DATA_DIR  覆蓋 platformdirs user-data 位置（plugin lockfile
                            + session.token + asset cache）。執行 scripts/dev.py
                            的任何子命令都會自動設成 <repo>/.codefyui_dev/。
                            外部明確設定的值（譬如 CI）會優先生效。
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
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


# ── i18n + ANSI styling ───────────────────────────────────────────────────────

def _detect_lang() -> str:
    """Decide between zh / en. CODEFYUI_LANG > LANG/LC_ALL > python locale > en."""
    explicit = os.environ.get("CODEFYUI_LANG", "").strip().lower()
    if explicit in ("en", "english"):
        return "en"
    if explicit in ("zh", "zh-tw", "zh_tw", "zh-hk", "zh_hk", "zh-cn", "zh_cn", "chinese"):
        return "zh"
    raw = (os.environ.get("LANG") or os.environ.get("LC_ALL") or "").lower()
    if raw.startswith("zh"):
        return "zh"
    if raw.startswith("en"):
        return "en"
    try:
        import locale
        # getlocale() replaces the deprecated getdefaultlocale() (removed in
        # Python 3.15). Fall back to the LANG/LC_* env vars it reads from when
        # the C library reports no locale (common on minimal images).
        loc = (locale.getlocale()[0] or "").lower()
        if not loc:
            loc = (os.environ.get("LC_ALL") or os.environ.get("LANG") or "").lower()
        if loc.startswith("zh"):
            return "zh"
    except Exception:
        pass
    return "en"


LANG = _detect_lang()


def t(zh: str, en: str) -> str:
    """Pick the localized message for the current LANG."""
    return zh if LANG == "zh" else en


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if not sys.stdout.isatty():
        return False
    return True


USE_COLOR = _supports_color()


def _enable_windows_vt() -> None:
    """Switch the legacy Windows console into VT mode so ANSI escapes render."""
    if sys.platform != "win32" or not USE_COLOR:
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        h = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_ulong()
        if kernel32.GetConsoleMode(h, ctypes.byref(mode)):
            # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            kernel32.SetConsoleMode(h, mode.value | 0x0004)
    except Exception:
        pass


_enable_windows_vt()


def _ansi(*codes: int) -> str:
    return f"\x1b[{';'.join(map(str, codes))}m" if USE_COLOR else ""


RESET  = _ansi(0)
BOLD   = _ansi(1)
DIM    = _ansi(2)
RED    = _ansi(31)
GREEN  = _ansi(32)
YELLOW = _ansi(33)
BLUE   = _ansi(34)
MAGENTA = _ansi(35)
CYAN   = _ansi(36)
GRAY   = _ansi(90)


def _display_width(s: str) -> int:
    """Visual column width — Chinese / fullwidth chars count as 2."""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in ("F", "W") else 1 for c in s)


def section(zh: str, en: str) -> None:
    """Coloured `=== heading ===` line, picks language."""
    print(f"{BOLD}{CYAN}=== {t(zh, en)} ==={RESET}")


def banner(zh: str, en: str) -> None:
    """Top-of-screen banner used at the start of install."""
    msg = t(zh, en)
    w = _display_width(msg) + 2
    print()
    print(f"{BOLD}{MAGENTA}┌{'─' * w}┐{RESET}")
    print(f"{BOLD}{MAGENTA}│ {msg} │{RESET}")
    print(f"{BOLD}{MAGENTA}└{'─' * w}┘{RESET}")
    print()


def warn(zh: str, en: str) -> None:
    print(f"{YELLOW}! {t(zh, en)}{RESET}", file=sys.stderr)


def err(zh: str, en: str) -> None:
    print(f"{RED}✗ {t(zh, en)}{RESET}", file=sys.stderr)


ROOT = Path(__file__).resolve().parent.parent  # dev.py lives in <root>/scripts/
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"
DIST_DIR = FRONTEND_DIR / "dist"
DIST_INDEX = DIST_DIR / "index.html"
VENV = BACKEND_DIR / ".venv"
VENV_BIN = VENV / ("Scripts" if sys.platform == "win32" else "bin")
VENV_PY = VENV_BIN / ("python.exe" if sys.platform == "win32" else "python")

# In-repo user-data dir for dev-mode plugin installs. Backend reads this via
# the CODEFYUI_USER_DATA_DIR env var (see plugin_loader.plugins_user_root).
# Gitignored so each dev clone manages its own lockfile.
DEV_USER_DATA_DIR = ROOT / ".codefyui_dev"
DEV_LOCKFILE = DEV_USER_DATA_DIR / "plugins" / "installed.json"


def _apply_dev_env() -> None:
    """Force dev-mode user-data dir.

    Running ``scripts/dev.py`` from inside a clone is itself the dev-mode
    signal — set ``CODEFYUI_USER_DATA_DIR=<repo>/.codefyui_dev/`` so plugin
    install, the running server, hot-reload's session token, and the asset
    cache all land in the same repo-local sandbox. The global
    ``cdui plugin install`` path (which writes to
    ``%LOCALAPPDATA%\\codefyui``) stays untouched, so contributors can
    keep a separate production install on the same machine.

    Idempotent and safe to call multiple times — only ever writes to
    ``os.environ`` if the variable isn't already set, so an outer caller
    that intentionally sets it (e.g. CI pointing at a tmp dir) wins.
    """
    os.environ.setdefault("CODEFYUI_USER_DATA_DIR", str(DEV_USER_DATA_DIR))
    DEV_USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DEV_USER_DATA_DIR / "plugins").mkdir(parents=True, exist_ok=True)

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


def _require_venv_tool(tool_name: str) -> str:
    """Resolve a venv-installed executable, or exit with a clean repair hint.

    Many users land here after a partial install (network blip during
    ``cdui install``, interrupted GPU index download, etc.). Surfacing a raw
    ``FileNotFoundError`` from subprocess is hostile; a single sentence
    explaining the fix is far better.
    """
    exe = VENV_BIN / (f"{tool_name}.exe" if sys.platform == "win32" else tool_name)
    if exe.exists():
        return str(exe)
    if not VENV.exists():
        msg = (
            f"錯誤：找不到虛擬環境（{VENV}）。\n"
            f"  請先安裝後再執行此指令：\n"
            f"    cdui install\n"
        )
    else:
        msg = (
            f"錯誤：虛擬環境存在但找不到 {tool_name}（{exe}）。\n"
            f"  上次 'cdui install' 可能未完成。建議：\n"
            f"    cdui clean && cdui install\n"
        )
    print(msg, file=sys.stderr)
    sys.exit(1)


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


def _resolve_release_tag() -> "str | None":
    """Resolve the release tag to install (``latest`` → concrete version).

    Returns the concrete tag, or ``None`` when the GitHub API can't be
    reached. Used to pin the backend checkout to the same release the
    prebuilt frontend dist comes from, so the two never drift apart.
    """
    tag = os.environ.get("CODEFYUI_RELEASE_TAG", "latest").strip() or "latest"
    if tag != "latest":
        return tag
    url = f"https://api.github.com/repos/{RELEASE_REPO}/releases/latest"
    try:
        req = Request(url, headers={"User-Agent": "cdui-installer",
                                    "Accept": "application/vnd.github+json"})
        with urlopen(req, timeout=30) as resp:
            data = json.load(resp)
        name = data.get("tag_name")
        return name or None
    except (URLError, HTTPError, TimeoutError, ValueError) as e:
        print(f"  無法解析 latest release tag：{e}")
        return None


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
        f"\n警告：frontend/dist 比 src 舊 {delta_min:.0f} 分鐘\n"
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
    p.add_argument(
        "--lang", choices=["en", "zh"], default=None,
        help="Output language (en/zh). Auto-detected from CODEFYUI_LANG / LANG / locale otherwise.",
    )
    return p.parse_args(argv_tail)


def _prompt_install_options(detected_label: str, detected_gpu: str) -> tuple[str, bool]:
    """Interactive menu for GPU + dev choice. Stays inside the terminal — no curses."""
    options = ["auto", "cpu", "cu118", "cu121", "cu124", "cu126", "cu128",
               "rocm6.1", "rocm6.2", "mps", "skip"]
    descriptions = {
        "auto":    t("依偵測自動選擇", "auto-pick from detection"),
        "cpu":     "CPU only",
        "cu118":   "CUDA 11.8",
        "cu121":   "CUDA 12.1",
        "cu124":   "CUDA 12.4",
        "cu126":   "CUDA 12.6",
        "cu128":   "CUDA 12.8",
        "rocm6.1": "ROCm 6.1 (AMD, Linux)",
        "rocm6.2": "ROCm 6.2 (AMD, Linux)",
        "mps":     t("Apple Silicon (MPS)", "Apple Silicon (MPS)"),
        "skip":    t("不動 torch（保留現有）", "leave torch untouched"),
    }

    banner("CodefyUI 安裝", "CodefyUI installer")
    print(f"  {DIM}{t('偵測到', 'Detected')}:{RESET} {GREEN}{detected_label}{RESET}")
    print(f"  {DIM}{t('語言', 'Language')}:{RESET}  {LANG}  {GRAY}{t('（用 --lang en 或 CODEFYUI_LANG=en 切換）', '(set --lang or CODEFYUI_LANG to switch)')}{RESET}")
    print()
    print(f"  {BOLD}{t('PyTorch wheel：', 'PyTorch wheel:')}{RESET}")
    for i, opt in enumerate(options, 1):
        is_default = (opt == "auto")
        is_detected = (opt == detected_gpu)
        # Build trailing annotation
        bits = [descriptions[opt]]
        if is_default:
            bits.append(t("預設", "default"))
            bits.append(f"→ {detected_gpu}")
        elif is_detected:
            bits.append(t("符合偵測結果", "matches detection"))
        annotation = f"  {GRAY}— {', '.join(bits)}{RESET}"

        num = f"{i:>2}"
        label_color = GREEN if is_default else (CYAN if is_detected else "")
        label_reset = RESET if label_color else ""
        print(f"   {DIM}{num}){RESET} {label_color}{opt:<8}{label_reset}{annotation}")
    print()

    while True:
        prompt = t("選擇（Enter = 1, auto）", "Choose [1]")
        raw = input(f"  {prompt}: ").strip() or "1"
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                gpu = options[idx]
                break
        except ValueError:
            pass
        print(f"  {YELLOW}{t('輸入無效，請填', 'Invalid choice — enter')} 1..{len(options)}{RESET}")

    dev_prompt = t("安裝 dev 測試工具（pytest, httpx 等）？[y/N]", "Install dev tooling (pytest, httpx, ...) [y/N]")
    raw = input(f"  {dev_prompt}: ").strip().lower()
    dev = raw in ("y", "yes")

    print()
    print(f"  {DIM}→{RESET} gpu={GREEN}{gpu}{RESET}, dev={GREEN}{dev}{RESET}")
    print()
    return gpu, dev


def _resolve_install_options(argv_tail: list[str]) -> tuple[str, bool]:
    """Combine CLI flags + env vars + interactive prompt into a final (gpu, dev)."""
    args = _parse_install_args(argv_tail)

    # --lang flag overrides env-var-based LANG detection done at module load.
    if args.lang:
        global LANG
        LANG = args.lang

    detected_label, detected_gpu = detect_gpu()

    gpu = args.gpu or os.environ.get("CODEFYUI_GPU", "").strip() or None
    if gpu is not None and gpu not in TORCH_INDEX_URLS:
        err(f"未知的 --gpu 值 {gpu!r}（合法值：{', '.join(TORCH_INDEX_URLS)}）",
            f"Unknown --gpu value {gpu!r} (valid: {', '.join(TORCH_INDEX_URLS)})")
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
        section(
            f"CodefyUI install: gpu={gpu}, dev={dev}（偵測：{detected_label}）",
            f"CodefyUI install: gpu={gpu}, dev={dev} (detected: {detected_label})",
        )

    if gpu == "auto":
        gpu = detected_gpu

    return gpu, dev


def _get_installed_torch_version() -> str | None:
    """Read torch's __version__ from the venv without importing torch."""
    candidates = [
        VENV / "Lib" / "site-packages" / "torch" / "version.py",            # Windows
        VENV / "lib" / "site-packages" / "torch" / "version.py",            # uv layout
    ]
    lib = VENV / "lib"
    if lib.exists():
        for entry in lib.iterdir():
            cand = entry / "site-packages" / "torch" / "version.py"
            if cand.exists():
                candidates.append(cand)
    for path in candidates:
        if not path.exists():
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("__version__"):
                    return line.split("=", 1)[1].strip().strip("'\"")
        except OSError:
            pass
    return None


def _print_post_install_summary(gpu: str, dev: bool) -> None:
    """Print the 'Installed / Next steps' panel — what was done + how to run it."""
    torch_ver = _get_installed_torch_version() or t("(尚未安裝)", "(not installed)")
    has_pnpm = bool(shutil.which("pnpm"))
    has_dist = DIST_INDEX.exists()
    cdui_cmd = ".\\cdui" if sys.platform == "win32" else "./cdui"

    print()
    print(f"{BOLD}{GREEN}✓ {t('安裝完成', 'Installation complete')}{RESET}")
    print()
    print(f"  {DIM}PyTorch:{RESET}  {torch_ver}  {GRAY}(gpu={gpu}){RESET}")
    print(f"  {DIM}Backend:{RESET}  {BACKEND_DIR}")
    print(f"  {DIM}Frontend:{RESET} {DIST_DIR if has_dist else t('(未建置)', '(not built)')}")
    if dev:
        print(f"  {DIM}Dev tools:{RESET} pytest, httpx, httpx-ws")
    print()
    print(f"{BOLD}{CYAN}▸ {t('下一步', 'Next steps')}{RESET}")
    print()

    if has_pnpm:
        print(f"  {BOLD}{t('開發模式', 'Development')}{RESET} {GRAY}({t('HMR、需要 pnpm', 'HMR, requires pnpm')}){RESET}")
        print(f"    {GREEN}{cdui_cmd} dev{RESET}")
        print(f"      {GRAY}→ backend  http://localhost:8000{RESET}")
        print(f"      {GRAY}→ frontend http://localhost:5173{RESET}")
        print()

    print(f"  {BOLD}{t('正式模式', 'Production')}{RESET} {GRAY}({t('單一 uvicorn 直接 serve dist', 'single uvicorn serving dist')}){RESET}")
    print(f"    {GREEN}{cdui_cmd} start{RESET}")
    print(f"      {GRAY}→ http://localhost:8000{RESET}")
    print()

    other_bits = [f"{cdui_cmd} stop", f"{cdui_cmd} clean"]
    if dev:
        other_bits.insert(1, f"{cdui_cmd} test")
    print(f"  {DIM}{t('其他', 'Other')}:{RESET} " + GRAY + " | ".join(other_bits) + RESET)

    if not has_pnpm:
        print()
        warn(
            "未偵測到 pnpm，僅可使用 production 模式。如需開發模式請安裝 Node.js 24+ 與 pnpm。",
            "pnpm not detected — only production mode available. Install Node.js 24+ and pnpm for dev mode.",
        )
    if gpu == "skip":
        print()
        warn(
            "已略過 PyTorch 安裝；請自行確保 venv 內已安裝合適的 torch。",
            "PyTorch install was skipped; ensure a suitable torch is already in the venv.",
        )
    print()


# ── Commands ──────────────────────────────────────────────────────────────────

def install(gpu: str, dev: bool) -> None:
    """Backend + frontend install. Caller resolves `gpu` / `dev` choices."""
    if VENV.exists():
        section("Backend: 虛擬環境已存在，跳過建立",
                "Backend: virtual env already exists, skipping create")
    else:
        section("Backend: 建立虛擬環境", "Backend: creating virtual env")
        run(["uv", "venv", "--python", "3.11"], cwd=BACKEND_DIR)

    # Step 1: PyTorch wheel — installed BEFORE `-e .` so the variant satisfies
    # the `torch>=2.0.0` dependency without re-resolving from PyPI default.
    index_url = TORCH_INDEX_URLS.get(gpu)
    if index_url == "__skip__":
        section("Backend: 略過 PyTorch 安裝（保留現有版本）",
                "Backend: skipping PyTorch install (keeping existing)")
    elif index_url is None:
        section(f"Backend: PyTorch 走 PyPI 預設（gpu={gpu}）",
                f"Backend: PyTorch from PyPI default (gpu={gpu})")
    else:
        section(f"Backend: 安裝 PyTorch（{gpu}）— {index_url}",
                f"Backend: installing PyTorch ({gpu}) — {index_url}")
        # `--reinstall-package` forces uv to drop the existing torch even when
        # the version constraint is already satisfied. Without it, swapping
        # variants (e.g. `--gpu cpu` after a previous `cu128` install) is a
        # no-op and the user keeps the wrong wheel.
        run(["uv", "pip", "install",
             "--reinstall-package", "torch",
             "--reinstall-package", "torchvision",
             "torch", "torchvision",
             "--index-url", index_url], cwd=BACKEND_DIR)

    # Step 2: project + every node's runtime deps. `gymnasium` / `safetensors` /
    # `tiktoken` etc. are all in [project.dependencies] now — no separate
    # explicit install needed.
    spec = ".[dev]" if dev else "."
    section(f"Backend: 安裝依賴（{spec}）", f"Backend: installing dependencies ({spec})")
    run(["uv", "pip", "install", "-e", spec], cwd=BACKEND_DIR)

    # Frontend: three branches in priority order.
    #   1. dist already present — nothing to do
    #   2. CODEFYUI_FORCE_BUILD=1 — local build path (developer)
    #   3. pnpm available — local build path (developer)
    #   4. fall back to downloading the release asset (end user, no Node)
    force_build = os.environ.get("CODEFYUI_FORCE_BUILD", "").strip() in ("1", "true", "yes")

    if DIST_INDEX.exists() and not force_build:
        section("Frontend: dist 已存在，略過", "Frontend: dist already exists, skipping")
    elif force_build or shutil.which("pnpm"):
        if not shutil.which("pnpm"):
            err("CODEFYUI_FORCE_BUILD=1 但找不到 pnpm",
                "CODEFYUI_FORCE_BUILD=1 but pnpm not found")
            sys.exit(1)
        section("Frontend: 安裝 node_modules", "Frontend: installing node_modules")
        run(["pnpm", "install"], cwd=FRONTEND_DIR)
        section("Frontend: 建置 dist", "Frontend: building dist")
        run(["pnpm", "build"], cwd=FRONTEND_DIR)
    else:
        section("Frontend: 未偵測到 pnpm，改下載 release dist",
                "Frontend: pnpm not found, downloading release dist instead")
        if not fetch_release_dist():
            err("無法取得 frontend dist", "cannot fetch frontend dist")
            print(
                t(
                    "\n  可選擇其一：\n"
                    "    1. 安裝 Node.js 24+ 與 pnpm 後重跑 cdui install\n"
                    "    2. 設定 CODEFYUI_RELEASE_TAG=<tag> 指定特定 release\n"
                    "    3. 檢查網路連線後重試",
                    "\n  Try one of:\n"
                    "    1. Install Node.js 24+ and pnpm, then re-run cdui install\n"
                    "    2. Set CODEFYUI_RELEASE_TAG=<tag> to pin a specific release\n"
                    "    3. Check your network and retry",
                ),
                file=sys.stderr,
            )
            sys.exit(1)

    _print_post_install_summary(gpu, dev)


def install_command() -> None:
    """Entry-point shim for `cdui install`: parse argv → resolve → install."""
    gpu, dev = _resolve_install_options(sys.argv[2:])
    install(gpu=gpu, dev=dev)


def update() -> None:
    """拉取 main branch 的最新版本並重新同步依賴。Accepts the same flags as install."""
    if not (ROOT / ".git").exists():
        err("此目錄不是 git clone，無法 update",
            "Not a git checkout — cannot update")
        sys.exit(1)
    # Decide whether this install will use a prebuilt release dist (no Node) or
    # build the frontend from source (pnpm available / forced). On the prebuilt
    # path we MUST pin the backend to the same release tag as the dist — pulling
    # `main` while fetching an older release's frontend leaves the SPA out of
    # sync with the API (e.g. it never bootstraps the session token, so every
    # mutating request is rejected 403 and the app "loads but doesn't work").
    force_build = os.environ.get("CODEFYUI_FORCE_BUILD", "").strip() in ("1", "true", "yes")
    will_build_from_source = force_build or bool(shutil.which("pnpm"))

    pinned_tag = None if will_build_from_source else _resolve_release_tag()

    if pinned_tag:
        section(f"切換至 release {pinned_tag}（前後端同版）",
                f"Checking out release {pinned_tag} (frontend/backend in sync)")
        run(["git", "fetch", "--tags", "origin"], cwd=ROOT)
        run(["git", "checkout", "-f", pinned_tag], cwd=ROOT)
        # install() reads this to fetch the matching dist.
        os.environ["CODEFYUI_RELEASE_TAG"] = pinned_tag
    else:
        section("拉取最新版本（main）", "Pulling latest (main)")
        # install.sh makes a *shallow* (`--depth 1`), tag-pinned clone, so the
        # local history is grafted and `main` can share no common ancestor with
        # the fetched tip — `git merge --ff-only origin/main` then dies with
        # "refusing to merge unrelated histories". An install dir is a
        # deployment, not a dev checkout, so just hard-realign `main` to the
        # fetched commit regardless of ancestry. `checkout -B` from FETCH_HEAD
        # works whether or not the branch existed / tracked upstream.
        run(["git", "fetch", "origin", "main", "--depth", "1"], cwd=ROOT)
        run(["git", "checkout", "-B", "main", "FETCH_HEAD"], cwd=ROOT)

    # Old dist is for the previous source — wipe it so install re-downloads
    # (or re-builds, when pnpm is on PATH) for the new code.
    if DIST_DIR.exists():
        section("移除舊 frontend/dist", "Removing stale frontend/dist")
        shutil.rmtree(DIST_DIR, ignore_errors=True)

    gpu, dev = _resolve_install_options(sys.argv[2:])
    install(gpu=gpu, dev=dev)


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


# ── Background server management ───────────────────────────────────────
# `cdui start` daemonizes by default so users can close the terminal and keep
# the server running, then manage it with `cdui status` / `cdui stop`. The PID
# + log live under the repo-local dev data dir alongside the session token.
SERVER_PIDFILE = DEV_USER_DATA_DIR / "server.pid"
SERVER_LOG = DEV_USER_DATA_DIR / "server.log"
SERVER_HEALTH_URL = "http://127.0.0.1:8000/api/health"


def _read_server_pid() -> "int | None":
    try:
        return int(SERVER_PIDFILE.read_text().strip())
    except (OSError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    """True if a process with *pid* currently exists."""
    if sys.platform == "win32":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True,
        )
        return str(pid) in out.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user
    return True


def _server_healthy(timeout: float = 1.0) -> bool:
    return _server_health_info(timeout) is not None


def _server_health_info(timeout: float = 1.0) -> "dict | None":
    """Fetch and parse /api/health. Returns the JSON dict, or None if the
    server isn't responding (or returned a non-200 / unparseable body)."""
    try:
        with urlopen(SERVER_HEALTH_URL, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            import json  # noqa: PLC0415 — only needed here
            return json.loads(resp.read().decode("utf-8", "replace"))
    except (URLError, HTTPError, TimeoutError, OSError, ValueError):
        return None


def _running_server_pid() -> "int | None":
    """Return the PID of the live background server, or None. Clears a stale
    pidfile as a side effect so callers don't act on a dead PID."""
    pid = _read_server_pid()
    if pid is None:
        return None
    if _pid_alive(pid):
        return pid
    # Stale pidfile (server crashed / was killed externally) — tidy up.
    SERVER_PIDFILE.unlink(missing_ok=True)
    return None


def start() -> None:
    """Production 模式：單一 uvicorn 由 FastAPI 直接 serve dist。

    預設背景執行（daemon）；加 --foreground / -f 改在前景執行。
    """
    if not DIST_INDEX.exists():
        print(
            "錯誤：找不到 frontend/dist/index.html\n"
            "  請執行 'cdui install'（下載 release dist）"
            " 或 'cdui build'（本地 build，需 pnpm）。",
            file=sys.stderr,
        )
        sys.exit(1)

    foreground = any(a in ("-f", "--foreground") for a in sys.argv[2:])

    existing = _running_server_pid()
    if existing is not None:
        print(f"CodefyUI 已在背景執行（PID {existing}）。")
        print("  查看狀態：cdui status    停止：cdui stop")
        return

    _warn_if_dist_stale()
    _apply_dev_env()
    uvicorn = _require_venv_tool("uvicorn")
    cmd = [uvicorn, "app.main:app", "--host", "127.0.0.1", "--port", "8000"]

    if foreground:
        print("=== CodefyUI 啟動（前景；Ctrl+C 停止）===")
        print("    開啟 → http://localhost:8000")
        print(f"    dev lockfile → {DEV_LOCKFILE}")
        print("")
        run(cmd, cwd=BACKEND_DIR)
        return

    # ── Background / daemon path ──────────────────────────────────────
    SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    logf = open(SERVER_LOG, "a", buffering=1)  # noqa: SIM115 — handed to child
    popen_kw: dict = {}
    if sys.platform == "win32":
        # New process group + detached so closing the console doesn't kill it.
        DETACHED_PROCESS = 0x00000008
        popen_kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS
    else:
        # New session → no controlling terminal, so SIGHUP on terminal close
        # doesn't reach the server.
        popen_kw["start_new_session"] = True

    proc = subprocess.Popen(
        cmd,
        cwd=BACKEND_DIR,
        stdout=logf,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        **popen_kw,
    )
    SERVER_PIDFILE.write_text(str(proc.pid))

    # Wait for the server to become healthy (or die) before reporting.
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            SERVER_PIDFILE.unlink(missing_ok=True)
            print("錯誤：伺服器啟動後隨即結束。最後的日誌：", file=sys.stderr)
            _print_log_tail(20)
            sys.exit(1)
        if _server_healthy():
            break
        time.sleep(0.5)
    else:
        print("警告：等候逾時，伺服器尚未回應健康檢查（仍在背景嘗試啟動）。")

    print("=== CodefyUI 已在背景啟動 ===")
    print(f"    PID         → {proc.pid}")
    print("    開啟        → http://localhost:8000")
    print(f"    日誌        → {SERVER_LOG}")
    print(f"    dev lockfile → {DEV_LOCKFILE}")
    print("")
    print("    管理：cdui status / cdui stop")


def _print_log_tail(n: int) -> None:
    try:
        lines = SERVER_LOG.read_text(errors="replace").splitlines()
        for ln in lines[-n:]:
            print("    " + ln, file=sys.stderr)
    except OSError:
        pass


# ── System status dashboard (`cdui status`) ───────────────────────────────
# A btop / k9s-style snapshot: host + OS, CPU (overall + per-core bars),
# memory, swap, disk, GPU (via nvidia-smi when present) and the top processes,
# followed by the CodefyUI server's own PID / health. Built on psutil when it's
# installed (it ships with the backend); degrades to a stdlib-only view when not.

def _human_bytes(n: "float | None") -> str:
    """Human-readable size, e.g. 1.5 GiB. Returns '—' for None."""
    if n is None:
        return "—"
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    size = float(n)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PiB"  # pragma: no cover — loop always returns first


def _pct_color(pct: float) -> str:
    """Green < 60% < yellow < 85% < red — the usual saturation gradient."""
    if pct >= 85:
        return RED
    if pct >= 60:
        return YELLOW
    return GREEN


def _bar(pct: "float | None", width: int = 24) -> str:
    """A coloured [████░░░░] usage bar. pct is 0–100; None renders empty."""
    if pct is None:
        return f"{GRAY}[{'░' * width}]{RESET}"
    pct = max(0.0, min(100.0, pct))
    filled = int(round(pct / 100 * width))
    color = _pct_color(pct)
    return f"{GRAY}[{color}{'█' * filled}{GRAY}{'░' * (width - filled)}{GRAY}]{RESET}"


def _fmt_uptime(seconds: float) -> str:
    secs = int(seconds)
    days, secs = divmod(secs, 86400)
    hours, secs = divmod(secs, 3600)
    mins, _ = divmod(secs, 60)
    if days:
        return t(f"{days} 天 {hours} 小時 {mins} 分", f"{days}d {hours}h {mins}m")
    if hours:
        return t(f"{hours} 小時 {mins} 分", f"{hours}h {mins}m")
    return t(f"{mins} 分", f"{mins}m")


def _kv(label: str, value: str) -> None:
    """Aligned `label  value` line; label padded to a fixed visual width."""
    pad = max(0, 14 - _display_width(label))
    print(f"  {DIM}{label}{RESET}{' ' * pad}  {value}")


def _gpu_stats() -> "list[dict]":
    """Per-GPU utilisation via `nvidia-smi` (fast, no torch import). Empty list
    when nvidia-smi is missing or errors (CPU-only / macOS / AMD machines)."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    gpus: list[dict] = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            gpus.append({
                "name": parts[0],
                "util": float(parts[1]),
                "mem_used": float(parts[2]) * 1024 * 1024,
                "mem_total": float(parts[3]) * 1024 * 1024,
                "temp": float(parts[4]),
            })
        except ValueError:
            continue
    return gpus


def _render_dashboard(interval: float, first: bool) -> None:
    """Print one frame of the status dashboard.

    *interval* is the psutil CPU sampling window (also the watch refresh gap);
    *first* gates a one-line hint that's pointless to repeat every frame.
    """
    try:
        import psutil  # noqa: PLC0415 — optional, ships with the backend
    except ImportError:
        psutil = None

    # Prime per-process CPU counters *before* the blocking CPU sample below so
    # the first read returns a real percentage rather than psutil's initial
    # 0.0. We hold the Process objects; the cpu_percent(interval=…) call in the
    # CPU section provides the sampling gap, then we read them back later.
    primed_procs: list = []
    if psutil is not None:
        for p in psutil.process_iter():
            try:
                p.cpu_percent(None)
                primed_procs.append(p)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    section("CodefyUI 系統狀態", "CodefyUI System Status")

    # ── Host / OS ─────────────────────────────────────────────────────────
    import platform  # noqa: PLC0415
    _kv(t("主機", "Host"), platform.node() or "—")
    _kv(t("作業系統", "OS"),
        f"{platform.system()} {platform.release()} ({platform.machine()})")
    if psutil is not None:
        try:
            _kv(t("開機時間", "Uptime"),
                _fmt_uptime(time.time() - psutil.boot_time()))
        except (OSError, AttributeError):
            pass
    if hasattr(os, "getloadavg"):
        try:
            la = os.getloadavg()
            _kv(t("負載平均", "Load avg"),
                f"{la[0]:.2f}  {la[1]:.2f}  {la[2]:.2f}")
        except OSError:
            pass

    # ── CPU ───────────────────────────────────────────────────────────────
    print()
    section("CPU", "CPU")
    cores = os.cpu_count() or 1
    if psutil is not None:
        overall = psutil.cpu_percent(interval=interval)
        per_core = psutil.cpu_percent(interval=None, percpu=True)
        _kv(t("總使用率", "Overall"),
            f"{_bar(overall)} {_pct_color(overall)}{overall:5.1f}%{RESET}"
            f"  {DIM}{cores} {t('核心', 'cores')}{RESET}")
        try:
            freq = psutil.cpu_freq()
            if freq and freq.current:
                _kv(t("時脈", "Freq"), f"{freq.current/1000:.2f} GHz")
        except (OSError, AttributeError):
            pass
        for i, cpct in enumerate(per_core):
            print(f"    {DIM}core {i:>2}{RESET} {_bar(cpct, 18)} "
                  f"{_pct_color(cpct)}{cpct:5.1f}%{RESET}")
    else:
        _kv(t("核心數", "Cores"), str(cores))
        print(f"    {DIM}{t('安裝 psutil 以顯示即時使用率', 'install psutil for live usage')}{RESET}")

    # ── Memory ────────────────────────────────────────────────────────────
    print()
    section("記憶體", "Memory")
    if psutil is not None:
        vm = psutil.virtual_memory()
        _kv("RAM",
            f"{_bar(vm.percent)} {_pct_color(vm.percent)}{vm.percent:5.1f}%{RESET}"
            f"  {_human_bytes(vm.used)} / {_human_bytes(vm.total)}")
        sm = psutil.swap_memory()
        if sm.total:
            _kv("Swap",
                f"{_bar(sm.percent)} {_pct_color(sm.percent)}{sm.percent:5.1f}%{RESET}"
                f"  {_human_bytes(sm.used)} / {_human_bytes(sm.total)}")
    else:
        print(f"    {DIM}{t('安裝 psutil 以顯示記憶體用量', 'install psutil for memory usage')}{RESET}")

    # ── Disk ──────────────────────────────────────────────────────────────
    print()
    section("磁碟", "Disk")
    root = "C:\\" if sys.platform == "win32" else "/"
    try:
        du = shutil.disk_usage(root)
        pct = du.used / du.total * 100 if du.total else 0.0
        _kv(root,
            f"{_bar(pct)} {_pct_color(pct)}{pct:5.1f}%{RESET}"
            f"  {_human_bytes(du.used)} / {_human_bytes(du.total)}"
            f"  ({_human_bytes(du.free)} {t('可用', 'free')})")
    except OSError:
        pass

    # ── GPU ───────────────────────────────────────────────────────────────
    gpus = _gpu_stats()
    if gpus:
        print()
        section("GPU", "GPU")
        for i, g in enumerate(gpus):
            mem_pct = g["mem_used"] / g["mem_total"] * 100 if g["mem_total"] else 0.0
            _kv(f"GPU {i}", f"{g['name']}  {g['temp']:.0f}°C")
            print(f"    {DIM}util {RESET}{_bar(g['util'], 18)} "
                  f"{_pct_color(g['util'])}{g['util']:5.1f}%{RESET}")
            print(f"    {DIM}vram {RESET}{_bar(mem_pct, 18)} "
                  f"{_pct_color(mem_pct)}{mem_pct:5.1f}%{RESET}  "
                  f"{_human_bytes(g['mem_used'])} / {_human_bytes(g['mem_total'])}")

    # ── Top processes ─────────────────────────────────────────────────────
    if psutil is not None:
        print()
        section("行程（依 CPU 排序）", "Top processes (by CPU)")
        # psutil reports per-process CPU% relative to a single core, so a busy
        # core can read >100%; normalise by core count for a system-wide view.
        cores = os.cpu_count() or 1
        procs = []
        for p in primed_procs:
            try:
                procs.append({
                    "pid": p.pid,
                    "name": p.name(),
                    "cpu": p.cpu_percent(None) / cores,
                    "mem": p.memory_percent(),
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        procs.sort(key=lambda x: x["cpu"], reverse=True)
        print(f"    {DIM}{'PID':>7}  {'CPU%':>6}  {'MEM%':>6}  {t('名稱', 'NAME')}{RESET}")
        for info in procs[:8]:
            name = (info["name"] or "?")[:28]
            cpu, mem = info["cpu"], info["mem"]
            print(f"    {info['pid']:>7}  "
                  f"{_pct_color(cpu)}{cpu:6.1f}{RESET}  {mem:6.1f}  {name}")

    # ── CodefyUI server ───────────────────────────────────────────────────
    print()
    section("CodefyUI 伺服器", "CodefyUI Server")
    pid = _running_server_pid()
    info = _server_health_info()
    healthy = info is not None
    if pid is not None:
        _kv(t("狀態", "State"),
            f"{GREEN}● {t('背景執行中', 'running (background)')}{RESET}  PID {pid}")
        _kv(t("健康檢查", "Health"),
            f"{GREEN}✓ {t('正常', 'ok')}{RESET}" if healthy
            else f"{RED}✗ {t('尚未回應', 'not responding')}{RESET}")
        if info:
            _kv(t("節點 / 預設", "Nodes / Presets"),
                f"{info.get('nodes_loaded', '?')} / {info.get('presets_loaded', '?')}")
        _kv("URL", "http://localhost:8000")
        _kv(t("日誌", "Log"), str(SERVER_LOG))
    elif healthy:
        orphan = t("有伺服器回應，但非 cdui 背景啟動（無 PID 檔）",
                   "responding, but not a cdui background server (no PID file)")
        _kv(t("狀態", "State"), f"{YELLOW}● {orphan}{RESET}")
        _kv("URL", "http://localhost:8000")
        if info:
            _kv(t("節點 / 預設", "Nodes / Presets"),
                f"{info.get('nodes_loaded', '?')} / {info.get('presets_loaded', '?')}")
    else:
        _kv(t("狀態", "State"),
            f"{GRAY}○ {t('未執行', 'not running')}{RESET}  "
            f"{DIM}{t('用 cdui start 啟動', 'start with: cdui start')}{RESET}")

    if first and _watch_disabled():
        tip = t("提示：直接執行 cdui status 會持續刷新（像 btop）",
                "tip: plain `cdui status` refreshes live (like btop)")
        print()
        print(f"  {DIM}{tip}{RESET}")


def _watch_disabled() -> bool:
    """True when we must print a single frame instead of looping: an explicit
    --once, or a non-interactive stdout (pipe / CI) where a clearing loop and
    its never-returning exit code would be useless or harmful."""
    if any(a in ("-1", "--once") for a in sys.argv[2:]):
        return True
    return not sys.stdout.isatty()


def _continuous_default() -> bool:
    """Whether `cdui status` should loop. Continuous is the default; only an
    explicit --once or a non-TTY stdout falls back to a single frame. An
    explicit --watch / -w forces the loop even past those (e.g. for testing)."""
    if any(a in ("-w", "--watch") for a in sys.argv[2:]):
        return True
    return not _watch_disabled()


def _parse_watch_interval() -> float:
    """Read the optional numeric refresh interval (default 2.0s).

    Accepts it after --watch / -w, or as a bare positional number so plain
    `cdui status 1` works: `cdui status`, `cdui status 1`, `cdui status -w 0.5`.
    """
    argv = sys.argv[2:]
    for i, a in enumerate(argv):
        if a in ("-w", "--watch"):
            if i + 1 < len(argv):
                try:
                    return max(0.5, float(argv[i + 1]))
                except ValueError:
                    pass
            return 2.0
    # Bare positional number, e.g. `cdui status 1`.
    for a in argv:
        if not a.startswith("-"):
            try:
                return max(0.5, float(a))
            except ValueError:
                continue
    return 2.0


def status() -> None:
    """系統與伺服器狀態儀表板（btop / k9s 風格，預設持續刷新）。"""
    if not _continuous_default():
        # Single frame (--once, or stdout isn't a TTY). Use a short CPU
        # sampling window so the reading is real (psutil's first non-blocking
        # call always returns 0.0).
        _render_dashboard(interval=0.3, first=True)
        # Mirror the old contract: exit non-zero when nothing is serving :8000,
        # so scripts can still gate on `cdui status`.
        if _running_server_pid() is None and not _server_healthy():
            sys.exit(1)
        return

    interval = _parse_watch_interval()
    _watch_loop(interval)


def _render_frame_text(interval: float, first: bool) -> str:
    """Render one dashboard frame into a string (incl. the header line) by
    temporarily redirecting stdout. Lets the watch loop repaint atomically."""
    import io  # noqa: PLC0415
    buf = io.StringIO()
    real = sys.stdout
    sys.stdout = buf
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"{DIM}{t('刷新間隔', 'refresh')} {interval:g}s · {ts} · "
              f"{t('按 Ctrl+C 離開', 'Ctrl+C to quit')}{RESET}")
        _render_dashboard(interval=interval, first=first)
    finally:
        sys.stdout = real
    return buf.getvalue()


def _watch_loop(interval: float) -> None:
    """btop-style live refresh without the full-screen-clear flicker.

    Each frame is rendered into a buffer, then painted by homing the cursor
    (``\\x1b[H``) and overwriting line by line — each line cleared to its end
    (``\\x1b[K``) so leftover characters from a longer previous frame vanish —
    and finally erasing anything below (``\\x1b[J``). The screen is only fully
    cleared once, up front, so there's never a blank flash between frames.
    """
    hide = "\x1b[?25l" if USE_COLOR else ""
    showp = "\x1b[?25h" if USE_COLOR else ""
    try:
        if USE_COLOR:
            sys.stdout.write(hide + "\x1b[2J\x1b[H")
            sys.stdout.flush()
        first = True
        while True:
            frame = _render_frame_text(interval, first)
            first = False
            if USE_COLOR:
                lines = frame.split("\n")
                # Home, then overwrite each line (clearing trailing leftovers),
                # then clear everything below the shorter-or-equal new frame.
                painted = "\x1b[H" + "\x1b[K\n".join(lines) + "\x1b[J"
                sys.stdout.write(painted)
            else:
                sys.stdout.write(frame)
            sys.stdout.flush()
            # When psutil is absent there's no blocking cpu sample, so the loop
            # would spin hot — pace it ourselves in that case.
            if not _has_psutil():
                time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        if showp:
            sys.stdout.write(showp)
            sys.stdout.flush()


def _has_psutil() -> bool:
    try:
        import psutil  # noqa: F401, PLC0415
        return True
    except ImportError:
        return False


def dev() -> None:
    if not shutil.which("pnpm"):
        print(
            "錯誤：dev 模式需要 pnpm（HMR）。請安裝 Node.js 24+ 與 pnpm，\n"
            "  或改用 'cdui start' 跑 production 模式（不需 Node）。",
            file=sys.stderr,
        )
        sys.exit(1)
    _install_frontend_deps_if_needed()
    _apply_dev_env()

    uvicorn = _require_venv_tool("uvicorn")
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
    # First, stop the tracked background server gracefully via its PID. On
    # POSIX it was started with start_new_session, so its PID is also its
    # process-group leader — kill the whole group to catch any children.
    pid = _read_server_pid()
    if pid is not None and _pid_alive(pid):
        print(f"  停止背景伺服器（PID {pid}）...")
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                           capture_output=True)
        else:
            _terminate_posix(pid)
    SERVER_PIDFILE.unlink(missing_ok=True)

    # Sweep up anything else (foreground starts, dev-mode vite, stray workers).
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/IM", "uvicorn.exe"], capture_output=True)
        subprocess.run(["taskkill", "/F", "/FI", "WINDOWTITLE eq vite*"], capture_output=True)
    else:
        subprocess.run(["pkill", "-f", "uvicorn app.main:app"], capture_output=True)
        subprocess.run(["pkill", "-f", "vite"], capture_output=True)
    print("=== 完成 ===")


def _terminate_posix(pid: int) -> None:
    """SIGTERM the process group, then SIGKILL anything still alive."""
    import signal  # noqa: PLC0415 — only needed here, POSIX only

    def _signal_group(sig: int) -> None:
        try:
            os.killpg(os.getpgid(pid), sig)
        except ProcessLookupError:
            pass
        except OSError:
            # Couldn't resolve/kill the group — fall back to the bare PID.
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass

    _signal_group(signal.SIGTERM)
    for _ in range(20):  # up to ~2s for a graceful shutdown
        if not _pid_alive(pid):
            return
        time.sleep(0.1)
    _signal_group(signal.SIGKILL)


def test() -> None:
    pytest = _require_venv_tool("pytest")
    run([pytest], cwd=BACKEND_DIR)


    # No bulk `dev-install` shortcut: official packs are opt-in. Contributors
    # decide per-chapter what they need and run ``plugin install`` themselves
    # (matches what an end user would do via the global ``cdui plugin``).


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
    "status": status,
    "stop": stop,
    "test": test,
    "clean": clean,
    "uninstall": uninstall,
}

# Commands that mutate or remove the venv must run from the outer interpreter,
# never from the venv's Python (Windows can't delete a running exe; update
# rewrites deps in-place).
_SKIP_VENV_EXEC = {"install", "update", "clean", "uninstall"}


def _dispatch_plugin_subcommand() -> int:
    """Hand off ``cdui plugin <subcmd> ...`` to scripts/plugins.py.

    The plugin CLI imports ``app.core.plugin_loader`` and ``platformdirs`` —
    both require the codefyui venv, so we must be running inside it before
    delegating. ``_exec_into_venv_if_available()`` is the same hop the
    other top-level commands take.
    """
    _exec_into_venv_if_available()
    _ensure_uv()
    _apply_dev_env()

    # scripts/ is not normally on sys.path when dev.py is invoked directly,
    # so bootstrap it before importing the sibling module.
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    import plugins as plugin_cli  # noqa: PLC0415 — late import: needs venv
    return plugin_cli.main(sys.argv[2:])


if __name__ == "__main__":
    # Long-form sub-grouped commands come first.
    if len(sys.argv) >= 2 and sys.argv[1] == "plugin":
        sys.exit(_dispatch_plugin_subcommand())

    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    if sys.argv[1] not in _SKIP_VENV_EXEC:
        _exec_into_venv_if_available()
    _ensure_uv()
    COMMANDS[sys.argv[1]]()
