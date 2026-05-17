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
    dev-install 把六個官方 chapter pack (c1–c6) 裝到 repo 內 .codefyui_dev/
                — 給專案內開發者用，與全域 cdui plugin 完全隔離
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
    CODEFYUI_USER_DATA_DIR  覆蓋 platformdirs user-data 位置（plugin lockfile
                            + session.token），dev-install 會自動設定到
                            <repo>/.codefyui_dev/
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
        loc = (locale.getdefaultlocale()[0] or "").lower()
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
    print(f"{YELLOW}⚠ {t(zh, en)}{RESET}", file=sys.stderr)


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


def _apply_dev_env_if_active() -> None:
    """Point the server at the repo-local plugin lockfile when it exists.

    The user explicitly opts into dev-mode plugins by running
    ``cdui dev-install`` once — that creates ``DEV_LOCKFILE``. From then on,
    every ``cdui start`` / ``cdui dev`` from this checkout uses that lockfile
    instead of the global one in user-data-dir. Other dev clones on the same
    machine stay isolated.
    """
    if DEV_LOCKFILE.exists():
        os.environ.setdefault("CODEFYUI_USER_DATA_DIR", str(DEV_USER_DATA_DIR))

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
    section("拉取最新版本（main）", "Pulling latest (main)")
    # Explicit remote/branch so the command works even on a detached HEAD or
    # a branch that doesn't track upstream.
    run(["git", "fetch", "origin", "main"], cwd=ROOT)
    run(["git", "checkout", "main"], cwd=ROOT)
    run(["git", "merge", "--ff-only", "origin/main"], cwd=ROOT)

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
    _apply_dev_env_if_active()
    uvicorn = _require_venv_tool("uvicorn")
    print("=== CodefyUI 啟動（Ctrl+C 停止）===")
    print("    開啟 → http://localhost:8000")
    if os.environ.get("CODEFYUI_USER_DATA_DIR"):
        print(f"    dev plugins → {DEV_USER_DATA_DIR}")
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
    _apply_dev_env_if_active()

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
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/IM", "uvicorn.exe"], capture_output=True)
        subprocess.run(["taskkill", "/F", "/FI", "WINDOWTITLE eq vite*"], capture_output=True)
    else:
        subprocess.run(["pkill", "-f", "uvicorn app.main:app"], capture_output=True)
        subprocess.run(["pkill", "-f", "vite"], capture_output=True)
    print("=== 完成 ===")


def test() -> None:
    pytest = _require_venv_tool("pytest")
    run([pytest], cwd=BACKEND_DIR)


_DEV_BUILTIN_PACKS = ("c1", "c2", "c3", "c4", "c5", "c6")


def dev_install() -> None:
    """Install the six built-in chapter packs into a repo-local lockfile.

    Designed for contributors working **inside the cloned repo** — different
    from the global ``cdui plugin install`` which writes the lockfile to
    ``%LOCALAPPDATA%\\codefyui\\plugins\\`` (shared across every clone).

    Effect:
        1. Creates ``./.codefyui_dev/plugins/`` in the repo (gitignored).
        2. Sets ``CODEFYUI_USER_DATA_DIR`` for the install + every subsequent
           ``cdui start`` / ``cdui dev`` from this checkout.
        3. Activates c1, c2, c3, c4, c5, c6 via the existing plugin install
           machinery (lockfile-only — no file copy for builtin sources).

    Idempotent: re-running just refreshes the lockfile timestamp.
    """
    _exec_into_venv_if_available()
    _ensure_uv()

    os.environ["CODEFYUI_USER_DATA_DIR"] = str(DEV_USER_DATA_DIR)
    DEV_USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DEV_USER_DATA_DIR / "plugins").mkdir(parents=True, exist_ok=True)

    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    import plugins as plugin_cli  # noqa: PLC0415 — late import needs venv

    print("=== Dev plugin install — repo-local lockfile ===")
    print(f"    target → {DEV_LOCKFILE}")
    print(f"    packs  → {' '.join(_DEV_BUILTIN_PACKS)}")
    print("")

    rc = plugin_cli.main(["install", *_DEV_BUILTIN_PACKS, "--no-confirm"])
    if rc != 0:
        print("=== 安裝失敗（rc={}）===".format(rc), file=sys.stderr)
        sys.exit(rc)

    print("")
    print("=== 完成 — 下一步 ===")
    print("    cdui start       # 用 dev lockfile 啟 server")
    print("    cdui dev         # 同上 + Vite HMR")
    print("    cdui plugin list # 確認列出 c1–c6")


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
    "dev-install": dev_install,
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


def _dispatch_plugin_subcommand() -> int:
    """Hand off ``cdui plugin <subcmd> ...`` to scripts/plugins.py.

    The plugin CLI imports ``app.core.plugin_loader`` and ``platformdirs`` —
    both require the codefyui venv, so we must be running inside it before
    delegating. ``_exec_into_venv_if_available()`` is the same hop the
    other top-level commands take.
    """
    _exec_into_venv_if_available()
    _ensure_uv()
    _apply_dev_env_if_active()

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
