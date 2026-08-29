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
                伺服器還在執行時會拒絕（會刪掉它正在服務的 dist），
                請先 cdui stop。
    build       建置 frontend dist（需 Node + pnpm，給開發者）
    dev         啟動開發伺服器（HMR；需 Node + pnpm）
    start       啟動 production（單一 uvicorn，用 frontend/dist；不需 Node）
                預設在背景執行（關掉 terminal 也會繼續跑），用 cdui status /
                cdui stop 管理。加 --foreground / -f 則在前景執行（Ctrl+C 停止）。
                旗標：--host <addr>   綁定位址（預設 127.0.0.1）。0.0.0.0 或
                                      區網 IP 可讓其他裝置存取 — 任何能連到該埠
                                      的人都能控制此實例，只在信任的網路使用。
                      --port <n>      埠號（預設 8000）
                      --project <dir> 指定專案目錄（codefyui.project.toml）
                      --              其後的參數原樣轉給 uvicorn，例如
                                      cdui start -- --proxy-headers
                                        --forwarded-allow-ips 127.0.0.1
                                      放在反向代理後面時，還必須設定環境變數
                                      CODEFYUI_EXTRA_ALLOWED_HOSTS 為對外主機名，
                                      否則每個請求（含網頁本身）都會回 421。
                                      詳見文件的 Deployment 頁。
    status      顯示系統與伺服器狀態儀表板（像 btop / k9s：CPU、記憶體、
                磁碟、GPU、行程、伺服器 PID 與健康檢查）
                預設持續刷新（每 2 秒，Ctrl+C 離開）；輸出被導向管線或非互動
                環境時自動改為只輸出一次。
                旗標：[秒] 或 -w [秒]    自訂刷新間隔（如 cdui status 1）
                      --once / -1       只輸出一次
    run         把圖檔送到執行中的伺服器排隊執行（server-owned run）
                每個裝置一條 FIFO 佇列；關掉 terminal 也不會中斷。
                用法：cdui run <graph.json> [旗標]
                旗標：--name <字串>     Runs 面板顯示的名稱
                      --device <裝置>   cpu | auto | cuda | cuda:N | mps
                                        （預設 auto，目前會解析成 cpu；
                                        解析後的裝置就是佇列 key）
                      --seed <n>        隨機種子
                      --record-outputs  保留節點輸出供事後檢視
                      --wait            串流進度直到結束（預設）
                      --detach          只印出 run id 就離開
                      --timeout <秒>    等待上限（0 = 不限；逾時後 run 仍繼續）
                      --host / --port   伺服器位址（預設沿用上次 start 的）
                離開碼：0 成功、1 失敗/取消/無法送出、2 參數錯誤、130 Ctrl+C
                （Ctrl+C 只是停止跟隨，run 仍在伺服器上繼續執行）。
                離線、不需要伺服器的單次執行請用 backend/run_graph.py。
    stop        停止「這個安裝」的服務：pidfile 記錄的背景伺服器，加上從這個
                目錄啟動的殘留行程（前景 start、dev 模式的 vite、遺留 worker）。
                旗標：--all   改為停止整台機器上所有 CodefyUI 與 Vite 行程。
                              會波及其他使用者、以及與 CodefyUI 無關的 Vite
                              dev server；共用主機請勿使用。
    test        執行整個專案的測試：backend（pytest）+ frontend（vitest）
                沒有 pnpm 時會略過前端測試並在結果表明講「略過」，不會失敗
                （release 安裝本來就不需要 Node）。兩邊都會跑完才回報，
                任一邊失敗就以離開碼 1 結束。
                旗標：--backend    只跑後端
                      --frontend   只跑前端
    clean       移除虛擬環境、node_modules 與 frontend/dist
    uninstall   解除安裝：clean + 移除全域 cdui launcher

    plugin <subcmd> ...
                與 cdui plugin 完全相同的介面，但 lockfile 寫到 repo 內的
                <repo>/.codefyui_dev/plugins/ 而不是 %LOCALAPPDATA%\\codefyui，
                讓多個 dev clone 互不干擾。官方 foundations/deep/rl direction
                pack 預設不會安裝，需要時逐一裝即可。範例：
                    python scripts/dev.py plugin install deep
                    python scripts/dev.py plugin install owner/repo@main
                    python scripts/dev.py plugin sync --dry-run  # 列出尚未決定的內建包
                    python scripts/dev.py plugin sync            # 全部安裝（會先確認）
                    python scripts/dev.py plugin list
                    python scripts/dev.py plugin enable deep     # 啟用
                    python scripts/dev.py plugin disable deep    # 停用（檔案保留）
                    python scripts/dev.py plugin uninstall deep  # 從 lockfile 移除
                                                                # （sync 之後不會再裝回）

    packs <subcmd> ...
                套件中心（Package Center）的終端機介面：安裝選用的模型與套件。
                只能安裝 catalog 內建清單裡的項目，不接受 pip spec 或網址。
                    cdui packs list                      # 列出所有套件與狀態
                    cdui packs status                    # 加上 torch 版本與建議指令
                    cdui packs install sentence-embeddings --yes
                    cdui packs install sentence-embeddings --items all-MiniLM-L6-v2
                    cdui packs remove word-vectors glove-50d
                離開碼：0 成功、1 失敗、2 拒絕執行（id 錯誤、相依未裝、無法確認）、
                3 需要重啟伺服器（會印出該執行的指令）、130 Ctrl+C 取消。

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
import re
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

RELEASE_REPO = "CodefyUI/CodefyUI"
RELEASE_ASSET = "frontend-dist.tar.gz"


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def _has_console_window() -> bool:
    """Windows: does this process own a console window?

    A `cdui` run typed at a terminal does. The restart helper does not: it is
    spawned DETACHED by a server that is about to exit, precisely so that
    closing the console it came from cannot take it with it.

    Never raises, and an unanswerable probe reads as "there is one" -- that
    is the behaviour every `cdui` run had before this question was asked, and
    the cost of the wrong answer is a console window rather than a launcher
    that no longer works.
    """
    if sys.platform != "win32":
        return True
    try:
        import ctypes  # noqa: PLC0415 — Windows-only, and only on this path
        return bool(ctypes.windll.kernel32.GetConsoleWindow())
    except (AttributeError, OSError, ValueError):
        return True


def _reexec(executable: str, argv: list) -> None:
    """Replace the current process with ``executable argv...`` (cross-platform).

    POSIX ``os.execv`` is a true in-place replace: the launching shell keeps
    waiting on the same PID and the console (stdin/stdout) stays attached, so an
    interactive ``input()`` later in the run still reads the terminal.

    Windows has no real ``exec`` — there ``os.execv`` *spawns* a new process and
    immediately exits the current one. The shell waiting on us (``cdui.cmd`` ->
    python) then sees its child exit and returns to its prompt, while the
    re-exec'd child runs orphaned and races the shell for the console. Any later
    ``input()`` in the child reads EOF — which is exactly why
    ``cdui plugin install <owner/repo>`` silently "cancelled" at the [y/N]
    prompt and dropped back to the command line. So on Windows we run the child
    synchronously and forward its exit code: one clean chain, one owner of the
    console at a time.

    The child is handed THIS process's own stdio, because a Windows child
    given none does not inherit a console-less parent's handles -- it
    attaches to a new console instead. The parent that has none is the
    restart helper, whose stdout IS the job log: without this, every line the
    relaunched ``start()`` prints (the reach lines, "the server exited right
    after start", the log tail explaining why) goes to a window that closes
    with it, and the log a person opens holds the helper's fourteen lines and
    nothing else. ``CREATE_NO_WINDOW`` goes with it in that case, and ONLY in
    that case: an interactive run owns its console, and taking it away is how
    the ``[y/N]`` bug above comes back.
    """
    if sys.platform == "win32":
        forwarded: dict = {"stdin": sys.stdin, "stdout": sys.stdout,
                           "stderr": sys.stderr}
        if not _has_console_window():
            forwarded["creationflags"] = subprocess.CREATE_NO_WINDOW
        sys.exit(subprocess.run([executable, *argv], **forwarded).returncode)
    os.execv(executable, [executable, *argv])


#: Where the interpreter that launched a `cdui` run is written down, before
#: `_exec_into_venv_if_available` replaces this process with the venv's
#: python. `cdui start` hands it to the server as half of CODEFYUI_LAUNCHER,
#: and the restart helper is started with it: a restart-mode install
#: REWRITES backend/.venv, and on Windows the interpreter doing the
#: rewriting must not be the one running out of the directory being
#: rewritten.
OUTER_PYTHON_ENV = "CODEFYUI_OUTER_PYTHON"


def _outer_python() -> str:
    """The interpreter to start a fresh `cdui` run with.

    NOT `sys.executable` in most commands: `start` is not in
    `_SKIP_VENV_EXEC`, so by the time it runs `_exec_into_venv_if_available`
    has already re-exec'd this process as backend/.venv's python and
    `sys.executable` is that. The outer one is recorded on the way through
    and read back here.

    Falls back to `sys.executable` when nothing was recorded (somebody ran
    the venv's python directly, so there IS no outer interpreter — the
    honest answer, and the best one available) or when what was recorded is
    no longer a file. A launcher that does not exist is the worse failure of
    the two: `restart.restart_available()` refuses it outright and the
    Package Center offers no restart at all.
    """
    recorded = os.environ.get(OUTER_PYTHON_ENV)
    if recorded and Path(recorded).is_file():
        return recorded
    return sys.executable


def _exec_into_venv_if_available() -> None:
    """Re-exec into backend/.venv's Python when it exists.

    Lets `python dev.py <cmd>` work transparently with any outer interpreter
    (uv-managed, system, or a temp env) — we hand off to the venv's Python so
    subprocess calls run against the installed deps.

    Records the outer interpreter FIRST (see `OUTER_PYTHON_ENV`). After the
    hop it is unrecoverable, and a restart-mode install needs it. `setdefault`
    rather than assignment because the re-exec'd child runs this same function
    again, from inside the venv, and would otherwise overwrite the answer with
    the one value it must never be.
    """
    os.environ.setdefault(OUTER_PYTHON_ENV, sys.executable)
    if not VENV_PY.exists():
        return
    # Are we already running *inside* this venv? Discriminate on sys.prefix
    # (the venv root), NOT the executable path. `uv venv` symlinks
    # .venv/bin/python straight to the uv-managed base interpreter, so
    # Path(sys.executable).resolve() collapses the venv python and that base
    # interpreter to the *same* real binary — a genuine outer interpreter then
    # compares equal to VENV_PY and the hop is wrongly skipped, leaving the run
    # on the outer interpreter where `app` is not importable (cdui plugin <cmd>
    # then dies with "ModuleNotFoundError: No module named 'app'"). sys.prefix
    # points at the venv only when its python is actually the running one.
    try:
        if Path(sys.prefix).resolve() == VENV.resolve():
            return
    except OSError:
        return
    _reexec(str(VENV_PY), sys.argv)


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


def _uv_install_timeout() -> int:
    """Seconds to allow for the uv bootstrap download. 0 disables the limit."""
    raw = os.environ.get("CODEFYUI_UV_INSTALL_TIMEOUT", "").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 180


def _ensure_uv() -> None:
    if shutil.which("uv"):
        return
    # This runs before EVERY command, so an unbounded download here wedges
    # `cdui status`, `cdui stop`, everything. School and lab networks commonly
    # drop packets rather than refusing them, which is indistinguishable from
    # a slow link until something imposes a deadline.
    timeout = _uv_install_timeout()
    limit = t(f"（最多等待 {timeout} 秒）", f" (waiting up to {timeout}s)") if timeout else ""
    print(t(
        f"=== uv 未安裝，正在從 https://astral.sh 自動安裝{limit} ===",
        f"=== uv is not installed; downloading it from https://astral.sh{limit} ===",
    ))
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["powershell", "-c", "irm https://astral.sh/uv/install.ps1 | iex"],
                check=True,
                timeout=timeout or None,
            )
        else:
            # curl's own deadlines matter as well as the outer one: without
            # them a black-holed connection sits in connect() and the process
            # is only ever torn down from outside.
            connect = min(30, timeout) if timeout else 30
            max_time = f" --max-time {timeout}" if timeout else ""
            subprocess.run(
                f"curl -LsSf --connect-timeout {connect}{max_time}"
                " https://astral.sh/uv/install.sh | sh",
                shell=True,
                check=True,
                timeout=timeout or None,
            )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as exc:
        timed_out = isinstance(exc, subprocess.TimeoutExpired)
        why = t(
            f"逾時（超過 {timeout} 秒）" if timed_out else "失敗",
            f"timed out after {timeout}s" if timed_out else "failed",
        )
        print(t(
            f"\n錯誤：自動安裝 uv {why}。\n"
            f"  CodefyUI 需要 uv 才能執行任何指令。若這台機器連不到網際網路\n"
            f"  （教室或實驗室網路常見），請改用下列任一方式：\n"
            f"    1. 自行安裝 uv 後重試：https://docs.astral.sh/uv/getting-started/installation/\n"
            f"    2. 從別處下載 uv，放進 PATH 之後再執行同一個指令\n"
            f"    3. 網路很慢但可用時，加大逾時：CODEFYUI_UV_INSTALL_TIMEOUT=600\n",
            f"\nError: installing uv {why}.\n"
            f"  CodefyUI needs uv before it can run any command. If this machine\n"
            f"  cannot reach the internet (common on school and lab networks),\n"
            f"  use one of these instead:\n"
            f"    1. Install uv yourself, then retry:\n"
            f"       https://docs.astral.sh/uv/getting-started/installation/\n"
            f"    2. Copy a uv binary onto this machine, put it on PATH, rerun the command\n"
            f"    3. On a slow but working link, raise the limit:\n"
            f"       CODEFYUI_UV_INSTALL_TIMEOUT=600\n",
        ), file=sys.stderr)
        sys.exit(1)
    # 安裝後重新啟動自身，讓新 PATH 生效
    _reexec(sys.executable, sys.argv)


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


# ── Dist build stamp ──────────────────────────────────────────────────────────
# frontend/dist/build-info.json records which commit/tag the dist was built
# from. Schema is shared with the "Stamp dist with build provenance" step in
# .github/workflows/release-build.yml — keep both writers in sync:
#   {"tag": str|null, "commit": str|null, "built_at": iso8601, "source": str}


# git emits UTF-8 (e.g. non-ASCII paths with core.quotepath=false); decoding
# with the locale codepage (cp950/cp1252) would crash the reader thread, so
# force utf-8 with replacement — a mangled char only degrades a display string.
_GIT_TEXT_KW: dict = {"encoding": "utf-8", "errors": "replace"}

# The same defence, for a different family of children: the Windows console
# tools. `tasklist`, `taskkill` and friends write their STATUS messages in the
# console code page AND translate them -- on a zh-TW box a pid that no longer
# exists answers `tasklist` with a Chinese "no tasks match" line in cp950.
#
# What makes that a crash rather than mojibake is where the decode happens.
# With a bare `text=True` Python decodes on subprocess's reader THREAD, using
# `locale.getencoding()` -- which `PYTHONUTF8=1` (set by every install path
# here) has already forced to utf-8. The UnicodeDecodeError is raised in that
# thread, printed as "Exception in thread Thread-1 (_readerthread)", and
# swallowed; `communicate` then hands the caller `stdout=None`. `cdui start`
# died exactly there, on `str(pid) in out.stdout`, every time a restart-mode
# install left a stale pidfile behind.
#
# So: decode as utf-8 and REPLACE what does not fit. Everything these callers
# actually read out of the output is ASCII -- a pid, a GPU name, a number --
# so a replacement character in a message nobody parses costs nothing, and
# `�` beats an exception on another thread. Callers still guard with
# `(out.stdout or "")`: a child that dies before writing anything gives None
# too, and that has nothing to do with encodings.
_CONSOLE_TEXT_KW: dict = {"text": True, "encoding": "utf-8", "errors": "replace"}


def _git_head_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT, capture_output=True, timeout=5, **_GIT_TEXT_KW,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = (out.stdout or "").strip()
    return commit if out.returncode == 0 and commit else None


def _git_exact_tag() -> str | None:
    try:
        out = subprocess.run(
            ["git", "describe", "--tags", "--exact-match"],
            cwd=ROOT, capture_output=True, timeout=5, **_GIT_TEXT_KW,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    tag = (out.stdout or "").strip()
    return tag if out.returncode == 0 and tag else None


def _git_frontend_src_dirty() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--", "frontend/src"],
            cwd=ROOT, capture_output=True, timeout=5, **_GIT_TEXT_KW,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or out.stdout is None:
        return None
    return any(line.strip() for line in out.stdout.splitlines())


def _git_frontend_unchanged_since(commit: str) -> bool | None:
    """Whether tracked frontend/ files are identical between `commit` and HEAD.

    True/False when git can prove it; None when undecidable (e.g. a shallow
    clone that no longer has the stamped commit). Only exit codes matter here.
    """
    try:
        out = subprocess.run(
            ["git", "diff", "--quiet", commit, "HEAD", "--", "frontend"],
            cwd=ROOT, capture_output=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode == 0:
        return True
    if out.returncode == 1:
        return False
    return None


def _read_build_stamp() -> dict | None:
    try:
        stamp = json.loads((DIST_DIR / "build-info.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(stamp, dict):
        return None
    if not isinstance(stamp.get("commit"), (str, type(None))):
        return None  # foreign schema — a non-string commit would crash the [:12] display
    return stamp


def _write_build_stamp(source: str) -> None:
    """Best-effort: a failed stamp must never fail the build itself."""
    from datetime import datetime, timezone

    stamp = {
        "tag": _git_exact_tag(),
        "commit": _git_head_commit(),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
    }
    try:
        (DIST_DIR / "build-info.json").write_text(
            json.dumps(stamp, indent=2) + "\n", encoding="utf-8"
        )
    except OSError:
        pass


def _warn_if_dist_stale() -> None:
    """Warn when frontend/dist does not correspond to the checked-out code.

    Release installs clone the full repo (frontend/src included) and unpack a
    prebuilt dist whose mtimes predate the checkout, so mtimes alone cannot
    tell "stale" from "freshly installed". Trust the build stamp first: a dist
    stamped with the current HEAD on a clean checkout is in sync. The mtime
    heuristic remains only for dirty or unstamped developer trees.
    """
    src = FRONTEND_DIR / "src"
    if not src.is_dir():
        return
    try:
        dist_mtime = DIST_INDEX.stat().st_mtime
    except OSError:
        return

    if shutil.which("pnpm"):
        advice = t(
            "若想看到最新前端，請先執行 'cdui build' 重新打包。",
            "Run 'cdui build' to rebuild the frontend.",
        )
    else:
        advice = t(
            "執行 'cdui update' 重新下載對應版本的前端。",
            "Run 'cdui update' to re-download the matching frontend.",
        )

    stamp = _read_build_stamp()
    if stamp and stamp.get("commit"):
        head = _git_head_commit()
        if head is None:
            return  # can't judge without git — stay quiet
        in_sync = stamp["commit"] == head
        if not in_sync:
            # A backend-only commit after the build leaves the dist valid:
            # frontend/ unchanged between the stamped commit and HEAD counts
            # as in sync. None (undecidable) does not.
            in_sync = _git_frontend_unchanged_since(stamp["commit"]) is True
        if in_sync:
            if not _git_frontend_src_dirty():
                return  # checkout matches the stamped frontend == in sync
            # dirty tree: fall through to the mtime comparison below
        else:
            stamp_tag = stamp.get("tag")
            built_desc = (
                f"{stamp_tag} ({stamp['commit'][:12]})" if stamp_tag
                else stamp["commit"][:12]
            )
            print(
                "\n"
                + t(
                    f"警告：frontend/dist 建置自其他版本\n"
                    f"    dist 建置自：{built_desc}\n"
                    f"    目前程式碼：{head[:12]}\n",
                    f"Warning: frontend/dist was built from a different version\n"
                    f"    dist built from: {built_desc}\n"
                    f"    current code:    {head[:12]}\n",
                )
                + f"    {advice}\n",
                file=sys.stderr,
            )
            return
    else:
        # Unstamped dist (pre-1.4.1 release asset or hand-placed). Without
        # pnpm the user can't rebuild anyway and the dist is release-managed
        # — stay quiet instead of pointing at an impossible fix.
        if not shutil.which("pnpm"):
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
        "\n"
        + t(
            f"警告：frontend/dist 比 src 舊 {delta_min:.0f} 分鐘",
            f"Warning: frontend/dist is {delta_min:.0f} minutes older than src",
        )
        + f"\n    dist mtime: {dist_when}\n"
        f"    src  mtime: {src_when}\n"
        f"    {advice}\n",
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
                capture_output=True, timeout=5, check=True, **_CONSOLE_TEXT_KW,
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


def _parse_install_args(argv_tail: list[str],
                        prog: str = "cdui install") -> argparse.Namespace:
    """Parse the flags passed to `cdui install` / `cdui update`."""
    p = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Select the PyTorch wheel variant and dev tooling for the backend "
            "venv. `cdui install` prompts interactively from a TTY when no "
            "flags are given; `cdui update` never prompts and reuses whatever "
            "the venv already has."
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


def _apply_lang(args: argparse.Namespace) -> None:
    """--lang overrides the env-var-based LANG detection done at module load."""
    if args.lang:
        global LANG
        LANG = args.lang


def _explicit_options(args: argparse.Namespace) -> tuple[str | None, bool | None]:
    """The gpu/dev choices the user stated outright — flags first, then env
    vars. None means "not stated"; each caller picks its own default."""
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
    return gpu, dev


def _resolve_install_options(argv_tail: list[str]) -> tuple[str, bool]:
    """Combine CLI flags + env vars + interactive prompt into a final (gpu, dev)."""
    args = _parse_install_args(argv_tail)
    _apply_lang(args)

    detected_label, detected_gpu = detect_gpu()
    gpu, dev = _explicit_options(args)

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


def _venv_site_packages() -> list[Path]:
    """Every site-packages dir that actually exists in the backend venv."""
    dirs = [
        VENV / "Lib" / "site-packages",                                     # Windows
        VENV / "lib" / "site-packages",                                     # uv layout
    ]
    lib = VENV / "lib"
    if lib.exists():
        # POSIX: lib/python3.11/site-packages
        dirs += [entry / "site-packages" for entry in lib.iterdir()]
    return [d for d in dirs if d.exists()]


def _get_installed_torch_version() -> str | None:
    """Read torch's __version__ from the venv without importing torch."""
    for site in _venv_site_packages():
        path = site / "torch" / "version.py"
        if not path.exists():
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("__version__"):
                    return line.split("=", 1)[1].strip().strip("'\"")
        except OSError:
            pass
    return None


def _installed_torch_variant() -> str | None:
    """Which TORCH_INDEX_URLS key produced the torch currently in the venv.

    torch stamps its wheel index into the version's local build tag —
    ``2.11.0+cu128``, ``2.6.0+cpu``, ``2.5.1+rocm6.2`` — so the variant is
    readable without importing torch or re-guessing from the hardware.

    Returns None when torch isn't installed at all. An installed torch we
    can't place (untagged PyPI wheel — Apple Silicon, or a hand-built one)
    resolves to "skip": leaving an unrecognized wheel alone beats
    overwriting what may well be a deliberate choice.
    """
    version = _get_installed_torch_version()
    if version is None:
        return None
    _, sep, local = version.partition("+")
    if not sep:
        return "skip"
    # Longest key first so "rocm6.2" can't be shadowed by a shorter prefix;
    # the tag may carry extra segments (e.g. "cpu.cxx11.abi").
    indexed = sorted(
        (k for k, v in TORCH_INDEX_URLS.items() if (v or "").startswith("https://")),
        key=len, reverse=True,
    )
    for key in indexed:
        if local == key or local.startswith(key + "."):
            return key
    return "skip"


def _venv_has_dev_extra() -> bool:
    """Was the venv installed with the [dev] extra? pytest is the marker —
    httpx would false-positive, since the LLM clients depend on it anyway."""
    return any(any(site.glob("pytest-*.dist-info")) for site in _venv_site_packages())


def _resolve_update_options(argv_tail: list[str]) -> tuple[str, bool]:
    """Final (gpu, dev) for `cdui update` — never prompts.

    An update is not a re-install. The user already chose their PyTorch
    variant and dev tooling, so reuse what the venv actually has instead of
    asking again (the installer menu has no business appearing here) or
    re-deriving from hardware detection, which would silently overwrite a
    deliberate choice. Flags and env vars still override.
    """
    args = _parse_install_args(argv_tail, prog="cdui update")
    _apply_lang(args)
    gpu, dev = _explicit_options(args)

    if gpu is None:
        gpu = _installed_torch_variant()
    if gpu is None:
        # No torch in the venv at all — a half-built install. Fall back to
        # detection so `cdui update` can still repair it.
        gpu = "auto"
    if gpu == "auto":
        _, gpu = detect_gpu()

    if dev is None:
        dev = _venv_has_dev_extra()

    current = _get_installed_torch_version() or t("尚未安裝", "not installed")
    section(
        f"CodefyUI update: gpu={gpu}, dev={dev}（目前 torch：{current}）",
        f"CodefyUI update: gpu={gpu}, dev={dev} (current torch: {current})",
    )
    return gpu, dev


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
        # `--reinstall-package` forces uv to drop the existing torch even when
        # the version constraint is already satisfied. Without it, swapping
        # variants (e.g. `--gpu cpu` after a previous `cu128` install) is a
        # no-op and the user keeps the wrong wheel. It is *only* needed for
        # that switch though: when the installed variant already matches, the
        # flag buys nothing but a multi-GB re-download on every `cdui update`.
        # Dropping it still runs against the right index, so a raised torch
        # floor upgrades from there rather than falling back to default PyPI
        # (which on Windows would quietly swap a CUDA build for a CPU one).
        switching = _installed_torch_variant() != gpu
        cmd = ["uv", "pip", "install"]
        if switching:
            section(f"Backend: 安裝 PyTorch（{gpu}）— {index_url}",
                    f"Backend: installing PyTorch ({gpu}) — {index_url}")
            cmd += ["--reinstall-package", "torch",
                    "--reinstall-package", "torchvision"]
        else:
            section(f"Backend: 沿用現有 PyTorch（{gpu}）— 只檢查更新",
                    f"Backend: keeping existing PyTorch ({gpu}) — checking for updates only")
        cmd += ["torch", "torchvision", "--index-url", index_url]
        run(cmd, cwd=BACKEND_DIR)

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
        _write_build_stamp("local-build")
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

    # Resolve options *before* touching git: `--help` and bad flags must exit
    # without hard-resetting the working tree, and the summary belongs above
    # the long fetch/build output. Nothing here depends on the new source —
    # this process already imported the old dev.py either way.
    gpu, dev = _resolve_update_options(sys.argv[2:])

    # Nothing below this line is survivable by a running server: the checkout
    # is hard-realigned under it, the frontend/dist it is serving is deleted,
    # and its dependencies are rewritten in the venv it imported from. The
    # result is not a crash — it is a server that stays up, keeps answering,
    # and returns 404 for its own JavaScript, which is far harder to
    # recognise than a clean refusal. On one laptop that was survivable
    # because you knew you had started it; on a shared box the server you
    # break is usually not yours.
    #
    # `_running_server_pid()` is the same liveness check `cdui status` runs,
    # deliberately: two answers to "is it running" would eventually disagree.
    # It also clears a stale pidfile as a side effect, so a crashed server
    # cannot block an update forever.
    running = _running_server_pid()
    if running is not None:
        err(f"伺服器正在執行中（PID {running}，{_display_url(*_server_addr())}），"
            f"不能在此時 update。請先執行 cdui stop。",
            f"A CodefyUI server is running (PID {running}, "
            f"{_display_url(*_server_addr())}) — refusing to update underneath "
            f"it. Stop it first: cdui stop")
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
    _write_build_stamp("local-build")
    print(f"=== 建置完成：{DIST_DIR} ===")


# ── Background server management ───────────────────────────────────────
# `cdui start` daemonizes by default so users can close the terminal and keep
# the server running, then manage it with `cdui status` / `cdui stop`. The PID
# + log live under the repo-local dev data dir alongside the session token.
SERVER_PIDFILE = DEV_USER_DATA_DIR / "server.pid"
SERVER_LOG = DEV_USER_DATA_DIR / "server.log"
# host:port of the last-started server, so status/stop report real URLs.
SERVER_ADDRFILE = DEV_USER_DATA_DIR / "server.addr"


def _parse_host_port(argv: list) -> "tuple[str, int]":
    """Read --host/--port from start's argv (same lightweight style as
    --foreground). Defaults unchanged: 127.0.0.1:8000."""
    host, port = "127.0.0.1", 8000
    for i, a in enumerate(argv):
        if a == "--host" and i + 1 < len(argv):
            host = argv[i + 1]
        elif a.startswith("--host="):
            host = a.split("=", 1)[1]
        elif a == "--port" and i + 1 < len(argv):
            try:
                port = int(argv[i + 1])
            except ValueError:
                pass
        elif a.startswith("--port="):
            try:
                port = int(a.split("=", 1)[1])
            except ValueError:
                pass
    return host, port


def _split_forwarded_args(argv: list) -> "tuple[list, list]":
    """Split start's argv at the first bare ``--``.

    Everything before the separator belongs to ``cdui start``; everything
    after it is forwarded to uvicorn verbatim, so flags this launcher knows
    nothing about (``--proxy-headers``, ``--root-path``,
    ``--forwarded-allow-ips``, ``--timeout-keep-alive``) become reachable
    without giving up the daemon, the pidfile, ``cdui status`` and
    ``cdui stop``.

    The separator is what makes the two namespaces non-colliding, and it has
    to cut in BOTH directions:

    * ``_parse_host_port`` / ``_parse_project`` / the ``-f`` scan are
      positional scanners that would otherwise happily read a forwarded
      ``--host`` as cdui's own. They are handed the head only.
    * A future ``cdui start`` flag can never shadow a uvicorn flag of the
      same name, because the tail is never scanned for cdui flags.

    Only the FIRST ``--`` is consumed. Any later one is a uvicorn argument
    and is forwarded untouched.
    """
    if "--" not in argv:
        return list(argv), []
    cut = argv.index("--")
    return list(argv[:cut]), list(argv[cut + 1:])


# uvicorn flags that `cdui start` owns and must not receive twice. --host and
# --port are mirrored into three places that would silently disagree with the
# real bind: CODEFYUI_HOST/PORT in the child env (which is what app.core.auth
# builds the Host whitelist from), SERVER_ADDRFILE (what `cdui status` and
# `cdui stop` report), and the health poll below. --ws-max-size is mirrored
# into settings.WS_MAX_MESSAGE_BYTES, which is the number the docs quote and
# the number the editor's "graph too large" message is about. A forwarded copy
# wins on uvicorn's argparse and desyncs whichever set it belongs to, so refuse
# it and name the real knob.
_UVICORN_FLAGS_CDUI_OWNS = ("--host", "--port", "--ws-max-size")

# Owned flag -> the remedy to name when it is forwarded after `--`. A refusal
# that does not point at the knob that DOES work is just a dead end.
# --host/--port have cdui flags of their own; --ws-max-size is configured by
# environment like every other app setting.
_UVICORN_FLAG_ALTERNATIVE = {
    "--host": "使用 cdui start --host。",
    "--port": "使用 cdui start --port。",
    "--ws-max-size": "改設環境變數 CODEFYUI_WS_MAX_MESSAGE_BYTES。",
}
_UVICORN_FLAG_ALTERNATIVE_EN = {
    "--host": "use `cdui start --host` instead.",
    "--port": "use `cdui start --port` instead.",
    "--ws-max-size": "set CODEFYUI_WS_MAX_MESSAGE_BYTES instead.",
}

# Mirrors app.config.Settings.WS_MAX_MESSAGE_BYTES, including its fallback to
# MAX_RUN_BODY_BYTES. Duplicated rather than imported because dev.py runs on a
# bare interpreter before the venv exists and must never import the backend.
# `backend/tests/test_ws_max_size.py` fails if the two ever disagree.
_WS_MAX_MESSAGE_BYTES_DEFAULT = 64 * 1024 * 1024  # 64 MB


def _ws_max_size() -> int:
    """The WS frame ceiling to hand uvicorn, mirroring Settings' precedence.

    CODEFYUI_WS_MAX_MESSAGE_BYTES wins; otherwise the WS cap follows
    CODEFYUI_MAX_RUN_BODY_BYTES so one graph ceiling covers both transports;
    otherwise the shared default. A malformed or non-positive value is
    ignored HERE and left to pydantic to reject in the child, so the two
    layers cannot disagree about what counts as valid (core#274).
    """
    for var in ("CODEFYUI_WS_MAX_MESSAGE_BYTES", "CODEFYUI_MAX_RUN_BODY_BYTES"):
        raw = os.environ.get(var)
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if value > 0:
            return value
    return _WS_MAX_MESSAGE_BYTES_DEFAULT


def _reject_owned_uvicorn_flags(extra: list) -> None:
    """Exit(2) if the forwarded tail redefines a bind flag cdui owns."""
    for a in extra:
        name = a.split("=", 1)[0]
        if name in _UVICORN_FLAGS_CDUI_OWNS:
            err(
                f"{name} 不能透過 -- 轉發給 uvicorn，"
                f"{_UVICORN_FLAG_ALTERNATIVE[name]}",
                f"{name} cannot be forwarded to uvicorn after `--`; "
                f"{_UVICORN_FLAG_ALTERNATIVE_EN[name]}",
            )
            if name == "--ws-max-size":
                print(
                    t(
                        "  cdui 會用 CODEFYUI_WS_MAX_MESSAGE_BYTES 推導這個旗標，"
                        "轉發的複本會讓實際上限與文件、以及編輯器的「圖太大」"
                        "訊息不一致。",
                        "  cdui derives this flag from "
                        "CODEFYUI_WS_MAX_MESSAGE_BYTES; a forwarded copy would "
                        "desync the real ceiling from the documented one and "
                        "from the editor's \"graph too large\" message.",
                    ),
                    file=sys.stderr,
                )
            else:
                print(
                    t(
                        "  cdui 會把綁定位址寫進 server.addr 與子行程環境變數，"
                        "轉發的複本會讓 cdui status / cdui stop 與實際綁定不一致。",
                        "  cdui records the bind address in server.addr and in "
                        "the child environment; a forwarded copy would desync "
                        "it from `cdui status` and `cdui stop`.",
                    ),
                    file=sys.stderr,
                )
            sys.exit(2)


def _parse_project(argv: list) -> "str | None":
    """Read --project <dir> from start/dev argv (same lightweight style as
    --host)."""
    for i, a in enumerate(argv):
        if a == "--project" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--project="):
            return a.split("=", 1)[1]
    return None


def _activate_project(raw: str) -> None:
    """Validate the project manifest and export CODEFYUI_PROJECT_DIR (abs) into
    the child env so uvicorn derives its roots (spec 7.1). Exits on a missing
    manifest."""
    proj = Path(raw).expanduser().resolve()
    manifest = proj / "codefyui.project.toml"
    if not manifest.exists():
        print(t(f"錯誤：找不到專案 manifest：{manifest}",
                f"Error: project manifest not found: {manifest}"),
              file=sys.stderr)
        print(t("  用 'cdui project init <dir>' 建立專案。",
                "  Create one with 'cdui project init <dir>'."), file=sys.stderr)
        sys.exit(1)
    os.environ["CODEFYUI_PROJECT_DIR"] = str(proj)
    print(t(f"    專案 → {proj}", f"    Project -> {proj}"))


def _probe_host(host: str) -> str:
    """The address to PROBE for a bind host: 0.0.0.0/:: listen everywhere
    but answer on loopback; a concrete LAN IP answers only on itself."""
    return "127.0.0.1" if host in ("0.0.0.0", "::") else host


def _print_uninstalled_builtin_packs() -> None:
    """Name any built-in pack that shipped on disk but was never installed.

    A release can add a pack — `stats` did — and `cdui update` puts its files
    in place, but the server loads only what the lockfile records and nothing
    re-syncs it. The pack is then fully installable and completely invisible:
    a class follows the update instructions, the new chapter's nodes are not
    in the palette, and no message anywhere explains why.

    This is discoverability only. Nothing is enabled on the user's behalf —
    running code someone did not ask for because a release shipped it is a
    consent decision, not a startup detail, so the pack still installs by hand
    (`cdui plugin sync`, one verb for all of them, #175). A pack the user
    uninstalled is not listed: `available_builtin_packs` subtracts the
    uninstall tombstones, so this notice stops nagging once you have said no.

    `cdui start` has already hopped into the venv (`_SKIP_VENV_EXEC` excludes
    it), so scripts/plugins.py's `app.core.*` imports resolve here. Guarded
    anyway: a notice must never be the reason a server fails to start.
    """
    try:
        scripts_dir = str(Path(__file__).resolve().parent)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import plugins as plugin_cli  # noqa: PLC0415 — late import: needs venv

        available = plugin_cli.available_builtin_packs()
    except Exception:
        return
    if not available:
        return
    names = ", ".join(f"{pack_id} ({name})" for pack_id, name in available)
    print(t(
        f"    有尚未安裝的內建外掛：{names}",
        f"    Built-in packs available but not installed: {names}",
    ))
    # One verb, not the id list this used to print (#175). The list grows with
    # every release, and retyping it is the step people skip — after which the
    # chapter's nodes are missing and this notice was the only warning.
    print(t(
        "    安裝：cdui plugin sync",
        "    Install them with: cdui plugin sync",
    ))


def _display_url(host: str, port: int) -> str:
    """Clickable URL for a bind host: wildcard/loopback render as
    localhost; a concrete LAN IP renders as itself."""
    shown = "localhost" if host in (
        "127.0.0.1", "0.0.0.0", "::", "::1", "localhost") else host
    return f"http://{shown}:{port}"


def _local_ips() -> "list[str]":
    """Best-effort local IPv4 addresses. Stdlib-only duplicate of
    app.core.auth.local_interface_ips — dev.py must run without the venv."""
    import socket
    ips: "set[str]" = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None,
                                       socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("192.0.2.1", 80))  # TEST-NET-1: never routed
            ips.add(s.getsockname()[0])
        finally:
            s.close()
    except OSError:
        pass
    ips.discard("127.0.0.1")
    return sorted(ips)


def _server_addr() -> "tuple[str, int]":
    """The last-started server's (host, port) from server.addr; defaults
    for pre-Stage-2 servers or when never started."""
    try:
        raw = SERVER_ADDRFILE.read_text().strip()
        host, _, port = raw.rpartition(":")
        return (host or "127.0.0.1"), int(port)
    except (OSError, ValueError):
        return "127.0.0.1", 8000


def _server_health_url(host: str, port: int) -> str:
    return f"http://{_probe_host(host)}:{port}/api/health"


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
            capture_output=True, **_CONSOLE_TEXT_KW,
            # `packs-run-pending` polls this every half second for up to two
            # minutes, from a DETACHED process with no console of its own --
            # without this each poll pops a console window over whatever the
            # user is looking at while their server is away.
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        # `or ""`: a dead pid makes tasklist print a translated "no tasks
        # match" message, and any child that writes nothing at all leaves
        # None here too. Either way the answer is "no such process", not a
        # traceback out of a launcher.
        return str(pid) in (out.stdout or "")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user
    return True


def _server_healthy(host: "str | None" = None, port: "int | None" = None,
                    timeout: float = 1.0) -> bool:
    return _server_health_info(host, port, timeout) is not None


def _server_health_info(host: "str | None" = None,
                        port: "int | None" = None,
                        timeout: float = 1.0) -> "dict | None":
    """Fetch and parse /api/health. Returns the JSON dict, or None if the
    server isn't responding (or returned a non-200 / unparseable body).
    Host/port default to the recorded server.addr of the last start."""
    if host is None or port is None:
        addr_host, addr_port = _server_addr()
        host = host if host is not None else addr_host
        port = port if port is not None else addr_port
    try:
        with urlopen(_server_health_url(host, port), timeout=timeout) as resp:
            if resp.status != 200:
                return None
            import json  # noqa: PLC0415 — only needed here
            return json.loads(resp.read().decode("utf-8", "replace"))
    except (URLError, HTTPError, TimeoutError, OSError, ValueError):
        return None


def _running_server_pid() -> "int | None":
    """Return the PID of the live background server, or None. Clears a stale
    pidfile (and its recorded server.addr) as a side effect so callers don't
    act on a dead PID or report a stale address."""
    pid = _read_server_pid()
    if pid is None:
        return None
    if _pid_alive(pid):
        return pid
    # Stale pidfile (server crashed / was killed externally) — tidy up both
    # files so `cdui status` can't report a dead server's last-known address.
    SERVER_PIDFILE.unlink(missing_ok=True)
    SERVER_ADDRFILE.unlink(missing_ok=True)
    return None


def start() -> None:
    """Production 模式：單一 uvicorn 由 FastAPI 直接 serve dist。

    預設背景執行（daemon）；加 --foreground / -f 改在前景執行。
    `--` 之後的參數原樣轉給 uvicorn（反向代理用）。
    """
    if not DIST_INDEX.exists():
        print(
            "錯誤：找不到 frontend/dist/index.html\n"
            "  請執行 'cdui install'（下載 release dist）"
            " 或 'cdui build'（本地 build，需 pnpm）。",
            file=sys.stderr,
        )
        sys.exit(1)

    # Everything after a bare `--` is uvicorn's; cdui's own flags are only
    # ever read from the head, so the two sets cannot collide.
    own_argv, uvicorn_extra = _split_forwarded_args(sys.argv[2:])
    _reject_owned_uvicorn_flags(uvicorn_extra)
    foreground = any(a in ("-f", "--foreground") for a in own_argv)
    host, port = _parse_host_port(own_argv)

    existing = _running_server_pid()
    if existing is not None:
        print(f"CodefyUI 已在背景執行（PID {existing}）。")
        print("  查看狀態：cdui status    停止：cdui stop")
        return

    _warn_if_dist_stale()
    _apply_dev_env()
    # Before anything is started. Two cases, one file: a restart that is
    # still running must not get a second server on its port while it is
    # replacing the files the first one holds open, and a claim left behind
    # by a server that died mid-restart must not refuse every future
    # restart-mode install for fifteen minutes.
    if not _restart_preflight():
        return
    project = _parse_project(own_argv)
    if project is not None:
        _activate_project(project)
    # settings.HOST/PORT (and therefore init_allowed_hosts) must agree
    # with the actual bind — binding a concrete LAN IP whitelists it
    # automatically (app.core.auth.init_allowed_hosts).
    os.environ["CODEFYUI_HOST"] = host
    os.environ["CODEFYUI_PORT"] = str(port)
    # Only this launcher knows the server it is about to start is one it
    # supervises. The Package Center reads this back as `launch_mode` to tell
    # a server that could be restarted for the user from a bare
    # `uvicorn app.main:app`, which nothing here can bring back up.
    os.environ["CODEFYUI_MANAGED"] = "start"
    # ...and only this launcher knows HOW it would launch it again. Both
    # paths get them: a foreground server can be restarted too, it simply
    # comes back as a daemon (see `_restart_relaunch_argv`).
    _export_restart_env(own_argv, uvicorn_extra)
    uvicorn = _require_venv_tool("uvicorn")
    # Extras go last so `app.main:app` keeps its position — the process
    # matchers in `cdui stop` key on it. --ws-max-size is passed explicitly
    # because uvicorn's own default (16 MB) is stricter than this project's
    # body ceiling, so the canvas socket would otherwise refuse graphs the
    # HTTP routes accept (core#274).
    cmd = [uvicorn, "app.main:app", "--host", host, "--port", str(port),
           "--ws-max-size", str(_ws_max_size()), *uvicorn_extra]
    SERVER_ADDRFILE.parent.mkdir(parents=True, exist_ok=True)
    SERVER_ADDRFILE.write_text(f"{host}:{port}")

    def _print_reach_lines() -> None:
        print(f"    開啟 → {_display_url(host, port)}")
        if host not in ("127.0.0.1", "localhost", "::1"):
            lan_ips = _local_ips() if host in ("0.0.0.0", "::") else [host]
            for ip in lan_ips:
                print(f"    LAN  → http://{ip}:{port}")
            print(t(
                "    注意：任何能連到這個埠的人都能控制此實例；只在信任的網路使用。",
                "    NOTE: anyone who can reach this port controls the "
                "instance; use only on trusted networks.",
            ))
        if uvicorn_extra:
            print(t(
                f"    uvicorn 額外參數 → {' '.join(uvicorn_extra)}",
                f"    extra uvicorn args → {' '.join(uvicorn_extra)}",
            ))
        _print_uninstalled_builtin_packs()

    if foreground:
        print("=== CodefyUI 啟動（前景；Ctrl+C 停止）===")
        _print_reach_lines()
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
        if _server_healthy(host, port):
            break
        time.sleep(0.5)
    else:
        print("警告：等候逾時，伺服器尚未回應健康檢查（仍在背景嘗試啟動）。")

    print("=== CodefyUI 已在背景啟動 ===")
    print(f"    PID         → {proc.pid}")
    _print_reach_lines()
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


# ── Restart-mode installs ─────────────────────────────────────────────────────
#
# A pack whose install REPLACES something the server has already imported --
# the GPU torch wheel above all -- cannot be installed by that server. On
# Windows the files are locked; everywhere else the process keeps running the
# code it just replaced. So the install happens across the gap where the
# server does not exist: the server writes down what it wanted
# (`<user data>/packs/pending_restart.json`), starts `packs-run-pending`
# detached, and shuts itself down. This half waits for it to go, runs the
# install, records the outcome where the next server reads it
# (`last_restart_job.json`), and starts the server again.
#
# Nothing here imports the backend, and that is the design rather than the
# usual dev.py constraint: for part of this command's run, the venv it is
# installing into has no working torch in it. The schema numbers, the file
# names and the two environment variable names are therefore a DUPLICATE of
# `backend/app/core/packs/restart.py`, and the JSON between them is the
# interface. `test_dev_and_restart_agree_on_the_restart_handshake` fails the
# day the two copies drift apart.

#: JSON list `[<outer python>, <abs path of this file>]`, exported by
#: `cdui start`. Not the `cdui` shim: the shim's whole job is to FIND an
#: interpreter, and asking a detached child to find one again is a second
#: chance to find a different one -- on a box with two checkouts, the wrong
#: one. Mirrors `restart.LAUNCHER_ENV`.
LAUNCHER_ENV = "CODEFYUI_LAUNCHER"

#: JSON list: the arguments THIS `cdui start` was given. The helper relaunches
#: with exactly these, so the server comes back on the address the browser is
#: still pointing at. Mirrors `restart.RELAUNCH_ARGV_ENV`.
RELAUNCH_ARGV_ENV = "CODEFYUI_RELAUNCH_ARGV"

#: Mirrors `restart.PENDING_SCHEMA` / `restart.OUTCOME_SCHEMA`. The numbers
#: are the handshake: a helper from an older install refuses a file it does
#: not understand rather than guessing at it.
PENDING_SCHEMA = 1
OUTCOME_SCHEMA = 1

#: The subcommand the server spawns. Mirrors `restart.HELPER_COMMAND`.
HELPER_COMMAND = "packs-run-pending"

#: How long a pending claim may sit before it is abandoned. Mirrors
#: `restart.STALE_PENDING_S`.
STALE_PENDING_S = 15 * 60

#: How long a claim with no `helper_pid` in it yet is still believed. The
#: helper is spawned detached and writes its pid in as its first act, so this
#: only has to cover process creation -- but on a cold Windows box with a
#: virus scanner in the way that is seconds, not milliseconds. A minute is an
#: order of magnitude more than that, and it is the whole cost of the case
#: this window exists to cover being wrong: a helper that never started
#: leaves the user with no server and a launcher that refuses, and every
#: second of the grace is a second they spend not knowing why.
HELPER_START_GRACE_S = 60

#: The two control files, under `<user data>/packs`. Mirrors `packs.paths`.
PENDING_FILE_NAME = "pending_restart.json"
OUTCOME_FILE_NAME = "last_restart_job.json"

#: How long the helper waits for the server to shut itself down before it
#: stops asking nicely, and how long it then waits for the kill to land.
#: Generous: the server finishes in-flight requests and closes its database
#: on the way out, and installing while it still holds the files is the exact
#: failure this whole mechanism exists to avoid.
RESTART_WAIT_S = 120.0
RESTART_KILL_GRACE_S = 10.0
RESTART_POLL_S = 0.5

#: Free bytes the install needs before it is allowed to start. A torch wheel
#: set unpacks to several GB; a pack's pip specs are a couple of hundred MB
#: of wheels plus room to build. Refusing up front is the cheap failure: an
#: install that runs out of disk halfway leaves the venv with a torch that
#: does not import, which is strictly worse than the one the user had.
RESTART_MIN_FREE_TORCH = 3 * 1024 ** 3
RESTART_MIN_FREE_PIP = 1 * 1024 ** 3

#: Lines of installer output kept in the outcome record. The full log is
#: megabytes of resolver output and lives in the job log file; these are the
#: lines that say why, and they are read back in a browser.
RESTART_LOG_TAIL_LINES = 40

#: How long a finished restart stays worth mentioning in `cdui status`.
RESTART_NOTICE_S = 60 * 60


def _packs_control_dir() -> Path:
    """`<user data>/packs` -- what `app.core.packs.paths.control_dir()` returns.

    `CODEFYUI_USER_DATA_DIR` is the same switch the backend reads, and every
    dev.py command sets it (`_apply_dev_env`). The fallback is the value that
    function would have written, so a command that has not called it yet
    still looks in the right place.
    """
    override = os.environ.get("CODEFYUI_USER_DATA_DIR")
    root = Path(override) if override else DEV_USER_DATA_DIR
    return root / "packs"


def _pending_restart_file() -> Path:
    return _packs_control_dir() / PENDING_FILE_NAME


def _last_restart_file() -> Path:
    return _packs_control_dir() / OUTCOME_FILE_NAME


def _restart_log_file(job_id: str, control: "Path | None" = None) -> Path:
    """The log for one restart job. Mirrors `restart._log_file_name`, including
    the substitution: the id is read back out of a file on disk and then
    concatenated into a path, which is the shape of every directory-traversal
    bug ever written. A substitution rather than a rejection, because an odd
    job id should cost an ugly log name, not a server that never comes back.

    *control* is passed by the helper, which knows which directory the claim
    it is acting on came out of; everybody else asks the environment."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", job_id or "unknown")
    root = control if control is not None else _packs_control_dir()
    return root / "logs" / f"restart-{safe}.log"


def _read_json_file(path: Path) -> "dict | None":
    """A JSON object from *path*, or None for anything that is not one.

    `ValueError` covers both halves deliberately: a bad parse, and bytes that
    are not UTF-8 (`UnicodeDecodeError` IS a `ValueError`). Every caller here
    is one whose job is to get PAST a file in that state.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _write_json_atomic(path: Path, data: dict) -> bool:
    """Write where a reader may look at any moment. False if it could not.

    Never raises: the caller is on its way to starting a server again, and a
    record nobody could write is not a reason to skip that.
    """
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def _iso_now() -> str:
    from datetime import datetime, timezone  # noqa: PLC0415 — only needed here
    return datetime.now(timezone.utc).isoformat()


def _iso_age_seconds(stamp) -> "float | None":
    """Seconds since an ISO-8601 timestamp, or None when it is not one."""
    from datetime import datetime, timezone  # noqa: PLC0415 — only needed here
    if not isinstance(stamp, str):
        return None
    try:
        moment = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - moment).total_seconds()


def _pending_age_seconds(path: Path, data: "dict | None") -> "float | None":
    """How long ago this claim was made. `created_at`, then the file's mtime.

    Two clocks because the first one can be missing or nonsense in exactly
    the file this has to judge. None when neither answers, which every caller
    reads as "old" -- an age nobody can establish must not be the reason a
    user cannot get a server back.
    """
    age = _iso_age_seconds((data or {}).get("created_at"))
    if age is not None:
        return age
    try:
        return max(0.0, time.time() - path.stat().st_mtime)
    except OSError:
        return None


def _pending_state(path: Path, data: "dict | None") -> str:
    """Is a restart still happening (`finishing`) or did it die (`abandoned`)?

    One predicate, two callers -- `_restart_preflight` decides whether to
    start a server on it and `cdui status` decides what to call it, and those
    two must never disagree in front of a user comparing them.

    Mirrors `app.core.packs.restart._is_stale`, which the SERVER decides the
    same question with -- see the parity test in `test_dev_packs_cli.py`.

    The rules, in the order they are applied:

    * `helper_pid` and/or `installer_pid` present -> finishing while ANY of
      them is alive, abandoned once all of them are dead, at any age. The
      helper writes its own pid in as its first act and the installer's the
      moment `uv` starts, so between them they are the real answer whenever
      one exists -- and they OUTRANK the clock, because a torch download
      over a slow line is still going at minute sixteen and that is an
      install finishing, not one abandoned. Calling it dead there would
      start a second server into the venv `uv` is mid-way through rewriting,
      which is the one outcome all of this exists to prevent. Both pids and
      not just the helper's, because a helper that was killed (`End task`,
      a closed console) does not take its `uv` with it: that install carries
      on as an orphan, rewriting the venv, with a dead pid in the claim.
    * No pid at all and younger than `HELPER_START_GRACE_S` -> finishing.
      The helper is a detached process that was spawned seconds ago and has
      not written its pid yet; starting a second server into that window is
      how two `uv` runs end up in one site-packages.
    * No pid at all and older than that -> abandoned. The server wrote the
      claim and then died before its helper ever ran. (`STALE_PENDING_S` is
      the outer bound on the same case: an age nobody can read at all counts
      as old, and so does one past the fifteen-minute mark.)

    A file nothing can parse is abandoned outright: both writers are atomic,
    so an unreadable file is not a half-written claim -- it is not a claim.
    """
    if data is None:
        return "abandoned"
    pids = [pid for pid in (data.get("helper_pid"), data.get("installer_pid"))
            if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0]
    if pids:
        return "finishing" if any(_pid_alive(pid) for pid in pids) else "abandoned"
    age = _pending_age_seconds(path, data)
    if age is None or age > STALE_PENDING_S:
        return "abandoned"
    return "finishing" if age <= HELPER_START_GRACE_S else "abandoned"


def _restart_preflight() -> bool:
    """May `cdui start` go ahead? Clears an abandoned claim on the way.

    (This is the old `_clear_stale_pending`, renamed because it no longer
    only clears: the case it used to get wrong is a `cdui start` typed WHILE
    a restart is finishing, which would put a second server on the port the
    helper is about to relaunch onto, holding open the very files it is
    replacing.)

    False means a restart is in flight and this command must stand down --
    reported and returned WITHOUT an error, because the user asked for a
    running server and one is on its way. True means either there was no
    claim, or there was an abandoned one and it has been deleted; leaving
    that behind would refuse every future restart-mode install with "one is
    already pending".
    """
    path = _pending_restart_file()
    if not path.exists():
        return True

    data = _read_json_file(path)
    if _pending_state(path, data) == "finishing":
        section("正在完成一次重啟安裝，先不啟動伺服器",
                "A restart install is finishing; not starting a server")
        print(t(f"    套件 → {(data or {}).get('pack_id')}",
                f"    pack → {(data or {}).get('pack_id')}"))
        print(t("    完成後它會自己把伺服器啟動回來；用 cdui status 查看進度。",
                "    it starts the server again itself when it is done; "
                "watch it with `cdui status`."))
        # The way out, named. A `helper_pid` reads as alive whenever the OS
        # has handed that number to something else, and this branch has no
        # time limit -- so without the path on screen a recycled pid is a
        # launcher that refuses forever and never says what to delete.
        print(t(f"    待處理檔 → {path}（若確定沒有安裝在進行中，可刪除它強制啟動）",
                f"    claim -> {path} (delete it to force a start)"))
        return False

    try:
        path.unlink()
    except OSError:
        # Not a reason to refuse: the claim is dead either way, and the
        # server the user asked for matters more than the tidying.
        warn("無法刪除殘留的重啟安裝紀錄",
             "could not delete the leftover restart-install record")
        return True
    section("已清除上一次沒有完成的重啟安裝紀錄",
            "Cleared a leftover restart install that never finished")
    return True


def _restart_relaunch_argv(own_argv: list, uvicorn_extra: list) -> list:
    """What the helper should hand a fresh `cdui start`.

    Exactly what THIS start was given, minus `--foreground` / `-f`: the
    helper has no console to hand over, and a foreground server parented by
    a process that is about to exit would go with it. The forwarded tail is
    put back behind its `--`, so the relaunched start splits it the same way
    this one did.

    `--project` is the one argument that cannot be replayed verbatim -- see
    `_absolutise_project`.
    """
    argv = _absolutise_project(
        [a for a in own_argv if a not in ("-f", "--foreground")])
    if uvicorn_extra:
        argv += ["--", *uvicorn_extra]
    return argv


def _absolutise_project(argv: list) -> list:
    """Replace a `--project` value with the directory it actually opened.

    Everything else in the argv means the same thing from anywhere;
    `--project ./lab` does not. `_relaunch_server` runs the new `cdui start`
    with `cwd=ROOT` and `_activate_project` resolves against the current
    directory, so a relative path typed in any other directory comes back
    meaning `<repo>/lab`: usually nothing (the manifest check exits 1 and
    the server never returns from the restart) and, on a box that happens to
    have one there, somebody else's project.

    The value used is the one `_activate_project` computed and exported --
    not a second resolution here, which could differ from it. Nothing to
    rewrite when it never ran (no `--project`, or a caller that stubbed it),
    and the flag's two spellings are both handled because `_parse_project`
    accepts both.
    """
    resolved = os.environ.get("CODEFYUI_PROJECT_DIR")
    if not resolved:
        return argv
    out = list(argv)
    for i, arg in enumerate(out):
        if arg == "--project" and i + 1 < len(out):
            out[i + 1] = resolved
            return out
        if arg.startswith("--project="):
            out[i] = f"--project={resolved}"
            return out
    return out


def _export_restart_env(own_argv: list, uvicorn_extra: list) -> None:
    """Tell the server how to be started again.

    Nothing inside the server knows how it was launched; this launcher is the
    only process that does. Without these two variables
    `restart.restart_available()` is False and the Package Center refuses a
    restart-mode install with the command to type instead -- which is the
    right answer for a `uvicorn app.main:app` somebody started by hand.

    JSON rather than a space-joined string so a path with a space in it (the
    usual Windows one) survives the round trip.
    """
    os.environ[LAUNCHER_ENV] = json.dumps(
        [_outer_python(), str(Path(__file__).resolve())])
    os.environ[RELAUNCH_ARGV_ENV] = json.dumps(
        _restart_relaunch_argv(own_argv, uvicorn_extra))


#: A PEP 508 requirement, and nothing else: a name, optional extras, and an
#: optional comma-joined version range. An ALLOWLIST, because the alternative
#: -- banning the shapes we thought of -- is a list somebody adds a package
#: manager flag to next year. It excludes by construction everything the
#: review named: a leading `-` (uv would read `--index-url`, `-r`, `-e` as
#: flags), whitespace (one string carrying two arguments), and `/ \ @ ; #`
#: (a path, a local build, a URL, a shell line).
_REQUIREMENT_RE = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"           # name
    r"(?:\[[A-Za-z0-9._-]+(?:,[A-Za-z0-9._-]+)*\])?"         # extras
    r"(?:(?:===|==|!=|<=|>=|~=|<|>)[A-Za-z0-9.*+!-]+"        # a specifier
    r"(?:,(?:===|==|!=|<=|>=|~=|<|>)[A-Za-z0-9.*+!-]+)*)?$"  # ...and more
)


def _is_plain_requirement(spec: str) -> bool:
    """Is *spec* a package requirement, rather than an argument to uv?

    `specs` is the one field of the pending file that reaches an installer's
    argv verbatim -- it is the pack's package list and there is nowhere else
    for it to come from -- so it is also the one field that could widen what
    gets installed. `["--index-url", "https://evil.example/simple", "torch"]`
    is three strings uv reads as a flag, its value and a package; `["-r",
    "req.txt"]` installs a file; `["-e", "."]` or a bare path builds whatever
    is in a directory the file names.

    The catalog's only shape is `sentence-transformers>=3.0,<6`, and
    `test_every_spec_the_catalog_ships_passes_the_helper` runs the real
    catalog through this so a spec added there cannot silently start being
    refused.
    """
    return bool(isinstance(spec, str) and _REQUIREMENT_RE.match(spec))


def _validate_pending(data: "dict | None") -> "str | None":
    """Why this pending file must not be acted on, or None when it may be.

    Every check answers one question: did THIS installation's server write
    this file? It names an interpreter to install into, a package index to
    install from, and a program to start afterwards -- so it is read the way
    any other input is, not trusted for having been found in the right place.

    A failure here means no install AND NO RELAUNCH (see `_run_pending_job`):
    a file that is not ours describes a server we never took down, and
    starting one "back" would be starting a second one.
    """
    if data is None:
        return "the pending file is missing, empty or unreadable"

    schema = data.get("schema")
    if isinstance(schema, bool) or schema != PENDING_SCHEMA:
        return f"schema {schema!r} is not {PENDING_SCHEMA}"

    kind = data.get("kind")
    if kind not in ("torch", "pip"):
        return f"kind {kind!r} is not 'torch' or 'pip'"

    launcher = data.get("launcher")
    if (not isinstance(launcher, list) or len(launcher) != 2
            or not all(isinstance(part, str) and part for part in launcher)):
        return "launcher is not exactly an interpreter and a script"
    try:
        mine = Path(launcher[1]).resolve() == Path(__file__).resolve()
    except OSError:
        mine = False
    if not mine:
        return f"launcher {launcher[1]!r} is not this installation's dev.py"
    try:
        # THIS interpreter, not merely one that exists. `spawn_helper` starts
        # the helper as `[launcher[0], dev.py, packs-run-pending, <file>]`, so
        # a claim this process is acting on names the interpreter running it;
        # `is_file()` alone accepts any python on the box, including another
        # checkout's venv, and the relaunch would then bring the server back
        # on an environment that is not the one just installed into.
        # `samefile` also answers the older question -- it raises
        # FileNotFoundError for a path that is gone, which is the case
        # `restart_available` checked when the panel was DRAWN and which
        # minutes may have since changed.
        runnable = os.path.samefile(launcher[0], sys.executable)
    except (OSError, ValueError):
        runnable = False
    if not runnable:
        return (f"launcher {launcher[0]!r} is not the interpreter this helper "
                f"is running on")

    relaunch = data.get("relaunch_argv")
    if (not isinstance(relaunch, list)
            or not all(isinstance(part, str) for part in relaunch)):
        return "relaunch_argv is not a list of strings"

    venv_python = data.get("venv_python")
    if not isinstance(venv_python, str) or not venv_python:
        return "venv_python is not a path"
    # The DIRECTORY is resolved and the leaf name is not. Resolving the whole
    # path would follow the interpreter itself, and on POSIX `uv venv`
    # symlinks `.venv/bin/python` straight at the uv-managed base interpreter
    # (the same fact `_exec_into_venv_if_available` discriminates on
    # `sys.prefix` for) -- so `.resolve()` lands somewhere under
    # ~/.local/share/uv, every genuine pending file is refused, and the user
    # loses their server on the one platform this was never tested on.
    # Resolving the parent still closes the hole that matters: `..` segments
    # and a symlinked DIRECTORY cannot smuggle the install out of this venv.
    try:
        leaf = Path(venv_python)
        inside = (leaf.parent.resolve() / leaf.name).is_relative_to(VENV.resolve())
    except OSError:
        inside = False
    if not inside:
        return f"venv_python {venv_python!r} is not inside {VENV}"

    pid = data.get("server_pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return f"server_pid {pid!r} is not a process id"

    if kind == "torch":
        index_url = data.get("index_url")
        if index_url is not None and index_url not in set(TORCH_INDEX_URLS.values()):
            return (f"index_url {index_url!r} is not one of the PyTorch wheel "
                    f"indexes this launcher installs from")
    else:
        specs = data.get("specs")
        if (not isinstance(specs, list)
                or not all(isinstance(s, str) and s for s in specs)):
            return "specs is not a list of package requirements"
        for spec in specs:
            if not _is_plain_requirement(spec):
                return (f"spec {spec!r} is not a package requirement; only a "
                        f"name with an optional version range is installed "
                        f"from here")
    return None


def _wait_for_server_exit(pid: int) -> str:
    """Wait for the server to go, and say which way it went.

    `exited` -- it shut itself down, which is the design: it schedules a
    SIGINT at itself the moment this helper is spawned, so its lifespan
    shutdown runs and the database closes. `terminated` -- it did not, within
    `RESTART_WAIT_S`, and was stopped. `alive` -- it survived that too.
    """
    deadline = time.monotonic() + RESTART_WAIT_S
    while True:
        if not _pid_alive(pid):
            return "exited"
        if time.monotonic() >= deadline:
            break
        time.sleep(RESTART_POLL_S)

    print(t(f"    伺服器 PID {pid} 超過 {RESTART_WAIT_S:.0f} 秒仍未結束，強制停止",
            f"    server PID {pid} did not exit within {RESTART_WAIT_S:.0f}s; "
            f"stopping it"), flush=True)
    _terminate_pid(pid)

    deadline = time.monotonic() + RESTART_KILL_GRACE_S
    while True:
        if not _pid_alive(pid):
            return "terminated"
        if time.monotonic() >= deadline:
            return "alive"
        time.sleep(RESTART_POLL_S)


def _forget_stopped_server() -> None:
    """Drop the pidfile and address of the server this helper just outlived.

    Nothing else does. A server that shuts ITSELF down (which is exactly what
    a restart-mode install asks it to do) leaves both files behind, and the
    `start()` this helper is about to run consults the pidfile FIRST: a pid
    the OS has since handed to something else reads as alive, so it prints
    "already running", returns 0, and the outcome record says `relaunch: ok`
    with no server anywhere.

    Safe here and nowhere else: this is the window in which
    `_restart_preflight` stands every other `cdui start` down, so no other
    process can be writing files this would delete out from under it.
    """
    for path in (SERVER_PIDFILE, SERVER_ADDRFILE):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            # Not worth a word to the user: the relaunch below is what they
            # are waiting for, and a leftover file costs at most one
            # "already running" they can act on with `cdui status`.
            pass


def _restart_disk_shortfall(kind: str) -> "str | None":
    """Why there is not enough room for this install, or None when there is.

    Measured on the VENV's filesystem -- where the wheels are UNPACKED, which
    is the space that decides whether the venv ends up with a torch that
    imports. It is deliberately not uv's cache volume: uv downloads and
    unpacks to `UV_CACHE_DIR` (often another disk, or a tmpfs), so a check
    there would answer a different question and pass on a machine whose venv
    disk is full. Neither is checked exhaustively -- this is a floor against
    the obviously doomed case, not a guarantee.

    An unreadable answer is not a "no": a network mount that will not report
    its size must not cost the user their install.
    """
    need = RESTART_MIN_FREE_TORCH if kind == "torch" else RESTART_MIN_FREE_PIP
    try:
        free = shutil.disk_usage(VENV).free
    except OSError:
        return None
    if free >= need:
        return None
    return (f"not enough free disk for the install: "
            f"{free / 1024 ** 3:.1f} GB free, "
            f"about {need // 1024 ** 3} GB needed")


def _pending_install_cmd(data: dict) -> "tuple[list | None, str]":
    """The `uv pip install` argv for this job, or (None, why not).

    The torch branch mirrors `install()` exactly. `--reinstall-package` is
    what makes uv DROP a wheel whose version constraint is already satisfied,
    which is the entire point of a variant switch; `--python` is pinned at the
    server's interpreter, because the helper's own is deliberately a
    different one.

    The package NAMES for a torch job are spelled out here rather than read
    from the file. Every other field describes the job; a list of package
    names read off disk and spliced into an installer's argv would be the one
    field that IS the job, and this file is exactly the input that must not be
    able to widen what gets installed. `specs` for a pip job is the deliberate
    exception -- it is the pack's package list and there is nowhere else for it
    to come from -- and it is bounded on the writing side by the catalog.
    """
    uv = shutil.which("uv")
    if uv is None:
        return None, "uv is not on PATH, so nothing could be installed"

    python = data["venv_python"]
    if data["kind"] == "torch":
        index_url = data.get("index_url")
        if not index_url or index_url == "__skip__":
            return None, "this torch job names no wheel index to install from"
        return [uv, "pip", "install", "--python", python,
                "--reinstall-package", "torch",
                "--reinstall-package", "torchvision",
                "torch", "torchvision", "--index-url", index_url], ""

    specs = [s for s in (data.get("specs") or ()) if isinstance(s, str)]
    if not specs:
        return None, "this pip job lists no packages to install"
    # No constraints file, on purpose: this is the install that REPLACES what
    # the server had loaded, and constraining it to what was already there
    # would pin the very versions it exists to move.
    return [uv, "pip", "install", "--python", python, *specs], ""


def _restart_child_env() -> dict:
    """This process's environment, minus the pointers to its own stdlib.

    Sanitised, never rebuilt. `CODEFYUI_USER_DATA_DIR` is not something the
    relaunched `start()` can rederive, and `CODEFYUI_OUTER_PYTHON` is how the
    restart AFTER this one finds a launcher at all — handing a child a fresh
    environment is how the second restart of a session refuses.

    What does come out are the three variables that tell a Python process
    where its standard library and its `sys.executable` are. Every one of
    them is a statement about the interpreter that set it, and every child
    started here is a different one: the installer runs against the venv, and
    the relaunched server is the outer interpreter hopping into that venv.
    A `PYTHONHOME` carried across that gap makes an `import` deep in the
    startup path die with "AssertionError: SRE module mismatch", detached,
    with a log file for its only witness. Mirrors `runner.pip_env`, which
    drops the same three (and `PYTHONPATH`, which matters on the server side
    and not here).
    """
    env = os.environ.copy()
    for name in ("PYTHONHOME", "__PYVENV_LAUNCHER__", "PYTHONEXECUTABLE"):
        env.pop(name, None)
    return env


def _raise_on_sigterm() -> None:
    """Make a SIGTERM at this helper raise, the way a Ctrl-C already does.

    POSIX kills a process on SIGTERM by default -- no `except`, no `finally`,
    no last words. For this helper that is the worst possible exit: `uv` is
    left rewriting the venv as an orphan, the claim on disk still says an
    install is under way, and no server is ever started again. Raising
    instead puts a `kill`, a service stop and a Ctrl-C on the one path that
    stops the installer, records the failure and relaunches.

    Windows has nothing to arm (its SIGTERM is never delivered; a Task
    Manager "End task" is a TerminateProcess nothing can intercept), and a
    handler can only be installed from the main thread -- neither is worth a
    word to a user, so both simply do nothing.
    """
    if sys.platform == "win32":
        return
    import signal  # noqa: PLC0415 — POSIX only, and only on this path

    def _stop(signum, frame):
        raise KeyboardInterrupt("the restart helper was asked to stop")

    try:
        signal.signal(signal.SIGTERM, _stop)
    except (OSError, ValueError):
        pass


def _run_pending_install(cmd: list, on_started=None) -> "tuple[int, list]":
    """Run the installer, echo every line, and keep the tail of what it said.

    stdout here IS the job log file -- the server redirected it when it
    spawned this process -- so echoing is what makes an install nobody watched
    reviewable afterwards.

    `CREATE_NO_WINDOW` because this helper was spawned DETACHED and has no
    console to lend: Windows gives such a child a console of its OWN, so a
    window sits over whatever the user is looking at for the length of the
    install -- and closing it sends CTRL_CLOSE_EVENT to `uv` mid-rewrite,
    which is the corruption the whole mechanism exists to prevent.
    `runner.creation_flags()` passes the same flag for live installs.

    *on_started* is called with the process the moment it exists, and before
    a single line is read: it is how the caller writes the installer's pid
    into the claim, which is the only thing that says "this restart is still
    happening" once the helper itself has been killed.
    """
    tail: list = []
    quiet: dict = {}
    if sys.platform == "win32":
        quiet["creationflags"] = subprocess.CREATE_NO_WINDOW
    proc = subprocess.Popen(
        cmd, cwd=BACKEND_DIR,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        env=_restart_child_env(),
        **quiet,
    )
    if on_started is not None:
        on_started(proc)
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip()
        print(line, flush=True)
        tail.append(line)
        if len(tail) > RESTART_LOG_TAIL_LINES:
            del tail[0]
    return proc.wait(), tail


def _write_restart_outcome(outcome_path: Path, data: "dict | None", *,
                           status: str, returncode, message: str,
                           log_tail: list, log_file: "Path | None" = None) -> dict:
    """Record how the job ended, where the server that comes back will look.

    The schema is `restart.write_last_restart`'s, and `status` and `message`
    are the contract the SPA reads -- so `message` is written for a person who
    was not watching, not for a log parser. `log_file` and `relaunch` are
    additions of this writer's; R1's reader ignores keys it does not know, on
    purpose, and the SPA reads neither.

    *outcome_path* is handed in rather than looked up: the helper knows which
    directory this job's claim came out of, and re-deriving it from the
    environment would let a `CODEFYUI_USER_DATA_DIR` that changed between the
    server's launch and this process file the report where nobody looks.

    Returns the record, so the caller can amend it once the relaunch has
    happened.
    """
    data = data or {}

    def _text(key: str) -> "str | None":
        value = data.get(key)
        return value if isinstance(value, str) else None

    record = {
        "schema": OUTCOME_SCHEMA,
        "job_id": _text("job_id"),
        "pack_id": _text("pack_id"),
        "kind": data.get("kind") if data.get("kind") in ("torch", "pip") else None,
        "status": status,
        "returncode": returncode,
        "message": message,
        "log_tail": "\n".join(log_tail),
        "log_file": str(log_file) if log_file is not None else None,
        # Filled in by `_note_relaunch` once there is an answer. Present and
        # null rather than absent, so "not recorded yet" and "an older dev.py
        # wrote this" stay different facts.
        "relaunch": None,
        "finished_at": _iso_now(),
    }
    if not _write_json_atomic(outcome_path, record):
        err("無法寫入重啟安裝的結果紀錄",
            "could not write the restart-install outcome record")
    return record


def _finish_pending_job(pending_path: Path, data: "dict | None", *, status: str,
                        returncode, message: str, log_tail: list,
                        log_file: "Path | None" = None) -> dict:
    """Record the outcome, then drop the claim. In that order: the claim is
    what stops a second install starting, and it must not be released before
    there is something to read in its place."""
    record = _write_restart_outcome(
        pending_path.parent / OUTCOME_FILE_NAME, data, status=status,
        returncode=returncode, message=message, log_tail=log_tail,
        log_file=log_file)
    try:
        pending_path.unlink(missing_ok=True)
    except OSError:
        warn("無法刪除待處理的重啟安裝紀錄",
             "could not delete the pending restart file")
    return record


def _note_relaunch(outcome_path: Path, record: dict, pid: "int | None") -> None:
    """Add the relaunch's own result to the record that was just written.

    `status` is deliberately left alone: the install really did succeed or
    fail, and overwriting that would destroy the one field the SPA reads to
    tell the user what happened to their package. What a failed relaunch adds
    is a second fact -- there is no server -- and the only person who can act
    on it is at a terminal, so it goes in `relaunch`, in `message`, and (via
    `log_file`) points at the log that says why.
    """
    record["relaunch"] = "ok" if pid is not None else "failed"
    if pid is None:
        record["message"] = (
            f"{record.get('message') or ''} — and the server could not be "
            f"started again; see {record.get('log_file')}").strip(" —")
    if not _write_json_atomic(outcome_path, record):
        err("無法更新重啟安裝的結果紀錄",
            "could not update the restart-install outcome record")


def _relaunch_server(launcher: list, relaunch_argv: list, log_path: Path) -> "int | None":
    """Start the server again, detached, and return its pid.

    Runs even when the install failed. A user who asked for a package and got
    no server back has lost more than the package: the runs they had queued,
    and the page they were looking at. Whatever went wrong is in the outcome
    record, which the server that comes back reads and shows.

    Detached the same two ways `start()` daemonises: no console to inherit and
    no console to be Ctrl-C'd through on Windows, its own session on POSIX. Its
    output goes to the job log FILE and never a pipe -- this process exits
    seconds later, and a pipe with nobody left to read it fills up and blocks
    the server mid-start.

    The environment is this process's, minus the stdlib pointers -- see
    `_restart_child_env`. Nearly all of it is carried over on purpose: it is
    what the original `cdui start` exported and the server handed down
    (`CODEFYUI_LAUNCHER`, `CODEFYUI_RELAUNCH_ARGV`, `CODEFYUI_OUTER_PYTHON`,
    `CODEFYUI_USER_DATA_DIR`). `start()` re-exports the first two from scratch
    anyway, but the user-data root is not one it can rederive -- and handing
    over a scrubbed environment is how the SECOND restart of a session finds
    no launcher and refuses.
    """
    cmd = [*launcher, "start", *relaunch_argv]
    detach: dict = {}
    if sys.platform == "win32":
        DETACHED_PROCESS = 0x00000008
        detach["creationflags"] = (subprocess.CREATE_NEW_PROCESS_GROUP
                                   | DETACHED_PROCESS)
    else:
        detach["start_new_session"] = True

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "ab") as log_file:
            proc = subprocess.Popen(
                cmd, cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log_file, stderr=subprocess.STDOUT,
                env=_restart_child_env(),
                **detach,
            )
    except OSError as exc:
        err(f"無法重新啟動伺服器：{exc}",
            f"could not relaunch the server: {exc}")
        return None
    print(t(f"    已重新啟動伺服器（PID {proc.pid}）",
            f"    relaunched the server (PID {proc.pid})"), flush=True)
    return proc.pid


def _run_pending_job(pending_path: Path) -> int:
    """Finish one restart-mode install, and say so on the way out.

    The closing line matters more than it looks: this runs detached, into a
    log file nobody is watching, and "the last thing in the log" is how a
    person tells an install that ended from one that was killed halfway.
    """
    code = _run_pending_steps(pending_path)
    print(t(f"=== 重啟安裝結束：離開碼 {code} ===",
            f"=== restart install finished: exit code {code} ==="), flush=True)
    return code


def _run_pending_steps(pending_path: Path) -> int:
    """The job itself. Returns the process exit code.

    0 installed, 1 the install did not happen or failed (and the server was
    started again anyway), 2 refused before anything ran.

    The relaunch is in a `finally` on purpose, and the refusal returns BEFORE
    that block is entered: those are the two halves of one rule. Once this
    helper has taken a server down it owes the user one back, whatever
    happened in between -- and a file that is not ours means it took nothing
    down and has nothing to give back.
    """
    data = _read_json_file(pending_path)
    problem = _validate_pending(data)
    if problem is not None:
        err(f"拒絕執行這個重啟安裝：{problem}",
            f"refusing to run this restart install: {problem}")
        if data is not None:
            # A file that could not even be parsed names no job, and writing
            # "refused" over `last_restart_job.json` would erase the report of
            # the last restart that DID run -- which is the thing the user is
            # most likely opening the panel to read. No job, no record.
            _write_restart_outcome(
                pending_path.parent / OUTCOME_FILE_NAME, data, status="failed",
                returncode=None, message=f"refused: {problem}", log_tail=[])
        # And the claim goes too. A refusal stamps no `helper_pid`, so a file
        # left here reads as "a helper is on its way" for the whole start
        # grace: the next `cdui start` stands down and tells the user a
        # server is coming from a helper that has already refused to run.
        # After the outcome record, for the same reason `_finish_pending_job`
        # writes in that order -- the claim is what stops a second install,
        # and it must not be released before there is something to read in
        # its place.
        try:
            pending_path.unlink(missing_ok=True)
        except OSError:
            warn("無法刪除這個被拒絕的重啟安裝紀錄",
                 "could not delete the refused restart-install record")
        return 2

    section(f"重啟安裝：{data.get('pack_id')}（{data['kind']}）",
            f"Restart install: {data.get('pack_id')} ({data['kind']})")
    job_id = data.get("job_id") if isinstance(data.get("job_id"), str) else ""
    log_path = _restart_log_file(job_id, pending_path.parent)

    # Claim the job by name, before the wait. `cdui start` reads this back to
    # tell a restart that is still working from one whose helper never
    # started -- and this is the only moment at which that can be written
    # safely: the server has been told to exit and is not coming back to its
    # own claim, and no second helper exists yet.
    data["helper_pid"] = os.getpid()
    if not _write_json_atomic(pending_path, data):
        warn("無法在重啟紀錄中登記 helper PID；"
             "這段期間的 cdui start 會以為安裝已中斷",
             "could not record this helper's PID in the pending file; "
             "a `cdui start` meanwhile will read the restart as abandoned")

    record: dict = {}

    def _finish(*, status: str, returncode, message: str, log_tail: list) -> None:
        """Write the outcome and drop the claim, keeping the record so the
        `finally` below can add what happened to the relaunch."""
        record.clear()
        record.update(_finish_pending_job(
            pending_path, data, status=status, returncode=returncode,
            message=message, log_tail=log_tail, log_file=log_path))

    try:
        how = _wait_for_server_exit(data["server_pid"])
        if how != "alive":
            _forget_stopped_server()
        if how == "exited":
            print(t("    伺服器已結束，開始安裝",
                    "    the server has exited; installing"), flush=True)
        elif how == "alive":
            warn("伺服器仍在執行；安裝可能會因檔案被佔用而失敗",
                 "the server is still running; the install may fail on "
                 "files it still holds open")

        shortfall = _restart_disk_shortfall(data["kind"])
        if shortfall is not None:
            err(f"磁碟空間不足，取消這次安裝：{shortfall}", shortfall)
            _finish(status="failed", returncode=None, message=shortfall,
                    log_tail=[])
            return 1

        cmd, why = _pending_install_cmd(data)
        if cmd is None:
            err(f"無法組出安裝指令：{why}",
                f"cannot build the install command: {why}")
            _finish(status="failed", returncode=None, message=why, log_tail=[])
            return 1

        print("    " + " ".join(cmd), flush=True)
        installer: list = []

        def _stamp(proc) -> None:
            """Name the installer in the claim, while it is running.

            `helper_pid` alone is not the answer to "is this restart still
            happening?": a helper that is killed does not take its `uv` with
            it, and that orphan keeps rewriting the venv. Both `cdui start`
            and the server read the claim as live while EITHER pid is.
            """
            installer.append(proc)
            data["installer_pid"] = proc.pid
            if not _write_json_atomic(pending_path, data):
                warn("無法在重啟紀錄中登記安裝程式 PID",
                     "could not record the installer's PID in the pending file")

        try:
            code, tail = _run_pending_install(cmd, on_started=_stamp)
        except OSError as exc:
            message = f"could not run the installer: {exc}"
            err(f"無法執行安裝程式：{exc}", message)
            _finish(status="failed", returncode=None, message=message,
                    log_tail=[])
            return 1
        except BaseException:
            # Ctrl-C, a SIGTERM (see `_raise_on_sigterm`), a closed console:
            # this process is going, and `uv` is not -- it is mid-way through
            # replacing the packages the next server has to import. Stop it,
            # say so where the panel and `cdui status` will read it, and let
            # the exception out: the `finally` below still relaunches, and a
            # server that comes back and reports beats none at all.
            for proc in installer:
                _terminate_pid(proc.pid)
            message = "the install was interrupted before it finished"
            err("安裝被中斷，已停止安裝程式", message)
            _finish(status="failed", returncode=None, message=message,
                    log_tail=[])
            raise

        if code == 0:
            _finish(status="ok", returncode=0,
                    message=f"{data.get('pack_id')} installed", log_tail=tail)
            return 0
        _finish(status="failed", returncode=code,
                message=f"the installer exited with {code}", log_tail=tail)
        return 1
    finally:
        pid = _relaunch_server(list(data["launcher"]),
                               list(data["relaunch_argv"]), log_path)
        if record:
            _note_relaunch(pending_path.parent / OUTCOME_FILE_NAME, record, pid)


def packs_run_pending() -> None:
    """`cdui packs-run-pending <pending_restart.json>` -- INTERNAL.

    Deliberately absent from the help text. It is started detached by a server
    that is about to exit (`app.core.packs.restart.spawn_helper`), never by a
    person: the file it is handed names a process to WAIT FOR, so running it
    by hand against a live server would wait two minutes and then stop it.

    In `_SKIP_VENV_EXEC` because it exists to rewrite that venv, and stdlib-only
    for the same reason -- for part of its run the environment it is installing
    into has no working torch in it.
    """
    # stdout here is a FILE, so it is block-buffered while stderr is not, and
    # a log whose warnings sort before the lines that caused them is a log
    # somebody debugs the wrong thing from. This is the only record of an
    # install nobody watched; make the two streams interleave truthfully.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass
    # Before anything is read or waited on: a signal that arrives while the
    # claim is being parsed has to land in the same place as one that
    # arrives during the install.
    _raise_on_sigterm()

    argv = sys.argv[2:]
    if len(argv) != 1:
        err(f"用法：cdui {HELPER_COMMAND} <pending_restart.json>",
            f"usage: cdui {HELPER_COMMAND} <pending_restart.json>")
        sys.exit(2)
    sys.exit(_run_pending_job(Path(argv[0])))


def _needs_uv_bootstrap(command: str) -> bool:
    """Should `__main__` install uv before running *command*?

    Every command but one. `_ensure_uv` exits(1) when it cannot bootstrap uv,
    and `packs-run-pending` must never exit before it has started the server
    again -- a box that lost uv between `cdui start` and the restart would
    otherwise lose its server for good, which is the one outcome this whole
    mechanism is built to prevent. The helper looks for uv itself and records
    "uv is not on PATH" as a failed job, and a failed job still relaunches.
    """
    return command != HELPER_COMMAND


def _print_restart_notice() -> None:
    """`cdui status`: one line for a restart that is pending or just ran.

    A restart-mode install is the one operation that happens while nobody can
    see it -- the server it was started from no longer exists, so there is no
    panel to report into. `cdui status` is where somebody looks when the page
    did not come back.

    Never raises: this is a dashboard.
    """
    try:
        pending_path = _pending_restart_file()
        pending = _read_json_file(pending_path)
        if pending is not None:
            # The SAME predicate `cdui start` decides on, so the two commands
            # cannot describe one file differently to one confused user.
            if _pending_state(pending_path, pending) == "finishing":
                # The word is `_pending_state`'s own, and the docs' -- three
                # places name these two states and a user comparing them
                # must not have to work out that "in progress" and
                # "finishing" were ever the same thing.
                _kv(t("重啟安裝", "Restart install"),
                    f"{YELLOW}● {t('收尾中', 'finishing')}{RESET}  "
                    f"{pending.get('pack_id')}  "
                    f"{DIM}{t('等待 PID', 'waiting for PID')} "
                    f"{pending.get('server_pid')}{RESET}")
            else:
                hint = t("下次 cdui start 會清掉它",
                         "the next `cdui start` clears it")
                _kv(t("重啟安裝", "Restart install"),
                    f"{RED}✗ {t('已中斷', 'abandoned')}{RESET}  "
                    f"{pending.get('pack_id')}  {DIM}{hint}{RESET}")

        last = _read_json_file(_last_restart_file())
        if last is None:
            return
        age = _iso_age_seconds(last.get("finished_at"))
        if age is None or age > RESTART_NOTICE_S:
            return
        failed = last.get("status") != "ok" or last.get("relaunch") == "failed"
        mark = f"{RED}✗{RESET}" if failed else f"{GREEN}✓{RESET}"
        _kv(t("上次重啟安裝", "Last restart"),
            f"{mark} {last.get('pack_id')}  {DIM}{last.get('message')}{RESET}")
        # Only when something went wrong: the log is thousands of lines of uv
        # output, and pointing at it after a clean install is noise.
        if failed and last.get("log_file"):
            _kv("", f"{DIM}{t('紀錄', 'log')} → {last['log_file']}{RESET}")
    except Exception:
        return


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
            capture_output=True, timeout=3, **_CONSOLE_TEXT_KW,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    gpus: list[dict] = []
    for line in (out.stdout or "").strip().splitlines():
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
        _kv("URL", _display_url(*_server_addr()))
        _kv(t("日誌", "Log"), str(SERVER_LOG))
    elif healthy:
        orphan = t("有伺服器回應，但非 cdui 背景啟動（無 PID 檔）",
                   "responding, but not a cdui background server (no PID file)")
        _kv(t("狀態", "State"), f"{YELLOW}● {orphan}{RESET}")
        _kv("URL", _display_url(*_server_addr()))
        if info:
            _kv(t("節點 / 預設", "Nodes / Presets"),
                f"{info.get('nodes_loaded', '?')} / {info.get('presets_loaded', '?')}")
    else:
        _kv(t("狀態", "State"),
            f"{GRAY}○ {t('未執行', 'not running')}{RESET}  "
            f"{DIM}{t('用 cdui start 啟動', 'start with: cdui start')}{RESET}")

    # Right here rather than in a section of its own: a restart-mode install
    # is a fact ABOUT the server line above it -- either why it is missing, or
    # what happened while it was.
    _print_restart_notice()

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
    project = _parse_project(sys.argv[2:])
    if project is not None:
        _activate_project(project)

    # `--reload` restarts the worker on every edit, so a pack install that
    # needs a restart is a different proposition here than under `cdui start`
    # — the panel tells the two apart by this marker (see start()).
    os.environ["CODEFYUI_MANAGED"] = "dev"
    uvicorn = _require_venv_tool("uvicorn")
    # Same WS ceiling as `cdui start` — a limit that only holds in production
    # is a limit developers discover from a bug report (core#274).
    backend_cmd = [uvicorn, "app.main:app", "--reload",
                   "--ws-max-size", str(_ws_max_size())]
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
    """停止此安裝的服務。加 --all 才會掃掉整台機器上的 CodefyUI / Vite。"""
    sweep_everything = "--all" in sys.argv[2:]
    print("=== 停止服務 ===")
    # First, stop the tracked background server gracefully via its PID. On
    # POSIX it was started with start_new_session, so its PID is also its
    # process-group leader — kill the whole group to catch any children.
    pid = _read_server_pid()
    if pid is not None and _pid_alive(pid):
        # Printing the stopped URL is a small NEW feature (stop printed no
        # URL before Stage 2), so later shells know what just went away.
        print(f"  停止背景伺服器（PID {pid}，{_display_url(*_server_addr())}）...")
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                           capture_output=True)
        else:
            _terminate_posix(pid)
    SERVER_PIDFILE.unlink(missing_ok=True)
    SERVER_ADDRFILE.unlink(missing_ok=True)

    # Then sweep up anything the pidfile does not know about — a foreground
    # `cdui start`, a `cdui dev` vite, a worker that outlived its parent.
    if sweep_everything:
        _sweep_every_install()
    else:
        _sweep_this_install(already_stopped=pid)
    print("=== 完成 ===")


def _sweep_this_install(already_stopped: "int | None" = None) -> None:
    """Stop leftover processes started FROM THIS CHECKOUT, and only those.

    The sweep exists for a real reason: a foreground start writes no pidfile,
    so without it `cdui stop` cannot stop what `cdui start -f` began. What it
    must not do is reach outside this install. On one laptop those are the
    same set; on a shared server they are not, and the untargeted version of
    this sweep ends everyone else's training — plus every unrelated Vite dev
    server on the box, which was never CodefyUI's to kill.

    Scoped by ownership rather than by name: a process counts only when it
    both looks like ours AND runs out of ``ROOT``.
    """
    strays = [p for p in _this_install_pids() if p != already_stopped]
    for stray in strays:
        print(t(f"  停止此安裝的殘留行程（PID {stray}）...",
                f"  Stopping leftover process from this install (PID {stray})..."))
        _terminate_pid(stray)
    if not _has_psutil():
        print(t("  （未安裝 psutil，無法辨識殘留行程；只停止了 pidfile 記錄的伺服器）",
                "  (psutil unavailable — leftover processes cannot be "
                "identified; only the pidfile server was stopped)"))


def _sweep_every_install() -> None:
    """`--all`: the pre-#250 machine-wide sweep, kept but no longer default.

    Still the right thing on a single-user machine whose pidfile has been
    lost, and the only thing that catches a server started from a checkout
    that no longer exists. It is opt-in because it cannot tell whose server
    it is stopping, and `pkill -f vite` is not even scoped to CodefyUI.
    """
    print(t("  --all：停止這台機器上所有 CodefyUI 與 Vite 行程（含其他使用者的）...",
            "  --all: stopping EVERY CodefyUI and Vite process on this "
            "machine (other people's included)..."))
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/IM", "uvicorn.exe"], capture_output=True)
        subprocess.run(["taskkill", "/F", "/FI", "WINDOWTITLE eq vite*"], capture_output=True)
    else:
        subprocess.run(["pkill", "-f", "uvicorn app.main:app"], capture_output=True)
        subprocess.run(["pkill", "-f", "vite"], capture_output=True)


def _under_root(path: "str | None") -> bool:
    """True when *path* is ROOT itself or lives inside it.

    ``normcase`` so Windows' case and separator variants compare equal, and
    the explicit ``+ os.sep`` so a sibling checkout called ``codefyui-old``
    is not mistaken for a child of ``codefyui``.
    """
    if not path:
        return False
    root = os.path.normcase(str(ROOT))
    candidate = os.path.normcase(str(path))
    return candidate == root or candidate.startswith(root + os.sep)


#: `vite` as a whole path component (``node_modules/vite/…``) or as the
#: executable itself (``…/vite``, ``…/vite.js``, ``…\vite.cmd``). Never a
#: bare substring: `invite.py` is not a dev server, and neither is a graph
#: file someone happened to name `vite.json`.
_VITE_RE = re.compile(
    r"(^|[\\/])vite[\\/]"
    r"|(^|[\\/])vite(\.(js|cjs|mjs|cmd|exe|bat|ps1))?$"
)


def _looks_like_a_codefyui_service(cmdline: "list[str]") -> bool:
    """Name-only test: is this the shape of a CodefyUI backend or a Vite?

    ``app.main:app`` is the uvicorn target every start path uses; ``vite``
    covers the dev-mode frontend. Says nothing about WHOSE it is — that is
    ``_is_this_install_process``'s job, and answering this one first keeps
    the ownership check (which costs a ``cwd()`` syscall) off every process
    on the machine.
    """
    parts = [os.path.normcase(part) for part in cmdline]
    return any("app.main:app" in part or _VITE_RE.search(part)
               for part in parts)


def _is_this_install_process(cmdline: "list[str]", cwd: "str | None") -> bool:
    """True for a CodefyUI backend / Vite dev server owned by THIS checkout.

    Two independent questions, both of which must answer yes: does it look
    like one of ours, and is it *this* install's? The second is satisfied
    either by a launch path inside ROOT (the venv's ``uvicorn`` and
    ``node_modules/vite`` both are) or by a working directory inside ROOT
    (``cdui dev`` starts the backend in ``backend/`` and the frontend in
    ``frontend/``).

    A pure function of what a process reports, so the matching rule can be
    tested without spawning anything.
    """
    if not _looks_like_a_codefyui_service(cmdline):
        return False
    return any(_under_root(part) for part in cmdline) or _under_root(cwd)


def _this_install_pids() -> "list[int]":
    """PIDs matching ``_is_this_install_process``, minus us and our parents.

    Needs psutil — a declared backend dependency, and ``stop`` runs inside
    the venv, so it is there in every supported install. When it is not, the
    list is empty and only the pidfile server is stopped: failing to stop a
    stray is recoverable, stopping the wrong process is not.
    """
    try:
        import psutil  # noqa: PLC0415 — optional, ships with the backend
    except ImportError:
        return []

    ours = {os.getpid()}
    try:
        ours.update(parent.pid for parent in psutil.Process().parents())
    except Exception:  # noqa: BLE001 — a missing ancestor must not stop the sweep
        pass

    found: list[int] = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            pid = proc.info["pid"]
            if pid in ours:
                continue
            cmdline = proc.info.get("cmdline") or []
            if not _looks_like_a_codefyui_service(cmdline):
                continue
            if _is_this_install_process(cmdline, None):
                found.append(pid)
                continue
            # The launch path did not settle ownership, so fall back to the
            # working directory — a second syscall, and one that is denied
            # for other users' processes, which is why it is asked last and
            # only of the handful that got this far.
            try:
                cwd = proc.cwd()
            except Exception:  # noqa: BLE001 — psutil raises several types
                continue
            if _is_this_install_process(cmdline, cwd):
                found.append(pid)
        except Exception:  # noqa: BLE001 — a process that vanished mid-scan
            continue
    return found


def _terminate_pid(pid: int) -> None:
    """Stop ONE process (plus its children on Windows).

    Deliberately not ``_terminate_posix``: that signals the whole process
    GROUP, which is correct for the background server (started with
    ``start_new_session``, so its group is exactly itself and its children)
    and wrong for a swept stray, whose group is whatever shell job launched
    it — potentially the caller's own script.
    """
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       capture_output=True,
                       # The restart helper calls this from a DETACHED
                       # process with no console of its own, and without
                       # this Windows opens one for `taskkill` -- a window
                       # over the user's screen at the one moment they are
                       # already waiting on a server that went away.
                       creationflags=subprocess.CREATE_NO_WINDOW)
        return
    import signal  # noqa: PLC0415 — only needed here, POSIX only
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    for _ in range(20):  # up to ~2s for a graceful shutdown
        if not _pid_alive(pid):
            return
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


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


# ── cdui run: submit a graph to the running server (#123) ────────────────────
#
# Deliberately a CLIENT, not a runner. `backend/run_graph.py` still executes a
# graph in-process with no server at all, and stays the answer for a box with
# no daemon. This command is the other half: the graph goes to the SERVER, so
# it joins the per-device queue, survives the terminal that submitted it, and
# shows up in the Runs panel next to everything the canvas started. Submitting
# five overnight jobs and closing the laptop is the workflow it exists for.

#: How long one long-poll of /events parks server-side. Five seconds keeps a
#: Ctrl+C responsive without turning the tail into a busy loop; the server
#: returns the moment an event lands, so this is a ceiling, not a cadence.
RUN_POLL_WAIT_S = 5.0

#: Statuses that mean the run is over. Mirrors run_store.TERMINAL_STATUSES —
#: not imported from it because this command must work when the backend
#: package cannot be imported (no venv yet), and the vocabulary is the REST
#: contract either way.
RUN_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled",
                                   "interrupted"})

#: Ceiling on the post-terminal drain (see ``_drain_run_events``).
RUN_DRAIN_MAX_PAGES = 20

#: Shell convention for "killed by SIGINT" (128 + 2). `cdui run --wait` uses
#: it so a Ctrl+C is distinguishable from a run that actually failed — the
#: run is still going on the server, and a script must not read that as one.
EXIT_INTERRUPTED = 130


def _parse_run_args(argv_tail: list, prog: str = "cdui run"):
    """Parse the flags of `cdui run`. Real argparse, like `cdui install`.

    Takes the tail as a PARAMETER rather than reading ``sys.argv`` so it can
    be tested without patching global state — the same reason
    ``_parse_install_args`` does.
    """
    p = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Submit a saved graph file to the running CodefyUI server. The "
            "run is owned by the server: it joins that device's queue, "
            "survives this terminal, and is visible in the Runs panel."
        ),
    )
    p.add_argument("graph", help="path to a graph .json file")
    p.add_argument("--name", default=None,
                   help="label for the run (shown in the Runs panel)")
    p.add_argument("--device", default="auto",
                   help="cpu | auto | cuda | cuda:N | mps (default: auto, "
                        "which the server currently resolves to cpu). The "
                        "RESOLVED device is the queue this run joins.")
    p.add_argument("--seed", type=int, default=None,
                   help="seed for random / numpy / torch. Every node is "
                        "seeded from it, and the run executes serially so "
                        "the same seed gives the same numbers.")
    p.add_argument("--deterministic", action="store_true",
                   help="also ask torch for deterministic kernels; ops with "
                        "no deterministic implementation warn instead of "
                        "failing the run")
    p.add_argument("--record-outputs", action="store_true",
                   help="capture node outputs for later inspection")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--wait", dest="wait", action="store_true", default=True,
                      help="stream progress and exit with the run (default)")
    mode.add_argument("--detach", dest="wait", action="store_false",
                      help="print the run id and exit 0 immediately")
    p.add_argument("--timeout", type=float, default=0.0,
                   help="give up waiting after N seconds (0 = no limit). The "
                        "run keeps going on the server.")
    p.add_argument("--host", default=None, help="server host (default: the "
                                                "last-started server)")
    p.add_argument("--port", type=int, default=None, help="server port")
    return p.parse_args(argv_tail)


def _session_token() -> "str | None":
    """The running server's session token, or None if it cannot be read.

    Same file, same env override and same failure mode as the plugin CLI's
    reader (``scripts/plugins.py``): a rotated-per-process 0600 file under the
    user-data dir. Duplicated rather than imported so `cdui run` does not drag
    the whole plugin CLI (and its imports) in for fifteen lines.

    Call ``_apply_dev_env()`` first — in a dev clone the token lives in
    ``<repo>/.codefyui_dev/`` and reading the global one would authenticate
    against a server that is not the one running.
    """
    try:
        from platformdirs import user_data_dir  # noqa: PLC0415 — needs venv
    except ImportError:
        return None
    override = os.environ.get("CODEFYUI_USER_DATA_DIR")
    base = (Path(override) if override
            else Path(user_data_dir("codefyui", appauthor=False)))
    try:
        return (base / "session.token").read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError):
        return None


def _api_request(url: str, host: str, *, token: "str | None" = None,
                 body: "dict | None" = None, timeout: float = 30.0) -> tuple:
    """One API call. Returns ``(status, parsed_body)``; ``(0, None)`` if down.

    The Host header is set explicitly because the server whitelists hosts and
    we always connect on loopback, which is always whitelisted — matching what
    the plugin and project CLIs already do. A 4xx/5xx comes back as a status
    plus its parsed ``detail`` rather than an exception, so callers report the
    server's own message instead of "HTTP Error 400: Bad Request".
    """
    headers = {"User-Agent": "cdui-run", "Host": host}
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(data))
    if token:
        headers["X-CodefyUI-Token"] = token
    req = Request(url, data=data, headers=headers,
                  method="POST" if body is not None else "GET")
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            return resp.status, (json.loads(raw) if raw else None)
    except HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8", "replace"))
        except (ValueError, OSError):
            return e.code, None
    except (URLError, TimeoutError, OSError, ValueError):
        return 0, None


def _run_submit_body(args) -> dict:
    """The POST /api/runs envelope for parsed args. Pure — hence testable.

    ``lane`` is left unset on purpose: the server's default IS the queued
    lane, and naming it here would hardcode a policy this command has no
    opinion about. ``interactive`` is the canvas's, and carries process-local
    state a CLI has none of.
    """
    graph = json.loads(Path(args.graph).read_text(encoding="utf-8"))
    options = {"device": args.device, "record_outputs": bool(
        getattr(args, "record_outputs", False))}
    if args.seed is not None:
        options["seed"] = args.seed
    if getattr(args, "deterministic", False):
        options["deterministic"] = True
    body = {"graph": graph, "options": options}
    if args.name:
        body["name"] = args.name
    return body


def _run_status_color(status: str) -> str:
    if status == "succeeded":
        return GREEN
    if status == "failed":
        return RED
    if status in ("cancelled", "interrupted"):
        return YELLOW
    return CYAN


def _render_run_event(event: dict) -> None:
    """Print one event from the run's log. ASCII + the house glyph set only."""
    kind = event.get("type")
    payload = event.get("payload") or {}
    if kind == "execution_start":
        print(f"  {CYAN}▸ {t('開始執行', 'started')}{RESET}")
    elif kind == "node_status":
        _render_node_status(payload)
    elif kind == "run_warning":
        print(f"  {YELLOW}! {payload.get('detail') or payload.get('kind')}"
              f"{RESET}")
    elif kind == "artifact":
        print(f"  {GRAY}+ {payload.get('kind')}: {payload.get('path')}{RESET}")
    elif kind == "execution_complete":
        print(f"  {GREEN}✓ {t('執行完成', 'run complete')}{RESET}")
    elif kind == "execution_error":
        print(f"  {RED}✗ {payload.get('error') or t('執行失敗', 'run failed')}"
              f"{RESET}")
    elif kind == "execution_stopped":
        print(f"  {YELLOW}○ {t('已停止', 'stopped')} "
              f"({payload.get('reason', '?')}){RESET}")


def _render_node_status(payload: dict) -> None:
    node = payload.get("node_id", "?")
    status = payload.get("status")
    if status == "progress":
        detail = _format_progress(payload)
        if detail:
            print(f"    {DIM}{node}{RESET}  {detail}")
    elif status == "completed":
        print(f"  {GREEN}✓{RESET} {node}")
    elif status == "cached":
        print(f"  {GRAY}= {node} {t('（快取）', '(cached)')}{RESET}")
    elif status == "skipped":
        print(f"  {GRAY}- {node} {t('（略過）', '(skipped)')}{RESET}")
    elif status == "error":
        print(f"  {RED}✗{RESET} {node}: {payload.get('error', '')}")


def _format_progress(payload: dict) -> str:
    """A progress payload as one compact line, loop counters first.

    Everything numeric that is not a loop counter is a measurement, which is
    the same rule the server uses to decide what becomes a metric series — so
    what the terminal shows and what the charts record cannot drift.
    """
    counters = []
    for key in ("epoch", "step", "batch"):
        value = payload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total = payload.get(f"total_{key}s")
            counters.append(f"{key} {value:g}"
                            + (f"/{total:g}" if isinstance(total, (int, float))
                               else ""))
    skip = {"epoch", "step", "batch", "total_epochs", "total_batches",
            "total_steps", "start_epoch", "event"}
    metrics = [f"{k}={v:.4g}" for k, v in payload.items()
               if k not in skip and isinstance(v, (int, float))
               and not isinstance(v, bool)]
    return "  ".join(counters + metrics)


def _tail_run(base: str, host: str, run_id: str, timeout: float) -> str:
    """Follow a run to its end, printing as it goes. Returns its final status.

    Long-polls ``/events`` from a cursor, so this is event-driven rather than
    a sleep loop — and the server parks a poll on a QUEUED run properly, so
    waiting for a device costs no requests. Returns the last status seen when
    *timeout* runs out: the run keeps going server-side, which is the point of
    a server-owned run, and the caller says so.
    """
    cursor = 0
    status = "queued"
    announced_queue = False
    deadline = (time.monotonic() + timeout) if timeout > 0 else None
    while True:
        code, body = _api_request(
            f"{base}/api/runs/{run_id}/events?cursor={cursor}"
            f"&wait={RUN_POLL_WAIT_S}", host,
            timeout=RUN_POLL_WAIT_S + 25.0)
        if code == 0 or body is None:
            err("與伺服器的連線中斷；run 仍在伺服器上執行",
                "lost the connection to the server; the run is still going")
            return status
        status = body.get("status", status)
        for event in body.get("events") or []:
            _render_run_event(event)
        cursor = body.get("cursor", cursor)
        if status in RUN_TERMINAL_STATUSES:
            _drain_run_events(base, host, run_id, cursor)
            return status
        if status == "queued" and not announced_queue:
            announced_queue = _announce_queue_position(base, host, run_id)
        if deadline is not None and time.monotonic() > deadline:
            warn(f"等待逾時（{timeout:g}s）；run 仍在伺服器上執行",
                 f"stopped waiting after {timeout:g}s; the run continues on "
                 "the server")
            return status


def _drain_run_events(base: str, host: str, run_id: str, cursor: int) -> None:
    """Print whatever is left in the log after the run reached a terminal.

    A page is bounded by BYTES as well as by ``limit``, so the poll that
    first saw the terminal status may have been cut short — and the event it
    cut is quite often the ``execution_error`` carrying the traceback, i.e.
    exactly the line the user ran the command to read. Terminal status makes
    the log final, so these polls do not wait: they read what is there and
    stop when a page comes back empty.

    Bounded so a server that somehow never stops producing cannot park the
    command here; the exit code is already decided by the status either way.
    """
    for _ in range(RUN_DRAIN_MAX_PAGES):
        code, body = _api_request(
            f"{base}/api/runs/{run_id}/events?cursor={cursor}&wait=0", host,
            timeout=30.0)
        events = (body or {}).get("events") or []
        if code != 200 or not events:
            return
        for event in events:
            _render_run_event(event)
        cursor = body.get("cursor", cursor)


def _announce_queue_position(base: str, host: str, run_id: str) -> bool:
    """Say where in line a queued run is. True once it has been reported."""
    code, body = _api_request(f"{base}/api/runs/{run_id}", host, timeout=10.0)
    if code != 200 or not isinstance(body, dict):
        return False
    position = body.get("queue_position")
    if position is None:
        return False
    device = body.get("queue_key") or "?"
    print(f"  {YELLOW}○ {t('排隊中', 'queued')}{RESET}  "
          f"{t('位置', 'position')} {position} {t('於', 'on')} {device}")
    return True


def run_graph() -> None:
    """`cdui run <graph.json>` — submit a graph to the running server.

    Named ``run_graph`` because ``run`` is this module's subprocess helper.
    Registered as ``COMMANDS["run"]``, like ``install_command``.
    """
    args = _parse_run_args(sys.argv[2:])
    _apply_dev_env()

    path = Path(args.graph)
    if not path.is_file():
        err(f"找不到圖檔：{path}", f"no such graph file: {path}")
        sys.exit(1)
    try:
        body = _run_submit_body(args)
    except (OSError, ValueError) as e:
        err(f"無法讀取圖檔：{e}", f"could not read the graph file: {e}")
        sys.exit(1)

    addr_host, addr_port = _server_addr()
    host = _probe_host(args.host if args.host is not None else addr_host)
    port = args.port if args.port is not None else addr_port
    base = f"http://{host}:{port}"
    netloc = f"{host}:{port}"

    token = _session_token()
    if token is None:
        err("找不到 session token — 伺服器沒有在執行？先執行 cdui start",
            "no session token found -- is the server running? Run `cdui start`")
        sys.exit(1)

    code, response = _api_request(f"{base}/api/runs", netloc, token=token,
                                  body=body)
    if code == 0:
        err(f"無法連線到 {base}", f"cannot reach {base}")
        sys.exit(1)
    if code != 200 or not isinstance(response, dict):
        detail = (response or {}).get("detail") if isinstance(response, dict) \
            else None
        err(f"提交失敗（HTTP {code}）：{detail or ''}",
            f"submit failed (HTTP {code}): {detail or ''}")
        sys.exit(1)

    run_id = response.get("run_id", "")
    status = response.get("status", "queued")
    print()
    section("提交執行", "Run submitted")
    _kv(t("Run ID", "Run ID"), run_id)
    _kv(t("圖檔", "Graph"), str(path))
    _kv(t("裝置", "Device"), args.device)
    if args.name:
        _kv(t("名稱", "Name"), args.name)
    _kv(t("狀態", "Status"),
        f"{_run_status_color(status)}{status}{RESET}")

    if not args.wait:
        print(f"  {DIM}{t('追蹤進度', 'follow it with')}: "
              f"cdui status  |  {base}{RESET}")
        return

    print()
    try:
        final = _tail_run(base, netloc, run_id, args.timeout)
    except KeyboardInterrupt:
        # Ctrl+C stops WATCHING, never the run — the whole point of a
        # server-owned run, and what the docs promise. Caught here rather
        # than left to the interpreter because the interrupt lands inside
        # ``urlopen`` and would otherwise print a socket traceback over the
        # progress output, which reads exactly like a crash.
        print()
        warn("已停止跟隨；run 仍在伺服器上執行",
             "stopped following; the run continues on the server")
        print(f"  {DIM}{t('重新跟隨', 'follow it again with')}: "
              f"cdui status  |  {base}{RESET}")
        sys.exit(EXIT_INTERRUPTED)
    print()
    _kv(t("結果", "Result"), f"{_run_status_color(final)}{final}{RESET}")
    if final != "succeeded":
        sys.exit(1)


def _run_status(cmd: list, cwd: Path) -> int:
    """Run *cmd* and return its exit code instead of raising.

    ``run()`` passes ``check=True``, so the first non-zero exit aborts the
    whole command. ``test`` deliberately wants the opposite: a red backend
    must not hide the frontend result, because learning about both halves in
    one pass is the entire point of running both.
    """
    # Our own headings go through Python's buffered stdout; the child writes
    # to the same fd directly. When stdout is a pipe rather than a terminal --
    # a file, a CI log -- the buffer is not flushed at each print, so
    # "=== Backend: pytest ===" lands AFTER pytest's own output and the
    # headings appear to label the wrong half. Observed, then fixed.
    sys.stdout.flush()
    sys.stderr.flush()
    if sys.platform == "win32":
        # Same reason as run(): Windows won't resolve pnpm.cmd without a shell.
        return subprocess.run(
            subprocess.list2cmdline(cmd), cwd=cwd, shell=True).returncode
    return subprocess.run(cmd, cwd=cwd).returncode


#: Marker for a half that was not run at all, as distinct from 0 (passed) or
#: any non-zero exit code (failed). ``None`` keeps the three states separate
#: in the summary, which is the thing the old command could not say.
_SKIPPED = None


def _run_backend_tests() -> int:
    pytest = _require_venv_tool("pytest")
    section("Backend：pytest", "Backend: pytest")
    return _run_status([pytest], cwd=BACKEND_DIR)


def _run_frontend_tests() -> "int | None":
    """Run ``pnpm test`` (vitest), or return ``_SKIPPED`` with a reason printed.

    CodefyUI has a deliberate no-Node install path — ``cdui install`` fetches
    ``frontend-dist.tar.gz`` from the release when pnpm is absent, so a
    perfectly healthy install can have no Node at all. Such a machine must
    still get a useful ``cdui test``, so a missing pnpm is a documented skip,
    never an error.
    """
    if not (FRONTEND_DIR / "package.json").exists():
        warn("找不到 frontend/package.json，略過前端測試",
             "frontend/package.json not found — skipping frontend tests")
        return _SKIPPED
    if not shutil.which("pnpm"):
        warn(
            "未偵測到 pnpm，略過前端測試（vitest）。安裝 Node.js 24+ 與 pnpm 後\n"
            "  即可在本機跑完整套測試；CI 的 frontend-build.yml 仍然會跑它們。",
            "pnpm not detected — skipping the frontend tests (vitest). Install\n"
            "  Node.js 24+ and pnpm to run the full suite locally; CI's\n"
            "  frontend-build.yml runs them either way.",
        )
        return _SKIPPED

    section("Frontend：pnpm test（vitest）", "Frontend: pnpm test (vitest)")
    if not (FRONTEND_DIR / "node_modules").exists():
        print(t("=== Frontend: 首次執行，安裝 node_modules ===",
                "=== Frontend: first run, installing node_modules ==="))
        code = _run_status(["pnpm", "install"], cwd=FRONTEND_DIR)
        if code != 0:
            err("pnpm install 失敗，無法執行前端測試",
                "pnpm install failed — cannot run the frontend tests")
            return code
    return _run_status(["pnpm", "test"], cwd=FRONTEND_DIR)


def _test_summary_line(label: str, code: "int | None") -> str:
    if code is _SKIPPED:
        return f"  {DIM}{label:<10}{RESET}{YELLOW}{t('略過', 'SKIPPED')}{RESET}"
    if code == 0:
        return f"  {DIM}{label:<10}{RESET}{GREEN}{t('通過', 'PASS')}{RESET}"
    return (f"  {DIM}{label:<10}{RESET}{RED}{t('失敗', 'FAIL')}{RESET}"
            f" {GRAY}(exit {code}){RESET}")


def test() -> None:
    """Run the backend suite and the frontend suite, and say which ones ran.

    Until core#245 this was ``pytest`` in ``backend/`` and nothing else, while
    138 frontend test files went unrun — so a contributor could see a green
    ``cdui test``, push, and fail ``pnpm test`` in CI. CONTRIBUTING.md points
    people at this command, so the name has to mean what it says.

    Both halves always run to completion; the exit code is non-zero if either
    failed. A skipped half is reported as skipped, never as a pass.

    ``--backend`` / ``--frontend`` narrow the run. They are selectors, so
    passing both is the same as passing neither: run everything. Unknown
    arguments are an error rather than being ignored -- the old command
    silently swallowed ``-k foo``, which looks exactly like a filter that
    worked.
    """
    args = sys.argv[2:]
    valid = {"--backend", "--frontend"}
    unknown = [a for a in args if a not in valid]
    if unknown:
        err(f"未知的參數：{' '.join(unknown)}。只接受 --backend / --frontend。",
            f"unrecognised argument(s): {' '.join(unknown)}. "
            "Only --backend / --frontend are accepted.")
        print(t("  要篩選個別測試請直接執行 pytest 或 pnpm test。",
                "  To filter individual tests, run pytest or pnpm test directly."),
              file=sys.stderr)
        sys.exit(2)
    selected = valid & set(args)
    want_backend = not selected or "--backend" in selected
    want_frontend = not selected or "--frontend" in selected

    backend_code: "int | None" = _run_backend_tests() if want_backend else _SKIPPED
    frontend_code: "int | None" = _run_frontend_tests() if want_frontend else _SKIPPED

    print()
    section("測試結果", "Test summary")
    print(_test_summary_line("backend", backend_code))
    print(_test_summary_line("frontend", frontend_code))
    print()

    failed = [c for c in (backend_code, frontend_code) if c is not _SKIPPED and c != 0]
    if failed:
        sys.exit(1)


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


def _codefyui_version() -> str:
    """Version string for `cdui --version`.

    Reads `backend/pyproject.toml` directly rather than importing
    `app.core.version`, because this must answer without the venv -- a
    half-finished install is one of the cases where someone needs the number.
    Falls back to the dist build stamp, which a no-Node install always has
    even when the checkout is missing.
    """
    try:
        text = (BACKEND_DIR / "pyproject.toml").read_text(encoding="utf-8")
        m = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
        if m:
            return m.group(1)
    except OSError:
        pass
    stamp = _read_build_stamp()
    if isinstance(stamp, dict) and stamp.get("tag"):
        return f"{stamp['tag']} (from dist build stamp)"
    return "unknown"


# ── Entry point ───────────────────────────────────────────────────────────────

COMMANDS = {
    "install": install_command,
    "update": update,
    "build": build,
    "dev": dev,
    "start": start,
    "status": status,
    # NOT `run`: that name is this module's subprocess helper. Same shim
    # pattern as install -> install_command.
    "run": run_graph,
    "stop": stop,
    "test": test,
    "clean": clean,
    "uninstall": uninstall,
    # Internal, and absent from `__doc__` on purpose: a server that is about
    # to exit starts this, handing it the file that names the process to wait
    # for. See `packs_run_pending`.
    HELPER_COMMAND: packs_run_pending,
}

# Commands that mutate or remove the venv must run from the outer interpreter,
# never from the venv's Python (Windows can't delete a running exe; update
# rewrites deps in-place). `packs-run-pending` replaces the venv's torch, which
# is the same problem: it is spawned with the interpreter `cdui start` recorded
# in OUTER_PYTHON_ENV and must stay on it.
_SKIP_VENV_EXEC = {"install", "update", "clean", "uninstall", HELPER_COMMAND}


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


def _dispatch_project_subcommand() -> int:
    """Hand off `cdui project <subcmd> ...` to scripts/project.py.

    Same venv hop as the plugin subgroup: project.py imports app.core.* so it
    must run inside the backend venv with token/env resolution matching the
    server (spec Section 5).
    """
    _exec_into_venv_if_available()
    _ensure_uv()
    _apply_dev_env()
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import project as project_cli  # noqa: PLC0415 — late import: needs venv
    return project_cli.main(sys.argv[2:])


def _dispatch_packs_subcommand() -> int:
    """Hand off `cdui packs <subcmd> ...` to scripts/packs.py.

    Same venv hop as the other subgroups, and it is load-bearing here:
    packs.py reads `app.core.packs` for the catalog and drives
    `flows.install_pack_live`, so the interpreter has to be the venv's before
    the import — an outer Python has no `app` to find.

    `backend/` joins `scripts/` on sys.path because that is where `app` lives
    when the backend is not installed as a package (a half-finished install is
    exactly the machine somebody reaches for the Package Center on).
    """
    _exec_into_venv_if_available()
    _ensure_uv()
    _apply_dev_env()
    for path in (Path(__file__).resolve().parent, BACKEND_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import packs as packs_cli  # noqa: PLC0415 — late import: needs venv
    return packs_cli.main(sys.argv[2:])


#: `cdui <group> <subcmd> ...` — commands that hand their tail to a sibling
#: script rather than to a function in this module. A table rather than a
#: chain of `if`s in `__main__`, because `__main__` is not importable: the
#: routing is what a test can reach, and a subcommand that is registered
#: nowhere is one nobody discovers until they type it.
SUBCOMMAND_GROUPS = {
    "plugin": _dispatch_plugin_subcommand,
    "project": _dispatch_project_subcommand,
    "packs": _dispatch_packs_subcommand,
}


def _subcommand_group(argv: list):
    """The sub-group dispatcher *argv* selects, or None when it selects none."""
    if len(argv) >= 2 and argv[1] in SUBCOMMAND_GROUPS:
        return SUBCOMMAND_GROUPS[argv[1]]
    return None


if __name__ == "__main__":
    # Before anything else, including the uv bootstrap: `--version` has to
    # answer on a broken or half-installed machine, because that is exactly
    # when someone is being asked which version they are on.
    if len(sys.argv) >= 2 and sys.argv[1] in ("--version", "-V", "version"):
        print(f"CodefyUI {_codefyui_version()}")
        sys.exit(0)

    # Long-form sub-grouped commands come first.
    _group = _subcommand_group(sys.argv)
    if _group is not None:
        sys.exit(_group())

    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    if sys.argv[1] not in _SKIP_VENV_EXEC:
        _exec_into_venv_if_available()
    if _needs_uv_bootstrap(sys.argv[1]):
        _ensure_uv()
    COMMANDS[sys.argv[1]]()
