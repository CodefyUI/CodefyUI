"""``cdui packs <subcommand>`` -- the Package Center from a terminal::

    cdui packs list                                  # what is in the catalog
    cdui packs status                                # ... plus torch and hints
    cdui packs install sentence-embeddings --yes     # install a whole pack
    cdui packs install sentence-embeddings --items all-MiniLM-L6-v2
    cdui packs remove word-vectors glove-50d         # delete one download

The in-app panel and this command are two front ends over ONE install:
``app.core.packs.flows.install_pack_live``. Nothing about what an install
does, in which order, or what a failure is called lives here -- this module
turns the flow's events into lines on a console and its exceptions into exit
codes, and that is all it does.

Which matters most for what this module deliberately CANNOT do. There is no
way to hand it a package spec, an index URL or a repo id: the only thing a
subcommand accepts is an id that has to be in ``catalog.CATALOG``, and the
catalog is the whole allowlist. This file spawns no processes at all, so
there is no path from an argument to a package manager to be found.

Exit codes, because scripts and CI read them:

* 0   -- done
* 1   -- the install failed (or the person said no at the prompt)
* 2   -- refused before anything ran: unknown id, wrong pack, unmet
         dependency, or no terminal to confirm at
* 3   -- this cannot be done while the server is running; the command to
         type instead is printed
* 130 -- cancelled with Ctrl-C, the shell convention for SIGINT

ASCII only. This prints on a legacy Windows console where a box-drawing
character or an emoji raises ``UnicodeEncodeError`` and takes the command
down with it.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import sys
from pathlib import Path

# NOTE: nothing from ``app`` is imported at module level, and nothing may be.
# ``cdui packs --help`` has to answer on a half-installed venv -- which is
# exactly the machine whose owner is reaching for the Package Center -- so
# every backend import lives inside the command that needs it.

# ── colour + i18n (self-contained, like scripts/plugins.py) ────────────────

_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
RESET = "\033[0m" if _USE_COLOR else ""
BOLD = "\033[1m" if _USE_COLOR else ""
DIM = "\033[2m" if _USE_COLOR else ""
RED = "\033[31m" if _USE_COLOR else ""
GREEN = "\033[32m" if _USE_COLOR else ""
YELLOW = "\033[33m" if _USE_COLOR else ""
CYAN = "\033[36m" if _USE_COLOR else ""


def _reconfigure_stdio() -> None:
    """Best-effort safety net so output never crashes on an unencodable char.

    Keeps the console's native encoding -- so Traditional Chinese still
    renders on a cp950 console -- but replaces anything it cannot encode
    rather than raising. Called from ``main`` so importing this module (in
    tests, say) leaves the captured stdio untouched.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError, OSError):
            pass


def _lang() -> str:
    forced = os.environ.get("CODEFYUI_LANG")
    if forced:
        return forced
    locale = (os.environ.get("LANG") or os.environ.get("LC_ALL") or "").lower()
    return "zh" if locale.startswith("zh") else "en"


def t(zh: str, en: str) -> str:
    return zh if _lang() == "zh" else en


def section(zh: str, en: str) -> None:
    print(f"{BOLD}{CYAN}=== {t(zh, en)} ==={RESET}")


def info(zh: str, en: str) -> None:
    print(f"  {DIM}{t(zh, en)}{RESET}")


def warn(zh: str, en: str) -> None:
    print(f"  {YELLOW}! {t(zh, en)}{RESET}")


def err(zh: str, en: str) -> None:
    print(f"  {RED}x {t(zh, en)}{RESET}", file=sys.stderr)


def ok(zh: str, en: str) -> None:
    print(f"  {GREEN}+ {t(zh, en)}{RESET}")


def raw_err(message: str) -> None:
    """Report a message that is already written: a backend failure carries
    its own words, and translating them here would put the installer's
    vocabulary in two places."""
    print(f"  {RED}x {message}{RESET}", file=sys.stderr)


def _mb(num_bytes: int | float | None) -> str:
    """Bytes as megabytes, the way the rest of the feature counts them
    (decimal MB -- the number on the download page, not the one in
    Explorer)."""
    return f"{(num_bytes or 0) / 1_000_000:.1f}"


def _stdin_is_interactive() -> bool:
    """Whether there is a human on the other end of ``input()``.

    A pytest run, a CI job and ``| tee`` all answer False, and all of them
    must take the non-interactive branch rather than blocking forever on a
    prompt nobody will see. On Windows the prompt is reachable at all only
    because ``dev.py`` re-enters the venv interpreter synchronously (see
    ``_reexec`` there): the in-place process replacement it used before
    orphaned the console, and every prompt read EOF.
    """
    try:
        return bool(sys.stdin) and sys.stdin.isatty()
    except (AttributeError, ValueError, OSError):
        return False


# ── rendering pack state ──────────────────────────────────────────────────


def _status_word(pack, probe) -> str:
    """``installed`` / ``partial`` / ``not installed`` for one pack.

    ``partial`` is not cosmetic: the four sentence-embedding models are
    alternatives, so a pack with one of them downloaded is a working pack
    with more available -- which is a different thing to say than either
    "installed" or "not installed".
    """
    if probe.installed:
        return "installed"
    if any(item.present for item in probe.items) or (pack.pip and probe.pip_ready):
        return "partial"
    return "not installed"


def _print_catalog(probed: dict) -> None:
    from app.core.packs import catalog

    packs = catalog.iter_packs()
    id_w = max(len(pack.pack_id) for pack in packs)
    title_w = max(len(pack.title) for pack in packs)
    item_w = max((len(item.item_id) for pack in packs for item in pack.items),
                 default=0)

    for pack in packs:
        probe = probed[pack.pack_id]
        print(f"  {pack.pack_id:<{id_w}}  {pack.title:<{title_w}}  "
              f"{_status_word(pack, probe)}")
        present = {item.item_id for item in probe.items if item.present}
        for item in pack.items:
            mark = "present" if item.item_id in present else "missing"
            print(f"    - {item.item_id:<{item_w}}  {mark:<7}  "
                  f"({_mb(item.approx_bytes)} MB, {item.license})")


def cmd_list(args: argparse.Namespace) -> int:
    from app.core.packs import state

    section("套件中心", "Package Center")
    _print_catalog(state.probe_all())
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from app.core.packs import catalog, restart, state

    probed = state.probe_all()
    section("套件中心", "Package Center")
    _print_catalog(probed)

    gpu = restart.gpu_info()
    print()
    section("PyTorch", "PyTorch")
    installed = gpu["installed_variant"]
    info(f"已安裝的版本：{installed or '無法判斷'}",
         f"installed variant : {installed or 'unknown'}")
    info(f"偵測到的裝置：{gpu['detected_label'] or '無法判斷'}",
         f"detected device   : {gpu['detected_label'] or 'unknown'}")
    info(f"建議的版本：{gpu['recommended_variant']}",
         f"recommended       : {gpu['recommended_variant']}")

    pending = [pack for pack in catalog.iter_packs()
               if not probed[pack.pack_id].installed]
    if pending:
        print()
        section("後續步驟", "Next steps")
        for pack in pending:
            # The command is spelled by the backend, so the panel and this
            # terminal never disagree about what to type.
            print(f"  {restart.install_command_for(pack, gpu['recommended_variant'])}")
    return 0


# ── install ───────────────────────────────────────────────────────────────


class _ConsoleReporter:
    """Turns install events into terminal output.

    Progress redraws ONE line (carriage return) instead of scrolling: a 470 MB
    download emits a frame every few hundred milliseconds, and a thousand
    lines of bar would bury the log lines that matter. Everything else prints
    normally, and any open bar is closed first so the two never collide.
    """

    BAR_WIDTH = 10

    def __init__(self) -> None:
        self._open = False
        self._width = 0

    def __call__(self, payload: dict) -> None:
        kind = payload.get("type")
        if kind == "progress":
            self._progress(payload)
            return
        self.close()
        if kind == "step_started":
            print(f"  {payload.get('label') or payload.get('step') or ''}")
        elif kind == "log":
            line = str(payload.get("line", "")).rstrip()
            if line:
                print(f"      {DIM}{line}{RESET}")
        # step_done needs no line of its own: closing the bar above is what
        # it means on a console.

    def close(self) -> None:
        """End the current progress line, if there is one."""
        if self._open:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._open = False
            self._width = 0

    def _progress(self, payload: dict) -> None:
        done = payload.get("bytes_done") or 0
        total = payload.get("bytes_total") or 0
        percent = payload.get("percent")
        if percent is None and total:
            percent = 100.0 * done / total

        if percent is None:
            bar = "?" * self.BAR_WIDTH
            pct = "  ?%"
        else:
            clamped = max(0.0, min(100.0, float(percent)))
            filled = int(round(self.BAR_WIDTH * clamped / 100.0))
            bar = "#" * filled + "." * (self.BAR_WIDTH - filled)
            pct = f"{clamped:>3.0f}%"

        # A frame that describes itself is not counting bytes. The GloVe
        # conversion counts LINES, and rendering 400,000 of those as
        # "0.4/0.4 MB" is not a rounding error -- it is a different quantity
        # with somebody else's unit on it. Such a frame gets its own words,
        # and only a frame without them gets the megabytes.
        text = str(payload.get("text") or "").strip()
        if text:
            tail = text
        else:
            sizes = (f"{_mb(done)}/{_mb(total)} MB" if total
                     else f"{_mb(done)} MB")
            tail = f"{sizes} {payload.get('item') or ''}"
        line = f"    [{bar}] {pct} {tail}".rstrip()
        # Pad to the previous width: a carriage return moves the cursor back
        # but leaves whatever was longer on the line behind it.
        sys.stdout.write("\r" + line + " " * max(0, self._width - len(line)))
        sys.stdout.flush()
        self._width = len(line)
        self._open = True


def _refuse_unknown_pack(pack_id: str) -> int:
    from app.core.packs import catalog

    err(f"未知的套件：{pack_id}", f"Unknown pack: {pack_id}")
    known = ", ".join(pack.pack_id for pack in catalog.iter_packs())
    info(f"已知的套件：{known}", f"Known packs: {known}")
    return 2


def _parse_items(pack, raw: str | None) -> list[str] | None:
    """``--items a,b`` as a list of item ids the pack actually declares.

    ``None`` means "the whole pack", which is what the flow reads as "every
    item that is not already here". Raises ``KeyError`` for an id the pack
    does not have -- checked here so a typo is refused before the disk check
    rather than after it.
    """
    from app.core.packs import catalog

    if raw is None:
        return None
    ids = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    for item_id in ids:
        catalog.get_item(pack, item_id)
    return ids


def _pending_items(pack, probe, item_ids: list[str] | None) -> list:
    """What this install would download, for the size prompt.

    Mirrors ``flows._resolve_items``: named items, or everything not already
    on disk. The flow decides again for itself from a fresh probe -- this
    copy only ever feeds a sentence shown to a human.
    """
    from app.core.packs import catalog

    if item_ids is not None:
        return [catalog.get_item(pack, item_id) for item_id in item_ids]
    present = {item.item_id for item in probe.items if item.present}
    return [item for item in pack.items if item.item_id not in present]


def _confirm(total_bytes: int) -> bool | None:
    """Ask before spending someone's bandwidth. ``None`` = nobody to ask.

    Fails closed with no terminal, like ``cdui plugin sync``: a CI job must
    not block on an ``input()`` nobody will answer, and must not decide yes on
    the user's behalf either.
    """
    def _no_terminal() -> None:
        err("沒有終端可確認。要直接安裝請加 --yes。",
            "No terminal to confirm at. Pass --yes to install.")

    if not _stdin_is_interactive():
        _no_terminal()
        return None
    prompt = t(f"要下載約 {_mb(total_bytes)} MB 嗎？",
               f"This downloads about {_mb(total_bytes)} MB. Continue?")
    try:
        answer = input(f"  {prompt} [y/N]: ").strip().lower()
    except EOFError:
        # isatty() said yes and stdin was closed anyway -- `< /dev/null` does
        # exactly this. Same fail-closed answer, because nothing was asked.
        print()
        _no_terminal()
        return None
    return answer in ("y", "yes")


def cmd_install(args: argparse.Namespace) -> int:
    from app.core.packs import catalog, flows, restart, state
    from app.core.packs.errors import (
        PackCancelled,
        PackInstallError,
        PackInsufficientDisk,
        PackNeedsRestart,
    )

    pack = catalog.find_pack(args.pack_id)
    if pack is None:
        return _refuse_unknown_pack(args.pack_id)

    if pack.install_mode == "restart":
        # Not something this command can do: the wheel it replaces is already
        # imported by whatever is running. The installer owns that swap.
        command = restart.install_command_for(pack)
        err(f"{pack.title} 不是用 cdui packs 安裝的",
            f"{pack.title} is not installed with cdui packs")
        info(f"{pack.title} 的切換方式：{command}",
             f"{pack.title} is switched with: {command}")
        return 2

    probe = state.probe_all()[pack.pack_id]
    if probe.blocked_by:
        err(f"{pack.title} 需要先安裝其他套件",
            f"{pack.title} needs another pack first")
        for dep in probe.blocked_by:
            print(f"  cdui packs install {dep}")
        return 2

    try:
        item_ids = _parse_items(pack, args.items)
    except KeyError:
        err(f"{pack.pack_id} 沒有你指定的項目：{args.items}",
            f"pack {pack.pack_id} has no such item: {args.items}")
        known = ", ".join(item.item_id for item in pack.items) or "(none)"
        info(f"可用的項目：{known}", f"Items in this pack: {known}")
        return 2

    total = sum(item.approx_bytes for item in _pending_items(pack, probe, item_ids))
    if total and not args.yes:
        try:
            answered = _confirm(total)
        except KeyboardInterrupt:
            print()
            return 130
        if answered is None:
            return 2
        if not answered:
            warn("已取消（沒有安裝任何東西）", "Cancelled (nothing was installed)")
            return 1

    section(f"安裝 {pack.title}", f"Installing {pack.title}")
    reporter = _ConsoleReporter()
    cancelled = {"requested": False}

    def _on_sigint(signum, frame) -> None:
        # Set a flag; never raise. The flow polls this between and during its
        # steps and unwinds through its OWN cancellation path, which removes
        # the half-written download -- a KeyboardInterrupt thrown from here
        # would skip that and print a traceback where "cancelled" belongs.
        cancelled["requested"] = True
        # End the progress line first: this fires mid-download, and the
        # message would otherwise land on top of the bar.
        reporter.close()
        warn("正在取消……（等目前的步驟收尾）",
             "Cancelling... (finishing the current step)")

    previous = None
    owns_sigint = False
    try:
        previous = signal.signal(signal.SIGINT, _on_sigint)
        owns_sigint = True
    except (AttributeError, OSError, ValueError):
        # Not the main thread, or a platform without SIGINT. Ctrl-C keeps
        # whatever behaviour it already had.
        pass

    try:
        outcome = flows.install_pack_live(
            pack, item_ids, emit=reporter,
            cancel_check=lambda: cancelled["requested"])
    except PackCancelled:
        reporter.close()
        warn("已取消", "Cancelled")
        return 130
    except PackNeedsRestart as exc:
        reporter.close()
        raw_err(str(exc))
        info(f"請改在終端機執行：{exc.command}",
             f"Run this in a terminal instead: {exc.command}")
        _print_hint(exc.hint)
        return 3
    except PackInsufficientDisk as exc:
        reporter.close()
        raw_err(str(exc))
        info(f"需要 {_mb(exc.needed)} MB，可用 {_mb(exc.free)} MB",
             f"needs {_mb(exc.needed)} MB, {_mb(exc.free)} MB free")
        _print_hint(exc.hint)
        return 1
    except PackInstallError as exc:
        reporter.close()
        raw_err(str(exc))
        _print_hint(exc.hint)
        return 1
    finally:
        reporter.close()
        if owns_sigint:
            signal.signal(signal.SIGINT,
                          previous if previous is not None else signal.SIG_DFL)

    ok(f"{pack.title} 安裝完成", f"{pack.title} is installed")
    if outcome.pip_installed:
        info("已安裝 Python 套件", "Python packages were installed")
    if outcome.items_done:
        done = ", ".join(outcome.items_done)
        info(f"已下載：{done}", f"Downloaded: {done}")
    return 0


def _print_hint(hint: str | None) -> None:
    """The operator-facing tail of a failure -- what to paste into an issue."""
    if not hint:
        return
    for line in str(hint).splitlines():
        if line.strip():
            print(f"      {DIM}{line}{RESET}", file=sys.stderr)


# ── remove ────────────────────────────────────────────────────────────────

#: A PEP 508 spec down to its distribution name: everything before the first
#: version marker, extra or comment. ``sentence-transformers>=3.0,<6`` names
#: ``sentence-transformers``, which is what an uninstall takes.
_DIST_NAME = re.compile(r"^[A-Za-z0-9._-]+")


def _dist_names(pack) -> list[str]:
    names = []
    for spec in pack.pip:
        match = _DIST_NAME.match(spec.strip())
        if match:
            names.append(match.group(0))
    return names


def cmd_remove(args: argparse.Namespace) -> int:
    from app.core.packs import catalog, flows, state

    pack = catalog.find_pack(args.pack_id)
    if pack is None:
        return _refuse_unknown_pack(args.pack_id)

    try:
        item = catalog.get_item(pack, args.item_id)
    except KeyError:
        err(f"{pack.pack_id} 沒有這個項目：{args.item_id}",
            f"pack {pack.pack_id} has no item {args.item_id}")
        known = ", ".join(one.item_id for one in pack.items) or "(none)"
        info(f"可用的項目：{known}", f"Items in this pack: {known}")
        return 2

    # Asked BEFORE the removal, because afterwards there is no way to tell
    # the two Falses apart: ``remove_item`` reports only whether bytes went,
    # and "there were none" is a different thing to tell somebody than
    # "something is holding them open".
    was_present = any(probe.item_id == args.item_id and probe.present
                      for probe in state.probe_all()[pack.pack_id].items)

    removed = flows.remove_item(pack, args.item_id)
    if removed:
        # Download PLUS what was derived from it: ``remove_item`` deletes the
        # GloVe npz along with the gz it came from, so reporting the gz alone
        # would under-count the space returned by more than half.
        freed = _mb(item.approx_bytes + item.derived_bytes)
        ok(f"已刪除 {item.item_id}（約釋出 {freed} MB）",
           f"Removed {item.item_id} (about {freed} MB freed)")
    elif not was_present:
        info(f"{item.item_id} 本來就沒有下載，沒有東西需要移除",
             f"{item.item_id} was not downloaded; nothing to remove")
    else:
        # Windows holds open files; the record is gone either way, so the
        # pack stops claiming a download it cannot use.
        warn(f"{item.item_id} 的紀錄已清除，但檔案仍在磁碟上（可能有程式正開著它）",
             f"{item.item_id} is no longer registered, but its files are "
             f"still on disk (something may be holding them open)")

    names = _dist_names(pack)
    if names:
        info("Python 套件不會一併移除。要移除的話：",
             "Python packages are not removed. To remove them:")
        print(f"      uv pip uninstall --python {sys.executable} {' '.join(names)}")
    return 0


# ── argparse routing ──────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """The whole CLI surface. Imports nothing from ``app``: ``--help`` has to
    answer on a machine whose venv is half-installed."""
    p = argparse.ArgumentParser(
        prog="cdui packs",
        description="Install the optional model packs the Package Center offers")
    sub = p.add_subparsers(dest="packs_cmd", required=True)

    p_list = sub.add_parser("list", help="List every pack and what is installed")
    p_list.set_defaults(_func=cmd_list)

    p_status = sub.add_parser(
        "status",
        help="Like list, plus the installed PyTorch build and what to run next")
    p_status.set_defaults(_func=cmd_status)

    p_install = sub.add_parser(
        "install",
        help="Install one pack from the catalog (no pip specs or URLs: the "
             "catalog is the allowlist)")
    p_install.add_argument("pack_id", help="a pack id from `cdui packs list`")
    p_install.add_argument(
        "--items", default=None,
        help="comma-separated item ids to download (default: everything the "
             "pack is missing)")
    p_install.add_argument("--yes", "-y", action="store_true",
                           help="skip the download-size confirmation")
    p_install.set_defaults(_func=cmd_install)

    p_remove = sub.add_parser(
        "remove", help="Delete one downloaded item and forget it")
    p_remove.add_argument("pack_id")
    p_remove.add_argument("item_id")
    p_remove.set_defaults(_func=cmd_remove)

    return p


def main(argv: list[str] | None = None) -> int:
    _reconfigure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args._func(args)


if __name__ == "__main__":
    # Direct runs need the backend on the path; `cdui packs` arrives here
    # already inside the venv, with dev.py having done this.
    _BACKEND = Path(__file__).resolve().parent.parent / "backend"
    if str(_BACKEND) not in sys.path:
        sys.path.insert(0, str(_BACKEND))
    sys.exit(main())
