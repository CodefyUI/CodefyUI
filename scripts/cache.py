"""``cdui cache <subcommand>`` -- the disk CodefyUI can rebuild, and how to
get it back::

    cdui cache list                      # what is cached, and how much of it
    cdui cache list --project ./lab      # ... of a project, not this install
    cdui cache prune                     # delete it (asks first)
    cdui cache prune --older-than 30     # ... only what has sat there a month
    cdui cache prune --yes               # ... without asking

A DERIVED cache is one whose every byte can be recomputed from inputs that
are still on disk, so deleting it costs CPU and nothing else. Today that is
``lm_blocks``: ``LMTokenizedDataset`` writes one packed token stream per
distinct (corpus, tokenizer, seq_len, append_eos, max_tokens) key, at 8 bytes
per token, and nothing has ever evicted one. Sweeping ``seq_len`` over three
values leaves three full copies of the same corpus -- around 800 MB apiece for
a 100M-token corpus (#306).

What is deliberately NOT in :data:`DERIVED_CACHES`, because ``prune`` deletes
everything the list names:

* the pack model cache (``<user data>/cache/hf``) and the pack state beside
  it -- a different root, and a DOWNLOAD rather than a derivation: deleting
  one costs bandwidth, which is the resource the machines this runs on have
  least of. ``cdui packs remove`` is that command;
* the downloaded assets (word vectors, tokenizer tables) beside them, for the
  same reason;
* run outputs, saved models, graphs and the database, none of which anything
  can rebuild.

Exit codes, because scripts and CI read them:

* 0   -- done (including "there was nothing to delete")
* 1   -- declined at the prompt, or something could not be deleted
* 2   -- refused before anything ran: a negative ``--older-than``, or no
         terminal to confirm at
* 3   -- a BACKGROUND server is running from this checkout; a graph in it may
         be reading a cache file, so nothing is deleted
* 130 -- cancelled with Ctrl-C

Background, and only background: the check is ``dev._running_server_pid``,
which reads the ``server.pid`` that solely the daemon branch of ``start``
writes. A foreground session (``cdui dev``, ``cdui start -f``) leaves no
pidfile anywhere, so nothing here can see one -- stop it yourself before
pruning, and say so wherever this refusal is described.

ASCII only. This prints on a legacy Windows console where a box-drawing
character or an emoji raises ``UnicodeEncodeError`` and takes the command
down with it.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

# NOTE: nothing from ``app`` is imported at module level, and nothing may be.
# ``cdui cache --help`` has to answer on a half-installed venv, and so does
# the refusal when a server is running.

# ---- colour + i18n (self-contained, like scripts/packs.py) ----------------

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
    """Report a message that is already written -- an OS error carries its
    own words, and translating them here would guess at what they said."""
    print(f"  {RED}x {message}{RESET}", file=sys.stderr)


# ---- the inventory -------------------------------------------------------

#: Every derived cache, in the order it prints: (directory name under
#: :func:`derived_cache_root`, (zh label, en label)).
#:
#: This tuple is the whole of what ``prune`` will delete, which is why the
#: test asserts it WHOLE rather than checking that ``lm_blocks`` is in it: a
#: list that can grow by accident is how a cleanup command ends up taking
#: somebody's downloads with it. See the module docstring for what stays out
#: and why.
DERIVED_CACHES = (
    ("lm_blocks", ("LMTokenizedDataset 打包好的 token 區塊",
                   "packed token blocks (LMTokenizedDataset)")),
)

#: Seconds in a day, for ``--older-than``.
DAY_S = 86400.0


def derived_cache_root() -> Path:
    """``<data root>/cache`` -- where nodes put what they can rebuild.

    Asked of ``app`` on every call rather than captured at import: project
    mode repoints ``MODELS_DIR`` during settings validation, and this command
    has to follow the cache the node is actually writing to.
    """
    from app.core.data_paths import data_root  # noqa: PLC0415 -- needs venv

    return data_root() / "cache"


@dataclass(frozen=True)
class Entry:
    """One thing ``prune`` would delete: a file, or a whole subdirectory."""

    path: Path
    size: int
    mtime: float


def _tree_size_and_mtime(path: Path) -> tuple[int, float]:
    """Bytes under *path*, and the newest mtime anywhere in it.

    An entry is usually one ``.pt`` file, but ``cache_dir`` on the node nests
    a directory per experiment, and that directory is one thing to keep or
    delete. The mtime is the newest file's rather than the directory's own,
    because a directory's timestamp does not move when a file inside it is
    rewritten -- which would make ``--older-than`` delete a cache that was
    refreshed yesterday.

    mtime, not atime: ``relatime`` and ``noatime`` are common enough that a
    read time cannot be trusted, so "old" here means LAST WRITTEN, and a
    long-lived entry that every run hits still counts as old.
    """
    try:
        stat = path.stat()
    except OSError:
        return 0, 0.0
    if not path.is_dir():
        return stat.st_size, stat.st_mtime
    total = 0
    newest = stat.st_mtime
    for parent, _dirs, files in os.walk(path, onerror=lambda _e: None):
        for name in files:
            try:
                child = os.stat(os.path.join(parent, name))
            except OSError:
                continue
            total += child.st_size
            newest = max(newest, child.st_mtime)
    return total, newest


def entries(directory: Path) -> list[Entry]:
    """One :class:`Entry` per direct child of *directory*, sorted by name.

    A missing or unreadable directory has no entries rather than being an
    error: a fresh install has never run the node that fills it.
    """
    try:
        children = sorted(directory.iterdir())
    except OSError:
        return []
    found = []
    for child in children:
        size, mtime = _tree_size_and_mtime(child)
        found.append(Entry(path=child, size=size, mtime=mtime))
    return found


def human_bytes(count: int) -> str:
    """``1610612736`` -> ``1.5 GB``. Binary units, the ones a file manager
    shows, so the number matches what somebody sees in their file browser."""
    size = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def _n_entries(n: int) -> str:
    """``1 entry`` / ``3 entries``, or the zh equivalent."""
    return t(f"{n} 個項目", f"{n} entry" if n == 1 else f"{n} entries")


def _pad(text: str, width: int) -> str:
    """Left-justify *text* to *width* terminal COLUMNS, not characters.

    A Chinese character occupies two columns, so ``ljust`` pads the zh
    rendering one column short per character and every column after it drifts
    right. Same rule as ``dev.py``'s ``_display_width``.
    """
    columns = sum(2 if unicodedata.east_asian_width(c) in ("F", "W") else 1
                  for c in text)
    return text + " " * max(0, width - columns)


def _row(name: str, label: str, found: list[Entry]) -> None:
    """One cache's line: what it is called, how much of it there is, what
    wrote it. The label is last because it is the only variable-width part."""
    total = sum(entry.size for entry in found)
    print(f"  {_pad(name, 11)} {_pad(_n_entries(len(found)), 11)} "
          f"{human_bytes(total):>9}  {label}")


def _server_pid() -> int | None:
    """The pid of a server started from this checkout, or None.

    ``dev.py`` owns this question: its answer also CLEARS a stale pidfile, so
    a second reading here would be a second thing to fix the day that file
    moves. Imported inside the call because ``--help`` must answer without
    ``scripts/`` having been put on the path by anything.
    """
    import dev  # noqa: PLC0415 -- see the docstring

    return dev._running_server_pid()  # noqa: SLF001 -- dev.py IS this CLI


def _activate_project(raw: str) -> None:
    """Point this command at the project at *raw*, the way ``start`` is.

    ``dev.py`` owns the flag: it resolves the directory, refuses with exit 1
    and one sentence when there is no ``codefyui.project.toml`` in it, echoes
    the resolved path, and exports ``CODEFYUI_PROJECT_DIR``. That variable is
    what ``app.config`` reads to repoint ``MODELS_DIR`` at
    ``<project>/assets/models`` -- and therefore what moves the cache this
    command measures. A second reading of the manifest here would be a second
    place to fix the day project mode changes shape, and a chance for the two
    of them to disagree about which directory the user meant.

    Called from :func:`main`, before either subcommand: ``settings`` is built
    once, at the first import of ``app.config``, which is the one
    :func:`derived_cache_root` does. Exporting after that is too late.
    """
    import dev  # noqa: PLC0415 -- see _server_pid

    dev._activate_project(raw)  # noqa: SLF001 -- dev.py IS this CLI


def _stdin_is_interactive() -> bool:
    """Whether there is a human on the other end of ``input()``.

    A pytest run, a CI job and ``| tee`` all answer False, and all of them
    must take the non-interactive branch rather than blocking forever on a
    prompt nobody will see.
    """
    try:
        return bool(sys.stdin) and sys.stdin.isatty()
    except (AttributeError, ValueError, OSError):
        return False


def _confirm(count: int, total_bytes: int) -> bool | None:
    """Ask before deleting. ``None`` = nobody to ask.

    Fails closed with no terminal, like ``cdui packs install``: a CI job must
    not block on an ``input()`` nobody will answer, and must not decide yes on
    the user's behalf either.
    """
    def _no_terminal() -> None:
        err("沒有終端可確認。要直接刪除請加 --yes。",
            "No terminal to confirm at. Pass --yes to delete.")

    if not _stdin_is_interactive():
        _no_terminal()
        return None
    prompt = t(f"要刪除 {_n_entries(count)}（{human_bytes(total_bytes)}）嗎？",
               f"Delete {_n_entries(count)} ({human_bytes(total_bytes)})?")
    try:
        answer = input(f"  {prompt} [y/N]: ").strip().lower()
    except EOFError:
        # isatty() said yes and stdin was closed anyway -- `< /dev/null` does
        # exactly this. Same fail-closed answer, because nothing was asked.
        print()
        _no_terminal()
        return None
    return answer in ("y", "yes")


def _delete(doomed: list[Entry]) -> tuple[int, int]:
    """Delete every entry. Returns (bytes freed, entries that would not go).

    One failure does not stop the rest: a single file held open by something
    outside this process should cost that file, not the whole clean-up.
    """
    freed = 0
    failed = 0
    for entry in doomed:
        try:
            if entry.path.is_dir():
                shutil.rmtree(entry.path)
            else:
                entry.path.unlink()
        except OSError as exc:
            failed += 1
            raw_err(f"{entry.path}: {exc}")
            continue
        freed += entry.size
    return freed, failed


# ---- subcommands ---------------------------------------------------------


def cmd_list(args: argparse.Namespace) -> int:
    root = derived_cache_root()
    section("衍生快取", "Derived caches")
    print(f"  {DIM}{root}{RESET}")
    total = 0
    for name, (zh, en) in DERIVED_CACHES:
        found = entries(root / name)
        total += len(found)
        _row(name, t(zh, en), found)
    if total:
        info("這些不會自動清除，要清請用：cdui cache prune",
             "Nothing deletes these automatically. Clear them: cdui cache prune")
    return 0


def cmd_prune(args: argparse.Namespace) -> int:
    if args.older_than is not None and args.older_than < 0:
        err("--older-than 不能是負數。", "--older-than cannot be negative.")
        return 2

    # Before anything is measured, let alone deleted: a graph running in that
    # server may be part-way through reading a block file, and losing it
    # mid-run is a failed training run rather than a slow one.
    pid = _server_pid()
    if pid is not None:
        err(f"伺服器還在執行（PID {pid}），現在不能清快取。",
            f"A server is running (PID {pid}); the cache cannot be cleared.")
        info("執行中的圖可能正在讀這些檔案。請先 cdui stop。",
             "A running graph may be reading these files. Run cdui stop first.")
        return 3

    root = derived_cache_root()
    cutoff = (None if args.older_than is None
              else time.time() - args.older_than * DAY_S)

    section("清除衍生快取", "Prune derived caches")
    doomed: list[Entry] = []
    kept = 0
    for name, (zh, en) in DERIVED_CACHES:
        found = entries(root / name)
        old = found if cutoff is None else [e for e in found if e.mtime < cutoff]
        kept += len(found) - len(old)
        doomed.extend(old)
        if old:
            _row(name, t(zh, en), old)

    if kept:
        days = args.older_than
        info(f"另外 {_n_entries(kept)} 比 {days:g} 天新，予以保留。",
             f"{_n_entries(kept)} newer than {days:g} days kept.")
    if not doomed:
        info("沒有需要刪除的東西。", "Nothing to delete.")
        return 0

    total = sum(entry.size for entry in doomed)
    if not args.yes:
        answered = _confirm(len(doomed), total)
        if answered is None:
            return 2
        if not answered:
            warn("已取消（沒有刪除任何東西）。", "Cancelled (nothing was deleted).")
            return 1

    freed, failed = _delete(doomed)
    ok(f"已刪除 {_n_entries(len(doomed) - failed)}，釋放 {human_bytes(freed)}。",
       f"deleted {_n_entries(len(doomed) - failed)}, freed {human_bytes(freed)}.")
    return 1 if failed else 0


# ---- argparse routing ----------------------------------------------------


def _add_project_flag(p: argparse.ArgumentParser) -> None:
    """``--project DIR``, on every subcommand.

    ``cdui start|dev --project <dir>`` puts the cache in
    ``<dir>/assets/cache`` and exports that inside its own process only, so
    a ``cdui cache`` typed in any shell had neither the flag nor the
    variable: it answered about ``backend/data/cache`` -- ``0 entries`` --
    while the copies filling the disk sat in the project the lesson was run
    from (#306). Spelled and resolved exactly like ``start``'s, because it
    is the same flag: somebody who started a server with it should not have
    to learn a second way to name the same directory.
    """
    p.add_argument(
        "--project", metavar="DIR", default=None,
        help="the project whose cache this is about, as passed to `cdui "
             "start --project`; its cache is <DIR>/assets/cache")


def build_parser() -> argparse.ArgumentParser:
    """The whole CLI surface. Imports nothing from ``app``: ``--help`` has to
    answer on a machine whose venv is half-installed."""
    p = argparse.ArgumentParser(
        prog="cdui cache",
        description="Show and clear the caches CodefyUI can rebuild from your "
                    "inputs")
    sub = p.add_subparsers(dest="cache_cmd", required=True)

    p_list = sub.add_parser(
        "list", help="What each derived cache holds, and how much disk it is on")
    _add_project_flag(p_list)
    p_list.set_defaults(_func=cmd_list)

    p_prune = sub.add_parser(
        "prune", help="Delete those entries (asks first)",
        # Raw, so the two command names below survive: the default formatter
        # re-wraps a description and would break `cdui start -f` over a line
        # end, which is where somebody copying it loses the flag.
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Delete every derived cache entry, after a [y/N] confirmation.\n"
            "\n"
            "Refused while a BACKGROUND server (`cdui start`) is running: a\n"
            "graph in it may be part-way through reading a block file. Only\n"
            "background -- a foreground server (`cdui dev`, `cdui start -f`)\n"
            "writes no pidfile and cannot be detected here, so stop one\n"
            "yourself before pruning."))
    p_prune.add_argument(
        "--older-than", type=float, default=None, metavar="DAYS",
        help="only delete entries last written more than DAYS days ago")
    p_prune.add_argument("--yes", "-y", action="store_true",
                         help="skip the confirmation prompt")
    _add_project_flag(p_prune)
    p_prune.set_defaults(_func=cmd_prune)

    return p


def main(argv: list[str] | None = None) -> int:
    _reconfigure_stdio()
    args = build_parser().parse_args(argv)
    if args.project:
        # Before the subcommand, and so before anything imports `app`: the
        # data root this measures is derived while `app.config` is first
        # imported, and only from the environment as it stands then.
        _activate_project(args.project)
    try:
        return args._func(args)
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == "__main__":
    # Direct runs need the backend and this directory on the path; `cdui
    # cache` arrives here already inside the venv, with dev.py having done it.
    _SCRIPTS = Path(__file__).resolve().parent
    for _path in (_SCRIPTS, _SCRIPTS.parent / "backend"):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))
    sys.exit(main())
