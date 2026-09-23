"""Copy the canonical plugin contract into the vendored SDK type files.

The single source of truth for the plugin frontend API is
``frontend/src/plugins/contract.ts`` (guarded against the host's real
implementation types by ``contract.assert.ts``). Plugins, however, vendor their
own ``ui/src/sdk/types.ts`` (clone-and-own, no host import). This script stamps a
"generated" banner on the contract and writes it into every vendored copy that
ships inside this repo::

    python scripts/sync_plugin_sdk.py            # write the copies
    python scripts/sync_plugin_sdk.py --check    # exit 1 if any copy is stale

``--check`` runs in CI (``backend/tests/test_plugin_dx.py``) so a contract change
that forgets to re-sync fails the build instead of silently shipping stale types
in the ``cdui plugin new`` scaffold.

The template repository (CodefyUI/CodefyUI-Plugin-Official) vendors the whole
SDK, the React bindings in ``react.tsx`` and ``index.ts`` as well as
``types.ts``, and CI cannot see it. ``--template`` points this script at a
local checkout of it instead::

    python scripts/sync_plugin_sdk.py --template ../CodefyUI-Plugin-Official
    python scripts/sync_plugin_sdk.py --template ../CodefyUI-Plugin-Official --check

The checkout's ``ui/src/sdk/`` gets ``types.ts`` from the contract and every
other file from the scaffold's ``ui/src/sdk/``, the canonical copy of the
bindings; files the scaffold does not have (the template's own tests) are left
alone. Exit codes: 0 in step, 1 stale, 2 not a plugin checkout with a
``ui/src/sdk/``. It is a release step (``.github/RELEASING.md``), not a CI job.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT = REPO_ROOT / "frontend" / "src" / "plugins" / "contract.ts"

# Vendored copies inside this repo, kept byte-for-byte in sync with CONTRACT.
TARGETS = [
    REPO_ROOT / "scripts" / "templates" / "plugin" / "ui" / "src" / "sdk" / "types.ts",
]

# The scaffold's whole SDK, which a template checkout's ui/src/sdk/ mirrors.
SCAFFOLD_SDK = REPO_ROOT / "scripts" / "templates" / "plugin" / "ui" / "src" / "sdk"
# Where a template checkout keeps it, relative to the checkout's root.
TEMPLATE_SDK = Path("ui") / "src" / "sdk"

BANNER = (
    "// CodefyUI plugin SDK — type contract (mirrors the host's plugin API).\n"
    "// Generated from CodefyUI's frontend/src/plugins/contract.ts — do not edit\n"
    "// by hand; refresh it when you target a newer CodefyUI release.\n\n"
)


def _norm(text: str) -> str:
    """Normalize line endings so the LF/CRLF git checkout setting can't cause a
    spurious drift report."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def rendered() -> str:
    """The vendored copy: the plugin-facing BANNER plus the contract body, with
    the canonical file's host-internal header comment (which references
    contract.assert.ts / this script) stripped — a plugin author shouldn't see
    files they don't have."""
    body = CONTRACT.read_text(encoding="utf-8")
    body = re.sub(r"\A\s*/\*\*.*?\*/\s*", "", body, count=1, flags=re.DOTALL)
    return _norm(BANNER + body)


def check() -> int:
    """Return 0 when every vendored copy matches the canonical contract, else 1."""
    if not CONTRACT.exists():
        print(f"missing canonical contract: {CONTRACT}")
        return 1
    want = rendered()
    stale = [
        t for t in TARGETS
        if not t.exists() or _norm(t.read_text(encoding="utf-8")) != want
    ]
    for t in stale:
        print(f"stale: {t.relative_to(REPO_ROOT)}")
    if stale:
        print("Run: python scripts/sync_plugin_sdk.py")
        return 1
    return 0


def write() -> int:
    want = rendered()
    for t in TARGETS:
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_text(want, encoding="utf-8", newline="\n")
        print(f"wrote: {t.relative_to(REPO_ROOT)}")
    return 0


def template_files() -> dict[str, str]:
    """What a template checkout's ``ui/src/sdk/`` has to hold, by file name.

    ``types.ts`` comes from the contract rather than from the scaffold's copy,
    so the template is right even before the scaffold has been re-synced.
    """
    files = {
        p.name: _norm(p.read_text(encoding="utf-8"))
        for p in sorted(SCAFFOLD_SDK.iterdir())
        if p.is_file()
    }
    files["types.ts"] = rendered()
    return files


def _template_sdk(template: Path) -> Path | None:
    """The checkout's SDK directory, or ``None`` when *template* is not a
    plugin checkout that has one. Never created here: a mistyped path must
    not grow a ``ui/src/sdk/``."""
    sdk = template / TEMPLATE_SDK
    if (template / "cdui.plugin.toml").is_file() and sdk.is_dir():
        return sdk
    print(f"not a plugin checkout with a {TEMPLATE_SDK.as_posix()}/ directory: {template}")
    return None


def _stale_in(sdk: Path) -> dict[str, str]:
    return {
        name: want
        for name, want in template_files().items()
        if not (sdk / name).is_file()
        or _norm((sdk / name).read_text(encoding="utf-8")) != want
    }


def check_template(template: Path) -> int:
    """Return 0 when the checkout's SDK matches this repo's, 1 when it does
    not, 2 when *template* is not a plugin checkout."""
    sdk = _template_sdk(template)
    if sdk is None:
        return 2
    stale = _stale_in(sdk)
    for name in stale:
        print(f"stale: {(TEMPLATE_SDK / name).as_posix()}")
    if stale:
        print(f"Run: python scripts/sync_plugin_sdk.py --template {template}")
        return 1
    return 0


def write_template(template: Path) -> int:
    """Bring the checkout's SDK in step, writing only the files that differ."""
    sdk = _template_sdk(template)
    if sdk is None:
        return 2
    stale = _stale_in(sdk)
    for name, want in stale.items():
        (sdk / name).write_text(want, encoding="utf-8", newline="\n")
        print(f"wrote: {(TEMPLATE_SDK / name).as_posix()}")
    if not stale:
        print(f"{TEMPLATE_SDK.as_posix()}/ is already in step")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Sync vendored plugin SDK types from the canonical contract.",
    )
    ap.add_argument(
        "--check", action="store_true",
        help="exit 1 if any vendored copy is stale (does not write)",
    )
    ap.add_argument(
        "--template", type=Path, metavar="DIR",
        help="a local checkout of CodefyUI-Plugin-Official: sync (or, with "
             "--check, check) its whole ui/src/sdk/ instead of this repo's copy",
    )
    args = ap.parse_args(argv)
    if args.template is not None:
        return check_template(args.template) if args.check else write_template(args.template)
    return check() if args.check else write()


if __name__ == "__main__":
    sys.exit(main())
