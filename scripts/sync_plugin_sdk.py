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

The clone-and-own template repo (``CodefyUI-Plugin-Official/ui/src/sdk/types.ts``)
lives in a separate repo this script can't see; refresh it from the same
``contract.ts`` whenever the contract changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT = REPO_ROOT / "frontend" / "src" / "plugins" / "contract.ts"

# Vendored copies inside this repo, kept byte-for-byte in sync with CONTRACT.
TARGETS = [
    REPO_ROOT / "scripts" / "templates" / "plugin" / "ui" / "src" / "sdk" / "types.ts",
]

BANNER = (
    "// AUTO-GENERATED from frontend/src/plugins/contract.ts — DO NOT EDIT.\n"
    "// Refresh with:  python scripts/sync_plugin_sdk.py\n\n"
)


def _norm(text: str) -> str:
    """Normalize line endings so the LF/CRLF git checkout setting can't cause a
    spurious drift report."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def rendered() -> str:
    return _norm(BANNER + CONTRACT.read_text(encoding="utf-8"))


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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Sync vendored plugin SDK types from the canonical contract.",
    )
    ap.add_argument(
        "--check", action="store_true",
        help="exit 1 if any vendored copy is stale (does not write)",
    )
    args = ap.parse_args(argv)
    return check() if args.check else write()


if __name__ == "__main__":
    sys.exit(main())
