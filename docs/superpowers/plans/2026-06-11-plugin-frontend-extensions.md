# Plugin Frontend Extensions Implementation Plan (Graph Copilot PR A)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let installed plugins ship a JavaScript bundle that the CodefyUI SPA loads at startup, receiving a stable `CodefyUIPluginAPI` object (floating-widget mount point, graph read/apply operations, authenticated fetch, toast, namespaced storage).

**Architecture:** Backend serves `<plugin>/frontend/` files through a dynamic route (no static mounts, so hot-install works) and exposes `frontend_entry` in `GET /api/plugins`. The SPA's new `PluginHost` component dynamically `import()`s each enabled plugin's entry and calls its default export with the API object. Graph mutations go through a pure reducer (`applyGraphOps`) committed as a single undo snapshot.

**Tech Stack:** FastAPI + pytest (backend), React 19 + Zustand + vitest (frontend), no new dependencies.

**Working directory:** `D:\Github\CodefyUI\.claude\worktrees\feat+plugin-frontend-extensions` (git branch `feat/plugin-frontend-extensions`). Backend commands run from `backend/` with `uv run`, frontend commands from `frontend/` with `pnpm`.

**Spec:** `docs/superpowers/specs/2026-06-11-graph-copilot-design.md` (Part A).

---

### Task 1: Backend — manifest `[frontend]` entry validation helper

**Files:**
- Modify: `backend/app/core/plugin_loader.py` (add function after `is_enabled`, ~line 107)
- Test: `backend/tests/test_plugin_frontend.py` (new file)

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_plugin_frontend.py`:

```python
"""Tests for plugin frontend-extension support (manifest validation,
bundle serving, /api/plugins frontend_entry field)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.plugin_loader import frontend_entry_rel


# ── frontend_entry_rel ──────────────────────────────────────────────────────

def test_entry_rel_returns_normalized_path():
    m = {"frontend": {"entry": "frontend/index.js"}}
    assert frontend_entry_rel(m) == "frontend/index.js"


def test_entry_rel_accepts_nested_path():
    m = {"frontend": {"entry": "frontend/dist/main.js"}}
    assert frontend_entry_rel(m) == "frontend/dist/main.js"


def test_entry_rel_none_when_table_missing():
    assert frontend_entry_rel({}) is None
    assert frontend_entry_rel({"plugin": {"id": "x"}}) is None


def test_entry_rel_none_when_entry_missing_or_not_string():
    assert frontend_entry_rel({"frontend": {}}) is None
    assert frontend_entry_rel({"frontend": {"entry": 3}}) is None
    assert frontend_entry_rel({"frontend": {"entry": ""}}) is None


def test_entry_rel_rejects_traversal_and_absolute():
    assert frontend_entry_rel({"frontend": {"entry": "frontend/../secrets.py"}}) is None
    assert frontend_entry_rel({"frontend": {"entry": "../frontend/index.js"}}) is None
    assert frontend_entry_rel({"frontend": {"entry": "/etc/passwd"}}) is None


def test_entry_rel_rejects_paths_outside_frontend_dir():
    assert frontend_entry_rel({"frontend": {"entry": "nodes/evil.js"}}) is None
    assert frontend_entry_rel({"frontend": {"entry": "frontend"}}) is None


def test_entry_rel_normalizes_backslashes():
    assert frontend_entry_rel({"frontend": {"entry": "frontend\\index.js"}}) == "frontend/index.js"
```

- [ ] **Step 2: Run tests to verify they fail**

Run (from `backend/`): `uv run pytest tests/test_plugin_frontend.py -q`
Expected: ImportError — `frontend_entry_rel` does not exist.

- [ ] **Step 3: Implement `frontend_entry_rel`**

In `backend/app/core/plugin_loader.py`, add `PurePosixPath` to the `pathlib` import (line 24: `from pathlib import Path, PurePosixPath`) and add after `is_enabled` (line 107):

```python
def frontend_entry_rel(manifest: dict[str, Any]) -> str | None:
    """Validated ``[frontend].entry`` path from a plugin manifest, or ``None``.

    The entry must be a relative POSIX-style path that stays inside the
    plugin's ``frontend/`` directory — anything else (traversal, absolute
    paths, other directories) is treated as "no frontend" rather than an
    error, so a malformed third-party manifest can't break startup.
    """
    fe = manifest.get("frontend")
    if not isinstance(fe, dict):
        return None
    entry = fe.get("entry")
    if not isinstance(entry, str) or not entry:
        return None
    p = PurePosixPath(entry.replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts:
        return None
    if p.parts[:1] != ("frontend",) or len(p.parts) < 2:
        return None
    return str(p)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_plugin_frontend.py -q`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/core/plugin_loader.py backend/tests/test_plugin_frontend.py
git commit -m "feat(plugins): validate [frontend].entry manifest field"
```

---

### Task 2: Backend — serve plugin frontend bundles

**Files:**
- Create: `backend/app/api/routes_plugin_frontend.py`
- Modify: `backend/app/main.py` (router registration block, lines 237–248)
- Test: `backend/tests/test_plugin_frontend.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_plugin_frontend.py`. The fixture mirrors `test_plugin_api.py`'s `direction_lockfile` pattern: redirect `plugins_user_root` to a tmp dir and write a deterministic lockfile, then build a fake installed plugin on disk.

```python
from app.core import plugin_loader


def _write_frontend_plugin(root: Path, plugin_id: str, *, enabled: bool = True,
                           with_entry: bool = True) -> None:
    """Create a fake installed third-party plugin with a frontend bundle."""
    pdir = root / plugin_id
    (pdir / "frontend").mkdir(parents=True)
    manifest = [
        "[plugin]",
        f'id = "{plugin_id}"',
        f'name = "{plugin_id}"',
        'version = "0.1.0"',
        "schema_version = 1",
    ]
    if with_entry:
        manifest += ["", "[frontend]", 'entry = "frontend/index.js"']
    (pdir / "cdui.plugin.toml").write_text("\n".join(manifest), encoding="utf-8")
    (pdir / "frontend" / "index.js").write_text(
        "export default function activate(api) {}", encoding="utf-8"
    )
    (pdir / "frontend" / "style.css").write_text(".x{}", encoding="utf-8")
    # A file OUTSIDE frontend/ that must never be reachable via the route.
    (pdir / "secret.txt").write_text("nope", encoding="utf-8")

    lock_path = root / "installed.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {
        "schema": 1, "plugins": {},
    }
    lock["plugins"][plugin_id] = {
        "source_kind": "github_url",
        "source": f"someone/{plugin_id}",
        "installed_at": "2026-06-11T00:00:00Z",
        "manifest": {"id": plugin_id, "name": plugin_id, "version": "0.1.0"},
        "trusted_modules": [],
        "enabled": enabled,
    }
    lock_path.write_text(json.dumps(lock), encoding="utf-8")


@pytest.fixture
def frontend_plugin_env(tmp_path, monkeypatch):
    root = tmp_path / "plugins"
    root.mkdir()
    _write_frontend_plugin(root, "fe-pack")
    _write_frontend_plugin(root, "fe-disabled", enabled=False)
    _write_frontend_plugin(root, "no-fe", with_entry=False)
    monkeypatch.setattr(plugin_loader, "plugins_user_root", lambda: root)
    yield root


@pytest.fixture
def fe_client(frontend_plugin_env):
    from app.config import settings
    from app.core.auth import TOKEN_HEADER, session_token
    from app.main import app
    from fastapi.testclient import TestClient
    with TestClient(app, base_url=f"http://127.0.0.1:{settings.PORT}") as c:
        c.headers[TOKEN_HEADER] = session_token()
        yield c


# ── GET /plugins/<id>/frontend/<path> ───────────────────────────────────────

def test_serves_frontend_js_with_module_mime(fe_client):
    r = fe_client.get("/plugins/fe-pack/frontend/index.js")
    assert r.status_code == 200
    assert "activate" in r.text
    assert r.headers["content-type"].startswith("text/javascript")


def test_serves_css(fe_client):
    r = fe_client.get("/plugins/fe-pack/frontend/style.css")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/css")


def test_404_for_file_outside_frontend_dir(fe_client):
    # Encoded traversal — TestClient does not collapse %2e%2e.
    r = fe_client.get("/plugins/fe-pack/frontend/%2e%2e/secret.txt")
    assert r.status_code == 404


def test_404_for_disabled_plugin(fe_client):
    r = fe_client.get("/plugins/fe-disabled/frontend/index.js")
    assert r.status_code == 404


def test_404_when_manifest_has_no_frontend_table(fe_client):
    r = fe_client.get("/plugins/no-fe/frontend/index.js")
    assert r.status_code == 404


def test_404_for_unknown_plugin_and_missing_file(fe_client):
    assert fe_client.get("/plugins/ghost/frontend/index.js").status_code == 404
    assert fe_client.get("/plugins/fe-pack/frontend/missing.js").status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_plugin_frontend.py -q`
Expected: the 7 Task-1 tests pass; the 6 new ones fail with 404-vs-200 mismatches or router-not-found (FastAPI returns 404 for unknown paths, so `test_serves_frontend_js_with_module_mime`, `test_serves_css` fail; the 404 tests may pass vacuously — that is fine, they pin behavior).

- [ ] **Step 3: Create the serving route**

Create `backend/app/api/routes_plugin_frontend.py`:

```python
"""Serve plugin-shipped frontend bundles.

Unlike the ``assets/`` StaticFiles mounts (created once at startup), this
route resolves the plugin directory on every request from the lockfile, so
a plugin installed while the server is running is servable immediately
after ``POST /api/plugins/reload`` — no restart, no remount.

Windows note: ``mimetypes`` can map ``.js`` to ``text/plain`` depending on
registry state, which makes browsers reject ESM imports. Media types for
the handful of bundle extensions are pinned explicitly.
"""

from __future__ import annotations

import sys
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..config import settings
from ..core.plugin_loader import (
    MANIFEST_FILENAME,
    frontend_entry_rel,
    iter_plugin_dirs,
    load_lockfile,
)

router = APIRouter(prefix="/plugins", tags=["plugin-frontend"])

_MEDIA_TYPES = {
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".css": "text/css",
    ".map": "application/json",
    ".json": "application/json",
}


def _read_manifest(plugin_dir) -> dict[str, Any]:
    try:
        return tomllib.loads(
            (plugin_dir / MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
    except (OSError, tomllib.TOMLDecodeError):
        return {}


@router.get("/{plugin_id}/frontend/{resource_path:path}")
async def serve_plugin_frontend(plugin_id: str, resource_path: str) -> FileResponse:
    lockfile = load_lockfile()
    for pid, plugin_dir in iter_plugin_dirs(
        settings.PLUGINS_BUILTIN_DIR, settings.PLUGINS_USER_DIR, lockfile
    ):
        if pid != plugin_id:
            continue
        if frontend_entry_rel(_read_manifest(plugin_dir)) is None:
            break
        base = (plugin_dir / "frontend").resolve()
        target = (base / resource_path).resolve()
        if not target.is_relative_to(base) or not target.is_file():
            break
        return FileResponse(
            target, media_type=_MEDIA_TYPES.get(target.suffix.lower())
        )
    raise HTTPException(status_code=404, detail="Plugin frontend resource not found")
```

In `backend/app/main.py`, add the import next to the other route imports and register it in the block at lines 237–248:

```python
app.include_router(routes_plugin_frontend.router)
```

(Match the local import style: the file imports route modules as `from .api import routes_nodes, ...` or individual imports — follow whatever pattern is at the top of `main.py`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_plugin_frontend.py -q`
Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/api/routes_plugin_frontend.py backend/app/main.py backend/tests/test_plugin_frontend.py
git commit -m "feat(plugins): serve plugin frontend bundles via dynamic route"
```

---

### Task 3: Backend — `frontend_entry` field in `GET /api/plugins`

**Files:**
- Modify: `backend/app/api/routes_plugins.py` (`list_plugins`, lines 62–99)
- Test: `backend/tests/test_plugin_frontend.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_plugin_frontend.py`:

```python
# ── /api/plugins frontend_entry ─────────────────────────────────────────────

def test_list_plugins_exposes_frontend_entry(fe_client):
    by_id = {p["id"]: p for p in fe_client.get("/api/plugins").json()}
    assert by_id["fe-pack"]["frontend_entry"] == "/plugins/fe-pack/frontend/index.js"


def test_list_plugins_frontend_entry_null_when_absent_or_disabled(fe_client):
    by_id = {p["id"]: p for p in fe_client.get("/api/plugins").json()}
    assert by_id["no-fe"]["frontend_entry"] is None
    assert by_id["fe-disabled"]["frontend_entry"] is None


def test_list_plugins_frontend_entry_null_when_file_missing(frontend_plugin_env):
    # Declared in manifest but the bundle file is gone.
    (frontend_plugin_env / "fe-pack" / "frontend" / "index.js").unlink()
    from app.config import settings
    from app.core.auth import TOKEN_HEADER, session_token
    from app.main import app
    from fastapi.testclient import TestClient
    with TestClient(app, base_url=f"http://127.0.0.1:{settings.PORT}") as c:
        c.headers[TOKEN_HEADER] = session_token()
        by_id = {p["id"]: p for p in c.get("/api/plugins").json()}
    assert by_id["fe-pack"]["frontend_entry"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_plugin_frontend.py -q`
Expected: 3 new failures (KeyError `frontend_entry`).

- [ ] **Step 3: Implement the field**

In `backend/app/api/routes_plugins.py`:

1. Extend the `plugin_loader` import (lines 25–32) with `frontend_entry_rel`.
2. In `list_plugins` (line 83 `out.append({...})`), compute before the append:

```python
        enabled = is_enabled(entry)
        entry_rel = frontend_entry_rel(manifest)
        frontend_entry = None
        if enabled and entry_rel and (plugin_dir / entry_rel).is_file():
            frontend_entry = f"/plugins/{plugin_id}/{entry_rel}"
```

and replace the existing `"enabled": is_enabled(entry),` line with `"enabled": enabled,`, adding the new key:

```python
            "enabled": enabled,
            "frontend_entry": frontend_entry,
```

- [ ] **Step 4: Run tests — new ones pass, no regressions**

Run: `uv run pytest tests/test_plugin_frontend.py tests/test_plugin_api.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add backend/app/api/routes_plugins.py backend/tests/test_plugin_frontend.py
git commit -m "feat(plugins): expose frontend_entry in GET /api/plugins"
```

---

### Task 4: CLI — trust notice when installing a frontend-shipping plugin

**Files:**
- Modify: `scripts/plugins.py` (`_install_github`)

No automated test (the scripts/ CLI has no pytest harness today); verified manually in Task 9's e2e and by the printed-output check below.

- [ ] **Step 1: Locate the insertion point**

Open `scripts/plugins.py`, find `_install_github` (~line 494). Locate the point right after the manifest is validated (`validate_manifest(...)` call) and before files are copied to the user plugins dir.

- [ ] **Step 2: Add the notice**

Immediately after the `validate_manifest` call in `_install_github`, insert (match the function's existing `print` style and indentation):

```python
    if frontend_entry_rel(manifest) is not None:
        print("NOTE: this plugin ships frontend UI code (JavaScript).")
        print("      After install it runs in your browser inside CodefyUI with")
        print("      full access to the editor UI. Only install plugins you trust.")
```

Import `frontend_entry_rel` from the backend loader the same way `scripts/plugins.py` already imports backend helpers — check the import block at the top of the file; if it cannot import `app.core.plugin_loader` (scripts run outside the backend package), duplicate the small validation inline instead:

```python
def _manifest_has_frontend(manifest: dict) -> bool:
    fe = manifest.get("frontend")
    return isinstance(fe, dict) and isinstance(fe.get("entry"), str) and bool(fe.get("entry"))
```

and gate the prints on `_manifest_has_frontend(manifest)`.

- [ ] **Step 3: Smoke-check by import**

Run (from repo root): `uv run --project backend python -c "import ast; ast.parse(open('scripts/plugins.py', encoding='utf-8').read()); print('OK')"`
Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add scripts/plugins.py
git commit -m "feat(cli): warn when an installed plugin ships frontend code"
```

---

### Task 5: Frontend — extract `buildFlowNode` from `tabStore.addNode`

**Files:**
- Modify: `frontend/src/utils/index.ts` (add export), `frontend/src/store/tabStore.ts` (`addNode`, lines 470–496)
- Test: `frontend/src/utils/index.test.ts` (append)

`applyGraphOps` (Task 6) must create nodes identically to `addNode` but without the per-node undo snapshot, so the construction moves to a shared helper.

- [ ] **Step 1: Write the failing tests**

Append to `frontend/src/utils/index.test.ts` (it already imports from `./index`; extend the import list with `buildFlowNode`):

```typescript
describe('buildFlowNode', () => {
  const def = {
    node_name: 'Conv2d',
    category: 'Layer',
    description: '',
    inputs: [],
    outputs: [],
    params: [
      { name: 'out_channels', param_type: 'int' as const, default: 32, description: '', options: [], min_value: 1, max_value: null },
    ],
  };

  it('builds a baseNode with default params and idle status', () => {
    const n = buildFlowNode(def, { x: 10, y: 20 });
    expect(n.type).toBe('baseNode');
    expect(n.position).toEqual({ x: 10, y: 20 });
    expect(n.data.type).toBe('Conv2d');
    expect(n.data.label).toBe('Conv2d');
    expect(n.data.params).toEqual({ out_channels: 32 });
    expect(n.data.definition).toBe(def);
    expect(n.data.executionStatus).toBe('idle');
    expect(n.id).toBeTruthy();
  });

  it('maps Start to the start renderer', () => {
    const n = buildFlowNode({ ...def, node_name: 'Start' }, { x: 0, y: 0 });
    expect(n.type).toBe('start');
  });

  it('strips plugin namespace when resolving the viz renderer', () => {
    // VIZ_NODE_TYPES is keyed by bare names; a namespaced plugin node must
    // still resolve (falls back to baseNode when no viz renderer exists).
    const n = buildFlowNode({ ...def, node_name: 'somepack:FancyNode' }, { x: 0, y: 0 });
    expect(n.type).toBe('baseNode');
    expect(n.data.type).toBe('somepack:FancyNode');
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run (from `frontend/`): `pnpm vitest run src/utils/index.test.ts`
Expected: FAIL — `buildFlowNode` is not exported.

- [ ] **Step 3: Implement and rewire `addNode`**

In `frontend/src/utils/index.ts`, add (near `resolveSerializedNodes`; reuse the existing imports — the file already has `generateId` and `VIZ_NODE_TYPES` locally):

```typescript
import type { Node } from '@xyflow/react';
import type { NodeData, NodeDefinition } from '../types';

export function buildFlowNode(
  definition: NodeDefinition,
  position: { x: number; y: number },
): Node<NodeData> {
  const defaultParams: Record<string, any> = {};
  for (const p of definition.params) {
    defaultParams[p.name] = p.default;
  }
  const name = definition.node_name;
  const bare = name.includes(':') ? name.slice(name.lastIndexOf(':') + 1) : name;
  return {
    id: generateId(),
    type: name === 'Start' ? 'start' : (VIZ_NODE_TYPES[bare] ?? 'baseNode'),
    position,
    data: {
      label: name,
      type: name,
      params: defaultParams,
      definition,
      executionStatus: 'idle',
    },
  };
}
```

(If `frontend/src/utils/index.ts` already imports these types under different aliases, reuse them — do not duplicate imports.)

In `frontend/src/store/tabStore.ts` replace the body of `addNode` (lines 470–496) with:

```typescript
  addNode: (definition, position) => {
    get().pushUndoSnapshot();
    const node = buildFlowNode(definition, position);
    set({
      tabs: updateTab(get().tabs, get().activeTabId, (tab) => ({
        nodes: [...tab.nodes, node],
      })),
    });
  },
```

and extend the line-4 import: `import { generateId, VIZ_NODE_TYPES, buildFlowNode } from '../utils';` (drop `VIZ_NODE_TYPES` from the import if nothing else in the file still uses it — check with search before removing).

Behavior note: `addNode` previously resolved the renderer with the namespaced name (`VIZ_NODE_TYPES[definition.node_name]`); `buildFlowNode` strips the `pack:` prefix the same way `resolveSerializedNodes` (line 208) does. This is a deliberate small fix, not a regression — keep it.

- [ ] **Step 4: Run the full frontend suite**

Run: `pnpm vitest run`
Expected: all pass (1573 baseline + 3 new).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/utils/index.ts frontend/src/utils/index.test.ts frontend/src/store/tabStore.ts
git commit -m "refactor(frontend): extract buildFlowNode for shared node construction"
```

---

### Task 6: Frontend — `applyGraphOps` pure reducer

**Files:**
- Create: `frontend/src/plugins/ops.ts`
- Test: `frontend/src/plugins/ops.test.ts` (new)

This is the heart of the plugin graph API: a pure function over `{nodes, edges}` so it unit-tests without React.

- [ ] **Step 1: Write the failing tests**

Create `frontend/src/plugins/ops.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import type { Node, Edge } from '@xyflow/react';
import type { NodeData, NodeDefinition } from '../types';
import { applyGraphOps, type GraphOp } from './ops';
import { buildFlowNode } from '../utils';

function def(name: string, overrides: Partial<NodeDefinition> = {}): NodeDefinition {
  return {
    node_name: name,
    category: 'Layer',
    description: '',
    inputs: [],
    outputs: [],
    params: [],
    ...overrides,
  };
}

const DEFS: NodeDefinition[] = [
  def('Source', {
    outputs: [{ name: 'out', data_type: 'TENSOR', description: '', optional: false }],
    params: [
      { name: 'size', param_type: 'int', default: 8, description: '', options: [], min_value: 1, max_value: 64 },
      { name: 'mode', param_type: 'select', default: 'a', description: '', options: ['a', 'b'], min_value: null, max_value: null },
    ],
  }),
  def('Sink', {
    inputs: [{ name: 'x', data_type: 'TENSOR', description: '', optional: false }],
  }),
  def('ModelSink', {
    inputs: [{ name: 'm', data_type: 'MODEL', description: '', optional: false }],
  }),
];

function run(ops: GraphOp[], nodes: Node<NodeData>[] = [], edges: Edge[] = []) {
  return applyGraphOps({ nodes, edges }, DEFS, ops);
}

describe('applyGraphOps — add_node', () => {
  it('adds a node with defaults and returns its id', () => {
    const r = run([{ op: 'add_node', node_type: 'Source', ref: 's' }]);
    expect(r.results[0]).toMatchObject({ ok: true });
    expect(r.nodes).toHaveLength(1);
    expect(r.refs.s).toBe(r.nodes[0].id);
    expect(r.nodes[0].data.params).toEqual({ size: 8, mode: 'a' });
    expect(r.mutated).toBe(true);
  });

  it('applies provided params and position', () => {
    const r = run([{ op: 'add_node', node_type: 'Source', params: { size: 16 }, position: { x: 5, y: 6 } }]);
    expect(r.nodes[0].data.params.size).toBe(16);
    expect(r.nodes[0].position).toEqual({ x: 5, y: 6 });
  });

  it('fails on unknown node type without adding', () => {
    const r = run([{ op: 'add_node', node_type: 'Nope' }]);
    expect(r.results[0].ok).toBe(false);
    expect(r.results[0].error).toContain('Unknown node type');
    expect(r.nodes).toHaveLength(0);
    expect(r.mutated).toBe(false);
  });

  it('fails on bad params (unknown name, range, options, type)', () => {
    const cases: Array<Record<string, unknown>> = [
      { ghost: 1 },
      { size: 0 },
      { size: 999 },
      { size: 'big' },
      { mode: 'z' },
    ];
    for (const params of cases) {
      const r = run([{ op: 'add_node', node_type: 'Source', params }]);
      expect(r.results[0].ok).toBe(false);
      expect(r.nodes).toHaveLength(0);
    }
  });
});

describe('applyGraphOps — connect', () => {
  it('connects two refs created in the same batch', () => {
    const r = run([
      { op: 'add_node', node_type: 'Source', ref: 'a' },
      { op: 'add_node', node_type: 'Sink', ref: 'b' },
      { op: 'connect', source: 'a', source_handle: 'out', target: 'b', target_handle: 'x' },
    ]);
    expect(r.results.map((x) => x.ok)).toEqual([true, true, true]);
    expect(r.edges).toHaveLength(1);
    expect(r.edges[0]).toMatchObject({ source: r.refs.a, target: r.refs.b, sourceHandle: 'out', targetHandle: 'x' });
  });

  it('rejects type-incompatible connections', () => {
    const r = run([
      { op: 'add_node', node_type: 'Source', ref: 'a' },
      { op: 'add_node', node_type: 'ModelSink', ref: 'b' },
      { op: 'connect', source: 'a', source_handle: 'out', target: 'b', target_handle: 'm' },
    ]);
    expect(r.results[2].ok).toBe(false);
    expect(r.results[2].error).toMatch(/TENSOR.*MODEL|incompatible/i);
    expect(r.edges).toHaveLength(0);
  });

  it('rejects unknown nodes, unknown ports, and duplicates', () => {
    const base: GraphOp[] = [
      { op: 'add_node', node_type: 'Source', ref: 'a' },
      { op: 'add_node', node_type: 'Sink', ref: 'b' },
    ];
    const dup = run([
      ...base,
      { op: 'connect', source: 'a', source_handle: 'out', target: 'b', target_handle: 'x' },
      { op: 'connect', source: 'a', source_handle: 'out', target: 'b', target_handle: 'x' },
    ]);
    expect(dup.results[3].ok).toBe(false);
    expect(dup.edges).toHaveLength(1);

    const ghost = run([{ op: 'connect', source: 'nope', source_handle: 'out', target: 'nope2', target_handle: 'x' }]);
    expect(ghost.results[0].ok).toBe(false);

    const badPort = run([
      ...base,
      { op: 'connect', source: 'a', source_handle: 'ghost', target: 'b', target_handle: 'x' },
    ]);
    expect(badPort.results[2].ok).toBe(false);
    expect(badPort.results[2].error).toContain('ghost');
  });
});

describe('applyGraphOps — set_params / remove_node / remove_edge / clear / layout', () => {
  function seeded() {
    const a = buildFlowNode(DEFS[0], { x: 0, y: 0 });
    const b = buildFlowNode(DEFS[1], { x: 100, y: 0 });
    const e: Edge = { id: 'e1', source: a.id, target: b.id, sourceHandle: 'out', targetHandle: 'x' };
    return { nodes: [a, b], edges: [e], a, b };
  }

  it('set_params merges valid values and reports invalid ones', () => {
    const { nodes, edges, a } = seeded();
    const ok = run([{ op: 'set_params', node_id: a.id, params: { size: 32 } }], nodes, edges);
    expect(ok.results[0].ok).toBe(true);
    expect(ok.nodes.find((n) => n.id === a.id)!.data.params.size).toBe(32);

    const bad = run([{ op: 'set_params', node_id: a.id, params: { size: -1 } }], nodes, edges);
    expect(bad.results[0].ok).toBe(false);
  });

  it('remove_node drops the node and its edges', () => {
    const { nodes, edges, a } = seeded();
    const r = run([{ op: 'remove_node', node_id: a.id }], nodes, edges);
    expect(r.results[0].ok).toBe(true);
    expect(r.nodes).toHaveLength(1);
    expect(r.edges).toHaveLength(0);
  });

  it('remove_edge matches by endpoints (handles optional)', () => {
    const { nodes, edges, a, b } = seeded();
    const r = run([{ op: 'remove_edge', source: a.id, target: b.id }], nodes, edges);
    expect(r.results[0].ok).toBe(true);
    expect(r.edges).toHaveLength(0);

    const miss = run([{ op: 'remove_edge', source: b.id, target: a.id }], nodes, edges);
    expect(miss.results[0].ok).toBe(false);
  });

  it('clear_graph empties everything; auto_layout repositions', () => {
    const { nodes, edges } = seeded();
    const cleared = run([{ op: 'clear_graph' }], nodes, edges);
    expect(cleared.nodes).toHaveLength(0);
    expect(cleared.edges).toHaveLength(0);

    const laid = run([{ op: 'auto_layout' }], nodes, edges);
    expect(laid.results[0].ok).toBe(true);
    expect(laid.nodes).toHaveLength(2);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pnpm vitest run src/plugins/ops.test.ts`
Expected: FAIL — module `./ops` does not exist.

- [ ] **Step 3: Implement the reducer**

Create `frontend/src/plugins/ops.ts`:

```typescript
/**
 * Pure graph-operation reducer behind CodefyUIPluginAPI.graph.applyOperations.
 *
 * Operates on copies of the active tab's nodes/edges and never touches the
 * store — the commit wrapper in ./api.ts handles the undo snapshot and the
 * store write. Failing ops are skipped (and reported) rather than aborting
 * the batch, so an agent driving this API can self-correct from per-op
 * errors.
 */
import type { Edge, Node } from '@xyflow/react';
import type { NodeData, NodeDefinition, ParamDefinition } from '../types';
import { buildFlowNode, generateId, isValidConnection } from '../utils';
import { autoLayout } from '../utils/autoLayout';

export type GraphOp =
  | { op: 'add_node'; node_type: string; ref?: string;
      params?: Record<string, unknown>; position?: { x: number; y: number } }
  | { op: 'connect'; source: string; source_handle: string;
      target: string; target_handle: string }
  | { op: 'set_params'; node_id: string; params: Record<string, unknown> }
  | { op: 'remove_node'; node_id: string }
  | { op: 'remove_edge'; source: string; target: string;
      source_handle?: string; target_handle?: string }
  | { op: 'clear_graph' }
  | { op: 'auto_layout' };

export interface OpResult {
  index: number;
  ok: boolean;
  error?: string;
  node_id?: string;
}

export interface ApplyOutcome {
  nodes: Node<NodeData>[];
  edges: Edge[];
  results: OpResult[];
  refs: Record<string, string>;
  dirtyIds: string[];
  mutated: boolean;
}

function validateParamValue(p: ParamDefinition, value: unknown): string | null {
  switch (p.param_type) {
    case 'int':
      if (typeof value !== 'number' || !Number.isInteger(value)) {
        return `param '${p.name}' expects an integer`;
      }
      break;
    case 'float':
      if (typeof value !== 'number' || Number.isNaN(value)) {
        return `param '${p.name}' expects a number`;
      }
      break;
    case 'bool':
      if (typeof value !== 'boolean') return `param '${p.name}' expects a boolean`;
      break;
    case 'select':
      if (typeof value !== 'string' || !p.options.includes(value)) {
        return `param '${p.name}' must be one of: ${p.options.join(', ')}`;
      }
      break;
    case 'string':
      if (typeof value !== 'string') return `param '${p.name}' expects a string`;
      break;
    default:
      // model_file / image_file / tensor_grid carry editor-managed payloads;
      // accept whatever the caller sends.
      return null;
  }
  if (typeof value === 'number') {
    if (p.min_value !== null && value < p.min_value) {
      return `param '${p.name}' must be >= ${p.min_value}`;
    }
    if (p.max_value !== null && value > p.max_value) {
      return `param '${p.name}' must be <= ${p.max_value}`;
    }
  }
  return null;
}

function validateParams(
  def: NodeDefinition,
  params: Record<string, unknown>,
): string | null {
  for (const [name, value] of Object.entries(params)) {
    const pd = def.params.find((p) => p.name === name);
    if (!pd) {
      const known = def.params.map((p) => p.name).join(', ') || '(none)';
      return `unknown param '${name}' for ${def.node_name}; known params: ${known}`;
    }
    const err = validateParamValue(pd, value);
    if (err) return err;
  }
  return null;
}

export function applyGraphOps(
  current: { nodes: Node<NodeData>[]; edges: Edge[] },
  definitions: NodeDefinition[],
  ops: GraphOp[],
): ApplyOutcome {
  let nodes = [...current.nodes];
  let edges = [...current.edges];
  const results: OpResult[] = [];
  const refs: Record<string, string> = {};
  const dirty = new Set<string>();
  let mutated = false;
  let staggered = 0;

  const defByName = new Map(definitions.map((d) => [d.node_name, d]));
  const resolveId = (idOrRef: string): string | null => {
    const viaRef = refs[idOrRef];
    if (viaRef && nodes.some((n) => n.id === viaRef)) return viaRef;
    return nodes.some((n) => n.id === idOrRef) ? idOrRef : null;
  };

  ops.forEach((op, index) => {
    const fail = (error: string) => results.push({ index, ok: false, error });

    switch (op.op) {
      case 'add_node': {
        const def = defByName.get(op.node_type);
        if (!def) {
          fail(`Unknown node type '${op.node_type}' — use exact names from the node catalog`);
          return;
        }
        if (op.params) {
          const err = validateParams(def, op.params);
          if (err) {
            fail(err);
            return;
          }
        }
        const position = op.position ?? { x: 160 + (staggered % 4) * 90, y: 120 + staggered * 70 };
        staggered += 1;
        const node = buildFlowNode(def, position);
        if (op.params) {
          node.data.params = { ...node.data.params, ...op.params };
        }
        nodes = [...nodes, node];
        if (op.ref) refs[op.ref] = node.id;
        dirty.add(node.id);
        mutated = true;
        results.push({ index, ok: true, node_id: node.id });
        return;
      }

      case 'connect': {
        const sourceId = resolveId(op.source);
        const targetId = resolveId(op.target);
        if (!sourceId) return fail(`connect: unknown source node '${op.source}'`);
        if (!targetId) return fail(`connect: unknown target node '${op.target}'`);
        const sourceNode = nodes.find((n) => n.id === sourceId)!;
        const targetNode = nodes.find((n) => n.id === targetId)!;
        if (sourceNode.type === 'noteNode' || targetNode.type === 'noteNode') {
          return fail('connect: note nodes cannot be connected');
        }

        const isTrigger = op.source_handle === 'trigger';
        const targetHandle = isTrigger ? '__trigger' : op.target_handle;

        if (!isTrigger) {
          const sDef = sourceNode.data.definition;
          const tDef = targetNode.data.definition;
          if (sDef) {
            const out = sDef.outputs.find((o) => o.name === op.source_handle);
            if (!out) {
              const names = sDef.outputs.map((o) => o.name).join(', ') || '(none)';
              return fail(`connect: '${sDef.node_name}' has no output '${op.source_handle}'; outputs: ${names}`);
            }
            if (tDef) {
              const inp = tDef.inputs.find((i) => i.name === op.target_handle);
              if (!inp) {
                const names = tDef.inputs.map((i) => i.name).join(', ') || '(none)';
                return fail(`connect: '${tDef.node_name}' has no input '${op.target_handle}'; inputs: ${names}`);
              }
              if (!isValidConnection(out.data_type, inp.data_type)) {
                return fail(`connect: incompatible types ${out.data_type} -> ${inp.data_type}`);
              }
            }
          }
        }

        const duplicate = edges.some(
          (e) => e.source === sourceId && e.target === targetId
            && (e.sourceHandle ?? '') === op.source_handle
            && (e.targetHandle ?? '') === targetHandle,
        );
        if (duplicate) return fail('connect: edge already exists');

        const edge: Edge = isTrigger
          ? { id: generateId(), source: sourceId, target: targetId,
              sourceHandle: 'trigger', targetHandle: '__trigger',
              animated: false, type: 'triggerEdge', data: { type: 'trigger' } }
          : { id: generateId(), source: sourceId, target: targetId,
              sourceHandle: op.source_handle, targetHandle,
              animated: false, style: { stroke: '#555', strokeWidth: 2 } };
        edges = [...edges, edge];
        dirty.add(targetId);
        mutated = true;
        results.push({ index, ok: true });
        return;
      }

      case 'set_params': {
        const id = resolveId(op.node_id);
        if (!id) return fail(`set_params: unknown node '${op.node_id}'`);
        const node = nodes.find((n) => n.id === id)!;
        const def = node.data.definition;
        if (def) {
          const err = validateParams(def, op.params);
          if (err) return fail(err);
        }
        nodes = nodes.map((n) =>
          n.id === id
            ? { ...n, data: { ...n.data, params: { ...n.data.params, ...op.params } } }
            : n,
        );
        dirty.add(id);
        mutated = true;
        results.push({ index, ok: true, node_id: id });
        return;
      }

      case 'remove_node': {
        const id = resolveId(op.node_id);
        if (!id) return fail(`remove_node: unknown node '${op.node_id}'`);
        nodes = nodes
          .filter((n) => n.id !== id)
          .map((n) =>
            n.type === 'noteNode' && n.data.boundToNodeId === id
              ? { ...n, data: { ...n.data, boundToNodeId: null, boundOffset: null } }
              : n,
          );
        edges = edges.filter((e) => e.source !== id && e.target !== id);
        mutated = true;
        results.push({ index, ok: true });
        return;
      }

      case 'remove_edge': {
        const sourceId = resolveId(op.source);
        const targetId = resolveId(op.target);
        if (!sourceId || !targetId) {
          return fail('remove_edge: unknown source or target node');
        }
        const matches = edges.filter(
          (e) => e.source === sourceId && e.target === targetId
            && (op.source_handle === undefined || (e.sourceHandle ?? '') === op.source_handle)
            && (op.target_handle === undefined || (e.targetHandle ?? '') === op.target_handle),
        );
        if (matches.length === 0) return fail('remove_edge: no matching edge');
        const drop = new Set(matches.map((e) => e.id));
        edges = edges.filter((e) => !drop.has(e.id));
        mutated = true;
        results.push({ index, ok: true });
        return;
      }

      case 'clear_graph': {
        nodes = [];
        edges = [];
        for (const k of Object.keys(refs)) delete refs[k];
        mutated = true;
        results.push({ index, ok: true });
        return;
      }

      case 'auto_layout': {
        nodes = autoLayout(nodes, edges, 'all') as Node<NodeData>[];
        mutated = true;
        results.push({ index, ok: true });
        return;
      }

      default:
        fail(`Unknown op '${(op as { op?: string }).op}'`);
    }
  });

  return {
    nodes,
    edges,
    results,
    refs,
    dirtyIds: [...dirty].filter((id) => nodes.some((n) => n.id === id)),
    mutated,
  };
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pnpm vitest run src/plugins/ops.test.ts`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/plugins/ops.ts frontend/src/plugins/ops.test.ts
git commit -m "feat(frontend): pure graph-operation reducer for the plugin API"
```

---

### Task 7: Frontend — plugin API object (`buildPluginAPI`) + commit wrapper

**Files:**
- Create: `frontend/src/plugins/api.ts`
- Test: `frontend/src/plugins/api.test.ts` (new)

- [ ] **Step 1: Write the failing tests**

Create `frontend/src/plugins/api.test.ts` (store-reset pattern mirrors `tabStore.test.ts`):

```typescript
import { describe, it, expect, beforeEach } from 'vitest';
import { useTabStore } from '../store/tabStore';
import { useNodeDefStore } from '../store/nodeDefStore';
import { buildPluginAPI } from './api';
import type { NodeDefinition } from '../types';

const DEFS: NodeDefinition[] = [
  {
    node_name: 'Source', category: 'Layer', description: '',
    inputs: [],
    outputs: [{ name: 'out', data_type: 'TENSOR', description: '', optional: false }],
    params: [],
  },
  {
    node_name: 'Sink', category: 'Layer', description: '',
    inputs: [{ name: 'x', data_type: 'TENSOR', description: '', optional: false }],
    outputs: [], params: [],
  },
];

function freshApi() {
  return buildPluginAPI('test-plugin', () => document.createElement('div'));
}

beforeEach(() => {
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  useTabStore.getState().addTab('test');
  useNodeDefStore.setState({ definitions: DEFS });
  window.localStorage.clear();
});

describe('graph surface', () => {
  it('applyOperations commits as a single undo step', () => {
    const api = freshApi();
    const result = api.graph.applyOperations([
      { op: 'add_node', node_type: 'Source', ref: 'a' },
      { op: 'add_node', node_type: 'Sink', ref: 'b' },
      { op: 'connect', source: 'a', source_handle: 'out', target: 'b', target_handle: 'x' },
    ]);
    expect(result.results.every((r) => r.ok)).toBe(true);
    expect(result.node_count).toBe(2);
    expect(result.edge_count).toBe(1);

    const tab = useTabStore.getState().getActiveTab();
    expect(tab.nodes).toHaveLength(2);
    expect(tab.edges).toHaveLength(1);

    useTabStore.getState().undo();
    const after = useTabStore.getState().getActiveTab();
    expect(after.nodes).toHaveLength(0);
    expect(after.edges).toHaveLength(0);
  });

  it('does not push an undo snapshot when nothing mutates', () => {
    const api = freshApi();
    const before = useTabStore.getState().getActiveTab().undoStack.length;
    api.graph.applyOperations([{ op: 'add_node', node_type: 'Ghost' }]);
    expect(useTabStore.getState().getActiveTab().undoStack.length).toBe(before);
  });

  it('getGraph returns the serialized active tab', () => {
    const api = freshApi();
    api.graph.applyOperations([{ op: 'add_node', node_type: 'Source' }]);
    const g = api.graph.getGraph();
    expect(g.nodes).toHaveLength(1);
    expect(g.nodes[0].type).toBe('Source');
  });

  it('getNodeDefinitions returns the store definitions', () => {
    expect(freshApi().graph.getNodeDefinitions()).toEqual(DEFS);
  });

  it('onGraphChanged fires on graph mutations and unsubscribes cleanly', () => {
    const api = freshApi();
    let calls = 0;
    const off = api.graph.onGraphChanged(() => { calls += 1; });
    api.graph.applyOperations([{ op: 'add_node', node_type: 'Source' }]);
    expect(calls).toBeGreaterThan(0);
    const seen = calls;
    off();
    api.graph.applyOperations([{ op: 'add_node', node_type: 'Sink' }]);
    expect(calls).toBe(seen);
  });
});

describe('storage surface', () => {
  it('namespaces keys per plugin', () => {
    const api = freshApi();
    api.storage.set('conversations', '[]');
    expect(window.localStorage.getItem('plugin:test-plugin:conversations')).toBe('[]');
    expect(api.storage.get('conversations')).toBe('[]');
    api.storage.remove('conversations');
    expect(api.storage.get('conversations')).toBeNull();
  });
});

describe('meta', () => {
  it('exposes apiVersion and pluginId', () => {
    const api = freshApi();
    expect(api.apiVersion).toBe(1);
    expect(api.pluginId).toBe('test-plugin');
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pnpm vitest run src/plugins/api.test.ts`
Expected: FAIL — module `./api` does not exist.

- [ ] **Step 3: Implement the API object**

Create `frontend/src/plugins/api.ts`:

```typescript
/**
 * CodefyUIPluginAPI — the object handed to every plugin frontend entry.
 *
 * This is a public, versioned surface: changing or removing anything here
 * breaks installed plugins. Add, don't mutate; bump apiVersion on breaking
 * changes.
 */
import { useTabStore } from '../store/tabStore';
import { useNodeDefStore } from '../store/nodeDefStore';
import { useToastStore } from '../store/toastStore';
import { apiFetch } from '../api/_auth';
import type { NodeDefinition } from '../types';
import { applyGraphOps, type ApplyOutcome, type GraphOp, type OpResult } from './ops';

export interface ApplyResult {
  results: OpResult[];
  refs: Record<string, string>;
  node_count: number;
  edge_count: number;
}

export type SerializedGraph = ReturnType<
  ReturnType<typeof useTabStore.getState>['getSerializedGraph']
>;

export interface CodefyUIPluginAPI {
  apiVersion: 1;
  pluginId: string;
  ui: {
    addFloatingWidget(opts: { id: string }): HTMLElement;
    toast(message: string, type?: 'info' | 'success' | 'error' | 'warning'): void;
  };
  graph: {
    getGraph(): SerializedGraph;
    getNodeDefinitions(): NodeDefinition[];
    applyOperations(ops: GraphOp[]): ApplyResult;
    onGraphChanged(cb: () => void): () => void;
  };
  http: {
    fetch(url: string, init?: RequestInit): Promise<Response>;
  };
  storage: {
    get(key: string): string | null;
    set(key: string, value: string): void;
    remove(key: string): void;
  };
}

export function commitGraphOperations(ops: GraphOp[]): ApplyResult {
  const store = useTabStore.getState();
  const tab = store.getActiveTab();
  const definitions = useNodeDefStore.getState().definitions;
  const outcome: ApplyOutcome = applyGraphOps(
    { nodes: tab.nodes, edges: tab.edges },
    definitions,
    ops,
  );
  if (outcome.mutated) {
    store.pushUndoSnapshot();
    store.setNodes(outcome.nodes);
    store.setEdges(outcome.edges);
    for (const id of outcome.dirtyIds) {
      useTabStore.getState().markDirty(id);
    }
  }
  return {
    results: outcome.results,
    refs: outcome.refs,
    node_count: outcome.nodes.length,
    edge_count: outcome.edges.length,
  };
}

function subscribeGraphChanged(cb: () => void): () => void {
  let prevTabId = useTabStore.getState().activeTabId;
  let prevTab = useTabStore.getState().tabs.find((t) => t.id === prevTabId);
  return useTabStore.subscribe((state) => {
    const tab = state.tabs.find((t) => t.id === state.activeTabId);
    const changed =
      state.activeTabId !== prevTabId
      || tab?.nodes !== prevTab?.nodes
      || tab?.edges !== prevTab?.edges;
    prevTabId = state.activeTabId;
    prevTab = tab;
    if (changed) cb();
  });
}

export function buildPluginAPI(
  pluginId: string,
  getWidgetContainer: (id: string) => HTMLElement,
): CodefyUIPluginAPI {
  const ns = (key: string) => `plugin:${pluginId}:${key}`;
  return {
    apiVersion: 1,
    pluginId,
    ui: {
      addFloatingWidget: ({ id }) => getWidgetContainer(id),
      toast: (message, type = 'info') =>
        useToastStore.getState().addToast(message, type),
    },
    graph: {
      getGraph: () => useTabStore.getState().getSerializedGraph(),
      getNodeDefinitions: () => useNodeDefStore.getState().definitions,
      applyOperations: (ops) => commitGraphOperations(ops),
      onGraphChanged: (cb) => subscribeGraphChanged(cb),
    },
    http: {
      fetch: (url, init) => apiFetch(url, init),
    },
    storage: {
      get: (key) => window.localStorage.getItem(ns(key)),
      set: (key, value) => window.localStorage.setItem(ns(key), value),
      remove: (key) => window.localStorage.removeItem(ns(key)),
    },
  };
}
```

Check `useToastStore`'s `addToast` signature in `frontend/src/store/toastStore.ts` before finishing — if the type union differs (e.g. includes `'warning'` or uses a different order), align the `toast` wrapper's type with it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pnpm vitest run src/plugins/`
Expected: ops + api suites pass.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/plugins/api.ts frontend/src/plugins/api.test.ts
git commit -m "feat(frontend): CodefyUIPluginAPI object with batched graph commits"
```

---

### Task 8: Frontend — `PluginHost` loader + widget stack + app wiring

**Files:**
- Create: `frontend/src/plugins/PluginHost.tsx`, `frontend/src/plugins/PluginHost.module.css`
- Modify: `frontend/src/App.tsx` (render `<PluginHost />`), `frontend/vite.config.ts` (proxy `/plugins`)
- Test: `frontend/src/plugins/PluginHost.test.tsx` (new)

- [ ] **Step 1: Write the failing test**

Create `frontend/src/plugins/PluginHost.test.tsx`. Loading real ESM via dynamic `import()` is not unit-testable in jsdom, so the unit test pins the fetch/activation control flow through the exported `loadPluginFrontends` with an injected importer; the in-browser path is covered by the Task 9 e2e.

```typescript
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { loadPluginFrontends } from './PluginHost';
import { useNodeDefStore } from '../store/nodeDefStore';

beforeEach(() => {
  useNodeDefStore.setState({
    definitions: [{
      node_name: 'X', category: 'c', description: '',
      inputs: [], outputs: [], params: [],
    }],
  });
});

function mockPluginsResponse(plugins: unknown[]) {
  vi.stubGlobal('fetch', vi.fn(async () => ({
    ok: true,
    json: async () => plugins,
  })) as unknown as typeof fetch);
}

describe('loadPluginFrontends', () => {
  it('activates enabled plugins with a frontend entry', async () => {
    mockPluginsResponse([
      { id: 'a', enabled: true, frontend_entry: '/plugins/a/frontend/index.js' },
      { id: 'b', enabled: true, frontend_entry: null },
      { id: 'c', enabled: false, frontend_entry: '/plugins/c/frontend/index.js' },
    ]);
    const activate = vi.fn();
    const importer = vi.fn(async () => ({ default: activate }));
    const loaded = await loadPluginFrontends(
      () => document.createElement('div'), importer,
    );
    expect(importer).toHaveBeenCalledTimes(1);
    expect(importer).toHaveBeenCalledWith('/plugins/a/frontend/index.js');
    expect(activate).toHaveBeenCalledTimes(1);
    expect(activate.mock.calls[0][0].pluginId).toBe('a');
    expect(loaded).toEqual(['a']);
  });

  it('isolates a failing plugin without breaking the rest', async () => {
    mockPluginsResponse([
      { id: 'bad', enabled: true, frontend_entry: '/plugins/bad/frontend/index.js' },
      { id: 'good', enabled: true, frontend_entry: '/plugins/good/frontend/index.js' },
    ]);
    const activate = vi.fn();
    const importer = vi.fn(async (url: string) => {
      if (url.includes('bad')) throw new Error('boom');
      return { default: activate };
    });
    const loaded = await loadPluginFrontends(
      () => document.createElement('div'), importer,
    );
    expect(loaded).toEqual(['good']);
    expect(activate).toHaveBeenCalledTimes(1);
  });

  it('rejects entries whose default export is not a function', async () => {
    mockPluginsResponse([
      { id: 'a', enabled: true, frontend_entry: '/plugins/a/frontend/index.js' },
    ]);
    const importer = vi.fn(async () => ({ default: 42 }));
    const loaded = await loadPluginFrontends(
      () => document.createElement('div'), importer,
    );
    expect(loaded).toEqual([]);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm vitest run src/plugins/PluginHost.test.tsx`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement PluginHost**

Create `frontend/src/plugins/PluginHost.module.css`:

```css
/* Floating widget stack: bottom-right, above the React Flow MiniMap
   (which sits inside the canvas at the viewport's bottom-right corner),
   below menus (z 200+) is wrong — widgets must float over the canvas but
   under modals (z 9000). 300 slots between popovers (250) and modals. */
.stack {
  position: fixed;
  right: 16px;
  bottom: 188px;
  z-index: 300;
  display: flex;
  flex-direction: column-reverse;
  align-items: flex-end;
  gap: 10px;
  pointer-events: none;
}

.stack > * {
  pointer-events: auto;
}
```

Create `frontend/src/plugins/PluginHost.tsx`:

```tsx
/**
 * Loads installed plugins' frontend bundles and hosts their floating
 * widgets in a fixed bottom-right stack.
 *
 * Activation is once per page load (module-level guard — React StrictMode
 * double-mounts effects in dev). A plugin that throws during import or
 * activate() is reported and skipped; it cannot break the app or other
 * plugins.
 */
import { useEffect, useRef } from 'react';
import { useNodeDefStore } from '../store/nodeDefStore';
import { useToastStore } from '../store/toastStore';
import { buildPluginAPI } from './api';
import styles from './PluginHost.module.css';

interface PluginListItem {
  id: string;
  enabled: boolean;
  frontend_entry: string | null;
}

type Importer = (url: string) => Promise<{ default?: unknown }>;

let hostStarted = false;
let stackEl: HTMLElement | null = null;

function widgetContainer(pluginId: string, widgetId: string): HTMLElement {
  const host = stackEl ?? document.body;
  const domId = `plugin-widget-${pluginId}-${widgetId}`;
  const existing = document.getElementById(domId);
  if (existing) return existing;
  const el = document.createElement('div');
  el.id = domId;
  host.appendChild(el);
  return el;
}

/** Wait (bounded) for node definitions so plugins see a usable catalog. */
async function waitForNodeDefinitions(timeoutMs = 15000): Promise<void> {
  const start = Date.now();
  while (useNodeDefStore.getState().definitions.length === 0) {
    if (Date.now() - start > timeoutMs) return;
    await new Promise((r) => setTimeout(r, 250));
  }
}

export async function loadPluginFrontends(
  getContainer: (pluginId: string, widgetId: string) => HTMLElement
    = widgetContainer,
  importer: Importer = (url) => import(/* @vite-ignore */ url),
): Promise<string[]> {
  let plugins: PluginListItem[];
  try {
    const res = await fetch('/api/plugins');
    if (!res.ok) return [];
    plugins = await res.json();
  } catch {
    return [];
  }

  await waitForNodeDefinitions();

  const activated: string[] = [];
  for (const p of plugins) {
    if (!p.enabled || !p.frontend_entry) continue;
    try {
      const mod = await importer(p.frontend_entry);
      if (typeof mod.default !== 'function') {
        throw new Error('frontend entry has no default export function');
      }
      mod.default(buildPluginAPI(p.id, (widgetId) => getContainer(p.id, widgetId)));
      activated.push(p.id);
    } catch (err) {
      console.warn(`[plugins] failed to activate '${p.id}' frontend:`, err);
      useToastStore.getState().addToast(
        `Plugin "${p.id}" UI failed to load`, 'error',
      );
    }
  }
  return activated;
}

export function PluginHost() {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    stackEl = ref.current;
    if (hostStarted) return;
    hostStarted = true;
    void loadPluginFrontends();
  }, []);

  return <div ref={ref} className={styles.stack} data-testid="plugin-widget-stack" />;
}
```

Note the test signature: in the test, `getContainer` is called as `getContainer(p.id, widgetId)` through the lambda — the injected mock `() => document.createElement('div')` accepts any args. Keep `loadPluginFrontends(getContainer, importer)` parameter order exactly as above.

In `frontend/src/App.tsx`:
- Add import: `import { PluginHost } from './plugins/PluginHost';`
- Render `<PluginHost />` right after `<DialogContainer />` (line 95).

In `frontend/vite.config.ts`, add to `server.proxy`:

```typescript
      '/plugins': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
```

- [ ] **Step 4: Run the frontend suite + typecheck/build**

Run: `pnpm vitest run` then `pnpm build`
Expected: all tests pass; `tsc -b && vite build` clean.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/plugins/ frontend/src/App.tsx frontend/vite.config.ts
git commit -m "feat(frontend): PluginHost loads plugin frontend bundles into a floating widget stack"
```

---

### Task 9: Full-suite verification + in-browser e2e with a demo plugin

**Files:**
- Create (uncommitted, scratch): a demo plugin under the dev user-data plugins dir

- [ ] **Step 1: Run both full suites**

From `backend/`: `uv run pytest -q` — expected: 1246+ passed (1230 baseline + 16 new), 11 skipped.
From `frontend/`: `pnpm vitest run` — expected: all pass (1573 baseline + ~15 new).

- [ ] **Step 2: Create the demo plugin fixture**

Find the dev user-data dir: `scripts/dev.py` sets `CODEFYUI_USER_DATA_DIR=<repo>/.codefyui_dev/` (check `scripts/dev.py` for the exact path it exports — search for `CODEFYUI_USER_DATA_DIR`). Inside the worktree create `<user_data>/plugins/demo-frontend/` with:

`cdui.plugin.toml`:

```toml
[plugin]
id = "demo-frontend"
name = "Demo Frontend Plugin"
version = "0.1.0"
description = "Scratch plugin used to e2e-verify frontend extensions."
schema_version = 1

[frontend]
entry = "frontend/index.js"
```

`frontend/index.js` (plain ESM, no build step — picks two compatible nodes from the live catalog so it works whatever packs are installed):

```javascript
export default function activate(api) {
  const el = api.ui.addFloatingWidget({ id: 'demo' });
  const btn = document.createElement('button');
  btn.textContent = 'AI';
  btn.style.cssText = [
    'width:48px', 'height:48px', 'border-radius:50%', 'border:none',
    'background:#06b6d4', 'color:#0b1220', 'font-weight:700',
    'cursor:pointer', 'box-shadow:0 4px 12px rgba(0,0,0,.45)',
  ].join(';');
  btn.onclick = () => {
    const defs = api.graph.getNodeDefinitions();
    const src = defs.find((d) => d.inputs.length === 0 && d.outputs.length > 0);
    const dst = src && defs.find((d) =>
      d.inputs.some((i) => i.data_type === src.outputs[0].data_type));
    if (!src || !dst) { api.ui.toast('No compatible node pair found', 'error'); return; }
    const result = api.graph.applyOperations([
      { op: 'add_node', node_type: src.node_name, ref: 'a' },
      { op: 'add_node', node_type: dst.node_name, ref: 'b' },
      { op: 'connect', source: 'a', source_handle: src.outputs[0].name,
        target: 'b',
        target_handle: dst.inputs.find((i) => i.data_type === src.outputs[0].data_type).name },
      { op: 'auto_layout' },
    ]);
    console.log('[demo-frontend] apply result:', JSON.stringify(result));
    api.ui.toast(`Demo applied: ${result.results.filter((r) => r.ok).length}/4 ops ok`, 'success');
  };
  el.appendChild(btn);
}
```

Then register it in the dev lockfile `<user_data>/plugins/installed.json` (create or extend):

```json
{
  "schema": 1,
  "plugins": {
    "demo-frontend": {
      "source_kind": "github_url",
      "source": "local/demo-frontend",
      "installed_at": "2026-06-11T00:00:00Z",
      "manifest": {"id": "demo-frontend", "name": "Demo Frontend Plugin", "version": "0.1.0"},
      "trusted_modules": [],
      "enabled": true
    }
  }
}
```

(If the lockfile already exists with other plugins, merge the entry instead of overwriting.)

- [ ] **Step 3: Start the dev environment and verify in Chrome**

Start backend + frontend dev servers per `scripts/dev.py` conventions (inspect `uv run python scripts/dev.py --help` from the worktree root; the dev command starts uvicorn on 8000 and vite on 5173 — run it with `run_in_background`).

In Chrome (mcp__claude-in-chrome tools), against the vite URL:
1. `GET /api/plugins` (via the app) — demo plugin listed with `frontend_entry`.
2. The teal "AI" button renders bottom-right above the MiniMap; nothing overlaps it. Adjust `.stack { bottom: … }` in `PluginHost.module.css` if it collides with the MiniMap or Results panel, and re-verify.
3. Click the button — two nodes appear connected, auto-laid-out; toast shows "4/4 ops ok"; console log shows per-op results.
4. Ctrl+Z — both nodes and the edge disappear in ONE undo step.
5. Console shows no plugin-related errors; check `read_console_messages` with pattern `plugins|demo-frontend`.
6. Disable the plugin via `POST /api/plugins/demo-frontend/disable` (or curl with token), reload the page — button gone, `frontend_entry` null.
   Re-enable afterwards.

- [ ] **Step 4: Record evidence**

Screenshot the canvas after step 3 (widget + generated nodes visible) for the PR description.

- [ ] **Step 5: Clean up scratch state**

The demo plugin lives under `.codefyui_dev/` (gitignored — verify with `git status`); leave it for Plan C reuse but confirm the working tree is clean of unintended files.

---

### Task 10: Push branch + open PR A

- [ ] **Step 1: Final review of the diff**

`git log --oneline main..HEAD` and `git diff main --stat` — confirm only intended files changed (backend plugin loader/routes/tests, scripts/plugins.py, frontend plugins/* + App.tsx + vite.config.ts + utils + tabStore).

- [ ] **Step 2: Push and create PR**

```bash
git push -u origin feat/plugin-frontend-extensions
gh pr create --title "feat: plugin frontend extensions (CodefyUIPluginAPI v1)" --body "<summary per template below>"
```

PR body must cover: motivation (first consumer: Graph Copilot plugin), manifest `[frontend].entry` format, serving route + security (path validation, enabled gate, MIME pinning), `frontend_entry` in /api/plugins, CodefyUIPluginAPI v1 surface, batched undo semantics, CLI trust notice, test coverage summary, e2e evidence screenshot, and a link to `docs/superpowers/specs/2026-06-11-graph-copilot-design.md`. End with the standard Claude Code attribution footer.

Do NOT merge — the maintainer reviews and merges. Plan B continues from this branch (stacked) without waiting.
