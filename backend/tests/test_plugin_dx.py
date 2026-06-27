"""Tests for the plugin developer-experience batch:

- ``_reload_target`` — hot-reload POST honors the configured server port.
- plugin SDK contract sync — the vendored scaffold copy matches the canonical
  ``frontend/src/plugins/contract.ts``.
- ``cdui plugin new`` — the scaffold generates a valid, loadable plugin.
"""

from __future__ import annotations

import plugins as plugin_cli


# ── item 4: reload targets the configured port ───────────────────────────────

def test_reload_target_defaults_to_8000(monkeypatch):
    monkeypatch.delenv("CODEFYUI_PORT", raising=False)
    url, host = plugin_cli._reload_target()
    assert url == "http://127.0.0.1:8000/api/plugins/reload"
    assert host == "127.0.0.1:8000"


def test_reload_target_honors_codefyui_port(monkeypatch):
    monkeypatch.setenv("CODEFYUI_PORT", "8200")
    url, host = plugin_cli._reload_target()
    assert url == "http://127.0.0.1:8200/api/plugins/reload"
    assert host == "127.0.0.1:8200"


def test_reload_target_ignores_non_numeric_port(monkeypatch):
    # A garbage override must not crash or produce a malformed URL — fall back.
    monkeypatch.setenv("CODEFYUI_PORT", "not-a-port")
    url, host = plugin_cli._reload_target()
    assert url.startswith("http://127.0.0.1:")
    assert url.endswith("/api/plugins/reload")
    assert host == url[len("http://"):-len("/api/plugins/reload")]


# ── item 3: vendored SDK types stay in sync with the canonical contract ───────

def test_plugin_sdk_types_in_sync():
    """The scaffold's vendored ``ui/src/sdk/types.ts`` must match the canonical
    ``frontend/src/plugins/contract.ts``. If this fails, run:
    ``python scripts/sync_plugin_sdk.py``."""
    import sync_plugin_sdk

    assert sync_plugin_sdk.check() == 0


# ── item 1: cdui plugin new scaffold ─────────────────────────────────────────

def _no_unrendered_placeholders(root):
    """Every generated text file must have its {{...}} tokens substituted."""
    for f in root.rglob("*"):
        if f.is_file() and "__pycache__" not in f.parts:
            assert "{{" not in f.read_text(encoding="utf-8"), f"unrendered token in {f}"


def _assert_python_compiles(path):
    """The generated .py is at least syntactically valid post-substitution.

    Uses the builtin ``compile`` (not ``py_compile``) so no .pyc is written into
    the scaffolded directory.
    """
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_new_scaffold_backend_only(tmp_path):
    from app.core.plugin_validator import validate_python_source

    rc = plugin_cli.main(["new", "my-test-plugin", "--dir", str(tmp_path)])
    assert rc == 0
    root = tmp_path / "my-test-plugin"

    # Core files exist; the ui/ subtree is absent without --ui.
    assert (root / "cdui.plugin.toml").is_file()
    assert (root / "nodes" / "example_node.py").is_file()
    assert (root / "tests" / "conftest.py").is_file()
    assert (root / "tests" / "test_example_node.py").is_file()
    assert not (root / "ui").exists()

    # Manifest validates and carries the substituted id; no frontend stanza.
    manifest = plugin_cli.read_manifest(root)
    plugin_cli.validate_manifest(manifest)
    assert manifest["plugin"]["id"] == "my-test-plugin"
    assert manifest["plugin"]["name"] == "My Test Plugin"
    assert "frontend" not in manifest

    # conftest wires the right namespace id; generated python is valid.
    conftest = (root / "tests" / "conftest.py").read_text(encoding="utf-8")
    assert 'PLUGIN_ID = "my-test-plugin"' in conftest
    _assert_python_compiles(root / "nodes" / "example_node.py")
    _assert_python_compiles(root / "tests" / "conftest.py")
    _assert_python_compiles(root / "tests" / "test_example_node.py")

    # The example node passes the AST security gate installs run.
    validate_python_source(
        (root / "nodes" / "example_node.py").read_bytes(),
        "example_node.py",
        allowed_modules=[],
    )
    _no_unrendered_placeholders(root)


def test_new_scaffold_with_ui(tmp_path):
    rc = plugin_cli.main(["new", "ui-plugin", "--ui", "--dir", str(tmp_path)])
    assert rc == 0
    root = tmp_path / "ui-plugin"

    assert (root / "ui" / "src" / "index.tsx").is_file()
    assert (root / "ui" / "src" / "sdk" / "types.ts").is_file()
    assert (root / "ui" / "src" / "sdk" / "react.tsx").is_file()
    assert (root / "ui" / "package.json").is_file()

    # --ui appends the [frontend] entry to the manifest.
    manifest = plugin_cli.read_manifest(root)
    plugin_cli.validate_manifest(manifest)
    assert manifest["frontend"]["entry"] == "frontend/index.js"

    # The node renderer registration uses the snake_case namespace.
    index_tsx = (root / "ui" / "src" / "index.tsx").read_text(encoding="utf-8")
    assert "ui_plugin:Example" in index_tsx
    # The vendored types match the canonical contract.
    import sync_plugin_sdk

    assert sync_plugin_sdk._norm((root / "ui" / "src" / "sdk" / "types.ts").read_text(encoding="utf-8")) \
        == sync_plugin_sdk.rendered()
    _no_unrendered_placeholders(root)


def test_new_scaffold_rejects_invalid_id(tmp_path):
    assert plugin_cli.main(["new", "Bad_Id", "--dir", str(tmp_path)]) == 2
    assert not (tmp_path / "Bad_Id").exists()


def test_new_scaffold_refuses_existing_nonempty(tmp_path):
    existing = tmp_path / "dupe"
    existing.mkdir()
    (existing / "keep.txt").write_text("do not clobber", encoding="utf-8")
    assert plugin_cli.main(["new", "dupe", "--dir", str(tmp_path)]) == 1
    # The pre-existing file is untouched.
    assert (existing / "keep.txt").read_text(encoding="utf-8") == "do not clobber"
