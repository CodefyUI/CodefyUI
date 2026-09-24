"""Tests for the plugin developer-experience batch:

- ``_reload_target`` — hot-reload POST honors the configured server port.
- plugin SDK contract sync — the vendored scaffold copy matches the canonical
  ``frontend/src/plugins/contract.ts``.
- ``cdui plugin new`` — the scaffold generates a valid, loadable plugin.
"""

from __future__ import annotations

import pytest

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


# ── #461: the template repository's copy, checked and synced on demand ───────

def _template_checkout(root, sdk_files):
    """A directory shaped like a CodefyUI-Plugin-Official checkout."""
    sdk = root / "ui" / "src" / "sdk"
    sdk.mkdir(parents=True)
    (root / "cdui.plugin.toml").write_text(
        '[plugin]\nid = "official-template"\nschema_version = 1\n', encoding="utf-8"
    )
    for name, text in sdk_files.items():
        (sdk / name).write_bytes(text.encode("utf-8"))
    return sdk


def test_sync_plugin_sdk_checks_and_syncs_a_template_checkout(tmp_path, capsys, monkeypatch):
    """The template repository vendors the whole SDK -- the contract's types
    plus the React bindings -- and fell three API versions behind with
    nothing in this repository noticing (#461). ``--template`` checks a local
    checkout of it against the scaffold, and brings it in step."""
    import sync_plugin_sdk

    # The in-repo copy is not this mode's business: nothing may be written to it.
    untouched = tmp_path / "scaffold-types.ts"
    monkeypatch.setattr(sync_plugin_sdk, "TARGETS", [untouched])

    scaffold_index = (sync_plugin_sdk.SCAFFOLD_SDK / "index.ts").read_text(encoding="utf-8")
    sdk = _template_checkout(tmp_path / "template", {
        "types.ts": "// apiVersion 2\n",
        "react.tsx": "export {};\n",
        # In step already; a CRLF checkout is not drift.
        "index.ts": scaffold_index.replace("\r\n", "\n").replace("\n", "\r\n"),
        # The template's own file, which the scaffold does not have.
        "react.test.tsx": "// the template's own test\n",
    })
    template = str(tmp_path / "template")

    assert sync_plugin_sdk.main(["--check", "--template", template]) == 1
    out = capsys.readouterr().out
    assert "stale: ui/src/sdk/types.ts" in out
    assert "stale: ui/src/sdk/react.tsx" in out
    assert "index.ts" not in out and "react.test.tsx" not in out

    assert sync_plugin_sdk.main(["--template", template]) == 0
    capsys.readouterr()
    assert sync_plugin_sdk._norm((sdk / "types.ts").read_text(encoding="utf-8")) \
        == sync_plugin_sdk.rendered()
    for scaffold_file in sync_plugin_sdk.SCAFFOLD_SDK.iterdir():
        if scaffold_file.name != "types.ts":
            assert sync_plugin_sdk._norm((sdk / scaffold_file.name).read_text(encoding="utf-8")) \
                == sync_plugin_sdk._norm(scaffold_file.read_text(encoding="utf-8"))
    assert (sdk / "react.test.tsx").read_text(encoding="utf-8") == "// the template's own test\n"
    assert not untouched.exists()

    assert sync_plugin_sdk.main(["--check", "--template", template]) == 0


def test_sync_plugin_sdk_refuses_a_directory_that_is_not_a_template_checkout(tmp_path):
    """A mistyped path must not grow a ui/src/sdk/ somewhere it does not belong."""
    import sync_plugin_sdk

    (tmp_path / "ui" / "src").mkdir(parents=True)  # no manifest, no sdk/
    assert sync_plugin_sdk.main(["--template", str(tmp_path)]) == 2
    assert sync_plugin_sdk.main(["--check", "--template", str(tmp_path)]) == 2
    assert not (tmp_path / "ui" / "src" / "sdk").exists()


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


def _scan_like_an_install(root):
    """Run the security scan an install runs over a plugin directory.

    ``flows._install_from_github`` hands the gate the WHOLE extracted tree
    with the manifest's own ``allowed_modules`` and the capabilities it
    declares (granted when the user accepts them). So does this: ``tests/``
    is scanned like ``nodes/``, because a node can import any file in the
    plugin (core#182).
    """
    from app.core.plugins.manifest import manifest_allowed_modules, manifest_capabilities

    manifest = plugin_cli.read_manifest(root)
    plugin_cli.validate_plugin_dir(
        root,
        list(manifest_allowed_modules(manifest)),
        manifest_capabilities(manifest),
    )


def test_new_scaffold_backend_only(tmp_path):
    rc = plugin_cli.main(["new", "my-test-plugin", "--dir", str(tmp_path)])
    assert rc == 0
    root = tmp_path / "my-test-plugin"

    # Core files exist; the ui/ subtree is absent without --ui.
    assert (root / "cdui.plugin.toml").is_file()
    assert (root / "pytest.ini").is_file()
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

    # The generated python is valid.
    _assert_python_compiles(root / "nodes" / "example_node.py")
    _assert_python_compiles(root / "tests" / "conftest.py")
    _assert_python_compiles(root / "tests" / "test_example_node.py")
    _no_unrendered_placeholders(root)


@pytest.mark.parametrize("ui", [False, True], ids=["backend-only", "with-ui"])
def test_new_scaffold_passes_the_install_scan_as_a_whole(tmp_path, ui):
    """A plugin fresh from ``cdui plugin new`` has to install (#413).

    The scan an install runs covers ``tests/`` as well as ``nodes/``, and the
    scaffold's ``tests/conftest.py`` used to fake the ``cdui_plugins.<id>``
    package with ``import sys`` and ``setattr``: refused at install, and by
    ``setattr`` at every tier, so no ``[security]`` grant could have let it
    through. This test used to check ``nodes/`` alone and passed all along.
    """
    args = ["new", "my-test-plugin", "--dir", str(tmp_path)] + (["--ui"] if ui else [])
    assert plugin_cli.main(args) == 0
    root = tmp_path / "my-test-plugin"

    _scan_like_an_install(root)

    # Not vacuous: the same scan refuses a test file the scaffold might have
    # shipped, so tests/ really was read above.
    (root / "tests" / "probe.py").write_text("import sys\n", encoding="utf-8")
    with pytest.raises(plugin_cli.PluginValidationError, match="tests/probe.py"):
        _scan_like_an_install(root)


def test_new_scaffold_tests_run_and_pass(tmp_path):
    """The scaffold's own tests pass, run the way its README says: with the
    CodefyUI backend's Python, from outside this repository's test setup.

    A separate interpreter, so nothing this suite has already imported (its
    ``cdui_plugins`` namespace, its ``tests`` package) can make them pass.
    """
    import os
    import subprocess
    import sys

    assert plugin_cli.main(["new", "my-test-plugin", "--dir", str(tmp_path)]) == 0
    root = tmp_path / "my-test-plugin"

    env = dict(os.environ)
    env.pop("PYTEST_ADDOPTS", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=300,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, output
    assert "2 passed" in output, output


def test_new_scaffold_example_node_fits_the_palette_and_has_details(tmp_path):
    """The example node is the first node a plugin author reads and copies, so
    it has to show the split every shipped node follows (#461): a DESCRIPTION
    the palette row shows whole, and the longer explanation in DETAILS, which
    the config panel and the Docs tab show. It used to carry a 70-character
    DESCRIPTION the row cut off and no DETAILS, so the description in every
    scaffolded node's Docs tab stopped at that one line."""
    # Imported, not restated: the cap is the palette row's measured width,
    # and it lives beside the ratchet that holds every shipped node to it, so
    # a re-measured row moves this test along with it.
    from tests.test_api_nodes import MAX_DESCRIPTION_CHARS

    assert plugin_cli.main(["new", "my-test-plugin", "--dir", str(tmp_path)]) == 0
    source = tmp_path / "my-test-plugin" / "nodes" / "example_node.py"

    # The class the rendered file defines, whose attributes are what
    # GET /api/nodes serves. compile + exec rather than an import, so no .pyc
    # lands in the scaffold (the same reason as _assert_python_compiles).
    namespace: dict = {"__name__": "scaffolded_example_node"}
    exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), namespace)
    node = namespace["ExampleNode"]

    description = node.DESCRIPTION
    assert len(description) <= MAX_DESCRIPTION_CHARS and "\n" not in description, (
        f"the example node's DESCRIPTION is {len(description)} characters; a "
        f"palette summary is one line of at most {MAX_DESCRIPTION_CHARS} and the "
        "row cuts off the rest. Move the rest into DETAILS.")
    assert node.DETAILS.strip(), (
        "the example node has no DETAILS, so the description in every "
        "scaffolded node's Docs tab stops at the one-line summary")
    # DETAILS tells the author where the row cuts off, as a number, so a
    # re-measured row has to move that sentence along with the cap.
    assert f"{MAX_DESCRIPTION_CHARS} characters" in node.DETAILS, (
        "the example node's DETAILS no longer states the palette row's cap of "
        f"{MAX_DESCRIPTION_CHARS} characters; update the number it gives")


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

    # The renderer is registered under the example node's real type: the
    # manifest id exactly as written, hyphens included, then NODE_NAME. Under
    # any other type it never mounts. It used to say `ui_plugin:Example`, and
    # the docs told every author with a hyphenated id to fix it by hand.
    import re

    source = root / "nodes" / "example_node.py"
    namespace: dict = {"__name__": "scaffolded_example_node"}
    exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), namespace)
    node_type = f"{manifest['plugin']['id']}:{namespace['ExampleNode'].NODE_NAME}"
    assert node_type == "ui-plugin:Example"
    index_tsx = (root / "ui" / "src" / "index.tsx").read_text(encoding="utf-8")
    registered = re.findall(r"registerRenderer\(\s*'([^']+)'", index_tsx)
    assert registered == [node_type], registered
    # The vendored types match the canonical contract.
    import sync_plugin_sdk

    assert sync_plugin_sdk._norm((root / "ui" / "src" / "sdk" / "types.ts").read_text(encoding="utf-8")) \
        == sync_plugin_sdk.rendered()
    _no_unrendered_placeholders(root)


def test_new_scaffold_rejects_invalid_id(tmp_path):
    assert plugin_cli.main(["new", "Bad_Id", "--dir", str(tmp_path)]) == 2
    assert not (tmp_path / "Bad_Id").exists()


def test_new_scaffold_refuses_a_route_name(tmp_path, capsys):
    """``new``, ``link`` and ``install`` agree about which ids this build
    owns, because they ask the same function. ``install`` is a fixed path
    under /api/plugins/, so a plugin scaffolded under it could never be
    installed -- the router would decide which one answered."""
    assert plugin_cli.main(["new", "install", "--dir", str(tmp_path)]) == 2
    assert not (tmp_path / "install").exists()
    printed = capsys.readouterr()
    assert "install" in printed.out + printed.err


def test_new_scaffold_allows_the_id_of_an_official_github_pack(tmp_path):
    """The other side of the same rule: a ``github`` catalog id is not
    reserved, because it is the id the author of an official plugin
    scaffolds and installs their own repository under. Refusing it here was
    refusing the author their own name."""
    assert plugin_cli.main(["new", "official-template", "--dir", str(tmp_path)]) == 0
    assert (tmp_path / "official-template" / "cdui.plugin.toml").is_file()


def test_new_scaffold_refuses_existing_nonempty(tmp_path):
    existing = tmp_path / "dupe"
    existing.mkdir()
    (existing / "keep.txt").write_text("do not clobber", encoding="utf-8")
    assert plugin_cli.main(["new", "dupe", "--dir", str(tmp_path)]) == 1
    # The pre-existing file is untouched.
    assert (existing / "keep.txt").read_text(encoding="utf-8") == "do not clobber"


# ── cp950: status glyphs must degrade to ASCII, never crash ───────────────────

def test_cli_output_survives_cp950_console(tmp_path):
    """On a console whose encoding can't represent ▶/✓ (legacy Windows cp950, or
    a redirected pipe), the CLI must print ASCII markers instead of raising
    UnicodeEncodeError. Runs the real entry point in a subprocess with stdout
    forced to cp950."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    scripts = Path(plugin_cli.__file__).resolve().parent
    backend = scripts.parent / "backend"

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "cp950"   # reproduce the crashing encoding
    env["PYTHONPATH"] = str(backend)    # so the subprocess can import `app`
    env["CODEFYUI_LANG"] = "en"         # deterministic ASCII messages
    env["NO_COLOR"] = "1"

    proc = subprocess.run(
        [sys.executable, str(scripts / "plugins.py"), "new", "cp-probe", "--dir", str(tmp_path)],
        capture_output=True,
        env=env,
    )
    out = proc.stdout.decode("cp950", "replace")
    assert proc.returncode == 0, proc.stderr.decode("cp950", "replace")
    assert "Traceback" not in proc.stderr.decode("cp950", "replace")
    assert "Creating new plugin" in out
    assert "▶" not in out and "✓" not in out      # glyphs fell back
    assert "> Creating new plugin" in out          # ASCII section marker
    assert "+ Created" in out                       # ASCII ok marker
