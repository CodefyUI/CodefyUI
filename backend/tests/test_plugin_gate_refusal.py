"""What the install scan says when it refuses a file (#413).

The validator decides whether a line may stand; the gate decides what the
person who has to fix it is told. A refusal names the file by its path in the
plugin and the line, says why a file outside ``nodes/`` is read at all, and
only gives advice that works: it never tells the author to declare a
``[security]`` grant that cannot let the file through, nor to declare one for
a file outside ``nodes/``, where a grant would widen what the whole plugin may
do.
"""

from __future__ import annotations

import pytest

from app.core.plugin_validator import (
    PluginValidationError,
    dangerous_modules,
    validate_python_source,
)
from app.core.plugins import gate
from app.core.security_tiers import CAPABILITIES

# The conftest `cdui plugin new` wrote until #413: the `cdui_plugins.<id>`
# package faked by hand. `import sys` is refused unless the author is trusted,
# and `setattr` with a computed name is refused at every tier.
OLD_SCAFFOLD_CONFTEST = """\
from __future__ import annotations

import sys
import types
from pathlib import Path

PLUGIN_ID = "my-plugin"
_PY_ID = PLUGIN_ID.replace("-", "_")
_REPO_ROOT = Path(__file__).resolve().parents[1]

_pkg = sys.modules.get("cdui_plugins")
if _pkg is None:
    _pkg = types.ModuleType("cdui_plugins")
    _pkg.__path__ = []
    sys.modules["cdui_plugins"] = _pkg

_sub_name = f"cdui_plugins.{_PY_ID}"
if _sub_name not in sys.modules:
    _sub = types.ModuleType(_sub_name)
    _sub.__path__ = [str(_REPO_ROOT)]
    sys.modules[_sub_name] = _sub
    setattr(_pkg, _PY_ID, _sub)
"""

WHY_A_FILE_OUTSIDE_NODES_IS_READ = "tests/ included, because a node can import any file"
TESTS_THAT_PASS = "https://docs.codefyui.com/advanced/plugins#tests-are-scanned-too"

# Refused cleanly at line 1 under the install's grants. Line 3 is too deeply
# nested for `ast.unparse`, which the validator calls on a `weights_only=`
# value it rejects, and the pass under every grant walks on to it.
DEEPLY_NESTED_PAST_THE_REFUSAL = (
    "import sys\nimport torch\ntorch.load('p', weights_only=" + "-" * 2500 + "1)\n"
)


def _line_of(source: str, text: str) -> int:
    return source.splitlines().index(text) + 1


def _plugin(tmp_path, files: dict[str, str]):
    root = tmp_path / "pack"
    (root / "nodes").mkdir(parents=True)
    (root / "nodes" / "__init__.py").write_text("", encoding="utf-8")
    for rel, source in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return root


def _refusal(root, allowed=(), capabilities=()) -> PluginValidationError:
    with pytest.raises(PluginValidationError) as excinfo:
        gate.validate_plugin_dir(root, list(allowed), list(capabilities))
    return excinfo.value


def _every_grant(root) -> None:
    """The most a manifest can ask for: every capability, and every module on
    the blocklist listed under allowed_modules (with --trust-author)."""
    gate.validate_plugin_dir(root, sorted(dangerous_modules()), list(CAPABILITIES))


# ── where ──────────────────────────────────────────────────────────────────

def test_a_refusal_names_the_file_by_its_path_in_the_plugin_and_the_line(tmp_path):
    root = _plugin(tmp_path, {"nodes/net.py": "import json\nimport requests\n"})
    exc = _refusal(root)
    assert str(exc).startswith("nodes/net.py, line 2: "), str(exc)
    assert exc.lineno == 2


def test_validate_nodes_dir_names_the_file_under_nodes_too(tmp_path):
    root = _plugin(tmp_path, {"nodes/bad.py": "import subprocess\n"})
    with pytest.raises(PluginValidationError) as excinfo:
        gate.validate_nodes_dir(root / "nodes", [])
    assert str(excinfo.value).startswith("nodes/bad.py, line 1: "), str(excinfo.value)


# ── node files: the validator's grant stays when it is one that works ──────

def test_a_node_file_keeps_the_capability_that_lets_it_through(tmp_path):
    root = _plugin(tmp_path, {"nodes/net.py": "import requests\n"})
    message = str(_refusal(root))
    assert 'capabilities = ["network"]' in message
    assert WHY_A_FILE_OUTSIDE_NODES_IS_READ not in message
    assert TESTS_THAT_PASS not in message
    # ...and it is advice that works:
    gate.validate_plugin_dir(root, [], ["network"])


def test_a_node_file_keeps_the_trust_author_advice_when_it_works(tmp_path):
    root = _plugin(tmp_path, {"nodes/tool.py": "import subprocess\n"})
    message = str(_refusal(root))
    assert "--trust-author" in message and "allowed_modules" in message
    gate.validate_plugin_dir(root, ["subprocess"])


def test_no_grant_is_suggested_when_none_could_let_the_file_through(tmp_path):
    """Granting `sys` here would only move the refusal to `setattr`, which no
    tier allows: following the validator's advice would be a dead end."""
    source = "import sys\n\n\ndef put(obj, name, value):\n    setattr(obj, name, value)\n"
    root = _plugin(tmp_path, {"nodes/tool.py": source})
    exc = _refusal(root)
    message = str(exc)

    assert message.startswith("nodes/tool.py, line 1: Importing 'sys' is not allowed")
    assert "--trust-author" not in message and "allowed_modules" not in message
    setattr_line = _line_of(source, "    setattr(obj, name, value)")
    assert "No [security] setting in cdui.plugin.toml lets this file through" in message
    assert f"line {setattr_line} is refused whatever the manifest declares" in message
    assert "Use of 'setattr'() is not allowed" in message
    # The claim is true: the most a manifest can ask for still refuses it.
    with pytest.raises(PluginValidationError, match="setattr"):
        _every_grant(root)


def test_a_refusal_no_grant_lifts_says_so(tmp_path):
    root = _plugin(tmp_path, {"nodes/calc.py": "def run(s):\n    return eval(s)\n"})
    message = str(_refusal(root))
    assert message.startswith("nodes/calc.py, line 2: Use of 'eval'() is not allowed")
    assert "No [security] setting in cdui.plugin.toml lifts this refusal" in message
    with pytest.raises(PluginValidationError, match="eval"):
        _every_grant(root)


# ── files outside nodes/: why they are read, and no grant ──────────────────

@pytest.mark.parametrize("rel", ["tests/test_things.py", "helpers.py"])
@pytest.mark.parametrize(
    ("statement", "kept"),
    [
        ("import sys", "no capability grants 'sys'"),
        ("from pathlib import Path", "plain open() needs no declaration"),
        ("import os", "'from os.path import join, basename' needs no capability"),
    ],
    ids=["tier-2", "capability", "capability-with-hint"],
)
def test_a_file_outside_nodes_is_told_why_it_was_read_and_not_to_declare_a_grant(
    tmp_path, rel, statement, kept
):
    root = _plugin(tmp_path, {rel: f"{statement}\n"})
    message = str(_refusal(root))

    assert message.startswith(f"{rel}, line 1: "), message
    assert WHY_A_FILE_OUTSIDE_NODES_IS_READ in message
    assert "--trust-author" not in message and "allowed_modules" not in message
    assert "capabilities = [" not in message
    assert "If only tests or tooling use this file, change it" in message
    # The validator's own advice that is not a grant survives the cut.
    assert kept in message
    # It ends with where a test setup that needs no grant is written down.
    *sentences, last = message.splitlines()
    assert last.endswith(TESTS_THAT_PASS), message
    # Every line before it is a whole sentence: the Plugin Center shows the
    # refusal as one paragraph, so the lines run together there.
    assert all(line.endswith(".") for line in sentences), message


def test_the_conftest_cdui_plugin_new_used_to_write_is_refused_with_advice_that_works(
    tmp_path,
):
    """#413 as a user met it: the refusal named `conftest.py` and advised
    `--trust-author` plus `allowed_modules = ["sys"]`, and following that
    advice only moved the refusal to `setattr`, which no tier allows."""
    root = _plugin(tmp_path, {"tests/conftest.py": OLD_SCAFFOLD_CONFTEST})
    exc = _refusal(root)
    message = str(exc)

    sys_line = _line_of(OLD_SCAFFOLD_CONFTEST, "import sys")
    setattr_line = _line_of(OLD_SCAFFOLD_CONFTEST, "    setattr(_pkg, _PY_ID, _sub)")
    assert message.startswith(
        f"tests/conftest.py, line {sys_line}: Importing 'sys' is not allowed"
    ), message
    assert exc.lineno == sys_line
    assert WHY_A_FILE_OUTSIDE_NODES_IS_READ in message
    assert "--trust-author" not in message and "allowed_modules" not in message
    assert f"line {setattr_line} is refused whatever the manifest declares" in message
    assert message.endswith(TESTS_THAT_PASS), "the refusal points at the fix"
    with pytest.raises(PluginValidationError, match="setattr"):
        _every_grant(root)


@pytest.mark.parametrize("rel", ["nodes/deep.py", "tests/test_deep.py"])
def test_a_file_the_every_grant_pass_cannot_walk_is_still_refused_cleanly(tmp_path, rel):
    """Working out the advice must never cost the verdict. The every-grant
    pass reads further than the refusal did, so it can fail where the scan
    itself did not; the refusal then stands with the validator's own words,
    instead of a traceback in the CLI and a failed Plugin Center job with no
    hint."""
    root = _plugin(tmp_path, {rel: DEEPLY_NESTED_PAST_THE_REFUSAL})
    exc = _refusal(root)
    assert str(exc).startswith(f"{rel}, line 1: Importing 'sys' is not allowed"), str(exc)
    assert exc.lineno == 1


def test_the_rewrite_does_not_hold_the_first_parse_while_it_parses_again(tmp_path):
    """The refusal is rewritten outside the handler of the first one, whose
    traceback is dropped first: its frames hold the file's parsed tree, and
    the rewrite parses the file a second time. On a large refused file,
    holding both trees at once is the difference in peak memory."""
    root = _plugin(tmp_path, {"nodes/net.py": "import requests\n"})
    exc = _refusal(root)
    assert exc.__context__ is None, "rewritten while the first refusal was being handled"
    assert exc.__cause__ is not None and exc.__cause__.__traceback__ is None


# ── the one place the gate reads the validator's words ─────────────────────

@pytest.mark.parametrize("module", ["sys", "pathlib", "os"])
def test_the_grant_advice_the_gate_cuts_is_still_what_the_validator_writes(module):
    """The gate cuts the sentence the validator ends an import refusal with,
    by how that sentence starts. If the validator rewords it, the cut stops
    silently, and files outside nodes/ are advised to ask for a grant again."""
    with pytest.raises(PluginValidationError) as excinfo:
        validate_python_source(f"import {module}\n", "this file")
    sentences = str(excinfo.value).split(". ")
    assert any(s.startswith(gate._GRANT_ADVICE_OPENINGS) for s in sentences), (
        "plugin_validator reworded the grant it names at the end of an import "
        "refusal; update gate._GRANT_ADVICE_OPENINGS to match: "
        f"{excinfo.value}"
    )
