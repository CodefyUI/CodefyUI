"""``cdui packs`` -- the terminal entry point to the Package Center.

The same install flow the in-app panel drives has to be reachable from a
shell: companies pre-provision machines with no browser open, and CI has no
panel to click. So ``scripts/packs.py`` is a THIN renderer over
``app.core.packs.flows`` -- it decides nothing about how an install works,
and these tests pin exactly that:

* every refusal the CLI can make on its own (unknown pack, the GPU pack,
  an unmet dependency) happens BEFORE the flow is called, with exit 2, so a
  mistyped id never starts a download;
* the flow's events render as ASCII on a terminal that may be cp950;
* the four failure shapes map to the four exit codes a script can act on
  (0 done, 1 failed, 2 refused, 3 restart needed, 130 cancelled);
* nothing the user typed can reach a package manager -- the module has no
  process-spawning machinery at all, so ``flows.install_pack_live`` is the
  only install path there is.

Plus the marker ``cdui dev`` stamps into the environment (``CODEFYUI_MANAGED``),
which is how the running server can tell it was launched by a supervisor that
could restart it. ``cdui start``'s half of that lives in
``test_dev_start_command.py``, next to the fixture that fakes its spawn.
"""

from __future__ import annotations

import ast
import io
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import dev  # scripts/dev.py -- conftest puts scripts/ on sys.path
import packs  # scripts/packs.py -- same


@pytest.fixture(autouse=True)
def _english(monkeypatch):
    """Pin the CLI's language so assertions are about behaviour, not locale."""
    monkeypatch.setenv("CODEFYUI_LANG", "en")


@pytest.fixture(autouse=True)
def _restore_managed_env():
    """``start()``/``dev()`` write raw ``os.environ`` -- monkeypatch cannot
    undo a write it never saw."""
    saved = os.environ.get("CODEFYUI_MANAGED")
    yield
    if saved is None:
        os.environ.pop("CODEFYUI_MANAGED", None)
    else:
        os.environ["CODEFYUI_MANAGED"] = saved


# ── dev.py dispatch ───────────────────────────────────────────────────────


def test_packs_dispatch_is_registered():
    """``cdui packs ...`` routes to the packs CLI, like plugin and project."""
    assert dev._subcommand_group(["cdui", "packs", "list"]) is dev._dispatch_packs_subcommand


def test_the_other_sub_groups_still_route():
    assert dev._subcommand_group(["cdui", "plugin", "list"]) is dev._dispatch_plugin_subcommand
    assert dev._subcommand_group(["cdui", "project", "list"]) is dev._dispatch_project_subcommand


def test_a_plain_command_is_not_a_sub_group():
    """`start` must fall through to COMMANDS, and a bare `cdui` to the help."""
    assert dev._subcommand_group(["cdui", "start"]) is None
    assert dev._subcommand_group(["cdui"]) is None


def test_packs_is_listed_in_the_help_text():
    """`cdui` with no command prints ``__doc__``; an unlisted subcommand is
    one nobody finds."""
    assert "packs" in dev.__doc__


def test_dispatch_enters_the_venv_before_importing_the_packs_cli(monkeypatch):
    """Order is the point: ``packs.py`` imports ``app.core.packs``, which
    only exists inside the backend venv."""
    calls: list[str] = []
    monkeypatch.setattr(dev, "_exec_into_venv_if_available",
                        lambda: calls.append("venv"))
    monkeypatch.setattr(dev, "_ensure_uv", lambda: calls.append("uv"))
    monkeypatch.setattr(dev, "_apply_dev_env", lambda: calls.append("env"))
    monkeypatch.setattr(packs, "main",
                        lambda argv: calls.append(f"main:{argv}") or 7)
    monkeypatch.setattr(sys, "argv", ["cdui", "packs", "install", "rag", "--yes"])

    assert dev._dispatch_packs_subcommand() == 7
    assert calls == ["venv", "uv", "env", "main:['install', 'rag', '--yes']"]


def test_dev_records_managed_env(monkeypatch, tmp_path):
    """`cdui dev` stamps the launch mode the API reports as ``launch_mode``."""
    seen: list = []

    class _FakeProc:
        stdout = None

        def wait(self):
            return 0

    def _fake_popen(cmd, **kw):
        seen.append(os.environ.get("CODEFYUI_MANAGED"))
        return _FakeProc()

    monkeypatch.setattr(dev.shutil, "which", lambda name: f"/fake/{name}")
    monkeypatch.setattr(dev, "_install_frontend_deps_if_needed", lambda: None)
    monkeypatch.setattr(dev, "_apply_dev_env", lambda: None)
    monkeypatch.setattr(dev, "_require_venv_tool", lambda name: f"/fake/{name}")
    monkeypatch.setattr(dev, "_stream", lambda *a, **kw: None)
    monkeypatch.setattr(dev, "BACKEND_DIR", tmp_path)
    monkeypatch.setattr(dev, "FRONTEND_DIR", tmp_path)
    monkeypatch.setattr(dev.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(sys, "argv", ["cdui", "dev"])

    dev.dev()

    # Both children -- the marker is set before either is spawned, so the
    # backend inherits it however the two are ordered.
    assert seen == ["dev", "dev"]


# ── fake pack states ──────────────────────────────────────────────────────


def _states(**overrides) -> dict:
    """A ``probe_all()`` answer for the whole catalog.

    Per pack: ``present`` (item ids on disk), ``pip_ready``, ``installed``,
    ``usable``, ``blocked_by``. Everything unnamed is "nothing installed".
    """
    from app.core.packs import catalog
    from app.core.packs.state import ItemState, PackState

    out: dict = {}
    for pack in catalog.iter_packs():
        spec = overrides.get(pack.pack_id, {})
        present = set(spec.get("present", ()))
        out[pack.pack_id] = PackState(
            pack_id=pack.pack_id,
            pip_ready=spec.get("pip_ready", False),
            items=tuple(
                ItemState(item_id=item.item_id, present=item.item_id in present,
                          sentinel=Path("nowhere"), snapshot_dir=None)
                for item in pack.items),
            installed=spec.get("installed", False),
            usable=spec.get("usable", False),
            blocked_by=tuple(spec.get("blocked_by", ())),
        )
    return out


@pytest.fixture
def probed(monkeypatch):
    """Install a fake ``probe_all`` and hand back a setter for it."""
    from app.core.packs import state

    holder: dict = {"value": _states()}
    monkeypatch.setattr(state, "probe_all", lambda: holder["value"])

    def _set(**overrides):
        holder["value"] = _states(**overrides)

    return _set


@pytest.fixture
def no_gpu_probe(monkeypatch):
    """Never shell out to a driver from a test."""
    from app.core.packs import restart, state

    monkeypatch.setattr(restart, "_detected", ("Fake GPU (driver 999)", "cu128"))
    monkeypatch.setattr(state, "torch_variant", lambda: "cpu")


# ── list / status ─────────────────────────────────────────────────────────


def test_cli_list_prints_every_pack(probed, capsys):
    from app.core.packs import catalog

    probed(**{"word-vectors": {"present": ["glove-50d"], "pip_ready": True,
                               "installed": True, "usable": True}})
    assert packs.main(["list"]) == 0

    out = capsys.readouterr().out
    assert out.isascii(), "the pack listing must survive a cp1252 console"
    for pack in catalog.iter_packs():
        assert pack.pack_id in out
        for item in pack.items:
            assert item.item_id in out
            assert item.license in out
    # Status words, and the sizes that make a download worth thinking about.
    assert "installed" in out
    assert "not installed" in out
    assert "present" in out
    assert "missing" in out
    assert "90.0 MB" in out          # all-MiniLM-L6-v2, 90_000_000 bytes


def test_cli_list_reports_a_half_installed_pack_as_partial(probed, capsys):
    probed(**{"sentence-embeddings": {"present": ["all-MiniLM-L6-v2"],
                                      "pip_ready": True, "usable": True}})
    assert packs.main(["list"]) == 0
    line = [ln for ln in capsys.readouterr().out.splitlines()
            if ln.strip().startswith("sentence-embeddings")][0]
    assert "partial" in line


def test_cli_status_adds_the_torch_variant_and_the_install_hints(
        probed, no_gpu_probe, capsys):
    assert packs.main(["status"]) == 0
    out = capsys.readouterr().out
    assert out.isascii()
    assert "cpu" in out                                  # installed variant
    assert "cdui packs install sentence-embeddings" in out
    assert "cdui install --gpu cu128" in out             # the GPU pack's hint


# ── install: the refusals, all before the flow runs ───────────────────────


@pytest.fixture
def flow_never_runs(monkeypatch):
    """Any call to the install flow is a test failure."""
    from app.core.packs import flows

    def _boom(*a, **kw):
        raise AssertionError("install_pack_live must not be reached")

    monkeypatch.setattr(flows, "install_pack_live", _boom)


def test_cli_install_unknown_pack_exits_2(probed, flow_never_runs, capsys):
    assert packs.main(["install", "sentance-embedings"]) == 2
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert "sentance-embedings" in text
    for known in ("sentence-embeddings", "word-vectors", "rag", "gpu-torch"):
        assert known in text


def test_cli_install_gpu_torch_prints_cdui_install_hint_exits_2(
        probed, no_gpu_probe, flow_never_runs, capsys):
    assert packs.main(["install", "gpu-torch"]) == 2
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert "GPU PyTorch is switched with: cdui install --gpu cu128" in text


def test_cli_install_blocked_dependency_exits_2(probed, flow_never_runs, capsys):
    probed(**{"rag": {"blocked_by": ("sentence-embeddings",)}})
    assert packs.main(["install", "rag"]) == 2
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert "cdui packs install sentence-embeddings" in text


def test_cli_install_unknown_item_exits_2(probed, flow_never_runs, capsys):
    assert packs.main(["install", "sentence-embeddings", "--items", "nope"]) == 2
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert "nope" in text
    assert "all-MiniLM-L6-v2" in text


# ── install: driving the flow ─────────────────────────────────────────────


@pytest.fixture
def fake_flow(monkeypatch):
    """Replace the install flow with a scripted event sequence.

    Returns a recorder; set ``rec["events"]`` to change the script and
    ``rec["raise"]`` to make the flow fail.
    """
    from app.core.packs import flows

    rec: dict = {
        "calls": [],
        "raise": None,
        "events": [
            {"type": "step_started", "step": "download:all-MiniLM-L6-v2",
             "label": "Downloading all-MiniLM-L6-v2"},
            {"type": "log", "line": "fetching config.json"},
            {"type": "progress", "item": "all-MiniLM-L6-v2",
             "bytes_done": 12_300_000, "bytes_total": 30_000_000,
             "percent": 41.0},
            {"type": "progress", "item": "all-MiniLM-L6-v2",
             "bytes_done": 12_600_000, "bytes_total": 30_000_000,
             "percent": 42.0},
            {"type": "step_done", "step": "download:all-MiniLM-L6-v2"},
            # Not every progress frame counts bytes. The GloVe conversion
            # counts LINES and says so in ``text``; 400,000 of them rendered
            # as "0.4/0.4 MB" is a number with somebody else's unit on it.
            {"type": "progress", "item": "glove-50d",
             "bytes_done": 400_000, "bytes_total": 400_000, "percent": 100.0,
             "text": "Converting GloVe text to npz (one-time)"},
        ],
    }

    def _fake(pack, item_ids, *, emit, cancel_check):
        rec["calls"].append({"pack_id": pack.pack_id, "item_ids": item_ids,
                             "cancelled": cancel_check()})
        for event in rec["events"]:
            emit(dict(event))
        if rec["raise"] is not None:
            raise rec["raise"]
        return flows.InstallOutcome(
            pack_id=pack.pack_id, pip_installed=True,
            items_done=("all-MiniLM-L6-v2",))

    monkeypatch.setattr(flows, "install_pack_live", _fake)
    return rec


def test_cli_install_drives_flow_and_renders_progress(probed, fake_flow, capsys):
    assert packs.main(["install", "sentence-embeddings", "--yes"]) == 0

    assert fake_flow["calls"] == [
        {"pack_id": "sentence-embeddings", "item_ids": None, "cancelled": False}]

    out = capsys.readouterr().out
    assert out.isascii(), "install output must survive a cp1252 console"
    assert "Downloading all-MiniLM-L6-v2" in out    # the step label
    assert "fetching config.json" in out            # the log line
    assert "42%" in out                             # the progress percentage
    assert "12.6/30.0 MB" in out
    # A frame that says what it is doing says it INSTEAD of a size: the
    # conversion counts lines, and "0.4/0.4 MB" would be those lines wearing
    # a unit that belongs to a different quantity.
    assert "Converting GloVe text to npz (one-time)" in out
    assert "0.4/0.4 MB" not in out, "line counts rendered as megabytes"
    assert "[" in out and "#" in out                # the ASCII bar
    assert "\r" in out, "progress must redraw one line, not scroll"
    # The bar is closed before the summary, so the two never share a line.
    assert out.endswith("\n")


def test_cli_install_passes_only_the_items_the_pack_declares(
        probed, fake_flow, capsys):
    assert packs.main([
        "install", "sentence-embeddings", "--items",
        "all-MiniLM-L6-v2, bge-small-zh-v1.5", "--yes"]) == 0
    assert fake_flow["calls"][0]["item_ids"] == [
        "all-MiniLM-L6-v2", "bge-small-zh-v1.5"]


def test_cli_install_without_yes_and_without_a_terminal_refuses(
        probed, fake_flow, monkeypatch, capsys):
    """Fails closed rather than blocking on a prompt nobody will answer."""
    monkeypatch.setattr(packs, "_stdin_is_interactive", lambda: False)
    assert packs.main(["install", "sentence-embeddings"]) == 2
    assert fake_flow["calls"] == []
    captured = capsys.readouterr()
    assert "--yes" in captured.out + captured.err


def test_cli_install_declined_at_the_prompt_installs_nothing(
        probed, fake_flow, monkeypatch, capsys):
    monkeypatch.setattr(packs, "_stdin_is_interactive", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    assert packs.main(["install", "sentence-embeddings"]) == 1
    assert fake_flow["calls"] == []


def test_cli_install_prompt_shows_the_download_size(
        probed, fake_flow, monkeypatch, capsys):
    asked: list[str] = []
    monkeypatch.setattr(packs, "_stdin_is_interactive", lambda: True)
    monkeypatch.setattr("builtins.input",
                        lambda prompt="": asked.append(prompt) or "y")
    assert packs.main(["install", "word-vectors"]) == 0
    assert "69" in "".join(asked), "the prompt must say how many MB this is"


def test_cli_install_needs_restart_exits_3_with_command(
        probed, fake_flow, capsys):
    from app.core.packs.errors import PackNeedsRestart

    # The real command, in the shape ``flows._restart_command`` builds: a
    # ``uv pip install`` line to run with the server stopped. There is no
    # ``cdui packs install --restart``, so a CLI that printed one would send
    # the user from exit code 3 to a usage error.
    command = f"uv pip install --python {sys.executable} sentence-transformers"
    fake_flow["raise"] = PackNeedsRestart(
        "sentence-transformers cannot be installed while the server is running",
        command=command,
        hint=f"stop the server, then run:\n{command}\n\nResolutionImpossible")
    assert packs.main(["install", "sentence-embeddings", "--yes"]) == 3
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert command in text
    assert "stop the server, then run:" in text


def test_cli_install_failure_exits_1_with_hint(probed, fake_flow, capsys):
    from app.core.packs.errors import PackInstallError

    fake_flow["raise"] = PackInstallError("installing Sentence embeddings failed",
                                          hint="error: no such package")
    assert packs.main(["install", "sentence-embeddings", "--yes"]) == 1
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert "installing Sentence embeddings failed" in text
    assert "error: no such package" in text


def test_cli_install_out_of_disk_exits_1_and_says_how_short(
        probed, fake_flow, capsys):
    from app.core.packs.errors import PackInsufficientDisk

    fake_flow["raise"] = PackInsufficientDisk(
        "not enough free disk space: 700 MB needed, 120 MB free",
        needed=700_000_000, free=120_000_000)
    assert packs.main(["install", "sentence-embeddings", "--yes"]) == 1
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert "700" in text and "120" in text


def test_cli_install_cancelled_exits_130(probed, fake_flow, capsys):
    from app.core.packs.errors import PackCancelled

    fake_flow["raise"] = PackCancelled("install of sentence-embeddings cancelled")
    assert packs.main(["install", "sentence-embeddings", "--yes"]) == 130


def test_ctrl_c_sets_the_cancel_flag_instead_of_tracebacking(
        probed, monkeypatch, capsys):
    """SIGINT during an install must reach the flow as a cancel REQUEST.

    Raising KeyboardInterrupt out of the handler would unwind through the
    downloader's own cleanup, leaving a half-written cache and a traceback
    where a learner expects "cancelled".
    """
    import signal

    from app.core.packs import flows
    from app.core.packs.errors import PackCancelled

    def _fake(pack, item_ids, *, emit, cancel_check):
        assert cancel_check() is False
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler), "the CLI must own SIGINT while installing"
        handler(signal.SIGINT, None)                     # simulate Ctrl-C
        assert cancel_check() is True
        raise PackCancelled("cancelled")

    before = signal.getsignal(signal.SIGINT)
    monkeypatch.setattr(flows, "install_pack_live", _fake)
    assert packs.main(["install", "sentence-embeddings", "--yes"]) == 130
    assert signal.getsignal(signal.SIGINT) is before, "SIGINT was not restored"


# ── remove ────────────────────────────────────────────────────────────────


def test_cli_remove_calls_flow_and_prints_pip_hint(probed, monkeypatch, capsys):
    from app.core.packs import flows

    calls: list = []
    monkeypatch.setattr(flows, "remove_item",
                        lambda pack, item_id: calls.append((pack.pack_id, item_id)) or True)

    assert packs.main(["remove", "sentence-embeddings", "all-MiniLM-L6-v2"]) == 0
    assert calls == [("sentence-embeddings", "all-MiniLM-L6-v2")]

    out = capsys.readouterr().out
    assert out.isascii()
    assert "all-MiniLM-L6-v2" in out
    # pip packages are the user's to remove, so the CLI spells the command.
    assert "uv pip uninstall" in out
    assert "sentence-transformers" in out
    assert sys.executable in out


def test_cli_remove_says_so_when_the_bytes_are_still_there(
        probed, monkeypatch, capsys):
    from app.core.packs import flows

    probed(**{"word-vectors": {"present": ["glove-50d"], "pip_ready": True,
                               "installed": True, "usable": True}})
    monkeypatch.setattr(flows, "remove_item", lambda pack, item_id: False)
    assert packs.main(["remove", "word-vectors", "glove-50d"]) == 0
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert "glove-50d" in text
    # Removing the record is not freeing the space; say which happened.
    assert "disk" in text.lower()


def test_cli_remove_an_item_that_was_never_downloaded_says_nothing_to_remove(
        probed, monkeypatch, capsys):
    """"Nothing was there" and "we could not delete it" are different facts.

    ``remove_item`` returns False for both -- it only reports whether BYTES
    went -- so the CLI answers from what the probe already knew, rather than
    telling somebody who never downloaded a 69 MB table that a process is
    holding it open.
    """
    from app.core.packs import flows

    monkeypatch.setattr(flows, "remove_item", lambda pack, item_id: False)
    assert packs.main(["remove", "word-vectors", "glove-50d"]) == 0
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert "glove-50d" in text
    assert "not downloaded" in text.lower()
    assert "disk" not in text.lower()


def test_cli_remove_unknown_item_exits_2(probed, monkeypatch, capsys):
    from app.core.packs import flows

    def _boom(*a, **kw):
        raise AssertionError("remove_item must not be reached")

    monkeypatch.setattr(flows, "remove_item", _boom)
    assert packs.main(["remove", "word-vectors", "glove-51d"]) == 2
    captured = capsys.readouterr()
    assert "glove-50d" in captured.out + captured.err


def test_cli_remove_unknown_pack_exits_2(probed, capsys):
    assert packs.main(["remove", "nope", "glove-50d"]) == 2


# ── the allowlist is the whole attack surface ─────────────────────────────


def test_cli_never_passes_user_strings_to_uv():
    """``packs.py`` cannot run a package manager, so it cannot be talked into
    running one with a spec somebody typed.

    The catalog is the allowlist; ``flows.install_pack_live`` is the only way
    in. This is a source-level check on purpose: a behavioural test can only
    show that today's arguments are refused, not that no path exists.
    """
    source = Path(packs.__file__).read_text(encoding="utf-8")
    for forbidden in ("subprocess", "os.system", "os.spawn", "os.exec",
                      "popen", "shell=True"):
        assert forbidden not in source, f"packs.py must not reference {forbidden}"

    tree = ast.parse(source)
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert "subprocess" not in imported

    # The only install entry point it reaches for.
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "install_pack_live" in called
    assert not {"run_pip", "download_hf_item", "download_asset_item"} & called


def test_help_needs_no_backend_import():
    """``cdui packs --help`` has to answer on a half-installed venv, so the
    parser must not depend on ``app`` -- every import of it is inside a
    command function."""
    tree = ast.parse(Path(packs.__file__).read_text(encoding="utf-8"))
    top_level = [node for node in tree.body
                 if isinstance(node, (ast.Import, ast.ImportFrom))]
    names = [alias.name for node in top_level
             if isinstance(node, ast.Import) for alias in node.names]
    names += [node.module or "" for node in top_level
              if isinstance(node, ast.ImportFrom)]
    assert not [n for n in names if n == "app" or n.startswith("app.")]

    with pytest.raises(SystemExit) as exc:
        packs.main(["--help"])
    assert exc.value.code == 0


@pytest.mark.parametrize("module", ["dev", "packs"])
def test_the_launcher_scripts_parse_on_the_oldest_python_we_support(module):
    """`requires-python = ">=3.10"`, and these two scripts are the ONLY files
    that have to prove it here rather than in CI.

    Every other module in this repo is imported by a test, so a syntax error
    stops the suite on any interpreter. `scripts/dev.py` and `scripts/packs.py`
    are different: they are what a user runs BEFORE there is a venv, on
    whatever Python they have, and this suite always runs on the venv's — so a
    3.11+-only construct in either one passes every check here and fails on
    the machine of the person trying to install. It has happened: a nested-
    quote f-string (3.12) went in and only the 3.11 venv noticed.

    `feature_version` makes the floor a fact this file states, on any
    interpreter, without one of each installed.
    """
    source = Path(sys.modules[module].__file__).read_text(encoding="utf-8")
    ast.parse(source, feature_version=(3, 10))


# ══ `cdui packs-run-pending` — the half of a restart-mode install that runs
# ══ while the server does not exist.
#
# The server writes down what it wanted, starts this command detached, and
# exits. This command waits for that process to go, runs the install, records
# what happened where the NEXT server will read it, and starts the server
# again. It runs from the OUTER interpreter with nothing importable, because
# for part of its run the venv it is installing into has no working torch in
# it — which is also why everything it needs from `app.core.packs.restart` is
# duplicated rather than imported.
#
# Two properties matter more than any other and are asserted several ways:
#
#   1. A pending file that is not ours is REFUSED and nothing is relaunched.
#      The file names an interpreter, a package index and a program to start;
#      it is read the way any other input is.
#   2. Once the server has been taken down, it comes back — install failed,
#      installer missing, disk full, does not matter. A user who asked for a
#      package and got no server back has lost more than the package.


def _restart_state():
    """The names dev.py and app.core.packs.restart must agree on."""
    from app.core.packs import restart

    return restart


def test_dev_and_restart_agree_on_the_restart_handshake():
    """The handshake is duplicated on purpose (dev.py cannot import ``app``),
    so the day the two copies drift, this is what says so.

    Everything here is a value that crosses the process gap: two environment
    variable names the server reads back, two schema numbers that decide
    whether a file is understood, the subcommand name the server spawns, and
    the two file names both halves open.
    """
    restart = _restart_state()
    assert dev.LAUNCHER_ENV == restart.LAUNCHER_ENV
    assert dev.RELAUNCH_ARGV_ENV == restart.RELAUNCH_ARGV_ENV
    assert dev.PENDING_SCHEMA == restart.PENDING_SCHEMA
    assert dev.OUTCOME_SCHEMA == restart.OUTCOME_SCHEMA
    assert dev.HELPER_COMMAND == restart.HELPER_COMMAND
    assert dev.STALE_PENDING_S == restart.STALE_PENDING_S


def test_dev_and_restart_agree_on_where_the_two_files_live(tmp_path,
                                                           monkeypatch):
    """Not just the names — the whole path, derived from the same variable.
    A helper that writes its outcome record where nobody reads it is a
    restart that silently reports nothing."""
    from app.core.packs import paths

    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path))
    assert dev._pending_restart_file() == paths.pending_restart_file()
    assert dev._last_restart_file() == paths.last_restart_file()
    assert dev._restart_log_file("abc123").parent == paths.job_log_dir()


def test_packs_run_pending_is_in_commands_and_skips_venv_exec():
    """Registered, and registered as a command that must NOT hop into the
    venv: it exists to rewrite that venv, and on Windows an interpreter
    cannot rewrite the directory it is running out of."""
    assert dev.COMMANDS[dev.HELPER_COMMAND] is dev.packs_run_pending
    assert dev.HELPER_COMMAND in dev._SKIP_VENV_EXEC
    assert dev.HELPER_COMMAND not in dev.SUBCOMMAND_GROUPS


def test_the_helper_does_not_let_the_uv_bootstrap_exit_for_it():
    """`_ensure_uv` exits(1) when it cannot download uv, and an exit before
    the relaunch is the one outcome this mechanism exists to prevent. The
    helper looks for uv itself and records a failed job instead — which
    relaunches (`test_run_pending_relaunches_when_uv_is_missing`).

    Asserted through a helper rather than by running `__main__`, which is not
    importable — the same reason `_subcommand_group` exists.
    """
    assert dev._needs_uv_bootstrap("start") is True
    assert dev._needs_uv_bootstrap("install") is True
    assert dev._needs_uv_bootstrap(dev.HELPER_COMMAND) is False


def test_packs_run_pending_is_not_advertised_in_the_help():
    """Deliberately undocumented. It is started by a server that is about to
    exit and takes a file naming a process to wait for; run by hand against a
    live server it would wait two minutes and then kill it."""
    assert dev.HELPER_COMMAND not in dev.__doc__


# ── the sandbox ───────────────────────────────────────────────────────────


class _FakeInstall:
    """Just enough of Popen for the installer: lines, then a return code."""

    def __init__(self, lines, returncode):
        self.stdout = io.StringIO("".join(f"{line}\n" for line in lines))
        self._returncode = returncode

    def wait(self):
        return self._returncode


class _FakeRelaunch:
    pid = 9999


class _Usage:
    """`shutil.disk_usage`'s answer, as much of it as the check reads."""

    def __init__(self, free):
        self.total = free * 2
        self.used = free
        self.free = free


@pytest.fixture
def helper(tmp_path, monkeypatch):
    """Sandbox `packs-run-pending`: a fake checkout, a fake uv, no waiting.

    Returns a recorder. `rec["control"]` is the directory both control files
    live in; set `rec["lines"]` / `rec["returncode"]` before the call to
    script the installer.
    """
    root = tmp_path / "checkout"
    backend = root / "backend"
    venv = backend / ".venv"
    venv.mkdir(parents=True)
    data = tmp_path / "data"
    control = data / "packs"
    control.mkdir(parents=True)

    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(data))
    monkeypatch.setattr(dev, "ROOT", root)
    monkeypatch.setattr(dev, "BACKEND_DIR", backend)
    monkeypatch.setattr(dev, "VENV", venv)
    monkeypatch.setattr(dev, "DIST_DIR", root / "frontend" / "dist")
    # Wall-clock waits, shrunk. The deadlines are what production needs; a
    # test that spent two real minutes proving one would never be run.
    monkeypatch.setattr(dev, "RESTART_WAIT_S", 0.02)
    monkeypatch.setattr(dev, "RESTART_KILL_GRACE_S", 0.02)
    monkeypatch.setattr(dev, "RESTART_POLL_S", 0.0)

    rec: dict = {
        "control": control, "venv": venv,
        "install": None, "install_kwargs": None,
        "relaunch": None, "relaunch_kwargs": None,
        "terminated": [], "alive_calls": [],
        "lines": ["Resolved 2 packages", "Installed 2 packages"],
        "returncode": 0,
    }

    def _fake_popen(cmd, **kw):
        cmd = list(cmd)
        if cmd[:1] == ["/fake/uv"]:
            rec["install"] = cmd
            rec["install_kwargs"] = kw
            return _FakeInstall(rec["lines"], rec["returncode"])
        rec["relaunch"] = cmd
        rec["relaunch_kwargs"] = kw
        return _FakeRelaunch()

    def _fake_alive(pid):
        rec["alive_calls"].append(pid)
        return not rec["terminated"]        # gone once it has been stopped

    monkeypatch.setattr(dev.shutil, "which",
                        lambda name: "/fake/uv" if name == "uv" else None)
    monkeypatch.setattr(dev.shutil, "disk_usage",
                        lambda path: _Usage(50 * 1024 ** 3))
    monkeypatch.setattr(dev, "_pid_alive", lambda pid: False)
    monkeypatch.setattr(dev, "_terminate_pid",
                        lambda pid: rec["terminated"].append(pid))
    monkeypatch.setattr(dev.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(dev, "LANG", "en")
    rec["alive"] = _fake_alive
    return rec


def _pending(helper, **overrides) -> Path:
    """A pending file this installation's server could have written."""
    data = {
        "schema": 1,
        "job_id": "job-1",
        "pack_id": "gpu-torch",
        "kind": "torch",
        "index_url": "https://download.pytorch.org/whl/cu128",
        "packages": ["torch", "torchvision"],
        "specs": [],
        "venv_python": str(helper["venv"] / "python"),
        "server_pid": 4242,
        "launcher": [sys.executable, str(Path(dev.__file__).resolve())],
        "relaunch_argv": ["--host", "0.0.0.0", "--port", "9100"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    data.update(overrides)
    path = helper["control"] / "pending_restart.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _outcome(helper) -> dict:
    return json.loads(
        (helper["control"] / "last_restart_job.json").read_text(encoding="utf-8"))


# ── the happy path ────────────────────────────────────────────────────────


def test_run_pending_installs_records_and_relaunches(helper):
    path = _pending(helper)

    assert dev._run_pending_job(path) == 0

    record = _outcome(helper)
    assert record["schema"] == dev.OUTCOME_SCHEMA
    assert record["status"] == "ok"
    assert record["returncode"] == 0
    assert record["job_id"] == "job-1"
    assert record["pack_id"] == "gpu-torch"
    assert record["kind"] == "torch"
    assert "Installed 2 packages" in record["log_tail"]
    assert record["finished_at"]

    # The claim is gone, or it would refuse the next install for 15 minutes.
    assert not path.exists()
    # And the server is back, on the address the browser is still pointing at.
    assert helper["relaunch"] == [
        sys.executable, str(Path(dev.__file__).resolve()), "start",
        "--host", "0.0.0.0", "--port", "9100",
    ]


def test_run_pending_torch_and_pip_command_shapes(helper):
    """Two kinds, two command lines, and the helper must not have to guess
    which — that is what `kind` is in the file for."""
    dev._run_pending_job(_pending(helper))
    assert helper["install"] == [
        "/fake/uv", "pip", "install",
        "--python", str(helper["venv"] / "python"),
        # Without --reinstall-package uv leaves a wheel whose version
        # constraint is already satisfied alone, and a variant switch
        # (cu124 -> cu128) becomes a no-op.
        "--reinstall-package", "torch",
        "--reinstall-package", "torchvision",
        "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cu128",
    ]
    assert helper["install_kwargs"]["cwd"] == dev.BACKEND_DIR

    helper["install"] = None
    dev._run_pending_job(_pending(
        helper, kind="pip", index_url=None, packages=[],
        specs=["sentence-transformers>=3.0", "faiss-cpu"]))
    # No constraints file, on purpose: this is the install that REPLACES what
    # the server had loaded, and constraining it to what was already there
    # would pin the versions it exists to move.
    assert helper["install"] == [
        "/fake/uv", "pip", "install",
        "--python", str(helper["venv"] / "python"),
        "sentence-transformers>=3.0", "faiss-cpu",
    ]


def test_run_pending_never_installs_package_names_it_read_off_disk(helper):
    """`packages` describes the job; it is not spliced into an installer's
    argv. The file is the one input that must not be able to widen what gets
    installed."""
    dev._run_pending_job(_pending(helper, packages=["torch", "evil-package"]))
    assert "evil-package" not in helper["install"]
    assert helper["install"][-4:-2] == ["torch", "torchvision"]


# ── waiting for the server to go ──────────────────────────────────────────


def test_run_pending_waits_for_the_pid_then_terminates(helper, monkeypatch,
                                                       capsys):
    """Installing while the old process still holds the files is the exact
    failure a restart-mode install exists to avoid, so this is the one place
    the helper is allowed to be slow — and the one place it is allowed to
    stop being polite."""
    monkeypatch.setattr(dev, "_pid_alive", helper["alive"])

    assert dev._run_pending_job(_pending(helper)) == 0

    assert helper["terminated"] == [4242], "the server had to be stopped"
    assert helper["alive_calls"], "and it was asked nicely first"
    assert helper["install"] is not None, "the install still ran afterwards"
    assert "4242" in capsys.readouterr().out, "which branch happened is logged"


def test_run_pending_does_not_terminate_a_server_that_left_on_its_own(helper,
                                                                      capsys):
    """The design: the server schedules a SIGINT at itself the moment the
    helper is spawned. Killing it after that would skip its lifespan
    shutdown — the database never closes and in-flight runs are never
    retired."""
    assert dev._run_pending_job(_pending(helper)) == 0
    assert helper["terminated"] == []


def test_run_pending_installs_anyway_when_the_server_will_not_die(
        helper, monkeypatch, capsys):
    """uv will most likely fail on locked files, and that failure is worth
    recording. Giving up here would leave the user with neither the package
    nor (until the relaunch) a server."""
    monkeypatch.setattr(dev, "_pid_alive", lambda pid: True)

    assert dev._run_pending_job(_pending(helper)) == 0
    assert helper["terminated"] == [4242]
    assert helper["install"] is not None
    assert "still running" in capsys.readouterr().err


# ── refusals: no install, and NO RELAUNCH ─────────────────────────────────


@pytest.mark.parametrize("overrides, expected", [
    ({"schema": 2}, "schema"),
    ({"schema": True}, "schema"),
    ({"kind": "conda"}, "kind"),
    ({"server_pid": "4242"}, "server_pid"),
    ({"server_pid": 0}, "server_pid"),
    ({"launcher": []}, "launcher"),
    # Exactly two entries. A third is somebody's idea of extra arguments, and
    # the helper appends its own after them.
    ({"launcher": [sys.executable, str(Path(dev.__file__).resolve()), "-X"]},
     "launcher"),
    ({"relaunch_argv": "--host 0.0.0.0"}, "relaunch_argv"),
])
def test_run_pending_refuses_a_file_it_does_not_recognise(helper, overrides,
                                                          expected, capsys):
    path = _pending(helper, **overrides)

    assert dev._run_pending_job(path) == 2

    assert helper["install"] is None
    assert helper["relaunch"] is None, (
        "a file that is not ours names a server we did not take down")
    assert expected in capsys.readouterr().err
    assert _outcome(helper)["status"] == "failed"


def test_run_pending_accepts_a_venv_python_that_is_a_symlink(helper, tmp_path):
    """The bug this test exists for cost every POSIX user their server.

    `uv venv` symlinks `.venv/bin/python` straight at the uv-managed base
    interpreter (dev.py's own venv hop documents exactly this), so resolving
    the WHOLE path lands somewhere under ~/.local/share/uv -- outside the
    venv -- and every genuine pending file was refused with exit 2 and no
    relaunch. Windows copies the interpreter instead, which is why this was
    invisible on the machine it was written on.

    The parent is still resolved, so `..` and a symlinked DIRECTORY cannot
    smuggle the install elsewhere; only the leaf is left alone.
    """
    base = tmp_path / "uv-managed" / "python3.11"
    base.parent.mkdir(parents=True)
    base.write_text("")
    link = helper["venv"] / "python-linked"
    try:
        os.symlink(base, link)
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip("this platform or account cannot create symlinks")

    assert dev._run_pending_job(_pending(helper, venv_python=str(link))) == 0
    # And uv is pointed at the link, not at what it resolves to: the venv is
    # the environment being installed into.
    assert helper["install"][4] == str(link)


def test_run_pending_refuses_a_venv_python_that_climbs_out(helper, capsys):
    """The half of the check that resolving the parent still enforces."""
    escape = str(helper["venv"] / ".." / ".." / "python")

    assert dev._run_pending_job(_pending(helper, venv_python=escape)) == 2

    assert helper["install"] is None
    assert helper["relaunch"] is None
    assert "venv_python" in capsys.readouterr().err


def test_run_pending_refuses_a_launcher_interpreter_that_is_gone(helper, capsys):
    """`restart_available` checked this when the panel was drawn; minutes may
    have passed. Starting the install anyway would take the server down and
    then discover there is nothing to bring it back with."""
    path = _pending(helper, launcher=["/no/such/python",
                                      str(Path(dev.__file__).resolve())])

    assert dev._run_pending_job(path) == 2

    assert helper["relaunch"] is None
    assert "interpreter" in capsys.readouterr().err


def test_run_pending_refuses_a_foreign_venv_python(helper, capsys):
    """The interpreter to install INTO is the one field that decides which
    environment gets rewritten. A path outside this checkout's venv is a file
    somebody else wrote."""
    path = _pending(helper, venv_python="/usr/bin/python3")

    assert dev._run_pending_job(path) == 2

    assert helper["install"] is None
    assert helper["relaunch"] is None
    assert "venv_python" in capsys.readouterr().err
    record = _outcome(helper)
    assert record["status"] == "failed"
    assert "refused" in record["message"]


def test_run_pending_refuses_a_launcher_that_is_not_this_dev_py(helper, capsys):
    """The launcher is the program the helper STARTS when it is done. A
    pending file that names a different one is a request to run something
    else entirely, on a machine that may well have two checkouts."""
    path = _pending(helper, launcher=[sys.executable, "/somewhere/else/dev.py"])

    assert dev._run_pending_job(path) == 2

    assert helper["relaunch"] is None
    assert "launcher" in capsys.readouterr().err


def test_run_pending_refuses_a_torch_index_that_is_not_pytorchs(helper, capsys):
    """`--index-url` is where every wheel in the install comes from. It may
    only ever be one of the URLs this launcher itself would have used."""
    path = _pending(helper, index_url="https://evil.example/whl/cu128")

    assert dev._run_pending_job(path) == 2

    assert helper["install"] is None
    assert helper["relaunch"] is None
    assert "index_url" in capsys.readouterr().err


def test_run_pending_accepts_every_index_this_launcher_would_have_used(helper):
    """The allowlist is TORCH_INDEX_URLS itself, so a variant added to
    `cdui install --gpu` is installable by restart on the same day."""
    for variant, url in dev.TORCH_INDEX_URLS.items():
        if not url or url == "__skip__":
            continue
        helper["install"] = None
        assert dev._run_pending_job(_pending(helper, index_url=url)) == 0, variant
        assert helper["install"][-1] == url


def test_run_pending_refuses_a_file_that_is_not_there(helper, capsys):
    assert dev._run_pending_job(helper["control"] / "nope.json") == 2
    assert helper["relaunch"] is None


def test_run_pending_refuses_a_file_that_is_not_text(helper, capsys):
    path = helper["control"] / "pending_restart.json"
    path.write_bytes(b"\xff\xfe\x00not utf-8")
    assert dev._run_pending_job(path) == 2
    assert helper["relaunch"] is None


# ── failures that DO relaunch ─────────────────────────────────────────────


def test_run_pending_relaunches_even_when_install_fails(helper):
    """The single most important property in this file. Whatever went wrong
    is in the outcome record, which the server that comes back reads and
    shows; a user with no server has lost their queued runs and the page they
    were looking at as well."""
    helper["returncode"] = 1
    helper["lines"] = ["error: no such package"]
    path = _pending(helper)

    assert dev._run_pending_job(path) == 1

    record = _outcome(helper)
    assert record["status"] == "failed"
    assert record["returncode"] == 1
    assert "no such package" in record["log_tail"]
    assert helper["relaunch"][:3] == [
        sys.executable, str(Path(dev.__file__).resolve()), "start"]
    assert not path.exists(), "a failed job is still a finished job"


def test_run_pending_relaunches_when_uv_is_missing(helper, monkeypatch):
    monkeypatch.setattr(dev.shutil, "which", lambda name: None)
    assert dev._run_pending_job(_pending(helper)) == 1
    assert helper["install"] is None
    assert helper["relaunch"] is not None
    assert "uv" in _outcome(helper)["message"]


def test_run_pending_relaunches_when_the_installer_cannot_be_started(
        helper, monkeypatch):
    """`Popen` itself raising — a uv that `which` found and the OS then
    refused to execute."""
    def _boom(cmd, **kw):
        if list(cmd)[:1] == ["/fake/uv"]:
            raise OSError("Exec format error")
        helper["relaunch"] = list(cmd)
        return _FakeRelaunch()

    monkeypatch.setattr(dev.subprocess, "Popen", _boom)
    assert dev._run_pending_job(_pending(helper)) == 1
    assert helper["relaunch"] is not None
    assert _outcome(helper)["status"] == "failed"


def test_run_pending_relaunches_when_a_pip_job_lists_no_packages(helper):
    """Our server refuses this before writing the file (twice over), so it is
    a bug rather than a forgery — and the cure for a bug is a server the user
    can still reach."""
    assert dev._run_pending_job(
        _pending(helper, kind="pip", index_url=None, specs=[])) == 1
    assert helper["install"] is None
    assert helper["relaunch"] is not None


def test_run_pending_refuses_when_disk_is_short(helper, monkeypatch):
    """A torch install that runs out of disk halfway leaves a venv with no
    working torch in it — which is worse than the wheel the user already had.
    Checked against the venv's own filesystem, which is not necessarily the
    one the temp directory is on."""
    seen: list = []

    def _tight(path):
        seen.append(Path(path))
        return _Usage(int(1.2 * 1024 ** 3))

    monkeypatch.setattr(dev.shutil, "disk_usage", _tight)

    assert dev._run_pending_job(_pending(helper)) == 1

    assert helper["install"] is None, "nothing was downloaded"
    assert helper["relaunch"] is not None, "and the server still came back"
    assert seen and seen[0] == helper["venv"]
    message = _outcome(helper)["message"]
    assert "1.2 GB free" in message
    assert "3 GB" in message


def test_a_pip_job_needs_less_disk_than_a_torch_job(helper, monkeypatch):
    """A couple of hundred MB of wheels is not a multi-GB CUDA runtime, and
    refusing it on a machine that has room for it would be a refusal nobody
    could act on."""
    monkeypatch.setattr(dev.shutil, "disk_usage",
                        lambda path: _Usage(int(2 * 1024 ** 3)))

    assert dev._run_pending_job(_pending(helper)) == 1          # torch: needs 3
    assert helper["install"] is None

    assert dev._run_pending_job(_pending(
        helper, kind="pip", index_url=None, specs=["faiss-cpu"])) == 0
    assert helper["install"] is not None


def test_run_pending_installs_when_the_free_space_cannot_be_read(helper,
                                                                 monkeypatch):
    """An unknowable answer is not a "no". A network mount that will not
    report its size must not cost the user their install."""
    def _boom(path):
        raise OSError("not supported")

    monkeypatch.setattr(dev.shutil, "disk_usage", _boom)
    assert dev._run_pending_job(_pending(helper)) == 0


# ── the log tail ──────────────────────────────────────────────────────────


def test_run_pending_log_tail_is_capped(helper):
    """The record is read back by a browser. A full pip log is megabytes of
    resolver output, and the last forty lines are the part that says why."""
    helper["lines"] = [f"line {i}" for i in range(500)]
    dev._run_pending_job(_pending(helper))

    tail = _outcome(helper)["log_tail"].splitlines()
    assert len(tail) == dev.RESTART_LOG_TAIL_LINES
    assert tail[-1] == "line 499", "the END of the log, not the start"
    assert "line 0" not in tail


def test_run_pending_streams_the_installer_to_its_own_log(helper, capsys):
    """stdout here IS the job log file (the server redirected it), and it is
    the only record of an install nobody watched."""
    helper["lines"] = ["Resolved 12 packages", "Downloading torch (2.4 GB)"]
    dev._run_pending_job(_pending(helper))
    out = capsys.readouterr().out
    assert "Downloading torch (2.4 GB)" in out


# ── the relaunch ──────────────────────────────────────────────────────────


def test_the_relaunch_is_detached_and_logged(helper):
    dev._run_pending_job(_pending(helper))
    kwargs = helper["relaunch_kwargs"]
    if sys.platform == "win32":
        assert kwargs["creationflags"] & 0x00000008          # DETACHED_PROCESS
        assert kwargs["creationflags"] & dev.subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        assert kwargs["start_new_session"] is True
    # Never a pipe: this process is about to exit, and a pipe with nobody
    # left to read it fills up and blocks the server mid-start.
    assert kwargs["stdout"] is not dev.subprocess.PIPE
    assert (helper["control"] / "logs" / "restart-job-1.log").exists()


def test_a_relaunch_that_cannot_start_does_not_mask_the_outcome(helper,
                                                                monkeypatch):
    """The `finally` must not raise over the return value the exit code is
    built from -- and the record must say there is no server, without
    unsaying that the install worked.

    `status` stays "ok" on purpose: that is the one field the SPA reads to
    tell the user what happened to their package, and the install really did
    succeed. The second fact -- nobody can reach the server -- is only
    actionable from a terminal, so it goes where `cdui status` will find it.
    """
    def _popen(cmd, **kw):
        if list(cmd)[:1] == ["/fake/uv"]:
            return _FakeInstall(["ok"], 0)
        raise OSError("no such file")

    monkeypatch.setattr(dev.subprocess, "Popen", _popen)

    assert dev._run_pending_job(_pending(helper)) == 0

    record = _outcome(helper)
    assert record["status"] == "ok"
    assert record["relaunch"] == "failed"
    assert record["log_file"] in record["message"], (
        "the message has to name the log, for somebody with no server")
    assert "restart-job-1.log" in record["log_file"]


def test_a_relaunch_that_worked_is_recorded_as_well(helper):
    """Present and null while it is unknown, "ok" once it is -- so "not
    recorded yet" and "written by an older dev.py" stay different facts."""
    dev._run_pending_job(_pending(helper))
    assert _outcome(helper)["relaunch"] == "ok"


@pytest.mark.parametrize(
    "name", ["PYTHONHOME", "__PYVENV_LAUNCHER__", "PYTHONEXECUTABLE"])
def test_restart_child_env_drops_the_stdlib_pointers(monkeypatch, name):
    """A child must find its own standard library.

    `_relaunch_server` starts the OUTER interpreter, which then hops into the
    venv; a `PYTHONHOME` inherited from some earlier hop would aim both of
    them at a third interpreter's stdlib. That failure is invisible until an
    `import` deep in the startup path dies with "SRE module mismatch", in a
    detached process whose only output is a log file.
    """
    monkeypatch.setenv(name, "C:/uv/python/cpython-3.11-windows")
    assert name not in dev._restart_child_env()


@pytest.mark.parametrize("name", ["CODEFYUI_USER_DATA_DIR",
                                  "CODEFYUI_OUTER_PYTHON"])
def test_restart_child_env_keeps_what_the_handshake_needs(monkeypatch, name):
    """Sanitised, not rebuilt. `CODEFYUI_USER_DATA_DIR` is not something
    `start()` can rederive -- the relaunched server would read its outcome
    record out of a different directory than the one it was written in -- and
    `CODEFYUI_OUTER_PYTHON` is how the next restart finds a launcher at all.
    Handing over a scrubbed environment is how the SECOND restart of a session
    refuses with "this server was launched without CODEFYUI_LAUNCHER"."""
    monkeypatch.setenv(name, "kept")
    assert dev._restart_child_env()[name] == "kept"


def test_the_relaunch_carries_the_sanitised_environment(helper, monkeypatch):
    """Not `os.environ` and not a fresh one: this process's, minus the three
    variables that would point the new server at the wrong stdlib."""
    monkeypatch.setenv("PYTHONHOME", "C:/uv/python/cpython-3.11-windows")
    monkeypatch.setenv("CODEFYUI_MARKER", "kept")

    dev._run_pending_job(_pending(helper))

    env = helper["relaunch_kwargs"]["env"]
    assert "PYTHONHOME" not in env
    assert env["CODEFYUI_MARKER"] == "kept"
    assert env["CODEFYUI_USER_DATA_DIR"] == str(helper["control"].parent)


def test_the_installer_gets_the_sanitised_environment_too(helper, monkeypatch):
    """`uv` resolves `--python` itself, but everything it runs under that
    interpreter inherits from here."""
    monkeypatch.setenv("PYTHONHOME", "C:/uv/python/cpython-3.11-windows")
    monkeypatch.setenv("CODEFYUI_MARKER", "kept")

    dev._run_pending_job(_pending(helper))

    env = helper["install_kwargs"]["env"]
    assert "PYTHONHOME" not in env
    assert env["CODEFYUI_MARKER"] == "kept"


def test_the_outcome_is_written_before_the_claim_is_dropped(helper, monkeypatch):
    """The claim is what stops a second install starting; releasing it before
    there is something to read in its place leaves a window where the restart
    has neither a claim nor a result."""
    real = dev._write_restart_outcome
    seen: dict = {}

    def _spy(*args, **kwargs):
        seen["claim_still_there"] = (
            helper["control"] / "pending_restart.json").exists()
        return real(*args, **kwargs)

    monkeypatch.setattr(dev, "_write_restart_outcome", _spy)
    dev._run_pending_job(_pending(helper))

    assert seen["claim_still_there"] is True
    assert not (helper["control"] / "pending_restart.json").exists()


# ── the command wrapper ───────────────────────────────────────────────────


def test_the_helper_writes_its_own_pid_into_the_claim(helper, monkeypatch):
    """`cdui start` reads this back to tell a restart that is still working
    from one whose helper never started. Written after validation and BEFORE
    the wait, which is the only moment it can be written safely: the server
    has been told to exit and will not touch its own claim again, and no
    second helper exists yet.
    """
    path = _pending(helper)
    seen: dict = {}

    def _alive(pid):
        seen["claim"] = json.loads(path.read_text(encoding="utf-8"))
        return False

    monkeypatch.setattr(dev, "_pid_alive", _alive)
    dev._run_pending_job(path)

    assert seen["claim"]["helper_pid"] == os.getpid(), (
        "the pid must be on disk before the wait begins")
    assert seen["claim"]["job_id"] == "job-1", "and nothing else was lost"
    assert seen["claim"]["launcher"][1] == str(Path(dev.__file__).resolve())


def test_a_refused_claim_is_never_stamped_with_a_helper_pid(helper):
    """It is not ours. Writing into it would be claiming a job this process
    has no business finishing."""
    path = _pending(helper, schema=99)
    dev._run_pending_job(path)
    assert "helper_pid" not in json.loads(path.read_text(encoding="utf-8"))


def test_every_outcome_ends_the_log_with_its_exit_code(helper, capsys):
    """This runs detached into a log nobody is watching, and "the last thing
    in the log" is how a person tells an install that ended from one that was
    killed halfway."""
    dev._run_pending_job(_pending(helper))
    assert capsys.readouterr().out.strip().splitlines()[-1].endswith(
        "restart install finished: exit code 0 ===")

    helper["returncode"] = 1
    dev._run_pending_job(_pending(helper))
    assert capsys.readouterr().out.strip().splitlines()[-1].endswith(
        "restart install finished: exit code 1 ===")

    dev._run_pending_job(_pending(helper, schema=99))
    assert capsys.readouterr().out.strip().splitlines()[-1].endswith(
        "restart install finished: exit code 2 ===")


def test_the_outcome_is_written_next_to_the_claim_not_where_the_env_points(
        helper, monkeypatch, tmp_path):
    """Derived from the pending path handed in. A `CODEFYUI_USER_DATA_DIR`
    that changed between the server's launch and this process would otherwise
    file the report in a directory nobody reads."""
    elsewhere = tmp_path / "somewhere-else"
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(elsewhere))

    assert dev._run_pending_job(_pending(helper)) == 0

    assert (helper["control"] / "last_restart_job.json").exists()
    assert (helper["control"] / "logs" / "restart-job-1.log").exists()
    assert not elsewhere.exists(), "nothing was written where the env points"


def test_an_unreadable_claim_does_not_erase_the_last_real_outcome(helper):
    """A file that could not be parsed names no job. Writing "refused" over
    `last_restart_job.json` would erase the report of the last restart that
    DID run -- which is the thing the user is most likely opening the panel
    to read."""
    outcome = helper["control"] / "last_restart_job.json"
    outcome.write_text(json.dumps({
        "schema": 1, "pack_id": "gpu-torch", "status": "ok",
        "message": "keep me"}), encoding="utf-8")

    path = helper["control"] / "pending_restart.json"
    path.write_bytes(b"\xff\xfe not json at all")
    assert dev._run_pending_job(path) == 2
    assert json.loads(outcome.read_text(encoding="utf-8"))["message"] == "keep me"

    assert dev._run_pending_job(helper["control"] / "never-existed.json") == 2
    assert json.loads(outcome.read_text(encoding="utf-8"))["message"] == "keep me"


def test_a_claim_that_parses_but_is_refused_does_get_a_record(helper):
    """The other side of it: there IS a job named here, and the user asked
    for it, so they are owed an answer about it."""
    dev._run_pending_job(_pending(helper, schema=99))
    record = _outcome(helper)
    assert record["status"] == "failed"
    assert record["pack_id"] == "gpu-torch"
    assert "refused" in record["message"]


@pytest.mark.skipif(sys.platform != "win32",
                    reason="CREATE_NO_WINDOW is a Windows creation flag")
def test_pid_alive_does_not_flash_a_console_window(monkeypatch):
    """The helper polls this every half second for up to two minutes, from a
    detached process with no console of its own — one window per poll, over
    whatever the user is looking at while their server is away."""
    seen: dict = {}

    class _Out:
        stdout = ""

    def _run(cmd, **kwargs):
        seen.update(kwargs)
        return _Out()

    monkeypatch.setattr(dev.subprocess, "run", _run)
    dev._pid_alive(4242)

    assert seen["creationflags"] & dev.subprocess.CREATE_NO_WINDOW


class _Tasklist:
    """`subprocess.run`'s answer, with only what `_pid_alive` reads."""

    def __init__(self, stdout):
        self.stdout = stdout
        self.stderr = ""
        self.returncode = 0


@pytest.fixture
def tasklist(monkeypatch):
    """Force `_pid_alive` down its Windows branch and script what it read.

    `sys.platform` so the branch is a fact this file states rather than a
    property of the runner, and `CREATE_NO_WINDOW` because dev.py names the
    constant directly and POSIX has no such attribute. Returns a function that
    installs one canned stdout; the recorded kwargs come back with it.
    """
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(dev.subprocess, "CREATE_NO_WINDOW", 0x08000000,
                        raising=False)
    seen: dict = {}

    def _script(stdout):
        def _run(cmd, **kwargs):
            seen["cmd"] = list(cmd)
            seen.update(kwargs)
            return _Tasklist(stdout)

        monkeypatch.setattr(dev.subprocess, "run", _run)
        return seen

    return _script


def test_pid_alive_reads_tasklist_as_utf8_and_replaces_what_it_cannot(tasklist):
    """The decode happens on subprocess's reader THREAD, where an exception is
    printed and swallowed and the caller is handed `stdout=None` instead. Ask
    for utf-8 with replacement and there is nothing left to raise: every fact
    read out of this output is ASCII, so a mangled message is free."""
    seen = tasklist("")

    dev._pid_alive(4242)

    assert seen["encoding"] == "utf-8"
    assert seen["errors"] == "replace"


def test_pid_alive_is_false_when_tasklist_produced_no_output(tasklist):
    """`stdout=None` -- a decode that died on the reader thread, or a child
    that wrote nothing at all. `cdui start` crashed here with "argument of
    type 'NoneType' is not iterable" every time a restart-mode install left a
    stale pidfile behind, which is the one moment this path is load-bearing."""
    tasklist(None)

    assert dev._pid_alive(4242) is False


def test_pid_alive_is_false_for_a_translated_no_tasks_message(tasklist):
    """Windows TRANSLATES this one: a pid that is gone answers with a sentence
    in the console's own language, not an empty string. It names no pid, so
    the substring test is still the right question -- as long as reading it
    cannot raise."""
    tasklist("資訊: 沒有執行中的工作符合指定的條件。\n")

    assert dev._pid_alive(4242) is False


def test_pid_alive_is_true_for_the_row_that_names_the_pid(tasklist):
    """The other half, so "always False" cannot pass: a live pid comes back as
    a table row, and `/NH` means the row is all there is."""
    tasklist("python.exe                    4242 Console"
             "                    1    123,456 K\n")

    assert dev._pid_alive(4242) is True


def test_packs_run_pending_exits_with_the_jobs_code(helper, monkeypatch):
    path = _pending(helper)
    monkeypatch.setattr(sys, "argv", ["dev.py", dev.HELPER_COMMAND, str(path)])
    with pytest.raises(SystemExit) as exc:
        dev.packs_run_pending()
    assert exc.value.code == 0


def test_packs_run_pending_without_a_file_exits_2(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["dev.py", dev.HELPER_COMMAND])
    with pytest.raises(SystemExit) as exc:
        dev.packs_run_pending()
    assert exc.value.code == 2
    assert dev.HELPER_COMMAND in capsys.readouterr().err


# ── `cdui status` says a restart happened while nobody was looking ────────


def test_status_reports_a_pending_restart_and_a_recent_outcome(helper, capsys):
    _pending(helper)
    (helper["control"] / "last_restart_job.json").write_text(json.dumps({
        "schema": 1, "job_id": "job-0", "pack_id": "gpu-torch", "kind": "torch",
        "status": "failed", "returncode": 1,
        "message": "the installer exited with 1", "log_tail": "",
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }), encoding="utf-8")

    dev._print_restart_notice()

    out = capsys.readouterr().out
    assert "gpu-torch" in out
    assert "4242" in out, "which process the pending claim is waiting for"
    assert "the installer exited with 1" in out
    # The state word, not a synonym of it. `_pending_state` returns
    # "finishing"/"abandoned", `cdui start` says the same two, and the docs
    # teach them as the two states a claim can be in -- a dashboard that says
    # "in progress" instead makes a reader work out that they are one thing.
    assert "finishing" in out


def test_status_forgets_an_outcome_that_is_an_hour_old(helper, capsys):
    old = datetime.now(timezone.utc) - timedelta(seconds=dev.RESTART_NOTICE_S + 60)
    (helper["control"] / "last_restart_job.json").write_text(json.dumps({
        "schema": 1, "pack_id": "gpu-torch", "status": "ok",
        "message": "gpu-torch installed", "finished_at": old.isoformat(),
    }), encoding="utf-8")

    dev._print_restart_notice()
    assert capsys.readouterr().out == ""


def test_status_says_nothing_when_no_restart_has_ever_run(helper, capsys):
    dev._print_restart_notice()
    assert capsys.readouterr().out == ""


def test_the_status_dashboard_prints_the_restart_notice(monkeypatch):
    """Wiring, not rendering. A notice nothing calls is dead code, and the
    one place it has to appear is the screen somebody opens when the page
    did not come back."""
    called: list = []
    monkeypatch.setattr(dev, "_print_restart_notice", lambda: called.append(True))
    monkeypatch.setattr(dev, "_gpu_stats", lambda: [])
    monkeypatch.setattr(dev, "_running_server_pid", lambda: None)
    monkeypatch.setattr(dev, "_server_health_info", lambda *a, **kw: None)

    dev._render_dashboard(interval=0.0, first=False)

    assert called == [True]


def test_status_calls_an_abandoned_claim_abandoned(helper, capsys):
    """The same predicate `cdui start` decides on, so the two commands cannot
    describe one file differently to one confused user."""
    old = datetime.now(timezone.utc) - timedelta(seconds=dev.STALE_PENDING_S + 60)
    _pending(helper, created_at=old.isoformat())

    dev._print_restart_notice()

    out = capsys.readouterr().out
    assert "abandoned" in out
    assert "cdui start" in out, "and how to be rid of it"


def test_status_prints_the_log_to_read_when_something_failed(helper, capsys):
    (helper["control"] / "last_restart_job.json").write_text(json.dumps({
        "schema": 1, "pack_id": "gpu-torch", "status": "ok",
        "relaunch": "failed", "message": "gpu-torch installed — and the "
                                         "server could not be started again",
        "log_file": r"C:\logs\restart-job-1.log",
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }), encoding="utf-8")

    dev._print_restart_notice()

    out = capsys.readouterr().out
    assert "restart-job-1.log" in out, (
        "a failed relaunch leaves nobody a server to read the panel on")


def test_status_does_not_point_at_a_log_after_a_clean_install(helper, capsys):
    """Thousands of lines of uv output are noise after a success."""
    (helper["control"] / "last_restart_job.json").write_text(json.dumps({
        "schema": 1, "pack_id": "gpu-torch", "status": "ok", "relaunch": "ok",
        "message": "gpu-torch installed",
        "log_file": r"C:\logs\restart-job-1.log",
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }), encoding="utf-8")

    dev._print_restart_notice()
    assert "restart-job-1.log" not in capsys.readouterr().out


def test_status_survives_a_control_file_it_cannot_read(helper, capsys):
    """A dashboard that raises is a dashboard nobody can use to find out why
    the server did not come back."""
    (helper["control"] / "last_restart_job.json").write_bytes(b"\xff\xfe{")
    (helper["control"] / "pending_restart.json").write_text("[]", encoding="utf-8")
    dev._print_restart_notice()
    assert capsys.readouterr().out == ""
