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
import os
import sys
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
    assert "153" in "".join(asked), "the prompt must say how many MB this is"


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
