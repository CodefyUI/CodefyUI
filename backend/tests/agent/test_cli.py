"""``cdui agent``: argument parsing, same-host defaults, start/stop (#771)."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.agent import cli

BACKEND_DIR = Path(__file__).resolve().parents[2]


@pytest.fixture
def user_data(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "_REPO_ROOT", tmp_path / "no-repo")
    monkeypatch.delenv(cli.TOKEN_ENV, raising=False)
    return tmp_path


def test_parser_defaults_and_flags():
    args = cli.build_parser().parse_args(["start"])
    assert (args.control, args.token, args.name, args.devices,
            args.foreground, args.daemonized) == (
        None, None, None, None, False, False)
    args = cli.build_parser().parse_args(
        ["start", "--control", "http://h:1", "--token", "t", "--name", "gpu",
         "--devices", "cuda:0,cuda:1", "--foreground"])
    assert (args.control, args.token, args.name, args.devices,
            args.foreground) == ("http://h:1", "t", "gpu", "cuda:0,cuda:1",
                                 True)
    assert cli.build_parser().parse_args(["stop"]).command == "stop"
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


def test_names(monkeypatch):
    assert cli.validate_name(" GPU-Box ") == "gpu-box"
    for bad in ("local", "any", "-x", "a_b", "x" * 64):
        with pytest.raises(ValueError):
            cli.validate_name(bad)
    monkeypatch.setattr(cli.socket, "gethostname",
                        lambda: "My_MacBook Air.local")
    assert cli.default_name() == "my-macbook-air"
    monkeypatch.setattr(cli.socket, "gethostname", lambda: "local")
    assert cli.default_name() == "worker-local"


def test_devices():
    assert cli.parse_devices(None) is None
    assert cli.parse_devices(" CUDA:0, cuda:1 ,") == ["cuda:0", "cuda:1"]
    with pytest.raises(ValueError):
        cli.parse_devices(" , ")
    facts = [SimpleNamespace(value="cpu", kind="cpu"),
             SimpleNamespace(value="cuda:0", kind="cuda"),
             SimpleNamespace(value="cuda:1", kind="cuda")]
    assert cli.offered_reports(facts, None) == facts
    assert [f.value for f in cli.offered_reports(facts, ["cuda:1"])] == [
        "cpu", "cuda:1"]
    with pytest.raises(ValueError):
        cli.offered_reports(facts, ["cuda:7"])


def test_devices_are_parsed_in_the_protocols_spelling():
    """``--devices`` speaks ``worker_protocol.canonical_device``: ``mps``
    and ``mps:0`` both name the probe's ``mps``, ``cuda:01`` is ``cuda:1``,
    and a typo is refused with the accepted forms."""
    assert cli.parse_devices("MPS, cuda:01") == ["mps:0", "cuda:1"]
    with pytest.raises(ValueError, match="cuda:N"):
        cli.parse_devices("gpu0")
    facts = [SimpleNamespace(value="cpu", kind="cpu"),
             SimpleNamespace(value="mps", kind="mps")]
    for wanted in ("mps", "mps:0"):
        assert [f.value for f in cli.offered_reports(
            facts, cli.parse_devices(wanted))] == ["cpu", "mps"]


def test_control_defaults_to_the_last_cdui_start(user_data):
    asked: list[str] = []

    def ask(prompt):
        asked.append(prompt)
        return ""

    # No server.addr: the prompt offers the default; non-TTY takes it.
    assert cli.resolve_control(None, interactive=True,
                               ask=ask) == "http://127.0.0.1:8000"
    assert len(asked) == 1
    assert cli.resolve_control(None, interactive=False) == (
        "http://127.0.0.1:8000")
    (user_data / "server.addr").write_text("0.0.0.0:8123")
    control = cli.resolve_control(None, interactive=True, ask=ask)
    assert callable(control) and control() == "http://127.0.0.1:8123"
    # Re-read at every dial: a restarted server is found again.
    (user_data / "server.addr").write_text("::1:9000")
    assert control() == "http://[::1]:9000"
    assert cli.resolve_control("http://x:1", interactive=True) == "http://x:1"
    assert len(asked) == 1


def test_token_defaults_to_the_session_token_file(user_data, monkeypatch):
    prompts: list[str] = []

    def ask(prompt):
        prompts.append(prompt)
        return "typed"

    assert cli.resolve_token("given", interactive=True, ask=ask) == "given"
    monkeypatch.setenv(cli.TOKEN_ENV, "from-env")
    assert cli.resolve_token(None, interactive=True, ask=ask) == "from-env"
    monkeypatch.delenv(cli.TOKEN_ENV)
    assert cli.resolve_token(None, interactive=True, ask=ask) == "typed"
    reader = cli.resolve_token(None, interactive=False, ask=ask)
    assert callable(reader) and reader() is None
    (user_data / "session.token").write_text("sess\n")
    reader = cli.resolve_token(None, interactive=True, ask=ask)
    assert reader() == "sess"
    assert len(prompts) == 1


def test_stop_without_an_agent(user_data, capsys):
    (user_data / cli.PIDFILE_NAME).write_text(str(os.getpid()))  # not an agent
    assert cli.main(["stop"]) == 0
    assert not (user_data / cli.PIDFILE_NAME).exists()  # stale: cleared


def test_dev_routes_the_agent_group():
    import dev

    assert dev._subcommand_group(["dev.py", "agent", "start"]) is (
        dev._dispatch_agent_subcommand)


def test_start_in_background_then_stop(user_data):
    """``start`` daemonises with a pidfile; ``stop`` ends it. The control
    plane is unreachable, so the agent just keeps redialling."""
    env = {**os.environ, "PYTHONPATH": str(BACKEND_DIR),
           "CODEFYUI_USER_DATA_DIR": str(user_data)}
    started = subprocess.run(
        [sys.executable, "-m", "app.agent", "start", "--control",
         "http://127.0.0.1:9", "--token", "t", "--name", "bg-test"],
        cwd=BACKEND_DIR, env=env, capture_output=True, text=True, timeout=60,
        stdin=subprocess.DEVNULL)
    assert started.returncode == 0, started.stderr + started.stdout
    pid = int((user_data / cli.PIDFILE_NAME).read_text())
    try:
        assert cli._agent_process(pid) is not None
        # A second start finds it and does not launch another.
        again = subprocess.run(
            [sys.executable, "-m", "app.agent", "start", "--control",
             "http://127.0.0.1:9", "--token", "t"], cwd=BACKEND_DIR, env=env,
            capture_output=True, text=True, timeout=60,
            stdin=subprocess.DEVNULL)
        assert str(pid) in again.stdout
    finally:
        stopped = subprocess.run(
            [sys.executable, "-m", "app.agent", "stop"], cwd=BACKEND_DIR,
            env=env, capture_output=True, text=True, timeout=60)
    assert stopped.returncode == 0, stopped.stderr
    deadline = time.monotonic() + 10
    while cli._agent_process(pid) is not None and time.monotonic() < deadline:
        time.sleep(0.1)
    assert cli._agent_process(pid) is None
    assert not (user_data / cli.PIDFILE_NAME).exists()
    log = (user_data / cli.LOGFILE_NAME).read_text(encoding="utf-8")
    assert "agent bg-test starting" in log
    # The token never reaches the command line.
    assert "--token" not in log


def test_start_without_a_token_fails_fast(user_data, capsys, monkeypatch):
    monkeypatch.setattr(cli, "_interactive", lambda: False)
    code = cli.main(["start", "--foreground", "--control", "http://x:1"])
    assert code == 2
    err = capsys.readouterr().err
    assert "session.token" in err and "--token" in err


def test_apple_silicon_offers_mps_without_torch(monkeypatch):
    monkeypatch.setattr(cli, "apple_silicon", lambda: True)
    facts = cli._probe_facts()
    mps = [f for f in facts if f.kind == "mps"]
    assert [(f.value, f.total_bytes, f.free_bytes) for f in mps] == [
        ("mps", None, None)]
    monkeypatch.setattr(cli, "apple_silicon", lambda: False)
    assert not [f for f in cli._probe_facts() if f.kind == "mps"]


def test_a_device_report_carries_the_cards_uuid():
    """The control plane matches a same-host card by it (#773)."""
    from app.core.resources import DeviceFact

    report = cli._report(DeviceFact("cuda:0", "cuda", 0, "RTX", 8, 4, "nvml",
                                    "abcd"))
    assert (report.value, report.uuid, report.source) == ("cuda:0", "abcd",
                                                          "nvml")
    assert cli._report(DeviceFact("cpu", "cpu", None, "", None, None,
                                  "none")).uuid is None
