"""Tests for the `cdui status` system dashboard (scripts/dev.py).

The dashboard is a btop / k9s-style snapshot: host + OS, CPU, memory, disk,
GPU and top processes, plus the CodefyUI server's own PID / health. These
tests cover the pure formatting helpers, the health-JSON parser, the watch
flag parsing, and that a full frame renders without raising on this machine
(with and without psutil available).
"""

from __future__ import annotations

import io
import json

import pytest

import dev  # scripts/dev.py — put on sys.path by conftest


# ── Formatting helpers ─────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "value, expected",
    [
        (None, "—"),
        (0, "0 B"),
        (512, "512 B"),
        (1024, "1.0 KiB"),
        (1536, "1.5 KiB"),
        (1024 ** 3, "1.0 GiB"),
        (1024 ** 4, "1.0 TiB"),
    ],
)
def test_human_bytes(value, expected):
    assert dev._human_bytes(value) == expected


@pytest.mark.parametrize(
    "pct, expected",
    [(0, dev.GREEN), (59.9, dev.GREEN), (60, dev.YELLOW),
     (84.9, dev.YELLOW), (85, dev.RED), (100, dev.RED)],
)
def test_pct_color(pct, expected):
    assert dev._pct_color(pct) == expected


def test_bar_fill_proportion(monkeypatch):
    # Force colour off so the bar is plain text we can count.
    monkeypatch.setattr(dev, "USE_COLOR", False)
    monkeypatch.setattr(dev, "GRAY", "")
    monkeypatch.setattr(dev, "RESET", "")
    monkeypatch.setattr(dev, "GREEN", "")
    monkeypatch.setattr(dev, "YELLOW", "")
    monkeypatch.setattr(dev, "RED", "")
    bar = dev._bar(50, width=10)
    assert bar.count("█") == 5
    assert bar.count("░") == 5
    # Clamping: out-of-range values saturate the bar.
    assert dev._bar(150, width=10).count("█") == 10
    assert dev._bar(-5, width=10).count("█") == 0


def test_bar_none_is_empty():
    bar = dev._bar(None, width=8)
    assert bar.count("█") == 0
    assert bar.count("░") == 8


def test_fmt_uptime_units(monkeypatch):
    monkeypatch.setattr(dev, "LANG", "en")
    assert dev._fmt_uptime(90) == "1m"
    assert dev._fmt_uptime(3 * 3600 + 5 * 60) == "3h 5m"
    assert dev._fmt_uptime(2 * 86400 + 3 * 3600) == "2d 3h 0m"


# ── Health JSON parsing ────────────────────────────────────────────────────

class _FakeResp:
    def __init__(self, status, body):
        self.status = status
        self._body = body.encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_server_health_info_ok(monkeypatch):
    payload = {"status": "ok", "nodes_loaded": 42, "presets_loaded": 3}
    monkeypatch.setattr(dev, "urlopen",
                        lambda *a, **k: _FakeResp(200, json.dumps(payload)))
    info = dev._server_health_info()
    assert info == payload
    assert dev._server_healthy() is True


def test_server_health_info_non_200(monkeypatch):
    monkeypatch.setattr(dev, "urlopen", lambda *a, **k: _FakeResp(500, "nope"))
    assert dev._server_health_info() is None
    assert dev._server_healthy() is False


def test_server_health_info_unreachable(monkeypatch):
    def boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(dev, "urlopen", boom)
    assert dev._server_health_info() is None


def test_server_health_info_bad_json(monkeypatch):
    monkeypatch.setattr(dev, "urlopen",
                        lambda *a, **k: _FakeResp(200, "{not json"))
    assert dev._server_health_info() is None


# ── Watch flag parsing ─────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "argv, isatty, expected",
    [
        # Continuous is the default on an interactive terminal...
        (["dev.py", "status"], True, True),
        (["dev.py", "status", "1"], True, True),
        # ...but a single frame when stdout is piped/redirected...
        (["dev.py", "status"], False, False),
        # ...or when --once / -1 is explicit (even on a TTY)...
        (["dev.py", "status", "--once"], True, False),
        (["dev.py", "status", "-1"], True, False),
        # ...and --watch forces the loop even off a TTY.
        (["dev.py", "status", "--watch"], False, True),
        (["dev.py", "status", "-w"], False, True),
    ],
)
def test_continuous_default(monkeypatch, argv, isatty, expected):
    monkeypatch.setattr(dev.sys, "argv", argv)
    monkeypatch.setattr(dev.sys, "stdout", io.StringIO())
    # io.StringIO.isatty() is always False; override just that for the test.
    monkeypatch.setattr(dev.sys.stdout, "isatty", lambda: isatty)
    assert dev._continuous_default() is expected


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["dev.py", "status"], 2.0),                  # default
        (["dev.py", "status", "1"], 1.0),            # bare positional
        (["dev.py", "status", "0.1"], 0.5),         # clamped to floor
        (["dev.py", "status", "--watch"], 2.0),       # flag, no value
        (["dev.py", "status", "-w", "1"], 1.0),
        (["dev.py", "status", "-w", "0.1"], 0.5),     # clamped to floor
        (["dev.py", "status", "-w", "abc"], 2.0),     # non-numeric → default
        (["dev.py", "status", "--once"], 2.0),        # only flags → default
    ],
)
def test_parse_watch_interval(monkeypatch, argv, expected):
    monkeypatch.setattr(dev.sys, "argv", argv)
    assert dev._parse_watch_interval() == expected


# ── GPU stats ──────────────────────────────────────────────────────────────

def test_gpu_stats_no_nvidia_smi(monkeypatch):
    monkeypatch.setattr(dev.shutil, "which", lambda _: None)
    assert dev._gpu_stats() == []


def test_gpu_stats_parses_csv(monkeypatch):
    monkeypatch.setattr(dev.shutil, "which", lambda _: "/usr/bin/nvidia-smi")

    class _Out:
        returncode = 0
        stdout = "NVIDIA RTX 4090, 37, 2048, 24576, 55\nbad line\n"

    monkeypatch.setattr(dev.subprocess, "run", lambda *a, **k: _Out())
    gpus = dev._gpu_stats()
    assert len(gpus) == 1
    g = gpus[0]
    assert g["name"] == "NVIDIA RTX 4090"
    assert g["util"] == 37.0
    assert g["temp"] == 55.0
    assert g["mem_total"] == 24576 * 1024 * 1024


def test_gpu_stats_command_fails(monkeypatch):
    monkeypatch.setattr(dev.shutil, "which", lambda _: "/usr/bin/nvidia-smi")

    def boom(*a, **k):
        raise OSError("exec failed")

    monkeypatch.setattr(dev.subprocess, "run", boom)
    assert dev._gpu_stats() == []


# ── Full-frame rendering ───────────────────────────────────────────────────

def test_render_dashboard_runs(monkeypatch, capsys):
    monkeypatch.setattr(dev, "_running_server_pid", lambda: None)
    monkeypatch.setattr(dev, "_server_health_info", lambda *a, **k: None)
    monkeypatch.setattr(dev.sys, "argv", ["dev.py", "status"])
    dev._render_dashboard(interval=0.0, first=True)
    out = capsys.readouterr().out
    assert "CPU" in out
    assert "http" not in out or "not running" in out or "未執行" in out


def test_render_dashboard_server_running(monkeypatch, capsys):
    monkeypatch.setattr(dev, "_running_server_pid", lambda: 12345)
    monkeypatch.setattr(
        dev, "_server_health_info",
        lambda *a, **k: {"nodes_loaded": 9, "presets_loaded": 2},
    )
    monkeypatch.setattr(dev.sys, "argv", ["dev.py", "status"])
    dev._render_dashboard(interval=0.0, first=False)
    out = capsys.readouterr().out
    assert "12345" in out
    assert "9" in out and "2" in out


def test_render_dashboard_without_psutil(monkeypatch, capsys):
    """When psutil isn't importable the frame still renders a degraded view."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("no psutil")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(dev, "_running_server_pid", lambda: None)
    monkeypatch.setattr(dev, "_server_health_info", lambda *a, **k: None)
    monkeypatch.setattr(dev.sys, "argv", ["dev.py", "status"])
    dev._render_dashboard(interval=0.0, first=True)
    out = capsys.readouterr().out
    assert "CPU" in out


# ── status() entry point ───────────────────────────────────────────────────

def test_status_oneshot_exits_nonzero_when_down(monkeypatch):
    monkeypatch.setattr(dev.sys, "argv", ["dev.py", "status", "--once"])
    monkeypatch.setattr(dev, "_running_server_pid", lambda: None)
    monkeypatch.setattr(dev, "_server_healthy", lambda *a, **k: False)
    monkeypatch.setattr(dev, "_server_health_info", lambda *a, **k: None)
    with pytest.raises(SystemExit) as exc:
        dev.status()
    assert exc.value.code == 1


def test_status_oneshot_ok_when_up(monkeypatch):
    monkeypatch.setattr(dev.sys, "argv", ["dev.py", "status", "--once"])
    monkeypatch.setattr(dev, "_running_server_pid", lambda: 999)
    monkeypatch.setattr(dev, "_server_healthy", lambda *a, **k: True)
    monkeypatch.setattr(dev, "_server_health_info",
                        lambda *a, **k: {"nodes_loaded": 1, "presets_loaded": 1})
    # Should return cleanly (no SystemExit) when the server is up.
    dev.status()


def test_status_loops_by_default_on_tty(monkeypatch):
    """Plain `cdui status` on a terminal loops (continuous is the default)."""
    monkeypatch.setattr(dev.sys, "argv", ["dev.py", "status"])
    monkeypatch.setattr(dev, "_continuous_default", lambda: True)
    monkeypatch.setattr(dev, "_running_server_pid", lambda: None)
    monkeypatch.setattr(dev, "_server_health_info", lambda *a, **k: None)

    calls = {"n": 0}

    def fake_render(interval, first):
        calls["n"] += 1
        raise KeyboardInterrupt

    monkeypatch.setattr(dev, "_render_dashboard", fake_render)
    monkeypatch.setattr(dev.sys, "stdout", io.StringIO())
    dev.status()
    assert calls["n"] == 1


def test_status_watch_single_frame(monkeypatch):
    """--watch loops until KeyboardInterrupt; feed one frame then interrupt."""
    monkeypatch.setattr(dev.sys, "argv", ["dev.py", "status", "-w", "1"])
    monkeypatch.setattr(dev, "_running_server_pid", lambda: None)
    monkeypatch.setattr(dev, "_server_health_info", lambda *a, **k: None)

    calls = {"n": 0}

    def fake_render(interval, first):
        calls["n"] += 1
        raise KeyboardInterrupt  # break out after the first frame

    monkeypatch.setattr(dev, "_render_dashboard", fake_render)
    # Capture cursor show/hide writes without touching the real terminal.
    monkeypatch.setattr(dev.sys, "stdout", io.StringIO())
    dev.status()  # should swallow KeyboardInterrupt and return
    assert calls["n"] == 1
