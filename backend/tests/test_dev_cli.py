"""Unit tests for the cdui start --host/--port plumbing in scripts/dev.py
(spec Section 9). The full LAN behavior is exercised manually in the PR4
gate; these pin the pure helpers."""

from __future__ import annotations

import dev  # scripts/dev.py — conftest puts scripts/ on sys.path


def test_parse_host_port_defaults():
    assert dev._parse_host_port([]) == ("127.0.0.1", 8000)


def test_parse_host_port_flags_both_styles():
    assert dev._parse_host_port(
        ["--host", "0.0.0.0", "--port", "8080"]) == ("0.0.0.0", 8080)
    assert dev._parse_host_port(
        ["--host=192.168.1.20", "--port=9000"]) == ("192.168.1.20", 9000)
    # Unrelated flags pass through; a bogus port falls back to default.
    assert dev._parse_host_port(
        ["--foreground", "--port", "bogus"]) == ("127.0.0.1", 8000)


def test_probe_host_wildcard_answers_on_loopback():
    assert dev._probe_host("0.0.0.0") == "127.0.0.1"
    assert dev._probe_host("::") == "127.0.0.1"
    assert dev._probe_host("192.168.1.20") == "192.168.1.20"


def test_server_health_url():
    assert dev._server_health_url("0.0.0.0", 8080) == \
        "http://127.0.0.1:8080/api/health"
    assert dev._server_health_url("192.168.1.20", 8000) == \
        "http://192.168.1.20:8000/api/health"


def test_server_addr_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(dev, "SERVER_ADDRFILE", tmp_path / "server.addr")
    assert dev._server_addr() == ("127.0.0.1", 8000)   # absent -> defaults
    dev.SERVER_ADDRFILE.write_text("192.168.1.20:8080")
    assert dev._server_addr() == ("192.168.1.20", 8080)
    dev.SERVER_ADDRFILE.write_text("garbage")
    assert dev._server_addr() == ("127.0.0.1", 8000)   # corrupt -> defaults


def test_display_url():
    assert dev._display_url("127.0.0.1", 8000) == "http://localhost:8000"
    assert dev._display_url("0.0.0.0", 8080) == "http://localhost:8080"
    assert dev._display_url("192.168.1.20", 8080) == \
        "http://192.168.1.20:8080"


def test_local_ips_returns_non_loopback_strings():
    ips = dev._local_ips()
    assert isinstance(ips, list)
    assert "127.0.0.1" not in ips
    assert all(isinstance(ip, str) for ip in ips)
