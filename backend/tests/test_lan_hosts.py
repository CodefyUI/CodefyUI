"""EXTRA_ALLOWED_HOSTS splitting + wildcard-bind interface whitelisting
(spec Sections 8/9). init_allowed_hosts mutates a module-global whitelist,
so every test here restores the conftest seed afterwards."""

from __future__ import annotations

import pytest

from app.config import settings
from app.core.auth import (
    allowed_hosts,
    host_is_allowed,
    init_allowed_hosts,
    local_interface_ips,
)
from app.main import (
    _extra_allowed_host_entries,
    _has_port,
    _reachable_urls,
)


@pytest.fixture(autouse=True)
def _restore_whitelist():
    yield
    init_allowed_hosts(settings.HOST, settings.PORT)  # conftest seed


def test_local_interface_ips_no_loopback_no_raise():
    ips = local_interface_ips()
    assert isinstance(ips, list)
    assert "127.0.0.1" not in ips


def test_extra_allowed_hosts_split_and_stripped(monkeypatch):
    monkeypatch.setattr(
        "app.config.settings.EXTRA_ALLOWED_HOSTS",
        " 192.168.1.20:8000, mybox:8000 ,,",
    )
    monkeypatch.setattr("app.config.settings.HOST", "127.0.0.1")
    assert _extra_allowed_host_entries() == [
        "192.168.1.20:8000", "mybox:8000",
    ]


def test_wildcard_bind_whitelists_each_interface_ip(monkeypatch):
    monkeypatch.setattr("app.config.settings.EXTRA_ALLOWED_HOSTS", "")
    monkeypatch.setattr("app.config.settings.HOST", "0.0.0.0")
    monkeypatch.setattr("app.config.settings.PORT", 8000)
    monkeypatch.setattr(
        "app.main.local_interface_ips", lambda: ["192.168.9.9"])
    assert _extra_allowed_host_entries() == ["192.168.9.9:8000"]


def test_extra_entries_reach_the_whitelist():
    init_allowed_hosts("127.0.0.1", 8000, extra=["192.168.1.20:8000"])
    assert "192.168.1.20:8000" in allowed_hosts()
    assert host_is_allowed("192.168.1.20:8000")
    assert not host_is_allowed("attacker.example:8000")  # rebinding stays closed


# ── reachable-at startup log (robust IPv6 filtering) ─────────────────────


def test_has_port_is_structural():
    assert not _has_port("::1")
    assert not _has_port("[::1]")
    assert not _has_port("127.0.0.1")
    assert not _has_port("localhost")
    assert _has_port("127.0.0.1:8000")
    assert _has_port("[::1]:8000")
    assert _has_port("localhost:8000")
    # Five-digit ports were the regression trap for a parse-based check:
    # "::1:8000" parses as a valid IPv6 address (4-digit hextet) while
    # "::1:54321" does not, so ipaddress-based filtering flipped between
    # ports. Structure does not care about digit count.
    assert _has_port("[::1]:54321")
    assert _has_port("192.168.1.20:54321")


def test_reachable_urls_excludes_malformed_bare_ipv6_lines():
    init_allowed_hosts("192.168.1.20", 8000)
    urls = _reachable_urls()
    # Bare loopback aliases (no port) must never appear — these produced
    # the malformed "http://::1" line under the old bare ':' substring
    # filter (allowed_hosts() always carries a portless "::1" / "[::1]"
    # alongside their port-suffixed siblings; see init_allowed_hosts).
    assert "http://::1" not in urls
    assert "http://[::1]" not in urls
    assert "http://::1:8000" not in urls
    # The correctly bracketed host:port form IS printed.
    assert "http://[::1]:8000" in urls
    assert "http://127.0.0.1:8000" in urls
    assert "http://192.168.1.20:8000" in urls


def test_reachable_urls_clean_on_five_digit_ports():
    # Regression: with the old ipaddress-parse filter, "::1:54321" failed
    # to parse as IPv6 (five-digit "hextet") and the malformed line came
    # back on high ports. The whitelist no longer even contains the
    # unbracketed form, and the printable set stays clean.
    init_allowed_hosts("192.168.1.20", 54321)
    assert "::1:54321" not in allowed_hosts()
    urls = _reachable_urls()
    assert "http://::1:54321" not in urls
    assert "http://[::1]:54321" in urls
    assert "http://192.168.1.20:54321" in urls


def test_concrete_ipv6_bind_whitelists_bracketed_host_headers():
    # A concrete IPv6 bind must accept what clients actually send:
    # "Host: [fe80::1]:8123" (RFC 3986 brackets), never "fe80::1:8123".
    init_allowed_hosts("fe80::1", 8123)
    assert host_is_allowed("[fe80::1]:8123")
    assert host_is_allowed("[fe80::1]")
    assert "fe80::1:8123" not in allowed_hosts()
