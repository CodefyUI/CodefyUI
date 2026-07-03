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
from app.main import _extra_allowed_host_entries


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
