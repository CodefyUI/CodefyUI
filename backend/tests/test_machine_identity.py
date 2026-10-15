"""``core.machine_identity``: is a worker on this machine? (#773)."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from app.core import machine_identity as mi


def test_the_digest_is_stable_prefixed_and_never_the_raw_id():
    one = mi.digest("abc123")
    assert one == mi.digest("abc123") and one != mi.digest("abc124")
    assert "abc123" not in one and len(one) == 64
    assert mi.digest(None) is None and mi.digest("") is None


def test_only_an_equal_known_identity_is_the_same_machine():
    assert mi.same_machine("x", "x")
    assert not mi.same_machine("x", "y")
    assert not mi.same_machine(None, None)
    assert not mi.same_machine(None, "x")


def test_linux_reads_etc_machine_id_then_dbus(tmp_path, monkeypatch):
    missing, dbus = tmp_path / "machine-id", tmp_path / "dbus-id"
    dbus.write_text(" ABCDEF\n", encoding="ascii")
    monkeypatch.setattr(mi, "_LINUX_PATHS", (missing, dbus))
    assert mi.raw_machine_id("linux") == "abcdef"
    missing.write_text("1234\n", encoding="ascii")
    assert mi.raw_machine_id("linux") == "1234"
    monkeypatch.setattr(mi, "_LINUX_PATHS", (tmp_path / "none",))
    assert mi.raw_machine_id("linux") is None


def test_macos_reads_the_platform_uuid_from_ioreg(monkeypatch):
    output = ('+-o J314sAP  <class IOPlatformExpertDevice>\n'
              '    "IOPlatformUUID" = "0A1B2C3D-AAAA-BBBB-CCCC-0123456789AB"\n')
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: SimpleNamespace(stdout=output))
    assert mi.raw_machine_id("darwin") == (
        "0a1b2c3d-aaaa-bbbb-cccc-0123456789ab")
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: SimpleNamespace(stdout="nothing"))
    assert mi.raw_machine_id("darwin") is None

    def boom(*_a, **_k):
        raise FileNotFoundError("ioreg")

    monkeypatch.setattr(subprocess, "run", boom)
    assert mi.raw_machine_id("darwin") is None


def test_windows_reads_machine_guid(monkeypatch):
    class _Key:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    fake = SimpleNamespace(
        HKEY_LOCAL_MACHINE=1, KEY_READ=1, KEY_WOW64_64KEY=2,
        OpenKey=lambda *a: _Key(),
        QueryValueEx=lambda key, name: ("GUID-1", 1))
    monkeypatch.setitem(__import__("sys").modules, "winreg", fake)
    assert mi.raw_machine_id("win32") == "guid-1"
    fake.QueryValueEx = lambda key, name: (_ for _ in ()).throw(OSError())
    assert mi.raw_machine_id("win32") is None


@pytest.mark.parametrize("raw", [None, "id-1"])
def test_machine_identity_is_the_digest_of_the_raw_id(monkeypatch, raw):
    mi.machine_identity.cache_clear()
    monkeypatch.setattr(mi, "raw_machine_id", lambda: raw)
    try:
        assert mi.machine_identity() == mi.digest(raw)
    finally:
        mi.machine_identity.cache_clear()
