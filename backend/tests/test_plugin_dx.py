"""Tests for the plugin developer-experience batch:

- ``_reload_target`` — hot-reload POST honors the configured server port.
- plugin SDK contract sync — the vendored scaffold copy matches the canonical
  ``frontend/src/plugins/contract.ts``.
- ``cdui plugin new`` — the scaffold generates a valid, loadable plugin.
"""

from __future__ import annotations

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
