"""Tests for the custom-nodes upload route (``POST /api/custom-nodes/upload``).

This is the untrusted-user upload surface: whoever is sitting at the browser
picks the file, and ``upload_custom_node`` is the only gate between that file
and the interpreter. core#179 -- a library VALUE's own method escaping
import-time capability gating -- applies here exactly as it does to the CLI
install path in ``test_plugin_cli.py``; this file proves it for the HTTP
surface specifically, since ``routes_custom_nodes.py`` calls
``validate_python_source`` independently and does not inherit whatever
``scripts/plugins.py`` was told.
"""

from __future__ import annotations

import pytest

from app.config import settings
from app.core.node_registry import registry


@pytest.fixture
def custom_nodes_dir(tmp_path, monkeypatch):
    """Redirect settings.CUSTOM_NODES_DIR at a temp dir for each test.

    A SUCCESSFUL upload calls ``_reload_all()`` -> ``rediscover_all()``,
    which clears and rebuilds the process-global node registry from whatever
    ``settings.CUSTOM_NODES_DIR`` points at *right then*. That mutation
    outlives ``monkeypatch``, which only reverts the setting on teardown, not
    the registry state built while it pointed at this temp dir -- so a test
    here that reaches the success path wipes real custom nodes (e.g.
    ``AddScalar`` from ``app/custom_nodes/example_custom_node.py``) out of
    the registry for every test that runs afterward in the same session.
    Repair it on teardown the same way conftest.py's ``_ensure_registry_intact``
    repairs the built-in/plugin side.
    """
    d = tmp_path / "custom_nodes"
    d.mkdir()
    real_dir = settings.CUSTOM_NODES_DIR
    monkeypatch.setattr(settings, "CUSTOM_NODES_DIR", d)
    yield d
    registry.discover(real_dir, "app.custom_nodes")


async def test_upload_rejects_numpy_dump_to_an_arbitrary_path(test_client, custom_nodes_dir):
    """core#179: ``numpy.zeros(3).dump(path)`` is a Tier-0-library value's own
    method, not an import -- the capability gate never sees it, so it must be
    closed by ``denied_attributes``, exactly as it already is for in-canvas
    scripts. Before the fix ``upload_custom_node`` passed no kwargs at all
    to ``validate_python_source``, so this content was written to disk and
    the node registered."""
    payload = (
        b"import numpy\n"
        b"def pwn(path):\n"
        b"    numpy.zeros(3).dump(path)\n"
    )
    resp = await test_client.post(
        "/api/custom-nodes/upload",
        files={"file": ("bad.py", payload, "text/x-python")},
    )
    assert resp.status_code == 400
    assert not (custom_nodes_dir / "bad.py").exists()


async def test_upload_still_accepts_ordinary_node_code(test_client, custom_nodes_dir):
    """Non-regression: the new kwarg must not condemn ordinary plugin code
    that happens to have a method named the same as a builtin, matching the
    existing false-positive guard the shared walker already gives calls."""
    payload = (
        b"from app.core.node_base import BaseNode\n"
        b"class Ok(BaseNode):\n"
        b"    NODE_NAME = 'Ok'\n"
        b"    CATEGORY = 'Test'\n"
        b"    DESCRIPTION = ''\n"
    )
    resp = await test_client.post(
        "/api/custom-nodes/upload",
        files={"file": ("ok.py", payload, "text/x-python")},
    )
    assert resp.status_code == 200
    assert (custom_nodes_dir / "ok.py").exists()
