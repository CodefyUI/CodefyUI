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
from app.core import plugin_loader
from app.core.node_registry import registry
from app.core.preset_registry import preset_registry


@pytest.fixture
def custom_nodes_dir(tmp_path, monkeypatch):
    """Redirect settings.CUSTOM_NODES_DIR at a temp dir for each test.

    A SUCCESSFUL upload calls ``_reload_all()`` -> ``rediscover_all()``,
    which clears and rebuilds the process-global node and preset registries
    from whatever the settings point at *right then*. That mutation outlives
    ``monkeypatch``, which only reverts the setting on teardown, not the
    registry state built while it pointed at this temp dir -- so a test here
    that reaches the success path wipes real custom nodes (e.g. ``AddScalar``
    from ``app/custom_nodes/example_custom_node.py``) out of the registry for
    every test that runs afterward in the same session.

    Both registries are therefore put back exactly as they were. Rediscovering
    the real directory is not enough since #519: a node uploaded here now
    really registers, and ``Ok`` below, which has no ports, would break every
    later ``GET /api/nodes`` in the session.
    """
    d = tmp_path / "custom_nodes"
    d.mkdir()
    monkeypatch.setattr(settings, "CUSTOM_NODES_DIR", d)
    # The rediscovery would also re-import every pack in this machine's real
    # lockfile; CI has none, and neither do these tests.
    monkeypatch.setattr(plugin_loader, "load_lockfile",
                        plugin_loader.empty_lockfile)
    nodes = dict(registry._nodes)
    presets = dict(preset_registry._presets)
    try:
        yield d
    finally:
        registry._nodes.clear()
        registry._nodes.update(nodes)
        preset_registry._presets.clear()
        preset_registry._presets.update(presets)


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


async def test_upload_still_accepts_code_with_no_denied_construct(test_client, custom_nodes_dir):
    """Non-regression: an ordinary node with no denied-attribute-name method
    at all still uploads fine -- proves ``denied_attributes`` did not turn
    into a blanket false-positive over every custom node, without claiming
    to test anything more specific than that. (A docstring claiming this
    guards "a method sharing a builtin's name" would be describing the test
    below, not this one -- this payload calls no such method, so it would
    pass identically whether or not ``denied_attributes`` were wired in at
    all.)"""
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


async def test_upload_rejects_the_plugins_own_method_sharing_a_denied_name(
    test_client, custom_nodes_dir
):
    """core#179 follow-up: ``denied_attributes`` is receiver-independent by
    design (the same cost ``script_policy.TIER0_DENIED_ATTRS`` already
    imposes on a script's own ``obj.save()``), so a custom node's OWN
    ``self.save(...)`` method -- not a call into numpy/torch at all -- is
    refused here too, and correctly so: unlike the CLI install path
    (``scripts/plugins.py``, which lifts this at ``--trust-author`` --
    see ``test_denied_attributes_is_refused_at_tier0_but_liftable_at_tier2``
    in ``test_plugin_cli.py``), a browser upload has no trust tier at all.
    Every upload is the zero-declaration case, so this stays refused
    unconditionally -- there is no "Tier 2" here to lift it at.

    The prior version of this test asserted the opposite outcome with a
    payload (a bare class, no method call at all) that could not have
    caught either answer being wrong -- it passed identically whether
    ``denied_attributes`` was wired in or not. This one actually calls the
    denied-named method, so it fails if the wiring is ever removed.
    """
    payload = (
        b"from app.core.node_base import BaseNode\n"
        b"class Ok(BaseNode):\n"
        b"    NODE_NAME = 'Ok'\n"
        b"    CATEGORY = 'Test'\n"
        b"    DESCRIPTION = ''\n"
        b"    def pwn(self, path):\n"
        b"        self.save(path)\n"
    )
    resp = await test_client.post(
        "/api/custom-nodes/upload",
        files={"file": ("ok.py", payload, "text/x-python")},
    )
    assert resp.status_code == 400
    assert not (custom_nodes_dir / "ok.py").exists()


@pytest.mark.parametrize("filename", ["__init__.py", "__main__.py"])
async def test_upload_refuses_a_name_kept_for_system_files(
    test_client, custom_nodes_dir, filename
):
    """The manager hides every name starting with ``__`` and refuses to
    delete one, so an upload under such a name could never be seen or
    removed from the UI. ``__init__.py`` is the one that did damage: in the
    default configuration this directory IS the ``app.custom_nodes`` package,
    and the upload answered 200 and replaced its tracked ``__init__.py``."""
    seeded = custom_nodes_dir / filename
    seeded.write_bytes(b"# seed\n")

    resp = await test_client.post(
        "/api/custom-nodes/upload",
        files={"file": (filename, b"x = 1\n", "text/x-python")},
    )
    assert resp.status_code == 400, resp.text
    assert "'__'" in resp.json()["detail"]
    assert seeded.read_bytes() == b"# seed\n"


@pytest.mark.parametrize("filename", ["__init__.py", "__helper.py.disabled"])
async def test_toggle_refuses_a_name_kept_for_system_files(
    test_client, custom_nodes_dir, filename
):
    """As delete does. Disabling ``__init__.py`` renamed the package's own
    file, and the list hides ``__`` names, so it could not be turned back on
    from the UI."""
    seeded = custom_nodes_dir / filename
    seeded.write_bytes(b"# seed\n")

    resp = await test_client.post("/api/custom-nodes/toggle",
                                  json={"filename": filename})

    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == "Cannot enable or disable system files"
    assert sorted(p.name for p in custom_nodes_dir.iterdir()) == [filename]
