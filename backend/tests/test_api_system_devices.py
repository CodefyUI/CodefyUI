"""``GET /api/system/devices`` carries per-device memory facts (#767)."""

from __future__ import annotations

import sys

import pytest

from app.api import routes_system


@pytest.mark.asyncio
async def test_every_device_has_a_memory_key(test_client):
    resp = await test_client.get("/api/system/devices")

    assert resp.status_code == 200
    data = resp.json()
    assert data["default"] in {d["value"] for d in data["devices"]}
    for device in data["devices"]:
        assert "memory" in device
        assert {"value", "label", "detail", "available"} <= device.keys()
    cpu = next(d for d in data["devices"] if d["value"] == "cpu")
    assert cpu["memory"]["source"] == "psutil"
    assert 0 < cpu["memory"]["free_bytes"] <= cpu["memory"]["total_bytes"]


@pytest.mark.asyncio
async def test_broken_probes_degrade_to_null_not_500(test_client, monkeypatch):
    """NVML raising and psutil raising: still 200, memory null."""

    class BrokenNVML:
        def nvmlInit(self):
            raise RuntimeError("NVML_ERROR_LIB_RM_VERSION_MISMATCH")

    def boom():
        raise OSError("no meminfo")

    monkeypatch.setitem(sys.modules, "pynvml", BrokenNVML())
    monkeypatch.setitem(sys.modules, "psutil",
                        type("P", (), {"virtual_memory": staticmethod(boom)}))
    monkeypatch.setattr(routes_system, "describe_accelerator", lambda: {
        "default": "cpu",
        "devices": [{"value": "cpu", "label": "CPU", "detail": "", "available": True}],
    })

    resp = await test_client.get("/api/system/devices")

    assert resp.status_code == 200
    assert resp.json() == {
        "default": "cpu",
        "devices": [{"value": "cpu", "label": "CPU", "detail": "",
                     "available": True, "memory": None}],
    }


@pytest.mark.asyncio
async def test_describe_accelerator_shape_is_unchanged(test_client):
    """The route adds ``memory`` and nothing else to what it already served."""
    from app.core.device_utils import describe_accelerator

    expected = describe_accelerator()
    resp = await test_client.get("/api/system/devices")

    served = resp.json()
    assert served["default"] == expected["default"]
    assert [{k: v for k, v in d.items() if k != "memory"} for d in served["devices"]] \
        == expected["devices"]
