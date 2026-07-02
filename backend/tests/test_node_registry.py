"""Tests for the node registry."""

from app.config import settings
from app.core.node_base import BaseNode, DataType, PortDefinition
from app.core.node_registry import NodeRegistry


class DummyNode(BaseNode):
    NODE_NAME = "Dummy"
    CATEGORY = "Test"
    DESCRIPTION = "A test node"

    @classmethod
    def define_inputs(cls):
        return [PortDefinition(name="input", data_type=DataType.TENSOR)]

    @classmethod
    def define_outputs(cls):
        return [PortDefinition(name="output", data_type=DataType.TENSOR)]

    def execute(self, inputs, params):
        return {"output": inputs.get("input")}


def test_register_and_get():
    reg = NodeRegistry()
    reg.register(DummyNode)
    assert reg.get("Dummy") is DummyNode
    assert reg.get("NonExistent") is None


def test_discover_builtin_nodes():
    reg = NodeRegistry()
    count = reg.discover(settings.NODES_DIR, "app.nodes")
    assert count >= 23
    assert reg.get("Conv2d") is not None
    assert reg.get("Print") is not None
    assert reg.get("TrainingLoop") is not None


def test_discover_skips_non_issubclassable_members(monkeypatch, tmp_path):
    class GoodNode(DummyNode):
        NODE_NAME = "Good"

    marker = object()

    monkeypatch.setattr(
        "app.core.node_registry.pkgutil.walk_packages",
        lambda *args, **kwargs: [(None, "fake_nodes.good", False)],
    )
    monkeypatch.setattr(
        "app.core.node_registry.importlib.import_module",
        lambda name: object(),
    )
    monkeypatch.setattr(
        "app.core.node_registry.inspect.getmembers",
        lambda module, predicate: [("bad", marker), ("good", GoodNode)],
    )

    reg = NodeRegistry()
    assert reg.discover(tmp_path, "fake_nodes") == 1
    assert reg.get("Good") is GoodNode

def test_clear():
    reg = NodeRegistry()
    reg.register(DummyNode)
    assert len(reg.nodes) == 1
    reg.clear()
    assert len(reg.nodes) == 0


def test_node_definitions():
    """Verify all discovered nodes have valid definitions."""
    reg = NodeRegistry()
    reg.discover(settings.NODES_DIR, "app.nodes")
    for name, cls in reg.nodes.items():
        assert cls.NODE_NAME, f"{name} missing NODE_NAME"
        assert cls.CATEGORY, f"{name} missing CATEGORY"
        inputs = cls.define_inputs()
        outputs = cls.define_outputs()
        assert isinstance(inputs, list), f"{name} define_inputs didn't return list"
        assert isinstance(outputs, list), f"{name} define_outputs didn't return list"
        # At least one input or output
        assert len(inputs) + len(outputs) > 0, f"{name} has no ports"
