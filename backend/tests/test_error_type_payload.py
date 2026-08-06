"""The error payload must carry the exception's class, not just its message.

`str(exc)` drops the class name -- `str(KeyError('tensor'))` is `"'tensor'"`.
The UI's beginner-friendly error mapping keyed off a `KeyError:` prefix that
this payload could never contain, so every rule in it was unreachable while
its own tests passed against strings the backend does not emit.

Sending `error_type` alongside `error` makes the mapping structural rather
than a guess at the message's shape.
"""

from __future__ import annotations

import pytest

from app.core.graph_engine import execute_graph
from app.core.node_base import BaseNode, DataType, PortDefinition
from app.core.node_registry import registry


def _start_node(nid="start"):
    return {"id": nid, "type": "Start", "data": {"params": {}}}


def _trigger(eid, src, tgt):
    return {"id": eid, "source": src, "target": tgt, "sourceHandle": "trigger", "type": "trigger"}


def _raising_node(name: str, exc: BaseException):
    class _Raiser(BaseNode):
        NODE_NAME = name
        CATEGORY = "Test"
        DESCRIPTION = "Raises a specific exception type"

        @classmethod
        def define_inputs(cls):
            return []

        @classmethod
        def define_outputs(cls):
            return [PortDefinition(name="out", data_type=DataType.ANY)]

        def execute(self, inputs, params):
            raise exc

    return _Raiser


async def _run_and_capture(name: str, exc: BaseException) -> dict:
    """Run a one-node graph that raises, return the emitted error detail."""
    captured: dict = {}

    async def on_progress(node_id, status, data):
        if status == "error":
            captured.update(data or {})

    registry._nodes[name] = _raising_node(name, exc)
    try:
        await execute_graph(
            [_start_node(), {"id": "1", "type": name, "data": {"params": {}}}],
            [_trigger("et", "start", "1")],
            on_progress=on_progress,
            error_mode="continue",
        )
    finally:
        registry._nodes.pop(name, None)
    return captured


@pytest.mark.asyncio
async def test_error_payload_names_the_exception_class():
    detail = await _run_and_capture("_TestKeyError", KeyError("tensor"))
    assert detail["error_type"] == "KeyError"


@pytest.mark.asyncio
async def test_message_stays_the_bare_str_of_the_exception():
    """The class name belongs in its own field, not glued onto the message."""
    detail = await _run_and_capture("_TestKeyError2", KeyError("tensor"))
    # This is exactly why the prefix scan could never work.
    assert detail["error"] == "'tensor'"
    assert "KeyError" not in detail["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc, expected",
    [
        (RuntimeError("mat1 and mat2 shapes cannot be multiplied (64x784 and 512x10)"), "RuntimeError"),
        (ValueError("epochs must be positive"), "ValueError"),
        (TypeError("unsupported operand"), "TypeError"),
        (ZeroDivisionError("division by zero"), "ZeroDivisionError"),
    ],
)
async def test_error_type_is_reported_for_every_exception_class(exc, expected):
    detail = await _run_and_capture(f"_TestRaise{expected}", exc)
    assert detail["error_type"] == expected
    assert detail["error"] == str(exc)


@pytest.mark.asyncio
async def test_a_custom_exception_class_reports_its_own_name():
    class GraphMisconfigured(RuntimeError):
        pass

    detail = await _run_and_capture(
        "_TestCustomExc", GraphMisconfigured("the optimizer has no parameters")
    )
    assert detail["error_type"] == "GraphMisconfigured"
