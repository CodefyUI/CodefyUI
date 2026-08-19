import sys
from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


def _console_safe(text: str) -> str:
    """Return *text* with anything the console cannot encode replaced.

    ``print`` encodes with ``sys.stdout.encoding``, which on Windows is the
    ANSI codepage of the machine's locale -- cp950 on a Traditional Chinese
    install, cp932 on Japanese, and so on. Those encode Han characters fine,
    so a Chinese label works; what they do NOT cover is the long tail of
    Unicode a graph legitimately carries. A superscript ``ᵀ`` in a label
    (U+1D40, in Phonetic Extensions) is enough, and so is one stray character
    in text a language model generated.

    Before this, that raised ``UnicodeEncodeError`` out of ``execute`` -- so
    the node failed, and with it the whole run. Losing a five-minute training
    run at the final Print because the model wrote a character the console
    could not spell is the wrong trade by a wide margin: the console is a
    *convenience* mirror of the log, and the value itself is passed through
    untouched either way.

    So the console gets a lossy rendering and the run continues. The
    ``__log__`` string the UI displays is never passed through here -- it
    keeps the exact text, because the browser has no such limitation.
    """
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    return text


class PrintNode(BaseNode):
    NODE_NAME = "Print"
    CATEGORY = "Utility"
    DESCRIPTION = "Print input value to console and pass through"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY, description="Any value to print")]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY, description="Pass-through")]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="label", param_type=ParamType.STRING, default="", description="Label prefix"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        value = inputs.get("value")
        label = params.get("label", "")
        prefix = f"[{label}] " if label else ""
        text = f"{prefix}{value}"
        try:
            print(_console_safe(text))
        except Exception:
            # Nothing about echoing to a console is worth failing a run for --
            # a closed or redirected stdout included. The UI still gets
            # ``__log__``, which is where anyone actually reads this.
            pass
        return {"value": value, "__log__": text}
