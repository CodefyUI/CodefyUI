from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from .execution_context import ExecutionContext


class DataType(str, Enum):
    TENSOR = "TENSOR"
    MODEL = "MODEL"
    DATASET = "DATASET"
    DATALOADER = "DATALOADER"
    OPTIMIZER = "OPTIMIZER"
    LOSS_FN = "LOSS_FN"
    SCALAR = "SCALAR"
    STRING = "STRING"
    IMAGE = "IMAGE"
    LIST = "LIST"
    ANY = "ANY"
    TRIGGER = "TRIGGER"
    # A callable that maps one sample to one sample (core#136). What the
    # transform-chain nodes pass to each other and what a dataset node
    # accepts on its ``train_transform`` / ``eval_transform`` ports. The
    # value on the wire is always a ``torchvision.transforms.Compose``, so a
    # port that receives one can assign it straight to ``dataset.transform``.
    TRANSFORM = "TRANSFORM"


class ParamType(str, Enum):
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    BOOL = "bool"
    SELECT = "select"
    MODEL_FILE = "model_file"
    IMAGE_FILE = "image_file"
    # Tabular / plain-data upload (CSV, TSV, TXT, JSON). Same editor widget as
    # IMAGE_FILE — a dropdown of files already uploaded to DATA_FILES_DIR plus
    # an upload button — so a reader node never requires the learner to type a
    # filesystem path.
    DATA_FILE = "data_file"
    TENSOR_GRID = "tensor_grid"
    # Session-only secret (e.g. an API key). The editor masks the input and
    # never persists the value; the save endpoint and the publish pre-flight
    # both scrub SECRET params to "" so saved graphs never contain secrets.
    SECRET = "secret"
    # Multi-line Python source (core#131). The editor renders a real code
    # editor with syntax highlighting instead of a one-line text input, and
    # the node card shows the first lines as a preview. The value is still an
    # ordinary string param, so it participates in the cache key and in saved
    # graph JSON like any other.
    CODE = "code"


# Renderable-media kinds a port may declare (#117). Plain strings rather
# than a closed Enum so plugins and later node packs can introduce their own
# kind without a core release.
#
# ``MEDIA_IMAGE`` contract: the port's value is a base64-encoded PNG string
# (no ``data:`` prefix). A node that wants a different container should
# declare a new kind rather than overload this one.
MEDIA_IMAGE = "image"

# ``MEDIA_CHART`` contract (#130): the port's value is a JSON-safe chart
# *spec* — a dict the client renders with its own plotting components, so the
# picture is crisp, themed and hoverable instead of a server-rendered PNG.
#
#   {"kind": "bar" | "line" | "scatter" | "heatmap",   # required
#    "title": str, "x_label": str, "y_label": str,     # optional captions
#    ...payload keys for that kind}
#
# Every number in the spec must be finite: the run store serialises events
# with ``allow_nan=False``, so a NaN would be rewritten to ``null`` and the
# renderer would draw a hole. Producers substitute a real value (usually 0.0)
# and say so in the caption. See ``plugins/stats/README.md`` for the per-kind
# payload keys.
MEDIA_CHART = "chart"

# ``MEDIA_VIDEO`` contract (#310): the port's value is a small REFERENCE dict,
# never the bytes. Video cannot ride the event stream the way MEDIA_IMAGE
# does -- one ``node_status`` event is capped at
# ``RUN_EVENT_PAYLOAD_CAP_BYTES`` (128 KB), two orders of magnitude under a
# clip -- so the file lives under ``settings.MEDIA_DIR`` and the event carries
# where to find it:
#
#   {"path": str,     # required: POSIX-style path RELATIVE to MEDIA_DIR
#    "url": str,      # required: "/api/media/<path>", served with a real
#                     #   Content-Type so <video>/<img> can play it inline
#    "format": "mp4" | "gif" | "webm",                # required
#    "fps": float, "frames": int,                     # optional metadata the
#    "width": int, "height": int, "bytes": int}       #   client shows as-is
MEDIA_VIDEO = "video"


@dataclass
class PortDefinition:
    name: str
    data_type: DataType
    description: str = ""
    optional: bool = False
    # Declares that this port carries renderable media rather than plain
    # data, so the transport layer never has to guess. Before #117 the WS
    # layer sniffed "string longer than 200 chars whose first 20 chars are
    # alphanumeric => base64 PNG", which mislabelled ordinary long text (an
    # LLM answer, CJK prose, a token dump) as a broken image. Declaring is
    # the node's job; nothing downstream may infer it from the value.
    media: str | None = None


@dataclass
class ParamDefinition:
    name: str
    param_type: ParamType
    default: Any = None
    description: str = ""
    options: list[str] = field(default_factory=list)  # for SELECT type
    min_value: float | None = None
    max_value: float | None = None
    # Conditional visibility, evaluated client-side. When set, the param
    # only renders in the config panel / node card if every key in this
    # dict matches the live value of the corresponding sibling param.
    # Example: ``visible_when={"preset": "Custom"}`` on Conv2dExplicit's
    # ``weights`` param means the editor only shows when ``preset`` is
    # set to ``"Custom"``. None means always visible.
    #
    # A value may also be a LIST, meaning "any of these" (#134):
    # ``visible_when={"type": ["Adam", "AdamW"]}`` on Optimizer's
    # ``amsgrad``. Comparison is by string form on both sides, so a select
    # option and its literal spelling always agree.
    visible_when: dict[str, Any] | None = None
    # Two-tier parameter UI (#134). An advanced param is fully real —
    # persisted, exported, identical in every way — it is simply collected
    # behind a collapsed "Advanced" section so the default view of a node
    # stays readable in a classroom. Use it for the knobs that wrap a
    # library default nobody changes until they need to; leave the ones a
    # lesson actually turns basic.
    advanced: bool = False


def is_param_visible(
    definition: "ParamDefinition", params: dict[str, Any] | None,
) -> bool:
    """Would the editor show this param for *params*? The backend twin of
    the frontend's ``isParamVisible``, and it must agree with it.

    It exists because a hidden param is still PRESENT: the canvas writes
    every default onto a new node, and switching the sibling that hides one
    does not clear it. So a node asked to reject an inapplicable value would
    name a field the user cannot see and cannot reset -- "Optimizer
    'Adagrad' does not accept eps" on a form with no eps on it. For a
    param the current configuration hides, the honest reading of a leftover
    value is "not set".

    Comparison is by string form on both sides, and a list value means "any
    of", exactly as the frontend does it.
    """
    rule = definition.visible_when
    if not rule:
        return True
    live = params or {}
    for sibling, expected in rule.items():
        actual = str(live.get(sibling))
        accepted = expected if isinstance(expected, list) else [expected]
        if not any(str(candidate) == actual for candidate in accepted):
            return False
    return True


#: JS ``parseInt``'s numeric prefix: optional whitespace, optional sign,
#: then ASCII digits, and whatever follows is ignored. ``[0-9]`` rather than
#: ``\d``, which in Python also matches Devanagari and other Unicode digits
#: that ``parseInt`` rejects.
_LEADING_INT = re.compile(r"\s*([+-]?[0-9]+)")


def resolve_count_param(
    params: dict[str, Any] | None,
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    """A port-count param, clamped. The backend twin of ``clampCount``.

    ONE convention for every param that decides how many ports a node has --
    ``ComposeTransform``'s ``steps``, ``PythonScript``'s ``input_ports`` /
    ``output_ports``, ``Split``'s ``chunks``. Each used to spell its own
    ``int(raw)``, and each therefore disagreed with the canvas in the same
    way: the frontend reads these with ``parseInt(String(raw), 10)``, which
    takes a NUMERIC PREFIX, while ``int()`` demands the whole string. So
    ``{"steps": "5.7"}`` drew five handles and validated two, and
    ``{"steps": "6 chains"}`` drew six and validated two -- the canvas
    offering ports the engine then rejected as invalid input ports.

    Only reachable through hand-edited or externally-generated graph JSON;
    the INT widget cannot produce any of these. Clamps rather than raises,
    because the callers are the palette, the validator and the renderer,
    none of which has anywhere to put an exception.

    Deliberately NOT ``int(float(raw))``, which was the obvious fix and is
    the wrong one: it agrees on ``"5.7"`` but then reads ``"1e3"`` as 1000
    where ``parseInt("1e3", 10)`` is 1, trading one divergence for another.

    Booleans go to *default* -- ``String(true)`` is ``"true"``, which
    ``parseInt`` rejects -- rather than to Python's ``int(True) == 1``.
    """
    raw = (params or {}).get(name, default)
    parsed: int | None
    if isinstance(raw, bool):
        parsed = None
    elif isinstance(raw, (int, float)):
        parsed = math.floor(raw) if math.isfinite(raw) else None
    else:
        match = _LEADING_INT.match("" if raw is None else str(raw))
        parsed = int(match.group(1)) if match else None
    if parsed is None:
        parsed = default
    return max(minimum, min(maximum, parsed))


class BaseNode(ABC):
    NODE_NAME: str = ""
    CATEGORY: str = ""
    DESCRIPTION: str = ""

    # If False, graph_engine bypasses ExecutionCache for this node type.
    # The cache relies on "same params + same upstream => same output",
    # and there are four ways a node breaks it, each with its own issue:
    #
    #   1. it OWNS trainable parameters, which drift between runs
    #      (StatefulModuleMixin; #253 added SequentialModel);
    #   2. it HANDS OUT a live mutable handle -- a MODEL or an OPTIMIZER,
    #      or a scheduler -- that something downstream then mutates in
    #      place, so the recorded handle describes an object that no
    #      longer exists (#254);
    #   3. its product is a SIDE EFFECT the return value does not carry:
    #      a file write (#143), a metric point (#254), a trained model
    #      (#253);
    #   4. it reads external state a fingerprint cannot describe -- a
    #      network resource, credentials (#116, #144).
    #
    # ``test_cache_live_handle_nodes.py`` asserts 2 and 3 over the whole
    # registry, so a new node that breaks either fails there rather than
    # in somebody's loss curve.
    cacheable: ClassVar[bool] = True

    # Whether ``graph_engine.invoke_node`` moves the tensors in this node's
    # inputs onto the device the node runs on. True is right for anything
    # that computes with torch: it is what makes "one device per run" a
    # property of the ENGINE rather than of each author's memory.
    #
    # Set it False when the node's work is host-side by nature -- it hands
    # its input straight to numpy, sklearn, matplotlib or PIL. For those,
    # alignment is not merely wasted (a device round-trip per call); it is
    # WRONG, because ``Tensor.numpy()`` raises on anything but the CPU. That
    # is a crash the node cannot see coming and did not ask for, on a machine
    # its author may not own.
    #
    # It is also the answer for a node that takes a whole collection on a
    # LIST port and consumes it one element at a time (``Map``): aligning the
    # port materialises every element on the device at once, which is the
    # opposite of what streaming it was for. The elements still get aligned,
    # one at a time, when the body node is invoked.
    align_inputs: ClassVar[bool] = True

    @classmethod
    def cache_fingerprint(cls, params: dict[str, Any]) -> Any:
        """Extra state to fold into the cache key, beyond params and
        upstream keys (#144, #145).

        Returns ``None`` by default -- most nodes are fully described by
        their params. A node whose output depends on external state that
        params name but do not CONTAIN -- almost always a file or directory
        path -- overrides this to return something that changes when that
        state does, typically ``core.cache_fingerprint.path_fingerprint``
        (or the ``paths_``/``directory_`` variant) applied to the path(s)
        the params resolve to.

        The engine folds the return value into
        ``ExecutionCache.compute_key`` verbatim, so it only needs to be
        JSON-serialisable (a dict/tuple/scalar is fine). This is
        independent of ``cacheable``: returning a non-None fingerprint
        does not make an otherwise non-cacheable node cacheable, and a
        cacheable node with no external state simply returns None here.
        """
        return None

    @classmethod
    @abstractmethod
    def define_inputs(cls) -> list[PortDefinition]:
        ...

    @classmethod
    @abstractmethod
    def define_outputs(cls) -> list[PortDefinition]:
        ...

    @classmethod
    def define_outputs_dynamic(
        cls,
        params: dict[str, Any] | None = None,
    ) -> list[PortDefinition]:
        """Per-instance output ports, optionally expanded based on params.

        Default delegates to the static ``define_outputs()``. Nodes whose
        port count depends on a parameter (e.g. ``SplitNode`` whose
        ``chunks`` param decides how many ``chunk_i`` outputs exist) override
        this to return the live port list. Callers that have access to a
        node instance's params (graph validator, renderer) should prefer
        this; callers that only know the class (palette template, preset
        registry) keep using the static method.
        """
        return cls.define_outputs()

    @classmethod
    def define_inputs_dynamic(
        cls,
        params: dict[str, Any] | None = None,
    ) -> list[PortDefinition]:
        """Per-instance input ports. The mirror of ``define_outputs_dynamic``.

        Added for ``PythonScriptNode`` (core#131), whose ``input_ports`` param
        decides how many ``inN`` ports exist. Everything written before it
        varies only its outputs, so the default delegates to the static
        ``define_inputs()`` and nothing had to change.

        Overriders carry the same contract as the outputs form: tolerate
        ``None``/``{}``/garbage param values, clamp rather than raise, and
        keep ``define_inputs()`` returning the port set for the DEFAULT
        params, because that is what the palette template shows.
        """
        return cls.define_inputs()

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return []

    @abstractmethod
    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        progress_callback: Any | None = None,
        *,
        context: "ExecutionContext | None" = None,
    ) -> dict[str, Any]:
        ...
