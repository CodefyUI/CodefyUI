"""PythonScript -- write Python directly on the canvas (core#131).

The deliberate escape hatch. Everything else in CodefyUI is a node someone
wrote as a file; this one is a node whose body the user types into the
browser. It exists because statistics and research work always outrun the
node library, and the alternatives (author a custom node, publish a plugin
pack) cost a round-trip through the filesystem for what is often four lines
of numpy.

The contract
------------
The script defines one function::

    def run(inputs, params):
        return {"out1": ...}

``inputs`` holds one key per *connected* input port (``in1``, ``in2``, ...);
an unwired port is simply absent, so use ``inputs.get("in2")`` when a port is
optional in your own design. ``params`` is a copy of this node's parameters.
A dict return maps keys to output ports; anything else becomes ``out1``.
Keys naming no declared port are dropped, and the drop is reported in the
Execution Log rather than silently swallowed.

The policy
----------
Two locks, and only the second one is a boundary.

Code is checked by :func:`app.core.script_policy.validate_script_source`
BEFORE it is compiled -- the same AST walker that gates plugin packs, run in
allowlist mode. The editor runs that same check on every keystroke, so a
rejection is a red banner while typing rather than a failed run ten minutes
later. Its rules are keyed on names, which is why it is the *first* line and
not the last one.

The namespace the script actually runs in never contains the host's real
module objects: :mod:`app.core.script_proxy` hands out restricted proxies
that judge what an attribute RESOLVES to -- a module off the Tier-0 list is
refused whatever it was called, so ``collections._sys``,
``statistics.random._os`` and ``sys.modules['os']`` all fail on the same
rule. Builtins are an explicit allowlist and ``__import__`` is replaced with
a guarded one that also returns proxies.

Read :mod:`app.core.script_policy` for the honest framing: this is a
guardrail, not a sandbox; it bounds the library surface a script can
navigate, not what the reachable functions do; and it contains neither CPU
nor memory.

Output capture
--------------
``print()`` is rebound per execution to a private buffer that becomes the
node's ``__log__`` entry. It is deliberately NOT
``contextlib.redirect_stdout``: nodes run on a shared thread pool, so
swapping the process-wide ``sys.stdout`` would swallow whatever a Print node
in the same level wrote at the same moment. The cost of the honest version
is that a library writing straight to ``sys.stdout`` is not captured; the
docs say so.
"""

from __future__ import annotations

import builtins
import inspect
import types
from typing import Any

from ...core.execution_context import ExecutionContext
from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)
from ...core.script_policy import (
    ESCAPE_HATCH_HINT,
    SCRIPT_FILENAME,
    TIER0_MODULES,
    validate_script_source,
)
from ...core.script_proxy import module_proxy, tier0_module_namespace

#: Upper bound on ports per side. Eight is past the point where a script is
#: really a subgraph, and it keeps the node card a sane height.
MAX_PORTS = 8

#: Characters of captured ``print()`` output kept per execution. A script
#: printing inside a training loop can produce megabytes; the log line is a
#: diagnostic, not a data channel.
MAX_CAPTURED_CHARS = 64_000

#: How much of the captured output is quoted back in a failure message.
_ERROR_LOG_TAIL = 800

DEFAULT_CODE = '''def run(inputs, params):
    """Edit me. Returns one value per output port.

    inputs -- one key per CONNECTED input port: in1, in2, ...
    params -- a copy of this node's parameters
    return -- {"out1": value, ...}; a bare value is treated as out1
    """
    x = inputs.get("in1")
    return {"out1": x}
'''

#: Builtins the script namespace exposes. An explicit ALLOWLIST, and it has
#: to be: the first cut of this node built ``vars(builtins)`` minus a
#: blocklist, which left ``__loader__`` -- CPython's BuiltinImporter -- bound
#: in the namespace, and::
#:
#:     m = __loader__.load_module('nt')   # the real os module
#:     getattr(m, 'getcwd')()             # a literal getattr is permitted
#:
#: was a complete escape from a node whose whole job is to not be one. A
#: blocklist over a namespace someone else owns is only ever correct until
#: that namespace grows, so this lists what goes IN.
#:
#: ``getattr``/``setattr``/``delattr`` are here deliberately -- the AST gate
#: permits them with a literal attribute name, and removing them would make
#: code the editor accepted fail at run time.
_ALLOWED_BUILTINS: tuple[str, ...] = (
    # Types and constructors
    "bool", "bytearray", "bytes", "complex", "dict", "float", "frozenset",
    "int", "list", "object", "set", "slice", "str", "tuple", "type",
    "memoryview", "range", "super", "property", "classmethod", "staticmethod",
    # Functions
    "abs", "all", "any", "ascii", "bin", "callable", "chr", "delattr",
    "divmod", "enumerate", "filter", "format", "getattr", "hasattr", "hash",
    "hex", "id", "isinstance", "issubclass", "iter", "len", "map", "max",
    "min", "next", "oct", "ord", "pow", "repr", "reversed", "round",
    "setattr", "sorted", "sum", "zip",
    # Constants
    "Ellipsis", "NotImplemented", "__debug__",
    # ``class X:`` compiles to a call to this; a script defining a small
    # helper class is ordinary Python, and ``type`` is already exposed.
    "__build_class__",
)


def _script_builtins_base() -> dict[str, Any]:
    """The allowlisted builtins plus the ``Exception`` hierarchy.

    Exceptions come from a live scan rather than a hard-coded list so a new
    CPython exception type does not silently become unavailable. Only
    ``Exception`` subclasses: ``SystemExit``, ``KeyboardInterrupt`` and
    ``GeneratorExit`` are BaseException-only for a reason, and a script has
    no business raising something the engine's ``except Exception`` cannot
    catch.
    """
    namespace: dict[str, Any] = {
        name: getattr(builtins, name)
        for name in _ALLOWED_BUILTINS
        if hasattr(builtins, name)
    }
    namespace.update({
        name: value
        for name, value in vars(builtins).items()
        if isinstance(value, type) and issubclass(value, Exception)
    })
    return namespace


def resolve_port_count(params: dict[str, Any] | None, name: str) -> int:
    """Clamp a port-count param into ``1..MAX_PORTS``, tolerating garbage."""
    raw = (params or {}).get(name, 1)
    try:
        count = int(raw)
    except (TypeError, ValueError):
        count = 1
    return max(1, min(MAX_PORTS, count))


def resolve_port_types(
    spec: Any,
    count: int,
    default: DataType,
) -> list[DataType]:
    """Resolve a per-port type list from the comma-separated *spec*.

    A short list repeats its last entry, so raising the port count never
    leaves the new ports untyped, and lowering it never loses the types of
    the ports that remain. Unknown names -- and ``TRIGGER``, which is not a
    data type any script could legally emit -- fall back to *default*.
    """
    if isinstance(spec, (list, tuple)):
        names = [str(item).strip() for item in spec]
    else:
        names = [part.strip() for part in str(spec or "").split(",")]
    names = [name for name in names if name]

    resolved: list[DataType] = []
    for index in range(count):
        raw = names[index] if index < len(names) else (names[-1] if names else "")
        try:
            data_type = DataType(raw.upper())
        except ValueError:
            data_type = default
        if data_type is DataType.TRIGGER:
            data_type = default
        resolved.append(data_type)
    return resolved


class _OutputCapture:
    """Per-execution ``print()`` sink with a hard size cap."""

    __slots__ = ("_chunks", "_length", "_truncated", "_limit")

    def __init__(self, limit: int = MAX_CAPTURED_CHARS) -> None:
        self._chunks: list[str] = []
        self._length = 0
        self._truncated = False
        self._limit = limit

    def write(self, text: str) -> None:
        if self._truncated:
            return
        remaining = self._limit - self._length
        if len(text) >= remaining:
            self._chunks.append(text[:remaining])
            self._truncated = True
        else:
            self._chunks.append(text)
            self._length += len(text)

    def print(
        self,
        *args: Any,
        sep: str = " ",
        end: str = "\n",
        file: Any = None,
        flush: bool = False,
    ) -> None:
        """Stand-in for the builtin. A caller-supplied *file* still wins."""
        text = sep.join(str(arg) for arg in args) + end
        if file is not None and hasattr(file, "write"):
            file.write(text)
            return
        self.write(text)

    def text(self) -> str:
        body = "".join(self._chunks)
        if self._truncated:
            body += f"\n... [output truncated at {self._limit} characters]"
        return body


def _guarded_import(
    name: str,
    globals: dict | None = None,   # noqa: A002 - matches builtins.__import__
    locals: dict | None = None,    # noqa: A002 - matches builtins.__import__
    fromlist: Any = (),
    level: int = 0,
) -> Any:
    """``__import__`` restricted to the Tier-0 allowlist, returning a PROXY.

    Two jobs. The allowlist check is the second lock on the door the AST gate
    already guards, for a graph that arrived by some path other than the
    editor. Wrapping the result is the important one: without it, ``import
    collections`` would hand back the real module and every attribute rule the
    proxy enforces would be one import statement away from irrelevant.

    ``builtins.__import__`` returns the root package for a bare ``import
    a.b`` and the named module when a ``fromlist`` is given; either way the
    proxy is built from what it returned, and ``IMPORT_FROM`` then resolves
    its names through the proxy's own policy.
    """
    if level:
        raise ImportError(f"Relative imports are not allowed.{ESCAPE_HATCH_HINT}")
    if name.split(".")[0] not in TIER0_MODULES:
        raise ImportError(f"Importing '{name}' is not allowed.{ESCAPE_HATCH_HINT}")
    imported = builtins.__import__(name, globals, locals, fromlist, level)
    if isinstance(imported, types.ModuleType):
        return module_proxy(imported)
    return imported  # pragma: no cover - __import__ always returns a module


def _tier0_namespace() -> dict[str, Any]:
    """The pre-bound allowlisted modules, each wrapped in its proxy."""
    return tier0_module_namespace()


class PythonScriptNode(BaseNode):
    NODE_NAME = "PythonScript"
    CATEGORY = "Utility"
    DESCRIPTION = (
        "Run Python you write on the canvas. Define run(inputs, params) and "
        "return a dict keyed by output port. The script may use "
        + ", ".join(TIER0_MODULES)
        + " and nothing else. That limits which LIBRARIES it can reach, not "
        "what they can do: this is a guardrail, not a sandbox, and the code "
        "runs in the CodefyUI process with your permissions. Only run "
        "scripts you trust."
    )

    # The code is an ordinary param, so ExecutionCache already keys on it:
    # editing the script re-executes this node and everything downstream.
    cacheable = True

    # ── Schema ───────────────────────────────────────────────────────────

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        # Baseline for the palette template: matches the default params.
        return cls.define_inputs_dynamic(None)

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return cls.define_outputs_dynamic(None)

    @classmethod
    def define_inputs_dynamic(
        cls,
        params: dict[str, Any] | None = None,
    ) -> list[PortDefinition]:
        count = resolve_port_count(params, "input_ports")
        types = resolve_port_types(
            (params or {}).get("input_types"), count, DataType.TENSOR
        )
        return [
            PortDefinition(
                name=f"in{i + 1}",
                data_type=types[i],
                description=f"inputs['in{i + 1}']",
                # A script decides for itself which ports it needs; requiring
                # every declared port would turn the port-count knob into a
                # trap (bump it to 4, graph stops validating).
                optional=True,
            )
            for i in range(count)
        ]

    @classmethod
    def define_outputs_dynamic(
        cls,
        params: dict[str, Any] | None = None,
    ) -> list[PortDefinition]:
        count = resolve_port_count(params, "output_ports")
        types = resolve_port_types(
            (params or {}).get("output_types"), count, DataType.ANY
        )
        return [
            PortDefinition(
                name=f"out{i + 1}",
                data_type=types[i],
                description=f"return {{'out{i + 1}': ...}}",
            )
            for i in range(count)
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        selectable = [dt.value for dt in DataType if dt is not DataType.TRIGGER]
        return [
            ParamDefinition(
                name="code",
                param_type=ParamType.CODE,
                default=DEFAULT_CODE,
                description=(
                    "Python source defining run(inputs, params). Checked "
                    "against the Tier-0 policy on every edit."
                ),
            ),
            ParamDefinition(
                name="input_ports",
                param_type=ParamType.INT,
                default=1,
                description=f"Number of input ports in1..inN (1..{MAX_PORTS})",
                min_value=1,
                max_value=MAX_PORTS,
            ),
            ParamDefinition(
                name="output_ports",
                param_type=ParamType.INT,
                default=1,
                description=f"Number of output ports out1..outN (1..{MAX_PORTS})",
                min_value=1,
                max_value=MAX_PORTS,
            ),
            ParamDefinition(
                name="input_types",
                param_type=ParamType.STRING,
                default=DataType.TENSOR.value,
                description=(
                    "Comma-separated data type per input port ("
                    + ", ".join(selectable)
                    + "). A short list repeats its last entry."
                ),
            ),
            ParamDefinition(
                name="output_types",
                param_type=ParamType.STRING,
                default=DataType.ANY.value,
                description=(
                    "Comma-separated data type per output port. ANY connects "
                    "to anything; name a real type to have the graph "
                    "validator check the wiring for you."
                ),
            ),
        ]

    # ── Execution ────────────────────────────────────────────────────────

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        *,
        context: ExecutionContext | None = None,
    ) -> dict[str, Any]:
        code = str(params.get("code") or "")
        # Gate first, always: a policy violation must not run module-level
        # statements on its way to being reported.
        validate_script_source(code)
        raw, captured = self._invoke(code, inputs, params, context=context)
        return self._shape_outputs(raw, params, captured)

    def _invoke(
        self,
        code: str,
        inputs: dict[str, Any],
        params: dict[str, Any],
        *,
        context: ExecutionContext | None = None,
    ) -> tuple[Any, _OutputCapture]:
        """Compile and run *code*, returning its raw result and its output.

        Split out from :meth:`execute` so the runtime guards can be tested
        without going through the AST gate first -- they are two independent
        locks and a test that can only reach one of them proves little.
        """
        capture = _OutputCapture()
        script_globals: dict[str, Any] = _tier0_namespace()
        script_globals.update({
            "__name__": "codefyui_script",
            "__file__": SCRIPT_FILENAME,
            "__builtins__": self._script_builtins(capture),
            # The run's compute device, so `torch.zeros(3, device=device)`
            # lands where the rest of the graph is.
            "device": context.device if context is not None else "cpu",
            # The engine's cooperative stop flag. Nothing can interrupt a
            # script from outside -- it runs on the interpreter's default
            # thread pool, so a runaway loop starves every node execution in
            # the process -- but a loop that asks can bail out on Stop.
            "should_stop": (
                context.should_stop if context is not None else (lambda: False)
            ),
        })

        try:
            compiled = compile(code, SCRIPT_FILENAME, "exec")
        except SyntaxError as exc:
            raise RuntimeError(
                f"PythonScript has a syntax error at line {exc.lineno}: {exc.msg}"
            ) from exc

        try:
            exec(compiled, script_globals)  # noqa: S102 - the entire point
        except Exception as exc:
            raise self._failure(exc, capture) from exc

        entry = script_globals.get("run")
        if not callable(entry):
            raise RuntimeError(
                "PythonScript needs a 'def run(inputs, params):' function; "
                "the script defines none."
            )
        # The gate refuses `async def run` while it is typed; this catches the
        # shapes an AST cannot see (a run() rebound to a coroutine function,
        # a graph that never passed through the editor). Without it the port
        # carries an un-awaited coroutine object downstream.
        if inspect.iscoroutinefunction(entry):
            raise RuntimeError(
                "PythonScript's run() is a coroutine function; the node calls "
                "it on a worker thread with no event loop, so nothing would "
                "await it. Define a plain 'def run(inputs, params)'."
            )

        try:
            return entry(dict(inputs), dict(params)), capture
        except Exception as exc:
            raise self._failure(exc, capture) from exc

    @staticmethod
    def _script_builtins(capture: _OutputCapture) -> dict[str, Any]:
        namespace = _script_builtins_base()
        namespace["print"] = capture.print
        namespace["__import__"] = _guarded_import
        return namespace

    @staticmethod
    def _failure(exc: Exception, capture: _OutputCapture) -> RuntimeError:
        """Turn a script exception into the message the node reports.

        The engine surfaces ``str(exception)`` and keeps the real traceback
        only under DEBUG, so the line number has to live in the message. We
        report the DEEPEST frame belonging to the script: for a failure
        inside a library call that is the calling line, and for a failure in
        a helper the user wrote it is the helper -- both times, the line the
        user should look at.
        """
        line: int | None = None
        tb = exc.__traceback__
        while tb is not None:
            if tb.tb_frame.f_code.co_filename == SCRIPT_FILENAME:
                line = tb.tb_lineno
            tb = tb.tb_next

        where = f" at line {line}" if line is not None else ""
        message = f"PythonScript failed{where}: {type(exc).__name__}: {exc}"

        printed = capture.text()
        if printed.strip():
            tail = printed[-_ERROR_LOG_TAIL:]
            message += f"\n--- output before the failure ---\n{tail.rstrip()}"
        return RuntimeError(message)

    @classmethod
    def _shape_outputs(
        cls,
        raw: Any,
        params: dict[str, Any],
        capture: _OutputCapture,
    ) -> dict[str, Any]:
        """Map the script's return value onto the declared output ports."""
        names = [port.name for port in cls.define_outputs_dynamic(params)]
        notes: list[str] = []

        if isinstance(raw, dict):
            result = {key: value for key, value in raw.items() if key in names}
            dropped = [str(key) for key in raw if key not in names]
            if dropped:
                notes.append(
                    "[PythonScript] ignored returned key(s) matching no output "
                    f"port: {', '.join(sorted(dropped))}. Declared ports: "
                    f"{', '.join(names)}."
                )
        else:
            # "Return one thing" is the common case; make it work without
            # ceremony rather than failing on a missing dict wrapper.
            result = {names[0]: raw}

        log = "\n".join(part for part in [capture.text().rstrip("\n"), *notes] if part)
        if log:
            result["__log__"] = log
        return result
