"""The reads of a run's recorded outputs, over any store (#789).

``api.routes_execution_outputs`` answers ``/api/execution/outputs/*`` with
these for a run held in this process's ``RunOutputStore``. A worker run's
captures are on its worker: the control plane forwards the read
(``fetch_output``) and the worker answers it with the SAME functions, over
the run subprocess's own store while the run goes (``agent.child``) and
over the files that store was written to when it ended afterwards
(:class:`DiskRunReader`, read by ``agent.outputs``). So every answer, the
4xx details included, is the same wherever the run executed.

A *store* here is anything with ``RunOutputStore``'s read methods:
``has_run``, ``list_ports``, ``get`` and ``get_with_version``, all async.

Light on purpose: no FastAPI and no engine import, and torch only inside
the functions that look at a value, because the worker agent process
imports this module and must never import torch (only its run subprocess
and its outputs helper do).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from .run_output_store import CaptureAlias


class OutputQueryError(Exception):
    """A read answered with an HTTP error: *status* and its *detail*."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


#: Returned by a read whose port produced ``None``: answered 204, no body.
NO_VALUE = object()

#: The operations a ``fetch_output`` frame names (one per read route).
OPS = ("list", "value", "stats", "steps", "grads")


def not_captured(what: str) -> str:
    """404 detail that names the switch the user probably left off."""
    return (
        f"{what}. Turn on Record outputs (the Rec toggle in the toolbar) and "
        "re-run the graph to capture port data."
    )


# -- what stands in for a value the disk cannot hold ------------------------


@dataclass(frozen=True)
class ModuleStub:
    """An ``nn.Module`` capture, as written to a worker's disk: the facts
    the Inspector shows for one (its class, parameter counts and repr),
    not the module, whose class a reader may not be able to import."""

    class_name: str
    params: int
    trainable: int
    repr: str


@dataclass(frozen=True)
class OpaqueStub:
    """A capture of a type the disk format does not carry (a numpy array, a
    set, a custom object): what the Inspector shows for one, its type name
    and repr."""

    type_name: str
    repr: str


_STUB_KEY = "__codefyui_stub__"
_PRIMITIVES = (str, int, float, bool, type(None))


def portable(value: Any) -> Any:
    """*value* as plain data ``torch.load(weights_only=True)`` reads back.

    Tensors are detached, moved to the CPU and stand alone (a view of a
    larger storage would save all of it); lists, tuples and dicts keep
    their shape; a module and anything else becomes a marked dict that
    :func:`restore` turns into a stub. Imports torch.
    """
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        try:
            whole = tensor.untyped_storage().nbytes()
            if whole > tensor.numel() * tensor.element_size() \
                    or not tensor.is_contiguous():
                tensor = tensor.clone(memory_format=torch.contiguous_format)
        except (RuntimeError, NotImplementedError):  # sparse, quantized
            pass
        return tensor
    if isinstance(value, _PRIMITIVES):
        return value
    if isinstance(value, torch.nn.Module):
        return {_STUB_KEY: "module",
                "class": value.__class__.__name__,
                "params": sum(p.numel() for p in value.parameters()),
                "trainable": sum(p.numel() for p in value.parameters()
                                 if p.requires_grad),
                "repr": repr(value)[:4000]}
    if type(value) in (list, tuple):
        items = [portable(item) for item in value]
        return items if type(value) is list else tuple(items)
    if type(value) is dict and _STUB_KEY not in value \
            and all(isinstance(key, _PRIMITIVES) for key in value):
        return {key: portable(item) for key, item in value.items()}
    return {_STUB_KEY: "opaque", "type": type(value).__name__,
            "repr": repr(value)[:4000]}


def restore(value: Any) -> Any:
    """:func:`portable`'s output with its marked dicts turned into stubs."""
    if isinstance(value, dict):
        kind = value.get(_STUB_KEY)
        if kind == "module":
            return ModuleStub(str(value.get("class")), int(value.get("params", 0)),
                              int(value.get("trainable", 0)),
                              str(value.get("repr", "")))
        if kind == "opaque":
            return OpaqueStub(str(value.get("type")), str(value.get("repr", "")))
        return {key: restore(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        items = [restore(item) for item in value]
        return items if type(value) is list else tuple(items)
    return value


# -- the reads ---------------------------------------------------------------


async def read_capture(
    store: Any, run_id: str, node_id: str, port: str,
) -> tuple[Any, int] | None:
    """The captured ``(value, write serial)`` for a port, or None.

    A preset card, a block instance and a bypassed node never run, so what a
    run keeps under their ports is a :class:`CaptureAlias` naming the port
    that holds the value (#553). Followed here, for every route that reads a
    port, so all of them answer for the port the canvas draws -- the card's
    own outputs, and the inputs of whatever it feeds. One step: an alias
    names a port that ran, and anything else is not a value to serve.
    """
    slot = await store.get_with_version(run_id, node_id, port)
    if slot is not None and isinstance(slot[0], CaptureAlias):
        alias = slot[0]
        slot = await store.get_with_version(run_id, alias.node_id, alias.port)
        if slot is not None and isinstance(slot[0], CaptureAlias):
            return None
    return slot


def parse_slice(slice_str: str) -> tuple[Any, ...] | None:
    """Parse a slice string like '0,:,:,0' into an indexing tuple.

    Each comma-separated piece is either an int or a slice (``start:stop:step``).
    Returns None for empty input. Raises ``ValueError`` on malformed input.
    """
    if not slice_str:
        return None
    pieces: list[Any] = []
    for raw in slice_str.split(","):
        part = raw.strip()
        if part == ":" or part == "":
            pieces.append(slice(None))
            continue
        if ":" in part:
            bits = part.split(":")
            if len(bits) > 3:
                raise ValueError(f"bad slice piece: {part!r}")
            conv = [int(b) if b.strip() else None for b in bits]
            while len(conv) < 3:
                conv.append(None)
            pieces.append(slice(conv[0], conv[1], conv[2]))
            continue
        try:
            pieces.append(int(part))
        except ValueError as e:
            raise ValueError(f"bad slice piece: {part!r}") from e
    return tuple(pieces)


def preview_slice(shape: list[int], max_elements: int) -> str:
    """The leading slice of a tensor of ``shape`` that fits ``max_elements``.

    Walks the dims from the front: a dim whose trailing block alone is still
    too big is pinned to index 0; the first dim whose trailing block fits
    keeps as many leading entries as fit (pinned to 0 when that is one). So
    ``[16,3,224,224]`` previews as ``0,0`` (one 224x224 plane), ``[4,70000]``
    as ``0,0:65536`` and ``[100000]`` as ``0:65536`` at a limit of 65,536.
    Called only for a tensor larger than the limit, so the walk always ends
    on a dim it cuts short.
    """
    pieces: list[str] = []
    for i in range(len(shape)):
        trailing = 1
        for d in shape[i + 1:]:
            trailing *= d
        if trailing > max_elements:
            pieces.append("0")
            continue
        keep = max_elements // trailing
        pieces.append("0" if keep == 1 else f"0:{keep}")
        break
    return ",".join(pieces)


def serialize_tensor(
    value: Any, slice_str: str, max_elements: int, preview: bool = False,
) -> dict[str, Any]:
    import torch

    tensor: torch.Tensor = value
    full_shape = list(tensor.shape)
    dtype = str(tensor.dtype)

    # A bounded preview (#640): the server knows the shape, so it picks the
    # slice instead of the client guessing one rank-blind. A tensor within the
    # limit is sent whole, exactly as without ``preview``.
    truncated = False
    if preview:
        if slice_str:
            raise OutputQueryError(400, "'preview' and 'slice' cannot be combined")
        if tensor.numel() > max_elements:
            slice_str = preview_slice(full_shape, max_elements)
            truncated = True

    try:
        slicer = parse_slice(slice_str)
    except ValueError as e:
        raise OutputQueryError(400, f"invalid slice: {e}")

    if slicer is not None:
        try:
            sliced = tensor[slicer]
        except (IndexError, TypeError) as e:
            raise OutputQueryError(400, f"slice failed: {e}")
    else:
        sliced = tensor
    # A value a node trains still tracks gradients, and turning one into a
    # float below makes torch warn (once per process). Detaching shares the
    # storage.
    sliced = sliced.detach()

    if sliced.numel() > max_elements:
        raise OutputQueryError(
            413,
            f"output has {sliced.numel()} elements (max {max_elements}); "
            "supply a narrower 'slice' parameter",
        )

    summary: dict[str, Any] = {
        "type": "tensor",
        "full_shape": full_shape,
        "dtype": dtype,
        "slice": slice_str or "",
        "sliced_shape": list(sliced.shape),
        "values": sliced.cpu().tolist(),
        "truncated": truncated,
    }
    if sliced.numel() > 0 and sliced.is_floating_point():
        summary["min"] = round(float(sliced.min()), 6)
        summary["max"] = round(float(sliced.max()), 6)
        summary["mean"] = round(float(sliced.mean()), 6)
    elif sliced.numel() > 0 and sliced.dtype != torch.bool:
        summary["min"] = int(sliced.min())
        summary["max"] = int(sliced.max())
    return summary


def serialize_value(
    value: Any, slice_str: str, max_elements: int, preview: bool = False,
) -> dict[str, Any]:
    if isinstance(value, ModuleStub):
        return {"type": "model", "class": value.class_name,
                "params": value.params, "trainable": value.trainable,
                "repr": value.repr}
    if isinstance(value, OpaqueStub):
        return {"type": value.type_name, "repr": value.repr}
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return serialize_tensor(value, slice_str, max_elements, preview)
        if isinstance(value, torch.nn.Module):
            total = sum(p.numel() for p in value.parameters())
            trainable = sum(p.numel() for p in value.parameters() if p.requires_grad)
            return {
                "type": "model",
                "class": value.__class__.__name__,
                "params": total,
                "trainable": trainable,
                "repr": repr(value)[:4000],
            }
    except ImportError:
        pass
    if isinstance(value, bool):
        return {"type": "scalar", "value": value}
    if isinstance(value, (int, float)):
        return {"type": "scalar", "value": value}
    if isinstance(value, str):
        return {"type": "string", "value": value[:4000]}
    if isinstance(value, (list, tuple)):
        out: dict[str, Any] = {
            "type": "list",
            "length": len(value),
            "repr": repr(value)[:4000],
        }
        # Include the actual values when the list is small and JSON-friendly so
        # the Inspector can render token chips, token IDs, and offset pairs
        # without a separate /list endpoint. Anything else stays repr-only.
        if len(value) <= 1024:
            primitive_types = (str, int, float, bool, type(None))
            if all(isinstance(x, primitive_types) for x in value):
                out["values"] = list(value)
            elif all(
                isinstance(x, (list, tuple))
                and len(x) == 2
                and all(isinstance(y, (int, float)) for y in x)
                for x in value
            ):
                out["values"] = [list(x) for x in value]
        return out
    return {"type": type(value).__name__, "repr": repr(value)[:4000]}


def shape_of(value: Any) -> list[int] | None:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return list(value.shape)
    except ImportError:
        pass
    return None


def type_label(value: Any) -> str:
    if isinstance(value, ModuleStub):
        return "model"
    if isinstance(value, OpaqueStub):
        return value.type_name
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return "tensor"
        if isinstance(value, torch.nn.Module):
            return "model"
    except ImportError:
        pass
    if isinstance(value, bool):
        return "scalar"
    if isinstance(value, (int, float)):
        return "scalar"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (list, tuple)):
        return "list"
    return type(value).__name__


def producer_of(nid: str, node_id: str, inner: bool) -> bool:
    """Whether ``nid``'s captures answer a read of ``node_id`` (#559).

    Its own, always. With ``inner``, also every node a block or preset card
    ``node_id`` holds: a run names those ``<card>/<node>`` (a block, nested
    ones deeper) and ``<card>__<node>`` (a preset card). A card never runs,
    so what it did is what those nodes did.
    """
    if nid == node_id:
        return True
    return inner and (
        nid.startswith(node_id + "/") or nid.startswith(node_id + "__"))


async def list_outputs(store: Any, run_id: str) -> list[dict[str, Any]]:
    ports = await store.list_ports(run_id)
    if ports is None:
        raise OutputQueryError(404, f"run '{run_id}' not found")
    result = []
    for node_id, port in ports:
        slot = await read_capture(store, run_id, node_id, port)
        value = None if slot is None else slot[0]
        result.append(
            {
                "node_id": node_id,
                "port": port,
                "type": type_label(value),
                "full_shape": shape_of(value),
            }
        )
    return result


async def steps_index(store: Any, run_id: str, node_id: str,
                      inner: bool = False) -> list[dict[str, Any]]:
    """The ``__steps_index`` read: see the route for the contract."""
    if not await store.has_run(run_id):
        raise OutputQueryError(404, f"run '{run_id}' not found")
    ports = await store.list_ports(run_id)
    if ports is None:
        raise OutputQueryError(404, f"run '{run_id}' not found")
    metas: dict[str, dict[int, dict[str, Any]]] = {}
    for nid, port in ports:
        if not producer_of(nid, node_id, inner) or not port.startswith("__step__"):
            continue
        if not port.endswith("__meta"):
            continue
        # port format: __step__{i}__meta
        try:
            idx = int(port[len("__step__"):-len("__meta")])
        except ValueError:
            continue
        meta = await store.get(run_id, nid, port)
        if isinstance(meta, dict):
            metas.setdefault(nid, {})[idx] = meta
    return [
        {"index": idx, **by_index[idx], **({"node_id": nid} if inner else {})}
        for nid, by_index in metas.items()
        for idx in sorted(by_index)
    ]


async def grad_index(store: Any, run_id: str, node_id: str,
                     inner: bool = False) -> list[dict[str, Any]]:
    """The ``__grad_index`` read: see the route for the contract."""
    if not await store.has_run(run_id):
        raise OutputQueryError(404, f"run '{run_id}' not found")
    ports = await store.list_ports(run_id)
    if ports is None:
        raise OutputQueryError(404, f"run '{run_id}' not found")

    entries: list[dict[str, Any]] = []
    for nid, port in ports:
        if not producer_of(nid, node_id, inner):
            continue
        if port.endswith("__grad__meta") or port.startswith("__weight_grad__") and port.endswith("__meta"):
            continue
        producer = {"node_id": nid} if inner else {}
        if port.endswith("__grad"):
            forward_port = port[:-len("__grad")]
            health = await store.get(run_id, nid, port + "__meta")
            entries.append({
                "port": forward_port,
                "kind": "port",
                "has_grad": True,
                "health": health if isinstance(health, dict) else None,
                **producer,
            })
        elif port.startswith("__weight_grad__"):
            param_name = port[len("__weight_grad__"):]
            health = await store.get(run_id, nid, port + "__meta")
            entries.append({
                "port": param_name,
                "kind": "weight",
                "has_grad": True,
                "health": health if isinstance(health, dict) else None,
                **producer,
            })
    return entries


async def output_value(store: Any, run_id: str, node_id: str, port: str, *,
                       slice_str: str = "", max_elements: int = 4096,
                       preview: bool = False) -> Any:
    """The value read, JSON-safe; :data:`NO_VALUE` for a port that produced
    ``None``."""
    from .run_store import json_safe

    if not await store.has_run(run_id):
        raise OutputQueryError(404, f"run '{run_id}' not found")
    slot = await read_capture(store, run_id, node_id, port)
    if slot is None:
        raise OutputQueryError(
            404, f"output '{node_id}.{port}' not found in run '{run_id}'")
    value = slot[0]
    if value is None:
        # The node ran and the port produced None -- TrainingLoop's
        # grad_scaler_state outside fp16, for one (#602). An empty 204 rather
        # than the 404 above, which stays for a port with nothing recorded:
        # the browser logs every 404 as a failed resource, and the Inspector
        # reads one as an expired run.
        return NO_VALUE
    payload = serialize_value(value, slice_str, max_elements, preview)
    payload["run_id"] = run_id
    payload["node_id"] = node_id
    payload["port"] = port
    # A diverged tensor's raw values/min/max/mean can be NaN/Inf. Starlette
    # renders with allow_nan=False, so one leaked value 500s the whole
    # response -- same hazard as #129, same fix: the run_store.json_safe
    # convention established there. The stats read follows the same
    # convention but not here -- its sanitisation lives inside
    # core.port_stats.compute_port_stats (the _num() helper in that module).
    return json_safe(payload)


#: ``compute(value) -> stats dict``, awaited (the routes run it on a pool).
StatsCompute = Callable[[Any], Awaitable[dict[str, Any]]]


def stub_stats(value: Any) -> dict[str, Any] | None:
    """The stats of a stub (always ``unsupported``, as the value's would
    be), or None for anything else."""
    if isinstance(value, ModuleStub):
        return {"kind": "unsupported", "type": value.class_name}
    if isinstance(value, OpaqueStub):
        return {"kind": "unsupported", "type": value.type_name}
    return None


async def output_stats(store: Any, run_id: str, node_id: str, port: str, *,
                       compute: StatsCompute, cache: Any = None) -> Any:
    """The stats read; :data:`NO_VALUE` for a port that produced ``None``.

    *cache* (a ``PortStatsCache``) is keyed on the store's write serial:
    a node inside a loop overwrites one port several times in a run, and
    the replacement can land on the freed storage of the value before it,
    so nothing derived from the object itself may key it.
    """
    if not await store.has_run(run_id):
        raise OutputQueryError(
            404, not_captured(f"run '{run_id}' has no captured outputs"))
    slot = await read_capture(store, run_id, node_id, port)
    if slot is None:
        raise OutputQueryError(404, not_captured(
            f"nothing captured for '{node_id}.{port}' in run '{run_id}'"))
    value, version = slot
    if value is None:
        # The node ran and the port produced None: nothing to summarise, and
        # nothing missing either -- the same empty 204 as the value read.
        return NO_VALUE
    key = (run_id, node_id, port, version)
    payload = None if cache is None else cache.get(key)
    if payload is None:
        stats = stub_stats(value)
        if stats is None:
            stats = await compute(value)
        payload = {"run_id": run_id, "node_id": node_id, "port": port, **stats}
        if cache is not None:
            cache.put(key, payload)
    return payload


async def dispatch(store: Any, op: str, run_id: str, params: dict[str, Any],
                   *, compute: StatsCompute,
                   cache: Any = None) -> tuple[int, Any]:
    """One ``fetch_output`` read as ``(HTTP status, JSON body)``: 200 with
    the payload, 204 with None, or the error status with ``{"detail"}``.
    Never raises but cancel."""
    try:
        node_id = str(params.get("node_id") or "")
        port = str(params.get("port") or "")
        inner = bool(params.get("inner"))
        if op == "list":
            body: Any = await list_outputs(store, run_id)
        elif op == "steps":
            body = await steps_index(store, run_id, node_id, inner)
        elif op == "grads":
            body = await grad_index(store, run_id, node_id, inner)
        elif op == "value":
            body = await output_value(
                store, run_id, node_id, port,
                slice_str=str(params.get("slice") or ""),
                max_elements=int(params.get("max_elements") or 4096),
                preview=bool(params.get("preview")))
        elif op == "stats":
            body = await output_stats(store, run_id, node_id, port,
                                      compute=compute, cache=cache)
        else:
            return 400, {"detail": f"unknown outputs read {op!r}"}
    except OutputQueryError as exc:
        return exc.status, {"detail": exc.detail}
    except Exception as exc:  # noqa: BLE001 -- answered, not raised
        return 500, {"detail": f"reading the recorded outputs failed: "
                               f"{type(exc).__name__}: {exc}"}
    if body is NO_VALUE:
        return 204, None
    return 200, body


def encode_body(body: Any) -> str:
    """A read's JSON body as text, NaN-free (``json_safe``) and compact."""
    from .run_store import json_safe

    return json.dumps(json_safe(body), allow_nan=False, separators=(",", ":"),
                      default=str)


# -- a run's captures on disk (a worker, #789) -------------------------------

#: Under a run's directory on the worker: ``<runs_dir>/<run_id>/outputs/``.
OUTPUTS_DIRNAME = "outputs"
#: The file that lists a run's captures; written last, so a directory
#: without it holds no complete dump and is read as holding nothing.
INDEX_FILE = "index.json"
INDEX_FORMAT = 1


async def dump_run(store: Any, run_id: str, directory: Path) -> dict[str, Any]:
    """Write *run_id*'s captures in *store* to *directory*. Imports torch.

    One ``torch.save`` file per slot (:func:`portable` values), then
    :data:`INDEX_FILE` naming them, in the store's order, with each slot's
    write serial and bytes (as the store measured them, so the worker's
    budget counts what the in-process store counts). Blocking work goes to
    a thread. Returns the index. A run with no captures still gets an
    index (an empty one): it ran and stored nothing.
    """
    import asyncio

    slots = await store.snapshot(run_id)
    return await asyncio.to_thread(_write_dump, run_id, slots, directory)


def _write_dump(run_id: str, slots: list[tuple[str, str, Any, int, int]],
                directory: Path) -> dict[str, Any]:
    import os

    import torch

    directory.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    total = 0
    for number, (node_id, port, value, version, nbytes) in enumerate(slots):
        entry: dict[str, Any] = {"node_id": node_id, "port": port,
                                 "version": int(version), "bytes": int(nbytes)}
        if isinstance(value, CaptureAlias):
            entry["alias"] = [value.node_id, value.port]
        else:
            name = f"{number:06d}.pt"
            torch.save(portable(value), directory / name)
            entry["file"] = name
        total += int(nbytes)
        entries.append(entry)
    index = {"format": INDEX_FORMAT, "run_id": run_id, "bytes": total,
             "slots": entries}
    temporary = directory / (INDEX_FILE + ".tmp")
    temporary.write_text(json.dumps(index), encoding="utf-8")
    os.replace(temporary, directory / INDEX_FILE)
    return index


def read_index(directory: Path) -> dict[str, Any] | None:
    """A dump's index, or None when there is no complete one."""
    try:
        index = json.loads((directory / INDEX_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(index, dict) or not isinstance(index.get("slots"), list):
        return None
    return index


class DiskRunReader:
    """A store over one run's dump (:func:`dump_run`), for :func:`dispatch`.

    Values are loaded per read, memory-mapped (``torch.load(mmap=True,
    weights_only=True)``), so a slice of a large tensor reads only its
    pages. Imports torch on the first value read.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._index = read_index(directory)
        self._slots: dict[tuple[str, str], dict[str, Any]] = {}
        if self._index is not None:
            for entry in self._index["slots"]:
                self._slots[(str(entry.get("node_id")),
                             str(entry.get("port")))] = entry

    def _holds(self, run_id: str) -> bool:
        return self._index is not None and self._index.get("run_id") == run_id

    async def has_run(self, run_id: str) -> bool:
        return self._holds(run_id)

    async def list_ports(self, run_id: str) -> list[tuple[str, str]] | None:
        if not self._holds(run_id):
            return None
        return list(self._slots)

    async def get_with_version(self, run_id: str, node_id: str,
                               port: str) -> tuple[Any, int] | None:
        if not self._holds(run_id):
            return None
        entry = self._slots.get((node_id, port))
        if entry is None:
            return None
        version = int(entry.get("version") or 0)
        alias = entry.get("alias")
        if alias is not None:
            return CaptureAlias(str(alias[0]), str(alias[1])), version
        import asyncio

        value = await asyncio.to_thread(self._load, str(entry.get("file")))
        return value, version

    async def get(self, run_id: str, node_id: str, port: str) -> Any | None:
        slot = await self.get_with_version(run_id, node_id, port)
        return None if slot is None else slot[0]

    def _load(self, name: str) -> Any:
        import torch

        path = self.directory / Path(name).name
        return restore(torch.load(path, map_location="cpu",
                                  weights_only=True, mmap=True))
