"""Optional node provenance: which data and which weights a run read (#344, hook 4).

The environment fingerprint (``env_fingerprint``) says which code and
libraries executed a run; it says nothing about the dataset or the model
the graph loaded. This module records that, per node, for the node types
that read a dataset or load a model, when the run asks for it with the
``provenance`` run option:

- ``"identity"``: cheap facts only. Where the files are, how many, how
  large, when last modified, and their **stat identity**
  (``content_hash.stat_identity``: path, size and mtime hashed together,
  ``content.method: "stat"``). No content is read.
- ``"content"``: the same plus a real **content hash**
  (``content_hash.content_identity``, sha256 of every byte,
  ``content.method: "sha256"``). Incremental: a file whose size, mtime and
  inode are unchanged since an earlier hash reuses that hash, so a large
  dataset is read in full once, not on every run.

Absent (the default) records nothing and changes nothing: the run's
options row, its events and its outputs are those of a run without it.

How it is collected. Nothing here touches a node's ``execute``: the
executor's progress bridge hears every node finish (``completed`` or
``cached``) with its outputs, and for a node type that has a provider in
:data:`PROVIDERS` it calls :func:`describe` on the node's params (defaults
filled in) and outputs, in a thread, after the node's own status event.
Path resolution reuses the node class's own resolver, so a provider looks
where the node looked. The record goes to ``RunSinks.record_provenance``:
the run service stores it (``exec_run_provenance``); a worker's run
subprocess sends it over the pipe and the control plane stores it.

Record shape (``schema_version`` 1). Every key is present in every record;
a fact that is not known is ``None``, and ``notes`` says why when the
reason is not obvious::

    {"schema_version": 1, "node_id", "node_type", "kind": "dataset"|"model",
     "mode": "identity"|"content",
     "source": {"type", "name", "location", "path", "subset"},
     "split",
     "revision": {"requested", "resolved", "kind"},
     "content": {"method": "stat"|"sha256", "digest", "exists", "files",
                 "bytes", "latest_mtime", "truncated", "unreadable",
                 ["reused", "hashed"]},
     "rows", "notes": [...]}

``source.path`` is never absolute: it is relative to the CodefyUI
directory it was found under, named by ``source.location`` (``models``,
``files``, ``data``, ``project``, ``working-dir``, ``hf-cache``,
``packs``); a path outside all of them is only its file name, with
``location: "external"``. A record is meant to be shared like a
fingerprint.

``content.method`` is the honesty rule: ``"stat"`` is an identity, which
an in-place edit within one timestamp tick or a copy that resets mtimes
can fool; ``"sha256"`` is the content. A reader must not treat a stat
digest as a content hash.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any, Awaitable, Callable

from . import content_hash

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

#: The run option's values. Absent / None: off.
MODES = ("identity", "content")

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_KAGGLE_VERSION = re.compile(r"[\\/]versions[\\/](\d+)(?:[\\/]|$)")

Provider = Callable[[dict[str, Any], dict[str, Any] | None, str],
                    dict[str, Any]]


# ── the record ────────────────────────────────────────────────────────────


def blank_record(node_id: str, node_type: str, kind: str,
                 mode: str) -> dict[str, Any]:
    """A record with every key present and nothing known."""
    return {
        "schema_version": SCHEMA_VERSION,
        "node_id": node_id,
        "node_type": node_type,
        "kind": kind,
        "mode": mode,
        "source": {"type": None, "name": None, "location": None,
                   "path": None, "subset": None},
        "split": None,
        "revision": {"requested": None, "resolved": None, "kind": None},
        "content": None,
        "rows": None,
        "notes": [],
    }


def _content(paths: list[Path], mode: str) -> dict[str, Any]:
    return content_hash.identity(paths, mode)


def locate(path: Path | str) -> tuple[str, str]:
    """``(location, path relative to it)`` for *path*, never absolute.

    The first CodefyUI directory containing it wins, most specific first;
    outside every one: ``("external", <file name>)``.
    """
    from ..config import settings

    target = Path(path).expanduser()
    try:
        target = target.resolve()
    except OSError:
        target = target.absolute()
    roots: list[tuple[str, Path | None]] = [
        ("models", settings.MODELS_DIR),
        ("files", settings.DATA_FILES_DIR),
        ("packs", _packs_cache()),
        ("hf-cache", _hf_hub_cache()),
        ("project", settings.PROJECT_DIR),
        ("data", settings.MODELS_DIR.parent),
        ("working-dir", Path.cwd()),
    ]
    for label, root in roots:
        if root is None:
            continue
        try:
            resolved_root = Path(root).resolve()
        except OSError:
            continue
        if target == resolved_root or target.is_relative_to(resolved_root):
            relative = target.relative_to(resolved_root).as_posix()
            return label, relative or "."
    return "external", target.name


def _packs_cache() -> Path | None:
    try:
        from .packs.paths import cache_dir

        return cache_dir()
    except Exception:  # noqa: BLE001
        return None


def _hf_hub_cache() -> Path | None:
    import os

    explicit = os.environ.get("HF_HUB_CACHE")
    if explicit:
        return Path(explicit)
    home = os.environ.get("HF_HOME")
    base = Path(home) if home else Path.home() / ".cache" / "huggingface"
    return base / "hub"


def _set_path(record: dict[str, Any], path: Path | str | None) -> None:
    if path is None:
        return
    location, relative = locate(path)
    record["source"]["location"] = location
    record["source"]["path"] = relative


def _rows(value: Any) -> int | None:
    """``len(value)`` when it has one that is a non-negative int, else None.

    ``len`` of a map-style dataset is its size and reads no data. An
    iterable dataset has none, which is an honest unknown.
    """
    try:
        rows = len(value)
    except Exception:  # noqa: BLE001 -- no __len__, or one that raises
        return None
    return rows if isinstance(rows, int) and rows >= 0 else None


def _output(outputs: dict[str, Any] | None, port: str) -> Any:
    return None if not isinstance(outputs, dict) else outputs.get(port)


# ── providers ─────────────────────────────────────────────────────────────


def _file_record(record: dict[str, Any], path: Path | None,
                 mode: str) -> None:
    record["source"]["type"] = "file"
    if path is None:
        record["notes"].append("no file selected")
        return
    _set_path(record, path)
    record["content"] = _content([path], mode)


def csv_reader(params: dict[str, Any], outputs: dict[str, Any] | None,
               mode: str) -> dict[str, Any]:
    from ..nodes.data.csv_reader_node import CSVReaderNode

    record = blank_record("", "CSVReader", "dataset", mode)
    raw = str(params.get("path") or "").strip()
    _file_record(record, CSVReaderNode._resolve_path(raw) if raw else None,
                 mode)
    tensor = _output(outputs, "tensor")
    shape = getattr(tensor, "shape", None)
    if shape is not None and len(shape) >= 1:
        record["rows"] = int(shape[0])
    record["notes"].append("rows counts the rows the node kept")
    return record


def torchvision_dataset(params: dict[str, Any],
                        outputs: dict[str, Any] | None,
                        mode: str) -> dict[str, Any]:
    from ..nodes.data.dataset_node import DatasetNode

    record = blank_record("", "Dataset", "dataset", mode)
    name = str(params.get("name") or "MNIST")
    split = str(params.get("split") or "train")
    root = DatasetNode._resolve_data_dir(
        str(params.get("data_dir") or "./data"))
    owned = DatasetNode._owned_paths(name, split, root) or [Path(root)]
    record["source"].update(type="torchvision", name=name)
    record["split"] = split
    _set_path(record, owned[0] if len(owned) == 1 else root)
    record["content"] = _content([Path(p) for p in owned], mode)
    record["rows"] = _rows(_output(outputs, "dataset"))
    record["notes"].append(
        "torchvision datasets carry no revision; the content identity "
        "covers the files the loader reads")
    return record


def image_folder_dataset(params: dict[str, Any],
                         outputs: dict[str, Any] | None,
                         mode: str) -> dict[str, Any]:
    from ..nodes.data.image_folder_dataset_node import ImageFolderDatasetNode

    record = blank_record("", "ImageFolderDataset", "dataset", mode)
    record["source"]["type"] = "directory"
    record["split"] = str(params.get("split") or "train")
    root = ImageFolderDatasetNode._resolve_root(params)
    if root is None:
        record["notes"].append("no folder selected")
    else:
        _set_path(record, root)
        record["content"] = _content([root], mode)
    record["rows"] = _rows(_output(outputs, "dataset"))
    return record


def _hf_dataset_cache_hash(dataset: Any) -> str | None:
    """The revision hash in an HF ``datasets`` Arrow cache path, or None.

    ``datasets`` stores a Hub dataset under
    ``<name>/<config>/<version>/<hash>/``, where ``<hash>`` is the commit
    of the dataset repository the files came from.
    """
    for entry in getattr(dataset, "cache_files", None) or []:
        filename = entry.get("filename") if isinstance(entry, dict) else None
        if not filename:
            continue
        for part in reversed(Path(filename).parent.parts):
            if _HEX40.match(part):
                return part
    return None


def _hub_ref(repo_type: str, repo_id: str, ref: str = "main") -> str | None:
    """The commit the local Hugging Face hub cache resolved *ref* to."""
    cache = _hf_hub_cache()
    if cache is None:
        return None
    folder = f"{repo_type}s--" + repo_id.replace("/", "--")
    try:
        value = (cache / folder / "refs" / ref).read_text(
            encoding="utf-8").strip()
    except OSError:
        return None
    return value if _HEX40.match(value) else None


def huggingface_dataset(params: dict[str, Any],
                        outputs: dict[str, Any] | None,
                        mode: str) -> dict[str, Any]:
    record = blank_record("", "HuggingFaceDataset", "dataset", mode)
    name = str(params.get("dataset_name") or "ylecun/mnist")
    record["source"].update(type="huggingface", name=name,
                            subset=str(params.get("subset") or "") or None)
    record["split"] = str(params.get("split") or "train")
    record["notes"].append(
        "the node pins no revision, so the repository's default branch "
        "was requested")
    wrapped = _output(outputs, "dataset")
    hf = getattr(wrapped, "_ds", None)
    resolved = _hf_dataset_cache_hash(hf) or _hub_ref("dataset", name)
    if resolved is not None:
        record["revision"].update(resolved=resolved, kind="hf-commit")
    else:
        record["notes"].append("the resolved revision is not recorded in "
                               "the local cache")
    files = sorted({Path(entry["filename"])
                    for entry in getattr(hf, "cache_files", None) or []
                    if isinstance(entry, dict) and entry.get("filename")})
    if files:
        _set_path(record, files[0].parent)
        record["content"] = _content(files, mode)
    else:
        record["notes"].append("the dataset has no local cache files")
    record["rows"] = _rows(wrapped)
    return record


def kaggle_dataset(params: dict[str, Any], outputs: dict[str, Any] | None,
                   mode: str) -> dict[str, Any]:
    record = blank_record("", "KaggleDataset", "dataset", mode)
    slug = str(params.get("dataset_slug") or "")
    record["source"].update(type="kaggle", name=slug or None)
    requested = _KAGGLE_VERSION.search("/" + slug + "/")
    record["revision"]["requested"] = requested.group(1) if requested else None
    dataset = _output(outputs, "dataset")
    root = getattr(dataset, "root", None)
    if root:
        found = _KAGGLE_VERSION.search(str(root))
        if found:
            record["revision"].update(resolved=found.group(1),
                                      kind="kaggle-version")
        _set_path(record, root)
        record["content"] = _content([Path(root)], mode)
    record["rows"] = _rows(dataset)
    return record


def model_loader(params: dict[str, Any], outputs: dict[str, Any] | None,
                 mode: str) -> dict[str, Any]:
    from ..nodes.io.model_loader_node import ModelLoaderNode

    record = blank_record("", "ModelLoader", "model", mode)
    raw = str(params.get("path") or "").strip()
    _file_record(record, ModelLoaderNode._resolve_path(raw) if raw else None,
                 mode)
    record["notes"].append("a local checkpoint has no revision; the "
                           "content identity is its identity")
    return record


def checkpoint_loader(params: dict[str, Any],
                      outputs: dict[str, Any] | None,
                      mode: str) -> dict[str, Any]:
    from .checkpoints import resolve_checkpoint_path

    record = blank_record("", "CheckpointLoader", "model", mode)
    raw = str(params.get("path") or "").strip()
    _file_record(record, resolve_checkpoint_path(raw) if raw else None, mode)
    record["notes"].append("a local checkpoint has no revision; the "
                           "content identity is its identity")
    return record


def hf_text_generate(params: dict[str, Any],
                     outputs: dict[str, Any] | None,
                     mode: str) -> dict[str, Any]:
    from ..nodes.llm._hf_generators import DEFAULT_GENERATOR
    from ..nodes.llm._packs_bridge import model_dir

    record = blank_record("", "HFTextGenerate", "model", mode)
    repo = str(params.get("model") or DEFAULT_GENERATOR)
    record["source"].update(type="huggingface", name=repo)
    record["revision"]["requested"] = _catalog_revision(repo)
    snapshot = model_dir(repo)
    if snapshot is None:
        record["notes"].append("the model snapshot is not downloaded")
        return record
    if _HEX40.match(snapshot.name):
        record["revision"].update(resolved=snapshot.name, kind="hf-commit")
    _set_path(record, snapshot)
    record["content"] = _content([snapshot], mode)
    return record


def _catalog_revision(repo_id: str) -> str | None:
    try:
        from .packs.catalog import iter_packs

        for pack in iter_packs():
            for item in pack.items:
                if item.kind == "hf" and item.repo_id == repo_id:
                    return item.revision
    except Exception:  # noqa: BLE001
        return None
    return None


#: Node type -> provider. A provider returns a record (``node_id`` and
#: ``node_type`` are filled in by :func:`describe`); it may raise, which
#: :func:`describe` turns into a record that says so.
PROVIDERS: dict[str, Provider] = {
    "CSVReader": csv_reader,
    "Dataset": torchvision_dataset,
    "ImageFolderDataset": image_folder_dataset,
    "HuggingFaceDataset": huggingface_dataset,
    "KaggleDataset": kaggle_dataset,
    "ModelLoader": model_loader,
    "CheckpointLoader": checkpoint_loader,
    "HFTextGenerate": hf_text_generate,
}

#: What each provider's records are, for the record of a provider that
#: failed before it could say.
_KINDS = {"ModelLoader": "model", "CheckpointLoader": "model",
          "HFTextGenerate": "model"}


def register_provider(node_type: str, provider: Provider) -> None:
    """Add or replace the provider for *node_type* (a plugin's own node)."""
    PROVIDERS[node_type] = provider


def _with_defaults(node_type: str, params: dict[str, Any]) -> dict[str, Any]:
    from .node_registry import registry

    merged: dict[str, Any] = {}
    cls = registry.get(node_type)
    if cls is not None:
        try:
            for definition in cls.define_params():
                merged[definition.name] = definition.default
        except Exception:  # noqa: BLE001
            pass
    merged.update(params or {})
    return merged


def describe(node_id: str, node_type: str, params: dict[str, Any],
             outputs: dict[str, Any] | None, mode: str) -> dict[str, Any] | None:
    """The record for one finished node, or None (no provider). Never raises."""
    provider = PROVIDERS.get(node_type)
    if provider is None:
        return None
    try:
        record = provider(_with_defaults(node_type, params), outputs, mode)
    except Exception as exc:  # noqa: BLE001 -- a fact, never a failure
        logger.info("provenance of node %s (%s) not collected: %s",
                    node_id, node_type, exc)
        record = blank_record(node_id, node_type,
                              _KINDS.get(node_type, "dataset"), mode)
        record["notes"].append(f"not collected: {type(exc).__name__}: {exc}")
    record["node_id"] = node_id
    record["node_type"] = node_type
    record["mode"] = mode
    return record


# ── the recorder the executor uses ────────────────────────────────────────


class ProvenanceRecorder:
    """Hears node statuses, describes each provided node once, and hands
    the record to *sink*. One per run."""

    #: The statuses after which a node's outputs are what it read.
    FINISHED = ("completed", "cached")

    def __init__(self, nodes: list[dict[str, Any]], mode: str,
                 sink: Callable[[dict[str, Any]], Awaitable[Any]]) -> None:
        self.mode = mode
        self.sink = sink
        self.nodes: dict[str, tuple[str, dict[str, Any]]] = {}
        self.seen: set[str] = set()
        self.add(nodes)

    def add(self, nodes: list[dict[str, Any]]) -> None:
        """Index *nodes* (the graph's, then the expanded ones) by id."""
        for node in nodes or []:
            if not isinstance(node, dict):
                continue
            node_type = node.get("type")
            if node_type not in PROVIDERS:
                continue
            data = node.get("data") if isinstance(node.get("data"), dict) else {}
            params = data.get("params") if isinstance(data.get("params"), dict) else {}
            self.nodes[str(node.get("id"))] = (str(node_type), dict(params))

    async def observe(self, node_id: str, status: str,
                      result: dict[str, Any] | None) -> None:
        """Record *node_id* the first time it finishes. Never raises."""
        if status not in self.FINISHED or node_id in self.seen:
            return
        known = self.nodes.get(node_id)
        if known is None:
            return
        self.seen.add(node_id)
        node_type, params = known
        try:
            record = await asyncio.to_thread(describe, node_id, node_type,
                                             params, result, self.mode)
            if record is not None:
                await self.sink(record)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 -- provenance never fails a run
            logger.warning("could not record the provenance of node %s",
                           node_id, exc_info=True)
