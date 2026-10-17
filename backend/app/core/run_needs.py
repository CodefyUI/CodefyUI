"""What storage a run needs, read off its graph at submit (#782).

``needs(graph)`` answers ``{"mount": <label> | None, "files": [...],
"inputs": [...], "paths": [...]}``: whether the run reads or writes files
under the data directory (so it may only run where that directory is
mounted, see ``scheduling.WorkerScheduler``), and which files it reads.
``files`` is every one of them as the graph spells it. The ``transfer``
backend (#785) splits them: ``inputs`` are the ones a typed param names
(``{"kind": "data_file" | "image" | "model", "name"}``), stored uploads it
can ship by content hash; ``paths`` are the ones a node rule found in a
plain string param (a ``FileReader`` path, an image folder), which it
cannot, so a run with any of them needs the mount itself.

Only a run that targets a worker is asked: a local run is on the data
directory by definition, so the 3.0 submit path never calls this.

The rules, and where they come from
-----------------------------------
Every node file under ``app/nodes`` that touches ``DATA_FILES_DIR``,
``IMAGES_DIR``, ``MODELS_DIR`` or ``MEDIA_DIR`` (or the data root above
them) is listed in :data:`COVERAGE` with how it is decided;
``tests/test_run_needs.py`` re-derives that list from the source and fails
when a node is missing from it. Three kinds of rule:

1. **Typed params** (any node, plugins included): a ``DATA_FILE``,
   ``IMAGE_FILE`` or ``MODEL_FILE`` param the editor shows, with a value,
   references a stored upload. A bundled sample (``data/samples/...``,
   resolved against the install, which every host has) does not.
2. **Writers**: ``CheckpointSaver``, ``ModelSaver``, ``ImageWriter``,
   ``VideoWrite`` always write under the data directory.
3. **Node rules** (:data:`NODE_RULES`): a ``TrainingLoop`` with
   ``checkpoint_every`` / ``checkpoint_every_steps`` / ``tensorboard``; a
   reader whose plain string path names a file or folder under the data
   directory (``FileReader``, ``ImageFolderDataset``, ``ImageBatchReader``,
   ``VideoLoad``, ``DocumentLoader`` reading a folder other than the
   bundled sample, ``GraphInput`` of type image).

``record_outputs`` is not read here: a run on a worker cannot ask for it
(``RunService`` refuses it at submit until #789).

Deliberately NOT a need: nodes that only download into a cache of their
own (``Dataset``, ``HuggingFaceDataset``, ``KaggleDataset``,
``TextCorpusDataset`` from the Hub, ``LMTokenizedDataset``'s token cache),
which a worker fills on its own disk, and the interrupt checkpoint a
training loop writes only when it is stopped.

The graph is read as it will execute: subgraphs and presets expanded,
bypassed and unselected nodes dropped, every missing param at its default
(``graph_engine.prepare_executable_graph``). A graph that does not prepare
(it will fail at run time anyway) is read node by node, every nested node
included, which can only find more needs, never fewer.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator

from .node_base import ParamType, is_param_visible

#: Param types whose value names a stored upload.
FILE_PARAM_TYPES = (ParamType.DATA_FILE, ParamType.IMAGE_FILE,
                    ParamType.MODEL_FILE)

#: Typed param -> the ``transfer`` input kind (#785) its upload store is.
INPUT_KINDS = {ParamType.DATA_FILE: "data_file",
               ParamType.IMAGE_FILE: "image",
               ParamType.MODEL_FILE: "model"}

#: Paths that resolve against the install (every host has them), not the
#: data directory: the bundled samples.
BUNDLED_PREFIXES = ("data/samples/",)

#: Nodes that always write under the data directory.
WRITERS = frozenset({"CheckpointSaver", "ModelSaver", "ImageWriter",
                     "VideoWrite"})


def _text(value: Any) -> str:
    return str(value or "").strip()


def _bundled(path: str) -> bool:
    spelled = path.replace("\\", "/")
    return any(spelled.startswith(prefix) for prefix in BUNDLED_PREFIXES)


def _path_param(name: str) -> Callable[[dict[str, Any]], list[str] | None]:
    """Needs the mount when *name* holds a path that is not a bundled sample."""
    def rule(params: dict[str, Any]) -> list[str] | None:
        value = _text(params.get(name))
        return [value] if value and not _bundled(value) else None
    return rule


def _training_loop(params: dict[str, Any]) -> list[str] | None:
    def positive(key: str) -> bool:
        try:
            return int(params.get(key) or 0) > 0
        except (TypeError, ValueError):
            return False

    writes = (positive("checkpoint_every")
              or positive("checkpoint_every_steps")
              or bool(params.get("tensorboard")))
    return [] if writes else None


def _document_loader(params: dict[str, Any]) -> list[str] | None:
    if _text(params.get("source")) != "directory":
        return None  # uploaded_file: its DATA_FILE param is the typed rule
    return _path_param("directory")(params)


def _graph_input(params: dict[str, Any]) -> list[str] | None:
    if _text(params.get("type")) != "image" or "value" in params:
        return None  # an API caller sends the pixels, not a path
    return _path_param("default")(params)


#: Node type -> rule: ``None`` (no need) or the files it reads (``[]`` for a
#: writer or a loop that only writes).
NODE_RULES: dict[str, Callable[[dict[str, Any]], list[str] | None]] = {
    "TrainingLoop": _training_loop,
    "FileReader": _path_param("path"),
    "ImageFolderDataset": _path_param("path"),
    "ImageBatchReader": _path_param("directory"),
    "VideoLoad": _path_param("path"),
    "DocumentLoader": _document_loader,
    "GraphInput": _graph_input,
}

#: Every built-in node that touches the data directories, and how this
#: module decides it. Kept explicit so a new reader or writer has to be
#: placed here (``tests/test_run_needs.py`` checks the list is complete).
COVERAGE: dict[str, str] = {
    # rule 1: a typed param
    "CSVReader": "typed param path (DATA_FILE)",
    "ImageReader": "typed param path (IMAGE_FILE)",
    "ModelLoader": "typed param path (MODEL_FILE)",
    "CheckpointLoader": "typed param path (MODEL_FILE)",
    "TextCorpusDataset": "typed param local_path (DATA_FILE) when "
                         "source=local_file; the Hub source downloads",
    # rule 2: writers
    "CheckpointSaver": "writer (MODELS_DIR)",
    "ModelSaver": "writer (MODELS_DIR)",
    "ImageWriter": "writer (<data>/output)",
    "VideoWrite": "writer (MEDIA_DIR)",
    # rule 3: node rules
    "TrainingLoop": "checkpoint_every / checkpoint_every_steps / "
                    "tensorboard",
    "FileReader": "path",
    "ImageFolderDataset": "path",
    "ImageBatchReader": "directory",
    "VideoLoad": "path",
    "DocumentLoader": "file (DATA_FILE) or a directory other than the "
                      "bundled sample",
    "GraphInput": "type=image with a default path",
    # not a need
    "Dataset": "none: downloads into its own cache",
    "LMTokenizedDataset": "none: a token cache each host fills itself",
    "DiffusionTrainingLoop": "none: writes only the interrupt checkpoint, "
                             "on a stop",
}


def _typed_files(node_cls: Any, params: dict[str, Any]
                 ) -> list[tuple[str, str]]:
    """``(kind, value)`` for every typed file param with a value."""
    files: list[tuple[str, str]] = []
    try:
        definitions = node_cls.define_params()
    except Exception:  # noqa: BLE001 -- a broken plugin node: no typed param
        return files
    for definition in definitions:
        if definition.param_type not in FILE_PARAM_TYPES:
            continue
        if not is_param_visible(definition, params):
            continue
        value = _text(params.get(definition.name))
        if value and not _bundled(value):
            files.append((INPUT_KINDS[definition.param_type], value))
    return files


def executable_nodes(graph: dict[str, Any]) -> list[dict[str, Any]]:
    """The nodes a run of *graph* executes, every param filled in (see the
    module docstring); also read by ``core.device_fallback`` (#839)."""
    from .graph_engine import build_preset_fallback, prepare_executable_graph

    try:
        nodes, _, _ = prepare_executable_graph(
            list(graph.get("nodes") or []), list(graph.get("edges") or []),
            preset_fallback=build_preset_fallback(graph.get("presets") or []),
            subgraphs=graph.get("subgraphs") or [], fill_defaults=True)
        return nodes
    except Exception:  # noqa: BLE001 -- read it node by node instead
        return list(_every_node(graph))


def _every_node(value: Any) -> Iterator[dict[str, Any]]:
    """Every node-shaped dict anywhere in *value*, nested ones included."""
    if isinstance(value, dict):
        if isinstance(value.get("type"), str) and isinstance(
                value.get("data"), dict):
            yield value
        for child in value.values():
            yield from _every_node(child)
    elif isinstance(value, list):
        for child in value:
            yield from _every_node(child)


def _node_needs(node_type: str, params: dict[str, Any]
                ) -> tuple[list[tuple[str, str]], list[str]] | None:
    """One node's need: ``None``, or ``(typed inputs, ruled paths)``."""
    from .node_registry import registry

    from .param_defaults import fill_missing_params

    node_cls = registry.get(node_type)
    if node_cls is not None:
        params = fill_missing_params(node_cls, params)
    typed = _typed_files(node_cls, params) if node_cls is not None else []
    rule = NODE_RULES.get(node_type)
    ruled = rule(params) if rule is not None else None
    if typed or ruled is not None or node_type in WRITERS:
        return typed, list(ruled or [])
    return None


def node_needs(node_type: str, params: dict[str, Any]) -> list[str] | None:
    """One node's need: ``None``, or the stored files it reads (``[]`` for
    a writer)."""
    found = _node_needs(node_type, params)
    if found is None:
        return None
    typed, ruled = found
    return [value for _, value in typed] + ruled


def needs(graph: dict[str, Any], *,
          label: str | None = None) -> dict[str, Any]:
    """``{"mount": label | None, "files": [...], "inputs": [...],
    "paths": [...]}`` for a run of *graph* (see the module docstring).

    *label* is the mount a need names (the control plane's,
    ``core.mounts.control_plane_label()``, by default).
    """
    if label is None:
        from .mounts import control_plane_label

        label = control_plane_label()
    mounted = False
    inputs: set[tuple[str, str]] = set()
    paths: set[str] = set()
    for node in executable_nodes(graph):
        node_type = node.get("type")
        data = node.get("data")
        params = data.get("params") if isinstance(data, dict) else None
        if not isinstance(node_type, str):
            continue
        found = _node_needs(node_type, params if isinstance(params, dict)
                            else {})
        if found is None:
            continue
        mounted = True
        inputs.update(found[0])
        paths.update(found[1])
    return {"mount": label if mounted else None,
            "files": sorted({name for _, name in inputs} | paths),
            "inputs": [{"kind": kind, "name": name}
                       for kind, name in sorted(inputs)],
            "paths": sorted(paths)}
