"""EvalJSONLDatasetNode -- one JSON object per line, as an eval set with an identity.

Every item gets a stable ``id`` (the record's own id field, or its line
number when the file has none) and the set as a whole gets an identity: the
path it was read from, the sha256 of the file's bytes and its row count. The
id is what CompareRuns aligns two runs on; the hash is what lets a reader
tell, months later, whether two results were computed on the same file.

Duplicate ids are refused at load rather than discovered at comparison
time, because a duplicated id makes "item s07" mean two different rows.

With ``image_field`` set, each record also names an image file, relative to
the JSONL file's own folder (the layout ``samples/shapes-vqa-12.jsonl`` next
to ``samples/shapes-vqa/`` uses). Every image is read and hashed at load: a
missing or unreadable image is refused with its line number, the item
carries the image's sha256 into its prediction row, and the set's identity
gains ``images_sha256`` -- the JSONL's own hash cannot see an image that
changed under an unchanged name.
"""

from __future__ import annotations

import json
from os.path import dirname, isabs, join
from typing import Any

from app.core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)

from ._eval_core import first_duplicates, make_dataset, sha256_bytes, sha256_text

#: Prefix that names a file shipped inside this pack.
PACK_PREFIX = "eval:"
_PACK_DIR = dirname(dirname(__file__))


def _label_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


class EvalJSONLDatasetNode(BaseNode):
    NODE_NAME = "Eval-JSONLDataset"
    CATEGORY = "Data"
    DESCRIPTION = "A JSONL file as an eval set with ids and a content hash"
    DETAILS = (
        "Reads one JSON object per line. `input_field` is the text the model "
        "sees, `label_field` the expected answer (optional per row), and "
        "`id_field` the stable item id; a file without ids gets `line-N`. "
        "Duplicate ids, blank inputs and malformed lines are refused with the "
        "line number. The dataset carries its identity -- path, sha256 of the "
        "file and row count -- into every prediction table built from it. A "
        "path starting with `eval:` names a file shipped in this pack, such as "
        "`eval:samples/sst2-mini-20.jsonl`. `image_field` names a field holding "
        "an image path relative to the JSONL file's folder, for VQA; each image "
        "is hashed into the item's identity."
    )

    # The file can change under an unchanged path, and the run is cheap.
    cacheable = False
    align_inputs = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="dataset",
                data_type=DataType.ANY,
                description="Eval dataset: items with id, input and label, plus the file's identity.",
            ),
            PortDefinition(
                name="labels",
                data_type=DataType.LIST,
                description="The label of each item, in file order (None where a row has none).",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="path",
                param_type=ParamType.DATA_FILE,
                default="",
                description=(
                    "JSONL file. An uploaded file, a path relative to the backend "
                    "working directory, or `eval:samples/...` for a file this pack ships."
                ),
            ),
            ParamDefinition(
                name="input_field",
                param_type=ParamType.STRING,
                default="text",
                description="Field holding the model input text.",
            ),
            ParamDefinition(
                name="label_field",
                param_type=ParamType.STRING,
                default="label",
                description="Field holding the expected answer. Rows without it have no label.",
            ),
            ParamDefinition(
                name="id_field",
                param_type=ParamType.STRING,
                default="id",
                description="Field holding a stable item id. Missing ids become line-N.",
            ),
            ParamDefinition(
                name="image_field",
                param_type=ParamType.STRING,
                default="",
                description=(
                    "Field holding an image path, relative to the JSONL file's folder "
                    "(VQA). Empty reads no images."
                ),
            ),
            ParamDefinition(
                name="limit",
                param_type=ParamType.INT,
                default=0,
                min_value=0,
                description="Use only the first N items. 0 uses every item.",
                advanced=True,
            ),
        ]

    @staticmethod
    def resolve_path(path: str) -> str:
        if path.startswith(PACK_PREFIX):
            rest = path[len(PACK_PREFIX):].lstrip("/\\")
            parts = [p for p in rest.replace("\\", "/").split("/") if p]
            if not parts or any(p in (".", "..") for p in parts):
                raise ValueError(
                    f"Eval-JSONLDataset: {path!r} does not name a file inside the eval pack."
                )
            return join(_PACK_DIR, *parts)
        from app.nodes.data.csv_reader_node import CSVReaderNode

        # The same resolution CSVReader uses: uploaded files by bare name,
        # project mode, and the project-escape check.
        return str(CSVReaderNode._resolve_path(path))

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        progress_callback: Any | None = None,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        node = self.NODE_NAME
        path = str(params.get("path", "") or "").strip()
        if not path:
            raise ValueError(f"{node}: set `path` to a JSONL file.")
        input_field = str(params.get("input_field", "text") or "").strip()
        label_field = str(params.get("label_field", "label") or "").strip()
        id_field = str(params.get("id_field", "id") or "").strip()
        if not input_field:
            raise ValueError(f"{node}: `input_field` cannot be empty.")
        image_field = str(params.get("image_field", "") or "").strip()
        limit = max(0, int(params.get("limit", 0) or 0))

        resolved = self.resolve_path(path)
        try:
            with open(resolved, "rb") as handle:
                raw = handle.read()
        except FileNotFoundError:
            raise ValueError(f"{node}: no file at {path!r} (resolved to {resolved}).") from None
        except OSError as exc:
            raise ValueError(f"{node}: cannot read {path!r}: {exc}") from exc
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{node}: {path!r} is not UTF-8 text ({exc}).") from exc

        items: list[dict[str, Any]] = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{node}: line {lineno} is not valid JSON ({exc.msg}).") from None
            if not isinstance(record, dict):
                raise ValueError(
                    f"{node}: line {lineno} is a JSON {type(record).__name__}, "
                    "not an object."
                )
            value = record.get(input_field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{node}: line {lineno} has no text in `{input_field}` "
                    f"(fields: {sorted(record)})."
                )
            raw_id = record.get(id_field) if id_field else None
            item_id = str(raw_id) if raw_id is not None and str(raw_id) != "" else f"line-{lineno}"
            item = {
                "id": item_id,
                "input": value,
                "label": _label_text(record.get(label_field)) if label_field else None,
                "line": lineno,
                "fields": record,
            }
            if image_field:
                item.update(_read_image(
                    node, record.get(image_field), field=image_field,
                    base=dirname(resolved), lineno=lineno,
                ))
            items.append(item)

        if not items:
            raise ValueError(f"{node}: {path!r} has no records.")
        duplicates = first_duplicates(item["id"] for item in items)
        if duplicates:
            raise ValueError(
                f"{node}: duplicate ids in {path!r}: {duplicates}. Every item "
                "needs its own id -- runs are compared by id, never by position."
            )

        total = len(items)
        every_item = items
        if limit:
            items = items[:limit]
        meta = {
            "path": path,
            "resolved_path": resolved,
            "sha256": sha256_bytes(raw),
            "rows": total,
            "used": len(items),
            "limit": limit,
            "fields": {"input": input_field, "label": label_field, "id": id_field},
            "template": None,
        }
        if image_field:
            meta["fields"]["image"] = image_field
            # Over every row of the file, like `sha256`, so `limit` does not
            # change the set's identity.
            meta["images_sha256"] = sha256_text("".join(
                f"{item['id']}:{item['image_sha256']};" for item in every_item))
        log = (
            f"{len(items)} item(s) from {path} (of {total}; sha256 "
            f"{meta['sha256'][:12]})"
            + (f", one image each (images sha256 {meta['images_sha256'][:12]})"
               if image_field else "")
        )
        return {
            "dataset": make_dataset(items, meta),
            "labels": [item["label"] for item in items],
            "__log__": log,
        }


def _read_image(node: str, value: Any, *, field: str, base: str, lineno: int) -> dict[str, Any]:
    """Resolve and hash one item's image; refuse what cannot be read.

    The path is relative to the JSONL file's folder and may not leave it: an
    eval set describes its own images, and a ``..`` or absolute path would
    let a shared file point the server at anything it can read.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{node}: line {lineno} has no image path in `{field}`.")
    normalized = value.strip().replace("\\", "/")
    parts = [p for p in normalized.split("/") if p]
    if (isabs(value) or normalized.startswith("/") or ":" in normalized
            or any(p in (".", "..") for p in parts)):
        raise ValueError(
            f"{node}: line {lineno}: image {value!r} must be a relative path inside "
            "the JSONL file's folder, without '..'."
        )
    path = join(base, *parts)
    # The string check above cannot see a symlink: `link/hosts` with
    # `link -> /etc` passes it. So containment is checked again on the
    # resolved paths, with the rule the API routes use (#483).
    from app.core.data_paths import resolve_under

    if resolve_under(base, "/".join(parts)) is None:
        raise ValueError(
            f"{node}: line {lineno}: image {value!r} resolves outside the JSONL "
            "file's folder (through a symbolic link)."
        )
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError as exc:
        raise ValueError(
            f"{node}: line {lineno}: cannot read image {value!r} ({type(exc).__name__})."
        ) from None
    return {"image": value, "image_path": path, "image_sha256": sha256_bytes(data)}
