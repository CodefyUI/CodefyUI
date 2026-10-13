"""Hugging Face revision resolution and loading for the eval pack.

**A model id is not an identity.** ``main`` moves every time the owner
pushes, so a result recorded against ``bert-tiny@main`` cannot be
reproduced next month. Every load here therefore RESOLVES the requested
revision to one exact snapshot first and loads that snapshot, and the
loaders record both what was asked for and what it resolved to:

=========  ==============================================================
``hub``    asked the Hub what the revision points at now (a commit sha)
``cache``  the Hub was off (``HF_HUB_OFFLINE``) or unreachable, so the sha
           of the snapshot already in the cache is used, and the reason
           is recorded beside it
``pinned`` the revision given was already a full commit sha (still asked
           of the Hub when it is reachable, for its file list)
``local``  the id is a directory on disk; its identity is the sha256 of
           the model and tokenizer files in it (``local-sha256:...``)
=========  ==============================================================

**Loaded from the snapshot directory.** The resolved snapshot is fetched
(or found) with ``snapshot_download`` and the model and processor are loaded
from that ``snapshots/<sha>`` directory, never by repo id; see
:func:`snapshot_dir` for why.

**Where the bytes live.** Downloads go to CodefyUI's own Hugging Face cache
(``app.core.packs.paths.hf_cache_dir()``, under the user data dir, honouring
``CODEFYUI_USER_DATA_DIR``) -- the directory the Package Center downloads
into -- and never to ``HF_HOME``, which belongs to the whole machine.
``HF_HUB_OFFLINE=1`` is respected: nothing here touches the network then,
and a snapshot that is not already cached is an error that says so.

``transformers`` and ``huggingface_hub`` are imported inside the functions
that use them, so the node modules import on an install without
``transformers`` (the palette scan imports every node module at startup) and
fail only when a loader actually runs.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from os.path import basename, dirname, join
from typing import Any

from ._eval_core import is_commit_sha

#: Files that make up a model or processor directory, hashed (by name and
#: content, in this order) into a local directory's identity.
LOCAL_IDENTITY_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
    "preprocessor_config.json",
    "processor_config.json",
)

#: Files whose presence in the cache proves a snapshot of a repo is there.
_PROBE_FILES = ("config.json", "tokenizer_config.json", "preprocessor_config.json")

TASKS = ("auto", "classification", "generation")
DTYPES = ("float32", "auto", "float16", "bfloat16")

#: Attribute the loaders stamp on what they hand out.
PROVENANCE_ATTR = "cdui_eval_provenance"

#: Loaded models kept per process. Two, because the pack's reason to exist
#: is an A/B comparison; a third load evicts the oldest.
MAX_CACHED_MODELS = 2
_MODELS: OrderedDict[tuple[str, ...], Any] = OrderedDict()

INSTALL_HINT = (
    "The eval pack's Hugging Face nodes need the `transformers` package. "
    "`cdui plugin install eval` installs it; on an existing install run "
    "`uv pip install \"transformers>=5,<6\"` in the CodefyUI environment."
)


def _transformers() -> Any:
    try:
        import transformers
    except ImportError as exc:
        raise RuntimeError(INSTALL_HINT) from exc
    return transformers


def hf_cache_dir() -> str:
    from app.core.packs.paths import hf_cache_dir as _cache_dir

    return str(_cache_dir())


def hub_offline() -> bool:
    from huggingface_hub import is_offline_mode

    return bool(is_offline_mode())


def local_identity(directory: str) -> str | None:
    """``local-sha256:<hex>`` for a model directory on disk, else None."""
    if not directory:
        return None
    digest = hashlib.sha256()
    found = 0
    for name in LOCAL_IDENTITY_FILES:
        try:
            handle = open(join(directory, name), "rb")
        except OSError:
            continue
        with handle:
            digest.update(name.encode("utf-8") + b"\0")
            while True:
                chunk = handle.read(1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
        found += 1
    if not found:
        return None
    return "local-sha256:" + digest.hexdigest()


def cached_snapshot(repo_id: str, revision: str, cache_dir: str) -> str | None:
    """The cached ``snapshots/<sha>`` directory *revision* points at, or None.

    Found through a file every model or processor snapshot has, rather than
    ``snapshot_download(local_files_only=True)``: that call compares the
    snapshot with the repository's full file list and refuses one this pack
    deliberately downloaded only part of (one weights format, no README).
    """
    from huggingface_hub import try_to_load_from_cache

    for probe in _PROBE_FILES:
        try:
            found = try_to_load_from_cache(
                repo_id, probe, cache_dir=cache_dir, revision=revision
            )
        except Exception:  # noqa: BLE001 - a malformed id is simply "not cached"
            return None
        if isinstance(found, str) and is_commit_sha(basename(dirname(found))):
            return dirname(found)
    return None


def cached_sha(repo_id: str, revision: str, cache_dir: str) -> str | None:
    """The commit sha of the cached snapshot *revision* points at, or None."""
    directory = cached_snapshot(repo_id, revision, cache_dir)
    return basename(directory) if directory else None


def _hub_info(repo_id: str, revision: str) -> tuple[str, list[str]]:
    """(commit sha, file names) of ``repo_id@revision`` on the Hub."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, revision=revision, timeout=15)
    files = [str(s.rfilename) for s in (info.siblings or [])]
    return str(info.sha or ""), files


def _resolved(model_id: str, requested: str, revision: str, source: str,
              note: str = "", files: list[str] | None = None) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "requested_revision": requested,
        "revision": revision,
        "source": source,
        "note": note,
        # Files the snapshot needs, when the Hub listed them; None loads
        # whatever the cached snapshot already holds.
        "files": files,
        # Only the cache may be read: offline, or the Hub did not answer.
        "cache_only": source == "cache" or bool(note),
    }


def resolve_revision(model_id: str, revision: str) -> dict[str, Any]:
    """Resolve ``model_id@revision`` to one exact snapshot. See module doc."""
    model_id = str(model_id or "").strip()
    if not model_id:
        raise ValueError("a model_id is required (a Hub id such as 'org/name', or a local directory).")
    requested = str(revision or "").strip() or "main"
    local = local_identity(model_id)
    if local is not None:
        return _resolved(model_id, requested, local, "local")

    cache = hf_cache_dir()
    pinned = is_commit_sha(requested)
    if hub_offline():
        note = "HF_HUB_OFFLINE is set; using the cached snapshot"
    else:
        try:
            sha, files = _hub_info(model_id, requested)
        except Exception as exc:  # noqa: BLE001 - classified below
            kind = type(exc).__name__
            if kind in ("RepositoryNotFoundError", "RevisionNotFoundError", "GatedRepoError"):
                raise ValueError(
                    f"{model_id}@{requested} does not resolve on the Hugging Face Hub "
                    f"({kind}: {str(exc).splitlines()[0] if str(exc) else kind})."
                ) from exc
            note = f"the Hub was unreachable ({kind}); using the cached snapshot"
        else:
            if is_commit_sha(sha):
                return _resolved(model_id, requested, sha, "pinned" if pinned else "hub",
                                 files=files)
            note = "the Hub returned no commit sha; using the cached snapshot"

    if pinned:
        # A full sha names its snapshot whether or not the Hub answered.
        return _resolved(model_id, requested, requested, "pinned", note)
    sha = cached_sha(model_id, requested, cache)
    if sha is None:
        raise ValueError(
            f"{model_id}@{requested} is not in the cache ({cache}) and "
            + ("HF_HUB_OFFLINE is set, so nothing is downloaded."
               if hub_offline() else f"{note.split(';')[0]}.")
            + " Run once with network access to download it."
        )
    return _resolved(model_id, requested, sha, "cache", note)


def _wanted_files(files: list[str]) -> list[str]:
    """The top-level files a model and its processor load from.

    Configs, vocabularies and tokenizer models always; weights in ONE format
    -- safetensors when the repository has it, else ``pytorch_model*.bin`` --
    so a repository that ships both, or Flax / TF / ONNX copies besides, is
    not downloaded twice over.
    """
    top = [f for f in files if "/" not in f]
    has_safetensors = any(f.endswith(".safetensors") for f in top)
    keep = []
    for name in top:
        if name.endswith((".json", ".txt", ".model", ".tiktoken")):
            keep.append(name)
        elif has_safetensors and name.endswith(".safetensors"):
            keep.append(name)
        elif not has_safetensors and name.startswith("pytorch_model") and name.endswith(".bin"):
            keep.append(name)
    return keep


def snapshot_dir(resolved: dict[str, Any]) -> str:
    """The directory holding exactly ``resolved['revision']``, downloading if allowed.

    Everything is loaded from this directory rather than by repo id. That
    is what makes the revision exact -- the directory IS
    ``snapshots/<sha>`` -- and it keeps transformers' Hub-side behaviour out
    of a graph run: loading a ``.bin``-only repository by id starts a
    background request asking the Hub's converter bot to open a safetensors
    pull request on that repository.
    """
    if resolved["source"] == "local":
        return resolved["model_id"]
    from huggingface_hub import snapshot_download

    from_hub = resolved["source"] == "hub"
    kwargs: dict[str, Any] = {
        # A branch or tag is downloaded BY NAME, so the cache records which
        # commit the name pointed at (refs/<name>) and an offline run later
        # can resolve the same name from the cache.
        "revision": resolved["requested_revision"] if from_hub else resolved["revision"],
        "cache_dir": hf_cache_dir(),
        "local_files_only": bool(resolved.get("cache_only")) or hub_offline(),
    }
    if kwargs["local_files_only"]:
        directory = cached_snapshot(resolved["model_id"], resolved["revision"], kwargs["cache_dir"])
        if directory is None:
            raise ValueError(
                f"{resolved['model_id']}@{resolved['revision']} is not in the cache "
                f"({kwargs['cache_dir']}). Run once with network access to download it."
            )
        return directory
    if resolved.get("files") is not None:
        kwargs["allow_patterns"] = _wanted_files(resolved["files"])
    try:
        directory = str(snapshot_download(resolved["model_id"], **kwargs))
    except Exception as exc:  # noqa: BLE001 - reported with the model it was for
        raise ValueError(
            f"{resolved['model_id']}@{resolved['revision']} could not be downloaded "
            f"({type(exc).__name__}: {exc})"
        ) from exc
    landed = basename(directory.rstrip("/\\"))
    if from_hub and is_commit_sha(landed) and landed != resolved["revision"]:
        # The branch moved between asking and downloading: the files on
        # disk are the truth, so the record follows them.
        resolved["revision"] = landed
        resolved["note"] = "the branch moved while downloading; recorded the commit downloaded"
    return directory


def infer_task(directory: str) -> str:
    """``classification`` or ``generation``, read off the config's architectures."""
    config = _transformers().AutoConfig.from_pretrained(directory)
    architectures = [str(a) for a in (config.architectures or [])]
    if any(a.endswith("ForSequenceClassification") for a in architectures):
        return "classification"
    if any(a.endswith(("ForCausalLM", "LMHeadModel")) for a in architectures):
        return "generation"
    raise ValueError(
        f"cannot tell the task from the architectures {architectures}; "
        "set `task` to classification or generation."
    )


def load_model(
    resolved: dict[str, Any], *, task: str, dtype: str, device: str
) -> tuple[Any, str]:
    """The model at exactly ``resolved['revision']``, cast and moved, and its task.

    ``task`` may be ``auto``; the task actually loaded is returned with it.
    """
    import torch

    transformers = _transformers()
    directory = snapshot_dir(resolved)
    key = (resolved["model_id"], resolved["revision"], task, dtype, device)
    cached = _MODELS.get(key)
    if cached is not None:
        _MODELS.move_to_end(key)
        return cached

    loaded_task = infer_task(directory) if task == "auto" else task
    auto = (
        transformers.AutoModelForSequenceClassification
        if loaded_task == "classification"
        else transformers.AutoModelForCausalLM
    )
    model = auto.from_pretrained(directory)
    # Cast after the load, never through a from_pretrained keyword: the
    # keyword was renamed between transformers releases, `.to` was not.
    if dtype != "auto":
        model = model.to(dtype={"float32": torch.float32, "float16": torch.float16,
                                "bfloat16": torch.bfloat16}[dtype])
    model = model.to(device)
    model.eval()
    _MODELS[key] = (model, loaded_task)
    while len(_MODELS) > MAX_CACHED_MODELS:
        _MODELS.popitem(last=False)
    return model, loaded_task


def load_processor(resolved: dict[str, Any], *, kind: str) -> Any:
    transformers = _transformers()
    auto = transformers.AutoTokenizer if kind == "tokenizer" else transformers.AutoProcessor
    return auto.from_pretrained(snapshot_dir(resolved))



def provenance_of(obj: Any) -> dict[str, Any]:
    """The provenance a loader stamped on *obj*, or an empty dict."""
    try:
        value = obj.cdui_eval_provenance
    except AttributeError:
        return {}
    return dict(value) if isinstance(value, dict) else {}


def stamp(obj: Any, provenance: dict[str, Any]) -> None:
    obj.cdui_eval_provenance = dict(provenance)


def clear_model_cache() -> None:
    _MODELS.clear()
