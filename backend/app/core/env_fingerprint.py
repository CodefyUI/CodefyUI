"""The environment fingerprint: which software and hardware executed a run (#344).

``RunProvenance`` (``run_store``) records the USER's project commit and the
plugin lockfile as they were at SUBMISSION. Neither identifies the code that
performed the run: the project commit is not the CodefyUI checkout serving
it, and a queued run can start hours after it was submitted, on a server
that was updated in between. This module answers the other question --
what environment is executing right now -- as a plain, JSON-safe dict.

Three rules shape it:

- **Facts, never failures.** :func:`collect_fingerprint` never raises. A
  fact that cannot be read (no git, no CUDA, a package that is not
  installed) is an explicit ``None``, never an omitted key: an omitted key
  and a key that was never collected look identical to a reader comparing
  two fingerprints, and a ``None`` says "asked, unknown".
- **Nothing identifying about the person or the host.** No hostname, no
  username, no absolute path, no environment value outside a fixed
  allowlist of numerics-relevant variables. A linked local plugin records
  an absolute path in the lockfile; the fingerprint drops it. A fingerprint
  is meant to be pasted into a bug report or a paper's appendix.
- **Canonical bytes.** :func:`canonical_json` is sorted keys, no
  whitespace, UTF-8, no NaN; :func:`fingerprint_digest` is the sha256 of
  those bytes. The digest covers EVERY key of the document, so two
  environments that agree on every recorded fact get the same digest and a
  change to any of them (a CodefyUI commit, a torch version, a plugin pin,
  a driver) gets a different one. Nothing time-varying is collected --
  no timestamp, no free memory, no load -- precisely so that holds.

What it is not: a promise of numerical equality. Two runs with the same
digest and the same seed can still differ (nondeterministic kernels,
threads); two different digests can still agree. The digest says whether
the environments are the same, nothing more.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from .version import get_version

logger = logging.getLogger(__name__)

#: Bump when a key is added, removed or changes meaning. A stored
#: fingerprint carries the version it was written under, so a reader can
#: tell an old document from a new one that lacks a fact.
#:
#: 2 (#344, hook 5) adds ``content_sha256`` and ``immutable`` to every
#: ``plugins`` entry, and gives a linked local pack its content digest as
#: its ``revision`` (``revision_kind: "content-sha256"``).
SCHEMA_VERSION = 2

#: ``backend/app/core/env_fingerprint.py`` -> repository root.
_REPO_ROOT = Path(__file__).resolve().parents[3]

#: The device-visibility variables. A worker's run subprocess sees its own
#: narrowed value; its fingerprint takes the agent's (see
#: :func:`collect_fingerprint`'s *host*).
VISIBILITY_ENV_VARS: tuple[str, ...] = ("CUDA_VISIBLE_DEVICES",
                                        "HIP_VISIBLE_DEVICES")

#: Environment variables that change numerics or device selection. Only
#: these are read, and only their values; nothing else in ``os.environ``
#: reaches the fingerprint (it is where secrets live).
NUMERICS_ENV_VARS: tuple[str, ...] = (
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "ONEDNN_MAX_CPU_ISA",
    "PYTORCH_ENABLE_MPS_FALLBACK",
)

_DIST_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


# ── canonical form ────────────────────────────────────────────────────────


def canonical_json(fingerprint: dict[str, Any]) -> bytes:
    """The one byte string a fingerprint digests to.

    Sorted keys and compact separators make it independent of insertion
    order and formatting; ``ensure_ascii=False`` plus UTF-8 makes it one
    encoding of each string rather than two; ``allow_nan=False`` keeps it
    valid JSON (raises ``ValueError`` on a NaN, which no collected fact
    can be).
    """
    return json.dumps(fingerprint, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def fingerprint_digest(fingerprint: dict[str, Any]) -> str:
    """sha256 hex of :func:`canonical_json` -- covers every key."""
    return hashlib.sha256(canonical_json(fingerprint)).hexdigest()


# ── individual facts ──────────────────────────────────────────────────────


def _dist_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except Exception:  # noqa: BLE001 -- PackageNotFoundError or a broken dist
        return None


def _read_build_info(root: Path) -> dict[str, Any] | None:
    try:
        data = json.loads((root / "frontend" / "dist" / "build-info.json")
                          .read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _code_identity(root: Path) -> dict[str, Any]:
    """Which CodefyUI code is serving: version, commit, dirty, and from where.

    ``commit_source`` says how the commit was learned: ``"git"`` when the
    server runs from a checkout (``root/.git`` exists -- checked first so
    that an install sitting inside some unrelated repository, a home
    directory under git for instance, never reports THAT repository's
    commit), ``"build-info"`` when only ``frontend/dist/build-info.json``
    names one (it describes the bundle the release shipped, the closest
    thing an installed copy has), else ``None``.
    """
    from .project import git_provenance

    commit: str | None = None
    dirty: bool | None = None
    source: str | None = None
    if (root / ".git").exists():
        commit, dirty = git_provenance(root)
        if commit is not None:
            source = "git"
    if commit is None:
        info = _read_build_info(root)
        stamped = info.get("commit") if info else None
        if isinstance(stamped, str) and stamped.strip():
            commit, source = stamped.strip(), "build-info"
    return {"version": get_version(), "commit": commit, "dirty": dirty,
            "commit_source": source}


@lru_cache(maxsize=1)
def _static_facts() -> dict[str, Any]:
    """What cannot change while this process lives. Computed once.

    The code identity belongs here too: the Python that is executing was
    imported at startup, so a checkout edited or pulled afterwards is not
    what runs until the server restarts -- re-reading git per run would
    report code the process never loaded.
    """
    torch_build: dict[str, Any] = {
        "git_version": None, "cuda": None, "hip": None, "cudnn": None,
        "debug": None,
    }
    try:
        import torch

        torch_build["git_version"] = getattr(torch.version, "git_version", None)
        torch_build["cuda"] = getattr(torch.version, "cuda", None)
        torch_build["hip"] = getattr(torch.version, "hip", None)
        torch_build["debug"] = bool(getattr(torch.version, "debug", False))
        try:
            torch_build["cudnn"] = torch.backends.cudnn.version()
        except Exception:  # noqa: BLE001
            torch_build["cudnn"] = None
    except Exception:  # noqa: BLE001 -- torch missing or broken
        pass

    mac_version = platform.mac_ver()[0] if sys.platform == "darwin" else ""
    return {
        "codefyui": _code_identity(_REPO_ROOT),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            # platform.node() is the hostname and is deliberately absent.
            "system": platform.system() or None,
            "release": platform.release() or None,
            "machine": platform.machine() or None,
            "mac_version": mac_version or None,
        },
        "torch_build": torch_build,
    }


def _libraries() -> dict[str, str | None]:
    return {name: _dist_version(name)
            for name in ("torch", "torchvision", "numpy", "mlx")}


def _cuda_driver() -> str | None:
    """The NVIDIA driver version, when ``pynvml`` can say. Else None."""
    try:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        try:
            version = pynvml.nvmlSystemGetDriverVersion()
        finally:
            pynvml.nvmlShutdown()
        return version.decode() if isinstance(version, bytes) else str(version)
    except Exception:  # noqa: BLE001 -- not installed, no driver, no GPU
        return None


def _accelerators() -> dict[str, Any]:
    from .device_utils import cuda_device_count, mlx_available

    facts: dict[str, Any] = {
        "cuda_available": False, "rocm": False, "cuda_devices": [],
        "cuda_driver": None, "mps_built": None, "mps_available": None,
        "mlx_available": mlx_available(),
    }
    try:
        import torch
    except Exception:  # noqa: BLE001
        return facts

    try:
        if torch.cuda.is_available():
            facts["cuda_available"] = True
            facts["rocm"] = getattr(torch.version, "hip", None) is not None
            devices = []
            for index in range(cuda_device_count()):
                try:
                    name = torch.cuda.get_device_name(index)
                except Exception:  # noqa: BLE001
                    name = None
                try:
                    major, minor = torch.cuda.get_device_capability(index)
                    capability: str | None = f"{major}.{minor}"
                except Exception:  # noqa: BLE001
                    capability = None
                devices.append({"index": index, "name": name,
                                "capability": capability})
            facts["cuda_devices"] = devices
            if not facts["rocm"]:
                facts["cuda_driver"] = _cuda_driver()
    except Exception:  # noqa: BLE001
        pass
    mps = getattr(torch.backends, "mps", None)
    if mps is not None:
        try:
            facts["mps_built"] = bool(mps.is_built())
            facts["mps_available"] = bool(mps.is_available())
        except Exception:  # noqa: BLE001
            pass
    return facts


def _devices() -> dict[str, Any]:
    """The device list the server offers -- ``GET /api/system/devices``'s."""
    from .device_utils import describe_accelerator

    try:
        described = describe_accelerator()
    except Exception:  # noqa: BLE001
        return {"default": None, "offered": []}
    return {
        "default": described.get("default"),
        "offered": [d.get("value") for d in described.get("devices", [])],
    }


def _plugin_identity(plugin_id: str, entry: dict[str, Any],
                     code: dict[str, Any],
                     directory: Path | None = None) -> dict[str, Any]:
    """One enabled lockfile entry, as an identity that names no local path.

    ``revision`` is the thing the pack's code is identified by, and
    ``revision_kind`` says what kind of thing it is:

    - ``"git-sha"`` for a pack installed from a repository at a commit.
    - ``"codefyui-commit"`` for a builtin pack: it ships inside the
      CodefyUI checkout, so the checkout's commit (and its ``dirty`` flag,
      under ``codefyui``) is its revision. ``source`` is its path relative
      to the repository.
    - ``"content-sha256"`` for a linked local pack (#344, hook 5): it has
      no immutable revision, so its revision is the digest of its files as
      they were when the run started. Its ``source`` is withheld (it is an
      absolute path on the author's machine).
    - ``None`` when nothing identifies the code (a local pack whose files
      could not be read, a builtin pack outside a git checkout).

    ``content_sha256`` is the digest of the pack's files for every kind
    (``content_hash.content_identity``, incremental, skipping VCS, cache,
    virtual-environment and build directories; a pack inside a git
    checkout is listed by ``git ls-files``, so what its ``.gitignore``
    excludes is left out), so a pack edited in place after
    it was installed is visible as such; ``None`` when unreadable.
    ``immutable`` says whether ``revision`` names something the code can
    be recovered from later: True for a git commit (a builtin pack's only
    while the checkout is clean), False for a content digest, which
    identifies the files but cannot bring them back.
    """
    kind = str(entry.get("source_kind") or "") or None
    manifest = entry.get("manifest")
    version = manifest.get("version") if isinstance(manifest, dict) else None
    sha = str(entry.get("sha") or "") or None
    ref = str(entry.get("ref") or "") or None
    source: str | None = str(entry.get("source") or "") or None
    content = _pack_content(directory)
    revision: str | None = None
    revision_kind: str | None = None
    immutable = False
    if kind == "builtin":
        source = f"plugins/{plugin_id}"
        revision = code.get("commit")
        revision_kind = "codefyui-commit" if revision else None
        immutable = bool(revision) and code.get("dirty") is False
    elif kind == "local":
        source = None
        if content is not None:
            revision, revision_kind = content, "content-sha256"
    elif sha:
        revision, revision_kind, immutable = sha, "git-sha", True
    return {
        "source_kind": kind,
        "source": source,
        "ref": ref,
        "version": version if isinstance(version, str) else None,
        "revision": revision,
        "revision_kind": revision_kind,
        "immutable": immutable,
        "content_sha256": content,
    }


def _pack_content(directory: Path | None) -> str | None:
    """The content digest of a pack directory, or None."""
    if directory is None:
        return None
    try:
        from .content_hash import PACK_SKIPS, content_identity

        if not directory.is_dir():
            return None
        return content_identity([directory], use_git=True,
                                skips=PACK_SKIPS).get("digest")
    except Exception:  # noqa: BLE001 -- a fact, never a failure
        return None


def _plugins(code: dict[str, Any]) -> dict[str, Any]:
    from .plugin_loader import (
        _resolve_plugin_dir,
        is_enabled,
        load_lockfile,
        plugins_builtin_root,
        plugins_user_root,
    )

    try:
        plugins = load_lockfile().get("plugins")
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(plugins, dict):
        return {}

    def directory(plugin_id: str, entry: dict[str, Any]) -> Path | None:
        try:
            return _resolve_plugin_dir(plugin_id, entry,
                                       plugins_builtin_root(),
                                       plugins_user_root())
        except Exception:  # noqa: BLE001
            return None

    return {
        str(plugin_id): _plugin_identity(str(plugin_id), entry, code,
                                         directory(str(plugin_id), entry))
        for plugin_id, entry in plugins.items()
        if isinstance(entry, dict) and is_enabled(entry)
    }


def _custom_nodes() -> dict[str, str]:
    """Enabled custom node files by name, each with its content's sha256.

    Custom nodes have no version at all, so their bytes are the identity.
    ``__init__.py`` is the package's own and not a node.
    """
    from ..config import settings

    out: dict[str, str] = {}
    try:
        files = sorted(Path(settings.CUSTOM_NODES_DIR).glob("*.py"))
    except Exception:  # noqa: BLE001
        return out
    for path in files:
        if path.name == "__init__.py":
            continue
        try:
            out[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue
    return out


def _optional_packages() -> dict[str, dict[str, str | None]]:
    """Package Center packs: each pip distribution's installed version.

    Every catalog pack is listed whether or not it is installed, with
    ``None`` for a distribution that is not, so installing one changes the
    digest instead of adding a key a reader might not know to look for.
    """
    try:
        from .packs.catalog import iter_packs

        packs = iter_packs()
    except Exception:  # noqa: BLE001
        return {}
    out: dict[str, dict[str, str | None]] = {}
    for pack in packs:
        dists: dict[str, str | None] = {}
        for spec in pack.pip:
            match = _DIST_NAME.match(spec)
            if match:
                dists[match.group(1)] = _dist_version(match.group(1))
        out[pack.pack_id] = dists
    return out


def _environment_variables() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in NUMERICS_ENV_VARS}


# ── the document ──────────────────────────────────────────────────────────


def collect_fingerprint(host: dict[str, Any] | None = None) -> dict[str, Any]:
    """The fingerprint of this process, as a fresh JSON-safe dict.

    Blocking (a lockfile read, a few package-metadata lookups, git once per
    process) -- call it in a thread from async code. Never raises.

    Top-level keys, all present in every document:

    ``schema_version``, ``codefyui`` (version, commit, dirty,
    commit_source), ``python``, ``platform``, ``libraries`` (torch,
    torchvision, numpy, mlx), ``torch_build`` (git_version, cuda, hip,
    cudnn, debug), ``accelerators`` (cuda/rocm/mps/mlx availability, the
    CUDA cards' names and compute capabilities, the NVIDIA driver),
    ``devices`` (the server's ``default`` and the ``offered`` list),
    ``plugins`` (enabled packs, see :func:`_plugin_identity`),
    ``custom_nodes`` (file name -> sha256), ``optional_packages`` (Package
    Center pip versions) and ``env`` (:data:`NUMERICS_ENV_VARS`).

    *host* is the fingerprint of the process that started this one, for a
    worker's run subprocess (#344): the agent narrows each run's
    ``CUDA_VISIBLE_DEVICES`` to the run's cards, so the subprocess would
    describe a machine with only those cards. With *host*, ``accelerators``,
    ``devices`` and the :data:`VISIBILITY_ENV_VARS` come from it (the
    machine as the agent sees it, as on the control plane, where a run's
    own device set is in its ``execution_facts``), and everything else --
    the code, the libraries, the plugins and custom nodes the subprocess
    loaded -- is read here. A subprocess on an unchanged machine therefore
    reports the agent's digest exactly.
    """
    static = _static_facts()
    # Deep enough a copy: the cached dict's nested values are dicts of
    # scalars, and a caller mutating the result must not edit the cache.
    document: dict[str, Any] = {
        key: dict(value) for key, value in static.items()
    }
    host = host if isinstance(host, dict) else None
    document["schema_version"] = SCHEMA_VERSION
    document["libraries"] = _libraries()
    if host is not None and isinstance(host.get("accelerators"), dict) \
            and isinstance(host.get("devices"), dict):
        document["accelerators"] = json.loads(json.dumps(host["accelerators"]))
        document["devices"] = json.loads(json.dumps(host["devices"]))
    else:
        host = None
        document["accelerators"] = _accelerators()
        document["devices"] = _devices()
    document["plugins"] = _plugins(document["codefyui"])
    document["custom_nodes"] = _custom_nodes()
    document["optional_packages"] = _optional_packages()
    document["env"] = _environment_variables()
    if host is not None:
        host_env = host.get("env") if isinstance(host.get("env"), dict) else {}
        for name in VISIBILITY_ENV_VARS:
            document["env"][name] = host_env.get(name)
    return document


def collect_with_digest() -> tuple[dict[str, Any], str]:
    """``(fingerprint, digest)`` in one call, for the endpoint and the run."""
    fingerprint = collect_fingerprint()
    return fingerprint, fingerprint_digest(fingerprint)


# ── what one run resolved ─────────────────────────────────────────────────


#: Bump with :data:`SCHEMA_VERSION`'s rule, for ``execution_facts``.
#:
#: 2 adds keys merged in after the run starts: ``peak_memory`` (#770, at
#: finish) and, for a run on a worker, ``visible_devices`` and
#: ``device_aliases`` (#774, from ``run_started``). A key a run never got
#: is absent, as in version 1.
#:
#: 3 adds ``fingerprint_source`` (#344): which process the run's
#: ``fingerprint_digest`` describes (:data:`FINGERPRINT_PROCESS`,
#: :data:`FINGERPRINT_WORKER_HELLO`, :data:`FINGERPRINT_WORKER_RUN`; None
#: when none was recorded), and, for a worker run, ``resolved.numerics``
#: from the run subprocess.
EXECUTION_FACTS_VERSION = 3

#: ``fingerprint_source``: the control plane process executed the run and
#: collected the fingerprint when it started.
FINGERPRINT_PROCESS = "process"
#: A worker run, before its subprocess reported: the fingerprint the
#: worker agent's ``hello`` carried (collected when the agent started).
FINGERPRINT_WORKER_HELLO = "worker-hello"
#: A worker run whose subprocess reported the fingerprint of the code and
#: plugins it loaded when the run started (``agent.child``).
FINGERPRINT_WORKER_RUN = "worker-run"


def numerics_settings() -> dict[str, Any]:
    """Process-wide torch numerics switches, read now.

    CodefyUI has no run-level precision option: mixed precision is a
    per-node parameter (the training nodes' AMP setting), recorded in the
    graph snapshot. What a run does inherit from the process are these
    switches, so they are recorded as they stood when the run started. A
    node can still change them mid-run.
    """
    facts: dict[str, Any] = {
        "default_dtype": None, "float32_matmul_precision": None,
        "cuda_matmul_allow_tf32": None, "cudnn_allow_tf32": None,
        "cudnn_benchmark": None, "num_threads": None,
    }
    try:
        import torch
    except Exception:  # noqa: BLE001
        return facts
    readers = {
        "default_dtype": lambda: str(torch.get_default_dtype()),
        "float32_matmul_precision": torch.get_float32_matmul_precision,
        "cuda_matmul_allow_tf32":
            lambda: bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": lambda: bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": lambda: bool(torch.backends.cudnn.benchmark),
        "num_threads": torch.get_num_threads,
    }
    for key, read in readers.items():
        try:
            facts[key] = read()
        except Exception:  # noqa: BLE001
            facts[key] = None
    return facts


def worker_execution_facts(
    *, options: dict[str, Any], graph: dict[str, Any] | None,
    worker_id: str, devices: list[str],
) -> dict[str, Any]:
    """:func:`execution_facts` for a run placed on a worker agent (#772).

    Only what this process knows without being the one that executes:
    ``requested`` as usual, and ``resolved`` with the placement's first
    device, the full device set, the seed and the determinism flag -- no
    ``numerics``, which would be this process's switches. ``facts_source``
    says the worker is the authority for the rest; its ``run_started``
    frame merges its own facts in. Never raises.
    """
    facts = execution_facts(options=options, graph=graph,
                            resolved_device=devices[0] if devices else None,
                            numerics=False)
    facts["facts_source"] = "worker"
    facts["worker_id"] = worker_id
    facts["resolved"]["devices"] = list(devices)
    return facts


def execution_facts(
    *, options: dict[str, Any], graph: dict[str, Any] | None,
    resolved_device: str | None, numerics: bool = True,
) -> dict[str, Any]:
    """Submitted intent next to what the run actually got. Never raises.

    ``requested`` is what was asked for: the stored options' ``device``,
    ``seed`` and ``deterministic`` (after the graph's ``settings`` filled
    any the submitter left out) plus the graph's own ``settings.device`` /
    ``settings.seed``, so a reader can see whether the submitter overrode
    them. ``resolved`` is what the run executes with: the device the
    scheduler resolved (``auto`` and an unavailable accelerator are gone by
    then; a node with its own ``device`` parameter can still override it
    for that node), the seed and determinism flag handed to the engine, and
    :func:`numerics_settings`.

    A seed fixes the random draws, not the arithmetic: the same seed on
    different hardware or library versions can produce different numbers.
    """
    from .device_utils import graph_settings_device, graph_settings_seed

    def _safe(read):
        try:
            return read()
        except Exception:  # noqa: BLE001
            return None

    seed = options.get("seed")
    deterministic = bool(options.get("deterministic"))
    return {
        "schema_version": EXECUTION_FACTS_VERSION,
        "requested": {
            "device": options.get("device"),
            "seed": seed,
            "deterministic": deterministic,
            "graph_settings_device": _safe(lambda: graph_settings_device(graph)),
            "graph_settings_seed": _safe(lambda: graph_settings_seed(graph)),
        },
        "resolved": {
            "device": resolved_device,
            "seed": seed,
            "deterministic": deterministic,
            **({"numerics": numerics_settings()} if numerics else {}),
        },
    }
