"""The Package Center allowlist: every optional pack, and what is in it.

An ALLOWLIST, not a package manager. The server installs what is written
here and nothing else -- no arbitrary pip spec or Hugging Face repo id ever
reaches a subprocess from a request body -- so this file is the entire
attack surface of the feature, and it is deliberately plain data.

Stdlib only, and no imports from the rest of the app, so routes, node code
and the CLI can all read it without an import cycle. Nothing here touches
the network or the filesystem; where the bytes land is ``paths.py`` and
putting them there is a later layer's job.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

#: How a pack's install finishes. "live" packs are usable the moment the
#: install ends; "restart" packs replace something already imported into the
#: running process (torch), so the server has to come back up first.
INSTALL_MODES = frozenset({"live", "restart"})

#: Where an item's bytes come from: a Hugging Face repo snapshot, or a
#: single file fetched over HTTPS into the asset cache.
ITEM_KINDS = frozenset({"hf", "asset"})


@dataclass(frozen=True)
class ModelItem:
    """One downloadable artifact inside a pack."""

    item_id: str                 # stable id used in URLs and sentinels
    kind: str                    # "hf" | "asset"
    approx_bytes: int            # for the disk precheck and progress fallback
    license: str                 # e.g. "Apache-2.0", "MIT", "PDDL-1.0"
    repo_id: str | None = None   # hf items
    revision: str = "main"       # hf items
    url: str | None = None       # asset items
    sha256: str | None = None    # asset items (None until recorded)
    filename: str | None = None  # asset items: file name inside the asset dir


@dataclass(frozen=True)
class Pack:
    """One installable bundle: some pip specs, some model files, or both."""

    pack_id: str
    title: str                       # English; the frontend may override by i18n key
    description: str                 # English, one sentence
    pip: tuple[str, ...]             # PEP 508 specs
    probe_modules: tuple[str, ...]   # import names checked with find_spec
    items: tuple[ModelItem, ...]
    depends_on: tuple[str, ...]      # pack ids
    install_mode: str                # "live" | "restart"


CATALOG: tuple[Pack, ...] = (
    Pack(
        pack_id="sentence-embeddings",
        title="Sentence embeddings",
        description=(
            "sentence-transformers plus four small embedding models "
            "(English, multilingual, Chinese) for TextEmbedding and WordVector."
        ),
        pip=("sentence-transformers>=3.0,<6",),
        probe_modules=("sentence_transformers", "transformers"),
        items=(
            ModelItem(
                item_id="all-MiniLM-L6-v2",
                kind="hf",
                repo_id="sentence-transformers/all-MiniLM-L6-v2",
                approx_bytes=90_000_000,
                license="Apache-2.0",
            ),
            ModelItem(
                item_id="paraphrase-multilingual-MiniLM-L12-v2",
                kind="hf",
                repo_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                approx_bytes=470_000_000,
                license="Apache-2.0",
            ),
            ModelItem(
                item_id="bge-small-zh-v1.5",
                kind="hf",
                repo_id="BAAI/bge-small-zh-v1.5",
                approx_bytes=95_000_000,
                license="MIT",
            ),
            ModelItem(
                item_id="multilingual-e5-small",
                kind="hf",
                repo_id="intfloat/multilingual-e5-small",
                approx_bytes=470_000_000,
                license="MIT",
            ),
        ),
        depends_on=(),
        install_mode="live",
    ),
    Pack(
        pack_id="word-vectors",
        title="Word vectors (GloVe)",
        description=(
            "Real 400k-word GloVe-50d table for WordVector; "
            "no Python packages needed."
        ),
        pip=(),
        probe_modules=(),
        items=(
            ModelItem(
                item_id="glove-50d",
                kind="asset",
                url=(
                    "https://github.com/RaRe-Technologies/gensim-data/releases/"
                    "download/glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz"
                ),
                filename="glove-wiki-gigaword-50.gz",
                # Measured off the release asset itself (69 182 535 bytes on
                # the wire, 400k words x 50 dimensions once unzipped), then
                # confirmed by a second independent download.
                # ``tests/test_packs_network.py`` re-checks it against the
                # live URL whenever a maintainer runs the opt-in suite; that
                # is also the test that PRINTS a digest to record when an
                # asset item has none yet.
                sha256="5c55f98957aa9fed8d2ac5fb1dcff57af3b23c5a3ee7af3f7945f8d49198eb24",
                approx_bytes=66_000_000,
                license="PDDL-1.0",
            ),
        ),
        depends_on=(),
        install_mode="live",
    ),
    Pack(
        pack_id="rag",
        title="RAG stack",
        description=(
            "Local generator model Qwen2.5-0.5B-Instruct for HFTextGenerate; "
            "needs Sentence embeddings first."
        ),
        pip=(),
        probe_modules=("transformers",),
        items=(
            ModelItem(
                item_id="qwen2.5-0.5b-instruct",
                kind="hf",
                repo_id="Qwen/Qwen2.5-0.5B-Instruct",
                approx_bytes=1_000_000_000,
                license="Apache-2.0",
            ),
        ),
        depends_on=("sentence-embeddings",),
        install_mode="live",
    ),
    Pack(
        pack_id="gpu-torch",
        title="GPU PyTorch",
        description=(
            "Switch PyTorch to the CUDA/ROCm build that matches this machine; "
            "the server restarts."
        ),
        pip=(),
        probe_modules=(),
        items=(),
        depends_on=(),
        install_mode="restart",
    ),
)

#: Packs whose ``pip`` list is also a pyproject extra. The two have to stay
#: equal: one is what the in-app installer runs, the other is what
#: ``pip install -e .[llm-sentence]`` gets, and they describe the same
#: decision about which versions this codebase is tested against.
PYPROJECT_EXTRA_FOR_PACK: dict[str, str] = {"sentence-embeddings": "llm-sentence"}


def _validate(packs: Sequence[Pack]) -> None:
    """Raise ``ValueError`` if *packs* could not be installed as written.

    Runs over ``CATALOG`` at import time, so a malformed entry fails the
    server at startup and every test run -- rather than halfway through a
    download, on a learner's machine.
    """
    by_id: dict[str, Pack] = {}
    for pack in packs:
        if pack.pack_id in by_id:
            raise ValueError(f"duplicate pack id: {pack.pack_id!r}")
        by_id[pack.pack_id] = pack

    # Across the WHOLE catalog, not per pack. Uninstalling a Hugging Face
    # item deletes the repo folder -- the one directory where that repo's
    # bytes live, shared by all of its revisions -- so two items naming one
    # repo would make removing either of them delete the other's model,
    # while the other's sentinel went on claiming it was there.
    repo_owner: dict[str, str] = {}
    for pack in packs:
        for item in pack.items:
            if item.kind != "hf" or not item.repo_id:
                continue
            owner = repo_owner.get(item.repo_id)
            if owner is not None:
                raise ValueError(
                    f"repo_id {item.repo_id!r} is used by item "
                    f"{item.item_id!r} in pack {pack.pack_id!r} and by {owner}")
            repo_owner[item.repo_id] = f"{pack.pack_id}/{item.item_id}"

    for pack in packs:
        if pack.install_mode not in INSTALL_MODES:
            raise ValueError(
                f"pack {pack.pack_id!r} has install_mode {pack.install_mode!r}; "
                f"expected one of {sorted(INSTALL_MODES)}")

        seen_items: set[str] = set()
        for item in pack.items:
            if item.item_id in seen_items:
                raise ValueError(
                    f"duplicate item id {item.item_id!r} in pack {pack.pack_id!r}")
            seen_items.add(item.item_id)

            if item.kind not in ITEM_KINDS:
                raise ValueError(
                    f"item {item.item_id!r} has kind {item.kind!r}; "
                    f"expected one of {sorted(ITEM_KINDS)}")
            if item.kind == "hf" and not item.repo_id:
                raise ValueError(f"hf item {item.item_id!r} has no repo_id")
            if item.kind == "asset" and not item.url:
                raise ValueError(f"asset item {item.item_id!r} has no url")
            if item.kind == "asset" and not item.filename:
                raise ValueError(f"asset item {item.item_id!r} has no filename")

        for dep in pack.depends_on:
            if dep == pack.pack_id:
                raise ValueError(f"pack {pack.pack_id!r} depends on itself")
            if dep not in by_id:
                raise ValueError(
                    f"pack {pack.pack_id!r} depends on unknown pack {dep!r}")

    _reject_cycles(by_id)


def _reject_cycles(by_id: dict[str, Pack]) -> None:
    """Depth-first walk; a pack still on the stack when reached again is a
    cycle, and the installer would follow it forever."""
    DONE, ON_STACK = 2, 1
    state: dict[str, int] = {}

    def visit(pack_id: str, trail: list[str]) -> None:
        if state.get(pack_id) == DONE:
            return
        if state.get(pack_id) == ON_STACK:
            raise ValueError(
                "pack dependency cycle: " + " -> ".join([*trail, pack_id]))
        state[pack_id] = ON_STACK
        for dep in by_id[pack_id].depends_on:
            visit(dep, [*trail, pack_id])
        state[pack_id] = DONE

    for pack_id in by_id:
        visit(pack_id, [])


def iter_packs() -> tuple[Pack, ...]:
    """Every pack, in catalog order (which is the order the UI lists them)."""
    return CATALOG


def find_pack(pack_id: str) -> Pack | None:
    """The pack with this id, or None. Use for "is this id known?" checks."""
    for pack in CATALOG:
        if pack.pack_id == pack_id:
            return pack
    return None


def get_pack(pack_id: str) -> Pack:
    """The pack with this id. Raises ``KeyError`` if it is not in the allowlist."""
    pack = find_pack(pack_id)
    if pack is None:
        raise KeyError(f"unknown pack: {pack_id!r}")
    return pack


def get_item(pack: Pack, item_id: str) -> ModelItem:
    """One item of *pack*. Raises ``KeyError`` if the pack does not have it."""
    for item in pack.items:
        if item.item_id == item_id:
            return item
    raise KeyError(f"pack {pack.pack_id!r} has no item {item_id!r}")


PACK_IDS: frozenset[str] = frozenset(pack.pack_id for pack in CATALOG)

_validate(CATALOG)
