"""Example gallery tags: the main-tag vocabulary, its order, and the tag rules.

The order of :data:`MAIN_TAGS` IS the gallery's order: most practical first,
papers and courses after every practical category. It is defined here and
nowhere else; the frontend reads it from ``GET /api/examples/tags``.

A ``graph.json`` declares where it belongs with a ``gallery`` block::

    "gallery": {"main_tag": "training", "tags": ["vision", "cnn"], "order": 30}

``main_tag`` must be one of :data:`MAIN_TAGS`; ``tags`` are free words
(lowercase, hyphenated, 1-24 characters, at most :data:`MAX_TAGS`).
``family`` sub-groups ``architectures``. An older file that only says
``"section"`` keeps working: a known ``section`` is read as the main tag.
"""

from __future__ import annotations

import re
from typing import Any

#: The controlled main-tag vocabulary, in gallery order.
MAIN_TAGS: tuple[str, ...] = (
    "quickstart",
    "training",
    "llm",
    "data",
    "evaluation",
    "research",
    "deploy",
    "architectures",
    "concepts",
    "papers",
    "courses",
    # A plugin example with no readable main tag is filed here automatically.
    "plugin",
    # A built-in with no readable main tag. The metadata tests keep this empty.
    "other",
)

MAIN_TAG_RANK: dict[str, int] = {tag: index + 1 for index, tag in enumerate(MAIN_TAGS)}

#: Main tags a file may declare. ``plugin`` and ``other`` are fallbacks the
#: server assigns, never values an author picks.
DECLARABLE_MAIN_TAGS: frozenset[str] = frozenset(MAIN_TAGS) - {"plugin", "other"}

MAX_TAGS = 6
TAG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
MAX_TAG_LENGTH = 24


def is_valid_tag(value: Any) -> bool:
    """A free tag: lowercase words joined by hyphens, 1-24 characters."""
    return (isinstance(value, str) and 0 < len(value) <= MAX_TAG_LENGTH
            and TAG_PATTERN.match(value) is not None)


def declared_main_tag(block: Any) -> str | None:
    """The main tag a ``gallery`` block declares, or ``None``.

    ``main_tag`` wins; a file with only the older ``section`` key is read
    through it, so third-party packs written before tags still sort into
    the category they named. Unknown or unreadable values say nothing.
    """
    if not isinstance(block, dict):
        return None
    for key in ("main_tag", "section"):
        value = block.get(key)
        # ``isinstance`` first: ``in`` on a set raises for unhashable values.
        if isinstance(value, str) and value in DECLARABLE_MAIN_TAGS:
            return value
    return None


def declared_tags(block: Any) -> list[str]:
    """The valid free tags of a ``gallery`` block, deduplicated, in file order.

    A bad entry costs that entry only. At most :data:`MAX_TAGS` are kept.
    """
    if not isinstance(block, dict):
        return []
    raw = block.get("tags")
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for value in raw:
        if is_valid_tag(value) and value not in out:
            out.append(value)
        if len(out) == MAX_TAGS:
            break
    return out


def resolve_gallery(data: Any, *, builtin: bool) -> dict[str, Any]:
    """The gallery fields the list serves for one ``graph.json`` payload.

    Always the same keys, read field by field so one bad value costs that
    value alone:

    - ``main_tag``: the declared main tag, else ``plugin`` for a plugin
      example and ``other`` for a built-in.
    - ``tags``: the main tag first, then the declared tags, deduplicated.
    - ``order``: an int (not a bool), else ``None``.
    - ``family``: a string, else ``None``.
    - ``section``: deprecated, equal to ``main_tag``. Kept so older
      frontends and plugins that group on it keep working.
    """
    block = data.get("gallery") if isinstance(data, dict) else None
    main_tag = declared_main_tag(block) or ("other" if builtin else "plugin")
    tags = [main_tag] + [tag for tag in declared_tags(block) if tag != main_tag]
    order = block.get("order") if isinstance(block, dict) else None
    family = block.get("family") if isinstance(block, dict) else None
    return {
        "main_tag": main_tag,
        "tags": tags,
        # ``bool`` is an ``int`` in Python and would sort as 0 or 1.
        "order": order if isinstance(order, int) and not isinstance(order, bool) else None,
        "family": family if isinstance(family, str) else None,
        "section": main_tag,
    }


def sort_key(entry: dict[str, Any]) -> tuple:
    """Gallery order: main-tag rank, then ``order`` (unset last), then name."""
    order = entry.get("order")
    return (
        MAIN_TAG_RANK.get(entry.get("main_tag"), len(MAIN_TAGS) + 1),
        0 if isinstance(order, int) else 1,
        order if isinstance(order, int) else 0,
        str(entry.get("name", "")).casefold(),
        str(entry.get("path", "")),
    )
