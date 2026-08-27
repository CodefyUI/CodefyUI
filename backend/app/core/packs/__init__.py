"""Optional pack support (Package Center). See catalog.py for the allowlist.

A pack is a curated bundle of pip packages and model files that a stock
CodefyUI install deliberately does NOT ship: the base install stays small
enough to hand to a classroom, and the four hundred megabytes a sentence
embedder needs arrive only when a lesson asks for them.

This package stays free of imports from ``app.api`` so node code and the
routes can both depend on it.
"""


class PackMissingError(RuntimeError):
    """A node needs an optional pack that is not installed.

    The message always ends with ``(pack=<id>)`` so the frontend can extract
    the id; ``pack_id`` carries it for Python callers.
    """

    def __init__(self, pack_id: str, message: str):
        self.pack_id = pack_id
        super().__init__(f"{message} (pack={pack_id})")
