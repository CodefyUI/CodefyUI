"""VectorIndex -- the [N, D] matrix plus the N chunk texts that IS the "database".

``VectorStore`` produces exactly one object and ``Retriever`` consumes exactly
one object, and this is it: the embeddings of every chunk stacked into a
matrix, the chunk strings themselves, and whatever metadata the chunker
attached. Searching it is a single matrix multiply followed by ``topk`` --
there is no index structure, no approximate nearest neighbour, nothing to
build. At teaching scale (a few hundred chunks of a few hundred dimensions)
brute force is not a compromise; it is the honest implementation, and it is
the one a learner can read.

Keeping it here rather than inside either node is what makes both nodes thin:
``VectorStore.execute`` is a call to :func:`build_index`, ``Retriever.execute``
is a search over this class, and a later ``VectorStoreFile`` node
is :meth:`save` / :meth:`load` with a path widget. ``search`` is itself
:meth:`VectorIndex.scores` plus :meth:`VectorIndex.top_hits`, and the split
is public because ``Retriever`` wants both halves out of ONE matrix multiply:
its verbose trace shows the whole ``[Q, N]``, because the chunks that lost
are half of what makes a top score readable, and its outputs are the winners.

Four decisions worth knowing about.

**The error messages name the wiring mistake, not the shapes.** The two ways
a RAG graph goes wrong are embedding the question with a different model than
the documents, and feeding chunks from one ``TextChunker`` run and embeddings
from another. Both surface here as a shape or length mismatch, and a message
that only reports the numbers ("expected 768, got 384") leaves the learner to
work out which of their edges is wrong. Both messages therefore end in the
fix.

**``normalize`` is stored, not just applied.** Unit rows make a cosine search
a plain ``q @ V.T``, so it is worth doing once at build time rather than on
every query. But ``scores`` normalises anyway when the metric is cosine: an
index assembled by hand, or loaded from a file somebody wrote with
``normalize=False``, must still answer cosine questions correctly. Normalising
a unit vector is a no-op, so the defensive pass costs one kernel launch and
buys the invariant. The flag is what the UI reports, and what tells a reader
of a saved file how the rows got there.

**``normalize`` is ignored for the dot metric.** Scaling every row to unit
length is precisely the information ``dot`` was chosen to keep, so a graph
asking for ``dot`` with the default ``normalize=True`` gets its raw vectors
and ``normalized=False`` -- what the parameter's help text promises. The
alternative (obeying both) would silently make ``dot`` identical to ``cosine``
and leave nothing in the UI to say so.

**The file format holds the strings as one JSON blob.** ``np.savez`` stores
arrays, and a list of variable-length UTF-8 chunk texts is not one -- a numpy
unicode array pads every entry to the longest, which for one long chunk among
hundreds is most of the file. ``chunks`` and ``metadata`` therefore travel as
a single JSON document encoded to UTF-8 bytes and stored as ``uint8``, which
round-trips CJK exactly and lets the archive load with ``allow_pickle=False``
-- the same shape as the GloVe table's vocabulary blob in ``_glove.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

METRICS = ("cosine", "dot")

#: What a query that is not a vector at all is answered with. One string
#: because ``scores`` raises it from two places (nothing wired, and something
#: wired that no tensor can be made of) and both are the same mistake seen
#: from different sides. Named for ``Retriever`` because that is the only
#: node a learner reaches this class through.
_QUERY_NOT_A_TENSOR = (
    "Retriever: the query input must be the embedding of the question (a "
    "tensor) -- connect TextEmbedding.embeddings"
)

# ``np.savez`` member names. Changing one changes the file format, so they are
# named once here rather than spelled inline in save and again in load.
_KEY_VECTORS = "vectors"
_KEY_METRIC = "metric"
_KEY_NORMALIZED = "normalized"
_KEY_TEXTS = "texts_json"


@dataclass(eq=False)
class VectorIndex:
    """An in-memory corpus: ``vectors[i]`` is the embedding of ``chunks[i]``.

    ``vectors`` is CPU float32 ``[N, D]`` -- unit rows when ``normalized`` is
    True. ``chunks`` and ``metadata`` are both length N; a chunk with no
    metadata carries ``{}`` rather than being absent, so ``metadata[i]`` is
    always safe to index.

    Every field is required, deliberately. A default for ``metadata`` would
    hand out indexes whose three lists disagree about N, and :func:`build_index`
    exists precisely so nobody has to assemble one by hand.

    ``eq=False`` because a field is a tensor. The generated ``__eq__`` compares
    the fields as a tuple, and ``tensor == tensor`` gives an elementwise tensor
    whose truth value RAISES -- so ``index_a == index_b`` would blow up inside
    anything that merely checks whether a port's value changed, and defining
    ``__eq__`` at all would set ``__hash__`` to None and make the object
    unusable as a dict key or in a set. Identity is the right answer anyway:
    one build produces one index, and two indexes are the same one or they are
    not.
    """

    vectors: torch.Tensor
    chunks: list[str]
    metadata: list[dict[str, Any]]
    metric: str
    normalized: bool

    def __len__(self) -> int:
        return int(self.vectors.shape[0]) if self.vectors.ndim else 0

    def __repr__(self) -> str:
        """One line, because this is what the Inspector shows.

        ``output_entries._summarize_single`` has no case for this class, so it
        falls through to ``{"type": ..., "repr": repr(value)[:200]}``. The
        dataclass-generated repr would put the whole matrix and every chunk
        into that field and the learner would see the first 200 characters of
        a float dump. These three numbers are the ones worth reading.
        """
        return f"VectorIndex(size={len(self)}, dim={self.dim}, metric={self.metric})"

    @property
    def dim(self) -> int:
        """Vector width D. 0 for a degenerate index, so ``repr`` never raises."""
        return int(self.vectors.shape[1]) if self.vectors.ndim >= 2 else 0

    def scores(self, query: torch.Tensor) -> torch.Tensor:
        """Similarity of every chunk to every query row: the whole ``[Q, N]``.

        The search itself, before anything is thrown away. ``Retriever``
        shows this matrix in its verbose trace for a small corpus, because
        the chunks that did NOT win are half of what a learner is looking at
        -- a top hit of 0.71 means one thing when the runners-up are at 0.68
        and another when they are at 0.20.

        Every coercion and every check lives here rather than in
        :meth:`search`, so the two cannot disagree about what a query is: a
        ``[D]`` row becomes one question, the dtype and device are settled
        (CPU float32, matching ``vectors``, which is why the nodes can leave
        ``align_inputs`` off), and a query from the wrong encoder is refused
        with the message that names the fix.

        A query that is not a tensor at all gets a message about the WIRING
        rather than torch's. An unwired port is ``None`` and
        ``torch.as_tensor(None)`` raises "Could not infer dtype of
        NoneType", which is a true sentence about a node the learner did not
        write and says nothing about the edge they forgot to draw.
        """
        try:
            q = torch.as_tensor(query).detach().float().cpu()
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(_QUERY_NOT_A_TENSOR) from exc
        if q.ndim == 1:
            q = q.unsqueeze(0)
        if q.ndim != 2:
            raise ValueError(
                f"query must be a [D] or [Q, D] tensor, got shape {list(q.shape)} "
                "-- pass the embedding of one question, or of a batch of them"
            )
        if q.shape[1] != self.dim:
            raise ValueError(
                f"query dimension {q.shape[1]} does not match the index dimension "
                f"{self.dim} -- embed the question with the same model the documents "
                "were embedded with"
            )

        vectors = self.vectors
        if self.metric == "cosine":
            # Both sides, every time: see the module docstring. ``eps`` keeps a
            # zero row (an empty chunk that embedded to nothing) from dividing
            # by zero and poisoning the whole score column with NaN.
            q = torch.nn.functional.normalize(q, dim=1, eps=1e-12)
            vectors = torch.nn.functional.normalize(vectors, dim=1, eps=1e-12)
        return q @ vectors.T  # [Q, N]

    def top_hits(self, matrix: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The best *top_k* columns of a ``[Q, N]`` matrix from :meth:`scores`.

        The second half of :meth:`search`, public for the same reason
        ``scores`` is: ``Retriever`` needs BOTH the whole matrix (for its
        verbose trace) and the winners (for its outputs), and calling the two
        halves gets it both from one matrix multiply. Recomputing the matrix
        for the trace would be a second matmul that is allowed to disagree
        with the results printed under it.

        *top_k* is clamped to ``[1, N]``: asking for three chunks from a
        two-chunk corpus is a reasonable thing for a default parameter to do,
        and ``torch.topk`` would raise instead.
        """
        k = max(1, min(int(top_k), len(self)))
        return torch.topk(matrix, k=k, dim=1)

    def search(self, query: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Score every chunk against *query* and return the best *top_k*.

        Returns ``(scores, indices)``, both ``[Q, k]`` and both sorted best
        first; ``indices`` is int64 and indexes into ``chunks``. A ``[D]``
        query counts as one question and comes back as ``[1, k]``.
        """
        return self.top_hits(self.scores(query), top_k)

    def save(self, path: Path) -> None:
        """Write the whole index to *path* as an ``.npz`` archive.

        ``np.savez`` is handed an open FILE rather than the path: given a name
        that does not already end in ``.npz`` it appends one, and a caller who
        asked for ``corpus`` would find ``corpus.npz`` with nothing at the name
        they were told about. The same trap ``_glove._save_npz`` documents.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"chunks": self.chunks, "metadata": self.metadata},
            ensure_ascii=False,
        ).encode("utf-8")
        with open(path, "wb") as handle:
            np.savez(
                handle,
                **{
                    _KEY_VECTORS: self.vectors.detach().cpu().numpy(),
                    _KEY_METRIC: np.frombuffer(self.metric.encode("utf-8"), dtype=np.uint8),
                    _KEY_NORMALIZED: np.bool_(self.normalized),
                    _KEY_TEXTS: np.frombuffer(payload, dtype=np.uint8),
                },
            )

    @classmethod
    def load(cls, path: Path) -> "VectorIndex":
        """Read back what :meth:`save` wrote.

        ``allow_pickle=False`` is the point of the JSON blob: nothing in the
        archive can execute, so an index file from somewhere else is data and
        not code.

        The two checks at the end are the same ones :func:`build_index`
        applies at the other door into this class. A file is not a value some
        node just produced -- it can be truncated, hand-edited, or written by
        something that is not this class -- and both faults are silent
        otherwise: a short ``chunks`` list raises ``IndexError`` from inside
        ``Retriever``, which points the learner at the wrong node, and an
        unknown metric falls through to the dot branch and answers a question
        nobody asked.
        """
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            # Read INSIDE the context -- an NpzFile fetches a member when it is
            # asked for one, and the zip closes on the way out.
            matrix = np.asarray(data[_KEY_VECTORS], dtype=np.float32)
            metric = data[_KEY_METRIC].tobytes().decode("utf-8")
            normalized = bool(data[_KEY_NORMALIZED])
            payload = data[_KEY_TEXTS].tobytes().decode("utf-8")

        texts = json.loads(payload)
        chunks = [str(c) for c in texts.get("chunks", [])]
        metadata = [dict(m) if isinstance(m, dict) else {}
                    for m in texts.get("metadata", [])]
        rows = int(matrix.shape[0]) if matrix.ndim >= 1 else 0
        # Only the list that is actually wrong: "3 vectors but 3 chunks and 1
        # metadata entry" invites the reader to check the count that already
        # agrees, and the plural of "1 entry" is the difference between a
        # message that was written and one that was concatenated.
        disagree = []
        if len(chunks) != rows:
            disagree.append(
                f"{len(chunks)} {'chunk' if len(chunks) == 1 else 'chunks'}")
        if len(metadata) != rows:
            disagree.append(
                f"{len(metadata)} metadata "
                f"{'entry' if len(metadata) == 1 else 'entries'}")
        if disagree:
            raise ValueError(
                f"{path} holds {rows} {'vector' if rows == 1 else 'vectors'} "
                f"but {' and '.join(disagree)} -- the archive is truncated or "
                "was not written by VectorIndex.save"
            )
        if metric not in METRICS:
            raise ValueError(
                f"{path} declares metric {metric!r}, which must be one of "
                f"{', '.join(METRICS)} -- the archive was not written by "
                "VectorIndex.save"
            )

        return cls(
            # ``torch.tensor`` rather than ``from_numpy``: it copies, so the
            # index does not alias a buffer numpy may have handed out read-only.
            vectors=torch.tensor(matrix, dtype=torch.float32),
            chunks=chunks,
            metadata=metadata,
            metric=metric,
            normalized=normalized,
        )


def build_index(
    embeddings: Any,
    chunks: Any,
    metadata: Any = None,
    *,
    metric: str = "cosine",
    normalize: bool = True,
) -> VectorIndex:
    """Coerce and validate what ``VectorStore`` was wired, or say what is wrong.

    *embeddings* is anything ``torch.as_tensor`` accepts -- a ``[N, D]`` tensor
    from ``TextEmbedding``, or a ``[D]`` row for a one-chunk corpus. *chunks*
    must be N strings and *metadata* either N dicts or nothing at all.
    """
    if metric not in METRICS:
        raise ValueError(
            f"VectorStore metric must be one of {', '.join(METRICS)}, got {metric!r}"
        )

    vectors = torch.as_tensor(embeddings).detach().float().cpu()
    if vectors.ndim == 1:
        vectors = vectors.unsqueeze(0)
    if vectors.ndim != 2:
        raise ValueError(
            f"VectorStore expected an [N, D] embedding matrix, got shape "
            f"{list(vectors.shape)} -- connect TextEmbedding.embeddings"
        )

    n = int(vectors.shape[0])
    # Checked before the length comparison: with no embeddings at all the
    # chunk count is a symptom, and "0 embeddings but 12 chunks" would point
    # at the chunker when the encoder is what produced nothing.
    if n == 0:
        raise ValueError(
            "VectorStore got an empty embedding matrix -- there is nothing to index; "
            "check that DocumentLoader found files and TextChunker produced chunks"
        )

    chunk_list = [str(c) for c in chunks]
    if len(chunk_list) != n:
        raise ValueError(
            f"VectorStore got {n} embeddings but {len(chunk_list)} chunks; "
            "both must come from the same TextChunker output"
        )

    meta_list = [] if metadata is None else list(metadata)
    if not meta_list:
        meta_list = [{} for _ in range(n)]
    elif len(meta_list) != n:
        raise ValueError(
            f"VectorStore got {n} chunks but {len(meta_list)} metadata entries; "
            "leave metadata unconnected or wire it from the same TextChunker"
        )
    # A non-dict entry becomes {} rather than raising: metadata is decoration
    # (Retriever falls back to "?" for a missing source), and losing a label is
    # not worth failing a run that has already paid for the embeddings.
    meta_list = [dict(m) if isinstance(m, dict) else {} for m in meta_list]

    unit = bool(normalize) and metric == "cosine"
    if unit:
        vectors = torch.nn.functional.normalize(vectors, dim=1, eps=1e-12)

    return VectorIndex(
        vectors=vectors,
        chunks=chunk_list,
        metadata=meta_list,
        metric=metric,
        normalized=unit,
    )
