"""VectorStore -- the chunk embeddings and their texts, packed into an index.

The fourth node of the RAG chain::

    DocumentLoader -> TextChunker -> TextEmbedding -> VectorStore -> Retriever

**Why this is a node at all.** Everything it does is one call to
:func:`~app.nodes.llm._vector_index.build_index`, and that is the point: the
"vector database" a RAG diagram draws as a cylinder is, at teaching scale, an
``[N, D]`` matrix and a list of N strings. Making it a node with its own port
puts that object on the canvas where a learner can click it and read
``VectorIndex(size=37, dim=384, metric=cosine)`` -- three numbers that say
exactly what a database everyone talks about actually contains.

**Why the index port is ANY.** A ``VectorIndex`` is not a tensor, and typing
the port ``TENSOR`` to make the wire look familiar would let the canvas
connect it to every node that takes one; each of those would then fail
somewhere inside torch instead of here. ANY is the honest declaration -- and
``Retriever`` guards it with an ``isinstance`` check on the way in, which is
the precedent ``TextGenerate`` set for its duck-typed ``tokenizer`` port.

**Why ``align_inputs`` is off.** ``build_index`` puts the matrix on the CPU
and keeps it there, because a corpus of vectors is data at rest, not a
computation. Letting the engine align the inputs would move the embeddings
onto the run device on the way in for this node to move them straight back --
the same reasoning ``TrainTestSplit`` gives for sklearn.

**What the memory budget does NOT see.** The index leaves through an ANY
port, and ``core.memory_budget._walk`` measures tensors, modules and the
usual containers but does not descend into arbitrary objects, so the stored
matrix is not counted against the cache's byte budget. That is a real
under-count, and it is acceptable at teaching scale: a few hundred chunks of
384 dimensions is under a megabyte. It would stop being acceptable for a
corpus large enough to matter, which is the point at which this node should
be storing a file instead of a field.
"""

from __future__ import annotations

import logging
from typing import Any

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)
from ...core.step_trace import StepRecorder
from ._vector_index import METRICS, build_index

logger = logging.getLogger(__name__)


class VectorStoreNode(BaseNode):
    NODE_NAME = "VectorStore"
    CATEGORY = "LLM"
    DESCRIPTION = (
        "Pack chunk embeddings and their texts into one searchable index. "
        "This is the 'database' of a RAG system: an [N, D] matrix plus the N "
        "chunk strings, with cosine as the default metric (rows are stored "
        "unit-length so a search is a single matrix multiply). Wire the index "
        "into Retriever. In-memory only; a re-run rebuilds it from the cached "
        "embeddings in a few milliseconds."
    )

    # Stated rather than inherited, like DocumentLoader's and TextChunker's:
    # this node is arithmetic over vectors the encoder has already produced,
    # so it runs on a machine that has downloaded nothing.
    REQUIRES_PACK = None

    # A pure function of its inputs: the same embeddings and the same params
    # build the same index, and nothing is written anywhere. There is no live
    # handle to invalidate either -- the index is immutable once built, and
    # ``Retriever`` only reads it.
    cacheable = True

    # See the module docstring: the index is host-side data at rest.
    align_inputs = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="embeddings",
                data_type=DataType.TENSOR,
                description="[N, D] from TextEmbedding",
            ),
            PortDefinition(
                name="chunks",
                data_type=DataType.LIST,
                description="N strings from TextChunker.chunks",
            ),
            PortDefinition(
                name="metadata",
                data_type=DataType.LIST,
                description="N dicts from TextChunker.metadata",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="index",
                data_type=DataType.ANY,
                description="VectorIndex object; connect to Retriever.index",
            ),
            PortDefinition(
                name="size",
                data_type=DataType.SCALAR,
                description="How many chunks the index holds (N).",
            ),
            PortDefinition(
                name="dim",
                data_type=DataType.SCALAR,
                description="Vector width (D), set by the embedding model.",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="metric",
                param_type=ParamType.SELECT,
                default="cosine",
                options=list(METRICS),
                description=(
                    "cosine ignores vector length and is what sentence "
                    "embeddings are trained for; dot is the raw product, for "
                    "embeddings whose length carries meaning"
                ),
            ),
            ParamDefinition(
                name="normalize",
                param_type=ParamType.BOOL,
                default=True,
                description=(
                    "Store unit-length rows (cosine then equals dot). "
                    "Ignored when metric is dot."
                ),
            ),
        ]

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        progress_callback: Any | None = None,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        metric = str(params.get("metric") or "cosine")
        normalize = bool(params.get("normalize", True))

        # Every check and every mandated message lives in build_index, which
        # is also what a future VectorStoreFile node will call. This node
        # adds no validation of its own: two nodes wording the same wiring
        # mistake differently is how a learner ends up with two theories.
        index = build_index(
            inputs.get("embeddings"),
            inputs.get("chunks"),
            inputs.get("metadata"),
            metric=metric,
            normalize=normalize,
        )

        size = len(index)
        dim = index.dim
        note = f"index of {size} chunks x {dim} dims ({index.metric})"

        result: dict[str, Any] = {
            "index": index,
            "size": size,
            "dim": dim,
            # The one result key the canvas Log tab renders; dunder keys are
            # filtered out of recorded outputs and port summaries.
            "__log__": note,
        }

        if context is not None and getattr(context, "verbose", False):
            recorder = StepRecorder()
            recorder.record(
                "build",
                f"Stack {size} chunk vectors into an [N, D] = [{size}, {dim}] "
                f"matrix and keep the {size} chunk texts beside it. Searching "
                f"it is one {index.metric} product against every row"
                + (" (rows stored unit-length)." if index.normalized
                   else "."),
                scalars={"N": float(size), "D": float(dim)},
            )
            result["__steps__"] = recorder.steps

        logger.info("VectorStore: %s", note)
        return result
