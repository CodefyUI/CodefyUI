from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition
from ...core.stateful_module import StatefulModuleMixin


class EmbeddingNode(StatefulModuleMixin, BaseNode):
    NODE_NAME = "Embedding"
    CATEGORY = "Utility"
    DESCRIPTION = (
        "Learnable embedding lookup (wraps nn.Embedding). Maps integer indices "
        "to vectors from a trainable weight matrix $W$ — conceptually "
        "$E[i] = W[i, :]$. For pre-trained word vectors (GloVe, etc.) use the "
        "`WordVector` node in the LLM category instead."
    )

    structural_params = ("num_embeddings", "embedding_dim", "padding_idx")

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Input tensor of integer indices (LongTensor)"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Embedding output tensor"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="num_embeddings", param_type=ParamType.INT, default=10000, description="Size of the vocabulary"),
            ParamDefinition(name="embedding_dim", param_type=ParamType.INT, default=256, description="Dimension of each embedding vector"),
            ParamDefinition(name="padding_idx", param_type=ParamType.INT, default=-1, description="Index for padding token (-1 for none)"),
        ]

    def _normalise_for_hash(self, params: dict[str, Any]) -> dict[str, Any]:
        # ``padding_idx < 0`` is the UI sentinel for "no padding". Collapse it
        # to None so different negative values don't produce different hashes.
        out = dict(params)
        pi = out.get("padding_idx")
        if isinstance(pi, (int, float)) and pi < 0:
            out["padding_idx"] = None
        return out

    def build_module(self, params: dict[str, Any]) -> Any:
        import torch.nn as nn
        kwargs: dict[str, Any] = {
            "num_embeddings": params.get("num_embeddings", 10000),
            "embedding_dim": params.get("embedding_dim", 256),
        }
        padding_idx = params.get("padding_idx", -1)
        if padding_idx is not None and padding_idx >= 0:
            kwargs["padding_idx"] = padding_idx
        return nn.Embedding(**kwargs)

    def execute(self, inputs: dict[str, Any], params: dict[str, Any], *, context: Any = None) -> dict[str, Any]:
        tensor = inputs["tensor"]
        emb = self.get_or_build_module(context, params)
        output = emb(tensor)
        result: dict[str, Any] = {"tensor": output}

        if context is not None and getattr(context, "verbose", False):
            from ...core.step_trace import StepRecorder
            recorder = StepRecorder()
            recorder.record(
                "indices",
                "Input integer indices into the embedding table.",
                indices=tensor,
            )
            recorder.record(
                "embedding_table",
                "Learnable embedding matrix $W$ of shape (num_embeddings, embedding_dim).",
                scalars={
                    "num_embeddings": float(params.get("num_embeddings", 10000)),
                    "embedding_dim": float(params.get("embedding_dim", 256)),
                },
                weight=emb.weight,
            )
            recorder.record(
                "lookup",
                "Gather rows: output[i] = $W$[indices[i], :].",
                output=output,
            )
            result["__steps__"] = recorder.steps

        return result
