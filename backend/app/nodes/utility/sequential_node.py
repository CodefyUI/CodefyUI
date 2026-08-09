"""
SequentialModel node — the bridge between layer nodes (TENSOR world) and
training nodes (MODEL world).

It reads a JSON layer specification and builds a real nn.Sequential that
can be passed to Optimizer and TrainingLoop.  Users describe the architecture
with familiar parameters; no Python coding required.

**Why this node is stateful (#253).** It OWNS trainable parameters, and
``TrainingLoop`` mutates them in place. That breaks the cache's
``f(params, upstream) -> output`` assumption exactly the way it breaks for
every layer node, so it carries :class:`StatefulModuleMixin` like they do --
which is also what makes ``Persist weights between runs`` and ``Reset all
weights now`` (Settings > Training Behavior) mean anything here. Before
#253 this node was a cacheable ROOT with no inputs, so nothing could ever
invalidate it: run 2 was handed run 1's already-trained module straight out
of ``ExecutionCache``, and with ``TrainingLoop`` cacheable too, no training
happened at all. ``DiffusionUNet`` is the same shape done right and was the
model for this.
"""

import logging
from typing import TYPE_CHECKING, Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition
from ...core.stateful_module import StatefulModuleMixin

if TYPE_CHECKING:
    # `torch` is imported lazily inside the functions below (startup cost);
    # this makes the "torch.nn.Module" annotations resolvable to a type
    # checker without paying that cost at import time.
    import torch

logger = logging.getLogger(__name__)


# ── Supported layer builders ────────────────────────────────────

def _build_layer(cfg: dict) -> "torch.nn.Module":
    """Convert a single layer dict into an nn.Module."""
    import torch.nn as nn

    # The Sequential-compatible wrappers live in their own module so they are
    # ordinary module-scope classes that pickle can name -- they used to be
    # defined inside this function's callers and could not be saved at all
    # (#283). Imported here, not at the top of the file, because that module
    # imports torch eagerly and the node registry imports THIS file just to
    # read node metadata.
    from .sequential_modules import (
        GRUBlock,
        LSTMBlock,
        MultiHeadAttentionBlock,
        Reshape,
        SelectIndex,
        TransformerDecoderBlock,
        TransformerEncoderBlock,
    )

    t = cfg.get("type", "")
    p = {k: v for k, v in cfg.items() if k != "type"}

    builders: dict[str, type] = {
        "Conv2d": nn.Conv2d,
        "Conv1d": nn.Conv1d,
        "ConvTranspose2d": nn.ConvTranspose2d,
        "BatchNorm2d": nn.BatchNorm2d,
        "BatchNorm1d": nn.BatchNorm1d,
        "LayerNorm": nn.LayerNorm,
        "GroupNorm": nn.GroupNorm,
        "InstanceNorm2d": nn.InstanceNorm2d,
        "MaxPool2d": nn.MaxPool2d,
        "AvgPool2d": nn.AvgPool2d,
        "AdaptiveAvgPool2d": nn.AdaptiveAvgPool2d,
        "Dropout": nn.Dropout,
        "Linear": nn.Linear,
        "Flatten": nn.Flatten,
        "Embedding": nn.Embedding,
    }

    # Sequential-compatible wrappers for complex modules
    wrappers: dict[str, type] = {
        "Reshape": Reshape,
        "SelectIndex": SelectIndex,
        "TransformerEncoder": TransformerEncoderBlock,
        "TransformerDecoder": TransformerDecoderBlock,
        "LSTM": LSTMBlock,
        "GRU": GRUBlock,
        "MultiHeadAttention": MultiHeadAttentionBlock,
    }

    # Activation functions
    activations: dict[str, nn.Module] = {
        "ReLU": nn.ReLU(inplace=True),
        "GELU": nn.GELU(),
        "Sigmoid": nn.Sigmoid(),
        "Tanh": nn.Tanh(),
        "LeakyReLU": nn.LeakyReLU(inplace=True),
        "ELU": nn.ELU(inplace=True),
        "SiLU": nn.SiLU(inplace=True),
        "Mish": nn.Mish(inplace=True),
        "SELU": nn.SELU(inplace=True),
        "PReLU": nn.PReLU(),
        "Hardswish": nn.Hardswish(inplace=True),
        "Softmax": nn.Softmax(dim=-1),
    }
    if t in activations:
        return activations[t]

    # Try wrapper modules first, then standard builders
    wcls = wrappers.get(t)
    if wcls is not None:
        return wcls(**p)

    cls = builders.get(t)
    if cls is None:
        raise ValueError(f"SequentialModel: unknown layer type '{t}'")
    return cls(**p)


class SequentialModelNode(StatefulModuleMixin, BaseNode):
    NODE_NAME = "SequentialModel"
    CATEGORY = "Training"
    DESCRIPTION = (
        "Build an nn.Sequential model visually. "
        "Double-click to open the architecture editor and drag-and-drop layers. "
        "Outputs a MODEL that can be connected to Optimizer and TrainingLoop."
    )

    #: The whole architecture lives in one param, so the layer spec IS the
    #: structure. Edit the architecture and the hash changes, which drops the
    #: persisted weights -- correct, since they no longer fit the new shapes.
    #: Changing anything else about the run (learning rate, epochs, batch
    #: size) leaves the hash alone and therefore keeps the weights, which is
    #: the point of the setting.
    structural_params = ("layers",)

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []  # no tensor input — purely declarative

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL, description="Built nn.Sequential model"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="layers",
                param_type=ParamType.STRING,
                default=(
                    '{"version":2,'
                    '"nodes":['
                    '{"id":"in","type":"Input","ports":[{"id":"p_x","name":"x"}]},'
                    '{"id":"c1","type":"Conv2d","params":{"in_channels":1,"out_channels":32,"kernel_size":3,"padding":1}},'
                    '{"id":"r1","type":"ReLU"},'
                    '{"id":"p1","type":"MaxPool2d","params":{"kernel_size":2,"stride":2}},'
                    '{"id":"c2","type":"Conv2d","params":{"in_channels":32,"out_channels":64,"kernel_size":3,"padding":1}},'
                    '{"id":"r2","type":"ReLU"},'
                    '{"id":"p2","type":"MaxPool2d","params":{"kernel_size":2,"stride":2}},'
                    '{"id":"f","type":"Flatten"},'
                    '{"id":"l1","type":"Linear","params":{"in_features":3136,"out_features":128}},'
                    '{"id":"r3","type":"ReLU"},'
                    '{"id":"l2","type":"Linear","params":{"in_features":128,"out_features":10}},'
                    '{"id":"out","type":"Output","ports":[{"id":"p_y","name":"y"}]}'
                    '],'
                    '"edges":['
                    '{"id":"e1","source":"in","sourceHandle":"p_x","target":"c1"},'
                    '{"id":"e2","source":"c1","target":"r1"},'
                    '{"id":"e3","source":"r1","target":"p1"},'
                    '{"id":"e4","source":"p1","target":"c2"},'
                    '{"id":"e5","source":"c2","target":"r2"},'
                    '{"id":"e6","source":"r2","target":"p2"},'
                    '{"id":"e7","source":"p2","target":"f"},'
                    '{"id":"e8","source":"f","target":"l1"},'
                    '{"id":"e9","source":"l1","target":"r3"},'
                    '{"id":"e10","source":"r3","target":"l2"},'
                    '{"id":"e11","source":"l2","target":"out","targetHandle":"p_y"}'
                    ']}'
                ),
                description="DAG JSON (v2 schema) describing the model graph.",
            ),
        ]

    def build_module(self, params: dict[str, Any]) -> "torch.nn.Module":
        import json

        from .graph_model import build_graph_model

        return build_graph_model(json.loads(params.get("layers", "{}")))

    def _weights_are_persisted(
        self, context: Any, params: dict[str, Any],
    ) -> bool:
        """True when THIS call will be handed weights an earlier run trained.

        Asked before :meth:`get_or_build_module`, because that call is what
        makes the answer stop being true. A miss inside the store drops the
        node's other structure hashes, so "my hash is already there" is
        exactly "the module I am about to get was built by an earlier run".

        Never raises -- an advisory log line must not be able to fail the
        node that produces it.
        """
        store = getattr(context, "node_state_store", None)
        if (
            context is None
            or not getattr(context, "weights_persistent", False)
            or store is None
            or not getattr(context, "current_node_id", "")
        ):
            return False
        try:
            wanted = self._structure_hash(params)
            return any(
                key[2] == wanted
                for key in store.keys_for_node(
                    context.graph_id, context.current_node_id)
            )
        except Exception:  # noqa: BLE001 - advisory only
            logger.debug("could not inspect persisted node state", exc_info=True)
            return False

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        resumed = self._weights_are_persisted(context, params)
        model = self.get_or_build_module(context, params)

        total = sum(p.numel() for p in model.parameters())
        layer_count = len(model.layers)
        logger.info("Built graph model with %d layers, %s parameters", layer_count, f"{total:,}")

        # #253: say which of the two happened. Whether a rerun continues
        # training or restarts it is a real choice with no obviously right
        # default, so the node does not make it silently -- the setting
        # decides and this line reports the decision. ``__log__`` is the one
        # result key the canvas Log tab renders, and dunder keys are filtered
        # out of recorded outputs and port summaries, so this adds a log line
        # and nothing else (same mechanism as TrainingLoop's schedule note).
        if resumed:
            note = (
                f"Continuing from weights an earlier run left behind "
                f"({total:,} parameters). Training resumes from those weights "
                f"rather than starting over. To start from a fresh "
                f"initialisation, use Settings > Training Behavior > "
                f"'Reset all weights now', or turn off 'Persist weights "
                f"between runs'."
            )
        else:
            note = (
                f"Fresh initialisation ({total:,} parameters across "
                f"{layer_count} layers)."
            )

        return {"model": model, "__log__": note}
