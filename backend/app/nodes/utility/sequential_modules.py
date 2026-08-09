"""Sequential-compatible wrappers for modules ``nn.Sequential`` can't chain.

Some torch layers do not fit a plain feed-forward chain: ``nn.LSTM`` returns
``(output, hidden)``, ``nn.MultiheadAttention`` wants three tensors, and there
is no stock layer for "reshape" or "take the CLS token". Each class here wraps
one of those into the single-tensor-in, single-tensor-out shape the layer
editor's graph builder assumes.

**Why these are here rather than in ``sequential_node.py`` (#283).** They used
to be *function-local* classes, defined inside a ``__new__`` on a private
wrapper in that module and all confusingly named ``Mod``. That made
``ModelSaver(save_mode="full_model")`` impossible for any graph using one:

    AttributeError: Can't pickle local object 'Reshape.__new__.<locals>.Mod'

Pickle stores a class by NAME -- module plus qualified name -- and a class
created inside a function has no name anyone can import it back by. So the
mode failed by construction on exactly these layers, with an internal message
naming an implementation detail, which is the same complaint #222 made about
the loader.

The closure turned out to be incidental. Nothing was captured from the
enclosing scope: every one of them already took its configuration through
``__init__`` arguments, and the only enclosing name they used was ``nn``
itself. The classes were nested to keep ``import torch`` off
``sequential_node``'s import path, which matters -- torch is seconds of
startup cost and the node registry imports every node module to read its
metadata. A separate module preserves that exactly: ``torch`` is imported at
the top HERE, and ``_build_layer`` imports this module inside the function
body, so nothing pays for torch until a model is actually built. That is the
same arrangement ``graph_model`` already uses.

One thing this does NOT buy, and it is worth being clear about: a saved file
containing these classes still will not come back through
``ModelLoader(load_mode="full_model")``. That path reads under torch's
restricted unpickler and widens it to ``torch.nn``'s own layer classes only
(see ``torch_nn_layer_globals``), and these are not torch's -- neither is
``GraphModelModule``, which every layer-editor model is. What changes is that
the SAVE now succeeds and produces a valid file, loadable by plain
``torch.load(..., weights_only=False)`` outside CodefyUI, instead of dying on
an unpicklable local class. ``state_dict`` remains the round-trip that works.

Attribute names are load-bearing: they are the ``state_dict`` key prefixes
(``encoder.``, ``lstm.``, ``attn.`` ...). Renaming one silently invalidates
every checkpoint saved before the rename.
"""

from __future__ import annotations

import torch.nn as nn


class Reshape(nn.Module):
    """Reshape a tensor, excluding the batch dimension.

    *shape* is the comma-separated spelling the layer editor stores, e.g.
    ``"64,7,7"``. Parsed here rather than by the caller so the module keeps
    the exact configuration it was given.
    """

    def __init__(self, shape: str):
        super().__init__()
        self._shape = [int(d) for d in shape.split(",")]

    def forward(self, x):
        return x.view(x.size(0), *self._shape)


class SelectIndex(nn.Module):
    """Select a single index along a dimension (e.g. the CLS token)."""

    def __init__(self, dim: int = 1, index: int = 0):
        super().__init__()
        self._dim = dim
        self._index = index

    def forward(self, x):
        return x.select(self._dim, self._index)


class TransformerEncoderBlock(nn.Module):
    """``nn.TransformerEncoder``, batch-first, as a one-in/one-out layer."""

    def __init__(self, d_model: int, nhead: int, num_layers: int = 1, dim_feedforward: int = 2048):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        return self.encoder(x)


class TransformerDecoderBlock(nn.Module):
    """``nn.TransformerDecoder`` in self-attention mode (memory is the input)."""

    def __init__(self, d_model: int, nhead: int, num_layers: int = 1, dim_feedforward: int = 2048):
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)

    def forward(self, x):
        return self.decoder(x, x)


class LSTMBlock(nn.Module):
    """``nn.LSTM`` returning only the output tensor (drops the hidden state)."""

    def __init__(self, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(**kwargs)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out


class GRUBlock(nn.Module):
    """``nn.GRU`` returning only the output tensor (drops the hidden state)."""

    def __init__(self, **kwargs):
        super().__init__()
        self.gru = nn.GRU(**kwargs)

    def forward(self, x):
        out, _ = self.gru(x)
        return out


class MultiHeadAttentionBlock(nn.Module):
    """``nn.MultiheadAttention`` in self-attention mode, weights discarded."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return out
