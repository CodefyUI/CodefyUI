"""Torch dataset behind LMTokenizedDataset.

Module-scope for the same reason as ``_lm_modules.py`` / (#283): pickle finds
classes by import path — a DataLoader with worker processes, or a future
serialization of the graph value, must be able to reimport this class.
``torch`` is imported at the top HERE and the node module imports this one
lazily inside ``execute``.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class PackedLMDataset(Dataset):
    """Fixed-length next-token blocks cut from one packed token stream.

    ``blocks`` is an int32 tensor of shape ``(num_blocks, seq_len + 1)``
    (int32 halves resident memory versus int64 for ~100M-token corpora; ids
    stay far below 2**31). ``__getitem__`` widens to the int64 the embedding
    lookup wants and returns ``(block[:-1], block[1:])`` — inputs and their
    one-step-shifted labels.
    """

    def __init__(self, blocks: torch.Tensor) -> None:
        if blocks.dim() != 2:
            raise ValueError(
                f"PackedLMDataset expects (num_blocks, seq_len+1), got {tuple(blocks.shape)}"
            )
        self.blocks = blocks

    @property
    def seq_len(self) -> int:
        return int(self.blocks.shape[1]) - 1

    def __len__(self) -> int:
        return int(self.blocks.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        block = self.blocks[idx].long()
        return block[:-1].contiguous(), block[1:].contiguous()
