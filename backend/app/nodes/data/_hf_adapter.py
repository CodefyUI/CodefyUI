"""Internal helper: wraps a HuggingFace `datasets.Dataset` as a torch Dataset.

This is a private module (leading underscore). It is reusable from any
HuggingFace-backed node — currently `HuggingFaceDatasetNode`, and a future
`HuggingFaceTextDataset` node could share the same shape with a different
column convention.
"""

from __future__ import annotations

from typing import Any, Callable

from torch.utils.data import Dataset


class HFTorchTextDataset(Dataset):
    """Adapt a HuggingFace `datasets.Dataset` to rows of raw text.

    The text sibling of `HFTorchImageDataset` (the docstring above always
    promised one): `__getitem__` returns one python string, which is what
    the LM packing node consumes. No transform hook — text preprocessing
    belongs to the tokenizer.
    """

    def __init__(self, hf_dataset: Any, text_column: str) -> None:
        self._ds = hf_dataset
        self._text_col = text_column

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> str:
        return str(self._ds[idx][self._text_col])


class LocalTextListDataset(Dataset):
    """A list of strings as a Dataset — the local-file corpus shape."""

    def __init__(self, rows: list[str]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> str:
        return self._rows[idx]


class HFTorchImageDataset(Dataset):
    """Adapt a HuggingFace `datasets.Dataset` to the torchvision Dataset convention.

    The class deliberately mirrors how torchvision's built-in datasets behave:
    `transform` is a public, mutable attribute that downstream nodes
    (e.g. `TransformNode`) may replace at any time.
    """

    def __init__(
        self,
        hf_dataset: Any,
        image_column: str,
        label_column: str,
        transform: Callable[[Any], Any] | None = None,
    ) -> None:
        self._ds = hf_dataset
        self._image_col = image_column
        self._label_col = label_column
        self.transform = transform

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        row = self._ds[idx]
        image = row[self._image_col]
        label = row[self._label_col]
        if self.transform is not None:
            image = self.transform(image)
        return image, label
