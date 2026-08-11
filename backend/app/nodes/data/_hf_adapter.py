"""Internal helper: wraps a HuggingFace `datasets.Dataset` as a torch Dataset.

This is a private module (leading underscore). It is reusable from any
HuggingFace-backed node — `HuggingFaceDatasetNode` for image classification,
and `TextCorpusDatasetNode` (#290) for the text variant this module's docstring
used to anticipate: same shape, one column instead of two, and no transform.
"""

from __future__ import annotations

from typing import Any, Callable

from torch.utils.data import Dataset


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


class HFTorchTextDataset(Dataset):
    """One text column of a HuggingFace `datasets.Dataset`, as raw strings (#290).

    `__getitem__` returns a `str`, NOT a `(data, target)` pair — the rows are a
    corpus, not supervised examples, so this cannot be handed to `DataLoader`
    directly. `LMTokenizedDataset` is what turns it into training pairs.

    No `transform` attribute, deliberately: the image variant has one because
    torchvision's convention is that `TransformNode` may replace it at any
    time, and there is no equivalent for text — tokenization is a node of its
    own with its own cache.
    """

    def __init__(self, hf_dataset: Any, text_column: str) -> None:
        self._ds = hf_dataset
        self._text_col = text_column

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> str:
        value = self._ds[idx][self._text_col]
        if isinstance(value, str):
            return value
        # A real corpus column is nullable, and `str(None)` would inject the
        # literal word "None" into the training stream. Anything else
        # (a number, a nested field) is stringified rather than refused: the
        # row is data the user pointed at, and failing the whole load over one
        # odd cell is worse than tokenizing its text form.
        return "" if value is None else str(value)
