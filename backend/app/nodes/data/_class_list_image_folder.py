"""``ImageFolder`` numbering a class list given to it, folders or not.

Kept in its own module so the node module does not import torchvision at
load time, and at module level so a DataLoader worker can unpickle it.
"""

from __future__ import annotations

import os
from typing import Any

from torchvision.datasets import DatasetFolder, ImageFolder


class ClassListImageFolder(ImageFolder):
    """Numbers *classes* in the order given; a class with no folder has no samples.

    ``ImageFolderDataset`` passes the class folders of every split, so a
    val folder missing ``banana`` still labels ``cherry`` with the index the
    training folder gives it.
    """

    def __init__(self, root: str, classes: list[str], **kwargs: Any) -> None:
        # Set before super().__init__, which calls find_classes.
        self._given_classes = list(classes)
        super().__init__(root, **kwargs)

    def find_classes(self, directory: Any) -> tuple[list[str], dict[str, int]]:
        classes = list(self._given_classes)
        return classes, {name: i for i, name in enumerate(classes)}

    @staticmethod
    def make_dataset(directory: Any, class_to_idx: dict[str, int], *args: Any, **kwargs: Any) -> Any:
        # torchvision refuses a class it finds no file for, so only the class
        # folders present in this split are walked; an empty folder that does
        # exist is still refused, as before.
        present = {
            name: index for name, index in class_to_idx.items()
            if os.path.isdir(os.path.join(directory, name))
        }
        return DatasetFolder.make_dataset(directory, present, *args, **kwargs)
