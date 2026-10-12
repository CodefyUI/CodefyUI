"""DatasetSplit — cut one DATASET into a train part and a validation part.

Added for #703: without it the only held-out data a graph could reach was
the dataset's own test split, so examples validated on the test set.

The split is a seeded permutation of the indices, so the same ``seed`` on
the same dataset always yields the same two parts, whatever the run seed.
``subset`` keeps only that many of the shuffled samples before splitting,
for a quick iteration on a large dataset.

Both outputs are ``torch.utils.data.Subset`` views of the input: no sample
is copied, and both parts read the input's ``transform``. A pipeline that
should differ between the two parts (augmentation on train only) belongs
upstream, on two Dataset nodes, rather than on the outputs here.
"""

from __future__ import annotations

from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class DatasetSplitNode(BaseNode):
    NODE_NAME = "DatasetSplit"
    CATEGORY = "Data"
    DESCRIPTION = "Split a dataset into seeded train and validation parts"
    DETAILS = (
        "A seeded shuffle of the sample indices, so the same seed gives the "
        "same split on every run. `subset` keeps only that many shuffled "
        "samples first, for quick experiments. Both parts share the input's "
        "transform."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="dataset", data_type=DataType.DATASET, description="Dataset to split"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="train", data_type=DataType.DATASET, description="The samples not held out"),
            PortDefinition(name="val", data_type=DataType.DATASET, description="The held-out samples"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="val_fraction",
                param_type=ParamType.FLOAT,
                default=0.1,
                min_value=0.0,
                max_value=0.99,
                description="Share of the samples held out for validation. 0 sends everything to train.",
            ),
            ParamDefinition(
                name="subset",
                param_type=ParamType.INT,
                default=0,
                min_value=0,
                description="Keep only this many randomly chosen samples before splitting. 0 keeps all.",
            ),
            ParamDefinition(
                name="seed",
                param_type=ParamType.INT,
                default=42,
                description="Seed of the shuffle. The same seed gives the same split.",
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch
        from torch.utils.data import Subset

        dataset = inputs["dataset"]
        val_fraction = float(params.get("val_fraction", 0.1))
        subset = int(params.get("subset", 0) or 0)
        seed = int(params.get("seed", 42))
        if not 0.0 <= val_fraction < 1.0:
            raise ValueError(
                f"DatasetSplit: val_fraction must be in [0, 1), got {val_fraction}.")

        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(dataset), generator=generator).tolist()
        if subset > 0:
            order = order[:subset]
        n_val = int(round(len(order) * val_fraction))
        return {
            "train": Subset(dataset, order[n_val:]),
            "val": Subset(dataset, order[:n_val]),
        }
