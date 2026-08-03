from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition
from ...core.seeding import make_worker_init_fn

#: torch's own default. Passed through only when there are workers to
#: prefetch for — ``DataLoader`` rejects any prefetch_factor at all when
#: ``num_workers == 0``, so a blanket pass-through would break every
#: single-process loader in the gallery.
DEFAULT_PREFETCH_FACTOR = 2


class DataLoaderNode(BaseNode):
    NODE_NAME = "DataLoader"
    CATEGORY = "Data"
    DESCRIPTION = "Wrap a dataset in a DataLoader for batched iteration"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="dataset", data_type=DataType.DATASET, description="Dataset to wrap"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="dataloader", data_type=DataType.DATALOADER, description="DataLoader for batched iteration"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="batch_size", param_type=ParamType.INT, default=32, description="Samples per batch", min_value=1),
            ParamDefinition(name="shuffle", param_type=ParamType.BOOL, default=True, description="Shuffle data at every epoch"),
            ParamDefinition(name="num_workers", param_type=ParamType.INT, default=0, description="Subprocesses for data loading", min_value=0),
            # Basic: it is the one-line fix for a GPU that spends its time
            # waiting on host-to-device copies, so people reach for it early.
            ParamDefinition(
                name="pin_memory",
                param_type=ParamType.BOOL,
                default=False,
                description=(
                    "Stage batches in page-locked memory for faster GPU "
                    "transfer (no effect on CPU)"
                ),
            ),
            ParamDefinition(
                name="drop_last",
                param_type=ParamType.BOOL,
                default=False,
                description=(
                    "Discard the final short batch so every batch has the "
                    "same size"
                ),
                advanced=True,
            ),
            ParamDefinition(
                name="persistent_workers",
                param_type=ParamType.BOOL,
                default=False,
                description=(
                    "Keep worker processes alive between epochs. Requires "
                    "num_workers > 0."
                ),
                advanced=True,
            ),
            ParamDefinition(
                name="prefetch_factor",
                param_type=ParamType.INT,
                default=DEFAULT_PREFETCH_FACTOR,
                description=(
                    "Batches each worker loads ahead. Ignored when "
                    "num_workers is 0."
                ),
                min_value=1,
                advanced=True,
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
        from torch.utils.data import DataLoader

        dataset = inputs["dataset"]
        batch_size = params.get("batch_size", 32)
        shuffle = params.get("shuffle", True)
        num_workers = int(params.get("num_workers", 0) or 0)

        kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": bool(params.get("pin_memory", False)),
            "drop_last": bool(params.get("drop_last", False)),
        }

        # Both of these are illegal at num_workers == 0 (torch raises), and
        # the honest reading of "prefetch 2 batches with no workers" is that
        # there is nothing to configure — so they are dropped rather than
        # rejected. ``persistent_workers`` IS surfaced when explicitly asked
        # for, because that one is a real misconfiguration: the user wanted
        # workers kept alive and there are none.
        if num_workers > 0:
            kwargs["persistent_workers"] = bool(
                params.get("persistent_workers", False))
            prefetch = params.get("prefetch_factor", DEFAULT_PREFETCH_FACTOR)
            kwargs["prefetch_factor"] = int(
                DEFAULT_PREFETCH_FACTOR if prefetch is None else prefetch)
        elif params.get("persistent_workers"):
            raise ValueError(
                "persistent_workers requires num_workers > 0; set "
                "num_workers or turn persistent_workers off.")

        # Reproducibility (#134). A shuffling loader builds a RandomSampler
        # that draws from the GLOBAL torch RNG at ITERATION time — inside
        # the training loop node, long after this node ran — so the epoch
        # order otherwise depends on everything the graph drew in between.
        # Its own generator, seeded from (run seed, this node's id), makes
        # the order a function of the seed alone.
        #
        # The generator is attached even when ``shuffle`` is off: it is also
        # what torch uses to seed worker processes, so a map-style dataset
        # doing random augmentation is covered too.
        node_id = getattr(context, "current_node_id", "") or "dataloader"
        generator = None
        if context is not None:
            generator = context.make_generator(f"dataloader:{node_id}")
            if generator is not None:
                kwargs["generator"] = generator
        if generator is not None and num_workers > 0:
            kwargs["worker_init_fn"] = make_worker_init_fn(
                context.derive_seed(f"dataloader:{node_id}"))

        dataloader = DataLoader(**kwargs)

        return {"dataloader": dataloader}
