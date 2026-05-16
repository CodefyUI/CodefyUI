"""SyntheticDatasetNode — produce 2D toy datasets via sklearn.datasets.

Designed for the chapter examples in C2-2 / C2-3 / C2-4 / C2-5, where the
textbook keeps reusing "concentric circles" / "two moons" / "blobs" as the
canonical hard-cases for classifiers. CSVReader requires a real CSV file;
this node lets a graph generate the data inline so an example ships with
zero data dependencies.

Output ports mirror :class:`CSVReaderNode` so existing TrainTestSplit and
classifier nodes drop in with no rewiring:

    tensor : (N, 2) float32 — feature matrix
    labels : list[str] — class labels (stringified ints)
    columns: ["x0", "x1"] — feature column names

The intentionally limited list of kinds keeps the surface tiny — anything
beyond circles/moons/blobs is better served by a real CSV.
"""

from __future__ import annotations

from typing import Any

import torch

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)


class SyntheticDatasetNode(BaseNode):
    NODE_NAME = "SyntheticDataset"
    CATEGORY = "Data"
    DESCRIPTION = (
        "Generate a 2D toy dataset (concentric circles, two moons, or "
        "isotropic blobs) via sklearn. Output matches CSVReader's shape, "
        "so TrainTestSplit and classifier nodes plug in directly."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tensor",
                data_type=DataType.TENSOR,
                description="Float32 [N, 2] feature matrix.",
            ),
            PortDefinition(
                name="labels",
                data_type=DataType.LIST,
                description="String class labels (e.g. ['0','1','0',...]).",
            ),
            PortDefinition(
                name="columns",
                data_type=DataType.LIST,
                description="Feature column names: ['x0', 'x1'].",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="kind",
                param_type=ParamType.SELECT,
                default="circles",
                options=["circles", "moons", "blobs", "classification"],
                description=(
                    "circles: two concentric rings (linearly inseparable). "
                    "moons: two interlocking half-moons. "
                    "blobs: isotropic Gaussian clusters (linearly separable). "
                    "classification: general sklearn make_classification."
                ),
            ),
            ParamDefinition(
                name="n_samples",
                param_type=ParamType.INT,
                default=200,
                min_value=10,
                description="Total number of samples to generate.",
            ),
            ParamDefinition(
                name="noise",
                param_type=ParamType.FLOAT,
                default=0.1,
                min_value=0.0,
                description="Gaussian noise added to the points (circles/moons/classification only).",
            ),
            ParamDefinition(
                name="factor",
                param_type=ParamType.FLOAT,
                default=0.5,
                min_value=0.0,
                description="Inner-circle radius ratio for 'circles' (0<factor<1). Ignored for other kinds.",
            ),
            ParamDefinition(
                name="centers",
                param_type=ParamType.INT,
                default=3,
                min_value=2,
                description="Number of blob centers (for 'blobs' only).",
            ),
            ParamDefinition(
                name="seed",
                param_type=ParamType.INT,
                default=42,
                description="Random seed for reproducibility.",
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
        from sklearn.datasets import (
            make_blobs,
            make_circles,
            make_classification,
            make_moons,
        )

        kind = str(params.get("kind", "circles"))
        n_samples = int(params.get("n_samples", 200))
        noise = float(params.get("noise", 0.1))
        factor = float(params.get("factor", 0.5))
        centers = int(params.get("centers", 3))
        seed = int(params.get("seed", 42))

        if kind == "circles":
            X, y = make_circles(
                n_samples=n_samples,
                noise=noise,
                factor=max(0.01, min(0.99, factor)),
                random_state=seed,
            )
        elif kind == "moons":
            X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
        elif kind == "blobs":
            X, y = make_blobs(
                n_samples=n_samples,
                centers=centers,
                cluster_std=max(0.1, noise * 5.0),
                random_state=seed,
            )
        elif kind == "classification":
            X, y = make_classification(
                n_samples=n_samples,
                n_features=2,
                n_informative=2,
                n_redundant=0,
                n_clusters_per_class=1,
                flip_y=noise,
                random_state=seed,
            )
        else:
            raise ValueError(f"SyntheticDataset: unknown kind {kind!r}")

        tensor = torch.from_numpy(X).float()
        labels = [str(int(v)) for v in y.tolist()]

        return {
            "tensor": tensor,
            "labels": labels,
            "columns": ["x0", "x1"],
        }
