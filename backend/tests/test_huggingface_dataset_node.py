"""Tests for HuggingFaceDatasetNode (metadata only — full load needs network).

Live download integration is exercised by `test_dataset_extension.py`. Here we
focus on the metadata + parameter-handling surface that should pass without
network access.
"""

from __future__ import annotations

from app.nodes.data.huggingface_dataset_node import HuggingFaceDatasetNode


def test_node_metadata():
    assert HuggingFaceDatasetNode.NODE_NAME == "HuggingFaceDataset"
    assert HuggingFaceDatasetNode.CATEGORY == "Data"


def test_no_inputs_defined():
    assert HuggingFaceDatasetNode.define_inputs() == []


def test_output_is_dataset():
    outputs = HuggingFaceDatasetNode.define_outputs()
    assert len(outputs) == 1
    assert outputs[0].name == "dataset"


def test_default_params_have_sensible_defaults():
    params = {p.name: p for p in HuggingFaceDatasetNode.define_params()}
    assert params["dataset_name"].default == "ylecun/mnist"
    assert params["split"].default == "train"
    assert params["image_column"].default == "image"
    assert params["label_column"].default == "label"
