"""Tests for SyntheticSequenceNode.

The node's job is to make one experiment possible: train the same graph twice,
changing only the recurrent layer, and see a gated cell succeed where a plain
one fails. That experiment is only valid if the *data* is identical between
the two runs and if the answer really is unreachable except by carrying it
across the whole sequence — so those two properties are what most of these
tests pin down.
"""

from __future__ import annotations

import pytest
import torch

from app.nodes.data.synthetic_sequence_node import SyntheticSequenceNode


def _run(**params):
    p = {
        "kind": "recall_first",
        "seq_len": 12,
        "n_samples": 40,
        "n_classes": 10,
        "n_distractors": 10,
        "seed": 0,
    }
    p.update(params)
    return SyntheticSequenceNode().execute({}, p)


def test_node_metadata():
    assert SyntheticSequenceNode.NODE_NAME == "SyntheticSequence"
    assert SyntheticSequenceNode.CATEGORY == "Data"
    assert [p.name for p in SyntheticSequenceNode.define_outputs()] == [
        "dataset",
        "vocab_size",
    ]


def test_sample_shape_and_dtype():
    ds = _run()["dataset"]
    seq, label = ds[0]
    assert seq.shape == (12,)
    assert seq.dtype == torch.long  # what nn.Embedding requires
    assert isinstance(label, int)


def test_len_matches_n_samples():
    assert len(_run(n_samples=37)["dataset"]) == 37


def test_vocab_size_covers_every_token_emitted():
    res = _run(n_classes=6, n_distractors=9)
    ds = res["dataset"]
    assert res["vocab_size"] == 15
    biggest = max(int(ds[i][0].max()) for i in range(len(ds)))
    assert biggest < res["vocab_size"]


def test_recall_first_puts_the_answer_at_position_zero():
    ds = _run(kind="recall_first")["dataset"]
    for i in range(len(ds)):
        seq, label = ds[i]
        assert int(seq[0]) == label


def test_recall_last_puts_the_answer_at_the_final_position():
    ds = _run(kind="recall_last")["dataset"]
    for i in range(len(ds)):
        seq, label = ds[i]
        assert int(seq[-1]) == label


def test_distractors_carry_no_answer_information():
    """Every non-answer position must be outside the label range.

    If a distractor could equal the label, a model could sometimes score
    above chance by reading a nearby token, and a plain RNN "solving" the
    task would prove nothing about long-range memory.
    """
    ds = _run(kind="recall_first", n_classes=10, n_distractors=10)["dataset"]
    for i in range(len(ds)):
        seq, _ = ds[i]
        assert torch.all(seq[1:] >= 10)


def test_labels_span_the_class_range():
    ds = _run(n_samples=400, n_classes=10)["dataset"]
    labels = {ds[i][1] for i in range(len(ds))}
    assert labels == set(range(10))


def test_same_seed_is_reproducible():
    a, b = _run(seed=7)["dataset"], _run(seed=7)["dataset"]
    assert torch.equal(a.sequences, b.sequences)
    assert torch.equal(a.labels, b.labels)


def test_different_seed_gives_different_data():
    a, b = _run(seed=1)["dataset"], _run(seed=2)["dataset"]
    assert not torch.equal(a.sequences, b.sequences)


def test_kind_does_not_change_the_data_apart_from_answer_position():
    """The controlled-variable guarantee, stated as a test.

    recall_first and recall_last at the same seed differ only in *where* the
    answer sits: same labels, same distractors everywhere else. Anything
    else varying between them would confound the distance comparison the
    node is built for.
    """
    first = _run(kind="recall_first", seed=3)["dataset"]
    last = _run(kind="recall_last", seed=3)["dataset"]
    assert torch.equal(first.labels, last.labels)
    assert torch.equal(first.sequences[:, 1:-1], last.sequences[:, 1:-1])


@pytest.mark.parametrize(
    "bad,match",
    [
        ({"kind": "recall_middle"}, "unknown kind"),
        ({"seq_len": 1}, "seq_len"),
        ({"n_samples": 0}, "n_samples"),
        ({"n_classes": 1}, "n_classes"),
        ({"n_distractors": 0}, "n_distractors"),
    ],
)
def test_invalid_params_fail_on_this_node_with_a_readable_message(bad, match):
    with pytest.raises(ValueError, match=match):
        _run(**bad)


def test_dataset_flows_through_a_dataloader():
    """The shape contract downstream: (batch, seq) longs + (batch,) labels."""
    from torch.utils.data import DataLoader

    ds = _run(n_samples=32, seq_len=9)["dataset"]
    x, y = next(iter(DataLoader(ds, batch_size=8)))
    assert x.shape == (8, 9)
    assert x.dtype == torch.long
    assert y.shape == (8,)
