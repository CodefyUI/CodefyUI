"""What ``value_bytes`` counts, and what it deliberately does not (#135).

The three stores' byte budgets are only as good as this measurement, and
the cases that decide whether it is any good are the boring ones: a dict of
lists of tensors (the ordinary shape of a node's outputs), a view that
shares its parent's storage, and a payload that is not tensors at all.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from app.core.memory_budget import (
    MAX_WALK_DEPTH,
    format_bytes,
    value_bytes,
)

KB = 1024


def test_a_contiguous_tensor_measures_element_size_times_count():
    """The formula the acceptance criterion names, on the case it names it for."""
    tensor = torch.zeros(1000, dtype=torch.float32)
    assert value_bytes(tensor) == tensor.element_size() * tensor.nelement()
    assert value_bytes(tensor) == 4000


def test_dtype_is_respected():
    assert value_bytes(torch.zeros(1000, dtype=torch.float64)) == 8000
    assert value_bytes(torch.zeros(1000, dtype=torch.bfloat16)) == 2000
    assert value_bytes(torch.zeros(1000, dtype=torch.int8)) == 1000


def test_containers_are_walked_through():
    """A dict of lists of tensors is the normal case, not an edge case."""
    payload = {
        "logits": [torch.zeros(250), torch.zeros(250)],
        "hidden": (torch.zeros(250), {"deep": torch.zeros(250)}),
    }
    # Four disjoint 250-float tensors, and nothing double-counted.
    assert 4000 <= value_bytes(payload) < 4400  # plus the dict keys' strings


def test_an_empty_container_is_free_of_tensor_cost():
    assert value_bytes([]) == 0
    assert value_bytes({}) == 0
    assert value_bytes(None) == 0
    assert value_bytes(3.14) == 0


def test_a_view_does_not_multiply_its_parents_storage():
    """The reason the walk measures STORAGE rather than the element formula.

    ``base[0:10]`` reports 40 bytes by ``element_size() * nelement()`` while
    holding the whole 4000-byte allocation alive. Counting the pair as 4040
    would be wrong in the harmless direction; counting the SLICE as 40 in a
    cache holding nothing else would be wrong in the direction that fills a
    graphics card.
    """
    base = torch.zeros(1000, dtype=torch.float32)
    assert value_bytes([base, base[0:10], base.view(-1)]) == 4000
    assert value_bytes(base[0:10]) == 4000


def test_two_references_to_one_tensor_count_once():
    tensor = torch.zeros(1000)
    # Measured against the SAME dict with the tensors removed, so the two
    # string keys (which are payload too, and counted) cancel out and what
    # is left is the tensor's contribution alone.
    keys_only = value_bytes({"a": None, "b": None})
    assert value_bytes({"a": tensor, "b": tensor}) - keys_only == 4000


def test_distinct_tensors_of_the_same_shape_both_count():
    """The dedup is on storage identity, not on shape."""
    assert value_bytes([torch.zeros(1000), torch.zeros(1000)]) == 8000


def test_a_module_is_its_parameters_and_buffers():
    module = nn.Linear(100, 50)  # 5000 weights + 50 biases, float32
    assert value_bytes(module) == (100 * 50 + 50) * 4


def test_a_modules_registered_buffer_counts():
    module = nn.BatchNorm1d(100)
    # weight + bias (100 each) as parameters; running_mean + running_var
    # (100 each) plus a scalar num_batches_tracked as buffers.
    assert value_bytes(module) == 4 * 100 * 4 + 8


def test_tied_weights_are_one_allocation():
    embedding = nn.Embedding(100, 16)
    decoder = nn.Linear(16, 100, bias=False)
    decoder.weight = embedding.weight
    tied = nn.ModuleList([embedding, decoder])
    assert value_bytes(tied) == 100 * 16 * 4


def test_bytes_and_strings_are_payload_too():
    """A base64 PNG on an IMAGE port is real residency, not zero."""
    assert value_bytes(b"x" * 4096) == 4096
    assert value_bytes(bytearray(4096)) == 4096
    assert value_bytes("x" * 4096) >= 4096


def test_a_numpy_array_counts():
    numpy = __import__("numpy")
    assert value_bytes(numpy.zeros(1000, dtype=numpy.float32)) == 4000


def test_things_that_are_not_payload_measure_zero():
    """Not an ``sys.getsizeof`` crawl -- a budget for what fills a card."""
    assert value_bytes(object()) == 0
    assert value_bytes(range(10_000)) == 0
    assert value_bytes({"n": 1, "flag": True, "ratio": 0.5}) == \
        value_bytes({}) + value_bytes("n") + value_bytes("flag") + \
        value_bytes("ratio")


def test_a_self_referential_structure_terminates():
    payload: dict = {"t": torch.zeros(1000)}
    payload["self"] = payload
    assert value_bytes(payload) >= 4000


def test_a_cyclic_list_terminates():
    payload: list = [torch.zeros(1000)]
    payload.append(payload)
    assert value_bytes(payload) >= 4000


def test_nesting_past_the_depth_cap_stops_rather_than_recursing():
    payload: object = torch.zeros(1000)
    for _ in range(MAX_WALK_DEPTH + 10):
        payload = [payload]
    # No RecursionError, and the answer is a lower bound rather than a lie
    # about the shape of the data.
    assert value_bytes(payload) == 0


def test_a_tensor_on_the_meta_device_falls_back_to_the_element_formula():
    with torch.device("meta"):
        tensor = torch.zeros(1000, dtype=torch.float32)
    assert value_bytes(tensor) == 4000


# ── the "never raises" promise, enforced rather than intended ─────────────


class _ExplodingSize:
    """A ``nbytes`` PROPERTY that raises -- a lazily-loaded array, a wrapper
    around an unmaterialised value, or any custom node's return type."""

    @property
    def nbytes(self):
        raise RuntimeError("nbytes exploded")


class _ExplodingIteration:
    """Something dict-like whose iteration fails."""

    def keys(self):
        raise RuntimeError("keys exploded")


def test_an_exploding_nbytes_property_measures_zero_instead_of_raising():
    """A bare ``getattr`` propagates anything that is not AttributeError.

    It matters because of WHERE this runs: inside ``ExecutionCache.put``,
    inside the node's own ``try``. An escape would report a node that
    computed its result perfectly well as a failure, and in fail_fast it
    would take the run down.
    """
    assert value_bytes(_ExplodingSize()) == 0


def test_an_exploding_object_inside_a_container_does_not_poison_the_walk():
    payload = {"real": torch.zeros(1000), "bad": _ExplodingSize()}
    assert value_bytes(payload) >= 4000


def test_nothing_the_walk_can_meet_escapes_as_an_exception():
    for hostile in (_ExplodingSize(), _ExplodingIteration(),
                    [_ExplodingSize()], {"k": _ExplodingIteration()}):
        assert value_bytes(hostile) >= 0  # the point is that it RETURNS


def test_a_module_whose_parameters_raise_is_survivable():
    class _BrokenModule(nn.Module):
        def parameters(self, recurse=True):
            raise RuntimeError("parameters exploded")

    assert value_bytes(_BrokenModule()) == 0


def test_format_bytes_reads_like_a_size():
    assert format_bytes(512) == "512 B"
    assert format_bytes(1536) == "1.5 KB"
    assert format_bytes(3 * 1024 * 1024) == "3.0 MB"
    assert format_bytes(2 * 1024 ** 3) == "2.0 GB"
