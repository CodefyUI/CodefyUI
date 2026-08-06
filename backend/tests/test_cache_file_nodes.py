"""Regression tests for #116 - nodes reading external state are never cached.

``ExecutionCache.compute_key()`` hashes ``{node_type, params, upstream keys,
device}``; a file's *contents* never enter that key by default. A cacheable
``CSVReader`` would therefore keep handing back the tensor it read on the
very first run, even after the user edits the CSV on disk. Nodes whose
output depends on state outside the graph (disk, network, process
environment) opt out with ``cacheable = False``, and ``graph_engine``
propagates that opt-out to every node downstream of them.

#116 (PR #142) set ``cacheable = False`` on nine reader nodes wholesale --
correctness first, at the cost of re-reading from disk on every run. #144
narrowed that: most of them now fold a content fingerprint (size, mtime, and
for small files a hash -- see ``core.cache_fingerprint``) into the cache key
instead, so they are cacheable again AND still notice a changed file. See
``test_cache_content_fingerprint.py`` for that half of the story.

What is left here is the residual: nodes whose external state a content
fingerprint of local files cannot fully describe -- ``KaggleDataset`` and
``HuggingFaceDataset`` hit the network and (for Kaggle) environment
credentials, neither of which a fingerprint of the local cache directory
can see. Fingerprinting the local cache would add cost without closing the
actual staleness gap (a changed remote revision, or changed credentials,
with an unchanged local cache), so these stay unconditionally non-cacheable.
"""

from __future__ import annotations

import pytest

from app.core.node_base import BaseNode
from app.core.node_registry import registry
from app.nodes.data.huggingface_dataset_node import HuggingFaceDatasetNode
from app.nodes.data.kaggle_dataset_node import KaggleDatasetNode
from app.nodes.data.synthetic_dataset_node import SyntheticDatasetNode
from app.nodes.tensor_ops.add_node import AddNode
from app.nodes.tensor_ops.mean_node import MeanNode
from app.nodes.utility.flatten_node import FlattenNode

# Nodes whose network/credential dependence a content fingerprint of local
# files cannot describe (#144 leaves these outside the restoration -- see
# the node-level comments on each for the specific reasoning).
EXTERNAL_STATE_NODES = [
    HuggingFaceDatasetNode,  # HF Hub download + on-disk HF cache
    KaggleDatasetNode,       # kagglehub download + KAGGLE_* env credentials
]

# Control group. These stay cacheable so the fix remains surgical (#116:
# "no other node's caching behavior changes"). SyntheticDataset is here on
# purpose: it sits in app/nodes/data/ beside the file readers but builds its
# samples from a seeded RNG, so caching it is correct.
PURE_NODES = [MeanNode, AddNode, FlattenNode, SyntheticDatasetNode]


def _node_name(node_cls: type[BaseNode]) -> str:
    return node_cls.NODE_NAME


@pytest.mark.parametrize("node_cls", EXTERNAL_STATE_NODES, ids=_node_name)
def test_external_state_node_is_not_cacheable(node_cls: type[BaseNode]) -> None:
    """Reading disk / network / env means the cache key cannot describe the
    output, so the node has to opt out.

    Checked through the registry as well: graph_engine reads ``cacheable`` off
    whatever ``registry.get(node_type)`` hands it, so the opt-out has to
    survive node discovery, not just the import at the top of this file.
    """
    assert node_cls.cacheable is False, (
        f"{node_cls.NODE_NAME} reads state outside the graph, so its cache key "
        "cannot describe its output. Set `cacheable = False` on the class."
    )
    assert registry.get(node_cls.NODE_NAME) is node_cls, (
        f"the registry serves a different {node_cls.NODE_NAME} class than the "
        "one asserted above, so the engine may not see the opt-out"
    )


@pytest.mark.parametrize("node_cls", PURE_NODES, ids=_node_name)
def test_pure_nodes_stay_cacheable(node_cls: type[BaseNode]) -> None:
    """Pure compute nodes keep the BaseNode default - the #116 fix must not
    widen into a blanket cache disable."""
    assert node_cls.cacheable is True
