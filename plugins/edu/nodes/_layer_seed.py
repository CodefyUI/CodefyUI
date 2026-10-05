"""Initial weights for the layers the edu MLP nodes create.

``FFNLayer`` and ``TrainAndEvaluate`` each build one new ``nn.Linear`` per
run. Its weights are drawn from ``derive_seed(run seed, node id)``, the value
the engine and an exported script seed the global RNG with right before each
node of a seeded run, so with a seed set they are bit for bit what they were
when the layer simply drew from that RNG (it was the node's first draw; see
``seeded_linear_init``).

With no run seed they are drawn from ``derive_seed(0, node id)``. An unseeded
run then repeats, gives the numbers a seed-0 run gives, and an exported
script -- a fresh process, whose RNG starts somewhere else than the editor's
-- prints the same. Pack code on purpose: core nodes keep "no seed means
torch's own entropy" (``ExecutionContext.derive_seed`` returns None).

A call with no context at all (``execute()`` called directly) has no node id
to derive from, and one shared fallback would give every such layer the same
weights, so the layer keeps what its constructor drew.
"""

from __future__ import annotations

from typing import Any

from app.core.seeding import derive_seed, seeded_linear_init

#: Stands in for the run seed when the run has none.
UNSEEDED_RUN_SEED = 0


def seed_new_layer(layer: Any, context: Any) -> Any:
    """Give *layer*, just built on the CPU, this node's initial weights."""
    if context is None:
        return layer
    node_id = getattr(context, "current_node_id", "") or ""
    seed = context.derive_seed(node_id)
    if seed is None:
        seed = derive_seed(UNSEEDED_RUN_SEED, node_id)
    return seeded_linear_init(layer, seed)
