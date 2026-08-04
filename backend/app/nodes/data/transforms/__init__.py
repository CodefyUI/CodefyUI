"""Composable image-transform nodes (core#136).

Each node here builds ONE step of a torchvision pipeline and hands it down a
``TRANSFORM`` edge, so an augmentation policy is something you can see and
rewire on the canvas instead of a fixed list of booleans on a single node.

See ``_base`` for the chaining contract and for how a seeded run keeps its
augmentations reproducible.
"""
