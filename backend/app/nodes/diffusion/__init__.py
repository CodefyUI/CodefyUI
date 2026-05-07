"""Diffusion / U-Net teaching nodes.

Educational counterparts to Stable-Diffusion-style building blocks. Tiny
defaults so a forward pass through a toy U-Net fits in inline tensor
previews; reverse-diffusion sampling loops are encapsulated in a single
``DDPMSampler`` node so students see the schedule dynamics without
needing a custom loop primitive in the graph editor.
"""
