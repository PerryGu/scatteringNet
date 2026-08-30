"""Occupancy MLP: raw XYZ → inside/outside logit."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

# Checkpoint schema tag (not a YAML knob).
CHECKPOINT_KIND = "occupancy_mlp"


def build_mlp(in_dim: int, hidden: int, depth: int) -> nn.Sequential:
    """Linear→ReLU × ``depth`` then a 1-logit head. Shared by OccupancyMLP."""
    if in_dim < 1:
        raise ValueError(f"in_dim must be >= 1, got {in_dim}")
    if hidden < 1:
        raise ValueError(f"hidden must be >= 1, got {hidden}")
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    layers: list[nn.Module] = []
    dim = in_dim
    for _ in range(depth):
        layers.append(nn.Linear(dim, hidden))
        layers.append(nn.ReLU(inplace=True))
        dim = hidden
    layers.append(nn.Linear(dim, 1))
    return nn.Sequential(*layers)


class OccupancyMLP(nn.Module):
    """
    Tiny fully-connected occupancy field.

    Maps a batch of 3D query coordinates to a single unnormalized logit per
    point. A later training step will apply ``binary_cross_entropy_with_logits``
    (do not softmax / sigmoid inside ``forward``).

    Shapes
    ------
    xyz:    ``(B, 3)``  batch of query points (device follows the caller)
    output: ``(B, 1)``  logits; positive → inside, negative → outside

    Device
    ------
    Parameters live on whatever device the module was moved to
    (``.to(device)`` / ``.cuda()``). ``xyz`` must already be on that same
    device; this module does not copy tensors.
    """

    def __init__(self, hidden: int = 64, depth: int = 4) -> None:
        """
        Build Linear→ReLU blocks then a 1-logit head.

        Parameters
        ----------
        hidden:
            Channel width of each hidden Linear (must be ``>= 1``).
        depth:
            Number of hidden Linear+ReLU blocks (must be ``>= 1``).
        """
        super().__init__()
        self.hidden = hidden
        self.depth = depth
        # First Linear is 3 → H; remaining blocks are H → H.
        self.net = build_mlp(3, hidden, depth)

    def forward(self, xyz: Tensor) -> Tensor:
        """
        Evaluate occupancy logits at query coordinates.

        Parameters
        ----------
        xyz:
            Float tensor of shape ``(B, 3)``. Last dim is Cartesian XYZ.

        Returns
        -------
        Tensor
            Float tensor of shape ``(B, 1)`` on the same device as ``xyz``.
        """
        if xyz.ndim != 2 or xyz.shape[-1] != 3:
            raise ValueError(
                f"xyz must have shape (B, 3), got {tuple(xyz.shape)}"
            )
        # Sequential Linear layers require matching dtype/device with parameters.
        return self.net(xyz)
