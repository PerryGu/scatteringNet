"""Geometry-conditioned occupancy: query XYZ plus a surface latent.

``OccupancyMLP`` stays xyz-only. This module is the Step 8 head:
``cat(xyz, z_surf)`` after a tiny PointNet on the envelope cloud.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from occupancy_mlp import build_mlp

# Distinct from OccupancyMLP so infer can tell the checkpoint apart.
CHECKPOINT_KIND = "occupancy_encoder"


class SurfaceEncoder(nn.Module):
    """
    Per-point MLP + max-pool (PointNet) over an envelope cloud.

    Shapes
    ------
    envelope: ``(U, N, 3)`` unique meshes in the batch
    output:   ``(U, D)``    one latent per mesh
    """

    def __init__(self, latent_dim: int = 64, hidden: int = 64) -> None:
        super().__init__()
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")
        if hidden < 1:
            raise ValueError(f"hidden must be >= 1, got {hidden}")
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.point_mlp = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, envelope: Tensor) -> Tensor:
        """Max-pool per-point features → one vector per unique mesh."""
        if envelope.ndim != 3 or envelope.shape[-1] != 3:
            raise ValueError(
                f"envelope must have shape (U, N, 3), got {tuple(envelope.shape)}"
            )
        features = self.point_mlp(envelope)
        return features.max(dim=1).values


class OccupancyEncoder(nn.Module):
    """
    Occupancy logits from query XYZ and a surface envelope.

    Unique ``shape_id`` values are encoded **once** per batch, then
    broadcast. That is the same rule the later face encoder will use.

    Shapes
    ------
    xyz:      ``(B, 3)``
    envelope: ``(B, N, 3)``
    shape_id: ``(B,)`` long
    output:   ``(B, 1)`` logits
    """

    def __init__(
        self,
        hidden: int = 64,
        depth: int = 4,
        latent_dim: int = 64,
    ) -> None:
        super().__init__()
        if hidden < 1:
            raise ValueError(f"hidden must be >= 1, got {hidden}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")
        self.hidden = hidden
        self.depth = depth
        self.latent_dim = latent_dim
        self.surface = SurfaceEncoder(latent_dim=latent_dim, hidden=hidden)
        # OccupancyMLP is 3 → H; here the first Linear is (3 + D) → H.
        self.head = build_mlp(3 + latent_dim, hidden, depth)

    def encode_unique(self, envelope: Tensor, shape_id: Tensor) -> Tensor:
        """
        Encode each distinct ``shape_id`` once and scatter back to ``(B, D)``.

        Parameters
        ----------
        envelope, shape_id:
            Batched clouds and integer mesh ids (same length B).
        """
        if shape_id.ndim != 1 or int(shape_id.shape[0]) != int(envelope.shape[0]):
            raise ValueError(
                f"shape_id must be (B,), got {tuple(shape_id.shape)} "
                f"for envelope {tuple(envelope.shape)}"
            )
        unique_ids, inverse = torch.unique(shape_id, sorted=True, return_inverse=True)
        # First row in the batch for each unique id (envelopes match per file).
        hits = shape_id.unsqueeze(0) == unique_ids.unsqueeze(1)
        first = hits.to(dtype=torch.int64).argmax(dim=1)
        z_unique = self.surface(envelope[first])
        return z_unique[inverse]

    def forward(
        self,
        xyz: Tensor,
        envelope: Tensor,
        shape_id: Tensor,
    ) -> Tensor:
        """``cat(xyz, z_surf)`` → occupancy logit."""
        if xyz.ndim != 2 or xyz.shape[-1] != 3:
            raise ValueError(f"xyz must have shape (B, 3), got {tuple(xyz.shape)}")
        if envelope.ndim != 3 or envelope.shape[-1] != 3:
            raise ValueError(
                f"envelope must have shape (B, N, 3), got {tuple(envelope.shape)}"
            )
        if int(xyz.shape[0]) != int(envelope.shape[0]):
            raise ValueError(
                f"xyz/envelope batch mismatch: {tuple(xyz.shape)} vs "
                f"{tuple(envelope.shape)}"
            )
        ids = shape_id.reshape(-1)
        z_surf = self.encode_unique(envelope, ids)
        return self.head(torch.cat([xyz, z_surf], dim=-1))
