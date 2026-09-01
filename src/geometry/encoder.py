"""PointNet over triangle face tokens (Step 10).

Input ``(B, F, 12)`` is ``[v0, v1, v2, n]``. Max-pool yields one ``z_face``
per mesh. Occupancy fusion lives on ``OccupancyEncoder``.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from geometry.face_tokens import FACE_FEAT_DIM


class MeshFaceEncoder(nn.Module):
    """
    Per-face MLP + max-pool.

    Shapes
    ------
    tokens: ``(U, F, 12)`` unique meshes in the batch
    output: ``(U, D)``     one latent per mesh
    """

    def __init__(
        self,
        latent_dim: int = 64,
        *,
        hidden: int = 64,
        depth: int = 4,
        in_dim: int = FACE_FEAT_DIM,
    ) -> None:
        super().__init__()
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")
        if hidden < 1:
            raise ValueError(f"hidden must be >= 1, got {hidden}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if in_dim < 1:
            raise ValueError(f"in_dim must be >= 1, got {in_dim}")
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.depth = depth
        self.in_dim = int(in_dim)
        layers: list[nn.Module] = []
        dim = self.in_dim
        for _ in range(depth):
            layers.append(nn.Linear(dim, hidden))
            layers.append(nn.ReLU(inplace=True))
            dim = hidden
        layers.append(nn.Linear(dim, latent_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, tokens: Tensor) -> Tensor:
        """Max-pool per-face features → one vector per unique mesh."""
        if tokens.ndim != 3 or tokens.shape[-1] != self.in_dim:
            raise ValueError(
                f"tokens must have shape (U, F, {self.in_dim}), got {tuple(tokens.shape)}"
            )
        features = self.mlp(tokens)
        return features.max(dim=1).values
