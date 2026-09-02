"""Geometry-conditioned occupancy: query XYZ plus a shape latent.

``OccupancyMLP`` stays xyz-only. ``shape_encoder: surface`` is the Step 8
envelope PointNet. ``shape_encoder: mesh`` is Step 10 ``MeshFaceEncoder``.
``knn_k > 0`` (surface only) adds per-query nearest envelope offsets.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from geometry.encoder import MeshFaceEncoder
from geometry.face_tokens import FACE_FEAT_DIM
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


def knn_offsets(xyz: Tensor, envelope: Tensor, k: int) -> Tensor:
    """
    Offsets from each query to its ``k`` nearest envelope points.

    Distances are Euclidean. ``k`` is clamped to the envelope length.
    Shapes: ``xyz (B, 3)``, ``envelope (B, N, 3)`` → ``(B, k, 3)``.
    """
    if xyz.ndim != 2 or xyz.shape[-1] != 3:
        raise ValueError(f"xyz must have shape (B, 3), got {tuple(xyz.shape)}")
    if envelope.ndim != 3 or envelope.shape[-1] != 3:
        raise ValueError(
            f"envelope must have shape (B, N, 3), got {tuple(envelope.shape)}"
        )
    if int(xyz.shape[0]) != int(envelope.shape[0]):
        raise ValueError(
            f"xyz/envelope batch mismatch: {tuple(xyz.shape)} vs {tuple(envelope.shape)}"
        )
    n_env = int(envelope.shape[1])
    if n_env < 1:
        raise ValueError("envelope length N must be >= 1")
    take = min(int(k), n_env)
    if take < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    # (B, N) straight-line distance in the shared AABB frame.
    dist = torch.linalg.norm(envelope - xyz.unsqueeze(1), dim=-1)
    idx = dist.topk(take, dim=-1, largest=False).indices
    nbrs = torch.gather(envelope, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    return nbrs - xyz.unsqueeze(1)


class OccupancyEncoder(nn.Module):
    """
    Occupancy logits from query XYZ and a geometry code.

    Unique ``shape_id`` values are encoded **once** per batch, then
    broadcast. ``surface`` uses envelope XYZ ``(B, N, 3)``;
    ``mesh`` uses face tokens ``(B, F, 12)``.
    ``knn_k > 0`` (surface) concatenates a local envelope code.

    Shapes
    ------
    xyz:      ``(B, 3)``
    geom:     ``(B, N, 3)`` or ``(B, F, 12)``
    shape_id: ``(B,)`` long
    output:   ``(B, 1)`` logits
    """

    def __init__(
        self,
        hidden: int = 64,
        depth: int = 4,
        latent_dim: int = 64,
        *,
        shape_encoder: str = "surface",
        encoder_hidden: int | None = None,
        encoder_depth: int = 4,
        knn_k: int = 0,
        knn_local_dim: int | None = None,
    ) -> None:
        super().__init__()
        if hidden < 1:
            raise ValueError(f"hidden must be >= 1, got {hidden}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")
        kind = str(shape_encoder).strip().lower()
        if kind not in ("surface", "mesh"):
            raise ValueError(
                f"shape_encoder must be 'surface' or 'mesh', got {shape_encoder!r}"
            )
        k = int(knn_k)
        if k < 0:
            raise ValueError(f"knn_k must be >= 0, got {k}")
        if k > 0 and kind != "surface":
            raise ValueError("knn_k > 0 requires shape_encoder='surface'")
        local_dim = int(knn_local_dim) if knn_local_dim is not None else int(latent_dim)
        if k > 0 and local_dim < 1:
            raise ValueError(f"knn_local_dim must be >= 1, got {local_dim}")
        self.hidden = hidden
        self.depth = depth
        self.latent_dim = latent_dim
        self.shape_encoder = kind
        self.knn_k = k
        self.knn_local_dim = local_dim if k > 0 else 0
        eh = int(encoder_hidden) if encoder_hidden is not None else int(hidden)
        ed = int(encoder_depth)
        if eh < 1:
            raise ValueError(f"encoder_hidden must be >= 1, got {eh}")
        if ed < 1:
            raise ValueError(f"encoder_depth must be >= 1, got {ed}")
        self.encoder_hidden = eh
        self.encoder_depth = ed
        # Keep ``surface`` as the module name so existing envelope checkpoints load.
        if kind == "surface":
            self.surface = SurfaceEncoder(latent_dim=latent_dim, hidden=hidden)
            self.token_dim = 3
        else:
            self.mesh = MeshFaceEncoder(
                latent_dim=latent_dim, hidden=eh, depth=ed, in_dim=FACE_FEAT_DIM
            )
            self.token_dim = FACE_FEAT_DIM
        head_in = 3 + latent_dim
        if k > 0:
            # Same PointNet block as the global envelope, over k neighbor offsets.
            self.local = SurfaceEncoder(latent_dim=local_dim, hidden=hidden)
            head_in += local_dim
        self.head = build_mlp(head_in, hidden, depth)

    def _geom_module(self) -> nn.Module:
        return self.surface if self.shape_encoder == "surface" else self.mesh

    def encode_unique(self, geom: Tensor, shape_id: Tensor) -> Tensor:
        """
        Encode each distinct ``shape_id`` once and scatter back to ``(B, D)``.

        Parameters
        ----------
        geom, shape_id:
            Batched tokens and integer mesh ids (same length B).
        """
        if shape_id.ndim != 1 or int(shape_id.shape[0]) != int(geom.shape[0]):
            raise ValueError(
                f"shape_id must be (B,), got {tuple(shape_id.shape)} "
                f"for geom {tuple(geom.shape)}"
            )
        unique_ids, inverse = torch.unique(shape_id, sorted=True, return_inverse=True)
        hits = shape_id.unsqueeze(0) == unique_ids.unsqueeze(1)
        first = hits.to(dtype=torch.int64).argmax(dim=1)
        z_unique = self._geom_module()(geom[first])
        return z_unique[inverse]

    def forward(
        self,
        xyz: Tensor,
        geom: Tensor,
        shape_id: Tensor,
    ) -> Tensor:
        """``cat(xyz, z_global[, z_local])`` → occupancy logit."""
        if xyz.ndim != 2 or xyz.shape[-1] != 3:
            raise ValueError(f"xyz must have shape (B, 3), got {tuple(xyz.shape)}")
        if geom.ndim != 3 or geom.shape[-1] != self.token_dim:
            raise ValueError(
                f"geom must have shape (B, K, {self.token_dim}), got {tuple(geom.shape)}"
            )
        if int(xyz.shape[0]) != int(geom.shape[0]):
            raise ValueError(
                f"xyz/geom batch mismatch: {tuple(xyz.shape)} vs {tuple(geom.shape)}"
            )
        ids = shape_id.reshape(-1)
        z_shape = self.encode_unique(geom, ids)
        pieces = [xyz, z_shape]
        if self.knn_k > 0:
            rel = knn_offsets(xyz, geom, self.knn_k)
            pieces.append(self.local(rel))
        return self.head(torch.cat(pieces, dim=-1))
