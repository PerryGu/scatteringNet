"""Geometry-conditioned occupancy: query XYZ plus a shape latent.

``OccupancyMLP`` stays xyz-only. ``shape_encoder: surface`` is the
envelope PointNet over ``(N, 6)`` XYZ + face normal (or ``(N, 3)``
for older checkpoints). ``knn_k > 0`` adds per-query nearest-neighbor
features (XYZ offset, plus the neighbor normal when the cloud is 6-D).
``knn_pool`` is ``max`` (PointNet) or ``concat`` (flatten the k rows).
The face-token head (``shape_encoder: mesh``) was removed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from scatteringnet.occupancy_mlp import build_mlp

# Distinct from OccupancyMLP so infer can tell the checkpoint apart.
CHECKPOINT_KIND = "occupancy_encoder"


class SurfaceEncoder(nn.Module):
    """
    Per-point MLP + max-pool (PointNet) over an envelope cloud.

    Shapes
    ------
    envelope: ``(U, N, C)`` unique meshes; ``C`` is 3 (XYZ) or 6 (XYZ+n)
    output:   ``(U, D)``    one latent per mesh
    """

    def __init__(
        self, latent_dim: int = 64, hidden: int = 64, *, in_dim: int = 3
    ) -> None:
        super().__init__()
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")
        if hidden < 1:
            raise ValueError(f"hidden must be >= 1, got {hidden}")
        dim = int(in_dim)
        if dim not in (3, 6):
            raise ValueError(f"in_dim must be 3 or 6, got {dim}")
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.in_dim = dim
        self.point_mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, envelope: Tensor) -> Tensor:
        """Max-pool per-point features → one vector per unique mesh."""
        if envelope.ndim != 3 or envelope.shape[-1] != self.in_dim:
            raise ValueError(
                f"envelope must have shape (U, N, {self.in_dim}), "
                f"got {tuple(envelope.shape)}"
            )
        features = self.point_mlp(envelope)
        return features.max(dim=1).values


def knn_offsets(xyz: Tensor, envelope: Tensor, k: int) -> Tensor:
    """
    Offsets from each query to its ``k`` nearest envelope points.

    Distances use XYZ only (AABB Euclidean). ``k`` is clamped to N.
    If the cloud is ``(B, N, 6)``, each neighbor is
    ``(dx, dy, dz, nx, ny, nz)`` — relative position plus that
    neighbor's stored normal. Query points have no normal.

    Shapes: ``xyz (B, 3)``, ``envelope (B, N, 3|6)`` → ``(B, k, 3|6)``.
    """
    if xyz.ndim != 2 or xyz.shape[-1] != 3:
        raise ValueError(f"xyz must have shape (B, 3), got {tuple(xyz.shape)}")
    if envelope.ndim != 3 or envelope.shape[-1] not in (3, 6):
        raise ValueError(
            f"envelope must have shape (B, N, 3 or 6), got {tuple(envelope.shape)}"
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
    feat = int(envelope.shape[-1])
    # k-NN is position-only; extras (normals) ride along after the gather.
    pos = envelope[..., :3]
    dist = torch.linalg.norm(pos - xyz.unsqueeze(1), dim=-1)
    idx = dist.topk(take, dim=-1, largest=False).indices
    nbrs = torch.gather(envelope, 1, idx.unsqueeze(-1).expand(-1, -1, feat))
    rel_xyz = nbrs[..., :3] - xyz.unsqueeze(1)
    if feat == 3:
        return rel_xyz
    return torch.cat([rel_xyz, nbrs[..., 3:]], dim=-1)


# Local k-NN pool. ``max`` is INSPECT / older best.pt. ``concat`` keeps the
# k neighbor rows so surround vs one-sided is still in the vector.
KNN_POOL_MAX = "max"
KNN_POOL_CONCAT = "concat"
KNN_POOLS = frozenset({KNN_POOL_MAX, KNN_POOL_CONCAT})


class LocalConcatEncoder(nn.Module):
    """
    Flatten ``k`` neighbor rows, then MLP → one local code.

    Max-pool (``SurfaceEncoder``) keeps only the strongest channel across
    neighbors and drops the set of directions. Concat keeps every
    ``(dx, dy, dz[, n])`` so the occupancy head can still tell many-sided
    (inside) from one-sided (air). Neighbors are sorted by offset length
    so the flatten layout is stable.

    Shapes
    ------
    rel: ``(B, take, C)`` from ``knn_offsets`` (``take`` ≤ ``k``)
    out: ``(B, D)``
    """

    def __init__(
        self,
        k: int,
        in_dim: int,
        latent_dim: int,
        hidden: int,
    ) -> None:
        super().__init__()
        kk = int(k)
        dim = int(in_dim)
        if kk < 1:
            raise ValueError(f"k must be >= 1, got {kk}")
        if dim not in (3, 6):
            raise ValueError(f"in_dim must be 3 or 6, got {dim}")
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")
        if hidden < 1:
            raise ValueError(f"hidden must be >= 1, got {hidden}")
        self.k = kk
        self.in_dim = dim
        self.latent_dim = int(latent_dim)
        self.hidden = int(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(kk * dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.latent_dim),
        )

    def forward(self, rel: Tensor) -> Tensor:
        """Pad/truncate to ``k``, sort by |offset|, flatten, MLP."""
        if rel.ndim != 3 or int(rel.shape[-1]) != self.in_dim:
            raise ValueError(
                f"rel must have shape (B, take, {self.in_dim}), "
                f"got {tuple(rel.shape)}"
            )
        batch, take, _ = rel.shape
        # knn_offsets may return fewer than k when N < k.
        if take < self.k:
            pad = rel.new_zeros(batch, self.k - take, self.in_dim)
            rel = torch.cat([rel, pad], dim=1)
        elif take > self.k:
            rel = rel[:, : self.k]
        dist = torch.linalg.norm(rel[..., :3], dim=-1)
        order = dist.argsort(dim=1)
        rel = torch.gather(rel, 1, order.unsqueeze(-1).expand_as(rel))
        return self.mlp(rel.reshape(batch, self.k * self.in_dim))


def knn_pool_from_ckpt(ckpt: dict) -> str:
    """
    Local pool recorded on ``best.pt``.

    Older files omit ``knn_pool``; those graphs are max-pool.
    """
    raw = ckpt.get("knn_pool")
    if raw is None:
        return KNN_POOL_MAX
    kind = str(raw).strip().lower()
    if kind not in KNN_POOLS:
        raise ValueError(f"knn_pool must be max or concat, got {raw!r}")
    return kind


# Catalog trains used YAML seed 1. Old best.pt files omit ``seed``.
DEFAULT_ENVELOPE_SEED = 1


def envelope_seed_from_ckpt(ckpt: dict) -> int:
    """
    Envelope RNG seed this checkpoint was trained with.

    New trains store ``seed`` on ``best.pt``. Older files omit it; do
    not fall back to live YAML (that knob may have changed since train).
    """
    raw = ckpt.get("seed")
    if raw is None:
        return DEFAULT_ENVELOPE_SEED
    return int(raw)


def envelope_dim_from_ckpt(ckpt: dict) -> int:
    """
    Envelope channel count this checkpoint was trained with.

    New trains store ``envelope_dim``. Older XYZ-only ``best.pt`` files
    omit it; the first SurfaceEncoder Linear in-features is then 3.
    """
    raw = ckpt.get("envelope_dim")
    if raw is not None:
        dim = int(raw)
        if dim not in (3, 6):
            raise ValueError(f"envelope_dim must be 3 or 6, got {dim}")
        return dim
    weight = (ckpt.get("state_dict") or {}).get("surface.point_mlp.0.weight")
    if weight is not None:
        dim = int(weight.shape[1])
        if dim in (3, 6):
            return dim
    return 3


class OccupancyEncoder(nn.Module):
    """
    Occupancy logits from query XYZ and an envelope code.

    Unique ``shape_id`` values are encoded **once** per batch, then
    broadcast. ``knn_k > 0`` concatenates a local envelope code.

    Shapes
    ------
    xyz:      ``(B, 3)``
    geom:     ``(B, N, C)`` envelope; ``C`` is ``envelope_dim`` (3 or 6)
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
        knn_k: int = 0,
        knn_local_dim: int | None = None,
        knn_pool: str = KNN_POOL_MAX,
        envelope_dim: int = 6,
    ) -> None:
        super().__init__()
        if hidden < 1:
            raise ValueError(f"hidden must be >= 1, got {hidden}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")
        kind = str(shape_encoder).strip().lower()
        if kind != "surface":
            raise ValueError(
                "OccupancyEncoder only supports shape_encoder='surface' "
                f"(face-token 'mesh' was removed), got {shape_encoder!r}"
            )
        k = int(knn_k)
        if k < 0:
            raise ValueError(f"knn_k must be >= 0, got {k}")
        local_dim = int(knn_local_dim) if knn_local_dim is not None else int(latent_dim)
        if k > 0 and local_dim < 1:
            raise ValueError(f"knn_local_dim must be >= 1, got {local_dim}")
        ed = int(envelope_dim)
        if ed not in (3, 6):
            raise ValueError(f"envelope_dim must be 3 or 6, got {ed}")
        self.hidden = hidden
        self.depth = depth
        self.latent_dim = latent_dim
        self.shape_encoder = kind
        self.knn_k = k
        self.knn_local_dim = local_dim if k > 0 else 0
        pool = str(knn_pool).strip().lower()
        if k > 0 and pool not in KNN_POOLS:
            raise ValueError(f"knn_pool must be max or concat, got {knn_pool!r}")
        # knn_k=0 has no local module; pool is stored as max for a stable ckpt.
        self.knn_pool = pool if k > 0 else KNN_POOL_MAX
        self.envelope_dim = ed
        # Name ``surface`` is load-stable for existing envelope checkpoints.
        self.surface = SurfaceEncoder(latent_dim=latent_dim, hidden=hidden, in_dim=ed)
        self.token_dim = ed
        head_in = 3 + latent_dim
        if k > 0:
            if self.knn_pool == KNN_POOL_CONCAT:
                # Flatten k neighbors — the Fill story (surround vs air).
                self.local = LocalConcatEncoder(
                    k=k, in_dim=ed, latent_dim=local_dim, hidden=hidden
                )
            else:
                # INSPECT / older trains: PointNet max-pool over k.
                self.local = SurfaceEncoder(
                    latent_dim=local_dim, hidden=hidden, in_dim=ed
                )
            head_in += local_dim
        self.head = build_mlp(head_in, hidden, depth)

    def encode_unique(self, geom: Tensor, shape_id: Tensor) -> Tensor:
        """
        Encode each distinct ``shape_id`` once and scatter back to ``(B, D)``.

        Parameters
        ----------
        geom, shape_id:
            Batched envelope clouds and integer mesh ids (same length B).
        """
        if shape_id.ndim != 1 or int(shape_id.shape[0]) != int(geom.shape[0]):
            raise ValueError(
                f"shape_id must be (B,), got {tuple(shape_id.shape)} "
                f"for geom {tuple(geom.shape)}"
            )
        unique_ids, inverse = torch.unique(shape_id, sorted=True, return_inverse=True)
        hits = shape_id.unsqueeze(0) == unique_ids.unsqueeze(1)
        first = hits.to(dtype=torch.int64).argmax(dim=1)
        z_unique = self.surface(geom[first])
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
