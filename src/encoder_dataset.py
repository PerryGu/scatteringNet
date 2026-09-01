"""Query points plus a cached surface envelope and unused face tokens.

``OccupancyPointDataset`` stays xyz + labels (+ optional mesh join).
This reader adds the Step 8 shell cloud used by ``OccupancyEncoder``.
Step 9 face tokens live on the dataset for the next encoder; ``__getitem__``
still returns ``(xyz, y, envelope, shape_id)``.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from dataset import OccupancyEncoderItem, OccupancyPointDataset
from geometry.face_tokens import apply_face_aabb, face_tokens_from_triangles
from geometry.surface import sample_surface_points
from normalize import apply_normalization


class OccupancyEncoderDataset(OccupancyPointDataset):
    """
    One NPZ plus an area-weighted envelope on the joined OBJ.

    ``__getitem__`` returns ``(xyz, y, envelope, shape_id)``.
    """

    def __init__(
        self,
        npz_path: Path,
        data_dir: Path | str,
        *,
        n_surface: int,
        seed: int = 1,
        shape_id: int = 0,
        n_faces: int = 256,
        item_geom: str = "surface",
    ) -> None:
        super().__init__(Path(npz_path), data_dir, shape_id=shape_id)
        if self.vertices is None or self.faces is None or self.mesh_path is None:
            raise ValueError("OccupancyEncoderDataset requires a mesh join")
        count = int(n_surface)
        if count < 1:
            raise ValueError(f"n_surface must be >= 1, got {count}")
        n_tok = int(n_faces)
        if n_tok < 1:
            raise ValueError(f"n_faces must be >= 1, got {n_tok}")
        cache_key = str(self.mesh_path.resolve())
        world = sample_surface_points(
            self.vertices,
            self.faces,
            count,
            seed=int(seed),
            cache_key=cache_key,
        )
        env = apply_normalization(world, self.center, self.scale)
        self.n_surface = count
        self.envelope = torch.from_numpy(env)
        # Tokens are AABB-aligned with queries; OccupancyEncoder does not read them.
        world_tok = face_tokens_from_triangles(
            self.vertices,
            self.faces,
            n_tok,
            cache_key=cache_key,
        )
        self.n_faces = n_tok
        self.face_tokens = torch.from_numpy(
            apply_face_aabb(world_tok, self.center, self.scale)
        )
        kind = str(item_geom).strip().lower()
        if kind not in ("surface", "mesh"):
            raise ValueError(f"item_geom must be 'surface' or 'mesh', got {item_geom!r}")
        self.item_geom = kind

    def __getitem__(self, index: int) -> OccupancyEncoderItem:
        xyz, y = super().__getitem__(index)
        if self.item_geom == "mesh":
            assert self.face_tokens is not None
            geom = self.face_tokens
        else:
            assert self.envelope is not None
            geom = self.envelope
        return (
            xyz,
            y,
            geom,
            torch.tensor(self.shape_id, dtype=torch.long),
        )
