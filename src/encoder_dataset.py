"""Query points plus a cached surface envelope.

``OccupancyPointDataset`` stays xyz + labels (+ optional mesh join).
This reader adds the envelope cloud used by ``OccupancyEncoder``.
``__getitem__`` returns ``(xyz, y, envelope, shape_id)``.
"""

from __future__ import annotations

from pathlib import Path

import torch

from scatteringnet.dataset import OccupancyEncoderItem, OccupancyPointDataset
from scatteringnet.geometry.surface import apply_envelope_aabb, sample_surface_points


class OccupancyEncoderDataset(OccupancyPointDataset):
    """
    One NPZ plus a surface envelope on the joined OBJ.

    ``__getitem__`` returns ``(xyz, y, envelope, shape_id)``.
    """

    def __init__(
        self,
        npz_path: Path,
        data_dir: Path | str | None,
        *,
        n_surface: int,
        seed: int = 1,
        shape_id: int = 0,
        load_queries: bool = True,
        sample_envelope: bool = True,
    ) -> None:
        super().__init__(
            Path(npz_path),
            data_dir,
            shape_id=shape_id,
            load_queries=load_queries,
        )
        if self.vertices is None or self.faces is None or self.mesh_path is None:
            raise ValueError("OccupancyEncoderDataset requires a mesh join")
        count = int(n_surface)
        if count < 1:
            raise ValueError(f"n_surface must be >= 1, got {count}")
        self.n_surface = count
        # Catalog binds one tensor per OBJ after the shared AABB is set.
        if not sample_envelope:
            self.envelope = None
            return
        cache_key = str(self.mesh_path.resolve())
        world = sample_surface_points(
            self.vertices,
            self.faces,
            count,
            seed=int(seed),
            cache_key=cache_key,
        )
        env = apply_envelope_aabb(world, self.center, self.scale)
        self.envelope = torch.from_numpy(env)

    def __getitem__(self, index: int) -> OccupancyEncoderItem:
        xyz, y = super().__getitem__(index)
        if self.envelope is None:
            raise RuntimeError("encoder part has no envelope")
        return (
            xyz,
            y,
            self.envelope,
            torch.tensor(self.shape_id, dtype=torch.long),
        )
