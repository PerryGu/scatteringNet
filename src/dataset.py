"""PyTorch Dataset for one occupancy NPZ (Step 5).

Loads ``points`` / ``labels`` via ``data_npz``, AABB-normalizes XYZ in
``__init__``, and stores ``center`` / ``scale`` for later checkpointing.

Item convention (locked here for the rest of the MVP):

- ``xyz``: ``float32`` tensor shape ``(3,)`` → collated ``(B, 3)``
- ``y``:   ``float32`` tensor shape ``(1,)`` → collated ``(B, 1)``
  (matches ``OccupancyMLP`` logits ``(B, 1)`` for BCE-with-logits)
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from data_npz import load_points_labels
from normalize import apply_normalization, compute_center_scale

DEFAULT_BATCH_SIZE = 1024


class OccupancyPointDataset(Dataset[tuple[Tensor, Tensor]]):
    """One scatter NPZ as normalized query points and occupancy labels."""

    def __init__(self, npz_path: Path) -> None:
        points, labels = load_points_labels(Path(npz_path))
        center, scale = compute_center_scale(points)
        normed = apply_normalization(points, center, scale)

        # CPU tensors: DataLoader workers copy to GPU in the train step, not here.
        self.xyz = torch.from_numpy(normed)  # (N, 3) float32
        # Keep a trailing singleton so default collate yields (B, 1), not (B,).
        self.y = torch.from_numpy(labels).unsqueeze(1)  # (N, 1) float32
        self.center = center
        self.scale = scale
        self.npz_path = Path(npz_path)

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.xyz[index], self.y[index]


def make_dataloader(
    dataset: OccupancyPointDataset,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = True,
) -> DataLoader:
    """Train-style loader: shuffle on, default collate, no extra workers."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


if __name__ == "__main__":
    from config import load_config

    sample = (
        load_config().data_dir
        / "exports"
        / "dataset_test"
        / "sphere__raycast_z_raut_s0.15_inout.npz"
    )
    ds = OccupancyPointDataset(sample)
    loader = make_dataloader(ds, batch_size=DEFAULT_BATCH_SIZE, shuffle=True)
    xyz, y = next(iter(loader))
    print(f"file={sample}")
    print(f"N={len(ds)} center={ds.center.tolist()} scale={ds.scale:.6f}")
    print(f"xyz.shape={tuple(xyz.shape)} xyz.dtype={xyz.dtype}")
    print(f"y.shape={tuple(y.shape)} y.dtype={y.dtype}")
