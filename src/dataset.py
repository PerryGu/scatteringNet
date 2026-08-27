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
from typing import Sequence

import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from data_npz import load_points_labels, resolve_npz_catalog
from normalize import apply_normalization, compute_center_scale

DEFAULT_BATCH_SIZE = 1024


class OccupancyPointDataset(Dataset[tuple[Tensor, Tensor]]):
    """
    One scatter NPZ as normalized query points and occupancy labels.

    Item convention: ``xyz`` is ``float32 (3,)`` (collated ``(B, 3)``);
    ``y`` is ``float32 (1,)`` (collated ``(B, 1)``) to match OccupancyMLP logits.
    """

    def __init__(self, npz_path: Path) -> None:
        """
        Load points, compute AABB ``center`` / ``scale``, store CPU tensors.

        Parameters
        ----------
        npz_path:
            Scatter NPZ with ``points (N, 3)`` and ``labels (N,)``.
        """
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


class OccupancyMultiNpzDataset(ConcatDataset[tuple[Tensor, Tensor]]):
    """
    Several occupancy NPZs, each AABB-normalized on its own mesh.

    Item convention matches :class:`OccupancyPointDataset`: ``xyz (3,)``,
    ``y (1,)``. Per-file ``center`` / ``scale`` stay on ``.parts[i]``.
    """

    def __init__(self, npz_paths: Sequence[Path | str]) -> None:
        """
        Load each NPZ as :class:`OccupancyPointDataset` and concatenate.

        Parameters
        ----------
        npz_paths:
            Existing occupancy files. Empty list is rejected.
        """
        paths = [Path(p) for p in npz_paths]
        if not paths:
            raise ValueError("OccupancyMultiNpzDataset needs at least one NPZ")
        # One AABB per file so later mesh join can reuse the same map.
        parts = [OccupancyPointDataset(path) for path in paths]
        super().__init__(parts)
        self.parts = parts
        self.npz_paths = [part.npz_path for part in parts]

    @classmethod
    def from_catalog(
        cls,
        data_dir: Path | str,
        *,
        npz_glob: str = "exports/dataset/*.npz",
        npz_paths: Sequence[str | Path] | None = None,
        max_files_per_shape: int | None = 2,
    ) -> OccupancyMultiNpzDataset:
        """
        Build from :func:`resolve_npz_catalog`.

        Parameters
        ----------
        data_dir, npz_glob, npz_paths, max_files_per_shape:
            Forwarded to :func:`resolve_npz_catalog`.

        Returns
        -------
        OccupancyMultiNpzDataset
            Pooled queries; ``len`` is the sum of file lengths.
        """
        catalog = resolve_npz_catalog(
            data_dir,
            npz_glob=npz_glob,
            npz_paths=npz_paths,
            max_files_per_shape=max_files_per_shape,
        )
        return cls(catalog)


def split_train_val_indices(
    n_points: int,
    val_fraction: float,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """
    Random disjoint train/val index tensors for one NPZ.

    Val size is ``round(n * val_fraction)``, at least 1 and at most n-1,
    so both splits stay non-empty. Same ``seed`` → same split.

    Parameters
    ----------
    n_points:
        Total points in the NPZ (must be ``>= 2``).
    val_fraction:
        Hold-out fraction in ``(0, 1)``.
    seed:
        RNG seed for ``torch.randperm``.

    Returns
    -------
    train_idx, val_idx:
        1-D long tensors that partition ``0 .. n_points-1``.
    """
    if n_points < 2:
        raise ValueError(f"need at least 2 points to split, got {n_points}")
    if val_fraction <= 0.0 or val_fraction >= 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    n_val = int(round(n_points * val_fraction))
    n_val = min(max(n_val, 1), n_points - 1)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    perm = torch.randperm(n_points, generator=generator)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


def make_dataloader(
    dataset: Dataset[tuple[Tensor, Tensor]],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = True,
) -> DataLoader:
    """
    Train-style loader: shuffle on, default collate, no extra workers.

    Accepts ``OccupancyPointDataset``, :class:`OccupancyMultiNpzDataset`,
    or a ``Subset`` of either (Phase 1 split).

    Parameters
    ----------
    dataset:
        Point dataset or a ``Subset``.
    batch_size:
        Must be ``>= 1``. Default :data:`DEFAULT_BATCH_SIZE`.
    shuffle:
        Shuffle each epoch (True for train, False for val).

    Returns
    -------
    DataLoader
        ``num_workers=0``.
    """
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
