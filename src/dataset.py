"""PyTorch Dataset for one occupancy NPZ.

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
from torch.utils.data import DataLoader, Dataset

from data_npz import (
    load_points_labels,
    load_points_labels_mesh,
    resolve_npz_catalog,
    shape_key,
)
from geometry.mesh_io import load_obj_triangles
from geometry.surface import sample_surface_points
from normalize import apply_normalization, compute_center_scale

DEFAULT_BATCH_SIZE = 1024


class OccupancyPointDataset(Dataset[tuple[Tensor, Tensor]]):
    """
    One scatter NPZ as normalized query points and occupancy labels.

    Item convention: ``xyz`` is ``float32 (3,)`` (collated ``(B, 3)``);
    ``y`` is ``float32 (1,)`` (collated ``(B, 1)``) to match OccupancyMLP logits.
    """

    def __init__(
        self,
        npz_path: Path,
        data_dir: Path | str | None = None,
        *,
        n_surface: int | None = None,
        seed: int = 1,
        shape_id: int = 0,
    ) -> None:
        """
        Load points, compute AABB ``center`` / ``scale``, store CPU tensors.

        When ``data_dir`` is set, also resolve ``mesh_path`` and load OBJ
        triangles. When ``n_surface`` is set, sample an envelope and
        ``__getitem__`` returns ``(xyz, y, envelope, shape_id)``.

        Occupancy queries and labels always come from the NPZ. Envelope
        points are sampled on the OBJ surface (no normal offset).

        Parameters
        ----------
        npz_path:
            Scatter NPZ with ``points (N, 3)`` and ``labels (N,)``.
        data_dir:
            Dataset root. ``None`` skips the OBJ join (unit tests that
            write points-only NPZs).
        n_surface:
            Envelope sample count. ``None`` keeps the xyz-only item.
        seed:
            Envelope RNG.
        shape_id:
            Integer id for encode-once (one per NPZ part).
        """
        npz_path = Path(npz_path)
        if data_dir is not None:
            points, labels, mesh_path = load_points_labels_mesh(npz_path, data_dir)
            vertices, faces = load_obj_triangles(mesh_path)
            self.mesh_path = mesh_path
            self.vertices = vertices
            self.faces = faces
        else:
            points, labels = load_points_labels(npz_path)
            self.mesh_path = None
            self.vertices = None
            self.faces = None
        # Same AABB for NPZ queries and the on-surface envelope.
        center, scale = compute_center_scale(points)
        normed = apply_normalization(points, center, scale)

        # CPU tensors: DataLoader workers copy to GPU in the train step, not here.
        self.xyz = torch.from_numpy(normed)  # (N, 3) float32
        # Keep a trailing singleton so default collate yields (B, 1), not (B,).
        self.y = torch.from_numpy(labels).unsqueeze(1)  # (N, 1) float32
        self.center = center
        self.scale = scale
        self.npz_path = npz_path
        # Same grouping key as the catalog cap (stem before ``__``).
        self.mesh_key = shape_key(npz_path)
        self.shape_id = int(shape_id)
        self.n_surface = n_surface
        self.envelope: Tensor | None = None
        if n_surface is not None:
            if self.vertices is None or self.faces is None or self.mesh_path is None:
                raise ValueError(
                    "n_surface requires a mesh join (pass data_dir and mesh_path)"
                )
            cache_key = str(self.mesh_path.resolve())
            world = sample_surface_points(
                self.vertices,
                self.faces,
                int(n_surface),
                seed=int(seed),
                cache_key=cache_key,
            )
            # Same AABB as query XYZ so the encoder sees one coordinate frame.
            env = apply_normalization(world, center, scale)
            self.envelope = torch.from_numpy(env)

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    def __getitem__(
        self, index: int
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.envelope is not None:
            return (
                self.xyz[index],
                self.y[index],
                self.envelope,
                torch.tensor(self.shape_id, dtype=torch.long),
            )
        return self.xyz[index], self.y[index]


class OccupancyMultiNpzDataset:
    """
    Catalog of per-file occupancy datasets.

    Each NPZ stays its own :class:`OccupancyPointDataset`. Files are **not**
    concatenated into one point cloud. ``len`` is the file count.
    """

    def __init__(
        self,
        npz_paths: Sequence[Path | str],
        data_dir: Path | str | None = None,
        *,
        n_surface: int | None = None,
        seed: int = 1,
    ) -> None:
        """
        Load each NPZ as :class:`OccupancyPointDataset` (no point pooling).

        Parameters
        ----------
        npz_paths:
            Existing occupancy files. Empty list is rejected.
        data_dir:
            Forwarded to each part so the OBJ join runs. ``None`` skips it.
        n_surface:
            Envelope count; ``None`` keeps xyz-only items.
        seed:
            Sample seed (forwarded to each part).
        """
        paths = [Path(p) for p in npz_paths]
        if not paths:
            raise ValueError("OccupancyMultiNpzDataset needs at least one NPZ")
        # One AABB and one reader per file; no ConcatDataset of points.
        parts = [
            OccupancyPointDataset(
                path,
                data_dir=data_dir,
                n_surface=n_surface,
                seed=seed,
                shape_id=i,
            )
            for i, path in enumerate(paths)
        ]
        self.parts = parts
        self.npz_paths = [part.npz_path for part in parts]
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.n_surface = n_surface

    def __len__(self) -> int:
        """Number of files (shapes), not pooled points."""
        return len(self.parts)

    @property
    def n_points(self) -> int:
        """Sum of query counts across files (for logs only)."""
        return sum(len(part) for part in self.parts)

    @property
    def n_meshes(self) -> int:
        """Unique resolved OBJs among parts that completed the join."""
        keys = {
            str(part.mesh_path.resolve())
            for part in self.parts
            if part.mesh_path is not None
        }
        return len(keys)

    @classmethod
    def from_catalog(
        cls,
        data_dir: Path | str,
        *,
        npz_glob: str = "exports/dataset/*.npz",
        npz_paths: Sequence[str | Path] | None = None,
        max_files_per_shape: int | None = 2,
        n_surface: int | None = None,
        seed: int = 1,
    ) -> OccupancyMultiNpzDataset:
        """
        Build from :func:`resolve_npz_catalog`.

        Parameters
        ----------
        data_dir, npz_glob, npz_paths, max_files_per_shape:
            Forwarded to :func:`resolve_npz_catalog`.
        n_surface, seed:
            Envelope sampling; ``None`` keeps xyz-only items.

        Returns
        -------
        OccupancyMultiNpzDataset
            One part per file; ``len`` is the file count.
        """
        catalog = resolve_npz_catalog(
            data_dir,
            npz_glob=npz_glob,
            npz_paths=npz_paths,
            max_files_per_shape=max_files_per_shape,
        )
        return cls(
            catalog,
            data_dir=data_dir,
            n_surface=n_surface,
            seed=seed,
        )


def split_train_test_files(
    n_files: int,
    test_fraction: float,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """
    Hold out whole files for the test set.

    Same math as :func:`split_train_val_indices`. Requires ``n_files >= 2``.
    """
    return split_train_val_indices(n_files, test_fraction, seed)


def split_train_val_files(
    n_files: int,
    val_fraction: float,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Deprecated name for :func:`split_train_test_files`."""
    return split_train_test_files(n_files, val_fraction, seed)


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

    Pass **one** :class:`OccupancyPointDataset` (one file). Do not pass
    the catalog — that would mix shapes.

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
