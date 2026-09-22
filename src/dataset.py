"""PyTorch Dataset for one occupancy NPZ.

Catalog parts index the NPZ and optional OBJ without holding every
query tensor. ``ensure_queries`` AABB-normalizes XYZ for the train step.
Item convention (locked here for the rest of the MVP):

- ``xyz``: ``float32`` tensor shape ``(3,)`` → collated ``(B, 3)``
- ``y``:   ``float32`` tensor shape ``(1,)`` → collated ``(B, 1)``
  (matches ``OccupancyMLP`` logits ``(B, 1)`` for BCE-with-logits)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from scatteringnet.data_npz import (
    count_npz_points,
    load_points_labels,
    read_npz_mesh_path,
    resolve_mesh_path,
    resolve_npz_catalog,
    shape_key,
)
from scatteringnet.geometry.mesh_io import load_obj_triangles
from scatteringnet.geometry.surface import apply_envelope_aabb, undo_envelope_aabb
from scatteringnet.normalize import apply_normalization, compute_center_scale

DEFAULT_BATCH_SIZE = 1024

OccupancyItem = tuple[Tensor, Tensor]
OccupancyEncoderItem = tuple[Tensor, Tensor, Tensor, Tensor]


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
        shape_id: int = 0,
        load_queries: bool = True,
    ) -> None:
        """
        Join the OBJ when ``data_dir`` is set. Query tensors are optional.

        ``load_queries=True`` (single-file tests) stores normalized XYZ/y.
        The catalog passes ``False`` and loads one NPZ when that file trains.

        Parameters
        ----------
        npz_path:
            Scatter NPZ with ``points (N, 3)`` and ``labels (N,)``.
        data_dir:
            Dataset root. ``None`` skips the OBJ join (unit tests that
            write points-only NPZs).
        shape_id:
            Integer id for encode-once. The catalog overwrites this with a
            stable id per mesh identity (not per file).
        load_queries:
            When False, only record ``n_queries``; call ``ensure_queries``.
        """
        npz_path = Path(npz_path)
        self.npz_path = npz_path
        self.mesh_key = shape_key(npz_path)
        self.shape_id = int(shape_id)
        self.n_surface: int | None = None
        self.envelope: Tensor | None = None
        self.xyz: Tensor | None = None
        self.y: Tensor | None = None
        self.shared_aabb = False
        if data_dir is not None:
            stored = read_npz_mesh_path(npz_path)
            mesh_path = resolve_mesh_path(stored, data_dir)
            vertices, faces = load_obj_triangles(mesh_path)
            self.mesh_path = mesh_path
            self.vertices = vertices
            self.faces = faces
        else:
            self.mesh_path = None
            self.vertices = None
            self.faces = None
        self.center = np.zeros(3, dtype=np.float32)
        self.scale = 1.0
        if load_queries:
            self.n_queries = -1
            self.ensure_queries()
        else:
            self.n_queries = count_npz_points(npz_path)

    def ensure_queries(self) -> None:
        """Load this file's XYZ/y and AABB-normalize. No-op if already loaded."""
        if self.xyz is not None:
            return
        points, labels = load_points_labels(self.npz_path)
        n = int(points.shape[0])
        if self.n_queries >= 0 and n != int(self.n_queries):
            raise ValueError(
                f"NPZ point count changed: {self.npz_path} had {self.n_queries}, now {n}"
            )
        self.n_queries = n
        if self.shared_aabb:
            center = np.asarray(self.center, dtype=np.float32).reshape(3)
            scale = float(self.scale)
        else:
            # Per-file query AABB. Catalog overwrites via apply_shared_mesh_aabb.
            center, scale = compute_center_scale(points)
            self.center = center
            self.scale = scale
        self.xyz = torch.from_numpy(apply_normalization(points, center, scale))
        self.y = torch.from_numpy(labels).unsqueeze(1)

    def release_queries(self) -> None:
        """Drop XYZ/y tensors. Envelope, AABB, and ``n_queries`` stay."""
        self.xyz = None
        self.y = None

    def __len__(self) -> int:
        return int(self.n_queries)

    def __getitem__(self, index: int) -> OccupancyItem:
        self.ensure_queries()
        assert self.xyz is not None and self.y is not None
        return self.xyz[index], self.y[index]


class OccupancyMultiNpzDataset:
    """
    Catalog of per-file occupancy datasets.

    Each NPZ stays its own :class:`OccupancyPointDataset`. Files are **not**
    concatenated into one point cloud. ``len`` is the file count.
    After load, parts that share a mesh share ``shape_id``, AABB, and (when
    used) one envelope tensor. Query XYZ is loaded per file, not at construct.
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
        Index each NPZ without keeping all query tensors.

        Lattice and jitter of one OBJ stay two parts (two query clouds).
        They share ``shape_id``, AABB, and one envelope tensor.

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
        # Paths + mesh join only. Query XYZ loads when that file trains.
        if n_surface is not None:
            from scatteringnet.encoder_dataset import OccupancyEncoderDataset

            parts = [
                OccupancyEncoderDataset(
                    path,
                    data_dir=data_dir,
                    n_surface=n_surface,
                    seed=seed,
                    load_queries=False,
                    sample_envelope=False,
                )
                for path in paths
            ]
        else:
            parts = [
                OccupancyPointDataset(path, data_dir=data_dir, load_queries=False)
                for path in paths
            ]
        ids = mesh_shape_ids([mesh_split_key(part) for part in parts])
        for part, sid in zip(parts, ids):
            part.shape_id = sid
        # One AABB per mesh: vertices when joined, else union of that key's queries.
        apply_shared_mesh_aabb(parts)
        if n_surface is not None:
            bind_shared_envelopes(parts, n_surface=int(n_surface), seed=int(seed))
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
        npz_catalog: Sequence[tuple[str, int | None]] | None = None,
        max_files_per_shape: int | None = 2,
        n_surface: int | None = None,
        seed: int = 1,
    ) -> OccupancyMultiNpzDataset:
        """
        Build from :func:`resolve_npz_catalog`.

        Parameters
        ----------
        data_dir, npz_glob, npz_paths, npz_catalog, max_files_per_shape:
            Forwarded to :func:`resolve_npz_catalog`.
        n_surface, seed:
            Envelope sampling; ``None`` keeps xyz-only items. ``seed`` also
            drives ``max_shapes`` catalog subsampling.

        Returns
        -------
        OccupancyMultiNpzDataset
            One part per file; ``len`` is the file count.
        """
        catalog = resolve_npz_catalog(
            data_dir,
            npz_glob=npz_glob,
            npz_paths=npz_paths,
            npz_catalog=npz_catalog,
            max_files_per_shape=max_files_per_shape,
            seed=seed,
        )
        return cls(
            catalog,
            data_dir=data_dir,
            n_surface=n_surface,
            seed=seed,
        )


def mesh_split_key(part: OccupancyPointDataset) -> str:
    """
    Identity used to keep every NPZ of one OBJ on the same split side.

    Prefer the resolved OBJ path when the mesh join ran. Fall back to
    ``mesh_key`` (filename stem before ``__``) for points-only parts.
    File-level splits leak: lattice and jitter of the same mesh can sit
    in both train and test.
    """
    if part.mesh_path is not None:
        return str(part.mesh_path.resolve())
    return str(part.mesh_key)


def mesh_shape_ids(keys: Sequence[str]) -> list[int]:
    """
    Stable integer per mesh identity (sorted unique keys → ``0 .. K-1``).

    Two files that share a key get the same id so ``encode_unique`` runs
    once per OBJ, not once per NPZ.
    """
    cleaned = [str(key).strip() for key in keys]
    if any(not text for text in cleaned):
        raise ValueError("mesh shape id key is empty")
    index = {key: i for i, key in enumerate(sorted(set(cleaned)))}
    return [index[text] for text in cleaned]


def _world_xyz(part: OccupancyPointDataset) -> np.ndarray:
    """Undo the part's current AABB map (queries already live as tensors)."""
    if part.xyz is None:
        raise RuntimeError("part has no query tensors")
    xyz = np.asarray(part.xyz.numpy(), dtype=np.float32)
    center = np.asarray(part.center, dtype=np.float32).reshape(3)
    return xyz * np.float32(part.scale) + center


def _world_queries_for_aabb(part: OccupancyPointDataset) -> np.ndarray:
    """World-space query XYZ: undo AABB if loaded, else read the NPZ once."""
    if part.xyz is not None:
        return _world_xyz(part)
    points, _labels = load_points_labels(part.npz_path)
    return points


def _aabb_for_mesh_group(
    group: Sequence[OccupancyPointDataset],
) -> tuple[np.ndarray, float]:
    """Mesh vertices when any part joined an OBJ; else union of query clouds."""
    for part in group:
        if part.vertices is not None:
            return compute_center_scale(part.vertices)
    stacked = np.concatenate([_world_queries_for_aabb(part) for part in group], axis=0)
    return compute_center_scale(stacked)


def apply_shared_mesh_aabb(parts: Sequence[OccupancyPointDataset]) -> None:
    """
    Give every NPZ of one mesh the same ``center`` / ``scale``.

    Lattice and jitter query AABBs differ. The encoder and the occupancy
    head must see one frame per OBJ. Envelope clouds are remapped too.
    Unloaded query tensors stay unloaded; ``ensure_queries`` uses this AABB.
    """
    groups: dict[str, list[OccupancyPointDataset]] = {}
    for part in parts:
        groups.setdefault(mesh_split_key(part), []).append(part)
    for group in groups.values():
        center, scale = _aabb_for_mesh_group(group)
        for part in group:
            if part.xyz is not None:
                world_xyz = _world_xyz(part)
                part.xyz = torch.from_numpy(apply_normalization(world_xyz, center, scale))
            if part.envelope is not None:
                env = np.asarray(part.envelope.numpy(), dtype=np.float32)
                old_c = np.asarray(part.center, dtype=np.float32).reshape(3)
                # Undo/remap XYZ only; unit normals stay directions.
                world_env = undo_envelope_aabb(env, old_c, float(part.scale))
                part.envelope = torch.from_numpy(
                    apply_envelope_aabb(world_env, center, scale)
                )
            part.center = center
            part.scale = scale
            part.shared_aabb = True


def bind_shared_envelopes(
    parts: Sequence[OccupancyPointDataset],
    *,
    n_surface: int,
    seed: int,
) -> None:
    """
    One AABB-normalized envelope tensor per mesh.

    Lattice and jitter NPZs of the same OBJ get the same object, not a copy.
    """
    from scatteringnet.geometry.surface import sample_surface_points

    count = int(n_surface)
    groups: dict[str, list[OccupancyPointDataset]] = {}
    for part in parts:
        groups.setdefault(mesh_split_key(part), []).append(part)
    for group in groups.values():
        first = group[0]
        if first.vertices is None or first.faces is None or first.mesh_path is None:
            raise ValueError("shared envelope requires a mesh join")
        world = sample_surface_points(
            first.vertices,
            first.faces,
            count,
            seed=int(seed),
            cache_key=str(first.mesh_path.resolve()),
        )
        env = torch.from_numpy(apply_envelope_aabb(world, first.center, first.scale))
        for part in group:
            part.n_surface = count
            part.envelope = env


def split_train_test_by_mesh(
    keys: Sequence[str],
    test_fraction: float,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """
    Hold out whole mesh identities, not individual NPZ files.

    ``keys[i]`` is the mesh identity of catalog file ``i``. Every file that
    shares a key goes to train or to test together. ``test_fraction`` applies
    to unique keys (sorted), not to the file count.

    Parameters
    ----------
    keys:
        One identity string per catalog file. Empty strings are rejected.
    test_fraction:
        Hold-out fraction of **unique meshes** in ``(0, 1)``.
    seed:
        RNG seed for the key permutation.

    Returns
    -------
    train_idx, test_idx:
        1-D long tensors of file indices that partition ``0 .. len(keys)-1``.
    """
    if not keys:
        raise ValueError("need at least one file key to split")
    groups: dict[str, list[int]] = {}
    for index, raw in enumerate(keys):
        text = str(raw).strip()
        if not text:
            raise ValueError(f"mesh split key at file {index} is empty")
        groups.setdefault(text, []).append(index)
    unique = sorted(groups)
    n_keys = len(unique)
    if n_keys < 2:
        raise ValueError(
            f"need at least 2 distinct meshes to hold out a mesh, got {n_keys}"
        )
    key_train, key_test = split_train_val_indices(n_keys, test_fraction, seed)
    train_files: list[int] = []
    test_files: list[int] = []
    for ki in key_train.tolist():
        train_files.extend(groups[unique[int(ki)]])
    for ki in key_test.tolist():
        test_files.extend(groups[unique[int(ki)]])
    # Stable file order on each side; train still shuffles files each epoch.
    train_files.sort()
    test_files.sort()
    return (
        torch.tensor(train_files, dtype=torch.long),
        torch.tensor(test_files, dtype=torch.long),
    )


def split_train_test_files(
    n_files: int,
    val_fraction: float,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """
    Hold out whole files. Same math as :func:`split_train_val_indices``.

    Catalog train uses :func:`split_train_test_by_mesh` instead so two NPZs
    of one OBJ cannot straddle the split.
    """
    return split_train_val_indices(n_files, val_fraction, seed)


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


def occupancy_collate(
    batch: list[OccupancyItem | OccupancyEncoderItem],
) -> OccupancyItem | OccupancyEncoderItem:
    """
    Collate xyz/y, and envelopes without stacking B independent copies.

    One-file loaders share a shape_id: expand one ``(N, 3)`` cloud to
    ``(B, N, 3)`` as a view. Mixed ids gather from the unique clouds.
    """
    first = batch[0]
    xyz = torch.stack([item[0] for item in batch], dim=0)
    y = torch.stack([item[1] for item in batch], dim=0)
    if len(first) == 2:
        return xyz, y
    shape_id = torch.stack(
        [
            item[3].reshape(()) if item[3].ndim > 0 else item[3]
            for item in batch
        ],
        dim=0,
    )
    if int(shape_id.min()) == int(shape_id.max()):
        # expand() is a view (one cloud, B aliases). pin_memory cannot pin
        # overlapping storage — CUDA catalog train hits this every batch.
        envelope = first[2].unsqueeze(0).expand(len(batch), -1, -1).contiguous()
        return xyz, y, envelope, shape_id
    env_by_id: dict[int, Tensor] = {}
    for item in batch:
        sid = int(item[3].reshape(()).item())
        if sid not in env_by_id:
            env_by_id[sid] = item[2]
    unique_ids, inverse = torch.unique(shape_id, sorted=True, return_inverse=True)
    unique_env = torch.stack([env_by_id[int(u)] for u in unique_ids.tolist()], dim=0)
    return xyz, y, unique_env[inverse].contiguous(), shape_id


def make_dataloader(
    dataset: Dataset[OccupancyItem] | Dataset[OccupancyEncoderItem],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = True,
    pin_memory: bool = False,
) -> DataLoader:
    """
    Train-style loader: custom occupancy collate, no extra workers.

    Pass **one** file dataset (queries already in RAM or loaded by
    ``ensure_queries``). Do not pass the catalog — that would mix shapes.

    Parameters
    ----------
    dataset:
        Point dataset, encoder dataset, or a ``Subset``.
    batch_size:
        Must be ``>= 1``. Default :data:`DEFAULT_BATCH_SIZE`.
    shuffle:
        Shuffle each epoch (True for train, False for val).
    pin_memory:
        Set True on CUDA so host tensors pin for the H2D copy.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=bool(pin_memory),
        collate_fn=occupancy_collate,
    )


if __name__ == "__main__":
    from scatteringnet.config import load_config

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
