"""Tests for OccupancyPointDataset + DataLoader."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import trimesh

from dataset import (
    OccupancyPointDataset,
    make_dataloader,
    split_train_test_files,
    split_train_val_indices,
)


def _write_npz(path: Path, n: int = 32) -> None:
    rng = np.random.default_rng(0)
    points = rng.uniform(-2.0, 2.0, size=(n, 3)).astype(np.float32)
    labels = (rng.random(n) > 0.5).astype(np.uint8)
    np.savez(path, points=points, labels=labels)


class OccupancyPointDatasetTests(unittest.TestCase):
    def test_getitem_float32_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.npz"
            _write_npz(path, n=16)
            ds = OccupancyPointDataset(path)
            xyz, y = ds[0]
        self.assertEqual(xyz.dtype, torch.float32)
        self.assertEqual(y.dtype, torch.float32)
        self.assertEqual(tuple(xyz.shape), (3,))
        self.assertEqual(tuple(y.shape), (1,))
        self.assertGreater(ds.scale, 0.0)
        self.assertEqual(tuple(ds.center.shape), (3,))

    def test_dataloader_batch_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.npz"
            _write_npz(path, n=20)
            ds = OccupancyPointDataset(path)
            loader = make_dataloader(ds, batch_size=8, shuffle=True)
            xyz, y = next(iter(loader))
        self.assertEqual(tuple(xyz.shape), (8, 3))
        self.assertEqual(tuple(y.shape), (8, 1))
        self.assertEqual(xyz.dtype, torch.float32)
        self.assertEqual(y.dtype, torch.float32)

    def test_split_train_val_covers_all_points(self) -> None:
        train_idx, val_idx = split_train_val_indices(20, 0.15, seed=1)
        n_val = int(val_idx.numel())
        n_train = int(train_idx.numel())
        self.assertEqual(n_train + n_val, 20)
        self.assertGreaterEqual(n_val, 1)
        combined = torch.cat([train_idx, val_idx])
        self.assertEqual(int(torch.unique(combined).numel()), 20)

    def test_split_train_test_files_holds_out_whole_shapes(self) -> None:
        train_idx, val_idx = split_train_test_files(10, 0.2, seed=1)
        self.assertEqual(int(train_idx.numel()) + int(val_idx.numel()), 10)
        overlap = set(train_idx.tolist()) & set(val_idx.tolist())
        self.assertEqual(len(overlap), 0)

    def test_data_dir_joins_obj_triangles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            obj = root / "box.obj"
            trimesh.creation.box(extents=[2.0, 2.0, 2.0]).export(obj)
            path = root / "sample.npz"
            rng = np.random.default_rng(0)
            points = rng.uniform(-2.0, 2.0, size=(8, 3)).astype(np.float32)
            labels = (rng.random(8) > 0.5).astype(np.uint8)
            np.savez(
                path,
                points=points,
                labels=labels,
                mesh_path=np.asarray("box.obj"),
            )
            ds = OccupancyPointDataset(path, data_dir=root)
            xyz, y = ds[0]
        self.assertEqual(tuple(xyz.shape), (3,))
        self.assertEqual(tuple(y.shape), (1,))
        self.assertIsNotNone(ds.mesh_path)
        self.assertEqual(ds.mesh_path.resolve(), obj.resolve())
        self.assertEqual(ds.vertices.shape[1], 3)
        self.assertEqual(ds.faces.shape[1], 3)
        self.assertEqual(ds.mesh_key, "sample")

    def test_n_surface_returns_envelope_and_shape_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            obj = root / "box.obj"
            trimesh.creation.box(extents=[2.0, 2.0, 2.0]).export(obj)
            path = root / "sample.npz"
            rng = np.random.default_rng(0)
            points = rng.uniform(-2.0, 2.0, size=(8, 3)).astype(np.float32)
            labels = (rng.random(8) > 0.5).astype(np.uint8)
            np.savez(
                path,
                points=points,
                labels=labels,
                mesh_path=np.asarray("box.obj"),
            )
            ds = OccupancyPointDataset(
                path, data_dir=root, n_surface=64, seed=1, shape_id=3
            )
            xyz, y, envelope, shape_id = ds[0]
            loader = make_dataloader(ds, batch_size=4, shuffle=False)
            b_xyz, b_y, b_env, b_id = next(iter(loader))
        self.assertEqual(tuple(xyz.shape), (3,))
        self.assertEqual(tuple(envelope.shape), (64, 3))
        self.assertEqual(int(shape_id.item()), 3)
        self.assertEqual(tuple(b_xyz.shape), (4, 3))
        self.assertEqual(tuple(b_env.shape), (4, 64, 3))
        self.assertEqual(tuple(b_id.shape), (4,))
        self.assertTrue(torch.equal(b_id, torch.tensor([3, 3, 3, 3])))

    def test_mesh_join_keeps_npz_occupancy_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            obj = root / "box.obj"
            trimesh.creation.box(extents=[2.0, 2.0, 2.0]).export(obj)
            path = root / "sample.npz"
            rng = np.random.default_rng(0)
            points = rng.uniform(-2.0, 2.0, size=(8, 3)).astype(np.float32)
            labels = (rng.random(8) > 0.5).astype(np.uint8)
            np.savez(
                path,
                points=points,
                labels=labels,
                mesh_path=np.asarray("box.obj"),
            )
            ds = OccupancyPointDataset(path, data_dir=root, n_surface=50, seed=1)
            _xyz, _y, envelope, _sid = ds[0]
        self.assertEqual(len(ds), 8)
        self.assertEqual(tuple(envelope.shape), (50, 3))


if __name__ == "__main__":
    unittest.main()
