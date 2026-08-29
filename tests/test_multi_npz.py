"""Multi-NPZ catalog and dataset. No occupancy training."""

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

from data_npz import resolve_npz_catalog, shape_key
from dataset import OccupancyMultiNpzDataset, make_dataloader


def _write_npz(path: Path, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    points = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    labels = (rng.random(n) > 0.5).astype(np.uint8)
    np.savez(path, points=points, labels=labels)


class MultiNpzCatalogTests(unittest.TestCase):
    def test_two_files_pool_point_count_and_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "box_a__occupancy_s0.15_inout.npz"
            b = root / "box_b__occupancy_s0.15_inout.npz"
            _write_npz(a, n=10, seed=1)
            _write_npz(b, n=15, seed=2)
            paths = resolve_npz_catalog(
                root,
                npz_glob="*.npz",
                max_files_per_shape=None,
            )
            self.assertEqual(len(paths), 2)
            ds = OccupancyMultiNpzDataset(paths)
            self.assertEqual(len(ds), 25)
            xyz, y = ds[0]
            self.assertEqual(tuple(xyz.shape), (3,))
            self.assertEqual(tuple(y.shape), (1,))
            loader = make_dataloader(ds, batch_size=8, shuffle=False)
            batch_xyz, batch_y = next(iter(loader))
            self.assertEqual(tuple(batch_xyz.shape), (8, 3))
            self.assertEqual(tuple(batch_y.shape), (8, 1))
            self.assertEqual(batch_xyz.dtype, torch.float32)
            self.assertEqual(len(ds.parts), 2)
            self.assertEqual(len(ds.npz_paths), 2)

    def test_max_files_per_shape_caps_same_mesh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_npz(root / "sphere__occupancy_s0.15_inout.npz", n=4, seed=1)
            _write_npz(root / "sphere__occupancy_s0.15_j0.04_inout.npz", n=4, seed=2)
            _write_npz(root / "cube__occupancy_s0.15_inout.npz", n=4, seed=3)
            paths = resolve_npz_catalog(
                root,
                npz_glob="*.npz",
                max_files_per_shape=1,
            )
            keys = [shape_key(p) for p in paths]
            self.assertEqual(len(paths), 2)
            self.assertEqual(set(keys), {"sphere", "cube"})

    def test_excludes_combo_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_npz(root / "sphere__occupancy_s0.15_inout.npz", n=4, seed=1)
            _write_npz(root / "combo_ab__occupancy_s0.15_inout.npz", n=4, seed=2)
            paths = resolve_npz_catalog(root, npz_glob="*.npz", max_files_per_shape=None)
            self.assertEqual(len(paths), 1)
            self.assertEqual(paths[0].name, "sphere__occupancy_s0.15_inout.npz")

    def test_explicit_list_and_from_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "a.npz"
            b = root / "b.npz"
            _write_npz(a, n=5, seed=1)
            _write_npz(b, n=7, seed=2)
            ds = OccupancyMultiNpzDataset.from_catalog(
                root,
                npz_paths=["a.npz", "b.npz"],
                max_files_per_shape=None,
            )
            self.assertEqual(len(ds), 12)

    def test_empty_catalog_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                resolve_npz_catalog(Path(tmp), npz_glob="*.npz")


if __name__ == "__main__":
    unittest.main()
