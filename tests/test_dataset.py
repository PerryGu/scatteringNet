"""Tests for OccupancyPointDataset + DataLoader (Step 5)."""

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

from dataset import OccupancyPointDataset, make_dataloader


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


if __name__ == "__main__":
    unittest.main()
