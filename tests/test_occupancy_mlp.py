"""Smoke tests for OccupancyMLP. No NPZ, no training."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

# Allow `from occupancy_mlp import OccupancyMLP` without installing a package.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from occupancy_mlp import OccupancyMLP


class OccupancyMLPTests(unittest.TestCase):
    def test_forward_shape_and_finite_on_cpu(self) -> None:
        model = OccupancyMLP(hidden=64, depth=4)
        model.eval()
        xyz = torch.randn(8, 3)
        with torch.no_grad():
            logits = model(xyz)
        self.assertEqual(tuple(logits.shape), (8, 1))
        self.assertTrue(torch.isfinite(logits).all().item())
        self.assertEqual(logits.device.type, "cpu")

    def test_rejects_bad_xyz_rank(self) -> None:
        model = OccupancyMLP()
        with self.assertRaises(ValueError):
            model(torch.randn(8, 4))
        with self.assertRaises(ValueError):
            model(torch.randn(3))

    def test_rejects_invalid_hyperparams(self) -> None:
        with self.assertRaises(ValueError):
            OccupancyMLP(hidden=0, depth=4)
        with self.assertRaises(ValueError):
            OccupancyMLP(hidden=64, depth=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_forward_on_cuda(self) -> None:
        # Parameters and input must share a device; .cuda() moves the module.
        model = OccupancyMLP(hidden=64, depth=4).cuda()
        model.eval()
        xyz = torch.randn(8, 3, device="cuda")
        with torch.no_grad():
            logits = model(xyz)
        self.assertEqual(tuple(logits.shape), (8, 1))
        self.assertTrue(logits.is_cuda)
        self.assertTrue(torch.isfinite(logits).all().item())


if __name__ == "__main__":
    unittest.main()
