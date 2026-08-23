"""Tests for single-NPZ overfit training (Step 6)."""

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

from config import OccupancyConfig
from train_one_npz import CHECKPOINT_KIND, train_one_npz


def _write_sphere_npz(path: Path, n: int = 256) -> None:
    """Synthetic unit-ball occupancy: inside if ||xyz|| < 0.5."""
    rng = np.random.default_rng(0)
    points = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    labels = (np.linalg.norm(points, axis=1) < 0.5).astype(np.uint8)
    np.savez(path, points=points, labels=labels)


def _cpu_cfg(
    *,
    epochs: int = 20,
    lr: float = 1e-2,
    checkpoint_path: Path,
) -> OccupancyConfig:
    return OccupancyConfig(
        data_dir=Path("."),
        device=torch.device("cpu"),
        hidden=32,
        depth=2,
        seed=1,
        epochs=epochs,
        lr=lr,
        checkpoint_path=checkpoint_path,
        sample_npz=Path("sphere.npz"),
    )


class TrainOneNpzTests(unittest.TestCase):
    def test_overfit_loss_drops_and_acc_above_chance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            npz_path = Path(tmp) / "sphere.npz"
            out_path = Path(tmp) / "one_npz.pt"
            _write_sphere_npz(npz_path, n=256)
            result = train_one_npz(npz_path, _cpu_cfg(checkpoint_path=out_path))
            self.assertTrue(out_path.is_file())
            self.assertEqual(result.out_path, out_path)
            self.assertEqual(len(result.losses), 20)
            self.assertLess(result.losses[-1], result.losses[0])
            self.assertGreater(result.accuracies[-1], 0.70)

            ckpt = torch.load(out_path, map_location="cpu", weights_only=False)
            self.assertEqual(ckpt["kind"], CHECKPOINT_KIND)
            self.assertIn("state_dict", ckpt)
            self.assertEqual(tuple(np.asarray(ckpt["center"]).shape), (3,))
            self.assertGreater(float(ckpt["scale"]), 0.0)
            self.assertEqual(int(ckpt["hidden"]), 32)
            self.assertEqual(int(ckpt["depth"]), 2)

    def test_rejects_zero_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            npz_path = Path(tmp) / "sphere.npz"
            _write_sphere_npz(npz_path, n=8)
            with self.assertRaises(ValueError):
                train_one_npz(
                    npz_path,
                    _cpu_cfg(epochs=0, checkpoint_path=Path(tmp) / "x.pt"),
                )


if __name__ == "__main__":
    unittest.main()
