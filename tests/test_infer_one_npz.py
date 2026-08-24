"""Tests for single-NPZ occupancy inference (Step 8)."""

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
from infer_one_npz import infer_one_npz, load_occupancy_checkpoint
from train_one_npz import CHECKPOINT_KIND, train_one_npz


def _write_sphere_npz(path: Path, n: int = 256) -> None:
    rng = np.random.default_rng(0)
    points = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    labels = (np.linalg.norm(points, axis=1) < 0.5).astype(np.uint8)
    np.savez(path, points=points, labels=labels)


def _cpu_cfg(*, checkpoint_path: Path, epochs: int = 20) -> OccupancyConfig:
    return OccupancyConfig(
        data_dir=Path("."),
        device=torch.device("cpu"),
        hidden=32,
        depth=2,
        seed=1,
        epochs=epochs,
        lr=1e-2,
        checkpoint_path=checkpoint_path,
        sample_npz=Path("sphere.npz"),
        val_fraction=0.15,
    )


class InferOneNpzTests(unittest.TestCase):
    def test_infer_matches_overfit_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            npz_path = Path(tmp) / "sphere.npz"
            ckpt_path = Path(tmp) / "one_npz.pt"
            _write_sphere_npz(npz_path, n=256)
            cfg = _cpu_cfg(checkpoint_path=ckpt_path)
            train_one_npz(npz_path, cfg)
            result = infer_one_npz(npz_path, cfg, write_pred=True)

            self.assertEqual(result.n, 256)
            self.assertEqual(result.n_pred_inside + result.n_pred_outside, 256)
            self.assertGreater(result.accuracy, 0.70)
            self.assertIsNotNone(result.pred_path)
            assert result.pred_path is not None
            self.assertTrue(result.pred_path.is_file())
            with np.load(result.pred_path) as pred:
                self.assertIn("pred_labels", pred.files)
                self.assertEqual(pred["pred_labels"].shape, (256,))

    def test_rejects_missing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            npz_path = Path(tmp) / "sphere.npz"
            _write_sphere_npz(npz_path, n=8)
            cfg = _cpu_cfg(checkpoint_path=Path(tmp) / "missing.pt")
            with self.assertRaises(FileNotFoundError):
                infer_one_npz(npz_path, cfg, write_pred=False)

    def test_rejects_wrong_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            npz_path = Path(tmp) / "sphere.npz"
            ckpt_path = Path(tmp) / "bad.pt"
            _write_sphere_npz(npz_path, n=8)
            torch.save({"kind": "not_occupancy_mlp", "state_dict": {}}, ckpt_path)
            cfg = _cpu_cfg(checkpoint_path=ckpt_path)
            with self.assertRaises(ValueError):
                infer_one_npz(npz_path, cfg, write_pred=False)

    def test_load_checkpoint_reads_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            npz_path = Path(tmp) / "sphere.npz"
            ckpt_path = Path(tmp) / "one_npz.pt"
            _write_sphere_npz(npz_path, n=64)
            cfg = _cpu_cfg(checkpoint_path=ckpt_path, epochs=2)
            train_one_npz(npz_path, cfg)
            payload = load_occupancy_checkpoint(ckpt_path, torch.device("cpu"))
            self.assertEqual(payload["kind"], CHECKPOINT_KIND)


if __name__ == "__main__":
    unittest.main()
