"""Tests for occupancy inference on one NPZ against a catalog checkpoint."""

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
from occupancy_mlp import CHECKPOINT_KIND
from train_multi_npz import train_multi_npz


def _write_sphere_npz(path: Path, n: int = 256) -> None:
    rng = np.random.default_rng(0)
    points = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    labels = (np.linalg.norm(points, axis=1) < 0.5).astype(np.uint8)
    np.savez(path, points=points, labels=labels)


def _cpu_cfg(root: Path, *, epochs: int = 20) -> OccupancyConfig:
    return OccupancyConfig(
        data_dir=root,
        device=torch.device("cpu"),
        hidden=32,
        depth=2,
        seed=1,
        epochs=epochs,
        lr=1e-2,
        checkpoint_path=root / "unused.pt",
        sample_npz=Path("sphere.npz"),
        val_fraction=0.15,
        run_name="infer",
        batch_size=64,
        optimizer="adam",
    )


class InferOneNpzTests(unittest.TestCase):
    def test_infer_matches_overfit_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            npz_path = root / "sphere.npz"
            _write_sphere_npz(npz_path, n=256)
            cfg = _cpu_cfg(root, epochs=20)
            trained = train_multi_npz(
                cfg,
                npz_paths=[npz_path],
                epochs=20,
                root=root,
                run_name="infer",
            )
            result = infer_one_npz(
                npz_path, cfg, ckpt_path=trained.best_path, write_pred=True
            )

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
            root = Path(tmp)
            npz_path = root / "sphere.npz"
            _write_sphere_npz(npz_path, n=8)
            cfg = _cpu_cfg(root)
            with self.assertRaises(FileNotFoundError):
                infer_one_npz(
                    npz_path,
                    cfg,
                    ckpt_path=root / "missing.pt",
                    write_pred=False,
                )

    def test_rejects_wrong_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            npz_path = root / "sphere.npz"
            ckpt_path = root / "bad.pt"
            _write_sphere_npz(npz_path, n=8)
            torch.save({"kind": "not_occupancy_mlp", "state_dict": {}}, ckpt_path)
            cfg = _cpu_cfg(root)
            with self.assertRaises(ValueError):
                infer_one_npz(npz_path, cfg, ckpt_path=ckpt_path, write_pred=False)

    def test_load_checkpoint_reads_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            npz_path = root / "sphere.npz"
            _write_sphere_npz(npz_path, n=64)
            cfg = _cpu_cfg(root, epochs=2)
            trained = train_multi_npz(
                cfg,
                npz_paths=[npz_path],
                epochs=2,
                root=root,
                run_name="infer",
            )
            payload = load_occupancy_checkpoint(
                trained.best_path, torch.device("cpu")
            )
            self.assertEqual(payload["kind"], CHECKPOINT_KIND)


if __name__ == "__main__":
    unittest.main()
