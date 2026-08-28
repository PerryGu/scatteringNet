"""Step 6 multi-NPZ occupancy train. Two tiny files, no OccupancyMLP edits."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import OccupancyConfig
from train_multi_npz import train_multi_npz
from train_one_npz import CHECKPOINT_KIND


def _write_npz(path: Path, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    points = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    labels = (np.linalg.norm(points, axis=1) < 0.5).astype(np.uint8)
    np.savez(path, points=points, labels=labels)


def _cpu_cfg(tmp: Path) -> OccupancyConfig:
    return OccupancyConfig(
        data_dir=tmp,
        device=torch.device("cpu"),
        hidden=16,
        depth=2,
        seed=1,
        epochs=2,
        lr=1e-2,
        checkpoint_path=Path("models/one_npz.pt"),
        sample_npz=Path("unused.npz"),
        val_fraction=0.2,
        run_name="test",
        smoke_epochs=2,
        checkpoint_metric="val_acc",
        batch_size=32,
        optimizer="adam",
    )


class TrainMultiNpzTests(unittest.TestCase):
    def test_two_files_train_logs_and_best(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "box_a__occupancy.npz"
            b = root / "box_b__occupancy.npz"
            _write_npz(a, n=40, seed=1)
            _write_npz(b, n=60, seed=2)
            result = train_multi_npz(
                _cpu_cfg(root),
                npz_paths=[a, b],
                epochs=2,
                root=root,
                run_name="test",
            )
            self.assertEqual(result.n_files, 2)
            self.assertEqual(result.n_train + result.n_val, 100)
            self.assertEqual(len(result.losses), 2)
            self.assertTrue(result.best_path.is_file())
            self.assertFalse((result.model_dir / "last.pt").is_file())
            self.assertTrue(result.run_dir.joinpath("metrics.jsonl").is_file())
            lines = result.run_dir.joinpath("metrics.jsonl").read_text(
                encoding="utf-8"
            ).strip().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertFalse(list(result.run_dir.rglob("*.pt")))
            ckpt = torch.load(result.best_path, map_location="cpu", weights_only=False)
            self.assertEqual(ckpt["kind"], CHECKPOINT_KIND)
            self.assertEqual(len(ckpt["npz_paths"]), 2)
            self.assertEqual(len(ckpt["parts"]), 2)
            for rel in ckpt["npz_paths"]:
                self.assertFalse(Path(rel).is_absolute(), rel)
            for part in ckpt["parts"]:
                self.assertFalse(Path(part["npz"]).is_absolute(), part["npz"])
            snap = yaml.safe_load(result.run_dir.joinpath("config.yaml").read_text(encoding="utf-8"))
            self.assertNotIn("catalog_npz_paths", snap)
            self.assertEqual(snap["catalog_file"], "catalog.txt")
            self.assertEqual(snap["catalog_n"], 2)
            self.assertIn("started_at", snap)
            self.assertIn("finished_at", snap)
            self.assertIn("wall", snap)
            self.assertGreater(float(snap["wall_seconds"]), 0.0)
            self.assertEqual(snap["device"], "cpu")
            self.assertIsNone(snap["gpu"])
            self.assertEqual(snap["batch_size"], 32)
            self.assertEqual(snap["optimizer"], "adam")
            catalog_lines = result.run_dir.joinpath("catalog.txt").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(catalog_lines), 2)
            for rel in catalog_lines:
                self.assertFalse(Path(rel).is_absolute(), rel)
            self.assertFalse(Path(snap["checkpoint_path"]).is_absolute())
            row = json.loads(lines[-1])
            self.assertIn("val_acc", row)
            self.assertIn("best_epoch", row)
            # Phase 1 one_npz.pt must not be created in the temp tree as the default out.
            self.assertFalse((root / "one_npz.pt").is_file())


if __name__ == "__main__":
    unittest.main()
