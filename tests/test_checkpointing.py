"""Step 5 checkpoints. Dummy weights — no OccupancyMLP, no occupancy train."""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import torch
import yaml

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from checkpointing import Checkpointer
from run_tracking import start_run


class CheckpointingTests(unittest.TestCase):
    def test_best_only_on_strict_improve(self) -> None:
        when = datetime(2026, 8, 27, 18, 0, 0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with start_run("dummy", root=root, created_at=when) as run:
                run.write_config({"seed": 1, "epochs": 3})
                saver = Checkpointer(run, total_epochs=3, root=root)
                # Epoch 2 is the apex; epoch 3 is worse so best.pt stays at 2.
                vals = (0.50, 0.90, 0.60)
                for epoch, val_acc in enumerate(vals, start=1):
                    saver.save(
                        {"w": torch.tensor([float(epoch)])},
                        epoch=epoch,
                        metric=val_acc,
                    )
                    run.log_epoch(
                        epoch=epoch,
                        loss=1.0 / epoch,
                        train_acc=0.5,
                        val_acc=val_acc,
                        wall_seconds=0.01,
                        best_epoch=saver.best_epoch,
                        best_metric=saver.best_metric,
                    )

            self.assertTrue(saver.best_path.is_file())
            self.assertFalse((saver.dir / "last.pt").is_file())
            best = torch.load(saver.best_path, map_location="cpu", weights_only=False)
            self.assertEqual(int(best["epoch"]), 2)
            self.assertEqual(int(best["total"]), 3)
            self.assertEqual(float(best["w"].item()), 2.0)

            snap = yaml.safe_load(run.config_path.read_text(encoding="utf-8"))
            self.assertEqual(snap["total"], 3)
            self.assertEqual(snap["checkpoint"], 2)
            self.assertEqual(snap["best_epoch"], 2)
            self.assertAlmostEqual(float(snap["best_metric"]), 0.90)

            pointer = (run.dir / "checkpoint_dir.txt").read_text(encoding="utf-8").strip()
            self.assertTrue(pointer.replace("\\", "/").endswith(f"models/{run.run_id}"))
            self.assertFalse(list(run.dir.rglob("*.pt")), "runs/ must not contain .pt")


if __name__ == "__main__":
    unittest.main()
