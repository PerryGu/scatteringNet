"""Run logs. Metrics only — no OccupancyMLP, no ``.pt``."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import torch
import yaml

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import OccupancyConfig
from run_tracking import format_duration, occupancy_config_snapshot, start_run


def _cpu_cfg() -> OccupancyConfig:
    return OccupancyConfig(
        data_dir=Path("."),
        device=torch.device("cpu"),
        hidden=32,
        depth=2,
        seed=1,
        epochs=3,
        lr=1e-3,
        test_fraction=0.2,
        run_name="dummy",
    )


def _tfevent_files(run_dir: Path) -> list[Path]:
    return list(run_dir.glob("events.out.tfevents*"))


class RunTrackingTests(unittest.TestCase):
    def test_format_duration(self) -> None:
        self.assertEqual(format_duration(0.4), "0.40s")
        self.assertEqual(format_duration(75.0), "1m 15.00s")
        self.assertEqual(format_duration(3661.2), "1h 01m 01.20s")

    def test_fake_three_epoch_loop_writes_logs_not_pt(self) -> None:
        when = datetime(2026, 8, 27, 16, 0, 0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with start_run("dummy", root=root, created_at=when) as run:
                self.assertEqual(run.run_id, "2026-08-27_16-00-00_dummy")
                self.assertTrue(run.dir.is_dir())
                snap = occupancy_config_snapshot(_cpu_cfg())
                run.write_config(snap)
                run.write_catalog(
                    ["exports/a.npz", "exports/b.npz"],
                    data_dir=Path("."),
                )
                for epoch in range(3):
                    run.log_epoch(
                        epoch=epoch,
                        loss=1.0 / (epoch + 1),
                        train_acc=0.5 + 0.1 * epoch,
                        val_acc=0.4 + 0.1 * epoch,
                        wall_seconds=0.01 * (epoch + 1),
                    )

            self.assertTrue(run.config_path.is_file())
            dumped = yaml.safe_load(run.config_path.read_text(encoding="utf-8"))
            self.assertEqual(dumped["run_id"], run.run_id)
            self.assertEqual(dumped["device"], "cpu")
            self.assertIsNone(dumped["gpu"])
            self.assertEqual(dumped["seed"], 1)
            self.assertNotIn("catalog_npz_paths", dumped)
            self.assertNotIn("npz_paths", dumped)
            self.assertEqual(dumped["catalog_file"], "catalog.txt")
            self.assertEqual(dumped["catalog_n"], 2)
            self.assertNotIn("checkpoint_path", dumped)
            self.assertNotIn("sample_npz", dumped)
            self.assertIn("started_at", dumped)
            self.assertIn("finished_at", dumped)
            self.assertIn("wall", dumped)
            self.assertGreaterEqual(float(dumped["wall_seconds"]), 0.0)
            catalog_lines = run.catalog_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(catalog_lines, ["exports/a.npz", "exports/b.npz"])

            lines = run.metrics_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 3)
            rows = [json.loads(line) for line in lines]
            self.assertEqual([row["epoch"] for row in rows], [0, 1, 2])
            self.assertIn("loss", rows[0])
            self.assertIn("train_acc", rows[0])
            self.assertIn("val_acc", rows[0])
            self.assertIn("wall_seconds", rows[0])

            self.assertTrue(_tfevent_files(run.dir), "TensorBoard event file missing")
            self.assertFalse(list(run.dir.rglob("*.pt")), "runs/ must not contain .pt")
            self.assertFalse(list(root.joinpath("runs").rglob("*.pt")))

    def test_sanitize_and_no_overwrite(self) -> None:
        when = datetime(2026, 8, 27, 16, 0, 1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = start_run("has a space", root=root, created_at=when)
            first.close()
            second = start_run("has a space", root=root, created_at=when)
            second.close()
            self.assertEqual(first.run_id, "2026-08-27_16-00-01_has_a_space")
            self.assertEqual(second.run_id, "2026-08-27_16-00-01_has_a_space_1")
            self.assertTrue(first.dir.is_dir())
            self.assertTrue(second.dir.is_dir())

    def test_snapshot_strips_absolute_data_and_repo_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            exports = data / "exports" / "dataset"
            exports.mkdir(parents=True)
            abs_a = (exports / "sphere__occupancy.npz").resolve()
            abs_a.write_bytes(b"")
            cfg = OccupancyConfig(
                data_dir=data.resolve(),
                device=torch.device("cpu"),
                hidden=32,
                depth=2,
                seed=1,
                epochs=3,
                lr=1e-3,
                test_fraction=0.2,
            )
            snap = occupancy_config_snapshot(cfg)
            self.assertNotIn("catalog_npz_paths", snap)
            self.assertNotIn("checkpoint_path", snap)
            self.assertNotIn("sample_npz", snap)
            self.assertIsNone(snap["gpu"])
            # data_dir outside this repo → absolute POSIX fallback.
            self.assertTrue(str(snap["data_dir"]).replace("\\", "/").endswith("/data"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_snapshot_records_cuda_gpu_name(self) -> None:
        cfg = replace(_cpu_cfg(), device=torch.device("cuda"))
        snap = occupancy_config_snapshot(cfg)
        self.assertEqual(snap["device"], "cuda")
        self.assertIsInstance(snap["gpu"], str)
        self.assertTrue(snap["gpu"])


if __name__ == "__main__":
    unittest.main()
