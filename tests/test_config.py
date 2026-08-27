"""Smoke tests for OccupancyConfig. No NPZ I/O."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import OccupancyConfig, format_config, load_config, load_yaml_knobs, sample_npz_path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_YAML = _REPO_ROOT / "config.yaml"


class OccupancyConfigTests(unittest.TestCase):
    def test_yaml_knobs_match_file(self) -> None:
        knobs = load_yaml_knobs(_YAML)
        disk = yaml.safe_load(_YAML.read_text(encoding="utf-8"))
        self.assertEqual(knobs["hidden"], int(disk["hidden"]))
        self.assertEqual(knobs["depth"], int(disk["depth"]))
        self.assertEqual(knobs["seed"], int(disk["seed"]))
        self.assertEqual(knobs["data_dir"], str(disk["data_dir"]).strip())
        self.assertEqual(knobs["epochs"], int(disk["epochs"]))
        self.assertAlmostEqual(knobs["lr"], float(disk["lr"]))
        self.assertEqual(knobs["checkpoint_path"], str(disk["checkpoint_path"]).strip())
        self.assertEqual(knobs["sample_npz"], str(disk["sample_npz"]).strip())
        self.assertAlmostEqual(knobs["val_fraction"], float(disk["val_fraction"]))
        self.assertEqual(knobs["run_name"], str(disk["run_name"]).strip())
        self.assertEqual(knobs["checkpoint_metric"], str(disk["checkpoint_metric"]).strip())

    def test_load_config_resolves_data_dir_and_device(self) -> None:
        cfg = load_config()
        self.assertIsInstance(cfg, OccupancyConfig)
        self.assertTrue(cfg.data_dir.is_dir())
        self.assertIsInstance(cfg.device, torch.device)
        self.assertIn(cfg.device.type, ("cuda", "cpu"))
        self.assertGreaterEqual(cfg.epochs, 1)
        self.assertGreater(cfg.lr, 0.0)
        self.assertTrue(str(cfg.checkpoint_path))
        self.assertEqual(sample_npz_path(cfg), cfg.data_dir / cfg.sample_npz)
        rendered = format_config(cfg)
        self.assertIn("data_dir=", rendered)
        self.assertIn("device=", rendered)
        self.assertIn("epochs=", rendered)
        self.assertIn("lr=", rendered)
        self.assertIn("val_fraction=", rendered)
        self.assertIn("npz_glob=", rendered)
        self.assertIn("run_name=", rendered)
        self.assertIn("checkpoint_metric=", rendered)
        print("\n" + rendered)

    def test_missing_data_dir_prints_and_raises(self) -> None:
        missing = Path(tempfile.gettempdir()) / "scatteringnet_v2_missing_data_dir"
        if missing.exists():
            self.skipTest(f"unexpected existing path: {missing}")
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "config.yaml"
            yaml_path.write_text(
                "\n".join(
                    [
                        "hidden: 64",
                        "depth: 4",
                        "seed: 1",
                        f'data_dir: "{missing.as_posix()}"',
                        "epochs: 30",
                        "lr: 0.001",
                        "checkpoint_path: models/one_npz.pt",
                        "sample_npz: exports/dataset_test/sphere.npz",
                        "val_fraction: 0.15",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaises(FileNotFoundError):
                load_config(yaml_path)


if __name__ == "__main__":
    unittest.main()
