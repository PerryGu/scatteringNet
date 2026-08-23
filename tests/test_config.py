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

from config import OccupancyConfig, format_config, load_config, load_yaml_knobs


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

    def test_load_config_resolves_data_dir_and_device(self) -> None:
        cfg = load_config()
        self.assertIsInstance(cfg, OccupancyConfig)
        self.assertTrue(cfg.data_dir.is_dir())
        self.assertIsInstance(cfg.device, torch.device)
        self.assertIn(cfg.device.type, ("cuda", "cpu"))
        rendered = format_config(cfg)
        self.assertIn("data_dir=", rendered)
        self.assertIn("device=", rendered)
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
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaises(FileNotFoundError):
                load_config(yaml_path)


if __name__ == "__main__":
    unittest.main()
