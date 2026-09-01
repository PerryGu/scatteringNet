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

from config import (
    OccupancyConfig,
    as_data_relative,
    as_repo_relative,
    format_config,
    gpu_name,
    load_config,
    load_yaml_knobs,
)


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
        self.assertAlmostEqual(knobs["val_fraction"], float(disk["val_fraction"]))
        self.assertIsNone(knobs["latent_dim"])
        self.assertEqual(knobs["run_name"], str(disk["run_name"]).strip())
        self.assertEqual(knobs["checkpoint_metric"], str(disk["checkpoint_metric"]).strip())
        self.assertEqual(knobs["batch_size"], int(disk["batch_size"]))
        self.assertEqual(knobs["optimizer"], str(disk["optimizer"]).strip().lower())
        self.assertEqual(knobs["n_surface"], int(disk["n_surface"]))
        self.assertEqual(knobs["n_faces"], int(disk["n_faces"]))
        self.assertEqual(knobs["encoder_hidden"], int(disk["encoder_hidden"]))
        self.assertEqual(knobs["encoder_depth"], int(disk["encoder_depth"]))
        self.assertEqual(
            knobs["shape_encoder"], str(disk["shape_encoder"]).strip().lower()
        )

    def test_load_config_resolves_data_dir_and_device(self) -> None:
        cfg = load_config()
        self.assertIsInstance(cfg, OccupancyConfig)
        self.assertTrue(cfg.data_dir.is_dir())
        self.assertIsInstance(cfg.device, torch.device)
        self.assertIn(cfg.device.type, ("cuda", "cpu"))
        self.assertGreaterEqual(cfg.epochs, 1)
        self.assertGreater(cfg.lr, 0.0)
        rendered = format_config(cfg)
        self.assertIn("data_dir=", rendered)
        self.assertIn("device=", rendered)
        self.assertIn("gpu=", rendered)
        self.assertIn("epochs=", rendered)
        self.assertIn("lr=", rendered)
        self.assertIn("val_fraction=", rendered)
        self.assertIn("latent_dim=", rendered)
        self.assertIn("npz_glob=", rendered)
        self.assertIn("run_name=", rendered)
        self.assertIn("checkpoint_metric=", rendered)
        self.assertIn("batch_size=", rendered)
        self.assertIn("optimizer=", rendered)
        self.assertIn("n_surface=", rendered)
        self.assertIn("n_faces=", rendered)
        self.assertIn("encoder_hidden=", rendered)
        self.assertIn("encoder_depth=", rendered)
        self.assertIn("shape_encoder=", rendered)
        self.assertIsNone(gpu_name(torch.device("cpu")))
        name = gpu_name(cfg.device)
        if cfg.device.type == "cuda":
            self.assertIsInstance(name, str)
            self.assertTrue(name)
        else:
            self.assertIsNone(name)
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
                        "test_fraction: 0.15",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaises(FileNotFoundError):
                load_config(yaml_path)

    def test_as_data_and_repo_relative(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        self.assertEqual(as_repo_relative("models/best.pt"), "models/best.pt")
        self.assertEqual(
            as_repo_relative(repo / "models" / "best.pt"),
            "models/best.pt",
        )
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            target = data / "exports" / "dataset" / "a.npz"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"")
            self.assertEqual(
                as_data_relative(target, data),
                "exports/dataset/a.npz",
            )
            self.assertEqual(
                as_data_relative("exports/dataset/a.npz", data),
                "exports/dataset/a.npz",
            )

    def test_unknown_optimizer_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "config.yaml"
            yaml_path.write_text(
                "\n".join(
                    [
                        "hidden: 64",
                        "depth: 4",
                        "seed: 1",
                        f'data_dir: "{Path(tmp).as_posix()}"',
                        "epochs: 30",
                        "lr: 0.001",
                        "test_fraction: 0.2",
                        "optimizer: human",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_yaml_knobs(yaml_path)

    def test_legacy_test_fraction_and_latent_dim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "config.yaml"
            yaml_path.write_text(
                "\n".join(
                    [
                        "hidden: 32",
                        "depth: 2",
                        "seed: 1",
                        f'data_dir: "{Path(tmp).as_posix()}"',
                        "epochs: 2",
                        "lr: 0.001",
                        "test_fraction: 0.25",
                        "checkpoint_metric: test_acc",
                        "latent_dim: 16",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            knobs = load_yaml_knobs(yaml_path)
            self.assertAlmostEqual(knobs["val_fraction"], 0.25)
            self.assertEqual(knobs["checkpoint_metric"], "val_acc")
            self.assertEqual(knobs["latent_dim"], 16)


if __name__ == "__main__":
    unittest.main()
