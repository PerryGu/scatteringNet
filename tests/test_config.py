"""Smoke tests for OccupancyConfig. No NPZ I/O."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml


from scatteringnet.config import (
    OccupancyConfig,
    as_data_relative,
    as_repo_relative,
    format_config,
    get_device,
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
        self.assertNotIn("envelope_mix", knobs)
        self.assertNotIn("envelope_mix", disk)
        self.assertEqual(knobs["knn_k"], int(disk["knn_k"]))
        self.assertEqual(len(knobs["npz_catalog"]), len(disk["npz_catalog"]))
        self.assertEqual(knobs["npz_catalog"][-1][0], disk["npz_catalog"][-1]["glob"])
        last_cap = disk["npz_catalog"][-1].get("max_shapes")
        if last_cap is None:
            self.assertIsNone(knobs["npz_catalog"][-1][1])
        else:
            self.assertEqual(knobs["npz_catalog"][-1][1], int(last_cap))
        self.assertEqual(
            knobs["shape_encoder"], str(disk["shape_encoder"]).strip().lower()
        )
        disk_pw = disk.get("pos_weight")
        if disk_pw is None:
            self.assertIsNone(knobs["pos_weight"])
            self.assertFalse(knobs["pos_weight_auto"])
        elif isinstance(disk_pw, str) and str(disk_pw).strip().lower() == "auto":
            self.assertIsNone(knobs["pos_weight"])
            self.assertTrue(knobs["pos_weight_auto"])
        else:
            self.assertAlmostEqual(float(knobs["pos_weight"]), float(disk_pw))
            self.assertFalse(knobs["pos_weight_auto"])

    def test_load_config_resolves_data_dir_and_device(self) -> None:
        # Live YAML points at this PC's catalog (E:/...). CI has no that disk.
        # require_data_dir still raises for a real train; this test only needs
        # a folder that exists so OccupancyConfig can resolve.
        knobs = load_yaml_knobs(_YAML)
        live = Path(knobs["data_dir"])
        if live.is_dir():
            self._assert_resolved_config(load_config())
            return
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            yaml_path = data / "config.yaml"
            disk = yaml.safe_load(_YAML.read_text(encoding="utf-8"))
            disk["data_dir"] = data.as_posix()
            yaml_path.write_text(yaml.safe_dump(disk), encoding="utf-8")
            self._assert_resolved_config(load_config(yaml_path))

    def _assert_resolved_config(self, cfg: OccupancyConfig) -> None:
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
        self.assertIn("npz_catalog=", rendered)
        self.assertIn("run_name=", rendered)
        self.assertIn("checkpoint_metric=", rendered)
        self.assertIn("batch_size=", rendered)
        self.assertIn("optimizer=", rendered)
        self.assertIn("n_surface=", rendered)
        self.assertNotIn("envelope_mix=", rendered)
        self.assertIn("knn_k=", rendered)
        self.assertIn("shape_encoder=", rendered)
        self.assertIn("pos_weight=", rendered)
        self.assertIn("pos_weight_auto=", rendered)
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

    def test_shape_encoder_mesh_rejected(self) -> None:
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
                        "shape_encoder: mesh",
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
            self.assertNotIn("envelope_mix", knobs)
            self.assertEqual(knobs["knn_k"], 0)
            self.assertIsNone(knobs["pos_weight"])
            self.assertFalse(knobs["pos_weight_auto"])

    def test_pos_weight_yaml_auto_and_float(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            auto_path = root / "auto.yaml"
            auto_path.write_text(
                "\n".join(
                    [
                        "hidden: 64",
                        "depth: 4",
                        "seed: 1",
                        f'data_dir: "{root.as_posix()}"',
                        "epochs: 2",
                        "lr: 0.001",
                        "val_fraction: 0.2",
                        "pos_weight: auto",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            auto = load_yaml_knobs(auto_path)
            self.assertIsNone(auto["pos_weight"])
            self.assertTrue(auto["pos_weight_auto"])
            num_path = root / "num.yaml"
            num_path.write_text(
                "\n".join(
                    [
                        "hidden: 64",
                        "depth: 4",
                        "seed: 1",
                        f'data_dir: "{root.as_posix()}"',
                        "epochs: 2",
                        "lr: 0.001",
                        "val_fraction: 0.2",
                        "pos_weight: 4.5",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            num = load_yaml_knobs(num_path)
            self.assertAlmostEqual(float(num["pos_weight"]), 4.5)
            self.assertFalse(num["pos_weight_auto"])
            bad_path = root / "bad.yaml"
            bad_path.write_text(
                "\n".join(
                    [
                        "hidden: 64",
                        "depth: 4",
                        "seed: 1",
                        f'data_dir: "{root.as_posix()}"',
                        "epochs: 2",
                        "lr: 0.001",
                        "val_fraction: 0.2",
                        "pos_weight: 0",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_yaml_knobs(bad_path)

    def test_stale_envelope_mix_yaml_is_ignored(self) -> None:
        """Old run snapshots still load; mix is no longer a knob."""
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "config.yaml"
            yaml_path.write_text(
                "\n".join(
                    [
                        "hidden: 64",
                        "depth: 4",
                        "seed: 1",
                        f'data_dir: "{Path(tmp).as_posix()}"',
                        "epochs: 2",
                        "lr: 0.001",
                        "test_fraction: 0.2",
                        "envelope_mix: 75",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            knobs = load_yaml_knobs(yaml_path)
            self.assertNotIn("envelope_mix", knobs)
            self.assertIsNone(knobs["pos_weight"])
            self.assertFalse(knobs["pos_weight_auto"])

    def test_get_device_cpu_env_override(self) -> None:
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"SCATTERINGNET_DEVICE": "cpu"}, clear=False):
            self.assertEqual(get_device().type, "cpu")


if __name__ == "__main__":
    unittest.main()
