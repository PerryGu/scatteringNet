"""Multi-NPZ occupancy train. Two tiny files, no OccupancyMLP edits."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import trimesh
import yaml

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dataclasses import replace

from config import OccupancyConfig
from occupancy_encoder import CHECKPOINT_KIND as ENCODER_KIND
from occupancy_mlp import CHECKPOINT_KIND
from train_multi_npz import train_multi_npz


def _write_box_obj(folder: Path) -> Path:
    path = folder / "box.obj"
    trimesh.creation.box(extents=[2.0, 2.0, 2.0]).export(path)
    return path


def _write_npz(path: Path, n: int, seed: int, mesh_rel: str = "box.obj") -> None:
    rng = np.random.default_rng(seed)
    points = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    labels = (np.linalg.norm(points, axis=1) < 0.5).astype(np.uint8)
    np.savez(
        path,
        points=points,
        labels=labels,
        mesh_path=np.asarray(mesh_rel),
    )


def _cpu_cfg(tmp: Path) -> OccupancyConfig:
    return OccupancyConfig(
        data_dir=tmp,
        device=torch.device("cpu"),
        hidden=16,
        depth=2,
        seed=1,
        epochs=2,
        lr=1e-2,
        test_fraction=0.2,
        run_name="test",
        checkpoint_metric="test_acc",
        batch_size=32,
        optimizer="adam",
    )


class TrainMultiNpzTests(unittest.TestCase):
    def test_two_files_train_logs_and_best(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_box_obj(root)
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
            self.assertGreater(result.n_train, 0)
            self.assertGreater(result.n_test, 0)
            self.assertEqual(result.n_train + result.n_test, 100)
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
                self.assertEqual(part["mesh"], "box.obj")
            snap = yaml.safe_load(result.run_dir.joinpath("config.yaml").read_text(encoding="utf-8"))
            self.assertNotIn("catalog_npz_paths", snap)
            self.assertEqual(snap["catalog_file"], "catalog.txt")
            self.assertEqual(snap["catalog_n"], 2)
            self.assertEqual(snap["n_files"], 2)
            self.assertEqual(snap["n_meshes"], 1)
            self.assertEqual(snap["split"], "shape")
            self.assertEqual(snap["n_train_files"] + snap["n_test_files"], 2)
            self.assertEqual(snap["n_train"], result.n_train)
            self.assertEqual(snap["n_test"], result.n_test)
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
            row = json.loads(lines[-1])
            self.assertIn("test_acc", row)
            self.assertIn("best_epoch", row)
            self.assertNotIn("smoke_epochs", snap)

    def test_surface_encoder_train_logs_iou(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_box_obj(root)
            a = root / "box_a__occupancy.npz"
            b = root / "box_b__occupancy.npz"
            _write_npz(a, n=40, seed=1)
            _write_npz(b, n=60, seed=2)
            cfg = replace(
                _cpu_cfg(root),
                shape_encoder="surface",
                n_surface=32,
            )
            result = train_multi_npz(
                cfg,
                npz_paths=[a, b],
                epochs=2,
                root=root,
                run_name="surf",
            )
            ckpt = torch.load(result.best_path, map_location="cpu", weights_only=False)
            self.assertEqual(ckpt["kind"], ENCODER_KIND)
            self.assertEqual(ckpt["shape_encoder"], "surface")
            self.assertEqual(ckpt["n_surface"], 32)
            snap = yaml.safe_load(
                result.run_dir.joinpath("config.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(snap["shape_encoder"], "surface")
            self.assertEqual(snap["n_surface"], 32)
            row = json.loads(
                result.run_dir.joinpath("metrics.jsonl")
                .read_text(encoding="utf-8")
                .strip()
                .splitlines()[-1]
            )
            self.assertIn("test_iou", row)
            self.assertIn("test_f1", row)

    def test_rejects_zero_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_box_obj(root)
            a = root / "box_a__occupancy.npz"
            _write_npz(a, n=8, seed=1)
            with self.assertRaises(ValueError):
                train_multi_npz(
                    _cpu_cfg(root),
                    npz_paths=[a],
                    epochs=0,
                    root=root,
                )


if __name__ == "__main__":
    unittest.main()
