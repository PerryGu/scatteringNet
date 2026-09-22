"""Infer occupancy from models/<id>/best.pt. No new checkpoints written."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import trimesh


from scatteringnet.config import OccupancyConfig
from scatteringnet.infer_multi_npz import infer_npz, resolve_best_pt
from scatteringnet.occupancy_mlp import CHECKPOINT_KIND
from scatteringnet.train_multi_npz import train_multi_npz


def _write_box_obj(folder: Path, name: str) -> Path:
    path = folder / name
    trimesh.creation.box(extents=[2.0, 2.0, 2.0]).export(path)
    return path


def _write_npz(path: Path, n: int, seed: int, mesh_rel: str) -> None:
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
        val_fraction=0.2,
        run_name="infer",
        checkpoint_metric="val_acc",
        batch_size=32,
        optimizer="adam",
    )


class InferMultiNpzTests(unittest.TestCase):
    def test_infer_from_best_pt_prints_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_box_obj(root, "box_a.obj")
            _write_box_obj(root, "box_b.obj")
            a = root / "box_a__occupancy.npz"
            b = root / "box_b__occupancy.npz"
            _write_npz(a, n=40, seed=1, mesh_rel="box_a.obj")
            _write_npz(b, n=60, seed=2, mesh_rel="box_b.obj")
            result = train_multi_npz(
                _cpu_cfg(root),
                npz_paths=[a, b],
                epochs=2,
                root=root,
                run_name="infer",
            )
            scores = infer_npz(
                _cpu_cfg(root),
                checkpoint=result.best_path,
                npz_path=a,
                root=root,
            )
            resolved = resolve_best_pt(run_id=result.run_dir.name, root=root)
            self.assertTrue(result.best_path.is_file())
            self.assertEqual(resolved, result.best_path)
            self.assertFalse(list(result.run_dir.rglob("*.pt")))
            ckpt = torch.load(result.best_path, map_location="cpu", weights_only=False)
            self.assertEqual(ckpt["kind"], CHECKPOINT_KIND)
            self.assertIn("accuracy", scores)
            self.assertGreaterEqual(scores["accuracy"], 0.0)
            self.assertLessEqual(scores["accuracy"], 1.0)
            self.assertIn("inside_iou", scores)
            self.assertIn("inside_f1", scores)


if __name__ == "__main__":
    unittest.main()
