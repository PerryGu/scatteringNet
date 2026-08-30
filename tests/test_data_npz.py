"""Tests for load_points_labels. Synthetic NPZ only — no training."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_npz import load_points_labels, load_points_labels_mesh


class LoadPointsLabelsTests(unittest.TestCase):
    def test_loads_float32_points_and_labels(self) -> None:
        points = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
        labels = np.array([0, 1], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.npz"
            # Extra keys must be ignored.
            np.savez(path, points=points, labels=labels, tags=np.array([0, 1], dtype=np.uint8))
            out_p, out_y = load_points_labels(path)
        self.assertEqual(out_p.dtype, np.float32)
        self.assertEqual(out_y.dtype, np.float32)
        self.assertEqual(out_p.shape, (2, 3))
        self.assertEqual(out_y.shape, (2,))
        np.testing.assert_array_equal(out_y, np.array([0.0, 1.0], dtype=np.float32))

    def test_rejects_bad_points_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.npz"
            np.savez(path, points=np.zeros((4, 2), dtype=np.float32), labels=np.zeros(4, dtype=np.uint8))
            with self.assertRaises(ValueError):
                load_points_labels(path)

    def test_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_points_labels(Path("this_file_does_not_exist.npz"))

    def test_load_points_labels_mesh_resolves_obj(self) -> None:
        points = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
        labels = np.array([0, 1], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            obj = root / "meshes" / "box.obj"
            obj.parent.mkdir(parents=True)
            obj.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
            path = root / "sample.npz"
            np.savez(
                path,
                points=points,
                labels=labels,
                mesh_path=np.asarray("meshes/box.obj"),
            )
            out_p, out_y, mesh = load_points_labels_mesh(path, root)
        self.assertEqual(out_p.shape, (2, 3))
        self.assertEqual(out_y.shape, (2,))
        self.assertEqual(mesh.resolve(), obj.resolve())

    def test_load_points_labels_mesh_requires_mesh_path(self) -> None:
        points = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        labels = np.array([1], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "no_mesh.npz"
            np.savez(path, points=points, labels=labels)
            with self.assertRaises(KeyError):
                load_points_labels_mesh(path, Path(tmp))

    def test_step2_sphere_npz_resolves_obj(self) -> None:
        """Plan test: a Step 2 sphere NPZ yields a loadable OBJ."""
        from config import load_config
        from geometry.mesh_io import load_obj_triangles

        cfg = load_config()
        candidates = [
            cfg.data_dir
            / "exports"
            / "dataset_test"
            / "sphere__raycast_z_raut_s0.15_inout.npz",
            cfg.data_dir
            / "exports"
            / "dataset"
            / "sphere_r0p5_sa16_sh16__occupancy_s0.15_inout.npz",
        ]
        sample = next((p for p in candidates if p.is_file()), None)
        if sample is None:
            self.skipTest(f"no Step 2 sphere NPZ under {cfg.data_dir}")
        _pts, _labs, mesh = load_points_labels_mesh(sample, cfg.data_dir)
        self.assertTrue(mesh.is_file())
        self.assertEqual(mesh.suffix.lower(), ".obj")
        vertices, faces = load_obj_triangles(mesh)
        self.assertEqual(vertices.shape[1], 3)
        self.assertEqual(faces.shape[1], 3)
        self.assertGreaterEqual(int(vertices.shape[0]), 3)
        self.assertGreaterEqual(int(faces.shape[0]), 1)


if __name__ == "__main__":
    unittest.main()
