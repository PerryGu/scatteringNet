"""Step 2 occupancy NPZ generation. Does not train OccupancyMLP."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_npz import load_points_labels
from scatter_generation.mesh_loader import load_mesh, to_data_relative
from scatter_generation.raycast_scatter import export_occupancy_npz, scatter_volume


def _write_box_obj(folder: Path, extents: tuple[float, float, float] = (2.0, 2.0, 2.0)) -> Path:
    mesh = trimesh.creation.box(extents=list(extents))
    path = folder / "box.obj"
    mesh.export(path)
    return path


class OccupancyNpzGenerationTests(unittest.TestCase):
    def test_npz_contract_and_phase1_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            obj = _write_box_obj(root)
            npz = export_occupancy_npz(
                obj,
                root / "box.npz",
                method="occupancy",
                point_spacing=0.35,
                random_range=0.0,
                seed=1,
                max_points=4000,
            )
            with np.load(npz, allow_pickle=True) as raw:
                points = np.asarray(raw["points"])
                labels = np.asarray(raw["labels"])
                self.assertEqual(points.ndim, 2)
                self.assertEqual(points.shape[1], 3)
                self.assertEqual(labels.shape, (points.shape[0],))
                self.assertIn("mesh_path", raw.files)
                mesh_path = str(np.asarray(raw["mesh_path"]).item())
                self.assertTrue(len(mesh_path) > 0)
                self.assertIn("random_range", raw.files)
            uniq = set(np.unique(labels).tolist())
            self.assertTrue(0 in uniq and 1 in uniq, f"need both classes, got {uniq}")
            pts, labs = load_points_labels(npz)
            self.assertEqual(pts.shape, points.shape)
            self.assertEqual(labs.shape, labels.shape)

    def test_mesh_path_relative_to_data_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            obj = data / "meshes" / "Primitives" / "box.obj"
            obj.parent.mkdir(parents=True)
            obj.write_text("x")
            rel = to_data_relative(obj, data_dir=data)
            self.assertEqual(rel, "meshes/Primitives/box.obj")
            outside = Path(tmp) / "elsewhere.obj"
            outside.write_text("x")
            stored = to_data_relative(outside, data_dir=data)
            self.assertTrue(Path(stored).is_absolute())

    def test_finer_spacing_yields_more_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            obj = _write_box_obj(root)
            coarse = export_occupancy_npz(
                obj, root / "coarse.npz", point_spacing=0.5, random_range=0.0, seed=1, max_points=8000
            )
            fine = export_occupancy_npz(
                obj, root / "fine.npz", point_spacing=0.25, random_range=0.0, seed=1, max_points=8000
            )
            n_coarse = int(np.load(coarse)["points"].shape[0])
            n_fine = int(np.load(fine)["points"].shape[0])
            self.assertGreater(n_fine, n_coarse)

    def test_random_range_zero_keeps_lattice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            obj = _write_box_obj(Path(tmp))
            mesh, _info = load_mesh(obj)
            a = scatter_volume(
                mesh, method="occupancy", point_spacing=0.4, random_range=0.0, seed=7, max_points=2000
            )
            b = scatter_volume(
                mesh, method="occupancy", point_spacing=0.4, jitter=0.0, seed=7, max_points=2000
            )
            np.testing.assert_allclose(a.points, b.points, rtol=0, atol=0)

    def test_random_range_moves_then_relabels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            obj = _write_box_obj(Path(tmp))
            mesh, _info = load_mesh(obj)
            noisy = scatter_volume(
                mesh, method="occupancy", point_spacing=0.35, random_range=0.2, seed=3, max_points=2500
            )
            noisy2 = scatter_volume(
                mesh, method="occupancy", point_spacing=0.35, random_range=0.2, seed=3, max_points=2500
            )
            np.testing.assert_allclose(noisy.points, noisy2.points, rtol=0, atol=1e-12)
            self.assertTrue(noisy.occupancy_verified)
            # Off-lattice: a second seed must not reproduce the same cloud.
            other = scatter_volume(
                mesh, method="occupancy", point_spacing=0.35, random_range=0.2, seed=99, max_points=2500
            )
            self.assertFalse(np.allclose(noisy.points, other.points, atol=1e-8))


if __name__ == "__main__":
    unittest.main()
