"""Envelope sampling. No occupancy training."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from geometry.surface import sample_surface_points


class SurfaceSampleTests(unittest.TestCase):
    def test_box_samples_lie_on_shell(self) -> None:
        mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
        pts = sample_surface_points(
            mesh.vertices, mesh.faces, 256, seed=1, cache_key=None
        )
        self.assertEqual(pts.shape, (256, 3))
        self.assertEqual(pts.dtype, np.float32)
        # Unit box: at least one axis should sit on a face at |coord| ≈ 1.
        face_dist = np.abs(np.abs(pts) - 1.0).min(axis=1)
        self.assertLess(float(face_dist.max()), 1e-4)

    def test_sphere_samples_have_unit_radius(self) -> None:
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        pts = sample_surface_points(
            mesh.vertices, mesh.faces, 512, seed=2, cache_key=None
        )
        radii = np.linalg.norm(pts, axis=1)
        self.assertEqual(pts.shape, (512, 3))
        self.assertTrue(np.all(radii > 0.85))
        self.assertTrue(np.all(radii < 1.15))

    def test_n_surface_controls_count(self) -> None:
        mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
        a = sample_surface_points(mesh.vertices, mesh.faces, 32, seed=1)
        b = sample_surface_points(mesh.vertices, mesh.faces, 64, seed=1)
        self.assertEqual(a.shape[0], 32)
        self.assertEqual(b.shape[0], 64)

    def test_rejects_zero_count(self) -> None:
        mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
        with self.assertRaises(ValueError):
            sample_surface_points(mesh.vertices, mesh.faces, 0)


if __name__ == "__main__":
    unittest.main()
