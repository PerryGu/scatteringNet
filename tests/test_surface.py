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

from geometry.surface import (
    _ENVELOPE_CACHE,
    clear_envelope_cache,
    crease_length_fraction,
    plan_envelope_counts,
    sample_surface_points,
    split_envelope_counts,
)


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

    def test_envelope_cache_hit_then_clear(self) -> None:
        mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
        clear_envelope_cache()
        a = sample_surface_points(
            mesh.vertices, mesh.faces, 16, seed=3, cache_key="box-cache"
        )
        b = sample_surface_points(
            mesh.vertices, mesh.faces, 16, seed=3, cache_key="box-cache"
        )
        key = ("box-cache", 16, 3, 0)
        self.assertIn(key, _ENVELOPE_CACHE)
        self.assertIs(a, b)
        clear_envelope_cache()
        self.assertNotIn(key, _ENVELOPE_CACHE)

    def test_mix_uses_separate_cache_slot(self) -> None:
        mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
        clear_envelope_cache()
        a = sample_surface_points(
            mesh.vertices, mesh.faces, 16, seed=3, mix=0, cache_key="box-mix"
        )
        b = sample_surface_points(
            mesh.vertices, mesh.faces, 16, seed=3, mix=100, cache_key="box-mix"
        )
        self.assertIn(("box-mix", 16, 3, 0), _ENVELOPE_CACHE)
        self.assertIn(("box-mix", 16, 3, 100), _ENVELOPE_CACHE)
        self.assertFalse(np.allclose(a, b))

    def test_rejects_zero_count(self) -> None:
        mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
        with self.assertRaises(ValueError):
            sample_surface_points(mesh.vertices, mesh.faces, 0)

    def test_mix_100_hugs_box_edges(self) -> None:
        mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
        pts = sample_surface_points(
            mesh.vertices, mesh.faces, 256, seed=1, mix=100
        )
        # Box faces sit at |coord|=1. Distance to the nearest of 12 edges.
        xyz = np.abs(np.asarray(pts, dtype=np.float64))
        d_edge = np.sort(np.abs(xyz - 1.0), axis=1)[:, 1]
        self.assertLess(float(np.median(d_edge)), 0.15)

    def test_rejects_bad_mix(self) -> None:
        mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
        with self.assertRaises(ValueError):
            sample_surface_points(mesh.vertices, mesh.faces, 8, mix=101)

    def test_plan_caps_sparse_creases_and_spills_to_area(self) -> None:
        # Mix 75 wants 768 crease dots; 10% sharp edges may take only 20%.
        self.assertEqual(split_envelope_counts(1024, 75), (256, 768))
        self.assertEqual(plan_envelope_counts(1024, 75, 0.10), (819, 205))
        self.assertEqual(plan_envelope_counts(1024, 75, 0.0), (1024, 0))
        # CAD box: almost every interior edge is sharp → Mix budget stands.
        self.assertEqual(plan_envelope_counts(1024, 75, 1.0), (256, 768))
        n_area, n_crease = plan_envelope_counts(1024, 75, 0.10)
        self.assertEqual(n_area + n_crease, 1024)

    def test_box_crease_fraction_keeps_mix_budget(self) -> None:
        mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
        # Face diagonals are coplanar, so frac is ~0.59, not 1. 2× still ≥ Mix 75.
        frac = crease_length_fraction(mesh.vertices, mesh.faces)
        self.assertGreater(frac, 0.5)
        n_area, n_crease = plan_envelope_counts(1024, 75, frac)
        self.assertEqual((n_area, n_crease), (256, 768))

    def test_sphere_mix_keeps_full_count(self) -> None:
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        frac = crease_length_fraction(mesh.vertices, mesh.faces)
        n_area, n_crease = plan_envelope_counts(512, 75, frac)
        self.assertEqual(n_area + n_crease, 512)
        self.assertLess(n_crease, int(round(512 * 0.75)))
        pts = sample_surface_points(
            mesh.vertices, mesh.faces, 512, seed=2, mix=75
        )
        self.assertEqual(pts.shape, (512, 3))


if __name__ == "__main__":
    unittest.main()
