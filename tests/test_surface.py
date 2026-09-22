"""Envelope sampling. No occupancy training."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh


from scatteringnet.geometry.surface import (
    _ENVELOPE_CACHE,
    apply_envelope_aabb,
    clear_envelope_cache,
    sample_surface_points,
)


class SurfaceSampleTests(unittest.TestCase):
    def test_box_samples_lie_on_shell(self) -> None:
        mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
        pts = sample_surface_points(
            mesh.vertices, mesh.faces, 256, seed=1, cache_key=None
        )
        self.assertEqual(pts.shape, (256, 6))
        self.assertEqual(pts.dtype, np.float32)
        xyz = pts[:, :3]
        nrm = pts[:, 3:]
        # Unit box: at least one axis should sit on a face at |coord| ≈ 1.
        face_dist = np.abs(np.abs(xyz) - 1.0).min(axis=1)
        self.assertLess(float(face_dist.max()), 1e-4)
        self.assertTrue(np.allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-5))
        # Dominant axis of the point matches a ±unit face normal.
        axis = np.argmax(np.abs(xyz), axis=1)
        self.assertTrue(
            np.allclose(np.abs(nrm[np.arange(len(pts)), axis]), 1.0, atol=1e-4)
        )

    def test_sphere_samples_have_unit_radius(self) -> None:
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        pts = sample_surface_points(
            mesh.vertices, mesh.faces, 512, seed=2, cache_key=None
        )
        xyz = pts[:, :3]
        nrm = pts[:, 3:]
        radii = np.linalg.norm(xyz, axis=1)
        self.assertEqual(pts.shape, (512, 6))
        self.assertTrue(np.all(radii > 0.85))
        self.assertTrue(np.all(radii < 1.15))
        self.assertTrue(np.allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-5))
        # Face normal should point outward (same hemisphere as the position).
        self.assertTrue(np.all(np.sum(xyz * nrm, axis=1) > 0.0))

    def test_aabb_remaps_xyz_only(self) -> None:
        mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
        pts = sample_surface_points(mesh.vertices, mesh.faces, 32, seed=1)
        center = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        out = apply_envelope_aabb(pts, center, 4.0)
        self.assertTrue(np.allclose(out[:, 3:], pts[:, 3:]))
        self.assertFalse(np.allclose(out[:, :3], pts[:, :3]))

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
        key = ("box-cache", 16, 3)
        self.assertIn(key, _ENVELOPE_CACHE)
        self.assertIs(a, b)
        clear_envelope_cache()
        self.assertNotIn(key, _ENVELOPE_CACHE)

    def test_rejects_zero_count(self) -> None:
        mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
        with self.assertRaises(ValueError):
            sample_surface_points(mesh.vertices, mesh.faces, 0)


if __name__ == "__main__":
    unittest.main()
