"""Tests for AABB normalization. No training."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


from scatteringnet.normalize import apply_normalization, compute_center_scale


class NormalizeTests(unittest.TestCase):
    def test_unit_cube_maps_to_minus_one_one(self) -> None:
        # Corners of an axis-aligned cube centered at the origin, side length 2.
        points = np.array(
            [
                [-1.0, -1.0, -1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        center, scale = compute_center_scale(points)
        np.testing.assert_allclose(center, np.zeros(3, dtype=np.float32), atol=1e-6)
        self.assertGreater(scale, 0.0)
        self.assertAlmostEqual(scale, 1.0, places=5)
        normed = apply_normalization(points, center, scale)
        self.assertTrue(np.all(normed >= -1.0 - 1e-5))
        self.assertTrue(np.all(normed <= 1.0 + 1e-5))

    def test_inverse_on_a_few_rows(self) -> None:
        rng = np.random.default_rng(0)
        points = rng.uniform(-4.0, 12.0, size=(16, 3)).astype(np.float32)
        center, scale = compute_center_scale(points)
        normed = apply_normalization(points, center, scale)
        recovered = normed * np.float32(scale) + center
        np.testing.assert_allclose(recovered[:5], points[:5], rtol=1e-5, atol=1e-5)

    def test_rejects_non_positive_scale(self) -> None:
        same = np.ones((4, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            compute_center_scale(same)
        with self.assertRaises(ValueError):
            apply_normalization(np.zeros((2, 3), dtype=np.float32), np.zeros(3), 0.0)


if __name__ == "__main__":
    unittest.main()
