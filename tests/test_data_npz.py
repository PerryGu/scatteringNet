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

from data_npz import load_points_labels


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


if __name__ == "__main__":
    unittest.main()
