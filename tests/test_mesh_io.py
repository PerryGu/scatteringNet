"""OBJ triangle loader. No occupancy training."""

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

from geometry.mesh_io import _TRIANGLE_CACHE, clear_triangle_cache, load_obj_triangles
from geometry import clear_geometry_caches


def _write_box_obj(folder: Path) -> Path:
    mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    path = folder / "box.obj"
    mesh.export(path)
    return path


class MeshIoTests(unittest.TestCase):
    def test_box_obj_returns_vertices_and_faces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            obj = _write_box_obj(Path(tmp))
            vertices, faces = load_obj_triangles(obj, cache=False)
        self.assertEqual(vertices.dtype, np.float32)
        self.assertEqual(faces.dtype, np.int32)
        self.assertEqual(vertices.shape[1], 3)
        self.assertEqual(faces.shape[1], 3)
        self.assertGreaterEqual(int(vertices.shape[0]), 3)
        self.assertGreaterEqual(int(faces.shape[0]), 1)

    def test_missing_obj_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_obj_triangles(Path("this_mesh_does_not_exist.obj"), cache=False)

    def test_cache_hit_then_clear(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            obj = _write_box_obj(Path(tmp))
            clear_triangle_cache()
            first_v, first_f = load_obj_triangles(obj, cache=True)
            second_v, second_f = load_obj_triangles(obj, cache=True)
            key = str(obj.resolve())
            self.assertIn(key, _TRIANGLE_CACHE)
            self.assertIs(first_v, second_v)
            self.assertIs(first_f, second_f)
            clear_geometry_caches()
            self.assertNotIn(key, _TRIANGLE_CACHE)

    def test_rejects_non_obj_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "box.stl"
            path.write_text("x")
            with self.assertRaises(ValueError):
                load_obj_triangles(path, cache=False)


if __name__ == "__main__":
    unittest.main()
