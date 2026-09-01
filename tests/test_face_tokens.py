"""Face tokens. No occupancy training."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from geometry.face_tokens import (
    FACE_FEAT_DIM,
    _TOKEN_CACHE,
    apply_face_aabb,
    clear_face_token_cache,
    face_tokens_from_triangles,
    undo_face_aabb,
)
from normalize import compute_center_scale


def _tetrahedron() -> tuple[np.ndarray, np.ndarray]:
    verts = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
        dtype=np.int64,
    )
    return verts, faces


class FaceTokenTests(unittest.TestCase):
    def test_box_shape_and_unit_normals(self) -> None:
        mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
        tok = face_tokens_from_triangles(mesh.vertices, mesh.faces, 12)
        self.assertEqual(tok.shape, (12, FACE_FEAT_DIM))
        self.assertEqual(tok.dtype, np.float32)
        lengths = np.linalg.norm(tok[:, 9:12], axis=1)
        self.assertTrue(np.allclose(lengths, 1.0, atol=1e-5))

    def test_tetrahedron_pads_by_repeat(self) -> None:
        verts, faces = _tetrahedron()
        tok = face_tokens_from_triangles(verts, faces, 8)
        self.assertEqual(tok.shape, (8, 12))
        # Four real faces, then the same four again.
        self.assertTrue(np.allclose(tok[:4], tok[4:]))

    def test_subsample_keeps_largest_area(self) -> None:
        verts = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [0, 3, 4]], dtype=np.int64)
        tok = face_tokens_from_triangles(verts, faces, 1)
        self.assertEqual(tok.shape, (1, 12))
        # The large triangle occupies [0,0,0], [2,0,0], [0,2,0].
        self.assertTrue(np.allclose(tok[0, 0:3], [0.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(tok[0, 3:6], [2.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(tok[0, 6:9], [0.0, 2.0, 0.0]))

    def test_aabb_inverse_on_corners(self) -> None:
        verts, faces = _tetrahedron()
        world = face_tokens_from_triangles(verts, faces, 4)
        center, scale = compute_center_scale(verts.astype(np.float32))
        normed = apply_face_aabb(world, center, scale)
        self.assertTrue(np.allclose(normed[:, 9:12], world[:, 9:12]))
        back = undo_face_aabb(normed, center, scale)
        self.assertTrue(np.allclose(back[:, 0:9], world[:, 0:9], atol=1e-5))
        self.assertTrue(np.allclose(back[:, 9:12], world[:, 9:12], atol=1e-5))

    def test_cache_hit_then_clear(self) -> None:
        mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
        clear_face_token_cache()
        a = face_tokens_from_triangles(
            mesh.vertices, mesh.faces, 16, cache_key="box-faces"
        )
        b = face_tokens_from_triangles(
            mesh.vertices, mesh.faces, 16, cache_key="box-faces"
        )
        self.assertIn(("box-faces", 16), _TOKEN_CACHE)
        self.assertIs(a, b)
        clear_face_token_cache()
        self.assertNotIn(("box-faces", 16), _TOKEN_CACHE)

    def test_rejects_zero_count(self) -> None:
        mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
        with self.assertRaises(ValueError):
            face_tokens_from_triangles(mesh.vertices, mesh.faces, 0)


if __name__ == "__main__":
    unittest.main()
