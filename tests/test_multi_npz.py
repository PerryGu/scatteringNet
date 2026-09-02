"""Multi-NPZ catalog and dataset. No occupancy training."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import trimesh

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_npz import resolve_npz_catalog, shape_key
from dataset import OccupancyMultiNpzDataset, OccupancyPointDataset, make_dataloader
from normalize import compute_center_scale


def _write_npz(
    path: Path,
    n: int,
    seed: int,
    mesh_rel: str | None = None,
) -> None:
    rng = np.random.default_rng(seed)
    points = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    labels = (rng.random(n) > 0.5).astype(np.uint8)
    payload: dict = {"points": points, "labels": labels}
    if mesh_rel is not None:
        payload["mesh_path"] = np.asarray(mesh_rel)
    np.savez(path, **payload)


def _write_points_npz(
    path: Path,
    points: np.ndarray,
    *,
    mesh_rel: str | None = None,
) -> None:
    labels = np.zeros((int(points.shape[0]),), dtype=np.uint8)
    payload: dict = {"points": np.asarray(points, dtype=np.float32), "labels": labels}
    if mesh_rel is not None:
        payload["mesh_path"] = np.asarray(mesh_rel)
    np.savez(path, **payload)


class MultiNpzCatalogTests(unittest.TestCase):
    def test_two_files_pool_point_count_and_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "box_a__occupancy_s0.15_inout.npz"
            b = root / "box_b__occupancy_s0.15_inout.npz"
            _write_npz(a, n=10, seed=1)
            _write_npz(b, n=15, seed=2)
            paths = resolve_npz_catalog(
                root,
                npz_glob="*.npz",
                max_files_per_shape=None,
            )
            self.assertEqual(len(paths), 2)
            ds = OccupancyMultiNpzDataset(paths)
            self.assertEqual(len(ds), 2)
            self.assertEqual(ds.n_points, 25)
            xyz, y = ds.parts[0][0]
            self.assertEqual(tuple(xyz.shape), (3,))
            self.assertEqual(tuple(y.shape), (1,))
            loader = make_dataloader(ds.parts[0], batch_size=8, shuffle=False)
            batch_xyz, batch_y = next(iter(loader))
            self.assertEqual(tuple(batch_xyz.shape), (8, 3))
            self.assertEqual(tuple(batch_y.shape), (8, 1))
            self.assertEqual(batch_xyz.dtype, torch.float32)
            self.assertEqual(len(ds.parts), 2)
            self.assertEqual(len(ds.npz_paths), 2)

    def test_npz_catalog_unions_globs_and_caps_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(6):
                _write_npz(root / f"box_{i}__occupancy.npz", n=4, seed=i)
            for i in range(4):
                _write_npz(root / f"pipe_{i}__occupancy.npz", n=4, seed=10 + i)
            paths = resolve_npz_catalog(
                root,
                npz_glob="missing_*.npz",
                npz_catalog=(
                    ("box_*.npz", 3),
                    ("pipe_*.npz", None),
                ),
                max_files_per_shape=None,
                seed=1,
            )
            keys = [shape_key(p) for p in paths]
            n_box = sum(1 for k in keys if k.startswith("box_"))
            n_pipe = sum(1 for k in keys if k.startswith("pipe_"))
            self.assertEqual(n_box, 3)
            self.assertEqual(n_pipe, 4)
            stray = root / "Cone__occupancy.npz"
            _write_npz(stray, n=4, seed=99)
            with self.assertRaises(FileNotFoundError):
                resolve_npz_catalog(
                    root,
                    npz_catalog=(("cone_*.npz", None),),
                    max_files_per_shape=None,
                    seed=1,
                )
            again = resolve_npz_catalog(
                root,
                npz_catalog=(("box_*.npz", 3), ("pipe_*.npz", None)),
                max_files_per_shape=None,
                seed=1,
            )
            self.assertEqual([p.name for p in again], [p.name for p in paths])

    def test_max_files_per_shape_caps_same_mesh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_npz(root / "sphere__occupancy_s0.15_inout.npz", n=4, seed=1)
            _write_npz(root / "sphere__occupancy_s0.15_j0.04_inout.npz", n=4, seed=2)
            _write_npz(root / "cube__occupancy_s0.15_inout.npz", n=4, seed=3)
            paths = resolve_npz_catalog(
                root,
                npz_glob="*.npz",
                max_files_per_shape=1,
            )
            keys = [shape_key(p) for p in paths]
            self.assertEqual(len(paths), 2)
            self.assertEqual(set(keys), {"sphere", "cube"})

    def test_excludes_combo_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_npz(root / "sphere__occupancy_s0.15_inout.npz", n=4, seed=1)
            _write_npz(root / "combo_ab__occupancy_s0.15_inout.npz", n=4, seed=2)
            paths = resolve_npz_catalog(root, npz_glob="*.npz", max_files_per_shape=None)
            self.assertEqual(len(paths), 1)
            self.assertEqual(paths[0].name, "sphere__occupancy_s0.15_inout.npz")

    def test_explicit_list_and_from_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trimesh.creation.box(extents=[2.0, 2.0, 2.0]).export(root / "box.obj")
            a = root / "a.npz"
            b = root / "b.npz"
            _write_npz(a, n=5, seed=1, mesh_rel="box.obj")
            _write_npz(b, n=7, seed=2, mesh_rel="box.obj")
            ds = OccupancyMultiNpzDataset.from_catalog(
                root,
                npz_paths=["a.npz", "b.npz"],
                max_files_per_shape=None,
            )
            self.assertEqual(len(ds), 2)
            self.assertEqual(ds.n_points, 12)
            self.assertEqual(ds.n_meshes, 1)
            self.assertIsNotNone(ds.parts[0].mesh_path)
            self.assertEqual(ds.parts[0].faces.shape[1], 3)
            self.assertEqual(ds.parts[0].shape_id, ds.parts[1].shape_id)

    def test_shape_id_is_per_mesh_not_per_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trimesh.creation.box(extents=[2.0, 2.0, 2.0]).export(root / "box_a.obj")
            trimesh.creation.box(extents=[2.0, 2.0, 2.0]).export(root / "box_b.obj")
            a0 = root / "box_a__s0.npz"
            a1 = root / "box_a__j.npz"
            b0 = root / "box_b__s0.npz"
            _write_npz(a0, n=5, seed=1, mesh_rel="box_a.obj")
            _write_npz(a1, n=6, seed=2, mesh_rel="box_a.obj")
            _write_npz(b0, n=7, seed=3, mesh_rel="box_b.obj")
            ds = OccupancyMultiNpzDataset(
                [a0, a1, b0],
                data_dir=root,
            )
            self.assertEqual(ds.parts[0].shape_id, ds.parts[1].shape_id)
            self.assertNotEqual(ds.parts[0].shape_id, ds.parts[2].shape_id)
            self.assertEqual({part.shape_id for part in ds.parts}, {0, 1})

    def test_shared_aabb_from_mesh_vertices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trimesh.creation.box(extents=[2.0, 2.0, 2.0]).export(root / "box.obj")
            tight = np.array(
                [[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, -0.1, 0.0]],
                dtype=np.float32,
            )
            wide = np.array(
                [[0.8, 0.0, 0.0], [-0.8, 0.0, 0.0], [0.0, 0.8, 0.0], [0.0, -0.8, 0.0]],
                dtype=np.float32,
            )
            a = root / "box__tight.npz"
            b = root / "box__wide.npz"
            _write_points_npz(a, tight, mesh_rel="box.obj")
            _write_points_npz(b, wide, mesh_rel="box.obj")
            alone_a = OccupancyPointDataset(a, data_dir=root)
            alone_b = OccupancyPointDataset(b, data_dir=root)
            self.assertNotAlmostEqual(float(alone_a.scale), float(alone_b.scale))
            ds = OccupancyMultiNpzDataset([a, b], data_dir=root, n_surface=16)
            expect_c, expect_s = compute_center_scale(ds.parts[0].vertices)
            self.assertTrue(np.allclose(ds.parts[0].center, ds.parts[1].center))
            self.assertAlmostEqual(float(ds.parts[0].scale), float(ds.parts[1].scale))
            self.assertTrue(np.allclose(ds.parts[0].center, expect_c))
            self.assertAlmostEqual(float(ds.parts[0].scale), float(expect_s))
            self.assertIsNotNone(ds.parts[0].envelope)
            self.assertIsNotNone(ds.parts[1].envelope)
            self.assertIsNotNone(ds.parts[0].face_tokens)
            self.assertEqual(tuple(ds.parts[0].face_tokens.shape), (256, 12))
            self.assertTrue(
                np.allclose(
                    ds.parts[0].face_tokens.numpy(),
                    ds.parts[1].face_tokens.numpy(),
                )
            )

    def test_shared_aabb_union_without_mesh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tight = np.array(
                [[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, -0.1, 0.0]],
                dtype=np.float32,
            )
            wide = np.array(
                [[0.8, 0.0, 0.0], [-0.8, 0.0, 0.0], [0.0, 0.8, 0.0], [0.0, -0.8, 0.0]],
                dtype=np.float32,
            )
            a = root / "sphere__tight.npz"
            b = root / "sphere__wide.npz"
            _write_points_npz(a, tight)
            _write_points_npz(b, wide)
            ds = OccupancyMultiNpzDataset([a, b])
            expect_c, expect_s = compute_center_scale(np.concatenate([tight, wide], axis=0))
            self.assertTrue(np.allclose(ds.parts[0].center, ds.parts[1].center))
            self.assertAlmostEqual(float(ds.parts[0].scale), float(ds.parts[1].scale))
            self.assertTrue(np.allclose(ds.parts[0].center, expect_c))
            self.assertAlmostEqual(float(ds.parts[0].scale), float(expect_s))

    def test_empty_catalog_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                resolve_npz_catalog(Path(tmp), npz_glob="*.npz")


if __name__ == "__main__":
    unittest.main()
