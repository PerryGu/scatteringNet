"""Viewer helper: mesh_path must resolve to an OBJ under data_dir."""

from __future__ import annotations

import base64
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_VIEWER = Path(__file__).resolve().parents[1] / "src" / "viewer"
if str(_VIEWER) not in sys.path:
    sys.path.insert(0, str(_VIEWER))

from mesh_access import resolve_viewer_mesh


class ResolveViewerMeshTests(unittest.TestCase):
    def test_relative_obj_under_data_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mesh_dir = root / "meshes"
            mesh_dir.mkdir()
            obj = mesh_dir / "part.obj"
            obj.write_text("# test\nv 0 0 0\n", encoding="utf-8")
            got = resolve_viewer_mesh("meshes/part.obj", root)
            self.assertEqual(got.resolve(), obj.resolve())

    def test_rejects_parent_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "data"
            root.mkdir()
            secret = base / "secret.obj"
            secret.write_text("# x\n", encoding="utf-8")
            with self.assertRaises(PermissionError):
                resolve_viewer_mesh("../secret.obj", root)

    def test_rejects_absolute_outside_data_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "data"
            root.mkdir()
            outside = base / "other.obj"
            outside.write_text("# x\n", encoding="utf-8")
            with self.assertRaises(PermissionError):
                resolve_viewer_mesh(str(outside.resolve()), root)

    def test_rejects_non_obj_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stl = root / "part.stl"
            stl.write_text("solid x\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                resolve_viewer_mesh("part.stl", root)

    def test_missing_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                resolve_viewer_mesh("meshes/missing.obj", root)


class ViewerModelAccessTests(unittest.TestCase):
    def test_list_newest_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "run_old"
            new = root / "run_new"
            old.mkdir()
            new.mkdir()
            (old / "best.pt").write_bytes(b"old")
            (new / "best.pt").write_bytes(b"new")
            os.utime(old / "best.pt", (1_000_000, 1_000_000))
            os.utime(new / "best.pt", (2_000_000, 2_000_000))
            from model_access import list_viewer_models

            ids = [row["id"] for row in list_viewer_models(root)]
            self.assertEqual(ids, ["run_new", "run_old"])

    def test_list_includes_shape_encoder_from_runs_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            models = base / "models"
            runs = base / "runs"
            run = models / "mesh_run"
            run.mkdir(parents=True)
            (run / "best.pt").write_bytes(b"x")
            snap = runs / "mesh_run"
            snap.mkdir(parents=True)
            (snap / "config.yaml").write_text(
                "shape_encoder: mesh\nhidden: 64\n", encoding="utf-8"
            )
            from model_access import list_viewer_models

            rows = list_viewer_models(models, runs_root=runs)
            self.assertEqual(rows[0]["shape_encoder"], "mesh")

    def test_list_missing_dir_is_empty(self) -> None:
        from model_access import list_viewer_models

        self.assertEqual(list_viewer_models(Path(tempfile.gettempdir()) / "no-such-models"), [])

    def test_resolve_checkpoint_under_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "a_run"
            run.mkdir()
            best = run / "best.pt"
            best.write_bytes(b"x")
            from model_access import resolve_viewer_checkpoint

            got = resolve_viewer_checkpoint("a_run", root)
            self.assertEqual(got.resolve(), best.resolve())
            got2 = resolve_viewer_checkpoint("models/a_run/best.pt", root)
            self.assertEqual(got2.resolve(), best.resolve())

    def test_resolve_rejects_parent_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from model_access import resolve_viewer_checkpoint

            with self.assertRaises(ValueError):
                resolve_viewer_checkpoint("../secret", Path(tmp))

    def test_match_part_by_npz_name(self) -> None:
        from model_access import match_checkpoint_part

        ckpt = {
            "parts": [
                {
                    "npz": "exports/dataset/Cone__occupancy.npz",
                    "mesh": "meshes/varied/Cone.obj",
                    "center": [0, 0, 0],
                    "scale": 1.0,
                }
            ]
        }
        part = match_checkpoint_part(ckpt, "Cone__occupancy.npz", "")
        self.assertIsNotNone(part)
        self.assertEqual(part["scale"], 1.0)
        self.assertIsNone(match_checkpoint_part(ckpt, "other.npz", "meshes/nope.obj"))
        by_mesh = match_checkpoint_part(ckpt, "other.npz", "meshes/varied/Cone.obj")
        self.assertIsNotNone(by_mesh)


# Unit cube Wavefront text for AABB lattice tests (12 triangles).
_CUBE_OBJ = """
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 0 0 1
v 1 0 1
v 1 1 1
v 0 1 1
f 1 2 3
f 1 3 4
f 5 8 7
f 5 7 6
f 1 5 6
f 1 6 2
f 4 3 7
f 4 7 8
f 1 4 8
f 1 8 5
f 2 6 7
f 2 7 3
"""


class ObjFillTests(unittest.TestCase):
    """Job B lattice: unlabeled AABB grid, no raycast GT, no torch."""

    def test_fill_does_not_import_torch(self) -> None:
        had_torch = "torch" in sys.modules
        import obj_fill

        self.assertEqual("torch" in sys.modules, had_torch)
        src = Path(obj_fill.__file__).read_text(encoding="utf-8")
        self.assertNotRegex(src, r"(?m)^(import torch|from torch\b)")
        self.assertNotIn("scatter_volume", src)

    def test_spacing_slider_maps_coarse_to_dense(self) -> None:
        from obj_fill import spacing_from_slider

        self.assertAlmostEqual(spacing_from_slider(0), 0.40)
        self.assertAlmostEqual(spacing_from_slider(100), 0.05)
        # Default UI value 71 ≈ training occupancy spacing 0.15.
        self.assertAlmostEqual(spacing_from_slider(71), 0.1515, places=4)

    def test_clamp_spacing_rejects_out_of_range(self) -> None:
        from obj_fill import clamp_spacing

        with self.assertRaises(ValueError):
            clamp_spacing(0.01)
        with self.assertRaises(ValueError):
            clamp_spacing(0.60)
        self.assertAlmostEqual(clamp_spacing(0.15), 0.15)

    def test_empty_obj_text_raises(self) -> None:
        from obj_fill import fill_from_obj_text

        with self.assertRaises(ValueError):
            fill_from_obj_text("  \n", 0.15)

    def test_fill_cube_returns_lattice_bytes(self) -> None:
        from obj_fill import fill_from_obj_text

        out = fill_from_obj_text(_CUBE_OBJ, 0.40)
        self.assertGreater(out["n"], 8)
        self.assertLessEqual(out["n"], 200_000)
        self.assertIn("points_b64", out)
        self.assertAlmostEqual(out["used_spacing"], 0.40)
        raw = base64.b64decode(out["points_b64"])
        self.assertEqual(len(raw), out["n"] * 3 * 4)


def _floor_and_flap_mesh():
    """3x3 XY floor (8 coplanar tris) plus a vertical flap on x=0."""
    verts = []
    for y in (0.0, 2.0, 4.0):
        for x in (0.0, 2.0, 4.0):
            verts.append([x, y, 0.0])
    # Vertical flap shares the two left floor edges (0-3 and 3-6).
    verts.append([0.0, 0.0, 1.0])
    verts.append([0.0, 2.0, 1.0])
    verts.append([0.0, 4.0, 1.0])
    faces = []
    def vid(ix: int, iy: int) -> int:
        return iy * 3 + ix

    for iy in range(2):
        for ix in range(2):
            a = vid(ix, iy)
            b = vid(ix + 1, iy)
            c = vid(ix + 1, iy + 1)
            d = vid(ix, iy + 1)
            faces.append([a, b, c])
            faces.append([a, c, d])
    faces.append([0, 3, 10])
    faces.append([0, 10, 9])
    faces.append([3, 6, 11])
    faces.append([3, 11, 10])
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


class EnvelopeOverlayTests(unittest.TestCase):
    """Purple overlay: same mix sampler as occupancy, no torch."""

    def test_envelope_job_does_not_import_torch(self) -> None:
        src = Path(__file__).resolve().parents[1] / "src" / "viewer" / "envelope_job.py"
        text = src.read_text(encoding="utf-8")
        self.assertNotRegex(text, r"(?m)^(import torch|from torch\b)")

    def test_clamp_n_surface(self) -> None:
        from envelope_job import clamp_n_surface

        self.assertEqual(clamp_n_surface(1024), 1024)
        with self.assertRaises(ValueError):
            clamp_n_surface(8)
        with self.assertRaises(ValueError):
            clamp_n_surface(99_000)

    def test_envelope_cube_count(self) -> None:
        from envelope_job import envelope_from_obj_text

        out = envelope_from_obj_text(_CUBE_OBJ, 256)
        self.assertEqual(out["n"], 256)
        self.assertEqual(out["n_creases"], 12)
        self.assertEqual(out["n_edge"], 256)
        self.assertEqual(out["n_area"], 0)
        raw = base64.b64decode(out["points_b64"])
        self.assertEqual(len(raw), 256 * 3 * 4)

    def test_split_envelope_counts(self) -> None:
        from envelope_job import split_envelope_counts

        self.assertEqual(split_envelope_counts(1024, 0), (1024, 0))
        self.assertEqual(split_envelope_counts(1024, 100), (0, 1024))
        self.assertEqual(split_envelope_counts(1024, 50), (512, 512))

    def test_envelope_mix_faces_only(self) -> None:
        from envelope_job import envelope_from_obj_text

        out = envelope_from_obj_text(_CUBE_OBJ, 256, mix=0)
        self.assertEqual(out["n"], 256)
        self.assertEqual(out["n_area"], 256)
        self.assertEqual(out["n_edge"], 0)
        self.assertEqual(out["mix"], 0)

    def test_cube_points_hug_edges(self) -> None:
        from envelope_job import sample_crease_surface
        from obj_fill import triangles_from_obj_text

        verts, faces = triangles_from_obj_text(_CUBE_OBJ)
        pts, _idx, n_creases = sample_crease_surface(verts, faces, 512, seed=1)
        self.assertEqual(n_creases, 12)
        # Unit cube [0,1]^3: 12 edges. Clustered samples sit near those edges.
        corners = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
             [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
            dtype=np.float64,
        )
        segs = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        dmin = np.full((pts.shape[0],), np.inf, dtype=np.float64)
        xyz = np.asarray(pts, dtype=np.float64)
        for i0, i1 in segs:
            a = corners[i0]
            b = corners[i1]
            ab = b - a
            den = float(np.dot(ab, ab)) + 1e-12
            t = np.clip(((xyz - a) @ ab) / den, 0.0, 1.0)
            proj = a + t[:, None] * ab
            dmin = np.minimum(dmin, np.linalg.norm(xyz - proj, axis=1))
        self.assertLess(float(np.median(dmin)), 0.12)

    def test_unwelded_cube_still_hugs_edges(self) -> None:
        from envelope_job import sample_crease_surface
        from obj_fill import triangles_from_obj_text

        verts, faces = triangles_from_obj_text(_CUBE_OBJ)
        exploded = np.asarray(verts, dtype=np.float64)[np.asarray(faces).reshape(-1)]
        exploded_faces = np.arange(faces.shape[0] * 3, dtype=np.int64).reshape(-1, 3)
        pts, _idx, n_creases = sample_crease_surface(
            exploded, exploded_faces, 256, seed=1
        )
        self.assertEqual(n_creases, 12)
        self.assertEqual(pts.shape, (256, 3))

    def test_flap_points_hug_the_fold(self) -> None:
        from envelope_job import sample_crease_surface
        from geometry.surface import crease_length_fraction, plan_envelope_counts

        verts, faces = _floor_and_flap_mesh()
        pts, _idx, n_creases = sample_crease_surface(verts, faces, 512, seed=1)
        self.assertGreaterEqual(n_creases, 1)
        # One 90° fold on a large floor: Mix 100 is capped; leftover is area.
        frac = crease_length_fraction(verts, faces)
        _n_area, n_edge = plan_envelope_counts(512, 100, frac)
        self.assertLess(n_edge, 512)
        # Only sharp shared edge is the 90° fold along x=0, y in [0,4], z=0.
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([0.0, 4.0, 0.0])
        ab = b - a
        xyz = np.asarray(pts, dtype=np.float64)
        t = np.clip(((xyz - a) @ ab) / (float(np.dot(ab, ab)) + 1e-12), 0.0, 1.0)
        proj = a + t[:, None] * ab
        dist = np.linalg.norm(xyz - proj, axis=1)
        near = np.sort(dist)[: max(1, n_edge)]
        self.assertLess(float(np.median(near)), 0.25)


class FaceTokenOverlayTests(unittest.TestCase):
    """Teal triangle + normal overlay: occupancy face tokens, no torch."""

    def test_faces_job_does_not_import_torch(self) -> None:
        src = Path(__file__).resolve().parents[1] / "src" / "viewer" / "faces_job.py"
        text = src.read_text(encoding="utf-8")
        self.assertNotRegex(text, r"(?m)^(import torch|from torch\b)")

    def test_clamp_n_faces(self) -> None:
        from faces_job import clamp_n_faces

        self.assertEqual(clamp_n_faces(256), 256)
        with self.assertRaises(ValueError):
            clamp_n_faces(8)
        with self.assertRaises(ValueError):
            clamp_n_faces(99_000)

    def test_faces_cube_tiles_to_count(self) -> None:
        from faces_job import faces_overlay_from_obj_text

        out = faces_overlay_from_obj_text(_CUBE_OBJ, 64)
        self.assertEqual(out["n"], 64)
        self.assertEqual(out["n_mesh"], 12)
        self.assertEqual(out["n_unique"], 12)
        raw = base64.b64decode(out["tokens_b64"])
        self.assertEqual(len(raw), 64 * 12 * 4)
        self.assertGreater(out["tick"], 0.0)


class UiPrefsTests(unittest.TestCase):
    def test_missing_file_is_defaults(self) -> None:
        from ui_prefs import DEFAULTS, load_ui_prefs

        with tempfile.TemporaryDirectory() as tmp:
            prefs = load_ui_prefs(tmp)
        self.assertEqual(prefs["opacity"], DEFAULTS["opacity"])
        self.assertTrue(prefs["mesh"])
        self.assertEqual(prefs["model_id"], "")

    def test_clamp_and_roundtrip(self) -> None:
        from ui_prefs import load_ui_prefs, save_ui_prefs

        with tempfile.TemporaryDirectory() as tmp:
            saved = save_ui_prefs(
                tmp,
                {
                    "mesh": False,
                    "opacity": 999,
                    "density": -3,
                    "model_id": "run_a",
                    "ignored": True,
                },
            )
            self.assertFalse(saved["mesh"])
            self.assertEqual(saved["opacity"], 100)
            self.assertEqual(saved["density"], 0)
            self.assertEqual(saved["inside_cut"], 50)
            self.assertEqual(saved["model_id"], "run_a")
            self.assertNotIn("ignored", saved)
            loaded = load_ui_prefs(tmp)
            self.assertEqual(loaded, saved)

    def test_clamp_inside_cut(self) -> None:
        from ui_prefs import clamp_ui_prefs

        self.assertEqual(clamp_ui_prefs({"inside_cut": 999})["inside_cut"], 100)
        self.assertEqual(clamp_ui_prefs({"inside_cut": -3})["inside_cut"], 0)
        self.assertEqual(clamp_ui_prefs({})["inside_cut"], 50)

    def test_clamp_envelope_n(self) -> None:
        from ui_prefs import clamp_ui_prefs

        self.assertEqual(clamp_ui_prefs({"envelope_n": 99999})["envelope_n"], 4096)
        self.assertEqual(clamp_ui_prefs({"envelope_n": 10})["envelope_n"], 256)
        self.assertEqual(clamp_ui_prefs({})["envelope_n"], 1024)

    def test_clamp_envelope_mix(self) -> None:
        from ui_prefs import clamp_ui_prefs

        self.assertEqual(clamp_ui_prefs({"envelope_mix": 999})["envelope_mix"], 100)
        self.assertEqual(clamp_ui_prefs({"envelope_mix": -3})["envelope_mix"], 0)
        self.assertEqual(clamp_ui_prefs({})["envelope_mix"], 100)

    def test_clamp_faces_n(self) -> None:
        from ui_prefs import clamp_ui_prefs

        self.assertEqual(clamp_ui_prefs({"faces_n": 99999})["faces_n"], 1024)
        self.assertEqual(clamp_ui_prefs({"faces_n": 8})["faces_n"], 64)
        self.assertEqual(clamp_ui_prefs({})["faces_n"], 256)

    def test_rejects_unsafe_model_id(self) -> None:
        from ui_prefs import clamp_ui_prefs

        out = clamp_ui_prefs({"model_id": "../secret"})
        self.assertEqual(out["model_id"], "")

    def test_invalid_json_is_defaults(self) -> None:
        from ui_prefs import DEFAULTS, PREFS_NAME, load_ui_prefs

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / PREFS_NAME).write_text("{not json", encoding="utf-8")
            prefs = load_ui_prefs(tmp)
        self.assertEqual(prefs["density"], DEFAULTS["density"])


def _viewer_cpu_cfg(data_dir: Path):
    """Tiny OccupancyConfig so infer tests do not read repo config.yaml."""
    import sys

    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import torch
    from config import OccupancyConfig

    return OccupancyConfig(
        data_dir=data_dir,
        device=torch.device("cpu"),
        hidden=8,
        depth=1,
        seed=1,
        epochs=1,
        lr=1e-3,
        val_fraction=0.2,
        batch_size=16,
        n_surface=16,
        latent_dim=4,
    )


def _save_encoder_ckpt(models: Path, run_id: str, *, shape_encoder: str) -> None:
    """Untrained OccupancyEncoder weights with the viewer checkpoint keys."""
    import sys

    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import numpy as np
    import torch
    from occupancy_encoder import CHECKPOINT_KIND, OccupancyEncoder

    folder = models / run_id
    folder.mkdir(parents=True)
    model = OccupancyEncoder(
        hidden=8,
        depth=1,
        latent_dim=4,
        shape_encoder=shape_encoder,
    )
    ckpt = {
        "kind": CHECKPOINT_KIND,
        "state_dict": model.state_dict(),
        "hidden": 8,
        "depth": 1,
        "latent_dim": 4,
        "shape_encoder": shape_encoder,
        "n_surface": 16,
        "seed": 1,
        "parts": [
            {
                "npz": "box.npz",
                "mesh": "box.obj",
                "center": np.array([0.5, 0.5, 0.5], dtype=np.float32),
                "scale": 1.0,
            }
        ],
    }
    torch.save(ckpt, folder / "best.pt")


class ViewerInferBothEncodersTests(unittest.TestCase):
    """Job A / B rebuild the envelope from the checkpoint."""

    def test_infer_npz_surface(self) -> None:
        import numpy as np
        from infer_job import infer_uploaded_npz

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "box.obj").write_text(_CUBE_OBJ, encoding="utf-8")
            models = root / "models"
            rng = np.random.default_rng(0)
            points = rng.uniform(0.0, 1.0, size=(24, 3)).astype(np.float32)
            labels = (points[:, 0] > 0.5).astype(np.uint8)
            cfg = _viewer_cpu_cfg(root)
            _save_encoder_ckpt(models, "run_surface", shape_encoder="surface")
            out = infer_uploaded_npz(
                run_id="run_surface",
                models_root=models,
                data_dir=root,
                npz_name="box.npz",
                mesh_path="box.obj",
                points=points,
                labels=labels,
                cfg=cfg,
            )
            self.assertEqual(out["n"], 24)
            self.assertEqual(out["shape_encoder"], "surface")
            raw = base64.b64decode(out["pred_b64"])
            self.assertEqual(len(raw), 24)

    def test_infer_obj_surface(self) -> None:
        import numpy as np
        from infer_job import infer_uploaded_obj

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = root / "models"
            _save_encoder_ckpt(models, "surf_job", shape_encoder="surface")
            points = np.array(
                [[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]], dtype=np.float32
            )
            out = infer_uploaded_obj(
                run_id="surf_job",
                models_root=models,
                data_dir=root,
                obj_name="cube.obj",
                obj_text=_CUBE_OBJ,
                points=points,
                cfg=_viewer_cpu_cfg(root),
            )
            self.assertEqual(out["shape_encoder"], "surface")
            self.assertEqual(out["n"], 2)
            self.assertEqual(out["n_inside"] + out["n_outside"], 2)

    def test_infer_rejects_mesh_checkpoint(self) -> None:
        import numpy as np
        import torch
        from infer_job import infer_uploaded_npz

        src = Path(__file__).resolve().parents[1] / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from occupancy_encoder import CHECKPOINT_KIND

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "box.obj").write_text(_CUBE_OBJ, encoding="utf-8")
            models = root / "models"
            folder = models / "mesh_job"
            folder.mkdir(parents=True)
            torch.save(
                {
                    "kind": CHECKPOINT_KIND,
                    "state_dict": {},
                    "hidden": 8,
                    "depth": 1,
                    "latent_dim": 4,
                    "shape_encoder": "mesh",
                    "n_surface": 16,
                    "seed": 1,
                    "parts": [],
                },
                folder / "best.pt",
            )
            points = np.zeros((4, 3), dtype=np.float32)
            labels = np.zeros((4,), dtype=np.uint8)
            with self.assertRaises(ValueError):
                infer_uploaded_npz(
                    run_id="mesh_job",
                    models_root=models,
                    data_dir=root,
                    npz_name="box.npz",
                    mesh_path="box.obj",
                    points=points,
                    labels=labels,
                    cfg=_viewer_cpu_cfg(root),
                )


if __name__ == "__main__":
    unittest.main()
