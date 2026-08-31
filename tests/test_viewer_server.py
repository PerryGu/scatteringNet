"""Viewer helper: mesh_path must resolve to an OBJ under data_dir."""

from __future__ import annotations

import base64
import os
import sys
import tempfile
import unittest
from pathlib import Path

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
            self.assertEqual(saved["model_id"], "run_a")
            self.assertNotIn("ignored", saved)
            loaded = load_ui_prefs(tmp)
            self.assertEqual(loaded, saved)

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


if __name__ == "__main__":
    unittest.main()
