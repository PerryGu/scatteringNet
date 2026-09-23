"""Gradio occupancy pipeline: fill + infer without launching Gradio."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_GRADIO = _REPO / "src" / "gradio"
if str(_GRADIO) not in sys.path:
    sys.path.insert(0, str(_GRADIO))

from pipeline import (  # noqa: E402
    INSPECT_RUN_ID,
    apply_cut,
    default_run_id,
    fill_and_infer,
    list_run_ids,
)

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


def _cpu_cfg(data_dir: Path):
    """Tiny OccupancyConfig so this test does not read repo config.yaml."""
    import torch
    from scatteringnet.config import OccupancyConfig

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


def _save_encoder_ckpt(models: Path, run_id: str) -> None:
    """Untrained envelope OccupancyEncoder (same keys as the viewer tests)."""
    import torch
    from scatteringnet.occupancy_encoder import CHECKPOINT_KIND, OccupancyEncoder

    folder = models / run_id
    folder.mkdir(parents=True)
    model = OccupancyEncoder(
        hidden=8,
        depth=1,
        latent_dim=4,
        shape_encoder="surface",
    )
    ckpt = {
        "kind": CHECKPOINT_KIND,
        "state_dict": model.state_dict(),
        "hidden": 8,
        "depth": 1,
        "latent_dim": 4,
        "shape_encoder": "surface",
        "n_surface": 16,
        "seed": 1,
        "parts": [],
    }
    torch.save(ckpt, folder / "best.pt")


class GradioFolderTests(unittest.TestCase):
    """Keep this directory from shadowing the pip gradio package."""

    def test_no_package_init(self) -> None:
        self.assertFalse((_GRADIO / "__init__.py").is_file())

    def test_pipeline_does_not_import_gradio(self) -> None:
        text = (_GRADIO / "pipeline.py").read_text(encoding="utf-8")
        self.assertNotRegex(text, r"(?m)^(import gradio|from gradio\b)")


class GradioFigureTests(unittest.TestCase):
    """Empty GLB is a real floor mesh (not a Plotly unit-cube camera)."""

    def test_empty_glb_exists(self) -> None:
        from figure import empty_figure

        path = empty_figure()
        self.assertTrue(path.is_file())
        self.assertGreater(path.stat().st_size, 20)
        self.assertEqual(path.suffix, ".glb")
        a = empty_figure()
        b = empty_figure()
        self.assertNotEqual(a.name, b.name)

    def test_floor_grid_lives_in_orbit_js(self) -> None:
        """Floor is a fixed Babylon helper, not a per-mesh GLB grid."""
        text = (_GRADIO / "orbit.js").read_text(encoding="utf-8")
        self.assertIn("addWorldHelpers", text)
        self.assertIn("sn_gx_", text)
        self.assertIn("CreateLines", text)
        self.assertIn("GRID_HALF = 16", text)
        self.assertIn("START_RADIUS = 60", text)

    def test_shell_opacity_writes_blend_material(self) -> None:
        import json
        import struct

        from figure import _clamp_opacity, occupancy_figure

        self.assertAlmostEqual(_clamp_opacity(50), 0.5)
        self.assertAlmostEqual(_clamp_opacity(-10), 0.0)
        self.assertAlmostEqual(_clamp_opacity(200), 1.0)

        verts = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=np.float32,
        )
        faces = np.array(
            [[0, 1, 2], [0, 2, 3], [4, 7, 6], [4, 6, 5]],
            dtype=np.int32,
        )
        path = occupancy_figure(
            verts,
            faces,
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.uint8),
            mesh_opacity=50,
        )
        self.assertTrue(path.is_file())
        data = path.read_bytes()
        chunk_len = struct.unpack_from("<I", data, 12)[0]
        header = json.loads(data[20 : 20 + chunk_len].rstrip(b" \x00"))
        shell = next(m for m in header["materials"] if m.get("name") == "shell")
        self.assertEqual(shell.get("alphaMode"), "BLEND")
        self.assertAlmostEqual(
            shell["pbrMetallicRoughness"]["baseColorFactor"][3],
            0.5,
            places=2,
        )

    def test_occupancy_points_node_is_named(self) -> None:
        """orbit.js keys off the occ_points node to set Babylon pointSize."""
        import json
        import struct

        from figure import occupancy_figure

        verts = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        pts = np.array([[0.5, 0.5, 0.5], [0.2, 0.2, 0.2]], dtype=np.float32)
        pred = np.array([1, 1], dtype=np.uint8)
        data = occupancy_figure(verts, faces, pts, pred).read_bytes()
        chunk_len = struct.unpack_from("<I", data, 12)[0]
        header = json.loads(data[20 : 20 + chunk_len].rstrip(b" \x00"))
        names = [n.get("name") for n in header.get("nodes", [])]
        self.assertIn("occ_points", names)
        self.assertIn("occ_shell", names)

    def test_large_shell_is_exported(self) -> None:
        """Load OBJ must draw Human2-scale meshes (old 80k cap wrote an empty GLB)."""
        import json
        import struct

        from figure import occupancy_figure

        n = 80_001
        verts = np.zeros((n, 3), dtype=np.float32)
        verts[:, 0] = np.linspace(0.0, 1.0, n, dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        data = occupancy_figure(
            verts,
            faces,
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.uint8),
        ).read_bytes()
        chunk_len = struct.unpack_from("<I", data, 12)[0]
        header = json.loads(data[20 : 20 + chunk_len].rstrip(b" \x00"))
        names = [node.get("name") for node in header.get("nodes", [])]
        self.assertIn("occ_shell", names)

    def test_empty_occupancy_does_not_raise(self) -> None:
        from figure import occupancy_figure

        path = occupancy_figure(
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.uint8),
        )
        self.assertTrue(path.is_file())

    def test_draw_cap_matches_fill_lattice(self) -> None:
        """GLB subsample must not drop points the 80k fill already classified."""
        from figure import MAX_PLOT_POINTS, _subsample
        from pipeline import MAX_FILL_POINTS

        self.assertEqual(MAX_PLOT_POINTS, MAX_FILL_POINTS)
        n = MAX_FILL_POINTS
        pts = np.zeros((n, 3), dtype=np.float32)
        pred = np.ones((n,), dtype=np.uint8)
        kept_pts, kept_pred = _subsample(pts, pred, MAX_PLOT_POINTS)
        self.assertEqual(kept_pts.shape[0], n)
        self.assertEqual(int(kept_pred.sum()), n)
        extra = np.zeros((n + 1, 3), dtype=np.float32)
        extra_pred = np.ones((n + 1,), dtype=np.uint8)
        capped, _ = _subsample(extra, extra_pred, MAX_PLOT_POINTS)
        self.assertEqual(capped.shape[0], MAX_PLOT_POINTS)


class GradioReplaceTests(unittest.TestCase):
    """A second OBJ must clear occupancy state (not keep the first mesh)."""

    def test_preview_clears_state_and_writes_new_glb(self) -> None:
        from app import accept_obj, preview_upload

        cube = _GRADIO / "examples" / "cube.obj"
        state, path, md = preview_upload(str(cube), 50)
        self.assertIsNone(state)
        self.assertTrue(Path(path).is_file())
        self.assertIn("cube.obj", md)
        steps = list(accept_obj(str(cube), 50))
        self.assertGreaterEqual(len(steps), 1)
        last = steps[-1]
        self.assertIsNone(last[0])
        self.assertTrue(Path(str(last[1])).is_file())
        self.assertEqual(last[2], 0)
        again = list(accept_obj(str(cube), 50, reset_n=4))
        self.assertEqual(again[-1][2], 4)
        self.assertIn("cube.obj", last[3])
        self.assertIsNone(last[4])
        self.assertTrue(str(last[5]).replace("\\", "/").endswith("cube.obj"))
        held = str(last[1])
        cleared = list(accept_obj(None, 50, current_glb=held))
        self.assertEqual(str(cleared[-1][1]), held)

    def test_orbit_js_keeps_one_engine(self) -> None:
        """Page JS must swap meshes, not construct a new Engine per GLB."""
        from app import _cmd_html

        text = (_GRADIO / "orbit.js").read_text(encoding="utf-8")
        self.assertIn("ImportMeshAsync", text)
        self.assertIn("applyCam", text)
        self.assertIn("applyDefaultCam", text)
        self.assertIn("addWorldHelpers", text)
        self.assertIn("__snOrbit", text)
        self.assertIn("loadCameras = false", text)
        self.assertIn("lastPath", text)
        self.assertIn("loadBusy", text)
        self.assertNotIn("sessionStorage", text)
        self.assertNotIn("span * 1.6", text)
        self.assertNotIn("forceDefault || !keep", text)
        self.assertNotIn("lastPath && !resetChanged", text)
        load_at = text.find("async function loadGlb")
        self.assertGreater(load_at, 0)
        self.assertNotIn("new BABYLON.Engine", text[load_at:])
        cube = _GRADIO / "examples" / "cube.obj"
        html = _cmd_html(str(cube), 3)
        self.assertIn("sn-cmd", html)
        self.assertIn("|3|8|0|0", html)
        self.assertIn("|0|8|0|1", _cmd_html(str(cube), 0, force_default=True))
        self.assertIn("applyPointSize", text)
        self.assertIn("applyWireframe", text)
        self.assertIn("enableEdgesRendering", text)
        self.assertNotIn("mats[m].wireframe = want", text)
        self.assertIn("occ_points", text)
        self.assertIn("occ_shell", text)
        self.assertNotIn("idx.length === nv", text)
        self.assertIn("sendObjToGradio", text)
        self.assertIn("sn-drop-over", text)
        self.assertIn("bindLoadBtn", text)
        app_text = (_GRADIO / "app.py").read_text(encoding="utf-8")
        self.assertIn("sn-obj-file-slot", app_text)
        self.assertNotIn("gr.UploadButton", app_text)
        self.assertIn("Drop an OBJ file here", app_text)
        self.assertIn("_EXAMPLE_OBJS", app_text)
        self.assertIn("inputs=[obj_in, opacity_in, reset_n, psize_in, wire_in, glb_held]", app_text)
        from app import _EXAMPLE_OBJS

        self.assertEqual(_EXAMPLE_OBJS[0], "Obese.obj")
        self.assertNotIn("cube.obj", _EXAMPLE_OBJS)
        for name in _EXAMPLE_OBJS:
            self.assertTrue((_GRADIO / "examples" / name).is_file(), name)


class GradioPipelineTests(unittest.TestCase):
    """Job B through the Gradio wrapper; Gradio UI is not started."""

    def test_list_and_default_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp) / "models"
            _save_encoder_ckpt(models, "older_run")
            inspect = models / INSPECT_RUN_ID
            inspect.mkdir(parents=True)
            (inspect / "best.pt").write_bytes((models / "older_run" / "best.pt").read_bytes())
            ids = list_run_ids(models)
            self.assertIn(INSPECT_RUN_ID, ids)
            self.assertEqual(default_run_id(models), INSPECT_RUN_ID)

    def test_fill_and_infer_cube(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = root / "models"
            _save_encoder_ckpt(models, "surf_job")
            out = fill_and_infer(
                _CUBE_OBJ,
                obj_name="cube.obj",
                run_id="surf_job",
                density=0,
                models=models,
                cfg=_cpu_cfg(root),
            )
            self.assertEqual(out["shape_encoder"], "surface")
            self.assertEqual(out["run_id"], "surf_job")
            self.assertGreater(out["n"], 8)
            self.assertLessEqual(out["n"], 80_000)
            self.assertEqual(out["points"].shape[1], 3)
            self.assertEqual(out["probs"].shape[0], out["n"])
            self.assertEqual(out["n_inside"] + out["n_outside"], out["n"])

    def test_apply_cut_no_gpu(self) -> None:
        probs = np.array([0.1, 0.6, 0.9], dtype=np.float32)
        pred, n_in, n_out = apply_cut(probs, 0.5)
        self.assertEqual(n_in, 2)
        self.assertEqual(n_out, 1)
        np.testing.assert_array_equal(pred, np.array([0, 1, 1], dtype=np.uint8))

    def test_rejects_empty_obj(self) -> None:
        with self.assertRaises(ValueError):
            fill_and_infer("", obj_name="x.obj", run_id="none", density=71)


if __name__ == "__main__":
    unittest.main()
