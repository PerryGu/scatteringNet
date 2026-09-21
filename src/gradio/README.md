# Occupancy Gradio demo

Thin Hugging Face Gradio UI for occupancy **fill**. The inspect tool is still the Three.js page in [`src/viewer`](../viewer/README.md) (`open_viewer.bat`). This folder does not replace that viewer and does not change occupancy training.

Upload an OBJ → mesh on an XZ **floor** of thin GridHelper-style lines plus RGB AxesHelper lines (Y-up). **Drop the OBJ on the 3D view** (or **Load OBJ**). **Run model** fills with occupancy points. **Mesh opacity** (default 50%) uses a GLTF blend material so inside points show through. **Wireframe** overlays crease edges on the solid shell (inspect-style, not every triangle). **Dot size** (default 8 px) scales occupancy points without remounting the camera. The 3D pane is a persistent Babylon canvas (`orbit.js`). Loading a sample, drop, **Load OBJ**, sliders, and **Run model** keep the orbit. **Reset view** returns to the start camera (45° / 70°, origin). Same Job B path as the helper (`obj_fill` + `infer_uploaded_obj`).

## Launch

Conda env **scatteringNet** (PyTorch already there):

```text
pip install -r src/gradio/requirements.txt
python src/gradio/app.py
```

Or double-click `open_gradio.bat` in this folder. Browser: `http://127.0.0.1:7860`. Sample meshes under `examples/` (cube, horse, Player, dog, Helix_bend, TorusX3_box) appear in the **Sample OBJ** row.

Put the Space checkpoint at `models/<run_id>/best.pt`. The demo loads `default_run_id()` (inspect run `2026-09-16_18-09-31_prim_extruded_nr45_knn24_n2048_n6` when that file exists, else the newest `best.pt`). There is no model picker.

## Hugging Face Space

Create a **Gradio** Space (not Static, not Docker). Set the app file to `src/gradio/app.py` if the Space is this repo; otherwise copy this folder and point `app.py` at a `best.pt`. Hardware: **CPU Basic** is enough for the example cube; **ZeroGPU** if you want CUDA for larger meshes. This demo is not the Three.js inspect UI.

## Why this folder is named `gradio`

It sits next to `src/viewer` on purpose. It is **not** a Python package (no `__init__.py`) so `pip install -e .` does not ship a fake `gradio` library. Always `import gradio` from site-packages **before** putting `src/` on `sys.path`. Launch `app.py` as a script.

## Limits (vs the local viewer)

- Lattice cap **80,000** points (viewer 200,000).
- The GLB draws at most **25,000** occupancy points.
- No NPZ Truth / Errors, no envelope overlay, no recents / IndexedDB.
- Inside cut recuts stored sigmoid probs (no extra GPU pass), same idea as the inspect slider.
